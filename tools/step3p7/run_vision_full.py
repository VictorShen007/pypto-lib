#!/usr/bin/env python3
"""Step3.7 VIT full-pipeline host driver — chained 8-card @pl.program vs vLLM golden.

B' fix: every stage is an 8-card replicated @pl.program (PatchEmbedProg /
LnPreProg / Ds1Prog / Ds2Prog / ProjProg) or the reused TP=8 tower
(Step3p7Vision), launched via golden.run with DistributedConfig(8 cards, 0
sub-workers). NO @pl.jit launchers anywhere in this process — the @pl.jit
launcher's simpler worker staying alive was what collided with a later
@pl.program's 8-card hierarchical init (SIGSEGV), proven by a two-8-card
in-process repro (both @pl.program launches PASS; golden.run tears down chip
workers between launches).

Two validation modes (env CHAIN_ACTUAL, default unset = ISOLATION):
  ISOLATION (default): each stage fed the REAL vLLM golden intermediate and
    validated INDEPENDENTLY vs the next vLLM intermediate (the original
    "strictest validation" methodology; tight 1/128 tol per single-op stage,
    loose 0.2/0.08 for the 47-layer tower). All 7 PASS — the clean B' proof
    (6 @pl.program launches in one process, no SIGSEGV, tower in-process).
  CHAIN (CHAIN_ACTUAL=1): real pixel -> ... -> image_features dataflow; each
    stage feeds its ACTUAL captured output to the next. Front (1-3) + tower (4)
    PASS (tight front / loose tower); back (5-7) drift-fail their tight 1/128
    tol because they inherit the tower's BF16-accumulation drift — expected,
    not a bug (each back stage is validated green in ISOLATION mode).
On a2a3sim (compile-only) there is no output, so every stage falls back to the
golden input and just compiles (in either mode).

  1. PatchEmbedProg  : model_input -> patch_out      (vs patch_embed_out)
  2. host posemb add : patch_out + posemb -> posed    (vs posemb_out, exact)
  3. LnPreProg       : posed -> ln_out               (vs layer_00_layer_input)
  4. Step3p7Vision   : ln_out -> tower_out            (vs tower_out)  [was SKIPped]
  5. Ds1Prog         : tower_out -> device im2col -> ds1_out  (vs downsampler1_out)
  6. Ds2Prog         : ds1_out -> device im2col -> ds2_out     (vs downsampler2_out)
  7. ProjProg        : ds2_out(zero-pad) -> image_features (vs image_features)

Stage 4 (tower) was previously SKIPped — the 1-card @pl.jit front/back stages
left a simpler worker alive that SIGSEGV'd the tower's 8-card hierarchical
init. With B' (no @pl.jit anywhere), the tower launches cleanly in-process.

Usage::

    export LD_LIBRARY_PATH="<PTOAS_LIB>:$LD_LIBRARY_PATH"
    python -m tools.step3p7.run_vision_full \\
      --ckpt <CKPT> \\
      --dump-root <GOLDEN_DIR> \\
      -p a2a3 -d 0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

# End-to-end bench granularity: golden.run defaults to 100 rounds per stage and
# this driver runs 7 stages, so cap via E2E_BENCH_ROUNDS (default 10).
import os
import golden.runner as _R
if os.environ.get("E2E_BENCH_ROUNDS"):
    _R._BENCH_ROUNDS = int(os.environ["E2E_BENCH_ROUNDS"])
    _R._BENCH_WARMUP = int(os.environ.get("E2E_BENCH_WARMUP", "2"))


def _load_dump(dump_root: Path, name: str) -> torch.Tensor:
    """Load a golden dump .pt by name substring match."""
    for p in sorted(dump_root.glob("*.pt")):
        if name in p.name:
            obj = torch.load(str(p), map_location="cpu", weights_only=False)
            return obj.get("hidden_states", obj)
    raise FileNotFoundError(f"no dump matching '{name}' in {dump_root}")


def _replicate(t: torch.Tensor, tp_size: int) -> torch.Tensor:
    """[...] per-rank (replicated) tensor -> [tp_size, ...].

    Every rank gets the SAME full tensor (vLLM replicated parallelism for
    patch_embed/posemb/ln/ds1/ds2/projector). This is the full-tower pattern
    (L122-126), NOT the per-rank-distinct stack_2d/stack_1d helpers (those are
    for TP-sliced tower weights).
    """
    return t.unsqueeze(0).expand(tp_size, *t.shape).contiguous()


def _shard_by_rank(images: torch.Tensor, rank: int, per_rank: int) -> torch.Tensor:
    """DP shard (``--mm-encoder-tp-mode data``): rank ``rank`` takes the
    contiguous slice ``images[rank*per_rank : (rank+1)*per_rank]``.

    ``images`` is ``[N, ...]`` (N images). vLLM ``run_dp_sharded_vision_model``
    semantics: each rank runs the FULL vision tower (full weights) over its own
    ⌈N/8⌉ images, then the row-axis ``AllGatherRowsProg`` reassembles the
    per-rank ``[per_rank*rows, d]`` outputs into ``[N, d]``. ``per_rank`` must be
    ≤ VB (the tower's B-axis capacity) or the driver must split the batch.
    """
    return images[rank * per_rank : (rank + 1) * per_rank]


def _stack_rows_2d(t: torch.Tensor, vb: int, active: int, rows_per_img: int) -> torch.Tensor:
    """Single [rows_per_img, K] -> [vb*rows_per_img, K]: `active` stacked copies
    (image b's rows at [b*rows_per_img:(b+1)*rows_per_img]), rest zero."""
    rep = t.unsqueeze(0).expand(active, *t.shape).reshape(active * t.shape[0], t.shape[1])
    out = torch.zeros(vb * rows_per_img, t.shape[1], dtype=t.dtype)
    out[:active * rows_per_img] = rep
    return out


def _patch_rows_to_tokens(patch_b: torch.Tensor, vb: int, rows_pad: int, vT: int) -> torch.Tensor:
    """[vb*rows_pad, D] -> [vb*vT, D]: drop each image's pad rows [vT:rows_pad]."""
    return patch_b.reshape(vb, rows_pad, -1)[:, :vT, :].reshape(vb * vT, -1)


def _capture_and_compare(buf, key, atol, rtol, max_ratio, active_rows=None):
    """compare_fn for the "hto" output: capture the NPU actual into buf[key]
    (for chaining to the next stage) AND compare to the golden via
    ratio_allclose. Mirrors the TP8 driver's _dump_cmp (minus the /tmp save).

    active_rows: for multi-image B-axis stages, mask the compare to
    [0:active_rows] on the row axis (dim 1). The FULL actual is still captured
    into buf[key] (the B-axis chaining source); only the compare is masked —
    inactive rows are never written by the `if b < active` guard and hold
    undefined values.
    """
    from golden import ratio_allclose
    base = ratio_allclose(atol=atol, rtol=rtol, max_error_ratio=max_ratio)

    def cmp(actual, expected, *, actual_outputs=None, expected_outputs=None,
            inputs=None, rtol=None, atol=None, **kw):
        buf[key] = actual.detach().cpu().clone()
        if active_rows is not None:
            actual = actual[:, :active_rows, :]
            expected = expected[:, :active_rows, :]
        return base(actual, expected, actual_outputs=actual_outputs,
                    expected_outputs=expected_outputs, inputs=inputs,
                    rtol=rtol, atol=atol, **kw)
    return cmp


def _stage_runtime_cfg(platform: str) -> dict:
    """runtime_cfg for golden.run. PYPTO_VIT_SWIMLANE=1 enables per-task L2
    swimlane capture (DFX) on every stage of this run: raw records +
    deps.json + the auto-converted merged_swimlane_*.json land under each
    stage's compiled build dir in ``dfx_outputs/rank{r}/d{k}/``. With
    PYPTO_BENCH unset each stage is exactly ONE dispatch (validation), i.e. a
    single-request profile."""
    rcfg: dict = {"platform": platform}
    if os.environ.get("PYPTO_VIT_SWIMLANE", "").strip() not in ("", "0", "false", "False"):
        rcfg["enable_l2_swimlane"] = True
    return rcfg


def _run_program_stage(program, specs, golden_fn, platform, atol, rtol,
                       max_ratio, cap, cap_key, name, active_rows=None,
                       runtime_dir=None):
    """golden.run an 8-card @pl.program stage; capture actual into cap[cap_key]."""
    from golden import run
    from pypto.ir.distributed_compiled_program import DistributedConfig
    is_sim = platform.endswith("sim")
    r = run(
        program=program,
        specs=specs,
        golden_fn=golden_fn,
        compile_cfg={"distributed_config": DistributedConfig(
            device_ids=list(range(8)), num_sub_workers=0)},
        runtime_cfg=_stage_runtime_cfg(platform),
        rtol=rtol, atol=atol,
        compare_fn={"hto": _capture_and_compare(cap, cap_key, atol, rtol,
                                                max_ratio, active_rows)},
        compile_only=is_sim,
        runtime_dir=runtime_dir,
    )
    status = "PASS" if r.passed else "FAIL"
    print(f"  [{name}] {status}", flush=True)
    if not r.passed and r.error:
        print(f"    {r.error[:500]}", flush=True)
    return r


def _permute_ds_weight(w: torch.Tensor, c_in: int) -> torch.Tensor:
    """[9*c_in, c_out] channel-major (k=c*9+t) -> tap-major (k=t*c_in+c).

    Matches the device tap-major im2col (im2col[r, t*c_in+c]); the matmul result
    is unchanged (only the FP32 accumulation order shifts, far below 1/128 rtol).
    """
    return w.reshape(c_in, 9, -1).permute(1, 0, 2).reshape(9 * c_in, -1).contiguous()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--dump-root", type=Path, required=True)
    p.add_argument("-p", "--platform", default="a2a3",
                   choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    p.add_argument("-d", "--device", type=int, default=0,
                   help="vestigial for 8-card runs (devices 0-7 come from the "
                        "DistributedConfig); kept for interface compatibility")
    p.add_argument("--rtol", type=float, default=1.0 / 128)
    p.add_argument("--atol", type=float, default=1e-4)
    p.add_argument("--active", type=int, default=1,
                   help="active images in the B axis (1 = single-image regression "
                        "gate; >1 stacks identical copies to exercise the multi-image "
                        "B-axis offsets — each active image must reproduce the single "
                        "image result)")
    p.add_argument("--mm-encoder-tp-mode", choices=["weights", "data"],
                   default="weights",
                   help="'weights' (default) = replicated VIT — every rank runs the "
                        "full image, zero cross-card sync (current). 'data' = DP shard "
                        "(vLLM run_dp_sharded_vision_model) — ⌈N/8⌉ images per rank "
                        "then a row-axis all_gather merge (Stage 8 synthetic smoke)")
    p.add_argument("--runtime-dir", type=str, default=None,
                   help="reuse a precompiled build_output/ dir for the tower stage "
                        "(skips the 23-49min tower compile; front/back @pl.program "
                        "stages still compile fresh, ~seconds each)")
    args = p.parse_args()

    from golden import TensorSpec, ScalarSpec
    from models.step3p7.vision.vision_config import (
        V_TOKENS, V_WIDTH, V_DS1_OUT, V_DS2_OUT, V_TOKENS_FINAL_PAD,
        V_GRID, V_GRID_DS1, TP_WORLD_SIZE, V_GLOBAL_BATCH, V_PATCH,
    )
    from models.step3p7.vision.vision_weight_loader import (
        load_step3p7_vision_weights_for_rank,
        KEY_V_PATCH_EMBED, KEY_V_POSEMB, KEY_V_LN_PRE_G, KEY_V_LN_PRE_B,
        KEY_V_DS1, KEY_V_DS1_B, KEY_V_DS2, KEY_V_DS2_B, KEY_V_PROJ,
    )
    from models.step3p7.vision.vision_full_fwd import (
        PatchEmbedProg, LnPreProg, Ds1Prog, Ds2Prog, ProjProg,
        V_GRID_DS1_PAD, PE_K_PAD, PE_ROWS_PAD,
        PE_ROWS_B, LN_ROWS_B, DS1_ROWS_B, DS2_ROWS_B,
        DS1_RAW_ROWS_B, DS2_RAW_ROWS_B,
        PATCH_IM2COL_FEAT_ROWS_G, PATCH_IM2COL_AREA_G, PATCH_KW_PAD, PE_K_VALID,
    )
    from models.step3p7.vision.conv_patch import PATCH_FEAT_PAD, PATCH_FEAT
    from models.step3p7.vision.host_im2col import patch_im2col
    from models.step3p7.vision.vision_fwd_repl_os import (
        Step3p7VisionReplOSChunked,
        vTp as _tower_vtp,
    )
    from tools.step3p7.run_l3_e2e_repl import _build_specs

    tp = TP_WORLD_SIZE
    D = V_WIDTH      # 1536
    vT = V_TOKENS    # 2704
    VB = V_GLOBAL_BATCH   # 4
    active = args.active
    dr = args.dump_root
    ok = True
    cap = {}   # stage captured actual outputs ([tp, VB*rows, ...] or None on sim)
    # ISOLATION (default): each stage fed the real vLLM golden intermediate and
    # validated INDEPENDENTLY vs the next vLLM intermediate (the original
    # "strictest validation" methodology). Sidesteps the tolerance-stack issue:
    # the tower's 47-layer BF16 drift (~0.05-0.2 rel, which passes the tower's
    # loose 0.2 tolerance) would otherwise propagate into ds1/ds2/projector and
    # false-fail their tight 1/128 (~7.8e-3) tolerance. CHAIN_ACTUAL=1 switches
    # to real pixel -> ... -> image_features dataflow (front+tower PASS; back
    # stages drift-fail at tight tol — expected, each back stage is validated
    # green independently in ISOLATION mode).
    import os
    CHAIN = bool(os.environ.get("CHAIN_ACTUAL"))
    print(f"[full] mode = {'CHAIN (actual forward)' if CHAIN else 'ISOLATION (golden-feed, independent validation)'}",
          flush=True)

    # ── Load rank-0 weights (replicated front/back) ──
    print("[full] loading rank-0 weights...", flush=True)
    b = load_step3p7_vision_weights_for_rank(args.ckpt, 0, tp)
    w_patch = b[KEY_V_PATCH_EMBED]    # [592, 1536]
    posemb_w = b[KEY_V_POSEMB]        # [2704, 1536]
    ln_g = b[KEY_V_LN_PRE_G].float()  # [1536]
    ln_b = b[KEY_V_LN_PRE_B].float()  # [1536]
    w_ds1 = b[KEY_V_DS1]              # [13824, 3072]
    w_ds2 = b[KEY_V_DS2]              # [27648, 6144]
    w_proj = b[KEY_V_PROJ]            # [6144, 4096]
    ds1_b = b[KEY_V_DS1_B].reshape(1, -1)            # [1, 3072]
    ds2_b = b[KEY_V_DS2_B].reshape(1, -1)            # [1, 6144]
    # Device im2col is tap-major -> permute the conv weights channel-major ->
    # tap-major ONCE on host (the matmul result is unchanged; only the FP32
    # accumulation order shifts, far below 1/128 rtol).
    w_ds1_perm = _permute_ds_weight(w_ds1, V_WIDTH)       # [13824, 3072] tap-major
    w_ds2_perm = _permute_ds_weight(w_ds2, V_DS1_OUT)     # [27648, 6144] tap-major

    # Small-stage weights are worker-resident stacked device tensors (uploaded
    # once inside prepare; mirrors the tower's _build_specs pattern). Without
    # this, every dispatch re-uploads ~512MB/rank of weights (ds2 hw 339.7MB +
    # ds1 84.9 + proj 50.3 + posemb 35.4 + patch 2.3) — the dominant part of
    # the non-tower per-forward host. Per-call IO (himage/hx/hfeat) and the
    # Out (hto) stay host shared (they change every forward in production).
    # MEASURED (per-forward wall, rounds=30 median, cross-mode bit-identical):
    # global 5 stages
    # 82.3->40.5ms (patch_embed 13.5->8.6, ln 11.8->11.9 [weights 12KB,
    # IO-bound, neutral], ds1 15.6->9.9, ds2 34.0->6.6, proj 7.4->3.6);
    # patch 5 B6 stages 67.4->29.8; both paths total 149.8->70.3ms (-53%).
    # Whole-VIT per-forward 1864.8->1785.3ms (-4.3%). 7-stage ISOLATION
    # gates all PASS through the resident path.
    _W = {"resident": "stacked"}

    # ── Load all real vLLM intermediates ──
    print("[full] loading real vLLM golden intermediates...", flush=True)
    model_input = _load_dump(dr, "model_input")               # [3, 728, 728]
    patch_embed_golden = _load_dump(dr, "patch_embed_out")     # [2704, 1536]
    if patch_embed_golden.dim() == 3: patch_embed_golden = patch_embed_golden.squeeze(0)
    posemb_golden = _load_dump(dr, "posemb_out")               # [2704, 1536]
    if posemb_golden.dim() == 3: posemb_golden = posemb_golden.squeeze(0)
    layer_00_in = _load_dump(dr, "layer_00_layer_input")        # [2704, 1536]
    if layer_00_in.dim() == 3: layer_00_in = layer_00_in.squeeze(0)
    tower_golden = _load_dump(dr, "tower_out")                 # [2704, 1536]
    if tower_golden.dim() == 3: tower_golden = tower_golden.squeeze(0)
    ds1_golden = _load_dump(dr, "downsampler1_out")             # [676, 3072]
    if ds1_golden.dim() == 3: ds1_golden = ds1_golden.squeeze(0)
    ds2_golden = _load_dump(dr, "downsampler2_out")             # [169, 6144]
    if ds2_golden.dim() == 3: ds2_golden = ds2_golden.squeeze(0)
    image_features_golden = _load_dump(dr, "image_features")    # [169, 4096]
    if image_features_golden.dim() == 3: image_features_golden = image_features_golden.squeeze(0)
    print(f"  model_input={model_input.shape} patch_embed={patch_embed_golden.shape} "
          f"posemb={posemb_golden.shape} tower={tower_golden.shape} "
          f"ds1={ds1_golden.shape} ds2={ds2_golden.shape} image_features={image_features_golden.shape}",
          flush=True)

    # ════════ Stage 1: PatchEmbedProg (model_input → patch_out + posemb → posed) ════════
    print("[full] Stage 1: patch_embed + posemb (folded, compare to posemb_out)", flush=True)
    # kw-pad 14->16 then flatten (cheap reshape+pad, NOT F.unfold):
    # [3,728,728] -> [3,52,14,52,14] -> F.pad kw -> [3,52,14,52,16] -> [3,605696].
    p_flat = model_input.reshape(3, V_GRID, V_PATCH, V_GRID, V_PATCH)
    p_flat = F.pad(p_flat, (0, PATCH_KW_PAD - V_PATCH))          # kw 14 -> 16
    image_flat = p_flat.reshape(PATCH_IM2COL_FEAT_ROWS_G, PATCH_IM2COL_AREA_G).to(torch.bfloat16)
    # w re-laid out channel-major [3,14,14] -> kw-padded [3,14,16] -> [672], then
    # zero-padded to [768] (pad taps -> 0 weight; matmul unchanged).
    w_patch_pad = torch.zeros(PE_K_PAD, D, dtype=torch.bfloat16)
    w_k = w_patch[:PATCH_FEAT].reshape(3, V_PATCH, V_PATCH, D)
    w_kpad = torch.zeros(3, V_PATCH, PATCH_KW_PAD, D, dtype=torch.bfloat16)
    w_kpad[:, :, :V_PATCH, :] = w_k
    w_patch_pad[:PE_K_VALID] = w_kpad.reshape(PE_K_VALID, D)
    # posemb is folded ONTO the NPU (was host Stage 2): the kernel quantises the
    # FP32 matmul acc to BF16 first (vLLM conv1 emits BF16), then adds posemb in a
    # second BF16 cast — bit-matching vLLM's `x + sample_abs_posemb`. Host just
    # zero-pads rows 2704->2880 (pad rows posemb=0) and stacks VB copies.
    posemb_pad = torch.zeros(PE_ROWS_PAD, D, dtype=torch.bfloat16)
    posemb_pad[:vT] = posemb_w.to(torch.bfloat16)
    posemb_b = _stack_rows_2d(posemb_pad, VB, active, PE_ROWS_PAD)   # [VB*2880, D]
    pe_golden_pad = torch.zeros(PE_ROWS_PAD, D, dtype=torch.bfloat16)
    pe_golden_pad[:vT] = posemb_golden                                # posed golden = posemb_out
    pe_golden_b = _stack_rows_2d(pe_golden_pad, VB, active, PE_ROWS_PAD)   # [VB*2880, D]
    pe_specs = [
        TensorSpec("himage", [tp, PATCH_IM2COL_FEAT_ROWS_G, PATCH_IM2COL_AREA_G], torch.bfloat16,
                   init_value=_replicate(image_flat, tp)),
        TensorSpec("hw", [tp, PE_K_PAD, D], torch.bfloat16, init_value=_replicate(w_patch_pad, tp), **_W),
        TensorSpec("hp", [tp, PE_ROWS_B, D], torch.bfloat16, init_value=_replicate(posemb_b, tp), **_W),
        TensorSpec("hto", [tp, PE_ROWS_B, D], torch.bfloat16, is_output=True),
        ScalarSpec("hactive", torch.int32, active),
    ]
    def pe_gfn(t): t["hto"][:] = _replicate(pe_golden_b, tp)
    r = _run_program_stage(PatchEmbedProg, pe_specs, pe_gfn, args.platform,
                           args.atol, args.rtol, 0.01, cap, "patch_out", "patch_embed+posemb",
                           active_rows=active * PE_ROWS_PAD)
    ok = ok and r.passed

    # ── Stage 2 (host posemb add) removed: posemb is now folded into Stage 1. ──

    # ════════ Stage 3: LnPreProg (posed → ln_out) ════════
    print("[full] Stage 3: ln_pre (posed, compare to layer_00_layer_input)", flush=True)
    # posed now comes straight from Stage 1 (posemb folded into patch_embed).
    if CHAIN and cap.get("patch_out") is not None:
        ln_in_b = _patch_rows_to_tokens(cap["patch_out"][0], VB, PE_ROWS_PAD, vT)   # [VB*vT, D]
    else:
        ln_in_b = _stack_rows_2d(posemb_golden, VB, active, vT)   # ISOLATION: posemb_out golden
    # ln_pre is a pure B-axis shape change ([VB*2704, D]) with NO active scalar:
    # the shared row-parallel layernorm (T_DYN) normalizes each row independently,
    # so inactive zero rows -> finite beta (masked out of the compare). Golden:
    # active rows = layer_00_in stacked, inactive rows = beta (ln(zero) == beta).
    ln_golden_b = _stack_rows_2d(layer_00_in, VB, active, vT)      # [VB*vT, D]
    ln_specs = [
        TensorSpec("hx", [tp, LN_ROWS_B, D], torch.bfloat16, init_value=_replicate(ln_in_b, tp)),
        TensorSpec("hg", [tp, D], torch.float32, init_value=_replicate(ln_g, tp), **_W),
        TensorSpec("hb", [tp, D], torch.float32, init_value=_replicate(ln_b, tp), **_W),
        TensorSpec("hto", [tp, LN_ROWS_B, D], torch.bfloat16, is_output=True),
    ]
    def ln_gfn(t):
        full = ln_b.to(torch.bfloat16).unsqueeze(0).expand(LN_ROWS_B, D).contiguous()  # inactive -> beta
        full[: active * vT] = ln_golden_b[: active * vT]
        t["hto"][:] = _replicate(full, tp)
    r = _run_program_stage(LnPreProg, ln_specs, ln_gfn, args.platform,
                           args.atol, args.rtol, 0.01, cap, "ln_out", "ln_pre",
                           active_rows=active * vT)
    ok = ok and r.passed

    # ════════ Stage 4: tower (ln_out → tower_out) — REPLICATED B-axis ════════
    # The full pipeline switches from the TP8 tower (Step3p7Vision, single-image)
    # to the REPLICATED B-axis tower (Step3p7VisionReplOSChunked): every card runs
    # the full 47-layer transformer over the STACKED-2D [VB*vT, D] input with an
    # `hactive` guard, zero cross-card sync (vLLM's replicated-ViT strategy).
    # Full-tower gate stays loose (atol=1.2e-1 / rtol=8e-2 / 1% max-error-ratio),
    # device-verified PASS on both towers — the repl may be cleaner
    # than TP8 (no 94-reduce BF16 accumulation). NO @pl.jit in this process, so
    # the tower launches cleanly in-process (two-8-card in-process repro proof).
    print("[full] Stage 4: tower (ln_out -> tower_out, replicated B-axis)", flush=True)
    # vtp: the tower's per-image cos/sin pad stride (QTM_G=128 M-decoupling made
    # it 2816); _build_specs' default is the stale 2736, which fails the runtime
    # hcos stack-shape check on a fresh (live) compile — mirror the patch driver.
    tower_specs, _, _ = _build_specs(args.ckpt, dr, active=active,
                                     vtp=_tower_vtp)   # stacks `active` copies of layer_00_in
    if CHAIN and cap.get("ln_out") is not None:
        tower_in_b = cap["ln_out"][0]   # [VB*vT, D] chained
        for s in tower_specs:
            if s.name == "hx_in":
                s.init_value = _replicate(tower_in_b, tp)
                break
        print("  [tower] input = chained ln_out", flush=True)
    else:
        print("  [tower] input = golden layer_00_in (isolation)", flush=True)
    tower_golden_b = _stack_rows_2d(tower_golden, VB, active, vT)
    def tower_gfn(t): t["hto"][:] = _replicate(tower_golden_b, tp)
    r = _run_program_stage(Step3p7VisionReplOSChunked, tower_specs, tower_gfn, args.platform,
                           0.12, 0.08, 0.01, cap, "tower_out", "tower",
                           active_rows=active * vT, runtime_dir=args.runtime_dir)
    ok = ok and r.passed

    # ════════ Stage 5: Ds1Prog (tower_out → device im2col → ds1_out) ════════
    print("[full] Stage 5: ds1 (tower_out -> device border-pad + im2col, compare to downsampler1_out)", flush=True)
    if CHAIN and cap.get("tower_out") is not None:
        tower_b = cap["tower_out"][0]                        # [VB*vT, D]
    else:
        tower_b = _stack_rows_2d(tower_golden, VB, active, vT)   # [VB*vT, D]
    ds1_specs = [
        TensorSpec("hfeat", [tp, DS1_RAW_ROWS_B, D], torch.bfloat16, init_value=_replicate(tower_b, tp)),
        TensorSpec("hw", [tp, 9 * D, V_DS1_OUT], torch.bfloat16, init_value=_replicate(w_ds1_perm, tp), **_W),
        TensorSpec("hbias", [tp, 1, V_DS1_OUT], torch.bfloat16, init_value=_replicate(ds1_b, tp), **_W),
        TensorSpec("hto", [tp, DS1_ROWS_B, V_DS1_OUT], torch.bfloat16, is_output=True),
        ScalarSpec("hactive", torch.int32, active),
    ]
    def ds1_gfn(t):
        # Golden [676, 3072]; pad to [688, 3072]. Padded rows = bias (matches
        # pypto: zero-input -> matmul=0 -> +bias = bias, not 0). Stack `active`
        # copies into the B axis.
        full = ds1_b.expand(V_GRID_DS1_PAD, -1).contiguous()
        full[:ds1_golden.shape[0]] = ds1_golden
        t["hto"][:] = _replicate(_stack_rows_2d(full, VB, active, V_GRID_DS1_PAD), tp)
    r = _run_program_stage(Ds1Prog, ds1_specs, ds1_gfn, args.platform,
                           args.atol, args.rtol, 0.01, cap, "ds1_out", "ds1",
                           active_rows=active * V_GRID_DS1_PAD)
    ok = ok and r.passed

    # ════════ Stage 6: Ds2Prog (ds1_out → device im2col → ds2_out) ════════
    print("[full] Stage 6: ds2 (ds1_out -> device border-pad + im2col, compare to downsampler2_out)", flush=True)
    if CHAIN and cap.get("ds1_out") is not None:
        # Drop each image's pad rows [676:688] -> [VB*676, V_DS1_OUT].
        ds1_real_b = cap["ds1_out"][0].reshape(VB, V_GRID_DS1_PAD, V_DS1_OUT)[:, :676, :].reshape(VB * 676, V_DS1_OUT)
    else:
        ds1_real_b = _stack_rows_2d(ds1_golden, VB, active, 676)   # [VB*676, V_DS1_OUT]
    ds2_specs = [
        TensorSpec("hfeat", [tp, DS2_RAW_ROWS_B, V_DS1_OUT], torch.bfloat16, init_value=_replicate(ds1_real_b, tp)),
        TensorSpec("hw", [tp, 9 * V_DS1_OUT, V_DS2_OUT], torch.bfloat16, init_value=_replicate(w_ds2_perm, tp), **_W),
        TensorSpec("hbias", [tp, 1, V_DS2_OUT], torch.bfloat16, init_value=_replicate(ds2_b, tp), **_W),
        TensorSpec("hto", [tp, DS2_ROWS_B, V_DS2_OUT], torch.bfloat16, is_output=True),
        ScalarSpec("hactive", torch.int32, active),
    ]
    def ds2_gfn(t):
        # Padded rows = bias (matches pypto: zero-input -> matmul=0 -> +bias = bias).
        full = ds2_b.expand(V_TOKENS_FINAL_PAD, -1).contiguous()
        full[:ds2_golden.shape[0]] = ds2_golden
        t["hto"][:] = _replicate(_stack_rows_2d(full, VB, active, V_TOKENS_FINAL_PAD), tp)
    r = _run_program_stage(Ds2Prog, ds2_specs, ds2_gfn, args.platform,
                           args.atol, args.rtol, 0.01, cap, "ds2_out", "ds2",
                           active_rows=active * V_TOKENS_FINAL_PAD)
    ok = ok and r.passed

    # ════════ Stage 7: ProjProg (ds2_out → zero-pad → image_features) ════════
    print("[full] Stage 7: projector (ds2_out zero-pad, compare to image_features)", flush=True)
    if CHAIN and cap.get("ds2_out") is not None:
        # Projector input pads rows [169:176] with ZEROS (vLLM zeroes the ds2 pad
        # before the projector; the padded rows' bias must NOT be fed forward).
        ds2_padded_b = cap["ds2_out"][0].reshape(VB, V_TOKENS_FINAL_PAD, V_DS2_OUT)
        ds2_padded_b[:, 169:176, :] = 0.0
        ds2_padded_b = ds2_padded_b.reshape(VB * V_TOKENS_FINAL_PAD, V_DS2_OUT)
    else:
        ds2_padded = torch.zeros(V_TOKENS_FINAL_PAD, V_DS2_OUT, dtype=torch.bfloat16)
        ds2_padded[:ds2_golden.shape[0]] = ds2_golden
        ds2_padded_b = _stack_rows_2d(ds2_padded, VB, active, V_TOKENS_FINAL_PAD)
    proj_specs = [
        TensorSpec("hx", [tp, DS2_ROWS_B, V_DS2_OUT], torch.bfloat16, init_value=_replicate(ds2_padded_b, tp)),
        TensorSpec("hw", [tp, V_DS2_OUT, 4096], torch.bfloat16, init_value=_replicate(w_proj, tp), **_W),
        TensorSpec("hto", [tp, DS2_ROWS_B, 4096], torch.bfloat16, is_output=True),
        ScalarSpec("hactive", torch.int32, active),
    ]
    def proj_gfn(t):
        full = torch.zeros(V_TOKENS_FINAL_PAD, 4096, dtype=torch.bfloat16)
        full[:image_features_golden.shape[0]] = image_features_golden
        t["hto"][:] = _replicate(_stack_rows_2d(full, VB, active, V_TOKENS_FINAL_PAD), tp)
    r = _run_program_stage(ProjProg, proj_specs, proj_gfn, args.platform,
                           args.atol, args.rtol, 0.01, cap, "image_features", "projector",
                           active_rows=active * V_TOKENS_FINAL_PAD)
    ok = ok and r.passed

    # ════════ Stage 8 (--mm-encoder-tp-mode data): DP row-axis all_gather ════════
    # Synthetic smoke of the DP merge primitive, decoupled from the image golden:
    # rank r holds a distinct [rows, cols] block (value r+1), and AllGatherRowsProg
    # must reassemble the rank-major concat [8*rows, cols] on every rank. Exercises
    # the cross-rank pull-side collective (compile + run + correctness) without
    # needing N distinct golden images. The production wiring — shard N pixel
    # images with _shard_by_rank, run the pipeline at active=per_rank, then merge —
    # is the same code path; only the input source differs.
    if args.mm_encoder_tp_mode == "data":
        print("[full] Stage 8: DP row-axis all_gather (synthetic per-rank shards)",
              flush=True)
        from models.step3p7.vision.vision_full_fwd import AllGatherRowsProg
        ag_rows = V_TOKENS_FINAL_PAD            # 176 (per_rank=1: one image/rank)
        ag_cols = 4096                          # LM_HIDDEN
        shards = torch.stack([
            torch.full((ag_rows, ag_cols), float(r) + 1.0, dtype=torch.bfloat16)
            for r in range(tp)
        ])                                      # [8, 176, 4096], rank r == value r+1
        ag_golden = torch.cat([
            torch.full((ag_rows, ag_cols), float(r) + 1.0, dtype=torch.bfloat16)
            for r in range(tp)
        ], dim=0)                               # [1408, 4096] rank-major concat
        ag_specs = [
            TensorSpec("hlocal", [tp, ag_rows, ag_cols], torch.bfloat16,
                       init_value=shards),
            TensorSpec("hto", [tp, tp * ag_rows, ag_cols], torch.bfloat16,
                       is_output=True),
        ]
        def ag_gfn(t):
            t["hto"][:] = ag_golden.unsqueeze(0).expand(tp, -1, -1).contiguous()
        r = _run_program_stage(AllGatherRowsProg, ag_specs, ag_gfn, args.platform,
                               args.atol, args.rtol, 0.01, cap, "dp_merged",
                               "all_gather_rows")
        ok = ok and r.passed

    # ── CHAIN tolerance sweep: ground-truth lowest-pass tol per stage (actual
    # chained output vs golden), instead of a single tight pass/fail. Front
    # stages pass tight (boring); sweep tower + back stages where drift lives.
    if CHAIN and not args.platform.endswith("sim"):
        print("\n[full] CHAIN tolerance sweep (actual chained rank-0 vs golden):",
              flush=True)
        def _sweep(key, golden, name, rows, n_real=None):
            a = cap.get(key)
            if a is None:
                print(f"  [sweep {name}] no capture", flush=True); return
            g = golden[:n_real] if n_real is not None else golden
            # rank-0 actual is B-axis [VB*rows, K]; take image 0's rows only.
            a0 = a[0][:rows][:g.shape[0]]
            af = a0.float(); gf = g.float()
            d = (af - gf).abs()
            print(f"  [sweep {name}] N={g.shape[0]} max_abs={d.max():.4f} "
                  f"mean_abs={d.mean():.6f} golden_abs_mean={gf.abs().mean():.4f}",
                  flush=True)
            for (at, rt) in [(1e-4, 1/128), (5e-3, 5e-3), (1e-2, 1e-2),
                             (2e-2, 2e-2), (5e-2, 5e-2), (8e-2, 0.2), (0.2, 0.2)]:
                thr = at + rt * gf.abs()
                bad = (d > thr).float().mean().item()
                print(f"      atol={at:g} rtol={rt:g} bad={bad*100:5.2f}% -> "
                      f"{'PASS(<2%)' if bad < 0.02 else 'fail'}", flush=True)
        _sweep("tower_out", tower_golden, "S4 tower", rows=vT)
        _sweep("ds1_out", ds1_golden, "S5 ds1", rows=V_GRID_DS1_PAD,
               n_real=ds1_golden.shape[0])
        _sweep("ds2_out", ds2_golden, "S6 ds2", rows=V_TOKENS_FINAL_PAD,
               n_real=ds2_golden.shape[0])
        _sweep("image_features", image_features_golden, "S7 image_features",
               rows=V_TOKENS_FINAL_PAD, n_real=image_features_golden.shape[0])

    print(f"\n[full] {'PASS' if ok else 'FAIL'} (7-stage 8-card @pl.program pipeline vs real vLLM golden, "
          f"mode={'CHAIN' if CHAIN else 'ISOLATION'})", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
