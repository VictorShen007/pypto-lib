#!/usr/bin/env python3
"""Step3.7 VIT **patch-path** full-pipeline host driver — chained 8-card
@pl.program per-crop vs vLLM golden / torch reference.

The patch path (504^2 local crops) front/back stages live in
``vision_full_fwd_patch.py`` (PatchEmbedPatchProg / LnPrePatchProg /
Ds1PatchProg / Ds2PatchProg / ProjPatchProg). The 47-layer patch TOWER is the
REPL tower (``vision_fwd_patch_repl_os_flat*``), chosen by CAPACITY — the crop
count is NEVER hardcoded: ``n_crops`` defaults to the dump's B dim (what
vLLM's ImagePatcher produced at capture time), and the tower is
  * ``n_crops == 6``              -> flat6 (VB=6 active-only tuned fast path)
  * ``n_crops <= V_PATCH_BATCH``  -> flat  (VB=16 capacity + `active` scalar)
  * ``n_crops >  V_PATCH_BATCH``  -> flat, chunked multi-dispatch
When ``n_crops == 6`` the front/back stages additionally run as B-axis-batched
B6 programs (6 crops concatenated along the token axis, ONE launch per stage —
mirrors vLLM b=6; see vision_full_fwd_patch.py docstring). For any other count
the generic per-crop host-loop path is used.

Pipeline (ISOLATION: each stage fed the real vLLM / torch reference and
validated INDEPENDENTLY vs the next; sidesteps tolerance-stack, mirror of
run_vision_full.py). CHAIN mode (env CHAIN_ACTUAL=1) feeds each stage the
ACTUAL rank-0 output of the previous pypto stage (real pixel -> ... ->
image_features dataflow) instead of the golden; the tower's 47-layer BF16
accumulation drift then propagates into ds1/ds2/projector and is expected to
drift-fail their tight 1/128 tolerance (NOT a bug — each back stage is
separately validated green in ISOLATION mode; mirror of run_vision_full.py
CHAIN):
  1. PatchEmbedPatchProg : patch_crop -> patch_out + posemb -> posed  (vs torch _golden_posed_crop)
  2. LnPrePatchProg      : posed -> ln_out             (vs torch layernorm; + chain vs patch_layer_00_layer_input)
  3. patch REPL tower    : ln_out -> patch_tower_out  (vs patch_tower_out, loose 0.2/0.08)
  4. host im2col ds1     : patch_tower_out -> ds1_cols
     Ds1PatchProg        : ds1_cols -> ds1_out         (vs patch_downsampler1_out)
  5. host im2col ds2     : ds1_out -> ds2_cols
     Ds2PatchProg        : ds2_cols -> ds2_out         (vs patch_downsampler2_out)
  6. ProjPatchProg       : ds2_out(zero-pad) -> image_features (vs torch vit_projector; no vLLM golden)

Each stage runs as an 8-card replicated @pl.program launch (B6: one launch
covers all 6 crops; per-crop: same-program serial, proven safe by a two-8-card
in-process repro). ``--stage`` selects a subset.

Usage::

    export LD_LIBRARY_PATH="<PTOAS_LIB>:$LD_LIBRARY_PATH"
    python -m tools.step3p7.run_vision_full_patch \\
      --ckpt <CKPT> \\
      --dump-root <PATCH_GOLDEN_DIR> \\
      --global-dump-root <GOLDEN_DIR> \\
      -p a2a3 -d 0 --stage patch_embed
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

# End-to-end bench granularity: golden.run defaults to 100 rounds per stage and
# this driver runs 7+ stages, so cap via E2E_BENCH_ROUNDS (default 10).
import golden.runner as _R
if os.environ.get("E2E_BENCH_ROUNDS"):
    _R._BENCH_ROUNDS = int(os.environ["E2E_BENCH_ROUNDS"])
    _R._BENCH_WARMUP = int(os.environ.get("E2E_BENCH_WARMUP", "2"))


def _load_dump(dump_root: Path, name: str) -> torch.Tensor:
    """Load a golden dump .pt by name substring match (returns hidden_states)."""
    for p in sorted(dump_root.glob("*.pt")):
        if name in p.name:
            obj = torch.load(str(p), map_location="cpu", weights_only=False)
            return obj.get("hidden_states", obj)
    raise FileNotFoundError(f"no dump matching '{name}' in {dump_root}")


def _replicate(t: torch.Tensor, tp_size: int) -> torch.Tensor:
    """[...] per-rank (replicated) tensor -> [tp_size, ...] (vLLM replicated)."""
    return t.unsqueeze(0).expand(tp_size, *t.shape).contiguous()


def _permute_ds_weight(w: torch.Tensor, c_in: int) -> torch.Tensor:
    """[9*c_in, c_out] channel-major (k=c*9+t) -> tap-major (k=t*c_in+c).

    Matches the device tap-major im2col (im2col[r, t*c_in+c]); the matmul result
    is unchanged (only the FP32 accumulation order shifts, far below 1/128 rtol).
    """
    return w.reshape(c_in, 9, -1).permute(1, 0, 2).reshape(9 * c_in, -1).contiguous()


def _golden_posed_crop(image: torch.Tensor, w_proj: torch.Tensor, posemb: torch.Tensor) -> torch.Tensor:
    """torch ref of patch_embed + posemb (folded): crop [3,504,504] -> posed [1296,1536].

    Mirrors vLLM's TWO-step rounding exactly: conv1 emits BF16 (F.linear under
    BF16), then `x + interp(posemb)` is a BF16 add. So golden = BF16(BF16(cols @ w)
    + posemb) — the patch_embed output is BF16-quantised BEFORE posemb is added,
    matching the device fold (pl.cast acc -> BF16, then + posemb -> BF16). Returns
    BF16.
    """
    from models.step3p7.vision.host_im2col import patch_crop_im2col
    cols = patch_crop_im2col(image).float()              # [1296, 592] FP32
    patch_out = (cols @ w_proj.float()).to(torch.bfloat16)   # step 1: conv1 BF16
    return (patch_out.float() + posemb.float()).to(torch.bfloat16)   # step 2: +posemb BF16


def _drift_stats(name, actual, expected, atol, rtol):
    """Print CHAIN drift stats (max_abs / mean_abs / bad% @tol) for a stage.

    Only fires when env CHAIN_STATS is set (ad-hoc dataflow-drift analysis vs the
    ISOLATION PASS/FAIL line). Compares rank-0 actual vs expected (golden or
    chained upstream actual) — used to quantify how the tower's BF16-accumulation
    drift propagates through ds1/ds2/projector (mirror of run_vision_full.py sweep).
    """
    import os
    if not os.environ.get("CHAIN_STATS"):
        return
    a = actual[0].float() if actual.dim() >= 1 and actual.shape[0] > 1 else actual.float()
    e = expected[0].float() if expected.dim() >= 1 and expected.shape[0] > 1 else expected.float()
    diff = (a - e).abs()
    thr = atol + rtol * e.abs()
    bad = (diff > thr).float().mean().item() * 100
    print(f"    [drift] {name}: max_abs={diff.max():.4f} mean_abs={diff.mean():.6f} "
          f"bad@{atol:g}/{rtol:g}={bad:.2f}%", flush=True)


def _capture_and_compare(buf, key, atol, rtol, max_ratio, name=None):
    """compare_fn for 'hto': capture the NPU actual into buf[key] AND compare."""
    from golden import ratio_allclose
    base = ratio_allclose(atol=atol, rtol=rtol, max_error_ratio=max_ratio)

    def cmp(actual, expected, *, actual_outputs=None, expected_outputs=None,
            inputs=None, rtol=None, atol=None, **kw):
        buf[key] = actual.detach().cpu().clone()
        if name is not None:
            _drift_stats(name, actual, expected, atol, rtol)
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
                       max_ratio, cap, cap_key, name):
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
        compare_fn={"hto": _capture_and_compare(cap, cap_key, atol, rtol, max_ratio, name)},
        compile_only=is_sim,
    )
    status = "PASS" if r.passed else "FAIL"
    print(f"  [{name}] {status}", flush=True)
    if not r.passed and r.error:
        print(f"    {r.error[:600]}", flush=True)
    return r


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--dump-root", type=Path, required=True,
                   help="dir with patch_* golden dumps (patch_model_input, "
                        "patch_tower_out, patch_downsampler1/2_out, "
                        "patch_layer_00_layer_input)")
    p.add_argument("--global-dump-root", type=Path, required=True,
                   help="dir with global tower_out (for tower stage input "
                        "feeding in isolation) and patch_embed weight context")
    p.add_argument("-p", "--platform", default="a2a3",
                   choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    p.add_argument("-d", "--device", type=int, default=0)
    p.add_argument("--active", type=int, default=None,
                   help="number of patch crops to run (default: the dump's B "
                        "dim — the count vLLM's ImagePatcher produced at "
                        "capture time, never hardcoded; pass fewer for quick "
                        "regression checks)")
    p.add_argument("--rtol", type=float, default=1.0 / 128)
    p.add_argument("--atol", type=float, default=1e-4)
    p.add_argument("--stage", type=str, default="all",
                   choices=["all", "patch_embed", "ln_pre", "front_chain",
                            "tower", "ds1", "ds2", "proj"])
    p.add_argument("--runtime-dir", type=str, default=None,
                   help="reuse a precompiled build_output/ dir for the patch "
                        "tower stage (skips the tower compile; front/back "
                        "@pl.program stages still compile fresh, ~seconds each)")
    args = p.parse_args()

    from golden import TensorSpec
    from models.step3p7.vision.vision_config import (
        V_TOKENS_PATCH, V_WIDTH, V_DS1_OUT, V_DS2_OUT, LM_HIDDEN,
        V_TOKENS_FINAL_PATCH_PAD, V_PATCH_BATCH, TP_WORLD_SIZE, V_LAYERS,
        V_GRID_DS1_PATCH, V_GRID_DS2_PATCH, V_TOKENS_FINAL_PATCH,
        V_GRID_PATCH, V_PATCH,
    )
    from models.step3p7.vision.vision_weight_loader import (
        load_step3p7_vision_weights_for_rank,
        KEY_V_PATCH_EMBED, KEY_V_POSEMB, KEY_V_LN_PRE_G, KEY_V_LN_PRE_B,
        KEY_V_DS1, KEY_V_DS1_B, KEY_V_DS2, KEY_V_DS2_B, KEY_V_PROJ,
    )
    from models.step3p7.vision.vision_full_fwd_patch import (
        PatchEmbedPatchProg, LnPrePatchProg, Ds1PatchProg, Ds2PatchProg,
        ProjPatchProg, V_GRID_DS1_PATCH_PAD,
        PatchEmbedPatchB6Prog, LnPrePatchB6Prog, Ds1PatchB6Prog, Ds2PatchB6Prog,
        ProjPatchB6Prog, V_TOKENS_PATCH_B, V_GRID_DS1_PATCH_PAD_B,
        V_TOKENS_FINAL_PATCH_PAD_B, PE_K_PAD_B6, PE_B6_ROWS_PAD,
        V_GRID_DS1_PATCH_PAD_B6,
        DS1_PATCH_RAW_ROWS_B6, DS2_PATCH_RAW_ROWS_B6,
        PATCH_IM2COL_FEAT_ROWS_B6, PATCH_IM2COL_AREA_PAD, PE_K_VALID_B6, PATCH_KW_PAD,
    )
    from models.step3p7.vision.host_im2col import (
        patch_crop_im2col, downsampler1_patch_im2col, downsampler2_patch_im2col,
        golden_patch_embed_crop, golden_posemb_patch,
        golden_downsample1_patch, golden_downsample2_patch,
    )
    from models.step3p7.vision.conv_patch import PATCH_FEAT_PAD, PATCH_FEAT   # 592, 588
    from models.step3p7.vision.layernorm import layernorm as _ln_torch_ref  # not directly callable; use F.layer_norm below

    tp = TP_WORLD_SIZE
    D = V_WIDTH                          # 1536
    vT = V_TOKENS_PATCH                  # 1296
    active = args.active
    dr = args.dump_root
    ok = True
    cap = {}     # per-crop captured actuals: cap[stage][c] = rank0 tensor
    is_sim = args.platform.endswith("sim")
    want = args.stage
    CHAIN = bool(os.environ.get("CHAIN_ACTUAL"))
    print(f"[patch_full] mode = {'CHAIN (actual forward)' if CHAIN else 'ISOLATION (golden-feed, independent validation)'}",
          flush=True)

    def run_stage(name):
        return want == "all" or want == name

    # ── Load rank-0 weights (replicated front/back, shared with global) ──
    print("[patch_full] loading rank-0 weights...", flush=True)
    b = load_step3p7_vision_weights_for_rank(args.ckpt, 0, tp)
    w_patch = b[KEY_V_PATCH_EMBED]        # [592, 1536]
    posemb_w = b[KEY_V_POSEMB]            # [2704, 1536]
    ln_g = b[KEY_V_LN_PRE_G].float()      # [1536]
    ln_b = b[KEY_V_LN_PRE_B].float()      # [1536]
    w_ds1 = b[KEY_V_DS1]                  # [13824, 3072]
    w_ds2 = b[KEY_V_DS2]                  # [27648, 6144]
    w_proj = b[KEY_V_PROJ]                # [6144, 4096]
    ds1_b = b[KEY_V_DS1_B].reshape(1, -1).float()      # [1, 3072]  (ds1 conv bias)
    ds2_b = b[KEY_V_DS2_B].reshape(1, -1).float()      # [1, 6144]
    # Device im2col is tap-major -> permute the conv weights channel-major ->
    # tap-major ONCE on host (matmul result unchanged; only the FP32 accumulation
    # order shifts, far below 1/128 rtol).
    w_ds1_perm = _permute_ds_weight(w_ds1, V_WIDTH)    # [13824, 3072] tap-major
    w_ds2_perm = _permute_ds_weight(w_ds2, V_DS1_OUT)  # [27648, 6144] tap-major
    # Small-stage weights are worker-resident stacked device tensors (uploaded
    # once inside prepare; mirrors the tower's _build_specs pattern) — without
    # this every dispatch re-uploads ~500MB/rank (ds2 hw 339.7MB dominant).
    # Per-call IO (himage/hx/hfeat/him2col) and the Out (hto) stay host shared.
    # MEASURED (per-forward wall, rounds=30 median, cross-mode bit-identical):
    # patch 5 B6 stages
    # 67.4->29.8ms (patch_embed 8.8->6.7, ln 6.0->5.6, ds1 13.6->7.8,
    # ds2 32.7->7.1, proj 6.6->2.9). Official patch-path ISOLATION gates
    # (patch_embed/ln_pre/tower/ds1/ds2/proj/front_chain c0-c5) all PASS
    # through the resident path.
    _W = {"resident": "stacked"}
    # torch layernorm reference (eps=1e-5, gamma+beta), matches vLLM nn.LayerNorm
    def torch_ln(x_bf16):
        return torch.nn.functional.layer_norm(x_bf16.float(), (D,), ln_g, ln_b, 1e-5).to(torch.bfloat16)

    # ── Load real patch golden dumps ──
    print("[patch_full] loading patch golden dumps...", flush=True)
    patch_pixels = _load_dump(dr, "patch_model_input")           # [B, 3, 504, 504], B = dump's crop count
    layer_00_in = _load_dump(dr, "patch_layer_00_layer_input")    # [B, 1296, 1536]
    patch_tower_golden = _load_dump(dr, "patch_tower_out")        # [B, 1296, 1536]
    ds1_golden = _load_dump(dr, "patch_downsampler1_out")        # [B, 3072, 18, 18]
    ds2_golden = _load_dump(dr, "patch_downsampler2_out")        # [B, 6144, 9, 9]
    print(f"  pixels={tuple(patch_pixels.shape)} layer_00_in={tuple(layer_00_in.shape)} "
          f"tower={tuple(patch_tower_golden.shape)} ds1={tuple(ds1_golden.shape)} "
          f"ds2={tuple(ds2_golden.shape)}", flush=True)
    # n_crops comes from the DUMP (mirror of vLLM's ImagePatcher at capture
    # time); --active only caps it for quick regression runs.
    n_crops = patch_pixels.shape[0] if active is None else min(active, patch_pixels.shape[0])
    print(f"  n_crops={n_crops} (dump B={patch_pixels.shape[0]})", flush=True)

    # interpolated patch posemb (same for all crops): [1296, 1536]
    posemb_patch = golden_posemb_patch(posemb_w).to(torch.bfloat16)

    # ════════ Stage 1: PatchEmbedPatchProg (per crop) ════════
    pe_out = [None] * n_crops
    if run_stage("patch_embed") or run_stage("front_chain"):
        if n_crops == 6:
            # B-axis batched: all 6 crops concatenated in ONE launch (B6 kernel,
            # weight streamed once per task instead of once per crop). posemb is
            # folded onto the NPU (was host Stage 2).
            print(f"\n[patch_full] Stage 1: patch_embed + posemb (B6 batched, folded, vs torch golden, "
                  f"tight {args.atol:g}/{args.rtol:g})", flush=True)
            # Feed the crop pixels kw-padded (14->16) then flattened:
            # [6,3,504,504] -> [6,3,36,14,36,14] -> F.pad kw -> [6,3,36,14,36,16]
            # -> [18, 290304] (row = crop*3+channel, col = oh*8064+kh*576+ow*16+kw).
            # Host does the cheap reshape + kw-pad ONLY (not F.unfold); the NPU
            # does the k14-s14-p0 im2col gather + K/row zero-pad on-device, so
            # the wall-clock matches vLLM's on-device patch_embed scope.
            p_flat = patch_pixels.reshape(n_crops, 3, V_GRID_PATCH, V_PATCH, V_GRID_PATCH, V_PATCH)
            p_flat = F.pad(p_flat, (0, PATCH_KW_PAD - V_PATCH))          # kw 14 -> 16
            image_flat = p_flat.reshape(PATCH_IM2COL_FEAT_ROWS_B6, PATCH_IM2COL_AREA_PAD).to(torch.bfloat16)
            # w re-laid out channel-major [3,14,14] -> kw-padded [3,14,16] -> [672],
            # then zero-padded to [768] (pad taps -> 0 weight; matmul unchanged).
            w_pe_b6 = torch.zeros(PE_K_PAD_B6, D, dtype=torch.bfloat16)
            w_k = w_patch[:PATCH_FEAT].reshape(3, V_PATCH, V_PATCH, D)   # [3,14,14,D]
            w_kpad = torch.zeros(3, V_PATCH, PATCH_KW_PAD, D, dtype=torch.bfloat16)
            w_kpad[:, :, :V_PATCH, :] = w_k
            w_pe_b6[:PE_K_VALID_B6] = w_kpad.reshape(PE_K_VALID_B6, D)
            # posemb [1296,D] is shared by all 6 crops; stack 6 + zero-pad rows
            # 7776 -> 8064 (pad rows posemb=0, output stays bias-free).
            posemb_b6 = torch.cat([posemb_patch for _ in range(6)], dim=0)
            posemb_b6 = torch.cat(
                [posemb_b6, torch.zeros(PE_B6_ROWS_PAD - V_TOKENS_PATCH_B, D, dtype=torch.bfloat16)], dim=0)
            pe_golden_b = torch.cat(
                [_golden_posed_crop(patch_pixels[c], w_patch, posemb_patch) for c in range(6)], dim=0)
            pe_golden_b = torch.cat(
                [pe_golden_b, torch.zeros(PE_B6_ROWS_PAD - V_TOKENS_PATCH_B, D, dtype=torch.bfloat16)], dim=0)
            specs = [
                TensorSpec("himage", [tp, PATCH_IM2COL_FEAT_ROWS_B6, PATCH_IM2COL_AREA_PAD], torch.bfloat16,
                           init_value=_replicate(image_flat, tp)),
                TensorSpec("hw", [tp, PE_K_PAD_B6, D], torch.bfloat16, init_value=_replicate(w_pe_b6, tp), **_W),
                TensorSpec("hp", [tp, PE_B6_ROWS_PAD, D], torch.bfloat16, init_value=_replicate(posemb_b6, tp), **_W),
                TensorSpec("hto", [tp, PE_B6_ROWS_PAD, D], torch.bfloat16, is_output=True),
            ]
            def pe_gfn(t): t["hto"][:] = _replicate(pe_golden_b, tp)
            cap_c = {}
            r = _run_program_stage(PatchEmbedPatchB6Prog, specs, pe_gfn, args.platform,
                                   args.atol, args.rtol, 0.01, cap_c, "patch_out", "patch_embed+posemb[b6]")
            ok = ok and r.passed
            pe_b = cap_c.get("patch_out", torch.zeros(tp, PE_B6_ROWS_PAD, D, dtype=torch.bfloat16))[0]
            pe_b = pe_b[:V_TOKENS_PATCH_B]
            pe_out = [pe_b[c * vT:(c + 1) * vT].float().to(torch.bfloat16) for c in range(6)]
        else:
            print(f"\n[patch_full] Stage 1: patch_embed + posemb (per-crop, folded, vs torch golden, "
                  f"tight {args.atol:g}/{args.rtol:g})", flush=True)
            for c in range(n_crops):
                cols = patch_crop_im2col(patch_pixels[c])             # [1296, 592] BF16
                pe_golden_c = _golden_posed_crop(patch_pixels[c], w_patch, posemb_patch)  # [1296, 1536]
                specs = [
                    TensorSpec("hx", [tp, vT, PATCH_FEAT_PAD], torch.bfloat16, init_value=_replicate(cols, tp)),
                    TensorSpec("hw", [tp, PATCH_FEAT_PAD, D], torch.bfloat16, init_value=_replicate(w_patch, tp), **_W),
                    TensorSpec("hp", [tp, vT, D], torch.bfloat16, init_value=_replicate(posemb_patch, tp), **_W),
                    TensorSpec("hto", [tp, vT, D], torch.bfloat16, is_output=True),
                ]
                def pe_gfn(t, _g=pe_golden_c): t["hto"][:] = _replicate(_g, tp)
                cap_c = {}
                r = _run_program_stage(PatchEmbedPatchProg, specs, pe_gfn, args.platform,
                                       args.atol, args.rtol, 0.01, cap_c, "patch_out", f"patch_embed+posemb[c{c}]")
                ok = ok and r.passed
                pe_out[c] = cap_c.get("patch_out", torch.zeros(tp, vT, D, dtype=torch.bfloat16))[0]
            pe_out = [p.float().to(torch.bfloat16) for p in pe_out]

    # ── Stage 2 (host posemb add) removed: posemb is folded into Stage 1. ──

    # ════════ Stage 3: LnPrePatchProg (posed -> ln_out) ════════
    ln_out = [None] * n_crops
    if run_stage("ln_pre") or run_stage("front_chain"):
        if n_crops == 6:
            # B-axis batched: 6 crops' posed stacked in ONE launch.
            print(f"\n[patch_full] Stage 3: ln_pre (B6 batched, vs torch nn.LayerNorm, "
                  f"tight {args.atol:g}/{args.rtol:g})", flush=True)
            posed_list = []
            for c in range(6):
                posed_list.append(pe_out[c] if pe_out[c] is not None else
                                  _golden_posed_crop(patch_pixels[c], w_patch, posemb_patch))
            posed_b = torch.cat(posed_list, dim=0)
            ln_golden_b = torch_ln(posed_b)
            specs = [
                TensorSpec("hx", [tp, V_TOKENS_PATCH_B, D], torch.bfloat16, init_value=_replicate(posed_b, tp)),
                TensorSpec("hg", [tp, D], torch.float32, init_value=_replicate(ln_g, tp), **_W),
                TensorSpec("hb", [tp, D], torch.float32, init_value=_replicate(ln_b, tp), **_W),
                TensorSpec("hto", [tp, V_TOKENS_PATCH_B, D], torch.bfloat16, is_output=True),
            ]
            def ln_gfn(t): t["hto"][:] = _replicate(ln_golden_b, tp)
            cap_c = {}
            r = _run_program_stage(LnPrePatchB6Prog, specs, ln_gfn, args.platform,
                                   args.atol, args.rtol, 0.01, cap_c, "ln_out", "ln_pre[b6]")
            ok = ok and r.passed
            ln_b = cap_c.get("ln_out", torch.zeros(tp, V_TOKENS_PATCH_B, D, dtype=torch.bfloat16))[0]
            ln_out = [ln_b[c * vT:(c + 1) * vT].float().to(torch.bfloat16) for c in range(6)]
        else:
            print(f"\n[patch_full] Stage 3: ln_pre (per-crop, vs torch nn.LayerNorm, "
                  f"tight {args.atol:g}/{args.rtol:g})", flush=True)
            for c in range(n_crops):
                posed_c = pe_out[c] if pe_out[c] is not None else \
                          _golden_posed_crop(patch_pixels[c], w_patch, posemb_patch)
                ln_golden_c = torch_ln(posed_c)                       # [1296, 1536]
                specs = [
                    TensorSpec("hx", [tp, vT, D], torch.bfloat16, init_value=_replicate(posed_c, tp)),
                    TensorSpec("hg", [tp, D], torch.float32, init_value=_replicate(ln_g, tp), **_W),
                    TensorSpec("hb", [tp, D], torch.float32, init_value=_replicate(ln_b, tp), **_W),
                    TensorSpec("hto", [tp, vT, D], torch.bfloat16, is_output=True),
                ]
                def ln_gfn(t, _g=ln_golden_c): t["hto"][:] = _replicate(_g, tp)
                cap_c = {}
                r = _run_program_stage(LnPrePatchProg, specs, ln_gfn, args.platform,
                                       args.atol, args.rtol, 0.01, cap_c, "ln_out", f"ln_pre[c{c}]")
                ok = ok and r.passed
                ln_out[c] = cap_c.get("ln_out", torch.zeros(tp, vT, D, dtype=torch.bfloat16))[0]
            ln_out = [x.float().to(torch.bfloat16) for x in ln_out]

    # ════════ Front chain: embed+posemb+ln vs patch_layer_00_layer_input ════════
    if run_stage("front_chain"):
        print(f"\n[patch_full] FRONT CHAIN: embed+posemb+ln vs patch_layer_00_layer_input "
              f"(per-crop, tight {args.atol:g}/{args.rtol:g})", flush=True)
        for c in range(n_crops):
            ln_c = ln_out[c] if ln_out[c] is not None else torch_ln(
                _golden_posed_crop(patch_pixels[c], w_patch, posemb_patch))
            g = layer_00_in[c]                                       # [1296, 1536]
            af, gf = ln_c.float(), g.float()
            diff = (af - gf).abs()
            thr = args.atol + args.rtol * gf.abs()
            bad = (diff > thr).float().mean().item()
            passed = bad < 0.01
            print(f"  [front_chain c{c}] {'PASS' if passed else 'FAIL'} "
                  f"bad={bad*100:5.2f}% max_abs={diff.max():.6f} (tol {args.atol:g}/{args.rtol:g})", flush=True)
            ok = ok and passed

    # ════════ Stage 4: patch tower (ln_out -> patch_tower_out) ════════
    if run_stage("tower"):
        # Tower selection is CAPACITY-based — the crop count is NEVER
        # hardcoded (it comes from the dump's B dim, which mirrors vLLM's
        # ImagePatcher at capture time):
        #   n_crops == 6              -> flat6 (VB=6 active-only tuned fast path)
        #   n_crops <= V_PATCH_BATCH  -> flat (VB=16 capacity + `active` scalar)
        #   n_crops >  V_PATCH_BATCH  -> flat, chunked multi-dispatch
        # Both towers share the REPL host_orch contract (hx_in [tp, vTe, D]
        # stacked crops, hactive scalar, hto [tp, vTe, D]) and the full-weight
        # REPL specs (run_l3_e2e_repl).
        from tools.step3p7.run_l3_e2e_repl import _build_specs as _build_specs_repl
        if n_crops == 6:
            from models.step3p7.vision.vision_fwd_patch_repl_os_flat6 import (
                Step3p7VisionPatchReplOSFlat6Chunked as _PatchTowerRepl,
                vTp as _tower_vtp,
            )
            VB = 6
            tower_label = "flat6 (VB=6 active-only, tuned 118.5ms fast path)"
        else:
            from models.step3p7.vision.vision_fwd_patch_repl_os_flat import (
                Step3p7VisionPatchReplOSFlatChunked as _PatchTowerRepl,
            )
            VB = V_PATCH_BATCH
            _tower_vtp = None
            tower_label = f"flat (VB={VB} capacity, active={n_crops})"
        print(f"\n[patch_full] Stage 4: patch tower {tower_label} "
              f"(vs patch_tower_out, loose 0.2/0.08/2%)", flush=True)

        from golden import ratio_allclose, run
        from pypto.ir.distributed_compiled_program import DistributedConfig
        base_loose = ratio_allclose(atol=0.12, rtol=0.08, max_error_ratio=0.01)
        # Chunked dispatch: ceil(n_crops/VB) launches of the SAME compiled
        # tower; each chunk stacks min(VB, remaining) crops into the fixed
        # [vTe] input (rest zero) and compares only its active rows. The
        # single-chunk case (n_crops <= VB, e.g. the 15-crop 2560x1440 image)
        # is the device-verified path; multi-chunk is the same loop with
        # n_chunks > 1, so any crop count works without recompiling.
        vTe = VB * vT
        n_chunks = (n_crops + VB - 1) // VB
        tower_pass = True
        for ch in range(n_chunks):
            c0 = ch * VB
            active_c = min(VB, n_crops - c0)
            # Stack this chunk's crops into the [vTe] input. CHAIN feeds the
            # ACTUAL front-chain ln_out (real dataflow); ISOLATION feeds the
            # REAL vLLM patch_layer_00_layer_input golden.
            ln_stack = torch.zeros(vTe, D, dtype=torch.bfloat16)
            if CHAIN:
                for c in range(c0, c0 + active_c):
                    src = ln_out[c] if ln_out[c] is not None else layer_00_in[c]
                    ln_stack[(c - c0) * vT:(c - c0 + 1) * vT] = src.to(torch.bfloat16)
                print(f"  [tower ch{ch}] input = chained ln_out (actual front-chain output)", flush=True)
            else:
                ln_stack[: active_c * vT] = layer_00_in[c0:c0 + active_c].reshape(active_c * vT, D)
                print(f"  [tower ch{ch}] input = golden layer_00_in (isolation)", flush=True)
            specs_p, _, _ = _build_specs_repl(args.ckpt, dr, patch=True,
                                              active=active_c, patch_vb=VB,
                                              vtp=_tower_vtp)
            for s in specs_p:
                if s.name == "hx_in":
                    s.init_value = ln_stack.unsqueeze(0).expand(tp, -1, -1).contiguous()
                    break
            # Golden: stack this chunk's patch_tower_out rows (rest zero);
            # compare active rows only.
            g_stack = torch.zeros(vTe, D, dtype=torch.bfloat16)
            g_stack[: active_c * vT] = patch_tower_golden[c0:c0 + active_c].reshape(active_c * vT, D)
            active_rows = active_c * vT
            def cmp_tower(actual, expected, _c0=c0, _ar=active_rows, **kw):
                # Keep a [tp, n_crops*vT, D] chaining buffer; this chunk fills
                # its crop slice with the active rows (rest stays zero).
                if "tower_out_full" not in cap:
                    cap["tower_out_full"] = torch.zeros(tp, n_crops * vT, D,
                                                        dtype=torch.bfloat16)
                cap["tower_out_full"][:, _c0 * vT:_c0 * vT + _ar, :] = \
                    actual[:, :_ar, :].detach().cpu()
                _drift_stats(f"tower ch{ch}", actual[:, :_ar, :],
                             expected[:, :_ar, :], 0.12, 0.08)
                a = actual[:, :_ar, :]; e = expected[:, :_ar, :]
                return base_loose(a, e, **kw)
            def tgfn(t): t["hto"][:] = g_stack.unsqueeze(0).expand(tp, -1, -1).contiguous()
            r = run(program=_PatchTowerRepl, specs=specs_p, golden_fn=tgfn,
                    compile_cfg={"distributed_config": DistributedConfig(
                        device_ids=list(range(8)), num_sub_workers=0)},
                    runtime_cfg=_stage_runtime_cfg(args.platform),
                    rtol=0.08, atol=0.12,
                    compare_fn={"hto": cmp_tower}, compile_only=is_sim,
                    runtime_dir=args.runtime_dir)
            tower_pass = tower_pass and r.passed
            if not r.passed and r.error:
                print(f"    [tower ch{ch}] {r.error[:600]}", flush=True)
        print(f"  [patch_tower] {'PASS' if tower_pass else 'FAIL'}", flush=True)
        ok = ok and tower_pass

    # ════════ Stage 5: Ds1PatchProg (patch_tower_out -> reshape+im2col -> ds1) ════════
    if run_stage("ds1"):
        if n_crops == 6:
            # B-axis batched: 6 crops' RAW FEAT stacked to 7776 rows; the device
            # folds BOTH the 1-zero-border pad AND the k3 s2 p1 im2col onto the
            # NPU (host no longer does F.pad/F.unfold).
            print(f"\n[patch_full] Stage 5: ds1 (B6 batched, device im2col, vs patch_downsampler1_out, "
                  f"tight {args.atol:g}/{args.rtol:g})", flush=True)
            tower_rows, full_list = [], []
            for c in range(6):
                # CHAIN: feed the ACTUAL tower_out slice for this crop; ISOLATION: golden.
                if CHAIN and "tower_out_full" in cap:
                    tower_c = cap["tower_out_full"][0][c * vT:(c + 1) * vT]   # [1296, 1536] actual
                else:
                    tower_c = patch_tower_golden[c]                      # [1296, 1536]
                tower_rows.append(tower_c)                               # token-major [1296, 1536]
                g_tokens = ds1_golden[c].permute(1, 2, 0).reshape(-1, V_DS1_OUT)
                full = ds1_b.expand(V_GRID_DS1_PATCH_PAD, -1).contiguous().to(torch.bfloat16)
                full[:324] = g_tokens
                full_list.append(full)
            tower_b = torch.cat(tower_rows, dim=0)                       # [7776, 1536]
            full_b = torch.cat(full_list, dim=0)                         # [2016, 3072]
            # trailing pad rows 2016 -> 2048: bias rows (device im2col rows are
            # zeroed -> matmul=0 -> +bias = bias).
            full_b = torch.cat(
                [full_b,
                 ds1_b.to(torch.bfloat16).expand(
                     V_GRID_DS1_PATCH_PAD_B6 - V_GRID_DS1_PATCH_PAD_B, -1).contiguous()], dim=0)
            specs = [
                TensorSpec("hfeat", [tp, DS1_PATCH_RAW_ROWS_B6, D], torch.bfloat16,
                           init_value=_replicate(tower_b, tp)),
                TensorSpec("hw", [tp, 9 * D, V_DS1_OUT], torch.bfloat16, init_value=_replicate(w_ds1_perm, tp), **_W),
                TensorSpec("hbias", [tp, 1, V_DS1_OUT], torch.bfloat16,
                           init_value=_replicate(ds1_b.to(torch.bfloat16), tp), **_W),
                TensorSpec("hto", [tp, V_GRID_DS1_PATCH_PAD_B6, V_DS1_OUT], torch.bfloat16, is_output=True),
            ]
            def ds1_gfn(t): t["hto"][:] = _replicate(full_b, tp)
            cap_c = {}
            r = _run_program_stage(Ds1PatchB6Prog, specs, ds1_gfn, args.platform,
                                   args.atol, args.rtol, 0.01, cap_c, "ds1_out", "ds1[b6]")
            ok = ok and r.passed
            # split rank0 actual [2016, 3072] into per-crop [336, 3072] for chaining
            act_b = cap_c.get("ds1_out", torch.zeros(tp, V_GRID_DS1_PATCH_PAD_B6, V_DS1_OUT,
                                                     dtype=torch.bfloat16))[0]
            for c in range(6):
                cap.setdefault("ds1_out", {})[c] = act_b[c * V_GRID_DS1_PATCH_PAD:(c + 1) * V_GRID_DS1_PATCH_PAD]
        else:
            print(f"\n[patch_full] Stage 5: ds1 (per-crop, vs patch_downsampler1_out, "
                  f"tight {args.atol:g}/{args.rtol:g})", flush=True)
            for c in range(n_crops):
                # CHAIN: feed the ACTUAL tower_out slice for this crop; ISOLATION: golden.
                if CHAIN and "tower_out_full" in cap:
                    tower_c = cap["tower_out_full"][0][c * vT:(c + 1) * vT]   # [1296, 1536] actual
                else:
                    tower_c = patch_tower_golden[c]                      # [1296, 1536]
                feat36 = tower_c.t().reshape(D, V_GRID_PATCH, V_GRID_PATCH)  # [1536, 36, 36]
                cols = downsampler1_patch_im2col(feat36)             # [336, 13824] padded
                # ds1 golden is NCHW [3072,18,18]; flatten to [324,3072] token-major
                # (token (r,c) -> row r*18+c) to match the kernel's im2col token order.
                g_tokens = ds1_golden[c].permute(1, 2, 0).reshape(-1, V_DS1_OUT)   # [324, 3072]
                # pad to [336, 3072] with bias (zero-input -> matmul=0 -> +bias = bias)
                full = ds1_b.expand(V_GRID_DS1_PATCH_PAD, -1).contiguous().to(torch.bfloat16)
                full[:324] = g_tokens
                specs = [
                    TensorSpec("him2col", [tp, V_GRID_DS1_PATCH_PAD, 9 * D], torch.bfloat16,
                               init_value=_replicate(cols, tp)),
                    TensorSpec("hw", [tp, 9 * D, V_DS1_OUT], torch.bfloat16, init_value=_replicate(w_ds1, tp), **_W),
                    TensorSpec("hbias", [tp, 1, V_DS1_OUT], torch.bfloat16,
                               init_value=_replicate(ds1_b.to(torch.bfloat16), tp), **_W),
                    TensorSpec("hto", [tp, V_GRID_DS1_PATCH_PAD, V_DS1_OUT], torch.bfloat16, is_output=True),
                ]
                def ds1_gfn(t, _g=full): t["hto"][:] = _replicate(_g, tp)
                cap_c = {}
                r = _run_program_stage(Ds1PatchProg, specs, ds1_gfn, args.platform,
                                       args.atol, args.rtol, 0.01, cap_c, "ds1_out", f"ds1[c{c}]")
                ok = ok and r.passed
                # capture rank0 actual [336, 3072] for chaining to ds2 (CHAIN mode)
                cap.setdefault("ds1_out", {})[c] = cap_c.get(
                    "ds1_out", torch.zeros(tp, V_GRID_DS1_PATCH_PAD, V_DS1_OUT, dtype=torch.bfloat16))[0]

    # ════════ Stage 6: Ds2PatchProg (ds1_out -> reshape+im2col -> ds2) ════════
    if run_stage("ds2"):
        if n_crops == 6:
            # B-axis batched: 6 crops' RAW FEAT stacked to 1944 rows; the device
            # folds BOTH the 1-zero-border pad AND the k3 s2 p1 im2col onto the NPU.
            print(f"\n[patch_full] Stage 6: ds2 (B6 batched, device im2col, vs patch_downsampler2_out, "
                  f"tight {args.atol:g}/{args.rtol:g})", flush=True)
            ds1_rows, full_list = [], []
            for c in range(6):
                # CHAIN: feed the ACTUAL ds1_out (token-major [324,3072] real rows);
                # ISOLATION: golden [3072,18,18] -> token-major [324,3072].
                if CHAIN and "ds1_out" in cap:
                    ds1_rows.append(cap["ds1_out"][c][:324])         # [324, 3072] real rows
                else:
                    ds1_rows.append(ds1_golden[c].permute(1, 2, 0).reshape(-1, V_DS1_OUT))
                g_tokens = ds2_golden[c].permute(1, 2, 0).reshape(-1, V_DS2_OUT)
                full = ds2_b.expand(V_TOKENS_FINAL_PATCH_PAD, -1).contiguous().to(torch.bfloat16)
                full[:81] = g_tokens
                full_list.append(full)
            ds1_b_cat = torch.cat(ds1_rows, dim=0)                   # [1944, 3072] (6*324)
            full_b = torch.cat(full_list, dim=0)                     # [576, 6144]
            specs = [
                TensorSpec("hfeat", [tp, DS2_PATCH_RAW_ROWS_B6, V_DS1_OUT], torch.bfloat16,
                           init_value=_replicate(ds1_b_cat, tp)),
                TensorSpec("hw", [tp, 9 * V_DS1_OUT, V_DS2_OUT], torch.bfloat16, init_value=_replicate(w_ds2_perm, tp), **_W),
                TensorSpec("hbias", [tp, 1, V_DS2_OUT], torch.bfloat16,
                           init_value=_replicate(ds2_b.to(torch.bfloat16), tp), **_W),
                TensorSpec("hto", [tp, V_TOKENS_FINAL_PATCH_PAD_B, V_DS2_OUT], torch.bfloat16, is_output=True),
            ]
            def ds2_gfn(t): t["hto"][:] = _replicate(full_b, tp)
            cap_c = {}
            r = _run_program_stage(Ds2PatchB6Prog, specs, ds2_gfn, args.platform,
                                   args.atol, args.rtol, 0.01, cap_c, "ds2_out", "ds2[b6]")
            ok = ok and r.passed
            # split rank0 actual [576, 6144] into per-crop [96, 6144] for chaining
            act_b = cap_c.get("ds2_out", torch.zeros(tp, V_TOKENS_FINAL_PATCH_PAD_B, V_DS2_OUT,
                                                     dtype=torch.bfloat16))[0]
            for c in range(6):
                cap.setdefault("ds2_out", {})[c] = act_b[c * V_TOKENS_FINAL_PATCH_PAD:(c + 1) * V_TOKENS_FINAL_PATCH_PAD]
        else:
            print(f"\n[patch_full] Stage 6: ds2 (per-crop, vs patch_downsampler2_out, "
                  f"tight {args.atol:g}/{args.rtol:g})", flush=True)
            for c in range(n_crops):
                # CHAIN: feed the ACTUAL ds1_out (token-major [324,3072] real rows)
                # reshaped to NCHW [3072,18,18]; ISOLATION: golden [3072,18,18].
                if CHAIN and "ds1_out" in cap:
                    ds1_act = cap["ds1_out"][c][:324]                 # [324, 3072] real rows
                    feat18 = ds1_act.t().reshape(V_DS1_OUT, V_GRID_DS1_PATCH, V_GRID_DS1_PATCH)
                else:
                    feat18 = ds1_golden[c]                            # [3072, 18, 18]
                cols = downsampler2_patch_im2col(feat18)            # [96, 27648] padded
                g_tokens = ds2_golden[c].permute(1, 2, 0).reshape(-1, V_DS2_OUT)   # [81, 6144]
                full = ds2_b.expand(V_TOKENS_FINAL_PATCH_PAD, -1).contiguous().to(torch.bfloat16)
                full[:81] = g_tokens
                specs = [
                    TensorSpec("him2col", [tp, V_TOKENS_FINAL_PATCH_PAD, 9 * V_DS1_OUT], torch.bfloat16,
                               init_value=_replicate(cols, tp)),
                    TensorSpec("hw", [tp, 9 * V_DS1_OUT, V_DS2_OUT], torch.bfloat16, init_value=_replicate(w_ds2, tp), **_W),
                    TensorSpec("hbias", [tp, 1, V_DS2_OUT], torch.bfloat16,
                               init_value=_replicate(ds2_b.to(torch.bfloat16), tp), **_W),
                    TensorSpec("hto", [tp, V_TOKENS_FINAL_PATCH_PAD, V_DS2_OUT], torch.bfloat16, is_output=True),
                ]
                def ds2_gfn(t, _g=full): t["hto"][:] = _replicate(_g, tp)
                cap_c = {}
                r = _run_program_stage(Ds2PatchProg, specs, ds2_gfn, args.platform,
                                       args.atol, args.rtol, 0.01, cap_c, "ds2_out", f"ds2[c{c}]")
                ok = ok and r.passed
                # capture rank0 actual [96, 6144] for chaining to projector (CHAIN mode)
                cap.setdefault("ds2_out", {})[c] = cap_c.get(
                    "ds2_out", torch.zeros(tp, V_TOKENS_FINAL_PATCH_PAD, V_DS2_OUT, dtype=torch.bfloat16))[0]

    # ════════ Stage 7: ProjPatchProg (ds2_out -> zero-pad -> image_features) ════════
    if run_stage("proj"):
        if n_crops == 6:
            # B-axis batched: 6 crops' proj_in stacked to 576 rows in ONE launch.
            print(f"\n[patch_full] Stage 7: projector (B6 batched, vs torch vit_projector on real ds2, "
                  f"tight {args.atol:g}/{args.rtol:g})", flush=True)
            proj_in_list = []
            for c in range(6):
                # CHAIN: feed the ACTUAL ds2_out token-major [81, 6144] (real rows);
                # ISOLATION: golden [6144,9,9] -> token-major [81, 6144].
                if CHAIN and "ds2_out" in cap:
                    feat = cap["ds2_out"][c][:81]                  # [81, 6144] actual real rows
                else:
                    ds2_c = ds2_golden[c]                          # [6144, 9, 9]
                    feat = ds2_c.permute(1, 2, 0).reshape(-1, V_DS2_OUT)  # [81, 6144]
                pin = torch.zeros(V_TOKENS_FINAL_PATCH_PAD, V_DS2_OUT, dtype=torch.bfloat16)
                pin[:81] = feat
                proj_in_list.append(pin)
            proj_in_b = torch.cat(proj_in_list, dim=0)
            g_b = (proj_in_b.float() @ w_proj.float()).to(torch.bfloat16)   # [576, 4096]
            specs = [
                TensorSpec("hx", [tp, V_TOKENS_FINAL_PATCH_PAD_B, V_DS2_OUT], torch.bfloat16,
                           init_value=_replicate(proj_in_b, tp)),
                TensorSpec("hw", [tp, V_DS2_OUT, LM_HIDDEN], torch.bfloat16, init_value=_replicate(w_proj, tp), **_W),
                TensorSpec("hto", [tp, V_TOKENS_FINAL_PATCH_PAD_B, LM_HIDDEN], torch.bfloat16, is_output=True),
            ]
            def proj_gfn(t): t["hto"][:] = _replicate(g_b, tp)
            cap_c = {}
            r = _run_program_stage(ProjPatchB6Prog, specs, proj_gfn, args.platform,
                                   args.atol, args.rtol, 0.01, cap_c, "image_features", "proj[b6]")
            ok = ok and r.passed
        else:
            print(f"\n[patch_full] Stage 7: projector (per-crop, vs torch vit_projector on real ds2, "
                  f"tight {args.atol:g}/{args.rtol:g})", flush=True)
            for c in range(n_crops):
                # CHAIN: feed the ACTUAL ds2_out token-major [81, 6144] (real rows);
                # ISOLATION: golden [6144,9,9] -> token-major [81, 6144].
                if CHAIN and "ds2_out" in cap:
                    feat = cap["ds2_out"][c][:81]                  # [81, 6144] actual real rows
                else:
                    ds2_c = ds2_golden[c]                          # [6144, 9, 9]
                    # flatten to [81, 6144] token-major (matches vLLM: flatten(2).transpose)
                    feat = ds2_c.permute(1, 2, 0).reshape(-1, V_DS2_OUT)  # [81, 6144]
                proj_in = torch.zeros(V_TOKENS_FINAL_PATCH_PAD, V_DS2_OUT, dtype=torch.bfloat16)
                proj_in[:81] = feat
                g = (proj_in.float() @ w_proj.float()).to(torch.bfloat16)   # [96, 4096]
                specs = [
                    TensorSpec("hx", [tp, V_TOKENS_FINAL_PATCH_PAD, V_DS2_OUT], torch.bfloat16,
                               init_value=_replicate(proj_in, tp)),
                    TensorSpec("hw", [tp, V_DS2_OUT, LM_HIDDEN], torch.bfloat16, init_value=_replicate(w_proj, tp), **_W),
                    TensorSpec("hto", [tp, V_TOKENS_FINAL_PATCH_PAD, LM_HIDDEN], torch.bfloat16, is_output=True),
                ]
                def proj_gfn(t, _g=g): t["hto"][:] = _replicate(_g, tp)
                cap_c = {}
                r = _run_program_stage(ProjPatchProg, specs, proj_gfn, args.platform,
                                       args.atol, args.rtol, 0.01, cap_c, "image_features", f"proj[c{c}]")
                ok = ok and r.passed

    print(f"\n[patch_full] {'PASS' if ok else 'FAIL'} (patch-path pipeline, "
          f"stage={want}, mode={'sim' if is_sim else 'a2a3'})", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
