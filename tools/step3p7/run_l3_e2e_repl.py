#!/usr/bin/env python3
"""L3 golden driver for the REPLICATED (8-card, cancel-TP8) vision towers.

Runs Step3p7VisionReplOS (global, 2704 tok, online-softmax) / the patch REPL
tower (B axis,
active crops; default = flat6 VB=6 active-only, see --patch-tower) on 8 real
cards: every card runs the FULL 47-layer
transformer with FULL (non-sliced) weights — vLLM's replicated-ViT strategy,
zero cross-card sync (no tp_all_reduce). Weights are loaded ONCE with
tp_world_size=1 (full dims: KO=4608, DLP=2048, ILP=8960) and replicated to all
8 ranks. Golden compare mirrors the TP8 driver: per-layer tight (5e-3), full tower
loose (atol=1.2e-1 / rtol=8e-2 / 1% max-error-ratio) —
the repl may be numerically cleaner than TP8 (no 94-reduce
BF16 accumulation), so try tight on full tower too via --rtol/--atol.

Usage:
  export LD_LIBRARY_PATH="<PTOAS_LIB>:$LD_LIBRARY_PATH"
  python -m tools.step3p7.run_l3_e2e_repl \
    --ckpt <CKPT> \
    --dump-root <GOLDEN_DIR> \
    [-p a2a3 -d 0] [--patch --active 6] [--layer 0]
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import torch

import golden.runner as _R


def _load_dump(dump_root: Path, name: str, key: str = "hidden_states"):
    import glob
    files = glob.glob(str(dump_root / f"*_{name}.pt"))
    if not files:
        raise FileNotFoundError(name)
    return torch.load(files[0], map_location="cpu")[key]


def _replicate(t: torch.Tensor, tp: int) -> torch.Tensor:
    """Single full tensor -> [tp, ...] (identical replicated copies)."""
    return t.unsqueeze(0).expand(tp, *t.shape).contiguous()


def _build_specs(ckpt_dir: str, dump_root: Path, tp_size: int = 8,
                 layer: int | None = None, patch: bool = False, active: int = 1,
                 num_layers: int | None = None, patch_vb: int | None = None,
                 vtp: int | None = None):
    """TensorSpec list for the replicated towers' host_orch (FULL dims).

    ``num_layers`` (prefix-depth probe): stack only layers 0..K-1 and set nl=K.
    No partial-depth golden exists, so the compare FAILS by design — the probe
    is for RUNNABILITY (does the task graph drain without a ring deadlock?).
    """
    from golden import TensorSpec, ScalarSpec
    from models.step3p7.vision.vision_weight_loader import (
        load_step3p7_vision_weights_for_rank,
        KEY_V_QKV, KEY_V_QKV_B, KEY_V_O, KEY_V_O_B,
        KEY_V_FC1, KEY_V_FC1_B, KEY_V_FC2, KEY_V_FC2_B,
        KEY_V_LN1_G, KEY_V_LN1_B, KEY_V_LN2_G, KEY_V_LN2_B,
        KEY_V_LS1, KEY_V_LS2,
    )
    from models.step3p7.vision.vision_config import (
        V_WIDTH, V_LAYERS, V_TOKENS, V_TOKENS_PATCH, V_PATCH_BATCH, V_GLOBAL_BATCH,
        V_HEAD_DIM, V_MLP_HIDDEN_PAD, V_WIDTH_PAD,
    )
    from models.step3p7.vision.rope2d import build_2d_rope_tables

    # ONE full-weight bundle (tp=1): identical for all 8 replicated ranks.
    print(f"[l3repl] loading FULL weights (tp=1) for rank 0 (replicated to {tp_size})...")
    b = load_step3p7_vision_weights_for_rank(ckpt_dir, 0, tp_world_size=1)

    num_layers = 1 if layer is not None else (num_layers or V_LAYERS)
    vT = V_TOKENS_PATCH if patch else V_TOKENS
    D = V_WIDTH
    KO = 3 * D                    # full qkv out (TP8: 3*192)
    DLP = V_WIDTH_PAD             # full padded attn ctx (TP8: 256)
    ILP = V_MLP_HIDDEN_PAD        # full MLP hidden (TP8: 1152)
    HD = V_HEAD_DIM
    vWQ = V_LAYERS * D
    vWF1 = V_LAYERS * D
    vWF2 = V_LAYERS * ILP

    # Tower input
    if layer is not None:
        nm = f"patch_layer_{layer:02d}_layer_input" if patch else f"layer_{layer:02d}_layer_input"
        x_in_full = _load_dump(dump_root, nm).squeeze(0)
    else:
        nm = "patch_layer_00_layer_input" if patch else "layer_00_layer_input"
        x_in_full = _load_dump(dump_root, nm).squeeze(0)
    if patch:
        VB = patch_vb or V_PATCH_BATCH
        vTe = VB * vT
        crops = x_in_full[:active] if x_in_full.dim() == 3 else x_in_full.unsqueeze(0)[:active]
        stacked = torch.zeros(vTe, V_WIDTH, dtype=crops.dtype)
        stacked[: active * vT] = crops.reshape(active * vT, V_WIDTH)
        x_in_full = stacked.contiguous()      # [vTe, 1536]
    else:
        # GLOBAL multi-image: the tower takes a STACKED-2D [vTe=VB*vT, D] input
        # + `hactive` guard scalar. A multi-image golden (x_in_full dim==3 after
        # squeeze(0) -> B distinct images) feeds B distinct images; a single-image
        # golden (dim==2) stacks `active` copies (rest zero; the `if b<active`
        # guard skips inactive rows). active=1 == the single-image regression
        # gate; active>1 exercises the B-axis offset (identical stacked images
        # must each reproduce the single-image result).
        VB = V_GLOBAL_BATCH
        vTe = VB * vT
        if x_in_full.dim() == 3:
            crops = x_in_full[:active]                                    # multi-image golden: active distinct images
        else:
            crops = x_in_full.unsqueeze(0).expand(active, vT, V_WIDTH)     # single golden: active copies
        stacked = torch.zeros(vTe, V_WIDTH, dtype=crops.dtype)
        stacked[: active * vT] = crops.reshape(active * vT, V_WIDTH)
        x_in_full = stacked.contiguous()      # [vTe, 1536]
    T = x_in_full.shape[0]
    print(f"[l3repl] x_in: {tuple(x_in_full.shape)} (layer={layer}, patch={patch}, active={active})")

    grid = 36 if patch else 52
    cos, sin = build_2d_rope_tables(grid, grid, HD)
    # The compiled tower's RoPE tables carry the tower's internal per-image
    # STRIDE (vTp: phase coverages padded to the QTM=48 GEMM tiling,
    # ceil(vT/48)*48 — the OS tower bakes 2736 for vT=2704; see its
    # distributed_meta.json: hcos__ssa_v0 In [8, 2736, 96]). The compiled
    # program demands the hcos/hsin INPUT match that padded shape — feeding
    # vT rows fails the runtime stack-shape check ("expects (8, 2736, 96);
    # got (8, 2704, 96)"). Pad rows only ever combine
    # with pad-row inputs, so their values can never reach a compared row
    # (the tower's pad-row confinement argument); zeros are safe. Patch
    # towers tile exactly (1296 = 27*48) -> this is a no-op there.
    vTp_spec = vtp if vtp is not None else ((vT + 47) // 48) * 48
    if cos.shape[0] < vTp_spec:
        pad_rows = vTp_spec - cos.shape[0]
        cos = torch.cat([cos, torch.zeros(pad_rows, HD, dtype=cos.dtype)])
        sin = torch.cat([sin, torch.zeros(pad_rows, HD, dtype=sin.dtype)])

    # Per-layer stacking (one_layer mode: layer `layer` rotated to position 0;
    # prefix mode: layers 0..K-1 at their natural positions, rest zero).
    def stack_2d(key_fn):
        w = b[key_fn.format(L=layer if layer is not None else 0)]
        if layer is not None:
            full = torch.zeros(V_LAYERS * w.shape[0], *w.shape[1:], dtype=w.dtype)
            full[:w.shape[0]] = w
        elif num_layers is not None:
            full = torch.zeros(V_LAYERS * w.shape[0], *w.shape[1:], dtype=w.dtype)
            for L_ in range(num_layers):
                w_ = b[key_fn.format(L=L_)]
                full[L_ * w.shape[0]:(L_ + 1) * w.shape[0]] = w_
        else:
            full = torch.cat([b[key_fn.format(L=L_)] for L_ in range(V_LAYERS)], dim=0)
        return _replicate(full, tp_size)

    def stack_1d(key_fn):
        if layer is not None:
            w = b[key_fn.format(L=layer)]
            full = torch.zeros(V_LAYERS, w.shape[0], dtype=w.dtype)
            full[0] = w
        elif num_layers is not None:
            w0 = b[key_fn.format(L=0)]
            full = torch.zeros(V_LAYERS, w0.shape[0], dtype=w0.dtype)
            for L_ in range(num_layers):
                full[L_] = b[key_fn.format(L=L_)]
        else:
            full = torch.stack([b[key_fn.format(L=L_)] for L_ in range(V_LAYERS)], dim=0)
        return _replicate(full, tp_size)

    rq = stack_2d(KEY_V_QKV)                       # [tp, 47*D, 4608]
    rbq = stack_1d(KEY_V_QKV_B)                    # [tp, 47, 4608]
    ro = stack_2d(KEY_V_O)                         # [tp, 47*DLP, D]
    rbo = stack_1d(KEY_V_O_B)                      # [tp, 47, D]
    rf1 = stack_2d(KEY_V_FC1)                      # [tp, 47*D, ILP]
    rbf1 = stack_1d(KEY_V_FC1_B)                   # [tp, 47, ILP]
    rf2 = stack_2d(KEY_V_FC2)                      # [tp, 47*ILP, D]
    rbf2 = stack_1d(KEY_V_FC2_B)                   # [tp, 47, D]
    rl1g = stack_1d(KEY_V_LN1_G).float(); rl1b = stack_1d(KEY_V_LN1_B).float()
    rl2g = stack_1d(KEY_V_LN2_G).float(); rl2b = stack_1d(KEY_V_LN2_B).float()
    rl1 = stack_1d(KEY_V_LS1).float(); rl2 = stack_1d(KEY_V_LS2).float()

    cos_exp = _replicate(cos, tp_size)
    sin_exp = _replicate(sin, tp_size)

    # Weights are worker-resident stacked device tensors (uploaded once inside
    # prepare; per-dispatch bind drops from ~205ms of host tensormap
    # registration to ~2ms — measured: main tower 1146->973ms/forward,
    # patch 1723->1512ms). Per-step IO (hx_in),
    # the Out (hto) and the chunk-chaining scratch (hmid) stay host shared.
    _W = {"resident": "stacked"}
    specs = [
        TensorSpec("hx_in", [tp_size, T, D], torch.bfloat16,
                   init_value=_replicate(x_in_full, tp_size)),
        TensorSpec("hq", [tp_size, vWQ, KO], torch.bfloat16, init_value=rq, **_W),
        TensorSpec("ho", [tp_size, V_LAYERS * DLP, D], torch.bfloat16, init_value=ro, **_W),
        TensorSpec("hbq", [tp_size, V_LAYERS, KO], torch.bfloat16, init_value=rbq, **_W),
        TensorSpec("hbo", [tp_size, V_LAYERS, D], torch.bfloat16, init_value=rbo, **_W),
        TensorSpec("hf1", [tp_size, vWF1, ILP], torch.bfloat16, init_value=rf1, **_W),
        TensorSpec("hf2", [tp_size, vWF2, D], torch.bfloat16, init_value=rf2, **_W),
        TensorSpec("hbf1", [tp_size, V_LAYERS, ILP], torch.bfloat16, init_value=rbf1, **_W),
        TensorSpec("hbf2", [tp_size, V_LAYERS, D], torch.bfloat16, init_value=rbf2, **_W),
        TensorSpec("hl1g", [tp_size, V_LAYERS, D], torch.float32, init_value=rl1g, **_W),
        TensorSpec("hl1b", [tp_size, V_LAYERS, D], torch.float32, init_value=rl1b, **_W),
        TensorSpec("hl2g", [tp_size, V_LAYERS, D], torch.float32, init_value=rl2g, **_W),
        TensorSpec("hl2b", [tp_size, V_LAYERS, D], torch.float32, init_value=rl2b, **_W),
        TensorSpec("hl1", [tp_size, V_LAYERS, D], torch.float32, init_value=rl1, **_W),
        TensorSpec("hl2", [tp_size, V_LAYERS, D], torch.float32, init_value=rl2, **_W),
        TensorSpec("hcos", [tp_size, cos.shape[0], HD], torch.float32, init_value=cos_exp, **_W),
        TensorSpec("hsin", [tp_size, sin.shape[0], HD], torch.float32, init_value=sin_exp, **_W),
        ScalarSpec("hnl", torch.int32, num_layers),
        TensorSpec("hto", [tp_size, T, D], torch.bfloat16, is_output=True),
    ]
    # Scratch chaining buffer for the chunked host_orch (hmid): written by
    # chunk k, read by chunk k+1. Zeros are fine — never read before write.
    specs.append(TensorSpec("hmid", [tp_size, T, D], torch.bfloat16))
    # B-axis active-image guard scalar (kernel: `if b < hactive`). Present for
    # BOTH the global tower (multi-image) and the patch tower (multi-crop); the
    # compare masks to [0:active*vT] rows.
    specs.append(ScalarSpec("hactive", torch.int32, active))
    return specs, x_in_full, T


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--dump-root", type=Path, required=True)
    p.add_argument("--rtol", type=float, default=5e-3)
    p.add_argument("--atol", type=float, default=5e-3)
    p.add_argument("--max-error-ratio", type=float, default=0.02)
    p.add_argument("--layer", type=int, default=-1,
                   help="single-layer mode: feed layer_NN input, nl=1, golden=layer_NN out")
    p.add_argument("-p", "--platform", default="a2a3", choices=["a2a3", "a2a3sim"])
    p.add_argument("-d", "--device", type=int, default=0)
    p.add_argument("--patch", action="store_true", default=False)
    p.add_argument("--active", type=int, default=1,
                   help="active images/crops in the B axis: patch = crop count "
                        "(6=real patch path); global = stacked copies of the "
                        "single golden image (1=regression gate, >1 exercises "
                        "the multi-image B axis)")
    p.add_argument("--scope-stats", action="store_true", default=False,
                   help="enable CallConfig.enable_scope_stats (locate ring offenders)")
    p.add_argument("--l2-swimlane", action="store_true", default=False,
                   help="enable CallConfig.enable_l2_swimlane (per-task L2 perf records)")
    p.add_argument("--chunk", type=int, default=6,
                   help="patch full-depth: 0 = single full-47 dispatch; non-zero = chunked "
                        "(sentinel only — chunk size RCH is hardcoded in "
                        "vision_fwd_patch_repl_os_flat6.py)")
    p.add_argument("--num-layers", type=int, default=-1,
                   help="prefix-depth probe: stack layers 0..K-1, nl=K (golden compare "
                        "is meaningless; success = runtime drains with no ring deadlock)")
    p.add_argument("--runtime-dir", type=str, default=None,
                   help="reuse a precompiled build_output/ dir (skips compile; must match "
                        "the selected tower/chunk/active program shape)")
    p.add_argument("--no-golden", action="store_true", default=False,
                   help="perf-only: skip golden load + validation (bench timing only)")
    p.add_argument("--tower", default="os", choices=["os", "osfa", "osfaslot", "osfixpipe", "osmlpf1", "osn256", "osmdec", "qkpv"],
                   help="global-tower program: os = production online-softmax tower; "
                        "the rest (osfa, osfaslot, osfixpipe, osmlpf1, osn256, osmdec, "
                        "qkpv) are retired experiment drafts")
    p.add_argument("--patch-tower", default="flat6", choices=["flat6", "fixpipe", "v1g128", "n256", "n256f1", "f1", "flat", "n256qkv", "n512", "patchmdec"],
                   help="patch-tower program: flat6 = production VB=6 active-only flat "
                        "attention; flat = VB=16 capacity + runtime active scalar; the "
                        "rest (fixpipe, v1g128, n256, n256f1, f1, n256qkv, n512, "
                        "patchmdec) are retired experiment drafts")
    args = p.parse_args()

    layer = args.layer if args.layer >= 0 else None
    num_layers = args.num_layers if args.num_layers > 0 else None
    _os_vtp = None  # global-tower RoPE stride override (osmdec: 2816 at M=128)
    _patch_vtp = None  # patch-tower RoPE stride override (patchmdec: 1408 at MQ_G=128)

    if args.patch:
        from models.step3p7.vision.vision_config import V_TOKENS_PATCH as vT_p, V_PATCH_BATCH
        # Crop capacity VB of the selected tower: flat6 is the VB=6
        # active-only specialization (vTe=7776); the flat tower carries
        # VB=V_PATCH_BATCH=16 capacity + a runtime `active` scalar, so any
        # --active <= 16 runs unmodified (e.g. 15-crop 2560x1440 images).
        # flat6 only matches the exactly-6-crop patch path — any other count
        # falls back to flat.
        _patch_tower = args.patch_tower
        if _patch_tower == "fixpipe" and args.active != 6:
            p.error(f"--patch-tower fixpipe requires --active 6 "
                    f"(VB=6 specialization; got active={args.active})")
        if _patch_tower == "v1g128" and args.active != 6:
            p.error(f"--patch-tower v1g128 requires --active 6 "
                    f"(VB=6 specialization; got active={args.active})")
        if _patch_tower == "n256" and args.active != 6:
            p.error(f"--patch-tower n256 requires --active 6 "
                    f"(VB=6 specialization; got active={args.active})")
        if _patch_tower == "n256f1" and args.active != 6:
            p.error(f"--patch-tower n256f1 requires --active 6 "
                    f"(VB=6 specialization; got active={args.active})")
        if _patch_tower == "f1" and args.active != 6:
            p.error(f"--patch-tower f1 requires --active 6 "
                    f"(VB=6 specialization; got active={args.active})")
        if _patch_tower == "n256qkv" and args.active != 6:
            p.error(f"--patch-tower n256qkv requires --active 6 "
                    f"(VB=6 specialization; got active={args.active})")
        if _patch_tower == "n512" and args.active != 6:
            p.error(f"--patch-tower n512 requires --active 6 "
                    f"(VB=6 specialization; got active={args.active})")
        if _patch_tower == "patchmdec" and args.active != 6:
            p.error(f"--patch-tower patchmdec requires --active 6 "
                    f"(VB=6 specialization; got active={args.active})")
        if _patch_tower == "flat6" and args.active != 6:
            print(f"[l3repl] patch-tower {_patch_tower} requires --active 6; "
                  f"falling back to flat (VB={V_PATCH_BATCH}) for active={args.active}")
            _patch_tower = "flat"
        if _patch_tower == "fixpipe":
            patch_vb = 6
            from models.step3p7.vision.vision_fwd_patch_repl_os_flat6_fixpipe_draft import (
                Step3p7VisionPatchReplOSFlat6 as TowerSingle,
                Step3p7VisionPatchReplOSFlat6Chunked as TowerChunked,
            )
        elif _patch_tower == "v1g128":
            patch_vb = 6
            from models.step3p7.vision.vision_fwd_patch_repl_os_flat6_v1g128_draft import (
                Step3p7VisionPatchReplOSFlat6 as TowerSingle,
                Step3p7VisionPatchReplOSFlat6Chunked as TowerChunked,
            )
        elif _patch_tower == "n256":
            # FP_N comes from the PYPTO_FP_N env (module-level read); FP_KC
            # derives to 64 for FP_N>128.
            patch_vb = 6
            from models.step3p7.vision.vision_fwd_patch_repl_os_flat6_n256_draft import (
                Step3p7VisionPatchReplOSFlat6 as TowerSingle,
                Step3p7VisionPatchReplOSFlat6Chunked as TowerChunked,
            )
        elif _patch_tower == "n256f1":
            # n256 + promoted mlp_f1 FixPipe split; FP_N env drives both.
            patch_vb = 6
            from models.step3p7.vision.vision_fwd_patch_repl_os_flat6_n256f1_draft import (
                Step3p7VisionPatchReplOSFlat6 as TowerSingle,
                Step3p7VisionPatchReplOSFlat6Chunked as TowerChunked,
            )
        elif _patch_tower == "f1":
            patch_vb = 6
            from models.step3p7.vision.vision_fwd_patch_repl_os_flat6_f1_draft import (
                Step3p7VisionPatchReplOSFlat6 as TowerSingle,
                Step3p7VisionPatchReplOSFlat6Chunked as TowerChunked,
            )
        elif _patch_tower == "n256qkv":
            # QKN_Q comes from the PYPTO_V1G_N env (module-level read);
            # QKK_Q derives to 64 for QKN_Q>128.
            patch_vb = 6
            from models.step3p7.vision.vision_fwd_patch_repl_os_flat6_n256qkv_draft import (
                Step3p7VisionPatchReplOSFlat6 as TowerSingle,
                Step3p7VisionPatchReplOSFlat6Chunked as TowerChunked,
            )
        elif _patch_tower == "n512":
            # flat6 production + PYPTO_FP_N (v3g/v4g/v4ag ladder, elementwise
            # passes stay RES_N<=256) + PYPTO_V1G_N (v1g) env knobs; defaults
            # 256/256 = production semantics (anchor), target 512/512.
            patch_vb = 6
            from models.step3p7.vision.vision_fwd_patch_repl_os_flat6_n512_draft import (
                Step3p7VisionPatchReplOSFlat6 as TowerSingle,
                Step3p7VisionPatchReplOSFlat6Chunked as TowerChunked,
            )
        elif _patch_tower == "flat6":
            # GEMM-M-decoupling promoted into flat6: the four
            # FixPipe GEMMs run MQ_G=128, so the tower's baked RoPE/GEMM
            # stride is vTp=1408 — the driver must pad cos/sin to it.
            patch_vb = 6
            from models.step3p7.vision.vision_fwd_patch_repl_os_flat6 import (
                Step3p7VisionPatchReplOSFlat6 as TowerSingle,
                Step3p7VisionPatchReplOSFlat6Chunked as TowerChunked,
                vTp as _patch_vtp,
            )
        elif _patch_tower == "patchmdec":
            # GEMM-M-decoupling draft: the four FixPipe GEMMs
            # raise M 48->128 (MQ_G=128), elementwise + attention stay MQ=48.
            # The tower's baked RoPE/GEMM stride is vTp=1408, so the driver
            # must pad cos/sin to it.
            patch_vb = 6
            from models.step3p7.vision.vision_fwd_patch_repl_os_flat6_mdec_draft import (
                Step3p7VisionPatchReplOSFlat6 as TowerSingle,
                Step3p7VisionPatchReplOSFlat6Chunked as TowerChunked,
                vTp as _patch_vtp,
            )
        else:
            # Flat attention over the full capacity (130.4ms baseline at
            # active=6): drop-in for the flat6 program — identical host_orch
            # contract. Handles any crop count <= VB via the active scalar;
            # counts > VB are chunked by the caller (run_vision_full_patch).
            patch_vb = V_PATCH_BATCH
            from models.step3p7.vision.vision_fwd_patch_repl_os_flat import (
                Step3p7VisionPatchReplOSFlat as TowerSingle,
                Step3p7VisionPatchReplOSFlatChunked as TowerChunked,
            )
        _pfx = "patch_"
        # Chunked chained dispatch only applies to the full-depth patch run
        # (layer/probe modes keep the single-dispatch program).
        use_chunked = args.chunk > 0 and layer is None and num_layers is None
        TowerProg = TowerChunked if use_chunked else TowerSingle
        print(f"[l3repl] patch program: {'chunked x%d' % args.chunk if use_chunked else 'single-dispatch'}"
              f" (attention: {_patch_tower}, VB={patch_vb})")
    else:
        if args.tower == "osfa":
            from models.step3p7.vision.vision_fwd_repl_os_fa_draft import (
                Step3p7VisionReplOS as TowerSingle,
                Step3p7VisionReplOSChunked as TowerChunked,
            )
        elif args.tower == "osfaslot":
            # Ring-depth probe: osfa V1 form with the vfa mixed
            # kernel's auto cube<->vec pipe deepened slot_num 4 -> 8
            # (GM ring depth; the reference flash_attention_cv_split.py
            # GM-staged-slot insight).
            from models.step3p7.vision.vision_fwd_repl_os_fa_slot_draft import (
                Step3p7VisionReplOS as TowerSingle,
                Step3p7VisionReplOSChunked as TowerChunked,
            )
        elif args.tower == "osfixpipe":
            from models.step3p7.vision.vision_fwd_repl_os_fixpipe_draft import (
                Step3p7VisionReplOS as TowerSingle,
                Step3p7VisionReplOSChunked as TowerChunked,
            )
        elif args.tower == "osmlpf1":
            from models.step3p7.vision.vision_fwd_repl_os_mlpf1_draft import (
                Step3p7VisionReplOS as TowerSingle,
                Step3p7VisionReplOSChunked as TowerChunked,
            )
        elif args.tower == "osn256":
            from models.step3p7.vision.vision_fwd_repl_os_n256_draft import (
                Step3p7VisionReplOS as TowerSingle,
                Step3p7VisionReplOSChunked as TowerChunked,
            )
        elif args.tower == "osmdec":
            # GEMM-M-decoupling draft: GEMM M-tiles env-driven
            # (PYPTO_QTM_G/PYPTO_QTM1_G, default 128 -> vTp=2816). The tower's
            # baked RoPE stride is vTp, so the driver must pad cos/sin to it.
            from models.step3p7.vision.vision_fwd_repl_os_mdec_draft import (
                Step3p7VisionReplOS as TowerSingle,
                Step3p7VisionReplOSChunked as TowerChunked,
                vTp as _os_vtp,
            )
        elif args.tower == "qkpv":
            # Fused qk_pv attention draft: v2o+v2f replaced by
            # vfa (fused QK+softmax+PV, KT=128 co-resident, M=48) + vfm
            # (pure-VEC online merge), rebased on the M-decoupling production
            # (QTM_G=128, vTp=2816). Golden compare = the same gates as os.
            from models.step3p7.vision.vision_fwd_repl_os_qkpv_draft import (
                Step3p7VisionReplOS as TowerSingle,
                Step3p7VisionReplOSChunked as TowerChunked,
                vTp as _os_vtp,
            )
        else:
            from models.step3p7.vision.vision_fwd_repl_os import (
                Step3p7VisionReplOS as TowerSingle,
                Step3p7VisionReplOSChunked as TowerChunked,
                vTp as _os_vtp,
            )
        _pfx = ""
        use_chunked = args.chunk > 0 and layer is None and num_layers is None
        TowerProg = TowerChunked if use_chunked else TowerSingle
        print(f"[l3repl] tower program: {'chunked x%d' % args.chunk if use_chunked else 'single-dispatch'}"
              f" (attention: {args.tower})")
    from models.step3p7.vision.vision_config import V_LAYERS, V_TOKENS, V_GLOBAL_BATCH
    from golden import run, ratio_allclose
    from pypto.ir.distributed_compiled_program import DistributedConfig

    specs, x_in, T = _build_specs(args.ckpt, args.dump_root,
                                  layer=layer, patch=args.patch, active=args.active,
                                  num_layers=num_layers, patch_vb=patch_vb if args.patch else None,
                                  vtp=_patch_vtp if args.patch else _os_vtp)

    nl = 1 if layer is not None else (num_layers or V_LAYERS)
    if layer is not None:
        _gname = f"{_pfx}layer_{layer:02d}_layer_out"
    elif num_layers is not None and num_layers < V_LAYERS:
        # prefix-depth probe: nl=K layers from the layer-00 input must
        # reproduce the (K-1)-th layer golden, NOT the full-tower golden
        # (the old patch_tower_out compare made any K<V_LAYERS probe fail
        # unconditionally).
        _gname = f"{_pfx}layer_{num_layers - 1:02d}_layer_out"
    else:
        _gname = f"{_pfx}tower_out"
    golden_tower = None
    _active_rows = None
    if args.no_golden:
        golden_fn = None
        print(f"[l3repl] layer={layer} num_layers={nl} perf-only "
              f"patch={args.patch} active={args.active} (golden skipped)")
    else:
        golden_tower = _load_dump(args.dump_root, _gname)
        if args.patch:
            vT = vT_p
            vTe = patch_vb * vT
            crops = golden_tower[:args.active] if golden_tower.dim() == 3 else golden_tower.unsqueeze(0)[:args.active]
            g = torch.zeros(vTe, golden_tower.shape[-1], dtype=crops.dtype)
            g[: args.active * vT] = crops.reshape(args.active * vT, -1)
            golden_tower = g.contiguous()
            _active_rows = args.active * vT
        else:
            # GLOBAL: a multi-image golden (>= active distinct images) feeds them
            # directly; a single-image golden (tower_out 2D, or per-layer [1,vT,D])
            # stacks `active` copies (mirror _build_specs).
            vT = V_TOKENS
            vTe = V_GLOBAL_BATCH * vT
            flat = golden_tower.reshape(-1, golden_tower.shape[-1])   # [B*vT, D] (B=1 single, B=2 multi)
            if flat.shape[0] >= args.active * vT:
                crops = flat[: args.active * vT]                                       # multi-image golden: distinct images
            else:
                crops = flat.unsqueeze(0).expand(args.active, vT, golden_tower.shape[-1]).reshape(args.active * vT, -1)
            g = torch.zeros(vTe, golden_tower.shape[-1], dtype=crops.dtype)
            g[: args.active * vT] = crops
            golden_tower = g.contiguous()
            _active_rows = args.active * vT
        print(f"[l3repl] layer={layer} num_layers={nl} golden={_gname} "
              f"patch={args.patch} active={args.active} tol={args.atol}/{args.rtol}")

        def golden_fn(tensors):
            gt = golden_tower.reshape(-1, golden_tower.shape[-1])
            tensors["hto"][:] = gt.unsqueeze(0).expand(8, -1, -1)

    base = ratio_allclose(atol=args.atol, rtol=args.rtol, max_error_ratio=args.max_error_ratio)

    def _dump_cmp(actual, expected, **kw):
        if _active_rows is not None:
            actual = actual[:, :_active_rows, :]
            expected = expected[:, :_active_rows, :]
        return base(actual, expected, **kw)

    rcfg: dict = {"platform": args.platform}
    if args.scope_stats:
        rcfg["enable_scope_stats"] = True
    if args.l2_swimlane:
        rcfg["enable_l2_swimlane"] = True

    # ── wall-clock timing (time.perf_counter): the ONLY figure not subject to the
    #    fallback_flattened per-dispatch collapse.  run() with a runtime_dir skips
    #    compile and performs: 1 validation dispatch + BENCH_WARMUP warmup +
    #    BENCH_ROUNDS timed rounds — every one of them a FULL 12-dispatch tower.
    _R._BENCH_ROUNDS = 10
    _R._BENCH_WARMUP = 3
    _t0 = time.perf_counter()
    result = run(
        program=TowerProg,
        specs=specs,
        golden_fn=golden_fn,
        compile_cfg={"distributed_config": DistributedConfig(device_ids=list(range(8)), num_sub_workers=0)},
        runtime_cfg=rcfg,
        rtol=args.rtol,
        atol=args.atol,
        compare_fn={"hto": _dump_cmp},
        compile_only=args.platform.endswith("sim"),
        runtime_dir=args.runtime_dir,
    )
    _t1 = time.perf_counter()
    _wall_ms = (_t1 - _t0) * 1000.0
    _n_fwd = 1 + _R._BENCH_WARMUP + _R._BENCH_ROUNDS
    print(f"\n[l3repl] ===== WALL-CLOCK (time.perf_counter, full-tower forwards) =====", flush=True)
    print(f"[l3repl]   run() total = {_wall_ms:.1f} ms  "
          f"({_n_fwd} forwards = 1 validation + {_R._BENCH_WARMUP} warmup + {_R._BENCH_ROUNDS} rounds)", flush=True)
    print(f"[l3repl]   per-forward wall = {_wall_ms / _n_fwd:.2f} ms  "
          f"(UPPER bound: amortises one-time worker prepare/fork overhead)", flush=True)
    if result.bench is not None:
        print(f"[l3repl]   harness device_wall per-dispatch (fallback_flattened) = "
              f"{statistics.fmean(result.bench.device_wall_us) / 1000:.2f} ms", flush=True)

    status = "PASS" if result.passed else "FAIL"
    print(f"\n[l3repl] {status}")
    if result.error:
        print(result.error[:500])
    sys.exit(0 if result.passed else 1)


if __name__ == "__main__":
    sys.exit(main())
