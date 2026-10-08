#!/usr/bin/env python3
"""Real multi-image B-axis test: 8 DIFFERENT images through the replicated tower.

Self-consistency check for the single-request multi-image feature: each of N
real images is preprocessed (pixel -> patch_embed im2col -> +posemb -> ln_pre)
into a tower input [2704, 1536], then:

  * reference: run the tower N times with --active 1 (image i alone) -> out_i
  * multi:     run the tower once with --active N (all N stacked)      -> out_N

The B axis is correct iff out_N[i*2704:(i+1)*2704] == out_i bit-for-bit for
every image (same kernel, same weights, same data -- only the row offset in the
stacked-2D [VB*2704, D] input changes). Any nonzero diff is a real bug: wrong
per-image row offset, cross-image attention/contamination, or a broken
``if b < active`` guard. No vLLM needed -- the single-image path is already
golden-validated, so it is the trusted reference.

Reuses a precompiled chunked build via --runtime-dir to skip the 23-49min
compile; hx_in shape is fixed at
[8, 21632, 1536] regardless of --active (vTe = VB*vT), only the `hactive`
runtime scalar changes between runs.

Usage:
  export LD_LIBRARY_PATH="<PTOAS_LIB>:$LD_LIBRARY_PATH"
  python -m tools.step3p7.run_l3_multi_image_real \
    --ckpt <CKPT> \
    --dump-root <GOLDEN_DIR> \
    --images-dir <IMAGES_DIR> \
    --num-images 8 \
    -p a2a3
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

IMAGE_MEAN = [0.48145466, 0.4578275, 0.40821073]
IMAGE_STD = [0.26862954, 0.26130258, 0.27577711]


def _load_tower_input(path: str, w_proj: torch.Tensor, posemb: torch.Tensor,
                      ln_g: torch.Tensor, ln_beta: torch.Tensor, eps: float,
                      image_size: int = 728) -> torch.Tensor:
    """raw image -> tower input [2704, 1536] BF16 (patch_embed + posemb + ln_pre)."""
    img = Image.open(path).convert("RGB").resize((image_size, image_size), Image.BILINEAR)
    x = np.asarray(img, dtype=np.float32) / 255.0                       # [728,728,3] in [0,1]
    x = (x - np.array(IMAGE_MEAN, dtype=np.float32)) / np.array(IMAGE_STD, dtype=np.float32)
    x = torch.from_numpy(x).permute(2, 0, 1).to(torch.bfloat16)          # [3,728,728]

    from models.step3p7.vision.host_im2col import patch_im2col
    cols = patch_im2col(x)                                               # [2704, 592] BF16
    pe = (cols.float() @ w_proj.float()).to(torch.bfloat16)              # [2704, 1536]
    posed = pe.float() + posemb.float()                                  # [2704, 1536] f32
    mu = posed.mean(-1, keepdim=True)
    var = posed.var(-1, keepdim=True, unbiased=False)
    ln = ((posed - mu) / torch.sqrt(var + eps) * ln_g.float() + ln_beta.float())
    return ln.to(torch.bfloat16)                                         # [2704, 1536]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--dump-root", type=Path, required=True)
    p.add_argument("--images-dir", type=Path, required=True)
    p.add_argument("--num-images", type=int, default=8)
    p.add_argument("-p", "--platform", default="a2a3", choices=["a2a3", "a2a3sim"])
    p.add_argument("--runtime-dir", type=str, default=None)
    p.add_argument("--tol", type=float, default=1e-3,
                   help="max allowed per-image abs diff (expect 0.0 for a correct B axis)")
    args = p.parse_args()

    from models.step3p7.vision.vision_config import V_WIDTH, V_TOKENS, V_GLOBAL_BATCH, V_EPS
    from models.step3p7.vision.vision_weight_loader import (
        load_step3p7_vision_weights_for_rank,
        KEY_V_PATCH_EMBED, KEY_V_POSEMB, KEY_V_LN_PRE_G, KEY_V_LN_PRE_B,
    )
    from models.step3p7.vision.vision_fwd_repl_os import Step3p7VisionReplOSChunked
    from tools.step3p7.run_l3_e2e_repl import _build_specs, _replicate
    from golden import run
    from pypto.ir.distributed_compiled_program import DistributedConfig

    D, vT, VB = V_WIDTH, V_TOKENS, V_GLOBAL_BATCH
    vTe = VB * vT
    N = args.num_images
    assert 1 <= N <= VB, f"--num-images {N} out of range [1, {VB}]"

    # ── images (pick N distinct base images) ────────────────────────────────
    imgs = sorted(glob.glob(str(args.images_dir / "v0[0-9].jpg")))
    if len(imgs) < N:
        imgs = sorted(glob.glob(str(args.images_dir / "*.jpg")))
    imgs = imgs[:N]
    if len(imgs) < N:
        sys.exit(f"need {N} images, found {len(imgs)} in {args.images_dir}")
    print(f"[multi-img] {N} images: {[os.path.basename(x) for x in imgs]}", flush=True)

    # ── preprocess (front-stage weights, tp=1) ──────────────────────────────
    print("[multi-img] loading front-stage weights (patch_embed/posemb/ln_pre)...", flush=True)
    fb = load_step3p7_vision_weights_for_rank(args.ckpt, 0, tp_world_size=1)
    w_proj = fb[KEY_V_PATCH_EMBED]           # [592, 1536]
    posemb = fb[KEY_V_POSEMB]                # [2704, 1536]
    ln_g = fb[KEY_V_LN_PRE_G].float()        # [1536]
    ln_beta = fb[KEY_V_LN_PRE_B].float()     # [1536]

    tower_inputs = []
    for im in imgs:
        t = _load_tower_input(im, w_proj, posemb, ln_g, ln_beta, V_EPS)
        assert tuple(t.shape) == (vT, D), t.shape
        tower_inputs.append(t)
    print(f"[multi-img] preprocessed {N} tower inputs, each {tuple(tower_inputs[0].shape)}", flush=True)

    # ── tower specs (weights) once; hx_in/hactive mutated per run ───────────
    print("[multi-img] building tower specs (full 47-layer replicated weights)...", flush=True)
    specs, _, _ = _build_specs(args.ckpt, args.dump_root, active=N)
    hx_spec = next(s for s in specs if s.name == "hx_in")
    ha_spec = next(s for s in specs if s.name == "hactive")

    def _stack(inputs: list[torch.Tensor], active: int) -> torch.Tensor:
        s = torch.zeros(vTe, D, dtype=torch.bfloat16)
        for i, t in enumerate(inputs[:active]):
            s[i * vT:(i + 1) * vT] = t
        return _replicate(s, 8)             # [8, vTe, D]

    cap: dict[str, torch.Tensor] = {}

    def _capture(key: str):
        def _cmp(actual, expected, **kw):
            cap[key] = actual.detach().cpu().clone()
            return (True, "capture-only")
        return _cmp

    def _gfn_zero(t):
        for v in t.values():
            if hasattr(v, "zero_"):
                v.zero_()

    dcfg = {"distributed_config": DistributedConfig(device_ids=list(range(8)), num_sub_workers=0)}
    rcfg = {"platform": args.platform}

    def _run_once(tag: str, inputs: list[torch.Tensor], active: int) -> torch.Tensor:
        hx_spec.init_value = _stack(inputs, active)
        # ScalarSpec.value must stay a 0-dim tensor (__post_init__ coerces it;
        # a plain int breaks to_python()/to_ctypes()).
        ha_spec.value = torch.tensor(active, dtype=ha_spec.dtype)
        run(
            program=Step3p7VisionReplOSChunked,
            specs=specs,
            golden_fn=_gfn_zero,
            compile_cfg=dcfg,
            runtime_cfg=rcfg,
            rtol=1.0, atol=1.0,
            compare_fn={"hto": _capture(tag)},
            runtime_dir=args.runtime_dir,
        )
        return cap[tag]

    # ── reference: active=1 per image ───────────────────────────────────────
    refs = []
    for i in range(N):
        print(f"[multi-img] reference run {i + 1}/{N} (active=1, image {i})...", flush=True)
        out = _run_once(f"ref_{i}", [tower_inputs[i]], 1)
        refs.append(out[0, :vT, :].float())   # rank 0, active rows

    # ── multi: active=N all stacked ─────────────────────────────────────────
    print(f"[multi-img] multi run (active={N}, all stacked)...", flush=True)
    multi = _run_once("multi", tower_inputs, N)[0].float()   # [vTe, D]

    # ── compare ─────────────────────────────────────────────────────────────
    worst = 0.0
    worst_i = -1
    per_img = []
    for i in range(N):
        d = (multi[i * vT:(i + 1) * vT] - refs[i]).abs().max().item()
        per_img.append(d)
        if d > worst:
            worst, worst_i = d, i
    print("\n[multi-img] per-image max abs diff (multi vs single):", flush=True)
    for i, d in enumerate(per_img):
        print(f"  image {i:2d} {os.path.basename(imgs[i]):<16} {d:.3e}", flush=True)
    print(f"  ------------------------------------------------", flush=True)
    print(f"  WORST = {worst:.3e} (image {worst_i})", flush=True)

    ok = worst <= args.tol
    print(f"\n[multi-img] {'PASS' if ok else 'FAIL'}: {N} different images "
          f"{'match' if ok else 'DIVERGE'} the single-image reference "
          f"(threshold {args.tol:.1e})", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    sys.exit(main())
