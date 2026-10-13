# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Step3.7 VIT **full-pipeline** Pypto version (pixel -> image_features).

This is a SEPARATE, self-contained full vision pipeline that does NOT
modify the original ``vision_fwd.py`` (the 47-layer tower, L3-green). It chains:

  host im2col(patch) -> patch_embed -> +posemb -> ln_pre -> [tower: Step3p7Vision] ->
  device im2col(ds1) -> ds1 -> device im2col(ds2) -> ds2 -> projector -> image_features

The tower (Step3p7Vision) is REUSED from ``vision_fwd.py`` (untouched). The front/back
stages are new ``@pl.jit`` wrappers around the standalone ``@pl.jit.inline`` kernels
(patch_embed/layernorm/vit_projector + 2 NEW padded ds1/ds2 kernels). patch_embed's
im2col runs in the host driver (``models/step3p7/vision/vision_full_fwd.py`` via
``host_im2col.py``); the ds1/ds2 im2cols are folded ONTO the NPU (``ds1_im2col_b`` /
``ds2_im2col_b`` tap-major gather kernels) so the host only does a cheap 1-zero-border
pad+reshape + a one-time weight permute (channel-major -> tap-major).

Two NEW padded kernels (ds1_pad/ds2_pad) are needed because the static
[676]/[169] shapes with ``n=N//16`` have tail bugs; we write padded copies
here (N=688/176, mult of 16, no tail).

Usage (host driver orchestrates the chain)::

    python -m models.step3p7.vision.vision_full_fwd --ckpt ... --dump-root ... -p a2a3 -d 0
"""

import pypto.language as pl
import pypto.language.distributed as pld

from .vision_config import (
    V_TOKENS, V_WIDTH, V_DS1_OUT, V_DS2_OUT, V_TOKENS_FINAL_PAD, V_TOKENS_FINAL,
    V_GRID, V_GRID_DS1, V_GRID_DS2, LM_HIDDEN, TP_WORLD_SIZE, V_GLOBAL_BATCH,
    V_PATCH, V_TOKENS_FINAL_PATCH_PAD,
)
from .conv_patch import patch_embed, PE_K_PAD, PE_ROWS_PAD
# pl.inline(<body>._func) inlines the body AST into THIS module's scope, so the
# cross-module bodies' free vars must be imported here to resolve (same-module
# bodies in vision_fwd don't hit this). layernorm uses D/D_TILE/T_TILE/
# V_WIDTH_INV/V_EPS; vit_projector uses N. patch_embed's free vars are only
# its own constants (PE_ROWS_PAD/PE_K_PAD, imported above).
from .layernorm import (
    layernorm, D as LN_D, T_DYN, D, D_TILE, T_TILE, V_WIDTH_INV, V_EPS,
)
from .vit_projector import vit_projector, N

# Padded token counts (mult of 16, cube M-tile safe; host zero-pads real counts).
V_GRID_DS1_PAD = ((V_GRID_DS1 * V_GRID_DS1 + 15) // 16) * 16   # 676 -> 688
# V_TOKENS_FINAL_PAD = 176 (from vision_config, 169 -> 176)

# ── multi-image B axis (mirror vision_fwd_repl_os): VB single source ──
# Each surrounding stage folds B into its ROW axis with a per-image stride of
# that stage's padded row count (row = b*stride + i). The per-image stride for
# ds1/ds2 is the stage's own padded rows (688/176) — NOT a uniform ×B across
# stages (design doc §3.7). patch_embed's pixel input also gains the B axis.
VB = V_GLOBAL_BATCH                         # 4
PE_ROWS_B = VB * PE_ROWS_PAD                # patch_embed stacked rows (4*2880)
LN_ROWS_B = VB * V_TOKENS                   # ln_pre stacked rows (4*2704)
DS1_ROWS_B = VB * V_GRID_DS1_PAD            # ds1 stacked rows (4*688)
DS2_ROWS_B = VB * V_TOKENS_FINAL_PAD        # ds2/projector stacked rows (4*176)

# patch_embed device im2col (k14 s14 p0, global): the [3,728,728] image is
# reshaped [3,52,14,52,14] then kw-padded 14->16 on the host and flattened to
# [3, 605696] (col = oh*11648 + kh*832 + ow*16 + kw_pad) — a cheap reshape+pad,
# no F.unfold. PE_K_VALID=672 valid cols (3*14*16); K-pad [672:768] and row-pad
# [2704:2880] per batch are zeroed in-kernel (create_tensor scratch is not
# zero-initialized).
PATCH_IM2COL_FEAT_ROWS_G = 3
# ptoas requires every tile row's byte size to be 32B-aligned (BF16 -> 16 cols),
# so the 14-px kw run is padded 14->16 on the host (F.pad, NOT unfold); the K
# layout becomes [ch][kh][kw_pad] = ch*224 + kh*16 + kw_pad, w re-laid out to
# match (pad taps -> 0 weight). Matmul result unchanged.
PATCH_KW_PAD = 16                                              # 14 + 2 pad
PATCH_IM2COL_AREA_G = V_GRID * V_PATCH * V_GRID * PATCH_KW_PAD  # 52*14*52*16 = 605696
PE_K_VALID = 3 * V_PATCH * PATCH_KW_PAD                        # 672 (3*14*16)


# ── device-side im2col gather (tap-major) constants ──────────────────────────
# k3 s2 p1: out grid = (in+1)//2. F.unfold's tap offset 2*oh-1+kh (kh in [0,3))
# reaches -1 at oh=0,kh=0, so the host pads a 1-zero spatial border (grid g ->
# g+2) and the gather reads ih=2*oh+kh (always in [0, g], no bounds check).
# Tap-major K layout (im2col[r, t*C+c] = feat[ih, iw, c], t=kh*3+kw) is a
# contiguous row-copy for the device; the host permutes the conv weight
# channel-major -> tap-major ONCE (w.reshape(C,9,N).permute(1,0,2).reshape(9C,N))
# so the matmul result is unchanged (only the FP32 accumulation order shifts,
# far below the 1/128 rtol).
DS1_FEAT_GRID = V_GRID + 2                      # 54  (52 + 1 border each side)
DS1_FEAT_TOKENS = DS1_FEAT_GRID * DS1_FEAT_GRID # 2916
DS1_FEAT_ROWS_B = VB * DS1_FEAT_TOKENS
DS2_FEAT_GRID = V_GRID_DS1 + 2                  # 28  (26 + 1 border each side)
DS2_FEAT_TOKENS = DS2_FEAT_GRID * DS2_FEAT_GRID # 784
DS2_FEAT_ROWS_B = VB * DS2_FEAT_TOKENS
# Un-padded (raw) input row counts. The 1-zero spatial border is now applied
# ON-DEVICE (ds1_border_pad_b / ds2_border_pad_b) instead of the host F.pad, so
# Ds1Prog/Ds2Prog take the raw tower_out [VB*2704] / ds1_out [VB*676] directly.
DS1_RAW_ROWS_B = VB * V_TOKENS                  # 2704*VB
DS2_RAW_ROWS_B = VB * V_GRID_DS1 * V_GRID_DS1   # 676*VB


# ── NEW padded downsampler kernels (N padded to mult-of-16, no tail, +bias) ──
# spmd restructured from row-groups to N-blocks (patch_embed idiom). Old: spmd
# over row groups (ds1 43 / ds2 only 11 tasks — ds2 left 70
# of 81 cores idle) and EVERY row-group task re-read the full weight (ds1 85MB
# ×43, ds2 340MB ×11). New: spmd over N-blocks (ds1 24 / ds2 48 tasks), each
# task owns one [K,128] weight column-slice, streams it once, and loops all row
# groups serially. K-chunk width sweep (global-path medians):
#   ds1: KC=512 = 2405us, 256 = 2318us (best, -3.6%) -> 256 (13824 % 1024 != 0)
#   ds2: KC=512 = 3348us, 256 = 3320us, 1024 = 3200us (best, -4.4%) -> 1024
# N-splitting never changes an output element's K accumulation order.
# M-tile: 16 -> 32 with static tail peel (same invariant: M-tiling
# never changes a row's K accumulation order either). Rationale: each N-block
# task's [K,128] weight slice (ds1 3.5MB / ds2 7MB) cannot fit on-chip, so it
# is re-streamed from L2/GM once PER ROW GROUP — 43x (ds1) / 11x (ds2) at
# M=16, ~2x less at M=32. Weight traffic was the wall (ds1 3.65GB / 2.3ms ~=
# 1.8TB/s aggregate, at the L2 bandwidth limit); KC sweeps (3-4%) confirmed
# compute/layout was NOT the limiter. M=64 was tried first and REJECTED by
# the compiler: ds1 Vec buffer 295,680B > 184KB AIV limit — and the usage is
# IDENTICAL at KC=128, i.e. purely M-driven, so no K-chunk shrink can buy a
# bigger M-tile here.
#   ds1: 688 = 21*32 + 16   (tail M=16)
#   ds2: 176 = 5*32 + 16    (tail M=16)
# K-loop pipelining (pl.pipeline stage=2, double
# buffer): overlaps the next weight-chunk load with cube compute — ds1 global
# 1576 -> 1030us on top of M=32 (the wall was NOT pure bandwidth; load/compute
# serialization was ~35% of it). Same chunks in the same order -> K
# accumulation order unchanged.
@pl.jit.inline
def ds1_pad(
    im2col: pl.Tensor[[V_GRID_DS1_PAD, 9 * V_WIDTH], pl.BF16],
    w: pl.Tensor[[9 * V_WIDTH, V_DS1_OUT], pl.BF16],
    bias: pl.Tensor[[1, V_DS1_OUT], pl.BF16],
    out: pl.Tensor[[V_GRID_DS1_PAD, V_DS1_OUT], pl.BF16],
):
    for nb_idx in pl.spmd(24, name_hint="ds1_pad"):   # V_DS1_OUT // 128
        n0 = nb_idx * 128
        # 4-way row-tile unroll: one pipeline stage's weight chunk
        # (shared `wt`) feeds FOUR adjacent 32-row tiles, cutting the per-task
        # weight re-stream count (22x -> 7x) WITHOUT growing any M-scaled
        # tensor (four [32,128] accs, not one [128,128]) — Vec usage stays at
        # the M=32 level that M=64 exceeded. 2-way measured 1018 -> 752us
        # (-26%, compiler CSEs the shared wt into one load). Same chunks in
        # the same order per row -> K accumulation order unchanged (bit-exact).
        #   688 = 5*128 + 32 + 16  (unpaired M=32 tail at 640, M=16 tail at 672)
        for tg_idx in pl.range(5):
            tg = tg_idx * 128
            acc = pl.matmul(im2col[tg : tg + 32, 0:256], w[0:256, n0:n0 + 128], out_dtype=pl.FP32)
            acc2 = pl.matmul(im2col[tg + 32 : tg + 64, 0:256], w[0:256, n0:n0 + 128], out_dtype=pl.FP32)
            acc3 = pl.matmul(im2col[tg + 64 : tg + 96, 0:256], w[0:256, n0:n0 + 128], out_dtype=pl.FP32)
            acc4 = pl.matmul(im2col[tg + 96 : tg + 128, 0:256], w[0:256, n0:n0 + 128], out_dtype=pl.FP32)
            for kb in pl.pipeline(53, stage=2):         # chunks 1..53
                k0 = (kb + 1) * 256
                wt = w[k0:k0 + 256, n0:n0 + 128]        # loaded once, used by all four tiles
                acc = pl.matmul_acc(acc, im2col[tg : tg + 32, k0:k0 + 256], wt)
                acc2 = pl.matmul_acc(acc2, im2col[tg + 32 : tg + 64, k0:k0 + 256], wt)
                acc3 = pl.matmul_acc(acc3, im2col[tg + 64 : tg + 96, k0:k0 + 256], wt)
                acc4 = pl.matmul_acc(acc4, im2col[tg + 96 : tg + 128, k0:k0 + 256], wt)
            # add conv bias (per-output-channel) before cast: out[m,n] = acc[m,n] + bias[0,n]
            lb = pl.slice(bias, [1, 128], [0, n0])
            ones_b = pl.full([32, 128], dtype=pl.FP32, value=1.0)
            acc = pl.add(acc, pl.col_expand_mul(ones_b, pl.cast(lb, target_type=pl.FP32)))
            out[tg : tg + 32, n0 : n0 + 128] = pl.cast(acc, target_type=pl.BF16, mode="rint")
            acc2 = pl.add(acc2, pl.col_expand_mul(ones_b, pl.cast(lb, target_type=pl.FP32)))
            out[tg + 32 : tg + 64, n0 : n0 + 128] = pl.cast(acc2, target_type=pl.BF16, mode="rint")
            acc3 = pl.add(acc3, pl.col_expand_mul(ones_b, pl.cast(lb, target_type=pl.FP32)))
            out[tg + 64 : tg + 96, n0 : n0 + 128] = pl.cast(acc3, target_type=pl.BF16, mode="rint")
            acc4 = pl.add(acc4, pl.col_expand_mul(ones_b, pl.cast(lb, target_type=pl.FP32)))
            out[tg + 96 : tg + 128, n0 : n0 + 128] = pl.cast(acc4, target_type=pl.BF16, mode="rint")
        # tail rows 640..671 (unpaired M=32; separate names: DSL forbids
        # reassigning a name with a different tensor shape)
        acc_u = pl.matmul(im2col[640:672, 0:256], w[0:256, n0:n0 + 128], out_dtype=pl.FP32)
        for kb_u in pl.pipeline(53, stage=2):           # chunks 1..53
            k0_u = (kb_u + 1) * 256
            acc_u = pl.matmul_acc(acc_u, im2col[640:672, k0_u:k0_u + 256], w[k0_u:k0_u + 256, n0:n0 + 128])
        lb_u = pl.slice(bias, [1, 128], [0, n0])
        ones_u = pl.full([32, 128], dtype=pl.FP32, value=1.0)
        acc_u = pl.add(acc_u, pl.col_expand_mul(ones_u, pl.cast(lb_u, target_type=pl.FP32)))
        out[640:672, n0 : n0 + 128] = pl.cast(acc_u, target_type=pl.BF16, mode="rint")
        # tail rows 672..687 (M=16)
        acc_t = pl.matmul(im2col[672:688, 0:256], w[0:256, n0:n0 + 128], out_dtype=pl.FP32)
        for kb_t in pl.pipeline(53, stage=2):          # chunks 1..53
            k0_t = (kb_t + 1) * 256
            acc_t = pl.matmul_acc(acc_t, im2col[672:688, k0_t:k0_t + 256], w[k0_t:k0_t + 256, n0:n0 + 128])
        lb_t = pl.slice(bias, [1, 128], [0, n0])
        ones_t = pl.full([16, 128], dtype=pl.FP32, value=1.0)
        acc_t = pl.add(acc_t, pl.col_expand_mul(ones_t, pl.cast(lb_t, target_type=pl.FP32)))
        out[672:688, n0 : n0 + 128] = pl.cast(acc_t, target_type=pl.BF16, mode="rint")
    return out


@pl.jit.inline
def ds2_pad(
    im2col: pl.Tensor[[V_TOKENS_FINAL_PAD, 9 * V_DS1_OUT], pl.BF16],
    w: pl.Tensor[[9 * V_DS1_OUT, V_DS2_OUT], pl.BF16],
    bias: pl.Tensor[[1, V_DS2_OUT], pl.BF16],
    out: pl.Tensor[[V_TOKENS_FINAL_PAD, V_DS2_OUT], pl.BF16],
):
    for nb_idx in pl.spmd(48, name_hint="ds2_pad"):   # V_DS2_OUT // 128
        n0 = nb_idx * 128
        # 2-way row-tile unroll (same lever as ds1_pad's 4-way):
        # shared `wt` feeds two adjacent tiles, weight re-stream 6x -> 4x.
        #   176 = 2*64 + 32 + 16  (unpaired M=32 tail at 128, M=16 tail at 160)
        for tg_idx in pl.range(2):
            tg = tg_idx * 64
            acc = pl.matmul(im2col[tg : tg + 32, 0:512], w[0:512, n0:n0 + 128], out_dtype=pl.FP32)
            acc2 = pl.matmul(im2col[tg + 32 : tg + 64, 0:512], w[0:512, n0:n0 + 128], out_dtype=pl.FP32)
            for kb in pl.pipeline(53, stage=2):       # chunks 1..53 (KC=512)
                k0 = (kb + 1) * 512
                wt = w[k0:k0 + 512, n0:n0 + 128]      # loaded once, used by both tiles
                acc = pl.matmul_acc(acc, im2col[tg : tg + 32, k0:k0 + 512], wt)
                acc2 = pl.matmul_acc(acc2, im2col[tg + 32 : tg + 64, k0:k0 + 512], wt)
            # add conv bias (per-output-channel) before cast
            lb = pl.slice(bias, [1, 128], [0, n0])
            ones_b = pl.full([32, 128], dtype=pl.FP32, value=1.0)
            acc = pl.add(acc, pl.col_expand_mul(ones_b, pl.cast(lb, target_type=pl.FP32)))
            out[tg : tg + 32, n0 : n0 + 128] = pl.cast(acc, target_type=pl.BF16, mode="rint")
            acc2 = pl.add(acc2, pl.col_expand_mul(ones_b, pl.cast(lb, target_type=pl.FP32)))
            out[tg + 32 : tg + 64, n0 : n0 + 128] = pl.cast(acc2, target_type=pl.BF16, mode="rint")
        # tail rows 128..159 (unpaired M=32)
        acc_u = pl.matmul(im2col[128:160, 0:512], w[0:512, n0:n0 + 128], out_dtype=pl.FP32)
        for kb_u in pl.pipeline(53, stage=2):          # chunks 1..53 (KC=512)
            k0_u = (kb_u + 1) * 512
            acc_u = pl.matmul_acc(acc_u, im2col[128:160, k0_u:k0_u + 512], w[k0_u:k0_u + 512, n0:n0 + 128])
        lb_u = pl.slice(bias, [1, 128], [0, n0])
        ones_u = pl.full([32, 128], dtype=pl.FP32, value=1.0)
        acc_u = pl.add(acc_u, pl.col_expand_mul(ones_u, pl.cast(lb_u, target_type=pl.FP32)))
        out[128:160, n0 : n0 + 128] = pl.cast(acc_u, target_type=pl.BF16, mode="rint")
        # tail rows 160..175 (M=16)
        acc_t = pl.matmul(im2col[160:176, 0:512], w[0:512, n0:n0 + 128], out_dtype=pl.FP32)
        for kb_t in pl.pipeline(53, stage=2):          # chunks 1..53 (KC=512)
            k0_t = (kb_t + 1) * 512
            acc_t = pl.matmul_acc(acc_t, im2col[160:176, k0_t:k0_t + 512], w[k0_t:k0_t + 512, n0:n0 + 128])
        lb_t = pl.slice(bias, [1, 128], [0, n0])
        ones_t = pl.full([16, 128], dtype=pl.FP32, value=1.0)
        acc_t = pl.add(acc_t, pl.col_expand_mul(ones_t, pl.cast(lb_t, target_type=pl.FP32)))
        out[160:176, n0 : n0 + 128] = pl.cast(acc_t, target_type=pl.BF16, mode="rint")
    return out


# ── multi-image B-axis kernels (local copies; single-image kernels stay for the
# @pl.jit wrappers + unit tests). Each folds B into the ROW axis with a per-image
# stride of that stage's padded row count: row = b*stride + i (design doc §3.7).
# The `for b in pl.parallel(VB): if b<active:` + row-offset idiom mirrors the
# tower's ln_fwd/qkv_gemm (vision_fwd_repl_os): active rows are bit-identical to the
# single-image kernel (identical K accumulation order + cast points per row), and
# inactive rows are never touched. patch_embed's pixel input gains the B axis
# (stride PE_ROWS_PAD=2880); ds1 stride V_GRID_DS1_PAD=688; ds2/projector stride
# V_TOKENS_FINAL_PAD=176 — NOT a uniform ×B (each stage pads differently).
@pl.jit.inline
def patch_im2col_b(
    image: pl.Tensor[[PATCH_IM2COL_FEAT_ROWS_G, PATCH_IM2COL_AREA_G], pl.BF16],  # [3, 605696]
    im2col: pl.Tensor[[PE_ROWS_B, PE_K_PAD], pl.BF16],                            # [VB*2880, 768]
    active: pl.Scalar[pl.INDEX],
):
    # image is [oh][kh][ow][kw_pad] flattened (kw padded 14->16 on the host), so
    # each (ch,oh,kh,ow) tap is a contiguous 16-col run in BOTH source and target
    # -> a [1,16] chunk copy (32B-aligned, mirror of ds1/ds2's [1,256] gather).
    # The two kw_pad cols are zero on both sides, contributing nothing. The VB
    # copies share one image; inactive batches are skipped via the `active` guard.
    for b in pl.parallel(VB):
        if b < active:
            rb = b * PE_ROWS_PAD
            for ch in pl.range(3):
                for oh in pl.spmd(V_GRID):          # 52
                    for ow in pl.range(V_GRID):     # 52
                        r = rb + oh * V_GRID + ow
                        for kh in pl.range(V_PATCH):   # 14
                            k0 = ch * (V_PATCH * PATCH_KW_PAD) + kh * PATCH_KW_PAD       # ch*224 + kh*16
                            p0 = oh * (V_PATCH * V_GRID * PATCH_KW_PAD) + kh * (V_GRID * PATCH_KW_PAD) + ow * PATCH_KW_PAD
                            im2col[r : r + 1, k0 : k0 + PATCH_KW_PAD] = image[ch : ch + 1, p0 : p0 + PATCH_KW_PAD]
    # zero K-pad cols [PE_K_VALID:PE_K_PAD] (672:768 = 96 cols)
    for rb_idx in pl.spmd(PE_ROWS_B // 64):         # VB*2880 = 180*64
        rb = rb_idx * 64
        im2col[rb : rb + 64, PE_K_VALID : PE_K_PAD] = pl.full(
            [64, PE_K_PAD - PE_K_VALID], dtype=pl.BF16, value=0.0)
    # zero row-pad rows [V_TOKENS:PE_ROWS_PAD] per batch (176 rows each)
    for b in pl.parallel(VB):
        if b < active:
            rb = b * PE_ROWS_PAD
            for pr in pl.spmd(PE_ROWS_PAD - V_TOKENS):   # 176
                im2col[rb + V_TOKENS + pr : rb + V_TOKENS + pr + 1, 0:PE_K_PAD] = pl.full(
                    [1, PE_K_PAD], dtype=pl.BF16, value=0.0)
    return im2col


@pl.jit.inline
def patch_embed_b(
    x_patches: pl.Tensor[[PE_ROWS_B, PE_K_PAD], pl.BF16],
    w_proj: pl.Tensor[[PE_K_PAD, D], pl.BF16],
    posemb: pl.Tensor[[PE_ROWS_B, D], pl.BF16],
    out: pl.Tensor[[PE_ROWS_B, D], pl.BF16],
    active: pl.Scalar[pl.INDEX],
):
    for b in pl.parallel(VB):
        if b < active:
            b_off = b * PE_ROWS_PAD
            for nb_idx in pl.spmd(24, name_hint="patch_embed_b"):
                r = nb_idx // 12
                n0 = (nb_idx - r * 12) * 128
                rb = b_off + r * 1440                                    # rows/share
                for tg_idx in pl.range(15):                              # 15*96 = 1440
                    tg = rb + tg_idx * 96
                    acc = pl.matmul(x_patches[tg : tg + 32, 0:256], w_proj[0:256, n0:n0 + 128], out_dtype=pl.FP32)
                    acc2 = pl.matmul(x_patches[tg + 32 : tg + 64, 0:256], w_proj[0:256, n0:n0 + 128], out_dtype=pl.FP32)
                    acc3 = pl.matmul(x_patches[tg + 64 : tg + 96, 0:256], w_proj[0:256, n0:n0 + 128], out_dtype=pl.FP32)
                    for kb in pl.pipeline(2, stage=2):                   # chunks 1..2
                        k0 = (kb + 1) * 256
                        wt = w_proj[k0:k0 + 256, n0:n0 + 128]
                        acc = pl.matmul_acc(acc, x_patches[tg : tg + 32, k0:k0 + 256], wt)
                        acc2 = pl.matmul_acc(acc2, x_patches[tg + 32 : tg + 64, k0:k0 + 256], wt)
                        acc3 = pl.matmul_acc(acc3, x_patches[tg + 64 : tg + 96, k0:k0 + 256], wt)
                    # vLLM two-step rounding: conv1 emits BF16, then `+ posemb`
                    # is a BF16 add -> quantise acc to BF16 FIRST, then add posemb.
                    pe = pl.cast(acc, target_type=pl.BF16, mode="rint")
                    out[tg : tg + 32, n0 : n0 + 128] = pl.cast(pl.add(pl.cast(pe, pl.FP32), pl.cast(posemb[tg : tg + 32, n0 : n0 + 128], pl.FP32)), target_type=pl.BF16, mode="rint")
                    pe2 = pl.cast(acc2, target_type=pl.BF16, mode="rint")
                    out[tg + 32 : tg + 64, n0 : n0 + 128] = pl.cast(pl.add(pl.cast(pe2, pl.FP32), pl.cast(posemb[tg + 32 : tg + 64, n0 : n0 + 128], pl.FP32)), target_type=pl.BF16, mode="rint")
                    pe3 = pl.cast(acc3, target_type=pl.BF16, mode="rint")
                    out[tg + 64 : tg + 96, n0 : n0 + 128] = pl.cast(pl.add(pl.cast(pe3, pl.FP32), pl.cast(posemb[tg + 64 : tg + 96, n0 : n0 + 128], pl.FP32)), target_type=pl.BF16, mode="rint")
    return out


@pl.jit.inline
def ds1_pad_b(
    im2col: pl.Tensor[[DS1_ROWS_B, 9 * V_WIDTH], pl.BF16],
    w: pl.Tensor[[9 * V_WIDTH, V_DS1_OUT], pl.BF16],
    bias: pl.Tensor[[1, V_DS1_OUT], pl.BF16],
    out: pl.Tensor[[DS1_ROWS_B, V_DS1_OUT], pl.BF16],
    active: pl.Scalar[pl.INDEX],
):
    for b in pl.parallel(VB):
        if b < active:
            rb = b * V_GRID_DS1_PAD
            for nb_idx in pl.spmd(24, name_hint="ds1_pad_b"):           # V_DS1_OUT // 128
                n0 = nb_idx * 128
                for tg_idx in pl.range(5):
                    tg = rb + tg_idx * 128
                    acc = pl.matmul(im2col[tg : tg + 32, 0:256], w[0:256, n0:n0 + 128], out_dtype=pl.FP32)
                    acc2 = pl.matmul(im2col[tg + 32 : tg + 64, 0:256], w[0:256, n0:n0 + 128], out_dtype=pl.FP32)
                    acc3 = pl.matmul(im2col[tg + 64 : tg + 96, 0:256], w[0:256, n0:n0 + 128], out_dtype=pl.FP32)
                    acc4 = pl.matmul(im2col[tg + 96 : tg + 128, 0:256], w[0:256, n0:n0 + 128], out_dtype=pl.FP32)
                    for kb in pl.pipeline(53, stage=2):                 # chunks 1..53
                        k0 = (kb + 1) * 256
                        wt = w[k0:k0 + 256, n0:n0 + 128]
                        acc = pl.matmul_acc(acc, im2col[tg : tg + 32, k0:k0 + 256], wt)
                        acc2 = pl.matmul_acc(acc2, im2col[tg + 32 : tg + 64, k0:k0 + 256], wt)
                        acc3 = pl.matmul_acc(acc3, im2col[tg + 64 : tg + 96, k0:k0 + 256], wt)
                        acc4 = pl.matmul_acc(acc4, im2col[tg + 96 : tg + 128, k0:k0 + 256], wt)
                    lb = pl.slice(bias, [1, 128], [0, n0])
                    ones_b = pl.full([32, 128], dtype=pl.FP32, value=1.0)
                    acc = pl.add(acc, pl.col_expand_mul(ones_b, pl.cast(lb, target_type=pl.FP32)))
                    out[tg : tg + 32, n0 : n0 + 128] = pl.cast(acc, target_type=pl.BF16, mode="rint")
                    acc2 = pl.add(acc2, pl.col_expand_mul(ones_b, pl.cast(lb, target_type=pl.FP32)))
                    out[tg + 32 : tg + 64, n0 : n0 + 128] = pl.cast(acc2, target_type=pl.BF16, mode="rint")
                    acc3 = pl.add(acc3, pl.col_expand_mul(ones_b, pl.cast(lb, target_type=pl.FP32)))
                    out[tg + 64 : tg + 96, n0 : n0 + 128] = pl.cast(acc3, target_type=pl.BF16, mode="rint")
                    acc4 = pl.add(acc4, pl.col_expand_mul(ones_b, pl.cast(lb, target_type=pl.FP32)))
                    out[tg + 96 : tg + 128, n0 : n0 + 128] = pl.cast(acc4, target_type=pl.BF16, mode="rint")
                # tail rows 640..671 (unpaired M=32)
                acc_u = pl.matmul(im2col[rb + 640:rb + 672, 0:256], w[0:256, n0:n0 + 128], out_dtype=pl.FP32)
                for kb_u in pl.pipeline(53, stage=2):                   # chunks 1..53
                    k0_u = (kb_u + 1) * 256
                    acc_u = pl.matmul_acc(acc_u, im2col[rb + 640:rb + 672, k0_u:k0_u + 256], w[k0_u:k0_u + 256, n0:n0 + 128])
                lb_u = pl.slice(bias, [1, 128], [0, n0])
                ones_u = pl.full([32, 128], dtype=pl.FP32, value=1.0)
                acc_u = pl.add(acc_u, pl.col_expand_mul(ones_u, pl.cast(lb_u, target_type=pl.FP32)))
                out[rb + 640:rb + 672, n0 : n0 + 128] = pl.cast(acc_u, target_type=pl.BF16, mode="rint")
                # tail rows 672..687 (M=16)
                acc_t = pl.matmul(im2col[rb + 672:rb + 688, 0:256], w[0:256, n0:n0 + 128], out_dtype=pl.FP32)
                for kb_t in pl.pipeline(53, stage=2):                   # chunks 1..53
                    k0_t = (kb_t + 1) * 256
                    acc_t = pl.matmul_acc(acc_t, im2col[rb + 672:rb + 688, k0_t:k0_t + 256], w[k0_t:k0_t + 256, n0:n0 + 128])
                lb_t = pl.slice(bias, [1, 128], [0, n0])
                ones_t = pl.full([16, 128], dtype=pl.FP32, value=1.0)
                acc_t = pl.add(acc_t, pl.col_expand_mul(ones_t, pl.cast(lb_t, target_type=pl.FP32)))
                out[rb + 672:rb + 688, n0 : n0 + 128] = pl.cast(acc_t, target_type=pl.BF16, mode="rint")
    return out


@pl.jit.inline
def ds2_pad_b(
    im2col: pl.Tensor[[DS2_ROWS_B, 9 * V_DS1_OUT], pl.BF16],
    w: pl.Tensor[[9 * V_DS1_OUT, V_DS2_OUT], pl.BF16],
    bias: pl.Tensor[[1, V_DS2_OUT], pl.BF16],
    out: pl.Tensor[[DS2_ROWS_B, V_DS2_OUT], pl.BF16],
    active: pl.Scalar[pl.INDEX],
):
    for b in pl.parallel(VB):
        if b < active:
            rb = b * V_TOKENS_FINAL_PAD
            for nb_idx in pl.spmd(48, name_hint="ds2_pad_b"):           # V_DS2_OUT // 128
                n0 = nb_idx * 128
                for tg_idx in pl.range(2):
                    tg = rb + tg_idx * 64
                    acc = pl.matmul(im2col[tg : tg + 32, 0:512], w[0:512, n0:n0 + 128], out_dtype=pl.FP32)
                    acc2 = pl.matmul(im2col[tg + 32 : tg + 64, 0:512], w[0:512, n0:n0 + 128], out_dtype=pl.FP32)
                    for kb in pl.pipeline(53, stage=2):                 # chunks 1..53 (KC=512)
                        k0 = (kb + 1) * 512
                        wt = w[k0:k0 + 512, n0:n0 + 128]
                        acc = pl.matmul_acc(acc, im2col[tg : tg + 32, k0:k0 + 512], wt)
                        acc2 = pl.matmul_acc(acc2, im2col[tg + 32 : tg + 64, k0:k0 + 512], wt)
                    lb = pl.slice(bias, [1, 128], [0, n0])
                    ones_b = pl.full([32, 128], dtype=pl.FP32, value=1.0)
                    acc = pl.add(acc, pl.col_expand_mul(ones_b, pl.cast(lb, target_type=pl.FP32)))
                    out[tg : tg + 32, n0 : n0 + 128] = pl.cast(acc, target_type=pl.BF16, mode="rint")
                    acc2 = pl.add(acc2, pl.col_expand_mul(ones_b, pl.cast(lb, target_type=pl.FP32)))
                    out[tg + 32 : tg + 64, n0 : n0 + 128] = pl.cast(acc2, target_type=pl.BF16, mode="rint")
                # tail rows 128..159 (unpaired M=32)
                acc_u = pl.matmul(im2col[rb + 128:rb + 160, 0:512], w[0:512, n0:n0 + 128], out_dtype=pl.FP32)
                for kb_u in pl.pipeline(53, stage=2):                   # chunks 1..53 (KC=512)
                    k0_u = (kb_u + 1) * 512
                    acc_u = pl.matmul_acc(acc_u, im2col[rb + 128:rb + 160, k0_u:k0_u + 512], w[k0_u:k0_u + 512, n0:n0 + 128])
                lb_u = pl.slice(bias, [1, 128], [0, n0])
                ones_u = pl.full([32, 128], dtype=pl.FP32, value=1.0)
                acc_u = pl.add(acc_u, pl.col_expand_mul(ones_u, pl.cast(lb_u, target_type=pl.FP32)))
                out[rb + 128:rb + 160, n0 : n0 + 128] = pl.cast(acc_u, target_type=pl.BF16, mode="rint")
                # tail rows 160..175 (M=16)
                acc_t = pl.matmul(im2col[rb + 160:rb + 176, 0:512], w[0:512, n0:n0 + 128], out_dtype=pl.FP32)
                for kb_t in pl.pipeline(53, stage=2):                   # chunks 1..53 (KC=512)
                    k0_t = (kb_t + 1) * 512
                    acc_t = pl.matmul_acc(acc_t, im2col[rb + 160:rb + 176, k0_t:k0_t + 512], w[k0_t:k0_t + 512, n0:n0 + 128])
                lb_t = pl.slice(bias, [1, 128], [0, n0])
                ones_t = pl.full([16, 128], dtype=pl.FP32, value=1.0)
                acc_t = pl.add(acc_t, pl.col_expand_mul(ones_t, pl.cast(lb_t, target_type=pl.FP32)))
                out[rb + 160:rb + 176, n0 : n0 + 128] = pl.cast(acc_t, target_type=pl.BF16, mode="rint")
    return out


# ── device-side im2col gather kernels (tap-major) ────────────────────────────
# Move the host F.unfold (~800ms) onto the NPU. feat is the 1-zero-border padded
# token layout (host does the cheap pad+reshape instead of unfold); im2col is a
# device scratch (create_tensor is NOT zero-initialized, so the pad rows are
# zeroed explicitly — the matmul needs them == 0 so out == bias). Valid rows are
# written by the gather; pad rows are zeroed in a second spmd pass.
@pl.jit.inline
def ds1_im2col_b(
    feat: pl.Tensor[[DS1_FEAT_ROWS_B, V_WIDTH], pl.BF16],          # [VB*2916, 1536]
    im2col: pl.Tensor[[DS1_ROWS_B, 9 * V_WIDTH], pl.BF16],         # [VB*688, 13824]
    active: pl.Scalar[pl.INDEX],
):
    for b in pl.parallel(VB):
        if b < active:
            fb = b * DS1_FEAT_TOKENS                       # 2916
            rb = b * V_GRID_DS1_PAD                        # 688
            for oh in pl.spmd(V_GRID_DS1):                 # 26
                for ow in pl.range(V_GRID_DS1):            # 26
                    r = rb + oh * V_GRID_DS1 + ow
                    for kh in pl.range(3):
                        for kw in pl.range(3):
                            src = fb + (2 * oh + kh) * DS1_FEAT_GRID + (2 * ow + kw)
                            k0 = (kh * 3 + kw) * V_WIDTH
                            for ck in pl.range(V_WIDTH // 256):   # 6
                                c0 = ck * 256
                                im2col[r : r + 1, k0 + c0 : k0 + c0 + 256] = feat[src : src + 1, c0 : c0 + 256]
            # zero pad rows [676:688] (12 rows; scratch is not zero-initialized)
            for pr in pl.spmd(V_GRID_DS1_PAD - V_GRID_DS1 * V_GRID_DS1):   # 12
                for ck in pl.range(9 * V_WIDTH // 256):                    # 54
                    c0 = ck * 256
                    im2col[rb + V_GRID_DS1 * V_GRID_DS1 + pr : rb + V_GRID_DS1 * V_GRID_DS1 + pr + 1, c0 : c0 + 256] = pl.full([1, 256], dtype=pl.BF16, value=0.0)
    return im2col


@pl.jit.inline
def ds2_im2col_b(
    feat: pl.Tensor[[DS2_FEAT_ROWS_B, V_DS1_OUT], pl.BF16],      # [VB*784, 3072]
    im2col: pl.Tensor[[DS2_ROWS_B, 9 * V_DS1_OUT], pl.BF16],     # [VB*176, 27648]
    active: pl.Scalar[pl.INDEX],
):
    for b in pl.parallel(VB):
        if b < active:
            fb = b * DS2_FEAT_TOKENS                       # 784
            rb = b * V_TOKENS_FINAL_PAD                    # 176
            for oh in pl.spmd(V_GRID_DS2):                 # 13
                for ow in pl.range(V_GRID_DS2):            # 13
                    r = rb + oh * V_GRID_DS2 + ow
                    for kh in pl.range(3):
                        for kw in pl.range(3):
                            src = fb + (2 * oh + kh) * DS2_FEAT_GRID + (2 * ow + kw)
                            k0 = (kh * 3 + kw) * V_DS1_OUT
                            for ck in pl.range(V_DS1_OUT // 256):  # 12
                                c0 = ck * 256
                                im2col[r : r + 1, k0 + c0 : k0 + c0 + 256] = feat[src : src + 1, c0 : c0 + 256]
            # zero pad rows [169:176] (7 rows)
            for pr in pl.spmd(V_TOKENS_FINAL_PAD - V_TOKENS_FINAL):   # 7
                for ck in pl.range(9 * V_DS1_OUT // 256):             # 108
                    c0 = ck * 256
                    im2col[rb + V_TOKENS_FINAL + pr : rb + V_TOKENS_FINAL + pr + 1, c0 : c0 + 256] = pl.full([1, 256], dtype=pl.BF16, value=0.0)
    return im2col


# ── device-side 1-zero-border pad (replaces the host F.pad + d2h/h2d) ─────────
# raw is the un-padded token layout [g x g] -> padded [g+2 x g+2] with a 1-zero
# spatial border; interior token (ih,iw) lands at (ih+1,iw+1). This feeds
# dsN_im2col_b whose gather reads ih=2*oh+kh (always in [0, g+2), no bounds
# check). Border rows/cols must be explicitly zeroed (create_tensor scratch is
# NOT zero-initialized).
@pl.jit.inline
def ds1_border_pad_b(
    raw: pl.Tensor[[DS1_RAW_ROWS_B, V_WIDTH], pl.BF16],          # [VB*2704, 1536]
    padded: pl.Tensor[[DS1_FEAT_ROWS_B, V_WIDTH], pl.BF16],       # [VB*2916, 1536]
    active: pl.Scalar[pl.INDEX],
):
    for b in pl.parallel(VB):
        if b < active:
            rb = b * V_TOKENS                        # 2704
            pb = b * DS1_FEAT_TOKENS                 # 2916
            # interior gather: 52x52 -> 54x54 offset (+1,+1)
            for ih in pl.spmd(V_GRID):               # 52
                for iw in pl.range(V_GRID):          # 52
                    src = rb + ih * V_GRID + iw
                    dst = pb + (ih + 1) * DS1_FEAT_GRID + (iw + 1)
                    for ck in pl.range(V_WIDTH // 256):   # 6
                        c0 = ck * 256
                        padded[dst : dst + 1, c0 : c0 + 256] = raw[src : src + 1, c0 : c0 + 256]
            # zero top (ih=0) + bottom (ih=53) border rows
            for iw in pl.spmd(DS1_FEAT_GRID):        # 54
                for ck in pl.range(V_WIDTH // 256):
                    c0 = ck * 256
                    padded[pb + iw : pb + iw + 1, c0 : c0 + 256] = pl.full([1, 256], dtype=pl.BF16, value=0.0)
                    padded[pb + (DS1_FEAT_GRID - 1) * DS1_FEAT_GRID + iw : pb + (DS1_FEAT_GRID - 1) * DS1_FEAT_GRID + iw + 1, c0 : c0 + 256] = pl.full([1, 256], dtype=pl.BF16, value=0.0)
            # zero left (iw=0) + right (iw=53) border cols (interior rows ih=1..52)
            for ih in pl.spmd(V_GRID):               # 52
                for ck in pl.range(V_WIDTH // 256):
                    c0 = ck * 256
                    padded[pb + (ih + 1) * DS1_FEAT_GRID : pb + (ih + 1) * DS1_FEAT_GRID + 1, c0 : c0 + 256] = pl.full([1, 256], dtype=pl.BF16, value=0.0)
                    padded[pb + (ih + 1) * DS1_FEAT_GRID + (DS1_FEAT_GRID - 1) : pb + (ih + 1) * DS1_FEAT_GRID + DS1_FEAT_GRID, c0 : c0 + 256] = pl.full([1, 256], dtype=pl.BF16, value=0.0)
    return padded


@pl.jit.inline
def ds2_border_pad_b(
    raw: pl.Tensor[[DS2_RAW_ROWS_B, V_DS1_OUT], pl.BF16],      # [VB*676, 3072]
    padded: pl.Tensor[[DS2_FEAT_ROWS_B, V_DS1_OUT], pl.BF16],   # [VB*784, 3072]
    active: pl.Scalar[pl.INDEX],
):
    for b in pl.parallel(VB):
        if b < active:
            rb = b * V_GRID_DS1 * V_GRID_DS1          # 676
            pb = b * DS2_FEAT_TOKENS                  # 784
            # interior gather: 26x26 -> 28x28 offset (+1,+1)
            for ih in pl.spmd(V_GRID_DS1):            # 26
                for iw in pl.range(V_GRID_DS1):       # 26
                    src = rb + ih * V_GRID_DS1 + iw
                    dst = pb + (ih + 1) * DS2_FEAT_GRID + (iw + 1)
                    for ck in pl.range(V_DS1_OUT // 256):   # 12
                        c0 = ck * 256
                        padded[dst : dst + 1, c0 : c0 + 256] = raw[src : src + 1, c0 : c0 + 256]
            # zero top (ih=0) + bottom (ih=27) border rows
            for iw in pl.spmd(DS2_FEAT_GRID):         # 28
                for ck in pl.range(V_DS1_OUT // 256):
                    c0 = ck * 256
                    padded[pb + iw : pb + iw + 1, c0 : c0 + 256] = pl.full([1, 256], dtype=pl.BF16, value=0.0)
                    padded[pb + (DS2_FEAT_GRID - 1) * DS2_FEAT_GRID + iw : pb + (DS2_FEAT_GRID - 1) * DS2_FEAT_GRID + iw + 1, c0 : c0 + 256] = pl.full([1, 256], dtype=pl.BF16, value=0.0)
            # zero left (iw=0) + right (iw=27) border cols (interior rows ih=1..26)
            for ih in pl.spmd(V_GRID_DS1):            # 26
                for ck in pl.range(V_DS1_OUT // 256):
                    c0 = ck * 256
                    padded[pb + (ih + 1) * DS2_FEAT_GRID : pb + (ih + 1) * DS2_FEAT_GRID + 1, c0 : c0 + 256] = pl.full([1, 256], dtype=pl.BF16, value=0.0)
                    padded[pb + (ih + 1) * DS2_FEAT_GRID + (DS2_FEAT_GRID - 1) : pb + (ih + 1) * DS2_FEAT_GRID + DS2_FEAT_GRID, c0 : c0 + 256] = pl.full([1, 256], dtype=pl.BF16, value=0.0)
    return padded


@pl.jit.inline
def vit_projector_b(
    x: pl.Tensor[[DS2_ROWS_B, V_DS2_OUT], pl.BF16],
    w: pl.Tensor[[V_DS2_OUT, LM_HIDDEN], pl.BF16],
    out: pl.Tensor[[DS2_ROWS_B, LM_HIDDEN], pl.BF16],
    active: pl.Scalar[pl.INDEX],
):
    for b in pl.parallel(VB):
        if b < active:
            rb = b * V_TOKENS_FINAL_PAD
            for nb_idx in pl.spmd(64, name_hint="vit_proj_b"):          # LM_HIDDEN // 64
                n0 = nb_idx * 64
                for tg_idx in pl.range(2):
                    tg = rb + tg_idx * 64
                    acc = pl.matmul(x[tg : tg + 32, 0:512], w[0:512, n0:n0 + 64], out_dtype=pl.FP32)
                    acc2 = pl.matmul(x[tg + 32 : tg + 64, 0:512], w[0:512, n0:n0 + 64], out_dtype=pl.FP32)
                    for kb in pl.pipeline(11, stage=2):                 # chunks 1..11
                        k0 = (kb + 1) * 512
                        wt = w[k0:k0 + 512, n0:n0 + 64]
                        acc = pl.matmul_acc(acc, x[tg : tg + 32, k0:k0 + 512], wt)
                        acc2 = pl.matmul_acc(acc2, x[tg + 32 : tg + 64, k0:k0 + 512], wt)
                    out[tg : tg + 32, n0 : n0 + 64] = pl.cast(acc, target_type=pl.BF16, mode="rint")
                    out[tg + 32 : tg + 64, n0 : n0 + 64] = pl.cast(acc2, target_type=pl.BF16, mode="rint")
                # tail rows 128..159 (unpaired M=32)
                acc_u = pl.matmul(x[rb + 128:rb + 160, 0:512], w[0:512, n0:n0 + 64], out_dtype=pl.FP32)
                for kb_u in pl.pipeline(11, stage=2):                   # chunks 1..11
                    k0_u = (kb_u + 1) * 512
                    acc_u = pl.matmul_acc(acc_u, x[rb + 128:rb + 160, k0_u:k0_u + 512], w[k0_u:k0_u + 512, n0:n0 + 64])
                out[rb + 128:rb + 160, n0 : n0 + 64] = pl.cast(acc_u, target_type=pl.BF16, mode="rint")
                # tail rows 160..175 (M=16)
                acc_t = pl.matmul(x[rb + 160:rb + 176, 0:512], w[0:512, n0:n0 + 64], out_dtype=pl.FP32)
                for kb_t in pl.pipeline(11, stage=2):                   # chunks 1..11
                    k0_t = (kb_t + 1) * 512
                    acc_t = pl.matmul_acc(acc_t, x[rb + 160:rb + 176, k0_t:k0_t + 512], w[k0_t:k0_t + 512, n0:n0 + 64])
                out[rb + 160:rb + 176, n0 : n0 + 64] = pl.cast(acc_t, target_type=pl.BF16, mode="rint")
    return out


# ── @pl.jit wrappers (cross-module inline, L1-verified pattern) ───────────────
@pl.jit
def patch_embed_full(
    x: pl.Tensor[[PE_ROWS_PAD, PE_K_PAD], pl.BF16],
    w: pl.Tensor[[PE_K_PAD, V_WIDTH], pl.BF16],
    out: pl.Out[pl.Tensor[[PE_ROWS_PAD, V_WIDTH], pl.BF16]],
):
    out = patch_embed(x, w, out)
    return out


@pl.jit
def ln_pre_full(
    x: pl.Tensor[[T_DYN, LN_D], pl.BF16],
    gamma: pl.Tensor[[LN_D], pl.FP32],
    beta: pl.Tensor[[LN_D], pl.FP32],
    out: pl.Out[pl.Tensor[[T_DYN, LN_D], pl.BF16]],
):
    x.bind_dynamic(0, T_DYN)
    out.bind_dynamic(0, T_DYN)
    out = layernorm(x, gamma, beta, out)
    return out


@pl.jit
def ds1_full(
    im2col: pl.Tensor[[V_GRID_DS1_PAD, 9 * V_WIDTH], pl.BF16],
    w: pl.Tensor[[9 * V_WIDTH, V_DS1_OUT], pl.BF16],
    bias: pl.Tensor[[1, V_DS1_OUT], pl.BF16],
    out: pl.Out[pl.Tensor[[V_GRID_DS1_PAD, V_DS1_OUT], pl.BF16]],
):
    out = ds1_pad(im2col, w, bias, out)
    return out


@pl.jit
def ds2_full(
    im2col: pl.Tensor[[V_TOKENS_FINAL_PAD, 9 * V_DS1_OUT], pl.BF16],
    w: pl.Tensor[[9 * V_DS1_OUT, V_DS2_OUT], pl.BF16],
    bias: pl.Tensor[[1, V_DS2_OUT], pl.BF16],
    out: pl.Out[pl.Tensor[[V_TOKENS_FINAL_PAD, V_DS2_OUT], pl.BF16]],
):
    out = ds2_pad(im2col, w, bias, out)
    return out


@pl.jit
def proj_full(
    x: pl.Tensor[[V_TOKENS_FINAL_PAD, V_DS2_OUT], pl.BF16],
    w: pl.Tensor[[V_DS2_OUT, LM_HIDDEN], pl.BF16],
    out: pl.Out[pl.Tensor[[V_TOKENS_FINAL_PAD, LM_HIDDEN], pl.BF16]],
):
    out = vit_projector(x, w, out)
    return out


# ── 8-card REPLICATED @pl.program stages (B' fix) ───────────────────────────
# Chain these in one process with NO @pl.jit launchers to kill the SIGSEGV: the
# simpler-runtime @pl.jit worker staying alive was what collided with a later
# @pl.program's 8-card hierarchical init (two 8-card @pl.program launches in one
# process both PASS, zero SIGSEGV; golden.run
# tears down chip workers between @pl.program launches). Each stage wraps an
# existing @pl.jit.inline body via pl.inline(body._func), mirroring vision_fwd
# ._build: per_rank (plain Orchestration — NOT inline_orchestration; the tower's
# per_rank is the same and calls pl.inline bodies directly at L240/L252) wraps the
# inline body; host_orch (HOST/Orchestrator) loops pl.range(pld.world_size()) and
# dispatches per_rank(..., device=r). inline_orchestration is only for leaf
# helpers (tower v_qkv/v_attn) called FROM per_rank — a helper marked
# inline_orchestration that host_orch calls directly is an unsupported caller
# (inline_functions_pass: "Submit uses after expansion"). Replicated = every rank
# collective, no window buffer) — matches vLLM replicated parallelism for these
# ops. The 8x redundant compute is the known projector gap (deferred phase-2).

_pe_in = pl.inline(patch_embed._func)
_ln_in = pl.inline(layernorm._func)
_ds1_in = pl.inline(ds1_pad._func)
_ds2_in = pl.inline(ds2_pad._func)
_proj_in = pl.inline(vit_projector._func)
# B-axis variants (local, mirror the tower's `for b in pl.parallel(VB)` guard).
_pe_in_b = pl.inline(patch_embed_b._func)
_ds1_in_b = pl.inline(ds1_pad_b._func)
_ds2_in_b = pl.inline(ds2_pad_b._func)
_proj_in_b = pl.inline(vit_projector_b._func)
# device-side im2col gather (folded into Ds1Prog/Ds2Prog before the matmul)
_ds1_im2col_in_b = pl.inline(ds1_im2col_b._func)
_ds2_im2col_in_b = pl.inline(ds2_im2col_b._func)
# device-side 1-zero-border pad (folded into Ds1Prog/Ds2Prog before the im2col)
_ds1_border_pad_in_b = pl.inline(ds1_border_pad_b._func)
_ds2_border_pad_in_b = pl.inline(ds2_border_pad_b._func)
# patch_embed device im2col gather (folded into PatchEmbedProg before the matmul)
_patch_im2col_b_in = pl.inline(patch_im2col_b._func)


def _build_patch_embed(tp_size: int = TP_WORLD_SIZE):
    @pl.program
    class PatchEmbedProg:
        @pl.function(type=pl.FunctionType.Orchestration)
        def per_rank(self,
            image: pl.Tensor[[PATCH_IM2COL_FEAT_ROWS_G, PATCH_IM2COL_AREA_G], pl.BF16],
            w: pl.Tensor[[PE_K_PAD, V_WIDTH], pl.BF16],
            posemb: pl.Tensor[[PE_ROWS_B, V_WIDTH], pl.BF16],
            out: pl.Out[pl.Tensor[[PE_ROWS_B, V_WIDTH], pl.BF16]],
            active: pl.Scalar[pl.INDEX],
        ) -> pl.Tensor[[PE_ROWS_B, V_WIDTH], pl.BF16]:
            im2col = pl.create_tensor([PE_ROWS_B, PE_K_PAD], dtype=pl.BF16)
            im2col = _patch_im2col_b_in(image, im2col, active)
            out = _pe_in_b(im2col, w, posemb, out, active); return out

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self,
            himage: pl.Tensor[[tp_size, PATCH_IM2COL_FEAT_ROWS_G, PATCH_IM2COL_AREA_G], pl.BF16],
            hw: pl.Tensor[[tp_size, PE_K_PAD, V_WIDTH], pl.BF16],
            hp: pl.Tensor[[tp_size, PE_ROWS_B, V_WIDTH], pl.BF16],
            hto: pl.Out[pl.Tensor[[tp_size, PE_ROWS_B, V_WIDTH], pl.BF16]],
            hactive: pl.Scalar[pl.INDEX],
        ):
            for r in pl.range(pld.world_size()):
                self.per_rank(himage[r], hw[r], hp[r], hto[r], hactive, device=r)
    return PatchEmbedProg


def _build_ln_pre(tp_size: int = TP_WORLD_SIZE):
    @pl.program
    class LnPreProg:
        # ln_pre is a pure shape change: the shared row-parallel `layernorm`
        # (T_DYN dynamic row dim) already normalizes each row independently, so a
        # STACKED-2D [LN_ROWS_B, D] input needs NO kernel change and active rows
        # stay bit-identical (design doc §3.7). No `active` guard: inactive rows
        # are host-zero-padded and normalize to a finite `beta` (compare masks
        # them out). LN_ROWS_B = VB*2704 = 10816, divisible by T_TILE=8.
        @pl.function(type=pl.FunctionType.Orchestration)
        def per_rank(self,
            x: pl.Tensor[[LN_ROWS_B, LN_D], pl.BF16],
            gamma: pl.Tensor[[LN_D], pl.FP32],
            beta: pl.Tensor[[LN_D], pl.FP32],
            out: pl.Out[pl.Tensor[[LN_ROWS_B, LN_D], pl.BF16]],
        ) -> pl.Tensor[[LN_ROWS_B, LN_D], pl.BF16]:
            out = _ln_in(x, gamma, beta, out); return out

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self,
            hx: pl.Tensor[[tp_size, LN_ROWS_B, LN_D], pl.BF16],
            hg: pl.Tensor[[tp_size, LN_D], pl.FP32],
            hb: pl.Tensor[[tp_size, LN_D], pl.FP32],
            hto: pl.Out[pl.Tensor[[tp_size, LN_ROWS_B, LN_D], pl.BF16]],
        ):
            for r in pl.range(pld.world_size()):
                self.per_rank(hx[r], hg[r], hb[r], hto[r], device=r)
    return LnPreProg


def _build_ds1(tp_size: int = TP_WORLD_SIZE):
    @pl.program
    class Ds1Prog:
        @pl.function(type=pl.FunctionType.Orchestration)
        def per_rank(self,
            feat: pl.Tensor[[DS1_RAW_ROWS_B, V_WIDTH], pl.BF16],
            w: pl.Tensor[[9 * V_WIDTH, V_DS1_OUT], pl.BF16],
            bias: pl.Tensor[[1, V_DS1_OUT], pl.BF16],
            out: pl.Out[pl.Tensor[[DS1_ROWS_B, V_DS1_OUT], pl.BF16]],
            active: pl.Scalar[pl.INDEX],
        ) -> pl.Tensor[[DS1_ROWS_B, V_DS1_OUT], pl.BF16]:
            feat_pad = pl.create_tensor([DS1_FEAT_ROWS_B, V_WIDTH], dtype=pl.BF16)
            feat_pad = _ds1_border_pad_in_b(feat, feat_pad, active)
            im2col = pl.create_tensor([DS1_ROWS_B, 9 * V_WIDTH], dtype=pl.BF16)
            im2col = _ds1_im2col_in_b(feat_pad, im2col, active)
            out = _ds1_in_b(im2col, w, bias, out, active)
            return out

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self,
            hfeat: pl.Tensor[[tp_size, DS1_RAW_ROWS_B, V_WIDTH], pl.BF16],
            hw: pl.Tensor[[tp_size, 9 * V_WIDTH, V_DS1_OUT], pl.BF16],
            hbias: pl.Tensor[[tp_size, 1, V_DS1_OUT], pl.BF16],
            hto: pl.Out[pl.Tensor[[tp_size, DS1_ROWS_B, V_DS1_OUT], pl.BF16]],
            hactive: pl.Scalar[pl.INDEX],
        ):
            for r in pl.range(pld.world_size()):
                self.per_rank(hfeat[r], hw[r], hbias[r], hto[r], hactive, device=r)
    return Ds1Prog


def _build_ds2(tp_size: int = TP_WORLD_SIZE):
    @pl.program
    class Ds2Prog:
        @pl.function(type=pl.FunctionType.Orchestration)
        def per_rank(self,
            feat: pl.Tensor[[DS2_RAW_ROWS_B, V_DS1_OUT], pl.BF16],
            w: pl.Tensor[[9 * V_DS1_OUT, V_DS2_OUT], pl.BF16],
            bias: pl.Tensor[[1, V_DS2_OUT], pl.BF16],
            out: pl.Out[pl.Tensor[[DS2_ROWS_B, V_DS2_OUT], pl.BF16]],
            active: pl.Scalar[pl.INDEX],
        ) -> pl.Tensor[[DS2_ROWS_B, V_DS2_OUT], pl.BF16]:
            feat_pad = pl.create_tensor([DS2_FEAT_ROWS_B, V_DS1_OUT], dtype=pl.BF16)
            feat_pad = _ds2_border_pad_in_b(feat, feat_pad, active)
            im2col = pl.create_tensor([DS2_ROWS_B, 9 * V_DS1_OUT], dtype=pl.BF16)
            im2col = _ds2_im2col_in_b(feat_pad, im2col, active)
            out = _ds2_in_b(im2col, w, bias, out, active)
            return out

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self,
            hfeat: pl.Tensor[[tp_size, DS2_RAW_ROWS_B, V_DS1_OUT], pl.BF16],
            hw: pl.Tensor[[tp_size, 9 * V_DS1_OUT, V_DS2_OUT], pl.BF16],
            hbias: pl.Tensor[[tp_size, 1, V_DS2_OUT], pl.BF16],
            hto: pl.Out[pl.Tensor[[tp_size, DS2_ROWS_B, V_DS2_OUT], pl.BF16]],
            hactive: pl.Scalar[pl.INDEX],
        ):
            for r in pl.range(pld.world_size()):
                self.per_rank(hfeat[r], hw[r], hbias[r], hto[r], hactive, device=r)
    return Ds2Prog


def _build_proj(tp_size: int = TP_WORLD_SIZE):
    @pl.program
    class ProjProg:
        @pl.function(type=pl.FunctionType.Orchestration)
        def per_rank(self,
            x: pl.Tensor[[DS2_ROWS_B, V_DS2_OUT], pl.BF16],
            w: pl.Tensor[[V_DS2_OUT, LM_HIDDEN], pl.BF16],
            out: pl.Out[pl.Tensor[[DS2_ROWS_B, LM_HIDDEN], pl.BF16]],
            active: pl.Scalar[pl.INDEX],
        ) -> pl.Tensor[[DS2_ROWS_B, LM_HIDDEN], pl.BF16]:
            out = _proj_in_b(x, w, out, active); return out

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self,
            hx: pl.Tensor[[tp_size, DS2_ROWS_B, V_DS2_OUT], pl.BF16],
            hw: pl.Tensor[[tp_size, V_DS2_OUT, LM_HIDDEN], pl.BF16],
            hto: pl.Out[pl.Tensor[[tp_size, DS2_ROWS_B, LM_HIDDEN], pl.BF16]],
            hactive: pl.Scalar[pl.INDEX],
        ):
            for r in pl.range(pld.world_size()):
                self.per_rank(hx[r], hw[r], hto[r], hactive, device=r)
    return ProjProg


def _build_allgather_rows(rows: int, cols: int, group_size: int = TP_WORLD_SIZE):
    """Row-axis (dim=0) all-gather ``@pl.program`` — the VIT DP merge primitive.

    Each rank holds its own ``[rows, cols]`` shard (its ⌈N/8⌉ images' final
    image_features); after the collective every rank holds the full
    ``[group_size * rows, cols]`` concatenation (all ranks' shards stacked along
    dim 0 in rank order). Pull-side single-step AtomicAdd/Ge barrier over a
    staged ``tw`` window + a ``sw`` signal window — the same self-method InCore
    pattern the TP8 tower's ``tp_all_reduce`` uses (``AtomicAdd`` notify for the
    cross-rank sync + ``Ge`` wait + tail reset; ``Set`` is the scalar *data*
    publish op, not a barrier). The copy is tiled ``[RT, CT]`` to stay
    within on-chip tile limits (cf. im2col's 256-col / projector's 512-col /
    tower's QT=16-row tiles).
    """
    FULL_ROWS = group_size * rows
    RT = 16             # row tile (rows must be a multiple of 16)
    CT = 512            # column tile (cols must be a multiple of 512)
    if rows % RT != 0 or cols % CT != 0:
        raise ValueError(
            f"_build_allgather_rows: rows={rows} must be a multiple of {RT} "
            f"and cols={cols} a multiple of {CT}"
        )
    n_rt = rows // RT
    n_ct = cols // CT

    @pl.program
    class AllGatherRowsProg:
        @pl.function(type=pl.FunctionType.InCore)
        def all_gather_rows(self,
            local: pl.Tensor[[rows, cols], pl.BF16],
            tw: pld.DistributedTensor[[FULL_ROWS, cols], pl.BF16],
            sw: pld.DistributedTensor[[group_size, 1], pl.INT32],
            mr: pl.Scalar[pl.INT32],
            out: pl.Tensor[[FULL_ROWS, cols], pl.BF16],
        ) -> pl.Tensor[[FULL_ROWS, cols], pl.BF16]:
            # 1) Stage own shard into tw[own slot] (peers pull from here) and
            #    copy it into out[own slot] (own output).
            for i in pl.range(n_rt):
                for c in pl.range(n_ct):
                    st = pl.load(local, [i * RT, c * CT], [RT, CT])
                    pl.store(st, [mr * rows + i * RT, c * CT], tw)
                    ot = pl.load(local, [i * RT, c * CT], [RT, CT])
                    pl.store(ot, [mr * rows + i * RT, c * CT], out)
            # 2) Phase-1 barrier: AtomicAdd(1) notify on our own signal cell,
            #    then Ge(1) wait on every peer's cell — all ranks have published
            #    their shard into tw. AtomicAdd (not Set) is the cross-rank SYNC
            #    primitive; Set is the scalar *data*-publish op (single-writer
            #    per cell, e.g. MoE route-table scatter), not a barrier. Mirrors
            #    the TP8 tower's tp_all_reduce.
            for peer in pl.parallel(group_size):
                if peer != mr:
                    pld.system.notify(target=sw, peer=peer, offsets=[mr, 0],
                                      value=1, op=pld.NotifyOp.AtomicAdd)
            for src in pl.parallel(group_size):
                if src != mr:
                    pld.system.wait(signal=sw, offsets=[src, 0], expected=1,
                                    cmp=pld.WaitCmp.Ge)
            # 3) Pull each peer's staged shard into out[peer slot].
            for peer in pl.parallel(group_size):
                if peer != mr:
                    for i in pl.range(n_rt):
                        for c in pl.range(n_ct):
                            recv = pld.tile.remote_load(
                                tw, peer=peer,
                                offsets=[peer * rows + i * RT, c * CT],
                                shape=[RT, CT],
                            )
                            pl.store(recv, [peer * rows + i * RT, c * CT], out)
            # 4) Phase-2 quiescence barrier: every rank has finished pulling, so
            #    the tail reset below can no longer clobber a cell another rank
            #    is still waiting on (a single Ge(1) barrier is not enough — rank
            #    A would pass Ge(1) on cell B as soon as B *publishes*, then reset
            #    B's cell while B is still pulling A's shard). Mirrors the tower's
            #    trailing Ge(n) barrier before its reset.
            for peer in pl.parallel(group_size):
                if peer != mr:
                    pld.system.notify(target=sw, peer=peer, offsets=[mr, 0],
                                      value=1, op=pld.NotifyOp.AtomicAdd)
            for src in pl.parallel(group_size):
                if src != mr:
                    pld.system.wait(signal=sw, offsets=[src, 0], expected=2,
                                    cmp=pld.WaitCmp.Ge)
            # 5) Tail reset: AtomicAdd counters don't self-clear, so zero the
            #    whole column for the next dispatch (reuse-safe), matching the
            #    tower's tp_all_reduce tail reset.
            _zero = pl.cast(0, pl.INT32)
            for _clr in pl.range(group_size):
                pl.write(sw, [_clr, 0], _zero)
            return out

        @pl.function(type=pl.FunctionType.Orchestration)
        def per_rank(self,
            local: pl.Tensor[[rows, cols], pl.BF16],
            tw: pld.DistributedTensor[[FULL_ROWS, cols], pl.BF16],
            sw: pld.DistributedTensor[[group_size, 1], pl.INT32],
            out: pl.Out[pl.Tensor[[FULL_ROWS, cols], pl.BF16]],
            mr: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[FULL_ROWS, cols], pl.BF16]:
            out = self.all_gather_rows(local, tw, sw, mr, out); return out

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self,
            hlocal: pl.Tensor[[group_size, rows, cols], pl.BF16],
            hto: pl.Out[pl.Tensor[[group_size, FULL_ROWS, cols], pl.BF16]],
        ):
            tb = pld.alloc_window_buffer(FULL_ROWS * cols * 2)
            sb = pld.alloc_window_buffer(group_size * 4)
            for r in pl.range(pld.world_size()):
                t = pld.window(tb, [FULL_ROWS, cols], dtype=pl.BF16)
                s = pld.window(sb, [group_size, 1], dtype=pl.INT32)
                self.per_rank(hlocal[r], t, s, hto[r], r, device=r)
    return AllGatherRowsProg


PatchEmbedProg = _build_patch_embed(TP_WORLD_SIZE)
LnPreProg = _build_ln_pre(TP_WORLD_SIZE)
Ds1Prog = _build_ds1(TP_WORLD_SIZE)
Ds2Prog = _build_ds2(TP_WORLD_SIZE)
ProjProg = _build_proj(TP_WORLD_SIZE)
AllGatherRowsProg = _build_allgather_rows(V_TOKENS_FINAL_PAD, LM_HIDDEN)
# Patch-path DP merge: per-rank shard = 6 crops' projector output, padded to
# 6 * V_TOKENS_FINAL_PATCH_PAD (96) = 576 rows (the B6 ProjPatchB6Prog output).
# Same builder, different row count (576 % 16 == 0, 4096 % 512 == 0).
AllGatherRowsPatchProg = _build_allgather_rows(6 * V_TOKENS_FINAL_PATCH_PAD, LM_HIDDEN)


# Reuse the 47-layer tower from vision_fwd (untouched, L3-green).
from .vision_fwd import Step3p7Vision  # noqa: E402  (TP=8 @pl.program)


# ============================================================================
# E2E run driver (merged from models/step3p7/run_vision_full.py).
# ============================================================================

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
    p = argparse.ArgumentParser(description="Step3.7 VIT full-pipeline driver (pixel -> image_features), 8-card vs vLLM golden.")
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
    from models.step3p7.vision.vision_fwd_repl_os import _build_specs

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
        TensorSpec("hw", [tp, V_DS2_OUT, LM_HIDDEN], torch.bfloat16, init_value=_replicate(w_proj, tp), **_W),
        TensorSpec("hto", [tp, DS2_ROWS_B, LM_HIDDEN], torch.bfloat16, is_output=True),
        ScalarSpec("hactive", torch.int32, active),
    ]
    def proj_gfn(t):
        full = torch.zeros(V_TOKENS_FINAL_PAD, LM_HIDDEN, dtype=torch.bfloat16)
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
        ag_cols = LM_HIDDEN
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
