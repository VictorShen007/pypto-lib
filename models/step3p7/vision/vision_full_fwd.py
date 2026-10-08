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
im2col runs in the host driver (``tools/step3p7/run_vision_full.py`` via
``host_im2col.py``); the ds1/ds2 im2cols are folded ONTO the NPU (``ds1_im2col_b`` /
``ds2_im2col_b`` tap-major gather kernels) so the host only does a cheap 1-zero-border
pad+reshape + a one-time weight permute (channel-major -> tap-major).

Two NEW padded kernels (ds1_pad/ds2_pad) are needed because the original
``conv_downsample1/2`` are static [676]/[169] with ``n=N//16`` tail bugs; leaving
the originals untouched, we write padded copies here (N=688/176, mult of 16,
no tail).

Usage (host driver orchestrates the chain)::

    python -m tools.step3p7.run_vision_full --ckpt ... --dump-root ... -p a2a3 -d 0
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


# ── NEW padded downsampler kernels (copies of conv_downsample1/2, N padded, no tail, +bias) ──
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
