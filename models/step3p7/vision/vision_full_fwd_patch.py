# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS PROGRAM IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See the License in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Step3.7 VIT **patch-path** front/back stages (pixel -> patch_tower_out -> image_features).

The patch path (6 x 504^2 local crops) reuses the SAME conv/ln/projector
weights as the global path (one PerceptionEncoder; vLLM ``vision_encoder.py``
``forward`` runs conv1 -> +interp(posemb) -> ln_pre -> transformer for each
crop). The 47-layer patch TOWER is in ``vision_fwd_patch.py``
(``Step3p7VisionPatch``); this module holds the front (patch_embed/ln_pre) and
back (ds1/ds2/projector) stages that chain around it.

Design — per-crop host-loop, single-sequence replicated @pl.programs:
  Each stage is an 8-card replicated @pl.program (mirror of the global
  ``vision_full_fwd.py`` stages) sized to ONE crop's token count
  (V_TOKENS_PATCH=1296 front; V_GRID_DS1_PATCH^2=324 / V_TOKENS_FINAL_PATCH=81
  back). The host driver loops the 6 crops, running one crop per launch, and
  stacks the per-crop outputs. This mirrors the proven global full-pipeline
  EXACTLY (same flat single-sequence spmd structure that compiles on a2a3),
  only the token dim differs; lowest risk. B-axis batching (all 6 crops in one
  launch, like the tower) is a deferred perf optimization — these stages are
  tiny (<ms compute) vs the 46s tower, so 6x launches are negligible.

  posemb is FOLDED ONTO THE NPU: the host only F.interpolates the global
  [2704,1536] posemb to [1296,1536] (bilinear align_corners=False, mirroring
  vLLM ``sample_abs_posemb`` vision_encoder.py:411) and feeds it as a kernel
  input; the kernel quantises the conv1 acc to BF16 then adds posemb in a second
  BF16 cast — bit-matching vLLM's two-step `x + interp(posemb)`.

Token-count pad (cube M-tile 16-align), mirror of global V_TOKENS_FINAL_PAD:
  - front patch_embed/ln_pre: 1296 % 16 == 0 -> no pad.
  - ds1: 324 -> 336 (V_GRID_DS1_PATCH_PAD).
  - ds2/projector: 81 -> 96 (V_TOKENS_FINAL_PATCH_PAD).

Usage (host driver orchestrates the chain)::

    python -m tools.step3p7.run_vision_full_patch --ckpt ... --dump-root ... -p a2a3 -d 0
"""

import pypto.language as pl
import pypto.language.distributed as pld

from .vision_config import (
    V_TOKENS_PATCH, V_WIDTH, V_DS1_OUT, V_DS2_OUT, LM_HIDDEN,
    V_TOKENS_FINAL_PATCH_PAD, V_GRID_DS1_PATCH, TP_WORLD_SIZE,
    V_GRID_PATCH, V_GRID_DS2_PATCH, V_TOKENS_FINAL_PATCH,
    V_PATCH, V_IMAGE_PATCH,
)
from .conv_patch import PATCH_FEAT, PATCH_FEAT_PAD   # 588 -> 592 (16-align)
# pl.inline(<body>._func) inlines the body AST into THIS module's scope; the
# cross-module body's free vars must be imported here to resolve. layernorm
# uses D/D_TILE/T_TILE/V_WIDTH_INV/V_EPS (and T_DYN in its signature annotation).
from .layernorm import (
    layernorm, layernorm_b6, D as LN_D, T_DYN, D, D_TILE, T_TILE,
    V_WIDTH_INV, V_EPS, LN_T_TILE_B6, LN_D_TILE_B6,
)

# Patch-path token-count pad (mult of 16, cube M-tile safe; host zero-pads).
V_GRID_DS1_PATCH_PAD = ((V_GRID_DS1_PATCH * V_GRID_DS1_PATCH + 15) // 16) * 16   # 324 -> 336


# ── patch-path padded downsampler kernels (copies of ds1_pad/ds2_pad, patch grid) ──
# spmd restructured from row-groups to N-blocks (patch_embed idiom).
# Old row-group spmd gave ds1 only 21 / ds2 only 6 tasks (ds2 left 75
# of 81 cores idle) and every task re-read the full 85MB/340MB weight.
# K-chunk width sweep (per-crop patch medians):
#   ds1: KC=512 = 1213us, 256 = 1164us (best, -4.0%) -> 256 (13824 % 1024 != 0)
#   ds2: KC=512 = 1892us, 256 = 1893us, 1024 = 1822us (best, -3.8%) -> 1024
# M-tile: 16 -> 32 with static tail peel (mirror of the global
# ds1_pad/ds2_pad; M-tiling never changes a row's K accumulation order, and the
# per-task [K,128] weight slice re-stream count drops from 21x/6x to 11x/3x;
# M=64 is rejected by the compiler on the global path — Vec buffer > 184KB):
#   ds1: 336 = 10*32 + 16   (tail M=16)
#   ds2: 96  = 3*32         (no tail)
@pl.jit.inline
def ds1_patch(
    im2col: pl.Tensor[[V_GRID_DS1_PATCH_PAD, 9 * V_WIDTH], pl.BF16],
    w: pl.Tensor[[9 * V_WIDTH, V_DS1_OUT], pl.BF16],
    bias: pl.Tensor[[1, V_DS1_OUT], pl.BF16],
    out: pl.Tensor[[V_GRID_DS1_PATCH_PAD, V_DS1_OUT], pl.BF16],
):
    for nb_idx in pl.spmd(24, name_hint="ds1_patch"):  # V_DS1_OUT // 128
        n0 = nb_idx * 128
        # 4-way row-tile unroll (mirror of global ds1_pad; shared
        # `wt` feeds four adjacent 32-row tiles, weight re-stream 11x -> 5x;
        # M-scaled tensors stay at M=32 shapes, same K order -> bit-exact).
        #   336 = 2*128 + 2*32 + 16  (unpaired M=32 tails at 256/288, M=16 at 320)
        for tg_idx in pl.range(2):
            tg = tg_idx * 128
            acc = pl.matmul(im2col[tg : tg + 32, 0:256], w[0:256, n0:n0 + 128], out_dtype=pl.FP32)
            acc2 = pl.matmul(im2col[tg + 32 : tg + 64, 0:256], w[0:256, n0:n0 + 128], out_dtype=pl.FP32)
            acc3 = pl.matmul(im2col[tg + 64 : tg + 96, 0:256], w[0:256, n0:n0 + 128], out_dtype=pl.FP32)
            acc4 = pl.matmul(im2col[tg + 96 : tg + 128, 0:256], w[0:256, n0:n0 + 128], out_dtype=pl.FP32)
            for kb in pl.pipeline(53, stage=2):        # chunks 1..53
                k0 = (kb + 1) * 256
                wt = w[k0:k0 + 256, n0:n0 + 128]       # loaded once, used by all four tiles
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
        # tail rows 256..287 (unpaired M=32; separate names per shape)
        acc_u = pl.matmul(im2col[256:288, 0:256], w[0:256, n0:n0 + 128], out_dtype=pl.FP32)
        for kb_u in pl.pipeline(53, stage=2):
            k0_u = (kb_u + 1) * 256
            acc_u = pl.matmul_acc(acc_u, im2col[256:288, k0_u:k0_u + 256], w[k0_u:k0_u + 256, n0:n0 + 128])
        lb_u = pl.slice(bias, [1, 128], [0, n0])
        ones_u = pl.full([32, 128], dtype=pl.FP32, value=1.0)
        acc_u = pl.add(acc_u, pl.col_expand_mul(ones_u, pl.cast(lb_u, target_type=pl.FP32)))
        out[256:288, n0 : n0 + 128] = pl.cast(acc_u, target_type=pl.BF16, mode="rint")
        # tail rows 288..319 (unpaired M=32)
        acc_v = pl.matmul(im2col[288:320, 0:256], w[0:256, n0:n0 + 128], out_dtype=pl.FP32)
        for kb_v in pl.pipeline(53, stage=2):
            k0_v = (kb_v + 1) * 256
            acc_v = pl.matmul_acc(acc_v, im2col[288:320, k0_v:k0_v + 256], w[k0_v:k0_v + 256, n0:n0 + 128])
        lb_v = pl.slice(bias, [1, 128], [0, n0])
        ones_v = pl.full([32, 128], dtype=pl.FP32, value=1.0)
        acc_v = pl.add(acc_v, pl.col_expand_mul(ones_v, pl.cast(lb_v, target_type=pl.FP32)))
        out[288:320, n0 : n0 + 128] = pl.cast(acc_v, target_type=pl.BF16, mode="rint")
        # tail rows 320..335 (M=16)
        acc_t = pl.matmul(im2col[320:336, 0:256], w[0:256, n0:n0 + 128], out_dtype=pl.FP32)
        for kb_t in pl.pipeline(53, stage=2):          # chunks 1..53
            k0_t = (kb_t + 1) * 256
            acc_t = pl.matmul_acc(acc_t, im2col[320:336, k0_t:k0_t + 256], w[k0_t:k0_t + 256, n0:n0 + 128])
        lb_t = pl.slice(bias, [1, 128], [0, n0])
        ones_t = pl.full([16, 128], dtype=pl.FP32, value=1.0)
        acc_t = pl.add(acc_t, pl.col_expand_mul(ones_t, pl.cast(lb_t, target_type=pl.FP32)))
        out[320:336, n0 : n0 + 128] = pl.cast(acc_t, target_type=pl.BF16, mode="rint")
    return out


@pl.jit.inline
def ds2_patch(
    im2col: pl.Tensor[[V_TOKENS_FINAL_PATCH_PAD, 9 * V_DS1_OUT], pl.BF16],
    w: pl.Tensor[[9 * V_DS1_OUT, V_DS2_OUT], pl.BF16],
    bias: pl.Tensor[[1, V_DS2_OUT], pl.BF16],
    out: pl.Tensor[[V_TOKENS_FINAL_PATCH_PAD, V_DS2_OUT], pl.BF16],
):
    for nb_idx in pl.spmd(48, name_hint="ds2_patch"):  # V_DS2_OUT // 128
        n0 = nb_idx * 128
        # 2-way row-tile unroll (mirror of global ds2_pad): shared
        # `wt` feeds two adjacent tiles, weight re-stream 3x -> 2x; same K
        # order per row -> bit-exact. 96 = 1*64 + 32 (unpaired M=32 tail at 64).
        acc = pl.matmul(im2col[0:32, 0:512], w[0:512, n0:n0 + 128], out_dtype=pl.FP32)
        acc2 = pl.matmul(im2col[32:64, 0:512], w[0:512, n0:n0 + 128], out_dtype=pl.FP32)
        for kb in pl.pipeline(53, stage=2):           # chunks 1..53 (KC=512)
            k0 = (kb + 1) * 512
            wt = w[k0:k0 + 512, n0:n0 + 128]          # loaded once, used by both tiles
            acc = pl.matmul_acc(acc, im2col[0:32, k0:k0 + 512], wt)
            acc2 = pl.matmul_acc(acc2, im2col[32:64, k0:k0 + 512], wt)
        # add conv bias (per-output-channel) before cast
        lb = pl.slice(bias, [1, 128], [0, n0])
        ones_b = pl.full([32, 128], dtype=pl.FP32, value=1.0)
        acc = pl.add(acc, pl.col_expand_mul(ones_b, pl.cast(lb, target_type=pl.FP32)))
        out[0:32, n0 : n0 + 128] = pl.cast(acc, target_type=pl.BF16, mode="rint")
        acc2 = pl.add(acc2, pl.col_expand_mul(ones_b, pl.cast(lb, target_type=pl.FP32)))
        out[32:64, n0 : n0 + 128] = pl.cast(acc2, target_type=pl.BF16, mode="rint")
        # tail rows 64..95 (unpaired M=32)
        acc_u = pl.matmul(im2col[64:96, 0:512], w[0:512, n0:n0 + 128], out_dtype=pl.FP32)
        for kb_u in pl.pipeline(53, stage=2):          # chunks 1..53 (KC=512)
            k0_u = (kb_u + 1) * 512
            acc_u = pl.matmul_acc(acc_u, im2col[64:96, k0_u:k0_u + 512], w[k0_u:k0_u + 512, n0:n0 + 128])
        lb_u = pl.slice(bias, [1, 128], [0, n0])
        ones_u = pl.full([32, 128], dtype=pl.FP32, value=1.0)
        acc_u = pl.add(acc_u, pl.col_expand_mul(ones_u, pl.cast(lb_u, target_type=pl.FP32)))
        out[64:96, n0 : n0 + 128] = pl.cast(acc_u, target_type=pl.BF16, mode="rint")
    return out


# ── @pl.jit.inline bodies (patch-dim; inlined into the @pl.programs below) ───
# patch_embed is reused from conv_patch.patch_embed but that body is statically
# shaped to V_TOKENS=2704 (global). Re-declare a patch-dim body here whose
# tiling is over V_TOKENS_PATCH=1296 rows (1296 % 16 == 0, no pad).
PE_NT_PATCH = 32  # N-tile = spmd task granularity; each task loads its [592, PE_NT]
                  # weight slice ONCE and reuses it across all 81 row groups
                  # (weight-traffic-optimal: 48x37.5KB total vs 81x1.8MB when
                  # spmd runs over row groups and every core re-reads the
                  # full weight). Tuned on 8x910B per-crop bench:
                  # row-parallel 81 tasks = 374us, NT=64/24 tasks = 243us,
                  # NT=32/48 tasks = 176us (best), NT=16/96 tasks = 288us
                  # (2 waves on 81 cores).


@pl.jit.inline
def patch_embed_patch(
    x_patches: pl.Tensor[[V_TOKENS_PATCH, PATCH_FEAT_PAD], pl.BF16],
    w_proj: pl.Tensor[[PATCH_FEAT_PAD, V_WIDTH], pl.BF16],
    posemb: pl.Tensor[[V_TOKENS_PATCH, V_WIDTH], pl.BF16],
    out: pl.Tensor[[V_TOKENS_PATCH, V_WIDTH], pl.BF16],
):
    # M-tile: 1296 = 40*32 + 16 -> 40 M=32 tiles + 1 M=16 tail (mirror of the
    # global patch_embed: -14.8% single-card vs M=16-only).
    for nb_idx in pl.spmd(48, name_hint="patch_embed_patch"):  # V_WIDTH // PE_NT_PATCH
        n0 = nb_idx * 32                                                # PE_NT_PATCH
        wt = w_proj[0:PATCH_FEAT_PAD, n0:n0 + 32]
        for tg_idx in pl.range(40):
            tg = tg_idx * 32
            xc = x_patches[tg : tg + 32, 0:PATCH_FEAT_PAD]
            acc = pl.matmul(xc, wt, out_dtype=pl.FP32)
            pe = pl.cast(acc, target_type=pl.BF16, mode="rint")   # two-step: conv1 BF16 first
            out[tg : tg + 32, n0 : n0 + 32] = pl.cast(pl.add(pl.cast(pe, pl.FP32), pl.cast(posemb[tg : tg + 32, n0 : n0 + 32], pl.FP32)), target_type=pl.BF16, mode="rint")
        # tail row group 1280..1295 (M=16; 1296 = 40*32 + 16)
        acc16 = pl.matmul(x_patches[1280:1296, 0:PATCH_FEAT_PAD], wt, out_dtype=pl.FP32)
        pe16 = pl.cast(acc16, target_type=pl.BF16, mode="rint")
        out[1280:1296, n0 : n0 + 32] = pl.cast(pl.add(pl.cast(pe16, pl.FP32), pl.cast(posemb[1280:1296, n0 : n0 + 32], pl.FP32)), target_type=pl.BF16, mode="rint")
    return out


# vit_projector body redeclared at patch N=96 (the global vit_projector is
# statically shaped to N=176). Same tiling, different token count. 96 % 16 == 0.
@pl.jit.inline
def vit_projector_patch_body(
    x: pl.Tensor[[V_TOKENS_FINAL_PATCH_PAD, V_DS2_OUT], pl.BF16],
    w: pl.Tensor[[V_DS2_OUT, LM_HIDDEN], pl.BF16],
    out: pl.Tensor[[V_TOKENS_FINAL_PATCH_PAD, LM_HIDDEN], pl.BF16],
):
    # spmd row-groups -> N-blocks (patch_embed idiom). Old: only 6
    # row-group tasks (96 rows), 75/81 cores idle, full 50MB weight re-read.
    # M-tile 16 -> 32 (mirror of vit_projector): halves
    # the per-task [6144,64] weight-slice re-stream count (6x -> 3x); 96 = 3*32,
    # no tail; M-tiling never changes a row's K accumulation order.
    # 2-way row-tile unroll (mirror of vit_projector): shared `wt`
    # feeds two adjacent 32-row tiles, weight re-stream 3x -> 2x; same K order
    # per row -> bit-exact. 96 = 1*64 + 32 (unpaired M=32 tail at 64).
    for nb_idx in pl.spmd(64, name_hint="vit_proj_patch"):  # LM_HIDDEN // 64
        n0 = nb_idx * 64
        acc = pl.matmul(x[0:32, 0:512], w[0:512, n0:n0 + 64], out_dtype=pl.FP32)
        acc2 = pl.matmul(x[32:64, 0:512], w[0:512, n0:n0 + 64], out_dtype=pl.FP32)
        for kb in pl.pipeline(11, stage=2):             # chunks 1..11
            k0 = (kb + 1) * 512
            wt = w[k0:k0 + 512, n0:n0 + 64]             # loaded once, used by both tiles
            acc = pl.matmul_acc(acc, x[0:32, k0:k0 + 512], wt)
            acc2 = pl.matmul_acc(acc2, x[32:64, k0:k0 + 512], wt)
        out[0:32, n0 : n0 + 64] = pl.cast(acc, target_type=pl.BF16, mode="rint")
        out[32:64, n0 : n0 + 64] = pl.cast(acc2, target_type=pl.BF16, mode="rint")
        # tail rows 64..95 (unpaired M=32)
        acc_u = pl.matmul(x[64:96, 0:512], w[0:512, n0:n0 + 64], out_dtype=pl.FP32)
        for kb_u in pl.pipeline(11, stage=2):           # chunks 1..11
            k0_u = (kb_u + 1) * 512
            acc_u = pl.matmul_acc(acc_u, x[64:96, k0_u:k0_u + 512], w[k0_u:k0_u + 512, n0:n0 + 64])
        out[64:96, n0 : n0 + 64] = pl.cast(acc_u, target_type=pl.BF16, mode="rint")
    return out


# ── 8-card REPLICATED @pl.program stages (mirror vision_full_fwd, patch dims) ──
# Each per_rank (plain Orchestration) wraps an inline body; host_orch
# (HOST/Orchestrator) loops pl.range(pld.world_size()) and dispatches per_rank
# (..., device=r). Replicated = every rank gets the same full tensor (vLLM
# replicated parallelism for patch_embed/ln/ds1/ds2/projector). The 8x
# redundant compute is the known projector gap (deferred phase-2).

_pe_in = pl.inline(patch_embed_patch._func)
_ln_in = pl.inline(layernorm._func)
_ds1_in = pl.inline(ds1_patch._func)
_ds2_in = pl.inline(ds2_patch._func)
_proj_in = pl.inline(vit_projector_patch_body._func)


def _build_patch_embed_patch(tp_size: int = TP_WORLD_SIZE):
    @pl.program
    class PatchEmbedPatchProg:
        @pl.function(type=pl.FunctionType.Orchestration)
        def per_rank(self,
            x: pl.Tensor[[V_TOKENS_PATCH, PATCH_FEAT_PAD], pl.BF16],
            w: pl.Tensor[[PATCH_FEAT_PAD, V_WIDTH], pl.BF16],
            posemb: pl.Tensor[[V_TOKENS_PATCH, V_WIDTH], pl.BF16],
            out: pl.Out[pl.Tensor[[V_TOKENS_PATCH, V_WIDTH], pl.BF16]],
        ) -> pl.Tensor[[V_TOKENS_PATCH, V_WIDTH], pl.BF16]:
            out = _pe_in(x, w, posemb, out); return out

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self,
            hx: pl.Tensor[[tp_size, V_TOKENS_PATCH, PATCH_FEAT_PAD], pl.BF16],
            hw: pl.Tensor[[tp_size, PATCH_FEAT_PAD, V_WIDTH], pl.BF16],
            hp: pl.Tensor[[tp_size, V_TOKENS_PATCH, V_WIDTH], pl.BF16],
            hto: pl.Out[pl.Tensor[[tp_size, V_TOKENS_PATCH, V_WIDTH], pl.BF16]],
        ):
            for r in pl.range(pld.world_size()):
                self.per_rank(hx[r], hw[r], hp[r], hto[r], device=r)
    return PatchEmbedPatchProg


def _build_ln_pre_patch(tp_size: int = TP_WORLD_SIZE):
    @pl.program
    class LnPrePatchProg:
        @pl.function(type=pl.FunctionType.Orchestration)
        def per_rank(self,
            x: pl.Tensor[[V_TOKENS_PATCH, LN_D], pl.BF16],
            gamma: pl.Tensor[[LN_D], pl.FP32],
            beta: pl.Tensor[[LN_D], pl.FP32],
            out: pl.Out[pl.Tensor[[V_TOKENS_PATCH, LN_D], pl.BF16]],
        ) -> pl.Tensor[[V_TOKENS_PATCH, LN_D], pl.BF16]:
            out = _ln_in(x, gamma, beta, out); return out

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self,
            hx: pl.Tensor[[tp_size, V_TOKENS_PATCH, LN_D], pl.BF16],
            hg: pl.Tensor[[tp_size, LN_D], pl.FP32],
            hb: pl.Tensor[[tp_size, LN_D], pl.FP32],
            hto: pl.Out[pl.Tensor[[tp_size, V_TOKENS_PATCH, LN_D], pl.BF16]],
        ):
            for r in pl.range(pld.world_size()):
                self.per_rank(hx[r], hg[r], hb[r], hto[r], device=r)
    return LnPrePatchProg


def _build_ds1_patch(tp_size: int = TP_WORLD_SIZE):
    @pl.program
    class Ds1PatchProg:
        @pl.function(type=pl.FunctionType.Orchestration)
        def per_rank(self,
            im2col: pl.Tensor[[V_GRID_DS1_PATCH_PAD, 9 * V_WIDTH], pl.BF16],
            w: pl.Tensor[[9 * V_WIDTH, V_DS1_OUT], pl.BF16],
            bias: pl.Tensor[[1, V_DS1_OUT], pl.BF16],
            out: pl.Out[pl.Tensor[[V_GRID_DS1_PATCH_PAD, V_DS1_OUT], pl.BF16]],
        ) -> pl.Tensor[[V_GRID_DS1_PATCH_PAD, V_DS1_OUT], pl.BF16]:
            out = _ds1_in(im2col, w, bias, out); return out

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self,
            him2col: pl.Tensor[[tp_size, V_GRID_DS1_PATCH_PAD, 9 * V_WIDTH], pl.BF16],
            hw: pl.Tensor[[tp_size, 9 * V_WIDTH, V_DS1_OUT], pl.BF16],
            hbias: pl.Tensor[[tp_size, 1, V_DS1_OUT], pl.BF16],
            hto: pl.Out[pl.Tensor[[tp_size, V_GRID_DS1_PATCH_PAD, V_DS1_OUT], pl.BF16]],
        ):
            for r in pl.range(pld.world_size()):
                self.per_rank(him2col[r], hw[r], hbias[r], hto[r], device=r)
    return Ds1PatchProg


def _build_ds2_patch(tp_size: int = TP_WORLD_SIZE):
    @pl.program
    class Ds2PatchProg:
        @pl.function(type=pl.FunctionType.Orchestration)
        def per_rank(self,
            im2col: pl.Tensor[[V_TOKENS_FINAL_PATCH_PAD, 9 * V_DS1_OUT], pl.BF16],
            w: pl.Tensor[[9 * V_DS1_OUT, V_DS2_OUT], pl.BF16],
            bias: pl.Tensor[[1, V_DS2_OUT], pl.BF16],
            out: pl.Out[pl.Tensor[[V_TOKENS_FINAL_PATCH_PAD, V_DS2_OUT], pl.BF16]],
        ) -> pl.Tensor[[V_TOKENS_FINAL_PATCH_PAD, V_DS2_OUT], pl.BF16]:
            out = _ds2_in(im2col, w, bias, out); return out

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self,
            him2col: pl.Tensor[[tp_size, V_TOKENS_FINAL_PATCH_PAD, 9 * V_DS1_OUT], pl.BF16],
            hw: pl.Tensor[[tp_size, 9 * V_DS1_OUT, V_DS2_OUT], pl.BF16],
            hbias: pl.Tensor[[tp_size, 1, V_DS2_OUT], pl.BF16],
            hto: pl.Out[pl.Tensor[[tp_size, V_TOKENS_FINAL_PATCH_PAD, V_DS2_OUT], pl.BF16]],
        ):
            for r in pl.range(pld.world_size()):
                self.per_rank(him2col[r], hw[r], hbias[r], hto[r], device=r)
    return Ds2PatchProg


def _build_proj_patch(tp_size: int = TP_WORLD_SIZE):
    @pl.program
    class ProjPatchProg:
        @pl.function(type=pl.FunctionType.Orchestration)
        def per_rank(self,
            x: pl.Tensor[[V_TOKENS_FINAL_PATCH_PAD, V_DS2_OUT], pl.BF16],
            w: pl.Tensor[[V_DS2_OUT, LM_HIDDEN], pl.BF16],
            out: pl.Out[pl.Tensor[[V_TOKENS_FINAL_PATCH_PAD, LM_HIDDEN], pl.BF16]],
        ) -> pl.Tensor[[V_TOKENS_FINAL_PATCH_PAD, LM_HIDDEN], pl.BF16]:
            out = _proj_in(x, w, out); return out

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self,
            hx: pl.Tensor[[tp_size, V_TOKENS_FINAL_PATCH_PAD, V_DS2_OUT], pl.BF16],
            hw: pl.Tensor[[tp_size, V_DS2_OUT, LM_HIDDEN], pl.BF16],
            hto: pl.Out[pl.Tensor[[tp_size, V_TOKENS_FINAL_PATCH_PAD, LM_HIDDEN], pl.BF16]],
        ):
            for r in pl.range(pld.world_size()):
                self.per_rank(hx[r], hw[r], hto[r], device=r)
    return ProjPatchProg


PatchEmbedPatchProg = _build_patch_embed_patch(TP_WORLD_SIZE)
LnPrePatchProg = _build_ln_pre_patch(TP_WORLD_SIZE)
Ds1PatchProg = _build_ds1_patch(TP_WORLD_SIZE)
Ds2PatchProg = _build_ds2_patch(TP_WORLD_SIZE)
ProjPatchProg = _build_proj_patch(TP_WORLD_SIZE)


# ── B-axis batched (B6) stages: all 6 crops in ONE launch ────────────────────
# the per-crop stages above are B-batched here — host concatenates
# the 6 crops along the token axis and one @pl.program launch covers them all
# (mirror of how the patch tower already runs active=6 in a single launch).
# Why: crops are independent (every output row depends only on its own input
# row + weights), so row-axis concatenation keeps each row's K accumulation
# order -> per-crop bit-exact. The win is weight traffic + launch tax: serial
# per-crop launches re-stream the SAME weight 6x (ds1 85MB x6); batched, the
# per-task weight re-stream count is ~unchanged while rows x6 (arithmetic
# intensity x6), and 6x dispatch collapses to 1. vLLM batches the same way
# (b=6 single forward).
# Row structures (all crop counts are 16-aligned so x6 keeps 32-alignment):
#   patch_embed/ln_pre: 7776 = 243*32            (no tail)
#   ds1:  2048 = 16*128                          (4-way, rows zero-padded from 2016)
#   ds2:   576 = 4*128 + 2*32                    (2-way + two M=32 tails)
#   proj:  576 = 3*192 exact                     (6-way, zero tails)
# Weight passes per task: ds1 16 (vs 4x6=24 serial), ds2/proj 3 (vs 2x6=12).
V_CROPS = 6
V_TOKENS_PATCH_B = V_TOKENS_PATCH * V_CROPS                    # 7776
V_GRID_DS1_PATCH_PAD_B = V_GRID_DS1_PATCH_PAD * V_CROPS        # 2016
V_TOKENS_FINAL_PATCH_PAD_B = V_TOKENS_FINAL_PATCH_PAD * V_CROPS  # 576
# patch_embed K zero-padded 592 -> 768 (3x256) so every K chunk is uniformly
# 256 wide: the inner K loop becomes a static-shape pipeline and the 80-col
# tail's sub-cache-line loads disappear. Zero columns contribute nothing.
PE_K_PAD_B6 = 768
# Rows zero-padded 7776 -> 8064 = 2 x (42 x 96): the 12 N-tile x 2 row-share
# split is an exact full wave (24 tasks on 24 cube cores; 36/60/72-task splits
# measured slower) with no row tails. Padded output rows are exact zeros; the
# driver slices the first 7776 rows when chaining.
PE_B6_ROWS_PAD = 8064
# patch_embed device im2col: image [6,3,504,504] is channel-major-flattened to
# [18, 254016] (row = crop*3 + channel, col = h*504+w) — a zero-copy host view,
# no F.unfold. PATCH_FEAT=588 valid cols (3*14*14); K-pad [588:768] and row-pad
# [7776:8064] are zeroed in-kernel (create_tensor scratch is not zero-initialized).
PATCH_IM2COL_FEAT_ROWS_B6 = V_CROPS * 3                        # 18
# ptoas requires every tile row's byte size to be 32B-aligned (BF16 -> 16 cols),
# so the 14-px kw run is padded 14->16 on the host (F.pad, NOT unfold) and the K
# layout becomes [ch][kh][kw_pad] = ch*224 + kh*16 + kw_pad. w is re-laid out to
# match (pad taps -> 0 weight); matmul result unchanged.
PATCH_KW_PAD = 16                                              # 14 + 2 pad
PATCH_IM2COL_AREA_PAD = V_GRID_PATCH * V_PATCH * V_GRID_PATCH * PATCH_KW_PAD   # 36*14*36*16 = 290304
PE_K_VALID_B6 = 3 * V_PATCH * PATCH_KW_PAD                     # 672 (3*14*16)
# ds1 rows zero-padded 2016 -> 2048: 16 uniform 128-row iterations (U=4), no
# M=32 tails. Padded rows are bias rows (im2col zeros) in the golden.
V_GRID_DS1_PATCH_PAD_B6 = 2048

# ── device-side im2col gather (tap-major) constants (B6, mirror global) ──────
# feat is the 1-zero-border padded token layout (host pad+reshape); the gather
# writes im2col[r, t*C+c] = feat[ih, iw, c] (t=kh*3+kw), a contiguous row-copy
# per (row, 256-col chunk). The conv weight is permuted channel-major -> tap-major
# ONCE on the host (matmul result unchanged; only the FP32 accumulation order
# shifts, far below 1/128 rtol). See vision_full_fwd.py ds1_im2col_b (global).
DS1_PATCH_FEAT_GRID = V_GRID_PATCH + 2                  # 38 (36 + 1 border each side)
DS1_PATCH_FEAT_TOKENS = DS1_PATCH_FEAT_GRID * DS1_PATCH_FEAT_GRID   # 1444
DS1_PATCH_FEAT_ROWS_B6 = V_CROPS * DS1_PATCH_FEAT_TOKENS            # 8664
DS2_PATCH_FEAT_GRID = V_GRID_DS1_PATCH + 2              # 20 (18 + 1 border each side)
DS2_PATCH_FEAT_TOKENS = DS2_PATCH_FEAT_GRID * DS2_PATCH_FEAT_GRID   # 400
DS2_PATCH_FEAT_ROWS_B6 = V_CROPS * DS2_PATCH_FEAT_TOKENS            # 2400
# raw (unpadded) token rows fed to the B6 downsamplers; the 1-zero border pad is
# done ON-DEVICE (ds1/ds2_patch_border_pad_b6) instead of the host F.pad (mirror
# of the global DS1_RAW_ROWS_B/DS2_RAW_ROWS_B).
DS1_PATCH_RAW_ROWS_B6 = V_CROPS * V_GRID_PATCH * V_GRID_PATCH          # 7776 (6*36*36)
DS2_PATCH_RAW_ROWS_B6 = V_CROPS * V_GRID_DS1_PATCH * V_GRID_DS1_PATCH  # 1944 (6*18*18)


@pl.jit.inline
def patch_embed_patch_b6(
    x_patches: pl.Tensor[[PE_B6_ROWS_PAD, PE_K_PAD_B6], pl.BF16],
    w_proj: pl.Tensor[[PE_K_PAD_B6, V_WIDTH], pl.BF16],
    posemb: pl.Tensor[[PE_B6_ROWS_PAD, V_WIDTH], pl.BF16],
    out: pl.Tensor[[PE_B6_ROWS_PAD, V_WIDTH], pl.BF16],
):
    # NT=32 -> 128 with a 2D split: 12 N-tiles x 2 row-shares (4032 = 42*96
    # exact per share) = 24 tasks — exactly one wave on 24 cube cores
    # (36/60/72-task splits all measured slower; extra tasks only add
    # scheduling overhead). Every wt piece is [256,128] = 64KB — the
    # L0B cap (a 74KB single-shot wt silently corrupted results; 128KB
    # crashed the IR compiler). The serial row loop cannot pipeline (stores
    # inside the loop crash the compiler), so the 3-way row unroll provides
    # the ILP instead. 376 -> 299us (measured, warm position).
    for nb_idx in pl.spmd(24, name_hint="patch_embed_patch_b6"):
        r = nb_idx // 12
        n0 = (nb_idx - r * 12) * 128
        rb = r * 4032
        for tg_idx in pl.range(42):
            tg = rb + tg_idx * 96
            acc = pl.matmul(x_patches[tg : tg + 32, 0:256], w_proj[0:256, n0:n0 + 128], out_dtype=pl.FP32)
            acc2 = pl.matmul(x_patches[tg + 32 : tg + 64, 0:256], w_proj[0:256, n0:n0 + 128], out_dtype=pl.FP32)
            acc3 = pl.matmul(x_patches[tg + 64 : tg + 96, 0:256], w_proj[0:256, n0:n0 + 128], out_dtype=pl.FP32)
            for kb in pl.pipeline(2, stage=2):             # chunks 1..2 (KC=256)
                k0 = (kb + 1) * 256
                wt = w_proj[k0:k0 + 256, n0:n0 + 128]      # loaded once, used by all three tiles
                acc = pl.matmul_acc(acc, x_patches[tg : tg + 32, k0:k0 + 256], wt)
                acc2 = pl.matmul_acc(acc2, x_patches[tg + 32 : tg + 64, k0:k0 + 256], wt)
                acc3 = pl.matmul_acc(acc3, x_patches[tg + 64 : tg + 96, k0:k0 + 256], wt)
            pe = pl.cast(acc, target_type=pl.BF16, mode="rint")   # two-step: conv1 BF16 first
            out[tg : tg + 32, n0 : n0 + 128] = pl.cast(pl.add(pl.cast(pe, pl.FP32), pl.cast(posemb[tg : tg + 32, n0 : n0 + 128], pl.FP32)), target_type=pl.BF16, mode="rint")
            pe2 = pl.cast(acc2, target_type=pl.BF16, mode="rint")
            out[tg + 32 : tg + 64, n0 : n0 + 128] = pl.cast(pl.add(pl.cast(pe2, pl.FP32), pl.cast(posemb[tg + 32 : tg + 64, n0 : n0 + 128], pl.FP32)), target_type=pl.BF16, mode="rint")
            pe3 = pl.cast(acc3, target_type=pl.BF16, mode="rint")
            out[tg + 64 : tg + 96, n0 : n0 + 128] = pl.cast(pl.add(pl.cast(pe3, pl.FP32), pl.cast(posemb[tg + 64 : tg + 96, n0 : n0 + 128], pl.FP32)), target_type=pl.BF16, mode="rint")
    return out


# ln_pre B6 reuses layernorm (row-wise; spmd over token tiles picks up the
# bigger T automatically via pl.tensor.dim), only the static shape differs.
@pl.jit.inline
def ds1_patch_b6(
    im2col: pl.Tensor[[V_GRID_DS1_PATCH_PAD_B6, 9 * V_WIDTH], pl.BF16],
    w: pl.Tensor[[9 * V_WIDTH, V_DS1_OUT], pl.BF16],
    bias: pl.Tensor[[1, V_DS1_OUT], pl.BF16],
    out: pl.Tensor[[V_GRID_DS1_PATCH_PAD_B6, V_DS1_OUT], pl.BF16],
):
    # 4-way row-tile unroll on rows zero-padded 2016 -> 2048: 16 uniform
    # 128-row iterations, no M=32 tails (the old tail pass re-streamed the
    # whole 13.5MB weight for 3 rows; 1690 -> 1646us measured). Padded rows
    # are bias rows in the golden (im2col zeros -> out = bias).
    for nb_idx in pl.spmd(24, name_hint="ds1_patch_b6"):   # V_DS1_OUT // 128
        n0 = nb_idx * 128
        for tg_idx in pl.range(16):
            tg = tg_idx * 128
            acc = pl.matmul(im2col[tg : tg + 32, 0:256], w[0:256, n0:n0 + 128], out_dtype=pl.FP32)
            acc2 = pl.matmul(im2col[tg + 32 : tg + 64, 0:256], w[0:256, n0:n0 + 128], out_dtype=pl.FP32)
            acc3 = pl.matmul(im2col[tg + 64 : tg + 96, 0:256], w[0:256, n0:n0 + 128], out_dtype=pl.FP32)
            acc4 = pl.matmul(im2col[tg + 96 : tg + 128, 0:256], w[0:256, n0:n0 + 128], out_dtype=pl.FP32)
            for kb in pl.pipeline(53, stage=2):            # chunks 1..53
                k0 = (kb + 1) * 256
                wt = w[k0:k0 + 256, n0:n0 + 128]           # loaded once, used by all four tiles
                acc = pl.matmul_acc(acc, im2col[tg : tg + 32, k0:k0 + 256], wt)
                acc2 = pl.matmul_acc(acc2, im2col[tg + 32 : tg + 64, k0:k0 + 256], wt)
                acc3 = pl.matmul_acc(acc3, im2col[tg + 64 : tg + 96, k0:k0 + 256], wt)
                acc4 = pl.matmul_acc(acc4, im2col[tg + 96 : tg + 128, k0:k0 + 256], wt)
            lb = pl.slice(bias, [1, 128], [0, n0])
            ones_b = pl.full([32, 128], dtype=pl.FP32, value=1.0)
            bias_e = pl.col_expand_mul(ones_b, pl.cast(lb, target_type=pl.FP32))
            out[tg : tg + 32, n0 : n0 + 128] = pl.cast(pl.add(acc, bias_e), target_type=pl.BF16, mode="rint")
            out[tg + 32 : tg + 64, n0 : n0 + 128] = pl.cast(pl.add(acc2, bias_e), target_type=pl.BF16, mode="rint")
            out[tg + 64 : tg + 96, n0 : n0 + 128] = pl.cast(pl.add(acc3, bias_e), target_type=pl.BF16, mode="rint")
            out[tg + 96 : tg + 128, n0 : n0 + 128] = pl.cast(pl.add(acc4, bias_e), target_type=pl.BF16, mode="rint")
    return out


@pl.jit.inline
def ds2_patch_b6(
    im2col: pl.Tensor[[V_TOKENS_FINAL_PATCH_PAD_B, 9 * V_DS1_OUT], pl.BF16],
    w: pl.Tensor[[9 * V_DS1_OUT, V_DS2_OUT], pl.BF16],
    bias: pl.Tensor[[1, V_DS2_OUT], pl.BF16],
    out: pl.Tensor[[V_TOKENS_FINAL_PATCH_PAD_B, V_DS2_OUT], pl.BF16],
):
    # 6-way row-tile unroll at KC=256: acc 6x[32,128] fp32 = 96KB <= 128KB Acc
    # budget. 576 = 3*192 exact -> zero tails; weight passes 6 -> 3 (U=6 beat
    # the committed U=4 by 9%; U>6 at NT=128 over-fills Acc — per-pass cost
    # grows superlinearly, measured on ds1). KC re-chunks K vs the per-crop
    # kernel but is validated against the torch golden at the tight gate.
    for nb_idx in pl.spmd(48, name_hint="ds2_patch_b6"):   # V_DS2_OUT // 128
        n0 = nb_idx * 128
        for tg_idx in pl.range(3):
            tg = tg_idx * 192
            acc = pl.matmul(im2col[tg : tg + 32, 0:256], w[0:256, n0:n0 + 128], out_dtype=pl.FP32)
            acc2 = pl.matmul(im2col[tg + 32 : tg + 64, 0:256], w[0:256, n0:n0 + 128], out_dtype=pl.FP32)
            acc3 = pl.matmul(im2col[tg + 64 : tg + 96, 0:256], w[0:256, n0:n0 + 128], out_dtype=pl.FP32)
            acc4 = pl.matmul(im2col[tg + 96 : tg + 128, 0:256], w[0:256, n0:n0 + 128], out_dtype=pl.FP32)
            acc5 = pl.matmul(im2col[tg + 128 : tg + 160, 0:256], w[0:256, n0:n0 + 128], out_dtype=pl.FP32)
            acc6 = pl.matmul(im2col[tg + 160 : tg + 192, 0:256], w[0:256, n0:n0 + 128], out_dtype=pl.FP32)
            for kb in pl.pipeline(107, stage=2):           # chunks 1..107 (KC=256)
                k0 = (kb + 1) * 256
                wt = w[k0:k0 + 256, n0:n0 + 128]           # loaded once, used by all six tiles
                acc = pl.matmul_acc(acc, im2col[tg : tg + 32, k0:k0 + 256], wt)
                acc2 = pl.matmul_acc(acc2, im2col[tg + 32 : tg + 64, k0:k0 + 256], wt)
                acc3 = pl.matmul_acc(acc3, im2col[tg + 64 : tg + 96, k0:k0 + 256], wt)
                acc4 = pl.matmul_acc(acc4, im2col[tg + 96 : tg + 128, k0:k0 + 256], wt)
                acc5 = pl.matmul_acc(acc5, im2col[tg + 128 : tg + 160, k0:k0 + 256], wt)
                acc6 = pl.matmul_acc(acc6, im2col[tg + 160 : tg + 192, k0:k0 + 256], wt)
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
            acc5 = pl.add(acc5, pl.col_expand_mul(ones_b, pl.cast(lb, target_type=pl.FP32)))
            out[tg + 128 : tg + 160, n0 : n0 + 128] = pl.cast(acc5, target_type=pl.BF16, mode="rint")
            acc6 = pl.add(acc6, pl.col_expand_mul(ones_b, pl.cast(lb, target_type=pl.FP32)))
            out[tg + 160 : tg + 192, n0 : n0 + 128] = pl.cast(acc6, target_type=pl.BF16, mode="rint")
    return out


# ── patch_embed device im2col gather (k14 s14 p0, B6) ────────────────────────
# k14 s14 p0 has no overlap/padding, so this is a pure reshape — but the image
# arrives channel-major-flattened [18, 254016] (row = crop*3+channel) and the
# im2col K is channel-major (c*196 + kh*14 + kw), so each (c,kh) row of a patch
# is a contiguous 14-px run in BOTH source and target -> a [1,14] chunk copy per
# (c,kh) (mirror of ds1/ds2's [1,256] gather, only the run width differs).
@pl.jit.inline
def patch_im2col_b6(
    image: pl.Tensor[[PATCH_IM2COL_FEAT_ROWS_B6, PATCH_IM2COL_AREA_PAD], pl.BF16],  # [18, 290304]
    im2col: pl.Tensor[[PE_B6_ROWS_PAD, PE_K_PAD_B6], pl.BF16],                       # [8064, 768]
):
    # image is [oh][kh][ow][kw_pad] flattened (kw padded 14->16 on the host), so
    # each (ch,oh,kh,ow) tap is a contiguous 16-col run in BOTH source and target
    # -> a [1,16] chunk copy (32B-aligned, mirror of ds1/ds2's [1,256] gather).
    # The two kw_pad cols are zero on both sides, contributing nothing.
    for c in pl.unroll(V_CROPS):                       # 6 crops
        rc = c * V_TOKENS_PATCH                        # 1296
        for c_ch in pl.range(3):                       # 3 channels
            src = c * 3 + c_ch                         # image row (crop*3+channel)
            for oh in pl.spmd(V_GRID_PATCH):           # 36
                for ow in pl.range(V_GRID_PATCH):      # 36
                    r = rc + oh * V_GRID_PATCH + ow
                    for kh in pl.range(V_PATCH):       # 14
                        k0 = c_ch * (V_PATCH * PATCH_KW_PAD) + kh * PATCH_KW_PAD       # ch*224 + kh*16
                        p0 = oh * (V_PATCH * V_GRID_PATCH * PATCH_KW_PAD) + kh * (V_GRID_PATCH * PATCH_KW_PAD) + ow * PATCH_KW_PAD
                        im2col[r : r + 1, k0 : k0 + PATCH_KW_PAD] = image[src : src + 1, p0 : p0 + PATCH_KW_PAD]
    # zero K-pad cols [PE_K_VALID_B6:PE_K_PAD_B6] (672:768 = 96 cols; scratch is
    # not zero-initialized and the matmul needs them == 0).
    for rb_idx in pl.spmd(PE_B6_ROWS_PAD // 64):       # 126 row-blocks of 64
        rb = rb_idx * 64
        im2col[rb : rb + 64, PE_K_VALID_B6 : PE_K_PAD_B6] = pl.full(
            [64, PE_K_PAD_B6 - PE_K_VALID_B6], dtype=pl.BF16, value=0.0)
    # zero row-pad rows [V_TOKENS_PATCH_B:PE_B6_ROWS_PAD] (288 rows)
    for pr in pl.spmd(PE_B6_ROWS_PAD - V_TOKENS_PATCH_B):   # 288
        im2col[V_TOKENS_PATCH_B + pr : V_TOKENS_PATCH_B + pr + 1, 0:PE_K_PAD_B6] = pl.full(
            [1, PE_K_PAD_B6], dtype=pl.BF16, value=0.0)
    return im2col


# ── device-side im2col gather kernels (tap-major, B6) ────────────────────────
# Move the host F.unfold onto the NPU (mirror of the global ds1_im2col_b /
# ds2_im2col_b). feat is the 1-zero-border padded token layout (host does the
# cheap pad+reshape instead of unfold); im2col is a device scratch
# (create_tensor is NOT zero-initialized, so pad rows are zeroed explicitly —
# the matmul needs them == 0 so out == bias). Valid rows are written by the
# gather; pad rows are zeroed in a second spmd pass. 6 crops are compile-time
# unrolled (all active; no runtime guard — the B6 path has no B axis, the crops
# are concatenated along the token axis).
@pl.jit.inline
def ds1_patch_b6_im2col(
    feat: pl.Tensor[[DS1_PATCH_FEAT_ROWS_B6, V_WIDTH], pl.BF16],          # [8664, 1536]
    im2col: pl.Tensor[[V_GRID_DS1_PATCH_PAD_B6, 9 * V_WIDTH], pl.BF16],   # [2048, 13824]
):
    for c in pl.unroll(V_CROPS):                       # 6 crops, all active
        fc = c * DS1_PATCH_FEAT_TOKENS                 # 1444
        rc = c * V_GRID_DS1_PATCH_PAD                  # 336
        for oh in pl.spmd(V_GRID_DS1_PATCH):           # 18
            for ow in pl.range(V_GRID_DS1_PATCH):      # 18
                r = rc + oh * V_GRID_DS1_PATCH + ow
                for kh in pl.range(3):
                    for kw in pl.range(3):
                        src = fc + (2 * oh + kh) * DS1_PATCH_FEAT_GRID + (2 * ow + kw)
                        k0 = (kh * 3 + kw) * V_WIDTH
                        for ck in pl.range(V_WIDTH // 256):   # 6
                            c0 = ck * 256
                            im2col[r : r + 1, k0 + c0 : k0 + c0 + 256] = feat[src : src + 1, c0 : c0 + 256]
        # zero per-crop pad rows [324:336] (12 rows; scratch is not zero-initialized)
        for pr in pl.spmd(V_GRID_DS1_PATCH_PAD - V_GRID_DS1_PATCH * V_GRID_DS1_PATCH):  # 12
            for ck in pl.range(9 * V_WIDTH // 256):             # 54
                c0 = ck * 256
                im2col[rc + V_GRID_DS1_PATCH * V_GRID_DS1_PATCH + pr : rc + V_GRID_DS1_PATCH * V_GRID_DS1_PATCH + pr + 1, c0 : c0 + 256] = pl.full([1, 256], dtype=pl.BF16, value=0.0)
    # zero trailing pad rows [2016:2048] (32 rows)
    for pr in pl.spmd(V_GRID_DS1_PATCH_PAD_B6 - V_GRID_DS1_PATCH_PAD_B):  # 32
        for ck in pl.range(9 * V_WIDTH // 256):             # 54
            c0 = ck * 256
            im2col[V_GRID_DS1_PATCH_PAD_B + pr : V_GRID_DS1_PATCH_PAD_B + pr + 1, c0 : c0 + 256] = pl.full([1, 256], dtype=pl.BF16, value=0.0)
    return im2col


@pl.jit.inline
def ds2_patch_b6_im2col(
    feat: pl.Tensor[[DS2_PATCH_FEAT_ROWS_B6, V_DS1_OUT], pl.BF16],         # [2400, 3072]
    im2col: pl.Tensor[[V_TOKENS_FINAL_PATCH_PAD_B, 9 * V_DS1_OUT], pl.BF16],  # [576, 27648]
):
    for c in pl.unroll(V_CROPS):                       # 6 crops, all active
        fc = c * DS2_PATCH_FEAT_TOKENS                 # 400
        rc = c * V_TOKENS_FINAL_PATCH_PAD              # 96
        for oh in pl.spmd(V_GRID_DS2_PATCH):           # 9
            for ow in pl.range(V_GRID_DS2_PATCH):      # 9
                r = rc + oh * V_GRID_DS2_PATCH + ow
                for kh in pl.range(3):
                    for kw in pl.range(3):
                        src = fc + (2 * oh + kh) * DS2_PATCH_FEAT_GRID + (2 * ow + kw)
                        k0 = (kh * 3 + kw) * V_DS1_OUT
                        for ck in pl.range(V_DS1_OUT // 256):  # 12
                            c0 = ck * 256
                            im2col[r : r + 1, k0 + c0 : k0 + c0 + 256] = feat[src : src + 1, c0 : c0 + 256]
        # zero per-crop pad rows [81:96] (15 rows)
        for pr in pl.spmd(V_TOKENS_FINAL_PATCH_PAD - V_TOKENS_FINAL_PATCH):  # 15
            for ck in pl.range(9 * V_DS1_OUT // 256):   # 108
                c0 = ck * 256
                im2col[rc + V_TOKENS_FINAL_PATCH + pr : rc + V_TOKENS_FINAL_PATCH + pr + 1, c0 : c0 + 256] = pl.full([1, 256], dtype=pl.BF16, value=0.0)
    return im2col


# ── device-side 1-zero-border pad kernels (B6, mirror of the global ──────────
# ds1/ds2_border_pad_b). Move the host F.pad+reshape ONTO the NPU: raw token
# layout [g*g, c] -> padded [(g+2)*(g+2), c] with interior (ih,iw) -> (ih+1,
# iw+1) and the 4 border edges explicitly zeroed (create_tensor scratch is NOT
# zero-initialized, so the border must be zeroed for the im2col gather to read
# ih=2*oh+kh in-bounds). 6 crops are compile-time unrolled (all active; no B
# axis — the B6 path concatenates crops along the token axis).
@pl.jit.inline
def ds1_patch_border_pad_b6(
    raw: pl.Tensor[[DS1_PATCH_RAW_ROWS_B6, V_WIDTH], pl.BF16],          # [7776, 1536]
    padded: pl.Tensor[[DS1_PATCH_FEAT_ROWS_B6, V_WIDTH], pl.BF16],      # [8664, 1536]
):
    for c in pl.unroll(V_CROPS):                      # 6 crops, all active
        rb = c * V_GRID_PATCH * V_GRID_PATCH          # 1296
        pb = c * DS1_PATCH_FEAT_TOKENS                # 1444
        # interior gather: 36x36 -> 38x38 offset (+1,+1)
        for ih in pl.spmd(V_GRID_PATCH):              # 36
            for iw in pl.range(V_GRID_PATCH):         # 36
                src = rb + ih * V_GRID_PATCH + iw
                dst = pb + (ih + 1) * DS1_PATCH_FEAT_GRID + (iw + 1)
                for ck in pl.range(V_WIDTH // 256):   # 6
                    c0 = ck * 256
                    padded[dst : dst + 1, c0 : c0 + 256] = raw[src : src + 1, c0 : c0 + 256]
        # zero top (ih=0) + bottom (ih=37) border rows
        for iw in pl.spmd(DS1_PATCH_FEAT_GRID):       # 38
            for ck in pl.range(V_WIDTH // 256):
                c0 = ck * 256
                padded[pb + iw : pb + iw + 1, c0 : c0 + 256] = pl.full([1, 256], dtype=pl.BF16, value=0.0)
                padded[pb + (DS1_PATCH_FEAT_GRID - 1) * DS1_PATCH_FEAT_GRID + iw : pb + (DS1_PATCH_FEAT_GRID - 1) * DS1_PATCH_FEAT_GRID + iw + 1, c0 : c0 + 256] = pl.full([1, 256], dtype=pl.BF16, value=0.0)
        # zero left (iw=0) + right (iw=37) border cols (interior rows ih=1..36)
        for ih in pl.spmd(V_GRID_PATCH):              # 36
            for ck in pl.range(V_WIDTH // 256):
                c0 = ck * 256
                padded[pb + (ih + 1) * DS1_PATCH_FEAT_GRID : pb + (ih + 1) * DS1_PATCH_FEAT_GRID + 1, c0 : c0 + 256] = pl.full([1, 256], dtype=pl.BF16, value=0.0)
                padded[pb + (ih + 1) * DS1_PATCH_FEAT_GRID + (DS1_PATCH_FEAT_GRID - 1) : pb + (ih + 1) * DS1_PATCH_FEAT_GRID + DS1_PATCH_FEAT_GRID, c0 : c0 + 256] = pl.full([1, 256], dtype=pl.BF16, value=0.0)
    return padded


@pl.jit.inline
def ds2_patch_border_pad_b6(
    raw: pl.Tensor[[DS2_PATCH_RAW_ROWS_B6, V_DS1_OUT], pl.BF16],        # [1944, 3072]
    padded: pl.Tensor[[DS2_PATCH_FEAT_ROWS_B6, V_DS1_OUT], pl.BF16],    # [2400, 3072]
):
    for c in pl.unroll(V_CROPS):                      # 6 crops, all active
        rb = c * V_GRID_DS1_PATCH * V_GRID_DS1_PATCH  # 324
        pb = c * DS2_PATCH_FEAT_TOKENS                # 400
        # interior gather: 18x18 -> 20x20 offset (+1,+1)
        for ih in pl.spmd(V_GRID_DS1_PATCH):          # 18
            for iw in pl.range(V_GRID_DS1_PATCH):     # 18
                src = rb + ih * V_GRID_DS1_PATCH + iw
                dst = pb + (ih + 1) * DS2_PATCH_FEAT_GRID + (iw + 1)
                for ck in pl.range(V_DS1_OUT // 256): # 12
                    c0 = ck * 256
                    padded[dst : dst + 1, c0 : c0 + 256] = raw[src : src + 1, c0 : c0 + 256]
        # zero top (ih=0) + bottom (ih=19) border rows
        for iw in pl.spmd(DS2_PATCH_FEAT_GRID):       # 20
            for ck in pl.range(V_DS1_OUT // 256):
                c0 = ck * 256
                padded[pb + iw : pb + iw + 1, c0 : c0 + 256] = pl.full([1, 256], dtype=pl.BF16, value=0.0)
                padded[pb + (DS2_PATCH_FEAT_GRID - 1) * DS2_PATCH_FEAT_GRID + iw : pb + (DS2_PATCH_FEAT_GRID - 1) * DS2_PATCH_FEAT_GRID + iw + 1, c0 : c0 + 256] = pl.full([1, 256], dtype=pl.BF16, value=0.0)
        # zero left (iw=0) + right (iw=19) border cols (interior rows ih=1..18)
        for ih in pl.spmd(V_GRID_DS1_PATCH):          # 18
            for ck in pl.range(V_DS1_OUT // 256):
                c0 = ck * 256
                padded[pb + (ih + 1) * DS2_PATCH_FEAT_GRID : pb + (ih + 1) * DS2_PATCH_FEAT_GRID + 1, c0 : c0 + 256] = pl.full([1, 256], dtype=pl.BF16, value=0.0)
                padded[pb + (ih + 1) * DS2_PATCH_FEAT_GRID + (DS2_PATCH_FEAT_GRID - 1) : pb + (ih + 1) * DS2_PATCH_FEAT_GRID + DS2_PATCH_FEAT_GRID, c0 : c0 + 256] = pl.full([1, 256], dtype=pl.BF16, value=0.0)
    return padded


@pl.jit.inline
def vit_projector_patch_b6(
    x: pl.Tensor[[V_TOKENS_FINAL_PATCH_PAD_B, V_DS2_OUT], pl.BF16],
    w: pl.Tensor[[V_DS2_OUT, LM_HIDDEN], pl.BF16],
    out: pl.Tensor[[V_TOKENS_FINAL_PATCH_PAD_B, LM_HIDDEN], pl.BF16],
):
    # NT=64 -> 128 with a 2D split: 32 N-tiles x 3 row-shares = 96 tasks,
    # each computing ONE whole U=6 group (576 = 3*192 exact, zero tails).
    # 96 tasks = 4 per cube core — the 32-task version left a straggler wave
    # on 24 cores (477 -> 336.5us measured). KC=512 -> 256 keeps the wt
    # chunk [256,128] at the 64KB L0B cap. 535 -> 336.5us (0.80x vLLM).
    for t_idx in pl.spmd(96, name_hint="vit_proj_patch_b6"):
        n0 = (t_idx % 32) * 128
        rb = (t_idx // 32) * 192
        acc = pl.matmul(x[rb : rb + 32, 0:256], w[0:256, n0:n0 + 128], out_dtype=pl.FP32)
        acc2 = pl.matmul(x[rb + 32 : rb + 64, 0:256], w[0:256, n0:n0 + 128], out_dtype=pl.FP32)
        acc3 = pl.matmul(x[rb + 64 : rb + 96, 0:256], w[0:256, n0:n0 + 128], out_dtype=pl.FP32)
        acc4 = pl.matmul(x[rb + 96 : rb + 128, 0:256], w[0:256, n0:n0 + 128], out_dtype=pl.FP32)
        acc5 = pl.matmul(x[rb + 128 : rb + 160, 0:256], w[0:256, n0:n0 + 128], out_dtype=pl.FP32)
        acc6 = pl.matmul(x[rb + 160 : rb + 192, 0:256], w[0:256, n0:n0 + 128], out_dtype=pl.FP32)
        for kb in pl.pipeline(23, stage=2):                 # chunks 1..23 (KC=256)
            k0 = (kb + 1) * 256
            wt = w[k0:k0 + 256, n0:n0 + 128]                # loaded once, used by all six tiles
            acc = pl.matmul_acc(acc, x[rb : rb + 32, k0:k0 + 256], wt)
            acc2 = pl.matmul_acc(acc2, x[rb + 32 : rb + 64, k0:k0 + 256], wt)
            acc3 = pl.matmul_acc(acc3, x[rb + 64 : rb + 96, k0:k0 + 256], wt)
            acc4 = pl.matmul_acc(acc4, x[rb + 96 : rb + 128, k0:k0 + 256], wt)
            acc5 = pl.matmul_acc(acc5, x[rb + 128 : rb + 160, k0:k0 + 256], wt)
            acc6 = pl.matmul_acc(acc6, x[rb + 160 : rb + 192, k0:k0 + 256], wt)
        out[rb : rb + 32, n0 : n0 + 128] = pl.cast(acc, target_type=pl.BF16, mode="rint")
        out[rb + 32 : rb + 64, n0 : n0 + 128] = pl.cast(acc2, target_type=pl.BF16, mode="rint")
        out[rb + 64 : rb + 96, n0 : n0 + 128] = pl.cast(acc3, target_type=pl.BF16, mode="rint")
        out[rb + 96 : rb + 128, n0 : n0 + 128] = pl.cast(acc4, target_type=pl.BF16, mode="rint")
        out[rb + 128 : rb + 160, n0 : n0 + 128] = pl.cast(acc5, target_type=pl.BF16, mode="rint")
        out[rb + 160 : rb + 192, n0 : n0 + 128] = pl.cast(acc6, target_type=pl.BF16, mode="rint")
    return out


_pe_b6_in = pl.inline(patch_embed_patch_b6._func)
# patch_embed device im2col gather (folded into PatchEmbedPatchB6Prog before matmul)
_patch_im2col_b6_in = pl.inline(patch_im2col_b6._func)
_ln_b6_in = pl.inline(layernorm_b6._func)   # B6 granularity T_TILE=24/D_TILE=512
_ds1_b6_in = pl.inline(ds1_patch_b6._func)
_ds2_b6_in = pl.inline(ds2_patch_b6._func)
# device-side im2col gather (folded into Ds1PatchB6Prog/Ds2PatchB6Prog before matmul)
_ds1_patch_b6_im2col_in = pl.inline(ds1_patch_b6_im2col._func)
_ds2_patch_b6_im2col_in = pl.inline(ds2_patch_b6_im2col._func)
# device-side 1-zero-border pad (folded into Ds1PatchB6Prog/Ds2PatchB6Prog before im2col)
_ds1_patch_border_pad_b6_in = pl.inline(ds1_patch_border_pad_b6._func)
_ds2_patch_border_pad_b6_in = pl.inline(ds2_patch_border_pad_b6._func)
_proj_b6_in = pl.inline(vit_projector_patch_b6._func)


def _build_patch_embed_patch_b6(tp_size: int = TP_WORLD_SIZE):
    @pl.program
    class PatchEmbedPatchB6Prog:
        @pl.function(type=pl.FunctionType.Orchestration)
        def per_rank(self,
            image: pl.Tensor[[PATCH_IM2COL_FEAT_ROWS_B6, PATCH_IM2COL_AREA_PAD], pl.BF16],
            w: pl.Tensor[[PE_K_PAD_B6, V_WIDTH], pl.BF16],
            posemb: pl.Tensor[[PE_B6_ROWS_PAD, V_WIDTH], pl.BF16],
            out: pl.Out[pl.Tensor[[PE_B6_ROWS_PAD, V_WIDTH], pl.BF16]],
        ) -> pl.Tensor[[PE_B6_ROWS_PAD, V_WIDTH], pl.BF16]:
            im2col = pl.create_tensor([PE_B6_ROWS_PAD, PE_K_PAD_B6], dtype=pl.BF16)
            im2col = _patch_im2col_b6_in(image, im2col)
            out = _pe_b6_in(im2col, w, posemb, out); return out

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self,
            himage: pl.Tensor[[tp_size, PATCH_IM2COL_FEAT_ROWS_B6, PATCH_IM2COL_AREA_PAD], pl.BF16],
            hw: pl.Tensor[[tp_size, PE_K_PAD_B6, V_WIDTH], pl.BF16],
            hp: pl.Tensor[[tp_size, PE_B6_ROWS_PAD, V_WIDTH], pl.BF16],
            hto: pl.Out[pl.Tensor[[tp_size, PE_B6_ROWS_PAD, V_WIDTH], pl.BF16]],
        ):
            for r in pl.range(pld.world_size()):
                self.per_rank(himage[r], hw[r], hp[r], hto[r], device=r)
    return PatchEmbedPatchB6Prog


def _build_ln_pre_patch_b6(tp_size: int = TP_WORLD_SIZE):
    @pl.program
    class LnPrePatchB6Prog:
        @pl.function(type=pl.FunctionType.Orchestration)
        def per_rank(self,
            x: pl.Tensor[[V_TOKENS_PATCH_B, LN_D], pl.BF16],
            gamma: pl.Tensor[[LN_D], pl.FP32],
            beta: pl.Tensor[[LN_D], pl.FP32],
            out: pl.Out[pl.Tensor[[V_TOKENS_PATCH_B, LN_D], pl.BF16]],
        ) -> pl.Tensor[[V_TOKENS_PATCH_B, LN_D], pl.BF16]:
            out = _ln_b6_in(x, gamma, beta, out); return out

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self,
            hx: pl.Tensor[[tp_size, V_TOKENS_PATCH_B, LN_D], pl.BF16],
            hg: pl.Tensor[[tp_size, LN_D], pl.FP32],
            hb: pl.Tensor[[tp_size, LN_D], pl.FP32],
            hto: pl.Out[pl.Tensor[[tp_size, V_TOKENS_PATCH_B, LN_D], pl.BF16]],
        ):
            for r in pl.range(pld.world_size()):
                self.per_rank(hx[r], hg[r], hb[r], hto[r], device=r)
    return LnPrePatchB6Prog


def _build_ds1_patch_b6(tp_size: int = TP_WORLD_SIZE):
    @pl.program
    class Ds1PatchB6Prog:
        @pl.function(type=pl.FunctionType.Orchestration)
        def per_rank(self,
            feat: pl.Tensor[[DS1_PATCH_RAW_ROWS_B6, V_WIDTH], pl.BF16],
            w: pl.Tensor[[9 * V_WIDTH, V_DS1_OUT], pl.BF16],
            bias: pl.Tensor[[1, V_DS1_OUT], pl.BF16],
            out: pl.Out[pl.Tensor[[V_GRID_DS1_PATCH_PAD_B6, V_DS1_OUT], pl.BF16]],
        ) -> pl.Tensor[[V_GRID_DS1_PATCH_PAD_B6, V_DS1_OUT], pl.BF16]:
            feat_pad = pl.create_tensor([DS1_PATCH_FEAT_ROWS_B6, V_WIDTH], dtype=pl.BF16)
            feat_pad = _ds1_patch_border_pad_b6_in(feat, feat_pad)
            im2col = pl.create_tensor([V_GRID_DS1_PATCH_PAD_B6, 9 * V_WIDTH], dtype=pl.BF16)
            im2col = _ds1_patch_b6_im2col_in(feat_pad, im2col)
            out = _ds1_b6_in(im2col, w, bias, out)
            return out

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self,
            hfeat: pl.Tensor[[tp_size, DS1_PATCH_RAW_ROWS_B6, V_WIDTH], pl.BF16],
            hw: pl.Tensor[[tp_size, 9 * V_WIDTH, V_DS1_OUT], pl.BF16],
            hbias: pl.Tensor[[tp_size, 1, V_DS1_OUT], pl.BF16],
            hto: pl.Out[pl.Tensor[[tp_size, V_GRID_DS1_PATCH_PAD_B6, V_DS1_OUT], pl.BF16]],
        ):
            for r in pl.range(pld.world_size()):
                self.per_rank(hfeat[r], hw[r], hbias[r], hto[r], device=r)
    return Ds1PatchB6Prog


def _build_ds2_patch_b6(tp_size: int = TP_WORLD_SIZE):
    @pl.program
    class Ds2PatchB6Prog:
        @pl.function(type=pl.FunctionType.Orchestration)
        def per_rank(self,
            feat: pl.Tensor[[DS2_PATCH_RAW_ROWS_B6, V_DS1_OUT], pl.BF16],
            w: pl.Tensor[[9 * V_DS1_OUT, V_DS2_OUT], pl.BF16],
            bias: pl.Tensor[[1, V_DS2_OUT], pl.BF16],
            out: pl.Out[pl.Tensor[[V_TOKENS_FINAL_PATCH_PAD_B, V_DS2_OUT], pl.BF16]],
        ) -> pl.Tensor[[V_TOKENS_FINAL_PATCH_PAD_B, V_DS2_OUT], pl.BF16]:
            feat_pad = pl.create_tensor([DS2_PATCH_FEAT_ROWS_B6, V_DS1_OUT], dtype=pl.BF16)
            feat_pad = _ds2_patch_border_pad_b6_in(feat, feat_pad)
            im2col = pl.create_tensor([V_TOKENS_FINAL_PATCH_PAD_B, 9 * V_DS1_OUT], dtype=pl.BF16)
            im2col = _ds2_patch_b6_im2col_in(feat_pad, im2col)
            out = _ds2_b6_in(im2col, w, bias, out)
            return out

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self,
            hfeat: pl.Tensor[[tp_size, DS2_PATCH_RAW_ROWS_B6, V_DS1_OUT], pl.BF16],
            hw: pl.Tensor[[tp_size, 9 * V_DS1_OUT, V_DS2_OUT], pl.BF16],
            hbias: pl.Tensor[[tp_size, 1, V_DS2_OUT], pl.BF16],
            hto: pl.Out[pl.Tensor[[tp_size, V_TOKENS_FINAL_PATCH_PAD_B, V_DS2_OUT], pl.BF16]],
        ):
            for r in pl.range(pld.world_size()):
                self.per_rank(hfeat[r], hw[r], hbias[r], hto[r], device=r)
    return Ds2PatchB6Prog


def _build_proj_patch_b6(tp_size: int = TP_WORLD_SIZE):
    @pl.program
    class ProjPatchB6Prog:
        @pl.function(type=pl.FunctionType.Orchestration)
        def per_rank(self,
            x: pl.Tensor[[V_TOKENS_FINAL_PATCH_PAD_B, V_DS2_OUT], pl.BF16],
            w: pl.Tensor[[V_DS2_OUT, LM_HIDDEN], pl.BF16],
            out: pl.Out[pl.Tensor[[V_TOKENS_FINAL_PATCH_PAD_B, LM_HIDDEN], pl.BF16]],
        ) -> pl.Tensor[[V_TOKENS_FINAL_PATCH_PAD_B, LM_HIDDEN], pl.BF16]:
            out = _proj_b6_in(x, w, out); return out

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self,
            hx: pl.Tensor[[tp_size, V_TOKENS_FINAL_PATCH_PAD_B, V_DS2_OUT], pl.BF16],
            hw: pl.Tensor[[tp_size, V_DS2_OUT, LM_HIDDEN], pl.BF16],
            hto: pl.Out[pl.Tensor[[tp_size, V_TOKENS_FINAL_PATCH_PAD_B, LM_HIDDEN], pl.BF16]],
        ):
            for r in pl.range(pld.world_size()):
                self.per_rank(hx[r], hw[r], hto[r], device=r)
    return ProjPatchB6Prog


PatchEmbedPatchB6Prog = _build_patch_embed_patch_b6(TP_WORLD_SIZE)
LnPrePatchB6Prog = _build_ln_pre_patch_b6(TP_WORLD_SIZE)
Ds1PatchB6Prog = _build_ds1_patch_b6(TP_WORLD_SIZE)
Ds2PatchB6Prog = _build_ds2_patch_b6(TP_WORLD_SIZE)
ProjPatchB6Prog = _build_proj_patch_b6(TP_WORLD_SIZE)
