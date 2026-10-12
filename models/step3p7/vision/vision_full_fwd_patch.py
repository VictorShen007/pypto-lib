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
crop). The 47-layer patch TOWER is the replicated (REPL) flat/flat6 tower
(``vision_fwd_patch_repl_os_flat.py`` / ``vision_fwd_patch_repl_os_flat6.py``,
REPL host_orch contract from ``vision_fwd_repl_os.py``); this module holds the
front (patch_embed/ln_pre) and back (ds1/ds2/projector) stages that chain
around it.

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

    python -m models.step3p7.vision.vision_full_fwd_patch --ckpt ... --dump-root ... -p a2a3 -d 0
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


# ============================================================================
# E2E run driver (merged from models/step3p7/run_vision_full_patch.py).
# ============================================================================

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
    drift propagates through ds1/ds2/projector (mirror of vision_full_fwd.py sweep).
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
    p = argparse.ArgumentParser(description="Step3.7 VIT patch-path full-pipeline driver (crops -> image_features).")
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
        # REPL specs (vision_fwd_repl_os).
        from models.step3p7.vision.vision_fwd_repl_os import _build_specs as _build_specs_repl
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
