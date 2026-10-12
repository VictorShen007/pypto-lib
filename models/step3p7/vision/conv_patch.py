# Copyright (c) PyPTO Contributors.
import pypto.language as pl
from .vision_config import V_PATCH, V_TOKENS, V_WIDTH

PATCH_FEAT = 3 * V_PATCH * V_PATCH     # 588
PATCH_FEAT_PAD = 592                    # zero-padded to 16-align
D = V_WIDTH                             # 1536

assert PATCH_FEAT_PAD % 16 == 0
assert D % 32 == 0

# Host zero-pads both reduction and row axes so the kernel sees uniform tiles:
#   K   592 -> PE_K_PAD   = 768 (3x256; KC=256 = L2 cache line, pipelineable)
#   rows 2704 (= 84*32+16) -> PE_ROWS_PAD = 2880 (30 groups of 96 = 3x M32).
# Zero rows/columns contribute nothing; the first V_TOKENS output rows are the
# valid result (the caller slices them off).
PE_K_PAD = 768
PE_ROWS_PAD = 2880

assert PE_K_PAD % 256 == 0 and PE_K_PAD >= PATCH_FEAT_PAD
assert PE_ROWS_PAD % 96 == 0 and PE_ROWS_PAD >= V_TOKENS

# Current recipe: NT=128 output slice, K host-zero-padded so the inner K loop
# pipelines (stage=2), 2D spmd split (12 N-tiles x 2 row-shares), and a 3-way
# M32 row-tile unroll that shares each wt chunk across three adjacent tiles.
# 24 tasks = exactly one full wave on the 24 cube cores. Constraint walls:
# wt chunk [256,128] bf16 = 64KB = L0B cap; Acc 3x[32,128] fp32 = 48KB < 128KB;
# the store-bearing row loop stays serial (pipelining it crashes the compiler).

@pl.jit.inline
def patch_embed(
    x_patches: pl.Tensor[[PE_ROWS_PAD, PE_K_PAD], pl.BF16],
    w_proj: pl.Tensor[[PE_K_PAD, D], pl.BF16],
    out: pl.Tensor[[PE_ROWS_PAD, D], pl.BF16],
):
    # 24 tasks = 12 N-tiles x 2 row-shares: exactly one full wave on the 24
    # cube cores (any extra tasks only add scheduling/wave-tail overhead).
    for nb_idx in pl.spmd(24, name_hint="patch_embed"):
        r = nb_idx // 12
        n0 = (nb_idx - r * 12) * 128
        rb = r * 1440                                                # rows/share
        for tg_idx in pl.range(15):                                  # 15*96 = 1440
            tg = rb + tg_idx * 96
            acc = pl.matmul(x_patches[tg : tg + 32, 0:256], w_proj[0:256, n0:n0 + 128], out_dtype=pl.FP32)
            acc2 = pl.matmul(x_patches[tg + 32 : tg + 64, 0:256], w_proj[0:256, n0:n0 + 128], out_dtype=pl.FP32)
            acc3 = pl.matmul(x_patches[tg + 64 : tg + 96, 0:256], w_proj[0:256, n0:n0 + 128], out_dtype=pl.FP32)
            for kb in pl.pipeline(2, stage=2):                       # chunks 1..2
                k0 = (kb + 1) * 256
                wt = w_proj[k0:k0 + 256, n0:n0 + 128]
                acc = pl.matmul_acc(acc, x_patches[tg : tg + 32, k0:k0 + 256], wt)
                acc2 = pl.matmul_acc(acc2, x_patches[tg + 32 : tg + 64, k0:k0 + 256], wt)
                acc3 = pl.matmul_acc(acc3, x_patches[tg + 64 : tg + 96, k0:k0 + 256], wt)
            out[tg : tg + 32, n0 : n0 + 128] = pl.cast(acc, target_type=pl.BF16, mode="rint")
            out[tg + 32 : tg + 64, n0 : n0 + 128] = pl.cast(acc2, target_type=pl.BF16, mode="rint")
            out[tg + 64 : tg + 96, n0 : n0 + 128] = pl.cast(acc3, target_type=pl.BF16, mode="rint")
    return out


# ── torch golden reference (consumed by tests/step3p7/unit/test_conv_patch.py) ──
# patch_embed is im2col(@host) -> matmul; golden is the same matmul in FP32.
def golden_patch_embed(x_patches, w_proj):
    import torch
    return (x_patches.float() @ w_proj.float()).to(torch.bfloat16)   # [V_TOKENS, 1536]
