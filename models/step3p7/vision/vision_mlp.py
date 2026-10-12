# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Step3p7 vision MLP (``PerceptionEncoderMLP``: fc1 ColPar -> quick_gelu -> fc2
RowPar, step_vl.py:146). FUSED inline kernel: the [T, 8960] intermediate cannot
be materialised in UB, so fc1 -> quick_gelu -> fc2 are fused over I-chunks.

quick_gelu is inlined (not called via a helper) because the standalone
bare-jit AST parser requires typed params on called inline helpers.

The ``@pl.jit`` test wrapper + ``build_tensor_specs`` + golden adapter + driver
live in ``tests/step3p7/unit/test_vision_mlp.py``. This module holds only the
kernel body and the torch golden reference.
"""

import pypto.language as pl

from .vision_config import V_MLP_HIDDEN, V_WIDTH

T_DYN = pl.dynamic("T_DYN")

D = V_WIDTH                 # 1536 (in/out)
I = V_MLP_HIDDEN            # 8960 (intermediate)
# Cube constraints: M (T_TILE) 16-aligned; K_CHUNK & N_CHUNK & I_CHUNK such that
# each weight slab [K_CHUNK, I_CHUNK] / [I_CHUNK, N_CHUNK] (BF16) <= 64KB Right.
T_TILE = 16
K_CHUNK = 128              # fc1 K over D (D/K_CHUNK = 12)
I_CHUNK = 128             # intermediate tile (I/I_CHUNK = 70); [128,128] slab = 32KB
N_CHUNK = 128             # fc2 output tile (D/N_CHUNK = 12); [128,128] slab = 32KB
_QGELU_SLOPE = 1.702

assert D % K_CHUNK == 0
assert I % I_CHUNK == 0
assert D % N_CHUNK == 0


@pl.jit.inline
def vision_mlp(
    x: pl.Tensor[[T_DYN, D], pl.BF16],          # [T, 1536]
    w_fc1: pl.Tensor[[D, I], pl.BF16],          # [1536, 8960]
    w_fc2: pl.Tensor[[I, D], pl.BF16],          # [8960, 1536]
    b_fc1: pl.Tensor[[1, I], pl.BF16],         # fc1 bias [1, 8960]
    b_fc2: pl.Tensor[[1, D], pl.BF16],         # fc2 bias [1, 1536]
    out: pl.Tensor[[T_DYN, D], pl.BF16],         # [T, 1536]
):
    """Fused fc1 -> quick_gelu -> fc2, I-chunked (no full intermediate in UB).

    vLLM PerceptionEncoderMLP has bias=True for both fc1 (ColPar) and fc2 (RowPar).
    fc1 bias added after matmul_acc, before quick_gelu.
    fc2 bias added after all matmul_acc, before cast output.
    """
    t_dim = pl.tensor.dim(x, 0)
    k_blocks_d = D // K_CHUNK    # 12 (fc1 K-reduction over D)
    i_blocks = I // I_CHUNK       # 70
    n_blocks = D // N_CHUNK       # 12 (fc2 output N-tiles)
    for tg_idx in pl.spmd(t_dim // T_TILE, name_hint="vmlp"):
        tg = tg_idx * T_TILE
        for nb in pl.range(n_blocks):
            n0 = nb * N_CHUNK
            # ── ib = 0: init down_acc with the first fc2 matmul (not pl.full) ──
            i0 = 0
            x0 = pl.slice(x, [T_TILE, K_CHUNK], [tg, 0])
            w1_0 = pl.slice(w_fc1, [K_CHUNK, I_CHUNK], [0, i0])
            fc1_acc = pl.matmul(x0, w1_0, out_dtype=pl.FP32)
            for kb in pl.range(1, k_blocks_d):
                k0 = kb * K_CHUNK
                xk = pl.slice(x, [T_TILE, K_CHUNK], [tg, k0])
                w1_k = pl.slice(w_fc1, [K_CHUNK, I_CHUNK], [k0, i0])
                fc1_acc = pl.matmul_acc(fc1_acc, xk, w1_k)
            # add fc1 bias (before quick_gelu)
            lb1_0 = pl.slice(b_fc1, [1, I_CHUNK], [0, i0])
            ones1_0 = pl.full([T_TILE, I_CHUNK], dtype=pl.FP32, value=1.0)
            fc1_acc = pl.add(fc1_acc, pl.col_expand_mul(ones1_0, pl.cast(lb1_0, target_type=pl.FP32)))
            t = pl.mul(fc1_acc, _QGELU_SLOPE)
            sigmoid = pl.recip(pl.add(pl.exp(pl.neg(t)), 1.0))
            gelu_bf16 = pl.cast(
                pl.mul(fc1_acc, sigmoid), target_type=pl.BF16, mode="rint",
            )
            w2_slab = pl.slice(w_fc2, [I_CHUNK, N_CHUNK], [i0, n0])
            down_acc = pl.matmul(gelu_bf16, w2_slab, out_dtype=pl.FP32)   # init
            # ── ib = 1..: matmul_acc the rest ──────────────────────────────
            for ib in pl.range(1, i_blocks):
                i0 = ib * I_CHUNK
                x0b = pl.slice(x, [T_TILE, K_CHUNK], [tg, 0])
                w1_0b = pl.slice(w_fc1, [K_CHUNK, I_CHUNK], [0, i0])
                fc1_acc_b = pl.matmul(x0b, w1_0b, out_dtype=pl.FP32)
                for kb in pl.range(1, k_blocks_d):
                    k0 = kb * K_CHUNK
                    xk = pl.slice(x, [T_TILE, K_CHUNK], [tg, k0])
                    w1_k = pl.slice(w_fc1, [K_CHUNK, I_CHUNK], [k0, i0])
                    fc1_acc_b = pl.matmul_acc(fc1_acc_b, xk, w1_k)
                # add fc1 bias (before quick_gelu)
                lb1_b = pl.slice(b_fc1, [1, I_CHUNK], [0, i0])
                ones1_b = pl.full([T_TILE, I_CHUNK], dtype=pl.FP32, value=1.0)
                fc1_acc_b = pl.add(fc1_acc_b, pl.col_expand_mul(ones1_b, pl.cast(lb1_b, target_type=pl.FP32)))
                t_b = pl.mul(fc1_acc_b, _QGELU_SLOPE)
                sig_b = pl.recip(pl.add(pl.exp(pl.neg(t_b)), 1.0))
                gelu_b = pl.cast(
                    pl.mul(fc1_acc_b, sig_b), target_type=pl.BF16, mode="rint",
                )
                w2_slab_b = pl.slice(w_fc2, [I_CHUNK, N_CHUNK], [i0, n0])
                down_acc = pl.matmul_acc(down_acc, gelu_b, w2_slab_b)
            # add fc2 bias (before cast output)
            lb2 = pl.slice(b_fc2, [1, N_CHUNK], [0, n0])
            ones2 = pl.full([T_TILE, N_CHUNK], dtype=pl.FP32, value=1.0)
            down_acc = pl.add(down_acc, pl.col_expand_mul(ones2, pl.cast(lb2, target_type=pl.FP32)))
            out[tg : tg + T_TILE, n0 : n0 + N_CHUNK] = pl.cast(
                down_acc, target_type=pl.BF16, mode="rint",
            )
    return out


# ── torch golden reference (consumed by tests/step3p7/unit/test_vision_mlp.py) ──
def golden_vision_mlp(x, w_fc1, w_fc2, b_fc1=None, b_fc2=None):
    import torch
    h = x.float() @ w_fc1.float()
    if b_fc1 is not None:
        h = h + b_fc1.float()
    h = h * torch.sigmoid(_QGELU_SLOPE * h)
    h = h.to(torch.bfloat16).float()
    out = h @ w_fc2.float()
    if b_fc2 is not None:
        out = out + b_fc2.float()
    return out.to(torch.bfloat16)
