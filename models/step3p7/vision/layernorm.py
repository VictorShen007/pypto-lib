# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Step3p7 vision LayerNorm (mean + variance + gamma + beta), inline kernel +
torch golden reference. Mirrors ``vllm.model_executor.models.step_vl.
_DEFAULT_NORM_LAYER = partial(nn.LayerNorm, eps=1e-5)``.

The ``@pl.jit`` test wrapper + ``build_tensor_specs`` + golden adapter + driver
live in ``tests/step3p7/unit/test_layernorm.py`` (step3p5 layout). This module
holds only the kernel body and the torch golden reference.
"""

import pypto.language as pl

from .vision_config import V_D_TILE, V_EPS, V_T_TILE, V_WIDTH, V_WIDTH_INV

# Dynamic token dimension (vision token count varies: 2704 patches / 169 final).
T_DYN = pl.dynamic("T_DYN")

# model config
D = V_WIDTH
D_TILE = V_D_TILE
T_TILE = V_T_TILE
assert D % D_TILE == 0, "V_WIDTH must be divisible by V_D_TILE"


@pl.jit.inline
def layernorm(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    gamma: pl.Tensor[[D], pl.FP32],
    beta: pl.Tensor[[D], pl.FP32],
    x_normed: pl.Tensor[[T_DYN, D], pl.BF16],
):
    """LayerNorm: ``gamma * (x - mean) / sqrt(var + eps) + beta`` per row.

    Uses only confirmed pypto primitives (row_sum / row_expand_mul /
    col_expand_mul / add / sub / mul / rsqrt / cast). Mean and variance are
    reduced over the full width D via a pipeline loop, then the per-row
    ``inv_std`` is broadcast back across D for the apply.
    """
    t_dim = pl.tensor.dim(x, 0)
    for tg_idx in pl.spmd(t_dim // T_TILE, name_hint="layernorm"):
        tg = tg_idx * T_TILE

            # ── reduce sum_x and sum_x2 over D ────────────────────────────────
        sum_x = pl.full([1, T_TILE], dtype=pl.FP32, value=0.0)
        sum_x2 = pl.full([1, T_TILE], dtype=pl.FP32, value=0.0)
        for db in pl.pipeline(D // D_TILE, stage=2):
            d0 = db * D_TILE
            x_chunk = pl.cast(
                x[tg : tg + T_TILE, d0 : d0 + D_TILE], target_type=pl.FP32,
            )
            sum_x = pl.add(
                sum_x, pl.reshape(pl.row_sum(x_chunk), [1, T_TILE]),
            )
            sum_x2 = pl.add(
                sum_x2,
                pl.reshape(pl.row_sum(pl.mul(x_chunk, x_chunk)), [1, T_TILE]),
            )

            # ── per-row mean / var / inv_std ──────────────────────────────────
        mean_col = pl.mul(sum_x, V_WIDTH_INV)                       # [1, T]
        mean_sq_col = pl.mul(sum_x2, V_WIDTH_INV)                    # [1, T]
        var_col = pl.sub(mean_sq_col, pl.mul(mean_col, mean_col))   # [1, T]
            # pl.rsqrt is the low-precision hardware vrsqrt (~1e-3 rel err, no
        # Newton refinement) — one NR step takes inv_std to ~1 ULP FP32:
            #   inv_std = iv0 * (3 - vpe * iv0^2) * 0.5
        vpe = pl.add(var_col, V_EPS)                                # [1, T]
        iv0 = pl.rsqrt(vpe)                                         # [1, T]
        iv2 = pl.mul(iv0, iv0)
        inv_std_col = pl.mul(
            iv0,
            pl.mul(pl.full([1, T_TILE], dtype=pl.FP32, value=0.5),
                   pl.sub(pl.full([1, T_TILE], dtype=pl.FP32, value=3.0),
                          pl.mul(vpe, iv2))),
        )                                                           # [1, T]
        mean_inv_col = pl.mul(mean_col, inv_std_col)                # [1, T]  mean*inv_std

        inv_std_t = pl.reshape(inv_std_col, [T_TILE, 1])
        mean_inv_t = pl.reshape(mean_inv_col, [T_TILE, 1])

            # ── apply: (x - mean) * inv_std * gamma + beta, per D-tile ─────────
        for apply_db in pl.pipeline(D // D_TILE, stage=2):
            d0 = apply_db * D_TILE
            x_chunk = pl.cast(
                x[tg : tg + T_TILE, d0 : d0 + D_TILE], target_type=pl.FP32,
            )
            gamma_chunk = pl.reshape(gamma[d0 : d0 + D_TILE], [1, D_TILE])
            beta_chunk = pl.reshape(beta[d0 : d0 + D_TILE], [1, D_TILE])
            ones = pl.full([T_TILE, D_TILE], dtype=pl.FP32, value=1.0)
            term1 = pl.row_expand_mul(x_chunk, inv_std_t)          # x * inv_std
            term2 = pl.row_expand_mul(ones, mean_inv_t)            # mean * inv_std
            centered_scaled = pl.sub(term1, term2)                 # (x - mean) * inv_std
            normed = pl.col_expand_mul(centered_scaled, gamma_chunk)   # * gamma
            beta_exp = pl.col_expand_mul(ones, beta_chunk)         # beta broadcast
            out_chunk = pl.add(normed, beta_exp)
            x_normed[tg : tg + T_TILE, d0 : d0 + D_TILE] = pl.cast(
                out_chunk, target_type=pl.BF16, mode="rint",
            )

    return x_normed


# B6 (6-crop batched) granularity: T_TILE=8 costs 211us on 7776 rows because
# 972 spmd tasks of 12 small chunk-iterations are dispatch-dominated (~83MB
# traffic would be ~52us). Sweep (two-read structure unchanged):
#   t8d256(committed)=211.5 t8d1536=221.7 t16d256=121.2 t16d512=115.7
#   t16d768=119.7 t24d512=93.3* t32d256=107.5 t32d384=98.6 t48d256=101.9
#   t48d128=109.7 t72d128=127.0 t96d128=106.4
# Sweet spot: <=~324 tasks AND few pipeline iterations. Tile size is capped
# by the Vec buffer wall (188416B/core, ~4 live tiles): [32,512] fp32 and
# [36,384] fp32 exceed it and fail to compile. Global path (2704 rows) keeps
# T_TILE=8 (2704 = 338*8; not divisible by 24).


# B6 granularity constants (see sweep table below).
LN_T_TILE_B6 = 24
LN_D_TILE_B6 = 512


@pl.jit.inline
def layernorm_b6(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    gamma: pl.Tensor[[D], pl.FP32],
    beta: pl.Tensor[[D], pl.FP32],
    x_normed: pl.Tensor[[T_DYN, D], pl.BF16],
):
    """B6-granularity LayerNorm (T_TILE=24, D_TILE=512); same math as
    :func:`layernorm` — only the spmd/pipeline granularity differs.

    LN_T_TILE_B6 / LN_D_TILE_B6 must also be importable at the ``pl.inline``
    call site (the inline tracer resolves free names in the caller's frame,
    not this module's globals)."""
    t_dim = pl.tensor.dim(x, 0)
    for tg_idx in pl.spmd(t_dim // LN_T_TILE_B6):
        tg = tg_idx * LN_T_TILE_B6

        # ── reduce sum_x and sum_x2 over D ────────────────────────────────
        sum_x = pl.full([1, LN_T_TILE_B6], dtype=pl.FP32, value=0.0)
        sum_x2 = pl.full([1, LN_T_TILE_B6], dtype=pl.FP32, value=0.0)
        for db in pl.pipeline(D // LN_D_TILE_B6, stage=2):
            d0 = db * LN_D_TILE_B6
            x_chunk = pl.cast(
                x[tg : tg + LN_T_TILE_B6, d0 : d0 + LN_D_TILE_B6], target_type=pl.FP32,
            )
            sum_x = pl.add(
                sum_x, pl.reshape(pl.row_sum(x_chunk), [1, LN_T_TILE_B6]),
            )
            sum_x2 = pl.add(
                sum_x2,
                pl.reshape(pl.row_sum(pl.mul(x_chunk, x_chunk)), [1, LN_T_TILE_B6]),
            )

        # ── per-row mean / var / inv_std ──────────────────────────────────
        mean_col = pl.mul(sum_x, V_WIDTH_INV)                       # [1, T]
        mean_sq_col = pl.mul(sum_x2, V_WIDTH_INV)                    # [1, T]
        var_col = pl.sub(mean_sq_col, pl.mul(mean_col, mean_col))   # [1, T]
        # pl.rsqrt is the low-precision hardware vrsqrt (~1e-3 rel err, no
        # Newton refinement) — one NR step takes inv_std to ~1 ULP FP32:
        #   inv_std = iv0 * (3 - vpe * iv0^2) * 0.5
        vpe = pl.add(var_col, V_EPS)                                # [1, T]
        iv0 = pl.rsqrt(vpe)                                         # [1, T]
        iv2 = pl.mul(iv0, iv0)
        inv_std_col = pl.mul(
            iv0,
            pl.mul(pl.full([1, LN_T_TILE_B6], dtype=pl.FP32, value=0.5),
                   pl.sub(pl.full([1, LN_T_TILE_B6], dtype=pl.FP32, value=3.0),
                          pl.mul(vpe, iv2))),
        )                                                           # [1, T]
        mean_inv_col = pl.mul(mean_col, inv_std_col)                # [1, T]  mean*inv_std

        inv_std_t = pl.reshape(inv_std_col, [LN_T_TILE_B6, 1])
        mean_inv_t = pl.reshape(mean_inv_col, [LN_T_TILE_B6, 1])

        # ── apply: (x - mean) * inv_std * gamma + beta, per D-tile ─────────
        for apply_db in pl.pipeline(D // LN_D_TILE_B6, stage=2):
            d0 = apply_db * LN_D_TILE_B6
            x_chunk = pl.cast(
                x[tg : tg + LN_T_TILE_B6, d0 : d0 + LN_D_TILE_B6], target_type=pl.FP32,
            )
            gamma_chunk = pl.reshape(gamma[d0 : d0 + LN_D_TILE_B6], [1, LN_D_TILE_B6])
            beta_chunk = pl.reshape(beta[d0 : d0 + LN_D_TILE_B6], [1, LN_D_TILE_B6])
            ones = pl.full([LN_T_TILE_B6, LN_D_TILE_B6], dtype=pl.FP32, value=1.0)
            term1 = pl.row_expand_mul(x_chunk, inv_std_t)          # x * inv_std
            term2 = pl.row_expand_mul(ones, mean_inv_t)            # mean * inv_std
            centered_scaled = pl.sub(term1, term2)                 # (x - mean) * inv_std
            normed = pl.col_expand_mul(centered_scaled, gamma_chunk)   # * gamma
            beta_exp = pl.col_expand_mul(ones, beta_chunk)         # beta broadcast
            out_chunk = pl.add(normed, beta_exp)
            x_normed[tg : tg + LN_T_TILE_B6, d0 : d0 + LN_D_TILE_B6] = pl.cast(
                out_chunk, target_type=pl.BF16, mode="rint",
            )

    return x_normed


# ── torch golden reference (consumed by tests/step3p7/unit/test_layernorm.py) ──
def golden_layernorm(x, gamma, beta):
    import torch

    return torch.nn.functional.layer_norm(
        x.float(), (D,), gamma.float(), beta.float(), V_EPS,
    ).to(torch.bfloat16)
