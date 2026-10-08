# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Step3p7 2D-RoPE for the vision attention. Inline kernel + host frequency
tables + torch golden reference. Mirrors vllm ``PerceptionEncoderRope2D``
(step_vl.py:65-133) + ``apply_rotary_emb`` (step_vl.py:39-62): head_dim split
in halves; full-head_dim ``rotate_half`` applied with cos/sin whose first half
encodes width-grid freqs and second half height-grid freqs.

``build_2d_rope_tables`` is also consumed by the L3 driver
(``tools/step3p7/run_l3_e2e.py``). The ``@pl.jit`` test wrapper + specs +
golden adapter + driver live in ``tests/step3p7/unit/test_rope2d.py``.
"""

import pypto.language as pl

from .vision_config import V_HEAD_DIM

T_DYN = pl.dynamic("T_DYN")

HD = V_HEAD_DIM          # 96
HALF = HD // 2            # 48
T_TILE = 8               # vector kernel (no cube), match rmsnorm T_TILE


def build_2d_rope_tables(grid_h: int, grid_w: int, head_dim: int = HD, theta: float = 10000.0):
    """Build 2D RoPE cos/sin tables **matching vLLM PerceptionEncoderRope2D**
    (interleaved rotate_half, NOT the old half-split form).

    Returns ``(cos_il, sin_signed)`` each ``[grid_h*grid_w, head_dim]`` FP32:
    - ``cos_il``: interleave-duplicated cos — each angle duplicated for the pair
      ``(j, j^1)`` so ``cos_il[2i]=cos_il[2i+1]=cos(theta_i)``.
    - ``sin_signed``: sign-folded sin — ``sin_signed[2i]=-sin(theta_i)``,
      ``sin_signed[2i+1]=+sin(theta_i)`` (sign=[-1,+1,-1,+1,...]).

    The rotation is then ``out[j] = x[j]*cos_il[j] + x[j^1]*sin_signed[j]``
    (j^1 = j XOR 1), the DSv4 ``rope_interleave`` form of vLLM's
    ``t*cos(freqs) + rotate_half(t)*sin(freqs)`` with interleaved pairs.

    freqs layout = cat([width_angles(dup), height_angles(dup)]) — first half (48)
    width, second half (48) height, each angle repeated for its pair — identical to
    vLLM ``_compute_2d_freqs`` (cat [freqs_w, freqs_h], repeat r=2).
    """
    import torch

    half = head_dim // 2          # 48
    q = half // 2                  # 24  (freq count per half)
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, q, dtype=torch.float32) / q)
    )  # [24]  (== vLLM _compute_inv_freq(base, dim//2): arange(0,48,2)[:24]/48)

    def half_ang(grid: torch.Tensor) -> torch.Tensor:
        ang = grid.unsqueeze(-1) * inv_freq.unsqueeze(0)        # [N, 24]
        return ang.repeat_interleave(2, dim=-1)                 # [N, 48] each angle dup

    rows = torch.arange(grid_h, dtype=torch.float32)
    cols = torch.arange(grid_w, dtype=torch.float32)
    rr, cc = torch.meshgrid(rows, cols, indexing="ij")
    fw = half_ang(cc.flatten())        # [n, 48]  width
    fh = half_ang(rr.flatten())        # [n, 48]  height
    freqs = torch.cat([fw, fh], dim=-1)    # [n, head_dim]
    cos_il = torch.cos(freqs)
    sign = torch.where(torch.arange(head_dim) % 2 == 0, -1.0, 1.0)   # [-1,+1,...]
    sin_signed = torch.sin(freqs) * sign
    return cos_il, sin_signed


@pl.jit.inline
def apply_2d_rope(
    qk: pl.Tensor[[T_DYN, HD], pl.BF16],          # one head's q or k, [T, 96]
    cos_il: pl.Tensor[[T_DYN, HD], pl.FP32],       # [T, 96] interleave-dup'd cos
    sin_signed: pl.Tensor[[T_DYN, HD], pl.FP32],   # [T, 96] sign-folded sin
    out: pl.Tensor[[T_DYN, HD], pl.BF16],           # [T, 96]
):
    """Apply interleaved 2D RoPE (vLLM form, DSv4 rope_interleave pattern):
    ``out[j] = qk[j]*cos_il[j] + qk[j^1]*sin_signed[j]``, ``j^1 = j XOR 1``.

    Mirrors the bug-free RoPE in ``vision_fwd.py`` ``vb2_all`` Pass B1 (the
    fix): swap index ``j^1`` built whole-tile [T_TILE,HD] via
    ``col_expand_mul`` + arithmetic, ``pl.gather`` for ``x[j^1]``, then the
    rotation APPLY is done **per-position [1,HD] inside ``pl.range(T_TILE)``**
    with ``col_expand_mul`` + ``pl.assemble``. The per-position inner loop has
    an ODD op count → PTOAS SyncSolver does not split it → all intra-iteration
    ``pipe_barrier``s are generated → no RAW hazard; and the [1,HD] form +
    per-position assemble avoids the N%64!=0 fractal/store bug that the old
    whole-tile [T_TILE,96] apply hit at HD=96. Equivalent to vLLM
    ``t*cos(freqs) + rotate_half(t)*sin(freqs)`` with interleaved pairs.
    """
    t_dim = pl.tensor.dim(qk, 0)
    for tg_idx in pl.spmd(t_dim // T_TILE, name_hint="rope2d"):
        tg = tg_idx * T_TILE
        # j^1 = j XOR 1 (pair partner), built whole-tile [T_TILE, HD] (DSv4 form)
        ones = pl.full([T_TILE, HD], dtype=pl.FP32, value=1.0)
        col = pl.col_expand_mul(ones, pl.cast(pl.arange(0, [1, HD], dtype=pl.INT32), target_type=pl.FP32))  # j
        dup = pl.cast(pl.cast(pl.mul(col, 0.5), target_type=pl.INT32, mode="trunc"), target_type=pl.FP32)    # j//2
        lane = pl.sub(col, pl.mul(dup, 2.0))                                                                       # j%2
        swap = pl.cast(pl.sub(pl.add(col, 1.0), pl.mul(lane, 2.0)), target_type=pl.INT32)                          # j^1
        xf = pl.cast(qk[tg : tg + T_TILE, 0:HD], target_type=pl.FP32)        # [T_TILE, HD]
        xs = pl.gather(xf, dim=-1, index=swap)                               # [T_TILE, HD]  x[j^1]
        # per-position [1,HD] apply (ODD op count → barriers generated)
        for qi in pl.range(T_TILE):
            c1 = pl.slice(cos_il, [1, HD], [tg + qi, 0])
            s1 = pl.slice(sin_signed, [1, HD], [tg + qi, 0])
            x1 = pl.slice(xf, [1, HD], [qi, 0])
            xs1 = pl.slice(xs, [1, HD], [qi, 0])
            out = pl.assemble(
                out,
                pl.cast(
                    pl.add(pl.col_expand_mul(x1, c1), pl.col_expand_mul(xs1, s1)),
                    target_type=pl.BF16, mode="rint",
                ),
                [tg + qi, 0],
            )
    return out


# ── torch golden reference (consumed by tests/step3p7/unit/test_rope2d.py) ─────
def golden_apply_2d_rope(qk, cos_il, sin_signed):
    """Torch interleaved RoPE reference (vLLM rotate_half form via swap-gather):
    ``out[j] = x[j]*cos_il[j] + x[j^1]*sin_signed[j]``, ``j^1 = j XOR 1``."""
    import torch
    x = qk.float()
    partner_idx = torch.arange(x.shape[-1]) ^ 1   # j^1
    partner = x[..., partner_idx]                   # x[j^1]
    out = x * cos_il.float() + partner * sin_signed.float()
    return out.to(torch.bfloat16)
