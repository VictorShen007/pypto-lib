# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Prefill-side shared-expert body — Step3.5 shared FFN (TP-sliced).

v2 lift (task #13): the ``_expert_shared_local`` method body from
``prefill_moe.py:1063-1141`` is lifted here as two module-level
``@pl.jit.inline`` bodies, split by activation:

  - ``prefill_expert_shared_silu``     — shared_lim=0.0 (silu gate)
  - ``prefill_expert_shared_swiglu16`` — shared_lim=16.0 (SwigluStep clamp)

The split replaces the runtime ``if _shared_swiglu_step`` branch in the
original body (lines 1109-1117) with compile-time body selection.

Math is unchanged. Lowercase closure variables (sh_inter_local) are
replaced with uppercase module-level constants from ``_moe_constants.py``.
"""

from __future__ import annotations

import pypto.language as pl

from ._moe_constants import (
    BATCH,
    HIDDEN,
    SH_INTER_LOCAL,
    SHARED_DOWN_N_CHUNK,
    SHARED_GATE_K_CHUNK,
    SHARED_SWIGLU_LIMIT,
    SHARED_SWIGLU_N_CHUNK,
)

__all__ = [
    "prefill_expert_shared_silu",
    "prefill_expert_shared_swiglu16",
    "select_prefill_expert_shared",
]


def select_prefill_expert_shared(shared_lim: float):
    """Return the right @pl.jit.inline body for the layer's shared_lim.

    Mirrors the factory-build-time ``_shared_swiglu_step`` closure choice
    in ``prefill_moe.py:217-224``.
    """
    if shared_lim == 0.0:
        return prefill_expert_shared_silu
    if shared_lim == 16.0:
        return prefill_expert_shared_swiglu16
    raise ValueError(
        f"shared_lim must be 0.0 or 16.0, got {shared_lim}",
    )


@pl.jit.inline
def prefill_expert_shared_silu(
    x: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
    w_gate: pl.Tensor[[HIDDEN, SH_INTER_LOCAL], pl.BF16],
    w_up: pl.Tensor[[HIDDEN, SH_INTER_LOCAL], pl.BF16],
    w_down: pl.Tensor[[SH_INTER_LOCAL, HIDDEN], pl.BF16],
    sh_y_shard: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
):
    """Shared-expert FFN with silu gate (shared_lim=0.0).

    NCHUNK-CLAMP-FIX: the full [BATCH,160] Vec-tile cast is miscompiled
    (wide-tile tmov misprune -> ~45% wrong) because 160 crosses a
    128-column block boundary. Compute 5 narrow [BATCH,32] chunks
    (h_c0..h_c4) as separate cast-result tiles and feed them straight into
    the down-proj K-loop, mirroring decode moe.py's [32]x5 pattern.
    """
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="sh_mlp"):
        x0_0 = pl.slice(x, [BATCH, SHARED_GATE_K_CHUNK], [0, 0])
        wg0_0 = pl.slice(
            w_gate, [SHARED_GATE_K_CHUNK, SHARED_SWIGLU_N_CHUNK], [0, 0],
        )
        wu0_0 = pl.slice(
            w_up, [SHARED_GATE_K_CHUNK, SHARED_SWIGLU_N_CHUNK], [0, 0],
        )
        gate_acc_0 = pl.matmul(x0_0, wg0_0, out_dtype=pl.FP32)
        up_acc_0 = pl.matmul(x0_0, wu0_0, out_dtype=pl.FP32)
        for kb in pl.range(1, HIDDEN // SHARED_GATE_K_CHUNK):
            k0 = kb * SHARED_GATE_K_CHUNK
            xk_0 = pl.slice(x, [BATCH, SHARED_GATE_K_CHUNK], [0, k0])
            wgk_0 = pl.slice(
                w_gate, [SHARED_GATE_K_CHUNK, SHARED_SWIGLU_N_CHUNK], [k0, 0],
            )
            wuk_0 = pl.slice(
                w_up, [SHARED_GATE_K_CHUNK, SHARED_SWIGLU_N_CHUNK], [k0, 0],
            )
            gate_acc_0 = pl.matmul_acc(gate_acc_0, xk_0, wgk_0)
            up_acc_0 = pl.matmul_acc(up_acc_0, xk_0, wuk_0)
        sigmoid_0 = pl.recip(pl.add(pl.exp(pl.neg(gate_acc_0)), 1.0))
        silu_0 = pl.mul(gate_acc_0, sigmoid_0)
        gated_0 = pl.mul(silu_0, up_acc_0)
        h_c0 = pl.cast(gated_0, target_type=pl.BF16)

        x0_1 = pl.slice(x, [BATCH, SHARED_GATE_K_CHUNK], [0, 0])
        wg0_1 = pl.slice(
            w_gate, [SHARED_GATE_K_CHUNK, SHARED_SWIGLU_N_CHUNK], [0, 32],
        )
        wu0_1 = pl.slice(
            w_up, [SHARED_GATE_K_CHUNK, SHARED_SWIGLU_N_CHUNK], [0, 32],
        )
        gate_acc_1 = pl.matmul(x0_1, wg0_1, out_dtype=pl.FP32)
        up_acc_1 = pl.matmul(x0_1, wu0_1, out_dtype=pl.FP32)
        for kb in pl.range(1, HIDDEN // SHARED_GATE_K_CHUNK):
            k0 = kb * SHARED_GATE_K_CHUNK
            xk_1 = pl.slice(x, [BATCH, SHARED_GATE_K_CHUNK], [0, k0])
            wgk_1 = pl.slice(
                w_gate, [SHARED_GATE_K_CHUNK, SHARED_SWIGLU_N_CHUNK], [k0, 32],
            )
            wuk_1 = pl.slice(
                w_up, [SHARED_GATE_K_CHUNK, SHARED_SWIGLU_N_CHUNK], [k0, 32],
            )
            gate_acc_1 = pl.matmul_acc(gate_acc_1, xk_1, wgk_1)
            up_acc_1 = pl.matmul_acc(up_acc_1, xk_1, wuk_1)
        sigmoid_1 = pl.recip(pl.add(pl.exp(pl.neg(gate_acc_1)), 1.0))
        silu_1 = pl.mul(gate_acc_1, sigmoid_1)
        gated_1 = pl.mul(silu_1, up_acc_1)
        h_c1 = pl.cast(gated_1, target_type=pl.BF16)

        x0_2 = pl.slice(x, [BATCH, SHARED_GATE_K_CHUNK], [0, 0])
        wg0_2 = pl.slice(
            w_gate, [SHARED_GATE_K_CHUNK, SHARED_SWIGLU_N_CHUNK], [0, 64],
        )
        wu0_2 = pl.slice(
            w_up, [SHARED_GATE_K_CHUNK, SHARED_SWIGLU_N_CHUNK], [0, 64],
        )
        gate_acc_2 = pl.matmul(x0_2, wg0_2, out_dtype=pl.FP32)
        up_acc_2 = pl.matmul(x0_2, wu0_2, out_dtype=pl.FP32)
        for kb in pl.range(1, HIDDEN // SHARED_GATE_K_CHUNK):
            k0 = kb * SHARED_GATE_K_CHUNK
            xk_2 = pl.slice(x, [BATCH, SHARED_GATE_K_CHUNK], [0, k0])
            wgk_2 = pl.slice(
                w_gate, [SHARED_GATE_K_CHUNK, SHARED_SWIGLU_N_CHUNK], [k0, 64],
            )
            wuk_2 = pl.slice(
                w_up, [SHARED_GATE_K_CHUNK, SHARED_SWIGLU_N_CHUNK], [k0, 64],
            )
            gate_acc_2 = pl.matmul_acc(gate_acc_2, xk_2, wgk_2)
            up_acc_2 = pl.matmul_acc(up_acc_2, xk_2, wuk_2)
        sigmoid_2 = pl.recip(pl.add(pl.exp(pl.neg(gate_acc_2)), 1.0))
        silu_2 = pl.mul(gate_acc_2, sigmoid_2)
        gated_2 = pl.mul(silu_2, up_acc_2)
        h_c2 = pl.cast(gated_2, target_type=pl.BF16)

        x0_3 = pl.slice(x, [BATCH, SHARED_GATE_K_CHUNK], [0, 0])
        wg0_3 = pl.slice(
            w_gate, [SHARED_GATE_K_CHUNK, SHARED_SWIGLU_N_CHUNK], [0, 96],
        )
        wu0_3 = pl.slice(
            w_up, [SHARED_GATE_K_CHUNK, SHARED_SWIGLU_N_CHUNK], [0, 96],
        )
        gate_acc_3 = pl.matmul(x0_3, wg0_3, out_dtype=pl.FP32)
        up_acc_3 = pl.matmul(x0_3, wu0_3, out_dtype=pl.FP32)
        for kb in pl.range(1, HIDDEN // SHARED_GATE_K_CHUNK):
            k0 = kb * SHARED_GATE_K_CHUNK
            xk_3 = pl.slice(x, [BATCH, SHARED_GATE_K_CHUNK], [0, k0])
            wgk_3 = pl.slice(
                w_gate, [SHARED_GATE_K_CHUNK, SHARED_SWIGLU_N_CHUNK], [k0, 96],
            )
            wuk_3 = pl.slice(
                w_up, [SHARED_GATE_K_CHUNK, SHARED_SWIGLU_N_CHUNK], [k0, 96],
            )
            gate_acc_3 = pl.matmul_acc(gate_acc_3, xk_3, wgk_3)
            up_acc_3 = pl.matmul_acc(up_acc_3, xk_3, wuk_3)
        sigmoid_3 = pl.recip(pl.add(pl.exp(pl.neg(gate_acc_3)), 1.0))
        silu_3 = pl.mul(gate_acc_3, sigmoid_3)
        gated_3 = pl.mul(silu_3, up_acc_3)
        h_c3 = pl.cast(gated_3, target_type=pl.BF16)

        x0_4 = pl.slice(x, [BATCH, SHARED_GATE_K_CHUNK], [0, 0])
        wg0_4 = pl.slice(
            w_gate, [SHARED_GATE_K_CHUNK, SHARED_SWIGLU_N_CHUNK], [0, 128],
        )
        wu0_4 = pl.slice(
            w_up, [SHARED_GATE_K_CHUNK, SHARED_SWIGLU_N_CHUNK], [0, 128],
        )
        gate_acc_4 = pl.matmul(x0_4, wg0_4, out_dtype=pl.FP32)
        up_acc_4 = pl.matmul(x0_4, wu0_4, out_dtype=pl.FP32)
        for kb in pl.range(1, HIDDEN // SHARED_GATE_K_CHUNK):
            k0 = kb * SHARED_GATE_K_CHUNK
            xk_4 = pl.slice(x, [BATCH, SHARED_GATE_K_CHUNK], [0, k0])
            wgk_4 = pl.slice(
                w_gate, [SHARED_GATE_K_CHUNK, SHARED_SWIGLU_N_CHUNK], [k0, 128],
            )
            wuk_4 = pl.slice(
                w_up, [SHARED_GATE_K_CHUNK, SHARED_SWIGLU_N_CHUNK], [k0, 128],
            )
            gate_acc_4 = pl.matmul_acc(gate_acc_4, xk_4, wgk_4)
            up_acc_4 = pl.matmul_acc(up_acc_4, xk_4, wuk_4)
        sigmoid_4 = pl.recip(pl.add(pl.exp(pl.neg(gate_acc_4)), 1.0))
        silu_4 = pl.mul(gate_acc_4, sigmoid_4)
        gated_4 = pl.mul(silu_4, up_acc_4)
        h_c4 = pl.cast(gated_4, target_type=pl.BF16)

        for db in pl.range(HIDDEN // SHARED_DOWN_N_CHUNK):
            d0 = db * SHARED_DOWN_N_CHUNK
            wd_c0 = pl.slice(
                w_down, [SHARED_SWIGLU_N_CHUNK, SHARED_DOWN_N_CHUNK], [0, d0],
            )
            y_acc = pl.matmul(h_c0, wd_c0, out_dtype=pl.FP32)
            wd_c1 = pl.slice(
                w_down, [SHARED_SWIGLU_N_CHUNK, SHARED_DOWN_N_CHUNK], [32, d0],
            )
            y_acc = pl.matmul_acc(y_acc, h_c1, wd_c1)
            wd_c2 = pl.slice(
                w_down, [SHARED_SWIGLU_N_CHUNK, SHARED_DOWN_N_CHUNK], [64, d0],
            )
            y_acc = pl.matmul_acc(y_acc, h_c2, wd_c2)
            wd_c3 = pl.slice(
                w_down, [SHARED_SWIGLU_N_CHUNK, SHARED_DOWN_N_CHUNK], [96, d0],
            )
            y_acc = pl.matmul_acc(y_acc, h_c3, wd_c3)
            wd_c4 = pl.slice(
                w_down, [SHARED_SWIGLU_N_CHUNK, SHARED_DOWN_N_CHUNK], [128, d0],
            )
            y_acc = pl.matmul_acc(y_acc, h_c4, wd_c4)
            sh_y_shard = pl.assemble(
                sh_y_shard,
                pl.cast(y_acc, target_type=pl.BF16),
                [0, d0],
            )

    return sh_y_shard


@pl.jit.inline
def prefill_expert_shared_swiglu16(
    x: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
    w_gate: pl.Tensor[[HIDDEN, SH_INTER_LOCAL], pl.BF16],
    w_up: pl.Tensor[[HIDDEN, SH_INTER_LOCAL], pl.BF16],
    w_down: pl.Tensor[[SH_INTER_LOCAL, HIDDEN], pl.BF16],
    sh_y_shard: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
):
    """Shared-expert FFN with SwigluStep@16.0 clamp (shared_lim=16.0).

    Pure-scalar specialization: shared_lim=16.0 is baked into the body
    at factory-build time, so the clamp is a compile-time constant
    (pl.minimum/pl.maximum with SHARED_SWIGLU_LIMIT) — no runtime
    branch. The companion ``prefill_expert_shared_silu`` is the
    shared_lim=0.0 (no-clamp) variant; ``select_prefill_expert_shared``
    picks one per layer. See module docstring for the split rationale.

    NCHUNK-CLAMP-FIX: same [32]x5 chunking as the silu variant — the full
    [BATCH,160] Vec-tile clamp+cast crosses a 128-column block boundary and
    is miscompiled (wide-tile tmov misprune), so each [BATCH,32] chunk is
    clamped and cast independently.
    """
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="sh_mlp"):
        x0_0 = pl.slice(x, [BATCH, SHARED_GATE_K_CHUNK], [0, 0])
        wg0_0 = pl.slice(
            w_gate, [SHARED_GATE_K_CHUNK, SHARED_SWIGLU_N_CHUNK], [0, 0],
        )
        wu0_0 = pl.slice(
            w_up, [SHARED_GATE_K_CHUNK, SHARED_SWIGLU_N_CHUNK], [0, 0],
        )
        gate_acc_0 = pl.matmul(x0_0, wg0_0, out_dtype=pl.FP32)
        up_acc_0 = pl.matmul(x0_0, wu0_0, out_dtype=pl.FP32)
        for kb in pl.range(1, HIDDEN // SHARED_GATE_K_CHUNK):
            k0 = kb * SHARED_GATE_K_CHUNK
            xk_0 = pl.slice(x, [BATCH, SHARED_GATE_K_CHUNK], [0, k0])
            wgk_0 = pl.slice(
                w_gate, [SHARED_GATE_K_CHUNK, SHARED_SWIGLU_N_CHUNK], [k0, 0],
            )
            wuk_0 = pl.slice(
                w_up, [SHARED_GATE_K_CHUNK, SHARED_SWIGLU_N_CHUNK], [k0, 0],
            )
            gate_acc_0 = pl.matmul_acc(gate_acc_0, xk_0, wgk_0)
            up_acc_0 = pl.matmul_acc(up_acc_0, xk_0, wuk_0)
        sigmoid_0 = pl.recip(pl.add(pl.exp(pl.neg(gate_acc_0)), 1.0))
        silu_0 = pl.mul(gate_acc_0, sigmoid_0)
        silu_c_0 = pl.minimum(silu_0, SHARED_SWIGLU_LIMIT)
        up_c_0 = pl.maximum(
            pl.minimum(up_acc_0, SHARED_SWIGLU_LIMIT),
            -SHARED_SWIGLU_LIMIT,
        )
        gated_0 = pl.mul(silu_c_0, up_c_0)
        h_c0 = pl.cast(gated_0, target_type=pl.BF16)

        x0_1 = pl.slice(x, [BATCH, SHARED_GATE_K_CHUNK], [0, 0])
        wg0_1 = pl.slice(
            w_gate, [SHARED_GATE_K_CHUNK, SHARED_SWIGLU_N_CHUNK], [0, 32],
        )
        wu0_1 = pl.slice(
            w_up, [SHARED_GATE_K_CHUNK, SHARED_SWIGLU_N_CHUNK], [0, 32],
        )
        gate_acc_1 = pl.matmul(x0_1, wg0_1, out_dtype=pl.FP32)
        up_acc_1 = pl.matmul(x0_1, wu0_1, out_dtype=pl.FP32)
        for kb in pl.range(1, HIDDEN // SHARED_GATE_K_CHUNK):
            k0 = kb * SHARED_GATE_K_CHUNK
            xk_1 = pl.slice(x, [BATCH, SHARED_GATE_K_CHUNK], [0, k0])
            wgk_1 = pl.slice(
                w_gate, [SHARED_GATE_K_CHUNK, SHARED_SWIGLU_N_CHUNK], [k0, 32],
            )
            wuk_1 = pl.slice(
                w_up, [SHARED_GATE_K_CHUNK, SHARED_SWIGLU_N_CHUNK], [k0, 32],
            )
            gate_acc_1 = pl.matmul_acc(gate_acc_1, xk_1, wgk_1)
            up_acc_1 = pl.matmul_acc(up_acc_1, xk_1, wuk_1)
        sigmoid_1 = pl.recip(pl.add(pl.exp(pl.neg(gate_acc_1)), 1.0))
        silu_1 = pl.mul(gate_acc_1, sigmoid_1)
        silu_c_1 = pl.minimum(silu_1, SHARED_SWIGLU_LIMIT)
        up_c_1 = pl.maximum(
            pl.minimum(up_acc_1, SHARED_SWIGLU_LIMIT),
            -SHARED_SWIGLU_LIMIT,
        )
        gated_1 = pl.mul(silu_c_1, up_c_1)
        h_c1 = pl.cast(gated_1, target_type=pl.BF16)

        x0_2 = pl.slice(x, [BATCH, SHARED_GATE_K_CHUNK], [0, 0])
        wg0_2 = pl.slice(
            w_gate, [SHARED_GATE_K_CHUNK, SHARED_SWIGLU_N_CHUNK], [0, 64],
        )
        wu0_2 = pl.slice(
            w_up, [SHARED_GATE_K_CHUNK, SHARED_SWIGLU_N_CHUNK], [0, 64],
        )
        gate_acc_2 = pl.matmul(x0_2, wg0_2, out_dtype=pl.FP32)
        up_acc_2 = pl.matmul(x0_2, wu0_2, out_dtype=pl.FP32)
        for kb in pl.range(1, HIDDEN // SHARED_GATE_K_CHUNK):
            k0 = kb * SHARED_GATE_K_CHUNK
            xk_2 = pl.slice(x, [BATCH, SHARED_GATE_K_CHUNK], [0, k0])
            wgk_2 = pl.slice(
                w_gate, [SHARED_GATE_K_CHUNK, SHARED_SWIGLU_N_CHUNK], [k0, 64],
            )
            wuk_2 = pl.slice(
                w_up, [SHARED_GATE_K_CHUNK, SHARED_SWIGLU_N_CHUNK], [k0, 64],
            )
            gate_acc_2 = pl.matmul_acc(gate_acc_2, xk_2, wgk_2)
            up_acc_2 = pl.matmul_acc(up_acc_2, xk_2, wuk_2)
        sigmoid_2 = pl.recip(pl.add(pl.exp(pl.neg(gate_acc_2)), 1.0))
        silu_2 = pl.mul(gate_acc_2, sigmoid_2)
        silu_c_2 = pl.minimum(silu_2, SHARED_SWIGLU_LIMIT)
        up_c_2 = pl.maximum(
            pl.minimum(up_acc_2, SHARED_SWIGLU_LIMIT),
            -SHARED_SWIGLU_LIMIT,
        )
        gated_2 = pl.mul(silu_c_2, up_c_2)
        h_c2 = pl.cast(gated_2, target_type=pl.BF16)

        x0_3 = pl.slice(x, [BATCH, SHARED_GATE_K_CHUNK], [0, 0])
        wg0_3 = pl.slice(
            w_gate, [SHARED_GATE_K_CHUNK, SHARED_SWIGLU_N_CHUNK], [0, 96],
        )
        wu0_3 = pl.slice(
            w_up, [SHARED_GATE_K_CHUNK, SHARED_SWIGLU_N_CHUNK], [0, 96],
        )
        gate_acc_3 = pl.matmul(x0_3, wg0_3, out_dtype=pl.FP32)
        up_acc_3 = pl.matmul(x0_3, wu0_3, out_dtype=pl.FP32)
        for kb in pl.range(1, HIDDEN // SHARED_GATE_K_CHUNK):
            k0 = kb * SHARED_GATE_K_CHUNK
            xk_3 = pl.slice(x, [BATCH, SHARED_GATE_K_CHUNK], [0, k0])
            wgk_3 = pl.slice(
                w_gate, [SHARED_GATE_K_CHUNK, SHARED_SWIGLU_N_CHUNK], [k0, 96],
            )
            wuk_3 = pl.slice(
                w_up, [SHARED_GATE_K_CHUNK, SHARED_SWIGLU_N_CHUNK], [k0, 96],
            )
            gate_acc_3 = pl.matmul_acc(gate_acc_3, xk_3, wgk_3)
            up_acc_3 = pl.matmul_acc(up_acc_3, xk_3, wuk_3)
        sigmoid_3 = pl.recip(pl.add(pl.exp(pl.neg(gate_acc_3)), 1.0))
        silu_3 = pl.mul(gate_acc_3, sigmoid_3)
        silu_c_3 = pl.minimum(silu_3, SHARED_SWIGLU_LIMIT)
        up_c_3 = pl.maximum(
            pl.minimum(up_acc_3, SHARED_SWIGLU_LIMIT),
            -SHARED_SWIGLU_LIMIT,
        )
        gated_3 = pl.mul(silu_c_3, up_c_3)
        h_c3 = pl.cast(gated_3, target_type=pl.BF16)

        x0_4 = pl.slice(x, [BATCH, SHARED_GATE_K_CHUNK], [0, 0])
        wg0_4 = pl.slice(
            w_gate, [SHARED_GATE_K_CHUNK, SHARED_SWIGLU_N_CHUNK], [0, 128],
        )
        wu0_4 = pl.slice(
            w_up, [SHARED_GATE_K_CHUNK, SHARED_SWIGLU_N_CHUNK], [0, 128],
        )
        gate_acc_4 = pl.matmul(x0_4, wg0_4, out_dtype=pl.FP32)
        up_acc_4 = pl.matmul(x0_4, wu0_4, out_dtype=pl.FP32)
        for kb in pl.range(1, HIDDEN // SHARED_GATE_K_CHUNK):
            k0 = kb * SHARED_GATE_K_CHUNK
            xk_4 = pl.slice(x, [BATCH, SHARED_GATE_K_CHUNK], [0, k0])
            wgk_4 = pl.slice(
                w_gate, [SHARED_GATE_K_CHUNK, SHARED_SWIGLU_N_CHUNK], [k0, 128],
            )
            wuk_4 = pl.slice(
                w_up, [SHARED_GATE_K_CHUNK, SHARED_SWIGLU_N_CHUNK], [k0, 128],
            )
            gate_acc_4 = pl.matmul_acc(gate_acc_4, xk_4, wgk_4)
            up_acc_4 = pl.matmul_acc(up_acc_4, xk_4, wuk_4)
        sigmoid_4 = pl.recip(pl.add(pl.exp(pl.neg(gate_acc_4)), 1.0))
        silu_4 = pl.mul(gate_acc_4, sigmoid_4)
        silu_c_4 = pl.minimum(silu_4, SHARED_SWIGLU_LIMIT)
        up_c_4 = pl.maximum(
            pl.minimum(up_acc_4, SHARED_SWIGLU_LIMIT),
            -SHARED_SWIGLU_LIMIT,
        )
        gated_4 = pl.mul(silu_c_4, up_c_4)
        h_c4 = pl.cast(gated_4, target_type=pl.BF16)

        for db in pl.range(HIDDEN // SHARED_DOWN_N_CHUNK):
            d0 = db * SHARED_DOWN_N_CHUNK
            wd_c0 = pl.slice(
                w_down, [SHARED_SWIGLU_N_CHUNK, SHARED_DOWN_N_CHUNK], [0, d0],
            )
            y_acc = pl.matmul(h_c0, wd_c0, out_dtype=pl.FP32)
            wd_c1 = pl.slice(
                w_down, [SHARED_SWIGLU_N_CHUNK, SHARED_DOWN_N_CHUNK], [32, d0],
            )
            y_acc = pl.matmul_acc(y_acc, h_c1, wd_c1)
            wd_c2 = pl.slice(
                w_down, [SHARED_SWIGLU_N_CHUNK, SHARED_DOWN_N_CHUNK], [64, d0],
            )
            y_acc = pl.matmul_acc(y_acc, h_c2, wd_c2)
            wd_c3 = pl.slice(
                w_down, [SHARED_SWIGLU_N_CHUNK, SHARED_DOWN_N_CHUNK], [96, d0],
            )
            y_acc = pl.matmul_acc(y_acc, h_c3, wd_c3)
            wd_c4 = pl.slice(
                w_down, [SHARED_SWIGLU_N_CHUNK, SHARED_DOWN_N_CHUNK], [128, d0],
            )
            y_acc = pl.matmul_acc(y_acc, h_c4, wd_c4)
            sh_y_shard = pl.assemble(
                sh_y_shard,
                pl.cast(y_acc, target_type=pl.BF16),
                [0, d0],
            )

    return sh_y_shard
