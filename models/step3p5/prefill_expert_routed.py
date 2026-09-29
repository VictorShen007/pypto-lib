# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Prefill-side routed-expert body — Step3.5 grouped FFN over N_LOCAL_EXPERTS.

v2 lift (task #13): the ``_expert_routed`` method body from
``prefill_moe.py:845-1032`` is lifted here as two module-level
``@pl.jit.inline`` bodies, split by activation:

  - ``prefill_expert_routed_silu``    — routed_lim=0.0 (silu gate)
  - ``prefill_expert_routed_swiglu7``  — routed_lim=7.0 (SwigluStep clamp)

The split replaces the runtime ``if _routed_swiglu_step`` branch in the
original body (lines 948-956) with compile-time body selection — the
caller (factory build) picks the right body via ``pl.inline(<variant>._func)``.

Math is unchanged. Lowercase closure variables (local_recv_max,
n_local_experts, inter) are replaced with uppercase module-level
constants from ``_moe_constants.py``.

h-BF16 removal (precision, PL-approved): the gated h is now kept as a
single FP32 h_tile [16,1280] = 80KB (no BF16 cast, no per-chunk cast
into h_bf16). The re-quant reads h_tile directly (already FP32), so
the BF16 narrowing of gated h — the only non-shared quant-path
difference vs the VLLM ``npu_dequant_swiglu_quant`` golden (which has
no BF16 narrowing) — is eliminated. UB is safe: the FP32 h_tile is
the original task #25 design, and dropping the h_bf16 buffer frees
40KB (moe_down Vec drops below the 160KB task #25 ceiling).

K1 row tiling (task #25, APPLIED): ROUTED_MAX_TILE=1024 rows made
h_tile [1024, 1280] FP32 = 5MB overflow UB (184KB). Row-tile to
ROUTED_ROW_TILE=16: h_tile [16, 1280] FP32 = 80KB; with the h_bf16
cast (also alive in moe_down) + weight reshape copies, moe_down
Vec = 160KB < 184KB; moe_gate_up Vec = 96KB. Math unchanged -- the
rt loop (pl.range, sequential within each expert) splits 1024 rows
into 64 row-tiles of 16 rows each. Mirrors the decode-side
expert_routed.py pattern (h_tile FP32 + cast to BF16 before the
down matmul). Decode side keeps MAX_TILE=8 (single token, no
overflow, not touched).

N=128 wide-cast fix: FP32 h_tile + full [16,1280] cast removed
(wide-tile tmov misprune). Each [16,64] gated chunk is now cast
in-place into h_bf16 (decode moe.py pattern); moe_down Vec budget
is now just h_bf16 [16,1280] BF16 = 40KB, no FP32 h_tile alive.

PL's directive was ROW_TILE=32, but ROW_TILE=32 with FP32 h_tile
overflows moe_down (h_tile 160KB + h_bf16 80KB = 240KB > 184KB --
both alive because h_bf16 = cast(h_tile) reads h_tile). ROW_TILE=16
is the largest power-of-2 that fits with FP32 h_tile + h_bf16 cast.

K_CHUNK=128 (was 256): the 3D-slice + reshape weight path creates
a copy in Vec ([K_CHUNK, N_CHUNK] BF16). K_CHUNK=256 made this 32KB
per tile; K_CHUNK=128 halves it to 16KB, recovering the Vec budget
that the h_tile + h_bf16 pair consumes. Math unchanged -- the kb
loop still covers full HIDDEN=4096 (32 iters at 128 vs 16 at 256).

tile_valid clamp (task #25 runtime fix): when n_rows < rt (e.g.
count=8, rt=16 -> tile_valid=-8), the negative valid_shape stalls
the cube unit (S1:running-stalled, stuck_core=5, completed=2305/
6912). Scalar pl.maximum doesn't exist; the clamp uses the identity
``x * cast(x > 0, INDEX) == max(x, 0)``. Exposed by K1 (UB fix) --
previously masked by the Vec overflow that prevented runtime.

K1-class drop_dims attempted (task #30): replacing the 3D-slice +
reshape with ``pl.slice(..., drop_dims=[0])`` FAILED -- the
tensor-to-tile lowering does not propagate ``drop_dims`` into the
matmul output type (``tile.set_validshape requires a 2D tile, but
got rank 3``). Reverted to the 3D-slice + reshape form. With K1
applied (ROUTED_ROW_TILE=16 + K_CHUNK=128), the Vec budget fits
without drop_dims; the drop_dims workaround is no longer required.
Tracked as a compiler limitation -- see PL ruling (task #30).

Task #30 (b) flatten weight sig (APPLIED): weight signature flattened
from 3D ``[N_LOCAL_EXPERTS, HIDDEN, INTER]`` to 2D
``[N_LOCAL_EXPERTS * HIDDEN, INTER]`` (w_down similarly
``[N_LOCAL_EXPERTS * INTER, HIDDEN]``). The body uses pure 2D slices
with ``gate_expert_base = e * HIDDEN`` / ``down_expert_base = e * INTER``
row offsets, eliminating the reshape copy that previously consumed
16-32 KB of Vec per weight tile. This mirrors the gate body
(``prefill_gate.py``) 2D-slice pattern. Math unchanged -- the matmul
operands are the same [K_CHUNK, N_CHUNK] BF16 tiles, just sourced
via 2D slice instead of 3D slice + reshape. The drop_dims failure is
now moot (no reshape to eliminate rank from).

T2a native W8A8 (task #11): the routed lane is converted from BF16 to
INT8. The gate/up matmul runs INT8@INT8 -> INT32, dequantized per tile
via ``row_expand_mul(x_scale) * col_expand_mul(w_scale)``; the gated
h stays FP32 (h_tile), then re-quantized per row (amax -> 127/amax,
rint -> FP16 round -> INT8 trunc) into h_i8 for the INT8 down matmul;
the down accumulator is dequantized via ``h_scale_dq * w_down_scale``
before the BF16 assemble. The route weight is NOT applied here (it stays
in ``prefill_combine.py``, unchanged). The per-token activation scale is
produced by the upstream norm/quant producer (``prefill_fwd.py``) and
routed through dispatch. Mirrors ``decode_fwd.py:1240-1756``.
"""

from __future__ import annotations

import pypto.language as pl

from ._moe_constants import (
    HIDDEN,
    INTER,
    LOCAL_RECV_MAX,
    N_LOCAL_EXPERTS,
    ROUTED_DOWN_K_CHUNK,
    ROUTED_DOWN_N_CHUNK,
    ROUTED_GATE_K_CHUNK,
    ROUTED_GATE_N_CHUNK,
    ROUTED_H_QUANT_N_CHUNK,
    ROUTED_MAX_TILE,
    ROUTED_ROW_TILE,
    ROUTED_SWIGLU_LIMIT,
)

__all__ = [
    "prefill_expert_routed_silu",
    "prefill_expert_routed_swiglu7",
    "select_prefill_expert_routed",
]


def select_prefill_expert_routed(routed_lim: float):
    """Return the right @pl.jit.inline body for the layer's routed_lim.

    Mirrors the factory-build-time ``_routed_swiglu_step`` closure choice
    in ``prefill_moe.py:208-215``.
    """
    if routed_lim == 0.0:
        return prefill_expert_routed_silu
    if routed_lim == 7.0:
        return prefill_expert_routed_swiglu7
    raise ValueError(
        f"routed_lim must be 0.0 or 7.0, got {routed_lim}",
    )


@pl.jit.inline
def prefill_expert_routed_silu(  # noqa: PLR0913, PLR0915
    local_routed_x: pl.Tensor[[LOCAL_RECV_MAX, HIDDEN], pl.INT8],
    local_routed_x_scale: pl.Tensor[[1, LOCAL_RECV_MAX], pl.FP32],
    local_expert_offset: pl.Tensor[[N_LOCAL_EXPERTS], pl.INT32],
    local_expert_count: pl.Tensor[[N_LOCAL_EXPERTS], pl.INT32],
    # Weight layout flattened from [N_LOCAL_EXPERTS, HIDDEN, INTER] to
    # [N_LOCAL_EXPERTS * HIDDEN, INTER] (task #30 (b) flatten weight sig).
    # Expert e occupies rows [e*HIDDEN, (e+1)*HIDDEN) in the K dim. This
    # eliminates the 3D-slice + reshape copy that previously consumed
    # 16-32 KB of Vec (UB) per gate/up weight tile. Per-output-channel
    # scales stay 2D [N_LOCAL_EXPERTS, *] (per-expert rows, decode-faithful).
    w_gate: pl.Tensor[
        [N_LOCAL_EXPERTS * HIDDEN, INTER], pl.INT8
    ],
    w_gate_scale: pl.Tensor[[N_LOCAL_EXPERTS, INTER], pl.FP32],
    w_up: pl.Tensor[
        [N_LOCAL_EXPERTS * HIDDEN, INTER], pl.INT8
    ],
    w_up_scale: pl.Tensor[[N_LOCAL_EXPERTS, INTER], pl.FP32],
    w_down: pl.Tensor[
        [N_LOCAL_EXPERTS * INTER, HIDDEN], pl.INT8
    ],
    w_down_scale: pl.Tensor[[N_LOCAL_EXPERTS, HIDDEN], pl.FP32],
    local_routed_y: pl.Tensor[
        [LOCAL_RECV_MAX, HIDDEN], pl.BF16
    ],
):
    """Routed-expert FFN with silu gate (routed_lim=0.0), INT8-native."""
    for e in pl.parallel(N_LOCAL_EXPERTS):
        n_rows = pl.read(local_expert_count, [e])
        offset_i32 = pl.read(local_expert_offset, [e])
        offset = pl.cast(offset_i32, pl.INDEX)
        valid_rows = pl.cast(n_rows, pl.INDEX)
        # Expert row base in the flattened K dim: gate/up weight rows
        # [gate_expert_base, gate_expert_base + HIDDEN) belong to expert e.
        # down weight rows [down_expert_base, down_expert_base + INTER)
        # belong to expert e.
        gate_expert_base = e * HIDDEN
        down_expert_base = e * INTER

        # Dynamic row-tile bound ceil(n_rows / ROUTED_ROW_TILE), decode-faithful
        # (decode_fwd.py n_tiles = (n_rows + RECV_TILE - 1) // RECV_TILE). The
        # trip count MUST stay a runtime scalar — `pl.range(constant)` fully
        # unrolls and overflows UB (pypto-dev-constraints §C:116).
        n_tiles = (n_rows + ROUTED_ROW_TILE - 1) // ROUTED_ROW_TILE
        for rt_idx in pl.range(n_tiles):
            rt = rt_idx * ROUTED_ROW_TILE
            # tile_valid = min(ROUTED_ROW_TILE, n_rows - rt), always in
            # [1, ROUTED_ROW_TILE] because the dynamic n_tiles bound keeps
            # rt < n_rows. Uses scalar pl.min (NOT `pl.cast(bool, INDEX)` —
            # on Ascend that yields TRUE=-1, pypto-dev-constraints §C:122,
            # which would produce a negative valid_shape and cube stall).
            tile_valid = pl.min(
                pl.cast(ROUTED_ROW_TILE, pl.INDEX),
                valid_rows - rt,
            )
            row_off = offset + rt
            if tile_valid > 0:
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="moe_gate_up"):
                    h_tile = pl.create_tensor(
                        [ROUTED_ROW_TILE, INTER], dtype=pl.FP32,
                    )

                    for nb in pl.range(INTER // ROUTED_GATE_N_CHUNK):
                        n0 = nb * ROUTED_GATE_N_CHUNK

                        x0 = pl.slice(
                            local_routed_x,
                            [ROUTED_ROW_TILE, ROUTED_GATE_K_CHUNK],
                            [row_off, 0],
                        )
                        wg0 = pl.slice(
                            w_gate,
                            [ROUTED_GATE_K_CHUNK, ROUTED_GATE_N_CHUNK],
                            [gate_expert_base, n0],
                        )
                        wu0 = pl.slice(
                            w_up,
                            [ROUTED_GATE_K_CHUNK, ROUTED_GATE_N_CHUNK],
                            [gate_expert_base, n0],
                        )
                        gate_acc = pl.matmul(x0, wg0, out_dtype=pl.INT32)
                        up_acc = pl.matmul(x0, wu0, out_dtype=pl.INT32)
                        for kb in pl.range(1, HIDDEN // ROUTED_GATE_K_CHUNK):
                            k0 = kb * ROUTED_GATE_K_CHUNK
                            xk = pl.slice(
                                local_routed_x,
                                [ROUTED_ROW_TILE, ROUTED_GATE_K_CHUNK],
                                [row_off, k0],
                            )
                            wgk = pl.slice(
                                w_gate,
                                [ROUTED_GATE_K_CHUNK, ROUTED_GATE_N_CHUNK],
                                [gate_expert_base + k0, n0],
                            )
                            wuk = pl.slice(
                                w_up,
                                [ROUTED_GATE_K_CHUNK, ROUTED_GATE_N_CHUNK],
                                [gate_expert_base + k0, n0],
                            )
                            gate_acc = pl.matmul_acc(gate_acc, xk, wgk)
                            up_acc = pl.matmul_acc(up_acc, xk, wuk)

                        # Per-token activation dequant (decode_fwd.py:1462-1501).
                        x_scale_col = pl.reshape(
                            pl.slice(
                                local_routed_x_scale,
                                [1, ROUTED_ROW_TILE],
                                [0, row_off],
                            ),
                            [ROUTED_ROW_TILE, 1],
                        )
                        wg_scale_row = pl.slice(
                            w_gate_scale,
                            [1, ROUTED_GATE_N_CHUNK],
                            [e, n0],
                        )
                        wu_scale_row = pl.slice(
                            w_up_scale,
                            [1, ROUTED_GATE_N_CHUNK],
                            [e, n0],
                        )
                        gate_2d = pl.col_expand_mul(
                            pl.row_expand_mul(
                                pl.cast(
                                    gate_acc,
                                    target_type=pl.FP32,
                                    mode="none",
                                ),
                                x_scale_col,
                            ),
                            wg_scale_row,
                        )
                        up_2d = pl.col_expand_mul(
                            pl.row_expand_mul(
                                pl.cast(
                                    up_acc,
                                    target_type=pl.FP32,
                                    mode="none",
                                ),
                                x_scale_col,
                            ),
                            wu_scale_row,
                        )

                        sigmoid = pl.recip(
                            pl.add(pl.exp(pl.neg(gate_2d)), 1.0),
                        )
                        silu = pl.mul(gate_2d, sigmoid)
                        gated = pl.mul(silu, up_2d)

                        gated_v = pl.set_validshape(
                            gated, tile_valid, ROUTED_GATE_N_CHUNK,
                        )
                        gated_m = pl.fillpad(
                            gated_v, pad_value=pl.PadValue.zero,
                        )
                        h_tile[
                            :, n0 : n0 + ROUTED_GATE_N_CHUNK
                        ] = gated_m

                # h re-quant (decode_fwd.py:1526-1627): per-row amax over the
                # FP32 gated h, then 127/amax scale, rint->FP16 round->INT8
                # trunc into h_i8 for the INT8 down matmul.
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="moe_h_quant"):
                    h_i8 = pl.create_tensor(
                        [ROUTED_ROW_TILE, INTER], dtype=pl.INT8,
                    )
                    eh_amax = pl.full(
                        [1, ROUTED_ROW_TILE], dtype=pl.FP32, value=1e-4,
                    )
                    for hqa in pl.range(INTER // ROUTED_H_QUANT_N_CHUNK):
                        hqa0 = hqa * ROUTED_H_QUANT_N_CHUNK
                        eh_a = pl.slice(
                            h_tile,
                            [ROUTED_ROW_TILE, ROUTED_H_QUANT_N_CHUNK],
                            [0, hqa0],
                        )
                        eh_amax = pl.maximum(
                            eh_amax,
                            pl.reshape(
                                pl.row_max(
                                    pl.maximum(eh_a, pl.neg(eh_a)),
                                ),
                                [1, ROUTED_ROW_TILE],
                            ),
                        )
                    eh_sq_row = pl.mul(
                        pl.recip(eh_amax),
                        pl.full(
                            [1, ROUTED_ROW_TILE],
                            dtype=pl.FP32,
                            value=127.0,
                        ),
                    )
                    h_scale_dq = pl.reshape(
                        pl.recip(eh_sq_row), [ROUTED_ROW_TILE, 1],
                    )
                    eh_sq_col = pl.reshape(eh_sq_row, [ROUTED_ROW_TILE, 1])
                    for hqn in pl.range(INTER // ROUTED_H_QUANT_N_CHUNK):
                        hqn0 = hqn * ROUTED_H_QUANT_N_CHUNK
                        eh_q = pl.slice(
                            h_tile,
                            [ROUTED_ROW_TILE, ROUTED_H_QUANT_N_CHUNK],
                            [0, hqn0],
                        )
                        eh_scaled = pl.row_expand_mul(eh_q, eh_sq_col)
                        eh_i32 = pl.cast(
                            eh_scaled, target_type=pl.INT32,
                        )
                        eh_half = pl.cast(
                            eh_i32, target_type=pl.FP16, mode="round",
                        )
                        h_i8[
                            :, hqn0 : hqn0 + ROUTED_H_QUANT_N_CHUNK
                        ] = pl.cast(
                            eh_half, target_type=pl.INT8, mode="trunc",
                        )

                with pl.at(level=pl.Level.CORE_GROUP, name_hint="moe_down"):
                    for db in pl.range(HIDDEN // ROUTED_DOWN_N_CHUNK):
                        d0 = db * ROUTED_DOWN_N_CHUNK
                        h0 = pl.slice(
                            h_i8,
                            [ROUTED_ROW_TILE, ROUTED_DOWN_K_CHUNK],
                            [0, 0],
                        )
                        # 2D slice into flattened w_down (task #30 (b)):
                        # [down_expert_base + 0 : +K_CHUNK, d0 : d0+N_CHUNK].
                        wd0 = pl.slice(
                            w_down,
                            [ROUTED_DOWN_K_CHUNK, ROUTED_DOWN_N_CHUNK],
                            [down_expert_base, d0],
                        )
                        y_acc = pl.matmul(h0, wd0, out_dtype=pl.INT32)
                        for kb2 in pl.range(1, INTER // ROUTED_DOWN_K_CHUNK):
                            k0 = kb2 * ROUTED_DOWN_K_CHUNK
                            hk = pl.slice(
                                h_i8,
                                [ROUTED_ROW_TILE, ROUTED_DOWN_K_CHUNK],
                                [0, k0],
                            )
                            wdk = pl.slice(
                                w_down,
                                [ROUTED_DOWN_K_CHUNK, ROUTED_DOWN_N_CHUNK],
                                [down_expert_base + k0, d0],
                            )
                            y_acc = pl.matmul_acc(y_acc, hk, wdk)

                        # Down dequant (decode_fwd.py:1711-1743) — h_scale_dq
                        # only; the route weight is applied in prefill_combine.py.
                        wd_scale_row = pl.slice(
                            w_down_scale,
                            [1, ROUTED_DOWN_N_CHUNK],
                            [e, d0],
                        )
                        y_2d = pl.col_expand_mul(
                            pl.row_expand_mul(
                                pl.cast(
                                    y_acc,
                                    target_type=pl.FP32,
                                    mode="none",
                                ),
                                h_scale_dq,
                            ),
                            wd_scale_row,
                        )

                        # OOB fix (task #6): fillpad resets valid_shape to the
                        # full 16-row physical shape, so assemble would write all
                        # 16 rows at [row_off, d0]. With prefix-sum CSR offsets and
                        # the fixed 64-iter rt loop, that writes zeros past the
                        # expert's rows (clobbering the next expert) and beyond
                        # LOCAL_RECV_MAX for offset>0. Keep the narrowed valid_shape
                        # instead so the store writes exactly tile_valid rows (0 for
                        # fully-invalid tiles).
                        y_v = pl.set_validshape(
                            y_2d, tile_valid, ROUTED_DOWN_N_CHUNK,
                        )
                        local_routed_y = pl.assemble(
                            local_routed_y,
                            pl.cast(y_v, target_type=pl.BF16),
                            [row_off, d0],
                        )

    return local_routed_y


@pl.jit.inline
def prefill_expert_routed_swiglu7(  # noqa: PLR0913, PLR0915
    local_routed_x: pl.Tensor[[LOCAL_RECV_MAX, HIDDEN], pl.INT8],
    local_routed_x_scale: pl.Tensor[[1, LOCAL_RECV_MAX], pl.FP32],
    local_expert_offset: pl.Tensor[[N_LOCAL_EXPERTS], pl.INT32],
    local_expert_count: pl.Tensor[[N_LOCAL_EXPERTS], pl.INT32],
    # Flattened 2D weight layout (task #30 (b)) — see silu body comment.
    w_gate: pl.Tensor[
        [N_LOCAL_EXPERTS * HIDDEN, INTER], pl.INT8
    ],
    w_gate_scale: pl.Tensor[[N_LOCAL_EXPERTS, INTER], pl.FP32],
    w_up: pl.Tensor[
        [N_LOCAL_EXPERTS * HIDDEN, INTER], pl.INT8
    ],
    w_up_scale: pl.Tensor[[N_LOCAL_EXPERTS, INTER], pl.FP32],
    w_down: pl.Tensor[
        [N_LOCAL_EXPERTS * INTER, HIDDEN], pl.INT8
    ],
    w_down_scale: pl.Tensor[[N_LOCAL_EXPERTS, HIDDEN], pl.FP32],
    local_routed_y: pl.Tensor[
        [LOCAL_RECV_MAX, HIDDEN], pl.BF16
    ],
):
    """Routed-expert FFN with SwigluStep@7.0 clamp (routed_lim=7.0)."""
    for e in pl.parallel(N_LOCAL_EXPERTS):
        n_rows = pl.read(local_expert_count, [e])
        offset_i32 = pl.read(local_expert_offset, [e])
        offset = pl.cast(offset_i32, pl.INDEX)
        valid_rows = pl.cast(n_rows, pl.INDEX)
        gate_expert_base = e * HIDDEN
        down_expert_base = e * INTER

        # Dynamic row-tile bound ceil(n_rows / ROUTED_ROW_TILE), decode-faithful
        # (decode_fwd.py n_tiles = (n_rows + RECV_TILE - 1) // RECV_TILE). The
        # trip count MUST stay a runtime scalar — `pl.range(constant)` fully
        # unrolls and overflows UB (pypto-dev-constraints §C:116).
        n_tiles = (n_rows + ROUTED_ROW_TILE - 1) // ROUTED_ROW_TILE
        for rt_idx in pl.range(n_tiles):
            rt = rt_idx * ROUTED_ROW_TILE
            # tile_valid = min(ROUTED_ROW_TILE, n_rows - rt), always in
            # [1, ROUTED_ROW_TILE] because the dynamic n_tiles bound keeps
            # rt < n_rows. Uses scalar pl.min (NOT `pl.cast(bool, INDEX)` —
            # on Ascend that yields TRUE=-1, pypto-dev-constraints §C:122,
            # which would produce a negative valid_shape and cube stall).
            tile_valid = pl.min(
                pl.cast(ROUTED_ROW_TILE, pl.INDEX),
                valid_rows - rt,
            )
            row_off = offset + rt
            if tile_valid > 0:
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="moe_gate_up"):
                    h_tile = pl.create_tensor(
                        [ROUTED_ROW_TILE, INTER], dtype=pl.FP32,
                    )

                    for nb in pl.range(INTER // ROUTED_GATE_N_CHUNK):
                        n0 = nb * ROUTED_GATE_N_CHUNK

                        x0 = pl.slice(
                            local_routed_x,
                            [ROUTED_ROW_TILE, ROUTED_GATE_K_CHUNK],
                            [row_off, 0],
                        )
                        wg0 = pl.slice(
                            w_gate,
                            [ROUTED_GATE_K_CHUNK, ROUTED_GATE_N_CHUNK],
                            [gate_expert_base, n0],
                        )
                        wu0 = pl.slice(
                            w_up,
                            [ROUTED_GATE_K_CHUNK, ROUTED_GATE_N_CHUNK],
                            [gate_expert_base, n0],
                        )
                        gate_acc = pl.matmul(x0, wg0, out_dtype=pl.INT32)
                        up_acc = pl.matmul(x0, wu0, out_dtype=pl.INT32)
                        for kb in pl.range(1, HIDDEN // ROUTED_GATE_K_CHUNK):
                            k0 = kb * ROUTED_GATE_K_CHUNK
                            xk = pl.slice(
                                local_routed_x,
                                [ROUTED_ROW_TILE, ROUTED_GATE_K_CHUNK],
                                [row_off, k0],
                            )
                            wgk = pl.slice(
                                w_gate,
                                [ROUTED_GATE_K_CHUNK, ROUTED_GATE_N_CHUNK],
                                [gate_expert_base + k0, n0],
                            )
                            wuk = pl.slice(
                                w_up,
                                [ROUTED_GATE_K_CHUNK, ROUTED_GATE_N_CHUNK],
                                [gate_expert_base + k0, n0],
                            )
                            gate_acc = pl.matmul_acc(gate_acc, xk, wgk)
                            up_acc = pl.matmul_acc(up_acc, xk, wuk)

                        # Per-token activation dequant (decode_fwd.py:1462-1501).
                        x_scale_col = pl.reshape(
                            pl.slice(
                                local_routed_x_scale,
                                [1, ROUTED_ROW_TILE],
                                [0, row_off],
                            ),
                            [ROUTED_ROW_TILE, 1],
                        )
                        wg_scale_row = pl.slice(
                            w_gate_scale,
                            [1, ROUTED_GATE_N_CHUNK],
                            [e, n0],
                        )
                        wu_scale_row = pl.slice(
                            w_up_scale,
                            [1, ROUTED_GATE_N_CHUNK],
                            [e, n0],
                        )
                        gate_2d = pl.col_expand_mul(
                            pl.row_expand_mul(
                                pl.cast(
                                    gate_acc,
                                    target_type=pl.FP32,
                                    mode="none",
                                ),
                                x_scale_col,
                            ),
                            wg_scale_row,
                        )
                        up_2d = pl.col_expand_mul(
                            pl.row_expand_mul(
                                pl.cast(
                                    up_acc,
                                    target_type=pl.FP32,
                                    mode="none",
                                ),
                                x_scale_col,
                            ),
                            wu_scale_row,
                        )

                        sigmoid = pl.recip(
                            pl.add(pl.exp(pl.neg(gate_2d)), 1.0),
                        )
                        silu = pl.mul(gate_2d, sigmoid)
                        # SwigluStep@7.0 clamp (replaces the runtime
                        # ``if _routed_swiglu_step`` branch from prefill_moe.py:948-954)
                        silu_c = pl.minimum(silu, ROUTED_SWIGLU_LIMIT)
                        up_c = pl.maximum(
                            pl.minimum(up_2d, ROUTED_SWIGLU_LIMIT),
                            -ROUTED_SWIGLU_LIMIT,
                        )
                        gated = pl.mul(silu_c, up_c)

                        gated_v = pl.set_validshape(
                            gated, tile_valid, ROUTED_GATE_N_CHUNK,
                        )
                        gated_m = pl.fillpad(
                            gated_v, pad_value=pl.PadValue.zero,
                        )
                        h_tile[
                            :, n0 : n0 + ROUTED_GATE_N_CHUNK
                        ] = gated_m

                # h re-quant (decode_fwd.py:1526-1627): per-row amax over the
                # FP32 gated h, then 127/amax scale, rint->FP16 round->INT8
                # trunc into h_i8 for the INT8 down matmul.
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="moe_h_quant"):
                    h_i8 = pl.create_tensor(
                        [ROUTED_ROW_TILE, INTER], dtype=pl.INT8,
                    )
                    eh_amax = pl.full(
                        [1, ROUTED_ROW_TILE], dtype=pl.FP32, value=1e-4,
                    )
                    for hqa in pl.range(INTER // ROUTED_H_QUANT_N_CHUNK):
                        hqa0 = hqa * ROUTED_H_QUANT_N_CHUNK
                        eh_a = pl.slice(
                            h_tile,
                            [ROUTED_ROW_TILE, ROUTED_H_QUANT_N_CHUNK],
                            [0, hqa0],
                        )
                        eh_amax = pl.maximum(
                            eh_amax,
                            pl.reshape(
                                pl.row_max(
                                    pl.maximum(eh_a, pl.neg(eh_a)),
                                ),
                                [1, ROUTED_ROW_TILE],
                            ),
                        )
                    eh_sq_row = pl.mul(
                        pl.recip(eh_amax),
                        pl.full(
                            [1, ROUTED_ROW_TILE],
                            dtype=pl.FP32,
                            value=127.0,
                        ),
                    )
                    h_scale_dq = pl.reshape(
                        pl.recip(eh_sq_row), [ROUTED_ROW_TILE, 1],
                    )
                    eh_sq_col = pl.reshape(eh_sq_row, [ROUTED_ROW_TILE, 1])
                    for hqn in pl.range(INTER // ROUTED_H_QUANT_N_CHUNK):
                        hqn0 = hqn * ROUTED_H_QUANT_N_CHUNK
                        eh_q = pl.slice(
                            h_tile,
                            [ROUTED_ROW_TILE, ROUTED_H_QUANT_N_CHUNK],
                            [0, hqn0],
                        )
                        eh_scaled = pl.row_expand_mul(eh_q, eh_sq_col)
                        eh_i32 = pl.cast(
                            eh_scaled, target_type=pl.INT32,
                        )
                        eh_half = pl.cast(
                            eh_i32, target_type=pl.FP16, mode="round",
                        )
                        h_i8[
                            :, hqn0 : hqn0 + ROUTED_H_QUANT_N_CHUNK
                        ] = pl.cast(
                            eh_half, target_type=pl.INT8, mode="trunc",
                        )

                with pl.at(level=pl.Level.CORE_GROUP, name_hint="moe_down"):
                    for db in pl.range(HIDDEN // ROUTED_DOWN_N_CHUNK):
                        d0 = db * ROUTED_DOWN_N_CHUNK
                        h0 = pl.slice(
                            h_i8,
                            [ROUTED_ROW_TILE, ROUTED_DOWN_K_CHUNK],
                            [0, 0],
                        )
                        # 2D slice into flattened w_down (task #30 (b)):
                        # [down_expert_base + 0 : +K_CHUNK, d0 : d0+N_CHUNK].
                        wd0 = pl.slice(
                            w_down,
                            [ROUTED_DOWN_K_CHUNK, ROUTED_DOWN_N_CHUNK],
                            [down_expert_base, d0],
                        )
                        y_acc = pl.matmul(h0, wd0, out_dtype=pl.INT32)
                        for kb2 in pl.range(1, INTER // ROUTED_DOWN_K_CHUNK):
                            k0 = kb2 * ROUTED_DOWN_K_CHUNK
                            hk = pl.slice(
                                h_i8,
                                [ROUTED_ROW_TILE, ROUTED_DOWN_K_CHUNK],
                                [0, k0],
                            )
                            wdk = pl.slice(
                                w_down,
                                [ROUTED_DOWN_K_CHUNK, ROUTED_DOWN_N_CHUNK],
                                [down_expert_base + k0, d0],
                            )
                            y_acc = pl.matmul_acc(y_acc, hk, wdk)

                        # Down dequant (decode_fwd.py:1711-1743) — h_scale_dq
                        # only; the route weight is applied in prefill_combine.py.
                        wd_scale_row = pl.slice(
                            w_down_scale,
                            [1, ROUTED_DOWN_N_CHUNK],
                            [e, d0],
                        )
                        y_2d = pl.col_expand_mul(
                            pl.row_expand_mul(
                                pl.cast(
                                    y_acc,
                                    target_type=pl.FP32,
                                    mode="none",
                                ),
                                h_scale_dq,
                            ),
                            wd_scale_row,
                        )

                        # OOB fix (task #6): see silu body — keep narrowed
                        # valid_shape so the store writes exactly tile_valid rows.
                        y_v = pl.set_validshape(
                            y_2d, tile_valid, ROUTED_DOWN_N_CHUNK,
                        )
                        local_routed_y = pl.assemble(
                            local_routed_y,
                            pl.cast(y_v, target_type=pl.BF16),
                            [row_off, d0],
                        )

    return local_routed_y
