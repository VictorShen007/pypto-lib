# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Prefill-side combine helpers — publish_src_route_table,
push_routed_y_to_sources, weighted_gather_and_add.

v2 lift (task #13): the 3 combine helpers from ``prefill_moe.py:1168-1329``
are lifted here as module-level ``@pl.jit.inline`` bodies using prefill-side
names (BATCH, HIDDEN, TOPK, N_RANKS, N_LOCAL_EXPERTS, N_ROUTES_PER_RANK,
LOCAL_RECV_MAX, PER_RANK_BUCKETS).

Math is unchanged. Lowercase closure variables (n_ranks, n_local_experts,
per_rank_buckets, local_recv_max, n_routes_per_rank) are replaced with
uppercase module-level constants from ``_moe_constants.py``.

Note: the original ``_publish_src_route_table`` and
``_push_routed_y_to_sources`` were ``@pl.function(type=InCore)``; they are
lifted here as ``@pl.jit.inline`` so the caller can ``pl.inline(<body>._func)``
them (module-level @pl.jit.inline is the only form that supports
``pl.inline(body._func)`` binding at factory build time).

Standalone execution note: this body is designed to be inlined into a
larger ``@pl.function`` with tile context. Standalone execution produces
an AICPU-only task graph -> runtime rejects (retCode=0x2a on real,
hangs on sim). Use (F) emulation + compile-only for L0, runtime
validation at L1.
"""

from __future__ import annotations

import pypto.language as pl
import pypto.language.distributed as pld

from ._moe_constants import (
    BATCH,
    HIDDEN,
    LOCAL_RECV_MAX,
    N_LOCAL_EXPERTS,
    N_RANKS,
    N_ROUTES_PER_RANK,
    PER_RANK_BUCKETS,
    TOPK,
)

__all__ = [
    "prefill_publish_src_route_table",
    "prefill_zero_routed_y_buf",
    "prefill_zero_routed_y_buf_chunked",
    "prefill_push_routed_y_to_sources",
    "prefill_weighted_gather_and_add",
]


@pl.jit.inline
def prefill_publish_src_route_table(
    indices: pl.Tensor[[BATCH, TOPK], pl.INT32],
    src_route_table: pld.DistributedTensor[
        [N_RANKS, N_LOCAL_EXPERTS, N_ROUTES_PER_RANK], pl.INT32
    ],
    my_rank: pl.Scalar[pl.INT32],
):
    """Publish per-route source index into the global src_route_table.

    AICPU-only warning: this body is pure-scalar control flow
    (pl.range + pl.read/pl.write/pl.store + pld.system.notify, no
    pl.at CORE_GROUP scope). Standalone execution produces an
    AICPU-only task graph (no AICore work) -> runtime rejects with
    retCode=0x2a on real device (correct reference behavior), hangs
    (SCHEDULER_TIMEOUT) on sim (sim defect, ignored per
    real-npu-first policy). Use (F) pure-Python emulation + compile-
    only for L0; runtime validation at L1 when integrated with tile
    context. See module docstring.
    """
    cursor = pl.create_tensor(
        [PER_RANK_BUCKETS], dtype=pl.INT32,
    )
    for i in pl.range(PER_RANK_BUCKETS):
        pl.write(cursor, [i], pl.cast(0, pl.INT32))

    for t in pl.range(BATCH):
        for k in pl.range(TOPK):
            eid = pl.read(indices, [t, k])
            dst = eid // N_LOCAL_EXPERTS
            loc_e = eid - dst * N_LOCAL_EXPERTS
            bkt = dst * N_LOCAL_EXPERTS + loc_e
            idx = pl.read(cursor, [bkt])
            pub_r_route = pl.cast(t * TOPK + k, pl.INT32)

            if dst == my_rank:
                # Self-rank publish: write the scalar pub_r_route directly
                # into the local view of the DistributedTensor. Avoids the
                # ``tmp = create_tensor; load; store`` dance that the InCore
                # tile verifier rejects with "tile.load requires TensorType
                # ... got TileType" (decode-side moe.py:_publish_src_route_table).
                pl.write(
                    src_route_table,
                    [my_rank, loc_e, pl.cast(idx, pl.INDEX)],
                    pub_r_route,
                )
            else:
                pld.system.notify(
                    target=src_route_table,
                    peer=dst,
                    offsets=[
                        my_rank, loc_e, pl.cast(idx, pl.INDEX),
                    ],
                    value=pub_r_route,
                    op=pld.NotifyOp.Set,
                )
            pl.write(cursor, [bkt], pl.cast(idx + 1, pl.INT32))

    return src_route_table


@pl.jit.inline
def prefill_zero_routed_y_buf(
    routed_y_buf: pld.DistributedTensor[
        [N_ROUTES_PER_RANK, HIDDEN], pl.BF16
    ],
):
    """Zero-init the routed_y_buf window before combine pushes.

    Data windows are NOT auto-zeroed (only signal windows are). The
    gather in ``prefill_weighted_gather_and_add`` reads all
    N_ROUTES_PER_RANK cells; any slot the combine push misses would
    otherwise read uninitialized ~5e4 garbage (mirrors decode
    moe.py:1838-1850 ``_zero_routed_y_buf``).

    NOTE: keep this body branch-free — static ``if`` conditions inside an
    inline body re-resolve in the SPLICING module's scope where the
    defining module's names are invisible. Variant selection happens at
    the ``pl.inline`` splice site in prefill_fwd.py instead.
    """
    for r in pl.range(N_ROUTES_PER_RANK):
        routed_y_buf[r : r + 1, :] = pl.full(
            [1, HIDDEN], dtype=pl.BF16, value=0.0,
        )

    return routed_y_buf


@pl.jit.inline
def prefill_zero_routed_y_buf_chunked(
    routed_y_buf: pld.DistributedTensor[
        [N_ROUTES_PER_RANK, HIDDEN], pl.BF16
    ],
):
    """Chunked-store variant of :func:`prefill_zero_routed_y_buf`.

    Same buffer, same zero values, same windows -- only the store
    granularity changes (R_CHUNK rows per store instead of 1, zero tile
    constructed once). [R_CHUNK, 4096] BF16 = 64KB at R_CHUNK=8, inside
    the per-core UB budget.
    """
    # NOTE: chunk size is a literal (4): cross-module constant resolution
    # at splice time is unreliable for names first used inside a body.
    zero_bf16 = pl.full([4, HIDDEN], dtype=pl.BF16, value=0.0)
    for rc in pl.range(N_ROUTES_PER_RANK // 4):
        r0 = rc * 4
        routed_y_buf[r0 : r0 + 4, :] = zero_bf16

    return routed_y_buf


@pl.jit.inline
def prefill_push_routed_y_to_sources(  # noqa: PLR0913
    local_routed_y: pl.Tensor[
        [LOCAL_RECV_MAX, HIDDEN], pl.BF16
    ],
    pub_counts: pld.DistributedTensor[
        [N_RANKS * N_RANKS, N_LOCAL_EXPERTS], pl.INT32
    ],
    routed_y_buf: pld.DistributedTensor[
        [N_ROUTES_PER_RANK, HIDDEN], pl.BF16
    ],
    combine_done: pld.DistributedTensor[
        [N_RANKS, 1], pl.INT32
    ],
    src_route_table: pld.DistributedTensor[
        [N_RANKS, N_LOCAL_EXPERTS, N_ROUTES_PER_RANK], pl.INT32
    ],
    my_rank: pl.Scalar[pl.INT32],
):
    """Push routed-expert output rows back to source ranks via routed_y_buf.

    AICPU-only warning: this body is pure-scalar control flow
    (pl.range + pl.read/pl.write/pl.store/pl.load + pld.system.notify/
    wait, no pl.at CORE_GROUP scope). Standalone execution produces an
    AICPU-only task graph (no AICore work) -> runtime rejects with
    retCode=0x2a on real device (correct reference behavior), hangs
    (SCHEDULER_TIMEOUT) on sim (sim defect, ignored per
    real-npu-first policy). Use (F) pure-Python emulation + compile-
    only for L0; runtime validation at L1 when integrated with tile
    context. See module docstring.
    """
    e_cursor = pl.cast(0, pl.INT32)
    for e in pl.range(N_LOCAL_EXPERTS):
        src_off = pl.cast(0, pl.INT32)
        for src in pl.range(N_RANKS):
            n = pl.cast(
                pl.read(
                    pub_counts, [src * N_RANKS + my_rank, e],
                ),
                pl.INDEX,
            )
            for row in pl.range(n):
                push_r_route = pl.read(
                    src_route_table,
                    [src, e, pl.cast(row, pl.INDEX)],
                )
                local_row = (
                    pl.cast(e_cursor, pl.INDEX)
                    + pl.cast(src_off, pl.INDEX) + row
                )
                if src == my_rank:
                    y_tile = pl.load(
                        local_routed_y,
                        [local_row, 0], [1, HIDDEN],
                    )
                    pl.store(y_tile, [push_r_route, 0], routed_y_buf)
                else:
                    # DeepSeek-style per-row push (decode-side
                    # moe.py:_push_routed_y_to_sources): a whole-tensor
                    # put materialises the 1MB routed_y_buf in UB and
                    # overflows the Vec buffer; use the per-row
                    # dst/src-offsets+shape form instead.
                    pld.tensor.put(
                        dst=routed_y_buf,
                        peer=src,
                        src=local_routed_y,
                        dst_offsets=[push_r_route, 0],
                        src_offsets=[local_row, 0],
                        shape=[1, HIDDEN],
                    )
            src_off = src_off + pl.cast(n, pl.INT32)
        total_e = pl.cast(0, pl.INT32)
        for src2 in pl.range(N_RANKS):
            total_e = total_e + pl.read(
                pub_counts, [src2 * N_RANKS + my_rank, e],
            )
        e_cursor = e_cursor + total_e

    # combine DMA fence (mirrors decode moe.py:1757-1763, §D:129 fence):
    # the cross-rank push above uses pld.tensor.put (remote_store / TPUT),
    # which has no trailing dsb (pto-isa TPut.hpp), so combine_done can
    # outrun the push DMA and let a consumer gather stale routed_y_buf.
    # A notify emits dsb(DSB_DDR)+pipe_barrier(PIPE_ALL) (TNotify.hpp)
    # draining the TPUTs. combine_done[my_rank, 0] is never waited by
    # waiters (they use src != my_rank); AtomicAdd+0 is a pure no-op fence.
    pld.system.notify(
        target=combine_done,
        peer=my_rank,
        offsets=[my_rank, 0],
        value=0,
        op=pld.NotifyOp.AtomicAdd,
    )
    for peer in pl.range(N_RANKS):
        if peer != my_rank:
            pld.system.notify(
                target=combine_done,
                peer=peer,
                offsets=[my_rank, 0],
                value=1,
                op=pld.NotifyOp.AtomicAdd,
            )
    for src in pl.range(N_RANKS):
        if src != my_rank:
            pld.system.wait(
                signal=combine_done,
                offsets=[src, 0],
                expected=1,
                cmp=pld.WaitCmp.Ge,
            )

    return routed_y_buf


@pl.jit.inline
def prefill_weighted_gather_and_add(
    routed_y_buf: pld.DistributedTensor[
        [N_ROUTES_PER_RANK, HIDDEN], pl.BF16
    ],
    expert_weights: pl.Tensor[[BATCH, TOPK], pl.FP32],
    sh_y: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
    moe_out: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
):
    """Weighted gather of routed-y rows + shared-y, summed into moe_out.

    AICPU/AICore mix warning: the gather loop is pure-scalar control
    flow (pl.range + pl.read + pl.mul/pl.add on scalars), but the
    epilogue uses pl.at CORE_GROUP + pl.store for the final moe_out
    tile. Standalone execution still produces a mostly-AICPU task
    graph (the single AICore dispatch is tiny) -> runtime rejects with
    retCode=0x2a on real device (correct reference behavior), hangs
    (SCHEDULER_TIMEOUT) on sim (sim defect, ignored per
    real-npu-first policy). Use (F) pure-Python emulation + compile-
    only for L0; runtime validation at L1 when integrated with tile
    context. See module docstring.
    """
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="moe_combine"):
        for b in pl.range(BATCH):
            acc = pl.cast(
                pl.load(sh_y, [b, 0], [1, HIDDEN]),
                target_type=pl.FP32,
            )
            for k in pl.range(TOPK):
                w_fp = pl.read(expert_weights, [b, k])

                gather_r_route = b * TOPK + k
                row_fp32 = pl.cast(
                    pl.load(
                        routed_y_buf, [gather_r_route, 0], [1, HIDDEN],
                    ),
                    target_type=pl.FP32,
                )
                weighted = pl.mul(row_fp32, w_fp)
                acc = pl.add(acc, weighted)

            pl.store(
                pl.cast(acc, target_type=pl.BF16),
                [b, 0],
                moe_out,
            )

    return moe_out
