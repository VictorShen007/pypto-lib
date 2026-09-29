# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Prefill-side dispatch helpers — histogram, prefix_sum, pack_payload,
build_local_expert_csr, build_inverse_map.

v2 lift (task #13): the 4 Inline dispatch helpers from
``prefill_moe.py:507-682`` are lifted here as module-level
``@pl.jit.inline`` bodies using prefill-side names (BATCH, HIDDEN, TOPK,
N_RANKS, N_LOCAL_EXPERTS, PER_RANK_BUCKETS, LOCAL_RECV_MAX).

Math is unchanged. Lowercase closure variables (n_ranks, n_local_experts,
per_rank_buckets, local_recv_max) are replaced with uppercase module-level
constants from ``_moe_constants.py``.

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
    PER_RANK_BUCKETS,
    SCALE_W_PAD,
    TOPK,
)

__all__ = [
    "prefill_histogram_and_prefix_sum",
    "prefill_pack_send_payload",
    "prefill_build_local_expert_csr",
    "prefill_build_inverse_map",
    "prefill_zero_dispatch_buffers",
]


@pl.jit.inline
def prefill_zero_dispatch_buffers(
    send_buf: pld.DistributedTensor[[LOCAL_RECV_MAX, HIDDEN], pl.INT8],
    send_scale_buf: pld.DistributedTensor[
        [LOCAL_RECV_MAX, SCALE_W_PAD], pl.FP32
    ],
    recv_x: pld.DistributedTensor[[LOCAL_RECV_MAX, HIDDEN], pl.INT8],
    recv_scale: pld.DistributedTensor[
        [LOCAL_RECV_MAX, SCALE_W_PAD], pl.FP32
    ],
):
    """Zero-init the 4 EP dispatch data windows before pack/a2a.

    Data windows are NOT auto-zeroed (only signal windows are). The
    symmetric fixed-slot a2a reads FULL PER_PEER_BOUND rows per (src,dst)
    block; rows past the real per-expert count are gap slots that
    otherwise carry uninitialized HBM residue (fix-as-diagnosis: remove
    the uninitialized-memory input from the dispatch path, mirrors
    ``prefill_zero_routed_y_buf`` on the combine side).

    The FP32 scale windows are zeroed with ``pl.full(dtype=FP32)``
    (``pto.texpands`` supports f32). The INT8 payload windows are zeroed
    via the INT32->FP16->INT8 cast chain (``pto.texpands`` rejects i8
    directly; mirrors decode_sparse_attn.py zero-pad). This body must be
    spliced into its OWN InCore step (NOT ``moe_dispatch_step``): the
    texpands+tstore loop trips pto-memory-consistency when inlined
    (ptoas 60s hang), same constraint as ``prefill_zero_routed_y_buf``.
    """
    for r in pl.range(LOCAL_RECV_MAX):
        zero_i8 = pl.cast(
            pl.cast(
                pl.full([1, HIDDEN], dtype=pl.INT32, value=0),
                target_type=pl.FP16, mode="round",
            ),
            target_type=pl.INT8, mode="trunc",
        )
        send_buf[r : r + 1, :] = zero_i8
        recv_x[r : r + 1, :] = zero_i8
        send_scale_buf[r : r + 1, :] = pl.full(
            [1, SCALE_W_PAD], dtype=pl.FP32, value=0.0,
        )
        recv_scale[r : r + 1, :] = pl.full(
            [1, SCALE_W_PAD], dtype=pl.FP32, value=0.0,
        )

    return send_buf


@pl.jit.inline
def prefill_histogram_and_prefix_sum(
    indices: pl.Tensor[[BATCH, TOPK], pl.INT32],
    send_counts_per_bucket: pl.Tensor[[PER_RANK_BUCKETS], pl.INT32],
    send_counts_per_rank: pl.Tensor[[N_RANKS], pl.INT32],
    send_offsets_per_rank: pl.Tensor[[N_RANKS], pl.INT32],
):
    """Per-tile histogram of expert destinations + prefix-sum offsets.

    Design note: this body is pure-scalar (pl.range + pl.read/pl.write,
    no tile ops, no pl.at scope). It is designed to be inlined into a
    larger @pl.function with tile context, NOT standalone-executable.
    Standalone execution produces an AICPU-only task graph (no AICore
    work) → runtime rejects with retCode=0x2a on real device (correct
    reference behavior), hangs (SCHEDULER_TIMEOUT) on sim (sim defect,
    filed against simpler). Use (F) pure-Python emulation + compile-only
    for L0; runtime validation at L1 when integrated with tile context.
    """
    for bkt in pl.range(PER_RANK_BUCKETS):
        pl.write(
            send_counts_per_bucket, [bkt], pl.cast(0, pl.INT32),
        )
    for r in pl.range(N_RANKS):
        pl.write(
            send_counts_per_rank, [r], pl.cast(0, pl.INT32),
        )

    for t in pl.range(BATCH):
        for k in pl.range(TOPK):
            eid = pl.read(indices, [t, k])
            dst = eid // N_LOCAL_EXPERTS
            loc_e = eid - dst * N_LOCAL_EXPERTS
            bkt = dst * N_LOCAL_EXPERTS + loc_e
            cur = pl.read(send_counts_per_bucket, [bkt])
            pl.write(
                send_counts_per_bucket, [bkt],
                pl.cast(cur + 1, pl.INT32),
            )
            r_cur = pl.read(send_counts_per_rank, [dst])
            pl.write(
                send_counts_per_rank, [dst],
                pl.cast(r_cur + 1, pl.INT32),
            )

    pl.write(send_offsets_per_rank, [0], pl.cast(0, pl.INT32))
    for r in pl.range(1, N_RANKS):
        prev_off = pl.read(send_offsets_per_rank, [r - 1])
        prev_cnt = pl.read(send_counts_per_rank, [r - 1])
        pl.write(
            send_offsets_per_rank, [r],
            pl.cast(prev_off + prev_cnt, pl.INT32),
        )

    return send_counts_per_bucket


@pl.jit.inline
def prefill_pack_send_payload(
    x: pl.Tensor[[BATCH, HIDDEN], pl.INT8],
    x_scale: pl.Tensor[[BATCH, SCALE_W_PAD], pl.FP32],
    indices: pl.Tensor[[BATCH, TOPK], pl.INT32],
    send_counts_per_bucket: pl.Tensor[[PER_RANK_BUCKETS], pl.INT32],
    send_offsets_per_rank: pl.Tensor[[N_RANKS], pl.INT32],
    send_buf: pld.DistributedTensor[[LOCAL_RECV_MAX, HIDDEN], pl.INT8],
    send_scale_buf: pld.DistributedTensor[
        [LOCAL_RECV_MAX, SCALE_W_PAD], pl.FP32
    ],
    cursor_per_bucket: pl.Tensor[[PER_RANK_BUCKETS], pl.INT32],
    bucket_offset: pl.Tensor[[PER_RANK_BUCKETS], pl.INT32],
):
    """Pack per-token INT8 rows + per-token FP32 scale into send buffers.

    T2a W8A8 (task #11 item 3): the activation payload is now INT8 with a
    separate per-token FP32 dequant scale riding alongside in its own a2a
    window (mirrors ``decode_fwd.py:916-924`` dispatch_step, which carries
    the scale in ``recv_aux`` column 0). The scale cannot be packed into
    the INT8 x buffer (different dtype + the INT8 payload needs 32-byte
    row alignment, ``cols % 32 == 0``), so it travels as a
    ``[LOCAL_RECV_MAX, SCALE_W_PAD]`` FP32 window sharing the same
    symmetric fixed-slot offsets.

    Design note: ``send_buf`` / ``send_scale_buf`` are DistributedTensor
    windows (HBM, allocated by host_orch via pld.alloc_window_buffer); each
    token row is copied into them with pl.load + pl.store (no pl.assemble —
    that would force an 8MB tile into UB and overflow memory space 1). The
    rest is scalar (pl.range + pl.read/pl.write, no pl.at scope). This body
    is designed to be inlined into a larger @pl.function with tile context,
    NOT standalone-executable. Standalone execution produces an AICPU-only
    task graph (no AICore work) → runtime rejects with retCode=0x2a on
    real device (correct reference behavior), hangs (SCHEDULER_TIMEOUT)
    on sim (sim defect, filed against simpler). Use (F) pure-Python
    emulation + compile-only for L0; runtime validation at L1 when
    integrated with tile context.
    """
    for r in pl.range(N_RANKS):
        # Symmetric fixed-slot dst-block base (§D:129, mirrors decode
        # moe.py:673): each (dst) rank owns a full BATCH*TOPK slot block
        # starting at r*BATCH*TOPK, so the receiver's fixed-slot a2a read
        # offset (my_rank*BATCH*TOPK) lines up without cross-rank offset
        # data. ``send_offsets_per_rank`` becomes an unused formal (kept
        # for signature parity with the variable-length original).
        rank_off = pl.cast(r * (BATCH * TOPK), pl.INT32)
        pl.write(
            bucket_offset, [r * N_LOCAL_EXPERTS],
            pl.cast(rank_off, pl.INT32),
        )
        pl.write(
            cursor_per_bucket, [r * N_LOCAL_EXPERTS],
            pl.cast(rank_off, pl.INT32),
        )
        for e in pl.range(1, N_LOCAL_EXPERTS):
            prev_off = pl.read(
                bucket_offset, [r * N_LOCAL_EXPERTS + e - 1],
            )
            prev_cnt = pl.read(
                send_counts_per_bucket,
                [r * N_LOCAL_EXPERTS + e - 1],
            )
            new_off = pl.cast(prev_off + prev_cnt, pl.INT32)
            pl.write(
                bucket_offset, [r * N_LOCAL_EXPERTS + e], new_off,
            )
            pl.write(
                cursor_per_bucket, [r * N_LOCAL_EXPERTS + e],
                new_off,
            )

    for t in pl.range(BATCH):
        for k in pl.range(TOPK):
            eid = pl.read(indices, [t, k])
            dst = eid // N_LOCAL_EXPERTS
            loc_e = eid - dst * N_LOCAL_EXPERTS
            bkt = dst * N_LOCAL_EXPERTS + loc_e
            slot_i32 = pl.read(cursor_per_bucket, [bkt])
            slot = pl.cast(slot_i32, pl.INDEX)
            x_tile = pl.load(x, [t, 0], [1, HIDDEN])
            pl.store(x_tile, [slot, 0], send_buf)
            # Carry the per-token dequant scale alongside the INT8 activation.
            # Full SCALE_W_PAD-wide tile (col 0 = scale, cols 1..7 = pad)
            # keeps the a2a window's load/store tile 32B-aligned.
            scale_tile = pl.load(x_scale, [t, 0], [1, SCALE_W_PAD])
            pl.store(scale_tile, [slot, 0], send_scale_buf)
            pl.write(
                cursor_per_bucket, [bkt],
                pl.cast(slot_i32 + 1, pl.INT32),
            )

    return send_buf


@pl.jit.inline
def prefill_build_local_expert_csr(
    pub_counts: pld.DistributedTensor[
        [N_RANKS * N_RANKS, N_LOCAL_EXPERTS], pl.INT32
    ],
    local_expert_offset: pl.Tensor[[N_LOCAL_EXPERTS], pl.INT32],
    local_expert_count: pl.Tensor[[N_LOCAL_EXPERTS], pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
):
    """Build local-expert CSR (count + offset) from published pub_counts.

    Design note: this body is pure-scalar (pl.range + pl.read/pl.write,
    no tile ops, no pl.at scope). It is designed to be inlined into a
    larger @pl.function with tile context, NOT standalone-executable.
    Standalone execution produces an AICPU-only task graph (no AICore
    work) → runtime rejects with retCode=0x2a on real device (correct
    reference behavior), hangs (SCHEDULER_TIMEOUT) on sim (sim defect,
    filed against simpler). Use (F) pure-Python emulation + compile-only
    for L0; runtime validation at L1 when integrated with tile context.
    """
    for e in pl.range(N_LOCAL_EXPERTS):
        acc = pl.cast(0, pl.INT32)
        for s in pl.range(N_RANKS):
            acc = acc + pl.read(
                pub_counts, [s * N_RANKS + my_rank, e],
            )
        pl.write(local_expert_count, [e], pl.cast(acc, pl.INT32))

    pl.write(local_expert_offset, [0], pl.cast(0, pl.INT32))
    for e in pl.range(1, N_LOCAL_EXPERTS):
        prev_off = pl.read(local_expert_offset, [e - 1])
        prev_cnt = pl.read(local_expert_count, [e - 1])
        pl.write(
            local_expert_offset, [e],
            pl.cast(prev_off + prev_cnt, pl.INT32),
        )

    return local_expert_count


@pl.jit.inline
def prefill_build_inverse_map(
    indices: pl.Tensor[[BATCH, TOPK], pl.INT32],
    pub_counts: pld.DistributedTensor[
        [N_RANKS * N_RANKS, N_LOCAL_EXPERTS], pl.INT32
    ],
    inverse_map: pl.Tensor[[BATCH, TOPK], pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
):
    """Build the inverse map: per-route destination-row in the recv buffer.

    Design note: this body is pure-scalar (pl.range + pl.read/pl.write,
    no tile ops, no pl.at scope). It is designed to be inlined into a
    larger @pl.function with tile context, NOT standalone-executable.
    Standalone execution produces an AICPU-only task graph (no AICore
    work) → runtime rejects with retCode=0x2a on real device (correct
    reference behavior), hangs (SCHEDULER_TIMEOUT) on sim (sim defect,
    filed against simpler). Use (F) pure-Python emulation + compile-only
    for L0; runtime validation at L1 when integrated with tile context.
    """
    cursor = pl.create_tensor(
        [PER_RANK_BUCKETS], dtype=pl.INT32,
    )
    for bkt in pl.range(PER_RANK_BUCKETS):
        pl.write(cursor, [bkt], pl.cast(0, pl.INT32))

    for t in pl.range(BATCH):
        for k in pl.range(TOPK):
            eid = pl.read(indices, [t, k])
            dst = eid // N_LOCAL_EXPERTS
            loc_e = eid - dst * N_LOCAL_EXPERTS
            bkt = dst * N_LOCAL_EXPERTS + loc_e

            src_off = pl.cast(0, pl.INT32)
            for s in pl.range(N_RANKS):
                if s < my_rank:
                    src_off = src_off + pl.read(
                        pub_counts, [s * N_RANKS + dst, loc_e],
                    )

            loc_e_off = pl.cast(0, pl.INT32)
            for prev_e in pl.range(N_LOCAL_EXPERTS):
                if prev_e < loc_e:
                    for s in pl.range(N_RANKS):
                        loc_e_off = loc_e_off + pl.read(
                            pub_counts,
                            [s * N_RANKS + dst, prev_e],
                        )

            my_cursor_val = pl.read(cursor, [bkt])
            dst_row = loc_e_off + src_off + my_cursor_val
            packed = (
                dst * pl.cast(LOCAL_RECV_MAX, pl.INT32) + dst_row
            )
            pl.write(inverse_map, [t, k], pl.cast(packed, pl.INT32))
            pl.write(
                cursor, [bkt],
                pl.cast(my_cursor_val + 1, pl.INT32),
            )
