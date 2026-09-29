# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""[中文摘要] 96 头 sliding-window attention 的多卡 @pl.program(每张卡 12 头,
窗口 512,partial RoPE 1.0;无 yarn 缩放);结构与 attention_full.py 镜像,
末尾同样以 `self.tp_all_reduce(...)` 汇集 o_proj 的 partial sum。
[关键装饰器] @pl.program +
   @pl.function(level=HOST, role=Orchestrator)  ← host_orch
   @pl.function(type=Orchestration)             ← chip_orch
   @pl.function(type=InCore)                    ← 各 InCore kernel body
   @pl.jit.inline 模块级 helper(本文件内 + _ops.py 复制)
[SPMD 角色] 跨卡 SPMD(TP=8 切头)+ 片上 SPMD(`pl.spmd(...)` 多核分派);
sliding-window mask 与 TP 切片正交,不影响通信。
[详见] 中文架构指南 §3, §4.2, §4.5, §6

────── 以下为英文原 docstring ──────

Step3p5 SWA (sliding-window) attention kernel — TP=8 in-place refactor (Phase 9 Wave 2).

Each rank holds a sliding-attention shard:

  - q_proj output: NUM_HEADS_SWA_LOCAL * HEAD_DIM = 12 * 128 = 1536
  - k_proj/v_proj output: KV_HEADS_LOCAL * HEAD_DIM = 1 * 128 = 128
  - o_proj input: NUM_HEADS_SWA_LOCAL * HEAD_DIM = 1536
  - w_g output:   NUM_HEADS_SWA_LOCAL          = 12
  - q_norm / k_norm gamma [HEAD_DIM=128] — REPLICATED on every rank

Compile-time constants baked in (LOCAL means per-rank-after-TP-slicing):

  - NUM_HEADS  = NUM_HEADS_SWA_LOCAL   (12)
  - HIDDEN_Q   = HIDDEN_Q_SWA_LOCAL    (1536)
  - KV_HIDDEN_DIM = KV_HIDDEN_LOCAL    (128)
  - NUM_KV_HEADS_DIM = KV_HEADS_LOCAL  (1)
  - Q_PER_KV   = Q_PER_KV_SWA          (12 ; invariant under TP)
  - Q_HEAD_BATCH = Q_HEAD_BATCH_SWA    (12)
  - Q_HEAD_PAD = Q_HEAD_PAD_SWA        (24)
  - ROTARY_HALF = ROTARY_HALF_SWA      (64 ; partial_rotary_factor = 1.0)
  - ROTARY_DIM  = 2 * ROTARY_HALF      (128 == HEAD_DIM, no pass-through)
  - SLIDING_WINDOW = 512 (per-position mask; orthogonal to TP slicing)

TP collective epilogue
----------------------
After the local o_proj (column-sliced) each rank holds a *partial*
``[BATCH, HIDDEN]`` BF16 sum. ``tp_all_reduce`` sums these across the
TP group so every rank ends up with the fully-reduced o_proj output;
the residual add (``+ current_hidden``) happens afterwards (the
residual is replicated across ranks, so adding it post-all-reduce keeps
the math correct).

The caller (Wave-3 ``decode_layer.py`` / ``decode_fwd.py``) must provide
a per-call-site scratch ``tmp_window`` and ``signal_window`` pair with
the documented shapes — see ``attention_full.py`` for the full contract,
re-stated here for symmetry:

  - ``tmp_window``    : ``pld.DistributedTensor`` view of a
                         ``BATCH * (HIDDEN // TP_WORLD_SIZE) * 2`` byte
                         ``alloc_window_buffer`` slot (BF16).
  - ``signal_window`` : ``pld.DistributedTensor[[SIGNAL_WINDOW_ROWS, 1],
                        pl.INT32]``, zero-initialised. Standalone/MTP bind
                        ``SIGNAL_WINDOW_ROWS=TP_WORLD_SIZE`` for independent
                        compact backings; canonical Main binds its 512B
                        stacked slot.
                        Each call site allocates a
                        fresh signal-window slot because the ring
                        all-reduce increments the cells across its
                        ``2 * (N - 1)`` steps; reusing a slot would
                        corrupt the wait thresholds in subsequent
                        collectives.

Per-layer ``rope_theta = 1e4`` (no yarn scaling on SWA layers per
``yarn_only_types = ["full_attention"]``). The sliding-window mask
selects ``[max(0, seq_len - SLIDING_WINDOW), seq_len)`` from the
chronological paged block table. An unaligned 512-token window spans
five physical blocks; aligned windows span four.

TODO(phase-3 SWA cache layout): linear layout retained for Wave 2; the
rotating-slot variant ``((start_pos + s) % WIN)`` is deferred until the
Wave-3 decode_fwd integration.

TODO(npu-tuning): Q_HEAD_PAD_SWA=24 / set_validshape(scores, 12, ...)
exercises the dual-AIV no-op replay path at half-pad == 12, which is
UNTESTED on hardware. Inherited from the single-card SWA draft.
"""

# pyright: reportUndefinedVariable=false

from __future__ import annotations

import pypto.language as pl
import pypto.language.distributed as pld

from ._ops import (
    build_plain_rope_tables,
    head_wise_gate_apply,
    partial_rope_rotate,
)
from .config import (
    ATTN_SCALE,
    BATCH,
    BATCH_TILE,
    BLOCK_SIZE,
    BLOCK_TABLE_FLAT_DYN,
    EPS,
    HEAD_DIM,
    HEAD_DIM_INV,
    HIDDEN,
    HIDDEN_INV,
    HIDDEN_Q_SWA_LOCAL,
    INPUT_PROJ_K_CHUNK,
    K_CHUNK,
    KV_PROJ_K_CHUNK,
    KV_PROJ_K_CHUNK_LOCAL,
    KV_CACHE_ROWS_DYN,
    KV_HEADS_LOCAL,
    KV_HIDDEN_LOCAL,
    LAYER_DYN,
    LAYER_HIDDEN_ROWS_DYN,
    LAYER_ROPE_THETA,
    LAYER_TYPE_SWA,
    LAYER_TYPES,
    MAX_BLOCKS_PER_SEQ,
    MAX_SEQ_DEFAULT,
    NUM_HEADS_SWA_LOCAL,
    NUM_HEADS_SWA_LOCAL_PAD,
    OUT_PROJ_K_CHUNK,
    SWA_OUT_PROJ_FUSE_CAST,
    SWA_OUT_PROJ_MATMUL_N_CHUNK,
    SWA_OUT_PROJ_MATMUL_TILES_PER_TASK,
    SWA_OUT_PROJ_VEC_N_CHUNK,
    Q_HEAD_BATCH_SWA,
    Q_HEAD_PAD_SWA,
    Q_OUT_CHUNK,
    Q_PER_KV_SWA,
    ROPE_SEQ_DYN,
    ROTARY_HALF_SWA,
    SLIDING_WINDOW,
    TP_WORLD_SIZE,
    USER_BATCH_DYN,
    is_full_attention,
)

NUM_HEADS = NUM_HEADS_SWA_LOCAL
HIDDEN_Q = HIDDEN_Q_SWA_LOCAL
KV_HIDDEN_DIM = KV_HIDDEN_LOCAL
NUM_KV_HEADS_DIM = KV_HEADS_LOCAL
Q_PER_KV = Q_PER_KV_SWA
Q_HEAD_BATCH = Q_HEAD_BATCH_SWA
Q_HEAD_PAD = Q_HEAD_PAD_SWA
ROTARY_HALF = ROTARY_HALF_SWA
ROTARY_DIM = ROTARY_HALF * 2
Q_GROUPS = Q_PER_KV // Q_HEAD_BATCH                # 1
TOTAL_Q_GROUPS = NUM_KV_HEADS_DIM * Q_GROUPS       # 1
WIN_BLOCKS = (SLIDING_WINDOW + BLOCK_SIZE - 1) // BLOCK_SIZE
# Standalone/MTP wrappers use compact independent signals.  Canonical Main
# overrides this symbolic formal while inlining the body.
SIGNAL_WINDOW_ROWS = TP_WORLD_SIZE

# Local override for KV projection's output chunk: KV_HIDDEN_LOCAL (128) is
# below the global KV_OUT_CHUNK=256 default, so we pick the whole local KV
# hidden in a single chunk.
KV_OUT_CHUNK_LOCAL = KV_HIDDEN_LOCAL

# Per-layer rows for the o_proj weight: model-bound (= n_sliding_attn ×
# HIDDEN_Q_SWA_LOCAL). Kept STATIC (not pl.dynamic) so the codegen-emitted L3
# host_orch does not reference an unresolved ``LAYER_QHIDDEN_ROWS_DYN`` global
# (pl.dynamic → NameError in the generated host_orch.py under DistributedWorker;
# same pypto limitation attention_full.py:153 works around with its 12288). It
# is computed from config so it tracks the per-rank width (HIDDEN_Q_SWA_LOCAL
# = 1536) and the tp1-unslice width automatically, matching the wo weight stack.
_N_SWA_LAYERS = sum(1 for _t in LAYER_TYPES if _t == LAYER_TYPE_SWA)
LAYER_QHIDDEN_ROWS_DYN = _N_SWA_LAYERS * HIDDEN_Q_SWA_LOCAL

assert Q_HEAD_PAD % 4 == 0 and Q_HEAD_PAD // 2 >= Q_HEAD_BATCH
assert SLIDING_WINDOW % BLOCK_SIZE == 0
assert BATCH % 2 == 0, (
    "fa_fused pipelines pairs of batches under TP, so BATCH must be even"
)
assert HIDDEN_Q % OUT_PROJ_K_CHUNK == 0
assert HIDDEN % SWA_OUT_PROJ_MATMUL_N_CHUNK == 0
assert HIDDEN % SWA_OUT_PROJ_VEC_N_CHUNK == 0
assert 0 < SWA_OUT_PROJ_MATMUL_TILES_PER_TASK <= (
    HIDDEN // SWA_OUT_PROJ_MATMUL_N_CHUNK
)
assert HIDDEN % TP_WORLD_SIZE == 0
assert KV_HIDDEN_DIM == KV_OUT_CHUNK_LOCAL


# =============================================================================
# Attention body — local compute through gated attn_out, partial o_proj,
# TP all-reduce, then residual add.
# =============================================================================
@pl.jit.inline
def attention_swa(
    current_hidden: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
    input_rms_weight: pl.Tensor[[LAYER_DYN, HIDDEN], pl.FP32],
    wq: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, HIDDEN_Q_SWA_LOCAL], pl.BF16],
    wk: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, KV_HIDDEN_LOCAL], pl.BF16],
    wv: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, KV_HIDDEN_LOCAL], pl.BF16],
    q_norm_weight: pl.Tensor[[LAYER_DYN, HEAD_DIM], pl.FP32],
    k_norm_weight: pl.Tensor[[LAYER_DYN, HEAD_DIM], pl.FP32],
    seq_lens: pl.Tensor[[USER_BATCH_DYN], pl.INT32],
    block_table: pl.Tensor[[BLOCK_TABLE_FLAT_DYN], pl.INT32],
    slot_mapping: pl.Tensor[[USER_BATCH_DYN], pl.INT32],
    rope_cos: pl.Tensor[[ROPE_SEQ_DYN, ROTARY_HALF_SWA * 2], pl.FP32],
    rope_sin: pl.Tensor[[ROPE_SEQ_DYN, ROTARY_HALF_SWA * 2], pl.FP32],
    k_cache: pl.Tensor[[KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16],
    v_cache: pl.Tensor[[KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16],
    wo: pl.Tensor[[LAYER_QHIDDEN_ROWS_DYN, HIDDEN], pl.BF16],
    w_g: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, NUM_HEADS_SWA_LOCAL_PAD], pl.BF16],
    gate_r: pl.Tensor[[NUM_HEADS_SWA_LOCAL_PAD, HIDDEN_Q_SWA_LOCAL], pl.BF16],
    resid1_out: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
    norm_layer_idx: pl.Scalar[pl.INT32],
    attn_layer_idx: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    # The collective stages and remotely loads the complete hidden row.
    # Keep the inline formal span equal to the real [BATCH,HIDDEN] access; a
    # narrowed HIDDEN//TP formal corrupts provenance for non-zero stacked slots.
    tmp_window: pld.DistributedTensor[[BATCH, HIDDEN], pl.BF16],
    signal_window: pld.DistributedTensor[[SIGNAL_WINDOW_ROWS, 1], pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
) -> pl.Tensor[[BATCH, HIDDEN], pl.BF16]:
    """Step3p5 SWA-attention layer through TP-reduced o_proj + residual."""

    decode_scope1_hidden_blocks = HIDDEN // INPUT_PROJ_K_CHUNK
    kv_proj_hidden_blocks = HIDDEN // KV_PROJ_K_CHUNK_LOCAL
    out_proj_k_blocks = HIDDEN_Q_SWA_LOCAL // OUT_PROJ_K_CHUNK
    decode_attn_scale = ATTN_SCALE
    num_layers_actual = pl.tensor.dim(input_rms_weight, 0)
    decode_layer_cache_rows = pl.tensor.dim(k_cache, 0) // num_layers_actual
    user_batch = pl.tensor.dim(seq_lens, 0)
    bt_stride = pl.tensor.dim(block_table, 0) // user_batch
    batch_padded = BATCH
    active_tokens = pl.cast(num_tokens, pl.INDEX)
    if active_tokens < 0:
        active_tokens = pl.cast(0, pl.INDEX)
    if active_tokens > BATCH:
        active_tokens = pl.cast(BATCH, pl.INDEX)

    # Persist the replicated residual before the long attention task chain.
    # Runtime-loop lowering must not recover current_hidden through a stale
    # pre-call SSA version after the TP collective; the caller-owned output
    # formal provides an explicit producer/consumer lineage.
    for resid_hold_task in pl.spmd(
        BATCH // BATCH_TILE,
        name_hint="attn_residual_hold",
        allow_early_resolve=True,
    ):
        resid_hold_b0 = resid_hold_task * BATCH_TILE
        resid1_out = pl.assemble(
            resid1_out,
            pl.slice(
                current_hidden,
                [BATCH_TILE, HIDDEN],
                [resid_hold_b0, 0],
            ),
            [resid_hold_b0, 0],
        )

    layer_hidden_base = attn_layer_idx * HIDDEN
    layer_qhidden_base = attn_layer_idx * HIDDEN_Q_SWA_LOCAL
    layer_cache_base = norm_layer_idx * decode_layer_cache_rows

    q_proj = pl.create_tensor([BATCH, HIDDEN_Q_SWA_LOCAL], dtype=pl.FP32)
    k_proj = pl.create_tensor([BATCH, KV_HIDDEN_LOCAL], dtype=pl.FP32)
    v_proj = pl.create_tensor([BATCH, KV_HIDDEN_LOCAL], dtype=pl.FP32)
    q_proj_norm = pl.create_tensor([BATCH, HIDDEN_Q_SWA_LOCAL], dtype=pl.FP32)
    k_proj_norm = pl.create_tensor([BATCH, KV_HIDDEN_LOCAL], dtype=pl.FP32)
    normed_all = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
    gate_score_t = pl.create_tensor([BATCH, NUM_HEADS_SWA_LOCAL_PAD], dtype=pl.BF16)
    gate_exp = pl.create_tensor([BATCH, HIDDEN_Q_SWA_LOCAL], dtype=pl.BF16)
    HEAD_GATE_K_SPLITS = 8
    HEAD_GATE_K_PER_SPLIT = HIDDEN // HEAD_GATE_K_SPLITS
    SWA_Q_OUT_CHUNK = Q_OUT_CHUNK // 2
    gate_logits_partial = pl.create_tensor(
        [BATCH, HEAD_GATE_K_SPLITS * NUM_HEADS_SWA_LOCAL_PAD], dtype=pl.FP32,
    )
    # Head-gate is computed on-device in Scope 1.f below (RESTORED, path (a)):
    # gate_exp = expand_per_head(sigmoid(normed_all @ w_g)) via block-diag R
    # (= gate_r input, layer-independent). The N=16 matmul_acc codegen bug that
    # once forced this worker-side is fixed on the current stack. Mirrors
    # attention_full.py. gate_r now holds R [NUM_HEADS_SWA_LOCAL_PAD,
    # HIDDEN_Q_SWA_LOCAL]; w_g is the per-attn-layer stacked gate weight.

    # ----- Scope 1.a — zero-centred input RMSNorm. -----
    # input_rms_weight is replicated across TP ranks (HIDDEN dim is not
    # sliced); every rank computes the same normed_all tile.
    for rms_spmd_idx in pl.spmd(BATCH // BATCH_TILE, name_hint="swa_rmsnorm_zc"):
        rms_b0 = rms_spmd_idx * BATCH_TILE
        partial_sq = pl.full([1, BATCH_TILE], dtype=pl.FP32, value=0.0)
        for kb in pl.range(decode_scope1_hidden_blocks):
            sq_k0 = kb * INPUT_PROJ_K_CHUNK
            sq_chunk = pl.cast(
                pl.slice(current_hidden, [BATCH_TILE, INPUT_PROJ_K_CHUNK], [rms_b0, sq_k0]),
                target_type=pl.FP32,
            )
            partial_sq = pl.add(
                partial_sq,
                pl.reshape(pl.row_sum(pl.mul(sq_chunk, sq_chunk)), [1, BATCH_TILE]),
            )
        variance = pl.reshape(
            pl.add(pl.mul(partial_sq, HIDDEN_INV), EPS), [BATCH_TILE, 1],
        )
        inv_rms = pl.recip(pl.sqrt(variance))
        for kb in pl.range(decode_scope1_hidden_blocks):
            norm_k0 = kb * INPUT_PROJ_K_CHUNK
            norm_chunk = pl.cast(
                pl.slice(current_hidden, [BATCH_TILE, INPUT_PROJ_K_CHUNK], [rms_b0, norm_k0]),
                target_type=pl.FP32,
            )
            gamma = pl.slice(input_rms_weight, [1, INPUT_PROJ_K_CHUNK], [norm_layer_idx, norm_k0])
            scaled = pl.row_expand_mul(norm_chunk, inv_rms)
            normed = pl.col_expand_mul(scaled, pl.add(gamma, 1.0))
            normed_all = pl.assemble(
                normed_all, pl.cast(normed, target_type=pl.BF16), [rms_b0, norm_k0],
            )

    # ----- Scope 1.f — on-device head-gate (RESTORED, path (a)). -----
    # gate_exp[b, h*HEAD_DIM + d] = sigmoid(normed_all @ w_g)[b, h], expanded
    # across HEAD_DIM via the block-diag constant R (= gate_r). Matches vLLM
    # modeling_step3p5 L489 + L527-531. Two scopes bound the UB working set.
    for hg_part in pl.spmd(
        HEAD_GATE_K_SPLITS,
        name_hint="swa_head_gate_logits_mm",
        allow_early_resolve=True,
    ):
        hg_k0 = hg_part * HEAD_GATE_K_PER_SPLIT
        hg_logits = pl.matmul(
            pl.slice(normed_all, [BATCH, INPUT_PROJ_K_CHUNK], [0, hg_k0]),
            pl.slice(w_g, [INPUT_PROJ_K_CHUNK, NUM_HEADS_SWA_LOCAL_PAD],
                     [layer_hidden_base + hg_k0, 0]),
            out_dtype=pl.FP32,
        )
        for kb in pl.range(1, HEAD_GATE_K_PER_SPLIT // INPUT_PROJ_K_CHUNK):
            k0 = hg_k0 + kb * INPUT_PROJ_K_CHUNK
            hg_logits = pl.matmul_acc(
                hg_logits,
                pl.slice(normed_all, [BATCH, INPUT_PROJ_K_CHUNK], [0, k0]),
                pl.slice(w_g, [INPUT_PROJ_K_CHUNK, NUM_HEADS_SWA_LOCAL_PAD],
                         [layer_hidden_base + k0, 0]),
            )
        gate_logits_partial = pl.assemble(
            gate_logits_partial,
            hg_logits,
            [0, hg_part * NUM_HEADS_SWA_LOCAL_PAD],
        )
    with pl.at(
        level=pl.Level.CORE_GROUP,
        name_hint="swa_head_gate_sigmoid_expand",
        allow_early_resolve=True,
    ):
        hg_logits = pl.slice(
            gate_logits_partial,
            [BATCH, NUM_HEADS_SWA_LOCAL_PAD],
            [0, 0],
        )
        for hg_part in pl.range(1, HEAD_GATE_K_SPLITS):
            hg_logits = pl.add(
                hg_logits,
                pl.slice(
                    gate_logits_partial,
                    [BATCH, NUM_HEADS_SWA_LOCAL_PAD],
                    [0, hg_part * NUM_HEADS_SWA_LOCAL_PAD],
                ),
            )
        hg_score = pl.recip(pl.add(pl.exp(pl.neg(hg_logits)), 1.0))
        gate_score_t[:, :] = pl.cast(hg_score, target_type=pl.BF16)
    swa_head_gate_chunks = HIDDEN_Q_SWA_LOCAL // K_CHUNK
    for hg_task in pl.spmd(
        (BATCH // BATCH_TILE) * swa_head_gate_chunks,
        name_hint="swa_head_gate_expand",
        allow_early_resolve=True,
    ):
        hg_b_idx = hg_task // swa_head_gate_chunks
        hg_n_idx = hg_task % swa_head_gate_chunks
        hg_b0 = hg_b_idx * BATCH_TILE
        hg_n0 = hg_n_idx * K_CHUNK
        hg_r = pl.slice(
            gate_r, [NUM_HEADS_SWA_LOCAL_PAD, K_CHUNK], [0, hg_n0],
        )
        hg_ge = pl.matmul(
            pl.slice(
                gate_score_t,
                [BATCH_TILE, NUM_HEADS_SWA_LOCAL_PAD],
                [hg_b0, 0],
            ),
            hg_r,
            out_dtype=pl.FP32,
        )
        gate_exp = pl.assemble(
            gate_exp,
            pl.cast(hg_ge, target_type=pl.BF16),
            [hg_b0, hg_n0],
        )

    # ----- Scope 1.b — Q projection. -----
    # wq is row-sliced (output dim → HIDDEN_Q_SWA_LOCAL = 1536 per rank).
    for q_spmd_idx in pl.spmd(
        (BATCH // BATCH_TILE) * (HIDDEN_Q_SWA_LOCAL // SWA_Q_OUT_CHUNK),
        name_hint="swa_q_proj",
        allow_early_resolve=True,
    ):
        q_b_idx = q_spmd_idx // (HIDDEN_Q_SWA_LOCAL // SWA_Q_OUT_CHUNK)
        q_ob = q_spmd_idx % (HIDDEN_Q_SWA_LOCAL // SWA_Q_OUT_CHUNK)
        q_b0 = q_b_idx * BATCH_TILE
        q_o0 = q_ob * SWA_Q_OUT_CHUNK
        q_tile_a_0 = pl.slice(normed_all, [BATCH_TILE, INPUT_PROJ_K_CHUNK], [q_b0, 0])
        q_tile_b_0 = pl.slice(
            wq, [INPUT_PROJ_K_CHUNK, SWA_Q_OUT_CHUNK], [layer_hidden_base, q_o0],
        )
        q_acc = pl.matmul(q_tile_a_0, q_tile_b_0, out_dtype=pl.FP32)
        for kb in pl.range(1, decode_scope1_hidden_blocks):
            q_k0 = kb * INPUT_PROJ_K_CHUNK
            q_tile_a = pl.slice(normed_all, [BATCH_TILE, INPUT_PROJ_K_CHUNK], [q_b0, q_k0])
            q_tile_b = pl.slice(
                wq,
                [INPUT_PROJ_K_CHUNK, SWA_Q_OUT_CHUNK],
                [layer_hidden_base + q_k0, q_o0],
            )
            q_acc = pl.matmul_acc(q_acc, q_tile_a, q_tile_b)
        q_proj = pl.assemble(q_proj, q_acc, [q_b0, q_o0])

    # ----- Scope 1.c/1.d — K/V projections. -----
    # wk is row-sliced (output dim → KV_HEADS_LOCAL * HEAD_DIM = 128 per rank).
    for kv_spmd_idx in pl.spmd(
        2 * (BATCH // BATCH_TILE),
        name_hint="swa_kv_proj",
        allow_early_resolve=True,
    ):
        kv_kind = kv_spmd_idx % 2
        kv_b0 = (kv_spmd_idx // 2) * BATCH_TILE
        if kv_kind == 0:
            k_o0 = 0
            k_tile_a = pl.slice(normed_all, [BATCH_TILE, KV_PROJ_K_CHUNK_LOCAL], [kv_b0, 0])
            k_acc = pl.matmul(
                k_tile_a,
                pl.slice(wk, [KV_PROJ_K_CHUNK_LOCAL, KV_HIDDEN_LOCAL], [layer_hidden_base, k_o0]),
                out_dtype=pl.FP32,
            )
            for kb in pl.range(1, kv_proj_hidden_blocks):
                k_k0 = kb * KV_PROJ_K_CHUNK_LOCAL
                k_acc = pl.matmul_acc(
                    k_acc, pl.slice(normed_all, [BATCH_TILE, KV_PROJ_K_CHUNK_LOCAL], [kv_b0, k_k0]),
                    pl.slice(wk, [KV_PROJ_K_CHUNK_LOCAL, KV_HIDDEN_LOCAL], [layer_hidden_base + k_k0, k_o0]),
                )
            k_proj = pl.assemble(k_proj, k_acc, [kv_b0, k_o0])
        else:
            v_o0 = 0
            v_tile_a = pl.slice(normed_all, [BATCH_TILE, KV_PROJ_K_CHUNK_LOCAL], [kv_b0, 0])
            v_acc = pl.matmul(
                v_tile_a,
                pl.slice(wv, [KV_PROJ_K_CHUNK_LOCAL, KV_HIDDEN_LOCAL], [layer_hidden_base, v_o0]),
                out_dtype=pl.FP32,
            )
            for kb in pl.range(1, kv_proj_hidden_blocks):
                v_k0 = kb * KV_PROJ_K_CHUNK_LOCAL
                v_acc = pl.matmul_acc(
                    v_acc, pl.slice(normed_all, [BATCH_TILE, KV_PROJ_K_CHUNK_LOCAL], [kv_b0, v_k0]),
                    pl.slice(wv, [KV_PROJ_K_CHUNK_LOCAL, KV_HIDDEN_LOCAL], [layer_hidden_base + v_k0, v_o0]),
                )
            v_proj = pl.assemble(v_proj, v_acc, [kv_b0, v_o0])

    # ----- Scope 1.e — per-head zero-centred q_norm / k_norm. -----
    # q_norm / k_norm gamma [HEAD_DIM] are REPLICATED across TP ranks; only
    # the per-head loop bounds shrink (KV_HEADS_LOCAL = 1 per rank).
    # Q and K branches run sequentially in the flat spmd body. The pl.range(2)
    # sub-tiling (6 heads per sub-tile, [BATCH_TILE*6, HEAD_DIM] FP32 = 49152 B)
    # keeps the Q-chain Vec/UB peak at ~98688 B (~97 KB clear of the 192 KB
    # ceiling). After pl.range(2) exits all Q tiles have been assembled into
    # q_proj_norm, so the K branch starts with a clean slate.
    for qkn_spmd_idx in pl.spmd(
        (BATCH // BATCH_TILE) * KV_HEADS_LOCAL, name_hint="swa_qk_norm_zc",
    ):
        qkn_b_idx = qkn_spmd_idx // KV_HEADS_LOCAL
        qkn_h = qkn_spmd_idx % KV_HEADS_LOCAL
        qkn_b0 = qkn_b_idx * BATCH_TILE

        # Q branch: 2 half-head sub-tiles (Q_HEAD_BATCH_SWA//2 = 6 heads each).
        qkn_q0 = qkn_h * Q_PER_KV_SWA * HEAD_DIM
        q_gamma = pl.slice(q_norm_weight, [1, HEAD_DIM], [norm_layer_idx, 0])
        for qh_sub in pl.range(2):
            q_sub_col0 = qkn_q0 + qh_sub * (Q_HEAD_BATCH_SWA // 2) * HEAD_DIM
            q_chunk_sub = pl.reshape(
                pl.slice(
                    q_proj,
                    [BATCH_TILE, (Q_HEAD_BATCH_SWA // 2) * HEAD_DIM],
                    [qkn_b0, q_sub_col0],
                ),
                [BATCH_TILE * (Q_HEAD_BATCH_SWA // 2), HEAD_DIM],
            )
            q_sq = pl.row_sum(pl.mul(q_chunk_sub, q_chunk_sub))
            q_inv = pl.rsqrt(pl.add(pl.mul(q_sq, HEAD_DIM_INV), EPS))
            q_scaled = pl.row_expand_mul(q_chunk_sub, q_inv)
            q_normed = pl.col_expand_mul(q_scaled, pl.add(q_gamma, 1.0))
            q_normed_flat = pl.reshape(
                q_normed, [BATCH_TILE, (Q_HEAD_BATCH_SWA // 2) * HEAD_DIM],
            )
            q_proj_norm = pl.assemble(q_proj_norm, q_normed_flat, [qkn_b0, q_sub_col0])

        # K branch: single head per rank (KV_HEADS_LOCAL = 1).
        qkn_k0 = qkn_h * HEAD_DIM
        k_chunk = pl.slice(k_proj, [BATCH_TILE, HEAD_DIM], [qkn_b0, qkn_k0])
        k_gamma = pl.slice(k_norm_weight, [1, HEAD_DIM], [norm_layer_idx, 0])
        k_sq = pl.row_sum(pl.mul(k_chunk, k_chunk))
        k_inv = pl.rsqrt(pl.add(pl.mul(k_sq, HEAD_DIM_INV), EPS))
        k_scaled = pl.row_expand_mul(k_chunk, k_inv)
        k_normed = pl.col_expand_mul(k_scaled, pl.add(k_gamma, 1.0))
        k_proj_norm = pl.assemble(k_proj_norm, k_normed, [qkn_b0, qkn_k0])

    # ----- Scope 1.f head-gate is now computed above (RESTORED, path (a)). ----
    # gate_exp was computed on-device right after Scope 1.a; o_proj (Scope 3.a)
    # applies it inline. Mirrors attention_full.py.

    # ----- Scope 2 — full RoPE + paged KV cache write + fa_fused (SWA). -----
    # Full RoPE (rotary_dim == HEAD_DIM, no pass-through tail). The KV cache
    # holds KV_HEADS_LOCAL = 1 KV head per rank, so the per-rank loop over
    # local KV heads collapses to a single iteration.
    attn_out = pl.create_tensor([BATCH, HIDDEN_Q_SWA_LOCAL], dtype=pl.BF16)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="swa_attn_out_zero"):
        attn_out[:, :] = pl.full(
            [BATCH, HIDDEN_Q_SWA_LOCAL], dtype=pl.BF16, value=0.0,
        )
    # Q_HEAD_PAD_SWA=24 is not a multiple of 16; use SWA_Q_PAD_ALIGNED=32 for
    # all alloc_tile row-dimension uses so the allocator alignment check passes.
    SWA_Q_PAD_ALIGNED = 32
    all_q_padded = pl.create_tensor(
        [BATCH * KV_HEADS_LOCAL * (Q_PER_KV_SWA // Q_HEAD_BATCH_SWA) * SWA_Q_PAD_ALIGNED, HEAD_DIM], dtype=pl.BF16,
    )

    # Separate active-row tasks write Q and update the resident K/V cache.
    with pl.spmd(
        active_tokens,
        name_hint="swa_rope_q",
        allow_early_resolve=True,
    ) as swa_rope_q_tid:
        b = pl.tile.get_block_idx()
        if b < active_tokens:
            ctx_len = pl.tensor.read(seq_lens, [b])
            pos = ctx_len - 1
            cos_row = pl.slice(rope_cos, [1, ROTARY_HALF_SWA * 2], [pos, 0])
            sin_row = pl.slice(rope_sin, [1, ROTARY_HALF_SWA * 2], [pos, 0])
            cos_lo = pl.slice(cos_row, [1, ROTARY_HALF_SWA], [0, 0])
            cos_hi = pl.slice(cos_row, [1, ROTARY_HALF_SWA], [0, ROTARY_HALF_SWA])
            sin_lo = pl.slice(sin_row, [1, ROTARY_HALF_SWA], [0, 0])
            sin_hi = pl.slice(sin_row, [1, ROTARY_HALF_SWA], [0, ROTARY_HALF_SWA])

            for ki in pl.range(KV_HEADS_LOCAL):
                # Per-head RoPE using CONTIGUOUS [1, ROTARY_HALF_SWA] slices of
                # q_proj_norm (mirrors the K path), replacing reshape([.,HEAD_DIM])
                # + col-offset slice which miscompiled the rot_q_hi write into
                # all_q_padded. See attention_full.py Scope 2 / _stage_scope12_qk.py.
                q_base = ki * Q_PER_KV_SWA
                pad_row_base = b * KV_HEADS_LOCAL * (Q_PER_KV_SWA // Q_HEAD_BATCH_SWA) * SWA_Q_PAD_ALIGNED + ki * SWA_Q_PAD_ALIGNED
                for qh in pl.range(Q_HEAD_BATCH_SWA):
                    qh_col = (q_base + qh) * HEAD_DIM
                    q_lo_h = pl.slice(q_proj_norm, [1, ROTARY_HALF_SWA], [b, qh_col])
                    q_hi_h = pl.slice(
                        q_proj_norm, [1, ROTARY_HALF_SWA], [b, qh_col + ROTARY_HALF_SWA],
                    )
                    rot_q_lo_h = pl.sub(
                        pl.col_expand_mul(q_lo_h, cos_lo),
                        pl.col_expand_mul(q_hi_h, sin_lo),
                    )
                    rot_q_hi_h = pl.add(
                        pl.col_expand_mul(q_hi_h, cos_hi),
                        pl.col_expand_mul(q_lo_h, sin_hi),
                    )
                    q_row = pad_row_base + qh
                    all_q_padded = pl.assemble(
                        all_q_padded, pl.cast(rot_q_lo_h, target_type=pl.BF16), [q_row, 0],
                    )
                    all_q_padded = pl.assemble(
                        all_q_padded, pl.cast(rot_q_hi_h, target_type=pl.BF16),
                        [q_row, ROTARY_HALF_SWA],
                    )
                all_q_padded = pl.assemble(
                    all_q_padded,
                    pl.cast(
                        pl.full([SWA_Q_PAD_ALIGNED - Q_HEAD_BATCH_SWA, HEAD_DIM],
                                dtype=pl.FP32, value=0.0),
                        target_type=pl.BF16,
                    ),
                    [pad_row_base + Q_HEAD_BATCH_SWA, 0],
                )

    with pl.spmd(
        active_tokens,
        name_hint="swa_rope_kv_cache",
        allow_early_resolve=True,
    ) as swa_rope_kv_tid:
        b = pl.tile.get_block_idx()
        if b < active_tokens:
            ctx_len = pl.tensor.read(seq_lens, [b])
            pos = ctx_len - 1
            slot = pl.tensor.read(slot_mapping, [b])
            slot_block = slot // BLOCK_SIZE
            slot_offset = slot - slot_block * BLOCK_SIZE
            cos_row = pl.slice(rope_cos, [1, ROTARY_HALF_SWA * 2], [pos, 0])
            sin_row = pl.slice(rope_sin, [1, ROTARY_HALF_SWA * 2], [pos, 0])
            cos_lo = pl.slice(cos_row, [1, ROTARY_HALF_SWA], [0, 0])
            cos_hi = pl.slice(cos_row, [1, ROTARY_HALF_SWA], [0, ROTARY_HALF_SWA])
            sin_lo = pl.slice(sin_row, [1, ROTARY_HALF_SWA], [0, 0])
            sin_hi = pl.slice(sin_row, [1, ROTARY_HALF_SWA], [0, ROTARY_HALF_SWA])

            for ki in pl.range(KV_HEADS_LOCAL):
                kv_col = ki * HEAD_DIM
                cache_row = (
                    layer_cache_base
                    + (slot_block * KV_HEADS_LOCAL + ki) * BLOCK_SIZE
                    + slot_offset
                )
                k_lo = pl.slice(k_proj_norm, [1, ROTARY_HALF_SWA], [b, kv_col])
                k_hi = pl.slice(
                    k_proj_norm, [1, ROTARY_HALF_SWA], [b, kv_col + ROTARY_HALF_SWA],
                )
                rot_k_lo = pl.sub(
                    pl.col_expand_mul(k_lo, cos_lo),
                    pl.col_expand_mul(k_hi, sin_lo),
                )
                rot_k_hi = pl.add(
                    pl.col_expand_mul(k_hi, cos_hi),
                    pl.col_expand_mul(k_lo, sin_hi),
                )
                k_cache = pl.assemble(
                    k_cache, pl.cast(rot_k_lo, target_type=pl.BF16), [cache_row, 0],
                )
                k_cache = pl.assemble(
                    k_cache, pl.cast(rot_k_hi, target_type=pl.BF16),
                    [cache_row, ROTARY_HALF_SWA],
                )
                v_cache = pl.assemble(
                    v_cache,
                    pl.cast(pl.slice(v_proj, [1, HEAD_DIM], [b, kv_col]),
                            target_type=pl.BF16),
                    [cache_row, 0],
                )


    # ----- fa_fused (SWA) — Phase A (2026-06-11): qwen3/32b-style 4-spmd. -----
    # Mirror of attention_full.py's Phase A rewrite. SWA differs only in:
    #   * Q_HEAD_BATCH_SWA=12 / Q_HEAD_PAD_SWA=24 / SWA_Q_PAD_ALIGNED=32
    #     (instead of full's 8/16/16)
    #   * the block-table range covers the trailing token window.
    # See docs/step3p5/phases/15-singlerank-npu.md "Phase A route decision".
    # Localise the module-level WIN_BLOCKS constant: pypto IR's frontend does
    # not lift bare module globals computed inside the file (only ``from
    # .config import`` names round-trip through the trace).
    SWA_WIN_BLOCKS = (SLIDING_WINDOW + BLOCK_SIZE - 1) // BLOCK_SIZE
    SWA_STORAGE_BLOCKS = SWA_WIN_BLOCKS + 1
    all_raw_scores = pl.create_tensor(
        [BATCH * SWA_STORAGE_BLOCKS * SWA_Q_PAD_ALIGNED, BLOCK_SIZE],
        dtype=pl.FP32,
    )
    all_exp_padded = pl.create_tensor(
        [BATCH * SWA_STORAGE_BLOCKS * SWA_Q_PAD_ALIGNED, BLOCK_SIZE],
        dtype=pl.BF16,
    )
    # mi/li 以 [1, SWA_Q_PAD_ALIGNED] 宽行落盘，避免从 GM 直接 slice
    # 出 FP32 [N,1] 窄列 tile。Padded heads 只参与中间计算，最终会裁掉。
    all_cur_mi = pl.create_tensor(
        [BATCH * SWA_STORAGE_BLOCKS, SWA_Q_PAD_ALIGNED], dtype=pl.FP32,
    )
    all_cur_li = pl.create_tensor(
        [BATCH * SWA_STORAGE_BLOCKS, SWA_Q_PAD_ALIGNED], dtype=pl.FP32,
    )
    all_oi_tmp = pl.create_tensor(
        [BATCH * SWA_STORAGE_BLOCKS * SWA_Q_PAD_ALIGNED, HEAD_DIM],
        dtype=pl.FP32,
    )
    # Launch exactly one logical task per active row. Keep the launch extent
    # as a loop-carried SSA value: older codegen can leave a direct
    # ``active_tokens`` alias unresolved in ``set_block_num`` when this inline
    # helper is embedded in chip orchestration. A zero-token request must be
    # handled by the graph-level no-op/reject contract because later
    # collectives are not valid for that case.
    swa_active_tasks = pl.cast(0, pl.INDEX)
    for swa_count_b in pl.range(active_tokens):
        swa_active_tasks = swa_active_tasks + 1
    # Stage 1: QK matmul (cube). One core per active batch row.
    with pl.spmd(
        swa_active_tasks,
        name_hint="swa_qk_matmul",
        deps=[swa_rope_q_tid, swa_rope_kv_tid],
        allow_early_resolve=True,
    ) as swa_qk_tid:
        fa_b = pl.tile.get_block_idx()
        if fa_b < active_tokens:
            fa_ctx_len = pl.tensor.read(seq_lens, [fa_b])
            fa_window_start = pl.max(0, fa_ctx_len - SLIDING_WINDOW)
            fa_first_block = fa_window_start // BLOCK_SIZE
            fa_end_block = (
                fa_ctx_len + BLOCK_SIZE - 1
            ) // BLOCK_SIZE
            fa_ctx_blocks = fa_end_block - fa_first_block
            fa_block_table_base = fa_b * bt_stride
            q_padded_row = fa_b * SWA_Q_PAD_ALIGNED  # KV_HEADS_LOCAL=1, Q_GROUPS=1
            q_padded = pl.slice(
                all_q_padded, [SWA_Q_PAD_ALIGNED, HEAD_DIM], [q_padded_row, 0],
            )
            for sb in pl.range(fa_ctx_blocks):
                fa_pbid = pl.cast(
                    pl.tensor.read(
                        block_table,
                        [fa_block_table_base + fa_first_block + sb],
                    ),
                    pl.INDEX,
                )
                fa_cache_row = layer_cache_base + fa_pbid * BLOCK_SIZE
                k_tile = pl.slice(
                    k_cache,
                    [BLOCK_SIZE, HEAD_DIM],
                    [fa_cache_row, 0],
                )
                raw_scores = pl.matmul(
                    q_padded, k_tile, b_trans=True, out_dtype=pl.FP32,
                )
                scratch_row = (
                    fa_b * SWA_STORAGE_BLOCKS + sb
                ) * SWA_Q_PAD_ALIGNED
                all_raw_scores = pl.assemble(all_raw_scores, raw_scores, [scratch_row, 0])

    # Stage 2: softmax (vec).
    with pl.spmd(
        swa_active_tasks,
        name_hint="swa_softmax",
        deps=[swa_qk_tid],
        allow_early_resolve=True,
    ) as swa_softmax_tid:
        fa_b = pl.tile.get_block_idx()
        if fa_b < active_tokens:
            fa_ctx_len = pl.tensor.read(seq_lens, [fa_b])
            fa_window_start = pl.max(0, fa_ctx_len - SLIDING_WINDOW)
            fa_first_block = fa_window_start // BLOCK_SIZE
            fa_end_block = (
                fa_ctx_len + BLOCK_SIZE - 1
            ) // BLOCK_SIZE
            fa_ctx_blocks = fa_end_block - fa_first_block
            for sb in pl.range(fa_ctx_blocks):
                fa_block = fa_first_block + sb
                fa_block_token0 = fa_block * BLOCK_SIZE
                valid_lo = pl.max(
                    0,
                    fa_window_start - fa_block_token0,
                )
                valid_hi = pl.min(
                    BLOCK_SIZE,
                    fa_ctx_len - fa_block_token0,
                )
                valid_len = valid_hi - valid_lo
                scratch_row = (
                    fa_b * SWA_STORAGE_BLOCKS + sb
                ) * SWA_Q_PAD_ALIGNED
                raw_scores = pl.slice(
                    all_raw_scores,
                    [SWA_Q_PAD_ALIGNED, BLOCK_SIZE],
                    [scratch_row, 0],
                )
                scores = pl.mul(raw_scores, decode_attn_scale)
                if valid_len < BLOCK_SIZE:
                    score_cols = pl.arange(
                        0,
                        [1, BLOCK_SIZE],
                        dtype=pl.INT32,
                    )
                    zero_i32 = pl.const(0, pl.INT32)
                    one_i32 = pl.const(1, pl.INT32)
                    valid_from_i32 = pl.minimum(
                        pl.maximum(
                            pl.add(
                                pl.sub(
                                    score_cols,
                                    pl.cast(valid_lo, pl.INT32),
                                ),
                                one_i32,
                            ),
                            zero_i32,
                        ),
                        one_i32,
                    )
                    valid_to_i32 = pl.minimum(
                        pl.maximum(
                            pl.neg(
                                pl.sub(
                                    score_cols,
                                    pl.cast(valid_hi, pl.INT32),
                                ),
                            ),
                            zero_i32,
                        ),
                        one_i32,
                    )
                    valid_mask = pl.cast(
                        pl.mul(valid_from_i32, valid_to_i32),
                        target_type=pl.FP32,
                    )
                    invalid_bias = pl.mul(
                        pl.sub(valid_mask, 1.0),
                        1.0e20,
                    )
                    scores = pl.col_expand_add(scores, invalid_bias)
                cur_mi = pl.row_max(scores)
                exp_scores = pl.exp(pl.row_expand_sub(scores, cur_mi))
                exp_scores_bf16 = pl.cast(exp_scores, target_type=pl.BF16)
                exp_scores_fp32 = pl.cast(exp_scores_bf16, target_type=pl.FP32)
                cur_li = pl.row_sum(exp_scores_fp32)
                all_exp_padded = pl.assemble(
                    all_exp_padded, exp_scores_bf16, [scratch_row, 0],
                )
                lm_row = fa_b * SWA_STORAGE_BLOCKS + sb
                all_cur_mi = pl.assemble(
                    all_cur_mi,
                    pl.reshape(cur_mi, [1, SWA_Q_PAD_ALIGNED]),
                    [lm_row, 0],
                )
                all_cur_li = pl.assemble(
                    all_cur_li,
                    pl.reshape(cur_li, [1, SWA_Q_PAD_ALIGNED]),
                    [lm_row, 0],
                )

    # Stage 3: SV matmul (cube). exp_tile uses the full SWA_Q_PAD_ALIGNED row
    # stride; matmul output's bottom (SWA_Q_PAD_ALIGNED - Q_HEAD_BATCH_SWA)
    # rows are garbage from un-initialised GM but are never read by Stage 4.
    with pl.spmd(
        swa_active_tasks,
        name_hint="swa_sv_matmul",
        deps=[swa_softmax_tid],
        allow_early_resolve=True,
    ) as swa_sv_tid:
        fa_b = pl.tile.get_block_idx()
        if fa_b < active_tokens:
            fa_ctx_len = pl.tensor.read(seq_lens, [fa_b])
            fa_window_start = pl.max(0, fa_ctx_len - SLIDING_WINDOW)
            fa_first_block = fa_window_start // BLOCK_SIZE
            fa_end_block = (
                fa_ctx_len + BLOCK_SIZE - 1
            ) // BLOCK_SIZE
            fa_ctx_blocks = fa_end_block - fa_first_block
            fa_block_table_base = fa_b * bt_stride
            for sb in pl.range(fa_ctx_blocks):
                fa_pbid = pl.cast(
                    pl.tensor.read(
                        block_table,
                        [fa_block_table_base + fa_first_block + sb],
                    ),
                    pl.INDEX,
                )
                fa_cache_row = layer_cache_base + fa_pbid * BLOCK_SIZE
                v_tile = pl.slice(
                    v_cache,
                    [BLOCK_SIZE, HEAD_DIM],
                    [fa_cache_row, 0],
                )
                scratch_row = (
                    fa_b * SWA_STORAGE_BLOCKS + sb
                ) * SWA_Q_PAD_ALIGNED
                exp_tile = pl.slice(
                    all_exp_padded,
                    [SWA_Q_PAD_ALIGNED, BLOCK_SIZE],
                    [scratch_row, 0],
                )
                oi_tmp = pl.matmul(exp_tile, v_tile, out_dtype=pl.FP32)
                all_oi_tmp = pl.assemble(all_oi_tmp, oi_tmp, [scratch_row, 0])

    # Stage 4: online softmax accumulation + final normalisation + attn_out
    # write. mi/li 从合法宽行读取后 reshape 成 reduction column。
    # Operations across padded rows produce garbage but are sliced off
    # before attn_out write (real heads only). valid_shape is omitted on
    # the stage-4 slices because pypto's frontend rejects re-assigning a
    # valid_shape-tagged tile back through arithmetic ops that drop the
    # valid_shape attribute (e.g., `li = pl.add(pl.mul(alpha, li), ...)`).
    with pl.spmd(
        swa_active_tasks,
        name_hint="swa_online_softmax",
        deps=[swa_sv_tid],
        allow_early_resolve=True,
    ) as _swa_online_softmax_tid:
        fa_b = pl.tile.get_block_idx()
        if fa_b < active_tokens:
            fa_ctx_len = pl.tensor.read(seq_lens, [fa_b])
            fa_window_start = pl.max(0, fa_ctx_len - SLIDING_WINDOW)
            fa_first_block = fa_window_start // BLOCK_SIZE
            fa_end_block = (
                fa_ctx_len + BLOCK_SIZE - 1
            ) // BLOCK_SIZE
            fa_ctx_blocks = fa_end_block - fa_first_block
            oi_row0 = (
                fa_b * SWA_STORAGE_BLOCKS * SWA_Q_PAD_ALIGNED
            )
            lm_row0 = fa_b * SWA_STORAGE_BLOCKS
            oi = pl.slice(
                all_oi_tmp, [SWA_Q_PAD_ALIGNED, HEAD_DIM], [oi_row0, 0],
            )
            mi = pl.reshape(
                pl.slice(
                    all_cur_mi, [1, SWA_Q_PAD_ALIGNED], [lm_row0, 0],
                ),
                [SWA_Q_PAD_ALIGNED, 1],
            )
            li = pl.reshape(
                pl.slice(
                    all_cur_li, [1, SWA_Q_PAD_ALIGNED], [lm_row0, 0],
                ),
                [SWA_Q_PAD_ALIGNED, 1],
            )
            for sb in pl.range(1, fa_ctx_blocks):
                sb_oi_row = oi_row0 + sb * SWA_Q_PAD_ALIGNED
                sb_lm_row = lm_row0 + sb
                # Use _blk-suffixed names to avoid pypto frontend's strict
                # type-equality check vs stage 2's `cur_mi`/`cur_li` (which
                # are row_max/row_sum products without valid_shape).
                oi_partial_blk = pl.slice(
                    all_oi_tmp, [SWA_Q_PAD_ALIGNED, HEAD_DIM], [sb_oi_row, 0],
                )
                cur_mi_blk = pl.reshape(
                    pl.slice(
                        all_cur_mi, [1, SWA_Q_PAD_ALIGNED], [sb_lm_row, 0],
                    ),
                    [SWA_Q_PAD_ALIGNED, 1],
                )
                cur_li_blk = pl.reshape(
                    pl.slice(
                        all_cur_li, [1, SWA_Q_PAD_ALIGNED], [sb_lm_row, 0],
                    ),
                    [SWA_Q_PAD_ALIGNED, 1],
                )
                mi_new = pl.maximum(mi, cur_mi_blk)
                alpha = pl.exp(pl.sub(mi, mi_new))
                beta = pl.exp(pl.sub(cur_mi_blk, mi_new))
                li = pl.add(pl.mul(alpha, li), pl.mul(beta, cur_li_blk))
                oi = pl.add(
                    pl.row_expand_mul(oi, alpha),
                    pl.row_expand_mul(oi_partial_blk, beta),
                )
                mi = mi_new
            ctx = pl.row_expand_div(oi, li)
            # ctx shape [SWA_Q_PAD_ALIGNED=32, HEAD_DIM] with valid_shape on the
            # first Q_HEAD_BATCH_SWA=12 rows. Cast to BF16 first (col_byte_size
            # 32*2=64B, 32B aligned), reshape to row-major flat [1, 32*HEAD_DIM],
            # then slice to the real Q_HEAD_BATCH_SWA*HEAD_DIM elements before
            # writing to attn_out (HIDDEN_Q_SWA_LOCAL = 12 * HEAD_DIM = 1536).
            ctx_bf16 = pl.cast(ctx, target_type=pl.BF16)
            ctx_padded_flat = pl.reshape(
                ctx_bf16, [1, SWA_Q_PAD_ALIGNED * HEAD_DIM],
            )
            ctx_flat_bf16 = pl.slice(
                ctx_padded_flat, [1, Q_HEAD_BATCH_SWA * HEAD_DIM], [0, 0],
            )
            # q_base = kvh * Q_PER_KV_SWA == 0 (KV_HEADS_LOCAL=1, kvh=0).
            attn_out = pl.assemble(attn_out, ctx_flat_bf16, [fa_b, 0])

    # ----- Scope 2.5 — head-wise gate is applied inline in o_proj below. -----
    # gate_exp was computed on-device in Scope 1.f (gate_exp = sigmoid(normed_all
    # @ w_g) @ R). The o_proj scope below multiplies attn_out by gate_exp per
    # K-chunk (wide element-wise, no [N,1] broadcast → no UB-align fault).
    # Mirrors attention_full.py Scope 3.a.

    # ----- Scope 3.a — local o_proj with inline head-gate (element-wise). -----
    # wo is column-sliced (input dim → HIDDEN_Q_SWA_LOCAL = 1536 per rank);
    # the output is a partial [BATCH, HIDDEN] BF16 tensor that must be
    # summed across the TP group via the all-reduce below before residual.
    #
    # Keep the fused mixed task unsplit across batch rows. BATCH_TILE=16 and
    # UP_DOWN would give each AIV lane an M=8 row fragment while AIC publishes
    # one full M=16 accumulator. Grouped N tiles would then reuse that split
    # C2V/V2C pipe for multiple publications, which can expose stale lower
    # rows. The unsplit [BATCH_TILE, N] accumulator fits the tile budget.
    # Declare both candidate destinations outside the compile-time feature
    # branches.  PyPTO converts the DSL to SSA before it folds config-backed
    # constant branches, so branch-local tensor declarations are otherwise
    # diagnosed as escaping their defining scope.
    partial_attn_proj = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
    partial_attn_proj_fp32 = pl.create_tensor([BATCH, HIDDEN], dtype=pl.FP32)
    swa_out_proj_n_tiles = HIDDEN // SWA_OUT_PROJ_MATMUL_N_CHUNK
    swa_out_proj_tasks = (
        swa_out_proj_n_tiles
        + SWA_OUT_PROJ_MATMUL_TILES_PER_TASK - 1
    ) // SWA_OUT_PROJ_MATMUL_TILES_PER_TASK
    for b0 in pl.parallel(0, batch_padded, BATCH_TILE):
        for out_task in pl.spmd(
            swa_out_proj_tasks,
            name_hint="swa_out_proj_matmul",
        ):
            # Same scheduler-grain control as full attention.  The SWA K
            # width is different, so it is tuned independently.
            for out_local in pl.range(
                SWA_OUT_PROJ_MATMUL_TILES_PER_TASK
            ):
                out_tile = (
                    out_task * SWA_OUT_PROJ_MATMUL_TILES_PER_TASK
                    + out_local
                )
                if out_tile < swa_out_proj_n_tiles:
                    o0 = out_tile * SWA_OUT_PROJ_MATMUL_N_CHUNK
                    # First K-chunk (kb=0). Gate applied inline.
                    hg_exp_0 = pl.cast(
                        pl.slice(
                            gate_exp,
                            [BATCH_TILE, OUT_PROJ_K_CHUNK],
                            [b0, 0],
                        ),
                        target_type=pl.FP32,
                    )
                    a_chunk_raw_0 = pl.slice(
                        attn_out,
                        [BATCH_TILE, OUT_PROJ_K_CHUNK],
                        [b0, 0],
                    )
                    a_gated_fp32_0 = pl.mul(
                        pl.cast(a_chunk_raw_0, target_type=pl.FP32),
                        hg_exp_0,
                    )
                    a_chunk_0 = pl.cast(
                        a_gated_fp32_0, target_type=pl.BF16,
                    )
                    w_chunk_0 = pl.slice(
                        wo,
                        [OUT_PROJ_K_CHUNK, SWA_OUT_PROJ_MATMUL_N_CHUNK],
                        [layer_qhidden_base, o0],
                    )
                    o_acc = pl.matmul(
                        a_chunk_0, w_chunk_0, out_dtype=pl.FP32,
                    )
                    for kb in pl.range(1, out_proj_k_blocks):
                        k0 = kb * OUT_PROJ_K_CHUNK
                        hg_exp = pl.cast(
                            pl.slice(
                                gate_exp,
                                [BATCH_TILE, OUT_PROJ_K_CHUNK],
                                [b0, k0],
                            ),
                            target_type=pl.FP32,
                        )
                        a_chunk_raw = pl.slice(
                            attn_out,
                            [BATCH_TILE, OUT_PROJ_K_CHUNK],
                            [b0, k0],
                        )
                        a_gated_fp32 = pl.mul(
                            pl.cast(a_chunk_raw, target_type=pl.FP32),
                            hg_exp,
                        )
                        a_chunk = pl.cast(
                            a_gated_fp32, target_type=pl.BF16,
                        )
                        w_chunk = pl.slice(
                            wo,
                            [
                                OUT_PROJ_K_CHUNK,
                                SWA_OUT_PROJ_MATMUL_N_CHUNK,
                            ],
                            [layer_qhidden_base + k0, o0],
                        )
                        o_acc = pl.matmul_acc(
                            o_acc, a_chunk, w_chunk,
                        )
                    if SWA_OUT_PROJ_FUSE_CAST != 0:
                        partial_attn_proj = pl.assemble(
                            partial_attn_proj,
                            pl.cast(o_acc, target_type=pl.BF16),
                            [b0, o0],
                        )
                    else:
                        partial_attn_proj_fp32 = pl.assemble(
                            partial_attn_proj_fp32, o_acc, [b0, o0],
                        )

    if SWA_OUT_PROJ_FUSE_CAST == 0:
        for b0 in pl.parallel(0, batch_padded, BATCH_TILE):
            for ob in pl.spmd(
                HIDDEN // SWA_OUT_PROJ_VEC_N_CHUNK,
                name_hint="swa_out_proj_cast",
            ):
                o0 = ob * SWA_OUT_PROJ_VEC_N_CHUNK
                fp32_chunk = pl.slice(
                    partial_attn_proj_fp32,
                    [BATCH_TILE, SWA_OUT_PROJ_VEC_N_CHUNK], [b0, o0],
                )
                partial_attn_proj = pl.assemble(
                    partial_attn_proj,
                    pl.cast(fp32_chunk, target_type=pl.BF16),
                    [b0, o0],
                )

    # ----- Scope 3.b — TP all-reduce(sum) of the partial o_proj output. -----
    # The pull-side ring all-reduce body now lives as a class method on
    # the enclosing @pl.program class (see TpAttentionSwa.tp_all_reduce
    # below). Phase X.2 lifted the call out of the @pl.jit.inline wrapper
    # in collectives.py — mixing the two pypto worlds is unsupported.
    # Phase 15.1 mirror of attention_full 15.B: at TP=1 skip the call so
    # orchestration codegen does not emit a stale SSA rename for the
    # SimplifyPass-elided ring loop body.
    if TP_WORLD_SIZE > 1:
        partial_attn_proj = self.tp_all_reduce(
            partial_attn_proj,
            tmp_window,
            signal_window,
            my_rank,
        )

    # ----- Scope 3.c — residual add (post-all-reduce). -----
    for b0 in pl.parallel(0, batch_padded, BATCH_TILE):
        for ob in pl.spmd(
            HIDDEN // SWA_OUT_PROJ_VEC_N_CHUNK,
            name_hint="swa_out_resid_add",
        ):
            o0 = ob * SWA_OUT_PROJ_VEC_N_CHUNK
            reduced = pl.cast(
                pl.slice(
                    partial_attn_proj,
                    [BATCH_TILE, SWA_OUT_PROJ_VEC_N_CHUNK],
                    [b0, o0],
                ),
                target_type=pl.FP32,
            )
            resid = pl.cast(
                pl.slice(
                    resid1_out,
                    [BATCH_TILE, SWA_OUT_PROJ_VEC_N_CHUNK],
                    [b0, o0],
                ),
                target_type=pl.FP32,
            )
            resid_sum = pl.add(reduced, resid)
            resid1_out = pl.assemble(
                resid1_out, pl.cast(resid_sum, target_type=pl.BF16), [b0, o0],
            )

    return resid1_out


# =============================================================================
# TP wrapper — Wave-2 program scaffolding (chip_orch + host_orch).
# =============================================================================
def _build_tp_attention_swa_program(tp_size: int = TP_WORLD_SIZE):
    """Return a freshly-built ``@pl.program`` class for the SWA-attention
    TP epilogue.

    Constructed inside a function so the module imports even on hosts that
    have not finished bringing up the pypto runtime (deferred-build
    pattern).
    """
    if HIDDEN % tp_size != 0:
        raise ValueError(
            f"HIDDEN={HIDDEN} must be divisible by tp_size={tp_size}"
        )
    attention_swa_inline = pl.inline(attention_swa._func)
    tp_chunk = HIDDEN // tp_size

    @pl.program
    class TpAttentionSwa:
        # ---------- Collective: TP all_reduce (barrier-style) ------------
        # Mirrors pypto/tests/st/distributed/test_l3_allreduce.py.
        @pl.function(type=pl.FunctionType.InCore)
        def tp_all_reduce(
            self,
            local: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
            tmp_window: pld.DistributedTensor[[BATCH, HIDDEN], pl.BF16],
            signal_window: pld.DistributedTensor[[tp_size, 1], pl.INT32],
            my_rank: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[BATCH, HIDDEN], pl.BF16]:
            """Barrier-style all-reduce(sum) across the TP group."""
            group_size = tp_size

            # All-reduce HIDDEN tiling width: fixed, INDEPENDENT of tp_size.
            # tp_chunk = HIDDEN // tp_size collapses to HIDDEN (4096) at
            # tp_size=1 (apply_tp1_patch single-card e2e), so [BATCH, 4096]
            # FP32 acc tiles (256KB) overflow the 188KB UB limit. A fixed
            # tile keeps the per-iteration working set bounded for every
            # tp_size (512 = canonical TP=8 chunk; HIDDEN is divisible by it).
            ar_chunk = HIDDEN // 8
            for k0 in pl.range(0, HIDDEN, ar_chunk):
                stage_tile = pl.load(local, [0, k0], [BATCH, ar_chunk])
                pl.store(stage_tile, [0, k0], tmp_window)

            for peer in pl.range(group_size):
                if peer != my_rank:
                    pld.system.notify(
                        target=signal_window, peer=peer,
                        offsets=[my_rank, 0], value=1,
                        op=pld.NotifyOp.AtomicAdd,
                    )
            for src in pl.range(group_size):
                if src != my_rank:
                    pld.system.wait(
                        signal=signal_window, offsets=[src, 0],
                        expected=1, cmp=pld.WaitCmp.Ge,
                    )

            for k0 in pl.range(0, HIDDEN, ar_chunk):
                own_tile = pl.load(tmp_window, [0, k0], [BATCH, ar_chunk])
                acc = pl.cast(own_tile, target_type=pl.FP32)
                for peer in pl.range(group_size):
                    if peer != my_rank:
                        recv = pld.tile.remote_load(
                            tmp_window, peer=peer,
                            offsets=[0, k0], shape=[BATCH, ar_chunk],
                        )
                        acc = pl.add(acc, pl.cast(recv, target_type=pl.FP32))
                pl.store(
                    pl.cast(acc, target_type=pl.BF16),
                    [0, k0], local,
                )
            return local

        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(  # noqa: PLR0913
            self,
            current_hidden: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
            input_rms_weight: pl.Tensor[[LAYER_DYN, HIDDEN], pl.FP32],
            wq: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, HIDDEN_Q], pl.BF16],
            wk: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, KV_HIDDEN_DIM], pl.BF16],
            wv: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, KV_HIDDEN_DIM], pl.BF16],
            q_norm_weight: pl.Tensor[[LAYER_DYN, HEAD_DIM], pl.FP32],
            k_norm_weight: pl.Tensor[[LAYER_DYN, HEAD_DIM], pl.FP32],
            seq_lens: pl.Tensor[[USER_BATCH_DYN], pl.INT32],
            block_table: pl.Tensor[[BLOCK_TABLE_FLAT_DYN], pl.INT32],
            slot_mapping: pl.Tensor[[USER_BATCH_DYN], pl.INT32],
            rope_cos: pl.Tensor[[ROPE_SEQ_DYN, ROTARY_DIM], pl.FP32],
            rope_sin: pl.Tensor[[ROPE_SEQ_DYN, ROTARY_DIM], pl.FP32],
            k_cache: pl.Tensor[[KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16],
            v_cache: pl.Tensor[[KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16],
            wo: pl.Tensor[[LAYER_QHIDDEN_ROWS_DYN, HIDDEN], pl.BF16],
            w_g: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, NUM_HEADS], pl.BF16],
            gate_r: pl.Tensor[[NUM_HEADS_SWA_LOCAL_PAD, HIDDEN_Q_SWA_LOCAL], pl.BF16],
            resid1_out: pl.Out[pl.Tensor[[BATCH, HIDDEN], pl.BF16]],
            tmp_window: pld.DistributedTensor[[BATCH, HIDDEN], pl.BF16],
            signal_window: pld.DistributedTensor[[tp_size, 1], pl.INT32],
            norm_layer_idx: pl.Scalar[pl.INT32],
            attn_layer_idx: pl.Scalar[pl.INT32],
            my_rank: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[BATCH, HIDDEN], pl.BF16]:
            resid1_out = attention_swa_inline(
                current_hidden,
                input_rms_weight,
                wq, wk, wv,
                q_norm_weight, k_norm_weight,
                seq_lens, block_table, slot_mapping,
                rope_cos, rope_sin,
                k_cache, v_cache,
                wo, w_g,
                gate_r,
                resid1_out,
                norm_layer_idx,
                attn_layer_idx,
                BATCH,
                tmp_window,
                signal_window,
                my_rank,
            )
            return resid1_out

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(  # noqa: PLR0913, PLR0915
            self,
            current_hidden: pl.Tensor[[tp_size, BATCH, HIDDEN], pl.BF16],
            input_rms_weight: pl.Tensor[[tp_size, LAYER_DYN, HIDDEN], pl.FP32],
            wq: pl.Tensor[[tp_size, LAYER_HIDDEN_ROWS_DYN, HIDDEN_Q], pl.BF16],
            wk: pl.Tensor[
                [tp_size, LAYER_HIDDEN_ROWS_DYN, KV_HIDDEN_DIM], pl.BF16
            ],
            wv: pl.Tensor[
                [tp_size, LAYER_HIDDEN_ROWS_DYN, KV_HIDDEN_DIM], pl.BF16
            ],
            q_norm_weight: pl.Tensor[[tp_size, LAYER_DYN, HEAD_DIM], pl.FP32],
            k_norm_weight: pl.Tensor[[tp_size, LAYER_DYN, HEAD_DIM], pl.FP32],
            seq_lens: pl.Tensor[[tp_size, USER_BATCH_DYN], pl.INT32],
            block_table: pl.Tensor[[tp_size, BLOCK_TABLE_FLAT_DYN], pl.INT32],
            slot_mapping: pl.Tensor[[tp_size, USER_BATCH_DYN], pl.INT32],
            rope_cos: pl.Tensor[[tp_size, ROPE_SEQ_DYN, ROTARY_DIM], pl.FP32],
            rope_sin: pl.Tensor[[tp_size, ROPE_SEQ_DYN, ROTARY_DIM], pl.FP32],
            k_cache: pl.Tensor[[tp_size, KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16],
            v_cache: pl.Tensor[[tp_size, KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16],
            wo: pl.Tensor[[tp_size, LAYER_QHIDDEN_ROWS_DYN, HIDDEN], pl.BF16],
            w_g: pl.Tensor[[tp_size, LAYER_HIDDEN_ROWS_DYN, NUM_HEADS], pl.BF16],
            gate_r: pl.Tensor[
                [tp_size, NUM_HEADS_SWA_LOCAL_PAD, HIDDEN_Q_SWA_LOCAL], pl.BF16
            ],
            resid1_out: pl.Out[pl.Tensor[[tp_size, BATCH, HIDDEN], pl.BF16]],
            norm_layer_idx: pl.Scalar[pl.INT32],
            attn_layer_idx: pl.Scalar[pl.INT32],
        ):
            tmp_buf = pld.alloc_window_buffer(BATCH * HIDDEN * 2)  # BF16
            sig_buf = pld.alloc_window_buffer(tp_size * 4)           # INT32

            for r in pl.range(pld.world_size()):
                tmp_window = pld.window(tmp_buf, [BATCH, HIDDEN], dtype=pl.BF16)
                signal_window = pld.window(sig_buf, [tp_size, 1], dtype=pl.INT32)
                self.chip_orch(
                    current_hidden[r],
                    input_rms_weight[r],
                    wq[r], wk[r], wv[r],
                    q_norm_weight[r], k_norm_weight[r],
                    seq_lens[r], block_table[r], slot_mapping[r],
                    rope_cos[r], rope_sin[r],
                    k_cache[r], v_cache[r],
                    wo[r], w_g[r],
                    gate_r[r],
                    resid1_out[r],
                    tmp_window,
                    signal_window,
                    norm_layer_idx,
                    attn_layer_idx,
                    r,
                    device=r,
                )

    return TpAttentionSwa


# =============================================================================
# Distributed-mock torch reference and harness.
# =============================================================================
def _torch_single_card_attention_swa(
    *,
    hidden_states,
    input_rms_weight,
    wq_full,
    wk_full,
    wv_full,
    q_norm_weight,
    k_norm_weight,
    wo_full,
    w_g_full,
    seq_lens,
    block_table,
    slot_mapping,
    rope_cos,
    rope_sin,
    k_cache_full,
    v_cache_full,
    num_heads_full,
    num_kv_heads_full,
    head_dim,
    rotary_half,
    q_per_kv,
    eps,
    block_size,
    sliding_window,
):
    """Pure-torch single-card oracle for the SWA path.

    SWA variant: ``rotary_dim == head_dim`` (no pass-through tail), and the
    visible range is ``[max(0, seq_len - sliding_window), seq_len)``.
    """
    import math

    import torch

    batch = hidden_states.shape[0]
    hidden_q = num_heads_full * head_dim
    scale = 1.0 / math.sqrt(head_dim)

    def zc(x, g):
        return x * (g + 1.0)

    x = hidden_states.float()
    var = x.pow(2).mean(dim=-1, keepdim=True)
    normed_bf16 = zc(x * torch.rsqrt(var + eps), input_rms_weight.float()).bfloat16()

    q_proj = normed_bf16.float() @ wq_full.float()
    k_proj = normed_bf16.float() @ wk_full.float()
    v_proj = normed_bf16.float() @ wv_full.float()

    def per_head(x_flat, num_h, gamma):
        gef = gamma.float() + 1.0
        xh = x_flat.view(batch, num_h, head_dim).float()
        return (
            xh * torch.rsqrt(xh.pow(2).mean(-1, keepdim=True) + eps) * gef
        ).view(batch, num_h * head_dim)

    q_proj_norm = per_head(q_proj, num_heads_full, q_norm_weight[0:1, :])
    k_proj_norm = per_head(k_proj, num_kv_heads_full, k_norm_weight[0:1, :])

    gate_logits = normed_bf16.float() @ w_g_full.float()
    k_cache = k_cache_full.clone()
    v_cache = v_cache_full.clone()
    max_ctx_blocks = MAX_BLOCKS_PER_SEQ
    attn_out = torch.zeros(batch, hidden_q, dtype=torch.bfloat16)

    for b in range(batch):
        ctx_len = int(seq_lens[b].item())
        window_start = max(0, ctx_len - sliding_window)
        first_block = window_start // block_size
        end_block = (ctx_len + block_size - 1) // block_size
        pos = ctx_len - 1

        cr = rope_cos[pos : pos + 1, :]
        sr = rope_sin[pos : pos + 1, :]
        c_lo, c_hi = cr[:, :rotary_half], cr[:, rotary_half:]
        s_lo, s_hi = sr[:, :rotary_half], sr[:, rotary_half:]

        slot = int(slot_mapping[b].item())
        sb_blk = slot // block_size
        sb_off = slot % block_size
        kh = k_proj_norm[b].view(num_kv_heads_full, head_dim).float()
        k_rot = torch.cat([
            kh[:, :rotary_half] * c_lo - kh[:, rotary_half:] * s_lo,
            kh[:, rotary_half:] * c_hi + kh[:, :rotary_half] * s_hi,
        ], dim=-1)
        for ki in range(num_kv_heads_full):
            row = (sb_blk * num_kv_heads_full + ki) * block_size + sb_off
            k_cache[row, :] = k_rot[ki].to(torch.bfloat16)
            v_cache[row, :] = v_proj[
                b, ki * head_dim : (ki + 1) * head_dim,
            ].to(torch.bfloat16)

        qh = q_proj_norm[b].view(num_heads_full, head_dim).float()
        q_rot = torch.cat([
            qh[:, :rotary_half] * c_lo - qh[:, rotary_half:] * s_lo,
            qh[:, rotary_half:] * c_hi + qh[:, :rotary_half] * s_hi,
        ], dim=-1)

        attn_row = torch.zeros(1, hidden_q, dtype=torch.bfloat16)
        for kvh in range(num_kv_heads_full):
            q_base = kvh * q_per_kv
            q_grp = q_rot[q_base : q_base + q_per_kv, :].to(torch.bfloat16)
            oi = torch.zeros(q_per_kv, head_dim)
            li = torch.zeros(q_per_kv, 1)
            mi = torch.zeros(q_per_kv, 1)
            for block in range(first_block, end_block):
                block_token0 = block * block_size
                valid_lo = max(window_start, block_token0) - block_token0
                valid_hi = min(ctx_len, block_token0 + block_size) - block_token0
                pbid = int(block_table[b * max_ctx_blocks + block].item())
                cr0 = (pbid * num_kv_heads_full + kvh) * block_size
                kt = k_cache[cr0 : cr0 + block_size, :]
                vt = v_cache[cr0 : cr0 + block_size, :]
                rs = q_grp.float() @ kt.float().T
                rs[:, :valid_lo] = torch.finfo(torch.float32).min
                rs[:, valid_hi:] = torch.finfo(torch.float32).min
                scores = rs * scale
                cm = scores.max(dim=-1, keepdim=True).values
                es = torch.exp(scores - cm)
                es_b = es.to(torch.bfloat16)
                cl = es_b.float().sum(dim=-1, keepdim=True)
                ot = es_b.float() @ vt.float()
                if block == first_block:
                    oi, li, mi = ot, cl, cm
                else:
                    mn = torch.maximum(mi, cm)
                    a = torch.exp(mi - mn)
                    bw = torch.exp(cm - mn)
                    li = a * li + bw * cl
                    oi = oi * a + ot * bw
                    mi = mn
            ctx = oi / li
            attn_row[
                :, q_base * head_dim : (q_base + q_per_kv) * head_dim,
            ] = ctx.reshape(1, -1).to(torch.bfloat16)
        attn_out[b : b + 1, :] = attn_row

    gate = torch.sigmoid(gate_logits).bfloat16().float().unsqueeze(-1)
    gated = (
        attn_out.view(batch, num_heads_full, head_dim).float() * gate
    ).view(batch, num_heads_full * head_dim).to(torch.bfloat16)

    o = gated.float() @ wo_full.float()
    resid1 = (o + hidden_states.float()).bfloat16()
    return resid1


def _torch_per_rank_partial_swa(
    *,
    rank,
    tp_world_size,
    hidden_states,
    input_rms_weight,
    wq_full,
    wk_full,
    wv_full,
    q_norm_weight,
    k_norm_weight,
    wo_full,
    w_g_full,
    seq_lens,
    block_table,
    slot_mapping,
    rope_cos,
    rope_sin,
    k_cache_full,
    v_cache_full,
    num_heads_full,
    num_kv_heads_full,
    head_dim,
    rotary_half,
    q_per_kv,
    eps,
    block_size,
    sliding_window,
):
    """Compute one rank's partial pre-all-reduce o_proj output (SWA path)."""
    import math

    import torch

    batch = hidden_states.shape[0]
    heads_local = num_heads_full // tp_world_size
    kv_heads_local = num_kv_heads_full // tp_world_size
    hidden_q_local = heads_local * head_dim
    kv_hidden_local = kv_heads_local * head_dim
    scale = 1.0 / math.sqrt(head_dim)

    wq_local = wq_full[:, rank * hidden_q_local : (rank + 1) * hidden_q_local]
    wk_local = wk_full[:, rank * kv_hidden_local : (rank + 1) * kv_hidden_local]
    wv_local = wv_full[:, rank * kv_hidden_local : (rank + 1) * kv_hidden_local]
    wo_local = wo_full[rank * hidden_q_local : (rank + 1) * hidden_q_local, :]
    w_g_local = w_g_full[:, rank * heads_local : (rank + 1) * heads_local]

    num_blocks_total = k_cache_full.shape[0] // (num_kv_heads_full * block_size)
    k_cache_full_view = k_cache_full.view(
        num_blocks_total, num_kv_heads_full, block_size, head_dim,
    )
    v_cache_full_view = v_cache_full.view(
        num_blocks_total, num_kv_heads_full, block_size, head_dim,
    )
    k_cache_local = k_cache_full_view[
        :, rank * kv_heads_local : (rank + 1) * kv_heads_local, :, :,
    ].contiguous().view(num_blocks_total * kv_heads_local * block_size, head_dim)
    v_cache_local = v_cache_full_view[
        :, rank * kv_heads_local : (rank + 1) * kv_heads_local, :, :,
    ].contiguous().view(num_blocks_total * kv_heads_local * block_size, head_dim)

    def zc(x, g):
        return x * (g + 1.0)

    x = hidden_states.float()
    var = x.pow(2).mean(dim=-1, keepdim=True)
    normed_bf16 = zc(x * torch.rsqrt(var + eps), input_rms_weight.float()).bfloat16()

    q_proj_local = normed_bf16.float() @ wq_local.float()
    k_proj_local = normed_bf16.float() @ wk_local.float()
    v_proj_local = normed_bf16.float() @ wv_local.float()

    q_h_local = q_proj_local.view(batch, heads_local, head_dim)
    q_h_local = zc(
        q_h_local * torch.rsqrt(q_h_local.pow(2).mean(-1, keepdim=True) + eps),
        q_norm_weight.float(),
    )
    k_h_local = k_proj_local.view(batch, kv_heads_local, head_dim)
    k_h_local = zc(
        k_h_local * torch.rsqrt(k_h_local.pow(2).mean(-1, keepdim=True) + eps),
        k_norm_weight.float(),
    )

    k_cache = k_cache_local.clone()
    v_cache = v_cache_local.clone()
    max_ctx_blocks = MAX_BLOCKS_PER_SEQ
    attn_out_local = torch.zeros(batch, hidden_q_local, dtype=torch.bfloat16)
    for b in range(batch):
        ctx_len = int(seq_lens[b].item())
        window_start = max(0, ctx_len - sliding_window)
        first_block = window_start // block_size
        end_block = (ctx_len + block_size - 1) // block_size
        pos = ctx_len - 1
        cr = rope_cos[pos : pos + 1, :]
        sr = rope_sin[pos : pos + 1, :]
        c_lo, c_hi = cr[:, :rotary_half], cr[:, rotary_half:]
        s_lo, s_hi = sr[:, :rotary_half], sr[:, rotary_half:]

        kh = k_h_local[b]
        k_rot = torch.cat([
            kh[:, :rotary_half] * c_lo - kh[:, rotary_half:] * s_lo,
            kh[:, rotary_half:] * c_hi + kh[:, :rotary_half] * s_hi,
        ], dim=-1)
        slot = int(slot_mapping[b].item())
        sb_blk = slot // block_size
        sb_off = slot % block_size
        for ki in range(kv_heads_local):
            row = (sb_blk * kv_heads_local + ki) * block_size + sb_off
            k_cache[row, :] = k_rot[ki].to(torch.bfloat16)
            v_cache[row, :] = v_proj_local[
                b, ki * head_dim : (ki + 1) * head_dim,
            ].to(torch.bfloat16)

        qh = q_h_local[b]
        q_rot = torch.cat([
            qh[:, :rotary_half] * c_lo - qh[:, rotary_half:] * s_lo,
            qh[:, rotary_half:] * c_hi + qh[:, :rotary_half] * s_hi,
        ], dim=-1)

        attn_row = torch.zeros(1, hidden_q_local, dtype=torch.bfloat16)
        for kvh in range(kv_heads_local):
            q_base = kvh * q_per_kv
            q_grp = q_rot[q_base : q_base + q_per_kv, :].to(torch.bfloat16)
            oi = torch.zeros(q_per_kv, head_dim)
            li = torch.zeros(q_per_kv, 1)
            mi = torch.zeros(q_per_kv, 1)
            for block in range(first_block, end_block):
                block_token0 = block * block_size
                valid_lo = max(window_start, block_token0) - block_token0
                valid_hi = min(ctx_len, block_token0 + block_size) - block_token0
                pbid = int(block_table[b * max_ctx_blocks + block].item())
                cr0 = (pbid * kv_heads_local + kvh) * block_size
                kt = k_cache[cr0 : cr0 + block_size, :]
                vt = v_cache[cr0 : cr0 + block_size, :]
                rs = q_grp.float() @ kt.float().T
                rs[:, :valid_lo] = torch.finfo(torch.float32).min
                rs[:, valid_hi:] = torch.finfo(torch.float32).min
                scores = rs * scale
                cm = scores.max(dim=-1, keepdim=True).values
                es = torch.exp(scores - cm)
                es_b = es.to(torch.bfloat16)
                cl = es_b.float().sum(dim=-1, keepdim=True)
                ot = es_b.float() @ vt.float()
                if block == first_block:
                    oi, li, mi = ot, cl, cm
                else:
                    mn = torch.maximum(mi, cm)
                    a = torch.exp(mi - mn)
                    bw = torch.exp(cm - mn)
                    li = a * li + bw * cl
                    oi = oi * a + ot * bw
                    mi = mn
            ctx = oi / li
            attn_row[
                :, q_base * head_dim : (q_base + q_per_kv) * head_dim,
            ] = ctx.reshape(1, -1).to(torch.bfloat16)
        attn_out_local[b : b + 1, :] = attn_row

    gate_local = (
        torch.sigmoid(normed_bf16.float() @ w_g_local.float())
        .bfloat16()
        .float()
    )
    attn_view = attn_out_local.view(batch, heads_local, head_dim).float()
    attn_gated = (attn_view * gate_local.unsqueeze(-1)).to(torch.bfloat16)
    attn_gated_flat = attn_gated.view(batch, hidden_q_local)
    partial_o = (attn_gated_flat.float() @ wo_local.float()).to(torch.bfloat16)
    return partial_o


def _run_distributed_mock(
    *,
    batch,
    max_seq,
    layer_idx,
    pass_rate,
    rtol,
    atol,
    seed,
):
    """Simulate TP=TP_WORLD_SIZE ranks in a torch loop and validate (SWA)."""
    import torch

    torch.manual_seed(seed)

    if is_full_attention(layer_idx):
        raise ValueError(
            f"layer_idx={layer_idx} is not a sliding-attention layer"
        )

    layer_rope_theta = LAYER_ROPE_THETA[layer_idx]
    num_blocks = batch * MAX_BLOCKS_PER_SEQ
    num_heads_full = NUM_HEADS_SWA_LOCAL * TP_WORLD_SIZE                  # 96
    num_kv_heads_full = KV_HEADS_LOCAL * TP_WORLD_SIZE                    # 8
    hidden_q_full = num_heads_full * HEAD_DIM
    kv_hidden_full = num_kv_heads_full * HEAD_DIM
    cache_rows_full = num_blocks * num_kv_heads_full * BLOCK_SIZE

    rope_cos, rope_sin = build_plain_rope_tables(
        max_seq, ROTARY_DIM, layer_rope_theta,
    )

    synthetic_proj_scale = 0.5
    hidden_states = (torch.rand(batch, HIDDEN) - 0.5).bfloat16()
    input_rms_weight = ((torch.rand(1, HIDDEN) - 0.5) * 0.1).float()
    wq_full = (torch.rand(HIDDEN, hidden_q_full) / HIDDEN ** 0.5).bfloat16()
    wk_full = (torch.rand(HIDDEN, kv_hidden_full) / HIDDEN ** 0.5).bfloat16()
    wv_full = (
        synthetic_proj_scale * torch.rand(HIDDEN, kv_hidden_full) / HIDDEN ** 0.5
    ).bfloat16()
    q_norm_weight = ((torch.rand(1, HEAD_DIM) - 0.5) * 0.1).float()
    k_norm_weight = ((torch.rand(1, HEAD_DIM) - 0.5) * 0.1).float()
    wo_full = (
        synthetic_proj_scale * (torch.rand(hidden_q_full, HIDDEN) - 0.5)
        / hidden_q_full ** 0.5
    ).bfloat16()
    w_g_full = (
        synthetic_proj_scale * (torch.rand(HIDDEN, num_heads_full) - 0.5)
        / HIDDEN ** 0.5
    ).bfloat16()

    # Cover the SWA seq_len regimes (single tile, multi tile, full window,
    # window+1, etc) — same pattern as the single-card SWA harness.
    seq_len_pattern = torch.tensor(
        [9, 31, 62, SLIDING_WINDOW - 1, SLIDING_WINDOW, SLIDING_WINDOW + 1,
         max_seq, max_seq // 2],
        dtype=torch.int32,
    )
    repeat = (batch + seq_len_pattern.numel() - 1) // seq_len_pattern.numel()
    seq_lens = seq_len_pattern.repeat(repeat)[:batch].clone()
    seq_lens = torch.clamp(seq_lens, min=1, max=max_seq)
    block_table = torch.arange(num_blocks, dtype=torch.int32)
    slot_mapping = torch.empty(batch, dtype=torch.int32)
    for b in range(batch):
        ctx_len = int(seq_lens[b].item())
        slot_pos = ctx_len - 1
        logical_block = slot_pos // BLOCK_SIZE
        page_offset = slot_pos % BLOCK_SIZE
        phys_block = b * MAX_BLOCKS_PER_SEQ + logical_block
        slot_mapping[b] = phys_block * BLOCK_SIZE + page_offset
    k_cache_full = (torch.rand(cache_rows_full, HEAD_DIM) - 0.5).bfloat16()
    v_cache_full = (
        synthetic_proj_scale * (torch.rand(cache_rows_full, HEAD_DIM) - 0.5)
    ).bfloat16()

    expected_resid1 = _torch_single_card_attention_swa(
        hidden_states=hidden_states,
        input_rms_weight=input_rms_weight,
        wq_full=wq_full, wk_full=wk_full, wv_full=wv_full,
        q_norm_weight=q_norm_weight,
        k_norm_weight=k_norm_weight,
        wo_full=wo_full, w_g_full=w_g_full,
        seq_lens=seq_lens, block_table=block_table,
        slot_mapping=slot_mapping,
        rope_cos=rope_cos, rope_sin=rope_sin,
        k_cache_full=k_cache_full.clone(),
        v_cache_full=v_cache_full.clone(),
        num_heads_full=num_heads_full,
        num_kv_heads_full=num_kv_heads_full,
        head_dim=HEAD_DIM,
        rotary_half=ROTARY_HALF,
        q_per_kv=Q_PER_KV,
        eps=EPS,
        block_size=BLOCK_SIZE,
        sliding_window=SLIDING_WINDOW,
    )

    summed_partial = torch.zeros(batch, HIDDEN, dtype=torch.float32)
    for r in range(TP_WORLD_SIZE):
        rank_partial = _torch_per_rank_partial_swa(
            rank=r,
            tp_world_size=TP_WORLD_SIZE,
            hidden_states=hidden_states,
            input_rms_weight=input_rms_weight,
            wq_full=wq_full, wk_full=wk_full, wv_full=wv_full,
            q_norm_weight=q_norm_weight,
            k_norm_weight=k_norm_weight,
            wo_full=wo_full, w_g_full=w_g_full,
            seq_lens=seq_lens, block_table=block_table,
            slot_mapping=slot_mapping,
            rope_cos=rope_cos, rope_sin=rope_sin,
            k_cache_full=k_cache_full.clone(),
            v_cache_full=v_cache_full.clone(),
            num_heads_full=num_heads_full,
            num_kv_heads_full=num_kv_heads_full,
            head_dim=HEAD_DIM,
            rotary_half=ROTARY_HALF,
            q_per_kv=Q_PER_KV,
            eps=EPS,
            block_size=BLOCK_SIZE,
            sliding_window=SLIDING_WINDOW,
        )
        summed_partial = summed_partial + rank_partial.float()

    tp_resid1 = (summed_partial + hidden_states.float()).bfloat16()

    close = torch.isclose(tp_resid1, expected_resid1, rtol=rtol, atol=atol)
    rate = close.float().mean().item()
    n_fail = int((~close).sum().item())
    ok = rate >= pass_rate
    status = "PASS" if ok else "FAIL"
    print(
        f"[{status}] attention_swa distributed-mock: pass_rate={rate:.6f} "
        f"threshold={pass_rate:.6f} "
        f"{n_fail}/{tp_resid1.numel()} mismatched rtol={rtol} atol={atol}"
    )
    return ok


def build_tp_attention_swa_program(tp_size: int = TP_WORLD_SIZE):
    """Public wrapper for the deferred @pl.program builder."""
    return _build_tp_attention_swa_program(tp_size)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", default="a2a3sim",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"],
                        help="Reserved for the Wave-3 real-distributed harness.")
    parser.add_argument("-d", "--device", type=int, default=0,
                        help="Reserved for the Wave-3 real-distributed harness.")
    parser.add_argument("-b", "--batch", type=int, default=BATCH)
    parser.add_argument("--max-seq", type=int, default=MAX_SEQ_DEFAULT)
    parser.add_argument("--layer-idx", type=int, default=1,
                        help="Which sliding-attention layer to specialise on.")
    parser.add_argument("--pass-rate", type=float, default=0.97)
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--build-program-only", action="store_true",
                        default=False,
                        help="Just construct the @pl.program scaffold and exit.")
    args = parser.parse_args()

    program_cls = build_tp_attention_swa_program(TP_WORLD_SIZE)
    print(
        f"[OK] built @pl.program TpAttentionSwa: {program_cls.__name__} "
        f"(tp_size={TP_WORLD_SIZE})"
    )

    if args.build_program_only:
        raise SystemExit(0)

    ok = _run_distributed_mock(
        batch=args.batch,
        max_seq=args.max_seq,
        layer_idx=args.layer_idx,
        pass_rate=args.pass_rate,
        rtol=args.rtol,
        atol=args.atol,
        seed=args.seed,
    )
    if not ok:
        raise SystemExit(1)


__all__ = [
    "attention_swa",
    "build_tp_attention_swa_program",
    "_build_tp_attention_swa_program",
    "_torch_single_card_attention_swa",
    "_torch_per_rank_partial_swa",
    "_run_distributed_mock",
    "NUM_HEADS",
    "HIDDEN_Q",
    "KV_HIDDEN_DIM",
    "NUM_KV_HEADS_DIM",
    "Q_PER_KV",
    "Q_HEAD_BATCH",
    "Q_HEAD_PAD",
    "ROTARY_HALF",
    "ROTARY_DIM",
    "Q_GROUPS",
    "TOTAL_Q_GROUPS",
    "WIN_BLOCKS",
    "LAYER_QHIDDEN_ROWS_DYN",
]
