# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""[中文摘要] 64 头 full-attention 的多卡 @pl.program(每张卡 8 头,partial RoPE 0.5,
llama3-yarn 缩放);包含 Wave-2 三层(host_orch / chip_orch / InCore body),
末尾用 tp_all_reduce 汇集 o_proj 的 partial sum。
[关键装饰器] @pl.program +
   @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)  ← host_orch
   @pl.function(type=pl.FunctionType.Orchestration)              ← chip_orch
   @pl.function(type=pl.FunctionType.InCore)                     ← 各 InCore kernel body
   @pl.jit.inline 模块级 helper(本文件内 + _ops.py 复制)
[SPMD 角色] 跨卡 SPMD(TP=8 切头)+ 片上 SPMD(`pl.spmd(...)` 多核分派);
chip_orch 用 `self.tp_all_reduce(...)` 汇集 partial。
[详见] 中文架构指南 §3, §4.2, §4.5, §6

────── 以下为英文原 docstring ──────

Step3p5 full-attention kernel — TP=8 in-place refactor (Phase 9 Wave 2).

Each rank holds an attention shard:

  - q_proj output: NUM_HEADS_FULL_LOCAL * HEAD_DIM = 8 * 128 = 1024
  - k_proj/v_proj output: KV_HEADS_LOCAL * HEAD_DIM = 1 * 128 = 128
  - o_proj input: NUM_HEADS_FULL_LOCAL * HEAD_DIM = 1024
  - w_g output:   NUM_HEADS_FULL_LOCAL          = 8
  - q_norm / k_norm gamma [HEAD_DIM=128] — REPLICATED on every rank

Compile-time constants baked in (LOCAL means per-rank-after-TP-slicing):

  - NUM_HEADS  = NUM_HEADS_FULL_LOCAL  (8)
  - HIDDEN_Q   = HIDDEN_Q_FULL_LOCAL   (1024)
  - KV_HIDDEN_DIM = KV_HIDDEN_LOCAL    (128)
  - NUM_KV_HEADS_DIM = KV_HEADS_LOCAL  (1)
  - Q_PER_KV   = Q_PER_KV_FULL         (8 ; invariant under TP)
  - Q_HEAD_BATCH = Q_HEAD_BATCH_FULL   (8)
  - Q_HEAD_PAD = Q_HEAD_PAD_FULL       (16)
  - ROTARY_HALF = ROTARY_HALF_FULL     (32 ; partial_rotary_factor = 0.5)
  - ROTARY_DIM  = 2 * ROTARY_HALF      (64)
  - ROTARY_PASS = HEAD_DIM - ROTARY_DIM (64 ; pass-through lanes)
  - Q_GROUPS  = Q_PER_KV // Q_HEAD_BATCH                  (1)
  - TOTAL_Q_GROUPS = NUM_KV_HEADS_DIM * Q_GROUPS          (1)

TP collective epilogue
----------------------
After the local o_proj (column-sliced) each rank holds a *partial* hidden
``[BATCH, HIDDEN]`` BF16 sum. ``tp_all_reduce`` sums these across the
TP group so every rank ends up with the fully-reduced o_proj output; the
residual add (``+ current_hidden``) happens afterwards (the residual is
replicated across ranks, so adding it post-all-reduce keeps the math
correct).

The caller (Wave-3 ``decode_layer.py`` / ``decode_fwd.py``) must provide
a per-call-site scratch ``tmp_window`` and ``signal_window`` pair, with
documented shapes:

  - ``tmp_window``    : ``pld.DistributedTensor`` view of a
                         ``BATCH * (HIDDEN // TP_WORLD_SIZE) * 2 bytes``
                         ``alloc_window_buffer`` slot (BF16).
  - ``signal_window`` : ``pld.DistributedTensor[[SIGNAL_WINDOW_ROWS, 1],
                        pl.INT32]``, zero-initialised. The standalone program
                        binds ``SIGNAL_WINDOW_ROWS=TP_WORLD_SIZE`` for an
                        independent compact backing. Canonical whole-net
                        inlining binds it to its 512B stacked slot.
                        Each call site allocates a fresh
                        signal-window slot because the ring all-reduce
                        increments the cells across its ``2 * (N - 1)``
                        steps; reusing a slot would corrupt the wait
                        thresholds in subsequent collectives.

Per-layer ``rope_theta`` / ``partial_rotary_factor`` selection and the
host-side rope-table build are unchanged from the single-card draft.
Yarn scaling is always on for full-attention layers (``yarn_only_types =
["full_attention"]``).
"""

# pyright: reportUndefinedVariable=false

from __future__ import annotations

import pypto.language as pl
import pypto.language.distributed as pld

from ._ops import (
    build_llama3_yarn_rope_tables,
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
    FULL_ATTN_QK_BLOCKS_PER_TASK,
    FULL_ATTN_QK_UNIFORM_O1,
    FULL_ATTN_SOFTMAX_BLOCKS_PER_TASK,
    FULL_ATTN_SOFTMAX_UNIFORM_O1,
    FULL_ATTN_ONLINE_SOFTMAX_BLOCKS_PER_TASK,
    FULL_ATTN_ONLINE_SOFTMAX_PARTIALS_PER_REDUCE_TASK,
    FULL_ATTN_ONLINE_SOFTMAX_UNIFORM_O1,
    FULL_ATTN_ONLINE_SOFTMAX_REDUCE_UNIFORM_O1,
    HEAD_DIM,
    HEAD_DIM_INV,
    HIDDEN,
    HIDDEN_INV,
    HIDDEN_Q_FULL_LOCAL,
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
    MAX_BLOCKS_PER_SEQ,
    MAX_SEQ_DEFAULT,
    NUM_HEADS_FULL_LOCAL,
    NUM_HEADS_FULL_LOCAL_PAD,
    FULL_ATTN_OUT_PROJ_FUSE_CAST,
    FULL_ATTN_OUT_PROJ_MATMUL_N_CHUNK,
    FULL_ATTN_OUT_PROJ_MATMUL_TILES_PER_TASK,
    FULL_ATTN_OUT_PROJ_VEC_N_CHUNK,
    Q_HEAD_BATCH_FULL,
    Q_HEAD_PAD_FULL,
    Q_OUT_CHUNK,
    Q_PER_KV_FULL,
    ROPE_SCALING,
    ROPE_SEQ_DYN,
    ROTARY_HALF_FULL,
    TP_WORLD_SIZE,
    USER_BATCH_DYN,
    is_full_attention,
)

NUM_HEADS = NUM_HEADS_FULL_LOCAL
HIDDEN_Q = HIDDEN_Q_FULL_LOCAL
KV_HIDDEN_DIM = KV_HIDDEN_LOCAL
NUM_KV_HEADS_DIM = KV_HEADS_LOCAL
Q_PER_KV = Q_PER_KV_FULL
Q_HEAD_BATCH = Q_HEAD_BATCH_FULL
Q_HEAD_PAD = Q_HEAD_PAD_FULL
ROTARY_HALF = ROTARY_HALF_FULL
ROTARY_DIM = ROTARY_HALF * 2
ROTARY_PASS = HEAD_DIM - ROTARY_DIM
Q_GROUPS = Q_PER_KV // Q_HEAD_BATCH                # 1
TOTAL_Q_GROUPS = NUM_KV_HEADS_DIM * Q_GROUPS       # 1
# This module's standalone wrapper uses a compact, independent signal
# allocation.  Canonical whole-net inlining overrides the same symbolic
# constant in its target module with COMM_SIGNAL_STRIDE_I32=128.
SIGNAL_WINDOW_ROWS = TP_WORLD_SIZE

# Local override for KV projection's output chunk: KV_HIDDEN_LOCAL (128) is
# below the global KV_OUT_CHUNK=256 default, so we pick the whole local KV
# hidden in a single chunk.
KV_OUT_CHUNK_LOCAL = KV_HIDDEN_LOCAL

# Per-layer rows for the o_proj weight: model-bound (= n_full_attn × HIDDEN_Q_FULL_LOCAL).
# Derivation: there are 12 full-attention layers (config.LAYER_TYPES) and each
# contributes HIDDEN_Q_FULL_LOCAL = 1024 rows of wo, so 12 * 1024 = 12288.
# Kept static (not pl.dynamic) for the same reasons documented in
# config.py's static-dim block.
LAYER_QHIDDEN_ROWS_DYN = 12288

assert Q_HEAD_PAD % 4 == 0 and Q_HEAD_PAD // 2 >= Q_HEAD_BATCH
assert BATCH % 2 == 0, (
    "fa_fused pipelines pairs of batches under TP, so BATCH must be even"
)
assert HIDDEN_Q % K_CHUNK == 0
assert HIDDEN % FULL_ATTN_OUT_PROJ_MATMUL_N_CHUNK == 0
assert HIDDEN % FULL_ATTN_OUT_PROJ_VEC_N_CHUNK == 0
assert 0 < FULL_ATTN_OUT_PROJ_MATMUL_TILES_PER_TASK <= (
    HIDDEN // FULL_ATTN_OUT_PROJ_MATMUL_N_CHUNK
)
assert HIDDEN % TP_WORLD_SIZE == 0
assert KV_HIDDEN_DIM == KV_OUT_CHUNK_LOCAL
assert Q_HEAD_PAD_FULL >= Q_HEAD_BATCH_FULL + 2


# =============================================================================
# Attention body — local compute through gated attn_out, partial o_proj,
# TP all-reduce, then residual add.
# =============================================================================
@pl.jit.inline
def attention_full(
    current_hidden: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
    input_rms_weight: pl.Tensor[[LAYER_DYN, HIDDEN], pl.FP32],
    wq: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, HIDDEN_Q_FULL_LOCAL], pl.BF16],
    wk: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, KV_HIDDEN_LOCAL], pl.BF16],
    wv: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, KV_HIDDEN_LOCAL], pl.BF16],
    q_norm_weight: pl.Tensor[[LAYER_DYN, HEAD_DIM], pl.FP32],
    k_norm_weight: pl.Tensor[[LAYER_DYN, HEAD_DIM], pl.FP32],
    seq_lens: pl.Tensor[[USER_BATCH_DYN], pl.INT32],
    block_table: pl.Tensor[[BLOCK_TABLE_FLAT_DYN], pl.INT32],
    slot_mapping: pl.Tensor[[USER_BATCH_DYN], pl.INT32],
    rope_cos: pl.Tensor[[ROPE_SEQ_DYN, ROTARY_HALF_FULL * 2], pl.FP32],
    rope_sin: pl.Tensor[[ROPE_SEQ_DYN, ROTARY_HALF_FULL * 2], pl.FP32],
    k_cache: pl.Tensor[[KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16],
    v_cache: pl.Tensor[[KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16],
    wo: pl.Tensor[[LAYER_QHIDDEN_ROWS_DYN, HIDDEN], pl.BF16],
    w_g: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, NUM_HEADS_FULL_LOCAL_PAD], pl.BF16],
    gate_r: pl.Tensor[[NUM_HEADS_FULL_LOCAL_PAD, HIDDEN_Q_FULL_LOCAL], pl.BF16],
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
    """Step3p5 full-attention layer through TP-reduced o_proj + residual.

    ``my_rank``, ``tmp_window`` and ``signal_window`` are passed in by the
    Wave-3 ``chip_orch`` wrapper (see module docstring for the shape
    contract). Inside this body they only ever flow through the
    :func:`tp_all_reduce` call below; the rest of the kernel is per-rank
    local compute on the rank's head slice.
    """

    decode_scope1_hidden_blocks = HIDDEN // INPUT_PROJ_K_CHUNK
    kv_proj_hidden_blocks = HIDDEN // KV_PROJ_K_CHUNK_LOCAL
    qhidden_blocks = HIDDEN_Q_FULL_LOCAL // K_CHUNK
    decode_attn_scale = ATTN_SCALE
    num_layers_actual = pl.tensor.dim(input_rms_weight, 0)
    decode_layer_cache_rows = pl.tensor.dim(k_cache, 0) // num_layers_actual
    user_batch = pl.tensor.dim(seq_lens, 0)
    bt_stride = pl.tensor.dim(block_table, 0) // user_batch
    # ``BATCH`` is only the static storage/formal capacity.  All per-request
    # row work is bounded by the runtime scalar supplied by the canonical
    # whole-net caller.  Do not use a clamped/padding row as a substitute for
    # an inactive request: that can alias its RoPE/KV write into another slot.
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
    layer_qhidden_base = attn_layer_idx * HIDDEN_Q_FULL_LOCAL
    layer_cache_base = norm_layer_idx * decode_layer_cache_rows

    q_proj = pl.create_tensor([BATCH, HIDDEN_Q_FULL_LOCAL], dtype=pl.FP32)
    k_proj = pl.create_tensor([BATCH, KV_HIDDEN_LOCAL], dtype=pl.FP32)
    v_proj = pl.create_tensor([BATCH, KV_HIDDEN_LOCAL], dtype=pl.FP32)
    q_proj_norm = pl.create_tensor([BATCH, HIDDEN_Q_FULL_LOCAL], dtype=pl.FP32)
    k_proj_norm = pl.create_tensor([BATCH, KV_HIDDEN_LOCAL], dtype=pl.FP32)
    normed_all = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
    gate_score_t = pl.create_tensor([BATCH, NUM_HEADS_FULL_LOCAL_PAD], dtype=pl.BF16)
    gate_exp = pl.create_tensor([BATCH, HIDDEN_Q_FULL_LOCAL], dtype=pl.BF16)
    HEAD_GATE_K_SPLITS = 8
    HEAD_GATE_K_PER_SPLIT = HIDDEN // HEAD_GATE_K_SPLITS
    FULL_Q_OUT_CHUNK = Q_OUT_CHUNK // 2
    gate_logits_partial = pl.create_tensor(
        [BATCH, HEAD_GATE_K_SPLITS * NUM_HEADS_FULL_LOCAL_PAD], dtype=pl.FP32,
    )

    # ----- Scope 1.a — zero-centred input RMSNorm. -----
    # input_rms_weight is replicated across TP ranks (HIDDEN dim is not
    # sliced). Every rank computes the same normed_all tile; this work is
    # duplicated but cheap relative to the projections.
    #
    # Phase A (2026-06-11): mirror qwen3/32b's CORE_GROUP + pl.pipeline form.
    # The previous `pl.spmd(BATCH//BATCH_TILE=1)` with a single worker was
    # the suspected source of the AICore 507018 VEC UB alignment crash at
    # the first MIX SQE task slot (`aicore_kernel_0_mix_aic`). qwen3/32b
    # uses CORE_GROUP + pipeline(stage=4) for the same shape and runs clean.
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="full_rmsnorm_zc"):
        partial_sq = pl.full([1, BATCH], dtype=pl.FP32, value=0.0)
        for kb in pl.pipeline(decode_scope1_hidden_blocks, stage=4):
            sq_k0 = kb * INPUT_PROJ_K_CHUNK
            sq_chunk = pl.cast(
                pl.slice(current_hidden, [BATCH, INPUT_PROJ_K_CHUNK], [0, sq_k0]),
                target_type=pl.FP32,
            )
            partial_sq = pl.add(
                partial_sq,
                pl.reshape(pl.row_sum(pl.mul(sq_chunk, sq_chunk)), [1, BATCH]),
            )
        variance = pl.reshape(
            pl.add(pl.mul(partial_sq, HIDDEN_INV), EPS), [BATCH, 1],
        )
        inv_rms = pl.recip(pl.sqrt(variance))
        for kb in pl.pipeline(decode_scope1_hidden_blocks, stage=4):
            norm_k0 = kb * INPUT_PROJ_K_CHUNK
            norm_chunk = pl.cast(
                pl.slice(current_hidden, [BATCH, INPUT_PROJ_K_CHUNK], [0, norm_k0]),
                target_type=pl.FP32,
            )
            gamma = pl.slice(input_rms_weight, [1, INPUT_PROJ_K_CHUNK], [norm_layer_idx, norm_k0])
            scaled = pl.row_expand_mul(norm_chunk, inv_rms)
            normed = pl.col_expand_mul(scaled, pl.add(gamma, 1.0))
            normed_all = pl.assemble(
                normed_all, pl.cast(normed, target_type=pl.BF16), [0, norm_k0],
            )

    # ----- Scope 1.f — on-device head-gate (RESTORED, path (a)). -----
    # gate = sigmoid(normed_all @ w_g) per head, then expanded across HEAD_DIM
    # via the block-diag constant R (= the ``gate_r`` input, layer-independent):
    # gate_exp[b, h*HEAD_DIM + d] = gate_score[b, h]. Matches vLLM
    # modeling_step3p5 L489 (g_proj on input_layernorm-normed hidden) + L527-531
    # (attn_out.view(.,num_heads,head_dim) * gate_states.unsqueeze(-1).sigmoid()
    # before o_proj). The N=16 matmul_acc codegen bug that once forced this
    # worker-side is fixed on the current stack (verified by
    # tests/step3p5/_probe_head_gate_full + _probe_matmul_acc_n16). Doing it
    # on-device lets the monolithic whole-net self-compute per-layer gate_r
    # (R is the same for every layer, so ``gate_r`` is fed once).
    #
    # w_g is per-attn-layer stacked (row base = layer_hidden_base); gate_r holds
    # R [NUM_HEADS_FULL_LOCAL_PAD, HIDDEN_Q_FULL_LOCAL]. Two scopes keep the UB
    # working set bounded (K-loop tiles free before the N-chunked expand).
    for hg_part in pl.spmd(
        HEAD_GATE_K_SPLITS,
        name_hint="full_head_gate_logits_mm",
        allow_early_resolve=True,
    ):
        hg_k0 = hg_part * HEAD_GATE_K_PER_SPLIT
        hg_logits = pl.matmul(
            pl.slice(normed_all, [BATCH, INPUT_PROJ_K_CHUNK], [0, hg_k0]),
            pl.slice(w_g, [INPUT_PROJ_K_CHUNK, NUM_HEADS_FULL_LOCAL_PAD],
                     [layer_hidden_base + hg_k0, 0]),
            out_dtype=pl.FP32,
        )
        for kb in pl.range(1, HEAD_GATE_K_PER_SPLIT // INPUT_PROJ_K_CHUNK):
            k0 = hg_k0 + kb * INPUT_PROJ_K_CHUNK
            hg_logits = pl.matmul_acc(
                hg_logits,
                pl.slice(normed_all, [BATCH, INPUT_PROJ_K_CHUNK], [0, k0]),
                pl.slice(w_g, [INPUT_PROJ_K_CHUNK, NUM_HEADS_FULL_LOCAL_PAD],
                         [layer_hidden_base + k0, 0]),
            )
        gate_logits_partial = pl.assemble(
            gate_logits_partial,
            hg_logits,
            [0, hg_part * NUM_HEADS_FULL_LOCAL_PAD],
        )
    with pl.at(
        level=pl.Level.CORE_GROUP,
        name_hint="full_head_gate_sigmoid_expand",
        allow_early_resolve=True,
    ):
        hg_logits = pl.slice(
            gate_logits_partial,
            [BATCH, NUM_HEADS_FULL_LOCAL_PAD],
            [0, 0],
        )
        for hg_part in pl.range(1, HEAD_GATE_K_SPLITS):
            hg_logits = pl.add(
                hg_logits,
                pl.slice(
                    gate_logits_partial,
                    [BATCH, NUM_HEADS_FULL_LOCAL_PAD],
                    [0, hg_part * NUM_HEADS_FULL_LOCAL_PAD],
                ),
            )
        hg_score = pl.recip(pl.add(pl.exp(pl.neg(hg_logits)), 1.0))
        gate_score_t[:, :] = pl.cast(hg_score, target_type=pl.BF16)
    full_head_gate_chunks = HIDDEN_Q_FULL_LOCAL // K_CHUNK
    for hg_task in pl.spmd(
        (BATCH // BATCH_TILE) * full_head_gate_chunks,
        name_hint="full_head_gate_expand",
        allow_early_resolve=True,
    ):
        hg_b_idx = hg_task // full_head_gate_chunks
        hg_n_idx = hg_task % full_head_gate_chunks
        hg_b0 = hg_b_idx * BATCH_TILE
        hg_n0 = hg_n_idx * K_CHUNK
        hg_r = pl.slice(
            gate_r, [NUM_HEADS_FULL_LOCAL_PAD, K_CHUNK], [0, hg_n0],
        )
        hg_ge = pl.matmul(
            pl.slice(
                gate_score_t,
                [BATCH_TILE, NUM_HEADS_FULL_LOCAL_PAD],
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
    # wq is row-sliced (output dim → HIDDEN_Q_FULL_LOCAL per rank), so the
    # SPMD bound shrinks accordingly.
    for q_spmd_idx in pl.spmd(
        (BATCH // BATCH_TILE) * (HIDDEN_Q_FULL_LOCAL // FULL_Q_OUT_CHUNK),
        name_hint="full_q_proj",
        allow_early_resolve=True,
    ):
        q_b_idx = q_spmd_idx // (HIDDEN_Q_FULL_LOCAL // FULL_Q_OUT_CHUNK)
        q_ob = q_spmd_idx % (HIDDEN_Q_FULL_LOCAL // FULL_Q_OUT_CHUNK)
        q_b0 = q_b_idx * BATCH_TILE
        q_o0 = q_ob * FULL_Q_OUT_CHUNK
        q_tile_a_0 = pl.slice(normed_all, [BATCH_TILE, INPUT_PROJ_K_CHUNK], [q_b0, 0])
        q_tile_b_0 = pl.slice(
            wq, [INPUT_PROJ_K_CHUNK, FULL_Q_OUT_CHUNK], [layer_hidden_base, q_o0],
        )
        q_acc = pl.matmul(q_tile_a_0, q_tile_b_0, out_dtype=pl.FP32)
        for kb in pl.range(1, decode_scope1_hidden_blocks):
            q_k0 = kb * INPUT_PROJ_K_CHUNK
            q_tile_a = pl.slice(normed_all, [BATCH_TILE, INPUT_PROJ_K_CHUNK], [q_b0, q_k0])
            q_tile_b = pl.slice(
                wq,
                [INPUT_PROJ_K_CHUNK, FULL_Q_OUT_CHUNK],
                [layer_hidden_base + q_k0, q_o0],
            )
            q_acc = pl.matmul_acc(q_acc, q_tile_a, q_tile_b)
        q_proj = pl.assemble(q_proj, q_acc, [q_b0, q_o0])

    # ----- Scope 1.c/1.d — K/V projections. -----
    # wk is row-sliced (output dim → KV_HEADS_LOCAL * HEAD_DIM = 128 per rank).
    # KV_HIDDEN_LOCAL is 128 (single local KV head), so we drop a single
    # output chunk of width KV_HIDDEN_LOCAL = 128.
    for kv_spmd_idx in pl.spmd(
        2 * (BATCH // BATCH_TILE),
        name_hint="full_kv_proj",
        allow_early_resolve=True,
    ):
        kv_kind = kv_spmd_idx % 2
        kv_b0 = (kv_spmd_idx // 2) * BATCH_TILE
        if kv_kind == 0:
            k_b0 = kv_b0
            k_o0 = 0
            k_tile_a_0 = pl.slice(normed_all, [BATCH_TILE, KV_PROJ_K_CHUNK_LOCAL], [k_b0, 0])
            k_tile_b = pl.slice(
                wk, [KV_PROJ_K_CHUNK_LOCAL, KV_HIDDEN_LOCAL], [layer_hidden_base, k_o0],
            )
            k_acc = pl.matmul(k_tile_a_0, k_tile_b, out_dtype=pl.FP32)
            for kb in pl.range(1, kv_proj_hidden_blocks):
                k_k0 = kb * KV_PROJ_K_CHUNK_LOCAL
                k_acc = pl.matmul_acc(
                    k_acc, pl.slice(normed_all, [BATCH_TILE, KV_PROJ_K_CHUNK_LOCAL], [k_b0, k_k0]),
                    pl.slice(wk, [KV_PROJ_K_CHUNK_LOCAL, KV_HIDDEN_LOCAL], [layer_hidden_base + k_k0, k_o0]),
                )
            k_proj = pl.assemble(k_proj, k_acc, [k_b0, k_o0])
        else:
            v_b0 = kv_b0
            v_o0 = 0
            v_tile_a_0 = pl.slice(normed_all, [BATCH_TILE, KV_PROJ_K_CHUNK_LOCAL], [v_b0, 0])
            v_tile_b = pl.slice(
                wv, [KV_PROJ_K_CHUNK_LOCAL, KV_HIDDEN_LOCAL], [layer_hidden_base, v_o0],
            )
            v_acc = pl.matmul(v_tile_a_0, v_tile_b, out_dtype=pl.FP32)
            for kb in pl.range(1, kv_proj_hidden_blocks):
                v_k0 = kb * KV_PROJ_K_CHUNK_LOCAL
                v_acc = pl.matmul_acc(
                    v_acc, pl.slice(normed_all, [BATCH_TILE, KV_PROJ_K_CHUNK_LOCAL], [v_b0, v_k0]),
                    pl.slice(wv, [KV_PROJ_K_CHUNK_LOCAL, KV_HIDDEN_LOCAL], [layer_hidden_base + v_k0, v_o0]),
                )
            v_proj = pl.assemble(v_proj, v_acc, [v_b0, v_o0])

    # ----- Scope 1.e — per-head zero-centred q_norm / k_norm. -----
    # q_norm / k_norm gamma [HEAD_DIM] are REPLICATED across TP ranks, so
    # this block runs unchanged on each rank — only the per-head loop
    # bounds shrink (KV_HEADS_LOCAL = 1 per rank).
    #
    # Q-heads are processed one-at-a-time inside each spmd block to keep
    # the Vec-memory footprint under the 192 KB platform limit. Packing
    # all Q_HEAD_BATCH_FULL=8 heads together blows past 222 KB (six FP32
    # intermediates × [128 × 128] BF16 → ~217 KB live).
    for qkn_spmd_idx in pl.spmd(
        (BATCH // BATCH_TILE) * KV_HEADS_LOCAL, name_hint="full_qk_norm_zc",
    ):
        qkn_b_idx = qkn_spmd_idx // KV_HEADS_LOCAL
        qkn_h = qkn_spmd_idx % KV_HEADS_LOCAL
        qkn_b0 = qkn_b_idx * BATCH_TILE

        qkn_q0_base = qkn_h * Q_PER_KV_FULL * HEAD_DIM
        q_gamma = pl.slice(q_norm_weight, [1, HEAD_DIM], [norm_layer_idx, 0])
        for qh in pl.range(Q_HEAD_BATCH_FULL):
            qh_q0 = qkn_q0_base + qh * HEAD_DIM
            q_chunk = pl.slice(q_proj, [BATCH_TILE, HEAD_DIM], [qkn_b0, qh_q0])
            q_sq = pl.row_sum(pl.mul(q_chunk, q_chunk))
            q_inv = pl.rsqrt(pl.add(pl.mul(q_sq, HEAD_DIM_INV), EPS))
            q_scaled = pl.row_expand_mul(q_chunk, q_inv)
            q_normed = pl.col_expand_mul(q_scaled, pl.add(q_gamma, 1.0))
            q_proj_norm = pl.assemble(q_proj_norm, q_normed, [qkn_b0, qh_q0])

        qkn_k0 = qkn_h * HEAD_DIM
        k_chunk = pl.slice(k_proj, [BATCH_TILE, HEAD_DIM], [qkn_b0, qkn_k0])
        k_gamma = pl.slice(k_norm_weight, [1, HEAD_DIM], [norm_layer_idx, 0])
        k_sq = pl.row_sum(pl.mul(k_chunk, k_chunk))
        k_inv = pl.rsqrt(pl.add(pl.mul(k_sq, HEAD_DIM_INV), EPS))
        k_scaled = pl.row_expand_mul(k_chunk, k_inv)
        k_normed = pl.col_expand_mul(k_scaled, pl.add(k_gamma, 1.0))
        k_proj_norm = pl.assemble(k_proj_norm, k_normed, [qkn_b0, qkn_k0])

    # ----- Scope 1.f — head-wise gate matmul: MOVED UP to run immediately after
    # Scope 1.a (normed_all build), before q/k/v proj + q_norm/k_norm. Empirically
    # the gate matmul here (last consumer of normed_all) read a stale/reused
    # normed_all buffer — kernel gate_logits came out ~20x too small (pre-norm
    # magnitude, varying per-head), so sigmoid never saturated and the gate never
    # suppressed (rank-5 hot-head ~40x blow-up). Running it first, while normed_all
    # is freshly live, fixes the value. See the block right after Scope 1.a.

    # ----- Scope 2 — partial RoPE + paged KV cache write + fa_fused. -----
    # k_cache / v_cache hold KV_HEADS_LOCAL = 1 KV head's history per rank,
    # so the loop over local KV heads collapses to a single iteration (the
    # math below mirrors the single-card draft but the slot/cache strides
    # use KV_HEADS_LOCAL).
    attn_out = pl.create_tensor([BATCH, HIDDEN_Q_FULL_LOCAL], dtype=pl.BF16)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="full_attn_out_zero"):
        attn_out[:, :] = pl.full(
            [BATCH, HIDDEN_Q_FULL_LOCAL], dtype=pl.BF16, value=0.0,
        )
    all_q_padded = pl.create_tensor(
        [BATCH * KV_HEADS_LOCAL * (Q_PER_KV_FULL // Q_HEAD_BATCH_FULL) * Q_HEAD_PAD_FULL, HEAD_DIM], dtype=pl.BF16,
    )

    # Separate active-row tasks write Q and update the resident K/V cache.
    with pl.spmd(
        active_tokens,
        name_hint="full_rope_q",
        allow_early_resolve=True,
    ) as full_rope_q_tid:
        b = pl.tile.get_block_idx()
        if b < active_tokens:
            ctx_len = pl.tensor.read(seq_lens, [b])
            pos = ctx_len - 1
            cos_lo = pl.slice(rope_cos, [1, ROTARY_HALF_FULL], [pos, 0])
            cos_hi = pl.slice(rope_cos, [1, ROTARY_HALF_FULL], [pos, ROTARY_HALF_FULL])
            sin_lo = pl.slice(rope_sin, [1, ROTARY_HALF_FULL], [pos, 0])
            sin_hi = pl.slice(rope_sin, [1, ROTARY_HALF_FULL], [pos, ROTARY_HALF_FULL])

            for ki in pl.range(KV_HEADS_LOCAL):
                # Per-head RoPE using CONTIGUOUS [1, ROTARY_HALF_FULL] slices of
                # q_proj_norm, mirroring the K path above. This replaces the
                # earlier reshape(q_proj_norm -> [Q_HEAD_BATCH_FULL, HEAD_DIM])
                # + [Q_HEAD_BATCH_FULL, ROTARY_HALF_FULL] col-offset slice, which
                # miscompiled the rot_q_hi (cols ROTARY_HALF_FULL..ROTARY_DIM)
                # write into all_q_padded -> wrong q.k scores for ctx>1 (invisible
                # at ctx=1 since output=V). Verified via _stage_scope12_qk.py:
                # per-rank crossrow scores bad_ratio 0.25/0.90 -> ~0.
                q_base = ki * Q_PER_KV_FULL
                pad_row_base = b * KV_HEADS_LOCAL * (Q_PER_KV_FULL // Q_HEAD_BATCH_FULL) * Q_HEAD_PAD_FULL + ki * Q_HEAD_PAD_FULL
                for qh in pl.range(Q_HEAD_BATCH_FULL):
                    qh_col = (q_base + qh) * HEAD_DIM
                    q_lo_h = pl.slice(q_proj_norm, [1, ROTARY_HALF_FULL], [b, qh_col])
                    q_hi_h = pl.slice(
                        q_proj_norm, [1, ROTARY_HALF_FULL], [b, qh_col + ROTARY_HALF_FULL],
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
                        all_q_padded,
                        pl.cast(pl.slice(q_proj_norm, [1, HEAD_DIM], [b, qh_col]),
                                target_type=pl.BF16),
                        [q_row, 0],
                    )
                    all_q_padded = pl.assemble(
                        all_q_padded, pl.cast(rot_q_lo_h, target_type=pl.BF16), [q_row, 0],
                    )
                    all_q_padded = pl.assemble(
                        all_q_padded, pl.cast(rot_q_hi_h, target_type=pl.BF16),
                        [q_row, ROTARY_HALF_FULL],
                    )
                all_q_padded = pl.assemble(
                    all_q_padded,
                    pl.cast(
                        pl.full([Q_HEAD_PAD_FULL - Q_HEAD_BATCH_FULL, HEAD_DIM],
                                dtype=pl.FP32, value=0.0),
                        target_type=pl.BF16,
                    ),
                    [pad_row_base + Q_HEAD_BATCH_FULL, 0],
                )

    with pl.spmd(
        active_tokens,
        name_hint="full_rope_kv_cache",
        allow_early_resolve=True,
    ) as full_rope_kv_tid:
        b = pl.tile.get_block_idx()
        if b < active_tokens:
            ctx_len = pl.tensor.read(seq_lens, [b])
            pos = ctx_len - 1
            slot = pl.tensor.read(slot_mapping, [b])
            slot_block = slot // BLOCK_SIZE
            slot_offset = slot - slot_block * BLOCK_SIZE
            cos_lo = pl.slice(rope_cos, [1, ROTARY_HALF_FULL], [pos, 0])
            cos_hi = pl.slice(rope_cos, [1, ROTARY_HALF_FULL], [pos, ROTARY_HALF_FULL])
            sin_lo = pl.slice(rope_sin, [1, ROTARY_HALF_FULL], [pos, 0])
            sin_hi = pl.slice(rope_sin, [1, ROTARY_HALF_FULL], [pos, ROTARY_HALF_FULL])

            for ki in pl.range(KV_HEADS_LOCAL):
                kv_col = ki * HEAD_DIM
                cache_row = (
                    layer_cache_base
                    + (slot_block * KV_HEADS_LOCAL + ki) * BLOCK_SIZE
                    + slot_offset
                )
                k_lo = pl.slice(k_proj_norm, [1, ROTARY_HALF_FULL], [b, kv_col])
                k_hi = pl.slice(
                    k_proj_norm, [1, ROTARY_HALF_FULL], [b, kv_col + ROTARY_HALF_FULL],
                )
                rot_k_lo = pl.sub(
                    pl.col_expand_mul(k_lo, cos_lo),
                    pl.col_expand_mul(k_hi, sin_lo),
                )
                rot_k_hi = pl.add(
                    pl.col_expand_mul(k_hi, cos_hi),
                    pl.col_expand_mul(k_lo, sin_hi),
                )
                # Phase A (2026-06-11): use qwen3/32b's full-row-cast-then-
                # overwrite idiom instead of the (compile-required, runtime-
                # broken) `pl.add(k_pass, 0.0)` workaround. Cast the entire
                # [1, HEAD_DIM] k_proj_norm row to BF16 once (which is the
                # exact pattern qwen3/32b's v_cache write uses and which
                # AICore lowers cleanly), then overwrite cols 0..2*HALF with
                # the RoPE'd halves. The pass-through tail (cols 2*HALF..)
                # is left as the initial full-row cast.
                k_cache = pl.assemble(
                    k_cache,
                    pl.cast(
                        pl.slice(k_proj_norm, [1, HEAD_DIM], [b, kv_col]),
                        target_type=pl.BF16,
                    ),
                    [cache_row, 0],
                )
                k_cache = pl.assemble(
                    k_cache, pl.cast(rot_k_lo, target_type=pl.BF16), [cache_row, 0],
                )
                k_cache = pl.assemble(
                    k_cache, pl.cast(rot_k_hi, target_type=pl.BF16),
                    [cache_row, ROTARY_HALF_FULL],
                )
                v_cache = pl.assemble(
                    v_cache,
                    pl.cast(pl.slice(v_proj, [1, HEAD_DIM], [b, kv_col]),
                            target_type=pl.BF16),
                    [cache_row, 0],
                )


    # ----- fa_fused — Phase A (2026-06-11): qwen3/32b-style 4-spmd split. -----
    # The previous fused mixed AIC+AIV single root tripped 507018 / VEC UB
    # not-aligned at this shape (NUM_HEADS_FULL_LOCAL=8, KV_HEADS_LOCAL=1,
    # Q_PER_KV_FULL=Q_HEAD_BATCH_FULL=8, Q_HEAD_PAD_FULL=16, HEAD_DIM=128).
    # We mirror qwen3/32b's split QK/softmax front-end, then fuse SV with the
    # first online-softmax reduction level. GM scratch carries raw scores and
    # softmax exp + mi/li between stages. ``pl.slice(..., valid_shape=...)``
    # replaces set_validshape +
    # fillpad so the VEC lowering goes through a different (proven-safe)
    # path. mi/li 以合法宽行落盘，读取后再 reshape 为 reduction column。
    # See docs/step3p5/phases/15-singlerank-npu.md "Phase A route decision".
    MAX_CTX_BLOCKS = MAX_SEQ_DEFAULT // BLOCK_SIZE
    full_qk_active_tasks = pl.cast(0, pl.INDEX)
    full_qk_uniform_tasks_per_row = pl.cast(0, pl.INDEX)
    full_qk_tasks_uniform = pl.cast(
        FULL_ATTN_QK_UNIFORM_O1, pl.INDEX,
    )
    full_softmax_active_tasks = pl.cast(0, pl.INDEX)
    full_softmax_uniform_tasks_per_row = pl.cast(0, pl.INDEX)
    full_softmax_tasks_uniform = pl.cast(
        FULL_ATTN_SOFTMAX_UNIFORM_O1, pl.INDEX,
    )
    full_online_softmax_active_tasks = pl.cast(0, pl.INDEX)
    full_online_uniform_tasks_per_row = pl.cast(0, pl.INDEX)
    full_online_tasks_uniform = pl.cast(
        FULL_ATTN_ONLINE_SOFTMAX_UNIFORM_O1, pl.INDEX,
    )
    # Launch exactly the logical tasks implied by active request rows.  A
    # zero-token request is a graph-level no-op/reject contract: adding a
    # dummy attention task here penalizes every non-empty request and does not
    # make the later collectives valid for num_tokens == 0.
    for fa_count_b in pl.range(active_tokens):
        fa_count_ctx_len = pl.tensor.read(seq_lens, [fa_count_b])
        fa_count_ctx_blocks = (
            fa_count_ctx_len + BLOCK_SIZE - 1
        ) // BLOCK_SIZE
        fa_count_qk_tasks = (
            fa_count_ctx_blocks + FULL_ATTN_QK_BLOCKS_PER_TASK - 1
        ) // FULL_ATTN_QK_BLOCKS_PER_TASK
        fa_count_softmax_tasks = (
            fa_count_ctx_blocks + FULL_ATTN_SOFTMAX_BLOCKS_PER_TASK - 1
        ) // FULL_ATTN_SOFTMAX_BLOCKS_PER_TASK
        fa_count_online_tasks = (
            fa_count_ctx_blocks
            + FULL_ATTN_ONLINE_SOFTMAX_BLOCKS_PER_TASK
            - 1
        ) // FULL_ATTN_ONLINE_SOFTMAX_BLOCKS_PER_TASK
        if fa_count_b == 0:
            full_qk_uniform_tasks_per_row = fa_count_qk_tasks
            full_softmax_uniform_tasks_per_row = fa_count_softmax_tasks
            full_online_uniform_tasks_per_row = fa_count_online_tasks
        else:
            # Each switch initializes its route predicate fail-closed.  A
            # mismatch can only clear the predicate; it can never re-enable a
            # disabled route.  This matters when a later row has exactly one
            # more task than row zero.
            if fa_count_qk_tasks != full_qk_uniform_tasks_per_row:
                full_qk_tasks_uniform = pl.cast(0, pl.INDEX)
            if (
                fa_count_softmax_tasks
                != full_softmax_uniform_tasks_per_row
            ):
                full_softmax_tasks_uniform = pl.cast(0, pl.INDEX)
            if (
                fa_count_online_tasks
                != full_online_uniform_tasks_per_row
            ):
                full_online_tasks_uniform = pl.cast(0, pl.INDEX)
        full_qk_active_tasks = full_qk_active_tasks + fa_count_qk_tasks
        full_softmax_active_tasks = (
            full_softmax_active_tasks + fa_count_softmax_tasks
        )
        full_online_softmax_active_tasks = (
            full_online_softmax_active_tasks + fa_count_online_tasks
        )
    all_raw_scores = pl.create_tensor(
        [BATCH * MAX_CTX_BLOCKS * Q_HEAD_PAD_FULL, BLOCK_SIZE], dtype=pl.FP32,
    )
    all_exp_padded = pl.create_tensor(
        [BATCH * MAX_CTX_BLOCKS * Q_HEAD_PAD_FULL, BLOCK_SIZE], dtype=pl.BF16,
    )
    all_cur_mi = pl.create_tensor(
        [BATCH * MAX_CTX_BLOCKS, Q_HEAD_PAD_FULL], dtype=pl.FP32,
    )
    all_cur_li = pl.create_tensor(
        [BATCH * MAX_CTX_BLOCKS, Q_HEAD_PAD_FULL], dtype=pl.FP32,
    )
    # Each segment owns Q_HEAD_PAD_FULL rows for O. Keep M/L in a dedicated
    # row-major scratch; packing them into O padding rows introduces narrow
    # row/column reshapes that are fragile in the 0162 lowering.
    online_partial = pl.create_tensor(
        [BATCH * MAX_CTX_BLOCKS * Q_HEAD_PAD_FULL, HEAD_DIM],
        dtype=pl.FP32,
    )
    # Pack M and L into one aligned row: [M heads | L heads].  The current
    # orchestration lowering on 0162 can permute two FP32 scratch lineages
    # mutated by the segment-reduce stage, even when their formal shapes differ.
    # A single InOut tensor removes that ambiguous parameter mapping while
    # retaining 32-byte-aligned [1, H] slices for all vector operations.
    online_partial_ml = pl.create_tensor(
        [BATCH * MAX_CTX_BLOCKS, 2 * Q_HEAD_BATCH_FULL], dtype=pl.FP32,
    )
    with pl.spmd(
        full_qk_active_tasks,
        name_hint="full_qk_matmul",
        deps=[full_rope_q_tid, full_rope_kv_tid],
        allow_early_resolve=True,
    ) as full_qk_tid:
        fa_task = pl.tile.get_block_idx()
        if fa_task < full_qk_active_tasks:
            fa_qk_b = pl.cast(0, pl.INDEX)
            fa_qk_task_in_b = fa_task
            if active_tokens != 1:
                if full_qk_tasks_uniform != 0:
                    # Traverse uniform rows task-major rather than row-major:
                    # complete block groups are queued before the shorter tail
                    # groups.  The runtime can then pull balanced work in the
                    # first wave and leave only tails for a partial next wave.
                    fa_qk_task_in_b = fa_task // active_tokens
                    fa_qk_b = (
                        fa_task - fa_qk_task_in_b * active_tokens
                    )
                else:
                    fa_qk_task_base = pl.cast(0, pl.INDEX)
                    for fa_qk_scan_b in pl.range(active_tokens):
                        fa_qk_scan_ctx_len = pl.tensor.read(
                            seq_lens, [fa_qk_scan_b],
                        )
                        fa_qk_scan_ctx_blocks = (
                            fa_qk_scan_ctx_len + BLOCK_SIZE - 1
                        ) // BLOCK_SIZE
                        fa_qk_scan_tasks = (
                            fa_qk_scan_ctx_blocks
                            + FULL_ATTN_QK_BLOCKS_PER_TASK - 1
                        ) // FULL_ATTN_QK_BLOCKS_PER_TASK
                        if fa_task >= fa_qk_task_base:
                            if (
                                fa_task
                                < fa_qk_task_base + fa_qk_scan_tasks
                            ):
                                fa_qk_b = fa_qk_scan_b
                                fa_qk_task_in_b = (
                                    fa_task - fa_qk_task_base
                                )
                        fa_qk_task_base = (
                            fa_qk_task_base + fa_qk_scan_tasks
                        )
            fa_qk_sb0 = (
                fa_qk_task_in_b * FULL_ATTN_QK_BLOCKS_PER_TASK
            )
            fa_qk_ctx_len = pl.tensor.read(seq_lens, [fa_qk_b])
            fa_qk_ctx_blocks = (
                fa_qk_ctx_len + BLOCK_SIZE - 1
            ) // BLOCK_SIZE
            fa_qk_block_table_base = fa_qk_b * bt_stride
            fa_qk_padded_row = fa_qk_b * Q_HEAD_PAD_FULL
            fa_qk_padded = pl.slice(
                all_q_padded,
                [Q_HEAD_PAD_FULL, HEAD_DIM],
                [fa_qk_padded_row, 0],
            )
            for fa_local in pl.range(FULL_ATTN_QK_BLOCKS_PER_TASK):
                fa_qk_sb = fa_qk_sb0 + fa_local
                if fa_qk_sb < fa_qk_ctx_blocks:
                    fa_qk_pbid = pl.cast(
                        pl.tensor.read(
                            block_table,
                            [fa_qk_block_table_base + fa_qk_sb],
                        ),
                        pl.INDEX,
                    )
                    fa_qk_cache_row = (
                        layer_cache_base + fa_qk_pbid * BLOCK_SIZE
                    )
                    fa_qk_k_tile = pl.slice(
                        k_cache,
                        [BLOCK_SIZE, HEAD_DIM],
                        [fa_qk_cache_row, 0],
                    )
                    fa_qk_raw_scores = pl.matmul(
                        fa_qk_padded,
                        fa_qk_k_tile,
                        b_trans=True,
                        out_dtype=pl.FP32,
                    )
                    fa_qk_scratch_row = (
                        fa_qk_b * MAX_CTX_BLOCKS + fa_qk_sb
                    ) * Q_HEAD_PAD_FULL
                    all_raw_scores = pl.assemble(
                        all_raw_scores,
                        fa_qk_raw_scores,
                        [fa_qk_scratch_row, 0],
                    )

    with pl.spmd(
        full_softmax_active_tasks,
        name_hint="full_softmax",
        deps=[full_qk_tid],
        allow_early_resolve=True,
    ) as full_softmax_tid:
        fa_task = pl.tile.get_block_idx()
        if fa_task < full_softmax_active_tasks:
            fa_sm_b = pl.cast(0, pl.INDEX)
            fa_sm_task_in_b = fa_task
            if active_tokens != 1:
                if full_softmax_tasks_uniform != 0:
                    fa_sm_task_in_b = fa_task // active_tokens
                    fa_sm_b = fa_task - fa_sm_task_in_b * active_tokens
                else:
                    fa_sm_task_base = pl.cast(0, pl.INDEX)
                    for fa_sm_scan_b in pl.range(active_tokens):
                        fa_sm_scan_ctx_len = pl.tensor.read(
                            seq_lens, [fa_sm_scan_b],
                        )
                        fa_sm_scan_ctx_blocks = (
                            fa_sm_scan_ctx_len + BLOCK_SIZE - 1
                        ) // BLOCK_SIZE
                        fa_sm_scan_tasks = (
                            fa_sm_scan_ctx_blocks
                            + FULL_ATTN_SOFTMAX_BLOCKS_PER_TASK - 1
                        ) // FULL_ATTN_SOFTMAX_BLOCKS_PER_TASK
                        if fa_task >= fa_sm_task_base:
                            if (
                                fa_task
                                < fa_sm_task_base + fa_sm_scan_tasks
                            ):
                                fa_sm_b = fa_sm_scan_b
                                fa_sm_task_in_b = (
                                    fa_task - fa_sm_task_base
                                )
                        fa_sm_task_base = (
                            fa_sm_task_base + fa_sm_scan_tasks
                        )
            fa_sm_sb0 = (
                fa_sm_task_in_b * FULL_ATTN_SOFTMAX_BLOCKS_PER_TASK
            )
            fa_sm_ctx_len = pl.tensor.read(seq_lens, [fa_sm_b])
            fa_sm_ctx_blocks = (
                fa_sm_ctx_len + BLOCK_SIZE - 1
            ) // BLOCK_SIZE
            for fa_local in pl.range(FULL_ATTN_SOFTMAX_BLOCKS_PER_TASK):
                fa_sm_sb = fa_sm_sb0 + fa_local
                if fa_sm_sb < fa_sm_ctx_blocks:
                    fa_sm_s0 = fa_sm_sb * BLOCK_SIZE
                    fa_sm_valid_len = pl.min(
                        BLOCK_SIZE, fa_sm_ctx_len - fa_sm_s0,
                    )
                    fa_sm_scratch_row = (
                        fa_sm_b * MAX_CTX_BLOCKS + fa_sm_sb
                    ) * Q_HEAD_PAD_FULL
                    fa_sm_scores_valid = pl.slice(
                        all_raw_scores,
                        [Q_HEAD_BATCH_FULL, BLOCK_SIZE],
                        [fa_sm_scratch_row, 0],
                        valid_shape=[
                            Q_HEAD_BATCH_FULL,
                            fa_sm_valid_len,
                        ],
                    )
                    fa_sm_scores_padded = pl.fillpad(
                        fa_sm_scores_valid,
                        pad_value=pl.PadValue.min,
                    )
                    fa_sm_scores = pl.mul(
                        fa_sm_scores_padded, decode_attn_scale,
                    )
                    fa_sm_cur_mi = pl.row_max(fa_sm_scores)
                    fa_sm_exp_scores = pl.exp(
                        pl.row_expand_sub(fa_sm_scores, fa_sm_cur_mi),
                    )
                    fa_sm_exp_scores_bf16 = pl.cast(
                        fa_sm_exp_scores, target_type=pl.BF16,
                    )
                    fa_sm_exp_scores_fp32 = pl.cast(
                        fa_sm_exp_scores_bf16, target_type=pl.FP32,
                    )
                    fa_sm_cur_li = pl.row_sum(fa_sm_exp_scores_fp32)
                    # SV uses the cube-legal padded head tile. Define every
                    # row explicitly instead of letting the padded half read
                    # uninitialized GM.
                    all_exp_padded = pl.assemble(
                        all_exp_padded,
                        fa_sm_exp_scores_bf16,
                        [fa_sm_scratch_row, 0],
                    )
                    fa_sm_exp_zero_heads = pl.full(
                        [
                            Q_HEAD_PAD_FULL - Q_HEAD_BATCH_FULL,
                            BLOCK_SIZE,
                        ],
                        dtype=pl.BF16,
                        value=0.0,
                    )
                    all_exp_padded = pl.assemble(
                        all_exp_padded,
                        fa_sm_exp_zero_heads,
                        [fa_sm_scratch_row + Q_HEAD_BATCH_FULL, 0],
                    )
                    fa_sm_lm_row = (
                        fa_sm_b * MAX_CTX_BLOCKS + fa_sm_sb
                    )
                    all_cur_mi = pl.assemble(
                        all_cur_mi,
                        pl.reshape(
                            fa_sm_cur_mi, [1, Q_HEAD_BATCH_FULL],
                        ),
                        [fa_sm_lm_row, 0],
                    )
                    all_cur_li = pl.assemble(
                        all_cur_li,
                        pl.reshape(
                            fa_sm_cur_li, [1, Q_HEAD_BATCH_FULL],
                        ),
                        [fa_sm_lm_row, 0],
                    )

    # Fuse SV and segment-local online recurrence into one mixed task per
    # segment. Each logical task owns a contiguous KV-block segment, computes
    # SV on the cube lane,
    # and immediately performs the segment-local online recurrence on the vector
    # lane. The SPMD grid is workload-derived; no fixed resident-worker count
    # or independent first-pass recurrence kernel is introduced.
    # Keep one AIV owner per mixed task. UP_DOWN would split O into disjoint
    # head rows but duplicate the unsplit M/L recurrence on both subblocks,
    # creating a cross-subblock GM read-after-write race between KV blocks.
    with pl.spmd(
        full_online_softmax_active_tasks,
        name_hint="full_sv_matmul",
        deps=[full_softmax_tid],
        allow_early_resolve=True,
    ) as full_sv_online_tid:
        fa_task = pl.tile.get_block_idx()
        if fa_task < full_online_softmax_active_tasks:
            fa_sv_b = pl.cast(0, pl.INDEX)
            if active_tokens == 1:
                fa_sv_task_in_b = fa_task
            else:
                if full_online_tasks_uniform != 0:
                    fa_sv_task_in_b = fa_task // active_tokens
                    fa_sv_b = fa_task - fa_sv_task_in_b * active_tokens
                else:
                    # Keep countdown as the heterogeneous fallback; cumulative
                    # task-base lowering previously overwrote row 0.
                    fa_sv_task_in_b = fa_task
                    for fa_sv_scan_b in pl.range(active_tokens):
                        fa_sv_scan_ctx_len = pl.tensor.read(
                            seq_lens, [fa_sv_scan_b],
                        )
                        fa_sv_scan_ctx_blocks = (
                            fa_sv_scan_ctx_len + BLOCK_SIZE - 1
                        ) // BLOCK_SIZE
                        fa_sv_scan_tasks = (
                            fa_sv_scan_ctx_blocks
                            + FULL_ATTN_ONLINE_SOFTMAX_BLOCKS_PER_TASK
                            - 1
                        ) // FULL_ATTN_ONLINE_SOFTMAX_BLOCKS_PER_TASK
                        if fa_sv_scan_b == fa_sv_b:
                            if fa_sv_task_in_b >= fa_sv_scan_tasks:
                                fa_sv_task_in_b = (
                                    fa_sv_task_in_b - fa_sv_scan_tasks
                                )
                                fa_sv_b = fa_sv_b + 1

            fa_sv_ctx_len = pl.tensor.read(seq_lens, [fa_sv_b])
            fa_sv_ctx_blocks = (
                fa_sv_ctx_len + BLOCK_SIZE - 1
            ) // BLOCK_SIZE
            fa_sv_sb0 = (
                fa_sv_task_in_b
                * FULL_ATTN_ONLINE_SOFTMAX_BLOCKS_PER_TASK
            )
            fa_sv_sb1 = pl.min(
                fa_sv_ctx_blocks,
                fa_sv_sb0 + FULL_ATTN_ONLINE_SOFTMAX_BLOCKS_PER_TASK,
            )
            fa_sv_block_table_base = fa_sv_b * bt_stride
            fa_sv_segment_row = (
                fa_sv_b * MAX_CTX_BLOCKS + fa_sv_task_in_b
            )
            fa_sv_segment_row0 = fa_sv_segment_row * Q_HEAD_PAD_FULL

            # Every block, including the first, follows the same mixed-lane
            # path: cube computes SV, C2V publishes it, and AIV either seeds
            # or updates the segment-local online state.  Keeping a single
            # cross-core boundary prevents cube-only stores from leaking into
            # the AIV function and vector-only stores from leaking into AIC.
            for fa_sv_sb in pl.range(fa_sv_sb0, fa_sv_sb1):
                fa_sv_pbid = pl.cast(
                    pl.tensor.read(
                        block_table,
                        [fa_sv_block_table_base + fa_sv_sb],
                    ),
                    pl.INDEX,
                )
                fa_sv_cache_row = (
                    layer_cache_base + fa_sv_pbid * BLOCK_SIZE
                )
                fa_sv_v = pl.slice(
                    v_cache,
                    [BLOCK_SIZE, HEAD_DIM],
                    [fa_sv_cache_row, 0],
                )
                fa_sv_scratch_row = (
                    fa_sv_b * MAX_CTX_BLOCKS + fa_sv_sb
                ) * Q_HEAD_PAD_FULL
                fa_sv_exp = pl.slice(
                    all_exp_padded,
                    [Q_HEAD_PAD_FULL, BLOCK_SIZE],
                    [fa_sv_scratch_row, 0],
                )
                fa_sv_oi = pl.matmul(
                    fa_sv_exp,
                    fa_sv_v,
                    out_dtype=pl.FP32,
                )
                fa_sv_lm_row = fa_sv_b * MAX_CTX_BLOCKS + fa_sv_sb
                fa_sv_mi_real = pl.slice(
                    all_cur_mi,
                    [1, Q_HEAD_BATCH_FULL],
                    [fa_sv_lm_row, 0],
                )
                fa_sv_li_real = pl.slice(
                    all_cur_li,
                    [1, Q_HEAD_BATCH_FULL],
                    [fa_sv_lm_row, 0],
                )
                fa_sv_zero_heads = pl.full(
                    [1, Q_HEAD_PAD_FULL - Q_HEAD_BATCH_FULL],
                    dtype=pl.FP32,
                    value=0.0,
                )
                fa_sv_mi = pl.concat(
                    fa_sv_mi_real, fa_sv_zero_heads,
                )
                fa_sv_li = pl.concat(
                    fa_sv_li_real, fa_sv_zero_heads,
                )
                if fa_sv_sb == fa_sv_sb0:
                    online_partial = pl.assemble(
                        online_partial,
                        fa_sv_oi,
                        [fa_sv_segment_row0, 0],
                    )
                    fa_sv_ml = pl.concat(
                        fa_sv_mi_real, fa_sv_li_real,
                    )
                    online_partial_ml = pl.assemble(
                        online_partial_ml,
                        fa_sv_ml,
                        [fa_sv_segment_row, 0],
                    )
                else:
                    fa_sv_acc_oi = pl.slice(
                        online_partial,
                        [Q_HEAD_PAD_FULL, HEAD_DIM],
                        [fa_sv_segment_row0, 0],
                    )
                    fa_sv_acc_mi_real = pl.slice(
                        online_partial_ml,
                        [1, Q_HEAD_BATCH_FULL],
                        [fa_sv_segment_row, 0],
                    )
                    fa_sv_acc_li_real = pl.slice(
                        online_partial_ml,
                        [1, Q_HEAD_BATCH_FULL],
                        [fa_sv_segment_row, Q_HEAD_BATCH_FULL],
                    )
                    fa_sv_acc_mi = pl.concat(
                        fa_sv_acc_mi_real, fa_sv_zero_heads,
                    )
                    fa_sv_acc_li = pl.concat(
                        fa_sv_acc_li_real, fa_sv_zero_heads,
                    )
                    fa_sv_mi_new = pl.maximum(
                        fa_sv_acc_mi, fa_sv_mi,
                    )
                    fa_sv_alpha_row = pl.exp(
                        pl.sub(fa_sv_acc_mi, fa_sv_mi_new),
                    )
                    fa_sv_beta_row = pl.exp(
                        pl.sub(fa_sv_mi, fa_sv_mi_new),
                    )
                    fa_sv_li_new = pl.add(
                        pl.mul(fa_sv_alpha_row, fa_sv_acc_li),
                        pl.mul(fa_sv_beta_row, fa_sv_li),
                    )
                    fa_sv_alpha = pl.reshape(
                        fa_sv_alpha_row, [Q_HEAD_PAD_FULL, 1],
                    )
                    fa_sv_beta = pl.reshape(
                        fa_sv_beta_row, [Q_HEAD_PAD_FULL, 1],
                    )
                    fa_sv_oi_new = pl.add(
                        pl.row_expand_mul(
                            fa_sv_acc_oi, fa_sv_alpha,
                        ),
                        pl.row_expand_mul(fa_sv_oi, fa_sv_beta),
                    )
                    online_partial = pl.assemble(
                        online_partial,
                        fa_sv_oi_new,
                        [fa_sv_segment_row0, 0],
                    )
                    fa_sv_mi_new_real = pl.slice(
                        fa_sv_mi_new,
                        [1, Q_HEAD_BATCH_FULL],
                        [0, 0],
                    )
                    fa_sv_li_new_real = pl.slice(
                        fa_sv_li_new,
                        [1, Q_HEAD_BATCH_FULL],
                        [0, 0],
                    )
                    fa_sv_ml_new = pl.concat(
                        fa_sv_mi_new_real, fa_sv_li_new_real,
                    )
                    online_partial_ml = pl.assemble(
                        online_partial_ml,
                        fa_sv_ml_new,
                        [fa_sv_segment_row, 0],
                    )


    # Reduce small contiguous groups of segment partials in parallel.
    # A task writes its result back to the first segment of its own group
    # (0, fan-in, 2*fan-in, ...). Those destinations are outside every other
    # group's source range, so concurrent tasks remain read/write disjoint.
    full_online_softmax_reduce_tasks = pl.cast(0, pl.INDEX)
    full_online_reduce_uniform_tasks_per_row = pl.cast(0, pl.INDEX)
    full_online_reduce_tasks_uniform = pl.cast(
        FULL_ATTN_ONLINE_SOFTMAX_REDUCE_UNIFORM_O1, pl.INDEX,
    )
    for fa_reduce_count_b in pl.range(active_tokens):
        fa_reduce_count_ctx_len = pl.tensor.read(
            seq_lens, [fa_reduce_count_b],
        )
        fa_reduce_count_ctx_blocks = (
            fa_reduce_count_ctx_len + BLOCK_SIZE - 1
        ) // BLOCK_SIZE
        fa_reduce_count_segments = (
            fa_reduce_count_ctx_blocks
            + FULL_ATTN_ONLINE_SOFTMAX_BLOCKS_PER_TASK
            - 1
        ) // FULL_ATTN_ONLINE_SOFTMAX_BLOCKS_PER_TASK
        fa_reduce_count_tasks = (
            fa_reduce_count_segments
            + FULL_ATTN_ONLINE_SOFTMAX_PARTIALS_PER_REDUCE_TASK
            - 1
        ) // FULL_ATTN_ONLINE_SOFTMAX_PARTIALS_PER_REDUCE_TASK
        if fa_reduce_count_b == 0:
            full_online_reduce_uniform_tasks_per_row = (
                fa_reduce_count_tasks
            )
        else:
            if (
                fa_reduce_count_tasks
                != full_online_reduce_uniform_tasks_per_row
            ):
                full_online_reduce_tasks_uniform = pl.cast(0, pl.INDEX)
        full_online_softmax_reduce_tasks = (
            full_online_softmax_reduce_tasks + fa_reduce_count_tasks
        )
    with pl.spmd(
        full_online_softmax_reduce_tasks,
        name_hint="full_online_softmax_reduce",
        deps=[full_sv_online_tid],
    ) as full_online_softmax_reduce_tid:
        fa_reduce_task = pl.tile.get_block_idx()
        if fa_reduce_task < full_online_softmax_reduce_tasks:
            fa_reduce_b = pl.cast(0, pl.INDEX)
            fa_reduce_task_in_b = fa_reduce_task
            if active_tokens != 1:
                if full_online_reduce_tasks_uniform != 0:
                    fa_reduce_task_in_b = (
                        fa_reduce_task // active_tokens
                    )
                    fa_reduce_b = (
                        fa_reduce_task
                        - fa_reduce_task_in_b * active_tokens
                    )
                else:
                    for fa_reduce_scan_b in pl.range(active_tokens):
                        fa_reduce_scan_ctx_len = pl.tensor.read(
                            seq_lens, [fa_reduce_scan_b],
                        )
                        fa_reduce_scan_ctx_blocks = (
                            fa_reduce_scan_ctx_len + BLOCK_SIZE - 1
                        ) // BLOCK_SIZE
                        fa_reduce_scan_segments = (
                            fa_reduce_scan_ctx_blocks
                            + FULL_ATTN_ONLINE_SOFTMAX_BLOCKS_PER_TASK
                            - 1
                        ) // FULL_ATTN_ONLINE_SOFTMAX_BLOCKS_PER_TASK
                        fa_reduce_scan_tasks = (
                            fa_reduce_scan_segments
                            + FULL_ATTN_ONLINE_SOFTMAX_PARTIALS_PER_REDUCE_TASK
                            - 1
                        ) // (
                            FULL_ATTN_ONLINE_SOFTMAX_PARTIALS_PER_REDUCE_TASK
                        )
                        if fa_reduce_scan_b == fa_reduce_b:
                            if (
                                fa_reduce_task_in_b
                                >= fa_reduce_scan_tasks
                            ):
                                fa_reduce_task_in_b = (
                                    fa_reduce_task_in_b
                                    - fa_reduce_scan_tasks
                                )
                                fa_reduce_b = fa_reduce_b + 1

            fa_reduce_ctx_len = pl.tensor.read(seq_lens, [fa_reduce_b])
            fa_reduce_ctx_blocks = (
                fa_reduce_ctx_len + BLOCK_SIZE - 1
            ) // BLOCK_SIZE
            fa_reduce_segment_count = (
                fa_reduce_ctx_blocks
                + FULL_ATTN_ONLINE_SOFTMAX_BLOCKS_PER_TASK
                - 1
            ) // FULL_ATTN_ONLINE_SOFTMAX_BLOCKS_PER_TASK
            fa_reduce_segment0 = (
                fa_reduce_task_in_b
                * FULL_ATTN_ONLINE_SOFTMAX_PARTIALS_PER_REDUCE_TASK
            )
            fa_reduce_segment1 = pl.min(
                fa_reduce_segment_count,
                fa_reduce_segment0
                + FULL_ATTN_ONLINE_SOFTMAX_PARTIALS_PER_REDUCE_TASK,
            )
            fa_reduce_acc_row = (
                fa_reduce_b * MAX_CTX_BLOCKS + fa_reduce_segment0
            )
            fa_reduce_acc_row0 = fa_reduce_acc_row * Q_HEAD_PAD_FULL
            # The first segment in each group doubles as its accumulator.
            # Reload/store makes the dynamic reduction dataflow explicit.
            for fa_reduce_segment in pl.range(
                fa_reduce_segment0 + 1,
                fa_reduce_segment1,
            ):
                fa_acc_oi = pl.slice(
                    online_partial,
                    [Q_HEAD_BATCH_FULL, HEAD_DIM],
                    [fa_reduce_acc_row0, 0],
                )
                fa_acc_mi_row = pl.slice(
                    online_partial_ml,
                    [1, Q_HEAD_BATCH_FULL],
                    [fa_reduce_acc_row, 0],
                )
                fa_acc_li_row = pl.slice(
                    online_partial_ml,
                    [1, Q_HEAD_BATCH_FULL],
                    [fa_reduce_acc_row, Q_HEAD_BATCH_FULL],
                )
                fa_reduce_source_row = (
                    fa_reduce_b * MAX_CTX_BLOCKS + fa_reduce_segment
                )
                fa_reduce_source_row0 = (
                    fa_reduce_source_row * Q_HEAD_PAD_FULL
                )
                fa_segment_oi = pl.slice(
                    online_partial,
                    [Q_HEAD_BATCH_FULL, HEAD_DIM],
                    [fa_reduce_source_row0, 0],
                )
                fa_segment_mi_row = pl.slice(
                    online_partial_ml,
                    [1, Q_HEAD_BATCH_FULL],
                    [fa_reduce_source_row, 0],
                )
                fa_segment_li_row = pl.slice(
                    online_partial_ml,
                    [1, Q_HEAD_BATCH_FULL],
                    [fa_reduce_source_row, Q_HEAD_BATCH_FULL],
                )
                fa_acc_mi_new_row = pl.maximum(
                    fa_acc_mi_row, fa_segment_mi_row,
                )
                fa_acc_alpha_row = pl.exp(
                    pl.sub(fa_acc_mi_row, fa_acc_mi_new_row),
                )
                fa_acc_beta_row = pl.exp(
                    pl.sub(fa_segment_mi_row, fa_acc_mi_new_row),
                )
                fa_acc_li_new_row = pl.add(
                    pl.mul(fa_acc_alpha_row, fa_acc_li_row),
                    pl.mul(fa_acc_beta_row, fa_segment_li_row),
                )
                fa_acc_alpha = pl.reshape(
                    fa_acc_alpha_row, [Q_HEAD_BATCH_FULL, 1],
                )
                fa_acc_beta = pl.reshape(
                    fa_acc_beta_row, [Q_HEAD_BATCH_FULL, 1],
                )
                fa_acc_oi_new = pl.add(
                    pl.row_expand_mul(fa_acc_oi, fa_acc_alpha),
                    pl.row_expand_mul(fa_segment_oi, fa_acc_beta),
                )
                online_partial = pl.assemble(
                    online_partial,
                    fa_acc_oi_new,
                    [fa_reduce_acc_row0, 0],
                )
                fa_acc_ml_new_row = pl.concat(
                    fa_acc_mi_new_row, fa_acc_li_new_row,
                )
                online_partial_ml = pl.assemble(
                    online_partial_ml,
                    fa_acc_ml_new_row,
                    [fa_reduce_acc_row, 0],
                )

    # Merge reduction-group outputs, normalize, and write attn_out.
    # The reduction stage leaves one FP32 partial per group. This task only
    # reads those partials and writes the final attention row, so normalization
    # stays in the same AIV task without mutating the reduction scratch.
    full_online_softmax_active_rows = pl.cast(0, pl.INDEX)
    for fa_online_count_b in pl.range(active_tokens):
        full_online_softmax_active_rows = (
            full_online_softmax_active_rows + 1
        )
    with pl.spmd(
        full_online_softmax_active_rows,
        name_hint="full_online_softmax_finalize",
        deps=[full_online_softmax_reduce_tid],
    ) as _full_online_softmax_finalize_tid:
        fa_b = pl.tile.get_block_idx()
        if fa_b < active_tokens:
            fa_ctx_len = pl.tensor.read(seq_lens, [fa_b])
            fa_ctx_blocks = (fa_ctx_len + BLOCK_SIZE - 1) // BLOCK_SIZE
            fa_segment_count = (
                fa_ctx_blocks
                + FULL_ATTN_ONLINE_SOFTMAX_BLOCKS_PER_TASK
                - 1
            ) // FULL_ATTN_ONLINE_SOFTMAX_BLOCKS_PER_TASK
            fa_reduce_group_count = (
                fa_segment_count
                + FULL_ATTN_ONLINE_SOFTMAX_PARTIALS_PER_REDUCE_TASK
                - 1
            ) // FULL_ATTN_ONLINE_SOFTMAX_PARTIALS_PER_REDUCE_TASK
            fa_acc_lm_row = fa_b * MAX_CTX_BLOCKS
            fa_acc_row0 = fa_acc_lm_row * Q_HEAD_PAD_FULL
            fa_acc_oi = pl.slice(
                online_partial,
                [Q_HEAD_BATCH_FULL, HEAD_DIM],
                [fa_acc_row0, 0],
            )
            fa_acc_mi_row = pl.slice(
                online_partial_ml,
                [1, Q_HEAD_BATCH_FULL],
                [fa_acc_lm_row, 0],
            )
            fa_acc_li_row = pl.slice(
                online_partial_ml,
                [1, Q_HEAD_BATCH_FULL],
                [fa_acc_lm_row, Q_HEAD_BATCH_FULL],
            )
            for fa_reduce_group in pl.range(1, fa_reduce_group_count):
                fa_group_segment = (
                    fa_reduce_group
                    * FULL_ATTN_ONLINE_SOFTMAX_PARTIALS_PER_REDUCE_TASK
                )
                fa_group_row = fa_b * MAX_CTX_BLOCKS + fa_group_segment
                fa_group_row0 = fa_group_row * Q_HEAD_PAD_FULL
                fa_group_oi = pl.slice(
                    online_partial,
                    [Q_HEAD_BATCH_FULL, HEAD_DIM],
                    [fa_group_row0, 0],
                )
                fa_group_mi_row = pl.slice(
                    online_partial_ml,
                    [1, Q_HEAD_BATCH_FULL],
                    [fa_group_row, 0],
                )
                fa_group_li_row = pl.slice(
                    online_partial_ml,
                    [1, Q_HEAD_BATCH_FULL],
                    [fa_group_row, Q_HEAD_BATCH_FULL],
                )
                fa_acc_mi_new_row = pl.maximum(
                    fa_acc_mi_row, fa_group_mi_row,
                )
                fa_acc_alpha_row = pl.exp(
                    pl.sub(fa_acc_mi_row, fa_acc_mi_new_row),
                )
                fa_acc_beta_row = pl.exp(
                    pl.sub(fa_group_mi_row, fa_acc_mi_new_row),
                )
                fa_acc_li_row = pl.add(
                    pl.mul(fa_acc_alpha_row, fa_acc_li_row),
                    pl.mul(fa_acc_beta_row, fa_group_li_row),
                )
                fa_acc_alpha = pl.reshape(
                    fa_acc_alpha_row, [Q_HEAD_BATCH_FULL, 1],
                )
                fa_acc_beta = pl.reshape(
                    fa_acc_beta_row, [Q_HEAD_BATCH_FULL, 1],
                )
                fa_acc_oi = pl.add(
                    pl.row_expand_mul(fa_acc_oi, fa_acc_alpha),
                    pl.row_expand_mul(fa_group_oi, fa_acc_beta),
                )
                fa_acc_mi_row = fa_acc_mi_new_row

            fa_final_li = pl.reshape(
                fa_acc_li_row, [Q_HEAD_BATCH_FULL, 1],
            )
            ctx = pl.row_expand_div(fa_acc_oi, fa_final_li)
            ctx_flat = pl.reshape(
                ctx, [1, Q_HEAD_BATCH_FULL * HEAD_DIM],
            )
            ctx_flat_bf16 = pl.cast(ctx_flat, target_type=pl.BF16)
            attn_out = pl.assemble(attn_out, ctx_flat_bf16, [fa_b, 0])

    # ----- Scope 2.5 — head-wise sigmoid gate. -----
    # The gate is computed on-device in Scope 1.f (gate_exp) and applied inline
    # in the o_proj Scope 3.a below (attn_out * gate_exp per K-chunk, wide
    # element-wise — no [BATCH_TILE,1] strided column load, so no AIV 32-B
    # align fault). No separate scope needed here.

    # ----- Scope 3.a — local o_proj with inline head-gate (element-wise). -----
    # Per (batch_tile, out_chunk): slice gate_exp (on-device computed in Scope
    # 1.f) per K-chunk, multiply attn_out * gate_exp inline (VEC, not a cube RHS
    # → no L0 overflow), then matmul with wo. No intermediate attn_out_gated
    # tensor → no alias.
    # Declare both candidate destinations outside the compile-time feature
    # branches.  PyPTO converts the DSL to SSA before it folds config-backed
    # constant branches, so branch-local tensor declarations are otherwise
    # diagnosed as escaping their defining scope.
    partial_attn_proj = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
    partial_attn_proj_fp32 = pl.create_tensor([BATCH, HIDDEN], dtype=pl.FP32)
    full_out_proj_n_tiles = HIDDEN // FULL_ATTN_OUT_PROJ_MATMUL_N_CHUNK
    full_out_proj_tasks = (
        full_out_proj_n_tiles
        + FULL_ATTN_OUT_PROJ_MATMUL_TILES_PER_TASK - 1
    ) // FULL_ATTN_OUT_PROJ_MATMUL_TILES_PER_TASK
    for b0 in pl.parallel(0, batch_padded, BATCH_TILE):
        # Keep the fused mixed task unsplit across batch rows.  UP_DOWN creates
        # two AIV subblocks; with grouped N tiles, the lower subblock can
        # intermittently publish only a suffix of the first tile.  The
        # unsplit [BATCH_TILE, N] accumulator stays within the tile budget and
        # preserves the workload-derived logical-task count.
        for out_task in pl.spmd(
            full_out_proj_tasks,
            name_hint="full_out_proj_matmul",
        ):
            # Keep the legal cube tile at N=64, but let one logical task
            # sequentially consume multiple output tiles.  This separates
            # compiler tile legality from scheduler grain: architecture
            # calibration changes task count without hard-coding core count.
            for out_local in pl.range(
                FULL_ATTN_OUT_PROJ_MATMUL_TILES_PER_TASK
            ):
                out_tile = (
                    out_task * FULL_ATTN_OUT_PROJ_MATMUL_TILES_PER_TASK
                    + out_local
                )
                if out_tile < full_out_proj_n_tiles:
                    o0 = out_tile * FULL_ATTN_OUT_PROJ_MATMUL_N_CHUNK
                    # First K-chunk (kb=0). Gate applied inline via gate_exp.
                    hg_exp_0 = pl.cast(
                        pl.slice(
                            gate_exp, [BATCH_TILE, K_CHUNK], [b0, 0],
                        ),
                        target_type=pl.FP32,
                    )
                    a_chunk_raw_0 = pl.slice(
                        attn_out, [BATCH_TILE, K_CHUNK], [b0, 0],
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
                        [K_CHUNK, FULL_ATTN_OUT_PROJ_MATMUL_N_CHUNK],
                        [layer_qhidden_base, o0],
                    )
                    o_acc = pl.matmul(
                        a_chunk_0, w_chunk_0, out_dtype=pl.FP32,
                    )
                    for kb in pl.range(1, qhidden_blocks):
                        k0 = kb * K_CHUNK
                        hg_exp = pl.cast(
                            pl.slice(
                                gate_exp,
                                [BATCH_TILE, K_CHUNK],
                                [b0, k0],
                            ),
                            target_type=pl.FP32,
                        )
                        a_chunk_raw = pl.slice(
                            attn_out,
                            [BATCH_TILE, K_CHUNK],
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
                            [K_CHUNK, FULL_ATTN_OUT_PROJ_MATMUL_N_CHUNK],
                            [layer_qhidden_base + k0, o0],
                        )
                        o_acc = pl.matmul_acc(
                            o_acc, a_chunk, w_chunk,
                        )
                    if FULL_ATTN_OUT_PROJ_FUSE_CAST != 0:
                        partial_attn_proj = pl.assemble(
                            partial_attn_proj,
                            pl.cast(o_acc, target_type=pl.BF16),
                            [b0, o0],
                        )
                    else:
                        partial_attn_proj_fp32 = pl.assemble(
                            partial_attn_proj_fp32, o_acc, [b0, o0],
                        )

    if FULL_ATTN_OUT_PROJ_FUSE_CAST == 0:
        for b0 in pl.parallel(0, batch_padded, BATCH_TILE):
            for ob in pl.spmd(
                HIDDEN // FULL_ATTN_OUT_PROJ_VEC_N_CHUNK,
                name_hint="full_out_proj_cast",
            ):
                o0 = ob * FULL_ATTN_OUT_PROJ_VEC_N_CHUNK
                oproj_fp32_chunk = pl.slice(
                    partial_attn_proj_fp32,
                    [BATCH_TILE, FULL_ATTN_OUT_PROJ_VEC_N_CHUNK], [b0, o0],
                )
                partial_attn_proj = pl.assemble(
                    partial_attn_proj,
                    pl.cast(oproj_fp32_chunk, target_type=pl.BF16),
                    [b0, o0],
                )

    # ----- Scope 3.b — TP all-reduce(sum) of the partial o_proj output. -----
    # Phase X.2: the pull-side ring body now lives as
    # TpAttentionFull.tp_all_reduce — see that class for the implementation.
    # After the call every TP rank holds the same fully-reduced
    # ``[BATCH, HIDDEN]`` o_proj output.
    # Phase 15.1 single-rank gate: at TP=1 the all-reduce is a no-op (no
    # peers); skip the function call entirely so the orchestration codegen
    # does not emit a stale SSA rename for the (now-empty) ring body.
    if TP_WORLD_SIZE > 1:
        partial_attn_proj = self.tp_all_reduce(
            partial_attn_proj,
            tmp_window,
            signal_window,
            my_rank,
        )

    # ----- Scope 3.c — residual add (post-all-reduce). -----
    # ``current_hidden`` is replicated across TP ranks, so each rank adds
    # the same residual to the same reduced sum — every rank ends up with
    # the same ``resid1_out``.
    for b0 in pl.parallel(0, batch_padded, BATCH_TILE):
        for ob in pl.spmd(
            HIDDEN // FULL_ATTN_OUT_PROJ_VEC_N_CHUNK,
            name_hint="full_out_resid_add",
        ):
            o0 = ob * FULL_ATTN_OUT_PROJ_VEC_N_CHUNK
            reduced = pl.cast(
                pl.slice(
                    partial_attn_proj,
                    [BATCH_TILE, FULL_ATTN_OUT_PROJ_VEC_N_CHUNK],
                    [b0, o0],
                ),
                target_type=pl.FP32,
            )
            resid = pl.cast(
                pl.slice(
                    resid1_out,
                    [BATCH_TILE, FULL_ATTN_OUT_PROJ_VEC_N_CHUNK],
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
#
# This `@pl.program` builder is the canonical entry point a Wave-3 forward
# pass uses to invoke ``attention_full`` with the TP collective threaded
# through.  It also serves as a compile-cleanness probe (importing this
# module triggers the deferred build inside the harness path).
# =============================================================================
def _build_tp_attention_full_program(tp_size: int = TP_WORLD_SIZE):
    """Return a freshly-built ``@pl.program`` class for the full-attention
    TP epilogue.

    Constructed inside a function so the module imports even on hosts that
    have not finished bringing up the pypto runtime (deferred-build
    pattern, matches the in-tree TP+EP MoE reference).
    """
    if HIDDEN % tp_size != 0:
        raise ValueError(
            f"HIDDEN={HIDDEN} must be divisible by tp_size={tp_size}"
        )
    attention_full_inline = pl.inline(attention_full._func)
    tp_chunk = HIDDEN // tp_size

    @pl.program
    class TpAttentionFull:
        # ---------- Collective: TP all_reduce (barrier-style) ------------
        # Mirrors pypto/tests/st/distributed/test_l3_allreduce.py — verified
        # PASS at TP=2/4/8 on real NPU. Replaces the previous ring all-reduce
        # which hit a codegen bug (multi-step monotonic AtomicAdd → 507018).
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
            gate_r: pl.Tensor[
                [NUM_HEADS_FULL_LOCAL_PAD, HIDDEN_Q_FULL_LOCAL], pl.BF16
            ],
            resid1_out: pl.Out[pl.Tensor[[BATCH, HIDDEN], pl.BF16]],
            tmp_window: pld.DistributedTensor[[BATCH, HIDDEN], pl.BF16],
            signal_window: pld.DistributedTensor[[tp_size, 1], pl.INT32],
            norm_layer_idx: pl.Scalar[pl.INT32],
            attn_layer_idx: pl.Scalar[pl.INT32],
            num_tokens: pl.Scalar[pl.INT32],
            my_rank: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[BATCH, HIDDEN], pl.BF16]:
            resid1_out = attention_full_inline(
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
                num_tokens,
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
                [tp_size, NUM_HEADS_FULL_LOCAL_PAD, HIDDEN_Q_FULL_LOCAL], pl.BF16
            ],
            resid1_out: pl.Out[pl.Tensor[[tp_size, BATCH, HIDDEN], pl.BF16]],
            norm_layer_idx: pl.Scalar[pl.INT32],
            attn_layer_idx: pl.Scalar[pl.INT32],
            num_tokens: pl.Scalar[pl.INT32],
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
                    num_tokens,
                    r,
                    device=r,
                )

    return TpAttentionFull


# =============================================================================
# Distributed-mock torch reference and harness.
#
# The Wave-2 acceptance criterion is that this file parses clean (which is
# verified at import time by ``_build_tp_attention_full_program`` being
# constructible) and that the ``__main__`` harness validates the math via
# a pure-torch 8-rank simulation against a single-card reference.
# =============================================================================
def _torch_single_card_attention_full(
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
    rotary_dim,
    rotary_half,
    rotary_pass,
    q_per_kv,
    eps,
    block_size,
    sliding_window=None,
):
    """Pure-torch single-card oracle (the value all 8 ranks should sum to).

    Mirrors the original single-card golden but is parametrised by
    world-level head/kv counts and an optional sliding window (re-used by
    the SWA sibling for symmetry).
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

    q_h = q_proj.view(batch, num_heads_full, head_dim)
    q_h = zc(q_h * torch.rsqrt(q_h.pow(2).mean(-1, keepdim=True) + eps),
             q_norm_weight.float())
    k_h = k_proj.view(batch, num_kv_heads_full, head_dim)
    k_h = zc(k_h * torch.rsqrt(k_h.pow(2).mean(-1, keepdim=True) + eps),
             k_norm_weight.float())

    k_cache = k_cache_full.clone()
    v_cache = v_cache_full.clone()
    max_ctx_blocks = MAX_BLOCKS_PER_SEQ
    attn_out = torch.zeros(batch, hidden_q, dtype=torch.bfloat16)
    for b in range(batch):
        ctx_len = int(seq_lens[b].item())
        eff_ctx_len = ctx_len if sliding_window is None else min(ctx_len, sliding_window)
        ctx_blocks = (eff_ctx_len + block_size - 1) // block_size
        pos = ctx_len - 1

        cr = rope_cos[pos : pos + 1, :]
        sr = rope_sin[pos : pos + 1, :]
        c_lo, c_hi = cr[:, :rotary_half], cr[:, rotary_half:rotary_dim]
        s_lo, s_hi = sr[:, :rotary_half], sr[:, rotary_half:rotary_dim]

        kh = k_h[b]
        if rotary_pass > 0:
            k_rot = torch.cat([
                kh[:, :rotary_half] * c_lo - kh[:, rotary_half:rotary_dim] * s_lo,
                kh[:, rotary_half:rotary_dim] * c_hi + kh[:, :rotary_half] * s_hi,
                kh[:, rotary_dim : rotary_dim + rotary_pass]
            ], dim=-1)
        else:
            k_rot = torch.cat([
                kh[:, :rotary_half] * c_lo - kh[:, rotary_half:] * s_lo,
                kh[:, rotary_half:] * c_hi + kh[:, :rotary_half] * s_hi,
            ], dim=-1)

        slot = int(slot_mapping[b].item())
        sb_blk = slot // block_size
        sb_off = slot % block_size
        for ki in range(num_kv_heads_full):
            row = (sb_blk * num_kv_heads_full + ki) * block_size + sb_off
            k_cache[row, :] = k_rot[ki].to(torch.bfloat16)
            v_cache[row, :] = v_proj[
                b, ki * head_dim : (ki + 1) * head_dim,
            ].to(torch.bfloat16)

        qh = q_h[b]
        if rotary_pass > 0:
            q_rot = torch.cat([
                qh[:, :rotary_half] * c_lo - qh[:, rotary_half:rotary_dim] * s_lo,
                qh[:, rotary_half:rotary_dim] * c_hi + qh[:, :rotary_half] * s_hi,
                qh[:, rotary_dim : rotary_dim + rotary_pass]
            ], dim=-1)
        else:
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
            for sb in range(ctx_blocks):
                valid_len = min(block_size, eff_ctx_len - sb * block_size)
                pbid = int(block_table[b * max_ctx_blocks + sb].item())
                cr0 = (pbid * num_kv_heads_full + kvh) * block_size
                kt = k_cache[cr0 : cr0 + block_size, :]
                vt = v_cache[cr0 : cr0 + block_size, :]
                rs = q_grp.float() @ kt.float().T
                if valid_len < block_size:
                    rs[:, valid_len:] = torch.finfo(torch.float32).min
                scores = rs * scale
                cm = scores.max(dim=-1, keepdim=True).values
                es = torch.exp(scores - cm)
                es_b = es.to(torch.bfloat16)
                cl = es_b.float().sum(dim=-1, keepdim=True)
                ot = es_b.float() @ vt.float()
                if sb == 0:
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

    gate = (
        torch.sigmoid(normed_bf16.float() @ w_g_full.float())
        .bfloat16()
        .float()
    )
    attn_view = attn_out.view(batch, num_heads_full, head_dim).float()
    attn_gated = (attn_view * gate.unsqueeze(-1)).to(torch.bfloat16)
    attn_gated_flat = attn_gated.view(batch, hidden_q)

    o = attn_gated_flat.float() @ wo_full.float()
    resid1 = (o + hidden_states.float()).bfloat16()
    return resid1


def _torch_per_rank_partial_full(
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
    rotary_dim,
    rotary_half,
    rotary_pass,
    q_per_kv,
    eps,
    block_size,
    sliding_window=None,
):
    """Compute one rank's partial pre-all-reduce o_proj output in pure torch.

    Slices the world-level weights along the TP axis, runs the rank's
    local computation, and returns ``rank_partial_attn`` of shape
    ``[batch, HIDDEN]`` (without the residual add — the harness sums
    these across ranks first, then adds the residual once).
    """
    import math

    import torch

    batch = hidden_states.shape[0]
    heads_local = num_heads_full // tp_world_size
    kv_heads_local = num_kv_heads_full // tp_world_size
    hidden_q_local = heads_local * head_dim
    scale = 1.0 / math.sqrt(head_dim)

    wq_local = wq_full[
        :, rank * hidden_q_local : (rank + 1) * hidden_q_local
    ]
    kv_hidden_local = kv_heads_local * head_dim
    wk_local = wk_full[
        :, rank * kv_hidden_local : (rank + 1) * kv_hidden_local
    ]
    wv_local = wv_full[
        :, rank * kv_hidden_local : (rank + 1) * kv_hidden_local
    ]
    wo_local = wo_full[
        rank * hidden_q_local : (rank + 1) * hidden_q_local, :
    ]
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
    ].contiguous().view(
        num_blocks_total * kv_heads_local * block_size, head_dim,
    )
    v_cache_local = v_cache_full_view[
        :, rank * kv_heads_local : (rank + 1) * kv_heads_local, :, :,
    ].contiguous().view(
        num_blocks_total * kv_heads_local * block_size, head_dim,
    )

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
        eff_ctx_len = ctx_len if sliding_window is None else min(ctx_len, sliding_window)
        ctx_blocks = (eff_ctx_len + block_size - 1) // block_size
        pos = ctx_len - 1
        cr = rope_cos[pos : pos + 1, :]
        sr = rope_sin[pos : pos + 1, :]
        c_lo, c_hi = cr[:, :rotary_half], cr[:, rotary_half:rotary_dim]
        s_lo, s_hi = sr[:, :rotary_half], sr[:, rotary_half:rotary_dim]

        kh = k_h_local[b]
        if rotary_pass > 0:
            k_rot = torch.cat([
                kh[:, :rotary_half] * c_lo - kh[:, rotary_half:rotary_dim] * s_lo,
                kh[:, rotary_half:rotary_dim] * c_hi + kh[:, :rotary_half] * s_hi,
                kh[:, rotary_dim : rotary_dim + rotary_pass]
            ], dim=-1)
        else:
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
        if rotary_pass > 0:
            q_rot = torch.cat([
                qh[:, :rotary_half] * c_lo - qh[:, rotary_half:rotary_dim] * s_lo,
                qh[:, rotary_half:rotary_dim] * c_hi + qh[:, :rotary_half] * s_hi,
                qh[:, rotary_dim : rotary_dim + rotary_pass]
            ], dim=-1)
        else:
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
            for sb in range(ctx_blocks):
                valid_len = min(block_size, eff_ctx_len - sb * block_size)
                pbid = int(block_table[b * max_ctx_blocks + sb].item())
                cr0 = (pbid * kv_heads_local + kvh) * block_size
                kt = k_cache[cr0 : cr0 + block_size, :]
                vt = v_cache[cr0 : cr0 + block_size, :]
                rs = q_grp.float() @ kt.float().T
                if valid_len < block_size:
                    rs[:, valid_len:] = torch.finfo(torch.float32).min
                scores = rs * scale
                cm = scores.max(dim=-1, keepdim=True).values
                es = torch.exp(scores - cm)
                es_b = es.to(torch.bfloat16)
                cl = es_b.float().sum(dim=-1, keepdim=True)
                ot = es_b.float() @ vt.float()
                if sb == 0:
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
    """Simulate TP=TP_WORLD_SIZE ranks in a torch loop and validate.

    The reference is the single-card oracle. The TP path computes each
    rank's partial o_proj output independently, sums them across the
    8 ranks (mocking ``tp_all_reduce``), and then adds the replicated
    residual.
    """
    import torch

    torch.manual_seed(seed)

    if not is_full_attention(layer_idx):
        raise ValueError(
            f"layer_idx={layer_idx} is not a full-attention layer"
        )

    layer_rope_theta = LAYER_ROPE_THETA[layer_idx]
    num_blocks = batch * MAX_BLOCKS_PER_SEQ
    num_heads_full = NUM_HEADS_FULL_LOCAL * TP_WORLD_SIZE                 # 64
    num_kv_heads_full = KV_HEADS_LOCAL * TP_WORLD_SIZE                    # 8
    hidden_q_full = num_heads_full * HEAD_DIM
    kv_hidden_full = num_kv_heads_full * HEAD_DIM
    cache_rows_full = num_blocks * num_kv_heads_full * BLOCK_SIZE

    rope_cos, rope_sin = build_llama3_yarn_rope_tables(
        max_seq, ROTARY_DIM, layer_rope_theta,
        factor=ROPE_SCALING["factor"],
        low=ROPE_SCALING["low_freq_factor"],
        high=ROPE_SCALING["high_freq_factor"],
        orig_max=ROPE_SCALING["original_max_position_embeddings"],
    )

    synthetic_proj_scale = 0.5
    hidden_states = (torch.rand(batch, HIDDEN) - 0.5).bfloat16()
    input_rms_weight = (torch.rand(1, HIDDEN) - 0.5).float()
    wq_full = (torch.rand(HIDDEN, hidden_q_full) / HIDDEN ** 0.5).bfloat16()
    wk_full = (torch.rand(HIDDEN, kv_hidden_full) / HIDDEN ** 0.5).bfloat16()
    wv_full = (
        synthetic_proj_scale * torch.rand(HIDDEN, kv_hidden_full) / HIDDEN ** 0.5
    ).bfloat16()
    q_norm_weight = (torch.rand(1, HEAD_DIM) - 0.5).float()
    k_norm_weight = (torch.rand(1, HEAD_DIM) - 0.5).float()
    wo_full = (
        synthetic_proj_scale * (torch.rand(hidden_q_full, HIDDEN) - 0.5)
        / hidden_q_full ** 0.5
    ).bfloat16()
    w_g_full = (
        synthetic_proj_scale * (torch.rand(HIDDEN, num_heads_full) - 0.5)
        / HIDDEN ** 0.5
    ).bfloat16()

    seq_lens = torch.randint(1, max_seq + 1, (batch,), dtype=torch.int32)
    block_table = torch.arange(num_blocks, dtype=torch.int32)
    slot_mapping = torch.empty(batch, dtype=torch.int32)
    for b in range(batch):
        pos = int(seq_lens[b].item()) - 1
        logical_block = pos // BLOCK_SIZE
        page_offset = pos % BLOCK_SIZE
        phys_block = b * MAX_BLOCKS_PER_SEQ + logical_block
        slot_mapping[b] = phys_block * BLOCK_SIZE + page_offset
    k_cache_full = (torch.rand(cache_rows_full, HEAD_DIM) - 0.5).bfloat16()
    v_cache_full = (
        synthetic_proj_scale * (torch.rand(cache_rows_full, HEAD_DIM) - 0.5)
    ).bfloat16()

    expected_resid1 = _torch_single_card_attention_full(
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
        rotary_dim=ROTARY_DIM,
        rotary_half=ROTARY_HALF,
        rotary_pass=ROTARY_PASS,
        q_per_kv=Q_PER_KV,
        eps=EPS,
        block_size=BLOCK_SIZE,
        sliding_window=None,
    )

    summed_partial = torch.zeros(batch, HIDDEN, dtype=torch.float32)
    for r in range(TP_WORLD_SIZE):
        rank_partial = _torch_per_rank_partial_full(
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
            rotary_dim=ROTARY_DIM,
            rotary_half=ROTARY_HALF,
            rotary_pass=ROTARY_PASS,
            q_per_kv=Q_PER_KV,
            eps=EPS,
            block_size=BLOCK_SIZE,
            sliding_window=None,
        )
        summed_partial = summed_partial + rank_partial.float()

    tp_resid1 = (summed_partial + hidden_states.float()).bfloat16()

    close = torch.isclose(tp_resid1, expected_resid1, rtol=rtol, atol=atol)
    rate = close.float().mean().item()
    n_fail = int((~close).sum().item())
    ok = rate >= pass_rate
    status = "PASS" if ok else "FAIL"
    print(
        f"[{status}] attention_full distributed-mock: pass_rate={rate:.6f} "
        f"threshold={pass_rate:.6f} "
        f"{n_fail}/{tp_resid1.numel()} mismatched rtol={rtol} atol={atol}"
    )
    return ok


def build_tp_attention_full_program(tp_size: int = TP_WORLD_SIZE):
    """Public wrapper for the deferred @pl.program builder."""
    return _build_tp_attention_full_program(tp_size)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", default="a2a3sim",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"],
                        help="Reserved for the Wave-3 real-distributed harness.")
    parser.add_argument("-d", "--device", type=int, default=0,
                        help="Reserved for the Wave-3 real-distributed harness.")
    parser.add_argument("-b", "--batch", type=int, default=BATCH)
    parser.add_argument("--max-seq", type=int, default=128)
    parser.add_argument("--layer-idx", type=int, default=0,
                        help="Which full-attention layer to specialise on.")
    parser.add_argument("--pass-rate", type=float, default=0.97)
    parser.add_argument("--rtol", type=float, default=5e-3)
    parser.add_argument("--atol", type=float, default=5e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--build-program-only", action="store_true",
                        default=False,
                        help="Just construct the @pl.program scaffold and exit.")
    args = parser.parse_args()

    if args.max_seq > MAX_SEQ_DEFAULT:
        raise ValueError(
            f"attention_full harness supports max_seq <= {MAX_SEQ_DEFAULT}"
        )

    program_cls = build_tp_attention_full_program(TP_WORLD_SIZE)
    print(
        f"[OK] built @pl.program TpAttentionFull: {program_cls.__name__} "
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
    "attention_full",
    "build_tp_attention_full_program",
    "_build_tp_attention_full_program",
    "_torch_single_card_attention_full",
    "_torch_per_rank_partial_full",
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
    "ROTARY_PASS",
    "Q_GROUPS",
    "TOTAL_Q_GROUPS",
    "LAYER_QHIDDEN_ROWS_DYN",
]
