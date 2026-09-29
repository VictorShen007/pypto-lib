# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""[中文摘要] Prefill 侧 full-attention(因果,无滑窗;每张卡 8 头,partial RoPE 0.5,
yarn 缩放);Scope 2 写 KV cache → causal flash;Scope 3 末尾
`tp_all_reduce` + 残差。结构与 attention_full.py 镜像,只是 [T, HIDDEN]
sequence-major,不再有 BATCH 凑齐。
[关键装饰器] @pl.program +
   @pl.function(level=HOST, role=Orchestrator)
   @pl.function(type=Orchestration)
   @pl.function(type=InCore)
[SPMD 角色] 跨卡(TP=8)+ 片上 SPMD。
[详见] 中文架构指南 §3

────── 以下为英文原 docstring ──────

Step3p5 prefill full-attention kernel — TP=8 (Phase 6).

Sequence-major counterpart of the decode-side ``attention_full.py``.
The prefill body carries ``T = PREFILL_BATCH * PREFILL_SEQ`` tokens and
emits a fully-reduced residual stream after the local o_proj +
``tp_all_reduce``. Per-rank head counts and Q/K/V slicing are identical
to the decode TP convention (``NUM_HEADS_FULL_LOCAL = 8``,
``KV_HEADS_LOCAL = 1``).

Layer flavour
-------------
Full-attention layers run **causal** attention over the entire prefix
``[0, t]`` for each query token ``t``; there is NO sliding-window
clamp. RoPE is the partial-0.5 variant
(``rotary_dim = 64`` with a 64-lane pass-through tail) and the cos/sin
tables carry the llama3-yarn scaling.

Pipeline (per rank, per layer)
------------------------------
  Scope 1 (delegated to ``prefill_qkv_proj_rope.py``):
      input_rmsnorm → wq/wk/wv → q_norm/k_norm → partial RoPE
      → emit q_rot / k_rot / v_proj + gate_logits
  Scope 2 (this file):
      KV cache write at ``positions[t]`` → causal flash attention
      → head-wise gate (per-rank heads only)
  Scope 3 (this file):
      local o_proj → tp_all_reduce → residual add (post-all-reduce)

Per-card weight bundle (host weight loader contract)
----------------------------------------------------
Same TP-slicing as the decode side:

  * ``input_rms_weight[LAYER, HIDDEN]`` FP32 (replicated)
  * ``wq[LAYER * HIDDEN, HIDDEN_Q_FULL_LOCAL=1024]`` BF16
  * ``wk[LAYER * HIDDEN, KV_HIDDEN_LOCAL=128]`` BF16
  * ``wv[LAYER * HIDDEN, KV_HIDDEN_LOCAL=128]`` BF16
  * ``q_norm_weight[LAYER, HEAD_DIM]`` FP32 (replicated)
  * ``k_norm_weight[LAYER, HEAD_DIM]`` FP32 (replicated)
  * ``w_g[LAYER * HIDDEN, NUM_HEADS_FULL_LOCAL=8]`` BF16
  * ``wo[LAYER * HIDDEN_Q_FULL_LOCAL, HIDDEN]`` BF16 (column-sliced
    by HIDDEN_Q axis; partial sum across TP group via tp_all_reduce)
  * ``rope_cos[ROPE_SEQ, ROTARY_DIM=64]`` FP32 (yarn-scaled, replicated)
  * ``rope_sin[ROPE_SEQ, ROTARY_DIM=64]`` FP32 (yarn-scaled, replicated)

TP collective epilogue
----------------------
Identical to the decode path: each rank holds a partial
``[T, HIDDEN]`` o_proj contribution; ``tp_all_reduce`` sums them into a
uniform reduced output, then the replicated residual is added.
"""

# pyright: reportUndefinedVariable=false

from __future__ import annotations

import os

import pypto.language as pl
import pypto.language.distributed as pld

from ._ops import (
    build_llama3_yarn_rope_tables,
    head_wise_gate_apply,
    tp_all_reduce,
)
from .config import (
    ATTN_SCALE,
    BATCH,
    BLOCK_SIZE,
    BLOCK_TABLE_FLAT_DYN,
    EPS,
    HEAD_DIM,
    HEAD_DIM_INV,
    HIDDEN,
    HIDDEN_INV,
    HIDDEN_Q_FULL_LOCAL,
    HIDDEN_Q_SWA_LOCAL,
    K_CHUNK,
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
    OUT_PROJ_N_CHUNK,
    Q_PER_KV_FULL,
    ROPE_SCALING,
    ROPE_SEQ_DYN,
    ROTARY_HALF_FULL,
    TP_WORLD_SIZE,
)
from .prefill_qkv_proj_rope import (
    PREFILL_BATCH,
    PREFILL_SEQ,
    PREFILL_T,
    TOK_TILE,
    _torch_prefill_qkv_oracle_impl,
)

PREFILL_TILE_COUNT = PREFILL_T // BATCH
assert PREFILL_T % BATCH == 0, (
    f"PREFILL_T={PREFILL_T} must be a multiple of BATCH={BATCH} "
    "so the token-tiling loop can chunk into whole BATCH rows"
)


NUM_HEADS = NUM_HEADS_FULL_LOCAL          # 8
HIDDEN_Q = HIDDEN_Q_FULL_LOCAL            # 1024
KV_HIDDEN_DIM = KV_HIDDEN_LOCAL           # 128
NUM_KV_HEADS_DIM = KV_HEADS_LOCAL         # 1
Q_PER_KV = Q_PER_KV_FULL                  # 8
ROTARY_HALF = ROTARY_HALF_FULL            # 32
ROTARY_DIM = ROTARY_HALF * 2              # 64

# Per-layer dyn dim for the o_proj weight rows.
# Staticized (was pl.dynamic("LAYER_QHIDDEN_ROWS_DYN_PREFILL_FULL")): the L3
# DistributedWorker runtime cannot resolve named pl.dynamic dims (NameError,
# same class as upstream pypto bugs #3/#4). The decode side works around this by
# static-baking model-bound dyn dims; mirror it here. 12 = full-attention layer
# count in LAYER_TYPES; wo bundle stacks all full layers at HIDDEN_Q_FULL_LOCAL
# rows each.
LAYER_QHIDDEN_ROWS_DYN = 12 * HIDDEN_Q_FULL_LOCAL

# Dump switch: default off (production); precision test flow sets
# PYPTO_STEP3P5_DUMP=1 to statically enable the dump assembles.
_DUMP_ENABLED = os.environ.get("PYPTO_STEP3P5_DUMP", "0") != "0"


assert HIDDEN_Q % K_CHUNK == 0
assert HIDDEN % OUT_PROJ_N_CHUNK == 0
assert HIDDEN % TP_WORLD_SIZE == 0


# =============================================================================
# Prefill full-attention body (Scope 2 + Scope 3).
#
# The Scope-1 prelude (input RMSNorm + Q/K/V projection + per-head q/k norm
# + partial RoPE + head-wise gate matmul) is inlined directly inside the
# body — see Phase X.9. The factory in ``prefill_qkv_proj_rope.py`` is no
# longer invoked; its math is materialised in-place so the pypto frontend
# can splice the body cleanly into ``PrefillLayerDense.chip_orch`` /
# ``PrefillLayerMoE.chip_orch`` ``@pl.function`` consumers.
# =============================================================================


@pl.jit.inline
def attention_full_prefill(
    current_hidden: pl.Tensor[[PREFILL_T, HIDDEN], pl.BF16],
    input_rms_weight: pl.Tensor[[LAYER_DYN, HIDDEN], pl.FP32],
    wq: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, HIDDEN_Q_FULL_LOCAL], pl.BF16],
    wk: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, KV_HIDDEN_LOCAL], pl.BF16],
    wv: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, KV_HIDDEN_LOCAL], pl.BF16],
    q_norm_weight: pl.Tensor[[LAYER_DYN, HEAD_DIM], pl.FP32],
    k_norm_weight: pl.Tensor[[LAYER_DYN, HEAD_DIM], pl.FP32],
    block_table: pl.Tensor[[BLOCK_TABLE_FLAT_DYN], pl.INT32],
    slot_mapping: pl.Tensor[[PREFILL_T], pl.INT32],
    rope_cos: pl.Tensor[[ROPE_SEQ_DYN, ROTARY_HALF_FULL * 2], pl.FP32],
    rope_sin: pl.Tensor[[ROPE_SEQ_DYN, ROTARY_HALF_FULL * 2], pl.FP32],
    k_cache: pl.Tensor[[KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16],
    v_cache: pl.Tensor[[KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16],
    wo: pl.Tensor[[LAYER_QHIDDEN_ROWS_DYN, HIDDEN], pl.BF16],
    w_g: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, NUM_HEADS_FULL_LOCAL_PAD], pl.BF16],
    gate_r: pl.Tensor[[NUM_HEADS_FULL_LOCAL_PAD, HIDDEN_Q_FULL_LOCAL], pl.BF16],
    positions: pl.Tensor[[PREFILL_T], pl.INT32],
    resid1_out: pl.Tensor[[PREFILL_T, HIDDEN], pl.BF16],
    qkv_dump: pl.Tensor[[PREFILL_T, HIDDEN_Q_FULL_LOCAL], pl.BF16],
    v_dump: pl.Tensor[[PREFILL_T, KV_HIDDEN_LOCAL], pl.BF16],
    attn_out_dump: pl.Tensor[[PREFILL_T, HIDDEN_Q_FULL_LOCAL], pl.BF16],
    attn_out_gated_dump: pl.Tensor[[PREFILL_T, HIDDEN_Q_FULL_LOCAL], pl.BF16],
    o_proj_dump: pl.Tensor[[PREFILL_T, HIDDEN], pl.BF16],
    o_proj_reduced_dump: pl.Tensor[[PREFILL_T, HIDDEN], pl.BF16],
    scores_dump: pl.Tensor[
        [PREFILL_T * 16, ((PREFILL_T + 127) // 128) * 128], pl.FP32
    ],
    input_norm_dump: pl.Tensor[[PREFILL_T, HIDDEN], pl.BF16],
    # q_proj_dump/q_norm_dump use the SWA width (1536) so the single
    # whole-network spec holds both full (1024 cols) and swa (1536 cols)
    # layers; full layers zero-fill [1024:1536] (see q_dump_pad_zero loop).
    q_proj_dump: pl.Tensor[[PREFILL_T, HIDDEN_Q_SWA_LOCAL], pl.FP32],
    k_proj_dump: pl.Tensor[[PREFILL_T, KV_HIDDEN_LOCAL], pl.FP32],
    v_proj_dump: pl.Tensor[[PREFILL_T, KV_HIDDEN_LOCAL], pl.FP32],
    v_tile_dump: pl.Tensor[[PREFILL_T, KV_HIDDEN_LOCAL], pl.BF16],
    q_norm_dump: pl.Tensor[[PREFILL_T, HIDDEN_Q_SWA_LOCAL], pl.FP32],
    k_norm_dump: pl.Tensor[[PREFILL_T, KV_HIDDEN_LOCAL], pl.FP32],
    gate_logits_dump: pl.Tensor[
        [PREFILL_T, NUM_HEADS_FULL_LOCAL_PAD], pl.BF16
    ],
    attn_delta_dump: pl.Tensor[[PREFILL_T, HIDDEN], pl.BF16],
    norm_layer_idx: pl.Scalar[pl.INT32],
    attn_layer_idx: pl.Scalar[pl.INT32],
    tmp_window: pld.DistributedTensor[
        [PREFILL_T, HIDDEN], pl.BF16
    ],
    signal_window: pld.DistributedTensor[[TP_WORLD_SIZE, 1], pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
):
    """Step3p5 full-attention prefill body — causal mask, head-wise gate.

    ``positions`` and ``slot_mapping`` are passed in by the caller. The
    cache layout matches the decode side (one KV head per card under
    TP=8, ``MAX_BLOCKS_PER_SEQ`` blocks per sequence row), so a single
    KV-head iteration handles the whole rank-local KV-cache shard.

    Phase X.9 — closure-undefined constants are inlined as numeric
    Python literals throughout the body. The pypto inline parser
    captures closure variables from ``prefill_fwd.py``'s frame at
    ``pl.inline(...)`` time, not from this module's globals; constants
    only present here therefore cannot be resolved when the body is
    spliced into ``PrefillLayerDense.chip_orch`` /
    ``PrefillLayerMoE.chip_orch``. Literals make every shape /
    arithmetic argument an unambiguous compile-time integer.
    """
    layer_qhidden_base = attn_layer_idx * HIDDEN_Q_FULL_LOCAL
    num_layers_actual = pl.tensor.dim(input_rms_weight, 0)
    layer_cache_rows = pl.tensor.dim(k_cache, 0) // num_layers_actual
    layer_cache_base = norm_layer_idx * layer_cache_rows

    # ── Scope 1 — inlined prefill QKV+RoPE body (full, Phase X.9). ───────
    # The factory ``select_prefill_qkv(full=True)`` body is materialised
    # directly here so the pypto frontend never sees a free function call
    # from inside a ``@pl.function`` consumer.
    normed_tile = pl.create_tensor([PREFILL_T, HIDDEN], dtype=pl.BF16)
    q_rot = pl.create_tensor([PREFILL_T, HIDDEN_Q_FULL_LOCAL], dtype=pl.BF16)
    k_rot = pl.create_tensor([PREFILL_T, KV_HIDDEN_LOCAL], dtype=pl.BF16)
    v_tile = pl.create_tensor([PREFILL_T, KV_HIDDEN_LOCAL], dtype=pl.BF16)
    gate_logits = pl.create_tensor(
        [PREFILL_T, NUM_HEADS_FULL_LOCAL_PAD], dtype=pl.FP32,
    )

    qkv_d_blocks = HIDDEN // 256
    qkv_q_blocks = HIDDEN_Q_FULL_LOCAL // 128
    layer_hidden_base = attn_layer_idx * HIDDEN

    # ── Stage 1.a — replicated zero-centred input RMSNorm. ───────────
    for tg_idx in pl.spmd(
        PREFILL_T // TOK_TILE, name_hint="prefill_full_rmsnorm_zc",
    ):
        tg = tg_idx * TOK_TILE
        partial_sq = pl.full([1, TOK_TILE], dtype=pl.FP32, value=0.0)
        for kb in pl.range(qkv_d_blocks):
            k0 = kb * 256
            chunk = pl.cast(
                pl.slice(
                    current_hidden,
                    [TOK_TILE, 256], [tg, k0],
                ),
                target_type=pl.FP32,
            )
            partial_sq = pl.add(
                partial_sq,
                pl.reshape(
                    pl.row_sum(pl.mul(chunk, chunk)), [1, TOK_TILE],
                ),
            )
        inv_rms = pl.reshape(
            pl.recip(
                pl.sqrt(
                    pl.add(pl.mul(partial_sq, HIDDEN_INV), EPS),
                ),
            ),
            [TOK_TILE, 1],
        )
        for kb in pl.range(qkv_d_blocks):
            k0 = kb * 256
            chunk = pl.cast(
                pl.slice(
                    current_hidden,
                    [TOK_TILE, 256], [tg, k0],
                ),
                target_type=pl.FP32,
            )
            gamma = pl.slice(
                input_rms_weight,
                [1, 256], [norm_layer_idx, k0],
            )
            scaled_rms = pl.row_expand_mul(chunk, inv_rms)
            normed_rms = pl.col_expand_mul(scaled_rms, pl.add(gamma, 1.0))
            normed_tile = pl.assemble(
                normed_tile,
                pl.cast(normed_rms, target_type=pl.BF16),
                [tg, k0],
            )

    # Module dump 1: input_norm (post-input RMSNorm hidden, replicated).
    if _DUMP_ENABLED:
        input_norm_dump = pl.assemble(input_norm_dump, normed_tile, [0, 0])

    # ── Stage 1.b — Q projection (per-rank heads). ───────────────────
    q_proj = pl.create_tensor(
        [PREFILL_T, HIDDEN_Q_FULL_LOCAL], dtype=pl.FP32,
    )
    for q_idx in pl.spmd(
        (PREFILL_T // TOK_TILE) * qkv_q_blocks,
        name_hint="prefill_full_q_proj",
    ):
        qb_idx = q_idx // qkv_q_blocks
        qo_idx = q_idx % qkv_q_blocks
        tg = qb_idx * TOK_TILE
        q_o0 = qo_idx * 128
        q_a0 = pl.slice(
            normed_tile, [TOK_TILE, 256], [tg, 0],
        )
        q_w0 = pl.slice(
            wq, [256, 128],
            [layer_hidden_base, q_o0],
        )
        q_acc = pl.matmul(q_a0, q_w0, out_dtype=pl.FP32)
        for kb in pl.range(1, qkv_d_blocks):
            k0 = kb * 256
            q_a = pl.slice(
                normed_tile, [TOK_TILE, 256], [tg, k0],
            )
            q_w = pl.slice(
                wq, [256, 128],
                [layer_hidden_base + k0, q_o0],
            )
            q_acc = pl.matmul_acc(q_acc, q_a, q_w)
        q_proj = pl.assemble(q_proj, q_acc, [tg, q_o0])
        # Module dump 2: qkv_proj.q (full = 1024 cols; dump buffer padded to
        # 1536, so write the per-tile chunk in-loop to preserve the target's
        # 1536-col row stride instead of an orchestration-level flat copy).
        if _DUMP_ENABLED:
            q_proj_dump = pl.assemble(q_proj_dump, q_acc, [tg, q_o0])

    # ── Stage 1.c — K projection. ────────────────────────────────────
    k_proj = pl.create_tensor(
        [PREFILL_T, KV_HIDDEN_LOCAL], dtype=pl.FP32,
    )
    for tg_idx in pl.spmd(
        PREFILL_T // TOK_TILE, name_hint="prefill_full_k_proj",
    ):
        tg = tg_idx * TOK_TILE
        k_a0 = pl.slice(
            normed_tile, [TOK_TILE, 256], [tg, 0],
        )
        k_w0 = pl.slice(
            wk, [256, KV_HIDDEN_LOCAL],
            [layer_hidden_base, 0],
        )
        k_acc = pl.matmul(k_a0, k_w0, out_dtype=pl.FP32)
        for kb in pl.range(1, qkv_d_blocks):
            k0 = kb * 256
            k_a = pl.slice(
                normed_tile, [TOK_TILE, 256], [tg, k0],
            )
            k_w = pl.slice(
                wk, [256, KV_HIDDEN_LOCAL],
                [layer_hidden_base + k0, 0],
            )
            k_acc = pl.matmul_acc(k_acc, k_a, k_w)
        k_proj = pl.assemble(k_proj, k_acc, [tg, 0])

    # Module dump 2: qkv_proj.k (single rank-local KV head).
    if _DUMP_ENABLED:
        k_proj_dump = pl.assemble(k_proj_dump, k_proj, [0, 0])

    # ── Stage 1.d — V projection. ────────────────────────────────────
    v_proj = pl.create_tensor(
        [PREFILL_T, KV_HIDDEN_LOCAL], dtype=pl.FP32,
    )
    for tg_idx in pl.spmd(
        PREFILL_T // TOK_TILE, name_hint="prefill_full_v_proj",
    ):
        tg = tg_idx * TOK_TILE
        v_a0 = pl.slice(
            normed_tile, [TOK_TILE, 256], [tg, 0],
        )
        v_w0 = pl.slice(
            wv, [256, KV_HIDDEN_LOCAL],
            [layer_hidden_base, 0],
        )
        v_acc = pl.matmul(v_a0, v_w0, out_dtype=pl.FP32)
        for kb in pl.range(1, qkv_d_blocks):
            k0 = kb * 256
            v_a = pl.slice(
                normed_tile, [TOK_TILE, 256], [tg, k0],
            )
            v_w = pl.slice(
                wv, [256, KV_HIDDEN_LOCAL],
                [layer_hidden_base + k0, 0],
            )
            v_acc = pl.matmul_acc(v_acc, v_a, v_w)
        v_proj = pl.assemble(v_proj, v_acc, [tg, 0])

    # Module dump 2: qkv_proj.v (single rank-local KV head).
    if _DUMP_ENABLED:
        v_proj_dump = pl.assemble(v_proj_dump, v_proj, [0, 0])

    # ── Stage 1.e — head-wise gate matmul (POST-input_layernorm hidden). ─
    # Consumes ``normed_tile`` (same POST-input_layernorm hidden that q/k/v
    # projections use) to match vLLM's Step3p5Attention.g_proj(hidden_states),
    # where hidden_states is already post-input_layernorm. Using pre-norm
    # ``current_hidden`` produces gate_logits ~20× too small and collapses
    # vanilla's saturating sigmoid to a mid-range one.
    for tg_idx in pl.spmd(
        PREFILL_T // TOK_TILE, name_hint="prefill_full_gate_proj",
    ):
        tg = tg_idx * TOK_TILE
        g_a0 = pl.slice(
            normed_tile, [TOK_TILE, 256], [tg, 0],
        )
        g_w0 = pl.slice(
            w_g, [256, NUM_HEADS_FULL_LOCAL_PAD],
            [layer_hidden_base, 0],
        )
        g_acc = pl.matmul(g_a0, g_w0, out_dtype=pl.FP32)
        for kb in pl.range(1, qkv_d_blocks):
            k0 = kb * 256
            g_a = pl.slice(
                normed_tile,
                [TOK_TILE, 256], [tg, k0],
            )
            g_w = pl.slice(
                w_g, [256, NUM_HEADS_FULL_LOCAL_PAD],
                [layer_hidden_base + k0, 0],
            )
            g_acc = pl.matmul_acc(g_acc, g_a, g_w)
        gate_logits = pl.assemble(gate_logits, g_acc, [tg, 0])
        # Module dump 4: attn_gate_logits.gate (FP32 -> BF16, padded to 16).
        if _DUMP_ENABLED:
            gate_logits_dump = pl.assemble(
                gate_logits_dump,
                pl.cast(g_acc, target_type=pl.BF16, mode="rint"),
                [tg, 0],
            )

    # ── Stage 1.f — per-head zero-centred q_norm / k_norm. ───────────
    q_proj_norm = pl.create_tensor(
        [PREFILL_T, HIDDEN_Q_FULL_LOCAL], dtype=pl.FP32,
    )
    k_proj_norm = pl.create_tensor(
        [PREFILL_T, KV_HIDDEN_LOCAL], dtype=pl.FP32,
    )
    # qk_norm folds 8 heads into the row dim ([T_TILE*8, HEAD_DIM]), so its Vec
    # footprint scales with heads too — use a quarter-size token tile to fit UB.
    QK_NORM_T_TILE = TOK_TILE // 4
    for qkn_idx in pl.spmd(
        (PREFILL_T // QK_NORM_T_TILE) * 1,
        name_hint="prefill_full_qk_norm_zc",
    ):
        tg_idx2 = qkn_idx // 1
        kh = qkn_idx % 1
        tg = tg_idx2 * QK_NORM_T_TILE
        q_col = kh * 8 * HEAD_DIM
        q_chunk = pl.reshape(
            pl.slice(
                q_proj, [QK_NORM_T_TILE, 8 * HEAD_DIM], [tg, q_col],
            ),
            [QK_NORM_T_TILE * 8, HEAD_DIM],
        )
        q_gamma = pl.slice(q_norm_weight, [1, HEAD_DIM], [norm_layer_idx, 0])
        # per_head_qk_norm helper deleted (Problem 19); logic inlined below.
        q_sq = pl.row_sum(pl.mul(q_chunk, q_chunk))
        q_inv = pl.recip(pl.sqrt(pl.add(pl.mul(q_sq, 0.0078125), EPS)))
        q_scaled = pl.row_expand_mul(q_chunk, q_inv)
        q_normed = pl.col_expand_mul(q_scaled, pl.add(q_gamma, 1.0))
        q_normed_flat = pl.reshape(
            q_normed, [QK_NORM_T_TILE, 8 * HEAD_DIM],
        )
        q_proj_norm = pl.assemble(
            q_proj_norm, q_normed_flat, [tg, q_col],
        )
        # Module dump 3: qk_norm.q (pre-RoPE; full q = 1024 cols, dump padded
        # to 1536 — write the per-tile chunk in-loop to preserve row stride).
        if _DUMP_ENABLED:
            q_norm_dump = pl.assemble(q_norm_dump, q_normed_flat, [tg, q_col])

        k_col = kh * HEAD_DIM
        k_chunk = pl.slice(k_proj, [QK_NORM_T_TILE, HEAD_DIM], [tg, k_col])
        k_gamma = pl.slice(k_norm_weight, [1, HEAD_DIM], [norm_layer_idx, 0])
        k_sq = pl.row_sum(pl.mul(k_chunk, k_chunk))
        k_inv = pl.recip(pl.sqrt(pl.add(pl.mul(k_sq, 0.0078125), EPS)))
        k_scaled = pl.row_expand_mul(k_chunk, k_inv)
        k_normed = pl.col_expand_mul(k_scaled, pl.add(k_gamma, 1.0))
        k_proj_norm = pl.assemble(k_proj_norm, k_normed, [tg, k_col])
        # Module dump 3: qk_norm.k (pre-RoPE).
        if _DUMP_ENABLED:
            k_norm_dump = pl.assemble(k_norm_dump, k_normed, [tg, k_col])

    # Module dumps 2/3 pad zeroing — full layers only fill the first 1024 q
    # columns (HIDDEN_Q_FULL_LOCAL); the single whole-network dump spec is
    # padded to 1536 (HIDDEN_Q_SWA_LOCAL) so it can also hold SWA layers. The
    # trailing [1024:1536] columns are therefore explicitly zeroed here rather
    # than silently relying on the harness to slice them off by golden width.
    q_dump_pad_blocks = (HIDDEN_Q_SWA_LOCAL - HIDDEN_Q_FULL_LOCAL) // 128
    if _DUMP_ENABLED:
        for p_idx in pl.spmd(
            (PREFILL_T // TOK_TILE) * q_dump_pad_blocks,
            name_hint="prefill_full_q_dump_pad_zero",
        ):
            p_t = p_idx // q_dump_pad_blocks
            p_c = p_idx % q_dump_pad_blocks
            p_tg = p_t * TOK_TILE
            p_col = HIDDEN_Q_FULL_LOCAL + p_c * 128
            q_proj_dump = pl.assemble(
                q_proj_dump,
                pl.full([TOK_TILE, 128], dtype=pl.FP32, value=0.0),
                [p_tg, p_col],
            )
            q_norm_dump = pl.assemble(
                q_norm_dump,
                pl.full([TOK_TILE, 128], dtype=pl.FP32, value=0.0),
                [p_tg, p_col],
            )

    # ── Stage 1.g — partial RoPE on Q and K (per-token positions). ───
    # ``positions[t]`` is the absolute cache row of token t. For the
    # prefill bring-up the host passes a contiguous arange. The full
    # variant uses partial-rotary (rotary_dim=64) with a 64-lane
    # pass-through tail.
    for t in pl.parallel(PREFILL_T):
        pos = pl.cast(pl.tensor.read(positions, [t]), pl.INDEX)
        cos_row = pl.slice(rope_cos, [1, 64], [pos, 0])
        sin_row = pl.slice(rope_sin, [1, 64], [pos, 0])
        cos_lo = pl.slice(cos_row, [1, 32], [0, 0])
        cos_hi = pl.slice(cos_row, [1, 32], [0, 32])
        sin_lo = pl.slice(sin_row, [1, 32], [0, 0])
        sin_hi = pl.slice(sin_row, [1, 32], [0, 32])

        with pl.at(
            level=pl.Level.CORE_GROUP,
            name_hint="prefill_full_rope_q_k",
        ):
            # K RoPE — single rank-local KV head per card under TP=8.
            for kh in pl.range(1):
                k_col = kh * HEAD_DIM
                k_lo = pl.slice(
                    k_proj_norm, [1, 32], [t, k_col],
                )
                k_hi = pl.slice(
                    k_proj_norm,
                    [1, 32],
                    [t, k_col + 32],
                )
                rot_k_lo = pl.sub(
                    pl.col_expand_mul(k_lo, cos_lo),
                    pl.col_expand_mul(k_hi, sin_lo),
                )
                rot_k_hi = pl.add(
                    pl.col_expand_mul(k_hi, cos_hi),
                    pl.col_expand_mul(k_lo, sin_hi),
                )
                k_rot = pl.assemble(
                    k_rot,
                    pl.cast(rot_k_lo, target_type=pl.BF16),
                    [t, k_col],
                )
                k_rot = pl.assemble(
                    k_rot,
                    pl.cast(rot_k_hi, target_type=pl.BF16),
                    [t, k_col + 32],
                )
                # rotary_pass = HEAD_DIM - 64 = 64 (full variant).
                k_pass = pl.slice(
                    k_proj_norm,
                    [1, HEAD_DIM - 64],
                    [t, k_col + 64],
                )
                k_rot = pl.assemble(
                    k_rot,
                    pl.cast(k_pass, target_type=pl.BF16),
                    [t, k_col + 64],
                )
                # V — copy through (no rotation).
                v_slice = pl.slice(
                    v_proj, [1, HEAD_DIM], [t, k_col],
                )
                v_tile = pl.assemble(
                    v_tile,
                    pl.cast(v_slice, target_type=pl.BF16),
                    [t, k_col],
                )

            # Q RoPE — Q_PER_KV consecutive heads per KV-head bundle.
            for kh in pl.range(1):
                q_base_col = kh * 8 * HEAD_DIM
                q_block_norm = pl.reshape(
                    pl.slice(
                        q_proj_norm,
                        [1, 8 * HEAD_DIM], [t, q_base_col],
                    ),
                    [8, HEAD_DIM],
                )
                q_lo = pl.slice(
                    q_block_norm, [8, 32], [0, 0],
                )
                q_hi = pl.slice(
                    q_block_norm,
                    [8, 32],
                    [0, 32],
                )
                rot_q_lo = pl.sub(
                    pl.col_expand_mul(q_lo, cos_lo),
                    pl.col_expand_mul(q_hi, sin_lo),
                )
                rot_q_hi = pl.add(
                    pl.col_expand_mul(q_hi, cos_hi),
                    pl.col_expand_mul(q_lo, sin_hi),
                )
                for qi in pl.range(8):
                    h_col = q_base_col + qi * HEAD_DIM
                    rl = pl.slice(
                        rot_q_lo, [1, 32], [qi, 0],
                    )
                    rh = pl.slice(
                        rot_q_hi, [1, 32], [qi, 0],
                    )
                    q_rot = pl.assemble(
                        q_rot,
                        pl.cast(rl, target_type=pl.BF16),
                        [t, h_col],
                    )
                    q_rot = pl.assemble(
                        q_rot,
                        pl.cast(rh, target_type=pl.BF16),
                        [t, h_col + 32],
                    )
                    # rotary_pass = HEAD_DIM - 64 = 64 (full).
                    q_pass = pl.slice(
                        q_proj_norm,
                        [1, HEAD_DIM - 64],
                        [t, h_col + 64],
                    )
                    q_rot = pl.assemble(
                        q_rot,
                        pl.cast(q_pass, target_type=pl.BF16),
                        [t, h_col + 64],
                    )

    # Module dump: q_rot (post-RoPE Q) and k_rot (post-RoPE K) — attention-core
    # dumps for task #2 core-vs-o_proj error decomposition.
    if _DUMP_ENABLED:
        qkv_dump = pl.assemble(qkv_dump, q_rot, [0, 0])
        v_dump = pl.assemble(v_dump, k_rot, [0, 0])
        v_tile_dump = pl.assemble(v_tile_dump, v_tile, [0, 0])

    # ── Scope 2.a — write KV cache for every prefill token. ──────────────
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_full_kv_write"):
        for t in pl.range(PREFILL_T):
            # Paged write: logical block -> physical block via block_table,
            # matching Scope 2.b's read side (block_table[bt_idx] * BLOCK_SIZE).
            block_idx = t // BLOCK_SIZE
            block = pl.tensor.read(block_table, [block_idx])
            offset = t - block_idx * BLOCK_SIZE
            for kh in pl.range(KV_HEADS_LOCAL):
                cache_row = (
                    layer_cache_base
                    + (block * KV_HEADS_LOCAL + kh) * BLOCK_SIZE
                    + offset
                )
                k_row = pl.slice(k_rot, [1, HEAD_DIM], [t, kh * HEAD_DIM])
                v_row = pl.slice(v_tile, [1, HEAD_DIM], [t, kh * HEAD_DIM])
                k_cache = pl.assemble(k_cache, k_row, [cache_row, 0])
                v_cache = pl.assemble(v_cache, v_row, [cache_row, 0])

    # ── Scope 2.b — causal flash attention per Q token (block-tiled K/V). ─
    # Q is viewed heads-in-rows (q_rot_flat [PREFILL_T*8, HEAD_DIM]) so the
    # per-token flash slices [8, HEAD_DIM] directly inside the InCore block —
    # mirrors deepseek v4 prefill_sparse_attn's ``q_flat = reshape(q,[T*H,D])``.
    # Folding column-heads into rows via reshape *inside* InCore is rejected by
    # codegen, so the reshape stays at orchestration level (here) and outputs go
    # to a heads-in-rows attn_out_flat that is reshaped back after the loop.
    q_rot_flat = pl.reshape(q_rot, [PREFILL_T * 8, HEAD_DIM])
    # Pad each token's 8 Q-heads to Q_HEAD_PAD_FULL=16 so the matmul
    # satisfies the Cube fractal minimum M=16. The 8 padding rows are
    # zero-filled; set_validshape(Q_HEAD_PAD_FULL // 2 = 8) masks them.
    q_rot_padded = pl.create_tensor(
        [PREFILL_T * 16, HEAD_DIM], dtype=pl.BF16,
    )
    for tp in pl.parallel(PREFILL_T):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_full_q_head_pad"):
            q_rot_padded = pl.assemble(
                q_rot_padded,
                pl.slice(q_rot_flat, [8, HEAD_DIM], [tp * 8, 0]),
                [tp * 16, 0],
            )
            q_rot_padded = pl.assemble(
                q_rot_padded,
                pl.full([8, HEAD_DIM], dtype=pl.BF16, value=0.0),
                [tp * 16 + 8, 0],
            )
    attn_out_flat = pl.create_tensor([PREFILL_T * 8, HEAD_DIM], dtype=pl.BF16)
    # DIAG: per-token masked scores (last KV block overwrites earlier ones);
    # rows t*16..t*16+16 hold token t's [16,128] FP32 scores for fill-sentinel
    # verification. Returned via the caller's scores_dump pl.Out.
    bt_stride = pl.cast(32, pl.INDEX)

    # ── Scope 2.b — causal flash attention (deterministic two-stage). ────
    # The prior single-loop form carried the online-softmax running
    # max/sum/output through GM read-modify-write buffers (mi_buf/li_buf/
    # oi_buf) across the dynamically-dispatched QK/PV kernels. For tokens with
    # ctx_blocks >= 3 that cross-kernel recurrence raced (run-to-run drift
    # pinned in Step 4). Split into a pure block-wise map (Stage 1, disjoint
    # per-(token, block) scratch) + an intra-kernel recurrence (Stage 2),
    # mirroring the decode-side attention_full online-softmax pipeline.
    MAX_CTX_BLOCKS = MAX_SEQ_DEFAULT // BLOCK_SIZE

    # Stage 1 scratch — one disjoint slot per (token, KV block). mi/li are
    # stored as [1, NUM_HEADS_FULL_LOCAL] rows (not the [16,1] column shape,
    # which cannot be materialized inside InCore); exp is padded to
    # NUM_HEADS_FULL_LOCAL_PAD rows for the SV cube matmul.
    all_cur_mi = pl.create_tensor(
        [PREFILL_T * MAX_CTX_BLOCKS, NUM_HEADS_FULL_LOCAL], dtype=pl.FP32,
    )
    all_cur_li = pl.create_tensor(
        [PREFILL_T * MAX_CTX_BLOCKS, NUM_HEADS_FULL_LOCAL], dtype=pl.FP32,
    )
    all_exp = pl.create_tensor(
        [PREFILL_T * MAX_CTX_BLOCKS * NUM_HEADS_FULL_LOCAL_PAD, HEAD_DIM],
        dtype=pl.BF16,
    )

    # Stage 1 — QK matmul + per-block softmax (pure map, disjoint writes).
    for t in pl.parallel(PREFILL_T):
        # Each token attends to all positions [0, position] within its
        # batch row. PREFILL_BATCH=1 simplifies the indexing — every
        # token shares batch index 0.
        pos = pl.cast(pl.tensor.read(positions, [t]), pl.INDEX)
        ctx_len = pos + 1
        ctx_blocks = (ctx_len + 128 - 1) // 128
        bt_base = pl.cast(0, pl.INDEX) * bt_stride
        for sb in pl.range(ctx_blocks):
            s0 = sb * 128
            valid_len = pl.min(128, ctx_len - s0)
            bt_idx = bt_base + sb
            pbid = pl.cast(
                pl.tensor.read(block_table, [bt_idx]), pl.INDEX,
            )
            cache_row0 = layer_cache_base + pbid * 128
            with pl.at(
                level=pl.Level.CORE_GROUP,
                name_hint="prefill_full_fa_qk",
            ):
                q_block = q_rot_padded[
                    t * 16 : t * 16 + 16, 0 : HEAD_DIM,
                ]
                k_tile = pl.slice(
                    k_cache, [128, HEAD_DIM], [cache_row0, 0],
                )
                raw_scores = pl.matmul(
                    q_block, k_tile, b_trans=True, out_dtype=pl.FP32,
                )
                scores_scaled = pl.mul(raw_scores, 0.08838834764831845)
                scores_clipped = pl.slice(
                    scores_scaled, [16, 128], [0, 0],
                    valid_shape=[8, valid_len],
                )
                scores = pl.fillpad(
                    scores_clipped, pad_value=pl.PadValue.min,
                )
                if _DUMP_ENABLED:
                    scores_dump = pl.assemble(
                        scores_dump, scores, [t * 16, s0],
                    )
                scores_real = pl.slice(
                    scores, [NUM_HEADS_FULL_LOCAL, 128], [0, 0],
                )
                cur_mi = pl.row_max(scores_real)
                exp_scores = pl.exp(
                    pl.row_expand_sub(scores_real, cur_mi)
                )
                exp_bf16 = pl.cast(exp_scores, target_type=pl.BF16)
                cur_li = pl.row_sum(
                    pl.cast(exp_bf16, target_type=pl.FP32)
                )
                scratch_row = t * MAX_CTX_BLOCKS + sb
                all_cur_mi = pl.assemble(
                    all_cur_mi,
                    pl.reshape(cur_mi, [1, NUM_HEADS_FULL_LOCAL]),
                    [scratch_row, 0],
                )
                all_cur_li = pl.assemble(
                    all_cur_li,
                    pl.reshape(cur_li, [1, NUM_HEADS_FULL_LOCAL]),
                    [scratch_row, 0],
                )
                exp_base = scratch_row * NUM_HEADS_FULL_LOCAL_PAD
                all_exp = pl.assemble(
                    all_exp, exp_bf16, [exp_base, 0],
                )
                all_exp = pl.assemble(
                    all_exp,
                    pl.full(
                        [
                            NUM_HEADS_FULL_LOCAL_PAD
                            - NUM_HEADS_FULL_LOCAL,
                            HEAD_DIM,
                        ],
                        dtype=pl.BF16,
                        value=0.0,
                    ),
                    [exp_base + NUM_HEADS_FULL_LOCAL, 0],
                )

    # Stage 2 — online recurrence + normalize, intra-kernel per token.
    online_oi = pl.create_tensor(
        [PREFILL_T * NUM_HEADS_FULL_LOCAL_PAD, HEAD_DIM], dtype=pl.FP32,
    )
    online_ml = pl.create_tensor(
        [PREFILL_T, 2 * NUM_HEADS_FULL_LOCAL], dtype=pl.FP32,
    )
    for t in pl.spmd(PREFILL_T, name_hint="prefill_full_fa_online"):
        pos = pl.cast(pl.tensor.read(positions, [t]), pl.INDEX)
        ctx_len = pos + 1
        ctx_blocks = (ctx_len + 128 - 1) // 128
        for sb in pl.range(ctx_blocks):
            pbid = pl.cast(
                pl.tensor.read(block_table, [sb]), pl.INDEX,
            )
            cache_row0 = layer_cache_base + pbid * 128
            scratch_row = t * MAX_CTX_BLOCKS + sb
            exp_base = scratch_row * NUM_HEADS_FULL_LOCAL_PAD
            exp_tile = pl.slice(
                all_exp,
                [NUM_HEADS_FULL_LOCAL_PAD, HEAD_DIM],
                [exp_base, 0],
            )
            v_tile_sb = pl.slice(
                v_cache, [128, HEAD_DIM], [cache_row0, 0],
            )
            oi_tmp = pl.matmul(exp_tile, v_tile_sb, out_dtype=pl.FP32)
            cur_mi_row = pl.slice(
                all_cur_mi, [1, NUM_HEADS_FULL_LOCAL], [scratch_row, 0],
            )
            cur_li_row = pl.slice(
                all_cur_li, [1, NUM_HEADS_FULL_LOCAL], [scratch_row, 0],
            )
            zero_heads = pl.full(
                [1, NUM_HEADS_FULL_LOCAL], dtype=pl.FP32, value=0.0,
            )
            cur_mi_ml = pl.concat(cur_mi_row, zero_heads)
            cur_li_ml = pl.concat(cur_li_row, zero_heads)
            if sb == 0:
                online_oi = pl.assemble(
                    online_oi, oi_tmp,
                    [t * NUM_HEADS_FULL_LOCAL_PAD, 0],
                )
                online_ml = pl.assemble(
                    online_ml,
                    pl.concat(cur_mi_row, cur_li_row),
                    [t, 0],
                )
            else:
                acc_oi = pl.slice(
                    online_oi,
                    [NUM_HEADS_FULL_LOCAL_PAD, HEAD_DIM],
                    [t * NUM_HEADS_FULL_LOCAL_PAD, 0],
                )
                acc_mi_row = pl.slice(
                    online_ml, [1, NUM_HEADS_FULL_LOCAL], [t, 0],
                )
                acc_li_row = pl.slice(
                    online_ml,
                    [1, NUM_HEADS_FULL_LOCAL],
                    [t, NUM_HEADS_FULL_LOCAL],
                )
                acc_mi = pl.concat(acc_mi_row, zero_heads)
                acc_li = pl.concat(acc_li_row, zero_heads)
                mi_new = pl.maximum(acc_mi, cur_mi_ml)
                alpha_row = pl.exp(pl.sub(acc_mi, mi_new))
                beta_row = pl.exp(pl.sub(cur_mi_ml, mi_new))
                li_new = pl.add(
                    pl.mul(alpha_row, acc_li),
                    pl.mul(beta_row, cur_li_ml),
                )
                alpha = pl.reshape(
                    alpha_row, [NUM_HEADS_FULL_LOCAL_PAD, 1],
                )
                beta = pl.reshape(
                    beta_row, [NUM_HEADS_FULL_LOCAL_PAD, 1],
                )
                oi_new = pl.add(
                    pl.row_expand_mul(acc_oi, alpha),
                    pl.row_expand_mul(oi_tmp, beta),
                )
                online_oi = pl.assemble(
                    online_oi, oi_new,
                    [t * NUM_HEADS_FULL_LOCAL_PAD, 0],
                )
                mi_new_real = pl.slice(
                    mi_new, [1, NUM_HEADS_FULL_LOCAL], [0, 0],
                )
                li_new_real = pl.slice(
                    li_new, [1, NUM_HEADS_FULL_LOCAL], [0, 0],
                )
                online_ml = pl.assemble(
                    online_ml,
                    pl.concat(mi_new_real, li_new_real),
                    [t, 0],
                )

        oi_final = pl.slice(
            online_oi,
            [NUM_HEADS_FULL_LOCAL, HEAD_DIM],
            [t * NUM_HEADS_FULL_LOCAL_PAD, 0],
        )
        li_final_row = pl.slice(
            online_ml,
            [1, NUM_HEADS_FULL_LOCAL],
            [t, NUM_HEADS_FULL_LOCAL],
        )
        li_final_col = pl.reshape(
            li_final_row, [NUM_HEADS_FULL_LOCAL, 1],
        )
        ctx = pl.row_expand_div(oi_final, li_final_col)
        attn_out_flat = pl.assemble(
            attn_out_flat,
            pl.cast(ctx, target_type=pl.BF16),
            [t * 8, 0],
        )

    attn_out = pl.reshape(attn_out_flat, [PREFILL_T, HIDDEN_Q_FULL_LOCAL])

    # Module dump: attn_out (post-PV normalize, pre-gate). Tile-by-tile
    # slice-assemble (mirrors rms_lm_head_chip_orch's Out writeback) — a
    # whole-tensor assemble into an Out has no codegen.
    if _DUMP_ENABLED:
        for tt in pl.range(PREFILL_T // TOK_TILE):
            t0 = tt * TOK_TILE
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="diag_attn_out_dump"):
                for kb in pl.range(HIDDEN_Q_FULL_LOCAL // K_CHUNK):
                    k0 = kb * K_CHUNK
                    attn_out_dump = pl.assemble(
                        attn_out_dump,
                        pl.slice(attn_out, [TOK_TILE, K_CHUNK], [t0, k0]),
                        [t0, k0],
                    )

    # ── Scope 2.5 — head-wise sigmoid gate via block-diagonal gate_r expand. ──
    # gate_logits is computed post-norm in Stage 1.e (normed_tile @ w_g),
    # matching vLLM's g_proj(hidden_states). The per-head score is expanded
    # across HEAD_DIM via the constant block-diagonal R (``gate_r``), then
    # attn_out is multiplied element-wise. This mirrors decode's
    # head_wise_gate_apply (gate_r fix) and avoids the [TOK_TILE,1] column
    # slice whose TLOAD hits the pto-isa ND2ND [N,1] VEC layout wall.
    gate_score_t = pl.create_tensor(
        [PREFILL_T, NUM_HEADS_FULL_LOCAL_PAD], dtype=pl.BF16,
    )
    for gs_idx in pl.spmd(
        PREFILL_T // TOK_TILE, name_hint="prefill_full_gate_sigmoid",
    ):
        tg = gs_idx * TOK_TILE
        gs_logits = pl.slice(
            gate_logits, [TOK_TILE, NUM_HEADS_FULL_LOCAL_PAD], [tg, 0],
        )
        gs_score = pl.recip(pl.add(pl.exp(pl.neg(gs_logits)), 1.0))
        gate_score_t = pl.assemble(
            gate_score_t, pl.cast(gs_score, target_type=pl.BF16), [tg, 0],
        )

    gate_exp = pl.create_tensor(
        [PREFILL_T, HIDDEN_Q_FULL_LOCAL], dtype=pl.BF16,
    )
    gate_exp_chunks = HIDDEN_Q_FULL_LOCAL // K_CHUNK
    # The [TOK_TILE, K_CHUNK] FP32 matmul accumulator would overflow Vec
    # (same class as o_proj); use a half-size token tile like OUT_PROJ_T_TILE.
    GATE_T_TILE = TOK_TILE // 2
    for ge_idx in pl.spmd(
        (PREFILL_T // GATE_T_TILE) * gate_exp_chunks,
        name_hint="prefill_full_gate_expand",
    ):
        tg_idx = ge_idx // gate_exp_chunks
        gn = ge_idx % gate_exp_chunks
        tg = tg_idx * GATE_T_TILE
        n0 = gn * K_CHUNK
        ge_acc = pl.matmul(
            pl.slice(
                gate_score_t,
                [GATE_T_TILE, NUM_HEADS_FULL_LOCAL_PAD], [tg, 0],
            ),
            pl.slice(gate_r, [NUM_HEADS_FULL_LOCAL_PAD, K_CHUNK], [0, n0]),
            out_dtype=pl.FP32,
        )
        gate_exp = pl.assemble(
            gate_exp, pl.cast(ge_acc, target_type=pl.BF16), [tg, n0],
        )

    attn_out_gated = pl.create_tensor(
        [PREFILL_T, HIDDEN_Q_FULL_LOCAL], dtype=pl.BF16,
    )
    for ag_idx in pl.spmd(
        (PREFILL_T // GATE_T_TILE) * gate_exp_chunks,
        name_hint="prefill_full_gate_apply",
    ):
        tg_idx = ag_idx // gate_exp_chunks
        an = ag_idx % gate_exp_chunks
        tg = tg_idx * GATE_T_TILE
        n0 = an * K_CHUNK
        a_slab = pl.cast(
            pl.slice(attn_out, [GATE_T_TILE, K_CHUNK], [tg, n0]),
            target_type=pl.FP32,
        )
        e_slab = pl.cast(
            pl.slice(gate_exp, [GATE_T_TILE, K_CHUNK], [tg, n0]),
            target_type=pl.FP32,
        )
        gated = pl.cast(pl.mul(a_slab, e_slab), target_type=pl.BF16)
        attn_out_gated = pl.assemble(attn_out_gated, gated, [tg, n0])

    # Module dump: attn_out_gated (post-gate, pre-o_proj).
    if _DUMP_ENABLED:
        attn_out_gated_dump = pl.assemble(attn_out_gated_dump, attn_out_gated, [0, 0])

    # ── Scope 3.a — local o_proj (per-rank partial). ─────────────────────
    qhidden_blocks = HIDDEN_Q_FULL_LOCAL // K_CHUNK
    partial_attn_proj = pl.create_tensor(
        [PREFILL_T, HIDDEN], dtype=pl.BF16,
    )
    # out_proj's FP32 matmul accumulator Vec footprint scales with the token
    # tile; use a half-size tile so it fits UB.
    # Phase A (2026-06-12): mirror of decode-side `full_out_proj` split — the
    # cube matmul and the BF16 cast are emitted as separate spmds (FP32 GM
    # scratch handoff) instead of a single mixed AIC+AIV scope. Same rationale
    # as decode: eliminate MixedKernels dispatch which was suspected of the
    # 507018 VEC UB align fault (see upstream-issues/step3p5-507018-vec-ub-align.md).
    OUT_PROJ_T_TILE = TOK_TILE // 2
    partial_attn_proj_fp32 = pl.create_tensor(
        [PREFILL_T, HIDDEN], dtype=pl.FP32,
    )
    for op_idx in pl.spmd(
        (PREFILL_T // OUT_PROJ_T_TILE) * (HIDDEN // 256),
        name_hint="prefill_full_out_proj_matmul",
        optimizations=[pl.split(pl.SplitMode.UP_DOWN)],
    ):
        tg_idx = op_idx // (HIDDEN // 256)
        ob = op_idx % (HIDDEN // 256)
        tg = tg_idx * OUT_PROJ_T_TILE
        o0 = ob * 256
        a0 = pl.slice(attn_out_gated, [OUT_PROJ_T_TILE, K_CHUNK], [tg, 0])
        w0 = pl.slice(
            wo, [K_CHUNK, 256],
            [layer_qhidden_base, o0],
        )
        o_acc = pl.matmul(a0, w0, out_dtype=pl.FP32)
        for kb in pl.range(1, qhidden_blocks):
            k0 = kb * K_CHUNK
            a = pl.slice(attn_out_gated, [OUT_PROJ_T_TILE, K_CHUNK], [tg, k0])
            w = pl.slice(
                wo, [K_CHUNK, 256],
                [layer_qhidden_base + k0, o0],
            )
            o_acc = pl.matmul_acc(o_acc, a, w)
        partial_attn_proj_fp32 = pl.assemble(
            partial_attn_proj_fp32, o_acc, [tg, o0],
        )

    for op_idx in pl.spmd(
        (PREFILL_T // OUT_PROJ_T_TILE) * (HIDDEN // 256),
        name_hint="prefill_full_out_proj_cast",
    ):
        tg_idx = op_idx // (HIDDEN // 256)
        ob = op_idx % (HIDDEN // 256)
        tg = tg_idx * OUT_PROJ_T_TILE
        o0 = ob * 256
        fp32_chunk = pl.slice(
            partial_attn_proj_fp32,
            [OUT_PROJ_T_TILE, 256], [tg, o0],
        )
        partial_attn_proj = pl.assemble(
            partial_attn_proj,
            pl.cast(fp32_chunk, target_type=pl.BF16, mode="rint"),
            [tg, o0],
        )

    # Module dump: o_proj local (pre-tp_all_reduce). Tile-by-tile
    # slice-assemble (mirrors rms_lm_head_chip_orch's Out writeback).
    if _DUMP_ENABLED:
        for tt in pl.range(PREFILL_T // TOK_TILE):
            t0 = tt * TOK_TILE
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="diag_o_proj_local_dump"):
                for kb in pl.range(HIDDEN // K_CHUNK):
                    k0 = kb * K_CHUNK
                    o_proj_dump = pl.assemble(
                        o_proj_dump,
                        pl.slice(partial_attn_proj, [TOK_TILE, K_CHUNK], [t0, k0]),
                        [t0, k0],
                    )

    # ── Scope 3.b — TP all-reduce of the partial o_proj. ─────────────────
    # Phase X.9: the pull-side ring body now lives as the consumer
    # class's ``tp_all_reduce`` ``@pl.function`` method (same pattern as
    # the decode-side ``attention_full``). The inlined body resolves
    # ``self`` from the enclosing ``chip_orch`` method's scope.
    # Phase A (2026-06-12): mirror of decode 15.B — at TP=1 the all-reduce
    # is a no-op (no peers); skip the call entirely so the orchestration
    # codegen does not emit a stale SSA rename for the (now-empty) ring
    # body.
    if tp_size > 1:
        partial_attn_proj = self.tp_all_reduce(
            partial_attn_proj,
            tmp_window,
            signal_window,
            my_rank,
        )

    # Module dump 5b: post_attn_residual.attn_delta (o_proj output,
    # post-all-reduce, pre-residual-add).
    if _DUMP_ENABLED:
        attn_delta_dump = pl.assemble(attn_delta_dump, partial_attn_proj, [0, 0])

    # NaN diag: o_proj output after TP all-reduce (pre-residual-add).
    # with pl.at(level=pl.Level.CORE_GROUP, name_hint="diag_o_proj_reduced_dump"):
        # for kb in pl.range(HIDDEN // K_CHUNK):
            # k0 = kb * K_CHUNK
            # o_proj_reduced_dump = pl.assemble(
                # o_proj_reduced_dump,
                # pl.slice(partial_attn_proj, [PREFILL_T, K_CHUNK], [0, k0]),
                # [0, k0],
            # )

    # ── Scope 3.c — residual add (post-all-reduce). ──────────────────────
    for ra_idx in pl.spmd(
        (PREFILL_T // TOK_TILE) * (HIDDEN // 256),
        name_hint="prefill_full_resid_add",
    ):
        tg_idx = ra_idx // (HIDDEN // 256)
        ob = ra_idx % (HIDDEN // 256)
        tg = tg_idx * TOK_TILE
        o0 = ob * 256
        reduced = pl.cast(
            pl.slice(
                partial_attn_proj,
                [TOK_TILE, 256], [tg, o0],
            ),
            target_type=pl.FP32,
        )
        resid = pl.cast(
            pl.slice(
                current_hidden,
                [TOK_TILE, 256], [tg, o0],
            ),
            target_type=pl.FP32,
        )
        resid1_out = pl.assemble(
            resid1_out,
            pl.cast(pl.add(reduced, resid), target_type=pl.BF16, mode="rint"),
            [tg, o0],
        )

    return resid1_out


# =============================================================================
# TP wrapper — @pl.program (chip_orch + host_orch).
# =============================================================================
def _build_tp_prefill_attention_full_program(tp_size: int = TP_WORLD_SIZE):
    """Return a freshly-built ``@pl.program`` for the full-attn prefill TP body.

    Deferred-build pattern: mirrors ``attention_full._build_tp_*`` so the
    module imports cleanly without a pypto runtime.
    """
    if HIDDEN % tp_size != 0:
        raise ValueError(
            f"HIDDEN={HIDDEN} must be divisible by tp_size={tp_size}"
        )
    body_inline = pl.inline(attention_full_prefill._func)

    @pl.program
    class PrefillAttentionFull:
        # ---------- Collective: TP all_reduce (Phase X.9, mirrors decode). ----
        # Barrier-style body (mirrors decode moe.py:270-313, PASS TP=8).
        # ``t_rows = PREFILL_T``, ``d_cols = HIDDEN``, ``group_size = tp_size``
        # are baked in from this factory's closure.
        @pl.function(type=pl.FunctionType.InCore)
        def tp_all_reduce(
            self,
            local: pl.Tensor[[PREFILL_T, HIDDEN], pl.BF16],
            tmp_window: pld.DistributedTensor[
                [PREFILL_T, HIDDEN], pl.BF16
            ],
            signal_window: pld.DistributedTensor[
                [tp_size, 1], pl.INT32
            ],
            my_rank: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[PREFILL_T, HIDDEN], pl.BF16]:
            # Barrier-style all-reduce (mirrors decode moe.py:270-313 tp_all_reduce,
            # PASS TP=8). Replaces the pull-side RING which deadlocked at TP=8
            # (monotonic AtomicAdd signal — see task #2). tmp_window is full
            # [PREFILL_T, HIDDEN] (not chunked) so each rank stages its whole
            # contribution before the barrier, then reduces every chunk across
            # all peers. ar_chunk = HIDDEN // 8 (fixed, §7a) keeps the fp32
            # accumulator [BATCH, ar_chunk] = 32KB inside UB at every TP.
            group_size = tp_size
            ar_chunk = HIDDEN // 8
            for tt in pl.range(PREFILL_TILE_COUNT):
                ttr = tt * BATCH
                for k0 in pl.range(0, HIDDEN, ar_chunk):
                    stage_tile = pl.load(local, [ttr, k0], [BATCH, ar_chunk])
                    pl.store(stage_tile, [ttr, k0], tmp_window)
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
            for tt in pl.range(PREFILL_TILE_COUNT):
                ttr = tt * BATCH
                for k0 in pl.range(0, HIDDEN, ar_chunk):
                    own_tile = pl.load(tmp_window, [ttr, k0], [BATCH, ar_chunk])
                    acc = pl.cast(own_tile, target_type=pl.FP32)
                    for peer in pl.range(group_size):
                        if peer != my_rank:
                            recv = pld.tile.remote_load(
                                tmp_window, peer=peer,
                                offsets=[ttr, k0], shape=[BATCH, ar_chunk],
                            )
                            acc = pl.add(acc, pl.cast(recv, target_type=pl.FP32))
                    pl.store(
                        pl.cast(acc, target_type=pl.BF16, mode="rint"), [ttr, k0], local,
                    )
            return local

        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(  # noqa: PLR0913
            self,
            current_hidden: pl.Tensor[[PREFILL_T, HIDDEN], pl.BF16],
            input_rms_weight: pl.Tensor[[LAYER_DYN, HIDDEN], pl.FP32],
            wq: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, HIDDEN_Q], pl.BF16],
            wk: pl.Tensor[
                [LAYER_HIDDEN_ROWS_DYN, KV_HIDDEN_DIM], pl.BF16
            ],
            wv: pl.Tensor[
                [LAYER_HIDDEN_ROWS_DYN, KV_HIDDEN_DIM], pl.BF16
            ],
            q_norm_weight: pl.Tensor[[LAYER_DYN, HEAD_DIM], pl.FP32],
            k_norm_weight: pl.Tensor[[LAYER_DYN, HEAD_DIM], pl.FP32],
            block_table: pl.Tensor[[BLOCK_TABLE_FLAT_DYN], pl.INT32],
            slot_mapping: pl.Tensor[[PREFILL_T], pl.INT32],
            rope_cos: pl.Tensor[[ROPE_SEQ_DYN, ROTARY_DIM], pl.FP32],
            rope_sin: pl.Tensor[[ROPE_SEQ_DYN, ROTARY_DIM], pl.FP32],
            k_cache: pl.Tensor[
                [KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16
            ],
            v_cache: pl.Tensor[
                [KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16
            ],
            wo: pl.Tensor[
                [LAYER_QHIDDEN_ROWS_DYN, HIDDEN], pl.BF16
            ],
            w_g: pl.Tensor[
                [LAYER_HIDDEN_ROWS_DYN, NUM_HEADS_FULL_LOCAL_PAD], pl.BF16
            ],
            gate_r: pl.Tensor[
                [NUM_HEADS_FULL_LOCAL_PAD, HIDDEN_Q_FULL_LOCAL], pl.BF16
            ],
            positions: pl.Tensor[[PREFILL_T], pl.INT32],
            resid1_out: pl.Out[
                pl.Tensor[[PREFILL_T, HIDDEN], pl.BF16]
            ],
            tmp_window: pld.DistributedTensor[
                [PREFILL_T, HIDDEN], pl.BF16
            ],
            signal_window: pld.DistributedTensor[
                [tp_size, 1], pl.INT32
            ],
            norm_layer_idx: pl.Scalar[pl.INT32],
            attn_layer_idx: pl.Scalar[pl.INT32],
            my_rank: pl.Scalar[pl.INT32],
        ):
            _dummy_v = pl.create_tensor(
                [PREFILL_T, KV_HIDDEN_LOCAL], dtype=pl.BF16,
            )
            _dummy_scores = pl.create_tensor(
                [PREFILL_T * 16, 128], dtype=pl.FP32,
            )
            _dummy_input_norm = pl.create_tensor(
                [PREFILL_T, HIDDEN], dtype=pl.BF16,
            )
            _dummy_q = pl.create_tensor(
                [PREFILL_T, HIDDEN_Q_SWA_LOCAL], dtype=pl.FP32,
            )
            _dummy_k = pl.create_tensor(
                [PREFILL_T, KV_HIDDEN_LOCAL], dtype=pl.FP32,
            )
            _dummy_v_proj = pl.create_tensor(
                [PREFILL_T, KV_HIDDEN_LOCAL], dtype=pl.FP32,
            )
            _dummy_v_tile = pl.create_tensor(
                [PREFILL_T, KV_HIDDEN_LOCAL], dtype=pl.BF16,
            )
            _dummy_qn = pl.create_tensor(
                [PREFILL_T, HIDDEN_Q_SWA_LOCAL], dtype=pl.FP32,
            )
            _dummy_kn = pl.create_tensor(
                [PREFILL_T, KV_HIDDEN_LOCAL], dtype=pl.FP32,
            )
            _dummy_gate = pl.create_tensor(
                [PREFILL_T, NUM_HEADS_FULL_LOCAL_PAD], dtype=pl.BF16,
            )
            _dummy_attn_delta = pl.create_tensor(
                [PREFILL_T, HIDDEN], dtype=pl.BF16,
            )
            qkv_dump = pl.create_tensor(
                [PREFILL_T, HIDDEN_Q_FULL_LOCAL], dtype=pl.BF16,
            )
            attn_out_dump = pl.create_tensor(
                [PREFILL_T, HIDDEN_Q_FULL_LOCAL], dtype=pl.BF16,
            )
            attn_out_gated_dump = pl.create_tensor(
                [PREFILL_T, HIDDEN_Q_FULL_LOCAL], dtype=pl.BF16,
            )
            o_proj_dump = pl.create_tensor(
                [PREFILL_T, HIDDEN], dtype=pl.BF16,
            )
            o_proj_reduced_dump = pl.create_tensor(
                [PREFILL_T, HIDDEN], dtype=pl.BF16,
            )
            resid1_out = body_inline(
                current_hidden,
                input_rms_weight,
                wq, wk, wv,
                q_norm_weight, k_norm_weight,
                block_table, slot_mapping,
                rope_cos, rope_sin,
                k_cache, v_cache,
                wo, w_g,
                gate_r,
                positions,
                resid1_out,
                qkv_dump,
                _dummy_v,
                attn_out_dump,
                attn_out_gated_dump,
                o_proj_dump,
                o_proj_reduced_dump,
                _dummy_scores,
                _dummy_input_norm,
                _dummy_q,
                _dummy_k,
                _dummy_v_proj,
                _dummy_v_tile,
                _dummy_qn,
                _dummy_kn,
                _dummy_gate,
                _dummy_attn_delta,
                norm_layer_idx,
                attn_layer_idx,
                tmp_window,
                signal_window,
                my_rank,
            )
            return resid1_out

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(  # noqa: PLR0913, PLR0915
            self,
            current_hidden: pl.Tensor[
                [tp_size, PREFILL_T, HIDDEN], pl.BF16
            ],
            input_rms_weight: pl.Tensor[
                [tp_size, LAYER_DYN, HIDDEN], pl.FP32
            ],
            wq: pl.Tensor[
                [tp_size, LAYER_HIDDEN_ROWS_DYN, HIDDEN_Q], pl.BF16
            ],
            wk: pl.Tensor[
                [tp_size, LAYER_HIDDEN_ROWS_DYN, KV_HIDDEN_DIM], pl.BF16
            ],
            wv: pl.Tensor[
                [tp_size, LAYER_HIDDEN_ROWS_DYN, KV_HIDDEN_DIM], pl.BF16
            ],
            q_norm_weight: pl.Tensor[
                [tp_size, LAYER_DYN, HEAD_DIM], pl.FP32
            ],
            k_norm_weight: pl.Tensor[
                [tp_size, LAYER_DYN, HEAD_DIM], pl.FP32
            ],
            block_table: pl.Tensor[
                [tp_size, BLOCK_TABLE_FLAT_DYN], pl.INT32
            ],
            slot_mapping: pl.Tensor[[tp_size, PREFILL_T], pl.INT32],
            rope_cos: pl.Tensor[
                [tp_size, ROPE_SEQ_DYN, ROTARY_DIM], pl.FP32
            ],
            rope_sin: pl.Tensor[
                [tp_size, ROPE_SEQ_DYN, ROTARY_DIM], pl.FP32
            ],
            k_cache: pl.Tensor[
                [tp_size, KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16
            ],
            v_cache: pl.Tensor[
                [tp_size, KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16
            ],
            wo: pl.Tensor[
                [tp_size, LAYER_QHIDDEN_ROWS_DYN, HIDDEN], pl.BF16
            ],
            w_g: pl.Tensor[
                [tp_size, LAYER_HIDDEN_ROWS_DYN, NUM_HEADS_FULL_LOCAL_PAD], pl.BF16
            ],
            gate_r: pl.Tensor[
                [tp_size, NUM_HEADS_FULL_LOCAL_PAD, HIDDEN_Q_FULL_LOCAL], pl.BF16
            ],
            positions: pl.Tensor[[tp_size, PREFILL_T], pl.INT32],
            resid1_out: pl.Out[
                pl.Tensor[[tp_size, PREFILL_T, HIDDEN], pl.BF16]
            ],
            norm_layer_idx: pl.Scalar[pl.INT32],
            attn_layer_idx: pl.Scalar[pl.INT32],
        ):
            tmp_buf = pld.alloc_window_buffer(PREFILL_T * HIDDEN * 2)
            sig_buf = pld.alloc_window_buffer(tp_size * 4)
            for r in pl.range(pld.world_size()):
                tmp_window = pld.window(
                    tmp_buf, [PREFILL_T, HIDDEN], dtype=pl.BF16,
                )
                signal_window = pld.window(
                    sig_buf, [tp_size, 1], dtype=pl.INT32,
                )
                self.chip_orch(
                    current_hidden[r],
                    input_rms_weight[r],
                    wq[r], wk[r], wv[r],
                    q_norm_weight[r], k_norm_weight[r],
                    block_table[r], slot_mapping[r],
                    rope_cos[r], rope_sin[r],
                    k_cache[r], v_cache[r],
                    wo[r], w_g[r],
                    gate_r[r],
                    positions[r],
                    resid1_out[r],
                    tmp_window, signal_window,
                    norm_layer_idx,
                    attn_layer_idx,
                    r,
                    device=r,
                )

    return PrefillAttentionFull


def _build_tp_prefill_attention_full_program_default():
    """Public default-size builder (TP_WORLD_SIZE = 8)."""
    return _build_tp_prefill_attention_full_program(TP_WORLD_SIZE)


# =============================================================================
# Torch reference + distributed-mock harness.
# =============================================================================
def _torch_single_card_prefill_full(
    *, hidden, input_rms_weight, wq_full, wk_full, wv_full,
    q_norm_weight, k_norm_weight, wo_full, w_g_full,
    rope_cos, rope_sin, positions,
):
    """Pure-torch single-card oracle for the full-attn prefill body."""
    import math

    import torch

    num_heads_full = NUM_HEADS_FULL_LOCAL * TP_WORLD_SIZE
    num_kv_heads_full = KV_HEADS_LOCAL * TP_WORLD_SIZE
    head_dim = HEAD_DIM
    q_per_kv = Q_PER_KV
    scale = 1.0 / math.sqrt(head_dim)

    qkv = _torch_prefill_qkv_oracle_impl(
        hidden=hidden,
        input_rms_weight=input_rms_weight,
        wq_full=wq_full, wk_full=wk_full, wv_full=wv_full,
        q_norm_weight=q_norm_weight, k_norm_weight=k_norm_weight,
        w_g_full=w_g_full,
        rope_cos=rope_cos, rope_sin=rope_sin,
        positions=positions,
        num_heads_full=num_heads_full,
        num_kv_heads_full=num_kv_heads_full,
        rotary_half=ROTARY_HALF,
    )
    q_rot = qkv["q_rot"].float()
    k_rot = qkv["k_rot"].float()
    v_proj = qkv["v_proj"].float()
    gate_logits = qkv["gate_logits"]

    t = hidden.shape[0]
    attn_out = torch.zeros(t, num_heads_full, head_dim)
    for ti in range(t):
        for kvh in range(num_kv_heads_full):
            q_base = kvh * q_per_kv
            q_grp = q_rot[ti, q_base : q_base + q_per_kv, :]
            scores = q_grp @ k_rot[: ti + 1, kvh, :].T  # [q_per_kv, ti+1]
            scores = scores * scale
            probs = torch.softmax(scores, dim=-1)
            ctx = probs @ v_proj[: ti + 1, kvh, :]
            attn_out[ti, q_base : q_base + q_per_kv, :] = ctx

    gate = torch.sigmoid(gate_logits).unsqueeze(-1)
    attn_gated = (attn_out * gate).to(torch.bfloat16)
    attn_gated_flat = attn_gated.view(t, num_heads_full * head_dim)
    o = attn_gated_flat.float() @ wo_full.float()
    resid1 = (o + hidden.float()).bfloat16()
    return resid1


def _torch_per_rank_partial_full(
    *, rank, hidden, input_rms_weight, wq_full, wk_full, wv_full,
    q_norm_weight, k_norm_weight, wo_full, w_g_full,
    rope_cos, rope_sin, positions,
):
    """Per-rank partial pre-all-reduce o_proj for the prefill full path."""
    import math

    import torch

    num_heads_full = NUM_HEADS_FULL_LOCAL * TP_WORLD_SIZE
    num_kv_heads_full = KV_HEADS_LOCAL * TP_WORLD_SIZE
    heads_local = num_heads_full // TP_WORLD_SIZE
    kv_heads_local = num_kv_heads_full // TP_WORLD_SIZE
    hidden_q_local = heads_local * HEAD_DIM
    kv_hidden_local = kv_heads_local * HEAD_DIM
    scale = 1.0 / math.sqrt(HEAD_DIM)

    wq_local = wq_full[
        :, rank * hidden_q_local : (rank + 1) * hidden_q_local,
    ]
    wk_local = wk_full[
        :, rank * kv_hidden_local : (rank + 1) * kv_hidden_local,
    ]
    wv_local = wv_full[
        :, rank * kv_hidden_local : (rank + 1) * kv_hidden_local,
    ]
    wo_local = wo_full[
        rank * hidden_q_local : (rank + 1) * hidden_q_local, :,
    ]
    w_g_local = w_g_full[
        :, rank * heads_local : (rank + 1) * heads_local,
    ]

    qkv = _torch_prefill_qkv_oracle_impl(
        hidden=hidden,
        input_rms_weight=input_rms_weight,
        wq_full=wq_local, wk_full=wk_local, wv_full=wv_local,
        q_norm_weight=q_norm_weight, k_norm_weight=k_norm_weight,
        w_g_full=w_g_local,
        rope_cos=rope_cos, rope_sin=rope_sin,
        positions=positions,
        num_heads_full=heads_local,
        num_kv_heads_full=kv_heads_local,
        rotary_half=ROTARY_HALF,
    )
    q_rot = qkv["q_rot"].float()
    k_rot = qkv["k_rot"].float()
    v_proj = qkv["v_proj"].float()
    gate_logits = qkv["gate_logits"]

    t = hidden.shape[0]
    attn_local = torch.zeros(t, heads_local, HEAD_DIM)
    q_per_kv = Q_PER_KV
    for ti in range(t):
        for kvh in range(kv_heads_local):
            q_base = kvh * q_per_kv
            q_grp = q_rot[ti, q_base : q_base + q_per_kv, :]
            scores = q_grp @ k_rot[: ti + 1, kvh, :].T
            scores = scores * scale
            probs = torch.softmax(scores, dim=-1)
            ctx = probs @ v_proj[: ti + 1, kvh, :]
            attn_local[ti, q_base : q_base + q_per_kv, :] = ctx

    gate = torch.sigmoid(gate_logits).unsqueeze(-1)
    attn_gated = (attn_local * gate).to(torch.bfloat16)
    attn_flat = attn_gated.view(t, hidden_q_local)
    partial_o = (attn_flat.float() @ wo_local.float()).to(torch.bfloat16)
    return partial_o


def _run_distributed_mock(
    *, norm_layer_idx: int = 0, pass_rate: float = 0.97,
    rtol: float = 1e-2, atol: float = 1e-2, seed: int = 0,
):
    """Mock 8-rank simulation of the prefill full-attn body."""
    import torch

    torch.manual_seed(seed)
    layer_rope_theta = LAYER_ROPE_THETA[norm_layer_idx]
    rope_cos, rope_sin = build_llama3_yarn_rope_tables(
        MAX_SEQ_DEFAULT, ROTARY_DIM, layer_rope_theta,
        factor=ROPE_SCALING["factor"],
        low=ROPE_SCALING["low_freq_factor"],
        high=ROPE_SCALING["high_freq_factor"],
        orig_max=ROPE_SCALING["original_max_position_embeddings"],
    )

    num_heads_full = NUM_HEADS_FULL_LOCAL * TP_WORLD_SIZE
    num_kv_heads_full = KV_HEADS_LOCAL * TP_WORLD_SIZE
    hidden_q_full = num_heads_full * HEAD_DIM
    kv_hidden_full = num_kv_heads_full * HEAD_DIM
    proj_scale = 0.5
    hidden = (torch.rand(PREFILL_T, HIDDEN) - 0.5).bfloat16()
    input_rms_weight = ((torch.rand(1, HIDDEN) - 0.5) * 0.1).float()
    wq_full = (
        (torch.rand(HIDDEN, hidden_q_full) - 0.5) / HIDDEN ** 0.5
    ).bfloat16()
    wk_full = (
        (torch.rand(HIDDEN, kv_hidden_full) - 0.5) / HIDDEN ** 0.5
    ).bfloat16()
    wv_full = (
        proj_scale * (torch.rand(HIDDEN, kv_hidden_full) - 0.5) / HIDDEN ** 0.5
    ).bfloat16()
    q_norm_weight = ((torch.rand(1, HEAD_DIM) - 0.5) * 0.1).float()
    k_norm_weight = ((torch.rand(1, HEAD_DIM) - 0.5) * 0.1).float()
    wo_full = (
        proj_scale * (torch.rand(hidden_q_full, HIDDEN) - 0.5)
        / hidden_q_full ** 0.5
    ).bfloat16()
    w_g_full = (
        proj_scale * (torch.rand(HIDDEN, num_heads_full) - 0.5)
        / HIDDEN ** 0.5
    ).bfloat16()

    positions = torch.arange(PREFILL_T, dtype=torch.int32)

    expected_resid1 = _torch_single_card_prefill_full(
        hidden=hidden,
        input_rms_weight=input_rms_weight,
        wq_full=wq_full, wk_full=wk_full, wv_full=wv_full,
        q_norm_weight=q_norm_weight, k_norm_weight=k_norm_weight,
        wo_full=wo_full, w_g_full=w_g_full,
        rope_cos=rope_cos, rope_sin=rope_sin,
        positions=positions,
    )

    summed_partial = torch.zeros(PREFILL_T, HIDDEN, dtype=torch.float32)
    for r in range(TP_WORLD_SIZE):
        rank_partial = _torch_per_rank_partial_full(
            rank=r,
            hidden=hidden,
            input_rms_weight=input_rms_weight,
            wq_full=wq_full, wk_full=wk_full, wv_full=wv_full,
            q_norm_weight=q_norm_weight, k_norm_weight=k_norm_weight,
            wo_full=wo_full, w_g_full=w_g_full,
            rope_cos=rope_cos, rope_sin=rope_sin,
            positions=positions,
        )
        summed_partial = summed_partial + rank_partial.float()
    tp_resid1 = (summed_partial + hidden.float()).bfloat16()

    close = torch.isclose(
        tp_resid1.float(), expected_resid1.float(),
        rtol=rtol, atol=atol,
    )
    rate = close.float().mean().item()
    n_fail = int((~close).sum().item())
    ok = rate >= pass_rate
    status = "PASS" if ok else "FAIL"
    print(
        f"[{status}] prefill_attention_full distributed-mock: "
        f"pass_rate={rate:.6f} threshold={pass_rate:.6f} "
        f"{n_fail}/{tp_resid1.numel()} mismatched "
        f"rtol={rtol} atol={atol}"
    )
    return ok


# =============================================================================
# L0 golden — single-card (tp_size=1) NPU body validation.
#
# The @pl.jit.inline body ``attention_full_prefill`` references ``self`` (the
# tp_all_reduce @pl.function peer method on the enclosing @pl.program class),
# so it cannot be compiled via a module-level @pl.jit wrapper like
# ``prefill_gate_test``. Instead the body is driven through its enclosing
# @pl.program ``_build_tp_prefill_attention_full_program(tp_size=1)``: at
# tp_size=1 the all-reduce ring is a no-op (group_size=1 -> pl.range(0)), so
# the body computes rank-0's local 8-head partial o_proj + residual, which
# ``golden_attention_full_prefill`` mirrors via the per-rank torch oracle.
# =============================================================================


def golden_attention_full_prefill(tensors):
    """Torch reference for the tp_size=1 NPU body: rank-0 local partial + residual.

    The body at tp_size=1 runs rank-0's local 8-head Q/K/V/gate/o projections;
    the all-reduce ring has ``group_size=1`` so ``pl.range(group_size - 1)`` is
    empty (no-op) and ``partial_attn_proj`` passes through unchanged. Scope 3.c
    then adds the residual. This golden mirrors that exactly: the per-rank
    oracle computes rank-0's partial o_proj (pre-residual), and the residual
    add closes the body.

    Host-orch tensors carry a leading ``[tp_size=1]`` axis; rank 0 is indexed
    at ``[0]``. Weight rows for ``layer_idx`` are sliced out of the stacked
    LAYER_*_DYN bundles (matching the body's ``layer_*_base`` row offsets).
    """
    import torch

    norm_layer_idx = tensors["norm_layer_idx"]
    attn_layer_idx = tensors["attn_layer_idx"]
    hidden = tensors["current_hidden"][0]                       # [PREFILL_T, HIDDEN] BF16
    input_rms_weight = tensors["input_rms_weight"][0][
        norm_layer_idx : norm_layer_idx + 1, :
    ]                                                          # [1, HIDDEN] FP32
    wq_full = tensors["wq"][0][
        attn_layer_idx * HIDDEN : attn_layer_idx * HIDDEN + HIDDEN, :
    ]                                                          # [HIDDEN, HIDDEN_Q] BF16
    wk_full = tensors["wk"][0][
        attn_layer_idx * HIDDEN : attn_layer_idx * HIDDEN + HIDDEN, :
    ]                                                          # [HIDDEN, KV_HIDDEN_DIM] BF16
    wv_full = tensors["wv"][0][
        attn_layer_idx * HIDDEN : attn_layer_idx * HIDDEN + HIDDEN, :
    ]                                                          # [HIDDEN, KV_HIDDEN_DIM] BF16
    q_norm_weight = tensors["q_norm_weight"][0][
        norm_layer_idx : norm_layer_idx + 1, :
    ]                                                          # [1, HEAD_DIM] FP32
    k_norm_weight = tensors["k_norm_weight"][0][
        norm_layer_idx : norm_layer_idx + 1, :
    ]                                                          # [1, HEAD_DIM] FP32
    wo_full = tensors["wo"][0][
        attn_layer_idx * HIDDEN_Q : attn_layer_idx * HIDDEN_Q + HIDDEN_Q, :
    ]                                                          # [HIDDEN_Q, HIDDEN] BF16
    w_g_full = tensors["w_g"][0][
        attn_layer_idx * HIDDEN : attn_layer_idx * HIDDEN + HIDDEN, :NUM_HEADS
    ]                                                          # [HIDDEN, NUM_HEADS] BF16
    rope_cos = tensors["rope_cos"][0]                          # [ROPE_SEQ_DYN, ROTARY_DIM]
    rope_sin = tensors["rope_sin"][0]                          # [ROPE_SEQ_DYN, ROTARY_DIM]
    positions = tensors["positions"][0]                        # [PREFILL_T] INT32

    # rank=0 + pass the already-rank-0-local slices as "full": the oracle's
    # rank-0 slice is then identity, yielding the rank-0 partial pre-residual.
    partial_o = _torch_per_rank_partial_full(
        rank=0,
        hidden=hidden,
        input_rms_weight=input_rms_weight,
        wq_full=wq_full, wk_full=wk_full, wv_full=wv_full,
        q_norm_weight=q_norm_weight, k_norm_weight=k_norm_weight,
        wo_full=wo_full, w_g_full=w_g_full,
        rope_cos=rope_cos, rope_sin=rope_sin,
        positions=positions,
    )                                                          # [PREFILL_T, HIDDEN] BF16

    resid1 = (partial_o.float() + hidden.float()).bfloat16()
    tensors["resid1_out"][0] = resid1


def build_tensor_specs(norm_layer_idx: int = 0, attn_layer_idx: int = 0, seed: int = 0):
    """Synthetic single-card (tp_size=1) specs for the L0 NPU golden run.

    Host-orch parameter order (leading ``[1]`` axis = tp_size=1 rank 0).
    Weights are stacked across all full-attn layers (LAYER_*_DYN rows); only
    ``layer_idx``'s row block is consumed by the body and the golden, the
    rest are padding (zero). KV cache / block_table / slot_mapping / positions
    are set up so the paged-cache round-trip (Scope 2.a write -> 2.b read) is
    identity: positions = arange, slot_mapping = arange, block_table[b] = b.
    """
    import torch
    from golden import TensorSpec, ScalarSpec

    torch.manual_seed(seed)
    layer_rope_theta = LAYER_ROPE_THETA[norm_layer_idx]
    rope_cos, rope_sin = build_llama3_yarn_rope_tables(
        int(ROPE_SEQ_DYN), ROTARY_DIM, layer_rope_theta,
        factor=ROPE_SCALING["factor"],
        low=ROPE_SCALING["low_freq_factor"],
        high=ROPE_SCALING["high_freq_factor"],
        orig_max=ROPE_SCALING["original_max_position_embeddings"],
    )
    proj_scale = 0.5

    def init_hidden():
        return (torch.rand(PREFILL_T, HIDDEN) - 0.5).bfloat16().unsqueeze(0)

    def init_input_rms_weight():
        return ((torch.rand(LAYER_DYN, HIDDEN) - 0.5) * 0.1).float().unsqueeze(0)

    def init_wq():
        return (
            (torch.rand(LAYER_HIDDEN_ROWS_DYN, HIDDEN_Q) - 0.5) / HIDDEN ** 0.5
        ).bfloat16().unsqueeze(0)

    def init_wkv():
        return (
            (torch.rand(LAYER_HIDDEN_ROWS_DYN, KV_HIDDEN_DIM) - 0.5) / HIDDEN ** 0.5
        ).bfloat16().unsqueeze(0)

    def init_wv():
        return (
            proj_scale
            * (torch.rand(LAYER_HIDDEN_ROWS_DYN, KV_HIDDEN_DIM) - 0.5)
            / HIDDEN ** 0.5
        ).bfloat16().unsqueeze(0)

    def init_qk_norm_weight():
        return ((torch.rand(LAYER_DYN, HEAD_DIM) - 0.5) * 0.1).float().unsqueeze(0)

    def init_wo():
        return (
            proj_scale * (torch.rand(LAYER_QHIDDEN_ROWS_DYN, HIDDEN) - 0.5)
            / HIDDEN_Q ** 0.5
        ).bfloat16().unsqueeze(0)

    def init_w_g():
        # NUM_HEADS real cols + padding to NUM_HEADS_FULL_LOCAL_PAD (16).
        real = (
            proj_scale * (torch.rand(LAYER_HIDDEN_ROWS_DYN, NUM_HEADS) - 0.5)
            / HIDDEN ** 0.5
        ).bfloat16()
        pad = torch.zeros(
            LAYER_HIDDEN_ROWS_DYN,
            NUM_HEADS_FULL_LOCAL_PAD - NUM_HEADS,
            dtype=torch.bfloat16,
        )
        return torch.cat([real, pad], dim=1).unsqueeze(0)

    def init_gate_r():
        # Block-diagonal expander R [NUM_HEADS_FULL_LOCAL_PAD, HIDDEN_Q_FULL_LOCAL]:
        # gate_exp[t, h*HEAD_DIM + d] = gate_score[t, h]. Real heads 0..7 carry a
        # 1-block across their HEAD_DIM; pad heads 8..15 are all zero.
        r = torch.zeros(
            NUM_HEADS_FULL_LOCAL_PAD, HIDDEN_Q_FULL_LOCAL, dtype=torch.bfloat16,
        )
        for h in range(NUM_HEADS_FULL_LOCAL):
            r[h, h * HEAD_DIM:(h + 1) * HEAD_DIM] = 1.0
        return r.unsqueeze(0)

    def init_positions():
        return torch.arange(PREFILL_T, dtype=torch.int32).unsqueeze(0)

    def init_slot_mapping():
        return torch.arange(PREFILL_T, dtype=torch.int32).unsqueeze(0)

    def init_block_table():
        return torch.arange(BLOCK_TABLE_FLAT_DYN, dtype=torch.int32).unsqueeze(0)

    return [
        TensorSpec(
            "current_hidden", [1, PREFILL_T, HIDDEN], torch.bfloat16,
            init_value=init_hidden,
        ),
        TensorSpec(
            "input_rms_weight", [1, LAYER_DYN, HIDDEN], torch.float32,
            init_value=init_input_rms_weight,
        ),
        TensorSpec(
            "wq", [1, LAYER_HIDDEN_ROWS_DYN, HIDDEN_Q], torch.bfloat16,
            init_value=init_wq,
        ),
        TensorSpec(
            "wk", [1, LAYER_HIDDEN_ROWS_DYN, KV_HIDDEN_DIM], torch.bfloat16,
            init_value=init_wkv,
        ),
        TensorSpec(
            "wv", [1, LAYER_HIDDEN_ROWS_DYN, KV_HIDDEN_DIM], torch.bfloat16,
            init_value=init_wv,
        ),
        TensorSpec(
            "q_norm_weight", [1, LAYER_DYN, HEAD_DIM], torch.float32,
            init_value=init_qk_norm_weight,
        ),
        TensorSpec(
            "k_norm_weight", [1, LAYER_DYN, HEAD_DIM], torch.float32,
            init_value=init_qk_norm_weight,
        ),
        TensorSpec(
            "block_table", [1, BLOCK_TABLE_FLAT_DYN], torch.int32,
            init_value=init_block_table,
        ),
        TensorSpec(
            "slot_mapping", [1, PREFILL_T], torch.int32,
            init_value=init_slot_mapping,
        ),
        TensorSpec(
            "rope_cos", [1, ROPE_SEQ_DYN, ROTARY_DIM], torch.float32,
            init_value=lambda: rope_cos.unsqueeze(0),
        ),
        TensorSpec(
            "rope_sin", [1, ROPE_SEQ_DYN, ROTARY_DIM], torch.float32,
            init_value=lambda: rope_sin.unsqueeze(0),
        ),
        TensorSpec(
            "k_cache", [1, KV_CACHE_ROWS_DYN, HEAD_DIM], torch.bfloat16,
            init_value=torch.zeros,
        ),
        TensorSpec(
            "v_cache", [1, KV_CACHE_ROWS_DYN, HEAD_DIM], torch.bfloat16,
            init_value=torch.zeros,
        ),
        TensorSpec(
            "wo", [1, LAYER_QHIDDEN_ROWS_DYN, HIDDEN], torch.bfloat16,
            init_value=init_wo,
        ),
        TensorSpec(
            "w_g", [1, LAYER_HIDDEN_ROWS_DYN, NUM_HEADS_FULL_LOCAL_PAD],
            torch.bfloat16, init_value=init_w_g,
        ),
        TensorSpec(
            "gate_r", [1, NUM_HEADS_FULL_LOCAL_PAD, HIDDEN_Q_FULL_LOCAL],
            torch.bfloat16, init_value=init_gate_r,
        ),
        TensorSpec(
            "positions", [1, PREFILL_T], torch.int32,
            init_value=init_positions,
        ),
        TensorSpec(
            "resid1_out", [1, PREFILL_T, HIDDEN], torch.bfloat16, is_output=True,
        ),
        ScalarSpec("norm_layer_idx", torch.int32, norm_layer_idx),
        ScalarSpec("attn_layer_idx", torch.int32, attn_layer_idx),
    ]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Step3p5 prefill full-attention body. Default path: single-card "
            "(tp_size=1) NPU L0 golden via golden.runner.run. The pure-torch "
            "8-rank distributed mock is available via --distributed-mock."
        ),
    )
    parser.add_argument(
        "-p", "--platform", default="a2a3", choices=["a2a3"],
    )
    parser.add_argument("-d", "--device", type=int, default=3)
    parser.add_argument("--norm-layer-idx", type=int, default=0)
    parser.add_argument("--attn-layer-idx", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rtol", type=float, default=0.05)
    parser.add_argument("--atol", type=float, default=0.05)
    parser.add_argument("--pass-rate", type=float, default=0.97)
    parser.add_argument("--build-program-only", action="store_true", default=False)
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument(
        "--distributed-mock", action="store_true", default=False,
        help="Run the pure-torch 8-rank distributed mock (no NPU).",
    )
    args = parser.parse_args()

    if args.distributed_mock:
        program_cls = _build_tp_prefill_attention_full_program(TP_WORLD_SIZE)
        _prog_name = getattr(program_cls, "name", None) or type(program_cls).__name__
        print(
            f"[OK] built @pl.program PrefillAttentionFull: {_prog_name} "
            f"(tp_size={TP_WORLD_SIZE}, T={PREFILL_T})"
        )
        if args.build_program_only:
            raise SystemExit(0)
        ok = _run_distributed_mock(
            norm_layer_idx=args.norm_layer_idx,
            pass_rate=args.pass_rate,
            rtol=args.rtol, atol=args.atol,
            seed=args.seed,
        )
        raise SystemExit(0 if ok else 1)

    from golden import ratio_allclose, run

    program_cls = _build_tp_prefill_attention_full_program(tp_size=1)
    _prog_name = getattr(program_cls, "name", None) or type(program_cls).__name__
    print(
        f"[OK] built @pl.program PrefillAttentionFull: {_prog_name} "
        f"(tp_size=1, T={PREFILL_T})"
    )
    if args.build_program_only:
        raise SystemExit(0)

    result = run(
        program=program_cls,
        specs=build_tensor_specs(
            norm_layer_idx=args.norm_layer_idx,
            attn_layer_idx=args.attn_layer_idx,
            seed=args.seed,
        ),
        golden_fn=golden_attention_full_prefill,
        runtime_cfg=dict(platform=args.platform, device_id=args.device),
        rtol=args.rtol, atol=args.atol,
        compile_only=args.compile_only,
        compare_fn={
            "resid1_out": ratio_allclose(
                atol=0.05, rtol=0.05, max_error_ratio=0.05,
            ),
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)


__all__ = [
    "PREFILL_BATCH",
    "PREFILL_SEQ",
    "PREFILL_T",
    "TOK_TILE",
    "NUM_HEADS",
    "HIDDEN_Q",
    "KV_HIDDEN_DIM",
    "NUM_KV_HEADS_DIM",
    "Q_PER_KV",
    "ROTARY_HALF",
    "ROTARY_DIM",
    "LAYER_QHIDDEN_ROWS_DYN",
    "attention_full_prefill",
    "_build_tp_prefill_attention_full_program",
    "_build_tp_prefill_attention_full_program_default",
    "_torch_single_card_prefill_full",
    "_torch_per_rank_partial_full",
    "_run_distributed_mock",
    "golden_attention_full_prefill",
    "build_tensor_specs",
]
