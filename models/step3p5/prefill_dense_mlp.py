# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""[中文摘要] Prefill dense-MLP body: TP-sliced gate/up + SiLU + down + tp_all_reduce,
内部 ``pl.range(PREFILL_TILE_COUNT)`` token-tiling 把 192KB Vec working set 控制在
BATCH 大小。从 prefill_fwd.py 的 ``_prefill_dense_mlp_body_tp`` 抽取,供
顶层 host_orch 与 ``prefill_fwd.py`` 的 dense-MLP 分支共享。
[关键装饰器] @pl.jit.inline
[SPMD 角色] TP 切分 + tp_all_reduce (self 方法, 由 enclosing @pl.program 提供)
[详见] 中文架构指南 §10; 设计文档 §4.3

────── 以下为英文原 docstring ──────

Step3p5 prefill dense-MLP inline body (TP=EP=8 BF16).

Token-tiled counterpart of ``decode_layer._dense_mlp_body_tp``: the body
receives the full ``[PREFILL_T, HIDDEN]`` post-attention residual and runs
an internal ``for tg in pl.range(PREFILL_TILE_COUNT)`` loop so each Vec
working set is BATCH-sized (fits the 192KB UB). This differs from the MoE
bodies (§4.4-4.8) which receive per-tile ``[BATCH, HIDDEN]`` input and are
driven by a caller-side ``pl.unroll(PREFILL_TILE_COUNT)`` loop — see
design §4.3 for the rationale (dense: single tp_all_reduce over the full
PREFILL_T partial; MoE: EP a2a is inherently per-tile).

The body references ``self.tp_all_reduce`` — bound to the enclosing
``@pl.program``'s method when inlined via ``pl.inline(body._func)`` (same
pattern as decode-side ``dense_mlp.py:dense_mlp_body_tp``). L0 compile
testing therefore requires a ``@pl.program`` wrapper that imports and
inlines this body.
"""

# pyright: reportUndefinedVariable=false

from __future__ import annotations

import os

import pypto.language as pl
import pypto.language.distributed as pld

from .config import (
    BATCH,
    EPS,
    HIDDEN,
    HIDDEN_INV,
    INTERMEDIATE_LOCAL,
    K_CHUNK,
    LAYER_DYN,
    LAYER_HIDDEN_ROWS_DYN,
    LAYER_INTER_ROWS_DYN,
    MLP_OUT_CHUNK,
    TP_WORLD_SIZE,
)
from .prefill_qkv_proj_rope import PREFILL_T

# Compile-time tile count (mirrors prefill_moe.py / prefill_fwd.py).
PREFILL_TILE_COUNT = PREFILL_T // BATCH
assert PREFILL_T % BATCH == 0, (
    f"PREFILL_T={PREFILL_T} must be a multiple of BATCH={BATCH} so the "
    "dense-MLP token-tiling loop can chunk into whole BATCH rows"
)

INTER_LOCAL = INTERMEDIATE_LOCAL

# Dump switch: default off (production); precision test flow sets
# PYPTO_STEP3P5_DUMP=1 to statically enable the dump assembles.
_DUMP_ENABLED = os.environ.get("PYPTO_STEP3P5_DUMP", "0") != "0"


@pl.jit.inline
def prefill_dense_mlp_body(
    resid1: pl.Tensor[[PREFILL_T, HIDDEN], pl.BF16],
    post_rms_weight: pl.Tensor[[LAYER_DYN, HIDDEN], pl.FP32],
    # Formal [LAYER_HIDDEN_ROWS_DYN] decl (decode dense_mlp_body_tp precedent):
    # the dense stack is 3*HIDDEN (12288) and the body selects a layer via
    # mlp_layer_idx * HIDDEN below; the whole-net caller pre-slices to a single
    # [HIDDEN] layer and passes mlp_layer_idx=0.
    w_gate: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, INTER_LOCAL], pl.BF16],
    w_up: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, INTER_LOCAL], pl.BF16],
    w_down: pl.Tensor[[LAYER_INTER_ROWS_DYN, HIDDEN], pl.BF16],
    next_hidden: pl.Tensor[[PREFILL_T, HIDDEN], pl.BF16],
    post_norm_dump: pl.Tensor[[PREFILL_T, HIDDEN], pl.BF16],
    ffn_output_dump: pl.Tensor[[PREFILL_T, HIDDEN], pl.BF16],
    norm_layer_idx: pl.Scalar[pl.INT32],
    mlp_layer_idx: pl.Scalar[pl.INT32],
    tmp_window: pld.DistributedTensor[[PREFILL_T, HIDDEN], pl.BF16],
    signal_window: pld.DistributedTensor[[TP_WORLD_SIZE, 1], pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
):
    """Post-attention zero-centred RMSNorm + TP-sliced SwiGLU MLP + residual.

    Token-tiled (DeepSeek-V4 style): process BATCH rows at a time so the
    per-scope Vec working set fits the 192KB UB. Staging tensors stay full
    ``[PREFILL_T, HIDDEN]``; only the Vec tiles are ``[BATCH, K_CHUNK]``.
    """
    hidden_blocks = HIDDEN // K_CHUNK
    mlp_out_blocks = INTER_LOCAL // MLP_OUT_CHUNK
    layer_hidden_base = mlp_layer_idx * HIDDEN
    layer_inter_base = mlp_layer_idx * INTER_LOCAL

    # ── Step 1: post-attention zero-centred RMSNorm. ──────────────────
    post_norm = pl.create_tensor([PREFILL_T, HIDDEN], dtype=pl.BF16)
    resid1_fp32 = pl.create_tensor([PREFILL_T, HIDDEN], dtype=pl.FP32)
    with pl.at(
        level=pl.Level.CORE_GROUP, name_hint="prefill_dense_post_rmsnorm_zc",
    ):
        for tg in pl.range(PREFILL_TILE_COUNT):
            t0 = tg * BATCH
            sq_sum = pl.full([1, BATCH], dtype=pl.FP32, value=0.0)
            for kb in pl.range(hidden_blocks):
                k0 = kb * K_CHUNK
                rchunk = pl.cast(
                    pl.slice(resid1, [BATCH, K_CHUNK], [t0, k0]),
                    target_type=pl.FP32,
                )
                resid1_fp32 = pl.assemble(resid1_fp32, rchunk, [t0, k0])
                sq_sum = pl.add(
                    sq_sum,
                    pl.reshape(pl.row_sum(pl.mul(rchunk, rchunk)), [1, BATCH]),
                )
            inv_rms_dense = pl.recip(
                pl.sqrt(pl.add(pl.mul(sq_sum, HIDDEN_INV), EPS)),
            )
            inv_rms_col = pl.reshape(inv_rms_dense, [BATCH, 1])
            for kb3 in pl.range(hidden_blocks):
                k0 = kb3 * K_CHUNK
                norm_chunk = pl.slice(
                    resid1_fp32, [BATCH, K_CHUNK], [t0, k0],
                )
                gamma = pl.slice(
                    post_rms_weight, [1, K_CHUNK], [norm_layer_idx, k0],
                )
                scaled = pl.row_expand_mul(norm_chunk, inv_rms_col)
                normed = pl.col_expand_mul(scaled, pl.add(gamma, 1.0))
                post_norm = pl.assemble(
                    post_norm,
                    pl.cast(normed, target_type=pl.BF16),
                    [t0, k0],
                )

    # Module dump 6: post_attn_norm.hidden_states (dense MLP input norm).
    if _DUMP_ENABLED:
        post_norm_dump = pl.assemble(post_norm_dump, post_norm, [0, 0])

    # ── Step 2: TP-sliced gate_up + SiLU. ─────────────────────────────
    mlp_tile = pl.create_tensor([PREFILL_T, INTER_LOCAL], dtype=pl.BF16)
    for ob in pl.spmd(
        mlp_out_blocks, name_hint="prefill_dense_gate_up_silu_tp",
        optimizations=[pl.split(pl.SplitMode.UP_DOWN)],
    ):
        mlp_o0 = ob * MLP_OUT_CHUNK
        # Token-tiled: matmul M = BATCH so the FP32 SwiGLU accumulators fit UB.
        for tg in pl.range(PREFILL_TILE_COUNT):
            t0 = tg * BATCH
            post_chunk_0 = pl.slice(post_norm, [BATCH, K_CHUNK], [t0, 0])
            wg_0 = pl.slice(
                w_gate, [K_CHUNK, MLP_OUT_CHUNK],
                [layer_hidden_base, mlp_o0],
            )
            wu_0 = pl.slice(
                w_up, [K_CHUNK, MLP_OUT_CHUNK],
                [layer_hidden_base, mlp_o0],
            )
            gate_acc = pl.matmul(post_chunk_0, wg_0, out_dtype=pl.FP32)
            up_acc = pl.matmul(post_chunk_0, wu_0, out_dtype=pl.FP32)
            for kb in pl.range(1, hidden_blocks):
                k0 = kb * K_CHUNK
                post_chunk = pl.slice(
                    post_norm, [BATCH, K_CHUNK], [t0, k0],
                )
                wg = pl.slice(
                    w_gate, [K_CHUNK, MLP_OUT_CHUNK],
                    [layer_hidden_base + k0, mlp_o0],
                )
                wu = pl.slice(
                    w_up, [K_CHUNK, MLP_OUT_CHUNK],
                    [layer_hidden_base + k0, mlp_o0],
                )
                gate_acc = pl.matmul_acc(gate_acc, post_chunk, wg)
                up_acc = pl.matmul_acc(up_acc, post_chunk, wu)
            sigmoid = pl.recip(pl.add(pl.exp(pl.neg(gate_acc)), 1.0))
            mlp_chunk = pl.mul(pl.mul(gate_acc, sigmoid), up_acc)
            mlp_tile = pl.assemble(
                mlp_tile,
                pl.cast(mlp_chunk, target_type=pl.BF16),
                [t0, mlp_o0],
            )

    # ── Step 3: TP-sliced w_down -> partial [PREFILL_T, HIDDEN]. ──────
    partial_hidden = pl.create_tensor([PREFILL_T, HIDDEN], dtype=pl.BF16)
    for dob in pl.spmd(
        hidden_blocks, name_hint="prefill_dense_down_proj_tp",
        optimizations=[pl.split(pl.SplitMode.UP_DOWN)],
    ):
        d0 = dob * K_CHUNK
        # Token-tiled: matmul M = BATCH so the FP32 down-proj accumulator fits UB.
        for tg in pl.range(PREFILL_TILE_COUNT):
            t0 = tg * BATCH
            mlp_chunk_0 = pl.slice(mlp_tile, [BATCH, MLP_OUT_CHUNK], [t0, 0])
            w_down_chunk_0 = pl.slice(
                w_down, [MLP_OUT_CHUNK, K_CHUNK],
                [layer_inter_base, d0],
            )
            down_acc = pl.matmul(
                mlp_chunk_0, w_down_chunk_0, out_dtype=pl.FP32,
            )
            for ob in pl.range(1, INTER_LOCAL // MLP_OUT_CHUNK):
                down_o0 = ob * MLP_OUT_CHUNK
                mlp_chunk_bf16 = pl.slice(
                    mlp_tile, [BATCH, MLP_OUT_CHUNK], [t0, down_o0],
                )
                w_down_chunk = pl.slice(
                    w_down, [MLP_OUT_CHUNK, K_CHUNK],
                    [layer_inter_base + down_o0, d0],
                )
                down_acc = pl.matmul_acc(
                    down_acc, mlp_chunk_bf16, w_down_chunk,
                )
            partial_hidden = pl.assemble(
                partial_hidden,
                pl.cast(down_acc, target_type=pl.BF16),
                [t0, d0],
            )

    # ── Step 4: TP all-reduce. ────────────────────────────────────────
    # Resolves to the enclosing @pl.program's tp_all_reduce method (same
    # pattern as decode-side dense_mlp.py). At TP=1 the all-reduce is a
    # no-op; skip the call so orchestration codegen does not emit a stale
    # SSA rename for the (now-empty) ring body.
    if tp_size > 1:
        self.tp_all_reduce(
            partial_hidden, tmp_window, signal_window, my_rank,
        )

    # Module dump 7a: ffn_out.ffn_output (dense MLP output, pre-residual-add).
    if _DUMP_ENABLED:
        ffn_output_dump = pl.assemble(ffn_output_dump, partial_hidden, [0, 0])

    # ── Step 5: residual add. ─────────────────────────────────────────
    with pl.at(
        level=pl.Level.CORE_GROUP, name_hint="prefill_dense_residual_add_tp",
    ):
        for tg5 in pl.range(PREFILL_TILE_COUNT):
            t0 = tg5 * BATCH
            for kb4 in pl.range(hidden_blocks):
                k0 = kb4 * K_CHUNK
                mlp_reduced = pl.cast(
                    pl.slice(partial_hidden, [BATCH, K_CHUNK], [t0, k0]),
                    target_type=pl.FP32,
                )
                r = pl.slice(resid1_fp32, [BATCH, K_CHUNK], [t0, k0])
                next_hidden = pl.assemble(
                    next_hidden,
                    pl.cast(pl.add(r, mlp_reduced), target_type=pl.BF16),
                    [t0, k0],
                )

    return next_hidden


# =============================================================================
# TP wrapper — @pl.program (chip_orch + host_orch).
# =============================================================================
def _build_tp_prefill_dense_mlp_program(tp_size: int = TP_WORLD_SIZE):
    """Return a freshly-built ``@pl.program`` for the prefill dense-MLP body.

    Mirrors ``_build_tp_prefill_attention_*_program``: the ``@pl.program``
    supplies ``self.tp_all_reduce`` (bound to ``chip_orch``'s ``self`` when
    the body is inlined); at ``tp_size=1`` the ring body is a no-op and the
    body's ``if tp_size > 1:`` guard skips the call entirely.
    """
    if HIDDEN % tp_size != 0:
        raise ValueError(
            f"HIDDEN={HIDDEN} must be divisible by tp_size={tp_size}"
        )
    body_inline = pl.inline(prefill_dense_mlp_body._func)

    @pl.program
    class PrefillDenseMlp:
        # ---------- Collective: TP all_reduce (InCore). ----
        # Barrier-style body (mirrors decode moe.py:270-313, PASS TP=8);
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
                        pl.cast(acc, target_type=pl.BF16), [ttr, k0], local,
                    )
            return local

        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(
            self,
            resid1: pl.Tensor[[PREFILL_T, HIDDEN], pl.BF16],
            post_rms_weight: pl.Tensor[[LAYER_DYN, HIDDEN], pl.FP32],
            w_gate: pl.Tensor[
                [LAYER_HIDDEN_ROWS_DYN, INTER_LOCAL], pl.BF16
            ],
            w_up: pl.Tensor[
                [LAYER_HIDDEN_ROWS_DYN, INTER_LOCAL], pl.BF16
            ],
            w_down: pl.Tensor[
                [LAYER_INTER_ROWS_DYN, HIDDEN], pl.BF16
            ],
            next_hidden: pl.Out[
                pl.Tensor[[PREFILL_T, HIDDEN], pl.BF16]
            ],
            tmp_window: pld.DistributedTensor[
                [PREFILL_T, HIDDEN], pl.BF16
            ],
            signal_window: pld.DistributedTensor[
                [tp_size, 1], pl.INT32
            ],
            norm_layer_idx: pl.Scalar[pl.INT32],
            mlp_layer_idx: pl.Scalar[pl.INT32],
            my_rank: pl.Scalar[pl.INT32],
        ):
            _dummy_post_norm = pl.create_tensor(
                [PREFILL_T, HIDDEN], dtype=pl.BF16,
            )
            _dummy_ffn = pl.create_tensor(
                [PREFILL_T, HIDDEN], dtype=pl.BF16,
            )
            next_hidden = body_inline(
                resid1, post_rms_weight,
                w_gate, w_up, w_down,
                next_hidden,
                _dummy_post_norm, _dummy_ffn,
                norm_layer_idx, mlp_layer_idx,
                tmp_window, signal_window, my_rank,
            )
            return next_hidden

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(
            self,
            resid1: pl.Tensor[
                [tp_size, PREFILL_T, HIDDEN], pl.BF16
            ],
            post_rms_weight: pl.Tensor[
                [tp_size, LAYER_DYN, HIDDEN], pl.FP32
            ],
            w_gate: pl.Tensor[
                [tp_size, LAYER_HIDDEN_ROWS_DYN, INTER_LOCAL], pl.BF16
            ],
            w_up: pl.Tensor[
                [tp_size, LAYER_HIDDEN_ROWS_DYN, INTER_LOCAL], pl.BF16
            ],
            w_down: pl.Tensor[
                [tp_size, LAYER_INTER_ROWS_DYN, HIDDEN], pl.BF16
            ],
            next_hidden: pl.Out[
                pl.Tensor[[tp_size, PREFILL_T, HIDDEN], pl.BF16]
            ],
            norm_layer_idx: pl.Scalar[pl.INT32],
            mlp_layer_idx: pl.Scalar[pl.INT32],
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
                    resid1[r],
                    post_rms_weight[r],
                    w_gate[r], w_up[r], w_down[r],
                    next_hidden[r],
                    tmp_window, signal_window,
                    norm_layer_idx,
                    mlp_layer_idx,
                    r,
                    device=r,
                )

    return PrefillDenseMlp


def _build_tp_prefill_dense_mlp_program_default():
    return _build_tp_prefill_dense_mlp_program(TP_WORLD_SIZE)


def golden_prefill_dense_mlp(tensors):
    """Torch reference: zero-centred RMSNorm + TP-sliced SwiGLU + down + residual.

    Mirrors the decode-side ``golden_dense_mlp`` math but operates on the full
    ``[PREFILL_T, HIDDEN]`` residual. The TP all-reduce is modelled as a
    sum-across-ranks (each rank holds INTER_LOCAL lanes of w_gate/w_up and the
    matching row slab of w_down). Single-rank (TP_WORLD_SIZE=1) falls through
    with no reduction.
    """
    import torch

    # Scratch tensors carry a leading rank dim ([1, ...]); index [0].
    resid1 = tensors["resid1"][0].float()                       # [PREFILL_T, HIDDEN]
    post_rms_weight = tensors["post_rms_weight"][0].float()     # [LAYER_DYN, HIDDEN]
    w_gate = tensors["w_gate"][0].float()                       # [LAYER_HIDDEN_ROWS, INTER_LOCAL]
    w_up = tensors["w_up"][0].float()                           # [LAYER_HIDDEN_ROWS, INTER_LOCAL]
    w_down = tensors["w_down"][0].float()                       # [LAYER_INTER_ROWS, HIDDEN]
    norm_layer_idx = int(tensors["norm_layer_idx"])
    mlp_layer_idx = int(tensors["mlp_layer_idx"])

    layer_hidden_base = mlp_layer_idx * HIDDEN
    layer_inter_base = mlp_layer_idx * INTER_LOCAL

    # ── Step 1: zero-centred RMSNorm ──────────────────────────────────
    # gamma_eff = stored_gamma + 1.0
    gamma_row = post_rms_weight[norm_layer_idx] + 1.0       # [HIDDEN]
    ms = resid1.pow(2).mean(dim=-1, keepdim=True)           # [PREFILL_T, 1]
    inv_rms = torch.rsqrt(ms + EPS)                         # [PREFILL_T, 1]
    post_norm = resid1 * inv_rms * gamma_row.unsqueeze(0)   # [PREFILL_T, HIDDEN]

    # ── Step 2: TP-sliced gate_up + SiLU ───────────────────────────────
    wg = w_gate[layer_hidden_base : layer_hidden_base + HIDDEN]   # [HIDDEN, INTER_LOCAL]
    wu = w_up[layer_hidden_base : layer_hidden_base + HIDDEN]     # [HIDDEN, INTER_LOCAL]
    gate_logits = post_norm @ wg                                  # [PREFILL_T, INTER_LOCAL]
    up_logits = post_norm @ wu                                    # [PREFILL_T, INTER_LOCAL]
    sigmoid = torch.sigmoid(gate_logits)
    inter = gate_logits * sigmoid * up_logits                     # SwiGLU/SiLU variant

    # ── Step 3: TP-sliced w_down -> partial ───────────────────────────
    wd = w_down[layer_inter_base : layer_inter_base + INTER_LOCAL]  # [INTER_LOCAL, HIDDEN]
    partial = inter @ wd                                            # [PREFILL_T, HIDDEN]

    # ── Step 4: TP all-reduce (sum across ranks) ───────────────────────
    # Each rank's partial is summed. Under TP=8 each rank holds a different
    # slice; the reference here is single-rank (mock), so we just pass through.
    # Real cross-rank validation happens in the L1 distributed-mock harness.
    reduced = partial

    # ── Step 5: residual add ───────────────────────────────────────────
    next_hidden = resid1 + reduced

    tensors["next_hidden"][0][:] = next_hidden.to(torch.bfloat16)


def build_tensor_specs(norm_layer_idx: int = 0, mlp_layer_idx: int = 0):
    """Synthetic tensor specs for L0 golden test of prefill_dense_mlp_body.

    Single-rank (TP_WORLD_SIZE=1) synthetic test: shapes carry a leading
    rank dim of 1 (the ``@pl.program`` host_orch convention for
    ``golden.runner.run``). w_gate/w_up/w_down carry only this rank's slice
    (INTER_LOCAL lanes). Real-ckpt TP-sliced specs are produced by
    ``weight_loader.py`` for L1/L2 testing.
    """
    import torch
    from golden import ScalarSpec, TensorSpec

    torch.manual_seed(0)

    def init_resid1():
        return (torch.randn(PREFILL_T, HIDDEN) * 0.5).bfloat16().unsqueeze(0)

    def init_post_rms_weight():
        return (torch.randn(LAYER_DYN, HIDDEN) * 0.1).float().unsqueeze(0)

    def init_w_gate():
        return (
            torch.randn(LAYER_HIDDEN_ROWS_DYN, INTER_LOCAL) / (HIDDEN ** 0.5)
        ).bfloat16().unsqueeze(0)

    def init_w_up():
        return (
            torch.randn(LAYER_HIDDEN_ROWS_DYN, INTER_LOCAL) / (HIDDEN ** 0.5)
        ).bfloat16().unsqueeze(0)

    def init_w_down():
        return (
            torch.randn(LAYER_INTER_ROWS_DYN, HIDDEN) / (INTER_LOCAL ** 0.5)
        ).bfloat16().unsqueeze(0)

    return [
        TensorSpec("resid1", [1, PREFILL_T, HIDDEN], torch.bfloat16, init_value=init_resid1),
        TensorSpec(
            "post_rms_weight", [1, LAYER_DYN, HIDDEN], torch.float32,
            init_value=init_post_rms_weight,
        ),
        TensorSpec(
            "w_gate", [1, LAYER_HIDDEN_ROWS_DYN, INTER_LOCAL], torch.bfloat16,
            init_value=init_w_gate,
        ),
        TensorSpec(
            "w_up", [1, LAYER_HIDDEN_ROWS_DYN, INTER_LOCAL], torch.bfloat16,
            init_value=init_w_up,
        ),
        TensorSpec(
            "w_down", [1, LAYER_INTER_ROWS_DYN, HIDDEN], torch.bfloat16,
            init_value=init_w_down,
        ),
        TensorSpec("next_hidden", [1, PREFILL_T, HIDDEN], torch.bfloat16, is_output=True),
        ScalarSpec("norm_layer_idx", torch.int32, norm_layer_idx),
        ScalarSpec("mlp_layer_idx", torch.int32, mlp_layer_idx),
    ]


def _run_tp1_golden(
    *, platform: str = "a2a3", device: int = 4,
    norm_layer_idx: int = 0, mlp_layer_idx: int = 0,
    atol: float = 0.05, rtol: float = 0.05, max_error_ratio: float = 0.05,
    compile_only: bool = False,
):
    """TP=1 NPU body verification on device ``device`` via golden.runner.run.

    Builds ``_build_tp_prefill_dense_mlp_program(tp_size=1)`` (the
    ``@pl.program`` supplies ``self.tp_all_reduce``; at TP=1 the ring body
    is a no-op), runs it on one card, and validates ``next_hidden`` against
    :func:`golden_prefill_dense_mlp` with ``ratio_allclose`` (5% cap,
    atol=rtol=0.05).
    """
    from golden.runner import run
    from golden.validation import ratio_allclose
    from pypto.ir.distributed_compiled_program import DistributedConfig

    program = _build_tp_prefill_dense_mlp_program(tp_size=1)
    specs = build_tensor_specs(
        norm_layer_idx=norm_layer_idx, mlp_layer_idx=mlp_layer_idx,
    )
    compile_cfg = {
        "distributed_config": DistributedConfig(
            device_ids=[device], num_sub_workers=0,
        ),
    }
    runtime_cfg = dict(platform=platform, device_id=device)
    compare_fn = {
        "next_hidden": ratio_allclose(
            atol=atol, rtol=rtol, max_error_ratio=max_error_ratio,
        ),
    }
    return run(
        program=program,
        specs=specs,
        golden_fn=golden_prefill_dense_mlp,
        compile_cfg=compile_cfg,
        runtime_cfg=runtime_cfg,
        rtol=rtol,
        atol=atol,
        compare_fn=compare_fn,
        compile_only=compile_only,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Step3p5 prefill dense-MLP body: TP=1 NPU golden verification "
            "(L0). Compiles the body via "
            "_build_tp_prefill_dense_mlp_program(tp_size=1) and validates "
            "next_hidden against golden_prefill_dense_mlp with "
            "ratio_allclose (5% cap, atol=rtol=0.05)."
        ),
    )
    parser.add_argument(
        "-p", "--platform", default="a2a3",
        choices=["a2a3"],
        help="Real-device platform only (sim forbidden).",
    )
    parser.add_argument("-d", "--device", type=int, default=4)
    parser.add_argument("--norm-layer-idx", type=int, default=0)
    parser.add_argument("--mlp-layer-idx", type=int, default=0)
    parser.add_argument("--atol", type=float, default=0.05)
    parser.add_argument("--rtol", type=float, default=0.05)
    parser.add_argument("--max-error-ratio", type=float, default=0.05)
    parser.add_argument(
        "--compile-only", action="store_true",
        help="Stop after codegen (no execute / no validate).",
    )
    args = parser.parse_args()

    res = _run_tp1_golden(
        platform=args.platform,
        device=args.device,
        norm_layer_idx=args.norm_layer_idx,
        mlp_layer_idx=args.mlp_layer_idx,
        atol=args.atol,
        rtol=args.rtol,
        max_error_ratio=args.max_error_ratio,
        compile_only=args.compile_only,
    )
    print(f"[prefill_dense_mlp] TP=1 NPU golden: {res}", flush=True)
    raise SystemExit(0 if res.passed else 1)


__all__ = [
    "prefill_dense_mlp_body",
    "_build_tp_prefill_dense_mlp_program",
    "_build_tp_prefill_dense_mlp_program_default",
    "golden_prefill_dense_mlp",
    "build_tensor_specs",
    "_run_tp1_golden",
    "PREFILL_T",
    "PREFILL_TILE_COUNT",
    "INTER_LOCAL",
]
