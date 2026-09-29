# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Prefill-side gate body — Step3.5 router (sigmoid + bias + top-K + renorm).

v2 lift (task #13): the ``_gate`` method body from ``prefill_moe.py:387-486``
is lifted here as a module-level ``@pl.jit.inline`` body using prefill-side
names (BATCH, ROUTER_SCORE_PAD, ROUTER_GATE_K_CHUNK, ROUTER_FP32_NEG_INF,
ROUTER_SCALE, ROUTER_TOPK_PAD, ROUTER_SORT_PAD, N_EXPERTS, TOPK).

BUG FIX (team-lead directive): the mrgsort call is lifted as the CANONICAL
form1 pattern (``mrgsort(srt, block_len=256)``), NOT the buggy form2
two-slice (``mrgsort(srt[:, 0:512], srt[:, 512:1024])``) that was at
prefill_moe.py:453. Per task #12 finding, form2 violates the format2
contract (each src must be a single sorted run; with ROUTER_SCORE_PAD=512
each 512-position slice contains 2 sorted runs of 256). Pure-Python
emulation shows form1 matches torch.argsort(stable=True) on all test
cases; form2 diverges on 3 of 4. See ``_tmp_mrgsort_emulation.py``.

The body is called from ``prefill_moe.py``'s ``gate_step`` wrapper and
from ``prefill_fwd.py``'s MoE layer builder via
``pl.inline(prefill_gate_body._func)`` (bound once at factory build time,
called by name inside the @pl.function body — the frontend rejects
``pl.inline(body._func)(args)`` written inline).
"""

from __future__ import annotations

import pypto.language as pl

from ._moe_constants import (
    BATCH,
    HIDDEN,
    LAYER_DYN,
    N_EXPERTS,
    ROUTER_FP32_NEG_INF,
    ROUTER_GATE_K_CHUNK,
    ROUTER_SCALE,
    ROUTER_SCORE_PAD,
    ROUTER_SORT_PAD,
    ROUTER_TOPK_PAD,
    TOPK,
)

__all__ = [
    "prefill_gate_body",
    "ROUTER_SCORE_PAD",
    "ROUTER_TOPK_PAD",
    "ROUTER_SORT_PAD",
    "ROUTER_GATE_K_CHUNK",
    "ROUTER_FP32_NEG_INF",
    "ROUTER_SCALE",
]


@pl.jit.inline
def prefill_gate_body(
    resid: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
    post_rms_weight: pl.Tensor[[LAYER_DYN, HIDDEN], pl.FP32],
    norm_layer_idx: pl.Scalar[pl.INT32],
    inv_rms: pl.Tensor[[BATCH, 1], pl.FP32],
    gate_w: pl.Tensor[[HIDDEN, N_EXPERTS], pl.FP32],
    router_bias: pl.Tensor[[N_EXPERTS], pl.FP32],
    expert_indices: pl.Tensor[[BATCH, TOPK], pl.INT32],
    expert_weights: pl.Tensor[[BATCH, TOPK], pl.FP32],
):
    """Step3.5 prefill router: sigmoid + additive bias + top-K + renorm + scale.

    Replicated body — same compute on every rank, identical output.

    T2a W8A8 (task #11 item 5): the gate now consumes the RAW residual and
    recomputes ``xg = resid * (gamma + 1)`` chunk-wise (mirroring the routed
    lane's producer, ``decode_fwd.py:593-596``), applying the shared
    ``inv_rms`` from the upstream norm/quant producer AFTER the FP32 matmul
    (``decode_fwd.py:622-630``). This keeps the gate's logits byte-identical
    to the V4 deferred-RMSNorm math while sharing one ``inv_rms`` with the
    routed lane (computed from ``sum(resid**2)``, not from xg).

    Pipeline (per row, fused into one InCore region):
      1. ``logits = xg @ gate_w`` (FP32 accumulator over HIDDEN).
      2. ``score = sigmoid(logits * inv_rms)`` (un-biased; renorm gather).
      3. ``biased = score + router_bias[None, :]`` (selection key).
      4. Pad biased to [1, ROUTER_SCORE_PAD] with -inf, sort32 + mrgsort
         cascade.
      5. Gather the top-K **global** indices, look up un-biased scores.
      6. Renormalise to sum=1, multiply by ``ROUTER_SCALE``, cast to BF16.

    The un-biased score (not the bias-shifted one) is what's renormalised
    into the per-expert routing weight, matching vllm's ``router_bias_func``
    (the bias only steers selection; the weighting uses the raw sigmoid).
    """
    score_buf = pl.create_tensor(
        [BATCH, ROUTER_SCORE_PAD], dtype=pl.FP32,
    )
    biased_buf = pl.create_tensor(
        [BATCH, ROUTER_SCORE_PAD], dtype=pl.FP32,
    )

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="gate_matmul"):
        # Slice-then-cast (not cast-then-slice): casting the full resid to
        # FP32 would allocate [BATCH, HIDDEN] FP32 = 256KB and overflow
        # the 184KB UB. Slicing BF16 first, then casting each tile,
        # keeps the FP32 tile at [BATCH, ROUTER_GATE_K_CHUNK] = 16KB.
        # Math unchanged — still FP32 accumulation over full HIDDEN.
        x0_bf = pl.slice(resid, [BATCH, ROUTER_GATE_K_CHUNK], [0, 0])
        x0_raw = pl.cast(x0_bf, target_type=pl.FP32)
        gamma0 = pl.slice(
            post_rms_weight, [1, ROUTER_GATE_K_CHUNK],
            [norm_layer_idx, 0],
        )
        x0 = pl.col_expand_mul(x0_raw, pl.add(gamma0, 1.0))
        # Apply inv_rms BEFORE matmul with BF16 round-trip to match VLLM's
        # standard RMSNorm precision (vLLM applies inv_rms in BF16 before
        # the matmul; deferred application in FP32 after matmul causes
        # ~0.026 gate-score differences that flip near-tie top-8 selections).
        x0_scaled = pl.row_expand_mul(x0, inv_rms)
        x0_bf16 = pl.cast(
            pl.cast(x0_scaled, target_type=pl.BF16),
            target_type=pl.FP32,
        )
        w0 = pl.slice(
            gate_w, [ROUTER_GATE_K_CHUNK, N_EXPERTS], [0, 0],
        )
        logits = pl.matmul(x0_bf16, w0, out_dtype=pl.FP32)
        for kb in pl.range(1, HIDDEN // ROUTER_GATE_K_CHUNK):
            k0 = kb * ROUTER_GATE_K_CHUNK
            xk_bf = pl.slice(
                resid, [BATCH, ROUTER_GATE_K_CHUNK], [0, k0],
            )
            xk_raw = pl.cast(xk_bf, target_type=pl.FP32)
            gammak = pl.slice(
                post_rms_weight, [1, ROUTER_GATE_K_CHUNK],
                [norm_layer_idx, k0],
            )
            xk = pl.col_expand_mul(xk_raw, pl.add(gammak, 1.0))
            xk_scaled = pl.row_expand_mul(xk, inv_rms)
            xk_bf16 = pl.cast(
                pl.cast(xk_scaled, target_type=pl.BF16),
                target_type=pl.FP32,
            )
            wk = pl.slice(
                gate_w, [ROUTER_GATE_K_CHUNK, N_EXPERTS], [k0, 0],
            )
            logits = pl.matmul_acc(logits, xk_bf16, wk)

        # GATE-PRECISION (vLLM-aligned): inv_rms is applied BEFORE the matmul
        # (above) with BF16 round-trip, matching vLLM's standard RMSNorm
        # precision.  The former deferred path (apply inv_rms AFTER the FP32
        # matmul) produced FP32-precision gate scores that differed from
        # vLLM's BF16 path by up to ~0.026, flipping near-tie top-8 selections.
        score_n = pl.recip(pl.add(pl.exp(pl.neg(logits)), 1.0))
        # NaN DIAG: raw sigmoid score — NaN here means upstream resid is bad.
        pl.dump_tag(score_n)
        bias_row = pl.reshape(router_bias, [1, N_EXPERTS])
        # ROUTER-BIAS-BF16 (align decode_fwd.py:657-664 / moe.py:485-490):
        # vLLM runs router_bias in BF16; the FP32 loader value's ~0.015
        # rounding decides the top-8 tail. Without the BF16 round-trip the
        # whole-net gate picks a different top-8 vs vLLM/decode.
        bias_row = pl.cast(
            pl.cast(bias_row, target_type=pl.BF16),
            target_type=pl.FP32,
        )
        biased_n = pl.add(
            score_n,
            pl.col_expand_mul(
                pl.full(
                    [BATCH, N_EXPERTS], dtype=pl.FP32, value=1.0,
                ),
                bias_row,
            ),
        )

        score_buf[:, :] = pl.full(
            [BATCH, ROUTER_SCORE_PAD], dtype=pl.FP32, value=0.0,
        )
        biased_buf[:, :] = pl.full(
            [BATCH, ROUTER_SCORE_PAD],
            dtype=pl.FP32, value=ROUTER_FP32_NEG_INF,
        )
        score_buf[:, 0:N_EXPERTS] = score_n
        biased_buf[:, 0:N_EXPERTS] = biased_n

    # NaN DIAG: dump the sigmoid scores OUTSIDE the CORE_GROUP (a dump_tag
    # inside pl.at(CORE_GROUP) is dropped by the dump infra). score_buf is
    # created outside and read by the renorm gather, so it survives DCE.
    # score == 0.0 exactly => logits < ~-88 (exp overflow in sigmoid);
    # score == 1.0 exactly => logits > ~+88. NaN => upstream resid is NaN.
    pl.dump_tag(score_buf)

    # --- Stage 2: per-row top-K via sort32 + mrgsort ------------------
    # pl.parallel (outer) + pl.range (inner) mirrors the canonical
    # pattern in examples/advanced/topk.py:33-44. The pl.range-only
    # scalar loop (decode-side gate.py:174) does not populate
    # topk_idx_tile on a2a3sim — the slice assignment silently no-ops.
    topk_idx_tile = pl.create_tensor(
        [BATCH, ROUTER_TOPK_PAD], dtype=pl.INT32,
    )
    GATE_ROW_TILE = 8  # rows per parallel tile (mirrors topk.py:ROW_TILE)
    for t0 in pl.parallel(0, BATCH, GATE_ROW_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="gate_topk_sort"):
            idx_init = pl.arange(
                0, [1, ROUTER_SCORE_PAD], dtype=pl.UINT32,
            )
            for ti in pl.range(GATE_ROW_TILE):
                tt = t0 + ti
                row = biased_buf[tt : tt + 1, :]
                srt = pl.sort32(row, idx_init)
                srt = pl.mrgsort(srt, block_len=64)
                # form1 canonical mrgsort (block_len=256) — 4 runs of 256
                # positions merge into 1 sorted run of 1024. This FIXES the
                # latent bug in prefill_moe.py:453 (form2 two-slice:
                # ``pl.mrgsort(srt[:, 0:512], srt[:, 512:1024])``), which
                # violates the format2 contract (each src must be a SINGLE
                # sorted run, but each 512-position slice contains 2 sorted
                # runs of 256). Per pure-Python emulation (task #12),
                # form1 matches torch.argsort(stable=True) on all tied-score
                # and no-ties test cases; form2 diverges on 3 of 4 cases.
                # See task #12 finding in _tmp_mrgsort_emulation.py.
                srt = pl.mrgsort(srt, block_len=256)
                pairs = srt[:, 0:ROUTER_SORT_PAD]
                top_idx = pl.gather(
                    pairs, mask_pattern=pl.tile.MaskPattern.P1010,
                    output_dtype=pl.INT32,
                )
                topk_idx_tile[tt : tt + 1, :] = top_idx
    # NaN DIAG: top-K expert indices — >= N_EXPERTS means sort picked padding.
    pl.dump_tag(topk_idx_tile)

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="gate_renorm"):
        gather_all = pl.gather(
            score_buf, dim=-1, index=topk_idx_tile,
        )
        gather_valid = pl.set_validshape(gather_all, BATCH, TOPK)
        topk_vals_pad = pl.fillpad(
            gather_valid, pad_value=pl.PadValue.zero,
        )

        denom = pl.reshape(pl.row_sum(topk_vals_pad), [BATCH, 1])
        # NaN DIAG: renorm denominator — 0 here means denom=0 → 0/0 NaN.
        pl.dump_tag(denom)
        weights_pad = pl.mul(
            pl.row_expand_div(topk_vals_pad, denom),
            ROUTER_SCALE,
        )

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="gate_scatter"):
        for tt in pl.range(BATCH):
            for k in pl.range(TOPK):
                pl.write(
                    expert_indices, [tt, k],
                    pl.read(topk_idx_tile, [tt, k]),
                )
                pl.write(
                    expert_weights, [tt, k],
                    pl.read(weights_pad, [tt, k]),
                )

    return expert_weights


# =============================================================================
# L0 golden test wrapper (mirrors decode-side gate.py:219-300 pattern).
#
# The @pl.jit.inline body cannot be compiled/run standalone via
# golden.runner.run_jit (the inline decorator's compile path produces no
# runnable artifact). This @pl.jit wrapper calls the inline body by name;
# the frontend auto-inlines the body into the wrapper at compile time.
# =============================================================================
@pl.jit
def prefill_gate_test(
    resid: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
    post_rms_weight: pl.Tensor[[LAYER_DYN, HIDDEN], pl.FP32],
    norm_layer_idx: pl.Scalar[pl.INT32],
    inv_rms: pl.Tensor[[BATCH, 1], pl.FP32],
    gate_w: pl.Tensor[[HIDDEN, N_EXPERTS], pl.FP32],
    router_bias: pl.Tensor[[N_EXPERTS], pl.FP32],
    expert_indices: pl.Out[pl.Tensor[[BATCH, TOPK], pl.INT32]],
    expert_weights: pl.Out[pl.Tensor[[BATCH, TOPK], pl.FP32]],
):
    prefill_gate_body(
        resid, post_rms_weight, norm_layer_idx, inv_rms,
        gate_w, router_bias, expert_indices, expert_weights,
    )
    return expert_indices, expert_weights


def golden_prefill_gate(tensors):
    """Torch reference: sigmoid + additive bias + topk + renorm + scale.

    Uses argsort(stable=True) to match the deterministic NPU sort32 +
    mrgsort form1 ordering (form1 matches torch stable argsort on all
    tied-score and no-ties cases per _tmp_mrgsort_emulation.py).
    """
    import torch

    resid = tensors["resid"].float()                    # [BATCH, HIDDEN]
    post_rms_weight = tensors["post_rms_weight"].float()  # [LAYER_DYN, HIDDEN]
    norm_layer_idx = int(tensors["norm_layer_idx"])
    inv_rms = tensors["inv_rms"].float()                 # [BATCH, 1]
    gate_w = tensors["gate_w"].float()                   # [HIDDEN, N_EXPERTS]
    router_bias = tensors["router_bias"].float()         # [N_EXPERTS]
    # ROUTER-BIAS-BF16 (align decode_fwd.py:657-664): vLLM runs router_bias
    # in BF16; round-trip the FP32 loader value so the golden matches the
    # kernel's cast(cast(bias, BF16), FP32) top-8 selection.
    router_bias = router_bias.to(torch.bfloat16).float()

    gamma = post_rms_weight[norm_layer_idx]              # [HIDDEN]
    xg = resid * (gamma + 1.0)                           # [BATCH, HIDDEN]
    logits = xg @ gate_w                                 # [BATCH, N_EXPERTS]
    score = torch.sigmoid(logits * inv_rms)              # raw sigmoid score
    biased = score + router_bias.view(1, -1)             # selection key

    indices = torch.argsort(-biased, dim=-1, stable=True)[:, :TOPK]
    topk_vals = torch.gather(score, dim=-1, index=indices.long())
    weights = (topk_vals / topk_vals.sum(dim=-1, keepdim=True)) * ROUTER_SCALE

    tensors["expert_indices"][:] = indices.to(torch.int32)
    tensors["expert_weights"][:] = weights.to(torch.float32)


def build_tensor_specs():
    import torch
    from golden import ScalarSpec, TensorSpec

    def init_resid():
        return torch.randn(BATCH, HIDDEN) * 0.5

    def init_post_rms_weight():
        # post_rms_weight stores (gamma - 1) so that xg = resid*(gamma+1)
        # recovers the standard RMSNorm gamma. Small values keep the
        # reference well-conditioned.
        return torch.randn(LAYER_DYN, HIDDEN) * 0.05

    def init_inv_rms():
        return torch.rand(BATCH, 1) * 0.5 + 0.5

    def init_gate_w():
        return torch.randn(HIDDEN, N_EXPERTS) / HIDDEN ** 0.5

    def init_router_bias():
        return torch.randn(N_EXPERTS) * 0.05

    return [
        TensorSpec(
            "resid", [BATCH, HIDDEN], torch.bfloat16, init_value=init_resid,
        ),
        TensorSpec(
            "post_rms_weight", [LAYER_DYN, HIDDEN], torch.float32,
            init_value=init_post_rms_weight,
        ),
        ScalarSpec("norm_layer_idx", torch.int32, 0),
        TensorSpec(
            "inv_rms", [BATCH, 1], torch.float32, init_value=init_inv_rms,
        ),
        TensorSpec(
            "gate_w", [HIDDEN, N_EXPERTS], torch.float32,
            init_value=init_gate_w,
        ),
        TensorSpec(
            "router_bias", [N_EXPERTS], torch.float32,
            init_value=init_router_bias,
        ),
        TensorSpec(
            "expert_indices", [BATCH, TOPK], torch.int32, is_output=True,
        ),
        TensorSpec(
            "expert_weights", [BATCH, TOPK], torch.float32, is_output=True,
        ),
    ]
