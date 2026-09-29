# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: Apache-2.0
"""Card-free cross-authority check: prefill gate vs decode (V4) gate.

The prefill router was rewritten (T2a task #17) to mirror the decode-side
V4 deferred-RMSNorm gate in ``decode_fwd.py`` (``_gate``, :584-680): it
consumes the RAW residual, recomputes ``xg = resid * (gamma + 1)``, runs the
FP32 matmul, then applies the shared ``inv_rms`` AFTER the matmul, sigmoid,
additive router bias, top-K and renorm.

Both authority sources (``decode_fwd.py`` and ``prefill_gate.py``) import
``pypto.language`` at module scope, which loads the ``pypto_core`` C
extension.  To stay card-free (no device, no C-extension import — matching
the ``test_five_layer_moe_contract.py`` convention of reading source text
rather than importing), this test:

  * re-implements both references as pure-torch transcriptions and asserts
    they produce byte-identical ``topk_idx`` / ``topk_w`` on the same inputs;
  * asserts, against the actual source files, that the three ROUTER-BIAS-BF16
    round-trip sites are all present (decode_fwd.py:657-664, prefill_gate.py
    kernel body + golden, prefill_fwd.py ``golden_whole_moe`` ``gate_fn``).

The load-bearing detail is the ROUTER-BIAS-BF16 round-trip: vLLM runs
``router_bias`` in BF16, and the ~0.015 rounding of the FP32 loader value
decides the top-8 tail.  Dropping it anywhere makes the routing diverge from
decode/vLLM.

Run::

    python -m tests.step3p5.unit.test_gate_cross_authority
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# Constant values transcribed from models/step3p5/config.py (which itself
# imports pypto.language, so we do NOT import it here).  config.py:154
# HIDDEN=4096, :189 MOE_NUM_EXPERTS=288, :190 MOE_TOP_K=8, :193
# MOE_ROUTER_SCALING_FACTOR=3.0, :361 BATCH=STORAGE_BATCH_CAPACITY=16,
# :142/:84 LAYER_DYN=KV_NUM_LAYERS=45.  Only the first LAYER_DYN row is read
# (norm_layer_idx=0), so the reference materializes a single row.
BATCH = 16
HIDDEN = 4096
N_EXPERTS = 288
TOPK = 8
ROUTER_SCALE = 3.0


def _decode_v4_gate_reference(
    resid: torch.Tensor,
    post_rms_weight: torch.Tensor,
    inv_rms: torch.Tensor,
    gate_w: torch.Tensor,
    router_bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode-side V4 deferred-RMSNorm gate, transcribed from decode_fwd.py
    ``_gate`` (:584-680).

      1. ``xg = resid * (gamma + 1)`` (decode_fwd.py:593-596).
      2. ``logits = xg @ gate_w`` (FP32, decode_fwd.py:602-621).
      3. ``logits_scaled = logits * inv_rms`` (deferred, decode_fwd.py:642-647).
      4. ``score = sigmoid(logits_scaled)`` (decode_fwd.py:648-650).
      5. ``biased = score + cast(cast(router_bias, BF16), FP32)``
         (decode_fwd.py:657-674).
      6. ``topk`` via ``argsort(-biased, stable=True)``, gather raw ``score``,
         renorm to sum=1, ``* ROUTER_SCALE``.
    """
    gamma = post_rms_weight[0].float()                   # [HIDDEN]
    xg = resid.float() * (gamma + 1.0)                   # [BATCH, HIDDEN]
    logits = xg @ gate_w.float()                         # [BATCH, N_EXPERTS]
    logits_scaled = logits * inv_rms.float()             # [BATCH, N_EXPERTS]
    score = torch.sigmoid(logits_scaled)
    bias = router_bias.float().to(torch.bfloat16).float()  # BF16 round-trip
    biased = score + bias.view(1, -1)
    indices = torch.argsort(-biased, dim=-1, stable=True)[:, :TOPK]
    topk_vals = torch.gather(score, dim=-1, index=indices.long())
    weights = (topk_vals / topk_vals.sum(dim=-1, keepdim=True)) * ROUTER_SCALE
    return indices.to(torch.int32), weights


def _prefill_gate_reference(
    resid: torch.Tensor,
    post_rms_weight: torch.Tensor,
    inv_rms: torch.Tensor,
    gate_w: torch.Tensor,
    router_bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prefill-side gate reference, transcribed from ``golden_prefill_gate``
    (prefill_gate.py:264-295).  Must be math-identical to the decode side.
    """
    gamma = post_rms_weight[0].float()                   # [HIDDEN]
    xg = resid.float() * (gamma + 1.0)                   # [BATCH, HIDDEN]
    logits = xg @ gate_w.float()                         # [BATCH, N_EXPERTS]
    score = torch.sigmoid(logits * inv_rms.float())      # raw sigmoid score
    bias = router_bias.float().to(torch.bfloat16).float()  # BF16 round-trip
    biased = score + bias.view(1, -1)
    indices = torch.argsort(-biased, dim=-1, stable=True)[:, :TOPK]
    topk_vals = torch.gather(score, dim=-1, index=indices.long())
    weights = (topk_vals / topk_vals.sum(dim=-1, keepdim=True)) * ROUTER_SCALE
    return indices.to(torch.int32), weights


def _make_inputs(seed: int) -> dict:
    torch.manual_seed(seed)
    return {
        "resid": torch.randn(BATCH, HIDDEN) * 0.5,
        "post_rms_weight": torch.randn(1, HIDDEN) * 0.05,
        "inv_rms": torch.rand(BATCH, 1) * 0.5 + 0.5,
        "gate_w": torch.randn(HIDDEN, N_EXPERTS) / HIDDEN ** 0.5,
        "router_bias": torch.randn(N_EXPERTS) * 0.05,
    }


class TestGateCrossAuthority(unittest.TestCase):
    def test_prefill_gate_matches_decode_v4_gate(self) -> None:
        for seed in (0, 1, 2):
            inp = _make_inputs(seed)
            dec_idx, dec_w = _decode_v4_gate_reference(
                inp["resid"], inp["post_rms_weight"], inp["inv_rms"],
                inp["gate_w"], inp["router_bias"],
            )
            pre_idx, pre_w = _prefill_gate_reference(
                inp["resid"], inp["post_rms_weight"], inp["inv_rms"],
                inp["gate_w"], inp["router_bias"],
            )
            self.assertEqual(pre_idx.dtype, torch.int32)
            self.assertTrue(torch.equal(pre_idx, dec_idx))
            self.assertTrue(torch.equal(pre_w, dec_w))

    def test_bf16_bias_roundtrip_is_load_bearing(self) -> None:
        # The BF16 round-trip must actually perturb the bias on realistic
        # inputs; otherwise the cross-authority check guards nothing.
        inp = _make_inputs(0)
        bias_fp32 = inp["router_bias"].float()
        bias_bf16 = bias_fp32.to(torch.bfloat16).float()
        self.assertFalse(
            torch.equal(bias_fp32, bias_bf16),
            "BF16 round-trip did not change router_bias — check is vacuous",
        )

    def test_bf16_bias_roundtrip_present_in_all_three_sites(self) -> None:
        # The three CRITICAL sites must all carry the BF16 round-trip.
        # (decode_fwd.py is the canonical authority; prefill_gate.py has both
        # the kernel body and the golden; prefill_fwd.py has golden_whole_moe's
        # gate_fn.)  Read source text rather than import to stay card-free.
        decode_src = (_REPO / "models" / "step3p5" / "decode_fwd.py").read_text(
            encoding="utf-8",
        )
        prefill_gate_src = (
            _REPO / "models" / "step3p5" / "prefill_gate.py"
        ).read_text(encoding="utf-8")
        prefill_fwd_src = (
            _REPO / "models" / "step3p5" / "prefill_fwd.py"
        ).read_text(encoding="utf-8")

        # decode_fwd.py gate kernel body: cast(cast(bias, BF16), FP32).
        self.assertIn("pl.cast(bias_row_chunk, target_type=pl.BF16)", decode_src)
        # prefill_gate.py: kernel body (nested pl.cast) + golden (torch).
        self.assertIn("pl.cast(bias_row, target_type=pl.BF16)", prefill_gate_src)
        self.assertIn(
            "router_bias = router_bias.to(torch.bfloat16).float()",
            prefill_gate_src,
        )
        # prefill_fwd.py golden_whole_moe gate_fn (torch).
        self.assertIn(
            "router_bias = router_bias.float().to(torch.bfloat16).float()",
            prefill_fwd_src,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
