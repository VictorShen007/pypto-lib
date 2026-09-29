#!/usr/bin/env python3
"""Efficient per-layer VLLM golden alignment.

Strategy:
1. Load ALL checkpoint weights ONCE (per rank) via weight_loader
2. For each layer: extract single-layer weights, compile, run, compare
3. Reuse weight cache across all 45 layers

Usage: python test_prefill_layer_align.py --start 0 --end 45 [--compile-only]
"""
from __future__ import annotations

import argparse, gc, glob, os, sys, time
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # pypto-lib 根目录

from models.step3p5.weight_loader import load_step3p5_weights_for_rank

CKPT_DIR = "/mnt/hw910test-jfs/models/step3p5_flash_release_hf_mtp3_w8a8_0328-copy-mtp"
VLLM_DUMP_DIR = "/mnt/hw910test-jfs/shenwx/pypto_jy_backup/golden_step3p5/prefill_golden_aime_P1P2P3"
CACHE_DIR = "/mnt/hw910test-jfs/shenwx/step3p5_weights_cache_tp8"
_DEFAULT_DEVICES = [0, 1, 2, 3, 4, 5, 6, 7]
HIDDEN = 4096
HEAD_DIM = 128
PREFILL_T = 128  # first N tokens from golden (golden has 331)


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--start", type=int, default=0, help="First layer to test")
    p.add_argument("--end", type=int, default=45, help="Last layer (exclusive)")
    p.add_argument("--tp-size", type=int, default=8)
    p.add_argument("--device", default=",".join(map(str, _DEFAULT_DEVICES)))
    p.add_argument("--compile-only", action="store_true")
    p.add_argument("--prefill-t", type=int, default=128, help="PREFILL_T (token count, padded to multiple of 32)")
    p.add_argument("--rtol", type=float, default=5e-3, help="Attention rtol (strict)")
    p.add_argument("--atol", type=float, default=5e-3, help="Attention atol (strict)")
    p.add_argument("--mlp-rtol", type=float, default=8e-2, help="MLP/FFN rtol (relaxed)")
    p.add_argument("--mlp-atol", type=float, default=2e-1, help="MLP/FFN atol (relaxed)")
    p.add_argument("--pass-rate", type=float, default=0.999, help="Pass rate threshold")
    p.add_argument("--cos-threshold", type=float, default=0.999, help="Min cosine similarity of layer output vs golden")
    return p.parse_args()


def _load_all_weights(tp_size):
    """Load ALL checkpoint weights for all ranks. Uses cache if available."""
    all_w = {}
    for rank in range(tp_size):
        matches = glob.glob(os.path.join(CACHE_DIR, f"rank{rank}.*.pt"))
        cache_path = matches[0] if matches else os.path.join(CACHE_DIR, f"rank{rank}.pt")
        t0 = time.time()
        if os.path.exists(cache_path):
            all_w[rank] = torch.load(cache_path, map_location="cpu", weights_only=True)
            print(f"  rank {rank} weights loaded from cache in {time.time() - t0:.1f}s", flush=True)
        else:
            all_w[rank] = load_step3p5_weights_for_rank(CKPT_DIR, rank=rank, tp_world_size=tp_size, int8_routed=True)
            print(f"  rank {rank} weights loaded from safetensors in {time.time() - t0:.1f}s", flush=True)
    return all_w


def _extract_layer_weights(all_w, layer_idx, tp_size, tensors):
    """Extract single-layer weights from the full weight cache into spec tensors."""
    from models.step3p5.prefill_fwd import NUM_DENSE_LAYERS

    # Layer-type classification. Step3.5 has 45 layers: 0/1/2 dense, 3..44 MoE;
    # full-attention at li % 4 == 0 (plus li==0), sliding-attention elsewhere.
    li = layer_idx
    # full attention: li==0 is a full layer (with dense MLP); li>=4 is full iff li%4==0.
    is_full = (li == 0) or (li >= 3 and li % 4 == 0)
    # dense MLP (vs MoE): layers 0/1/2 only.
    is_dense = li < 3
    # full_local: index into the per-full-layer weight stacks (0,4,8,... -> 0,1,2,...).
    full_local = li // 4 if (li >= 3 and is_full) else (0 if li == 0 else None)
    # swa_local: index into the per-SWA-layer weight stacks (1,2 -> 0,1; MoE swa -> li - #full_before - 1).
    swa_local = li - 1 if (li >= 1 and li < 3 and not is_full) else None
    if li >= 3 and not is_full:
        # MoE sliding layers: subtract the full layers seen so far (li//4) and the dense offset.
        swa_local = li - li // 4 - 1
        full_local = None
    # dense_local: index into the per-dense-layer MLP stacks (0,1,2 -> 0,1,2).
    dense_local = li if is_dense else None

    copied = set()
    skipped = set()
    # Iterate every TP rank; each rank's bundle holds its own TP/EP slice.
    for rank in range(tp_size):
        w = all_w[rank]

        # Norm weights are full-stack [45, ...] tables; select the single row for this layer.
        # Cast to FP32 because the device norm tensors are declared FP32.
        for wk, sn in [("input_rms_weight", "input_rms_weight"),
                        ("post_attn_rms_weight", "post_rms_weight"),
                        ("q_norm_weight", "q_norm_weight"),
                        ("k_norm_weight", "k_norm_weight")]:
            if sn in tensors:
                tensors[sn][rank][layer_idx].copy_(w[wk][layer_idx].to(torch.float32))

        # Attention projection weights. The device spec declares the full-attn and
        # swa-attn stacks separately (full_wq/full_wk/... vs swa_wq/swa_wk/...), each
        # sized by the number of layers of that attention type, so we write the
        # single-layer row into the per-type local slot (full_local / swa_local).
        if is_full and full_local is not None:
            for wk, sn in [("wq_full", "full_wq"), ("wk_full", "full_wk"),
                           ("wv_full", "full_wv"), ("w_g_full", "full_w_g")]:
                if sn in tensors:
                    src = w[wk][full_local]  # [H, ...] per-layer row in the full stack
                    off = full_local * HIDDEN
                    tensors[sn][rank][off:off + HIDDEN].copy_(src)
            if "full_wo" in tensors and "wo_full" in w:
                src = w["wo_full"][full_local]  # [1024, 4096] o_proj row
                off = full_local * src.shape[0]
                tensors["full_wo"][rank][off:off + src.shape[0]].copy_(src)
        elif swa_local is not None:
            for wk, sn in [("wq_swa", "swa_wq"), ("wk_swa", "swa_wk"),
                           ("wv_swa", "swa_wv"), ("w_g_swa", "swa_w_g")]:
                if sn in tensors:
                    src = w[wk][swa_local]
                    off = swa_local * HIDDEN
                    tensors[sn][rank][off:off + HIDDEN].copy_(src)
            if "swa_wo" in tensors and "wo_swa" in w:
                src = w["wo_swa"][swa_local]
                off = swa_local * src.shape[0]
                tensors["swa_wo"][rank][off:off + src.shape[0]].copy_(src)

        # Dense MLP weights (layers 0/1/2). gate/up are [H, 1408] per layer (TP-sliced),
        # down is [1408, H] per layer. Copy the single layer's row into the dense stack.
        if is_dense and dense_local is not None:
            for wk, sn in [("dense_w_gate", "dense_w_gate"), ("dense_w_up", "dense_w_up")]:
                if sn in tensors:
                    src = w[wk][dense_local]  # [4096, 1408]
                    off = dense_local * HIDDEN
                    tensors[sn][rank][off:off + HIDDEN].copy_(src)
            if "dense_w_down" in tensors:
                src = w["dense_w_down"][dense_local]  # [1408, 4096]
                off = dense_local * src.shape[0]
                tensors["dense_w_down"][rank][off:off + src.shape[0]].copy_(src)

        # MoE weights (layers 3-44). Two index systems:
        #   moe_abs = layer index into the weight_loader's MoE stack (0..41 for layers 3..44)
        #   moe_rel = index into this single-layer program's MoE spec (usually 0, since
        #             _run_layer builds n_moe=1 and the spec holds only one MoE layer)
        if not is_dense:
            moe_abs = li - 3  # absolute MoE layer index in weight_loader (0..41)
            moe_rel = li - max(layer_idx, NUM_DENSE_LAYERS)  # relative index in spec
            if moe_abs >= 0:
                # Router gate + bias (replicated, FP32).
                for wk, sn in [("moe_gate_w", "moe_gate_w"), ("moe_router_bias", "moe_router_bias")]:
                    if sn in tensors and wk in w:
                        src = w[wk][moe_abs]
                        if wk == "moe_gate_w":
                            off = moe_rel * HIDDEN
                            tensors[sn][rank][off:off + HIDDEN].copy_(src)
                        else:
                            off = moe_rel * src.shape[0]
                            tensors[sn][rank][off:off + src.shape[0]].copy_(src)
                # Routed expert gate/up: [E,H,INTER] -> flattened [E*H, INTER] (device layout).
                for wk, sn in [("moe_w_gate_r", "moe_w_gate_r"), ("moe_w_up_r", "moe_w_up_r")]:
                    if sn in tensors and wk in w:
                        src = w[wk][moe_abs]
                        flat = src.reshape(-1, src.shape[-1])
                        off = moe_rel * flat.shape[0]
                        tensors[sn][rank][off:off + flat.shape[0]].copy_(flat)
                # Routed expert per-output-channel scales (W8A8), [E, out] per layer.
                for wk, sn in [("moe_w_gate_r_scale", "moe_w_gate_r_scale"), ("moe_w_up_r_scale", "moe_w_up_r_scale")]:
                    if sn in tensors and wk in w:
                        src = w[wk][moe_abs]
                        off = moe_rel * src.shape[0]
                        tensors[sn][rank][off:off + src.shape[0]].copy_(src)
                # Routed expert down: [E,H,INTER] -> flattened [E*INTER, H].
                if "moe_w_down_r" in tensors and "moe_w_down_r" in w:
                    src = w["moe_w_down_r"][moe_abs]
                    flat = src.reshape(-1, src.shape[-1])
                    off = moe_rel * flat.shape[0]
                    tensors["moe_w_down_r"][rank][off:off + flat.shape[0]].copy_(flat)
                if "moe_w_down_r_scale" in tensors and "moe_w_down_r_scale" in w:
                    src = w["moe_w_down_r_scale"][moe_abs]
                    off = moe_rel * src.shape[0]
                    tensors["moe_w_down_r_scale"][rank][off:off + src.shape[0]].copy_(src)
                # Shared expert gate/up (TP-sliced [H, SH_INTER_LOCAL]) + down.
                for wk, sn in [("moe_w_gate_s", "moe_w_gate_s"), ("moe_w_up_s", "moe_w_up_s")]:
                    if sn in tensors and wk in w:
                        src = w[wk][moe_abs]
                        off = moe_rel * HIDDEN
                        tensors[sn][rank][off:off + HIDDEN].copy_(src)
                if "moe_w_down_s" in tensors and "moe_w_down_s" in w:
                    src = w["moe_w_down_s"][moe_abs]
                    off = moe_rel * src.shape[0]
                    tensors["moe_w_down_s"][rank][off:off + src.shape[0]].copy_(src)


def _load_vllm_router_golden(layer_idx, prefill_t):
    """Load VLLM golden router data for a layer.

    Returns (topk_ids, topk_weights) for first prefill_t tokens,
    or (None, None) if the file doesn't exist (dense layers 0-2).
    """
    vllm_path = f"{VLLM_DUMP_DIR}/1_rank0_layer_{layer_idx:02d}_moe_router.pt"
    if not os.path.exists(vllm_path):
        return None, None
    data = torch.load(vllm_path, map_location="cpu", weights_only=True)
    topk_ids = data["topk_ids"][:prefill_t]          # [T, 8] INT32
    topk_weights = data["topk_weights"][:prefill_t]   # [T, 8] FP32
    return topk_ids, topk_weights


def _compute_shared_expert_ref(all_w, tp_size, layer_idx, prefill_t):
    """Compute torch reference for shared expert output.

    Uses VLLM golden post_attn_norm as input and shared expert weights
    from the weight cache. Computation: silu(x @ w_gate_s) * (x @ w_up_s) @ w_down_s
    then TP all-reduce (sum across ranks).

    Returns: [PREFILL_T, HIDDEN] FP32 tensor, or None if not a MoE layer.
    """
    from models.step3p5.prefill_fwd import NUM_DENSE_LAYERS

    if layer_idx < NUM_DENSE_LAYERS:
        return None

    vllm_path = f"{VLLM_DUMP_DIR}/1_rank0_layer_{layer_idx:02d}_post_attn_norm.pt"
    if not os.path.exists(vllm_path):
        return None
    x = torch.load(vllm_path, map_location="cpu", weights_only=True)["hidden_states"]
    x = x[:prefill_t].float()  # [T, 4096]

    moe_abs = layer_idx - NUM_DENSE_LAYERS
    rank_outputs = []
    for rank in range(tp_size):
        w = all_w[rank]
        w_gate_s = w["moe_w_gate_s"][moe_abs].float()  # [4096, SH_INTER_LOCAL]
        w_up_s = w["moe_w_up_s"][moe_abs].float()      # [4096, SH_INTER_LOCAL]
        w_down_s = w["moe_w_down_s"][moe_abs].float()  # [SH_INTER_LOCAL, 4096]

        gate = torch.nn.functional.silu(x @ w_gate_s)  # [128, SH_INTER_LOCAL]
        up = x @ w_up_s                                  # [128, SH_INTER_LOCAL]
        y = gate * up                                    # [128, SH_INTER_LOCAL]
        out = y @ w_down_s                               # [128, 4096]
        rank_outputs.append(out)

    # TP all-reduce: sum across ranks
    return sum(rank_outputs)  # [128, 4096] FP32


def _compare_routing(pypto_indices, pypto_weights, golden_ids, golden_weights):
    """Compare pypto routing output with VLLM golden.

    pypto_indices: [PREFILL_T, 8] INT32  (expert_indices_dump)
    pypto_weights: [PREFILL_T, 8] FP32   (expert_weights_dump)
    golden_ids:    [PREFILL_T, 8] INT32  (vllm topk_ids)
    golden_weights:[PREFILL_T, 8] FP32   (vllm topk_weights)

    Returns dict with routing comparison metrics.
    """
    if golden_ids is None:
        return {"routing_note": "N/A (dense layer, no router data)"}

    n_tokens = pypto_indices.shape[0]
    exact = 0       # tokens where top-8 set matches exactly
    any_match = 0    # tokens where at least 1 expert matches
    total_intersection = 0
    idx_mismatches = []  # (token_idx, py_set, gold_set, intersection)

    for t in range(n_tokens):
        py_set = set(pypto_indices[t].tolist())
        gold_set = set(golden_ids[t].tolist())
        inter = len(py_set & gold_set)
        total_intersection += inter
        if inter == 8:
            exact += 1
        if inter > 0:
            any_match += 1
        if inter < 8:
            idx_mismatches.append((t, py_set, gold_set, inter))

    # Show first 5 mismatches
    mismatch_details = []
    for t, py_set, gold_set, inter in idx_mismatches[:5]:
        mismatch_details.append(
            f"  token {t:3d}: py={sorted(py_set)} gold={sorted(gold_set)} "
            f"intersection={inter}/8"
        )

    return {
        "routing_token_exact": f"{exact}/{n_tokens} ({100*exact/n_tokens:.2f}%)",
        "routing_token_any": f"{any_match}/{n_tokens} ({100*any_match/n_tokens:.2f}%)",
        "routing_idx_jaccard": f"{total_intersection}/{n_tokens*8} ({100*total_intersection/(n_tokens*8):.2f}%)",
        "routing_mismatch_count": len(idx_mismatches),
        "routing_mismatch_samples": mismatch_details,
    }


def _compare_module_dumps(layer_idx, tensors, prefill_t, rtol, atol, mlp_rtol, mlp_atol):
    """Compare the 11 per-module device dumps against VLLM golden detail dumps.

    Golden files live per-rank in ``VLLM_DUMP_DIR`` as ``1_rank0_layer_XX_<name>.pt``.
    Semantic mapping (device dump -> golden file key):

      input_norm_dump  -> input_norm.hidden_states            (BF16, replicated)
      q_proj_dump      -> qkv_proj.q                          (FP32 device vs BF16 golden)
      k_proj_dump      -> qkv_proj.k                          (FP32 device vs BF16 golden)
      v_proj_dump      -> qkv_proj.v                          (FP32 device vs BF16 golden)
      q_norm_dump      -> qk_norm.q                           (FP32 device vs BF16 golden)
      k_norm_dump      -> qk_norm.k                           (FP32 device vs BF16 golden)
      gate_logits_dump -> attn_gate_logits.gate               (BF16, device padded to 16 heads)
      resid1_dump      -> post_attn_residual.hidden_states    (BF16)
      attn_delta_dump  -> post_attn_residual.attn_delta       (BF16)
      post_norm_dump   -> post_attn_norm.hidden_states        (BF16)
      ffn_output_dump  -> ffn_out.ffn_output                  (BF16, pre-residual-add)

    PL rulings honoured here: gate is compared only on the leading real-head
    columns (full=8, swa=12 — sliced by the golden column count, not hardcoded);
    qkv/qk device tensors are FP32 and are compared directly against
    ``golden.float()`` (the FP32 value is the unrounded pre-BF16-cast, so casting
    the device side to BF16 would only add error).  The layer output
    ``ffn_out.hidden_states`` is intentionally NOT dumped here — it is already the
    main ``next_hidden_out`` comparison target (single-layer semantic reuse).
    """
    info = {}

    # Golden comparisons are rank0-only: the device dump is read at rank 0
    # (tensors[name][0]) and matched against ``1_rank0_layer_*``. The other
    # ranks are implied by TP symmetry and are NOT compared yet — per-rank
    # comparison is a known follow-up, not a completed check.
    def _golden(name, key):
        path = f"{VLLM_DUMP_DIR}/1_rank0_layer_{layer_idx:02d}_{name}.pt"
        if not os.path.exists(path):
            return None
        d = torch.load(path, map_location="cpu", weights_only=True)
        return d[key][:prefill_t]

    def _cmp(tag, dev, gold, rtol=rtol, atol=atol):
        if gold is None:
            info[tag] = {"golden": False}
            return
        dev = dev.float()
        gold = gold.float()
        diff = (dev - gold).abs()
        close = torch.isclose(dev, gold, rtol=rtol, atol=atol)
        info[tag] = {
            "golden": True,
            "max_abs": diff.max().item(),
            "mean_abs": diff.mean().item(),
            "pass_rate": close.float().mean().item(),
        }

    def _dump(name):
        return tensors[name][0, :prefill_t] if name in tensors else None

    # Golden self-consistency sanity check: post_attn_residual.hidden_states
    # must equal layer_input + attn_delta bitwise (0/0). This validates the
    # golden attn_delta field semantics (used as the o_proj reference) before
    # any device comparison relies on it.
    li = _golden("layer_input", "hidden_states")
    pa = _golden("post_attn_residual", "hidden_states")
    ad = _golden("post_attn_residual", "attn_delta")
    if li is not None and pa is not None and ad is not None:
        info["residual_identity"] = {
            "golden": True,
            "exact": bool(torch.equal(pa, li + ad)),
            "max_abs": (pa.float() - (li + ad).float()).abs().max().item(),
        }
    else:
        info["residual_identity"] = {"golden": False}

    # input_norm (BF16, replicated across ranks).
    d = _dump("input_norm_dump")
    if d is not None:
        _cmp("input_norm", d, _golden("input_norm", "hidden_states"))

    # qkv projections: slice device cols to the golden column count (full q is
    # 1024 but the padded single spec is 1536; swa q is 1536).
    d = _dump("q_proj_dump")
    if d is not None:
        gq = _golden("qkv_proj", "q")
        _cmp("q_proj", d[:, : gq.shape[1]] if gq is not None else d, gq)
    d = _dump("k_proj_dump")
    if d is not None:
        _cmp("k_proj", d, _golden("qkv_proj", "k"))
    d = _dump("v_proj_dump")
    if d is not None:
        _cmp("v_proj", d, _golden("qkv_proj", "v"))

    # kv cache: k_rot_dump (BF16, post-RoPE K) vs golden kv_cache.k (BF16);
    # v_tile_dump (BF16, the exact value written to v_cache) vs golden
    # kv_cache.v (BF16, no RoPE on V so it is bit-identical to qkv_proj.v).
    def _golden_kv(key):
        path = f"{VLLM_DUMP_DIR}/1_rank0_layer_{layer_idx:02d}_kv_cache.pt"
        if not os.path.exists(path):
            return None
        d = torch.load(path, map_location="cpu", weights_only=True)
        return d[key][:prefill_t]

    d = _dump("k_rot_dump")
    if d is not None:
        _cmp("kv_k", d, _golden_kv("k"))
    d = _dump("v_tile_dump")
    if d is not None:
        gv = _golden_kv("v")
        _cmp("kv_v", d, gv)
        if gv is not None:
            info["kv_v"]["bit_exact"] = bool(torch.equal(d, gv))

    # qk_norm (pre-RoPE).
    d = _dump("q_norm_dump")
    if d is not None:
        gq = _golden("qk_norm", "q")
        _cmp("q_norm", d[:, : gq.shape[1]] if gq is not None else d, gq)
    d = _dump("k_norm_dump")
    if d is not None:
        _cmp("k_norm", d, _golden("qk_norm", "k"))

    # gate (BF16, device padded to 16 heads; compare leading real-head cols).
    d = _dump("gate_logits_dump")
    if d is not None:
        gg = _golden("attn_gate_logits", "gate")
        _cmp("gate_logits", d[:, : gg.shape[1]] if gg is not None else d, gg)

    # residual / attention delta / post-norm / ffn output (BF16).
    d = _dump("resid1_dump")
    if d is not None:
        _cmp("resid1", d, _golden("post_attn_residual", "hidden_states"))
    d = _dump("attn_delta_dump")
    if d is not None:
        _cmp("attn_delta", d, _golden("post_attn_residual", "attn_delta"))
    d = _dump("post_norm_dump")
    if d is not None:
        _cmp("post_norm", d, _golden("post_attn_norm", "hidden_states"))
    d = _dump("ffn_output_dump")
    if d is not None:
        # ffn_output is the W8A8 MoE/dense FFN result: its error floor is the
        # quant floor (~8e-2), not the attention-path floor, so it must use the
        # relaxed MLP tolerance instead of the strict rtol/atol.
        _cmp("ffn_output", d, _golden("ffn_out", "ffn_output"), rtol=mlp_rtol, atol=mlp_atol)

    return info


def _run_layer(layer_idx, tp_size, devices, all_w, compile_only, rtol, atol, mlp_rtol, mlp_atol, pass_rate_threshold, cos_threshold, prefill_t, prefill_t_pad):
    """Run a single layer comparison."""
    from models.step3p5.prefill_fwd import _build_prefill_fwd_program, build_tensor_specs
    from models.step3p5.prefill_fwd import NUM_DENSE_LAYERS, NUM_HIDDEN_LAYERS

    base = max(layer_idx, NUM_DENSE_LAYERS)
    n_moe = max(1, min(layer_idx + 1, NUM_HIDDEN_LAYERS) - base)

    program = _build_prefill_fwd_program(tp_size, layer_idx, layer_idx + 1, n_moe_layers=n_moe)
    specs = build_tensor_specs(tp_size, seed=0, n_moe_layers=n_moe)

    tensors = {}
    for spec in specs:
        t = spec.create_tensor().contiguous()
        if not t.is_shared():
            t.share_memory_()
        tensors[spec.name] = t

    _extract_layer_weights(all_w, layer_idx, tp_size, tensors)

    # Override Full attention RoPE cos/sin tables with correct theta=5M + yarn scaling.
    # build_tensor_specs initialises them with build_plain_rope_tables(theta=10000),
    # which is wrong for Full attention layers (theta=5000000 with yarn).
    from models.step3p5._ops import build_llama3_yarn_rope_tables
    from models.step3p5.prefill_attention_full import (
        MAX_SEQ_DEFAULT, ROTARY_DIM, ROPE_SCALING,
    )
    rope_cos_full_fixed, rope_sin_full_fixed = build_llama3_yarn_rope_tables(
        MAX_SEQ_DEFAULT, ROTARY_DIM, 5_000_000.0,
        factor=ROPE_SCALING["factor"],
        low=ROPE_SCALING["low_freq_factor"],
        high=ROPE_SCALING["high_freq_factor"],
        orig_max=ROPE_SCALING["original_max_position_embeddings"],
    )
    for r in range(tp_size):
        tensors["rope_cos_full"][r].copy_(rope_cos_full_fixed)
        tensors["rope_sin_full"][r].copy_(rope_sin_full_fixed)

    # Load golden data from the VLLM dump directly.
    vllm_in = torch.load(f"{VLLM_DUMP_DIR}/1_rank0_layer_{layer_idx:02d}_layer_input.pt",
                          map_location="cpu", weights_only=True)["hidden_states"]
    vllm_out = torch.load(f"{VLLM_DUMP_DIR}/1_rank0_layer_{layer_idx:02d}_ffn_out.pt",
                           map_location="cpu", weights_only=True)["hidden_states"]
    golden_input = vllm_in[:prefill_t]
    golden_output = vllm_out[:prefill_t]
    if prefill_t_pad > prefill_t:
        golden_input = torch.cat([golden_input,
            torch.zeros(prefill_t_pad - prefill_t, HIDDEN, dtype=golden_input.dtype)])
        # golden_output NOT padded — only compare real tokens below
    for r in range(tp_size):
            tensors["hidden_states"][r].copy_(golden_input)

    # Compile the single-layer program to device IR. This is CPU-only codegen;
    # the actual MaxChildren=1024 limit is enforced later at compiled.prepare.
    from pypto.backend import BackendType, set_backend_type
    set_backend_type(BackendType.Ascend910B)
    from pypto import ir
    from pypto.ir.distributed_compiled_program import DistributedConfig

    compiled = ir.compile(
        program, platform="a2a3",
        distributed_config=DistributedConfig(device_ids=devices, num_sub_workers=0),
    )
    if compile_only:
        return {"layer": layer_idx, "status": "compile_ok", "time": 0}

    from pypto.runtime.device_tensor import StackedDeviceTensor

    # persistent=True keeps the compiled program's device allocations alive across
    # the single run; __enter__/__exit__ bracket the runtime context manager.
    prepare_cm = compiled.prepare(persistent=True)
    rt = prepare_cm.__enter__()

    # _HOST: tensor names that stay on the host (already-populated inputs, or host-
    # visible pl.Out results read back after run). Everything else is allocated on
    # device via dev() below. The 11 *_dump names are the per-module outputs (task
    # #116); the 6 attention-core names (q_rot/k_rot/attn_out/attn_out_gated/
    # o_proj_local/scores) are the operator-level dumps (task #2).
    _HOST = {"hidden_states", "block_table", "slot_mapping", "positions",
             "next_hidden_out", "resid1_dump",
             "logits_shard_out",
             # Per-module dump scheme (task #116): 12 host-visible pl.Out dumps.
             "input_norm_dump", "q_proj_dump", "k_proj_dump", "v_proj_dump",
             "v_tile_dump",
             "q_norm_dump", "k_norm_dump", "gate_logits_dump",
             "attn_delta_dump", "post_norm_dump", "ffn_output_dump",
             # Attention-core dumps (task #2): 6 host-visible pl.Out dumps.
             "q_rot_dump", "k_rot_dump", "attn_out_dump", "attn_out_gated_dump",
             "o_proj_local_dump", "scores_dump"}

    def dev(name, spec):
        # Allocate one device tensor per TP rank (spec.shape[1:] is the per-rank
        # shape, spec.shape[0]=tp_size), seeded with the host value, then stack
        # them into a single StackedDeviceTensor addressed across all ranks.
        shards = [rt.alloc_tensor(tuple(spec.shape[1:]), spec.dtype,
                                  init=tensors[name][r], worker_id=r)
                  for r in range(tp_size)]
        return StackedDeviceTensor(shards, tuple(spec.shape), list(range(tp_size)))

    try:
        arglist = []
        for spec in specs:
            arglist.append(tensors[spec.name] if spec.name in _HOST else dev(spec.name, spec))

        t_run = time.time()
        rt.run(compiled, *arglist)
        elapsed = time.time() - t_run

        # Save dump tensors for per-stage analysis
        dump_dir = f"/tmp/layer{layer_idx}_dumps"
        os.makedirs(dump_dir, exist_ok=True)
        for name in ["resid1_dump", "sh_y_dump", "tile_y_dump", "attn_out_dump", "attn_out_gated_dump", "o_proj_dump",
                      "local_routed_y_dump", "expert_indices_dump", "expert_weights_dump",
                      "full_qkv_dump", "full_v_dump", "full_o_proj_reduced_dump",
                      "full_scores_dump", "next_hidden_out",
                      "input_norm_dump", "q_proj_dump", "k_proj_dump", "v_proj_dump",
                      "v_tile_dump",
                      "q_norm_dump", "k_norm_dump", "gate_logits_dump",
                      "attn_delta_dump", "post_norm_dump", "ffn_output_dump",
                      "q_rot_dump", "k_rot_dump", "attn_out_dump", "attn_out_gated_dump",
                      "o_proj_local_dump", "scores_dump"]:
            if name in tensors:
                torch.save(tensors[name], f"{dump_dir}/{name}.pt")

        actual = tensors["next_hidden_out"][0, :prefill_t, :].to(torch.bfloat16)
        diff = (actual.float() - golden_output.float()).abs()
        max_abs = diff.max().item()
        # Full-layer output (next_hidden_out = resid + attn_delta + ffn_output) vs
        # golden under relaxed MLP tolerance; pass_rate_layer_hidden and pass_rate_mlp
        # alias the same quantity (kept as two names for backward-compat).
        close_layer_hidden = torch.isclose(actual.float(), golden_output.float(), rtol=mlp_rtol, atol=mlp_atol)
        close_mlp = torch.isclose(actual.float(), golden_output.float(), rtol=mlp_rtol, atol=mlp_atol)
        pass_rate_layer_hidden = close_layer_hidden.float().mean().item()
        pass_rate_mlp = close_mlp.float().mean().item()
        # Cosine similarity of the layer output vs golden: 1 - cos_sim is the
        # "cosine error", a shape/angle diagnostic that is robust to the BF16
        # per-channel magnitude rounding that inflates max_abs on outlier cols.
        a_flat = actual.float().reshape(-1)
        g_flat = golden_output.float().reshape(-1)
        cos_sim = float((a_flat @ g_flat) / (a_flat.norm() * g_flat.norm() + 1e-12))
        cos_sim = max(-1.0, min(1.0, cos_sim))  # clamp fp overshoot (cos <= 1)
        cos_err = 1.0 - cos_sim
        n = diff.numel()
        spread = float((tensors["next_hidden_out"][:, :prefill_t, :].float() - actual.float().unsqueeze(0)).abs().max().item())

        # Attention residual comparison (resid1_dump vs VLLM post_attn_residual)
        attn_info = {}
        if "resid1_dump" in tensors:
            pypto_resid = tensors["resid1_dump"][0, :prefill_t, :].to(torch.bfloat16)
            # Check if dump is non-zero (connected)
            if pypto_resid.abs().max() > 0:
                vllm_resid_path = f"{VLLM_DUMP_DIR}/1_rank0_layer_{layer_idx:02d}_post_attn_residual.pt"
                if os.path.exists(vllm_resid_path):
                    vllm_resid = torch.load(vllm_resid_path, map_location="cpu", weights_only=True)
                    vllm_resid_hs = vllm_resid["hidden_states"][:prefill_t].to(torch.bfloat16)
                    # golden ``attn_delta`` is the true o_proj output (pre-residual-add).
                    # ``hidden_states - input`` is corrupted by residual BF16
                    # quantization, so use the attn_delta field directly.
                    vllm_attn_delta = vllm_resid["attn_delta"][:prefill_t].to(torch.bfloat16)
                    diff_resid = (pypto_resid.float() - vllm_resid_hs.float()).abs()
                    close_resid = torch.isclose(pypto_resid.float(), vllm_resid_hs.float(), rtol=rtol, atol=atol)
                    attn_info["resid1_max_abs"] = diff_resid.max().item()
                    attn_info["resid1_pass_rate"] = close_resid.float().mean().item()
                    # pypto true o_proj output (post all-reduce) vs golden attn_delta.
                    # attn_delta_dump is the post-all-reduce o_proj output; the legacy
                    # full_o_proj_reduced_dump / o_proj_reduced_dump names no longer
                    # exist in build_tensor_specs, and (resid1 - input) is corrupted by
                    # residual BF16 quantization, so use attn_delta_dump directly
                    # (consistent with the dumps-line _compare_module_dumps metric).
                    pypto_attn_delta = tensors["attn_delta_dump"][0, :prefill_t, :].to(torch.bfloat16)
                    diff_delta = (pypto_attn_delta.float() - vllm_attn_delta.float()).abs()
                    close_delta = torch.isclose(pypto_attn_delta.float(), vllm_attn_delta.float(), rtol=rtol, atol=atol)
                    attn_info["attn_delta_max_abs"] = diff_delta.max().item()
                    attn_info["attn_delta_pass_rate"] = close_delta.float().mean().item()

        # Routing comparison (MoE layers only)
        routing = {}
        if "expert_indices_dump" in tensors:
            py_idx = tensors["expert_indices_dump"][0, :prefill_t, :]    # [T, 8]
            py_w = tensors["expert_weights_dump"][0, :prefill_t, :]       # [T, 8]
            gold_ids, gold_w = _load_vllm_router_golden(layer_idx, prefill_t)
            routing = _compare_routing(py_idx, py_w, gold_ids, gold_w)

        # MoE intermediate comparison (shared / routed / combine)
        moe_info = {}
        if "sh_y_dump" in tensors:
            pypto_sh = tensors["sh_y_dump"][0, :prefill_t, :].to(torch.bfloat16)
            if pypto_sh.abs().max() > 0:
                sh_ref = _compute_shared_expert_ref(all_w, tp_size, layer_idx, prefill_t)
                if sh_ref is not None:
                    diff_sh = (pypto_sh.float() - sh_ref).abs()
                    close_sh = torch.isclose(pypto_sh.float(), sh_ref, rtol=mlp_rtol, atol=mlp_atol)
                    moe_info["sh_y_max_abs"] = diff_sh.max().item()
                    moe_info["sh_y_pass_rate"] = close_sh.float().mean().item()
                    moe_info["sh_y_mean_abs"] = diff_sh.mean().item()

        if "local_routed_y_dump" in tensors:
            pypto_routed = tensors["local_routed_y_dump"][0, :prefill_t, :].to(torch.bfloat16)
            routed_max = pypto_routed.abs().max().item()
            moe_info["routed_y_nonzero"] = routed_max > 0
            moe_info["routed_y_max_abs_val"] = routed_max
            moe_info["routed_y_finite"] = torch.isfinite(pypto_routed).all().item()

        if "tile_y_dump" in tensors:
            pypto_tile = tensors["tile_y_dump"][0, :prefill_t, :].to(torch.bfloat16)
            tile_max = pypto_tile.abs().max().item()
            moe_info["tile_y_nonzero"] = tile_max > 0
            moe_info["tile_y_max_abs_val"] = tile_max
            moe_info["tile_y_finite"] = torch.isfinite(pypto_tile).all().item()

        # Per-substep attention dump comparison (dense layers).
        attn_diag = {}
        if "qkv_dump" in tensors and tensors["qkv_dump"][0].abs().sum() > 0:
            qkv = tensors["qkv_dump"][0]
            attn = tensors["attn_dump"][0]
            attn_diag["qkv_finite"] = torch.isfinite(qkv).all().item()
            attn_diag["attn_finite"] = torch.isfinite(attn).all().item()
            attn_diag["qkv_nonzero"] = (qkv.abs() > 0).sum().item()
            attn_diag["attn_nonzero"] = (attn.abs() > 0).sum().item()

        # Full attention per-stage dump comparison (MoE layers).
        full_attn_diag = {}
        for dump_name in ["full_qkv_dump", "full_v_dump", "full_attn_out_dump", "full_attn_out_gated_dump",
                          "full_o_proj_dump", "full_o_proj_reduced_dump"]:
            if dump_name in tensors and tensors[dump_name][0].abs().max() > 0:
                full_attn_diag[dump_name] = {
                    "nonzero": True,
                    "finite": torch.isfinite(tensors[dump_name][0]).all().item(),
                    "max_abs": tensors[dump_name][0].abs().max().item(),
                }
                # Save dump to file for torch reference comparison
                _save_dir = "/tmp/full_attn_dumps"
                os.makedirs(_save_dir, exist_ok=True)
                np.save(f"{_save_dir}/layer{layer_idx:02d}_{dump_name}.npy",
                        tensors[dump_name][0].float().cpu().numpy())
            else:
                full_attn_diag[dump_name] = {"nonzero": False}

        # SWA attention per-stage dump comparison.
        swa_attn_diag = {}
        for dump_name in ["attn_out_dump", "attn_out_gated_dump",
                          "o_proj_dump", "o_proj_reduced_dump"]:
            if dump_name in tensors and tensors[dump_name][0].abs().max() > 0:
                swa_attn_diag[dump_name] = {
                    "nonzero": True,
                    "finite": torch.isfinite(tensors[dump_name][0]).all().item(),
                    "max_abs": tensors[dump_name][0].abs().max().item(),
                }
                _save_dir = "/tmp/swa_attn_dumps"
                os.makedirs(_save_dir, exist_ok=True)
                np.save(f"{_save_dir}/layer{layer_idx:02d}_{dump_name}.npy",
                        tensors[dump_name][0].float().cpu().numpy())
            else:
                swa_attn_diag[dump_name] = {"nonzero": False}

        # Stage-2 localization (token0 = V[0] analytical path): V proj vs
        # golden qkv_proj.v, and o_proj_reduced vs golden true attn delta.
        stage_info = {}
        if "full_v_dump" in tensors:
            py_v = tensors["full_v_dump"][0, :prefill_t, :].to(torch.bfloat16)
            if py_v.abs().max() > 0:
                vllm_qkv_path = f"{VLLM_DUMP_DIR}/1_rank0_layer_{layer_idx:02d}_qkv_proj.pt"
                if os.path.exists(vllm_qkv_path):
                    vllm_qkv = torch.load(vllm_qkv_path, map_location="cpu", weights_only=True)
                    gold_v = vllm_qkv["v"][:prefill_t].to(torch.bfloat16)
                    diff_v = (py_v.float() - gold_v.float()).abs()
                    stage_info["v_max_abs"] = diff_v.max().item()
                    stage_info["v_token0_max_abs"] = diff_v[0].max().item()
                    stage_info["v_token0_pass_rate"] = torch.isclose(
                        py_v[0].float(), gold_v[0].float(), rtol=rtol, atol=atol,
                    ).float().mean().item()
        if "full_o_proj_reduced_dump" in tensors:
            py_opr = tensors["full_o_proj_reduced_dump"][0, :prefill_t, :].to(torch.bfloat16)
            if py_opr.abs().max() > 0:
                vllm_resid_path = f"{VLLM_DUMP_DIR}/1_rank0_layer_{layer_idx:02d}_post_attn_residual.pt"
                if os.path.exists(vllm_resid_path):
                    vllm_resid = torch.load(vllm_resid_path, map_location="cpu", weights_only=True)
                    gold_delta = vllm_resid["attn_delta"][:prefill_t].to(torch.bfloat16)
                    diff_opr = (py_opr.float() - gold_delta.float()).abs()
                    stage_info["o_proj_reduced_max_abs"] = diff_opr.max().item()
                    stage_info["o_proj_reduced_token0_max_abs"] = diff_opr[0].max().item()
                    stage_info["o_proj_reduced_token0_pass_rate"] = torch.isclose(
                        py_opr[0].float(), gold_delta[0].float(), rtol=rtol, atol=atol,
                    ).float().mean().item()

        result = {
            "layer": layer_idx,
            "status": (
                "pass"
                if (
                    pass_rate_mlp >= pass_rate_threshold
                    and cos_sim >= cos_threshold
                )
                else "fail"
            ),
            "time": elapsed,
            "max_abs": max_abs,
            "cos_sim": cos_sim,
            "cos_err": cos_err,
            "pass_rate_layer_hidden": pass_rate_layer_hidden,
            "pass_rate_mlp": pass_rate_mlp,
            "tp_spread": spread,
            "finite": torch.isfinite(actual).all().item(),
        }
        result.update(attn_info)
        result.update(routing)
        result.update(moe_info)
        result.update(full_attn_diag)
        result.update(swa_attn_diag)
        result.update(stage_info)
        result["module_dumps"] = _compare_module_dumps(
            layer_idx, tensors, prefill_t, rtol, atol, mlp_rtol, mlp_atol,
        )
        return result
    finally:
        prepare_cm.__exit__(None, None, None)
        # close() does not release the compiled program; drop it explicitly so
        # its per-layer GM buffer allocations do not accumulate across layers.
        del compiled
        gc.collect()


def main() -> int:
    args = _parse_args()
    tp = args.tp_size
    devices = [int(x) for x in args.device.split(",")]

    # Pad PREFILL_T to multiple of 32 (for TOK_TILE) and 16 (for BATCH).
    raw_t = args.prefill_t
    t_pad = ((raw_t + 31) // 32) * 32  # ceil to multiple of 32
    if t_pad != raw_t:
        print(f"  PREFILL_T: {raw_t} -> pad to {t_pad} (multiple of 32)", flush=True)
    else:
        print(f"  PREFILL_T: {raw_t}", flush=True)

    # Override PREFILL_T before any prefill_fwd import (clears pyc cache first).
    # Set KV cache rows BEFORE any pypto import (config.py reads the env var at import time).
    KV_NUM_LAYERS = 45
    kv_blocks = (t_pad + 127) // 128
    kv_cache_rows = KV_NUM_LAYERS * kv_blocks * 128
    os.environ["PYPTO_STEP3P5_KV_CACHE_ROWS"] = str(kv_cache_rows)
    os.environ["PYPTO_STEP3P5_DUMP"] = "1"  # 精度测试需要 device dump 对比 golden
    print(f"  KV_CACHE_ROWS: {kv_cache_rows} ({kv_blocks} blocks/layer × {KV_NUM_LAYERS} layers)", flush=True)

    import models.step3p5.prefill_qkv_proj_rope as pq
    pq.PREFILL_T = t_pad
    pq.PREFILL_SEQ = t_pad  # PREFILL_BATCH=1

    print(f"=== Layers {args.start}-{args.end - 1} VLLM Golden Alignment ===", flush=True)
    print(f"  tp={tp} devices={devices}", flush=True)

    # Phase 1: Load all weights once
    print("\n--- Loading all weights ---", flush=True)
    t0 = time.time()
    all_w = _load_all_weights(tp)
    print(f"  all weights loaded in {time.time() - t0:.1f}s", flush=True)

    # Phase 2: Per-layer compile + run + compare
    print(f"\n--- Per-layer alignment (layers {args.start}-{args.end - 1}) ---", flush=True)
    results = []
    total_start = time.time()

    for li in range(args.start, args.end):
        t0 = time.time()
        result = _run_layer(li, tp, devices, all_w, args.compile_only, args.rtol, args.atol, args.mlp_rtol, args.mlp_atol, args.pass_rate, args.cos_threshold, raw_t, t_pad)
        result["total_time"] = time.time() - t0
        results.append(result)

        if args.compile_only:
            print(f"  layer {li:2d}: compile OK ({result['total_time']:.1f}s)", flush=True)
        else:
            status = "PASS" if result["status"] == "pass" else "FAIL"
            route_info = ""
            if "routing_token_exact" in result:
                route_info = f" route_exact={result['routing_token_exact']} jaccard={result['routing_idx_jaccard']}"
            attn_delta_info = ""
            if "attn_delta_pass_rate" in result:
                attn_delta_info = f" attn_delta={result['attn_delta_max_abs']:.4f}/{result['attn_delta_pass_rate']:.6f}"
            resid1_info = ""
            if "resid1_pass_rate" in result:
                resid1_info = f" resid1={result['resid1_max_abs']:.4f}/{result['resid1_pass_rate']:.6f}"
            moe_info = ""
            if "sh_y_max_abs" in result:
                moe_info = f" sh_y={result['sh_y_max_abs']:.4f}/{result['sh_y_pass_rate']:.6f}"
            if "routed_y_max_abs_val" in result:
                moe_info += f" routed_y={result['routed_y_max_abs_val']:.4f}"
            if "tile_y_max_abs_val" in result:
                moe_info += f" tile_y={result['tile_y_max_abs_val']:.4f}"
            full_attn_info = ""
            if "full_qkv_dump" in result and result["full_qkv_dump"].get("nonzero"):
                full_attn_info = f" full_qkv={result['full_qkv_dump']['max_abs']:.4f}"
            if "full_attn_out_dump" in result and result["full_attn_out_dump"].get("nonzero"):
                full_attn_info += f" full_attn={result['full_attn_out_dump']['max_abs']:.4f}"
            if "full_o_proj_dump" in result and result["full_o_proj_dump"].get("nonzero"):
                full_attn_info += f" full_o_proj={result['full_o_proj_dump']['max_abs']:.4f}"
            swa_attn_info = ""
            if "attn_out_dump" in result and result["attn_out_dump"].get("nonzero"):
                swa_attn_info = f" attn_out={result['attn_out_dump']['max_abs']:.4f}"
            if "o_proj_dump" in result and result["o_proj_dump"].get("nonzero"):
                swa_attn_info += f" o_proj={result['o_proj_dump']['max_abs']:.4f}"
            stage_info = ""
            if "v_token0_max_abs" in result:
                stage_info = f" V0={result['v_token0_max_abs']:.4f}(pr={result['v_token0_pass_rate']:.4f})"
            if "o_proj_reduced_token0_max_abs" in result:
                stage_info += f" o_proj0={result['o_proj_reduced_token0_max_abs']:.4f}(pr={result['o_proj_reduced_token0_pass_rate']:.4f})"
            print(f"  layer {li:2d}: {status} | max_abs={result['max_abs']:.4f} cos={result['cos_sim']:.6f} "
                  f"next_hidden_out={result['pass_rate_layer_hidden']:.6f} mlp={result['pass_rate_mlp']:.6f} spread={result['tp_spread']:.4f}"
                  f"{attn_delta_info}{resid1_info}{route_info}{moe_info}{full_attn_info}{swa_attn_info}{stage_info} ({result['total_time']:.1f}s)", flush=True)
            md = result.get("module_dumps", {})
            if md:
                parts = []
                for tag in ("input_norm", "q_proj", "k_proj", "v_proj", "q_norm",
                            "k_norm", "gate_logits", "resid1", "attn_delta",
                            "post_norm", "ffn_output", "kv_k", "kv_v"):
                    e = md.get(tag)
                    if not e:
                        continue
                    if not e.get("golden"):
                        parts.append(f"{tag}=no-golden")
                    else:
                        s = f"{tag}={e['max_abs']:.4f}/{e['pass_rate']:.5f}"
                        if "bit_exact" in e:
                            s += f" bit_exact={e['bit_exact']}"
                        parts.append(s)
                print(f"    dumps: {' '.join(parts)}", flush=True)
                ri = md.get("residual_identity")
                if ri:
                    if not ri.get("golden"):
                        print(f"    residual_identity=no-golden", flush=True)
                    else:
                        print(f"    residual_identity=exact:{ri['exact']} max_abs:{ri['max_abs']:.4f}", flush=True)

    total_elapsed = time.time() - total_start
    passed = sum(1 for r in results if r["status"] == "pass")
    failed = sum(1 for r in results if r["status"] == "fail" and "compile" not in r["status"])
    print(f"\n=== Summary: {passed} PASS, {failed} FAIL out of {len(results)} layers "
          f"({total_elapsed:.0f}s total) ===", flush=True)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())