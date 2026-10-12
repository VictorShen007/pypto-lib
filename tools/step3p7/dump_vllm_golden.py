#!/usr/bin/env python3
"""Dump vLLM PerceptionEncoder (Step3.7 vision tower) intermediate tensors.

Uses vLLM's ACTUAL PerceptionEncoder implementation (from step_vl.py),
not a from-scratch reimplementation. This ensures the golden matches what
vLLM actually computes (same attention, same weight mapping, same RoPE).

Key differences from the old dump_vllm_vision_golden.py:
  - Uses vLLM's PerceptionEncoder (not self-written)
  - Uses vLLM's WeightsMapper for weight loading (not custom KEY mapping)
  - Uses vLLM's QKVParallelLinear/RowParallelLinear (not nn.Linear)
  - Uses vLLM's PerceptionEncoderRope2D (not custom Rope2D)
  - Uses vLLM's MMEncoderAttention (not torch SDPA directly)

Usage:
  # Must run in vLLM's Python environment (has vllm installed)
  export PYTHONPATH=<VLLM_SRC>:$PYTHONPATH
  export LD_LIBRARY_PATH="<PTOAS_LIB>:$LD_LIBRARY_PATH"
  python -m tools.step3p7.dump_vllm_golden \
    --ckpt <CKPT> \
    --out <OUT_DIR> \
    --device 0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch_npu  # noqa: F401 — registers NPU backend


# ── vLLM imports (must be in vLLM's Python env) ────────────────────────────
# These are vLLM's ACTUAL model implementations, not reimplementations.
from vllm.config import VllmConfig, ModelConfig, set_current_vllm_config
from vllm.model_executor.models.step_vl import (
    PerceptionEncoder,
    PerceptionEncoderVisionBlock,
    _DEFAULT_NORM_LAYER,
)
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.models.utils import WeightsMapper
from vllm.model_executor.layers.linear import ColumnParallelLinear


def _load_config(ckpt_dir: Path) -> dict:
    """Load config.json from checkpoint."""
    with open(ckpt_dir / "config.json") as f:
        return json.load(f)


def _build_vllm_perception_encoder(config: dict, ckpt_dir: Path, device: int):
    """Instantiate vLLM's PerceptionEncoder with real weights.

    This uses vLLM's ACTUAL implementation — same attention (MMEncoderAttention),
    same linear layers (QKVParallelLinear/RowParallelLinear), same RoPE
    (PerceptionEncoderRope2D), same weight mapping (WeightsMapper).
    """
    vision_config = type("VisionConfig", (), config["vision_config"])()

    # vLLM's get_act_fn for quick_gelu
    act_fn = get_act_fn(config["vision_config"]["hidden_act"])

    # Instantiate vLLM's PerceptionEncoder (TP=1, data_parallel mode)
    # disable_tp=True → no tensor parallel slicing (full-width, like single-rank)
    encoder = PerceptionEncoder(
        vision_config,
        act_layer=act_fn,
        norm_layer=_DEFAULT_NORM_LAYER,
        quant_config=None,  # BF16, no quantization
        prefix="vision_model",
    ).to(f"npu:{device}").to(torch.bfloat16)

    # Load weights using vLLM's WeightsMapper (the ACTUAL mapping vLLM uses)
    mapper = WeightsMapper(
        orig_to_new_prefix={
            "model.vision_model.": "vision_model.",
            "model.vit_large_projector.": "vit_large_projector.",
        },
        orig_to_new_substr={
            ".attn.in_proj_weight": ".attn.qkv_proj.weight",
            ".attn.in_proj_bias": ".attn.qkv_proj.bias",
            ".mlp.c_fc": ".mlp.fc1",
            ".mlp.c_proj": ".mlp.fc2",
        },
    )

    # Load all safetensors shards and apply mapper
    from safetensors import safe_open
    import glob

    index = json.load(open(ckpt_dir / "model.safetensors.index.json"))
    weight_map = index["weight_map"]

    # Group by shard
    shards: dict[str, dict[str, torch.Tensor]] = {}
    for hf_name, shard_file in weight_map.items():
        if not shard_file:
            continue
        if shard_file not in shards:
            shards[shard_file] = {}
        with safe_open(str(ckpt_dir / shard_file), framework="pt", device="cpu") as f:
            for key in f.keys():
                if key in weight_map and weight_map[key] == shard_file:
                    shards[shard_file][key] = f.get_tensor(key)

    # Merge all shards
    all_weights = {}
    for shard_weights in shards.values():
        all_weights.update(shard_weights)

    # Apply vLLM's WeightsMapper
    mapped_weights = mapper._map(all_weights)

    # Load into model
    missing, unexpected = encoder.load_state_dict(mapped_weights, strict=False)
    if missing:
        print(f"[WARN] missing keys: {len(missing)}", file=sys.stderr)
        for k in missing[:5]:
            print(f"  missing: {k}", file=sys.stderr)
    if unexpected:
        print(f"[WARN] unexpected keys: {len(unexpected)}", file=sys.stderr)
        for k in unexpected[:5]:
            print(f"  unexpected: {k}", file=sys.stderr)

    encoder.eval()
    return encoder


# ── Hook system: dump intermediate tensors ───────────────────────────────────
class DumpCollector:
    """Collect intermediate tensors via forward hooks."""

    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        self.idx = 0
        self.hooks = []
        self.layer_idx = 0

    def _save(self, name: str, tensor: torch.Tensor):
        """Save tensor in the same format as the old golden dump."""
        path = self.out_dir / f"{self.idx:04d}_rank0_{name}.pt"
        torch.save({"__meta__": {"name": name}, "hidden_states": tensor.cpu()}, str(path))
        self.idx += 1

    def hook_vision_block(self, block: PerceptionEncoderVisionBlock, layer_idx: int):
        """Hook a vision block to dump pre/post intermediate tensors."""
        prefix = f"layer_{layer_idx:02d}"

        def pre_hook(module, args, kwargs):
            x = args[0] if args else kwargs.get("x")
            if x is not None:
                self._save(f"{prefix}_layer_input", x)

        def ln1_hook(module, args, kwargs):
            out = module(*args, **kwargs) if not isinstance(args, tuple) else None
            return None  # don't modify, just observe via forward hook

        def post_attn_hook(module, args, kwargs, output):
            # module is the VisionBlock; output is x after attn+ls1+residual
            pass

        def post_mlp_hook(module, args, kwargs, output):
            pass

        def post_block_hook(module, args, kwargs, output):
            self._save(f"{prefix}_layer_out", output)

        self.hooks.append(block.register_forward_pre_hook(pre_hook))
        self.hooks.append(block.register_forward_hook(post_block_hook))

    def hook_encoder(self, encoder: PerceptionEncoder):
        """Hook the full encoder to dump inputs/outputs."""
        # Hook each transformer block
        transformer = encoder.transformer
        for i, block in enumerate(transformer.blocks):
            self.hook_vision_block(block, i)

        # Hook conv1 (patch embed)
        def conv1_hook(module, args, kwargs):
            self._save("model_input", args[0])

        self.hooks.append(encoder.conv1.register_forward_pre_hook(conv1_hook))

        # Hook forward_features output (after ln_post, before downsamplers)
        def forward_features_hook(module, args, kwargs, output):
            self._save("tower_out", output)

        self.hooks.append(encoder.forward_features.register_forward_hook(forward_features_hook))

        # Hook projector INPUT (downsampler2 output = projector's input, [169, 6144])
        # so the pypto projector can be fed the real vLLM downsampler2 output.
        def projector_pre_hook(module, args, kwargs):
            x = args[0] if args else (kwargs.get("input_") or kwargs.get("input"))
            if x is not None and x.dim() == 3:  # [B, 169, 6144] -> [169, 6144]
                x = x.squeeze(0)
            if x is not None:
                self._save("projector_input", x)

        if hasattr(encoder, "vit_large_projector"):
            self.hooks.append(
                encoder.vit_large_projector.register_forward_pre_hook(projector_pre_hook)
            )

        # Hook forward output (after downsamplers, the final image_features)
        def forward_hook(module, args, kwargs, output):
            self._save("image_features", output)

        self.hooks.append(encoder.register_forward_hook(forward_hook))

    def cleanup(self):
        for h in self.hooks:
            h.remove()


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    device = f"npu:{args.device}"
    torch.npu.set_device(args.device)

    print(f"[dump] loading config from {args.ckpt}", flush=True)
    config = _load_config(args.ckpt)

    print(f"[dump] building vLLM PerceptionEncoder (TP=1, BF16, device={device})", flush=True)
    # vLLM v1 CustomOp layers (ColumnParallelLinear/Conv2dLayer/MMEncoderAttention)
    # require an active vllm config context. Build + load + forward all run inside it.
    vllm_config = VllmConfig()
    with set_current_vllm_config(vllm_config):
        encoder = _build_vllm_perception_encoder(config, args.ckpt, args.device)

        print(f"[dump] registering hooks", flush=True)
        collector = DumpCollector(args.out)
        args.out.mkdir(parents=True, exist_ok=True)
        collector.hook_encoder(encoder)

        # Generate random input (same seed as old golden for comparability)
        torch.manual_seed(20260806)
        pixel_values = (torch.randn(1, 3, 728, 728, dtype=torch.float32) * 0.5).to(torch.bfloat16).to(device)

        print(f"[dump] running forward (input shape={pixel_values.shape})", flush=True)
        with torch.no_grad():
            output = encoder(pixel_values)

        print(f"[dump] output shape={output.shape}", flush=True)
        print(f"[dump] dumped {collector.idx} tensors to {args.out}", flush=True)

        collector.cleanup()
    print(f"[dump] DONE", flush=True)


if __name__ == "__main__":
    sys.exit(main() or 0)
