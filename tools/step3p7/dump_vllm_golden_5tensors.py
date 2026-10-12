#!/usr/bin/env python3
"""Dump the 5 missing Step3.7 VIT intermediate tensors from vLLM's REAL PerceptionEncoder.

Reuses vLLM's actual PerceptionEncoder + WeightsMapper (same as dump_vllm_golden.py)
but:
  - feeds the EXISTING golden model_input.pt (so new tensors are layer-consistent
    with the 96 existing golden files), NOT a freshly generated randn
  - builds vit_large_projector too (so #5 image_features = projector output)
  - hooks the 5 needed points: patch_embed_out, posemb_out, downsampler1_out,
    downsampler2_out, image_features
  - verifies standalone(TP=1) reproduces serve(rank0) by comparing recomputed
    tower_out vs the existing golden 0095_rank0_tower_out.pt

Usage:
  export PYTHONPATH=<VLLM_SRC>:$PYTHONPATH
  python dump_vllm_golden_5tensors.py --ckpt <ckpt> --device 8 \
      --out <OUT_DIR> \
      --model-input <MODEL_INPUT_PT> \
      --verify-tower <TOWER_OUT_PT>
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import torch
import torch_npu  # noqa

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.models.step_vl import PerceptionEncoder, _DEFAULT_NORM_LAYER
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.models.utils import WeightsMapper
from vllm.model_executor.layers.linear import ColumnParallelLinear
from safetensors import safe_open


def _load_config(ckpt_dir: Path) -> dict:
    with open(ckpt_dir / "config.json") as f:
        return json.load(f)


def _load_all_weights(ckpt_dir: Path) -> dict:
    index = json.load(open(ckpt_dir / "model.safetensors.index.json"))
    weight_map = index["weight_map"]
    all_w = {}
    seen_shards = set()
    for hf_name, shard_file in weight_map.items():
        if not shard_file or shard_file in seen_shards:
            continue
        seen_shards.add(shard_file)
        with safe_open(str(ckpt_dir / shard_file), framework="pt", device="cpu") as f:
            for key in f.keys():
                if key in weight_map:
                    all_w[key] = f.get_tensor(key)
    return all_w


def _build_encoder(hf_config, ckpt_dir: Path, device: str):
    """Build vLLM PerceptionEncoder using the REAL HF vision config (with all
    transformers defaults: use_abs_posemb/use_rope2d/use_ln_pre/mlp_ratio...)."""
    vision_config = hf_config.vision_config
    act_fn = get_act_fn(vision_config.hidden_act)
    encoder = PerceptionEncoder(
        vision_config, act_layer=act_fn, norm_layer=_DEFAULT_NORM_LAYER,
        quant_config=None, prefix="vision_model",
    ).to(device).to(torch.bfloat16)

    mapper = WeightsMapper(
        orig_to_new_prefix={"model.vision_model.": "vision_model.",
                            "model.vit_large_projector.": "vit_large_projector."},
        orig_to_new_substr={".attn.in_proj_weight": ".attn.qkv_proj.weight",
                            ".attn.in_proj_bias": ".attn.qkv_proj.bias",
                            ".mlp.c_fc": ".mlp.fc1", ".mlp.c_proj": ".mlp.fc2"},
    )
    all_w = _load_all_weights(ckpt_dir)
    mapped = dict(mapper.apply(all_w.items()))
    # Module params are named WITHOUT the prefix (conv1.weight, ...) while the
    # checkpoint keys carry it (vision_model.conv1.weight). Plain torch
    # load_state_dict matches by exact name, so strip the prefix and drop
    # non-vision keys; otherwise the load silently succeeds with EVERYTHING
    # missing (strict=False) and the tower runs on uninitialized weights.
    mapped = {k[len("vision_model."):]: v for k, v in mapped.items()
              if k.startswith("vision_model.")}
    missing, unexpected = encoder.load_state_dict(mapped, strict=False)
    enc_keys = len(mapped)
    print(f"[encoder] mapped vision keys={enc_keys} missing={missing} "
          f"unexpected={unexpected}", flush=True)
    if missing:
        for k in missing[:8]: print(f"  missing: {k}", file=sys.stderr)
    if unexpected:
        for k in unexpected[:8]: print(f"  unexpected: {k}", file=sys.stderr)
    encoder.eval()
    return encoder


def _build_projector(hf_config, ckpt_dir: Path, device: str):
    vc = hf_config.vision_config; tc = hf_config.text_config
    proj = ColumnParallelLinear(
        vc.width * 4, tc.hidden_size,
        bias=hf_config.projector_bias,
        gather_output=True, quant_config=None,
        prefix="vit_large_projector", disable_tp=True,
    ).to(device).to(torch.bfloat16)
    proj.eval()
    # load projector weight (ckpt key is 'vit_large_projector.weight', possibly
    # 'model.vit_large_projector.weight'); proj state_dict wants relative 'weight'.
    all_w = _load_all_weights(ckpt_dir)
    pw = all_w.get("vit_large_projector.weight")
    if pw is None:
        pw = all_w.get("model.vit_large_projector.weight")
    if pw is None:
        print("[projector] WARN: vit_large_projector.weight not found in ckpt", file=sys.stderr)
    else:
        miss2, _ = proj.load_state_dict({"weight": pw}, strict=False)
        print(f"[projector] loaded weight shape={tuple(pw.shape)} remaining_missing={miss2}", flush=True)
    return proj


class Dumper:
    def __init__(self, out_dir: Path):
        self.out_dir = out_dir; self.out_dir.mkdir(parents=True, exist_ok=True)
        self.idx = 96  # continue after existing 96 golden files
        self.hooks = []
        self._patch_embed = None
        self._posemb = None
        self._ds1 = None
        self._ds2 = None

    def _save(self, name, t):
        if not torch.is_tensor(t): return
        t = t.detach().cpu()
        if t.dtype != torch.bfloat16: t = t.to(torch.bfloat16)
        path = self.out_dir / f"{self.idx:04d}_rank0_{name}.pt"
        torch.save({"__meta__": {"name": name}, "hidden_states": t}, str(path))
        print(f"  saved {path.name} shape={tuple(t.shape)}", flush=True)
        self.idx += 1

    def attach(self, encoder):
        # patch_embed_out = conv1 output reshaped to [B,2704,1536] (before +posemb)
        def conv1_hook(mod, args, kwargs, output):
            b = output.shape[0]
            resh = output.permute(0, 2, 3, 1).reshape(b, -1, output.shape[1]).squeeze(0)
            self._patch_embed = resh.clone(); self._save("patch_embed_out", resh)
        self.hooks.append(encoder.conv1.register_forward_hook(conv1_hook, with_kwargs=True))

        # posemb_out = ln_pre INPUT (after +posemb, before ln_pre)
        def lnpre_pre(mod, args, kwargs):
            x = args[0] if args else kwargs.get("x")
            if x is not None:
                self._posemb = x.squeeze(0).clone(); self._save("posemb_out", x.squeeze(0))
        self.hooks.append(encoder.ln_pre.register_forward_pre_hook(lnpre_pre, with_kwargs=True))

        # downsampler1_out = vit_downsampler1 output [B,3072,26,26] -> [676,3072]
        def ds1_hook(mod, args, kwargs, output):
            o = output.squeeze(0)  # [3072,26,26]
            flat = o.permute(1, 2, 0).reshape(-1, o.shape[0]).contiguous()  # [676,3072]
            self._ds1 = flat.clone(); self._save("downsampler1_out", flat)
        self.hooks.append(encoder.vit_downsampler1.register_forward_hook(ds1_hook, with_kwargs=True))

        # downsampler2_out = vit_downsampler2 output [B,6144,13,13] -> [169,6144]
        def ds2_hook(mod, args, kwargs, output):
            b, c, h, w = output.shape
            o = output.view(b, c, h * w).transpose(1, 2).squeeze(0).contiguous()  # [169,6144]
            self._ds2 = o.clone(); self._save("downsampler2_out", o)
        self.hooks.append(encoder.vit_downsampler2.register_forward_hook(ds2_hook, with_kwargs=True))

    def attach_projector(self, proj):
        def proj_hook(mod, args, kwargs, output):
            o = output[0] if isinstance(output, tuple) else output
            self._save("image_features", o.squeeze(0))
        self.hooks.append(proj.register_forward_hook(proj_hook, with_kwargs=True))

    def cleanup(self):
        for h in self.hooks: h.remove()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", type=int, default=8)
    ap.add_argument("--model-input", type=Path, required=True)
    ap.add_argument("--verify-tower", type=Path, default=None)
    args = ap.parse_args()

    device = f"npu:{args.device}"
    torch.npu.set_device(args.device)

    # Init a TP=1 distributed env (gloo, single proc) so vLLM's ModelWeightParameter
    # / parallel_state accessors (get_tp_group, get_tensor_model_parallel_rank) work
    # standalone without a real HCCL group.
    import os
    for k, v in {"MASTER_ADDR": "127.0.0.1", "MASTER_PORT": "29512",
                  "RANK": "0", "WORLD_SIZE": "1", "LOCAL_RANK": "0"}.items():
        os.environ.setdefault(k, v)
    from vllm.distributed.parallel_state import (init_distributed_environment,
                                                  initialize_model_parallel)

    from transformers import AutoConfig
    print(f"[dump] load HF config from {args.ckpt}", flush=True)
    hf_config = AutoConfig.from_pretrained(str(args.ckpt), trust_remote_code=True)
    vllm_config = VllmConfig()
    # standalone (no distributed/TP group init): force vision attention to
    # data-parallel mode => tp_size=1, disable_tp=True, no TP group required.
    import vllm.model_executor.models.step_vl as _svl
    _svl.is_vit_use_data_parallel = lambda: True
    with set_current_vllm_config(vllm_config):
        init_distributed_environment(world_size=1, rank=0, backend="gloo")
        initialize_model_parallel(tensor_model_parallel_size=1)

        # Mechanism-consistency with the serve: the vLLM-Ascend worker calls
        # register_ascend_customop() at startup, which swaps the tower's
        # MMEncoderAttention -> AscendMMEncoderAttention (CANN fused op
        # _npu_flash_attention_unpad, head_dim 96 pad->128). The standalone
        # capture has no worker, so without this call the tower silently runs
        # torch SDPA instead of the serve's fused attention.
        from vllm_ascend.utils import register_ascend_customop
        register_ascend_customop()

        print(f"[dump] build encoder from {args.ckpt}", flush=True)
        encoder = _build_encoder(hf_config, args.ckpt, device)
        print(f"[dump] build projector", flush=True)
        projector = _build_projector(hf_config, args.ckpt, device)

        dumper = Dumper(args.out)
        dumper.attach(encoder)
        dumper.attach_projector(projector)

        # feed EXISTING golden model_input
        mi = torch.load(str(args.model_input), map_location="cpu", weights_only=False)
        x_in = (mi["hidden_states"] if isinstance(mi, dict) else mi).to(torch.bfloat16)
        x_in = x_in.unsqueeze(0).to(device)  # [1,3,728,728]
        print(f"[dump] input shape={tuple(x_in.shape)} dtype={x_in.dtype}", flush=True)

        with torch.no_grad():
            tower = encoder.forward_features(x_in)        # triggers conv1/ln_pre hooks + tower
            enc_out = encoder(x_in)                       # full forward incl downsamplers (triggers ds1/ds2)
            # projector on enc_out ([1,169,6144] -> [1,169,4096])
            image_features = projector(enc_out)           # triggers projector hook

        print(f"[dump] tower_out shape={tuple(tower.shape)} enc_out={tuple(enc_out.shape)} "
              f"image_features={tuple(image_features[0].shape if isinstance(image_features,tuple) else image_features.shape)}", flush=True)

        # save tower_out for verification
        dumper._save("tower_out_verify", tower.squeeze(0))

        if args.verify_tower and args.verify_tower.exists():
            ref = torch.load(str(args.verify_tower), map_location="cpu", weights_only=False)
            ref = (ref["hidden_states"] if isinstance(ref, dict) else ref).to(torch.bfloat16)
            mine = tower.squeeze(0).cpu().to(torch.bfloat16)
            print(f"[verify] ref tower_out shape={tuple(ref.shape)} mine={tuple(mine.shape)}", flush=True)
            if ref.shape == mine.shape:
                diff = (ref.float() - mine.float()).abs()
                print(f"[verify] max_abs_diff={diff.max().item():.6f} mean_abs_diff={diff.mean().item():.6f}", flush=True)
                print(f"[verify] allclose(bf16,atol=0.05,rtol=0.05)={torch.allclose(ref,mine,atol=0.05,rtol=0.05)}", flush=True)
            else:
                print(f"[verify] SHAPE MISMATCH — ckpt/encoder mismatch", flush=True)

        dumper.cleanup()
    print(f"[dump] DONE — {len(list(args.out.glob('*.pt')))} files in {args.out}", flush=True)


if __name__ == "__main__":
    sys.exit(main() or 0)
