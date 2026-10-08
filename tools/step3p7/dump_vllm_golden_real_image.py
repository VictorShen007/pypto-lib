#!/usr/bin/env python3
"""Capture Step3.7 VIT golden dumps for a REAL image via vLLM's actual encoder.

Mechanism-consistent capture for whole-net benchmarking: the crop algorithm is
vLLM's OWN ``ImagePatcher`` and the pixel preprocessing is vLLM's OWN
``Step3VisionProcessor``, both imported from the checkpoint's
``processing_step3.py`` (no reimplementation). The encoder/projector are
vLLM's real ``PerceptionEncoder`` + ``vit_large_projector`` (TP=1 standalone,
the proven ``dump_vllm_golden_5tensors.py`` recipe: data-parallel forced,
verified against the serve dump).

Crop mechanism (ImagePatcher.__call__): square-pad -> preprocess-resize ->
determine_window_size -> crop-resize -> slide_window of 504^2 crops. The crop
COUNT and newline mask are whatever ImagePatcher derives from the image at
capture time (e.g. 2560x1440 -> 15 crops, 5x3 grid) — nothing in this script
assumes a fixed count; both are recorded to meta.json, and the patch dump
tensors carry the real B dim so downstream pypto drivers read the count from
the dump shapes.

Dumps (global, 728^2 path):
  model_input [3,728,728], patch_embed_out [2704,1536], posemb_out [2704,1536],
  layer_00_layer_input [2704,1536], tower_out [2704,1536],
  downsampler1_out [676,3072], downsampler2_out [169,6144],
  image_features [169,4096]
Dumps (patch, 504^2 crops path; N = crop count for this image):
  patch_model_input [N,3,504,504], patch_layer_00_layer_input [N,1296,1536],
  patch_tower_out [N,1296,1536], patch_downsampler1_out [N,3072,18,18],
  patch_downsampler2_out [N,6144,9,9]

Usage (must run in vLLM's Python env; the CANN toolkit env is REQUIRED —
the downsampler conv2d triggers ACL graph compile which needs ``tbe``)::

    source <ASCEND_TOOLKIT>/set_env.sh
    export PYTHONPATH=<VLLM_SRC>:$PYTHONPATH
    python tools/step3p7/dump_vllm_golden_real_image.py \\
      --ckpt <CKPT> \\
      --image <IMAGE_PATH> \\
      --out <OUT_DIR> \\
      --device 0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch_npu  # noqa: F401 — registers NPU backend

# processing_step3.py imports TokenizersBackend from
# transformers.tokenization_utils_tokenizers, which does not exist in
# transformers 4.57 (the serve env's version has it). ImagePatcher /
# Step3VisionProcessor never use it (only Step3VLProcessor tokenizer loading
# does), so stub the symbol before importing.
import types as _types
if "transformers.tokenization_utils_tokenizers" not in sys.modules:
    _stub = _types.ModuleType("transformers.tokenization_utils_tokenizers")
    _stub.TokenizersBackend = type("TokenizersBackend", (), {})
    sys.modules["transformers.tokenization_utils_tokenizers"] = _stub

from vllm.config import VllmConfig, set_current_vllm_config  # noqa: E402
from vllm.model_executor.models.step_vl import (  # noqa: E402
    PerceptionEncoder, _DEFAULT_NORM_LAYER,
)
from vllm.model_executor.layers.activation import get_act_fn  # noqa: E402
from vllm.model_executor.models.utils import WeightsMapper  # noqa: E402
from vllm.model_executor.layers.linear import ColumnParallelLinear  # noqa: E402
from safetensors import safe_open  # noqa: E402


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
    mapped = dict(mapper.apply(_load_all_weights(ckpt_dir).items()))
    # Module params are named WITHOUT the prefix (conv1.weight, ...) while the
    # checkpoint keys carry it (vision_model.conv1.weight). Plain torch
    # load_state_dict matches by exact name, so strip the prefix and drop
    # non-vision keys; otherwise the load silently succeeds with EVERYTHING
    # missing (strict=False) and the tower runs on uninitialized weights (that
    # exact bug produced the 100%-NaN "polluted" dumps).
    mapped = {k[len("vision_model."):]: v for k, v in mapped.items()
              if k.startswith("vision_model.")}
    missing, unexpected = encoder.load_state_dict(mapped, strict=False)
    print(f"[encoder] mapped vision keys={len(mapped)} missing={missing} "
          f"unexpected={unexpected}", flush=True)
    if missing:
        for k in missing[:8]:
            print(f"  missing: {k}", file=sys.stderr)
    if unexpected:
        for k in unexpected[:8]:
            print(f"  unexpected: {k}", file=sys.stderr)
    encoder.eval()
    return encoder


def _build_projector(hf_config, ckpt_dir: Path, device: str):
    vc = hf_config.vision_config
    tc = hf_config.text_config
    proj = ColumnParallelLinear(
        vc.width * 4, tc.hidden_size,
        bias=hf_config.projector_bias,
        gather_output=True, quant_config=None,
        prefix="vit_large_projector", disable_tp=True,
    ).to(device).to(torch.bfloat16)
    proj.eval()
    all_w = _load_all_weights(ckpt_dir)
    pw = all_w.get("vit_large_projector.weight")
    if pw is None:
        pw = all_w.get("model.vit_large_projector.weight")
    if pw is None:
        print("[projector] WARN: vit_large_projector.weight not found", file=sys.stderr)
    else:
        miss2, _ = proj.load_state_dict({"weight": pw}, strict=False)
        print(f"[projector] loaded weight shape={tuple(pw.shape)} remaining_missing={miss2}", flush=True)
    return proj


class Dumper:
    """Forward-hook dump collector; mode controls the B-axis convention.

    global: single-image forward, B=1 squeezed away (matches the serve-format
            golden/ dir consumed by run_vision_full.py).
    patch:  batch-of-crops forward, B dim kept ([N, ...] per-crop tensors with
            N = this image's crop count, matching golden_permodule_patch/
            consumed by run_vision_full_patch.py).
    """

    def __init__(self, out_dir: Path, mode: str):
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.idx = 0
        self.hooks = []
        self.mode = mode
        assert mode in ("global", "patch")

    def _save(self, name: str, t):
        if not torch.is_tensor(t):
            return
        # patch mode: keep the serve-format "patch_*" file names the pypto
        # patch driver (run_vision_full_patch.py) expects for shared hook
        # tensors (patch_layer_00_layer_input / patch_downsampler*_out);
        # patch_model_input / patch_tower_out are already prefixed.
        if self.mode == "patch" and not name.startswith("patch_"):
            name = "patch_" + name
        t = t.detach().cpu()
        if t.dtype != torch.bfloat16:
            t = t.to(torch.bfloat16)
        path = self.out_dir / f"{self.idx:04d}_rank0_{name}.pt"
        torch.save({"__meta__": {"name": name}, "hidden_states": t}, str(path))
        print(f"  saved {path.name} shape={tuple(t.shape)}", flush=True)
        self.idx += 1

    def _squeeze(self, t):
        return t.squeeze(0) if self.mode == "global" else t

    def attach(self, encoder):
        # patch_embed_out = conv1 output, token-major [T, 1536] (before +posemb)
        def conv1_hook(mod, args, kwargs, output):
            b = output.shape[0]
            resh = output.permute(0, 2, 3, 1).reshape(b, -1, output.shape[1])
            self._save("patch_embed_out", self._squeeze(resh))
        self.hooks.append(encoder.conv1.register_forward_hook(conv1_hook, with_kwargs=True))

        # posemb_out = ln_pre INPUT (after +posemb, before ln_pre)
        def lnpre_pre(mod, args, kwargs):
            x = args[0] if args else kwargs.get("x")
            if x is not None:
                self._save("posemb_out", self._squeeze(x))
        self.hooks.append(encoder.ln_pre.register_forward_pre_hook(lnpre_pre, with_kwargs=True))

        # layer_00_layer_input = transformer block 0 input = ln_pre output
        def block0_pre(mod, args, kwargs):
            x = args[0] if args else kwargs.get("x")
            if x is not None:
                self._save("layer_00_layer_input", self._squeeze(x))
        self.hooks.append(
            encoder.transformer.resblocks[0].register_forward_pre_hook(block0_pre, with_kwargs=True))

        # tower_out = forward_features output (after the 47-layer transformer).
        # NOTE: forward_features is a METHOD on the encoder (not a submodule),
        # so it cannot be hooked; main() calls it explicitly and saves the
        # return value (use_ln_post=False -> transformer output == tower_out).

        # downsampler1_out: global -> token-major [676, 3072]; patch keeps NCHW
        # [B, 3072, 18, 18] (the patch driver indexes crop dim 0 directly).
        def ds1_hook(mod, args, kwargs, output):
            if self.mode == "global":
                o = output.squeeze(0)                       # [3072, 26, 26]
                self._save("downsampler1_out", o.permute(1, 2, 0).reshape(-1, o.shape[0]).contiguous())
            else:
                self._save("downsampler1_out", output)      # [B, 3072, 18, 18]
        self.hooks.append(encoder.vit_downsampler1.register_forward_hook(ds1_hook, with_kwargs=True))

        # downsampler2_out: global -> token-major [169, 6144]; patch NCHW [B, 6144, 9, 9]
        def ds2_hook(mod, args, kwargs, output):
            if self.mode == "global":
                b, c, h, w = output.shape
                self._save("downsampler2_out",
                           output.view(b, c, h * w).transpose(1, 2).squeeze(0).contiguous())
            else:
                self._save("downsampler2_out", output)
        self.hooks.append(encoder.vit_downsampler2.register_forward_hook(ds2_hook, with_kwargs=True))

    def attach_projector(self, proj):
        def proj_hook(mod, args, kwargs, output):
            o = output[0] if isinstance(output, tuple) else output
            self._save("image_features", o.squeeze(0))
        self.hooks.append(proj.register_forward_hook(proj_hook, with_kwargs=True))

    def cleanup(self):
        for h in self.hooks:
            h.remove()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--image", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", type=int, default=8)
    ap.add_argument("--skip-global", action="store_true")
    ap.add_argument("--skip-patch", action="store_true")
    args = ap.parse_args()

    # The checkpoint ships vLLM's custom preprocessor (processing_step3.py);
    # put its directory on sys.path so ImagePatcher/Step3VisionProcessor import.
    sys.path.insert(0, str(args.ckpt))
    from processing_step3 import ImagePatcher, Step3VisionProcessor  # noqa: E402

    device = f"npu:{args.device}"
    torch.npu.set_device(args.device)

    # ── vLLM standalone env (TP=1, gloop, data-parallel forced) ──
    import os
    for k, v in {"MASTER_ADDR": "127.0.0.1", "MASTER_PORT": "29512",
                 "RANK": "0", "WORLD_SIZE": "1", "LOCAL_RANK": "0"}.items():
        os.environ.setdefault(k, v)
    from vllm.distributed.parallel_state import (init_distributed_environment,
                                                 initialize_model_parallel)
    import vllm.model_executor.models.step_vl as _svl
    _svl.is_vit_use_data_parallel = lambda: True

    from transformers import AutoConfig
    print(f"[dump] load HF config from {args.ckpt}", flush=True)
    hf_config = AutoConfig.from_pretrained(str(args.ckpt), trust_remote_code=True)

    vllm_config = VllmConfig()
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

        print(f"[dump] build encoder + projector (device={device})", flush=True)
        encoder = _build_encoder(hf_config, args.ckpt, device)
        projector = _build_projector(hf_config, args.ckpt, device)

        # ── vLLM's OWN crop mechanism: ImagePatcher + Step3VisionProcessor ──
        print(f"[dump] preprocessing image {args.image} (vLLM ImagePatcher)", flush=True)
        from PIL import Image
        image = Image.open(str(args.image)).convert("RGB")
        patcher = ImagePatcher()
        img_global, patches, newlines = patcher(image)
        print(f"  original={image.size} global_base={img_global.size} "
              f"num_patches={len(patches)} newlines={newlines}", flush=True)
        # NO fixed crop count: the grid is whatever ImagePatcher derives from
        # this image. Only generic sanity checks; count + mask are recorded to
        # meta.json and the dump tensors carry the real B dim, so downstream
        # pypto drivers read the count from the dump shapes.
        if not patches:
            raise RuntimeError("ImagePatcher produced no crops")
        crop_sizes = {p.size for p in patches}
        if len(crop_sizes) != 1:
            raise RuntimeError(f"mixed crop sizes: {sorted(crop_sizes)}")
        print(f"  crop size={crop_sizes.pop()}", flush=True)

        preproc = Step3VisionProcessor(728, "bilinear", 504)
        out_root = args.out
        meta = {"image": str(args.image), "image_size": list(image.size),
                "global_base_size": list(img_global.size),
                "num_patches": len(patches), "newlines": list(newlines)}

        # ── global path: patcher's base image -> 728^2 -> full forward ──
        if not args.skip_global:
            gdir = out_root / "golden"
            dump = Dumper(gdir, "global")
            dump.attach(encoder)
            dump.attach_projector(projector)
            x_g = preproc(img_global, is_patch=False)["pixel_values"].to(torch.bfloat16).to(device)
            dump._save("model_input", x_g.squeeze(0))
            print(f"[dump] global forward input={tuple(x_g.shape)}", flush=True)
            with torch.no_grad():
                tower_g = encoder.forward_features(x_g)  # tower (no hooks: method)
                dump._save("tower_out", tower_g.squeeze(0))
                enc_out = encoder(x_g)                  # conv1 -> tower -> ds1 -> ds2
                image_features = projector(enc_out)     # [1, 169, 4096]
            ift = image_features[0] if isinstance(image_features, tuple) else image_features
            print(f"[dump] global enc_out={tuple(enc_out.shape)} "
                  f"image_features={tuple(ift.shape)}", flush=True)
            meta["global"] = {"input": list(x_g.shape),
                              "enc_out": list(enc_out.shape),
                              "image_features": list(ift.shape)}
            dump.cleanup()

        # ── patch path: one forward over ALL crops (B = ImagePatcher count) ──
        if not args.skip_patch:
            pdir = out_root / "golden_permodule_patch"
            dump = Dumper(pdir, "patch")
            dump.attach(encoder)
            x_p = torch.stack(
                [preproc(p, is_patch=True)["pixel_values"].squeeze(0) for p in patches],
                dim=0).to(torch.bfloat16).to(device)    # [N, 3, 504, 504], N dynamic
            dump._save("patch_model_input", x_p)
            print(f"[dump] patch forward input={tuple(x_p.shape)}", flush=True)
            with torch.no_grad():
                tower_p = encoder.forward_features(x_p)  # [N, 1296, 1536]
                dump._save("patch_tower_out", tower_p)
                enc_out_p = encoder(x_p)                # batch-N forward
            print(f"[dump] patch enc_out={tuple(enc_out_p.shape)}", flush=True)
            meta["patch"] = {"input": list(x_p.shape),
                             "enc_out": list(enc_out_p.shape)}
            dump.cleanup()

        with open(out_root / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[dump] DONE — wrote {out_root}/meta.json", flush=True)


if __name__ == "__main__":
    sys.exit(main() or 0)
