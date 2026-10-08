# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Step3p7 vision weight loader. Vision tower + projector are BF16 (NOT quantized
— vllm_ascend/quantization/modelslim_config.py:525: "vision tower and projector
are FLOAT"), so no dequant / scale keys.

HF stores weights as ``[out, in]``; pypto kernels consume ``[in, out]`` (matmul
``a[M,K] @ b[K,N]``), so every 2-D weight is transposed. TP-sliced weights
(qkv_proj by head, fc1 by intermediate, fc2 by intermediate, out_proj by head)
are sliced per rank; the rest (conv1, posemb, ln, ls, downsamplers, projector)
are REPLICATED.

in_proj_weight is the combined [3*D, D] qkv; per-rank slicing takes each rank's
2-head (D_LOC=192) slice from each of the q/k/v thirds, then stacks to
[3*D_LOC, D] and transposes to [D, 3*D_LOC] for the kernel.

Usage::

    from models.step3p7.weight_loader import load_step3p7_vision_weights_for_rank
    bundle = load_step3p7_vision_weights_for_rank(ckpt_dir, rank=0, tp_world_size=8)
"""

from __future__ import annotations

from .vision_config import (
    LM_HIDDEN, TP_WORLD_SIZE, V_DS1_OUT, V_DS2_OUT, V_HEAD_DIM,
    V_HEADS, V_HEADS_LOCAL, V_LAYERS, V_MLP_HIDDEN, V_MLP_HIDDEN_LOCAL,
    V_PATCH, V_TOKENS, V_WIDTH, V_WIDTH_LOCAL, V_WIDTH_LOCAL_PAD,
)

D = V_WIDTH                 # 1536
D_LOC = V_WIDTH_LOCAL      # 192
I = V_MLP_HIDDEN           # 8960
I_LOC = V_MLP_HIDDEN_LOCAL  # 1120
I_LOC_PAD = ((I_LOC + 127) // 128) * 128  # 1152 (pad to 128-multiple for MLP tiling)
HD = V_HEAD_DIM            # 96

# ── bundle key constants (vision subtree) ────────────────────────────────────
KEY_V_PATCH_EMBED = "v.patch_embed.weight"          # [592, 1536] (padded)
KEY_V_POSEMB = "v.posemb"                              # [2704, 1536]
KEY_V_LN_PRE_G = "v.ln_pre.gamma"
KEY_V_LN_PRE_B = "v.ln_pre.beta"
KEY_V_QKV = "v.block.{L}.qkv.weight"                  # [1536, 576] per rank
KEY_V_QKV_B = "v.block.{L}.qkv.bias"                  # [576] per rank
KEY_V_O = "v.block.{L}.out_proj.weight"               # [192, 1536] per rank
KEY_V_O_B = "v.block.{L}.out_proj.bias"                # [192] per rank
KEY_V_LN1_G = "v.block.{L}.ln1.gamma"
KEY_V_LN1_B = "v.block.{L}.ln1.beta"
KEY_V_FC1 = "v.block.{L}.fc1.weight"                   # [1536, 1120] per rank
KEY_V_FC1_B = "v.block.{L}.fc1.bias"                   # [1120] per rank
KEY_V_FC2 = "v.block.{L}.fc2.weight"                   # [1120, 1536] per rank
KEY_V_FC2_B = "v.block.{L}.fc2.bias"                   # [1536] (replicated)
KEY_V_LN2_G = "v.block.{L}.ln2.gamma"
KEY_V_LN2_B = "v.block.{L}.ln2.beta"
KEY_V_LS1 = "v.block.{L}.ls1"                          # [1536] replicated
KEY_V_LS2 = "v.block.{L}.ls2"
KEY_V_DS1 = "v.ds1.weight"                             # [9*1536, 3072]
KEY_V_DS1_B = "v.ds1.bias"
KEY_V_DS2 = "v.ds2.weight"                             # [9*3072, 6144]
KEY_V_DS2_B = "v.ds2.bias"
KEY_V_PROJ = "v.projector.weight"                     # [6144, 4096]


def _hf_block_keys(layer: int) -> dict:
    """On-disk HF tensor name for each per-block bundle key (real names)."""
    p = f"vision_model.transformer.resblocks.{layer}"
    return {
        KEY_V_QKV.format(L=layer): f"{p}.attn.in_proj_weight",
        KEY_V_QKV_B.format(L=layer): f"{p}.attn.in_proj_bias",
        KEY_V_O.format(L=layer): f"{p}.attn.out_proj.weight",
        KEY_V_O_B.format(L=layer): f"{p}.attn.out_proj.bias",
        KEY_V_LN1_G.format(L=layer): f"{p}.ln_1.weight",
        KEY_V_LN1_B.format(L=layer): f"{p}.ln_1.bias",
        KEY_V_FC1.format(L=layer): f"{p}.mlp.c_fc.weight",
        KEY_V_FC1_B.format(L=layer): f"{p}.mlp.c_fc.bias",
        KEY_V_FC2.format(L=layer): f"{p}.mlp.c_proj.weight",
        KEY_V_FC2_B.format(L=layer): f"{p}.mlp.c_proj.bias",
        KEY_V_LN2_G.format(L=layer): f"{p}.ln_2.weight",
        KEY_V_LN2_B.format(L=layer): f"{p}.ln_2.bias",
        KEY_V_LS1.format(L=layer): f"{p}.ls_1.gamma",
        KEY_V_LS2.format(L=layer): f"{p}.ls_2.gamma",
    }


# ── shard reading (mirror step3p5 weight_loader._read_index / _ShardCache) ───
def _read_index(ckpt_dir: str) -> dict:
    import json
    import os
    # Support both BF16 (model.safetensors.index.json) and W8A8
    # (quant_model_weights.safetensors.index.json) checkpoints.
    for name in ("model.safetensors.index.json",
                 "quant_model_weights.safetensors.index.json"):
        p = os.path.join(ckpt_dir, name)
        if os.path.exists(p):
            return json.load(open(p))["weight_map"]
    raise FileNotFoundError(f"no safetensors index in {ckpt_dir}")


class _ShardCache:
    def __init__(self, ckpt_dir: str):
        self.ckpt_dir = ckpt_dir
        self._cache: dict[str, object] = {}

    def open(self, shard: str):
        from safetensors import safe_open
        if shard not in self._cache:
            self._cache[shard] = safe_open(
                os.path.join(self.ckpt_dir, shard), framework="pt",
            )
        return self._cache[shard]


# ── slicing / orientation helpers ─────────────────────────────────────────────
import os  # noqa: E402
import torch  # noqa: E402

_PAD_TO = 592   # PATCH_FEAT 588 -> 592 (16-align)


def _to_bf16(t: torch.Tensor) -> torch.Tensor:
    return t.to(torch.bfloat16) if t.dtype != torch.bfloat16 else t


def _slice_qkv(w_qkv_hf: torch.Tensor, b_qkv_hf: torch.Tensor, rank: int, tp: int):
    """in_proj_weight [3*D, D] (q|k|v stacked, each [D, D]) -> per-rank
    [3*D_LOC, D] (rank's 2 heads from each of q/k/v), then kernel wants
    [D, 3*D_LOC] so transpose at the end. Bias similarly [3*D] -> [3*D_LOC].
    """
    heads_per = V_HEADS // tp           # 2
    head_start = rank * heads_per
    # row slices within each q/k/v third: [head_start*HD : (head_start+heads_per)*HD]
    r0 = head_start * HD
    r1 = r0 + heads_per * HD            # D_LOC = heads_per * HD
    thirds = [w_qkv_hf[r0:r1], w_qkv_hf[D + r0:D + r1], w_qkv_hf[2 * D + r0:2 * D + r1]]
    w = torch.cat(thirds, dim=0)         # [3*D_LOC, D]
    b = torch.cat([b_qkv_hf[r0:r1], b_qkv_hf[D + r0:D + r1], b_qkv_hf[2 * D + r0:2 * D + r1]], dim=0)
    return _to_bf16(w.t().contiguous()), _to_bf16(b.contiguous())   # [D, 3*D_LOC], [3*D_LOC]


def _slice_fc1(w_hf: torch.Tensor, b_hf: torch.Tensor, rank: int, tp: int):
    """c_fc [I, D] -> per-rank [D, I_LOC] (transpose + slice I). Bias [I]->[I_LOC].

    tp-aware: i_per = V_MLP_HIDDEN // tp. tp=8 -> 1120 (==I_LOC, byte-identical
    to the original TP8 path); tp=1 -> 8960 (full, for the REPLICATED tower).
    """
    i_per = V_MLP_HIDDEN // tp
    i_per_pad = ((i_per + 127) // 128) * 128
    i0 = rank * i_per
    w = w_hf[i0:i0 + i_per, :].t().contiguous()   # [D, i_per]
    b = b_hf[i0:i0 + i_per].contiguous()
    w_pad = torch.zeros(w.shape[0], i_per_pad, dtype=w.dtype)
    w_pad[:, :i_per] = w
    b_pad = torch.zeros(i_per_pad, dtype=b.dtype)
    b_pad[:i_per] = b
    return _to_bf16(w_pad), _to_bf16(b_pad)


def _slice_fc2(w_hf: torch.Tensor, rank: int, tp: int):
    """c_proj [D, I] -> per-rank [I_LOC, D] (transpose + slice I rows). Bias replicated.

    tp-aware (same i_per logic as _slice_fc1); tp=8 byte-identical, tp=1 full.
    """
    i_per = V_MLP_HIDDEN // tp
    i_per_pad = ((i_per + 127) // 128) * 128
    i0 = rank * i_per
    w = w_hf[:, i0:i0 + i_per].t().contiguous()   # [i_per, D]
    w_pad = torch.zeros(i_per_pad, w.shape[1], dtype=w.dtype)
    w_pad[:i_per, :] = w
    return _to_bf16(w_pad)


def _slice_out_proj(w_hf: torch.Tensor, b_hf: torch.Tensor, rank: int, tp: int):
    """out_proj RowParallelLinear [D, D] (out, in) -> per-rank [D_LOC, D]
    (slice INPUT columns, transpose). Bias [D] replicated (NOT sliced)."""
    heads_per = V_HEADS // tp
    r0 = rank * heads_per * HD
    r1 = r0 + heads_per * HD
    # RowParallelLinear splits INPUT dimension (columns in HF [out, in])
    w = w_hf[:, r0:r1].t().contiguous()           # [in_per_rank=192, out=D=1536]
    # v5 HD pad: wo [heads_per*96, D] -> [heads_per*128, D] (rows 96:128/head=0)
    _HD=96;_HD_PAD=128
    w_pad=torch.zeros(heads_per*_HD_PAD,w.shape[1],dtype=w.dtype)
    for _h in range(heads_per):
        w_pad[_h*_HD_PAD:_h*_HD_PAD+_HD,:]=w[_h*_HD:(_h+1)*_HD,:]
    w=w_pad
    # Bias: [out=D=1536] replicated across ranks (RowParallelLinear)
    b = b_hf.contiguous()  # [1536] full, NOT sliced
    return _to_bf16(w), _to_bf16(b)


def load_step3p7_vision_weights_for_rank(
    ckpt_dir: str,
    rank: int = 0,
    tp_world_size: int = TP_WORLD_SIZE,
) -> dict:
    """Load the vision subtree (BF16) for one rank. Returns {KEY: tensor}.

    REPLICATED: conv1, posemb, ln_pre, per-block ln1/ln2/ls1/ls2, fc2 bias,
    downsamplers, projector. TP-SLICED: qkv, fc1, fc2, out_proj (+ their bias).
    """
    wmap = _read_index(ckpt_dir)
    shards = _ShardCache(ckpt_dir)
    bundle: dict[str, torch.Tensor] = {}

    def load(hf_name: str, fallback: str | None = None) -> torch.Tensor:
        if hf_name not in wmap and fallback and fallback in wmap:
            hf_name = fallback
        return shards.open(wmap[hf_name]).get_tensor(hf_name)

    # ── replicated front ─────────────────────────────────────────────────
    conv1 = load("vision_model.conv1.weight")                       # [1536,3,14,14]
    w_proj = conv1.reshape(D, 3 * V_PATCH * V_PATCH).t().contiguous()  # [588, 1536]
    import torch.nn.functional as F
    w_proj = F.pad(w_proj, (0, 0, 0, _PAD_TO - w_proj.shape[0]))     # [592, 1536]
    bundle[KEY_V_PATCH_EMBED] = _to_bf16(w_proj)
    # W8A8 checkpoint stores this as "model.vision_model.positional_embedding";
    # BF16 checkpoint stores it as "vision_model.positional_embedding".
    bundle[KEY_V_POSEMB] = _to_bf16(load("vision_model.positional_embedding",
                                          fallback="model.vision_model.positional_embedding"))   # [2704, 1536]
    bundle[KEY_V_LN_PRE_G] = load("vision_model.ln_pre.weight").to(torch.float32)
    bundle[KEY_V_LN_PRE_B] = load("vision_model.ln_pre.bias").to(torch.float32)

    # ── per-block (47) ───────────────────────────────────────────────────
    for L in range(V_LAYERS):
        hk = _hf_block_keys(L)
        wqkv = load(hk[KEY_V_QKV.format(L=L)]); bqkv = load(hk[KEY_V_QKV_B.format(L=L)])
        w, b = _slice_qkv(wqkv, bqkv, rank, tp_world_size)
        bundle[KEY_V_QKV.format(L=L)] = w
        bundle[KEY_V_QKV_B.format(L=L)] = b
        wo = load(hk[KEY_V_O.format(L=L)]); bo = load(hk[KEY_V_O_B.format(L=L)])
        wo_s, bo_s = _slice_out_proj(wo, bo, rank, tp_world_size)
        bundle[KEY_V_O.format(L=L)] = wo_s
        bundle[KEY_V_O_B.format(L=L)] = bo_s
        bundle[KEY_V_LN1_G.format(L=L)] = load(hk[KEY_V_LN1_G.format(L=L)]).to(torch.float32)
        bundle[KEY_V_LN1_B.format(L=L)] = load(hk[KEY_V_LN1_B.format(L=L)]).to(torch.float32)
        wfc1 = load(hk[KEY_V_FC1.format(L=L)]); bfc1 = load(hk[KEY_V_FC1_B.format(L=L)])
        w1, b1 = _slice_fc1(wfc1, bfc1, rank, tp_world_size)
        bundle[KEY_V_FC1.format(L=L)] = w1
        bundle[KEY_V_FC1_B.format(L=L)] = b1
        wfc2 = load(hk[KEY_V_FC2.format(L=L)])
        bundle[KEY_V_FC2.format(L=L)] = _slice_fc2(wfc2, rank, tp_world_size)
        bundle[KEY_V_FC2_B.format(L=L)] = load(hk[KEY_V_FC2_B.format(L=L)]).to(torch.bfloat16)  # replicated bias
        bundle[KEY_V_LN2_G.format(L=L)] = load(hk[KEY_V_LN2_G.format(L=L)]).to(torch.float32)
        bundle[KEY_V_LN2_B.format(L=L)] = load(hk[KEY_V_LN2_B.format(L=L)]).to(torch.float32)
        bundle[KEY_V_LS1.format(L=L)] = load(hk[KEY_V_LS1.format(L=L)]).to(torch.float32)
        bundle[KEY_V_LS2.format(L=L)] = load(hk[KEY_V_LS2.format(L=L)]).to(torch.float32)

    # ── downsamplers + projector (replicated; conv weight [out,in,3,3] -> [9*in, out]) ──
    for key_w, key_b, hf_w, c_in, c_out in [
        (KEY_V_DS1, KEY_V_DS1_B, "vision_model.vit_downsampler1.weight", D, V_DS1_OUT),
        (KEY_V_DS2, KEY_V_DS2_B, "vision_model.vit_downsampler2.weight", V_DS1_OUT, V_DS2_OUT),
    ]:
        w = load(hf_w)                                                # [c_out, c_in, 3, 3]
        w = w.reshape(c_out, 9 * c_in).t().contiguous()             # [9*c_in, c_out]
        bundle[key_w] = _to_bf16(w)
        # bias lives in a different shard; load via index (may be absent -> zeros)
        bname = hf_w.replace(".weight", ".bias")
        if bname in wmap:
            bundle[key_b] = _to_bf16(load(bname))
        else:
            bundle[key_b] = torch.zeros(c_out, dtype=torch.bfloat16)

    proj = load("vit_large_projector.weight",
                fallback="model.vit_large_projector.weight")        # [4096, 6144]
    bundle[KEY_V_PROJ] = _to_bf16(proj.t().contiguous())             # [6144, 4096]
    return bundle


__all__ = [
    "load_step3p7_vision_weights_for_rank", "_hf_block_keys",
    "KEY_V_PATCH_EMBED", "KEY_V_POSEMB", "KEY_V_LN_PRE_G", "KEY_V_LN_PRE_B",
    "KEY_V_QKV", "KEY_V_QKV_B", "KEY_V_O", "KEY_V_O_B", "KEY_V_LN1_G", "KEY_V_LN1_B",
    "KEY_V_FC1", "KEY_V_FC1_B", "KEY_V_FC2", "KEY_V_FC2_B", "KEY_V_LN2_G", "KEY_V_LN2_B",
    "KEY_V_LS1", "KEY_V_LS2", "KEY_V_DS1", "KEY_V_DS1_B", "KEY_V_DS2", "KEY_V_DS2_B",
    "KEY_V_PROJ",
]
