# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Host-side im2col helpers for the step3p7 vision conv stages.

The pypto conv kernels (``conv_patch.patch_embed``, ``conv_downsample.conv_downsample1/2``)
take an **already-im2col'd** GM tensor as input (the kernel only does the matmul;
im2col is the host's job — see ``conv_patch.py`` / ``conv_downsample.py`` docstrings).
Since pypto ``host_orch`` is DSL (no torch), these im2cols run in the **host caller**
(outside the @pl.program) and feed the result as an input GM tensor.

Three im2cols, matching the vLLM PerceptionEncoder convs:
  - patch_embed:   image [3,728,728] --conv k14 s14 p0--> [2704, 588] -> pad [2704, 592]
  - downsampler1:  feat [1536,52,52] --conv k3 s2 p1--> [676, 9*1536=13824] -> pad [676..688]
  - downsampler2:  feat [3072,26,26] --conv k3 s2 p1--> [169, 9*3072=27648] -> pad [169..176]

The conv weight is stored as ``[9*c_in, c_out]`` (already flattened-transposed by
``vision_weight_loader``), so im2col produces ``[tokens, 9*c_in]`` = ``[M, K]`` to
match ``a[M,K] @ b[K,N]``.

All return CPU BF16 tensors (the caller moves them to GM / device as needed).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .vision_config import (
    V_PATCH, V_IMAGE, V_GRID, V_TOKENS, V_WIDTH,
    V_DS1_OUT, V_DS2_OUT, V_GRID_DS1, V_GRID_DS2, V_TOKENS_FINAL,
    V_DS_KERNEL, V_DS_STRIDE, V_DS_PAD,
    V_IMAGE_PATCH, V_GRID_PATCH, V_TOKENS_PATCH,
    V_GRID_DS1_PATCH, V_GRID_DS2_PATCH, V_TOKENS_FINAL_PATCH,
    V_TOKENS_FINAL_PATCH_PAD,
)

PATCH_FEAT = 3 * V_PATCH * V_PATCH          # 588
PATCH_FEAT_PAD = 592                        # 16-align (V_TOKENS_PAD rows not needed: 2704%16==0)


def patch_im2col(image: torch.Tensor) -> torch.Tensor:
    """image [3, 728, 728] (or [B,3,728,728]) -> im2col patches [2704, 588] -> pad [2704, 592].

    Matches vLLM ``conv1`` (Conv2d 3->1536, k14 s14 p0). im2col via F.unfold(k14, s14).
    """
    if image.dim() == 3:
        image = image.unsqueeze(0)            # [1, 3, 728, 728]
    # F.unfold: [B, C*k*k, L] where L = num patches = 52*52 = 2704
    cols = F.unfold(image, kernel_size=V_PATCH, stride=V_PATCH)   # [B, 588, 2704]
    cols = cols.transpose(1, 2).reshape(-1, PATCH_FEAT)             # [2704, 588]
    # pad cols 588 -> 592 (16-align); rows already 2704 (mult of 16, no row pad)
    cols = F.pad(cols, (0, PATCH_FEAT_PAD - PATCH_FEAT))           # [2704, 592]
    return cols.to(torch.bfloat16)


def _conv_im2col(feat: torch.Tensor, c_in: int, h: int, w: int, out_tokens: int,
                 pad_rows_to: int | None = None) -> torch.Tensor:
    """Generic k3 s2 p1 im2col for the downsamplers.

    feat: [c_in, h, w] (NCHW, no batch) -> im2col [out_tokens, 9*c_in] (-> pad rows),
    or feat: [B, c_in, h, w] (B-axis stacked images) -> [B*out_tokens, 9*c_in] with
    image-major row order (image b's rows at [b*rows:(b+1)*rows]) — matching the
    multi-image B-axis kernels' `row = b*stride + i` layout (design doc §3.7).
    """
    B = feat.shape[0] if feat.dim() == 4 else 1
    x = feat.reshape(B, c_in, h, w)
    cols = F.unfold(x, kernel_size=V_DS_KERNEL, stride=V_DS_STRIDE, padding=V_DS_PAD)  # [B, 9*c_in, out_tokens]
    cols = cols.transpose(1, 2).reshape(B, out_tokens, 9 * c_in)                        # [B, out_tokens, 9*c_in]
    if pad_rows_to is not None and out_tokens < pad_rows_to:
        cols = F.pad(cols, (0, 0, 0, pad_rows_to - out_tokens))                         # zero-pad rows -> mult of 16
    return cols.reshape(-1, 9 * c_in).to(torch.bfloat16)                                # [B*rows, 9*c_in]


def downsampler1_im2col(feat_52: torch.Tensor) -> torch.Tensor:
    """tower_out reshaped [1536, 52, 52] -> ds1 im2col [676, 13824] -> pad [688, 13824].

    k3 s2 p1: out grid = (52 + 2*1 - 3)//2 + 1 = 26 -> 26*26 = 676 tokens.
    """
    return _conv_im2col(feat_52, c_in=V_WIDTH, h=V_GRID, w=V_GRID,
                        out_tokens=V_GRID_DS1 * V_GRID_DS1,
                        pad_rows_to=((V_GRID_DS1 * V_GRID_DS1 + 15) // 16) * 16)   # 676 -> 688


def downsampler2_im2col(feat_26: torch.Tensor) -> torch.Tensor:
    """ds1_out reshaped [3072, 26, 26] -> ds2 im2col [169, 27648] -> pad [176, 27648].

    k3 s2 p1: out grid = (26 + 2*1 - 3)//2 + 1 = 13 -> 13*13 = 169 tokens.
    """
    return _conv_im2col(feat_26, c_in=V_DS1_OUT, h=V_GRID_DS1, w=V_GRID_DS1,
                        out_tokens=V_GRID_DS2 * V_GRID_DS2,
                        pad_rows_to=((V_TOKENS_FINAL + 15) // 16) * 16)            # 169 -> 176


# ── torch conv golden references (for stage-wise validation) ──────────────────
def golden_patch_embed(image: torch.Tensor, w_proj: torch.Tensor) -> torch.Tensor:
    """image [3,728,728] @ w_proj [592,1536] (im2col path) -> [2704, 1536] BF16.

    Equivalent to F.conv2d(image, conv1_weight, stride=14) reshaped.
    """
    cols = patch_im2col(image).float()                       # [2704, 592]
    out = (cols @ w_proj.float()).to(torch.bfloat16)         # [2704, 1536]
    return out


def golden_downsample1(feat_52: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """feat [1536,52,52] -> ds1 [676, 3072] (im2col path)."""
    cols = downsampler1_im2col(feat_52).float()             # [688, 13824] (padded)
    out = (cols @ w.float()).to(torch.bfloat16)             # [688, 3072]
    return out[: V_GRID_DS1 * V_GRID_DS1]                   # [676, 3072] drop pad


def golden_downsample2(feat_26: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """feat [3072,26,26] -> ds2 [169, 6144] (im2col path)."""
    cols = downsampler2_im2col(feat_26).float()             # [176, 27648]
    out = (cols @ w.float()).to(torch.bfloat16)             # [176, 6144]
    return out[: V_TOKENS_FINAL]                            # [169, 6144]


# ── PATCH path (6 x 504^2 local crops) im2cols ──────────────────────────────
# The patch path reuses the SAME conv weights as the global path; only the
# spatial grid differs (504->36x36=1296 / 36->18x18=324 / 18->9x9=81). See
# vision_config.py §PATCH path and vision_full_fwd_patch.py.

# Per-crop patch_embed im2col row pad: 1296 % 16 == 0, no row pad needed.
# (V_TOKENS_PATCH_PAD is just V_TOKENS_PATCH; PATCH_FEAT_PAD=592 shared.)


def patch_crop_im2col(image: torch.Tensor) -> torch.Tensor:
    """patch crop [3, 504, 504] (or [B,3,504,504]) -> im2col [1296, 588] -> pad [1296, 592].

    Matches vLLM ``conv1`` (Conv2d 3->1536, k14 s14 p0) on a 504^2 crop.
    F.unfold is dim-agnostic, so the global ``patch_im2col`` would also work
    on a 504 input; this is a named alias for the patch path for clarity.
    """
    return patch_im2col(image)                  # [1296, 592]


def downsampler1_patch_im2col(feat_36: torch.Tensor) -> torch.Tensor:
    """tower_out crop reshaped [1536, 36, 36] -> ds1 im2col [324, 13824] -> pad [336, 13824].

    k3 s2 p1: out grid = (36 + 2*1 - 3)//2 + 1 = 18 -> 18*18 = 324 tokens.
    """
    return _conv_im2col(feat_36, c_in=V_WIDTH, h=V_GRID_PATCH, w=V_GRID_PATCH,
                        out_tokens=V_GRID_DS1_PATCH * V_GRID_DS1_PATCH,
                        pad_rows_to=((V_GRID_DS1_PATCH * V_GRID_DS1_PATCH + 15) // 16) * 16)   # 324 -> 336


def downsampler2_patch_im2col(feat_18: torch.Tensor) -> torch.Tensor:
    """ds1_out crop reshaped [3072, 18, 18] -> ds2 im2col [81, 27648] -> pad [96, 27648].

    k3 s2 p1: out grid = (18 + 2*1 - 3)//2 + 1 = 9 -> 9*9 = 81 tokens.
    """
    return _conv_im2col(feat_18, c_in=V_DS1_OUT, h=V_GRID_DS1_PATCH, w=V_GRID_DS1_PATCH,
                        out_tokens=V_GRID_DS2_PATCH * V_GRID_DS2_PATCH,
                        pad_rows_to=V_TOKENS_FINAL_PATCH_PAD)            # 81 -> 96


# ── torch conv golden references (patch path, per-crop) ───────────────────────
def golden_patch_embed_crop(image: torch.Tensor, w_proj: torch.Tensor) -> torch.Tensor:
    """crop [3,504,504] @ w_proj [592,1536] (im2col path) -> [1296, 1536] BF16."""
    cols = patch_crop_im2col(image).float()                  # [1296, 592]
    out = (cols @ w_proj.float()).to(torch.bfloat16)         # [1296, 1536]
    return out


def golden_downsample1_patch(feat_36: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """feat [1536,36,36] -> ds1 [324, 3072] (im2col path)."""
    cols = downsampler1_patch_im2col(feat_36).float()        # [336, 13824] padded
    out = (cols @ w.float()).to(torch.bfloat16)             # [336, 3072]
    return out[: V_GRID_DS1_PATCH * V_GRID_DS1_PATCH]        # [324, 3072] drop pad


def golden_downsample2_patch(feat_18: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """feat [3072,18,18] -> ds2 [81, 6144] (im2col path)."""
    cols = downsampler2_patch_im2col(feat_18).float()        # [96, 27648]
    out = (cols @ w.float()).to(torch.bfloat16)             # [96, 6144]
    return out[: V_TOKENS_FINAL_PATCH]                      # [81, 6144]


def golden_posemb_patch(posemb_w_2704: torch.Tensor) -> torch.Tensor:
    """Interpolate global posemb [2704, 1536] (52x52 grid) -> patch [1296, 1536] (36x36).

    Mirrors vLLM ``sample_abs_posemb`` (vision_encoder.py:411): reshape to
    (1, 1536, 52, 52), F.interpolate(bilinear, align_corners=False) to
    (36, 36), permute+reshape -> [1296, 1536]. use_cls_token=False so no cls.
    """
    pe = posemb_w_2704.reshape(V_GRID, V_GRID, V_WIDTH).permute(2, 0, 1).unsqueeze(0)  # [1,1536,52,52]
    pe = F.interpolate(pe, size=(V_GRID_PATCH, V_GRID_PATCH), mode="bilinear",
                       align_corners=False)                                            # [1,1536,36,36]
    pe = pe.permute(0, 2, 3, 1).reshape(-1, V_WIDTH)                                   # [1296, 1536]
    return pe
