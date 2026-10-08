# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Step3p7 vision tower (PerceptionEncoder) + projector constants.

Source of truth for vision dims, topology and tiling. Pure Python (no ``pl``
decorators). Dims verbatim from the step3p7 checkpoint ``config.json`` +
``configuration_step3p7.py`` (vision_config defaults: use_rope2d=True,
use_abs_posemb=True, use_ln_pre=True, use_ln_post=False, ls_init_value=0.1,
hidden_act=quick_gelu, mlp_ratio=8960/1536).

The step3p7 LM is step3p5; ``LM_HIDDEN`` here must equal
``models/step3p5/config.py:HIDDEN`` (4096). Defined locally so this package's
standalone primitive tests do not drag the step3p5 import graph.
"""

# ── vision tower (perception_encoder) ────────────────────────────────────────
V_WIDTH = 1536            # embed dim
V_LAYERS = 47             # transformer blocks
V_HEADS = 16
V_HEAD_DIM = V_WIDTH // V_HEADS          # 96
V_PATCH = 14
V_IMAGE = 728
V_GRID = V_IMAGE // V_PATCH             # 52  -> 52*52 = 2704 patches
V_TOKENS = V_GRID * V_GRID              # 2704

V_MLP_RATIO = 8960 / 1536               # ≈5.83 (from configuration_step3p7.py:15)
V_MLP_HIDDEN = int(V_WIDTH * V_MLP_RATIO)   # 8960
# Full (non-TP-sliced) padded dims — for the REPLICATED tower (vision_fwd_repl_os /
# vision_fwd_patch_repl_os_flat*), which cancels TP8 and runs the full 47-layer transformer
# on each of the 8 cards (like vLLM). 8960 is already 70*128, so pad == hidden.
V_MLP_HIDDEN_PAD = ((V_MLP_HIDDEN + 127) // 128) * 128   # 8960
# V_WIDTH_PAD (all 16 heads * HD_PAD 128 = 2048) is defined below, after
# V_HEAD_DIM_PAD (line ~120) — it depends on it and this early position
# predates that definition, which broke the module import.

V_LS_INIT = 0.1                         # LayerScale gamma
V_EPS = 1e-5                            # LayerNorm eps
V_ATTN_SCALE = 1.0 / (V_HEAD_DIM ** 0.5)
V_HEAD_DIM_INV = 1.0 / V_HEAD_DIM
V_WIDTH_INV = 1.0 / V_WIDTH

# ── absolute positional embedding ───────────────────────────────────────────
V_USE_ABS_POSEMB = True                 # configuration_step3p7.py:23
V_USE_LN_PRE = True                    # configuration_step3p7.py:21
V_USE_LN_POST = False                  # configuration_step3p7.py:22
V_USE_CLS_TOKEN = False                # config.json vision_config

# ── conv downsamplers (after the 47-layer transformer) ───────────────────────
V_DS1_OUT = V_WIDTH * 2                 # 3072  (conv k3 s2 p1)
V_DS2_OUT = V_WIDTH * 4                 # 6144
V_DS_KERNEL = 3
V_DS_STRIDE = 2
V_DS_PAD = 1
# out = (in + 2*pad - k) // stride + 1 = (in + 1) // 2  for k3 s2 p1
V_GRID_DS1 = (V_GRID + 1) // 2          # 26
V_GRID_DS2 = (V_GRID_DS1 + 1) // 2     # 13
V_TOKENS_FINAL = V_GRID_DS2 * V_GRID_DS2   # 169  (= config.json image_token_len)
# Padded to a multiple of the cube M-tile (16) for the pypto projector kernel:
# 169 is NOT divisible by 16 (169 = 10*16 + 9), so a naive `n = N // 16` loop
# would skip the last 9 rows. The projector kernel takes V_TOKENS_FINAL_PAD rows;
# the host zero-pads the real 169 -> 176 (rows 169..175 = 0) before calling, and
# the caller discards output rows [169:176]. Kernel-side M is always 16 (cube-safe).
V_TOKENS_FINAL_PAD = ((V_TOKENS_FINAL + 15) // 16) * 16   # 176

# ── PATCH path (multi-resolution "local crops") ──────────────────────────────
# vLLM step3_vl runs TWO forward_features per image request: the global path
# above (1 x 728² -> 2704 tok) AND a patch path of 504² crops produced by
# vLLM's ImagePatcher (square-pad -> preprocess-resize -> crop-resize ->
# slide_window). The crop COUNT is image-dependent (e.g. 1280x720 -> 3x2=6,
# 2560x1440 -> 5x3=15) and is NEVER hardcoded in pypto: the capture script
# dumps whatever ImagePatcher produced, the drivers read n_crops from the
# dump's B dim, and the kernels take capacity VB + runtime `active` scalar
# (see V_PATCH_BATCH below). The patch path is ~63% of vision compute.
# pypto implements it separately (vision_fwd_patch.py) with a real B axis so
# all crops of one image run in one launch (the global tower is
# single-sequence, no B axis).
V_IMAGE_PATCH = 504                      # patch crop side (px)
V_GRID_PATCH = V_IMAGE_PATCH // V_PATCH  # 36  -> 36*36 = 1296 patches/crop
V_TOKENS_PATCH = V_GRID_PATCH * V_GRID_PATCH   # 1296
# Conv downsamplers k3 s2 p1 again, applied per-crop: 36 -> 18 -> 9.
V_GRID_DS1_PATCH = (V_GRID_PATCH + 1) // 2     # 18
V_GRID_DS2_PATCH = (V_GRID_DS1_PATCH + 1) // 2  # 9
V_TOKENS_FINAL_PATCH = V_GRID_DS2_PATCH * V_GRID_DS2_PATCH   # 81 image tokens/crop
# Projector cube M-tile padding (mirror V_TOKENS_FINAL_PAD): 81 -> 96.
V_TOKENS_FINAL_PATCH_PAD = ((V_TOKENS_FINAL_PATCH + 15) // 16) * 16   # 96
# Static storage CAPACITY for the patch B axis — NOT the crop count. Any
# n_crops <= V_PATCH_BATCH runs in one launch with `active` = n_crops
# (mirrors the step3p5 decode idiom: STORAGE_BATCH_CAPACITY + runtime active
# scalar guard — NOT pl.dynamic, symbolic dyn-dims have codegen bugs, see
# step3p5 notes). Bumped 8 -> 16 so the 15-crop 2560x1440 grid
# fits; drivers chunk n_crops > V_PATCH_BATCH into multiple dispatches.
# Kept a multiple of 8 for tile alignment. The flat6 tower (VB=6, active-only
# attention) remains the tuned fast path for exactly-6-crop images.
V_PATCH_BATCH = 16

# Static storage capacity for the GLOBAL tower's multi-image B axis (mirrors
# V_PATCH_BATCH for the patch path). A runtime `active` scalar guards the per-b
# bodies (images 0..active-1 run; inactive rows never read/written). VB is the
# single source of truth: every B-derived quantity (vTe = VB*vT, chunk RCH =
# max(1, 4//VB)) must derive from it symbolically — never hardcode the number 4
# anywhere else. First cut = 4; bumped to 8 to exercise 8-image
# requests (VB=8 -> RCH=1, the per-chunk task count doubles past the 48.7k
# device-proven envelope — see vision_fwd_repl_os.py (RCH=4)).
V_GLOBAL_BATCH = 8

# ── LM (step3p5) — projector output dim ─────────────────────────────────────
LM_HIDDEN = 4096                        # == step3p5 HIDDEN; projector target

# ── TP topology (mirror step3p5 TP_WORLD_SIZE=8) ─────────────────────────────
TP_WORLD_SIZE = 8
V_HEADS_LOCAL = V_HEADS // TP_WORLD_SIZE        # 2  -> local width 192
V_WIDTH_LOCAL = V_HEADS_LOCAL * V_HEAD_DIM       # 192
V_MLP_HIDDEN_LOCAL = V_MLP_HIDDEN // TP_WORLD_SIZE   # 1120
V_MLP_HIDDEN_LOCAL_PAD = ((V_MLP_HIDDEN_LOCAL + 127) // 128) * 128  # 1152 (pad to 128-multiple for MLP tiling)
V_DS2_OUT_LOCAL = V_DS2_OUT // TP_WORLD_SIZE          # 768

# PTOAS cube bf16 matmul N must be a multiple of 16; host loader zero-pads
# the local head columns up to this width (mirror step3p5 NUM_HEADS_FULL_LOCAL_PAD).
V_HEADS_LOCAL_PAD = 16
V_WIDTH_LOCAL_PAD = V_HEADS_LOCAL_PAD * V_HEAD_DIM   # 1536 (padded local width)

# v5 attention workaround: row_expand on cube-Acc [16,N] is safe iff N<64 or
# N%64==0. HD=96 (96%64=32) triggers a codegen mis-read; pad HD to the next
# multiple of 64 (128) so PV oi / merge / ctx operate on [16,128] (safe).
# q/k stay HD=96 (QK K-dim reduction safe, QK output [16,K_TILE=16] N<64 safe);
# only v / w_o / attn_ctx are padded to HD_PAD per head.
# WHY pad 96→128: row_expand_mul/div on cube-Acc [16,N] has a pypto lowering bug when
#   N>=64 and N%64!=0 (L0C-fractal→UB row-major misread, corrupts rows[0:8] of the first
#   partial 64-col block). HD=96 (96%64=32) hits it; 128 (mult-of-64) is safe. So PV oi/
#   merge/ctx run on [16,128] (safe) instead of [16,96] (buggy).
V_HEAD_DIM_PAD = 128                                  # next mult-of-64 >= 96
V_WIDTH_LOCAL_PAD_V5 = V_HEADS_LOCAL * V_HEAD_DIM_PAD # 256 (padded, 128/head)
# Full padded attention ctx width = all 16 heads * HD_PAD(128) = 2048; used by the
# replicated tower (vision_fwd_repl_os / vision_fwd_patch_repl_os_flat*).
V_WIDTH_PAD = V_HEADS * V_HEAD_DIM_PAD                # 2048

# ── tiling ───────────────────────────────────────────────────────────────────
V_D_TILE = 256                          # width-side cube/K tile (V_WIDTH % == 0). Sweep (T=2704, medians): 128=86.1us, 256=78.2us (best), 384=80.5us, 512=80.5us; all validate PASS.
V_T_TILE = 8                            # token-side tile
V_K_CHUNK = 512                         # matmul K-reduction chunk
V_MLP_OUT_CHUNK = 128                   # MLP intermediate output chunk
V_HEAD_DIM_TILE = 32                    # attention head_dim tile (96 % == 0)

assert V_WIDTH % V_D_TILE == 0, "V_WIDTH must be divisible by V_D_TILE"
assert V_MLP_HIDDEN % V_MLP_OUT_CHUNK == 0
assert V_HEAD_DIM % V_HEAD_DIM_TILE == 0
assert V_HEADS % TP_WORLD_SIZE == 0
assert V_MLP_HIDDEN % TP_WORLD_SIZE == 0

__all__ = [
    # vision dims
    "V_WIDTH", "V_LAYERS", "V_HEADS", "V_HEAD_DIM", "V_PATCH", "V_IMAGE",
    "V_GRID", "V_TOKENS", "V_MLP_RATIO", "V_MLP_HIDDEN", "V_MLP_HIDDEN_PAD",
    "V_WIDTH_PAD", "V_LS_INIT", "V_EPS",
    "V_ATTN_SCALE", "V_HEAD_DIM_INV", "V_WIDTH_INV",
    # flags
    "V_USE_ABS_POSEMB", "V_USE_LN_PRE", "V_USE_LN_POST", "V_USE_CLS_TOKEN",
    # downsample
    "V_DS1_OUT", "V_DS2_OUT", "V_DS_KERNEL", "V_DS_STRIDE", "V_DS_PAD",
    "V_GRID_DS1", "V_GRID_DS2", "V_TOKENS_FINAL",
    # patch path
    "V_IMAGE_PATCH", "V_GRID_PATCH", "V_TOKENS_PATCH", "V_GRID_DS1_PATCH",
    "V_GRID_DS2_PATCH", "V_TOKENS_FINAL_PATCH", "V_TOKENS_FINAL_PATCH_PAD",
    "V_PATCH_BATCH", "V_GLOBAL_BATCH",
    # LM
    "LM_HIDDEN",
    # TP
    "TP_WORLD_SIZE", "V_HEADS_LOCAL", "V_WIDTH_LOCAL", "V_MLP_HIDDEN_LOCAL",
    "V_DS2_OUT_LOCAL", "V_HEADS_LOCAL_PAD", "V_WIDTH_LOCAL_PAD",
    # tiling
    "V_D_TILE", "V_T_TILE", "V_K_CHUNK", "V_MLP_OUT_CHUNK", "V_HEAD_DIM_TILE",
]
