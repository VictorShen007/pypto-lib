# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Single-card golden UT for step3p7 patch embed (im2col@host -> 3->1536 matmul).

The kernel body + torch golden reference live in
``models/step3p7/vision/conv_patch.py``; this file holds only the ``@pl.jit``
test wrapper, specs, golden adapter, and driver. Static shape: rows padded
``V_TOKENS`` 2704 -> ``PE_ROWS_PAD`` = 2880 (96-row groups) and K
``PATCH_FEAT_PAD`` 592 -> ``PE_K_PAD`` = 768 (uniform 256-wide chunks); zero
rows/cols contribute nothing, the first 2704 output rows are the valid result.

Usage::

    python -m tests.step3p7.unit.test_conv_patch --smoke -p a2a3sim
    python -m tests.step3p7.unit.test_conv_patch -p a2a3 -d 0
"""

import argparse

import pypto.language as pl

from models.step3p7.vision.conv_patch import (
    PATCH_FEAT_PAD, PE_K_PAD, PE_ROWS_PAD, V_TOKENS,
    golden_patch_embed, patch_embed,
)
from models.step3p7.vision.vision_config import V_WIDTH

N = PE_ROWS_PAD          # 2880 (2704 zero-padded to 30 x 96-row groups)
K = PE_K_PAD             # 768 (592 zero-padded to 3 x 256-wide chunks)
D = V_WIDTH              # 1536


@pl.jit
def patch_embed_test(
    x: pl.Tensor[[N, K], pl.BF16],
    w: pl.Tensor[[K, D], pl.BF16],
    out: pl.Out[pl.Tensor[[N, D], pl.BF16]],
):
    out = patch_embed(x, w, out)
    return out


def golden_patch_embed_test(tensors):
    tensors["out"][:] = golden_patch_embed(tensors["x"], tensors["w"])


def build_tensor_specs():
    import torch
    from golden import TensorSpec

    def init_x():
        # Real rows/cols random, then zero-pad rows 2704->2880 and cols 592->768.
        x = torch.randn(V_TOKENS, PATCH_FEAT_PAD) * 0.5
        return torch.cat([
            torch.cat([x, torch.zeros(V_TOKENS, K - PATCH_FEAT_PAD)], dim=1),
            torch.zeros(N - V_TOKENS, K),
        ], dim=0).to(torch.bfloat16)

    def init_w():
        w = torch.randn(PATCH_FEAT_PAD, D) / (PATCH_FEAT_PAD ** 0.5)
        return torch.cat([w, torch.zeros(K - PATCH_FEAT_PAD, D)], dim=0).to(torch.bfloat16)

    return [
        TensorSpec("x", [N, K], torch.bfloat16, init_value=init_x),
        TensorSpec("w", [K, D], torch.bfloat16, init_value=init_w),
        TensorSpec("out", [N, D], torch.bfloat16, is_output=True),
    ]


def _parse_args():
    parser = argparse.ArgumentParser(description="Step3p7 patch embed validation.")
    parser.add_argument("-p", "--platform", default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--smoke", action="store_true", default=False)
    return parser.parse_args()


def main():
    args = _parse_args()
    from golden import ratio_allclose, run_jit

    print(f"--- patch_embed_test: N={N}, {K}->{D} ---")
    result = run_jit(
        fn=patch_embed_test,
        specs=build_tensor_specs(),
        golden_fn=golden_patch_embed_test,
        runtime_cfg=dict(platform=args.platform, device_id=args.device),
        rtol=1e-3, atol=1e-3,
        compare_fn={"out": ratio_allclose(atol=1e-4, rtol=1.0/128)},
        compile_only=args.smoke or args.platform.endswith("sim"),
    )
    if not result.passed:
        if result.error:
            print(result.error)
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
