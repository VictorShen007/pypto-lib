# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Single-card golden UT for step3p7 vision LayerNorm (eps=1e-5, gamma+beta).

The kernel body + torch golden reference live in
``models/step3p7/vision/layernorm.py``; this file holds only the ``@pl.jit``
test wrapper, ``build_tensor_specs``, the golden adapter, and the driver.

Usage::

    python -m tests.step3p7.unit.test_layernorm --smoke -p a2a3sim   # compile-only
    python -m tests.step3p7.unit.test_layernorm -p a2a3 -d 0          # real card + golden
"""

import argparse

import pypto.language as pl

from models.step3p7.vision.layernorm import D, T_DYN, golden_layernorm, layernorm
from models.step3p7.vision.vision_config import V_EPS, V_T_TILE


@pl.jit
def layernorm_test(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    gamma: pl.Tensor[[D], pl.FP32],
    beta: pl.Tensor[[D], pl.FP32],
    x_normed: pl.Out[pl.Tensor[[T_DYN, D], pl.BF16]],
):
    x.bind_dynamic(0, T_DYN)
    x_normed.bind_dynamic(0, T_DYN)
    x_normed = layernorm(x, gamma, beta, x_normed)
    return x_normed


def golden_layernorm_test(tensors):
    tensors["x_normed"][:] = golden_layernorm(
        tensors["x"], tensors["gamma"], tensors["beta"],
    )


def build_tensor_specs(T):
    import torch
    from golden import TensorSpec

    def init_x():
        return torch.randn(T, D) * 0.5

    def init_gamma():
        return torch.randn(D) * 0.1 + 1.0

    def init_beta():
        return torch.randn(D) * 0.1

    return [
        TensorSpec("x", [T, D], torch.bfloat16, init_value=init_x),
        TensorSpec("gamma", [D], torch.float32, init_value=init_gamma),
        TensorSpec("beta", [D], torch.float32, init_value=init_beta),
        TensorSpec("x_normed", [T, D], torch.bfloat16, is_output=True),
    ]


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Step3p7 vision LayerNorm standalone validation.",
    )
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--smoke", action="store_true", default=False,
                        help="compile-only (no device run)")
    parser.add_argument("--t", type=int, default=V_T_TILE * 4,
                        help="token count T (must be a multiple of T_TILE)")
    parser.add_argument("--runtime-dir", type=str, default=None)
    parser.add_argument("--golden-data", type=str, default=None)
    parser.add_argument("--dump-passes", action="store_true", default=False)
    return parser.parse_args()


def main():
    args = _parse_args()
    from golden import ratio_allclose, run_jit

    T = args.t
    assert T % V_T_TILE == 0, f"T={T} must be a multiple of T_TILE={V_T_TILE}"
    print(f"--- layernorm_test: T={T}, D={D} ---")
    result = run_jit(
        fn=layernorm_test,
        specs=build_tensor_specs(T),
        golden_fn=golden_layernorm_test,
        runtime_dir=args.runtime_dir,
        golden_data=args.golden_data,
        compile_cfg=dict(dump_passes=args.dump_passes),
        runtime_cfg=dict(platform=args.platform, device_id=args.device),
        rtol=1e-3,
        atol=1e-3,
        compare_fn={"x_normed": ratio_allclose(atol=1e-4, rtol=1.0 / 128)},
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
