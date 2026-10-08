# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Single-card golden UT for step3p7 vision MLP (fused fc1->quick_gelu->fc2).

The kernel body + torch golden reference live in
``models/step3p7/vision/vision_mlp.py``; this file holds only the ``@pl.jit``
test wrapper, ``build_tensor_specs``, the golden adapter, and the driver.

Usage::

    python -m tests.step3p7.unit.test_vision_mlp --smoke -p a2a3sim
    python -m tests.step3p7.unit.test_vision_mlp -p a2a3 -d 0
"""

import argparse

import pypto.language as pl

from models.step3p7.vision.vision_mlp import D, I, T_DYN, golden_vision_mlp, vision_mlp


@pl.jit
def vision_mlp_test(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    w_fc1: pl.Tensor[[D, I], pl.BF16],
    w_fc2: pl.Tensor[[I, D], pl.BF16],
    b_fc1: pl.Tensor[[1, I], pl.BF16],
    b_fc2: pl.Tensor[[1, D], pl.BF16],
    out: pl.Out[pl.Tensor[[T_DYN, D], pl.BF16]],
):
    x.bind_dynamic(0, T_DYN)
    out.bind_dynamic(0, T_DYN)
    out = vision_mlp(x, w_fc1, w_fc2, b_fc1, b_fc2, out)
    return out


def golden_vision_mlp_test(tensors):
    tensors["out"][:] = golden_vision_mlp(
        tensors["x"], tensors["w_fc1"], tensors["w_fc2"],
        tensors["b_fc1"], tensors["b_fc2"],
    )


def build_tensor_specs(T):
    import torch
    from golden import TensorSpec

    def init_x():
        return torch.randn(T, D) * 0.5

    def init_fc1():
        return torch.randn(D, I) / (D ** 0.5)

    def init_fc2():
        return torch.randn(I, D) / (I ** 0.5)

    return [
        TensorSpec("x", [T, D], torch.bfloat16, init_value=init_x),
        TensorSpec("w_fc1", [D, I], torch.bfloat16, init_value=init_fc1),
        TensorSpec("w_fc2", [I, D], torch.bfloat16, init_value=init_fc2),
        TensorSpec("b_fc1", [1, I], torch.bfloat16, init_value=lambda: torch.randn(1, I) * 0.02),
        TensorSpec("b_fc2", [1, D], torch.bfloat16, init_value=lambda: torch.randn(1, D) * 0.02),
        TensorSpec("out", [T, D], torch.bfloat16, is_output=True),
    ]


def _parse_args():
    parser = argparse.ArgumentParser(description="Step3p7 vision MLP validation.")
    parser.add_argument("-p", "--platform", default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--smoke", action="store_true", default=False)
    parser.add_argument("--t", type=int, default=16,
                        help="token count (default 16 = T_TILE)")
    return parser.parse_args()


def main():
    args = _parse_args()
    from golden import ratio_allclose, run_jit

    T = args.t
    assert T % 16 == 0, f"T={T} must be a multiple of T_TILE=16"
    print(f"--- vision_mlp_test: T={T}, {D}->{I}->{D} (fused) ---")
    result = run_jit(
        fn=vision_mlp_test,
        specs=build_tensor_specs(T),
        golden_fn=golden_vision_mlp_test,
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
