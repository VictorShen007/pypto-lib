# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Single-card golden UT for step3p7 2D-RoPE apply.

The kernel body + torch golden reference + ``build_2d_rope_tables`` (host
cos/sin) live in ``models/step3p7/vision/rope2d.py``; this file holds only the
``@pl.jit`` test wrapper, ``build_tensor_specs``, the golden adapter, and the
driver.

Usage::

    python -m tests.step3p7.unit.test_rope2d --smoke -p a2a3sim
    python -m tests.step3p7.unit.test_rope2d -p a2a3 -d 0
"""

import argparse

import pypto.language as pl

from models.step3p7.vision.rope2d import (
    HD,
    T_DYN,
    apply_2d_rope,
    build_2d_rope_tables,
    golden_apply_2d_rope,
)


@pl.jit
def apply_2d_rope_test(
    qk: pl.Tensor[[T_DYN, HD], pl.BF16],
    cos: pl.Tensor[[T_DYN, HD], pl.FP32],
    sin: pl.Tensor[[T_DYN, HD], pl.FP32],
    out: pl.Out[pl.Tensor[[T_DYN, HD], pl.BF16]],
):
    qk.bind_dynamic(0, T_DYN)
    out.bind_dynamic(0, T_DYN)
    out = apply_2d_rope(qk, cos, sin, out)
    return out


def golden_apply_2d_rope_test(tensors):
    tensors["out"][:] = golden_apply_2d_rope(
        tensors["qk"], tensors["cos"], tensors["sin"],
    )


def build_tensor_specs(T, grid_h=13, grid_w=13):
    import torch
    from golden import TensorSpec

    assert T == grid_h * grid_w, f"T={T} must equal grid_h*grid_w={grid_h*grid_w}"

    def init_qk():
        return torch.randn(T, HD) * 0.5

    cos, sin = build_2d_rope_tables(grid_h, grid_w, HD)
    return [
        TensorSpec("qk", [T, HD], torch.bfloat16, init_value=init_qk),
        TensorSpec("cos", [T, HD], torch.float32, init_value=cos),
        TensorSpec("sin", [T, HD], torch.float32, init_value=sin),
        TensorSpec("out", [T, HD], torch.bfloat16, is_output=True),
    ]


def _parse_args():
    parser = argparse.ArgumentParser(description="Step3p7 2D-RoPE apply validation.")
    parser.add_argument("-p", "--platform", default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--smoke", action="store_true", default=False)
    parser.add_argument("--grid", type=int, default=8,
                        help="grid_h=grid_w (default 8 -> T=64 for smoke; "
                             "real vision uses 52 -> T=2704 patches)")
    return parser.parse_args()


def main():
    args = _parse_args()
    from golden import ratio_allclose, run_jit

    T = args.grid * args.grid
    assert T % 8 == 0, f"T={T} must be a multiple of T_TILE=8"
    print(f"--- apply_2d_rope_test: T={T}, HD={HD} ---")
    result = run_jit(
        fn=apply_2d_rope_test,
        specs=build_tensor_specs(T, args.grid, args.grid),
        golden_fn=golden_apply_2d_rope_test,
        runtime_cfg=dict(platform=args.platform, device_id=args.device),
        rtol=1e-3, atol=1e-3,
        compare_fn={"out": ratio_allclose(atol=1e-4, rtol=1.0 / 128)},
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
