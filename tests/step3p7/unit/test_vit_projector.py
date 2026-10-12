# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Single-card golden UT for step3p7 vit projector (6144->4096 ColPar matmul).

The kernel body + torch golden reference live in
``models/step3p7/vision/vit_projector.py``; this file holds only the ``@pl.jit``
test wrapper, specs, golden adapter, and driver. Static shape
(``V_TOKENS_FINAL`` = 169).

Usage::

    python -m tests.step3p7.unit.test_vit_projector --smoke -p a2a3sim
    python -m tests.step3p7.unit.test_vit_projector -p a2a3 -d 0
"""

import argparse

import pypto.language as pl

from models.step3p7.vision.vision_config import LM_HIDDEN, V_DS2_OUT, V_TOKENS_FINAL, V_TOKENS_FINAL_PAD
from models.step3p7.vision.vit_projector import golden_vit_projector, vit_projector

IN = V_DS2_OUT       # 6144
OUT = LM_HIDDEN      # 4096
N_REAL = V_TOKENS_FINAL       # 169 (real image tokens)
N = V_TOKENS_FINAL_PAD       # 176 (169 zero-padded to 16-align for the cube M-tile)


@pl.jit
def vit_projector_test(
    x: pl.Tensor[[N, IN], pl.BF16],
    w: pl.Tensor[[IN, OUT], pl.BF16],
    out: pl.Out[pl.Tensor[[N, OUT], pl.BF16]],
):
    out = vit_projector(x, w, out)
    return out


def golden_vit_projector_test(tensors):
    tensors["out"][:] = golden_vit_projector(tensors["x"], tensors["w"])


def build_tensor_specs():
    import torch
    from golden import TensorSpec

    def init_x():
        # Real 169 rows random; pad rows 169..175 with zeros (production contract:
        # host zero-pads 169 -> 176 before calling the kernel).
        x = torch.randn(N, IN) * 0.5
        x[N_REAL:] = 0.0
        return x

    def init_w():
        return torch.randn(IN, OUT) / (IN ** 0.5)

    return [
        TensorSpec("x", [N, IN], torch.bfloat16, init_value=init_x),
        TensorSpec("w", [IN, OUT], torch.bfloat16, init_value=init_w),
        TensorSpec("out", [N, OUT], torch.bfloat16, is_output=True),
    ]


def _parse_args():
    parser = argparse.ArgumentParser(description="Step3p7 vit projector validation.")
    parser.add_argument("-p", "--platform", default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--smoke", action="store_true", default=False)
    return parser.parse_args()


def main():
    args = _parse_args()
    from golden import ratio_allclose, run_jit

    print(f"--- vit_projector_test: N_real={N_REAL} -> N_pad={N}, {IN}->{OUT} ---")
    result = run_jit(
        fn=vit_projector_test,
        specs=build_tensor_specs(),
        golden_fn=golden_vit_projector_test,
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
