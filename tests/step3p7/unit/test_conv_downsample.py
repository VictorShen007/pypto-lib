# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Single-card golden UT for step3p7 conv downsamplers (padded + bias, matching L3 full-pipeline).

Tests the FIXED ds1_pad/ds2_pad kernels from vision_full_fwd.py (N padded to mult-of-16,
conv bias added) — the SAME kernels used in the L3 full-pipeline driver. This aligns
L1 with L3 (no L1-broken-L3-passing contradiction).

Two 3x3 s2 p1 convs (im2col@host -> matmul + bias):
  - ds1_pad: 1536 -> 3072 (grid 676 -> pad 688)
  - ds2_pad: 3072 -> 6144 (grid 169 -> pad 176)

Usage::

    python -m tests.step3p7.unit.test_conv_downsample --smoke -p a2a3sim
    python -m tests.step3p7.unit.test_conv_downsample -p a2a3 -d 0
    python -m tests.step3p7.unit.test_conv_downsample -p a2a3 -d 0 --kernel 1   # only ds1
"""

import argparse

import pypto.language as pl

from models.step3p7.vision.vision_config import (
    V_DS1_OUT, V_DS2_OUT, V_TOKENS_FINAL_PAD, V_WIDTH, V_GRID_DS1,
)
from models.step3p7.vision.vision_full_fwd import (
    ds1_full, ds2_full, V_GRID_DS1_PAD,
)

# ds1: 1536 -> 3072
N1 = V_GRID_DS1_PAD                       # 688 (676 zero-padded to 16-align)
K1 = 9 * V_WIDTH                          # 9 * 1536 = 13824
M1 = V_DS1_OUT                            # 3072
# ds2: 3072 -> 6144
N2 = V_TOKENS_FINAL_PAD                   # 176 (169 zero-padded)
K2 = 9 * V_DS1_OUT                        # 9 * 3072 = 27648
M2 = V_DS2_OUT                            # 6144


def _golden_ds1(tensors):
    """torch golden: im2col @ w + bias (FP32 acc -> BF16)."""
    import torch
    out = (tensors["im2col"].float() @ tensors["w"].float()
           + tensors["bias"].float()).to(torch.bfloat16)
    tensors["out"][:] = out


def _golden_ds2(tensors):
    import torch
    out = (tensors["im2col"].float() @ tensors["w"].float()
           + tensors["bias"].float()).to(torch.bfloat16)
    tensors["out"][:] = out


def _specs(n, k, m):
    import torch
    from golden import TensorSpec

    def init_im2col():
        x = torch.randn(n, k) * 0.5
        # zero-pad the real rows (the padding rows are zeros — for ds1 676->688, ds2 169->176)
        real_n = (n // 16 * 16)  # actual token count before pad (approximate for test)
        return x  # full random is fine for standalone test (tests matmul correctness)

    def init_w():
        return torch.randn(k, m) / (k ** 0.5)

    def init_bias():
        return torch.randn(1, m) * 0.02  # small bias, matching real magnitude

    return [
        TensorSpec("im2col", [n, k], torch.bfloat16, init_value=init_im2col),
        TensorSpec("w", [k, m], torch.bfloat16, init_value=init_w),
        TensorSpec("bias", [1, m], torch.bfloat16, init_value=init_bias),
        TensorSpec("out", [n, m], torch.bfloat16, is_output=True),
    ]


def _parse_args():
    parser = argparse.ArgumentParser(description="Step3p7 conv downsample validation (padded + bias).")
    parser.add_argument("-p", "--platform", default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--smoke", action="store_true", default=False)
    parser.add_argument("--kernel", type=int, default=0, choices=[0, 1, 2],
                        help="0=both (default), 1=ds1 only, 2=ds2 only")
    return parser.parse_args()


def _run(fn, specs, golden_fn, args, atol, rtol):
    from golden import ratio_allclose, run_jit
    return run_jit(
        fn=fn, specs=specs, golden_fn=golden_fn,
        runtime_cfg=dict(platform=args.platform, device_id=args.device),
        rtol=rtol, atol=atol,
        compare_fn={"out": ratio_allclose(atol=atol, rtol=rtol, max_error_ratio=0.01)},
        compile_only=args.smoke or args.platform.endswith("sim"),
    )


def main():
    args = _parse_args()
    ok = True
    if args.kernel in (0, 1):
        print(f"--- ds1_pad_test: N={N1}, {K1}->{M1} (+bias) ---")
        r = _run(ds1_full, _specs(N1, K1, M1), _golden_ds1, args, atol=6e-3, rtol=6e-3)
        print(f"  ds1: {'PASS' if r.passed else 'FAIL'}")
        if not r.passed and r.error:
            print(r.error[:300])
        ok = ok and r.passed
    if args.kernel in (0, 2):
        print(f"--- ds2_pad_test: N={N2}, {K2}->{M2} (+bias) ---")
        r = _run(ds2_full, _specs(N2, K2, M2), _golden_ds2, args, atol=6e-3, rtol=6e-3)
        print(f"  ds2: {'PASS' if r.passed else 'FAIL'}")
        if not r.passed and r.error:
            print(r.error[:300])
        ok = ok and r.passed
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
