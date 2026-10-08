# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Pure-torch invariant test for the device-side tap-major im2col gather.

The device kernels ``ds1_im2col_b``/``ds2_im2col_b`` (vision_full_fwd.py) fold
the k3 s2 p1 im2col ONTO the NPU as a tap-major gather (``im2col[r, t*C+c] =
feat[ih, iw, c]``, t=kh*3+kw) over a 1-zero-border padded token layout, with the
conv weight permuted channel-major -> tap-major once on the host. This test
locks in the MATH invariant (dim-agnostic, so small dims are used to stay fast):

    tap-major gather @ permuted weight  ==  channel-major F.unfold @ original weight

The full-dim device correctness is covered end-to-end by
``tools/step3p7/run_vision_full.py`` (Stage 5/6 ISOLATION vs vLLM golden).

Usage::

    python -m tests.step3p7.unit.test_device_im2col
"""

import sys

import torch
import torch.nn.functional as F


def _tapmajor_reference(feat_pad_tokens, c, gp, gout):
    """torch reference of the device tap-major gather.

    feat_pad_tokens: [gp*gp, c] (1-zero-border padded, token row-major).
    Returns [gout*gout, 9*c] tap-major (im2col[r, t*c + c'] = feat[ih*gp+iw, c']).
    """
    x = feat_pad_tokens.reshape(1, gp, gp, c).permute(0, 3, 1, 2)  # [1, c, gp, gp]
    # F.unfold(k3 s2 p0) on the pre-padded input == the device gather (ih=2*oh+kh
    # is always in-bounds on the padded grid, no bounds check). Channel-major.
    cols = F.unfold(x, kernel_size=3, stride=2, padding=0)         # [1, 9*c, gout*gout]
    # channel-major (k=c*9+t) -> tap-major (k=t*c+c)
    cols = cols.reshape(1, c, 9, gout * gout).permute(0, 2, 1, 3).reshape(1, 9 * c, gout * gout)
    return cols.squeeze(0).t()                                      # [gout*gout, 9*c]


def _permute_weight(w, c):
    """[9*c, n] channel-major -> tap-major (matches host `_permute_ds_weight`)."""
    return w.reshape(c, 9, -1).permute(1, 0, 2).reshape(9 * c, -1).contiguous()


def _check(c, h, w, n):
    """Run the invariant on one (c, h, w, n) config."""
    gout = (h + 1) // 2
    gp = h + 2
    torch.manual_seed(0)
    feat = torch.randn(1, c, h, w)                    # NCHW
    wgt = torch.randn(9 * c, n)

    # channel-major (old host path): F.unfold k3 s2 p1
    cm = F.unfold(feat, kernel_size=3, stride=2, padding=1).squeeze(0).t()  # [gout^2, 9c]

    # tap-major (new device path): pad + F.unfold p0 + permute
    feat_pad = F.pad(feat, (1, 1, 1, 1))              # [1, c, h+2, w+2]
    feat_pad_tokens = feat_pad.permute(0, 2, 3, 1).reshape(-1, c)
    tm = _tapmajor_reference(feat_pad_tokens, c, gp, gout)
    w_perm = _permute_weight(wgt, c)

    # 1) exact gather mapping: tm[:, t*c+c'] == cm[:, c'*9+t]
    for t in range(9):
        for cc in range(c):
            d = (tm[:, t * c + cc] - cm[:, cc * 9 + t]).abs().max().item()
            assert d == 0.0, f"(c={c},h={h},w={w}) gather mismatch t={t} c={cc}: {d}"

    # 2) matmul equivalence (only FP32 accumulation order shifts; ~1e-5 for the
    # larger configs, far below the real 1/128 ~ 7.8e-3 rtol).
    out_cm = cm.float() @ wgt.float()
    out_tm = tm.float() @ w_perm.float()
    d = (out_cm - out_tm).abs().max().item()
    assert d < 1e-4, f"(c={c},h={h},w={w}) matmul mismatch: {d}"


def main():
    # (c_in, h, w, c_out) — dim-agnostic invariant, so small dims are enough.
    for cfg in [(8, 6, 6, 5), (16, 10, 10, 7), (3, 8, 8, 4), (32, 12, 12, 16)]:
        _check(*cfg)
    print("[test_device_im2col] PASS (tap-major gather @ permuted w == F.unfold @ w)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
