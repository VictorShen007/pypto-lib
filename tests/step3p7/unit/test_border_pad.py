# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Pure-torch invariant test for the device-side 1-zero-border pad.

The device kernels ``ds1_border_pad_b`` / ``ds2_border_pad_b`` (vision_full_fwd.py)
moved the host ``F.pad`` (1-pixel zero spatial border) ONTO the NPU: raw token
layout [g*g, c] -> padded [g+2 x g+2, c] with interior token (ih,iw) landing at
(ih+1, iw+1), and the 4 border edges explicitly zeroed (the scratch is NOT
zero-initialized). This test locks in the INDEX MATH (dim-agnostic, small dims to
stay fast) against the torch ``F.pad`` ground truth. The full-dim device
correctness is covered end-to-end by ``tools/step3p7/run_vision_full.py``
(Stage 5/6 ISOLATION vs vLLM golden).

The reference mirrors the kernel's explicit loops (interior gather + per-edge
zero), and starts from a NaN scratch so a missed border/interior write is caught
as a NaN diff rather than silently passing.

Usage::

    python -m tests.step3p7.unit.test_border_pad
"""

import sys

import torch
import torch.nn.functional as F


def _border_pad_reference(raw_tokens, c, g, vb=1, active=1):
    """torch reference of ds1/ds2_border_pad_b (explicit index math).

    raw_tokens: [vb*g*g, c] row-major (token = ih*g + iw).
    Returns [vb*(g+2)*(g+2), c] with a 1-zero spatial border.
    """
    gp = g + 2
    # Mirror `pl.create_tensor` (NOT zero-initialized): NaN scratch so any
    # position the kernel forgets to write stays NaN and fails the compare.
    padded = torch.full((vb * gp * gp, c), float("nan"), dtype=raw_tokens.dtype)
    for b in range(active):
        rb = b * g * g
        pb = b * gp * gp
        # interior gather: (ih,iw) -> (ih+1, iw+1)
        for ih in range(g):
            for iw in range(g):
                padded[pb + (ih + 1) * gp + (iw + 1)] = raw_tokens[rb + ih * g + iw]
        # top (ih=0) + bottom (ih=gp-1) rows
        for iw in range(gp):
            padded[pb + iw] = 0.0
            padded[pb + (gp - 1) * gp + iw] = 0.0
        # left (iw=0) + right (iw=gp-1) cols, interior rows ih=1..g
        for ih in range(g):
            padded[pb + (ih + 1) * gp] = 0.0
            padded[pb + (ih + 1) * gp + (gp - 1)] = 0.0
    return padded


def _check(c, g, vb=3, active=2):
    torch.manual_seed(0)
    raw = torch.randn(vb * g * g, c)
    got = _border_pad_reference(raw, c, g, vb, active)

    # Ground truth: F.pad 1-ring zero border on the [vb, c, g, g] view, then back
    # to token row-major.
    golden = raw.reshape(vb, g, g, c).permute(0, 3, 1, 2)        # [vb, c, g, g]
    golden = F.pad(golden, (1, 1, 1, 1))                         # [vb, c, g+2, g+2]
    golden = golden.permute(0, 2, 3, 1).reshape(-1, c)           # [vb*(g+2)^2, c]

    # Active images must match exactly; inactive images (b>=active) are never
    # written by the `if b < active` guard, so mask the compare to [0:active].
    n_active = active * (g + 2) * (g + 2)
    d = (got[:n_active] - golden[:n_active]).abs().max().item()
    assert d == 0.0, f"(c={c},g={g}) border-pad mismatch vs F.pad: {d}"
    # NaN check: every active padded position got written (no missed edge).
    assert not torch.isnan(got[:n_active]).any(), f"(c={c},g={g}) unwritten positions"


def main():
    # (c_in, grid) — dim-agnostic; covers even/odd g and the 26/52 real grids.
    for cfg in [(8, 6), (16, 10), (3, 7), (32, 5), (1536, 52), (3072, 26)]:
        _check(*cfg)
    print("[test_border_pad] PASS (device 1-zero-border pad == torch F.pad)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
