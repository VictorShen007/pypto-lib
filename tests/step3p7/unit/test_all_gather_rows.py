# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Pure-torch UT for the row-axis all_gather (dim=0) DP merge + ``_shard_by_rank``.

The device primitive lives in two places:
  * ``models/step3p5/collectives.tp_all_gather_rows`` (@pl.jit.inline reference)
    + ``_mock_tp_all_gather_rows`` (pure-torch reference).
  * ``models/step3p7/vision/vision_full_fwd.AllGatherRowsProg`` (@pl.program,
    the usable self-method InCore copy).

This UT validates the pure-torch reference of the offset math (rank-major
row-axis concat) and the host ``_shard_by_rank`` DP shard helper — both plain
Python, no NPU. It mirrors the ``_mock_tp_all_gather_rows`` smoke in
``collectives._run_mocks`` but as an importable test.

Usage::

    python -m tests.step3p7.unit.test_all_gather_rows
"""


def _test_mock_all_gather_rows():
    import torch
    from models.step3p5.collectives import TP_WORLD_SIZE, _mock_tp_all_gather_rows

    g = TP_WORLD_SIZE
    rows, cols = 5, 16

    # Random data: every rank's [r, :, :] slice must equal the rank-major concat.
    x = torch.randn(g, rows, cols, dtype=torch.bfloat16)
    out = _mock_tp_all_gather_rows(x)
    assert out.shape == (g, g * rows, cols), f"shape {tuple(out.shape)}"
    concat = x.reshape(g * rows, cols)
    for r in range(g):
        torch.testing.assert_close(out[r], concat)

    # Value-marker check: rank r's distinct block (value r+1) must land at rows
    # [r*rows:(r+1)*rows] of the concat — i.e. the row-axis offset math is right.
    marker = torch.stack([
        torch.full((rows, cols), float(r) + 1.0, dtype=torch.bfloat16)
        for r in range(g)
    ])
    mout = _mock_tp_all_gather_rows(marker)
    for r in range(g):
        torch.testing.assert_close(mout[0, r * rows:(r + 1) * rows], marker[r])
    print(f"[OK] _mock_tp_all_gather_rows ({g} ranks, {rows}x{cols})")


def _test_shard_by_rank():
    import torch
    from tools.step3p7.run_vision_full import _shard_by_rank

    images = torch.arange(16).reshape(8, 2)  # 8 images of 2 elems
    assert torch.equal(_shard_by_rank(images, 0, 1), images[0:1])
    assert torch.equal(_shard_by_rank(images, 7, 1), images[7:8])
    assert torch.equal(_shard_by_rank(images, 1, 2), images[2:4])
    assert torch.equal(_shard_by_rank(images, 3, 2), images[6:8])
    print("[OK] _shard_by_rank (per_rank 1 and 2)")


def main():
    _test_mock_all_gather_rows()
    _test_shard_by_rank()
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
