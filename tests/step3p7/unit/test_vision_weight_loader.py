# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Host-side UT for step3p7 vision weight loader (no NPU).

Loads the vision subtree (BF16) for one rank and asserts the key set +
per-tensor shapes match the kernel contract (mirrors
``tests/step3p5/unit/test_weight_loader_w8a8.py``). Supports both the BF16
(``model.safetensors.index.json``) and W8A8
(``quant_model_weights.safetensors.index.json``) checkpoints.

Usage::

    python -m tests.step3p7.unit.test_vision_weight_loader \\
        --ckpt <CKPT>
"""

import argparse


def _parse_args():
    parser = argparse.ArgumentParser(description="Step3p7 vision weight loader UT.")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--tp", type=int, default=8)
    return parser.parse_args()


def main():
    args = _parse_args()

    from models.step3p7.vision.vision_config import (
        V_LAYERS, V_TOKENS, V_WIDTH, V_HEADS, V_HEADS_LOCAL,
        V_HEAD_DIM_PAD, V_MLP_HIDDEN_LOCAL_PAD, V_DS1_OUT, V_DS2_OUT, LM_HIDDEN,
    )
    from models.step3p7.vision.vision_weight_loader import (
        KEY_V_PATCH_EMBED, KEY_V_POSEMB, KEY_V_QKV, KEY_V_O, KEY_V_FC1, KEY_V_FC2,
        KEY_V_DS1, KEY_V_DS2, KEY_V_PROJ,
        load_step3p7_vision_weights_for_rank,
    )

    bundle = load_step3p7_vision_weights_for_rank(args.ckpt, args.rank, args.tp)
    print(f"[loader] {len(bundle)} keys loaded from {args.ckpt} (rank={args.rank}, tp={args.tp})")

    # Expected key count: front(4) + per-block(14 * V_LAYERS) + downsamplers(4) + projector(1)
    expected_n = 4 + 14 * V_LAYERS + 4 + 1
    assert len(bundle) == expected_n, f"key count {len(bundle)} != expected {expected_n}"

    HD = V_WIDTH // V_HEADS                              # 96
    QKV_OUT = 3 * V_HEADS_LOCAL * HD                    # 576 per rank (3 * 2 heads * 96)
    OP_OUT = V_HEADS_LOCAL * V_HEAD_DIM_PAD             # 256 per rank (2 heads * 128 padded)
    # Shape contract (rank-0, tp=8 slices).
    checks = [
        (KEY_V_PATCH_EMBED, (592, V_WIDTH)),          # 588 -> 592 padded
        (KEY_V_POSEMB, (V_TOKENS, V_WIDTH)),          # (2704, 1536)
        (KEY_V_QKV.format(L=0), (V_WIDTH, QKV_OUT)),  # per-rank qkv (1536, 576)
        (KEY_V_O.format(L=0), (OP_OUT, V_WIDTH)),     # out_proj per rank (256, 1536)
        (KEY_V_FC1.format(L=0), (V_WIDTH, V_MLP_HIDDEN_LOCAL_PAD)),    # fc1 per-rank (1536, 1152)
        (KEY_V_FC2.format(L=0), (V_MLP_HIDDEN_LOCAL_PAD, V_WIDTH)),   # fc2 per-rank (1152, 1536)
        (KEY_V_DS1, (9 * V_WIDTH, V_DS1_OUT)),         # (13824, 3072)
        (KEY_V_DS2, (9 * V_DS1_OUT, V_DS2_OUT)),      # (27648, 6144)
        (KEY_V_PROJ, (V_DS2_OUT, LM_HIDDEN)),         # (6144, 4096)
    ]
    fail = 0
    for key, want in checks:
        if key not in bundle:
            print(f"  MISSING key: {key}")
            fail += 1
            continue
        got = tuple(bundle[key].shape)
        status = "OK" if got == want else "SHAPE MISMATCH"
        if got != want:
            fail += 1
        print(f"  {key}: {got} {status}" + (f" (expected {want})" if got != want else ""))

    # all 47 layers present
    for L in range(V_LAYERS):
        for tmpl in (KEY_V_QKV, KEY_V_O, KEY_V_FC1, KEY_V_FC2):
            k = tmpl.format(L=L)
            if k not in bundle:
                print(f"  MISSING layer key: {k}")
                fail += 1

    if fail:
        print(f"\n[loader] FAIL ({fail} problems)")
        return 1
    print(f"\n[loader] PASS ({len(bundle)} keys, all shapes match)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
