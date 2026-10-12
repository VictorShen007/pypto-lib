# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Step3p7 vision **full-pipeline** compile-only driver (a2a3sim, no NPU).

Compiles the full-pipeline version: 5 @pl.jit stage wrappers (patch_embed_full,
ln_pre_full, ds1_full, ds2_full, proj_full) + 5 replicated @pl.program stages
(PatchEmbedProg/LnPreProg/Ds1Prog/Ds2Prog/ProjProg, B' fix) + the reused tower
(Step3p7Vision). @pl.jit stages use ``run_jit(compile_only=True)`` (golden
harness); the @pl.program stages + tower use ``ir.compile``. Original
``_compile_vision.py`` untouched.

Usage::

    PYPTO_PROG_BUILD_DIR=<BUILD_DIR> \\
      python -m models.step3p7.vision._compile_vision_full -p a2a3sim
"""
from __future__ import annotations

import argparse
import sys

from pypto import ir
from pypto.ir.distributed_compiled_program import DistributedConfig

from .vision_config import (
    V_TOKENS, V_WIDTH, V_DS1_OUT, V_DS2_OUT, V_TOKENS_FINAL_PAD, TP_WORLD_SIZE,
    V_GRID_DS1, LM_HIDDEN,
)
from .vision_full_fwd import (
    patch_embed_full, ln_pre_full, ds1_full, ds2_full, proj_full,
    Step3p7Vision, V_GRID_DS1_PAD, PE_K_PAD, PE_ROWS_PAD,
    PatchEmbedProg, LnPreProg, Ds1Prog, Ds2Prog, ProjProg,
)
from .vision_fwd_patch import Step3p7VisionPatch


def _specs_for(name):
    """Dummy TensorSpecs for compile-only (shapes from kernel signatures)."""
    import torch
    from golden import TensorSpec
    z = torch.zeros  # compile-only: init not materialized
    if name == "patch_embed_full":
        return [
            TensorSpec("x", [PE_ROWS_PAD, PE_K_PAD], torch.bfloat16, init_value=lambda: z(PE_ROWS_PAD, PE_K_PAD, dtype=torch.bfloat16)),
            TensorSpec("w", [PE_K_PAD, V_WIDTH], torch.bfloat16, init_value=lambda: z(PE_K_PAD, V_WIDTH, dtype=torch.bfloat16)),
            TensorSpec("out", [PE_ROWS_PAD, V_WIDTH], torch.bfloat16, is_output=True),
        ]
    if name == "ln_pre_full":
        return [
            TensorSpec("x", [V_TOKENS, V_WIDTH], torch.bfloat16, init_value=lambda: z(V_TOKENS, V_WIDTH, dtype=torch.bfloat16)),
            TensorSpec("gamma", [V_WIDTH], torch.float32, init_value=lambda: z(V_WIDTH, dtype=torch.float32)),
            TensorSpec("beta", [V_WIDTH], torch.float32, init_value=lambda: z(V_WIDTH, dtype=torch.float32)),
            TensorSpec("out", [V_TOKENS, V_WIDTH], torch.bfloat16, is_output=True),
        ]
    if name == "ds1_full":
        return [
            TensorSpec("im2col", [V_GRID_DS1_PAD, 9 * V_WIDTH], torch.bfloat16, init_value=lambda: z(V_GRID_DS1_PAD, 9 * V_WIDTH, dtype=torch.bfloat16)),
            TensorSpec("w", [9 * V_WIDTH, V_DS1_OUT], torch.bfloat16, init_value=lambda: z(9 * V_WIDTH, V_DS1_OUT, dtype=torch.bfloat16)),
            TensorSpec("bias", [1, V_DS1_OUT], torch.bfloat16, init_value=lambda: z(1, V_DS1_OUT, dtype=torch.bfloat16)),
            TensorSpec("out", [V_GRID_DS1_PAD, V_DS1_OUT], torch.bfloat16, is_output=True),
        ]
    if name == "ds2_full":
        return [
            TensorSpec("im2col", [V_TOKENS_FINAL_PAD, 9 * V_DS1_OUT], torch.bfloat16, init_value=lambda: z(V_TOKENS_FINAL_PAD, 9 * V_DS1_OUT, dtype=torch.bfloat16)),
            TensorSpec("w", [9 * V_DS1_OUT, V_DS2_OUT], torch.bfloat16, init_value=lambda: z(9 * V_DS1_OUT, V_DS2_OUT, dtype=torch.bfloat16)),
            TensorSpec("bias", [1, V_DS2_OUT], torch.bfloat16, init_value=lambda: z(1, V_DS2_OUT, dtype=torch.bfloat16)),
            TensorSpec("out", [V_TOKENS_FINAL_PAD, V_DS2_OUT], torch.bfloat16, is_output=True),
        ]
    if name == "proj_full":
        return [
            TensorSpec("x", [V_TOKENS_FINAL_PAD, V_DS2_OUT], torch.bfloat16, init_value=lambda: z(V_TOKENS_FINAL_PAD, V_DS2_OUT, dtype=torch.bfloat16)),
            TensorSpec("w", [V_DS2_OUT, LM_HIDDEN], torch.bfloat16, init_value=lambda: z(V_DS2_OUT, LM_HIDDEN, dtype=torch.bfloat16)),
            TensorSpec("out", [V_TOKENS_FINAL_PAD, LM_HIDDEN], torch.bfloat16, is_output=True),
        ]
    raise ValueError(name)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", default="a2a3sim",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    args = parser.parse_args()

    from golden import run_jit
    ok = True
    fns = [
        (patch_embed_full, "patch_embed_full"),
        (ln_pre_full,      "ln_pre_full"),
        (ds1_full,         "ds1_full"),
        (ds2_full,         "ds2_full"),
        (proj_full,        "proj_full"),
    ]
    for fn, name in fns:
        try:
            r = run_jit(fn=fn, specs=_specs_for(name),
                        runtime_cfg=dict(platform=args.platform, device_id=0),
                        compile_only=True)
            print(f"[vision_full] {name}: {'OK' if r.passed else 'FAIL'}", flush=True)
            if not r.passed and r.error:
                print(f"  {r.error[:200]}", flush=True)
            ok = ok and r.passed
        except Exception as exc:  # noqa: BLE001
            print(f"[vision_full] {name}: EXC {exc}", flush=True)
            ok = False

    # Tower (TP=8 @pl.program, reused from vision_fwd).
    dist_cfg = DistributedConfig(device_ids=list(range(TP_WORLD_SIZE)), num_sub_workers=0)
    prog_name = getattr(Step3p7Vision, "name", None) or type(Step3p7Vision).__name__
    print(f"[vision_full] compiling tower {prog_name} tp={TP_WORLD_SIZE}", flush=True)
    try:
        compiled = ir.compile(
            Step3p7Vision, platform=args.platform,
            distributed_config=dist_cfg, skip_ptoas=False,
        )
        print(f"[vision_full] tower: OK output_dir={compiled.output_dir}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[vision_full] tower: FAIL {exc}", flush=True)
        ok = False

    # 8-card REPLICATED @pl.program stages (B' fix) — same ir.compile path as
    # the tower; no specs needed (shapes resolve from the per_rank annotations).
    replicated = [
        (PatchEmbedProg, "PatchEmbedProg"),
        (LnPreProg,      "LnPreProg"),
        (Ds1Prog,        "Ds1Prog"),
        (Ds2Prog,        "Ds2Prog"),
        (ProjProg,       "ProjProg"),
    ]
    for prog, name in replicated:
        print(f"[vision_full] compiling {name} (replicated tp={TP_WORLD_SIZE})", flush=True)
        try:
            compiled = ir.compile(
                prog, platform=args.platform,
                distributed_config=dist_cfg, skip_ptoas=False,
            )
            print(f"[vision_full] {name}: OK output_dir={compiled.output_dir}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[vision_full] {name}: FAIL {exc}", flush=True)
            ok = False

    # Patch-path tower (Phase 1a step 1: vT=1296, no B axis yet — structurally
    # the global tower specialized to the 36x36 grid). Validates the patch
    # specialization codegens before the B-axis retrofit. Reuses the SAME
    # dist_cfg (8 cards). See vision_fwd_patch.py docstring.
    patch_name = getattr(Step3p7VisionPatch, "name", None) or type(Step3p7VisionPatch).__name__
    print(f"[vision_full] compiling patch tower {patch_name} tp={TP_WORLD_SIZE}", flush=True)
    try:
        compiled = ir.compile(
            Step3p7VisionPatch, platform=args.platform,
            distributed_config=dist_cfg, skip_ptoas=False,
        )
        print(f"[vision_full] patch tower: OK output_dir={compiled.output_dir}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[vision_full] patch tower: FAIL {exc}", flush=True)
        ok = False

    print(f"[vision_full] {'OK' if ok else 'FAIL'} (5 @pl.jit + 5 @pl.program + tower + patch tower)", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
