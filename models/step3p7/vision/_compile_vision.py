# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Compile the step3p7 vision ``@pl.program`` (``Step3p7Vision``) for codegen
verification. Compile-only driver — no NPU execution. Mirrors
``models/step3p5/_compile_moe.py``.

STATUS: will fail at lowering until ``vision_fwd.chip_orch`` body is filled
(it currently raises NotImplementedError). Once the L2 body lands, this is the
compile gate.

Usage (from pypto-lib/)::

    PYPTO_PROG_BUILD_DIR=<BUILD_DIR> \\
      python -m models.step3p7._compile_vision -p a2a3sim
"""
from __future__ import annotations

import argparse
import sys

from pypto import ir
from pypto.ir.distributed_compiled_program import DistributedConfig

from .vision_config import TP_WORLD_SIZE
from .vision_fwd import Step3p7Vision


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--platform", default="a2a3sim",
        choices=["a2a3", "a2a3sim", "a5", "a5sim"],
    )
    args = parser.parse_args()

    prog_name = getattr(Step3p7Vision, "name", None) or type(Step3p7Vision).__name__
    print(f"[vision] compiling {prog_name} tp={TP_WORLD_SIZE}", flush=True)

    dist_cfg = DistributedConfig(
        device_ids=list(range(TP_WORLD_SIZE)),
        num_sub_workers=0,
    )
    compiled = ir.compile(
        Step3p7Vision,
        platform=args.platform,
        distributed_config=dist_cfg,
        skip_ptoas=False,
        dump_passes=True,
    )
    print(f"[vision] OK output_dir={compiled.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
