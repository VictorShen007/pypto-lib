# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Tests for the step3p7 **vision** side (vision tower + downsamplers + projector).

Mirrors the ``tests/step3p5/`` layout. The language model is NOT tested here —
it is reused from ``models/step3p5``. Currently only ``unit/`` (single-kernel
golden UTs) is populated; ``precision/`` / ``ci/`` / ``harnesses/`` are added
as the L3/L4 integration grows.
"""
