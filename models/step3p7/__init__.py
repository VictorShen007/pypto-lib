# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Step3p7 model: VIT (vision) + language model (reused from step3p5).

Package layout (split so readers can tell vision apart from the LM):

- ``models/step3p7/vision/`` — **vision-only** kernels: the vision tower
  (47-layer transformer), the two conv downsamplers, and the 6144->4096
  projector. All files are named with a ``vision_`` / ``vit_`` / ``conv_`` /
  ``rope2d`` prefix or are otherwise unambiguously vision. The vision config
  is ``vision/vision_config.py`` and vision weights load via
  ``vision/vision_weight_loader.py`` (``vision_`` prefix avoids collision with
  the step3p5 LM's ``config.py`` / ``weight_loader.py``).
- ``models/step3p7/`` (this top level) — reserved for the *integration glue*
  that wires vision output (image_features) into the step3p5 LM
  (holder scatter, combined VIT+LM program, etc.).
- The step3p7 *language model* is **not** re-implemented here; it is reused
  verbatim from ``models/step3p5`` (decode / prefill / MoE / attention).

See the repo-root ``plan.md`` for the full design and ``step3p7_vit_test.md``
for the L0-L4 test plan.
"""
