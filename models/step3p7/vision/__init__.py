# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Step3p7 **vision** subpackage: vision tower + downsamplers + projector.

Everything in this subpackage is the vision side of step3p7 (the
PerceptionEncoder + 6144->4096 projector). The step3p7 *language model* is
NOT here — it is reused from ``models/step3p5``. Top-level ``models/step3p7``
holds only the integration glue that wires vision output into the LM.

Naming: ``vision_config.py`` / ``vision_weight_loader.py`` carry the
``vision_`` prefix to avoid name collision with the step3p5 LM's own
``config.py`` / ``weight_loader.py``.
"""
