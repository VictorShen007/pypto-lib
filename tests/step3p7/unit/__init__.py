# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Single-kernel golden UTs for step3p7 vision primitives.

Each ``test_<kernel>.py`` is a ``__main__`` script (like ``tests/step3p5/unit/``;
the ``test_`` prefix is a filename convention, NOT a pytest hook)::

    python -m tests.step3p7.unit.test_layernorm --smoke -p a2a3sim   # compile-only
    python -m tests.step3p7.unit.test_layernorm -p a2a3 -d 0          # real card + golden

The kernel body + torch golden reference live in ``models/step3p7/vision/``;
these files hold only the ``@pl.jit`` test wrapper, ``build_tensor_specs``,
the golden adapter, and the ``__main__`` driver.
"""
