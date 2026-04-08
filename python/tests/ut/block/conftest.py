# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Shared fixtures for unit tests."""

from pathlib import Path
import pytest

def duration_estimate(seconds: float):
    """Annotate a test case with an estimated duration."""

    def decorator(func):
        func.duration_estimate = seconds
        return func

    return decorator


@pytest.fixture(autouse=True)
def pass_verification_context(request):
    """Enable pass verification only for migrated source-compatible UT areas."""

    path = Path(str(request.fspath)).as_posix()
    marker = "python/tests/ut/block/"
    if marker in path:
        from pypto_block.pypto_core import passes

        with passes.PassContext([passes.VerificationInstrument(passes.VerificationMode.BEFORE_AND_AFTER)]):
            yield
        return
    yield
