#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================
"""
"""
import pytest

from contextlib import contextmanager

import pto


@contextmanager
def pto_function(name: str, *args):
    print(f"Entering context: {name}")
    try:
        yield pto.begin_function(name, *args)
    except Exception as e:
        print(f"Caught exception: {e}")
        raise
    finally:
        # TODO(anastasios): make false input param.
        pto.end_function(name, False)
        print(f"Exiting context: {name}")


@pytest.mark.skip(
    reason="RuntimeError: ASSERTION FAILED: currentFunctionPtr_->IsGraphType(GraphType::LEAF_GRAPH)"
)
def test_pybind_context_manager():
    dtype = pto.DataType.DT_FP16
    shape = (8, 8)
    a = pto.tensor(dtype, shape, "tensor_a")
    b = pto.tensor(dtype, shape, "tensor_a")

    c = None
    with pto_function("fnc_name", a, b):
        pto.set_vec_tile_shapes(8, 8)
        c = pto.add(a, b)

    assert isinstance(c, pto.tensor)
