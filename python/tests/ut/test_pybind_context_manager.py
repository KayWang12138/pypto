#!/usr/bin/env python3
# coding: utf-8
# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================
"""
"""
import pto


def test_pybind_context_manager():
    dtype = pto.DT_FP16
    shape = (8, 8)
    a = pto.tensor(shape, dtype, "tensor_a")
    b = pto.tensor(shape, dtype, "tensor_a")
    c = pto.tensor(shape, dtype, "tensor_c")

    with pto.function("fnc_name", a, b, static=True):
        pto.set_vec_tile_shapes(8, 8)
        c.move(pto.add(a, b))

    assert isinstance(c, pto.tensor)
