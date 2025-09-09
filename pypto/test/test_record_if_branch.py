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
import pto
import sys
import os

common_dir = os.path.join(os.path.dirname(__file__), '..', 'common')
sys.path.insert(0, os.path.abspath(common_dir))

from utils import dyn_function, loop_function, record_if_branch


# FIXME(anastasios): Fix after pypackage.Monkey patching for now
pto.dyn_function = dyn_function
pto.loop_function = loop_function
pto.cond = record_if_branch


def test_record_if_branch():
    dtype = pto.DataType.DT_FP16
    shape = (32, 32)
    a = pto.tensor(dtype, shape, "tensor_a")
    b = pto.tensor(dtype, shape, "tensor_b")
    c = pto.tensor(dtype, shape, "tensor_c")

    with pto.dyn_function("ADD_IF", [a, b], [c]):
        pto.set_vec_tile_shapes(8, 8)
        loop_range = pto.loop_range(2)
        with pto.loop_function(
            "LOOP",
            "k",
            loop_range,
        ) as rlf:
            for k in rlf:
                if pto.cond(k < 10):
                    c = pto.add(a, b)

    assert isinstance(c, pto.tensor)
