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
from .test_base import BaseTest

dtype = pto.DT_FP16
shape = (64, 64)
tiles = (32, 32)


class TestOperator(BaseTest):

    def test_sin(self):
        A = pto.tensor(shape, dtype, 'A')
        C = pto.tensor(shape, dtype, 'C')
        with pto.function("sign", [A], [C]):
            pto.set_vec_tile_shapes(*tiles)
            C[:] = pto.sin(A)

    def test_cos(self):
        A = pto.tensor(shape, dtype, 'A')
        C = pto.tensor(shape, dtype, 'C')
        with pto.function("cos", [A], [C]):
            pto.set_vec_tile_shapes(*tiles)
            C[:] = pto.cos(A)

    def test_sigmoid(self):
        A = pto.tensor(shape, dtype, 'A')
        C = pto.tensor(shape, dtype, 'C')
        with pto.function("sigmoid", [A], [C]):
            pto.set_vec_tile_shapes(*tiles)
            C[:] = pto.sigmoid(A)

    def test_softmax(self):
        A = pto.tensor(shape, dtype, 'A')
        C = pto.tensor(shape, dtype, 'C')
        with pto.function("softmax", [A], [C]):
            pto.set_vec_tile_shapes(*tiles)
            C[:] = pto.softmax(A, dim=-1)
