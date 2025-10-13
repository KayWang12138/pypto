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


def init_tensors():
    dtype = pto.DataType.DT_FP32
    shape = (128, 128)
    a = pto.tensor(dtype, shape, "a")
    b = pto.tensor(dtype, shape, "b")
    c = pto.tensor(dtype, shape, "c")
    return a, b, c


def main():
    a, b, _ = init_tensors()
    with pto.dyn_function("main", [a], [b], []):
        pto.set_vec_tile_shapes(16, 16)
        for in_idx in pto.range(5, name="in_loop1"):
            if pto.cond(in_idx < 2):
                b.move(pto.add(a, a))

        for in_idx in pto.range(5, name="in_loop2"):
            if pto.cond(in_idx < 2):
                b.move(pto.add(b, a))

        for idx in pto.range(5, name="in_loop2"):
            if pto.cond(idx < 2):
                b.move(pto.add(b, a))


if __name__ == "__main__":
    main()
