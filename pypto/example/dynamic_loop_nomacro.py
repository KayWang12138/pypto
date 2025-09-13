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

from utils import dyn_function, loop_function, record_if_branch


# FIXME(anastasios): Fix after pypackage.Monkey patching for now
pto.dyn_function = dyn_function
pto.loop_function = loop_function
pto.cond = record_if_branch


def main():
    dtype = pto.DataType.DT_FP16
    shape = (128, 128)
    a = pto.tensor(dtype, shape, "PTO_TENSOR_a")
    b = pto.tensor(dtype, shape, "PTO_TENSOR_b")

    with pto.dyn_function("MAIN", [a], [b]):
        pto.set_vec_tile_shapes(64, 64)
        with pto.loop_function("Dynamic", "k", pto.loop_range_(10)) as rlf:
            for k in rlf:
                b.move(pto.add(a, a))

                if pto.cond(k < 2):
                    b.move(pto.add(b, a))
                else:
                    b.move(pto.sub(b, a))

                if pto.cond(k < 5):
                    b.move(pto.mul(b, a))
                else:
                    b.move(pto.div(b, a))
                b.move(pto.sub(b, a))
    print(pto.dump())
    assert isinstance(b, pto.tensor)


if __name__ == "__main__":
    main()
