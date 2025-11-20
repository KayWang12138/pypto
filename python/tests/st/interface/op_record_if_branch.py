# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
"""
import pypto


def op_record_if_branch(a, b, c, params):
    tile_shape = params[1]
    pypto.set_vec_tile_shapes(tile_shape[0], tile_shape[1])
    for k in pypto.loop(2, name="LOOP", idx_name="k"):
        if pypto.cond(k < 10):
            c.move(pypto.add(a, b))


def golden_if_branch(a, b):
    return a + b