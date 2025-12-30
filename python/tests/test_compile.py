#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import os
import pypto
import torch
import torch_npu


@pypto.jit(runtime_options={"run_mode": 0})
def simple_add(a, b, c, tiling=None):
    """最简单的add计算函数"""
    pypto.set_vec_tile_shapes(tiling, tiling)
    c[:] = a + b


def test_simple_add():
    """测试简单的add计算"""
    torch.npu.set_device(6)
    tiling = 4
    n, m = tiling * 1, tiling * 1

    # 准备数据
    a_data = torch.ones((n, m), dtype=torch.float32, device=f'npu:{6}') * 2
    b_data = torch.ones((n, m), dtype=torch.float32, device=f'npu:{6}') * 3
    c_data = torch.zeros((n, m), dtype=torch.float32, device=f'npu:{6}')

    # 定义输入和输出
    inputs = [a_data, b_data]
    outputs = [c_data]
    pto_inputs = [pypto.from_torch(tensor, f"IN_{idx}") for idx, tensor in enumerate(inputs)]
    pto_outputs = [pypto.from_torch(tensor, f"OUT_{idx}") for idx, tensor in enumerate(outputs)]

    # 调用带jit的add函数
    simple_add(pto_inputs[0], pto_inputs[1], pto_outputs[0], tiling)

    print(c_data)
    print("test_simple_add passed!")

if __name__ == "__main__":
    test_simple_add()
