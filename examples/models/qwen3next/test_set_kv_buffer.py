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
"""
"""

from dataclasses import dataclass
import os
import pypto
import torch
import torch_npu
import numpy as np
from numpy.testing import assert_allclose


# 1. 添加支持动态的config
@pypto.jit(
    host_options={"only_codegen": True}
)
def set_kv_buffer_func(k, v, loc, k_buffer, v_buffer):
    # 3. 得到动态tensor的shape
    bs_loop = k.shape[0]
    n = k_buffer.shape[1]
    d = k_buffer.shape[2]

    # 4. 实现kernel逻辑，循环展开BS动态轴
    for bs_idx in pypto.loop(bs_loop, name="LOOP_ATT_PRE_L0", idx_name="bs_idx"):
        # 5. 通过view得到x_tile、cos_tile、sin_tile
        pypto.set_vec_tile_shapes(128, 128)
        k_tile = pypto.view(k, [1, n * d], [bs_idx, 0])
        v_tile = pypto.view(v, [1, n * d], [bs_idx, 0])
        k_bs = pypto.reshape(k_tile, [1, n, d])
        v_bs = pypto.reshape(v_tile, [1, n, d])

        #将k,v按照loc映射位置分别存入k_buffer,v_buffer
        cacheindex = loc[bs_idx]
        pypto.assemble(k_bs, [cacheindex, 0, 0], k_buffer)
        pypto.assemble(v_bs, [cacheindex, 0, 0], v_buffer)


# 封装pypto实现的函数
def set_kv_buffer(**kwargs):
    k_torch = kwargs.get("k_torch")
    v_torch = kwargs.get("v_torch")
    loc_torch = kwargs.get("loc_torch")
    k_buffer_torch = kwargs.get("k_buffer_torch")
    v_buffer_torch = kwargs.get("v_buffer_torch")
    num_kv_heads = kwargs.get("num_kv_heads")
    # 从入参获取维度
    kv_size = k_torch.shape[-1]
    head_dim = kv_size // num_kv_heads
    # 将k_buffer_torch reshape为3维
    k_buffer_torch = k_buffer_torch.view(-1, num_kv_heads, head_dim)
    v_buffer_torch = v_buffer_torch.view(-1, num_kv_heads, head_dim)
    # 将loc_torch 数据类型变为int32
    loc_int32 = loc_torch.to(torch.int32)
    # 给v_torch分配连续内存
    v_torch_in = v_torch.contiguous()

    inputs = {
        k_torch: [0],
        v_torch_in: [0],
        loc_int32: []
    }
    outputs = {
        k_buffer_torch: [],
        v_buffer_torch: []
    }

    pto_inputs = [pypto.from_torch(tensor, dynamic_axis=axis) for tensor, axis in inputs.items()]
    pto_outputs = [pypto.from_torch(tensor, dynamic_axis=axis) for tensor, axis in outputs.items()]

    set_kv_buffer_func(*pto_inputs, *pto_outputs)


# torch实现的对比代码
def set_kv_buffer_torch(**kwargs):
    k_torch = kwargs.get("k_torch")
    v_torch = kwargs.get("v_torch")
    loc_torch = kwargs.get("loc_torch")
    k_buffer_c = kwargs.get("k_buffer_c")
    v_buffer_c = kwargs.get("v_buffer_c")
    num_kv_heads = kwargs.get("num_kv_heads")

    bs = k_torch.shape[0]
    kv_size = k_torch.shape[-1]
    head_dim = kv_size // num_kv_heads

    #set_kv_cache
    k_torch_c = k_torch.view(-1, num_kv_heads, head_dim)
    v_torch_c = v_torch.view(-1, num_kv_heads, head_dim)
    k_buffer_c = k_buffer_c.view(-1, num_kv_heads, head_dim)
    v_buffer_c = v_buffer_c.view(-1, num_kv_heads, head_dim)
    for i in range(bs):
        k_buffer_c[loc_torch[i]] = k_torch_c[i]
        v_buffer_c[loc_torch[i]] = v_torch_c[i]


def main():
    device_id = int(os.environ.get('TILE_FWK_STEST_DEVICE_ID', 0))
    torch.npu.set_device(device_id)

    # 设置参数
    bs = 64
    head_dim = 256
    kv_size = 256
    max_embbending_size = 262144
    page_size = 128
    page_num = max_embbending_size // page_size + 1
    loc_shape = [bs, ]
    buffer_shape = [page_num, page_size, 1, head_dim]
    k_shape = [bs, kv_size]

    # 准备测试数据
    np.random.seed(0)
    # inputs
    k_torch = torch.empty(k_shape, dtype=torch.bfloat16).uniform_(-1, 1).to(device=f'npu:{device_id}')
    v_torch = torch.empty(k_shape, dtype=torch.bfloat16).uniform_(-1, 1).to(device=f'npu:{device_id}')
    loc_torch = torch.randint(0, max_embbending_size, loc_shape, dtype=torch.int64).to(device=f'npu:{device_id}')
    k_buffer_torch = torch.full(buffer_shape, 9, dtype=torch.bfloat16, device=f'npu:{device_id}')
    v_buffer_torch = torch.full(buffer_shape, 9, dtype=torch.bfloat16, device=f'npu:{device_id}')
    num_kv_heads = 1

    k_buffer_c = k_buffer_torch.clone()
    v_buffer_c = v_buffer_torch.clone()

    set_kv_buffer(
                k_torch=k_torch,
                v_torch=v_torch,
                loc_torch=loc_torch,
                k_buffer_torch=k_buffer_torch,
                v_buffer_torch=v_buffer_torch,
                num_kv_heads=num_kv_heads
                )
    set_kv_buffer_torch(
                    k_torch=k_torch,
                    v_torch=v_torch,
                    loc_torch=loc_torch.clone(),
                    k_buffer_c=k_buffer_c,
                    v_buffer_c=v_buffer_c,
                    num_kv_heads=num_kv_heads
                    )
    # compare result
    assert_allclose(np.array(k_buffer_c.cpu().flatten().tolist()), np.array(k_buffer_torch.cpu().flatten().tolist()),
                    rtol=0.001, atol=0.001)
    assert_allclose(np.array(v_buffer_c.cpu().flatten().tolist()), np.array(v_buffer_torch.cpu().flatten().tolist()),
                    rtol=0.001, atol=0.001)

if __name__ == "__main__":
    main()
