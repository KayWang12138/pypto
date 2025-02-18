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
import os
import pypto
import torch
import numpy as np
from utils.np_compare import detailed_allclose_manual

def scatter_update_golden(value, cache, slots):
    bs = value.shape[0]
    block_number, block_size, n2, head_size = cache.shape
    for bs_idx in range(bs):
        index = slots[bs_idx]
        dim_0 = index // block_size
        dim_1 = index % block_size
        cache[dim_0, dim_1, :] = value[bs_idx][:]


def main():
    test_scatter_update()


@pypto.jit
def scatter_update(key, value, index, key_cache, value_cache):
    # 1. 添加支持动态的config
    pypto.set_host_options(only_codegen=True)

    # 3. 得到动态tensor的shape
    dtype = key.dtype
    b_scalar = index.shape[0]
    b_tile = 2
    b_loop = (b_scalar + b_tile - 1) // b_tile
    n2 = key.shape[1]
    d = key.shape[2]

    block_num_scalar = key_cache.shape[0]
    block_size = value_cache.shape[1]

    print(f'n2 {n2} d {d} b_loop {b_loop}')

    kv_2d_shape = (b_scalar, n2 * d)
    kv_cache_2d_shape = (block_num_scalar * block_size, n2 * d)

    for tmp_idx in pypto.loop(1, name="LOOP_RESHAPE_INPLACE", idx_name="tmp_idx"):
        key_2d = pypto.reshape(key, kv_2d_shape, inplace=True)
        value_2d = pypto.reshape(value, kv_2d_shape, inplace=True)
        key_cache_2d = pypto.reshape(key_cache, kv_cache_2d_shape, inplace=True)
        value_cache_2d = pypto.reshape(value_cache, kv_cache_2d_shape, inplace=True)
    for b_idx in pypto.loop(b_loop, name="LOOP_SCATTER_UPDATE", idx_name="b_idx"):
        def b_loop_func(b_idx):
            pypto.set_vec_tile_shapes(16, 16, 16)
            b_ofs = b_idx * b_tile
            b_valid = (b_scalar - b_idx * b_tile).min(b_tile)
            key_view = pypto.view(key_2d, [b_tile, n2 * d], [b_ofs, 0], valid_shape=[b_valid, n2 * d])
            value_view = pypto.view(value_2d, [b_tile, n2 * d], [b_ofs, 0], valid_shape=[b_valid, n2 * d])
            index_view = pypto.view(index, [b_tile], [b_ofs], valid_shape=[b_valid])
            index_view = pypto.reshape(index_view, [b_tile, 1], valid_shape=[b_valid, 1])
            pypto.set_vec_tile_shapes(16, 128)
            key_cache.move(pypto.scatter_update(key_cache_2d, -2, index_view, key_view))
            value_cache.move(pypto.scatter_update(value_cache_2d, -2, index_view, value_view))

        b_loop_func(b_idx)


def test_scatter_update():
    np.random.seed(0)
    torch.manual_seed(0)
    # 1. 设置参数
    b = 5
    n2 = 1
    d = 128
    block_num = 150
    block_size = 128

    device_id = int(os.environ.get('TILE_FWK_STEST_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    key = torch.rand((b, n2, d), dtype=torch.float16, device=f'npu:{device_id}') * 0 + 99
    value = torch.zeros((b, n2, d), dtype=torch.float16, device=f'npu:{device_id}') * 0 + 99
    index = torch.randperm(block_num * block_size)[:b].to(key.device)
    index = torch.tensor([128, 129, 130, 131, 132]).to(key.device)
    key_cache = torch.rand((block_num, block_size, n2, d), dtype=torch.float16, device=f'npu:{device_id}') * 0 + 2
    value_cache = torch.rand((block_num, block_size, n2, d), dtype=torch.float16, device=f'npu:{device_id}') * 0 + 2
    key_cache_clone = key_cache.clone()
    value_cache_clone = value_cache.clone()

    # 4. 执行kernel并获取结果
    inputs = {
        key: [0],
        value: [0],
        index: [0]
    }
    outputs = {
        key_cache: [0],
        value_cache: [0]
    }
    pto_inputs = [pypto.from_torch(tensor, dynamic_axis=axis) for tensor, axis in inputs.items()]
    pto_outputs = [pypto.from_torch(tensor, dynamic_axis=axis) for tensor, axis in outputs.items()]
    scatter_update(*pto_inputs, *pto_outputs)
    pypto.runtime._device_synchronize()

    # 5. 与PyTorch参考实现对比
    scatter_update_golden(key, key_cache_clone, index)
    scatter_update_golden(value, value_cache_clone, index)
    npu_key_cache_out = key_cache.cpu().tolist()
    npu_value_cache_out = value_cache.cpu().tolist()
    tmp = np.array(npu_key_cache_out).flatten().reshape(block_num * block_size, n2 * d)[:, :2]
    print(f'index {index} \nkey_cache {tmp}')
    # 6. 与PyTorch参考实现对比
    if block_num > 100:
        key_cache_clone_2d = np.array(key_cache_clone.cpu()).flatten().reshape(block_num * block_size, n2 * d)
        value_cache_clone_2d = np.array(value_cache_clone.cpu()).flatten().reshape(block_num * block_size, n2 * d)
        npu_key_cache_out_2d = np.array(npu_key_cache_out).flatten().reshape(block_num * block_size, n2 * d)
        npu_value_cache_out_2d = np.array(npu_value_cache_out).flatten().reshape(block_num * block_size, n2 * d)
        key_cache_clone_2d_select = key_cache_clone_2d[index.cpu().numpy()]
        npu_key_cache_out_2d_select = npu_key_cache_out_2d[index.cpu().numpy()]
        value_cache_clone_2d_select = value_cache_clone_2d[index.cpu().numpy()]
        npu_value_cache_out_2d_select = npu_value_cache_out_2d[index.cpu().numpy()]
        print('5555')
        detailed_allclose_manual(key_cache_clone_2d_select.flatten(), npu_key_cache_out_2d_select.flatten(),
                                 "key cache")
        detailed_allclose_manual(value_cache_clone_2d_select.flatten(), npu_value_cache_out_2d_select.flatten(),
                                 "key cache")
        print('6666')
    else:
        detailed_allclose_manual(np.array(key_cache_clone.cpu()).flatten(), np.array(npu_key_cache_out).flatten(),
                                 "key cache")
        detailed_allclose_manual(np.array(value_cache_clone.cpu()).flatten(), np.array(npu_value_cache_out).flatten(),
                                 "value cache")


if __name__ == "__main__":
    main()
