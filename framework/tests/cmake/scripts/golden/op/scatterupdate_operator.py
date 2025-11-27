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
""" ScatterUpdate Operator 相关用例 Golden 生成逻辑.

本脚本有 2 种执行模式:
1. CI批跑时, 由 tests/cmake/scripts/golden_ctrl.py 调用, 为避免日志过多, 此时 logging 级别为 logging.INFO;
2. 单独调试时, 本脚本单独被调用, 此时 logging 级别为 logging.DEBUG;
"""
import sys
import logging
from pathlib import Path
from typing import List

import numpy as np
from ml_dtypes import bfloat16
np.random.seed(0)

if __name__ == "__main__":
    """ 单独调试时配置 """
    # 日志级别
    logging.basicConfig(format='%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s: %(message)s',
                        level=logging.DEBUG)
    # 系统 import 路径
    g_src_root: Path = Path(Path(__file__).parent, "../../../../../").resolve()
    logging.debug("SrcRoot: %s", g_src_root)
    g_ctrl_path: Path = Path(g_src_root, "tests/cmake/scripts")
    if str(g_ctrl_path) not in sys.path:
        sys.path.append(str(g_ctrl_path))
    from golden_register import GoldenRegister  # 单独调试 import 失败, 需确认上文中 '系统 import 路径' 配置正确
else:
    from golden_register import GoldenRegister


def scatter_update_bnsd(inputs, axis):
    # inputs: cache, key_states, indices
    # cache shape: [b, 1, s2, d]
    # key_states shape: [b, 1, s1, d]
    # indices shape: [b, s1]
    cache, key_states, indices = inputs
    b, n2, s2, d = cache.shape  # n2=1
    s1 = indices.shape[1]
    res = cache

    if axis == -2:
        for b_i in range(b):
            for s2_i in range(s2):
                for s1_i in range(s1):
                    index_value = indices[b_i][s1_i]
                    if s2_i == index_value:
                        logging.debug("find the index value and to replace!")
                        res[b_i][0][s2_i][:] = key_states[b_i][0][s1_i][:]

    return res


def scatter_update_pa_bsnd(inputs, axis):
    # inputs: cache, key_states, indices
    # cache shape: [block_number,block_size,n2,d], n2=1
    # key_states shape: [b*s1*1, d]
    # indices shape: [b, s1], s1=1
    cache, key_states, indices = inputs
    block_number, block_size, n2, d = cache.shape
    res = cache.reshape(block_number * block_size * n2, d)
    b, s1 = indices.shape

    if axis == -2:
        for b_i in range(b):
            for s1_i in range(s1):
                index_value = indices[b_i][s1_i]
                res[index_value][:] = key_states[b_i * s1 + s1_i][:]

    res = res.reshape(block_number, block_size, n2, d)

    NzFrac = 16
    z_pabsnz = res.reshape((block_number,  block_size, d // NzFrac, NzFrac)).transpose(0, 2, 1, 3)
    logging.debug("========z_pabsnz=====", z_pabsnz)

    return z_pabsnz


def scatter_update_pro(inputs, axis, cache_mode="BNSD"):
    if cache_mode == "PA_NZ":
        return scatter_update_pa_bsnd(inputs, axis)
    else:
        return scatter_update_bnsd(inputs, axis)

def gen_scatterupdate_data_bf16(b, n, s, s2, kv_lora_rank, qk_rope_head_dim, axis, output_dir: Path):
    dtype = bfloat16
    indices_dtype = np.int64
    shape_params = [b, n, s2, kv_lora_rank + qk_rope_head_dim]
    shape_indices = [b, s]  # default support index dim=1

    src1_shape = [b, n, s, kv_lora_rank + qk_rope_head_dim]

    shape_res = [b, n, s2, kv_lora_rank + qk_rope_head_dim]
    logging.debug("shape params0 is ", shape_params)
    logging.debug("shape params1 is ", src1_shape)
    logging.debug("shape indices is ", shape_indices)
    logging.debug("shape res is ", shape_res)

    x_path = Path(output_dir, 'x.bin')
    y_path = Path(output_dir, 'y.bin')
    indices_path = Path(output_dir, 'indices.bin')
    z_path = Path(output_dir, 'z_golden.bin')

    x = np.random.uniform(1, 1, shape_params).astype(dtype)
    x.tofile(x_path)
    y = np.random.uniform(-2, 2, src1_shape).astype(dtype)
    y.tofile(y_path)

    indices = np.random.randint(0, shape_params[axis], size=shape_indices).astype(indices_dtype)
    indices.tofile(indices_path)
    logging.debug("====indices=====", indices)

    # numpy
    z = x
    logging.debug("zzz====", z)
    if axis == -2:  # now only support axis -2
        for _b in range(b):
            for _n in range(n):
                for _s in range(s2):
                    for index in range(s):  # [1,3,7]
                        idx_val = indices[_b][index]
                        if _s == idx_val:
                            logging.debug("find the index value and to replace!")
                            z[_b][_n][idx_val][:] = y[_b][_n][index][:]

    logging.debug("after zzz====", z)
    z.tofile(z_path)


def gen_scatterupdate_data(b, n, s, s2, kv_lora_rank, qk_rope_head_dim, axis, output_dir: Path):
    dtype = np.float32
    indices_dtype = np.int64
    shape_params = [b, n, s2, kv_lora_rank + qk_rope_head_dim]
    shape_indices = [b, s]  # default support index dim=1

    src1_shape = [b, n, s, kv_lora_rank + qk_rope_head_dim]

    shape_res = [b, n, s2, kv_lora_rank + qk_rope_head_dim]
    logging.debug("shape params0 is ", shape_params)
    logging.debug("shape params1 is ", src1_shape)
    logging.debug("shape indices is ", shape_indices)
    logging.debug("shape res is ", shape_res)

    x_path = Path(output_dir, 'x.bin')
    y_path = Path(output_dir, 'y.bin')
    indices_path = Path(output_dir, 'indices.bin')
    z_path = Path(output_dir, 'z_golden.bin')

    x = np.random.uniform(1, 1, shape_params).astype(dtype)
    x.tofile(x_path)
    y = np.random.uniform(-2, 2, src1_shape).astype(dtype)
    y.tofile(y_path)

    indices = np.random.randint(0, shape_params[axis], size=shape_indices).astype(indices_dtype)
    import random
    L1=random.sample(range(0, shape_params[axis]), src1_shape[axis]) # src1_shape[axis] <= shape_params[axis]
    for i in range(src1_shape[axis]):
        indices[0][i] = L1[i]
    indices.tofile(indices_path)
    logging.debug("====indices=====", indices)

    # numpy
    z = x

    if axis == -2:  # now only support axis -2
        for _b in range(b):
            for _n in range(n):
                for _s in range(s2):
                    for index in range(s):  # [1,3,7]
                        idx_val = indices[_b][index]
                        if _s == idx_val:
                            logging.debug("find the index value and to replace!")
                            z[_b][_n][idx_val][:] = y[_b][_n][index][:]

    z.tofile(z_path)
    logging.debug(z)

def gen_scatterupdate_data_bsnz(b, n, s, blockSize, blockNum , kv_lora_rank, qk_rope_head_dim, axis, output_dir: Path):
    np.set_printoptions(threshold=np.inf)
    dtype = np.float32
    s2 = blockSize * blockNum * n
    d = kv_lora_rank + qk_rope_head_dim
    indices_dtype = np.int64
    shape_params = [b, n, s2, d]
    shape_indices = [b, s]  # default support index dim=1

    src1_shape = [b, n, s, d]

    logging.debug("shape params0 is ", shape_params)
    logging.debug("shape params1 is ", src1_shape)
    logging.debug("shape indices is ", shape_indices)

    x_path = Path(output_dir, 'x.bin')
    y_path = Path(output_dir, 'y.bin')
    indices_path = Path(output_dir, 'indices.bin')
    z_path = Path(output_dir, 'z_golden.bin')

    x = np.random.uniform(5, 5, shape_params).astype(dtype)
    NzFrac = 8  # fp32 is 8

    y = np.random.uniform(2, 2, src1_shape).astype(dtype)
    y.tofile(y_path)

    indices = np.random.randint(0, shape_params[axis], size=shape_indices).astype(indices_dtype)
    import random
    L1=random.sample(range(0, shape_params[axis]), src1_shape[axis]) # src1_shape[axis] <= shape_params[axis]
    for i in range(src1_shape[axis]):
        indices[0][i] = L1[i]
    indices.tofile(indices_path)
    logging.debug("====indices=====", indices)

    # numpy
    z = x
    x_nz = x.reshape((blockNum,  blockSize, d // NzFrac, NzFrac)).transpose(0, 2, 1, 3)
    logging.debug("==after transpose======x_nz=====", x_nz)
    x_nz.tofile(x_path)

    logging.debug("========z=====", z)
    if axis == -2:  # now only support axis -2
        for _b in range(b):
            for _n in range(n):
                for _s in range(s2):
                    for index in range(s):  # [1,3,7]
                        idx_val = indices[_b][index]
                        if _s == idx_val:
                            logging.debug("find the index value and to replace!")
                            z[_b][_n][idx_val][:] = y[_b][_n][index][:]



    z_pabsnz = z.reshape((blockNum,  blockSize, d // NzFrac, NzFrac)).transpose(0, 2, 1, 3)
    z_pabsnz.tofile(z_path)


def gen_scatterupdate_data_bsnd(b, s, n, d, blockNum, blockSize, axis, output_dir: Path, dtype):
    np.set_printoptions(threshold=np.inf)
    if axis == 4:
        shape_params = [blockNum, blockSize, n, d] # dst
        shape_indices = [b, s]
        src1_shape = [b, s, n, d] # src1
    elif axis == 2:
        shape_params = [blockNum * blockSize * n, d] # dst
        shape_indices = [1, b * s]
        src1_shape = [b * s * n, d] # src1

    logging.info("shape params0 is ")
    logging.info(shape_params)
    logging.info("shape params1 is ")
    logging.info(src1_shape)
    logging.info("shape indices is ")
    logging.info(shape_indices)

    x_path = Path(output_dir, 'x.bin')
    y_path = Path(output_dir, 'y.bin')
    indices_path = Path(output_dir, 'indices.bin')
    z_path = Path(output_dir, 'z_golden.bin')

    x = np.random.randint(1, 2, shape_params).astype(dtype)
    x.tofile(x_path)
    y = np.random.randint(2, 3, src1_shape).astype(dtype)
    y.tofile(y_path)
    logging.info("====src=====\n")
    logging.info(y)

    indices = np.random.choice(range(0, blockNum * blockSize), shape_indices.shape, replace = False)
    indices.tofile(indices_path)
    logging.info("====indices=====\n")
    logging.info(indices)

    # numpy
    z = x

    logging.info("====before dst=====\n")
    logging.info(z)
    if axis == 4:  # now only support axis -2
        for _b in range(b):
            for _s in range(s):
                idx_val = indices[_b][_s]
                z[idx_val // blockSize][idx_val % blockSize][:] = y[_b][_s][:]
    elif axis == 2:
        for _bs in range(b * s):
            idx_val = indices[0][_bs]
            z[idx_val][:] = y[_bs][:]

    z.tofile(z_path)
    logging.info("====after dst=====\n")
    logging.info(z)


def gen_scatterupdate_data_bsnz_bf16(b, n, s, blockSize, blockNum , kv_lora_rank, qk_rope_head_dim, axis, output_dir: Path):
    np.set_printoptions(threshold=np.inf)
    dtype = bfloat16
    s2 = blockSize * blockNum * n
    d = kv_lora_rank + qk_rope_head_dim
    indices_dtype = np.int64
    shape_params = [b, n, s2, d]
    shape_indices = [b, s]  # default support index dim=1

    src1_shape = [b, n, s, d]

    logging.debug("shape params0 is ", shape_params)
    logging.debug("shape params1 is ", src1_shape)
    logging.debug("shape indices is ", shape_indices)

    x_path = Path(output_dir, 'x.bin')
    y_path = Path(output_dir, 'y.bin')
    indices_path = Path(output_dir, 'indices.bin')
    z_path = Path(output_dir, 'z_golden.bin')

    x = np.random.uniform(1, 1, shape_params).astype(dtype)
    NzFrac = 16  # fp32 is 8

    y = np.random.uniform(2, 2, src1_shape).astype(dtype)
    y.tofile(y_path)

    indices = np.random.randint(0, shape_params[axis], size=shape_indices).astype(indices_dtype)
    import random
    L1=random.sample(range(0, shape_params[axis]), src1_shape[axis]) # src1_shape[axis] <= shape_params[axis]
    for i in range(src1_shape[axis]):
        indices[0][i] = L1[i]
    indices.tofile(indices_path)
    logging.debug("====indices=====", indices)

    # numpy
    z = x

    x_nz = x.reshape((blockNum,  blockSize, d // NzFrac, NzFrac)).transpose(0, 2, 1, 3)
    logging.debug("==after transpose======x_nz=====", x_nz)
    x_nz.tofile(x_path)

    if axis == -2:  # now only support axis -2
        for _b in range(b):
            for _n in range(n):
                for _s in range(s2):
                    for index in range(s):  # [1,3,7]
                        idx_val = indices[_b][index]
                        if _s == idx_val:
                            logging.debug("find the index value and to replace!")
                            z[_b][_n][idx_val][:] = y[_b][_n][index][:]

    # x_nz = x.reshape((blockNum,  blockSize, d // NzFrac, NzFrac)).transpose(0, 2, 1, 3)
    # logging.debug("==after transpose======x_nz=====", x_nz)
    # x_nz.tofile(x_path)

    z_pabsnz = z.reshape((blockNum,  blockSize, d // NzFrac, NzFrac)).transpose(0, 2, 1, 3)
    logging.debug("========z_pabsnz=====", z_pabsnz)
    z_pabsnz.tofile(z_path)


def gen_scatterupdate_data_exp(b, n, s, s2, kv_lora_rank, qk_rope_head_dim, axis, output_dir: Path):
    dtype = np.float32
    indices_dtype = np.int64
    shape_params = [b, n, s2, kv_lora_rank + qk_rope_head_dim]
    shape_indices = [b, s]  # default support index dim=1

    src1_shape = [b, n, s, kv_lora_rank + qk_rope_head_dim]

    shape_res = [b, n, s2, kv_lora_rank + qk_rope_head_dim]
    logging.debug("shape params0 is ", shape_params)
    logging.debug("shape params1 is ", src1_shape)
    logging.debug("shape indices is ", shape_indices)
    logging.debug("shape res is ", shape_res)

    x_path = Path(output_dir, 'x.bin')
    y_path = Path(output_dir, 'y.bin')
    indices_path = Path(output_dir, 'indices.bin')
    z_path = Path(output_dir, 'z_golden.bin')

    x = np.random.uniform(1, 1, shape_params).astype(dtype)
    x.tofile(x_path)
    y = np.random.uniform(2, 2, src1_shape).astype(dtype)
    y.tofile(y_path)

    indices = np.random.randint(0, shape_params[axis], size=shape_indices).astype(indices_dtype)
    indices.tofile(indices_path)
    logging.debug("====indices=====", indices)

    # numpy
    z = x

    if axis == -2:  # now only support axis -2
        for _b in range(b):
            for _n in range(n):
                for _s in range(s2):
                    for index in range(s):  # [1,3,7]
                        idx_val = indices[_b][index]
                        if _s == idx_val:
                            logging.debug("find the index value and to replace!")
                            z[_b][_n][idx_val][:] = y[_b][_n][index][:]

    z = np.exp(z)
    z.tofile(z_path)

    # result = np.allclose(y, y_tf, rtol=1e-3, atol=1e-3)
    # logging.debug(f"====== golden precise: {result}")


def rms_norm(x):
    res = x / np.sqrt(np.mean(np.square(x), axis=-1, keepdims=True) + 1e-6)
    return res


def rms_norm_bf16(x):
    #   res = x / np.sqrt(np.mean(np.square(x), axis=-1, keepdims=True) + 1e-6)
    x = x.astype(np.float32)
    res = x / np.sqrt(np.mean(np.square(x), axis=-1, keepdims=True) + 1e-6)
    res = res.astype(bfloat16)

    return res


def scatter_update(past_key_states, key_states, indices, b, s, s2, kv_lora_rank, qk_rope_head_dim, axis):
    z = past_key_states

    if axis == -2:
        for _b in range(b):
            for _s in range(s2):
                for index in range(s):  # [1,3,7]
                    idx_val = indices[_b][index]
                    if _s == idx_val:
                        logging.debug("find the index value and to replace!")
                        z[_b][0][idx_val][:] = key_states[_b][0][index][:]

    return z


def gen_graph_d_data(b, s, s2, kv_lora_rank, qk_rope_head_dim, axis, output_dir: Path):
    dtype = np.float32
    indices_dtype = np.int64
    shape_params = [b, 1, s2, kv_lora_rank + qk_rope_head_dim]
    shape_indices = [b, s]

    src1_shape = [b, 1, s, kv_lora_rank + qk_rope_head_dim]

    shape_res = [b, 1, s2, kv_lora_rank + qk_rope_head_dim]

    shape_compressed_kv = [b, s, kv_lora_rank]
    shape_k_pe_rope = [b, 1, s, qk_rope_head_dim]
    shape_past_key_states = [b, 1, s2, kv_lora_rank + qk_rope_head_dim]

    logging.debug("shape params0 is %s", shape_params)
    logging.debug("shape params1 is %s", src1_shape)
    logging.debug("shape indices is %s", shape_indices)
    logging.debug("shape res is %s", shape_past_key_states)

    x_path = Path(output_dir, 'x.bin')  # last kery_states
    past_key_states = np.random.uniform(1, 1, shape_past_key_states).astype(
        dtype)  # [b, 1, s2, kv_lora_rank+qk_rope_head_dim]
    past_key_states.tofile(x_path)

    compressed_kv_path = Path(output_dir, 'compressed_kv.bin')
    k_pe_rope_path = Path(output_dir, 'y.bin')

    indices_path = Path(output_dir, 'indices.bin')
    z_path = Path(output_dir, 'z_golden.bin')

    indices = np.random.randint(0, shape_params[axis], size=shape_indices).astype(indices_dtype)
    indices.tofile(indices_path)
    logging.debug("====indices===== %s", indices)

    compressed_kv = np.random.uniform(-4, 4, shape_compressed_kv).astype(dtype)  # [b, s, kv_lora_rank]
    k_pe_rope = np.random.uniform(2, 2, shape_k_pe_rope).astype(dtype)  # [b, 1, s, qk_rope_head_dim]
    compressed_kv.tofile(compressed_kv_path)
    k_pe_rope.tofile(k_pe_rope_path)

    k_nope = rms_norm(compressed_kv)  # [b, s, kv_lora_rank]
    k_nope_new = k_nope.reshape(b, s, 1, kv_lora_rank).transpose(0, 2, 1, 3)  # [b, 1, s, kv_lora_rank]

    key_states = np.concatenate((k_nope_new, k_pe_rope), axis=-1)  # [b, 1, s, kv_lora_rank + qk_rope_head_dim]

    past_key_states_new = scatter_update(past_key_states, key_states, indices, b, s, s2, kv_lora_rank, qk_rope_head_dim,
                                         -2)

    past_key_states_new.tofile(z_path)


def gen_graph_d_data_bf16(b, s, s2, kv_lora_rank, qk_rope_head_dim, axis, output_dir: Path, k_pe=None):
    dtype = bfloat16
    # data = np.array([1.25, 3.5, -0.75], dtype=np.bfloat16)
    indices_dtype = np.int64
    shape_params = [b, 1, s2, kv_lora_rank + qk_rope_head_dim]
    shape_indices = [b, s]

    src1_shape = [b, 1, s, kv_lora_rank + qk_rope_head_dim]

    shape_res = [b, 1, s2, kv_lora_rank + qk_rope_head_dim]

    shape_compressed_kv = [b, s, kv_lora_rank]
    shape_k_pe_rope = [b, 1, s, qk_rope_head_dim]
    shape_past_key_states = [b, 1, s2, kv_lora_rank + qk_rope_head_dim]

    logging.debug("shape params0 is %s", shape_params)
    logging.debug("shape params1 is %s", src1_shape)
    logging.debug("shape indices is %s", shape_indices)
    logging.debug("shape res is %s", shape_past_key_states)

    x_path = Path(output_dir, 'x.bin')  # last kery_states
    past_key_states = np.random.uniform(1, 1, shape_past_key_states).astype(
        dtype)  # [b, 1, s2, kv_lora_rank+qk_rope_head_dim]
    past_key_states.tofile(x_path)

    compressed_kv_path = Path(output_dir, 'compressed_kv.bin')
    k_pe_rope_path = Path(output_dir, 'y.bin')

    indices_path = Path(output_dir, 'indices.bin')
    z_path = Path(output_dir, 'z_golden.bin')

    indices = np.random.randint(0, shape_params[axis], size=shape_indices).astype(indices_dtype)
    indices.tofile(indices_path)
    logging.debug("====indices===== %s", indices)

    compressed_kv = np.random.uniform(-4, 4, shape_compressed_kv).astype(dtype)  # [b, s, kv_lora_rank]
    k_pe_rope = np.random.uniform(2, 2, shape_k_pe_rope).astype(dtype)  # [b, 1, s, qk_rope_head_dim]
    if k_pe is not None:
        k_pe_rope = k_pe
    compressed_kv.tofile(compressed_kv_path)
    k_pe_rope.tofile(k_pe_rope_path)
    logging.debug("=======k_pe_rope=== %s", k_pe_rope)
    logging.debug("=======compressed_kv=== %s", compressed_kv)
    k_nope = rms_norm_bf16(compressed_kv)  # [b, s, kv_lora_rank]
    k_nope_new = k_nope.reshape(b, s, 1, kv_lora_rank).transpose(0, 2, 1, 3)  # [b, 1, s, kv_lora_rank]

    key_states = np.concatenate((k_nope_new, k_pe_rope), axis=-1)  # [b, 1, s, kv_lora_rank + qk_rope_head_dim]

    past_key_states_new = scatter_update(past_key_states, key_states, indices, b, s, s2, kv_lora_rank, qk_rope_head_dim,
                                         -2)
    logging.debug("=======past_key_states_new=== %s", compressed_kv)
    past_key_states_new.tofile(z_path)


@GoldenRegister.reg_golden_func(
    case_names=[
        "ScatterupdateOnBoardTest.test_scatter_update_1_1_16_16",
        "ScatterupdateOnBoardTest.test_scatter_update_1_1_20_20",
        "ScatterupdateOnBoardTest.test_scatter_update_1_1_16_16_bf16",
        "ScatterupdateOnBoardTest.test_scatter_update_1_1_64_64",
        "ScatterupdateOnBoardTest.test_scatter_update_1_1_16_16_exp",
        "ScatterupdateOnBoardTest.test_scatter_update_1_16_16_16",
        "ScatterupdateOnBoardTest.test_scatter_update_64_7168_64_moe",
        "ScatterupdateOnBoardTest.test_scatter_update_2_1_512_576",
        "ScatterupdateOnBoardTest.test_scatter_update_2_1_512_576_graphD",
        "ScatterupdateOnBoardTest.test_scatter_update_32_1_512_576_graphD",
        "ScatterupdateOnBoardTest.test_scatter_update_2_1_512_576_graphD_bf16",
        "ScatterupdateOnBoardTest.test_scatter_update_32_1_512_576_graphD_bf16",
        "ScatterupdateOnBoardTest.test_scatter_update_2_1_512_576_multi_row3",
        "ScatterupdateOnBoardTest.test_scatter_update_1_48_4096_512",
        "ScatterupdateOnBoardTest.test_scatter_update_1_48_4096_512_BSNZ",
        "ScatterupdateOnBoardTest.test_scatter_update_1_48_4096_512_BSNZ_bf16",
        "ScatterupdateOnBoardTest.test_scatter_update_1_1_1_64_BSND_2dims",
        "ScatterupdateOnBoardTest.test_scatter_update_1_1_1_64_BSND_4dims",
    ]
)
def gen_scatterupdate_op_date(case_name: str, output: Path) -> bool:
    dtype = np.float32
    indices_dtype = np.int64

    x_path = Path(output, 'x.bin')
    indices_path = Path(output, 'indices.bin')
    y_path = Path(output, 'y.bin')
    z_path = Path(output, 'z_golden.bin')

    # complete = x_path.exists() and indices_path.exists() and z_path.exists() and y_path.exists()

    # if complete:
    #     logging.debug("Case(%s), Golden complete.", case_name)
    # else:
    if case_name == "ScatterupdateOnBoardTest.test_scatter_update_1_1_16_16":
        s2, kv_lora_rank, qk_rope_head_dim = 16, 8, 8
        gen_scatterupdate_data(1, 1, 1, s2, kv_lora_rank, qk_rope_head_dim, -2, output)
    elif case_name == "ScatterupdateOnBoardTest.test_scatter_update_1_1_20_20":
        s2, kv_lora_rank, qk_rope_head_dim = 16, 10, 10
        gen_scatterupdate_data(1, 1, 1, s2, kv_lora_rank, qk_rope_head_dim, -2, output)
    elif case_name == "ScatterupdateOnBoardTest.test_scatter_update_1_1_16_16_bf16":
        s2, kv_lora_rank, qk_rope_head_dim = 16, 8, 8
        gen_scatterupdate_data_bf16(1, 1, 1, s2, kv_lora_rank, qk_rope_head_dim, -2, output)
    elif case_name == "ScatterupdateOnBoardTest.test_scatter_update_64_7168_64_moe":
        s2, kv_lora_rank, qk_rope_head_dim = 64, 7160, 8
        gen_scatterupdate_data(1, 1, 64, s2, kv_lora_rank, qk_rope_head_dim, -2, output)
    elif case_name == "ScatterupdateOnBoardTest.test_scatter_update_1_16_16_16":
        s2, kv_lora_rank, qk_rope_head_dim = 16, 8, 8
        gen_scatterupdate_data(1, 1, 16, s2, kv_lora_rank, qk_rope_head_dim, -2, output)
    elif case_name == "ScatterupdateOnBoardTest.test_scatter_update_1_1_16_16_exp":
        s2, kv_lora_rank, qk_rope_head_dim = 16, 8, 8
        gen_scatterupdate_data_exp(1, 1, 1, s2, kv_lora_rank, qk_rope_head_dim, -2, output)
    elif case_name == "ScatterupdateOnBoardTest.test_scatter_update_2_1_512_576":
        b, s, s2, kv_lora_rank, qk_rope_head_dim = 2, 1, 512, 512, 64
        gen_scatterupdate_data(b, 1, s, s2, kv_lora_rank, qk_rope_head_dim, -2, output)
    elif case_name == "ScatterupdateOnBoardTest.test_scatter_update_2_1_512_576_multi_row3":
        b, s, s2, kv_lora_rank, qk_rope_head_dim = 2, 3, 512, 512, 64
        gen_scatterupdate_data(b, 1, s, s2, kv_lora_rank, qk_rope_head_dim, -2, output)
    elif case_name == "ScatterupdateOnBoardTest.test_scatter_update_2_1_512_576_graphD":
        b, s, s2, kv_lora_rank, qk_rope_head_dim = 2, 1, 512, 512, 64
        gen_graph_d_data(b, s, s2, kv_lora_rank, qk_rope_head_dim, -2, output)
    elif case_name == "ScatterupdateOnBoardTest.test_scatter_update_32_1_512_576_graphD":
        b, s, s2, kv_lora_rank, qk_rope_head_dim = 32, 1, 512, 512, 64
        gen_graph_d_data(b, s, s2, kv_lora_rank, qk_rope_head_dim, -2, output)
    elif case_name == "ScatterupdateOnBoardTest.test_scatter_update_2_1_512_576_graphD_bf16":
        b, s, s2, kv_lora_rank, qk_rope_head_dim = 2, 1, 512, 512, 64
        gen_graph_d_data_bf16(b, s, s2, kv_lora_rank, qk_rope_head_dim, -2, output)
    elif case_name == "ScatterupdateOnBoardTest.test_scatter_update_32_1_512_576_graphD_bf16":
        b, s, s2, kv_lora_rank, qk_rope_head_dim = 32, 1, 512, 512, 64
        gen_graph_d_data_bf16(b, s, s2, kv_lora_rank, qk_rope_head_dim, -2, output)
    elif case_name == "ScatterupdateOnBoardTest.test_scatter_update_1_48_4096_512":
        S2, kv_lora_rank, qk_rope_head_dim = 48, 256, 256
        gen_scatterupdate_data(1, 1, 32, S2, kv_lora_rank, qk_rope_head_dim, -2, output)
    elif case_name == "ScatterupdateOnBoardTest.test_scatter_update_1_48_4096_512_BSNZ":
        blockSize, blockNum, kv_lora_rank, qk_rope_head_dim = 16, 3, 256, 256
        gen_scatterupdate_data_bsnz(1, 1, 32, blockSize, blockNum , kv_lora_rank, qk_rope_head_dim, -2, output)
    elif case_name == "ScatterupdateOnBoardTest.test_scatter_update_1_1_1_64_BSND_2dims":
        b, s, n, d, blockNum, blockSize = 20, 2, 1, 32, 20, 20
        gen_scatterupdate_data_bsnd(b, s, n, d, blockNum, blockSize, 2, output, np.float32)
    elif case_name == "ScatterupdateOnBoardTest.test_scatter_update_1_1_1_64_BSND_4dims":
        b, s, n, d, blockNum, blockSize = 20, 2, 1, 32, 20, 20
        gen_scatterupdate_data_bsnd(b, s, n, d, blockNum, blockSize, 4, output, np.float32)
    elif case_name == "ScatterupdateOnBoardTest.test_scatter_update_1_48_4096_512_BSNZ_bf16":
        b, s, kv_lora_rank = 32, 1, 512
        blockNum = 1920
        blockSize = 128
        NzFrac = 16
        dtype = bfloat16
        kv_cache_shape = [blockNum, blockSize, 1, kv_lora_rank]
        k_nope_shape = [b * s * 1, kv_lora_rank]
        kv_len_shape = [b, s]

        kv_cache = np.random.uniform(-1, 1, kv_cache_shape).astype(dtype)
        k_nope = np.random.uniform(-1, 1, k_nope_shape).astype(dtype)
        kv_len = np.random.randint(
            0, blockNum * blockSize, size=kv_len_shape).astype(np.int64)

        kv_cache_out = scatter_update_pro(
            [kv_cache, k_nope, kv_len], -2, "PA_NZ")

        x_path = Path(output, 'x.bin')
        y_path = Path(output, 'y.bin')
        indices_path = Path(output, 'indices.bin')
        z_path = Path(output, 'z_golden.bin')


        x_nz = kv_cache.reshape((blockNum,  blockSize, kv_lora_rank // NzFrac, NzFrac)).transpose(0, 2, 1, 3)
        logging.debug("==after transpose======x_nz=====", x_nz)
        x_nz.tofile(x_path)

        # kv_cache.tofile(x_path)
        k_nope.tofile(y_path)
        kv_len.tofile(indices_path)
        kv_cache_out.tofile(z_path)

    else:
        logging.error("Can't get func to gen golden, Case(%s)", case_name)
        return False
    return True


def main() -> bool:
    """
    单独调试 入口函数
    """
    # 用例名称
    case_name_list: List[str] = [
        "ScatterupdateOnBoardTest.test_scatter_update_1_1_16_16",
    ]
    # 函数调用
    ret: bool = True
    for cs in case_name_list:
        output: Path = Path(g_src_root, "build/output/bin/golden", cs).resolve()
        output.mkdir(parents=True, exist_ok=True)
        ret = gen_scatterupdate_op_date(case_name=cs, output=output)
    return ret


if __name__ == "__main__":
    exit(0 if main() else 1)
