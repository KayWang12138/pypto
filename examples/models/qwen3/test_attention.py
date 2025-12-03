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
import pypto
import torch
import numpy as np
import math
import os
from utils.np_compare import detailed_allclose_manual

np.random.seed(0)
torch.manual_seed(0)
np.set_printoptions(formatter={'float': '{:.6f}'.format})


@dataclass
class AttentionTileConfig:
    g_tile: int
    s2_tile: int
    c1_tile_shape: list
    v1_tile_shape: list
    c2_tile_shape: list
    v2_tile_shape: list


@dataclass
class AttentionConfig:
    b: int
    s1: int
    s2: int
    n1: int
    n2: int
    q_d: int
    kv_d: int
    block_size: int = 128
    max_num_blocks_per_query: int = 0
    softmax_scale: float = 1.0
    kv_layout: str = "PA_BSND"
    actual_seq: torch.Tensor = None  # 改为 torch.Tensor 类型
    block_table_batch: int = 0
    kv_num_blocks: int = 0


def get_qwen_common_config(device="cpu"):
    b = 24
    s1 = 2
    s2 = 4096
    q_d = 128
    nq = 4
    nkv = 1
    kv_layout = "PA_BSND"
    softmax_scale = q_d ** -0.5
    block_table_batch = 256
    kv_num_blocks = 100

    # 创建 torch tensor 类型的 actual_seq
    actual_seq_values = [8, 6, 6, 6, 8, 6, 6, 6, 8, 6, 6, 6, 8, 6, 6, 6, 8, 6, 6, 6, 8, 6, 6, 6]
    actual_seq_tensor = torch.tensor(actual_seq_values, dtype=torch.int32, device=device)

    atten_cfg = AttentionConfig(b=b, s1=s1, s2=s2, n1=nq, n2=nkv, softmax_scale=softmax_scale, kv_layout=kv_layout,
                                q_d=q_d, kv_d=q_d, block_table_batch=block_table_batch, kv_num_blocks=kv_num_blocks,
                                actual_seq=actual_seq_tensor)  # 传入 tensor
    atten_cfg.max_num_blocks_per_query = 32
    cube_tile = 128
    vector_tile = 128
    s2_tile = 128
    tile_cfg = AttentionTileConfig(
        nq,
        s2_tile,
        [[cube_tile, cube_tile], [cube_tile, cube_tile], [cube_tile, cube_tile]],
        [vector_tile, vector_tile],
        [[cube_tile, cube_tile], [cube_tile, cube_tile], [cube_tile, cube_tile]],
        [vector_tile, vector_tile])
    return atten_cfg, tile_cfg


def gen_block_table(actual_seq_len, block_size, block_table_shape):
    block_num_per_batch = []
    block_num = 0

    # 处理 torch tensor 类型的 actual_seq_len
    if isinstance(actual_seq_len, torch.Tensor):
        # 如果 tensor 在 GPU/NPU 上，先移动到 CPU
        if actual_seq_len.device.type != 'cpu':
            actual_seq_len_cpu = actual_seq_len.cpu()
        else:
            actual_seq_len_cpu = actual_seq_len

        # 转换为 numpy 数组进行处理，或者直接使用 torch 操作
        for actual_seq in actual_seq_len_cpu:
            block_num_per_batch.append(math.ceil(actual_seq.item() / block_size))
            block_num += math.ceil(actual_seq.item() / block_size)
    else:
        # 保持对 list 的兼容
        for actual_seq in actual_seq_len:
            block_num_per_batch.append(math.ceil(actual_seq / block_size))
            block_num += math.ceil(actual_seq / block_size)

    # 使用 torch 替换 numpy
    block_idx_list = torch.arange(0, block_num, dtype=torch.int32)
    block_idx_list = block_idx_list[torch.randperm(block_idx_list.size(0))]  # 随机排列

    # 创建 block_table 张量
    block_table = torch.full(block_table_shape, -1, dtype=torch.int32)
    block_idx = 0
    block_table_batch_idx = 0
    for idx in block_num_per_batch:
        for j in range(idx):
            block_table[block_table_batch_idx][j] = block_idx_list[block_idx]
            block_idx += 1
        block_table_batch_idx += 1
    return block_table


def kv_cache_concat_bsnd(kr_cache_out, kv_cache_out, block_table, atten_config):
    b = atten_config.b
    n2 = atten_config.n2
    kv_lora_rank = atten_config.q_d
    rope_dim = atten_config.kv_d
    block_size = atten_config.block_size
    kv_cache_actual_seq = atten_config.actual_seq
    dtype = kv_cache_out.dtype

    # 处理 torch tensor 类型的 kv_cache_actual_seq
    if isinstance(kv_cache_actual_seq, torch.Tensor):
        if kv_cache_actual_seq.device.type != 'cpu':
            kv_cache_actual_seq_cpu = kv_cache_actual_seq.cpu()
        else:
            kv_cache_actual_seq_cpu = kv_cache_actual_seq
        kv_max = (torch.max(kv_cache_actual_seq_cpu).item() + block_size - 1) // block_size * block_size
    else:
        kv_max = (max(kv_cache_actual_seq) + block_size - 1) // block_size * block_size

    # 使用 torch 创建张量，保持在同一设备上
    device = kr_cache_out.device
    k_cache = torch.zeros([b, kv_max, n2, kv_lora_rank], dtype=dtype, device=device)
    v_cache = torch.zeros([b, kv_max, n2, rope_dim], dtype=dtype, device=device)

    for b_idx in range(b):
        block_list = block_table[b_idx]
        kv_nope_temp_tensor = torch.zeros([1, kv_max, n2, kv_lora_rank], dtype=dtype, device=device)
        kv_rope_temp_tensor = torch.zeros([1, kv_max, n2, rope_dim], dtype=dtype, device=device)
        s_idx = 0

        for _, block_idx in enumerate(block_list):
            if block_idx == -1:
                break
            # 使用 torch 的切片操作
            start_idx = s_idx * block_size
            end_idx = (s_idx + 1) * block_size

            kv_nope_temp_tensor[:, start_idx:end_idx, :, :] = kv_cache_out[block_idx:block_idx + 1, :, :, :]
            kv_rope_temp_tensor[:, start_idx:end_idx, :, :] = kr_cache_out[block_idx:block_idx + 1, :, :, :]
            s_idx += 1

        v_cache[b_idx:b_idx + 1, :, :, :] = kv_nope_temp_tensor
        k_cache[b_idx:b_idx + 1, :, :, :] = kv_rope_temp_tensor

    return k_cache, v_cache


def get_special_array(m, n):
    q_shape = [m, n]

    # 生成递增的行值
    base = np.arange(1, m + 1)  # 生成 [1, 2, ..., m]

    # 将 base 扩展到二维形状 [m, n]
    q = base[:, np.newaxis]  # 增加一个新维度，形状变为 [m, 1]
    q = np.broadcast_to(q, q_shape)  # 广播到目标形状 [m, n]

    # 转换为 float16 类型
    q = q.astype(np.float16)
    return q


def softmax(x, is_fp16=False):
    # 使用 torch 的 softmax 实现
    if is_fp16:
        original_dtype = x.dtype
        x = x.float()
    x_max = x.max(dim=-1, keepdim=True).values
    x_sub = x - x_max
    y = torch.exp(x_sub)
    x_sum = y.sum(dim=-1, keepdim=True)
    ans = y / x_sum
    if is_fp16:
        ans = ans.to(original_dtype)
        x_max = x_max.to(original_dtype)
        x_sum = x_sum.to(original_dtype)

    return ans, x_max, x_sum


@pypto.jit
def ifa_func(inputs, outputs):
    # 1. 添加支持动态的config
    pypto.set_codegen_options(support_dynamic_unaligned=True)
    pypto.set_host_options(only_codegen=True)
    # pypto.set_option('profile_enable', True)

    # 2. 从入参拿到输入和输出tensor    
    q = inputs[0]
    k = inputs[1]
    v = inputs[2]
    block_table = inputs[3]
    kv_act_seqs = inputs[4]
    atten_out = outputs[0]

    # 3. 设置axis=0为动态shape     
    pypto.mark_dynamic(q, 0)
    pypto.mark_dynamic(k, 0)
    pypto.mark_dynamic(v, 0)
    pypto.mark_dynamic(kv_act_seqs, 0)

    shape_q = q.shape
    shape_k = k.shape
    shape_act_seqs = kv_act_seqs.shape

    atten_cfg, tile_cfg = get_qwen_common_config()
    softmax_scale = atten_cfg.softmax_scale

    bs_scalar = shape_q[0]
    nq = shape_q[1]
    block_num_scalar = shape_k[0]
    block_size = shape_k[1]
    nkv = shape_k[2]
    dn = shape_k[3]
    b_scalar = shape_act_seqs[0]

    dtype = q.dtype
    group = nq // nkv
    n2_sym = nkv

    g_tile = tile_cfg.g_tile
    s2_tile = tile_cfg.s2_tile
    c1_tile = tile_cfg.c1_tile_shape
    v1_tile = tile_cfg.v1_tile_shape
    c2_tile = tile_cfg.c2_tile_shape
    v2_tile = tile_cfg.v2_tile_shape

    # 4. 得到动态tensor的shape
    s1_scalar = bs_scalar // b_scalar
    g = nq // nkv
    g_loop = g // g_tile

    print(f"g_tile {g_tile} s2_tile {s2_tile} \n  \
    c1_tile_shape {c1_tile} v1_tile_shape {v1_tile} \n \
    c2_tile_shape {c2_tile} v2_tile_shape {v2_tile} \n \
    q_shape {q.shape} k_shape {k.shape} v_shape {v.shape} \n \
    block_table  {block_table.shape} kv_act_seqs_shape {kv_act_seqs.shape} \n \
        s1_sym {s1_scalar} n2_sym {n2_sym} g_loop_sym {g}  g_loop {g_loop}")

    k_2d_shape = (block_num_scalar * block_size, n2_sym * dn)
    q_2d_shape = (b_scalar * s1_scalar * nq, dn)

    for _ in pypto.loop(1, name="LOOP_RESHAPE_INPLACE", idx_name="tmp_idx"):
        k_2d = pypto.reshape(k, k_2d_shape, inplace=True)
        v_2d = pypto.reshape(v, k_2d_shape, inplace=True)
        q_2d = pypto.reshape(q, q_2d_shape, inplace=True)

    def fun():
        # 5. 实现kernel逻辑，循环展开B动态轴
        for b_idx in pypto.loop(b_scalar, name="LOOP_b", idx_name="b_idx"):
            def b_fun():
                for s1_idx in pypto.loop(s1_scalar, name="LOOP_s1", idx_name="s1_idx"):
                    def s1_fun():
                        cur_seq = kv_act_seqs[b_idx] - (s1_scalar - 1 - s1_idx)
                        s2_loop = (cur_seq + s2_tile - 1) // s2_tile
                        for n2_idx in pypto.loop(n2_sym, name="LOOP_n2", idx_name="n2_idx"):
                            def n2_fun():
                                for g_idx in pypto.loop(g_loop, name="LOOP_g", idx_name="g_idx"):
                                    def g_fun():
                                        oi_upd = pypto.tensor([g_tile, dn], pypto.DT_FP32, "oi_upd")
                                        li_upd = pypto.tensor([g_tile, 1], pypto.DT_FP32, "li_upd")
                                        mi_upd = pypto.tensor([g_tile, 1], pypto.DT_FP32, "mi_upd")
                                        for s2_idx in pypto.loop(s2_loop, name="LOOP_s2", idx_name="s2_idx"):
                                            def inside():
                                                block_idx = block_table[b_idx, s2_idx]
                                                bs_ofs = b_idx * s1_scalar + s1_idx
                                                n1g_ofs = n2_idx * group + g_idx * g_tile
                                                actual_s2_tile = (cur_seq - s2_idx * s2_tile).min(s2_tile)
                                                oi_ofs = [bs_ofs, n1g_ofs, 0]
                                                # 6. 按照计算图实现运算逻辑，设置set_vec_tile_shapes时应尽可能用满UB，但不要超过UB的大小。
                                                pypto.set_vec_tile_shapes(16, v1_tile[0], v1_tile[1])
                                                qi = pypto.view(q_2d, [g_tile, dn], [bs_ofs * nq + n1g_ofs, 0])
                                                kj = pypto.view(k_2d, [block_size, dn], [block_idx * block_size, 0],
                                                                valid_shape=[actual_s2_tile, dn])
                                                vj = pypto.view(v_2d, [block_size, dn], [block_idx * block_size, 0],
                                                                valid_shape=[actual_s2_tile, dn])
                                                # c1
                                                # 7. 下面是flash attention的计算逻辑
                                                pypto.set_cube_tile_shapes(c1_tile[0], c1_tile[1], c1_tile[2])
                                                sij = pypto.matmul(qi, kj, pypto.DT_FP32, a_trans=False,
                                                                    b_trans=True)
                                                sij = pypto.reshape(sij, [g_tile, s2_tile],
                                                                    valid_shape=[g_tile, actual_s2_tile])
                                                # v1
                                                pypto.set_vec_tile_shapes(v1_tile[0], v1_tile[1])
                                                sij_scale = pypto.mul(sij, softmax_scale)
                                                tilda_mij = pypto.amax(sij_scale, -1, True)
                                                tsub = pypto.sub(sij_scale, tilda_mij)
                                                tilda_pij = pypto.exp(tsub)
                                                tilda_pij_fp16 = pypto.cast(tilda_pij, dtype)
                                                tilda_lij = pypto.sum(tilda_pij, -1, True)
                                                pypto.set_vec_tile_shapes(16, v2_tile[0], v2_tile[1])
                                                oi_upd_3d = pypto.cast(pypto.reshape(tsub, [1, 1, g_tile * dn]),
                                                                        dtype)
                                                if pypto.cond(pypto.is_loop_begin(s2_idx)):
                                                    # c2
                                                    pypto.set_cube_tile_shapes(c2_tile[0], c2_tile[1], c2_tile[2])
                                                    oi_tmp = pypto.matmul(tilda_pij_fp16, vj, pypto.DT_FP32)
                                                    oi_upd[:] = pypto.tensor(oi_tmp.shape, pypto.DT_FP32, "oi_upd")
                                                    pypto.set_vec_tile_shapes(v2_tile[0], v2_tile[1])
                                                    if pypto.cond(pypto.is_loop_end(s2_idx)):
                                                        oi_upd[:] = pypto.div(oi_tmp, tilda_lij)
                                                        pypto.set_vec_tile_shapes(16, 16, v2_tile[0], v2_tile[1])
                                                        oi_upd_3d = pypto.cast(
                                                            pypto.reshape(oi_upd, [1, g_tile, dn]),
                                                            dtype)
                                                        # 8. 将结果搬运到输出tensor上
                                                        pypto.assemble(oi_upd_3d, oi_ofs, atten_out)
                                                    else:
                                                        oi_upd[:] = oi_tmp

                                                    li_upd[:] = tilda_lij
                                                    mi_upd[:] = tilda_mij

                                                else:
                                                    oi = oi_upd
                                                    li = li_upd
                                                    mi = mi_upd

                                                    mi_new = pypto.maximum(mi, tilda_mij)
                                                    t1 = pypto.sub(mi, mi_new)
                                                    t2 = pypto.exp(t1)
                                                    t3 = pypto.sub(tilda_mij, mi_new)
                                                    t4 = pypto.exp(t3)
                                                    t5 = pypto.mul(t4, tilda_lij)
                                                    t6 = pypto.mul(t2, li)
                                                    li_new = pypto.add(t6, t5)

                                                    q3 = pypto.mul(oi, t2)
                                                    pypto.set_cube_tile_shapes(c2_tile[0], c2_tile[1], c2_tile[2])
                                                    q1 = pypto.matmul(tilda_pij_fp16, vj, pypto.DT_FP32)
                                                    pypto.set_vec_tile_shapes(v2_tile[0], v2_tile[1])
                                                    q2 = pypto.mul(q1, t4)
                                                    oi_tmp = pypto.add(q3, q2)
                                                    if pypto.cond(pypto.is_loop_end(s2_idx)):
                                                        oi_upd[:] = pypto.div(oi_tmp, li_new)
                                                        pypto.set_vec_tile_shapes(16, v2_tile[0], v2_tile[1])
                                                        oi_upd_3d = pypto.cast(
                                                            pypto.reshape(oi_upd, [1, g_tile, dn]),
                                                            dtype)
                                                        # 9. 将结果搬运到输出tensor上
                                                        pypto.assemble(oi_upd_3d, oi_ofs, atten_out)
                                                    else:
                                                        oi_upd[:] = oi_tmp
                                                    li_upd[:] = li_new
                                                    mi_upd[:] = mi_new

                                            inside()

                                    g_fun()

                            n2_fun()

                    s1_fun()

            b_fun()

    fun()


def IFA(atten_cfg):
    device_id = os.environ.get('TILE_FWK_STEST_DEVICE_ID', 1)
    print(f'xxxxxx device id {int(device_id)}')
    torch_dtype = torch.float16
    torch.npu.set_device(int(device_id))
    b = atten_cfg.b
    s1 = atten_cfg.s1
    d = atten_cfg.q_d
    nq = atten_cfg.n1
    nkv = atten_cfg.n2

    block_size = atten_cfg.block_size
    max_num_blocks_per_query = atten_cfg.max_num_blocks_per_query

    # 获取 torch tensor 类型的 actual_seq
    kv_cache_actual_seq = atten_cfg.actual_seq

    q_shape = [b * s1, nq, d]
    kv_shape = [atten_cfg.kv_num_blocks, block_size, nkv, d]
    block_table_shape = [atten_cfg.block_table_batch, max_num_blocks_per_query]

    # 使用 torch 生成数据
    device = f'npu:{device_id}'
    q = torch.empty(q_shape, dtype=torch_dtype).uniform_(-1, 1).to(device=device)
    k = torch.empty(kv_shape, dtype=torch_dtype).uniform_(-1, 1).to(device=device)
    v = torch.empty(kv_shape, dtype=torch_dtype).uniform_(-1, 1).to(device=device)
    attention_output = torch.zeros(q_shape, dtype=torch_dtype).to(device=device)

    # 2. 生成block table - 传入 torch tensor
    block_table = gen_block_table(kv_cache_actual_seq, block_size, block_table_shape)

    # 3. 根据block table 将pa格式的数据转换成
    k_cache_bsnd, v_cache_bsnd = kv_cache_concat_bsnd(k, v, block_table, atten_cfg)

    for i in range(b):
        for j in range(s1):
            for n2_idx in range(nkv):
                # 从 torch tensor 获取值
                kv_seq_len = kv_cache_actual_seq[i].item()  # 使用 .item() 获取标量值
                seq_len = kv_seq_len - s1 + 1 + j
                q_bs = q[i * s1 + j]
                k_bs = k_cache_bsnd[i, :seq_len, n2_idx:n2_idx + 1].reshape(seq_len, d)
                v_bs = v_cache_bsnd[i, :seq_len, n2_idx:n2_idx + 1].reshape(seq_len, d)
                # MM1: 矩阵乘法
                qk_bmm_res = torch.matmul(q_bs, k_bs.transpose(1, 0))  # 1,nq, d  -> n_q,d @ d, s2_actual_len
                qk_ele_res = qk_bmm_res * atten_cfg.softmax_scale
                # Softmax计算
                softmax_res, _, _ = softmax(qk_ele_res, True)

                # MM2: 矩阵乘法
                bmm2_res = torch.matmul(softmax_res, v_bs)

                # 存储结果
                attention_output[i * s1 + j] = bmm2_res

    # 4. 准备测试数据 - 直接使用 torch 张量
    block_table_torch = block_table.to(dtype=torch.int32, device=device)
    act_seq_torch = kv_cache_actual_seq.to(dtype=torch.int32, device=device)  # 直接使用已有的 tensor

    out_torch = torch.full(q_shape, 9, dtype=torch_dtype, device=device)

    inputs = [q, k, v, block_table_torch, act_seq_torch]
    outputs = [out_torch]
    # 5. 执行kernel并获取结果
    pto_inputs = [pypto.from_torch(tensor, f"IN_{idx}") for idx, tensor in enumerate(inputs)]
    pto_outputs = [pypto.from_torch(tensor, f"OUT_{idx}") for idx, tensor in enumerate(outputs)]
    ifa_func(pto_inputs, pto_outputs)
    pypto.runtime._device_synchronize()

    y_data = out_torch.cpu()
    # 6. 与PyTorch参考实现对比
    detailed_allclose_manual(np.array(attention_output.cpu()).flatten(), np.array(y_data).flatten(), "attention")


def test_ifa():
    # 1. 设置参数
    device_id = os.environ.get('TILE_FWK_STEST_DEVICE_ID', 0)
    device = f'npu:{device_id}'
    atten_cfg, _ = get_qwen_common_config(device=device)

    # 检查 B 的大小和 actual_seq 长度是否相等
    assert atten_cfg.b == len(
        atten_cfg.actual_seq), f'{atten_cfg.b} {atten_cfg.actual_seq} B的大小必须和actual_seq长度相等'

    # 检查所有值是否都小于 s2
    if atten_cfg.actual_seq.device.type != 'cpu':
        actual_seq_cpu = atten_cfg.actual_seq.cpu()
    else:
        actual_seq_cpu = atten_cfg.actual_seq

    assert all(x <= atten_cfg.s2 for x in actual_seq_cpu), "所有值都必须小于s2"
    IFA(atten_cfg)


if __name__ == "__main__":
    test_ifa()
