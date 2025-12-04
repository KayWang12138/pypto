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
from dataclasses import dataclass
import math
import os
import pypto
import torch


@dataclass
class AttentionTileConfig:
    c1_tile_shape: list
    v1_tile_shape: list
    c2_tile_shape: list
    v2_tile_shape: list


def get_qwen_common_config():
    cube_tile = 128
    vector_tile = 128
    tile_cfg = AttentionTileConfig(
        [[cube_tile, cube_tile], [cube_tile, cube_tile], [cube_tile, cube_tile]],
        [vector_tile, vector_tile],
        [[cube_tile, cube_tile], [cube_tile, cube_tile], [cube_tile, cube_tile]],
        [vector_tile, vector_tile])
    return tile_cfg


def get_s1size(act_seqs, b_idx):
    if b_idx == 0:
        return act_seqs[b_idx]
    else:
        return act_seqs[b_idx] - act_seqs[b_idx - 1]


def get_bsoffset(act_seqs, b_idx):
    if b_idx == 0:
        return 0
    else:
        return act_seqs[b_idx - 1]


def kv_cache_concat_bsnd(k_cache, v_cache, kv_cache_actual_seq, block_table, b):
    block_size = k_cache.shape[1]
    n2 = k_cache.shape[2]
    d = k_cache.shape[3]
    dtype = k_cache.dtype

    kv_max = (max(kv_cache_actual_seq) + block_size - 1) // block_size * block_size

    # 使用 torch 创建张量，保持在同一设备上
    device = k_cache.device
    k_cache_bsnd = torch.zeros([b, kv_max, n2, d], dtype=dtype, device=device)
    v_cache_bsnd = torch.zeros([b, kv_max, n2, d], dtype=dtype, device=device)

    for b_idx in range(b):
        block_list = block_table[b_idx]
        k_temp_tensor = torch.zeros([1, kv_max, n2, d], dtype=dtype, device=device)
        v_temp_tensor = torch.zeros([1, kv_max, n2, d], dtype=dtype, device=device)
        s_idx = 0

        for _, block_idx in enumerate(block_list):
            if block_idx == -1:
                break
            # 使用 torch 的切片操作
            start_idx = s_idx * block_size
            end_idx = (s_idx + 1) * block_size

            k_temp_tensor[:, start_idx:end_idx, :, :] = k_cache[block_idx:block_idx + 1, :, :, :]
            v_temp_tensor[:, start_idx:end_idx, :, :] = v_cache[block_idx:block_idx + 1, :, :, :]
            s_idx += 1

        k_cache_bsnd[b_idx:b_idx + 1, :, :, :] = k_temp_tensor
        v_cache_bsnd[b_idx:b_idx + 1, :, :, :] = v_temp_tensor

    return k_cache_bsnd, v_cache_bsnd


@pypto.jit
def gated_attention_decode_func(inputs, outputs):
    # 1. 添加支持动态的config
    pypto.set_codegen_options(
        support_dynamic_unaligned=True,
        codegen_expression_fusion=True
    )

    # 2. 从入参拿到输入和输出tensor    
    q = inputs[0]
    k = inputs[1]
    v = inputs[2]
    block_table = inputs[3]
    q_act_seqs = inputs[4]
    kv_act_seqs = inputs[5]
    gate = inputs[6]
    weight = inputs[7]
    atten_out = outputs[0]

    # 3. 设置axis=0为动态shape     
    pypto.mark_dynamic(q, 0)
    pypto.mark_dynamic(k, 0)
    pypto.mark_dynamic(v, 0)
    pypto.mark_dynamic(q_act_seqs, 0)
    pypto.mark_dynamic(kv_act_seqs, 0)
    pypto.mark_dynamic(gate, 0)

    # 4. 定义动态函数
    tile_cfg = get_qwen_common_config()
    nq = q.shape[1]
    dn = q.shape[2]
    nkv = k.shape[2]
    softmax_scale = dn ** -0.5
    block_size = k.shape[1]
    dtype = q.dtype
    hidden_size = weight.shape[0]

    g_tile = nq // nkv
    s2_tile = block_size
    c1_tile = tile_cfg.c1_tile_shape
    v1_tile = tile_cfg.v1_tile_shape
    c2_tile = tile_cfg.c2_tile_shape
    v2_tile = tile_cfg.v2_tile_shape

    b_scalar = q_act_seqs.shape[0]
    n2_sym = nkv
    g_scalar = nq // nkv
    g_loop = g_scalar // g_tile

    # 5. 实现kernel逻辑，循环展开B动态轴
    for b_idx in pypto.loop(b_scalar, name="LOOP_b", idx_name="b_idx", submit_before_loop=True):
        cur_seq = kv_act_seqs[b_idx]
        s1_scalar = 0
        b_ofs = 0
        if pypto.cond(pypto.is_loop_begin(b_idx)):
            s1_scalar = q_act_seqs[b_idx]
            b_ofs = 0
        else:
            s1_scalar = q_act_seqs[b_idx] - q_act_seqs[b_idx - 1]
            b_ofs = q_act_seqs[b_idx - 1]
        for s1_idx in pypto.loop(s1_scalar, name="LOOP_s1", idx_name="s1_idx", submit_before_loop=True):
            s2_loop = (cur_seq + s2_tile - 1) // s2_tile
            bs_ofs = b_ofs + s1_idx
            bs_out = pypto.tensor([nq, dn], pypto.DT_FP32, "bs_out")
            for n2_idx in pypto.loop(n2_sym, name="LOOP_n2", idx_name="n2_idx"):
                for g_idx in pypto.loop(g_loop, name="LOOP_g", idx_name="g_idx"):
                    oi_upd = pypto.tensor([g_tile, dn], pypto.DT_FP32, "oi_upd")
                    li_upd = pypto.tensor([g_tile, 1], pypto.DT_FP32, "li_upd")
                    mi_upd = pypto.tensor([g_tile, 1], pypto.DT_FP32, "mi_upd")
                    for s2_idx in pypto.loop(s2_loop, name="LOOP_s2", idx_name="s2_idx"):
                        block_idx = block_table[b_idx, s2_idx]
                        n1g_ofs = n2_idx * g_scalar + g_idx * g_tile
                        actual_s2_tile = (cur_seq - s2_idx * s2_tile).min(s2_tile)
                        # 6. 按照计算图实现运算逻辑，设置set_vec_tile_shapes时应尽可能用满UB，但不要超过UB的大小。
                        pypto.set_vec_tile_shapes(16, v1_tile[0], v1_tile[1])
                        # 7. 通过view得到tile_q
                        qi = pypto.view(q, [1, g_tile, dn], [bs_ofs, n1g_ofs, 0],
                                        valid_shape=[1, g_tile, dn])
                        qi = pypto.reshape(qi, [g_tile, dn])
                        pypto.set_vec_tile_shapes(1, v1_tile[0], 1, v1_tile[1])
                        kj = pypto.view(k, [1, block_size, 1, dn], [block_idx, 0, n2_idx, 0],
                                        valid_shape=[1, actual_s2_tile, 1, dn])
                        kj = pypto.reshape(kj, [block_size, dn], valid_shape=[actual_s2_tile, dn])
                        vj = pypto.view(v, [1, block_size, 1, dn], [block_idx, 0, n2_idx, 0],
                                        valid_shape=[1, actual_s2_tile, 1, dn])
                        vj = pypto.reshape(vj, [block_size, dn], valid_shape=[actual_s2_tile, dn])

                        # c1
                        # 8. 下面是flash attention的计算逻辑
                        pypto.set_cube_tile_shapes(c1_tile[0], c1_tile[1], c1_tile[2])
                        sij = pypto.matmul(qi, kj, pypto.DT_FP32, a_trans=False, b_trans=True)
                        sij = pypto.reshape(sij, [g_tile, s2_tile], valid_shape=[g_tile, actual_s2_tile])
                        # v1
                        pypto.set_vec_tile_shapes(v1_tile[0], v1_tile[1])
                        sij_scale = pypto.mul(sij, softmax_scale)
                        tilda_mij = pypto.amax(sij_scale, keepdim=True)
                        tsub = pypto.sub(sij_scale, tilda_mij)
                        tilda_pij = pypto.exp(tsub)
                        tilda_pij_fp16 = pypto.cast(tilda_pij, dtype)
                        tilda_lij = pypto.sum(tilda_pij, keepdim=True)
                        if pypto.cond(pypto.is_loop_begin(s2_idx)):
                            # c2
                            pypto.set_cube_tile_shapes(c2_tile[0], c2_tile[1], c2_tile[2])
                            oi_tmp = pypto.matmul(tilda_pij_fp16, vj, pypto.DT_FP32)
                            oi_upd[:] = pypto.tensor(oi_tmp.shape, pypto.DT_FP32, "oi_upd")
                            pypto.set_vec_tile_shapes(v2_tile[0], v2_tile[1])
                            if pypto.cond(pypto.is_loop_end(s2_idx)):
                                oi_upd[:] = pypto.div(oi_tmp, tilda_lij)
                                # 9. 将结果搬运到输出tensor上
                                pypto.assemble(oi_upd, [n1g_ofs, 0], bs_out)
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
                                # 10. 将结果搬运到输出tensor上
                                pypto.assemble(oi_upd, [n1g_ofs, 0], bs_out)
                            else:
                                oi_upd[:] = oi_tmp
                            li_upd[:] = li_new
                            mi_upd[:] = mi_new

            for _ in pypto.loop(1, name="LOOP_gate_linear", idx_name="_", submit_before_loop=True):
                pypto.set_vec_tile_shapes(16, v1_tile[0], v1_tile[1])
                gate_bs = pypto.view(gate, [1, nq, dn], [bs_ofs, 0, 0])
                gate_bs = pypto.reshape(gate_bs, [nq, dn])
                gate_bs_fp32 = pypto.cast(gate_bs, pypto.DT_FP32)
                pypto.set_vec_tile_shapes(v1_tile[0], v1_tile[1])
                weight_bs = pypto.view(weight, [hidden_size, nq * dn], [0, 0])
                weight_fp32 = pypto.cast(weight_bs, pypto.DT_FP32)
                out_gate = pypto.mul(bs_out, gate_bs_fp32)
                out_gate = pypto.reshape(out_gate, [1, nq * dn])
                pypto.set_cube_tile_shapes(c2_tile[0], c2_tile[1], c2_tile[2])
                out_linear = pypto.matmul(out_gate, weight_fp32, pypto.DT_FP32, a_trans=False, b_trans=True)
                out_linear = pypto.cast(out_linear, dtype)
                pypto.assemble(out_linear, [bs_ofs, 0], atten_out)


def gated_attention_decode(
    q_torch,
    k_torch,
    v_torch,
    block_table_torch,
    q_act_seq_torch,
    kv_act_seq_torch,
    gate_torch,
    weight_torch
    ):

    device = q_torch.device
    bs1 = q_torch.shape[0]
    n1 = q_torch.shape[1]
    d = q_torch.shape[2]
    hidden_size = weight_torch.shape[0]
    gate_torch = gate_torch.view(bs1, n1, d)
    out_torch = torch.full([bs1, hidden_size], 9, dtype=q_torch.dtype, device=device)

    inputs = [q_torch, k_torch, v_torch, block_table_torch, q_act_seq_torch, kv_act_seq_torch, gate_torch, weight_torch]
    outputs = [out_torch]

    pto_inputs = [pypto.from_torch(tensor, f'IN_{idx}') for idx, tensor in enumerate(inputs)]
    pto_outputs = [pypto.from_torch(tensor, f'IN_{idx}') for idx, tensor in enumerate(outputs)]

    gated_attention_decode_func(pto_inputs, pto_outputs)

    return out_torch


def test_gated_attention_decode():
    device_id = os.environ.get('TILE_FWK_STEST_DEVICE_ID', 0)
    torch.npu.set_device(int(device_id))
    b = 4
    bs1 = 4
    nq = 2
    nkv = 1
    g = nq // nkv
    d = 256
    softmax_scale = d ** -0.5
    hidden_size = 2048
    block_size = 128
    kv_num_blocks = 2049
    block_table_batch = 4
    max_num_blocks_per_query = 1
    kv_actual_seq = [10, 10, 10, 10]
    q_actual_seq = [1, 2, 3, 4]

    q_shape = [bs1, nq, d]
    kv_shape = [kv_num_blocks, block_size, nkv, d]
    block_table_shape = [block_table_batch, max_num_blocks_per_query]
    gate_shape = [bs1, nq, d]
    weight_shape = [hidden_size, nq * d]
    out_shape = [bs1, hidden_size]

    torch_dtype = torch.float16
    device = f'npu:{device_id}'

    q_torch = torch.rand(q_shape, dtype=torch_dtype).to(device=device)
    k_torch = torch.rand(kv_shape, dtype=torch_dtype).to(device=device)
    v_torch = torch.rand(kv_shape, dtype=torch_dtype).to(device=device)
    block_table_torch = torch.randint(0, kv_num_blocks, block_table_shape, dtype=torch.int32)
    gate_torch = torch.rand(gate_shape, dtype=torch_dtype).to(device=device)
    weight_torch = torch.rand(weight_shape, dtype=torch_dtype).to(device=device)
    gate_fp32 = gate_torch.to(torch.float32)
    weight_fp32 = weight_torch.to(torch.float32)

    q_act_seq_torch = torch.tensor(q_actual_seq).to(dtype=torch.int32)
    kv_act_seq_torch = torch.tensor(kv_actual_seq).to(dtype=torch.int32)
    torch_golden = torch.zeros(out_shape, dtype=torch_dtype).to(device=device)

    k_bsnd, v_bsnd = kv_cache_concat_bsnd(k_torch, v_torch, kv_act_seq_torch, block_table_torch, b)

    for i in range(b):
        s1 = get_s1size(q_act_seq_torch, i)
        b_ofs = get_bsoffset(q_act_seq_torch, i)
        for j in range(s1):
            bs_ofs = b_ofs + j
            atten_out = torch.zeros([nq, d], dtype=torch.float32).to(device=device)
            for n2_idx in range(nkv):
                s2 = kv_act_seq_torch[i]
                # attention
                q_bs = q_torch[bs_ofs, g * n2_idx: g * (n2_idx + 1)]
                k_bs = k_bsnd[i, :s2, n2_idx]
                v_bs = v_bsnd[i, :s2, n2_idx]
                qk_mm = torch.matmul(q_bs, k_bs.T).to(torch.float32)
                qk_mm_scale = qk_mm * softmax_scale
                softmax_res = torch.nn.functional.softmax(qk_mm_scale, dim=-1).to(torch_dtype)
                atten_out[g * n2_idx: g * (n2_idx + 1), :] = torch.matmul(softmax_res, v_bs)
            # gate
            gate_fp32_bs = gate_fp32[bs_ofs]
            gate_out = (atten_out * gate_fp32_bs).reshape(1, nq * d)
            # linear
            linear_out = torch.matmul(gate_out, weight_fp32.T).to(torch.float32)
            torch_golden[bs_ofs] = linear_out.reshape(hidden_size).to(torch_dtype)

    block_table_torch = block_table_torch.to(device=device)
    q_act_seq_torch = q_act_seq_torch.to(device=device)
    kv_act_seq_torch = kv_act_seq_torch.to(device=device)
    pto_out = gated_attention_decode(q_torch, k_torch, v_torch, block_table_torch, 
        q_act_seq_torch, kv_act_seq_torch, gate_torch, weight_torch)

    assert torch.allclose(pto_out, torch_golden, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    test_gated_attention_decode()