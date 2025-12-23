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


def get_s1_size(act_seqs, b_idx):
    # torch 获取TND格式下每个batch的s1大小
    if b_idx == 0:
        return act_seqs[b_idx]
    else:
        return act_seqs[b_idx] - act_seqs[b_idx - 1]


def get_b_offset(act_seqs, b_idx):
    # torch 获取TND格式下每个batch的起始索引
    if b_idx == 0:
        return 0
    else:
        return act_seqs[b_idx - 1]


@pypto.jit(
        codegen_options={"codegen_expression_fusion": True}
)
def gated_attention_prefill_func(q, k, v, act_seqs, gate, weight, final_out):
    # 1. 获取参数信息
    tile_cfg = get_qwen_common_config()
    nq = q.shape[1]
    dn = q.shape[2]
    nkv = k.shape[1]
    dtype = q.dtype
    softmax_scale = dn ** -0.5
    hidden_size = weight.shape[0]

    g_tile = nq // nkv
    c1_tile = tile_cfg.c1_tile_shape
    v1_tile = tile_cfg.v1_tile_shape
    c2_tile = tile_cfg.c2_tile_shape
    v2_tile = tile_cfg.v2_tile_shape

    s2_tile = 128
    b_scalar = act_seqs.shape[0]
    n2_scalar = nkv
    g_scalar = nq // nkv
    g_loop = g_scalar // g_tile

    # 2. 实现kernel逻辑，循环展开B动态轴
    for b_idx in pypto.loop(b_scalar, name="LOOP_b", idx_name="b_idx", submit_before_loop=True):
        s1_scalar = 0
        b_ofs = 0
        # 3. 动态获取TND格式下每个batch的s1大小与起始索引位置
        if pypto.cond(pypto.is_loop_begin(b_idx)):
            s1_scalar = act_seqs[b_idx]
            b_ofs = 0
        else:
            s1_scalar = act_seqs[b_idx] - act_seqs[b_idx - 1]
            b_ofs = act_seqs[b_idx - 1]
        for s1_idx in pypto.loop(s1_scalar, name="LOOP_s1", idx_name="s1_idx", submit_before_loop=True):
            kv_seq_len = s1_idx + 1
            s2_loop = (kv_seq_len + s2_tile - 1) // s2_tile
            bs_ofs = b_ofs + s1_idx
            atten_out = pypto.tensor([nq, dn], pypto.DT_FP32, "atten_out")
            for n2_idx in pypto.loop(n2_scalar, name="LOOP_n2", idx_name="n2_idx"):
                for g_idx in pypto.loop(g_loop, name="LOOP_g", idx_name="g_idx"):
                    oi_upd = pypto.tensor([g_tile, dn], pypto.DT_FP32, "oi_upd")
                    li_upd = pypto.tensor([g_tile, 1], pypto.DT_FP32, "li_upd")
                    mi_upd = pypto.tensor([g_tile, 1], pypto.DT_FP32, "mi_upd")
                    for s2_idx in pypto.loop(s2_loop, name="LOOP_s2", idx_name="s2_idx"):
                        s2_bs_ofs = b_ofs + s2_idx * s2_tile
                        n1g_ofs = n2_idx * g_scalar + g_idx * g_tile
                        actual_s2_tile = (kv_seq_len - s2_idx * s2_tile).min(s2_tile)
                        # 4. 按照计算图实现运算逻辑，设置set_vec_tile_shapes时应尽可能用满UB，但不要超过UB的大小。
                        pypto.set_vec_tile_shapes(16, v1_tile[0], v1_tile[1])
                        # 5. 通过view得到tile_q
                        qi = pypto.view(q, [1, g_tile, dn], [bs_ofs, n1g_ofs, 0])
                        qi = pypto.reshape(qi, [g_tile, dn])
                        kj = pypto.view(k, [s2_tile, 1, dn], [s2_bs_ofs, n2_idx, 0],
                                        valid_shape=[actual_s2_tile, 1, dn])
                        kj = pypto.reshape(kj, [s2_tile, dn], valid_shape=[actual_s2_tile, dn])
                        vj = pypto.view(v, [s2_tile, 1, dn], [s2_bs_ofs, n2_idx, 0],
                                        valid_shape=[actual_s2_tile, 1, dn])
                        vj = pypto.reshape(vj, [s2_tile, dn], valid_shape=[actual_s2_tile, dn])

                        # c1
                        # 6. 下面是flash attention的计算逻辑
                        pypto.set_cube_tile_shapes(c1_tile[0], c1_tile[1], c1_tile[2])
                        sij = pypto.matmul(qi, kj, pypto.DT_FP32, a_trans=False, b_trans=True)
                        sij = pypto.reshape(sij, [g_tile, s2_tile], valid_shape=[g_tile, actual_s2_tile])
                        # v1
                        pypto.set_vec_tile_shapes(v1_tile[0], v1_tile[1])
                        sij_scale = pypto.mul(sij, softmax_scale)
                        tilda_mij = pypto.amax(sij_scale, -1, keepdim=True)
                        tsub = pypto.sub(sij_scale, tilda_mij)
                        tilda_pij = pypto.exp(tsub)
                        tilda_pij_fp16 = pypto.cast(tilda_pij, dtype)
                        tilda_lij = pypto.sum(tilda_pij, -1, keepdim=True)
                        if pypto.cond(pypto.is_loop_begin(s2_idx)):
                            # c2
                            pypto.set_cube_tile_shapes(c2_tile[0], c2_tile[1], c2_tile[2])
                            oi_tmp = pypto.matmul(tilda_pij_fp16, vj, pypto.DT_FP32)
                            oi_upd[:] = pypto.tensor(oi_tmp.shape, pypto.DT_FP32, "oi_upd")
                            pypto.set_vec_tile_shapes(v2_tile[0], v2_tile[1])
                            if pypto.cond(pypto.is_loop_end(s2_idx)):
                                oi_upd[:] = pypto.div(oi_tmp, tilda_lij)
                                # 7. 将attention结果搬运到临时tensor上
                                pypto.assemble(oi_upd, [n1g_ofs, 0], atten_out)
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
                                # 8. 将attention结果搬运到临时tensor上
                                pypto.assemble(oi_upd, [n1g_ofs, 0], atten_out)
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
                # 9、gate相乘
                gate_out = pypto.mul(atten_out, gate_bs_fp32)
                gate_out = pypto.reshape(gate_out, [1, nq * dn])
                pypto.set_cube_tile_shapes(c2_tile[0], c2_tile[1], c2_tile[2])
                # 10、出口linear
                linear_out = pypto.matmul(gate_out, weight_fp32, pypto.DT_FP32, a_trans=False, b_trans=True)
                linear_out = pypto.cast(linear_out, dtype)
                pypto.assemble(linear_out, [bs_ofs, 0], final_out)


def gated_attention_prefill(
    q_torch,
    k_torch,
    v_torch,
    act_seq_torch,
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

    inputs = {
        q_torch: [0],
        k_torch: [0],
        v_torch: [0],
        act_seq_torch: [0],
        gate_torch: [0],
        weight_torch: []
    }
    outputs = {
        out_torch: []
    }

    pto_inputs = [pypto.from_torch(tensor, dynamic_axis=axis) for tensor, axis in inputs.items()]
    pto_outputs = [pypto.from_torch(tensor, dynamic_axis=axis) for tensor, axis in outputs.items()]

    gated_attention_prefill_func(*pto_inputs, *pto_outputs)

    return out_torch


def test_gated_attention_prefill():
    device_id = os.environ.get('TILE_FWK_STEST_DEVICE_ID', 1)
    torch.npu.set_device(int(device_id))
    b = 4
    bs1 = 22
    nq = 2
    nkv = 1
    g = nq // nkv
    d = 256
    softmax_scale = d ** -0.5
    hidden_size = 2048
    actual_seq = [5, 8, 17, 22]

    q_shape = [bs1, nq, d]
    kv_shape = [bs1, nkv, d]
    gate_shape = [bs1, nq, d]
    weight_shape = [hidden_size, nq * d]
    out_shape = [bs1, hidden_size]

    torch_dtype = torch.float16
    device = f'npu:{device_id}'

    q_torch = torch.rand(q_shape, dtype=torch_dtype).to(device=device)
    k_torch = torch.rand(kv_shape, dtype=torch_dtype).to(device=device)
    v_torch = torch.rand(kv_shape, dtype=torch_dtype).to(device=device)
    gate_torch = torch.rand(gate_shape, dtype=torch_dtype).to(device=device)
    weight_torch = torch.rand(weight_shape, dtype=torch_dtype).to(device=device)
    gate_fp32 = gate_torch.to(torch.float32)
    weight_fp32 = weight_torch.to(torch.float32)

    act_seq_torch = torch.tensor(actual_seq).to(dtype=torch.int32, device=device)
    torch_golden = torch.zeros(out_shape, dtype=torch_dtype).to(device=device)

    for i in range(b):
        s1 = get_s1_size(actual_seq, i)
        b_ofs = get_b_offset(actual_seq, i)
        for j in range(s1):
            bs_ofs = b_ofs + j
            atten_out = torch.zeros([nq, d], dtype=torch.float32).to(device=device)
            for n2_idx in range(nkv):
                # attention
                kv_seq_len = j + 1
                q_bs = q_torch[bs_ofs, g * n2_idx: g * (n2_idx + 1)]
                k_bs = k_torch[b_ofs: b_ofs + kv_seq_len, n2_idx]
                v_bs = v_torch[b_ofs: b_ofs + kv_seq_len, n2_idx]
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

    pto_out = gated_attention_prefill(q_torch, k_torch, v_torch, act_seq_torch, gate_torch, weight_torch)

    assert torch.allclose(pto_out, torch_golden, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    test_gated_attention_prefill()
