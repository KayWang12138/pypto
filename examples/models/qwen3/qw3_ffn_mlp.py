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
import numpy as np
from numpy.testing import assert_allclose
import torch


def main():
    test_qwen3_ffn()


def ffn_golden_torch(topk, expand_x, expert_tokens, weight_gate_upper, weight_down_proj):
    bs = expand_x.shape[0] // topk
    h = expand_x.shape[1]
    d = weight_down_proj.shape[2]
    # 使用 torch.zeros_like 创建相同形状和类型的张量
    out = torch.zeros_like(expand_x)

    print("bs: ", bs)
    print("h: ", h)
    print("d: ", d)
    print("expert_tokens: ", expert_tokens)
    start_idx = 0  # 初始化起始索引
    for i in range(expert_tokens.shape[0]):
        token_count = expert_tokens[i].item()
        if token_count <= 0:
            print(f"专家 {i} 无token，跳过")
            continue
        end_idx = start_idx + token_count  # 计算结束索引
        print("start_idx: ", start_idx)
        print("end_idx: ", end_idx)
        # 确保不越界（安全保护）
        if end_idx > expand_x.shape[0]:
            print(f"警告：专家 {i} 的token计数超出范围，调整到数组末尾")
            end_idx = expand_x.shape[0]

        selected_x = expand_x[start_idx:end_idx, :]
        # 获取当前专家的权重
        current_gate_weight = weight_gate_upper[i]  # [h, d*2]
        current_down_weight = weight_down_proj[i]   # [h, d]
        # 矩阵乘法: [token_count, h] @ [h, d*2] = [token_count, d*2]
        gate_output = torch.matmul(selected_x.float(), current_gate_weight.float().T)
        # 分割为left和right
        split_dim = gate_output.shape[-1] // 2
        left, right = torch.split(gate_output, split_dim, dim=-1)
        # 计算Swish-GLU: x * sigmoid(x) - 使用PyTorch的sigmoid函数
        swiglu = left * 1 / (1 + torch.exp(-left))
        # Swish-GLU与right相乘
        swiglu_right = swiglu * right
        # 下层投影: [token_count, d] @ [d, h] = [token_count, h]
        expert_output = torch.matmul(
            swiglu_right.to(current_down_weight.dtype).float(),
            current_down_weight.float().T  # 转置
        )
        # 累加结果
        out[start_idx:end_idx, :] = expert_output.to(expand_x.dtype)
        # 更新下一个专家的起始索引
        start_idx = end_idx

    return out


def get_token_acc_table(expert_tokens):
    assert len(expert_tokens.shape) == 1
    token_acc_table = torch.zeros_like(expert_tokens)
    for i in range(1, expert_tokens.shape[0]):
        token_acc_table[i] = torch.sum(expert_tokens[0:i])
    return token_acc_table


def gen_input(b, s, topk, per_expert_num, hidden_size, intermediate_size, dtypes, device_id):
    expand_x_tensor = torch.randn((b * s * topk, hidden_size), dtype = dtypes, device = f'npu:{device_id}') * 0.01 * 2 - 0.01
    expert_tokens_tesnor =  torch.randint(0, 2, (per_expert_num, ), dtype = torch.int32, device = f'npu:{device_id}')

    token_acc_table_tensor = get_token_acc_table(expert_tokens_tesnor).to(torch.int32)
    weight_gate_upper_tensor =  torch.randn((per_expert_num, intermediate_size * 2, hidden_size), dtype = dtypes, device = f'npu:{device_id}')  * 0.01 * 2 - 0.01
    weight_down_proj_tensor =  torch.randn((per_expert_num, hidden_size, intermediate_size), dtype = dtypes, device = f'npu:{device_id}')  * 0.01 * 2 - 0.01
    out_tensor = torch.zeros_like(expand_x_tensor, device = f'npu:{device_id}')
    return expand_x_tensor, expert_tokens_tesnor, token_acc_table_tensor, weight_gate_upper_tensor, weight_down_proj_tensor, out_tensor


def expert_infer_base(**kwargs):
    # 入参信息获取
    exp_idx = kwargs.get("exp_idx")
    loop_base = kwargs.get("loop_base")
    token_loop_idx = kwargs.get("token_loop_idx")
    expand_x = kwargs.get("expand_x")
    expert_tokens = kwargs.get("expert_tokens")
    token_acc_table = kwargs.get("token_acc_table")
    weight_gate_upper = kwargs.get("weight_gate_upper")
    weight_down_proj = kwargs.get("weight_down_proj")
    vec_tile_shape = kwargs.get("vec_tile_shape")
    cube_tile_shape = kwargs.get("cube_tile_shape")
    ffn_out = kwargs.get("ffn_out")

    hidden_size = expand_x.shape[1]
    intermediate_size = weight_down_proj.shape[1]
    x_Dtype = expand_x.dtype

    # 计算对应激活专家的偏移地址
    pypto.set_vec_tile_shapes(32)
    # 获取该激活专家的当前loop参与计算的token
    token_num = expert_tokens[exp_idx,]
    # 获取该激活专家的当前loop参与计算的token的偏移地址
    expand_x_offset_start = token_acc_table[exp_idx, ]
    expand_x_offset = [expand_x_offset_start + token_loop_idx * loop_base, 0]
    # 获取该激活专家权重的偏移地址
    weight_13_offset = [exp_idx * (intermediate_size * 2), 0]
    weight_2_offset = [exp_idx * hidden_size, 0]
    # 获取当前专家的实际token数
    cur_valid_size = (token_num - token_loop_idx * loop_base).min(loop_base)
    pypto.set_vec_tile_shapes(vec_tile_shape[0], vec_tile_shape[1])
    x = pypto.view(expand_x, [loop_base, hidden_size], expand_x_offset, valid_shape=[cur_valid_size, hidden_size])
    # 获取当前专家的weght_13[up_weight + gate_weight]
    ffn_weight_2d = pypto.view(weight_gate_upper, [intermediate_size * 2, hidden_size], weight_13_offset)

    # # 获取当前专家的weght_2[down_weight]
    down_proj_2d = pypto.view(weight_down_proj, [hidden_size, intermediate_size], weight_2_offset)

    # up_proj的matmul计算
    pypto.set_cube_tile_shapes([cube_tile_shape[0], cube_tile_shape[0]], [cube_tile_shape[1], cube_tile_shape[1]], [cube_tile_shape[2], cube_tile_shape[2]])
    pypto.set_matrix_size({loop_base, ffn_weight_2d.shape[1], ffn_weight_2d.shape[0]})
    gate = pypto.matmul(x, ffn_weight_2d, pypto.DT_FP32, b_trans=True)

    pypto.set_vec_tile_shapes(vec_tile_shape[0], vec_tile_shape[1])
    gate_left = pypto.view(gate, [loop_base, intermediate_size], [0, 0])
    gate_right = pypto.view(gate, [loop_base, intermediate_size], [0, intermediate_size])

    # SwiGlu计算[x / (1 + e^(-x))]
    swiglu_a = pypto.mul(gate_left, -1.0)
    swiglu_b = pypto.exp(swiglu_a)
    swiglu_c = pypto.add(swiglu_b, 1.0)
    swiglu_out = pypto.div(gate_left, swiglu_c)

    # SwiGlu计算结果与gate_right相乘
    swiglu = pypto.mul(swiglu_out, gate_right)

    # down_proj的matmul计算
    swish_fp16 = pypto.cast(swiglu, x_Dtype)
    pypto.set_cube_tile_shapes([cube_tile_shape[0], cube_tile_shape[0]], [cube_tile_shape[1], cube_tile_shape[1]], [cube_tile_shape[2], cube_tile_shape[2]])
    pypto.set_matrix_size({loop_base, down_proj_2d.shape[1], down_proj_2d.shape[0]})
    out = pypto.matmul(swish_fp16, down_proj_2d, x_Dtype, b_trans=True)
    pypto.assemble(out, expand_x_offset, ffn_out)


# tiling config
vec_tile_shape = (64, 128)
cube_tile_shape = (64, 128, 128)
loop_base = 16


@pypto.jit(
    host_options={"only_codegen": True},
    codegen_options={"support_dynamic_unaligned": True}
)
def moe_main(inputs, outputs):

    expand_x = inputs[0]
    expert_tokens = inputs[1]
    token_acc_table = inputs[2]
    weight_gate_upper = inputs[3]
    weight_down_proj = inputs[4]
    ffn_out = outputs[0]
    weight_dtype = weight_down_proj.dtype

    # 获取当前device上专家总数
    expert_num = expert_tokens.shape[0]
    w1_2d_shape = (weight_gate_upper.shape[0] * weight_gate_upper.shape[1], weight_gate_upper.shape[2])
    w2_2d_shape = (weight_down_proj.shape[0] * weight_down_proj.shape[1], weight_down_proj.shape[2])
    for _ in pypto.loop(0, 1, 1, name="LOOP_RESHAPE", idx_name="reshape_inplace_1"):
        w1_2d = pypto.reshape(weight_gate_upper, w1_2d_shape, inplace=True)
        w2_2d = pypto.reshape(weight_down_proj, w2_2d_shape, inplace=True)
    for exp_idx in pypto.loop(0, expert_num, 1, name="LOOP_FFN_L0", idx_name="exp_idx"):
        def loop_expert(exp_idx):
            # 获取激活专家的token数
            token_num = expert_tokens[exp_idx, ]
            # 每个专家单次计算16token，不足部分会进行pad
            exp_loop_times = (token_num + loop_base - 1) / loop_base
            for token_loop_idx in pypto.loop(0, exp_loop_times, 1, name="LOOP_FFN_L1", idx_name="token_loop_idx"):
                def loop_token(exp_idx, token_loop_idx):
                    expert_infer_base(
                        exp_idx=exp_idx,
                        token_loop_idx=token_loop_idx,
                        loop_base=loop_base,
                        expand_x=expand_x,
                        expert_tokens=expert_tokens,
                        token_acc_table=token_acc_table,
                        weight_gate_upper=w1_2d,
                        weight_down_proj=w2_2d,
                        ffn_out=ffn_out,
                        vec_tile_shape=vec_tile_shape,
                        cube_tile_shape=cube_tile_shape,
                        )
                loop_token(exp_idx, token_loop_idx)
        loop_expert(exp_idx)


def test_qwen3_ffn():
    dtype = torch.bfloat16
    # parameter config
    b = 2
    s = 1
    intermediate_size = 768
    hidden_size = 2048
    per_expert_num = 16
    topk = 8
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)


    inputs_list = gen_input(b, s, topk, per_expert_num, hidden_size, intermediate_size, dtype, device_id)
    inputs = {
        inputs_list[0]: [0],
        inputs_list[1]: [],
        inputs_list[2]: [],
        inputs_list[3]: [],
        inputs_list[4]: []
    }
    outputs = {
        inputs_list[5]: []
    }

    pto_inputs = [pypto.from_torch(tensor, dynamic_axis=axis) for tensor, axis in inputs.items()]
    pto_outputs = [pypto.from_torch(tensor, dynamic_axis=axis) for tensor, axis in outputs.items()]
    moe_main(pto_inputs, pto_outputs)
    pypto.runtime._device_synchronize()

    # golden
    golden = ffn_golden_torch(topk, inputs_list[0], inputs_list[1], inputs_list[3], inputs_list[4])
    assert_allclose(np.array(inputs_list[5].cpu().flatten().tolist()), np.array(golden.cpu().flatten().tolist()), rtol=0.005, atol=0.005)

if __name__ == "__main__":
    main()