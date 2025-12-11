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
import torch
import numpy as np
import pypto
from numpy.testing import assert_allclose


def main():
    test_qwen3next_ffn()


def ffn_golden_torch_with_shared(
    expand_x,
    shared_gate_upper_weight,
    shared_down_weight,
    shared_router_weight
):
    # 可以考虑拼接shared_gate_upper_weight与shared_router_weight，只调用一次matmul算子
    # SwiGLU: (x @ shared_gate_weight.T) -> sigmoid, (x @ shared_up_weight.T) -> up
    gate_output = torch.matmul(expand_x.float(), shared_gate_upper_weight.float().T)
    split_dim = gate_output.shape[-1] // 2
    left, right = torch.split(gate_output, split_dim, dim=-1)
    swiglu = left * torch.sigmoid(left)
    swiglu_right = swiglu * right
    shared_output = torch.matmul(
        swiglu_right.to(shared_down_weight.dtype).float(), shared_down_weight.float().T
        )

    # Shared Gate (sigmoid(expand_x @ shared_router_weight.T) * shared_output)
    shared_gate_logits = torch.matmul(expand_x.float(), shared_router_weight.float().T)
    shared_gate = torch.sigmoid(shared_gate_logits)                                # [N, 1]

    final_output = (shared_gate * shared_output).to(expand_x.dtype)

    return final_output


def gen_input_with_shared(b, s, hidden_size, intermediate_size, dtypes, device_id):
    expand_x_tensor = torch.randn((b * s, hidden_size), dtype=dtypes, device=f'npu:{device_id}') * 0.01 * 2 - 0.01

    # 共享专家权重
    shared_gate_upper_weight = torch.randn(
        (intermediate_size * 2, hidden_size), dtype=dtypes, device=f'npu:{device_id}'
        ) * 0.01 * 2 - 0.01
    shared_down_weight = torch.randn(
        (hidden_size, intermediate_size), dtype=dtypes, device=f'npu:{device_id}'
        ) * 0.01 * 2 - 0.01
    shared_router_weight = torch.randn(
        (1, hidden_size), dtype=dtypes, device=f'npu:{device_id}'
        ) * 0.01 * 2 - 0.01  # 输出标量

    out_tensor = torch.zeros_like(expand_x_tensor, device=f'npu:{device_id}')

    return expand_x_tensor, shared_gate_upper_weight, shared_down_weight, shared_router_weight, out_tensor


def shared_expert_infer(**kwargs):
    expand_x = kwargs.get("expand_x")
    shared_gate_upper_weight = kwargs.get("shared_gate_upper_weight")
    shared_down_weight = kwargs.get("shared_down_weight")
    shared_router_weight = kwargs.get("shared_router_weight")
    token_loop_idx = kwargs.get("token_loop_idx")
    base_loop = kwargs.get("base_loop")
    ffn_out = kwargs.get("ffn_out")
    vec_tile_shape = kwargs.get("vec_tile_shape")
    cube_tile_shape = kwargs.get("cube_tile_shape")

    token_num = expand_x.shape[0]
    hidden_size = expand_x.shape[1]
    intermediate_size = shared_down_weight.shape[1]  # intermediate_size
    x_dtype = expand_x.dtype

    # 偏移值
    pypto.set_vec_tile_shapes(32)
    start_idx = token_loop_idx * base_loop
    cur_valid_size = (token_num - start_idx).min(base_loop)
    expand_x_offset = [start_idx, 0]

    # 偏移
    pypto.set_vec_tile_shapes(vec_tile_shape[0], vec_tile_shape[1])
    x_block = pypto.view(expand_x, [base_loop, hidden_size], expand_x_offset, valid_shape=[cur_valid_size, hidden_size])

    # 上投影
    pypto.set_cube_tile_shapes([cube_tile_shape[0], cube_tile_shape[0]], 
                            [cube_tile_shape[1], cube_tile_shape[1]], 
                            [cube_tile_shape[2], cube_tile_shape[2]])
    pypto.set_matrix_size({base_loop, shared_gate_upper_weight.shape[1], shared_gate_upper_weight.shape[0]})
    gate = pypto.matmul(x_block, shared_gate_upper_weight, pypto.DT_FP32, b_trans=True)

    pypto.set_vec_tile_shapes(vec_tile_shape[0], vec_tile_shape[1])
    gate_left = pypto.view(gate, [base_loop, intermediate_size], [0, 0])
    gate_right = pypto.view(gate, [base_loop, intermediate_size], [0, intermediate_size])

    # SwiGlu计算[x / (1 + e^(-x))] <==> [x * sigmoid(x)]
    swiglu_sg = pypto.sigmoid(gate_left)
    swiglu_out = pypto.mul(gate_left, swiglu_sg)
    # SwiGlu计算结果与gate_right相乘
    swiglu = pypto.mul(swiglu_out, gate_right)

    # 下投影
    swiglu_fp16 = pypto.cast(swiglu, x_dtype)
    pypto.set_cube_tile_shapes([cube_tile_shape[0], cube_tile_shape[0]], 
                            [cube_tile_shape[1], cube_tile_shape[1]], 
                            [cube_tile_shape[2], cube_tile_shape[2]])
    pypto.set_matrix_size({base_loop, shared_down_weight.shape[1], shared_down_weight.shape[0]})
    shared_output = pypto.matmul(swiglu_fp16, shared_down_weight, pypto.DT_FP32, b_trans=True)

    # --- Shared Gate: x_block @ shared_router_weight.T (output [N, 1]) ---
    pypto.set_matrix_size({base_loop, shared_router_weight.shape[1], shared_router_weight.shape[0]})
    gate_logits = pypto.matmul(x_block, shared_router_weight, pypto.DT_FP32, b_trans=True)

    pypto.set_vec_tile_shapes(vec_tile_shape[0], vec_tile_shape[1])
    shared_gate = pypto.sigmoid(gate_logits)

    # 对位乘
    gated_shared = pypto.mul(shared_gate, shared_output)
    gated_shared_fp16 = pypto.cast(gated_shared, x_dtype)

    pypto.assemble(gated_shared_fp16, [start_idx, 0], ffn_out)


# tiling config
vec_tile_shape_global = (64, 128)
cube_tile_shape_global = (64, 128, 128)
base_loop_global = 16


@pypto.jit(
    host_options={"only_codegen": True},
    codegen_options={"support_dynamic_unaligned": True}
)
def moe_with_shared_main(inputs, outputs):
    expand_x = inputs[0]
    shared_gate_upper_weight = inputs[1]
    shared_down_weight = inputs[2]
    shared_router_weight = inputs[3]
    ffn_out = outputs[0]

    total_token_num = expand_x.shape[0]
    
    exp_loop_times = (total_token_num + base_loop_global - 1) // base_loop_global
    for token_loop_idx in pypto.loop(0, exp_loop_times, 1, name="LOOP_Shared_Expert", idx_name="token_loop_idx_shared"):
        def loop_token_shared(token_loop_idx):
            shared_expert_infer(
                expand_x=expand_x,
                shared_gate_upper_weight=shared_gate_upper_weight,
                shared_down_weight=shared_down_weight,
                shared_router_weight=shared_router_weight,
                token_loop_idx=token_loop_idx,
                base_loop=base_loop_global,
                ffn_out=ffn_out,
                vec_tile_shape=vec_tile_shape_global,
                cube_tile_shape=cube_tile_shape_global
                )
        loop_token_shared(token_loop_idx)


def test_qwen3next_ffn():
    dtype = torch.bfloat16
    b, s = 16, 1
    intermediate_size = 512
    hidden_size = 2048
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)

    inputs_list = gen_input_with_shared(
        b, s, hidden_size, intermediate_size, dtype, device_id
    )
    inputs = {
        inputs_list[0]: [0],
        inputs_list[1]: [],
        inputs_list[2]: [],
        inputs_list[3]: []
    }
    outputs = {
        inputs_list[4]: []
    }

    pypto_inputs = [pypto.from_torch(tensor, dynamic_axis=axis) for tensor, axis in inputs.items()]
    pypto_outputs = [pypto.from_torch(tensor, dynamic_axis=axis) for tensor, axis in outputs.items()]
    moe_with_shared_main(pypto_inputs, pypto_outputs)
    pypto.runtime._device_synchronize()

    # golden
    golden = ffn_golden_torch_with_shared(inputs_list[0], inputs_list[1], inputs_list[2], inputs_list[3])

    # Compare
    pypto_out = np.array(inputs_list[4].cpu().flatten().tolist())
    golden_out = np.array(golden.cpu().flatten().tolist())
    assert_allclose(pypto_out, golden_out, rtol=0.005, atol=0.005)


if __name__ == "__main__":
    main()