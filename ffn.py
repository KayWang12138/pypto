# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import pypto 
import torch 
import torch_npu 
from dataclasses import dataclass
torch_npu.npu.config.allow_internal_format = True


@dataclass
class FFNParams:
    hidden_size: int 
    intermediate_size: int
    cube_tile_shapes: list 
    vector_tile_shapes: list 
    dtype: str 
    
    
def sigmoid_pto(input_tensor):
    exp_tensor = pypto.exp(input_tensor * -1.0)
    out_tensor = pypto.div(pypto.full(exp_tensor.shape, 1.0, exp_tensor.dtype), exp_tensor + 1)
    return out_tensor 


def gelu_pto(input_tensor):
    sigmod_x = sigmoid_pto(input_tensor * 1.702)
    output_tensor = input_tensor * sigmod_x
    return output_tensor 


def sigmoid_torch(input_tensor):
    exp_tensor = torch.exp(input_tensor * -1.0)
    output_tensor = 1.0 / (1 + exp_tensor)
    return output_tensor 


def gelu_torch(input_tensor):
    sigmoid_x = sigmoid_torch(input_tensor * 1.702)
    output_tensor = input_tensor * sigmoid_x
    return output_tensor 


# @pypto.jit(
#     debug_options={"compile_debug_mode": 1}
# )
@pypto.jit
def ffn_pto_custom(hidden_states, up_weights, gate_weights, down_weights, ffn_out, params: FFNParams):
    assert len(hidden_states.shape) == 2
    dtype = hidden_states.dtype 
    t = hidden_states.shape[0]
    tile_bs = pypto.symbolic_scalar(32)
    t_loop = (t + tile_bs - 1) // tile_bs
    
    for t_idx in pypto.loop(0, t_loop, 1, name="LOOP_T", idx_name="t_idx"):
        input_view = hidden_states[t_idx*tile_bs: (t_idx+1)*tile_bs, :]
        pypto.set_cube_tile_shapes(*params.cube_tile_shapes)
        up_proj = pypto.matmul(input_view, up_weights, out_dtype=dtype)
        gate_proj = pypto.matmul(input_view, gate_weights, out_dtype=dtype)
        pypto.set_vec_tile_shapes(*params.vector_tile_shapes)
        gate_proj_act = gelu_pto(gate_proj)
        up_gate_proj = pypto.mul(up_proj, gate_proj_act)
        down_proj = pypto.matmul(up_gate_proj, down_weights, out_dtype=dtype)
        ffn_out[t_idx*tile_bs: (t_idx+1)*tile_bs, :] = down_proj


def ffn_pytorch(hidden_states, up_weights, gate_weights, down_weights, params: FFNParams):
    up_proj = torch.matmul(hidden_states, up_weights)
    gate_proj = torch.matmul(hidden_states, gate_weights)
    print(up_proj.shape, gate_proj.shape)
    gate_proj_act = gelu_torch(gate_proj)
    up_gate_proj = up_proj * gate_proj_act
    down_proj = torch.matmul(up_gate_proj, down_weights)
    return down_proj


def test_main(device_id):
    # torch_npu.npu.set_device(device_id)
    batch_size = 32 
    params = FFNParams(
        hidden_size=2048,
        intermediate_size=2048,
        cube_tile_shapes=[[16, 16], [32, 32], [32, 32]],
        vector_tile_shapes=[16, 32],
        dtype="bfloat16"
    )
    
    hidden_states = torch.rand(batch_size, params.hidden_size, dtype=torch.bfloat16, device="cpu")
    up_weights = torch.randn(params.hidden_size, params.intermediate_size, dtype=torch.bfloat16, device="cpu")
    gate_weights = torch.randn(params.hidden_size, params.intermediate_size, dtype=torch.bfloat16, device="cpu")
    down_weights = torch.rand(params.intermediate_size, params.hidden_size, dtype=torch.bfloat16, device="cpu")
    ffn_out = torch.empty(size=(batch_size, params.hidden_size), dtype=torch.bfloat16, device="cpu")

    hidden_states_pto = pypto.from_torch(hidden_states, dynamic_axis=[0], name="hidden_states")
    up_weights_pto = pypto.from_torch(up_weights, name="up_weights")
    gate_weights_pto = pypto.from_torch(gate_weights, name="gate_weights")
    down_weights_pto = pypto.from_torch(down_weights, name="down_weights")
    ffn_out_pto = pypto.from_torch(ffn_out, dynamic_axis=[0], name="ffn_out")
    
    output_torch = ffn_pytorch(hidden_states, up_weights, gate_weights, down_weights, params)
    # torch_npu.npu.synchronize()
    print("pytorch_output: ", output_torch[0])
    
    ffn_pto_custom(hidden_states_pto, up_weights_pto, gate_weights_pto, down_weights_pto, ffn_out_pto, params)
    # torch_npu.npu.synchronize()
    print("pytorch_output: ", ffn_out[0])
    
    torch.testing.assert_close(output_torch.cpu(), ffn_out.cpu(), atol=1e-3, rtol=1e-3)


if __name__ == "__main__":
    import sys 
    device_id = int(0)
    test_main(device_id)