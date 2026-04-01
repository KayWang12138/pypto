#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
GLM-4.5 FFN Shared Expert Quantization Module

This module implements the quantized FFN computation for shared experts in MoE architecture.
Shared experts are used across all tokens and tasks, learning general feature representations
while reducing the total parameter count through weight sharing.

Main Functions:
    - ffn_shared_expert_quant: Main function for shared expert FFN quantization
    - share_expert_moe_main: JIT compiled kernel for shared expert computation
    - expert_infer_base: Base inference function for shared expert computation
"""
import os
import torch
import torch_npu
import numpy as np
from numpy.testing import assert_allclose
from glm_ffn_common_interface import dequant_dynamic, swiglu, symmetric_quantization_per_token_fp8_e4m3
from torch._subclasses.fake_tensor import FakeTensor
from torch._dynamo import allow_in_graph
import pypto
from utils.get_format import get_format


def check_args(
    hidden_states,
    w13,
    w13_scale,
    w2,
    w2_scale
):
    assert hidden_states.dim() == 2
    assert hidden_states.shape[1] == 5120
    assert get_format(hidden_states) == 'ND'
    assert hidden_states.dtype == torch.bfloat16

    assert w13.dim() == 2
    assert w13.shape[0] == 5120
    assert w13.shape[1] == 384
    assert get_format(w13) == 'NZ'
    assert w13.dtype == torch.int8

    assert w13_scale.dim() == 1
    assert w13_scale.shape[0] == 384
    assert get_format(w13_scale) == 'ND'
    assert w13_scale.dtype == torch.bfloat16

    assert w2.dim() == 2
    assert w2.shape[0] == 192
    assert w2.shape[1] == 5120
    assert get_format(w2) == 'NZ'
    assert w2.dtype == torch.int8

    assert w2_scale.dim() == 1
    assert w2_scale.shape[0] == 5120
    assert get_format(w2_scale) == 'ND'
    assert w2_scale.dtype == torch.bfloat16


def main():
    test_ffn_share()


def ffn_golden_quan_per_token(x):
    # y_int8 : int8  scale_dequant : x.dtype
    x_dtype = x.dtype
    x_fp32 = x.to(torch.float32)
    max_value = x_fp32.abs().max(dim=1, keepdim=True)[0]
    scale_quant = 127.0 / max_value
    y_fp32 = x_fp32 * scale_quant
    y_rint = torch.round(y_fp32).to(torch.int32)
    y_round = torch.round(y_rint).to(torch.float16)
    y_int8 = torch.trunc(y_round).to(torch.int8)
    scale_dequant = (1 / scale_quant)
    return y_int8, scale_dequant


def ffn_golden_quan_per_channel(x):
    # y_int8 : int8  scale_dequant : x.dtype
    x_dtype = x.dtype
    x_fp32 = x.to(torch.float32)
    max_value = x_fp32.abs().max(dim=0, keepdim=True)[0]
    scale_quant = 127.0 / max_value
    y_fp32 = x_fp32 * scale_quant
    y_rint = torch.round(y_fp32).to(torch.int32)
    y_round = torch.round(y_rint).to(torch.float16)
    y_int8 = torch.trunc(y_round).to(torch.int8)
    scale_dequant = (1 / scale_quant)
    return y_int8, scale_dequant

def quant_fp8e4m3_per_token(x: torch.Tensor):
    # perblock
    x_fp32 = x.to(torch.float32)
    max_value = torch.amax(torch.abs(x_fp32), dim=1, keepdim=True)
    scale_quant = 448.0 / max_value
    y_fp32 = x_fp32 * scale_quant
    y_fp32 = y_fp32.view(x.shape)
    y_fp8e4m3 = y_fp32.to(torch.float8_e4m3fn)
    scale_dequant = 1.0 / scale_quant
    # (b, s, n, d) fp8e4m3, (b, s, n, 1) fp32
    return y_fp8e4m3, scale_dequant


def quant_fp8e4m3_per_channel(x: torch.Tensor):
    # perblock
    x_fp32 = x.to(torch.float32)
    max_value = torch.amax(torch.abs(x_fp32), dim=0, keepdim=True)
    scale_quant = 448.0 / max_value
    y_fp32 = x_fp32 * scale_quant
    y_fp32 = y_fp32.view(x.shape)
    y_fp8e4m3 = y_fp32.to(torch.float8_e4m3fn)
    scale_dequant = 1.0 / scale_quant
    # (b, s, n, d) fp8e4m3, (b, s, n, 1) fp32
    return y_fp8e4m3, scale_dequant


def torch_swiglu(up_proj):
    intermediate_size = up_proj.shape[1] // 2
    up_proj_left = up_proj[:, :intermediate_size]
    up_proj_right = up_proj[:, intermediate_size:]
    swiglu_mul = torch.mul(up_proj_left, -1.0)
    swiglu_exp = torch.exp(swiglu_mul)
    swiglu_add = torch.add(swiglu_exp, 1.0)
    swiglu_div = torch.div(up_proj_left, swiglu_add)
    swiglu_out = torch.mul(swiglu_div, up_proj_right)
    return swiglu_out


def moe_torch_npu(hidden_states, w13, w13_scale, w2, w2_scale):
    x_dtype = hidden_states.dtype
    quantized_x1, dynamic_scale = quant_fp8e4m3_per_token(hidden_states)
    print(f"quantized_x1 device = {quantized_x1.device}")
    print(f"w13 device = {w13.device}")
    quantized_x1_cpu = quantized_x1.cpu()
    w13_cpu = w13.cpu()
    mm1_out = torch.matmul(quantized_x1_cpu.to(torch.float32).to('npu'), w13_cpu.to(torch.float32).to('npu'))

    w13_scale = w13_scale.to(torch.float32)
    w13_scale = torch.unsqueeze(w13_scale, 0)
    mm1_out_fp32 = mm1_out * dynamic_scale
    mm1_out_fp32 = mm1_out_fp32 * w13_scale

    swiglu_out = torch_swiglu(mm1_out_fp32)

    quantized_x2, x_scale = quant_fp8e4m3_per_token(swiglu_out)
    mm1_out = torch.matmul(quantized_x2.to(torch.float), w2.to(torch.float))

    w2_scale = w2_scale.to(torch.float32)
    w2_scale = torch.unsqueeze(w2_scale, 0)

    output = mm1_out * x_scale
    output = output * w2_scale
    output = output.to(x_dtype)
    return output


def moe_torch_golden(hidden_states, w13, w13_scale, w2, w2_scale):
    x_dtype = hidden_states.dtype
    # quantized_x, dynamic_scale = torch_npu.npu_dynamic_quant(hidden_states)
    w13_scale = w13_scale.to(torch.float32)
    w2_scale = w2_scale.to(torch.float32)
    quantized_x, dynamic_scale = quant_fp8e4m3_per_token(hidden_states)
    dynamic_scale = dynamic_scale.reshape(-1)
    output_w13 = torch_npu.npu_quant_matmul(
            quantized_x,
            w13,
            w13_scale,
            pertoken_scale=dynamic_scale,
            bias=None,
            output_dtype=torch.float32,
        )
    swiglu_out = torch_npu.npu_swiglu(output_w13)
    print(f"swiglu dtype = {swiglu_out.dtype}")
    # quantized_x, x_scale = torch_npu.npu_dynamic_quant(swiglu_out)
    quantized_x, x_scale = quant_fp8e4m3_per_token(swiglu_out)
    x_scale = x_scale.reshape(-1)
    output = torch_npu.npu_quant_matmul(
            quantized_x,
            w2,
            w2_scale,
            pertoken_scale=x_scale,
            bias=None,
            output_dtype=x_dtype,
        )
    return output


def gen_input(
    b: int,
    s: int,
    hidden_size: int,
    intermediate_size: int,
    dtypes: torch.dtype,
    device_id: int
) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(42)
    hidden_states = torch.randn((b * s, hidden_size), dtype=dtypes, device=f'npu:{device_id}') * 0.01 * 2 - 0.01

    weight_gate_upper_tensor = torch.randn((hidden_size, intermediate_size * 2),
                                           dtype=dtypes, device=f'npu:{device_id}') * 0.01 * 2 - 0.01
    # w13, w13_scale = ffn_golden_quan_per_channel(weight_gate_upper_tensor)
    w13, w13_scale = quant_fp8e4m3_per_channel(weight_gate_upper_tensor)
    w13_scale = w13_scale.reshape(-1).to(dtypes)

    weight_down_proj_tensor = torch.randn((intermediate_size, hidden_size),
                                          dtype=dtypes, device=f'npu:{device_id}') * 0.01 * 2 - 0.01
    # w2, w2_scale = ffn_golden_quan_per_channel(weight_down_proj_tensor)
    w2, w2_scale = quant_fp8e4m3_per_channel(weight_down_proj_tensor)
    w2_scale = w2_scale.reshape(-1).to(dtypes)


    ffn_res = torch.empty((b * s, hidden_size), dtype=dtypes, device=f'npu:{device_id}')
    return hidden_states, w13, w13_scale, w2, w2_scale, ffn_res


def expert_infer_base(hidden_states, w13_params, w2_params, ffn_res, tiling_params, offset_params):
    """
    Base inference function for shared expert computation.

    This function performs FFN computation for shared expert:
    1. Per-token quantization: hidden_states_quant = Quantize(hidden_states)
    2. Quantized matrix multiplication: up_proj = MatMul(hidden_states_quant, w13)
    3. Dequantization: up_proj_dequant = Dequantize(up_proj, w13_scale, hidden_states_scale)
    4. SwiGLU activation: swiglu_out = SwiGLU(up_proj_dequant)
    5. Per-token quantization: down_proj_quant = Quantize(swiglu_out)
    6. Quantized matrix multiplication: down_proj = MatMul(down_proj_quant, w2)
    7. Dequantization: output = Dequantize(down_proj, w2_scale, down_proj_scale)

    Args:
        hidden_states: Input hidden states [num_tokens, hidden_size]
        w13_params: Tuple of (w13, w13_scale)
        w2_params: Tuple of (w2, w2_scale)
        ffn_res: Output tensor [num_tokens, hidden_size]
        tiling_params: Tuple of (vec_tile_shape, mm1_cube_tile_shape, mm2_cube_tile_shape)
        offset_params: Tuple of (share_loop_idx, loop_base)

    Note:
        This function processes tokens in tiles of size loop_base (typically 8)
        to support efficient computation on NPU.
    """
    # 入参信息获取
    w13, w13_scale = w13_params
    w2, w2_scale = w2_params
    unroll_offset, unroll_level = offset_params
    vec_tile_shape, mm1_cube_tile_shape, mm2_cube_tile_shape = tiling_params

    hidden_size = hidden_states.shape[1]
    intermediate_size = w2.shape[0]
    x_dtype = hidden_states.dtype

    # offset
    pypto.set_vec_tile_shapes(vec_tile_shape[0], vec_tile_shape[1])
    hidden_states_offset = [unroll_offset, 0]
    hidden_states_actual = pypto.view(hidden_states, [unroll_level, hidden_size], hidden_states_offset)

    # dynamic per_token_quant
    hidden_states_quant, hidden_states_scale = symmetric_quantization_per_token_fp8_e4m3(hidden_states_actual)

    # up_proj的matmul计算
    pypto.set_cube_tile_shapes([unroll_level, unroll_level],
                               [mm1_cube_tile_shape[1], mm1_cube_tile_shape[1]],
                               [mm1_cube_tile_shape[2], mm1_cube_tile_shape[2]], False)
    up_proj = pypto.matmul(hidden_states_quant, w13, pypto.DT_FP32)

    # dequant
    w13_scale_2d = pypto.unsqueeze(w13_scale, 0)
    pypto.set_vec_tile_shapes(4, intermediate_size * 2)
    up_proj_dequant = dequant_dynamic(up_proj, w13_scale_2d, hidden_states_scale)
    swiglu_out = swiglu(up_proj_dequant)

    # dynamic per_token_quant
    down_proj_quant, down_proj_scale = symmetric_quantization_per_token_fp8_e4m3(swiglu_out)

    # down_proj
    pypto.set_cube_tile_shapes([unroll_level, unroll_level],
                               [mm2_cube_tile_shape[1], mm2_cube_tile_shape[1]],
                               [mm2_cube_tile_shape[2], mm2_cube_tile_shape[2]], False)
    down_proj = pypto.matmul(down_proj_quant, w2, pypto.DT_FP32)

    # dequant
    w2_scale_2d = pypto.unsqueeze(w2_scale, 0)
    pypto.set_vec_tile_shapes(4, hidden_size)
    down_proj_dequant = dequant_dynamic(down_proj, w2_scale_2d, down_proj_scale)
    out = pypto.cast(down_proj_dequant, x_dtype)
    pypto.assemble(out, hidden_states_offset, ffn_res)


@pypto.frontend.jit(
    runtime_options={
        "device_sched_mode": 1,
        "stitch_cfgcache_size": 2700000
    },
    debug_options={
        "compile_debug_mode": 1,
    }
)
def share_expert_moe_main(
    hidden_states: pypto.tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    w13: pypto.tensor(),
    w13_scale: pypto.tensor(),
    w2: pypto.tensor(),
    w2_scale: pypto.tensor(),
    ffn_res: pypto.tensor([pypto.DYNAMIC, ...], pypto.DT_BF16)
):

    vec_tile_shape = (4, 5120)
    mm1_cube_tile_shape = (8, 256, 256)
    mm2_cube_tile_shape = (8, 192, 256)

    token_nums = hidden_states.shape[0]
    for share_loop_idx, loop_base in pypto.loop_unroll(
        token_nums,
        unroll_list=[1, 2, 4, 8, 16, 32, 64, 128],
        name="share_loop_idx"):
        expert_infer_base(
            hidden_states=hidden_states,
            w13_params=[w13, w13_scale],
            w2_params=[w2, w2_scale],
            ffn_res=ffn_res,
            tiling_params=[vec_tile_shape, mm1_cube_tile_shape, mm2_cube_tile_shape],
            offset_params=[share_loop_idx, loop_base]
    )


@allow_in_graph
def ffn_shared_expert_quant(
    hidden_states: torch.Tensor,
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    ffn_res: torch.Tensor
) -> None:
    """
    Quantized FFN computation for shared experts in MoE architecture.

    This function computes FFN output using quantized operations for shared experts.
    Shared experts are used across all tokens and tasks, learning general feature
    representations while reducing the total parameter count through weight sharing.

    Args:
        hidden_states: Input hidden states [num_tokens, hidden_size]
        w13: Gate and up projection weights (int8) [hidden_size, intermediate_size * 2]
        w13_scale: w13 weight scales [intermediate_size * 2]
        w2: Down projection weights (int8) [intermediate_size, hidden_size]
        w2_scale: w2 weight scales [hidden_size]
        ffn_res: Output tensor [num_tokens, hidden_size]

    Note:
        This function is decorated with @allow_in_graph to enable integration
        with PyTorch's compilation graph. The computation uses per-token quantization
        for better accuracy compared to per-channel quantization.
    """
    # if not isinstance(hidden_states, FakeTensor):
    #     check_args(hidden_states, w13, w13_scale, w2, w2_scale)

    inputs = [hidden_states, w13, w13_scale, w2, w2_scale, ffn_res]
    share_expert_moe_main(*inputs)


def test_ffn_share() -> None:
    x_dtype = torch.bfloat16
    # parameter config
    s = 1
    intermediate_size = 192
    hidden_size = 5120
    torch_npu.npu.config.allow_internal_format = True
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)

    # Test with different batch sizes
    for b in [2]:
    # for b in [1, 2]:
        # hidden_states, w13, w13_scale, w2, w2_scale, ffn_res
        hidden_states, w13, w13_scale, w2, w2_scale, ffn_res = \
            gen_input(b, s, hidden_size, intermediate_size, x_dtype, device_id)
        # w13 = torch_npu.npu_format_cast(w13, 29)
        # w2 = torch_npu.npu_format_cast(w2, 29)
        ffn_shared_expert_quant(hidden_states, w13, w13_scale, w2, w2_scale, ffn_res)

        # golden
        golden = moe_torch_golden(hidden_states, w13, w13_scale, w2, w2_scale)
        assert_allclose(np.array(ffn_res.cpu().flatten().tolist()), np.array(golden.cpu().flatten().tolist()),
                        rtol=0.0078125, atol=0.0001)


if __name__ == "__main__":
    main()
