#!/usr/bin/env python3
# coding: utf-8
# Copyright(c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""GMM MXFP8 SwiGLU quantization example built with PyPTO."""

import math
from dataclasses import dataclass

import torch
from numpy.testing import assert_allclose

import pypto
import torch_npu  # noqa: F401  # Required to initialize the NPU runtime.


def compute_golden_result(
    x,
    weight,
    scaled_x_golden,
    scaled_weight_golden,
    a_trans,
    b_trans,
):
    """Compute the reference result for grouped scaled matmul + SwiGLU + int8 quant."""
    if a_trans:
        x = torch.swapaxes(x, -1, -2)
        scaled_x_golden = torch.swapaxes(scaled_x_golden, -1, -2)
        if scaled_x_golden.ndim == 3:
            scaled_x_golden = scaled_x_golden.reshape(
                scaled_x_golden.shape[0] * scaled_x_golden.shape[1],
                scaled_x_golden.shape[2],
            )
        scaled_x_golden = torch.swapaxes(scaled_x_golden, -1, -2)
    else:
        if scaled_x_golden.ndim == 3:
            scaled_x_golden = scaled_x_golden.reshape(
                scaled_x_golden.shape[0],
                scaled_x_golden.shape[1] * scaled_x_golden.shape[2],
            )

    if b_trans:
        weight = torch.swapaxes(weight, -1, -2)
        if scaled_weight_golden.ndim == 3:
            scaled_weight_golden = scaled_weight_golden.reshape(
                scaled_weight_golden.shape[0],
                scaled_weight_golden.shape[1] * scaled_weight_golden.shape[2],
            )
        scaled_weight_golden = torch.swapaxes(scaled_weight_golden, -1, -2)
    else:
        scaled_weight_golden = torch.swapaxes(scaled_weight_golden, -1, -2)
        if scaled_weight_golden.ndim == 3:
            scaled_weight_golden = scaled_weight_golden.reshape(
                scaled_weight_golden.shape[0] * scaled_weight_golden.shape[1],
                scaled_weight_golden.shape[2],
            )

    k_dim = x.shape[-1]
    if math.ceil(k_dim / 32) % 2 != 0:
        scaled_x_golden = scaled_x_golden[:, :-1]
        scaled_weight_golden = scaled_weight_golden[:-1, :]

    scaled_x_golden_broadcast = torch.repeat_interleave(
        scaled_x_golden, repeats=32, dim=-1
    )
    scaled_weight_golden_broadcast = torch.repeat_interleave(
        scaled_weight_golden, repeats=32, dim=-2
    )
    x1_dims = x.ndim
    x2_dims = weight.ndim
    x1_pad_len = scaled_x_golden_broadcast.shape[-1] - x.shape[-1]
    x2_pad_len = scaled_weight_golden_broadcast.shape[-2] - weight.shape[-2]
    x1_pad = [0, x1_pad_len]
    for _ in range(x1_dims - 1):
        x1_pad += [0, 0]
    x1_golden = torch.nn.functional.pad(x, x1_pad, mode="constant", value=0)

    weight_pad = [0, 0]
    weight_pad += [0, x2_pad_len]
    for _ in range(x2_dims - 2):
        weight_pad += [0, 0]
    weight_golden = torch.nn.functional.pad(weight, weight_pad, mode="constant", value=0)

    x_fp32 = x.to(torch.float32)
    scaled_x_golden_broadcast_fp32 = scaled_x_golden_broadcast.to(torch.float32)
    x1_golden = x_fp32 * scaled_x_golden_broadcast_fp32

    weight_fp32 = weight.to(torch.float32)
    scaled_weight_golden_broadcast_fp32 = scaled_weight_golden_broadcast.to(torch.float32)
    weight_golden = weight_fp32 * scaled_weight_golden_broadcast_fp32

    gmm_out = torch.matmul(x1_golden, weight_golden)

    swiglu_out = swiglu(gmm_out)
    swiglu_out = swiglu_out.to(torch.bfloat16)
    swiglu_out = swiglu_out.to(torch.float32)

    quant_output, quant_scale_output = quant_pertoken(swiglu_out)

    return quant_output, quant_scale_output

def swiglu(gmm_out):
    """Apply SwiGLU activation function."""
    value, gate = gmm_out.chunk(2, dim=-1)
    silu_value = value * torch.sigmoid(value)
    return silu_value * gate


def quant_pertoken(swiglu_out):
    """Quantize each token independently to int8 and return the dequant scale."""
    input_abs = torch.abs(swiglu_out)
    input_max = torch.amax(input_abs, dim=-1, keepdim=True)
    scale_max = torch.tensor(127.0, dtype=torch.float32, device=swiglu_out.device)
    scale = input_max / scale_max
    input_scaled = swiglu_out / scale
    round_data = torch.round(input_scaled)
    round_data_clamped = torch.clamp(round_data, min=-127, max=127)
    round_data_int8 = round_data_clamped.to(dtype=torch.int8)
    scale_tensor = scale.squeeze(-1).to(dtype=torch.float32)

    return round_data_int8, scale_tensor


def gen_golden(a, b, scaled_a, scaled_b, group_list, a_trans, b_trans):
    """Generate grouped reference outputs for all experts."""
    num_groups = b.shape[0]
    gmmswigluquant_output = []
    gmmswigluquant_output_scale = []
    begin = 0
    end = 0

    for i in range(num_groups):
        if group_list[i] <= 0:
            continue
        begin = end
        end = end + group_list[i]
        if a_trans:
            x = a[:, begin:end]
        else:
            x = a[begin:end, :]
        weight = b[i]
        if a_trans:
            scaled_x_golden = scaled_a[:, begin:end, :]
        else:
            scaled_x_golden = scaled_a[begin:end, :, :]
        scaled_weight_golden = scaled_b[i]

        golden_temp, golden_temp_quant = compute_golden_result(
            x=x,
            weight=weight,
            scaled_x_golden=scaled_x_golden,
            scaled_weight_golden=scaled_weight_golden,
            a_trans=a_trans,
            b_trans=b_trans,
        )
        gmmswigluquant_output.append(golden_temp)
        gmmswigluquant_output_scale.append(golden_temp_quant)

    gmmswigluquant_output = torch.cat(gmmswigluquant_output, dim=0)
    gmmswigluquant_output_scale = torch.cat(gmmswigluquant_output_scale, dim=0)

    return gmmswigluquant_output, gmmswigluquant_output_scale

@dataclass
class ShapeConfig:
    """Configuration for matmul tile shapes and layout flags."""

    ori_shape: list
    tile_size: int
    m_tile_shape: list
    k_tile_shape: list
    n_tile_shape: list
    vector_tile_shape: list
    a_trans: bool = False
    b_trans: bool = False
    a_format_nz: bool = False
    b_format_nz: bool = False
    c_format_nz: bool = False


@pypto.jit(
    debug_options={"runtime_debug_mode": 1, "compile_debug_mode": 1},
    runtime_options={"device_sched_mode": 3},
)
def scaled_matmul_kernel(
    a: pypto.Tensor,
    b: pypto.Tensor,
    scaled_a: pypto.Tensor,
    scaled_b: pypto.Tensor,
    out: pypto.Tensor,
    out_quant: pypto.Tensor,
    group_list,
    tile_config,
) -> None:
    """Run grouped scaled matmul, then apply SwiGLU and per-token quantization."""
    num_groups = b.shape[0]
    n_size = b.shape[-1]
    begin = 0
    end = 0

    for i in range(num_groups):
        begin = end
        end = end + group_list[i]

        x = a[begin:end, :]
        weight = b[i]
        scaled_x = scaled_a[begin:end, :, :]
        pypto.set_vec_tile_shapes(
            tile_config.vector_tile_shape[0],
            tile_config.vector_tile_shape[1],
            tile_config.vector_tile_shape[2],
            tile_config.vector_tile_shape[3],
        )
        scaled_weight = scaled_b[i]

        pypto.set_cube_tile_shapes(
            tile_config.m_tile_shape,
            tile_config.k_tile_shape,
            tile_config.n_tile_shape,
            enable_multi_data_load=True,
            enable_split_k=True,
        )
        current_mm_out = pypto.scaled_mm(x, weight, pypto.DT_FP32, scaled_x, scaled_weight)
        # Split the matmul output into the SwiGLU value and gate tensors.
        value = current_mm_out[:, : n_size // 2]
        gate = current_mm_out[:, n_size // 2:]
        pypto.set_vec_tile_shapes(64, 256)
        silu_value = value * pypto.sigmoid(value)
        swiglu_out = pypto.mul(silu_value, gate)

        # Quantize each token with its own scale.
        x_bf16 = pypto.cast(swiglu_out, pypto.DT_BF16, pypto.CastMode.CAST_RINT)
        x_fp32 = pypto.cast(x_bf16, pypto.DT_FP32)
        x_abs = pypto.abs(x_fp32)
        x_max = pypto.amax(x_abs, -1, True)
        shape_0, shape_1 = x_max.shape[:2]
        x_scale = pypto.div(pypto.full([shape_0, shape_1], 127.0, pypto.DT_FP32), x_max)
        x_mul = pypto.mul(x_fp32, x_scale)

        x_mul_round = pypto.round(x_mul)
        x_fp16 = pypto.cast(x_mul_round, pypto.DT_FP16, pypto.CastMode.CAST_RINT)
        x_int8 = pypto.cast(x_fp16, pypto.DT_INT8)
        x_scale_quant = pypto.div(pypto.full([shape_0, shape_1], 1.0, pypto.DT_FP32), x_scale)

        pypto.assemble(x_int8, [begin, 0], out)
        pypto.assemble(x_scale_quant, [begin, 0], out_quant)


def gen_mxfp8(a, b, scaled_a, scaled_b, group_list, tile_config):
    """Launch the PyPTO kernel and return the quantized output plus scales."""
    a = a.npu()
    b = b.npu()
    scaled_a = scaled_a.npu()
    scaled_b = scaled_b.npu()

    out_shape = (a.shape[0], b.shape[-1] // 2)
    out = torch.zeros(out_shape, dtype=torch.int8).npu()

    out_quant_shape = (a.shape[0], 1)
    out_quant = torch.zeros(out_quant_shape, dtype=torch.float32).npu()

    input_tensors = {
        a: [],
        b: [],
        scaled_a: [],
        scaled_b: [],
    }
    output_tensors = {
        out: [],
        out_quant: [],
    }

    pto_inputs = [pypto.from_torch(tensor, dynamic_axis=axis) for tensor, axis in input_tensors.items()]
    pto_outputs = [pypto.from_torch(tensor, dynamic_axis=axis) for tensor, axis in output_tensors.items()]

    scaled_matmul_kernel(*pto_inputs, *pto_outputs, group_list, tile_config)
    out = out.to(torch.float32)
    out_quant = out_quant.squeeze(dim=1)

    return out, out_quant


def test_gmm_mxfp8(params):
    """Validate the PyPTO kernel against the PyTorch reference implementation."""
    m = params["m"]
    k = params["k"]
    n = params["n"]
    group_list = params["group_list"]
    a_trans = params["a_trans"]
    b_trans = params["b_trans"]
    tile_size = params["tile_size"]
    m_tile_shape = params["m_tile_shape"]
    k_tile_shape = params["k_tile_shape"]
    n_tile_shape = params["n_tile_shape"]
    vector_tile_shape = params["vector_tile_shape"]

    tile_config = ShapeConfig(
        [m, k, n],
        tile_size,
        m_tile_shape,
        k_tile_shape,
        n_tile_shape,
        vector_tile_shape,
        a_trans,
        b_trans,
        False,
        False,
        False,
    )

    a = torch.randn((m, k), dtype=torch.float32).uniform_(0, 1).to(torch.float8_e4m3fn)
    scaled_a = torch.randn((m, k // 64, 2), dtype=torch.float32).uniform_(0, 1).to(torch.float8_e8m0fnu)

    if b_trans:
        b = torch.randn((len(group_list), n, k), dtype=torch.float32).uniform_(0, 1).to(torch.float8_e4m3fn)
    else:
        b = torch.randn((len(group_list), k, n), dtype=torch.float32).uniform_(0, 1).to(torch.float8_e4m3fn)

    if b_trans:
        scaled_b = torch.randn((len(group_list), n, k // 64, 2), dtype=torch.float32).uniform_(0, 1).to(torch.float8_e8m0fnu)
    else:
        scaled_b = torch.randn((len(group_list), k // 64, n, 2), dtype=torch.float32).uniform_(0, 1).to(torch.float8_e8m0fnu)

    golden, golden_quant = gen_golden(a, b, scaled_a, scaled_b, group_list, a_trans, b_trans)

    result, result_quant = gen_mxfp8(a, b, scaled_a, scaled_b, group_list, tile_config)

    assert_allclose(golden.cpu().numpy(), result.cpu().numpy(), rtol=1e-3, atol=1)
    assert_allclose(golden_quant.cpu().numpy(), result_quant.cpu().numpy(), rtol=1e-4, atol=1e-4)
    print("success")


def get_params(case_name):
    """Get test parameters for specified test case."""
    if case_name == "testcase6":
        params = {
            "m": 16,
            "k": 512,
            "n": 7168,
            "group_list": [7, 9],
            "a_trans": False,
            "b_trans": False,
            "tile_size": 256,
            "m_tile_shape": [9, 9],
            "k_tile_shape": [256, 256],
            "n_tile_shape": [256, 256],
            "vector_tile_shape": [1, 8, 256, 32],
        }
    else:
        raise RuntimeError(f"Cannot get parameters for case: {case_name}")
    return params


if __name__ == "__main__":
    params = get_params("testcase6")
    test_gmm_mxfp8(params)
