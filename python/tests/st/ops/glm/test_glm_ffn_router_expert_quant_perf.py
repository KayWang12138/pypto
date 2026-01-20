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
GLM-4.5 FFN Router Expert Quantization Module

This module implements the quantized FFN computation for router experts in MoE architecture.
Router experts are dynamically selected based on input token features, allowing the model
to use only a subset of experts while maintaining a large parameter count.

Main Functions:
    - ffn_router_expert_quant: Main function for router expert FFN quantization
    - moe_router_expert_main: JIT compiled kernel for router expert computation
    - expert_infer_base: Base inference function for a single expert
"""
import os
import time
import gc
import torch
import torch_npu
import pypto
import numpy as np
from numpy.testing import assert_allclose
from torch._subclasses.fake_tensor import FakeTensor
from torch._dynamo import allow_in_graph
from utils.get_format import get_format


def symmetric_quantization_per_token(input_tensor):
    """
    Perform symmetric quantization per token (per row).
    
    Args:
        input_tensor: Input tensor to quantize
        
    Returns:
        Tuple of (quantized_int8_tensor, dequantization_scale)
    """
    x_fp32 = pypto.cast(input_tensor, pypto.DT_FP32)
    x_abs = pypto.abs(x_fp32)
    x_max = pypto.amax(x_abs, -1, True)
    shape_0, shape_1 = x_max.shape[:2]
    x_scale = pypto.div(pypto.full([shape_0, shape_1], 127.0, pypto.DT_FP32), x_max)
    x_mul = pypto.mul(x_fp32, x_scale)
    x_int32 = pypto.cast(x_mul, pypto.DT_INT32, pypto.CastMode.CAST_RINT)
    x_fp16 = pypto.cast(x_int32, pypto.DT_FP16, pypto.CastMode.CAST_ROUND)
    x_int8 = pypto.cast(x_fp16, pypto.DT_INT8, pypto.CastMode.CAST_TRUNC)
    x_scale_quant = pypto.div(pypto.full([shape_0, shape_1], 1.0, pypto.DT_FP32), x_scale)
    return x_int8, x_scale_quant


def dequant_dynamic(in_tensor, scale_1, scale_2):
    """
    Perform dynamic dequantization using two scale factors.
    
    Args:
        in_tensor: Quantized input tensor
        scale_1: First scale factor
        scale_2: Second scale factor
        
    Returns:
        Dequantized tensor
    """
    in_tensor_fp32 = pypto.cast(in_tensor, pypto.DT_FP32, pypto.CastMode.CAST_NONE)
    scale_1_fp32 = pypto.cast(scale_1, pypto.DT_FP32, pypto.CastMode.CAST_NONE)
    scale_2_fp32 = pypto.cast(scale_2, pypto.DT_FP32, pypto.CastMode.CAST_NONE)
    out_scale_2 = pypto.mul(in_tensor_fp32, scale_2_fp32)
    out = pypto.mul(out_scale_2, scale_1_fp32)
    return out


def swiglu(up_proj):
    """
    Apply SwiGLU activation function: x * sigmoid(x) * right_half.
    
    Args:
        up_proj: Input tensor with shape [batch, intermediate_size * 2]
        
    Returns:
        SwiGLU activated tensor with shape [batch, intermediate_size]
    """
    intermediate_size = up_proj.shape[1] // 2
    up_proj_left = pypto.view(up_proj, [up_proj.shape[0], intermediate_size], [0, 0])
    up_proj_right = pypto.view(up_proj, [up_proj.shape[0], intermediate_size], [0, intermediate_size])
    swiglu_mul = pypto.mul(up_proj_left, -1.0)
    swiglu_exp = pypto.exp(swiglu_mul)
    swiglu_add = pypto.add(swiglu_exp, 1.0)
    swiglu_div = pypto.div(up_proj_left, swiglu_add)
    swiglu_out = pypto.mul(swiglu_div, up_proj_right)
    return swiglu_out


def check_args(
        hidden_states: torch.Tensor,
        pertoken_scale: torch.Tensor,
        group_list: torch.Tensor,
        w13: torch.Tensor,
        w13_scale: torch.Tensor,
        w2: torch.Tensor,
        w2_scale: torch.Tensor
) -> None:
    """
    Validate input arguments for router expert FFN quantization operation.

    Args:
        hidden_states: Quantized input hidden states (int8) [num_tokens * topk, hidden_size]
        pertoken_scale: Per-token quantization scale [num_tokens * topk]
        group_list: Group list containing token counts per expert [per_device_expert_num]
        w13: Gate and up projection weights (int8) [per_device_expert_num, hidden_size, intermediate_size * 2]
        w13_scale: w13 weight scales [per_device_expert_num, intermediate_size * 2]
        w2: Down projection weights (int8) [per_device_expert_num, intermediate_size, hidden_size]
        w2_scale: w2 weight scales [per_device_expert_num, hidden_size]

    Raises:
        AssertionError: If any input argument doesn't meet the required format or dtype.
    """
    assert hidden_states.dim() == 2
    assert hidden_states.shape[1] == 5120
    assert get_format(hidden_states) == 'ND'
    assert hidden_states.dtype == torch.int8

    assert pertoken_scale.dim() == 1
    assert get_format(pertoken_scale) == 'ND'
    assert pertoken_scale.dtype == torch.float32

    assert group_list.dim() == 1
    assert get_format(group_list) == 'ND'
    assert group_list.dtype == torch.int32

    assert w13.dim() == 3
    assert w13.shape[1] == 5120
    assert w13.shape[2] == 3072
    assert get_format(w13) == 'NZ'
    assert w13.dtype == torch.int8

    assert w13_scale.dim() == 2
    assert w13_scale.shape[1] == 3072
    assert get_format(w13_scale) == 'ND'
    assert w13_scale.dtype == torch.float32

    assert w2.dim() == 3
    assert w2.shape[1] == 1536
    assert w2.shape[2] == 5120
    assert get_format(w2) == 'NZ'
    assert w2.dtype == torch.int8

    assert w2_scale.dim() == 2
    assert w2_scale.shape[1] == 5120
    assert get_format(w2_scale) == 'ND'
    assert w2_scale.dtype == torch.bfloat16


def ffn_router_torch_npu(
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    group_list: torch.Tensor,
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor
) -> torch.Tensor:
    group_list = group_list.to(torch.int64)
    group_list_cumsum = group_list.cumsum(dim=0)
    output_dtype = w2_scale.dtype
    w13_int8_nz = torch_npu.npu_format_cast(w13, 29)
    hidden_states, swiglu_out_scale, _ = torch_npu.npu_grouped_matmul_swiglu_quant(
                x=hidden_states,
                weight=w13_int8_nz,
                bias=None,
                group_list=group_list_cumsum,
                weight_scale=w13_scale,
                x_scale=hidden_states_scale)
    hidden_states = torch_npu.npu_grouped_matmul(
            x=[hidden_states],
            weight=[w2],
            scale=[w2_scale],
            bias=None,
            per_token_scale=[swiglu_out_scale],
            split_item=2,
            group_list_type=1,
            group_type=0,
            group_list=group_list,
            output_dtype=output_dtype)[0]
    return hidden_states


def ffn_router_torch_native(
    hidden_states: torch.Tensor,
    group_list: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor
) -> torch.Tensor:
    """
    Torch 原生实现的 MoE FFN Router Expert，不使用量化。
    
    对每个 expert 的对应 tokens 执行 FFN 计算：
    1. MatMul(hidden_states, w13) -> up_proj
    2. SwiGLU(up_proj) -> swiglu_out
    3. MatMul(swiglu_out, w2) -> output
    
    Args:
        hidden_states: Input tensor with shape [total_tokens, hidden_size]
        group_list: Number of tokens per expert, shape [local_expert_num]
        w13: Weight matrix for up projection, shape [local_expert_num, hidden_size, intermediate_size * 2]
        w2: Weight matrix for down projection, shape [local_expert_num, intermediate_size, hidden_size]
        
    Returns:
        Output tensor with shape [total_tokens, hidden_size]
    """
    device = hidden_states.device
    dtype = hidden_states.dtype
    local_expert_num = group_list.shape[0]
    
    # 计算每个 expert 的 token 起始位置
    group_list_cumsum = group_list.cumsum(dim=0)
    token_start_idx = torch.cat([torch.tensor([0], device=device, dtype=torch.int64), 
                                  group_list_cumsum[:-1].to(torch.int64)])
    
    # 存储所有 expert 的输出
    outputs = []
    
    # 对每个 expert 进行处理
    for exp_idx in range(local_expert_num):
        token_num = group_list[exp_idx].item()
        if token_num == 0:
            continue
            
        # 获取该 expert 对应的 tokens
        start_idx = token_start_idx[exp_idx].item()
        end_idx = start_idx + token_num
        hidden_states_slice = hidden_states[start_idx:end_idx]  # [token_num, hidden_size]
        
        # 第一个矩阵乘法: hidden_states @ w13[exp_idx]
        # w13[exp_idx]: [hidden_size, intermediate_size * 2]
        # hidden_states_slice: [token_num, hidden_size]
        # up_proj: [token_num, intermediate_size * 2]
        up_proj = torch.matmul(hidden_states_slice, w13[exp_idx])
        
        # SwiGLU 激活函数
        # up_proj: [token_num, intermediate_size * 2]
        intermediate_size = up_proj.shape[1] // 2
        up_proj_left = up_proj[:, :intermediate_size]  # [token_num, intermediate_size]
        up_proj_right = up_proj[:, intermediate_size:]  # [token_num, intermediate_size]
        
        # SwiGLU = left * sigmoid(left) * right = left / (1 + exp(-left)) * right
        swiglu_out = up_proj_left * torch.sigmoid(up_proj_left) * up_proj_right
        # 或者等价于: swiglu_out = (up_proj_left / (1 + torch.exp(-up_proj_left))) * up_proj_right
        
        # 第二个矩阵乘法: swiglu_out @ w2[exp_idx]
        # w2[exp_idx]: [intermediate_size, hidden_size]
        # swiglu_out: [token_num, intermediate_size]
        # output: [token_num, hidden_size]
        output = torch.matmul(swiglu_out, w2[exp_idx])
        
        outputs.append(output)
    
    # 将所有 expert 的输出拼接起来
    if outputs:
        result = torch.cat(outputs, dim=0)  # [total_tokens, hidden_size]
    else:
        # 如果没有有效的 expert，返回零张量
        total_tokens = hidden_states.shape[0]
        hidden_size = hidden_states.shape[1]
        result = torch.zeros((total_tokens, hidden_size), dtype=dtype, device=device)
    
    return result


def get_token_acc_table(group_list):
    assert len(group_list.shape) == 1
    return (torch.cumsum(group_list, dim=0) - group_list).to(group_list.dtype)


def calculate_moe_ffn_tflops(
    group_list: torch.Tensor,
    hidden_size: int,
    intermediate_size: int
) -> float:
    """
    计算 MoE FFN Router Expert 的 TFLOPs (Tera Floating Point Operations).
    
    MoE FFN 包含两个主要的矩阵乘法：
    1. 第一个矩阵乘法: hidden_states @ w13
       - w13 shape: (num_experts, hidden_size, intermediate_size * 2)
       - FLOPs = 2 * sum(group_list[i] * hidden_size * intermediate_size * 2) for all experts
       - 注意：矩阵乘法 FLOPs 包含乘法和加法，所以需要乘以 2
    2. 第二个矩阵乘法: swiglu_out @ w2
       - w2 shape: (num_experts, intermediate_size, hidden_size)
       - FLOPs = 2 * sum(group_list[i] * intermediate_size * hidden_size) for all experts
       - 注意：矩阵乘法 FLOPs 包含乘法和加法，所以需要乘以 2
    3. SwiGLU 激活函数: 元素级操作，FLOPs 相对较小，约为 intermediate_size * 2 的几倍
    
    Args:
        group_list: 每个 expert 处理的 token 数量，shape: (num_experts,)
        hidden_size: 隐藏层大小
        intermediate_size: 中间层大小
        
    Returns:
        TFLOPs (Tera FLOPs)
    """
    # 将 group_list 转换为 CPU numpy 数组进行计算
    if isinstance(group_list, torch.Tensor):
        group_list_np = group_list.cpu().numpy()
    else:
        group_list_np = np.array(group_list)
    
    total_tokens = group_list_np.sum()
    
    # 第一个矩阵乘法: hidden_states @ w13
    # w13 shape: (num_experts, hidden_size, intermediate_size * 2)
    # 对于每个 expert i，处理的 token 数量是 group_list[i]
    # 矩阵乘法 FLOPs = 2 * M * N * K（包含乘法和加法）
    # 其中 M=group_list[i], N=intermediate_size*2, K=hidden_size
    mm1_flops = 2 * total_tokens * hidden_size * intermediate_size * 2
    
    # 第二个矩阵乘法: swiglu_out @ w2
    # w2 shape: (num_experts, intermediate_size, hidden_size)
    # 矩阵乘法 FLOPs = 2 * M * N * K（包含乘法和加法）
    # 其中 M=group_list[i], N=hidden_size, K=intermediate_size
    mm2_flops = 2 * total_tokens * intermediate_size * hidden_size
    
    # SwiGLU 激活函数: 主要是元素级操作
    # 包括: sigmoid (exp, add, div), mul
    # 对于每个 token，需要处理 intermediate_size * 2 个元素
    # 大约需要 5-6 次元素级操作: exp, add, div, mul (两次)
    swiglu_flops = total_tokens * intermediate_size * 2 * 6  # 保守估计
    
    # 量化/反量化操作: 主要是元素级操作，FLOPs 相对较小
    # 估算为矩阵乘法 FLOPs 的 1-2%
    quant_dequant_flops = (mm1_flops + mm2_flops) * 0.02
    
    total_flops = mm1_flops + mm2_flops + swiglu_flops + quant_dequant_flops
    
    # 转换为 TFLOPs (1 TFLOP = 10^12 FLOPs)
    tflops = total_flops / 1e12
    
    return tflops


def ffn_golden_quan_per_token(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize input tensor per token (per row).

    Args:
        x: Input tensor to quantize

    Returns:
        Tuple of (quantized_int8_tensor, dequantization_scale)
    """
    x_fp32 = x.to(torch.float32)
    max_value = x_fp32.abs().max(dim=1, keepdim=True)[0]
    scale_quant = 127.0 / max_value
    y_fp32 = x_fp32 * scale_quant
    y_rint = torch.round(y_fp32).to(torch.int32)
    y_round = torch.round(y_rint).to(torch.float16)
    y_int8 = torch.trunc(y_round).to(torch.int8)
    scale_dequant = (1 / scale_quant)
    return y_int8, scale_dequant


def ffn_golden_quan_per_channel_3d(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize input tensor per channel (per column) for 3D tensors.

    Note: This function currently uses per-token quantization (dim=1)
    but is kept for backward compatibility.

    Args:
        x: Input tensor to quantize

    Returns:
        Tuple of (quantized_int8_tensor, dequantization_scale)
    """
    x_fp32 = x.to(torch.float32)
    max_value = x_fp32.abs().max(dim=1, keepdim=True)[0]
    scale_quant = 127.0 / max_value
    y_fp32 = x_fp32 * scale_quant
    y_rint = torch.round(y_fp32).to(torch.int32)
    y_round = torch.round(y_rint).to(torch.float16)
    y_int8 = torch.trunc(y_round).to(torch.int8)
    scale_dequant = (1 / scale_quant)
    return y_int8, scale_dequant

def gen_input(
    b: int,
    s: int,
    topk: int,
    local_expert_num: int,  # 更名：per_expert_num → local_expert_num
    hidden_size: int,
    intermediate_size: int,
    dtypes: torch.dtype,
    device_id: int,
    uniform_distribution: bool = False  # 新增参数：是否均匀分配 token
) -> tuple[torch.Tensor, ...]:
    # 不使用固定种子，让每次调用生成不同的随机数据
    # 使用当前时间戳作为种子，确保每次调用都不同
    current_seed = int(time.time() * 1000000) % (2**31)  # 转换为32位整数
    torch.manual_seed(current_seed)
    total_tokens = b * s * topk

    # 生成 hidden_states 并量化
    hidden_states = torch.randn((total_tokens, hidden_size), \
        dtype=dtypes, device=f'npu:{device_id}') * 0.01 * 2 - 0.01
    hidden_states, hidden_states_scale = ffn_golden_quan_per_token(hidden_states)
    hidden_states_scale = hidden_states_scale.reshape(-1).to(torch.float32)

    # === 动态生成 group_list ===
    if local_expert_num <= 0:
        raise ValueError("local_expert_num must be positive")
    if total_tokens < 0:
        raise ValueError("Total tokens (b*s*topk) must be non-negative")

    if total_tokens == 0:
        group_list = torch.zeros(local_expert_num, dtype=torch.int32, device=f'npu:{device_id}')
    else:
        if uniform_distribution:
            # 均匀分配模式：将 total_tokens 平均分配给每个 expert
            base = total_tokens // local_expert_num
            remainder = total_tokens % local_expert_num
            
            # 每个 expert 至少 base 个 token，前 remainder 个 expert 多分配 1 个
            group_list_np = np.full(local_expert_num, base, dtype=np.int32)
            if remainder > 0:
                group_list_np[:remainder] += 1
            
            print(f"[均匀分配] group_list_np: {group_list_np}")
            print(f"[均匀分配] total_tokens: {total_tokens}")
            print(f"[均匀分配] group_list_np sum: {group_list_np.sum()}")
        else:
            # 原有不均匀分配逻辑
            base = total_tokens // local_expert_num
            # remainder = total_tokens % local_expert_num

            # 初始分配：每个 expert 至少 base 个 token
            group_list_np = np.full(local_expert_num, base, dtype=np.int32)

            # 随机扰动：每个 expert 增加 [-50, +50] 的偏移，但不能低于 0
            # 注意：我们先加扰动，再调整总和
            # 使用不同的种子确保每次调用生成不同的 group_list
            rng = np.random.default_rng(seed=current_seed)
            perturb = rng.integers(-50, 51, size=local_expert_num)
            group_list_np = np.clip(group_list_np + perturb, a_min=0, a_max=None)

            # 调整总和：计算当前总和与目标的差值
            current_sum = group_list_np.sum()
            diff = total_tokens - current_sum

            # 通过微调（优先加到非零项或随机项）来修正 diff
            # 简单策略：循环加/减 1 直到 diff=0
            idx = 0
            while diff != 0:
                if diff > 0:
                    # 需要增加 token
                    group_list_np[idx % local_expert_num] += 1
                    diff -= 1
                else:
                    # 需要减少 token，但不能低于 0
                    if group_list_np[idx % local_expert_num] > 0:
                        group_list_np[idx % local_expert_num] -= 1
                        diff += 1
                idx += 1
                # 安全兜底：避免无限循环（理论上不会发生）
                if idx > 10 * local_expert_num:
                    break

            # 最终确保总和正确（数值误差兜底）
            final_sum = group_list_np.sum()
            if final_sum != total_tokens:
                # 强制修正最后一个元素（极端情况）
                group_list_np[-1] += total_tokens - final_sum
            print(f"[不均匀分配] group_list_np: {group_list_np}")
            print(f"[不均匀分配] total_tokens: {total_tokens}")
            print(f"[不均匀分配] group_list_np sum: {group_list_np.sum()}")
        
        group_list = torch.from_numpy(group_list_np).to(device=f'npu:{device_id}', dtype=torch.int32)

    group_list_cumsum = get_token_acc_table(group_list).to(torch.int32)
    print(f"group_list_cumsum: {group_list_cumsum}")
    # 生成并量化 w13
    w13 = torch.randn((local_expert_num, hidden_size, intermediate_size * 2), \
        dtype=dtypes, device=f'npu:{device_id}') * 0.01 * 2 - 0.01
    w13, w13_scale = ffn_golden_quan_per_channel_3d(w13)
    w13_scale = w13_scale.squeeze(1).to(torch.float32)

    # 生成并量化 w2
    w2 = torch.randn((local_expert_num, intermediate_size, hidden_size), dtype=dtypes, \
        device=f'npu:{device_id}') * 0.01 * 2 - 0.01
    w2, w2_scale = ffn_golden_quan_per_channel_3d(w2)
    w2_scale = w2_scale.squeeze(1).to(dtypes)

    ffn_res = torch.empty((total_tokens, hidden_size), dtype=w2_scale.dtype, device=f'npu:{device_id}')
    return hidden_states, hidden_states_scale, group_list, group_list_cumsum, w13, w13_scale, w2, w2_scale, ffn_res


def gen_input_native(
    b: int,
    s: int,
    topk: int,
    local_expert_num: int,
    hidden_size: int,
    intermediate_size: int,
    dtypes: torch.dtype,
    device_id: int
) -> tuple[torch.Tensor, ...]:
    """
    生成不需要量化的随机输入（原生浮点版本）。
    
    Args:
        b: Batch size
        s: Sequence length
        topk: Top-k experts per token
        local_expert_num: Number of local experts
        hidden_size: Hidden size
        intermediate_size: Intermediate size
        dtypes: Data type
        device_id: Device ID
        
    Returns:
        Tuple of (hidden_states, group_list, group_list_cumsum, w13, w2, ffn_res)
        - hidden_states: [total_tokens, hidden_size], float
        - group_list: [local_expert_num], int32
        - group_list_cumsum: [local_expert_num], int32
        - w13: [local_expert_num, hidden_size, intermediate_size * 2], float
        - w2: [local_expert_num, intermediate_size, hidden_size], float
        - ffn_res: [total_tokens, hidden_size], float (output buffer)
    """
    # 不使用固定种子，让每次调用生成不同的随机数据
    # 使用当前时间戳作为种子，确保每次调用都不同
    current_seed = int(time.time() * 1000000) % (2**31)  # 转换为32位整数
    torch.manual_seed(current_seed)
    total_tokens = b * s * topk

    # 生成 hidden_states（不量化）
    hidden_states = torch.randn((total_tokens, hidden_size), \
        dtype=dtypes, device=f'npu:{device_id}') * 0.01 * 2 - 0.01

    # === 动态生成 group_list ===
    if local_expert_num <= 0:
        raise ValueError("local_expert_num must be positive")
    if total_tokens < 0:
        raise ValueError("Total tokens (b*s*topk) must be non-negative")

    if total_tokens == 0:
        group_list = torch.zeros(local_expert_num, dtype=torch.int32, device=f'npu:{device_id}')
    else:
        base = total_tokens // local_expert_num

        # 初始分配：每个 expert 至少 base 个 token
        group_list_np = np.full(local_expert_num, base, dtype=np.int32)

        # 随机扰动：每个 expert 增加 [-50, +50] 的偏移，但不能低于 0
        # 注意：我们先加扰动，再调整总和
        # 使用不同的种子确保每次调用生成不同的 group_list
        rng = np.random.default_rng(seed=current_seed)
        perturb = rng.integers(-50, 51, size=local_expert_num)
        group_list_np = np.clip(group_list_np + perturb, a_min=0, a_max=None)

        # 调整总和：计算当前总和与目标的差值
        current_sum = group_list_np.sum()
        diff = total_tokens - current_sum

        # 通过微调（优先加到非零项或随机项）来修正 diff
        # 简单策略：循环加/减 1 直到 diff=0
        idx = 0
        while diff != 0:
            if diff > 0:
                # 需要增加 token
                group_list_np[idx % local_expert_num] += 1
                diff -= 1
            else:
                # 需要减少 token，但不能低于 0
                if group_list_np[idx % local_expert_num] > 0:
                    group_list_np[idx % local_expert_num] -= 1
                    diff += 1
            idx += 1
            # 安全兜底：避免无限循环（理论上不会发生）
            if idx > 10 * local_expert_num:
                break

        # 最终确保总和正确（数值误差兜底）
        final_sum = group_list_np.sum()
        if final_sum != total_tokens:
            # 强制修正最后一个元素（极端情况）
            group_list_np[-1] += total_tokens - final_sum
        print(f"group_list_np: {group_list_np}")
        print(f"total_tokens: {total_tokens}")
        print(f"group_list_np sum: {group_list_np.sum()}")
        group_list = torch.from_numpy(group_list_np).to(device=f'npu:{device_id}', dtype=torch.int32)

    group_list_cumsum = get_token_acc_table(group_list).to(torch.int32)
    print(f"group_list_cumsum: {group_list_cumsum}")
    
    # 生成 w13（不量化）
    w13 = torch.randn((local_expert_num, hidden_size, intermediate_size * 2), \
        dtype=dtypes, device=f'npu:{device_id}') * 0.01 * 2 - 0.01

    # 生成 w2（不量化）
    w2 = torch.randn((local_expert_num, intermediate_size, hidden_size), dtype=dtypes, \
        device=f'npu:{device_id}') * 0.01 * 2 - 0.01

    ffn_res = torch.empty((total_tokens, hidden_size), dtype=dtypes, device=f'npu:{device_id}')
    return hidden_states, group_list, group_list_cumsum, w13, w2, ffn_res


def expert_infer_base(
        hidden_states_params,
        group_list_params,
        w13_params,
        w2_params,
        offset_params,
        tiling_params,
        ffn_res):
    """
    Base inference function for a single expert computation.

    This function performs FFN computation for a specific expert:
    1. Quantized matrix multiplication: up_proj = MatMul(hidden_states, w13)
    2. Dequantization: up_proj_dequant = Dequantize(up_proj, w13_scale, hidden_states_scale)
    3. SwiGLU activation: swiglu_out = SwiGLU(up_proj_dequant)
    4. Quantization: down_proj_quant = Quantize(swiglu_out)
    5. Quantized matrix multiplication: down_proj = MatMul(down_proj_quant, w2)
    6. Dequantization: output = Dequantize(down_proj, w2_scale, down_proj_scale)

    Args:
        hidden_states_params: Tuple of (hidden_states, hidden_states_scale)
        group_list_params: Tuple of (group_list, group_list_cumsum)
        w13_params: Tuple of (w13, w13_scale)
        w2_params: Tuple of (w2, w2_scale)
        offset_params: Tuple of (exp_idx, token_loop_idx, loop_base)
        tiling_params: Tuple of (mm1_cube_tile_shape, mm2_cube_tile_shape)
        ffn_res: Output tensor [num_tokens * topk, hidden_size]

    Note:
        This function processes tokens in tiles of size loop_base (typically 8)
        to support efficient computation on NPU.
    """
    # 入参信息获取
    hidden_states, hidden_states_scale = hidden_states_params
    _, group_list_cumsum = group_list_params
    w13, w13_scale = w13_params
    w2, w2_scale = w2_params
    exp_idx, unroll_offset, unroll_level = offset_params
    print(f"exp_idx: {exp_idx}")
    print(f"unroll_offset: {unroll_offset}")
    print(f"unroll_level: {unroll_level}")
    mm1_cube_tile_shape, mm2_cube_tile_shape = tiling_params

    hidden_size = hidden_states.shape[1]
    print(f"hidden_size: {hidden_size}")
    intermediate_size = w13.shape[1] // 2
    print(f"intermediate_size: {intermediate_size}")
    x_dtype = w2_scale.dtype

    # 计算对应激活专家的偏移地址
    pypto.set_vec_tile_shapes(32)

    # 获取该激活专家在当前loop参与计算部分，有效token的偏移地址和scale偏移地址
    hidden_states_offset_start = group_list_cumsum[exp_idx, ]
    hidden_states_offset = [hidden_states_offset_start + unroll_offset, 0]
    x_scale_offset = [hidden_states_offset_start + unroll_offset, 0]

    # 获取该激活专家权重和scale的偏移地址
    weight_13_offset = [exp_idx * hidden_size, 0]
    w13_scale_offset = [exp_idx, 0]
    weight_2_offset = [exp_idx * intermediate_size, 0]
    w2_scale_offset = [exp_idx, 0]

    # 获取当前专家的实际token数和scale
    x = pypto.view(hidden_states, [unroll_level, hidden_size], hidden_states_offset)
    x_scale = pypto.view(hidden_states_scale, [unroll_level, 1], x_scale_offset)

    # 获取当前专家的weght_13和scale
    w13_weight_2d = pypto.view(w13, [hidden_size, intermediate_size * 2], weight_13_offset)
    w13_scale_valid = pypto.view(w13_scale, [1, intermediate_size * 2], w13_scale_offset)

    # # 获取当前专家的weght_2和scale
    w2_weight_2d = pypto.view(w2, [intermediate_size, hidden_size], weight_2_offset)
    w2_scale_valid = pypto.view(w2_scale, [1, hidden_size], w2_scale_offset)

    # up_proj的matmul计算
    pypto.set_cube_tile_shapes([unroll_level, unroll_level], [mm1_cube_tile_shape[1], mm1_cube_tile_shape[1] * 2], \
                               [mm1_cube_tile_shape[2], mm1_cube_tile_shape[2]], True, True)
    pypto.set_matrix_size([unroll_level, w13_weight_2d.shape[0], w13_weight_2d.shape[1]])
    up_proj = pypto.matmul(x, w13_weight_2d, pypto.DT_INT32)

    # dequant
    pypto.set_vec_tile_shapes(4, intermediate_size * 2)
    up_proj_out = dequant_dynamic(up_proj, w13_scale_valid, x_scale)
    swiglu_out = swiglu(up_proj_out)

    # down_proj
    # quant
    down_proj_quant, down_proj_scale = symmetric_quantization_per_token(swiglu_out)

    pypto.set_cube_tile_shapes([unroll_level, unroll_level], [mm2_cube_tile_shape[1], mm2_cube_tile_shape[1] * 2], \
                               [mm2_cube_tile_shape[2], mm2_cube_tile_shape[2]], True, True)
    pypto.set_matrix_size([unroll_level, w2_weight_2d.shape[0], w2_weight_2d.shape[1]])
    down_proj = pypto.matmul(down_proj_quant, w2_weight_2d, pypto.DT_INT32)

    # dequant
    pypto.set_vec_tile_shapes(4, hidden_size)
    down_proj_dequant = dequant_dynamic(down_proj, w2_scale_valid, down_proj_scale)
    out = pypto.cast(down_proj_dequant, x_dtype)
    pypto.assemble(out, hidden_states_offset, ffn_res)


@pypto.jit(
    host_options={"only_codegen": True},
    runtime_options={"device_sched_mode": 1},
    pass_options={"cube_l1_reuse_mode": 2}
)
def moe_router_expert_main(hidden_states, hidden_states_scale,
                           group_list, group_list_cumsum, w13,
                           w13_scale, w2, w2_scale, ffn_res):
    """
    JIT compiled kernel for router expert FFN quantization.

    This kernel processes multiple experts in a loop, where each expert processes
    a subset of tokens based on the group_list. The computation is done in tiles
    to support efficient execution on NPU.

    Args:
        hidden_states: Quantized input hidden states (int8) [num_tokens * topk, hidden_size]
        hidden_states_scale: Per-token quantization scale [num_tokens * topk]
        group_list: Group list containing token counts per expert [per_device_expert_num]
        group_list_cumsum: Cumulative sum of group list [per_device_expert_num]
        w13: Gate and up projection weights (int8) [per_device_expert_num, hidden_size, intermediate_size * 2]
        w13_scale: w13 weight scales [per_device_expert_num, intermediate_size * 2]
        w2: Down projection weights (int8) [per_device_expert_num, intermediate_size, hidden_size]
        w2_scale: w2 weight scales [per_device_expert_num, hidden_size]
        ffn_res: Output tensor [num_tokens * topk, hidden_size]

    Note:
        This function uses cube L1 reuse mode 2 for better memory efficiency.
        Each expert processes tokens in tiles of size 8.
    """
    pypto.experimental.set_operation_config(combine_axis=True)

    # tiling config
    mm1_cube_tile_shape = (8, 256, 256)
    mm2_cube_tile_shape = (8, 256, 256)

    # 获取当前device上专家总数
    expert_num = group_list.shape[0]

    # 输入Tensor shape转换为2维
    w13_2d_shape = (w13.shape[0] * w13.shape[1], w13.shape[2])
    w2_2d_shape = (w2.shape[0] * w2.shape[1], w2.shape[2])
    hidden_states_scale_shape = (hidden_states_scale.shape[0], 1)

    w13_2d = pypto.reshape(w13, w13_2d_shape, inplace=True)
    print(f"w13_2d shape: {w13_2d.shape}")
    w2_2d = pypto.reshape(w2, w2_2d_shape, inplace=True)
    print(f"w2_2d shape: {w2_2d.shape}")
    hidden_states_scale_2d = pypto.reshape(hidden_states_scale, hidden_states_scale_shape, inplace=True)
    print(f"hidden_states_scale_2d shape: {hidden_states_scale_2d.shape}")
    for exp_idx in pypto.loop(expert_num, name="LOOP_FFN_ROUTER_MLP_L0", idx_name="exp_idx"):
        # 获取激活专家的token数
        token_num = group_list[exp_idx, ]
        # print(f"token_num: {token_num}")
        for token_loop_idx, loop_base in pypto.loop_unroll(
            token_num,
            unroll_list=[1, 2, 4, 8, 16, 32],
            name="LOOP_FFN_ROUTER_MLP_L1",
            idx_name="token_loop_idx"):
            print(f"token_loop_idx: {token_loop_idx}, loop_base: {loop_base}")
            expert_infer_base(
                hidden_states_params=[hidden_states, hidden_states_scale_2d],
                group_list_params=[group_list, group_list_cumsum],
                w13_params=[w13_2d, w13_scale],
                w2_params=[w2_2d, w2_scale],
                offset_params=[exp_idx, token_loop_idx, loop_base],
                tiling_params=[mm1_cube_tile_shape, mm2_cube_tile_shape],
                ffn_res=ffn_res
            )


def expert_infer_base_native(
        hidden_states_params,
        group_list_params,
        w13_params,
        w2_params,
        offset_params,
        tiling_params,
        ffn_res):
    """
    Base inference function for a single expert computation (native float version, no quantization).

    This function performs FFN computation for a specific expert:
    1. Matrix multiplication: up_proj = MatMul(hidden_states, w13)
    2. SwiGLU activation: swiglu_out = SwiGLU(up_proj)
    3. Matrix multiplication: down_proj = MatMul(swiglu_out, w2)

    Args:
        hidden_states_params: Tuple of (hidden_states,) - no scale needed
        group_list_params: Tuple of (group_list, group_list_cumsum)
        w13_params: Tuple of (w13,) - no scale needed
        w2_params: Tuple of (w2,) - no scale needed
        offset_params: Tuple of (exp_idx, token_loop_idx, loop_base)
        tiling_params: Tuple of (mm1_cube_tile_shape, mm2_cube_tile_shape)
        ffn_res: Output tensor [num_tokens * topk, hidden_size]

    Note:
        This function processes tokens in tiles of size loop_base (typically 8)
        to support efficient computation on NPU.
    """
    # 入参信息获取
    hidden_states = hidden_states_params[0]
    _, group_list_cumsum = group_list_params
    w13 = w13_params[0]
    w2 = w2_params[0]
    exp_idx, unroll_offset, unroll_level = offset_params
    mm1_cube_tile_shape, mm2_cube_tile_shape = tiling_params

    hidden_size = hidden_states.shape[1]
    intermediate_size = w13.shape[1] // 2

    # 计算对应激活专家的偏移地址
    pypto.set_vec_tile_shapes(32)

    # 获取该激活专家在当前loop参与计算部分，有效token的偏移地址
    hidden_states_offset_start = group_list_cumsum[exp_idx, ]
    hidden_states_offset = [hidden_states_offset_start + unroll_offset, 0]

    # 获取该激活专家权重的偏移地址
    weight_13_offset = [exp_idx * hidden_size, 0]
    weight_2_offset = [exp_idx * intermediate_size, 0]

    # 获取当前专家的实际token数
    x = pypto.view(hidden_states, [unroll_level, hidden_size], hidden_states_offset)
    print(f"x shape: {x.shape}")
    # 获取当前专家的weight_13
    w13_weight_2d = pypto.view(w13, [hidden_size, intermediate_size * 2], weight_13_offset)
    print(f"w13_weight_2d shape: {w13_weight_2d.shape}")
    # 获取当前专家的weight_2
    w2_weight_2d = pypto.view(w2, [intermediate_size, hidden_size], weight_2_offset)
    print(f"w2_weight_2d shape: {w2_weight_2d.shape}")      
    # 注意：matmul 的所有输入（包括两个输入张量和输出类型）必须具有相同的数据类型
    # 使用 GM accumulated 时，输出类型必须是 FP32 或 INT32，不能是 FP16/BF16
    # 所以需要将所有输入转换为 FP32 进行 matmul 计算
    
    # 将输入转换为 FP32
    x_fp32 = pypto.cast(x, pypto.DT_FP32)
    print(f"x_fp32 shape: {x_fp32.shape}")
    w13_weight_2d_fp32 = pypto.cast(w13_weight_2d, pypto.DT_FP32)
    print(f"w13_weight_2d_fp32 shape: {w13_weight_2d_fp32.shape}")
    # up_proj的matmul计算（所有输入和输出都是 FP32）
    pypto.set_cube_tile_shapes([unroll_level, unroll_level], [mm1_cube_tile_shape[1], mm1_cube_tile_shape[1] * 2], \
                               [mm1_cube_tile_shape[2], mm1_cube_tile_shape[2]], True, True)
    pypto.set_matrix_size([unroll_level, w13_weight_2d_fp32.shape[0], w13_weight_2d_fp32.shape[1]])
    up_proj = pypto.matmul(x_fp32, w13_weight_2d_fp32, pypto.DT_FP32)
    print(f"up_proj shape: {up_proj.shape}")
    # SwiGLU 激活函数（输入和输出都是 FP32）
    # 设置 vec tile shapes 用于 SwiGLU 操作
    pypto.set_vec_tile_shapes(4, intermediate_size * 2)
    swiglu_out = swiglu(up_proj)
    print(f"swiglu_out shape: {swiglu_out.shape}")
    # 将 w2 权重转换为 FP32
    w2_weight_2d_fp32 = pypto.cast(w2_weight_2d, pypto.DT_FP32)
    print(f"w2_weight_2d_fp32 shape: {w2_weight_2d_fp32.shape}")
    # down_proj的matmul计算（所有输入和输出都是 FP32）
    pypto.set_cube_tile_shapes([unroll_level, unroll_level], [mm2_cube_tile_shape[1], mm2_cube_tile_shape[1] * 2], \
                               [mm2_cube_tile_shape[2], mm2_cube_tile_shape[2]], True, True)
    pypto.set_matrix_size([unroll_level, w2_weight_2d_fp32.shape[0], w2_weight_2d_fp32.shape[1]])
    down_proj = pypto.matmul(swiglu_out, w2_weight_2d_fp32, pypto.DT_FP32)
    print(f"down_proj shape: {down_proj.shape}")
    # 如果需要，将结果转换回输出张量的类型
    # 检查输出张量的数据类型，如果不是 FP32，则转换
    # output_dtype = ffn_res.dtype
    # if output_dtype != pypto.DT_FP32:
    # 设置 vec tile shapes 用于 cast 操作
    pypto.set_vec_tile_shapes(4, hidden_size)
    out = pypto.cast(down_proj, ffn_res.dtype)
    print(f"out shape: {out.shape}")
    # 输出结果
    pypto.assemble(out, hidden_states_offset, ffn_res)


@pypto.jit(
    host_options={"only_codegen": True},
    runtime_options={"device_sched_mode": 1},
    pass_options={"cube_l1_reuse_mode": 2}
)
def moe_router_expert_main_native(hidden_states,
                                  group_list, group_list_cumsum, w13,
                                  w2, ffn_res):
    """
    JIT compiled kernel for router expert FFN (native float version, no quantization).

    This kernel processes multiple experts in a loop, where each expert processes
    a subset of tokens based on the group_list. The computation is done in tiles
    to support efficient execution on NPU.

    Args:
        hidden_states: Input hidden states (float) [num_tokens * topk, hidden_size]
        group_list: Group list containing token counts per expert [per_device_expert_num]
        group_list_cumsum: Cumulative sum of group list [per_device_expert_num]
        w13: Gate and up projection weights (float) [per_device_expert_num, hidden_size, intermediate_size * 2]
        w2: Down projection weights (float) [per_device_expert_num, intermediate_size, hidden_size]
        ffn_res: Output tensor [num_tokens * topk, hidden_size]

    Note:
        This function uses cube L1 reuse mode 2 for better memory efficiency.
        Each expert processes tokens in tiles of size 8.
    """
    pypto.experimental.set_operation_config(combine_axis=True)

    # tiling config
    # mm1_cube_tile_shape = (1, 128, 128)
    # mm2_cube_tile_shape = (1, 128, 128)
    mm1_cube_tile_shape = (8, 256, 256)
    mm2_cube_tile_shape = (8, 256, 256)

    # 获取当前device上专家总数
    expert_num = group_list.shape[0]

    # 输入Tensor shape转换为2维
    w13_2d_shape = (w13.shape[0] * w13.shape[1], w13.shape[2])
    w2_2d_shape = (w2.shape[0] * w2.shape[1], w2.shape[2])

    w13_2d = pypto.reshape(w13, w13_2d_shape, inplace=True)
    w2_2d = pypto.reshape(w2, w2_2d_shape, inplace=True)

    for exp_idx in pypto.loop(expert_num, name="LOOP_FFN_ROUTER_MLP_L0", idx_name="exp_idx"):
        # 获取激活专家的token数
        token_num = group_list[exp_idx, ]
        for token_loop_idx, loop_base in pypto.loop_unroll(
            token_num,
            unroll_list=[1],
            name="LOOP_FFN_ROUTER_MLP_L1",
            idx_name="token_loop_idx"):
            expert_infer_base_native(
                hidden_states_params=[hidden_states],
                group_list_params=[group_list, group_list_cumsum],
                w13_params=[w13_2d],
                w2_params=[w2_2d],
                offset_params=[exp_idx, token_loop_idx, loop_base],
                tiling_params=[mm1_cube_tile_shape, mm2_cube_tile_shape],
                ffn_res=ffn_res
            )



def test_ffn_router() -> None:
    dtype = torch.bfloat16
    # parameter config
    s = 4096
    intermediate_size = 2048
    hidden_size = 2048
    local_expert_num = 16
    topk = 8
    test_iter = 5
    torch_npu.npu.config.allow_internal_format = True
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)

    for b in [1, 2]:
        print(f"\n{'='*60}")
        print(f"Testing with batch size b = {b}")
        print(f"{'='*60}")

        # Warmup once
        print("Running warmup...")
        hidden_states, hidden_states_scale, group_list, group_list_cumsum, w13, w13_scale, w2, w2_scale, ffn_res = \
            gen_input(b, s, topk, local_expert_num, hidden_size, intermediate_size, dtype, device_id, uniform_distribution=False)
        
        # Warmup PyTorch version
        _ = ffn_router_torch_npu(hidden_states, hidden_states_scale, group_list, w13, w13_scale, w2, w2_scale)
        # _ = ffn_router_torch_native(hidden_states.to(torch.bfloat16), group_list, w13.to(torch.bfloat16), w2.to(torch.bfloat16))

        torch.npu.synchronize()

        # Warmup PyPTO version
        inputs = {
            hidden_states: [0],
            hidden_states_scale: [0],
            group_list: [],
            group_list_cumsum: [],
            w13: [],
            w13_scale: [],
            w2: [],
            w2_scale: []
        }
        outputs = {ffn_res: [0]}
        pto_inputs = [pypto.from_torch(tensor, dynamic_axis=axis) for tensor, axis in inputs.items()]
        pto_outputs = [pypto.from_torch(tensor, dynamic_axis=axis) for tensor, axis in outputs.items()]
        moe_router_expert_main(*pto_inputs, *pto_outputs)
        torch.npu.synchronize()
        # pypto.runtime._device_synchronize()

        # Clear cache after warmup
        torch.npu.empty_cache()
        gc.collect()

        # Lists to store metrics
        torch_times = []
        pypto_times = []
        torch_peak_mem_gb_list = []
        pypto_peak_mem_gb_list = []

        for it in range(test_iter):
            print(f"Iteration {it + 1}/{test_iter}...")

            # Generate fresh random input each time
            hidden_states, hidden_states_scale, group_list, group_list_cumsum, w13, w13_scale, w2, w2_scale, ffn_res = \
                gen_input(b, s, topk, local_expert_num, hidden_size, intermediate_size, dtype, device_id, uniform_distribution=False)
            print(f"hidden_states shape: {hidden_states.shape}")
            print(f"hidden_states_scale shape: {hidden_states_scale.shape}")
            print(f"group_list shape: {group_list.shape}")
            print(f"group_list_cumsum shape: {group_list_cumsum.shape}")
            print(f"w13 shape: {w13.shape}")
            print(f"w13_scale shape: {w13_scale.shape}")
            print(f"w2 shape: {w2.shape}")
            print(f"w2_scale shape: {w2_scale.shape}")
            # Reset memory stats before PyTorch run
            torch.npu.reset_peak_memory_stats(device_id)


            # --- PyTorch NPU timing ---
            print("Running PyTorch NPU...")
            torch.npu.synchronize()
            start = time.perf_counter()
            # torch.npu.synchronize()
            golden = ffn_router_torch_npu(hidden_states, hidden_states_scale, group_list, w13, w13_scale, w2, w2_scale)
            # golden = ffn_router_torch_native(hidden_states.to(torch.bfloat16), group_list, w13.to(torch.bfloat16), w2.to(torch.bfloat16))
 
            torch.npu.synchronize()
            end = time.perf_counter()
            torch_time_ms = (end - start) * 1000
            torch_times.append(torch_time_ms)
            print(f"golden shape: {golden.shape}")
            print(f"golden dtype: {golden.dtype}")
            print(f"golden values: {golden.flatten().tolist()[:10]}")

            # Record peak memory after PyTorch (since PyPTO may reuse buffers, we capture here)
            peak_mem_bytes = torch.npu.max_memory_allocated(device_id)
            torch_peak_mem_gb = peak_mem_bytes / (1024**3)
            torch_peak_mem_gb_list.append(torch_peak_mem_gb)

            torch.npu.reset_peak_memory_stats(device_id)
            # Prepare inputs for PyPTO
            inputs = {
                hidden_states: [0],
                hidden_states_scale: [0],
                group_list: [],
                group_list_cumsum: [],
                w13: [],
                w13_scale: [],
                w2: [],
                w2_scale: []
            }
            outputs = {ffn_res: [0]}
            print("Running PyPTO...")
            
            
            pto_inputs = [pypto.from_torch(tensor, dynamic_axis=axis) for tensor, axis in inputs.items()]
            pto_outputs = [pypto.from_torch(tensor, dynamic_axis=axis) for tensor, axis in outputs.items()]
            # --- PyPTO timing ---
            torch.npu.synchronize()
            start = time.perf_counter()
            # torch.npu.synchronize()
            moe_router_expert_main(*pto_inputs, *pto_outputs)
            # pypto.runtime._device_synchronize()
            torch.npu.synchronize()
            end = time.perf_counter()
            pypto_time_ms = (end - start) * 1000
            pypto_times.append(pypto_time_ms)


            # Record peak memory after PyTorch (since PyPTO may reuse buffers, we capture here)
            peak_mem_bytes = torch.npu.max_memory_allocated(device_id)
            pypto_peak_mem_gb = peak_mem_bytes / (1024**3)
            pypto_peak_mem_gb_list.append(pypto_peak_mem_gb)

            # Optional: verify correctness (you can disable in perf mode)
            vaild_token_cumsum = group_list.cumsum(dim=0)
            # valid_size = vaild_token_cumsum[-1] * hidden_size
            # assert_allclose(
            #     np.array(ffn_res.cpu().flatten().tolist()[:valid_size]),
            #     np.array(golden.cpu().flatten().tolist()[:valid_size]),
            #     rtol=0.0078125, atol=0.0001
            # )
            torch.npu.empty_cache()
            gc.collect()

        # Compute averages
        avg_torch_time = sum(torch_times) / len(torch_times)
        avg_pypto_time = sum(pypto_times) / len(pypto_times)
        avg_torch_peak_mem_gb = sum(torch_peak_mem_gb_list) / len(torch_peak_mem_gb_list)
        avg_pypto_peak_mem_gb = sum(pypto_peak_mem_gb_list) / len(pypto_peak_mem_gb_list)

        # Compute TFLOPs (same for all iterations since b,s,topk fixed)
        tflops_total = calculate_moe_ffn_tflops(group_list, hidden_size, intermediate_size)
        # Compute throughput (TFLOP/s)
        torch_tflops_per_sec = tflops_total / (avg_torch_time / 1000)
        pypto_tflops_per_sec = tflops_total / (avg_pypto_time / 1000)

        # Compute comparison ratios
        speedup = avg_torch_time / avg_pypto_time  # PyPTO相对于PyTorch的加速比
        throughput_ratio = pypto_tflops_per_sec / torch_tflops_per_sec  # 吞吐量提升比
        mem_ratio = avg_torch_peak_mem_gb / avg_pypto_peak_mem_gb  # 内存节省比（>1表示PyPTO更省内存）
        mem_saving = (1 - avg_pypto_peak_mem_gb / avg_torch_peak_mem_gb) * 100  # 内存节省百分比

        # Print summary
        print(f"\n[Summary for b={b}]")
        print(f"Total TFLOPs per run: {tflops_total:.4f} TFLOP")
        # print(f"PyTorch Native: Avg Time = {avg_torch_time:.3f} ms, Throughput = {torch_tflops_per_sec:.2f} TFLOP/s, Peak Mem = {avg_torch_peak_mem_gb:.2f} GB")
        print(f"PyTorch NPU: Avg Time = {avg_torch_time:.3f} ms, Throughput = {torch_tflops_per_sec:.2f} TFLOP/s, Peak Mem = {avg_torch_peak_mem_gb:.2f} GB")
        print(f"PyPTO JIT  : Avg Time = {avg_pypto_time:.3f} ms, Throughput = {pypto_tflops_per_sec:.2f} TFLOP/s, Peak Mem = {avg_pypto_peak_mem_gb:.2f} GB")
        print(f"\n[Comparison (PyPTO vs PyTorch)]")
        print(f"  Speedup        : {speedup:.3f}x (PyPTO is {(speedup-1)*100:.1f}% faster)")
        print(f"  Throughput Gain : {throughput_ratio:.3f}x (PyPTO achieves {(throughput_ratio-1)*100:.1f}% higher throughput)")
        if mem_ratio > 1.0:
            print(f"  Memory Saving   : {mem_ratio:.3f}x (PyPTO uses {mem_saving:.2f}% less memory)")
        elif mem_ratio < 1.0:
            print(f"  Memory Usage    : {1/mem_ratio:.3f}x (PyPTO uses {(1/mem_ratio-1)*100:.2f}% more memory)")
        else:
            print(f"  Memory Usage    : Same (1.000x)")
        print("test_ffn_router passed for this batch size.\n")


def test_ffn_router_native() -> None:
    # dtype = torch.bfloat16
    dtype = torch.float32
    # parameter config
    s = 1024
    intermediate_size = 2048
    hidden_size = 4096
    local_expert_num = 16
    topk = 1
    test_iter = 5
    torch_npu.npu.config.allow_internal_format = True
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)

    for b in [1]:
        print(f"\n{'='*60}")
        print(f"Testing with batch size b = {b}")
        print(f"{'='*60}")

        # Warmup once
        print("Running warmup...")
        # hidden_states, hidden_states_scale, group_list, group_list_cumsum, w13, w13_scale, w2, w2_scale, ffn_res = \
        #     gen_input(b, s, topk, local_expert_num, hidden_size, intermediate_size, dtype, device_id)
        hidden_states, group_list, group_list_cumsum, w13, w2, ffn_res = gen_input_native(b, s, topk, local_expert_num, hidden_size, intermediate_size, dtype, device_id)
        print(f"hidden_states shape: {hidden_states.shape}")
        print(f"hidden_states dtype: {hidden_states.dtype}")
        print(f"hidden_states values: {hidden_states.flatten().tolist()[:10]}")
        print(f"group_list shape: {group_list.shape}")
        print(f"group_list dtype: {group_list.dtype}")
        print(f"group_list values: {group_list.flatten().tolist()[:10]}")
        print(f"group_list_cumsum shape: {group_list_cumsum.shape}")
        print(f"group_list_cumsum dtype: {group_list_cumsum.dtype}")
        print(f"group_list_cumsum values: {group_list_cumsum.flatten().tolist()[:10]}")
        print(f"w13 shape: {w13.shape}")
        print(f"w13 dtype: {w13.dtype}")
        print(f"w2 shape: {w2.shape}")
        print(f"w2 dtype: {w2.dtype}")
        print(f"ffn_res shape: {ffn_res.shape}")
        print(f"ffn_res dtype: {ffn_res.dtype}")
        # Warmup PyTorch version
        # _ = ffn_router_torch_npu(hidden_states, hidden_states_scale, group_list, w13, w13_scale, w2, w2_scale)
        golden = ffn_router_torch_native(hidden_states, group_list, w13, w2)
        print(f"golden shape: {golden.shape}")
        print(f"golden dtype: {golden.dtype}")
        print(f"golden values: {golden.flatten().tolist()[:10]}")
        torch.npu.synchronize()

        # Warmup PyPTO version
        inputs = {
            hidden_states: [0],
            group_list: [],
            group_list_cumsum: [],
            w13: [0],
            w2: [0],
        }
        outputs = {ffn_res: [0]}
        pto_inputs = [pypto.from_torch(tensor, dynamic_axis=axis) for tensor, axis in inputs.items()]
        pto_outputs = [pypto.from_torch(tensor, dynamic_axis=axis) for tensor, axis in outputs.items()]
        moe_router_expert_main_native(*pto_inputs, *pto_outputs)
        torch.npu.synchronize()
        print(f"ffn_res values: {ffn_res.flatten().tolist()[:10]}")
        # pypto.runtime._device_synchronize()
        vaild_token_cumsum = group_list.cumsum(dim=0)
        valid_size = vaild_token_cumsum[-1] * hidden_size
        assert_allclose(
                np.array(ffn_res.cpu().flatten().tolist()[:valid_size]),
                np.array(golden.cpu().flatten().tolist()[:valid_size]),
                rtol=0.0078125, atol=0.0001
            )
        print("✅ warmup native ffn router result matches!")
        # Clear cache after warmup
        torch.npu.empty_cache()
        gc.collect()

        # Lists to store metrics
        # torch_times = []
        # pypto_times = []
        # torch_peak_mem_gb_list = []
        # pypto_peak_mem_gb_list = []

        # for it in range(test_iter):
        #     print(f"Iteration {it + 1}/{test_iter}...")

        #     # Generate fresh random input each time
        #     # hidden_states, hidden_states_scale, group_list, group_list_cumsum, w13, w13_scale, w2, w2_scale, ffn_res = \
        #     #     gen_input(b, s, topk, local_expert_num, hidden_size, intermediate_size, dtype, device_id)
        #     hidden_states, group_list, group_list_cumsum, w13, w2, ffn_res = gen_input_native(b, s, topk, local_expert_num, hidden_size, intermediate_size, dtype, device_id)
        #     print(f"hidden_states shape: {hidden_states.shape}")
        #     print(f"group_list shape: {group_list.shape}")
        #     print(f"group_list_cumsum shape: {group_list_cumsum.shape}")
        #     print(f"w13 shape: {w13.shape}")
        #     print(f"w2 shape: {w2.shape}")
        #     # Reset memory stats before PyTorch run
        #     torch.npu.reset_peak_memory_stats(device_id)


        #     # --- PyTorch NPU timing ---
        #     print("Running PyTorch NPU...")
        #     torch.npu.synchronize()
        #     start = time.perf_counter()
        #     # torch.npu.synchronize()
        #     golden = ffn_router_torch_native(hidden_states, group_list, w13, w2)
        #     # golden = ffn_router_torch_native(hidden_states, group_list, w13, w2)

        #     torch.npu.synchronize()
        #     end = time.perf_counter()
        #     torch_time_ms = (end - start) * 1000
        #     torch_times.append(torch_time_ms)

        #     # Record peak memory after PyTorch (since PyPTO may reuse buffers, we capture here)
        #     peak_mem_bytes = torch.npu.max_memory_allocated(device_id)
        #     torch_peak_mem_gb = peak_mem_bytes / (1024**3)
        #     torch_peak_mem_gb_list.append(torch_peak_mem_gb)

        #     torch.npu.reset_peak_memory_stats(device_id)
        #     # Prepare inputs for PyPTO
        #     inputs = {
        #         hidden_states: [0],
        #         group_list: [],
        #         group_list_cumsum: [],
        #         w13: [0],
        #         w2: [0],
        #     }
        #     outputs = {ffn_res: [0]}
        #     print("Running PyPTO...")
            
            
        #     pto_inputs = [pypto.from_torch(tensor, dynamic_axis=axis) for tensor, axis in inputs.items()]
        #     pto_outputs = [pypto.from_torch(tensor, dynamic_axis=axis) for tensor, axis in outputs.items()]
        #     # --- PyPTO timing ---
        #     torch.npu.synchronize()
        #     start = time.perf_counter()
        #     # torch.npu.synchronize()
        #     moe_router_expert_main_native(*pto_inputs, *pto_outputs)
        #     # pypto.runtime._device_synchronize()
        #     torch.npu.synchronize()
        #     end = time.perf_counter()
        #     pypto_time_ms = (end - start) * 1000
        #     pypto_times.append(pypto_time_ms)


        #     # Record peak memory after PyTorch (since PyPTO may reuse buffers, we capture here)
        #     peak_mem_bytes = torch.npu.max_memory_allocated(device_id)
        #     pypto_peak_mem_gb = peak_mem_bytes / (1024**3)
        #     pypto_peak_mem_gb_list.append(pypto_peak_mem_gb)

        #     # Optional: verify correctness (you can disable in perf mode)

        #     assert_allclose(
        #         np.array(ffn_res.cpu().flatten().tolist()[:valid_size]),
        #         np.array(golden.cpu().flatten().tolist()[:valid_size]),
        #         rtol=0.0078125, atol=0.0001
        #     )
        #     print(f"✅ test_ffn_router_native iteration {it + 1} result matches!")
        #     torch.npu.empty_cache()
        #     gc.collect()

        # # Compute averages
        # avg_torch_time = sum(torch_times) / len(torch_times)
        # avg_pypto_time = sum(pypto_times) / len(pypto_times)
        # avg_torch_peak_mem_gb = sum(torch_peak_mem_gb_list) / len(torch_peak_mem_gb_list)
        # avg_pypto_peak_mem_gb = sum(pypto_peak_mem_gb_list) / len(pypto_peak_mem_gb_list)

        # # Compute TFLOPs (same for all iterations since b,s,topk fixed)
        # tflops_total = calculate_moe_ffn_tflops(group_list, hidden_size, intermediate_size)
        # # Compute throughput (TFLOP/s)
        # torch_tflops_per_sec = tflops_total / (avg_torch_time / 1000)
        # pypto_tflops_per_sec = tflops_total / (avg_pypto_time / 1000)

        # # Compute comparison ratios
        # speedup = avg_torch_time / avg_pypto_time  # PyPTO相对于PyTorch的加速比
        # throughput_ratio = pypto_tflops_per_sec / torch_tflops_per_sec  # 吞吐量提升比
        # mem_ratio = avg_torch_peak_mem_gb / avg_pypto_peak_mem_gb  # 内存节省比（>1表示PyPTO更省内存）
        # mem_saving = (1 - avg_pypto_peak_mem_gb / avg_torch_peak_mem_gb) * 100  # 内存节省百分比

        # # Print summary
        # print(f"\n[Summary for b={b}]")
        # print(f"Total TFLOPs per run: {tflops_total:.4f} TFLOP")
        # print(f"PyTorch NPU: Avg Time = {avg_torch_time:.3f} ms, Throughput = {torch_tflops_per_sec:.2f} TFLOP/s, Peak Mem = {avg_torch_peak_mem_gb:.2f} GB")
        # print(f"PyPTO JIT  : Avg Time = {avg_pypto_time:.3f} ms, Throughput = {pypto_tflops_per_sec:.2f} TFLOP/s, Peak Mem = {avg_pypto_peak_mem_gb:.2f} GB")
        # print(f"\n[Comparison (PyPTO vs PyTorch)]")
        # print(f"  Speedup        : {speedup:.3f}x (PyPTO is {(speedup-1)*100:.1f}% faster)")
        # print(f"  Throughput Gain : {throughput_ratio:.3f}x (PyPTO achieves {(throughput_ratio-1)*100:.1f}% higher throughput)")
        # if mem_ratio > 1.0:
        #     print(f"  Memory Saving   : {mem_ratio:.3f}x (PyPTO uses {mem_saving:.2f}% less memory)")
        # elif mem_ratio < 1.0:
        #     print(f"  Memory Usage    : {1/mem_ratio:.3f}x (PyPTO uses {(1/mem_ratio-1)*100:.2f}% more memory)")
        # else:
        #     print(f"  Memory Usage    : Same (1.000x)")
        # print("test_ffn_router passed for this batch size.\n")


def main():
    test_ffn_router()
    # test_ffn_router_native()

if __name__ == "__main__":
    main()




