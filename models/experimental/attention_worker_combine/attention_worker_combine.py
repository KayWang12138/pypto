#!/usr/bin/env python3
# coding: utf-8
"""
Attention Worker Combine Operator - MoE Token Fusion

功能：MoE 模型中的注意力 token 融合算子，将多个专家的计算结果按权重加权求和

计算公式：
    y = Σ(expert_scales[i] × token_data[i]) + token_data[K]
    
其中：
    - K 为 TopK 选出的专家数量
    - expert_scales[i] 为第 i 个专家的权重
    - token_data[i] 为第 i 个专家处理的 token 数据
    - token_data[K] 为共享专家（shared expert）的输出数据

简化实现：
    - schedule_context 作为占位参数，暂不使用
    - token_data 作为显式输入参数
    - 暂不实现三轴切分策略
"""

import numpy as np
import torch
import torch_npu  # 必须导入 torch_npu 才能使用 NPU
import pytest
import pypto
from pypto import DataType as DT
from typing import Callable

np.random.seed(42)
torch.manual_seed(42)


def attention_worker_combine_golden(
    expert_scales: np.ndarray,
    token_data: np.ndarray,
    hidden_size: int,
    token_dtype: int,
) -> np.ndarray:
    """
    Golden 函数：使用 NumPy 实现专家加权融合
    
    参数：
        expert_scales: [BatchSize, K] float32 - 专家权重
        token_data: [K+1, BatchSize, HiddenSize] float16/bfloat16 - 专家输出数据
        hidden_size: int - 隐藏维度大小
        token_dtype: int - 数据类型（0=float16, 1=bfloat16）
    
    返回：
        y: [BatchSize, HiddenSize] float16/bfloat16 - 融合输出
    """
    batch_size = expert_scales.shape[0]
    K = expert_scales.shape[1]
    
    # 获取路由专家的输出（前 K 个）
    routed_expert_outputs = token_data[:K]  # [K, BatchSize, HiddenSize]
    
    # 获取共享专家的输出（第 K 个）
    shared_expert_output = token_data[K]  # [BatchSize, HiddenSize]
    
    # 转换为 float32 进行计算，避免精度损失
    if token_dtype == 0:  # float16
        routed_expert_outputs_fp32 = routed_expert_outputs.astype(np.float32)
        shared_expert_output_fp32 = shared_expert_output.astype(np.float32)
    else:  # bfloat16
        # NumPy 不直接支持 bfloat16，使用 float32 代替
        routed_expert_outputs_fp32 = routed_expert_outputs.astype(np.float32)
        shared_expert_output_fp32 = shared_expert_output.astype(np.float32)
    
    # 加权求和：y = Σ(expert_scales[i] × token_data[i])
    # expert_scales: [BatchSize, K]
    # routed_expert_outputs: [K, BatchSize, HiddenSize]
    # 结果：[BatchSize, HiddenSize]
    
    # 转置 routed_expert_outputs 为 [BatchSize, K, HiddenSize]
    routed_expert_outputs_transposed = np.transpose(routed_expert_outputs_fp32, (1, 0, 2))
    
    # 扩展 expert_scales 维度：[BatchSize, K, 1]
    expert_scales_expanded = expert_scales[:, :, np.newaxis]
    
    # 加权求和：[BatchSize, K, HiddenSize] * [BatchSize, K, 1] -> [BatchSize, K, HiddenSize]
    weighted_outputs = routed_expert_outputs_transposed * expert_scales_expanded
    
    # 沿 K 轴求和：[BatchSize, HiddenSize]
    y = np.sum(weighted_outputs, axis=1)
    
    # 加上共享专家输出
    y = y + shared_expert_output_fp32
    
    # 转换回目标数据类型
    if token_dtype == 0:  # float16
        y = y.astype(np.float16)
    else:  # bfloat16
        # 使用 torch 进行 bfloat16 转换（NumPy 不直接支持）
        y_tensor = torch.from_numpy(y).bfloat16()
        y = y_tensor.float().numpy()  # 转回 float32 以便后续验证
    
    return y


def attention_worker_combine_kernel(
    K: int,
    hidden_size: int,
    token_dtype: int,
) -> Callable:
    """
    Kernel 工厂函数：创建专家加权融合 kernel
    
    简化实现：固定 batch_size=1
    
    参数：
        K: int - 专家数量
        hidden_size: int - 隐藏维度大小
        token_dtype: int - 数据类型（0=float16, 1=bfloat16）
    
    返回：
        kernel 函数
    """
    # 设置数据类型
    if token_dtype == 0:
        dtype = pypto.DT_FP16
    else:
        dtype = pypto.DT_BF16
    
    @pypto.frontend.jit(debug_options={"runtime_debug_mode": 1})
    def kernel(
        expert_scales: pypto.Tensor([1, K], pypto.DT_FP32),
        routed_expert_data: pypto.Tensor([K, hidden_size], dtype),
        shared_expert_data: pypto.Tensor([1, hidden_size], dtype),
        y: pypto.Tensor([1, hidden_size], dtype),
    ):
        """
        Kernel 函数：PyPTO 实现专家加权融合（单批次）
        
        参数：
            expert_scales: [1, K] float32 - 专家权重
            routed_expert_data: [K, HiddenSize] float16/bfloat16 - 路由专家输出
            shared_expert_data: [1, HiddenSize] float16/bfloat16 - 共享专家输出
            y: [1, HiddenSize] float16/bfloat16 - 融合输出
        """
        # 设置 vec tile shapes
        pypto.set_vec_tile_shapes(K, hidden_size)
        
        # 转换为 float32
        pypto.set_vec_tile_shapes(K // 2, hidden_size)
        routed_out_fp32 = pypto.cast(routed_expert_data, pypto.DT_FP32)
        
        # 设置 cube tile shapes
        k_tile_shape = ((K + 15) // 16) * 16
        l0b_size = 65536
        n_tile_shape = min(l0b_size // pypto.bytes_of(pypto.DT_FP32) // k_tile_shape, hidden_size)
        n_tile_shape = ((n_tile_shape + 15) // 16) * 16
        pypto.set_cube_tile_shapes([1, 1], [k_tile_shape, k_tile_shape], [n_tile_shape, n_tile_shape])
        
        # 加权求和: [1, K] @ [K, HiddenSize] = [1, HiddenSize]
        weighted_sum_fp32 = expert_scales.matmul(routed_out_fp32, pypto.DT_FP32)
        
        # 获取共享专家输出并转换
        shared_out_fp32 = pypto.cast(shared_expert_data, pypto.DT_FP32)
        
        # 加上共享专家输出
        result_fp32 = pypto.add(weighted_sum_fp32, shared_out_fp32)
        
        # 转换回目标数据类型
        result_final = pypto.cast(result_fp32, dtype)
        
        # 写入输出
        y[0:, :] = result_final
    
    return kernel


def attention_worker_combine_wrapper(
    batch_size: int,
    K: int,
    hidden_size: int,
    token_dtype: int,
) -> Callable:
    """
    包装函数：处理多批次调用
    
    参数：
        batch_size: int - 批次大小
        K: int - 专家数量
        hidden_size: int - 隐藏维度大小
        token_dtype: int - 数据类型（0=float16, 1=bfloat16）
    
    返回：
        wrapper 函数
    """
    kernel = attention_worker_combine_kernel(K, hidden_size, token_dtype)
    
    def wrapper(
        expert_scales: torch.Tensor,
        routed_expert_data: torch.Tensor,
        shared_expert_data: torch.Tensor,
        y: torch.Tensor,
    ):
        """
        包装函数：逐批次调用 kernel
        
        参数：
            expert_scales: [BatchSize, K] float32
            routed_expert_data: [BatchSize * K, HiddenSize]
            shared_expert_data: [BatchSize, HiddenSize]
            y: [BatchSize, HiddenSize]
        """
        for i in range(batch_size):
            # 获取当前批次的数据
            scales_i = expert_scales[i:i+1, :]
            routed_i = routed_expert_data[i*K:(i+1)*K, :]
            shared_i = shared_expert_data[i:i+1, :]
            y_i = y[i:i+1, :]
            
            # 调用 kernel
            kernel(scales_i, routed_i, shared_i, y_i)
    
    return wrapper


def test_attention_worker_combine():
    """测试 attention_worker_combine 算子"""
    
    device = 'npu:0' if torch.npu.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    batch_size, K, hidden_size, token_dtype = 1, 8, 2048, 0
    print(f"\n配置: batch_size={batch_size}, K={K}, hidden_size={hidden_size}, dtype=float16")
    
    expert_scales_np = np.random.randn(batch_size, K).astype(np.float32)
    expert_scales_np = expert_scales_np / expert_scales_np.sum(axis=1, keepdims=True)
    
    token_data_np = np.random.randn(K + 1, batch_size, hidden_size).astype(np.float16)
    
    y_golden = attention_worker_combine_golden(expert_scales_np, token_data_np, hidden_size, token_dtype)
    
    expert_scales_torch = torch.from_numpy(expert_scales_np).to(device)
    routed_expert_3d = torch.from_numpy(token_data_np[:K]).transpose(0, 1)
    routed_expert_data = routed_expert_3d.reshape(batch_size * K, hidden_size).contiguous().to(device)
    shared_expert_data = torch.from_numpy(token_data_np[K]).transpose(0, 1).contiguous().to(device)
    y_torch = torch.zeros(batch_size, hidden_size, dtype=torch.float16, device=device)
    
    wrapper = attention_worker_combine_wrapper(batch_size, K, hidden_size, token_dtype)
    wrapper(expert_scales_torch, routed_expert_data, shared_expert_data, y_torch)
    
    y_output_np = y_torch.cpu().numpy()
    np.testing.assert_allclose(y_output_np, y_golden, atol=0.001, rtol=0.01)
    print(f"✓ 测试通过")


if __name__ == "__main__":
    test_attention_worker_combine()