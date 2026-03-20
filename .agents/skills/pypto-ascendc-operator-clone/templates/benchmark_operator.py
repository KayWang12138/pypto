#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AscendC算子性能测试脚本模板

使用方法:
1. 复制此模板到算子测试目录
2. 根据算子修改参数配置和算子调用
3. 使用msprof执行性能采集:
   msprof --output=/tmp/msprof_output --ai-core=on --task-time=l2 python benchmark_operator.py
"""

import torch
import torch_npu
import os

# ============== 配置区域 ==============
# NPU设备配置
DEVICE_ID = 0

# 算子参数配置（根据算子修改）
B = 2       # Batch size
N = 8       # Head num
Sq = 16     # Query sequence length
Skv = 16    # Key/Value sequence length
D = 64      # Head dimension
DTYPE = torch.bfloat16

# 测试配置
WARMUP_ITERATIONS = 5    # warmup迭代次数
TEST_ITERATIONS = 100    # 正式测试迭代次数


def setup_device(device_id):
    """配置NPU设备"""
    device = torch.device(f"npu:{device_id}")
    torch.npu.set_device(device_id)
    return device


def generate_inputs(B, N, Sq, Skv, D, dtype, device):
    """生成测试输入数据（根据算子修改）"""
    # 示例：Flash Attention的输入
    q = torch.randn(B, N, Sq, D, dtype=dtype, device=device)
    k = torch.randn(B, N, Skv, D, dtype=dtype, device=device)
    v = torch.randn(B, N, Skv, D, dtype=dtype, device=device)
    return q, k, v


def run_operator_iteration(inputs, device):
    """单次算子执行（根据算子修改）"""
    q, k, v = inputs
    
    # 示例：调用Flash Attention算子
    # out = torch_npu.npu_fusion_attention_v2(
    #     q, k, v, N, "BNSD",
    #     scale=1.0/(D**0.5)
    # )
    
    # TODO: 替换为实际算子调用
    out = torch_npu.npu_xxx(q, k, v)  # 修改为实际算子API
    
    return out


def run_benchmark(device, inputs, warmup_iters, test_iters):
    """执行性能测试"""
    print(f"Warmup iterations: {warmup_iters}")
    
    # Warmup阶段
    for _ in range(warmup_iters):
        _ = run_operator_iteration(inputs, device)
    torch.npu.synchronize()
    
    print(f"Test iterations: {test_iters}")
    
    # 正式测试阶段
    for i in range(test_iters):
        _ = run_operator_iteration(inputs, device)
    torch.npu.synchronize()
    
    print("Benchmark completed")


def main():
    """主函数"""
    print("=" * 60)
    print("AscendC Operator Benchmark")
    print("=" * 60)
    
    # 配置设备
    device = setup_device(DEVICE_ID)
    print(f"Device: npu:{DEVICE_ID}")
    
    # 打印参数配置
    print(f"\nOperator Parameters:")
    print(f"  B={B}, N={N}, Sq={Sq}, Skv={Skv}, D={D}")
    print(f"  dtype={DTYPE}")
    print(f"\nTest Configuration:")
    print(f"  Warmup: {WARMUP_ITERATIONS} iterations")
    print(f"  Test: {TEST_ITERATIONS} iterations")
    
    # 生成输入
    inputs = generate_inputs(B, N, Sq, Skv, D, DTYPE, device)
    
    # 执行测试
    print("\nRunning benchmark...")
    run_benchmark(device, inputs, WARMUP_ITERATIONS, TEST_ITERATIONS)
    
    print("=" * 60)


if __name__ == "__main__":
    main()