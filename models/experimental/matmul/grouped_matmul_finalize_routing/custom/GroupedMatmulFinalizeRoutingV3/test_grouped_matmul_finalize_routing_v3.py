#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""PyPTO GroupedMatmulFinalizeRoutingV3 operator test.

测试 MoE 融合算子的精度验证，包含：
1. MXFP8 基础场景测试
2. 转置权重测试
3. 可选参数测试（bias, shared_input）
4. 不同专家数量测试

精度验证使用 numpy.testing.assert_allclose。
"""

import os
import sys
import math
import argparse

import torch
import torch_npu
import numpy as np
from numpy.testing import assert_allclose

from grouped_matmul_finalize_routing_v3_golden import grouped_matmul_finalize_routing_v3_golden
from grouped_matmul_finalize_routing_v3_impl import grouped_matmul_finalize_routing_v3_wrapper


# ─────────────────────────────────────────────
# 1. 环境工具
# ─────────────────────────────────────────────

def get_device_id():
    """从环境变量获取 TILE_FWK_DEVICE_ID。"""
    if "TILE_FWK_DEVICE_ID" not in os.environ:
        print("Please set: export TILE_FWK_DEVICE_ID=0")
        return None
    try:
        return int(os.environ["TILE_FWK_DEVICE_ID"])
    except ValueError:
        print(f"ERROR: TILE_FWK_DEVICE_ID must be int, got: {os.environ['TILE_FWK_DEVICE_ID']}")
        return None


# ─────────────────────────────────────────────
# 2. 测试数据生成
# ─────────────────────────────────────────────

def generate_test_data(
    m: int,
    k: int,
    n: int,
    e: int,
    batch: int,
    group_list: list,
    transpose_x2: bool = False,
    include_bias: bool = False,
    include_shared: bool = False,
):
    """生成 MXFP8 格式测试数据。"""
    
    # 输入 x1 [M, K]
    x1 = torch.randn((m, k), dtype=torch.float32).uniform_(0, 1).to(torch.float8_e4m3fn)
    
    # pertoken_scale [M, Ceil(K/64), 2]
    pertoken_scale = torch.randn(
        (m, math.ceil(k / 64), 2), dtype=torch.float32
    ).uniform_(0.5, 1.5).to(torch.float8_e8m0fnu)
    
    # 权重 x2 [E, K, N] 或 [E, N, K]
    if transpose_x2:
        x2 = torch.randn((e, n, k), dtype=torch.float32).uniform_(0, 1).to(torch.float8_e4m3fn)
        scale = torch.randn(
            (e, n, math.ceil(k / 64), 2), dtype=torch.float32
        ).uniform_(0.5, 1.5).to(torch.float8_e8m0fnu)
    else:
        x2 = torch.randn((e, k, n), dtype=torch.float32).uniform_(0, 1).to(torch.float8_e4m3fn)
        scale = torch.randn(
            (e, math.ceil(k / 64), n, 2), dtype=torch.float32
        ).uniform_(0.5, 1.5).to(torch.float8_e8m0fnu)
    
    # 路由索引 [M]
    row_index = torch.randint(0, batch, (m,), dtype=torch.int64)
    
    # logit [M] (required per SPEC)
    logit = torch.randn((m,), dtype=torch.float32).uniform_(0.5, 2.0)
    
    # bias [E, N] (可选)
    bias = None
    if include_bias:
        bias = torch.randn((e, n), dtype=torch.bfloat16)
    
    # shared_input [bsdp, N] (可选)
    shared_input = None
    bsdp = batch // e if include_shared else 0
    if include_shared and bsdp > 0:
        shared_input = torch.randn((bsdp, n), dtype=torch.bfloat16)
    
    return {
        'x1': x1,
        'x2': x2,
        'scale': scale,
        'pertoken_scale': pertoken_scale,
        'group_list': group_list,
        'row_index': row_index,
        'logit': logit,
        'batch': batch,
        'n': n,
        'bias': bias,
        'shared_input': shared_input,
        'shared_input_weight': 1.0,
        'shared_input_offset': 0,
        'transpose_x1': False,
        'transpose_x2': transpose_x2,
        'group_list_type': 1,
    }


# ─────────────────────────────────────────────
# 3. 测试函数
# ─────────────────────────────────────────────

def test_mxfp8_basic(device_id=None, run_mode="npu"):
    """Test 1: MXFP8 基础配置验证。"""
    print("=" * 60)
    print("Test: GroupedMatmulFinalizeRoutingV3 - MXFP8 Basic")
    print("=" * 60)
    
    # 参数配置（来自 SPEC.md 典型配置）
    m, k, n, e, batch = 16, 512, 7168, 2, 8
    group_list = [7, 9]
    
    print(f"  Config: m={m}, k={k}, n={n}, e={e}, batch={batch}")
    print(f"  group_list: {group_list}")
    
    # 生成测试数据
    data = generate_test_data(
        m=m, k=k, n=n, e=e, batch=batch, group_list=group_list
    )
    
    # 执行 PyPTO wrapper
    result = grouped_matmul_finalize_routing_v3_wrapper(
        x1=data['x1'],
        x2=data['x2'],
        scale=data['scale'],
        pertoken_scale=data['pertoken_scale'],
        group_list=data['group_list'],
        row_index=data['row_index'],
        logit=data['logit'],
        batch=data['batch'],
        n=data['n'],
        bias=data['bias'],
        shared_input=data['shared_input'],
        transpose_x2=data['transpose_x2'],
    )
    
    # 执行 golden
    golden = grouped_matmul_finalize_routing_v3_golden(**data)
    
    # 精度对比
    print(f"  Output shape: {result.shape}")
    max_diff = np.abs(result.cpu().numpy() - golden.cpu().numpy()).max()
    print(f"  Max diff: {max_diff:.6e}")
    
    if run_mode == "npu":
        assert_allclose(
            result.cpu().numpy(),
            golden.cpu().numpy(),
            rtol=1e-3, atol=1e-3,
        )
    
    print("  ✓ Passed\n")
    print("[PRECISION_PASS]")


def test_mxfp8_transpose(device_id=None, run_mode="npu"):
    """Test 2: 转置权重场景验证。"""
    print("=" * 60)
    print("Test: GroupedMatmulFinalizeRoutingV3 - Transpose Weight")
    print("=" * 60)
    
    m, k, n, e, batch = 16, 512, 7168, 2, 8
    group_list = [7, 9]
    transpose_x2 = True
    
    print(f"  Config: m={m}, k={k}, n={n}, e={e}, batch={batch}")
    print(f"  group_list: {group_list}, transpose_x2=True")
    
    data = generate_test_data(
        m=m, k=k, n=n, e=e, batch=batch, group_list=group_list,
        transpose_x2=transpose_x2,
    )
    
    result = grouped_matmul_finalize_routing_v3_wrapper(
        x1=data['x1'],
        x2=data['x2'],
        scale=data['scale'],
        pertoken_scale=data['pertoken_scale'],
        group_list=data['group_list'],
        row_index=data['row_index'],
        logit=data['logit'],
        batch=data['batch'],
        n=data['n'],
        transpose_x2=data['transpose_x2'],
    )
    
    golden = grouped_matmul_finalize_routing_v3_golden(**data)
    
    print(f"  Output shape: {result.shape}")
    max_diff = np.abs(result.cpu().numpy() - golden.cpu().numpy()).max()
    print(f"  Max diff: {max_diff:.6e}")
    
    if run_mode == "npu":
        assert_allclose(
            result.cpu().numpy(),
            golden.cpu().numpy(),
            rtol=1e-3, atol=1e-3,
        )
    
    print("  ✓ Passed\n")
    print("[PRECISION_PASS]")


def test_mxfp8_with_shared_input(device_id=None, run_mode="npu"):
    """Test 3: 包含共享专家输出场景验证。"""
    print("=" * 60)
    print("Test: GroupedMatmulFinalizeRoutingV3 - With Shared Input")
    print("=" * 60)
    
    m, k, n, e, batch = 16, 512, 7168, 2, 8
    group_list = [7, 9]
    
    print(f"  Config: m={m}, k={k}, n={n}, e={e}, batch={batch}")
    print(f"  group_list: {group_list}, include_shared=True")
    
    data = generate_test_data(
        m=m, k=k, n=n, e=e, batch=batch, group_list=group_list,
        include_shared=True,
    )
    
    result = grouped_matmul_finalize_routing_v3_wrapper(
        x1=data['x1'],
        x2=data['x2'],
        scale=data['scale'],
        pertoken_scale=data['pertoken_scale'],
        group_list=data['group_list'],
        row_index=data['row_index'],
        logit=data['logit'],
        batch=data['batch'],
        n=data['n'],
        shared_input=data['shared_input'],
        shared_input_weight=data['shared_input_weight'],
        shared_input_offset=data['shared_input_offset'],
    )
    
    golden = grouped_matmul_finalize_routing_v3_golden(**data)
    
    print(f"  Output shape: {result.shape}")
    max_diff = np.abs(result.cpu().numpy() - golden.cpu().numpy()).max()
    print(f"  Max diff: {max_diff:.6e}")
    
    if run_mode == "npu":
        assert_allclose(
            result.cpu().numpy(),
            golden.cpu().numpy(),
            rtol=1e-3, atol=1e-3,
        )
    
    print("  ✓ Passed\n")
    print("[PRECISION_PASS]")


def test_mxfp8_with_bias(device_id=None, run_mode="npu"):
    """Test 4: 包含 bias 场景验证。"""
    print("=" * 60)
    print("Test: GroupedMatmulFinalizeRoutingV3 - With Bias")
    print("=" * 60)
    
    m, k, n, e, batch = 16, 512, 7168, 2, 8
    group_list = [7, 9]
    
    print(f"  Config: m={m}, k={k}, n={n}, e={e}, batch={batch}")
    print(f"  group_list: {group_list}, include_bias=True")
    
    data = generate_test_data(
        m=m, k=k, n=n, e=e, batch=batch, group_list=group_list,
        include_bias=True,
    )
    
    result = grouped_matmul_finalize_routing_v3_wrapper(
        x1=data['x1'],
        x2=data['x2'],
        scale=data['scale'],
        pertoken_scale=data['pertoken_scale'],
        group_list=data['group_list'],
        row_index=data['row_index'],
        logit=data['logit'],
        batch=data['batch'],
        n=data['n'],
        bias=data['bias'],
    )
    
    golden = grouped_matmul_finalize_routing_v3_golden(**data)
    
    print(f"  Output shape: {result.shape}")
    max_diff = np.abs(result.cpu().numpy() - golden.cpu().numpy()).max()
    print(f"  Max diff: {max_diff:.6e}")
    
    if run_mode == "npu":
        assert_allclose(
            result.cpu().numpy(),
            golden.cpu().numpy(),
            rtol=1e-3, atol=1e-3,
        )
    
    print("  ✓ Passed\n")
    print("[PRECISION_PASS]")


def test_mxfp8_large_experts(device_id=None, run_mode="npu"):
    """Test 5: 大专家数量场景验证。"""
    print("=" * 60)
    print("Test: GroupedMatmulFinalizeRoutingV3 - Large Experts (E=8)")
    print("=" * 60)
    
    m, k, n, e, batch = 128, 512, 7168, 8, 32
    group_list = [16, 16, 16, 16, 16, 16, 16, 16]
    
    print(f"  Config: m={m}, k={k}, n={n}, e={e}, batch={batch}")
    print(f"  group_list: {group_list}")
    
    data = generate_test_data(
        m=m, k=k, n=n, e=e, batch=batch, group_list=group_list,
    )
    
    result = grouped_matmul_finalize_routing_v3_wrapper(
        x1=data['x1'],
        x2=data['x2'],
        scale=data['scale'],
        pertoken_scale=data['pertoken_scale'],
        group_list=data['group_list'],
        row_index=data['row_index'],
        logit=data['logit'],
        batch=data['batch'],
        n=data['n'],
    )
    
    golden = grouped_matmul_finalize_routing_v3_golden(**data)
    
    print(f"  Output shape: {result.shape}")
    max_diff = np.abs(result.cpu().numpy() - golden.cpu().numpy()).max()
    print(f"  Max diff: {max_diff:.6e}")
    
    if run_mode == "npu":
        assert_allclose(
            result.cpu().numpy(),
            golden.cpu().numpy(),
            rtol=1e-3, atol=1e-3,
        )
    
    print("  ✓ Passed\n")
    print("[PRECISION_PASS]")


def test_mxfp8_full_config(device_id=None, run_mode="npu"):
    """Test 6: 完整配置验证（包含所有可选参数）。"""
    print("=" * 60)
    print("Test: GroupedMatmulFinalizeRoutingV3 - Full Config")
    print("=" * 60)
    
    m, k, n, e, batch = 16, 512, 7168, 2, 8
    group_list = [7, 9]
    
    print(f"  Config: m={m}, k={k}, n={n}, e={e}, batch={batch}")
    print(f"  group_list: {group_list}, bias=True, shared_input=True")
    
    data = generate_test_data(
        m=m, k=k, n=n, e=e, batch=batch, group_list=group_list,
        include_bias=True,
        include_shared=True,
    )
    
    result = grouped_matmul_finalize_routing_v3_wrapper(
        x1=data['x1'],
        x2=data['x2'],
        scale=data['scale'],
        pertoken_scale=data['pertoken_scale'],
        group_list=data['group_list'],
        row_index=data['row_index'],
        logit=data['logit'],
        batch=data['batch'],
        n=data['n'],
        bias=data['bias'],
        shared_input=data['shared_input'],
        shared_input_weight=data['shared_input_weight'],
        shared_input_offset=data['shared_input_offset'],
    )
    
    golden = grouped_matmul_finalize_routing_v3_golden(**data)
    
    print(f"  Output shape: {result.shape}")
    max_diff = np.abs(result.cpu().numpy() - golden.cpu().numpy()).max()
    print(f"  Max diff: {max_diff:.6e}")
    
    if run_mode == "npu":
        assert_allclose(
            result.cpu().numpy(),
            golden.cpu().numpy(),
            rtol=1e-3, atol=1e-3,
        )
    
    print("  ✓ Passed\n")
    print("[PRECISION_PASS]")


# ─────────────────────────────────────────────
# 4. CLI 入口
# ─────────────────────────────────────────────

EXAMPLES = {
    "GroupedMatmulFinalizeRoutingV3::test_mxfp8_basic": {
        "name": "MXFP8 Basic",
        "description": "MXFP8 基础配置验证",
        "function": test_mxfp8_basic,
    },
    "GroupedMatmulFinalizeRoutingV3::test_mxfp8_transpose": {
        "name": "MXFP8 Transpose",
        "description": "转置权重场景验证",
        "function": test_mxfp8_transpose,
    },
    "GroupedMatmulFinalizeRoutingV3::test_mxfp8_with_shared_input": {
        "name": "MXFP8 Shared Input",
        "description": "包含共享专家输出验证",
        "function": test_mxfp8_with_shared_input,
    },
    "GroupedMatmulFinalizeRoutingV3::test_mxfp8_with_bias": {
        "name": "MXFP8 Bias",
        "description": "包含 bias 场景验证",
        "function": test_mxfp8_with_bias,
    },
    "GroupedMatmulFinalizeRoutingV3::test_mxfp8_large_experts": {
        "name": "MXFP8 Large Experts",
        "description": "大专家数量场景验证",
        "function": test_mxfp8_large_experts,
    },
    "GroupedMatmulFinalizeRoutingV3::test_mxfp8_full_config": {
        "name": "MXFP8 Full Config",
        "description": "完整配置验证",
        "function": test_mxfp8_full_config,
    },
}


def main():
    parser = argparse.ArgumentParser(
        description="PyPTO GroupedMatmulFinalizeRoutingV3 operator test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s GroupedMatmulFinalizeRoutingV3::test_mxfp8_basic
  %(prog)s --list
        """,
    )
    parser.add_argument("example_id", type=str, nargs="?", help="Case ID to run")
    parser.add_argument("--list", action="store_true", help="List available cases")
    parser.add_argument(
        "--run_mode", "--run-mode",
        type=str, default="npu", choices=["npu", "sim"],
        help="Run mode (default: npu)",
    )
    args = parser.parse_args()
    
    # --list
    if args.list:
        print("\nAvailable cases:\n")
        for key, info in sorted(EXAMPLES.items()):
            print(f"  {key}  — {info['description']}")
        return
    
    # 选择用例
    if args.example_id:
        if args.example_id not in EXAMPLES:
            print(f"ERROR: unknown case '{args.example_id}'")
            print(f"Valid: {', '.join(sorted(EXAMPLES))}")
            sys.exit(1)
        to_run = [(args.example_id, EXAMPLES[args.example_id])]
    else:
        to_run = list(sorted(EXAMPLES.items()))
    
    # NPU 设备初始化
    device_id = None
    if args.run_mode == "npu":
        device_id = get_device_id()
        if device_id is None:
            sys.exit(1)
        torch.npu.set_device(device_id)
    
    # 执行测试
    try:
        for key, info in to_run:
            print(f"\n▸ Running {key}: {info['name']}")
            info["function"](device_id, args.run_mode)
        print("\n" + "=" * 60)
        print("All tests passed!")
        print("=" * 60)
    except AssertionError as e:
        print(f"\n[PRECISION_FAIL] {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\nRuntime error: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()