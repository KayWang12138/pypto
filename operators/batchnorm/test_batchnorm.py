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
"""PyPTO BatchNorm operator test.

测试说明:
  - 测试 batchnorm_wrapper 与 batchnorm_golden 的精度对比
  - 支持 3D 和 5D 输入
  - 使用 numpy.testing.assert_allclose 进行精度对比
  - 支持 NPU 和 sim 模式
"""

import os
import sys
import argparse
import torch
import numpy as np
from numpy.testing import assert_allclose

from batchnorm_golden import batchnorm_golden
from batchnorm_impl import batchnorm_wrapper


# -------------------------------------------------------------------------
# 1. 环境工具
# -------------------------------------------------------------------------

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


# -------------------------------------------------------------------------
# 2. 测试函数
# -------------------------------------------------------------------------

def test_batchnorm_3d(device_id=None, run_mode="npu"):
    """测试 3D 输入 [batch, seq_len, channels]。"""
    print("=" * 60)
    print("Test: BatchNorm 3D (BatchNorm1d)")
    print("=" * 60)

    device = f"npu:{device_id}" if (run_mode == "npu" and device_id is not None) else "cpu"

    # 测试数据
    torch.manual_seed(42)
    shape = (2, 512, 768)
    channels = 768
    dtype = torch.float32

    x = torch.randn(shape, dtype=dtype, device=device)
    gamma = torch.ones(channels, dtype=dtype, device=device)
    beta = torch.zeros(channels, dtype=dtype, device=device)

    print(f"  Input shape : {x.shape}")
    print(f"  Channels   : {channels}")

    # 执行 kernel wrapper
    result = batchnorm_wrapper(x, gamma, beta, eps=1e-5)

    # 执行 golden
    golden = batchnorm_golden(x, gamma, beta, eps=1e-5)

    # 精度对比
    print(f"  Output shape: {result.shape}")

    max_diff = np.abs(result.cpu().numpy() - golden.cpu().numpy()).max()
    print(f"  Max diff    : {max_diff:.6e}")

    if run_mode == "npu":
        try:
            assert_allclose(
                result.cpu().numpy(),
                golden.cpu().numpy(),
                rtol=1e-3,
                atol=1e-3,
            )
            print("[PRECISION_PASS]")
        except AssertionError as e:
            print(f"[PRECISION_FAIL] {e}", file=sys.stderr)
            sys.exit(1)
    else:
        print("  [SIM mode - skipping precision check")

    print("  Passed\n")


def test_batchnorm_5d(device_id=None, run_mode="npu"):
    """测试 5D 输入 [batch, seq_len, channels, H, W] - ResNet 典型配置。"""
    print("=" * 60)
    print("Test: BatchNorm 5D (ResNet typical)")
    print("=" * 60)

    device = f"npu:{device_id}" if (run_mode == "npu" and device_id is not None) else "cpu"

    # 测试数据
    torch.manual_seed(42)
    shape = (2, 32, 64, 28, 28)
    channels = 64
    dtype = torch.float32

    x = torch.randn(shape, dtype=dtype, device=device)
    gamma = torch.ones(channels, dtype=dtype, device=device)
    beta = torch.zeros(channels, dtype=dtype, device=device)

    print(f"  Input shape : {x.shape}")
    print(f"  Channels   : {channels}")

    # 执行 kernel wrapper
    result = batchnorm_wrapper(x, gamma, beta, eps=1e-5)

    # 执行 golden
    golden = batchnorm_golden(x, gamma, beta, eps=1e-5)

    # 精度对比
    print(f"  Output shape: {result.shape}")

    max_diff = np.abs(result.cpu().numpy() - golden.cpu().numpy()).max()
    print(f"  Max diff    : {max_diff:.6e}")

    if run_mode == "npu":
        try:
            assert_allclose(
                result.cpu().numpy(),
                golden.cpu().numpy(),
                rtol=1e-3,
                atol=1e-3,
            )
            print("[PRECISION_PASS]")
        except AssertionError as e:
            print(f"[PRECISION_FAIL] {e}", file=sys.stderr)
            sys.exit(1)
    else:
        print("  [SIM mode - skipping precision check")

    print("  Passed\n")


def test_batchnorm_bf16(device_id=None, run_mode="npu"):
    """测试 BF16 数据类型。"""
    print("=" * 60)
    print("Test: BatchNorm BF16 dtype")
    print("=" * 60)

    device = f"npu:{device_id}" if (run_mode == "npu" and device_id is not None) else "cpu"

    # 测试数据
    torch.manual_seed(42)
    shape = (2, 16, 64, 14, 14)
    channels = 64
    dtype = torch.bfloat16

    x = torch.randn(shape, dtype=dtype, device=device)
    gamma = torch.ones(channels, dtype=dtype, device=device)
    beta = torch.zeros(channels, dtype=dtype, device=device)

    print(f"  Input shape : {x.shape}")
    print(f"  Dtype      : {dtype}")

    # 执行 kernel wrapper
    result = batchnorm_wrapper(x, gamma, beta, eps=1e-5)

    # 执行 golden
    golden = batchnorm_golden(x, gamma, beta, eps=1e-5)

    # 精度对比
    print(f"  Output shape: {result.shape}")

    max_diff = np.abs(result.cpu().float().numpy() - golden.cpu().float().numpy()).max()
    print(f"  Max diff    : {max_diff:.6e}")

    if run_mode == "npu":
        try:
            assert_allclose(
                result.cpu().float().numpy(),
                golden.cpu().float().numpy(),
                rtol=1e-2,
                atol=1e-2,
            )
            print("[PRECISION_PASS]")
        except AssertionError as e:
            print(f"[PRECISION_FAIL] {e}", file=sys.stderr)
            sys.exit(1)
    else:
        print("  [SIM mode - skipping precision check")

    print("  Passed\n")


def test_batchnorm_large(device_id=None, run_mode="npu"):
    """测试大 batch 场景 - 性能测试配置。"""
    print("=" * 60)
    print("Test: BatchNorm large batch (performance)")
    print("=" * 60)

    device = f"npu:{device_id}" if (run_mode == "npu" and device_id is not None) else "cpu"

    # 测试数据
    torch.manual_seed(42)
    shape = (8, 64, 256, 56, 56)
    channels = 256
    dtype = torch.float32

    x = torch.randn(shape, dtype=dtype, device=device)
    gamma = torch.ones(channels, dtype=dtype, device=device)
    beta = torch.zeros(channels, dtype=dtype, device=device)

    print(f"  Input shape : {x.shape}")
    print(f"  Channels   : {channels}")

    # 执行 kernel wrapper
    result = batchnorm_wrapper(x, gamma, beta, eps=1e-5)

    # 执行 golden
    golden = batchnorm_golden(x, gamma, beta, eps=1e-5)

    # 精度对比
    print(f"  Output shape: {result.shape}")

    max_diff = np.abs(result.cpu().numpy() - golden.cpu().numpy()).max()
    print(f"  Max diff    : {max_diff:.6e}")

    if run_mode == "npu":
        try:
            assert_allclose(
                result.cpu().numpy(),
                golden.cpu().numpy(),
                rtol=1e-3,
                atol=1e-3,
            )
            print("[PRECISION_PASS]")
        except AssertionError as e:
            print(f"[PRECISION_FAIL] {e}", file=sys.stderr)
            sys.exit(1)
    else:
        print("  [SIM mode - skipping precision check")

    print("  Passed\n")


# -------------------------------------------------------------------------
# 3. CLI 入口
# -------------------------------------------------------------------------

# 用例注册表
EXAMPLES = {
    "batchnorm::test_batchnorm_3d": {
        "name": "BatchNorm 3D",
        "description": "3D input [batch, seq_len, channels]",
        "function": test_batchnorm_3d,
    },
    "batchnorm::test_batchnorm_5d": {
        "name": "BatchNorm 5D",
        "description": "5D input [batch, seq_len, channels, H, W] - ResNet typical",
        "function": test_batchnorm_5d,
    },
    "batchnorm::test_batchnorm_bf16": {
        "name": "BatchNorm BF16",
        "description": "BF16 dtype support",
        "function": test_batchnorm_bf16,
    },
    "batchnorm::test_batchnorm_large": {
        "name": "BatchNorm large",
        "description": "Large batch performance test",
        "function": test_batchnorm_large,
    },
}


def main():
    parser = argparse.ArgumentParser(
        description="PyPTO BatchNorm operator test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
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
            print(f"  {key}  - {info['description']}")
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
            return
        import torch_npu
        torch.npu.set_device(device_id)

    # 执行
    try:
        for key, info in to_run:
            print(f"\n> Running {key}: {info['name']}")
            info["function"](device_id, args.run_mode)
        print("\n" + "=" * 60)
        print("All tests passed!")
        print("=" * 60)
    except Exception as e:
        print(f"\nError: {e}")
        raise


if __name__ == "__main__":
    main()
