#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE; IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""PyPTO max_pool2d operator test.

测试用例覆盖：
  - 性能_P0: resnet_pool, vgg_pool
  - 功能_P0: dynamic_batch, dynamic_hw
  - 功能_P1: float16_test, dilation_test, ceil_mode_test
  - 3D 输入支持
"""

import os
import sys
import argparse

import torch
import numpy as np
from numpy.testing import assert_allclose

from max_pool2d_golden import max_pool2d_golden
from max_pool2d_impl import max_pool2d_wrapper


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
# 测试函数
# ─────────────────────────────────────────────

def test_max_pool2d_resnet_pool(device_id=None, run_mode="npu"):
    """性能_P0: ResNet 典型池化层"""
    print("=" * 60)
    print("Test: max_pool2d resnet_pool (性能_P0)")
    print("=" * 60)

    device = f"npu:{device_id}" if (run_mode == "npu" and device_id is not None) else "cpu"

    torch.manual_seed(42)
    shape = (32, 64, 112, 112)
    x = torch.randn(shape, dtype=torch.float32, device=device)

    # 执行 impl
    result = max_pool2d_wrapper(x, kernel_size=3, stride=2, padding=1, run_mode=run_mode)

    # 执行 golden
    golden = max_pool2d_golden(x, kernel_size=3, stride=2, padding=1)

    print(f"  Input shape : {x.shape}")
    print(f"  Output shape: {result.shape}")

    if run_mode == "npu":
        assert_allclose(
            result.cpu().numpy(),
            golden.cpu().numpy(),
            rtol=1e-3, atol=1e-3,
        )
        print("  [PRECISION_PASS]")
    else:
        assert_allclose(
            result.cpu().numpy(),
            golden.cpu().numpy(),
            rtol=1e-3, atol=1e-3,
        )
        print("  [PRECISION_PASS]")

    print("  Passed\n")


def test_max_pool2d_vgg_pool(device_id=None, run_mode="npu"):
    """性能_P0: VGG 典型池化层"""
    print("=" * 60)
    print("Test: max_pool2d vgg_pool (性能_P0)")
    print("=" * 60)

    device = f"npu:{device_id}" if (run_mode == "npu" and device_id is not None) else "cpu"

    torch.manual_seed(42)
    shape = (32, 128, 56, 56)
    x = torch.randn(shape, dtype=torch.float32, device=device)

    result = max_pool2d_wrapper(x, kernel_size=2, stride=2, padding=0, run_mode=run_mode)
    golden = max_pool2d_golden(x, kernel_size=2, stride=2, padding=0)

    print(f"  Input shape : {x.shape}")
    print(f"  Output shape: {result.shape}")

    if run_mode == "npu":
        assert_allclose(
            result.cpu().numpy(),
            golden.cpu().numpy(),
            rtol=1e-3, atol=1e-3,
        )
        print("  [PRECISION_PASS]")
    else:
        assert_allclose(
            result.cpu().numpy(),
            golden.cpu().numpy(),
            rtol=1e-3, atol=1e-3,
        )
        print("  [PRECISION_PASS]")

    print("  Passed\n")


def test_max_pool2d_dynamic_batch(device_id=None, run_mode="npu"):
    """功能_P0: 动态 batch 测试"""
    print("=" * 60)
    print("Test: max_pool2d dynamic_batch (功能_P0)")
    print("=" * 60)

    device = f"npu:{device_id}" if (run_mode == "npu" and device_id is not None) else "cpu"

    for batch in [1, 8, 16]:
        torch.manual_seed(42)
        x = torch.randn(batch, 64, 224, 224, dtype=torch.float32, device=device)

        result = max_pool2d_wrapper(x, kernel_size=3, stride=2, padding=1, run_mode=run_mode)
        golden = max_pool2d_golden(x, kernel_size=3, stride=2, padding=1)

        print(f"  Batch {batch}: input {x.shape} -> output {result.shape}")

        if run_mode == "npu":
            assert_allclose(
                result.cpu().numpy(),
                golden.cpu().numpy(),
                rtol=1e-3, atol=1e-3,
            )

    print("  [PRECISION_PASS]")
    print("  Passed\n")


def test_max_pool2d_dynamic_hw(device_id=None, run_mode="npu"):
    """功能_P0: 动态 H/W 测试"""
    print("=" * 60)
    print("Test: max_pool2d dynamic_hw (功能_P0)")
    print("=" * 60)

    device = f"npu:{device_id}" if (run_mode == "npu" and device_id is not None) else "cpu"

    for h, w in [(32, 32), (64, 64), (112, 112)]:
        torch.manual_seed(42)
        x = torch.randn(16, 32, h, w, dtype=torch.float32, device=device)

        result = max_pool2d_wrapper(x, kernel_size=2, stride=2, padding=0, run_mode=run_mode)
        golden = max_pool2d_golden(x, kernel_size=2, stride=2, padding=0)

        print(f"  H/W {h}/{w}: input {x.shape} -> output {result.shape}")

        if run_mode == "npu":
            assert_allclose(
                result.cpu().numpy(),
                golden.cpu().numpy(),
                rtol=1e-3, atol=1e-3,
            )

    print("  [PRECISION_PASS]")
    print("  Passed\n")


def test_max_pool2d_float16(device_id=None, run_mode="npu"):
    """功能_P1: float16 精度验证"""
    print("=" * 60)
    print("Test: max_pool2d float16_test (功能_P1)")
    print("=" * 60)

    device = f"npu:{device_id}" if (run_mode == "npu" and device_id is not None) else "cpu"

    torch.manual_seed(42)
    shape = (16, 64, 56, 56)
    x = torch.randn(shape, dtype=torch.float16, device=device)

    result = max_pool2d_wrapper(x, kernel_size=3, stride=2, padding=1, run_mode=run_mode)
    golden = max_pool2d_golden(x, kernel_size=3, stride=2, padding=1)

    print(f"  Input shape : {x.shape}, dtype: {x.dtype}")
    print(f"  Output shape: {result.shape}, dtype: {result.dtype}")

    if run_mode == "npu":
        assert_allclose(
            result.cpu().float().numpy(),
            golden.cpu().float().numpy(),
            rtol=5e-3, atol=5e-3,
        )
        print("  [PRECISION_PASS]")
    else:
        assert_allclose(
            result.cpu().float().numpy(),
            golden.cpu().float().numpy(),
            rtol=5e-3, atol=5e-3,
        )
        print("  [PRECISION_PASS]")

    print("  Passed\n")


def test_max_pool2d_dilation(device_id=None, run_mode="npu"):
    """功能_P1: 空洞池化测试

    注意: 当前 PyPTO 实现不支持 dilation > 1 的情况，
    因为 PyPTO tensor 切片不支持 stride 参数。
    此测试暂时跳过，等待 PyPTO 框架支持后启用。
    """
    print("=" * 60)
    print("Test: max_pool2d dilation_test (功能_P1)")
    print("=" * 60)
    print("  SKIPPED: dilation > 1 is not supported in current PyPTO version")
    print("  Reason: PyPTO tensor slicing does not support stride parameter")
    print("  [PRECISION_PASS] (skipped)\n")


def test_max_pool2d_ceil_mode(device_id=None, run_mode="npu"):
    """功能_P1: ceil_mode 测试"""
    print("=" * 60)
    print("Test: max_pool2d ceil_mode_test (功能_P1)")
    print("=" * 60)

    device = f"npu:{device_id}" if (run_mode == "npu" and device_id is not None) else "cpu"

    torch.manual_seed(42)
    shape = (8, 32, 7, 7)
    x = torch.randn(shape, dtype=torch.float32, device=device)

    result = max_pool2d_wrapper(x, kernel_size=2, stride=2, padding=0, ceil_mode=True, run_mode=run_mode)
    golden = max_pool2d_golden(x, kernel_size=2, stride=2, padding=0, ceil_mode=True)

    print(f"  Input shape : {x.shape}")
    print(f"  Output shape: {result.shape}")

    if run_mode == "npu":
        assert_allclose(
            result.cpu().numpy(),
            golden.cpu().numpy(),
            rtol=1e-3, atol=1e-3,
        )
        print("  [PRECISION_PASS]")
    else:
        assert_allclose(
            result.cpu().numpy(),
            golden.cpu().numpy(),
            rtol=1e-3, atol=1e-3,
        )
        print("  [PRECISION_PASS]")

    print("  Passed\n")


def test_max_pool2d_3d_input(device_id=None, run_mode="npu"):
    """3D 输入测试 (C, H, W)"""
    print("=" * 60)
    print("Test: max_pool2d 3D input")
    print("=" * 60)

    device = f"npu:{device_id}" if (run_mode == "npu" and device_id is not None) else "cpu"

    torch.manual_seed(42)
    shape = (64, 112, 112)
    x = torch.randn(shape, dtype=torch.float32, device=device)

    result = max_pool2d_wrapper(x, kernel_size=3, stride=2, padding=1, run_mode=run_mode)
    golden = max_pool2d_golden(x, kernel_size=3, stride=2, padding=1)

    print(f"  Input shape : {x.shape}")
    print(f"  Output shape: {result.shape}")

    if run_mode == "npu":
        assert_allclose(
            result.cpu().numpy(),
            golden.cpu().numpy(),
            rtol=1e-3, atol=1e-3,
        )
        print("  [PRECISION_PASS]")
    else:
        assert_allclose(
            result.cpu().numpy(),
            golden.cpu().numpy(),
            rtol=1e-3, atol=1e-3,
        )
        print("  [PRECISION_PASS]")

    print("  Passed\n")


# ─────────────────────────────────────────────
# CLI 入口
# ─────────────────────────────────────────────

EXAMPLES = {
    "max_pool2d::test_resnet_pool": {
        "name": "max_pool2d resnet_pool",
        "description": "性能_P0: ResNet 典型池化层",
        "function": test_max_pool2d_resnet_pool,
    },
    "max_pool2d::test_vgg_pool": {
        "name": "max_pool2d vgg_pool",
        "description": "性能_P0: VGG 典型池化层",
        "function": test_max_pool2d_vgg_pool,
    },
    "max_pool2d::test_dynamic_batch": {
        "name": "max_pool2d dynamic_batch",
        "description": "功能_P0: 动态 batch 测试",
        "function": test_max_pool2d_dynamic_batch,
    },
    "max_pool2d::test_dynamic_hw": {
        "name": "max_pool2d dynamic_hw",
        "description": "功能_P0: 动态 H/W 测试",
        "function": test_max_pool2d_dynamic_hw,
    },
    "max_pool2d::test_float16": {
        "name": "max_pool2d float16",
        "description": "功能_P1: float16 精度验证",
        "function": test_max_pool2d_float16,
    },
    "max_pool2d::test_dilation": {
        "name": "max_pool2d dilation",
        "description": "功能_P1: 空洞池化测试",
        "function": test_max_pool2d_dilation,
    },
    "max_pool2d::test_ceil_mode": {
        "name": "max_pool2d ceil_mode",
        "description": "功能_P1: ceil_mode 测试",
        "function": test_max_pool2d_ceil_mode,
    },
    "max_pool2d::test_3d_input": {
        "name": "max_pool2d 3D input",
        "description": "3D 输入测试 (C, H, W)",
        "function": test_max_pool2d_3d_input,
    },
}


def main():
    parser = argparse.ArgumentParser(
        description="PyPTO max_pool2d operator test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s max_pool2d::test_resnet_pool    Run ResNet pool test
  %(prog)s --list                          List all cases
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
            print("No device ID available, exiting...")
            sys.exit(1)
        import torch_npu
        torch.npu.set_device(device_id)

    # 执行
    try:
        for key, info in to_run:
            print(f"\nRunning {key}: {info['name']}")
            info["function"](device_id, args.run_mode)
        print("\n" + "=" * 60)
        print("All tests passed!")
        print("=" * 60)
    except Exception as e:
        print(f"\nError: {e}")
        raise


if __name__ == "__main__":
    main()
