#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# -----------------------------------------------------------------------------------------------------------
"""
adaptive_avg_pool2d 算子测试

测试用例覆盖 spec.md 中的典型配置。
"""

import os
import sys
import argparse

import torch
import numpy as np
from numpy.testing import assert_allclose

from adaptive_avg_pool2d_golden import adaptive_avg_pool2d_golden
from adaptive_avg_pool2d_impl import adaptive_avg_pool2d_wrapper


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
# 2. 测试函数
# ─────────────────────────────────────────────

def test_adaptive_avg_pool2d_level0(device_id=None, run_mode="npu"):
    """Level 0: 小数据量基础功能验证。"""
    print("=" * 60)
    print("Test: adaptive_avg_pool2d Level 0 (basic)")
    print("=" * 60)

    device = f"npu:{device_id}" if (run_mode == "npu" and device_id is not None) else "cpu"

    # 测试数据
    torch.manual_seed(0)
    shape = (2, 4, 8, 8)
    output_size = (4, 4)
    dtype = torch.float32
    x = torch.randn(shape, dtype=dtype, device=device)

    # 执行 kernel wrapper
    result = adaptive_avg_pool2d_wrapper(x, output_size)

    # 执行 golden
    golden = adaptive_avg_pool2d_golden(x, output_size)

    # 精度对比
    print(f"  Input shape : {x.shape}")
    print(f"  Output shape: {result.shape}")
    print(f"  Output size : {output_size}")
    max_diff = np.abs(result.cpu().numpy() - golden.cpu().numpy()).max()
    print(f"  Max diff    : {max_diff:.6e}")

    if run_mode == "npu":
        assert_allclose(
            result.cpu().numpy(),
            golden.cpu().numpy(),
            rtol=1e-3, atol=1e-3,
        )

    print("  ✓ PASS")
    print("[PRECISION_PASS]")


def test_adaptive_avg_pool2d_perf_p0(device_id=None, run_mode="npu"):
    """性能_P0: 典型 CNN 特征图下采样 [16, 256, 14, 14] -> [16, 256, 7, 7]"""
    print("=" * 60)
    print("Test: adaptive_avg_pool2d 性能_P0")
    print("=" * 60)

    device = f"npu:{device_id}" if (run_mode == "npu" and device_id is not None) else "cpu"

    torch.manual_seed(42)
    shape = (16, 256, 14, 14)
    output_size = (7, 7)
    dtype = torch.float32
    x = torch.randn(shape, dtype=dtype, device=device)

    result = adaptive_avg_pool2d_wrapper(x, output_size)
    golden = adaptive_avg_pool2d_golden(x, output_size)

    print(f"  Shape: {shape} -> {result.shape}")
    max_diff = np.abs(result.cpu().numpy() - golden.cpu().numpy()).max()
    print(f"  Max diff: {max_diff:.6e}")

    if run_mode == "npu":
        assert_allclose(
            result.cpu().numpy(),
            golden.cpu().numpy(),
            rtol=1e-3, atol=1e-3,
        )

    print("  ✓ PASS")


def test_adaptive_avg_pool2d_func_p0(device_id=None, run_mode="npu"):
    """功能_P0: 全局平均池化 [8, 512, 7, 7] -> [8, 512, 1, 1]"""
    print("=" * 60)
    print("Test: adaptive_avg_pool2d 功能_P0 (global pooling)")
    print("=" * 60)

    device = f"npu:{device_id}" if (run_mode == "npu" and device_id is not None) else "cpu"

    torch.manual_seed(123)
    shape = (8, 512, 7, 7)
    output_size = (1, 1)
    dtype = torch.float32
    x = torch.randn(shape, dtype=dtype, device=device)

    result = adaptive_avg_pool2d_wrapper(x, output_size)
    golden = adaptive_avg_pool2d_golden(x, output_size)

    print(f"  Shape: {shape} -> {result.shape}")
    max_diff = np.abs(result.cpu().numpy() - golden.cpu().numpy()).max()
    print(f"  Max diff: {max_diff:.6e}")

    if run_mode == "npu":
        assert_allclose(
            result.cpu().numpy(),
            golden.cpu().numpy(),
            rtol=1e-3, atol=1e-3,
        )

    print("  ✓ PASS")


def test_adaptive_avg_pool2d_single_value(device_id=None, run_mode="npu"):
    """单值尺寸_P1: output_size=8"""
    print("=" * 60)
    print("Test: adaptive_avg_pool2d 单值尺寸_P1")
    print("=" * 60)

    device = f"npu:{device_id}" if (run_mode == "npu" and device_id is not None) else "cpu"

    torch.manual_seed(456)
    shape = (4, 128, 16, 16)
    output_size = 8  # 单值
    dtype = torch.float32
    x = torch.randn(shape, dtype=dtype, device=device)

    result = adaptive_avg_pool2d_wrapper(x, output_size)
    golden = adaptive_avg_pool2d_golden(x, output_size)

    print(f"  Shape: {shape} -> {result.shape}")
    print(f"  Output size: {output_size} (single value)")
    max_diff = np.abs(result.cpu().numpy() - golden.cpu().numpy()).max()
    print(f"  Max diff: {max_diff:.6e}")

    if run_mode == "npu":
        assert_allclose(
            result.cpu().numpy(),
            golden.cpu().numpy(),
            rtol=1e-3, atol=1e-3,
        )

    print("  ✓ PASS")


def test_adaptive_avg_pool2d_non_aligned(device_id=None, run_mode="npu"):
    """非对齐_P1: 7x7 -> 5x5"""
    print("=" * 60)
    print("Test: adaptive_avg_pool2d 非对齐_P1")
    print("=" * 60)

    device = f"npu:{device_id}" if (run_mode == "npu" and device_id is not None) else "cpu"

    torch.manual_seed(789)
    shape = (2, 64, 7, 7)
    output_size = (5, 5)
    dtype = torch.float32
    x = torch.randn(shape, dtype=dtype, device=device)

    result = adaptive_avg_pool2d_wrapper(x, output_size)
    golden = adaptive_avg_pool2d_golden(x, output_size)

    print(f"  Shape: {shape} -> {result.shape}")
    max_diff = np.abs(result.cpu().numpy() - golden.cpu().numpy()).max()
    print(f"  Max diff: {max_diff:.6e}")

    if run_mode == "npu":
        assert_allclose(
            result.cpu().numpy(),
            golden.cpu().numpy(),
            rtol=1e-3, atol=1e-3,
        )

    print("  ✓ PASS")


# ─────────────────────────────────────────────
# 3. CLI 入口
# ─────────────────────────────────────────────

EXAMPLES = {
    "adaptive_avg_pool2d::test_level0": {
        "name": "Level 0 (basic)",
        "description": "小数据量基础功能验证 [2,4,8,8] -> [2,4,4,4]",
        "function": test_adaptive_avg_pool2d_level0,
    },
    "adaptive_avg_pool2d::test_perf_p0": {
        "name": "性能_P0",
        "description": "典型 CNN 特征图下采样 [16,256,14,14] -> [16,256,7,7]",
        "function": test_adaptive_avg_pool2d_perf_p0,
    },
    "adaptive_avg_pool2d::test_func_p0": {
        "name": "功能_P0",
        "description": "全局平均池化 [8,512,7,7] -> [8,512,1,1]",
        "function": test_adaptive_avg_pool2d_func_p0,
    },
    "adaptive_avg_pool2d::test_single_value": {
        "name": "单值尺寸_P1",
        "description": "单值 output_size=8",
        "function": test_adaptive_avg_pool2d_single_value,
    },
    "adaptive_avg_pool2d::test_non_aligned": {
        "name": "非对齐_P1",
        "description": "非对齐窗口 7x7 -> 5x5",
        "function": test_adaptive_avg_pool2d_non_aligned,
    },
}


def main():
    parser = argparse.ArgumentParser(
        description="PyPTO adaptive_avg_pool2d operator test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s adaptive_avg_pool2d::test_level0    Run Level 0
  %(prog)s --list                              List all cases
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
            return
        import torch_npu
        torch.npu.set_device(device_id)

    # 执行
    passed = 0
    failed = 0
    try:
        for key, info in to_run:
            print(f"\n▸ Running {key}: {info['name']}")
            try:
                info["function"](device_id, args.run_mode)
                passed += 1
            except Exception as e:
                print(f"  ✗ FAIL: {e}")
                failed += 1

        print("\n" + "=" * 60)
        print(f"Results: {passed} passed, {failed} failed")
        print("=" * 60)

        if failed > 0:
            sys.exit(1)

    except Exception as e:
        print(f"\nError: {e}")
        raise


if __name__ == "__main__":
    main()
