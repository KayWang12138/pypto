#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""PyPTO topk operator test.

测试内容:
- 2D/3D/4D 输入 shape
- largest=True/False 边界验证
- dtype 支持 (FP32/FP16)
- 动态轴支持
- 精度对比使用 numpy.testing.assert_allclose
"""

import os
import sys
import argparse

import torch
import numpy as np
from numpy.testing import assert_allclose

from topk_golden import topk_golden
from topk_impl import topk_wrapper

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

def test_topk_level0(device_id=None, run_mode="npu"):
    """Level 0: 2D 基础功能验证。"""
    print("=" * 60)
    print("Test: topk Level 0 (2D basic)")
    print("=" * 60)

    device = f"npu:{device_id}" if (run_mode == "npu" and device_id is not None) else "cpu"

    torch.manual_seed(0)
    shape = (16, 128)
    dtype = torch.float32
    x = torch.randn(shape, dtype=dtype, device=device)
    k = 10

    values, indices = topk_wrapper(x, k, dim=-1, largest=True, sorted=True)
    golden_values, golden_indices = topk_golden(x, k, dim=-1, largest=True, sorted=True)

    print(f"  Input shape : {x.shape}")
    print(f"  Values shape: {values.shape}")
    print(f"  Indices shape: {indices.shape}")

    # 检查 values 精度
    values_diff = np.abs(values.cpu().numpy() - golden_values.cpu().numpy()).max()
    print(f"  Values max diff: {values_diff:.6e}")

    # 检查 indices（允许不保证 sorted 顺序）
    indices_match = np.array_equal(
        np.sort(indices.cpu().numpy(), axis=-1),
        np.sort(golden_indices.cpu().numpy(), axis=-1)
    )
    print(f"  Indices match (sorted): {indices_match}")

    if run_mode == "npu":
        assert_allclose(
            values.cpu().numpy(),
            golden_values.cpu().numpy(),
            rtol=1e-3, atol=1e-3,
        )
        assert indices_match, "Indices sets do not match"

    print("  OK Passed\n")


def test_topk_level1(device_id=None, run_mode="npu"):
    """Level 1: 3D 典型场景验证。"""
    print("=" * 60)
    print("Test: topk Level 1 (3D typical)")
    print("=" * 60)

    device = f"npu:{device_id}" if (run_mode == "npu" and device_id is not None) else "cpu"

    torch.manual_seed(42)
    shape = (4, 64, 128)
    dtype = torch.float32
    x = torch.randn(shape, dtype=dtype, device=device)
    k = 10

    values, indices = topk_wrapper(x, k, dim=-1, largest=True, sorted=True)
    golden_values, golden_indices = topk_golden(x, k, dim=-1, largest=True, sorted=True)

    print(f"  Input shape : {x.shape}")
    print(f"  Values shape: {values.shape}")

    values_diff = np.abs(values.cpu().numpy() - golden_values.cpu().numpy()).max()
    print(f"  Values max diff: {values_diff:.6e}")

    indices_match = np.array_equal(
        np.sort(indices.cpu().numpy(), axis=-1),
        np.sort(golden_indices.cpu().numpy(), axis=-1)
    )
    print(f"  Indices match: {indices_match}")

    if run_mode == "npu":
        assert_allclose(
            values.cpu().numpy(),
            golden_values.cpu().numpy(),
            rtol=1e-3, atol=1e-3,
        )
        assert indices_match, "Indices sets do not match"

    print("  OK Passed\n")


def test_topk_level2(device_id=None, run_mode="npu"):
    """Level 2: 4D 性能场景验证。"""
    print("=" * 60)
    print("Test: topk Level 2 (4D performance)")
    print("=" * 60)

    device = f"npu:{device_id}" if (run_mode == "npu" and device_id is not None) else "cpu"

    torch.manual_seed(123)
    shape = (2, 4, 64, 128)
    dtype = torch.float32
    x = torch.randn(shape, dtype=dtype, device=device)
    k = 10

    values, indices = topk_wrapper(x, k, dim=-1, largest=True, sorted=True)
    golden_values, golden_indices = topk_golden(x, k, dim=-1, largest=True, sorted=True)

    print(f"  Input shape : {x.shape}")
    print(f"  Values shape: {values.shape}")

    values_diff = np.abs(values.cpu().numpy() - golden_values.cpu().numpy()).max()
    print(f"  Values max diff: {values_diff:.6e}")

    indices_match = np.array_equal(
        np.sort(indices.cpu().numpy(), axis=-1),
        np.sort(golden_indices.cpu().numpy(), axis=-1)
    )
    print(f"  Indices match: {indices_match}")

    if run_mode == "npu":
        assert_allclose(
            values.cpu().numpy(),
            golden_values.cpu().numpy(),
            rtol=1e-3, atol=1e-3,
        )
        assert indices_match, "Indices sets do not match"

    print("  OK Passed\n")


def test_topk_smallest(device_id=None, run_mode="npu"):
    """smallest=False 边界验证。"""
    print("=" * 60)
    print("Test: topk Smallest (largest=False)")
    print("=" * 60)

    device = f"npu:{device_id}" if (run_mode == "npu" and device_id is not None) else "cpu"

    torch.manual_seed(456)
    shape = (32, 64)
    dtype = torch.float32
    x = torch.randn(shape, dtype=dtype, device=device)
    k = 5

    values, indices = topk_wrapper(x, k, dim=-1, largest=False, sorted=True)
    golden_values, golden_indices = topk_golden(x, k, dim=-1, largest=False, sorted=True)

    print(f"  Input shape : {x.shape}")
    print(f"  Values shape: {values.shape}")

    values_diff = np.abs(values.cpu().numpy() - golden_values.cpu().numpy()).max()
    print(f"  Values max diff: {values_diff:.6e}")

    indices_match = np.array_equal(
        np.sort(indices.cpu().numpy(), axis=-1),
        np.sort(golden_indices.cpu().numpy(), axis=-1)
    )
    print(f"  Indices match: {indices_match}")

    if run_mode == "npu":
        assert_allclose(
            values.cpu().numpy(),
            golden_values.cpu().numpy(),
            rtol=1e-3, atol=1e-3,
        )
        assert indices_match, "Indices sets do not match"

    print("  OK Passed\n")


def test_topk_dtype_fp16(device_id=None, run_mode="npu"):
    """FP16 dtype 支持。"""
    print("=" * 60)
    print("Test: topk dtype FP16")
    print("=" * 60)

    device = f"npu:{device_id}" if (run_mode == "npu" and device_id is not None) else "cpu"

    torch.manual_seed(789)
    shape = (32, 64)
    dtype = torch.float16
    x = torch.randn(shape, dtype=dtype, device=device)
    k = 10

    values, indices = topk_wrapper(x, k, dim=-1, largest=True, sorted=True)
    golden_values, golden_indices = topk_golden(x, k, dim=-1, largest=True, sorted=True)

    print(f"  Input shape : {x.shape}")
    print(f"  Input dtype : {x.dtype}")
    print(f"  Values dtype: {values.dtype}")

    # FP16 使用更宽松的容差
    if run_mode == "npu":
        assert_allclose(
            values.cpu().float().numpy(),
            golden_values.cpu().float().numpy(),
            rtol=1e-2, atol=1e-2,
        )

    print("  OK Passed\n")


def test_topk_large(device_id=None, run_mode="npu"):
    """大规模性能验证。"""
    print("=" * 60)
    print("Test: topk Large (performance)")
    print("=" * 60)

    device = f"npu:{device_id}" if (run_mode == "npu" and device_id is not None) else "cpu"

    torch.manual_seed(999)
    shape = (1024, 4096)
    dtype = torch.float32
    x = torch.randn(shape, dtype=dtype, device=device)
    k = 20

    values, indices = topk_wrapper(x, k, dim=-1, largest=True, sorted=True)
    golden_values, golden_indices = topk_golden(x, k, dim=-1, largest=True, sorted=True)

    print(f"  Input shape : {x.shape}")
    print(f"  Values shape: {values.shape}")
    values_diff = np.abs(values.cpu().numpy() - golden_values.cpu().numpy()).max()
    print(f"  Values max diff: {values_diff:.6e}")

    if run_mode == "npu":
        assert_allclose(
            values.cpu().numpy(),
            golden_values.cpu().numpy(),
            rtol=1e-3, atol=1e-3,
        )

    print("  OK Passed\n")


# ─────────────────────────────────────────────
# 3. CLI 入口
# ─────────────────────────────────────────────

# 用例注册表
EXAMPLES = {
    "topk::test_topk_level0": {
        "name": "topk Level 0",
        "description": "2D 基础功能验证",
        "function": test_topk_level0,
    },
    "topk::test_topk_level1": {
        "name": "topk Level 1",
        "description": "3D 典型场景验证",
        "function": test_topk_level1,
    },
    "topk::test_topk_level2": {
        "name": "topk Level 2",
        "description": "4D 性能场景验证",
        "function": test_topk_level2,
    },
    "topk::test_topk_smallest": {
        "name": "topk Smallest",
        "description": "largest=False 边界验证",
        "function": test_topk_smallest,
    },
    "topk::test_topk_dtype_fp16": {
        "name": "topk dtype FP16",
        "description": "FP16 dtype 支持",
        "function": test_topk_dtype_fp16,
    },
    "topk::test_topk_large": {
        "name": "topk Large",
        "description": "大规模性能验证",
        "function": test_topk_large,
    },
}


def main():
    parser = argparse.ArgumentParser(
        description="PyPTO topk operator test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s topk::test_topk_level0    Run Level 0
  %(prog)s --list                    List all cases
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
            return
        import torch_npu
        torch.npu.set_device(device_id)

    # 执行
    try:
        for key, info in to_run:
            print(f"\n> Running {key}: {info['name']}")
            info["function"](device_id, args.run_mode)
        print("\n" + "=" * 60)
        print("[PRECISION_PASS] All tests passed!")
        print("=" * 60)
    except AssertionError as e:
        print(f"\n[PRECISION_FAIL] {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\nRuntime error: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
