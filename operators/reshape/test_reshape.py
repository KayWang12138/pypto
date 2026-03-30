#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
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
"""PyPTO reshape operator test.

测试 reshape 算子的实现，与 golden 参考实现进行精度对比。

运行方式:
    python test_reshape.py                          # 运行所有测试
    python test_reshape.py --list                   # 列出所有测试用例
    python test_reshape.py reshape::test_reshape_level0  # 运行特定测试
    python test_reshape.py --run_mode sim           # 使用模拟器模式
"""

import os
import sys
import argparse

import torch
import numpy as np
from numpy.testing import assert_allclose

from reshape_golden import reshape_golden
from reshape_impl import reshape_wrapper, reshape_dynamic_wrapper


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

def test_reshape_level0(device_id=None, run_mode="npu"):
    """Level 0: 小数据量基础功能验证（24 元素）。"""
    print("=" * 60)
    print("Test: reshape Level 0 (basic)")
    print("=" * 60)

    device = f"npu:{device_id}" if (run_mode == "npu" and device_id is not None) else "cpu"

    # 测试数据: [2, 3, 4] -> [2, 12]
    torch.manual_seed(0)
    x = torch.randn(2, 3, 4, dtype=torch.float32, device=device)
    target_shape = [2, 12]

    # 执行 kernel wrapper
    result = reshape_wrapper(x, target_shape)

    # 执行 golden
    golden = reshape_golden(x, target_shape)

    # 精度对比
    print(f"  Input shape : {list(x.shape)}")
    print(f"  Target shape: {target_shape}")
    print(f"  Output shape: {list(result.shape)}")

    # 转换为 numpy 进行对比（注意 bfloat16 需要先转 float）
    result_np = result.cpu().float().numpy() if result.dtype == torch.bfloat16 else result.cpu().numpy()
    golden_np = golden.cpu().float().numpy() if golden.dtype == torch.bfloat16 else golden.cpu().numpy()

    max_diff = np.abs(result_np - golden_np).max()
    print(f"  Max diff    : {max_diff:.6e}")

    if run_mode == "npu":
        try:
            assert_allclose(
                result_np,
                golden_np,
                rtol=1e-3, atol=1e-3,
            )
            print("  [PRECISION_PASS]")
        except AssertionError as e:
            print(f"  [PRECISION_FAIL] {e}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"  Runtime error: {e}", file=sys.stderr)
            sys.exit(2)

    print("  PASSED\n")


def test_reshape_level1(device_id=None, run_mode="npu"):
    """Level 1: 典型场景验证（1K+ 元素）。"""
    print("=" * 60)
    print("Test: reshape Level 1 (typical)")
    print("=" * 60)

    device = f"npu:{device_id}" if (run_mode == "npu" and device_id is not None) else "cpu"

    # 测试数据: [4, 256, 64] -> [4, 16384]
    torch.manual_seed(42)
    x = torch.randn(4, 256, 64, dtype=torch.float32, device=device)
    target_shape = [4, 16384]

    result = reshape_wrapper(x, target_shape)
    golden = reshape_golden(x, target_shape)

    result_np = result.cpu().float().numpy() if result.dtype == torch.bfloat16 else result.cpu().numpy()
    golden_np = golden.cpu().float().numpy() if golden.dtype == torch.bfloat16 else golden.cpu().numpy()

    max_diff = np.abs(result_np - golden_np).max()
    print(f"  Input shape : {list(x.shape)}")
    print(f"  Target shape: {target_shape}")
    print(f"  Max diff    : {max_diff:.6e}")

    if run_mode == "npu":
        try:
            assert_allclose(
                result_np,
                golden_np,
                rtol=1e-3, atol=1e-3,
            )
            print("  [PRECISION_PASS]")
        except AssertionError as e:
            print(f"  [PRECISION_FAIL] {e}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"  Runtime error: {e}", file=sys.stderr)
            sys.exit(2)

    print("  PASSED\n")


def test_reshape_negative_dim(device_id=None, run_mode="npu"):
    """测试负维度自动推断（-1）。"""
    print("=" * 60)
    print("Test: reshape negative dimension (-1)")
    print("=" * 60)

    device = f"npu:{device_id}" if (run_mode == "npu" and device_id is not None) else "cpu"

    # 测试数据: [2, 3, 4] -> [2, -1] => [2, 12]
    torch.manual_seed(100)
    x = torch.randn(2, 3, 4, dtype=torch.float32, device=device)
    target_shape = [2, -1]

    result = reshape_wrapper(x, target_shape)
    golden = reshape_golden(x, target_shape)

    result_np = result.cpu().float().numpy() if result.dtype == torch.bfloat16 else result.cpu().numpy()
    golden_np = golden.cpu().float().numpy() if golden.dtype == torch.bfloat16 else golden.cpu().numpy()

    max_diff = np.abs(result_np - golden_np).max()
    print(f"  Input shape : {list(x.shape)}")
    print(f"  Target shape: {target_shape} => {list(result.shape)}")
    print(f"  Max diff    : {max_diff:.6e}")

    if run_mode == "npu":
        try:
            assert_allclose(
                result_np,
                golden_np,
                rtol=1e-3, atol=1e-3,
            )
            print("  [PRECISION_PASS]")
        except AssertionError as e:
            print(f"  [PRECISION_FAIL] {e}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"  Runtime error: {e}", file=sys.stderr)
            sys.exit(2)

    print("  PASSED\n")


def test_reshape_flatten(device_id=None, run_mode="npu"):
    """测试展平为一维（flatten）。"""
    print("=" * 60)
    print("Test: reshape flatten")
    print("=" * 60)

    device = f"npu:{device_id}" if (run_mode == "npu" and device_id is not None) else "cpu"

    # 测试数据: [2, 3, 4] -> [-1] => [24]
    torch.manual_seed(200)
    x = torch.randn(2, 3, 4, dtype=torch.float32, device=device)
    target_shape = [-1]

    result = reshape_wrapper(x, target_shape)
    golden = reshape_golden(x, target_shape)

    result_np = result.cpu().float().numpy() if result.dtype == torch.bfloat16 else result.cpu().numpy()
    golden_np = golden.cpu().float().numpy() if golden.dtype == torch.bfloat16 else golden.cpu().numpy()

    max_diff = np.abs(result_np - golden_np).max()
    print(f"  Input shape : {list(x.shape)}")
    print(f"  Target shape: {target_shape} => {list(result.shape)}")
    print(f"  Max diff    : {max_diff:.6e}")

    if run_mode == "npu":
        try:
            assert_allclose(
                result_np,
                golden_np,
                rtol=1e-3, atol=1e-3,
            )
            print("  [PRECISION_PASS]")
        except AssertionError as e:
            print(f"  [PRECISION_FAIL] {e}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"  Runtime error: {e}", file=sys.stderr)
            sys.exit(2)

    print("  PASSED\n")


def test_reshape_bfloat16(device_id=None, run_mode="npu"):
    """测试 bfloat16 数据类型。"""
    print("=" * 60)
    print("Test: reshape bfloat16 dtype")
    print("=" * 60)

    device = f"npu:{device_id}" if (run_mode == "npu" and device_id is not None) else "cpu"

    # 测试数据: [4, 64, 64] -> [4, 4096]
    torch.manual_seed(300)
    x = torch.randn(4, 64, 64, dtype=torch.bfloat16, device=device)
    target_shape = [4, 4096]

    result = reshape_wrapper(x, target_shape)
    golden = reshape_golden(x, target_shape)

    # bfloat16 需要先转 float 再转 numpy
    result_np = result.cpu().float().numpy()
    golden_np = golden.cpu().float().numpy()

    max_diff = np.abs(result_np - golden_np).max()
    print(f"  Input shape : {list(x.shape)}")
    print(f"  Input dtype : {x.dtype}")
    print(f"  Max diff    : {max_diff:.6e}")

    if run_mode == "npu":
        try:
            # bfloat16 使用更大的容差
            assert_allclose(
                result_np,
                golden_np,
                rtol=0.01, atol=0.01,
            )
            print("  [PRECISION_PASS]")
        except AssertionError as e:
            print(f"  [PRECISION_FAIL] {e}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"  Runtime error: {e}", file=sys.stderr)
            sys.exit(2)

    print("  PASSED\n")


def test_reshape_float16(device_id=None, run_mode="npu"):
    """测试 float16 数据类型。"""
    print("=" * 60)
    print("Test: reshape float16 dtype")
    print("=" * 60)

    device = f"npu:{device_id}" if (run_mode == "npu" and device_id is not None) else "cpu"

    # 测试数据: [2, 128, 32] -> [2, 4096]
    torch.manual_seed(400)
    x = torch.randn(2, 128, 32, dtype=torch.float16, device=device)
    target_shape = [2, 4096]

    result = reshape_wrapper(x, target_shape)
    golden = reshape_golden(x, target_shape)

    # float16 需要先转 float 再转 numpy
    result_np = result.cpu().float().numpy()
    golden_np = golden.cpu().float().numpy()

    max_diff = np.abs(result_np - golden_np).max()
    print(f"  Input shape : {list(x.shape)}")
    print(f"  Input dtype : {x.dtype}")
    print(f"  Max diff    : {max_diff:.6e}")

    if run_mode == "npu":
        try:
            # float16 使用更大的容差
            assert_allclose(
                result_np,
                golden_np,
                rtol=0.01, atol=0.01,
            )
            print("  [PRECISION_PASS]")
        except AssertionError as e:
            print(f"  [PRECISION_FAIL] {e}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"  Runtime error: {e}", file=sys.stderr)
            sys.exit(2)

    print("  PASSED\n")


def test_reshape_dynamic(device_id=None, run_mode="npu"):
    """测试动态 shape 场景。"""
    print("=" * 60)
    print("Test: reshape dynamic shape")
    print("=" * 60)

    device = f"npu:{device_id}" if (run_mode == "npu" and device_id is not None) else "cpu"

    # 测试数据: 模拟动态 batch 和 seq_len
    # [b, s1, 64] -> [b, s2, 8]
    b, s1, s2 = 4, 16, 128
    torch.manual_seed(500)
    x = torch.randn(b, s1, 64, dtype=torch.float32, device=device)
    target_shape = [b, s2, 8]

    # 使用动态 wrapper（第 0 维是动态的 batch）
    result = reshape_dynamic_wrapper(x, target_shape, dynamic_axes=[0])
    golden = reshape_golden(x, target_shape)

    result_np = result.cpu().float().numpy() if result.dtype == torch.bfloat16 else result.cpu().numpy()
    golden_np = golden.cpu().float().numpy() if golden.dtype == torch.bfloat16 else golden.cpu().numpy()

    max_diff = np.abs(result_np - golden_np).max()
    print(f"  Input shape : {list(x.shape)}")
    print(f"  Target shape: {target_shape}")
    print(f"  Dynamic axes: [0] (batch)")
    print(f"  Max diff    : {max_diff:.6e}")

    if run_mode == "npu":
        try:
            assert_allclose(
                result_np,
                golden_np,
                rtol=1e-3, atol=1e-3,
            )
            print("  [PRECISION_PASS]")
        except AssertionError as e:
            print(f"  [PRECISION_FAIL] {e}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"  Runtime error: {e}", file=sys.stderr)
            sys.exit(2)

    print("  PASSED\n")


def test_reshape_boundary(device_id=None, run_mode="npu"):
    """测试边界情况。"""
    print("=" * 60)
    print("Test: reshape boundary cases")
    print("=" * 60)

    device = f"npu:{device_id}" if (run_mode == "npu" and device_id is not None) else "cpu"

    # 边界测试 1: 最小 shape
    torch.manual_seed(600)
    x = torch.randn(1, 1, 64, dtype=torch.float32, device=device)
    target_shape = [1, 8, 8]

    result = reshape_wrapper(x, target_shape)
    golden = reshape_golden(x, target_shape)

    result_np = result.cpu().float().numpy() if result.dtype == torch.bfloat16 else result.cpu().numpy()
    golden_np = golden.cpu().float().numpy() if golden.dtype == torch.bfloat16 else golden.cpu().numpy()

    max_diff = np.abs(result_np - golden_np).max()
    print(f"  Case 1 (min shape): [1, 1, 64] -> [1, 8, 8], Max diff: {max_diff:.6e}")

    if run_mode == "npu":
        try:
            assert_allclose(result_np, golden_np, rtol=1e-3, atol=1e-3)
            print("  [PRECISION_PASS]")
        except AssertionError as e:
            print(f"  [PRECISION_FAIL] {e}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"  Runtime error: {e}", file=sys.stderr)
            sys.exit(2)

    # 边界测试 2: 大 shape
    x = torch.randn(32, 128, 32, dtype=torch.float32, device=device)
    target_shape = [32, 4096]

    result = reshape_wrapper(x, target_shape)
    golden = reshape_golden(x, target_shape)

    result_np = result.cpu().float().numpy() if result.dtype == torch.bfloat16 else result.cpu().numpy()
    golden_np = golden.cpu().float().numpy() if golden.dtype == torch.bfloat16 else golden.cpu().numpy()

    max_diff = np.abs(result_np - golden_np).max()
    print(f"  Case 2 (large shape): [32, 128, 32] -> [32, 4096], Max diff: {max_diff:.6e}")

    if run_mode == "npu":
        try:
            assert_allclose(result_np, golden_np, rtol=1e-3, atol=1e-3)
            print("  [PRECISION_PASS]")
        except AssertionError as e:
            print(f"  [PRECISION_FAIL] {e}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"  Runtime error: {e}", file=sys.stderr)
            sys.exit(2)

    print("  PASSED\n")


# ─────────────────────────────────────────────
# 3. CLI 入口
# ─────────────────────────────────────────────

# 用例注册表
EXAMPLES = {
    "reshape::test_reshape_level0": {
        "name": "reshape Level 0",
        "description": "小数据量基础功能验证",
        "function": test_reshape_level0,
    },
    "reshape::test_reshape_level1": {
        "name": "reshape Level 1",
        "description": "典型场景验证",
        "function": test_reshape_level1,
    },
    "reshape::test_reshape_negative_dim": {
        "name": "reshape negative dimension",
        "description": "负维度自动推断测试",
        "function": test_reshape_negative_dim,
    },
    "reshape::test_reshape_flatten": {
        "name": "reshape flatten",
        "description": "展平为一维测试",
        "function": test_reshape_flatten,
    },
    "reshape::test_reshape_bfloat16": {
        "name": "reshape bfloat16",
        "description": "bfloat16 数据类型测试",
        "function": test_reshape_bfloat16,
    },
    "reshape::test_reshape_float16": {
        "name": "reshape float16",
        "description": "float16 数据类型测试",
        "function": test_reshape_float16,
    },
    "reshape::test_reshape_dynamic": {
        "name": "reshape dynamic",
        "description": "动态 shape 测试",
        "function": test_reshape_dynamic,
    },
    "reshape::test_reshape_boundary": {
        "name": "reshape boundary",
        "description": "边界情况测试",
        "function": test_reshape_boundary,
    },
}


def main():
    parser = argparse.ArgumentParser(
        description="PyPTO reshape operator test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s reshape::test_reshape_level0    Run Level 0
  %(prog)s --list                          List all cases
  %(prog)s --run_mode sim                  Run in simulator mode
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
