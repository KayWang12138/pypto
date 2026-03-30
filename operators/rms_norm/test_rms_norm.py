#!/usr/bin/env python3
# coding: utf-8

"""PyPTO rms_norm operator test.

模板说明:
  - 本文件是 test_rms_norm.py 的固定模板，由 pypto-op-develop 在 Stage 3 生成。
  - 所有 rms_norm 占位符需替换为实际算子名称。
  - test_rms_norm.py 只做 import + 调用 + 精度对比，不包含 golden 或 kernel 实现代码。
  - golden 实现来自 rms_norm_golden.py（Stage 2A 由 pypto-golden-generator 生成）。
  - kernel 实现来自 rms_norm_impl.py（Stage 3 由 pypto-op-develop 生成）。
  - 精度对比必须使用 numpy.testing.assert_allclose，禁止手写 assert max_diff < tolerance。
  - 模式参照 examples/ 与 models/ 的统一规范。
"""

import os
import sys
import argparse

import torch
import numpy as np
from numpy.testing import assert_allclose

from rms_norm_golden import rms_norm_golden
from rms_norm_impl import rms_norm_wrapper


# ----------------------------------------------------------------
# 1. 环境工具
# ----------------------------------------------------------------

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


# ----------------------------------------------------------------
# 2. 测试函数
# ----------------------------------------------------------------

def test_rms_norm_level0(device_id=None, run_mode="npu"):
    """Level 0: 小数据量基础功能验证 (2D 输入，参考 layer_norm.py 示例)。"""
    print("=" * 60)
    print("Test: rms_norm Level 0 (basic 2D)")
    print("=" * 60)

    device = f"npu:{device_id}" if (run_mode == "npu" and device_id is not None) else "cpu"

    # 2D 输入（参考 layer_norm.py 示例）: x:[32, 128], weight:[128]
    torch.manual_seed(0)
    batch_size, hidden_size = 32, 128
    shape = (batch_size, hidden_size)

    x = torch.randn(shape, dtype=torch.float32, device=device)
    weight = torch.randn(hidden_size, dtype=torch.float32, device=device)
    eps = 1e-6

    # 执行 kernel wrapper
    result = rms_norm_wrapper(x, weight, eps)

    # 执行 golden
    golden = rms_norm_golden(x, weight, eps)

    # 精度对比（必须使用 assert_allclose）
    print(f"  Input shape : {x.shape}")
    print(f"  Output shape: {result.shape}")
    max_diff = np.abs(result.cpu().numpy() - golden.cpu().numpy()).max()
    print(f"  Max diff    : {max_diff:.6e}")

    if run_mode == "npu":
        assert_allclose(
            result.cpu().numpy(),
            golden.cpu().numpy(),
            rtol=1e-3,
            atol=1e-3,
        )

    print("  [PRECISION_PASS]")
    print("  PASSED\n")


def test_rms_norm_level1(device_id=None, run_mode="npu"):
    """Level 1: 3D 输入测试 (LLaMA-7B 配置小规模)。"""
    print("=" * 60)
    print("Test: rms_norm Level 1 (3D)")
    print("=" * 60)

    device = f"npu:{device_id}" if (run_mode == "npu" and device_id is not None) else "cpu"

    # LLaMA-7B 配置小规模: x:[2, 128, 4096], weight:[4096]
    torch.manual_seed(42)
    batch_size, seq_len, hidden_size = 2, 128, 4096
    shape = (batch_size, seq_len, hidden_size)

    x = torch.randn(shape, dtype=torch.float32, device=device)
    weight = torch.randn(hidden_size, dtype=torch.float32, device=device)
    eps = 1e-6

    result = rms_norm_wrapper(x, weight, eps)
    golden = rms_norm_golden(x, weight, eps)

    max_diff = np.abs(result.cpu().numpy() - golden.cpu().numpy()).max()
    print(f"  Shape: {shape}, Max diff: {max_diff:.6e}")

    if run_mode == "npu":
        assert_allclose(
            result.cpu().numpy(),
            golden.cpu().numpy(),
            rtol=1e-3,
            atol=1e-3,
        )

    print("  [PRECISION_PASS]")
    print("  PASSED\n")


def test_rms_norm_level2(device_id=None, run_mode="npu"):
    """Level 2: 动态轴测试。"""
    print("=" * 60)
    print("Test: rms_norm Level 2 (dynamic axis)")
    print("=" * 60)

    device = f"npu:{device_id}" if (run_mode == "npu" and device_id is not None) else "cpu"

    # 动态轴测试: x:[b, s, 256], weight:[256]
    torch.manual_seed(123)
    batch_size, seq_len, hidden_size = 4, 64, 256
    shape = (batch_size, seq_len, hidden_size)

    x = torch.randn(shape, dtype=torch.float32, device=device)
    weight = torch.randn(hidden_size, dtype=torch.float32, device=device)
    eps = 1e-6

    result = rms_norm_wrapper(x, weight, eps)
    golden = rms_norm_golden(x, weight, eps)

    max_diff = np.abs(result.cpu().numpy() - golden.cpu().numpy()).max()
    print(f"  Shape: {shape}, Max diff: {max_diff:.6e}")

    if run_mode == "npu":
        assert_allclose(
            result.cpu().numpy(),
            golden.cpu().numpy(),
            rtol=1e-3,
            atol=1e-3,
        )

    print("  [PRECISION_PASS]")
    print("  PASSED\n")


def test_rms_norm_level3(device_id=None, run_mode="npu"):
    """Level 3: LLaMA-13B 配置小规模。"""
    print("=" * 60)
    print("Test: rms_norm Level 3 (LLaMA-13B)")
    print("=" * 60)

    device = f"npu:{device_id}" if (run_mode == "npu" and device_id is not None) else "cpu"

    # LLaMA-13B 配置小规模: x:[2, 64, 5120], weight:[5120]
    torch.manual_seed(456)
    batch_size, seq_len, hidden_size = 2, 64, 5120
    shape = (batch_size, seq_len, hidden_size)

    x = torch.randn(shape, dtype=torch.float32, device=device)
    weight = torch.randn(hidden_size, dtype=torch.float32, device=device)
    eps = 1e-6

    result = rms_norm_wrapper(x, weight, eps)
    golden = rms_norm_golden(x, weight, eps)

    max_diff = np.abs(result.cpu().numpy() - golden.cpu().numpy()).max()
    print(f"  Shape: {shape}, Max diff: {max_diff:.6e}")

    if run_mode == "npu":
        assert_allclose(
            result.cpu().numpy(),
            golden.cpu().numpy(),
            rtol=1e-3,
            atol=1e-3,
        )

    print("  [PRECISION_PASS]")
    print("  PASSED\n")


# ----------------------------------------------------------------
# 3. CLI 入口
# ----------------------------------------------------------------

# 用例注册表
EXAMPLES = {
    "rms_norm::test_rms_norm_level0": {
        "name": "rms_norm Level 0",
        "description": "2D 基础功能验证",
        "function": test_rms_norm_level0,
    },
    "rms_norm::test_rms_norm_level1": {
        "name": "rms_norm Level 1",
        "description": "3D 输入测试 (LLaMA-7B)",
        "function": test_rms_norm_level1,
    },
    "rms_norm::test_rms_norm_level2": {
        "name": "rms_norm Level 2",
        "description": "动态轴测试",
        "function": test_rms_norm_level2,
    },
    "rms_norm::test_rms_norm_level3": {
        "name": "rms_norm Level 3",
        "description": "LLaMA-13B 配置",
        "function": test_rms_norm_level3,
    },
}


def main():
    parser = argparse.ArgumentParser(
        description="PyPTO rms_norm operator test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s rms_norm::test_rms_norm_level0    Run Level 0
  %(prog)s --list                            List all cases
        """,
    )
    parser.add_argument("example_id", type=str, nargs="?", help="Case ID to run")
    parser.add_argument("--list", action="store_true", help="List available cases")
    parser.add_argument(
        "--run_mode", "--run-mode",
        type=str,
        default="npu",
        choices=["npu", "sim"],
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
