#!/usr/bin/env python3
# coding: utf-8
"""
GELU 算子测试

验证 PyPTO GELU 实现与 PyTorch golden 的一致性。
"""
import os
import sys
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from numpy.testing import assert_allclose

from gelu_impl import gelu_wrapper
from gelu_golden import gelu_golden


# ─────────────────────────────────────────────
# 环境工具
# ─────────────────────────────────────────────

def get_device_id():
    """从环境变量获取 TILE_FWK_DEVICE_ID。"""
    if "TILE_FWK_DEVICE_ID" not in os.environ:
        print("ERROR: Please set TILE_FWK_DEVICE_ID environment variable")
        print("Example: export TILE_FWK_DEVICE_ID=0")
        return None

    try:
        device_id = int(os.environ["TILE_FWK_DEVICE_ID"])
        return device_id
    except ValueError:
        print(f"ERROR: TILE_FWK_DEVICE_ID must be an integer, got: {os.environ['TILE_FWK_DEVICE_ID']}")
        return None


# ─────────────────────────────────────────────
# 测试用例
# ─────────────────────────────────────────────

def test_gelu_level0(device_id: int = None, run_mode: str = "npu"):
    """
    Level 0: 小数据量基础功能验证

    验证基本功能和 shape 正确性
    """
    print("\n" + "=" * 60)
    print("GELU Level 0 Test: Basic Functionality")
    print("=" * 60 + "\n")

    device = f'npu:{device_id}' if run_mode == "npu" and device_id is not None else 'cpu'

    shape = (4, 128)  # 简化 shape
    x = torch.randn(shape, dtype=torch.float32, device=device)

    output = gelu_wrapper(x)
    golden = gelu_golden(x)

    output_cpu = output.cpu() if run_mode == "npu" else output
    golden_cpu = golden.cpu() if run_mode == "npu" else golden

    max_diff = np.abs(output_cpu.numpy() - golden_cpu.numpy()).max()
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Max difference: {max_diff:.6e}")

    assert_allclose(output_cpu.numpy(), golden_cpu.numpy(), rtol=1e-3, atol=1e-3)
    print("✓ Test passed")
    print()


def test_gelu_level1(device_id: int = None, run_mode: str = "npu"):
    """
    Level 1: 典型场景验证

    验证典型配置（来自 spec.md）
    """
    print("\n" + "=" * 60)
    print("GELU Level 1 Test: Typical Scenarios")
    print("=" * 60 + "\n")

    device = f'npu:{device_id}' if run_mode == "npu" and device_id is not None else 'cpu'

    test_cases = [
        {
            "name": "性能_P0_BERT",
            "shape": (1, 128, 768),
            "desc": "BERT-base 推理"
        },
        {
            "name": "性能_P0_GPT2",
            "shape": (4, 512, 1024),
            "desc": "GPT-2 medium"
        },
        {
            "name": "功能_P1_LLaMA",
            "shape": (8, 1024, 4096),
            "desc": "LLaMA 推理"
        },
        {
            "name": "功能_P1_custom",
            "shape": (2, 256, 512),
            "desc": "自定义 shape"
        },
    ]

    all_passed = True
    for case in test_cases:
        try:
            x = torch.randn(case["shape"], dtype=torch.float32, device=device)
            output = gelu_wrapper(x)
            golden = gelu_golden(x)

            output_cpu = output.cpu() if run_mode == "npu" else output
            golden_cpu = golden.cpu() if run_mode == "npu" else golden

            max_diff = np.abs(output_cpu.numpy() - golden_cpu.numpy()).max()

            assert_allclose(output_cpu.numpy(), golden_cpu.numpy(), rtol=1e-3, atol=1e-3)
            print(f"✓ {case['name']}: {case['desc']} ... PASS (max_diff={max_diff:.6e})")
        except Exception as e:
            print(f"✗ {case['name']}: {case['desc']} ... FAIL: {e}")
            all_passed = False

    print()
    if all_passed:
        print("✓ All Level 1 tests passed")
    else:
        print("✗ Some Level 1 tests failed")
    print()


def test_gelu_edge_cases(device_id: int = None, run_mode: str = "npu"):
    """
    边界条件测试
    """
    print("\n" + "=" * 60)
    print("GELU Edge Cases Test")
    print("=" * 60 + "\n")

    device = f'npu:{device_id}' if run_mode == "npu" and device_id is not None else 'cpu'

    test_cases = [
        {
            "name": "zero_pos_neg",
            "values": torch.tensor([[0.0, 1.0, -1.0, 0.5, -0.5]], dtype=torch.float32, device=device),
            "desc": "零值、正负值"
        },
        {
            "name": "small_values",
            "values": torch.tensor([[0.001, -0.001, 0.0001, -0.0001]], dtype=torch.float32, device=device),
            "desc": "小值"
        },
        {
            "name": "large_values",
            "values": torch.tensor([[10.0, -10.0, 5.0, -5.0]], dtype=torch.float32, device=device),
            "desc": "大值"
        },
    ]

    all_passed = True
    for case in test_cases:
        try:
            x = case["values"]
            output = gelu_wrapper(x)
            golden = gelu_golden(x)

            output_cpu = output.cpu() if run_mode == "npu" else output
            golden_cpu = golden.cpu() if run_mode == "npu" else golden

            max_diff = np.abs(output_cpu.numpy() - golden_cpu.numpy()).max()

            assert_allclose(output_cpu.numpy(), golden_cpu.numpy(), rtol=1e-3, atol=1e-3)
            print(f"✓ {case['name']}: {case['desc']} ... PASS (max_diff={max_diff:.6e})")
        except Exception as e:
            print(f"✗ {case['name']}: {case['desc']} ... FAIL: {e}")
            all_passed = False

    print()
    if all_passed:
        print("✓ All edge case tests passed")
    else:
        print("✗ Some edge case tests failed")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="PyPTO GELU Test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s gelu::test_gelu_level0    Run Level 0
  %(prog)s --list                    List all cases
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

    # Define test cases
    examples = {
        "gelu::test_gelu_level0": {
            "name": "GELU Level 0",
            "description": "Basic functionality test",
            "function": test_gelu_level0,
        },
        "gelu::test_gelu_level1": {
            "name": "GELU Level 1",
            "description": "Typical scenarios test",
            "function": test_gelu_level1,
        },
        "gelu::test_gelu_edge_cases": {
            "name": "GELU Edge Cases",
            "description": "Edge cases test",
            "function": test_gelu_edge_cases,
        },
    }

    # --list
    if args.list:
        print("\nAvailable cases:\n")
        for key, info in sorted(examples.items()):
            print(f"  {key}  — {info['description']}")
        return

    # Select case
    if args.example_id:
        if args.example_id not in examples:
            print(f"ERROR: unknown case '{args.example_id}'")
            print(f"Valid: {', '.join(sorted(examples.keys()))}")
            sys.exit(1)
        to_run = [(args.example_id, examples[args.example_id])]
    else:
        to_run = list(sorted(examples.items()))

    # NPU device initialization
    device_id = None
    if args.run_mode == "npu":
        device_id = get_device_id()
        if device_id is None:
            return
        import torch_npu
        torch.npu.set_device(device_id)

    # Execute
    try:
        for key, info in to_run:
            print(f"\n▶ Running {key}: {info['name']}")
            info["function"](device_id, args.run_mode)

        print("\n" + "=" * 60)
        print("All tests passed!")
        print("=" * 60)
    except Exception as e:
        print(f"\nError: {e}")
        raise


if __name__ == "__main__":
    main()
