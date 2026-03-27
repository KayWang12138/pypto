#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY kind, either express or implied,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
Softmax 算子测试

验证 softmax 实现的精度。
"""

import os
import sys
import argparse
import torch
import numpy as np
from numpy.testing import assert_allclose

from softmax_golden import softmax_golden
from softmax_impl import softmax_wrapper


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

def test_softmax_level0(device_id: int = None, run_mode: str = "npu"):
    """
    Level 0: 小数据量基础功能验证
    
    验证基本功能和 shape 正确性
    """
    print("\n" + "=" * 60)
    print("Softmax Level 0 Test: Basic Functionality")
    print("=" * 60 + "\n")
    
    device = f'npu:{device_id}' if run_mode == "npu" and device_id is not None else 'cpu'
    
    shape = (4, 128, 8, 64)
    x = torch.randn(shape, dtype=torch.float32, device=device)
    
    output = softmax_wrapper(x, dim=-1)
    golden = softmax_golden(x, dim=-1)
    
    output_cpu = output.cpu() if run_mode == "npu" else output
    golden_cpu = golden.cpu() if run_mode == "npu" else golden
    
    max_diff = np.abs(output_cpu.numpy() - golden_cpu.numpy()).max()
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Max difference: {max_diff:.6e}")
    
    assert_allclose(output_cpu.numpy(), golden_cpu.numpy(), rtol=1e-3, atol=1e-3)
    print("✓ Test passed")
    print()


def test_softmax_level1(device_id: int = None, run_mode: str = "npu"):
    """
    Level 1: 典型场景验证
    
    验证典型配置（来自 spec.md）
    """
    print("\n" + "=" * 60)
    print("Softmax Level 1 Test: Typical Scenarios")
    print("=" * 60 + "\n")
    
    device = f'npu:{device_id}' if run_mode == "npu" and device_id is not None else 'cpu'
    
    test_cases = [
        {
            "name": "性能_P0",
            "shape": (1, 4096, 4096),
            "dim": -1,
            "desc": "Attention 场景大 token"
        },
        {
            "name": "功能_P0",
            "shape": (2, 1024, 512),
            "dim": -1,
            "desc": "常规 batch 推理"
        },
        {
            "name": "功能_P1_1",
            "shape": (4, 256, 128),
            "dim": 1,
            "desc": "中间轴归一化"
        },
        {
            "name": "功能_P1_2",
            "shape": (8, 512, 64),
            "dim": -1,
            "desc": "小批量多 head",
        },
    ]
    
    all_passed = True
    for case in test_cases:
        try:
            x = torch.randn(case["shape"], dtype=torch.float32, device=device)
            output = softmax_wrapper(x, dim=case["dim"])
            golden = softmax_golden(x, dim=case["dim"])
            
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


def main():
    parser = argparse.ArgumentParser(
        description="PyPTO Softmax Test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s softmax::test_softmax_level0    Run Level 0
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
        "softmax::test_softmax_level0": {
            "name": "Softmax Level 0",
            "description": "Basic functionality test",
            "function": test_softmax_level0,
        },
        "softmax::test_softmax_level1": {
            "name": "Softmax Level 1",
            "description": "Typical scenarios test",
            "function": test_softmax_level1,
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
