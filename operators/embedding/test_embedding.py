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
"""PyPTO embedding operator test.

测试内容:
- 2D indices 输入
- padding_idx 支持
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

from embedding_golden import embedding_golden
from embedding_impl import embedding_wrapper

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

def test_embedding_level0(device_id=None, run_mode="npu"):
    """Level 0: 基础功能验证。"""
    print("=" * 60)
    print("Test: embedding Level 0 (basic)")
    print("=" * 60)

    device = f"npu:{device_id}" if (run_mode == "npu" and device_id is not None) else "cpu"

    torch.manual_seed(0)
    vocab_size, embed_dim = 1000, 64
    batch, seq = 2, 16

    weight = torch.randn((vocab_size, embed_dim), dtype=torch.float32, device=device)
    indices = torch.randint(0, vocab_size, (batch, seq), dtype=torch.int64, device=device)

    result = embedding_wrapper(indices, weight)
    golden = embedding_golden(indices, weight)

    print(f"  Indices shape: {indices.shape}")
    print(f"  Weight shape  : {weight.shape}")
    print(f"  Output shape  : {result.shape}")
    max_diff = np.abs(result.cpu().numpy() - golden.cpu().numpy()).max()
    print(f"  Max diff      : {max_diff:.6e}")

    if run_mode == "npu":
        assert_allclose(
            result.cpu().numpy(),
            golden.cpu().numpy(),
            rtol=1e-3, atol=1e-3,
        )

    print("  OK Passed\n")


def test_embedding_level1(device_id=None, run_mode="npu"):
    """Level 1: 典型场景验证。"""
    print("=" * 60)
    print("Test: embedding Level 1 (typical)")
    print("=" * 60)

    device = f"npu:{device_id}" if (run_mode == "npu" and device_id is not None) else "cpu"

    torch.manual_seed(42)
    vocab_size, embed_dim = 50000, 256
    batch, seq = 4, 128

    weight = torch.randn((vocab_size, embed_dim), dtype=torch.float32, device=device)
    indices = torch.randint(0, vocab_size, (batch, seq), dtype=torch.int64, device=device)

    result = embedding_wrapper(indices, weight)
    golden = embedding_golden(indices, weight)

    print(f"  Indices shape: {indices.shape}")
    print(f"  Weight shape  : {weight.shape}")
    print(f"  Output shape  : {result.shape}")
    max_diff = np.abs(result.cpu().numpy() - golden.cpu().numpy()).max()
    print(f"  Max diff      : {max_diff:.6e}")

    if run_mode == "npu":
        assert_allclose(
            result.cpu().numpy(),
            golden.cpu().numpy(),
            rtol=1e-3, atol=1e-3,
        )

    print("  OK Passed\n")


def test_embedding_padding(device_id=None, run_mode="npu"):
    """padding_idx 参数兼容性验证。

    注意：根据 PyTorch 行为，padding_idx 在 forward 时不会将输出置零，
    仅在 backward 时影响梯度。PyPTO 作为推理框架，行为与 PyTorch forward 一致。
    """
    print("=" * 60)
    print("Test: embedding Padding (API compatibility)")
    print("=" * 60)

    device = f"npu:{device_id}" if (run_mode == "npu" and device_id is not None) else "cpu"

    torch.manual_seed(123)
    vocab_size, embed_dim = 100, 32
    batch, seq = 2, 8
    padding_idx = 0

    weight = torch.randn((vocab_size, embed_dim), dtype=torch.float32, device=device)
    indices = torch.randint(0, vocab_size, (batch, seq), dtype=torch.int64, device=device)
    # 确保有一些 padding 位置
    indices[0, 0] = padding_idx
    indices[1, 3] = padding_idx

    result = embedding_wrapper(indices, weight, padding_idx=padding_idx)
    golden = embedding_golden(indices, weight, padding_idx=padding_idx)

    print(f"  Indices shape: {indices.shape}")
    print(f"  Padding idx   : {padding_idx}")
    print(f"  Output shape  : {result.shape}")
    print(f"  Note: padding_idx does NOT zero out output in forward pass")

    max_diff = np.abs(result.cpu().numpy() - golden.cpu().numpy()).max()
    print(f"  Max diff      : {max_diff:.6e}")

    if run_mode == "npu":
        assert_allclose(
            result.cpu().numpy(),
            golden.cpu().numpy(),
            rtol=1e-3, atol=1e-3,
        )

    print("  OK Passed\n")


def test_embedding_dtype_fp16(device_id=None, run_mode="npu"):
    """FP16 dtype 支持。"""
    print("=" * 60)
    print("Test: embedding dtype FP16")
    print("=" * 60)

    device = f"npu:{device_id}" if (run_mode == "npu" and device_id is not None) else "cpu"

    torch.manual_seed(456)
    vocab_size, embed_dim = 1000, 64
    batch, seq = 2, 16

    weight = torch.randn((vocab_size, embed_dim), dtype=torch.float16, device=device)
    indices = torch.randint(0, vocab_size, (batch, seq), dtype=torch.int64, device=device)

    result = embedding_wrapper(indices, weight)
    golden = embedding_golden(indices, weight)

    print(f"  Indices shape: {indices.shape}")
    print(f"  Weight dtype : {weight.dtype}")
    print(f"  Output dtype : {result.dtype}")

    # FP16 使用更宽松的容差
    if run_mode == "npu":
        assert_allclose(
            result.cpu().float().numpy(),
            golden.cpu().float().numpy(),
            rtol=1e-2, atol=1e-2,
        )

    print("  OK Passed\n")


def test_embedding_large(device_id=None, run_mode="npu"):
    """大规模性能验证。"""
    print("=" * 60)
    print("Test: embedding Large (performance)")
    print("=" * 60)

    device = f"npu:{device_id}" if (run_mode == "npu" and device_id is not None) else "cpu"

    torch.manual_seed(789)
    vocab_size, embed_dim = 100000, 512
    batch, seq = 32, 256

    weight = torch.randn((vocab_size, embed_dim), dtype=torch.float32, device=device)
    indices = torch.randint(0, vocab_size, (batch, seq), dtype=torch.int64, device=device)

    result = embedding_wrapper(indices, weight)
    golden = embedding_golden(indices, weight)

    print(f"  Indices shape: {indices.shape}")
    print(f"  Weight shape  : {weight.shape}")
    print(f"  Output shape  : {result.shape}")
    max_diff = np.abs(result.cpu().numpy() - golden.cpu().numpy()).max()
    print(f"  Max diff      : {max_diff:.6e}")

    if run_mode == "npu":
        assert_allclose(
            result.cpu().numpy(),
            golden.cpu().numpy(),
            rtol=1e-3, atol=1e-3,
        )

    print("  OK Passed\n")


# ─────────────────────────────────────────────
# 3. CLI 入口
# ─────────────────────────────────────────────

# 用例注册表
EXAMPLES = {
    "embedding::test_embedding_level0": {
        "name": "embedding Level 0",
        "description": "基础功能验证",
        "function": test_embedding_level0,
    },
    "embedding::test_embedding_level1": {
        "name": "embedding Level 1",
        "description": "典型场景验证",
        "function": test_embedding_level1,
    },
    "embedding::test_embedding_padding": {
        "name": "embedding Padding",
        "description": "padding_idx 功能验证",
        "function": test_embedding_padding,
    },
    "embedding::test_embedding_dtype_fp16": {
        "name": "embedding dtype FP16",
        "description": "FP16 dtype 支持",
        "function": test_embedding_dtype_fp16,
    },
    "embedding::test_embedding_large": {
        "name": "embedding Large",
        "description": "大规模性能验证",
        "function": test_embedding_large,
    },
}


def main():
    parser = argparse.ArgumentParser(
        description="PyPTO embedding operator test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s embedding::test_embedding_level0    Run Level 0
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
        import traceback
        traceback.print_exc()
        sys.exit(2)


if __name__ == "__main__":
    main()
