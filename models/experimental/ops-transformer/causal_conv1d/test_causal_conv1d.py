#!/usr/bin/env python3
# coding: utf-8

"""PyPTO causal_conv1d operator test.

测试 Prefill 和 Decode 两种模式：
- Prefill_P0: seqlen=2048, dim=2048, batch=1
- Decode_P0: seqlen=1, dim=2048, batch=1

精度验证使用 assert_allclose，rtol=0.01, atol=0.01
"""

import os
import sys
import argparse

import torch
import numpy as np
from numpy.testing import assert_allclose

from causal_conv1d_golden import (
    causal_conv1d_prefill_golden,
    causal_conv1d_decode_golden,
)
from causal_conv1d_impl import (
    causal_conv1d_prefill_wrapper,
    causal_conv1d_decode_wrapper,
)


def get_device_id():
    """从环境变量获取 TILE_FWK_DEVICE_ID"""
    if "TILE_FWK_DEVICE_ID" not in os.environ:
        print("Please set: export TILE_FWK_DEVICE_ID={device_id}")
        return 0
    try:
        return int(os.environ["TILE_FWK_DEVICE_ID"])
    except ValueError:
        print(f"ERROR: TILE_FWK_DEVICE_ID must be int, got: {os.environ['TILE_FWK_DEVICE_ID']}")
        return 0


def test_prefill_p0(device_id=None, run_mode="npu"):
    """Prefill_P0: seqlen=2048, dim=2048, width=4, batch=1"""
    print("=" * 60)
    print("Test: causal_conv1d Prefill_P0")
    print("=" * 60)

    device = f"npu:{device_id}" if (run_mode == "npu" and device_id is not None) else "cpu"

    # 参数
    seqlen = 2048
    dim = 2048
    width = 4
    batch = 1
    state_len = width - 1  # 3

    # 测试数据
    torch.manual_seed(42)
    x = torch.randn(seqlen, dim, dtype=torch.float16, device=device)
    weight = torch.randn(width, dim, dtype=torch.float16, device=device)
    conv_state = torch.randn(batch, state_len, dim, dtype=torch.float16, device=device)
    cu_seqlens = torch.tensor([0, seqlen], dtype=torch.int32, device=device)

    # 复制 conv_state 用于 golden（因为 golden 会原地更新）
    conv_state_golden = conv_state.clone()

    # 执行 impl
    y_impl = causal_conv1d_prefill_wrapper(x, weight, conv_state, cu_seqlens, activation="silu", width=width)

    # 执行 golden
    y_golden = causal_conv1d_prefill_golden(x, weight, conv_state_golden, cu_seqlens, activation="silu", width=width)

    # 精度对比
    print(f"  Input shape      : {x.shape}")
    print(f"  Output shape     : {y_impl.shape}")
    y_diff = np.abs(y_impl.cpu().numpy() - y_golden.cpu().numpy()).max()
    print(f"  y max diff       : {y_diff:.6e}")
    state_diff = np.abs(conv_state.cpu().numpy() - conv_state_golden.cpu().numpy()).max()
    print(f"  conv_state diff  : {state_diff:.6e}")

    if run_mode == "npu":
        try:
            assert_allclose(
                y_impl.cpu().numpy(),
                y_golden.cpu().numpy(),
                rtol=0.01, atol=0.01,
            )
            assert_allclose(
                conv_state.cpu().numpy(),
                conv_state_golden.cpu().numpy(),
                rtol=0.01, atol=0.01,
            )
            print("[PRECISION_PASS]")
        except AssertionError as e:
            print(f"[PRECISION_FAIL] {e}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"Runtime error: {e}", file=sys.stderr)
            sys.exit(2)

    print("  ✓ Passed\n")


def test_prefill_tilelang_equiv(device_id=None, run_mode="npu"):
    """Prefill with TileLang equivalent config: seqlen=2048, dim=2048, width=4, batch=1, num_cache_lines=804"""
    print("=" * 60)
    print("Test: causal_conv1d Prefill TileLang Equivalent")
    print("=" * 60)

    device = f"npu:{device_id}" if (run_mode == "npu" and device_id is not None) else "cpu"

    # TileLang 测试用例规格
    seqlen = 2048
    dim = 2048
    width = 4
    batch = 1
    num_cache_lines = 804
    state_len = 3

    torch.manual_seed(42)
    x = torch.randn(seqlen, dim, dtype=torch.float16, device=device)
    weight = torch.randn(width, dim, dtype=torch.float16, device=device)
    conv_state = torch.randn(num_cache_lines, state_len, dim, dtype=torch.float16, device=device)
    cu_seqlens = torch.tensor([0, seqlen], dtype=torch.int32, device=device)
    cache_indices = torch.tensor([0], dtype=torch.int32, device=device)

    conv_state_golden = conv_state.clone()

    y_impl = causal_conv1d_prefill_wrapper(
        x, weight, conv_state, cu_seqlens,
        activation="silu", width=width, cache_indices=cache_indices
    )

    y_golden = causal_conv1d_prefill_golden(
        x, weight, conv_state_golden, cu_seqlens,
        activation="silu", width=width, cache_indices=cache_indices
    )

    print(f"  Input shape      : {x.shape}")
    print(f"  Output shape     : {y_impl.shape}")
    print(f"  num_cache_lines  : {num_cache_lines}")
    print(f"  cache_indices    : {cache_indices.tolist()}")
    y_diff = np.abs(y_impl.cpu().numpy() - y_golden.cpu().numpy()).max()
    print(f"  y max diff       : {y_diff:.6e}")
    state_diff = np.abs(conv_state.cpu().numpy() - conv_state_golden.cpu().numpy()).max()
    print(f"  conv_state diff  : {state_diff:.6e}")

    if run_mode == "npu":
        try:
            assert_allclose(
                y_impl.cpu().numpy(),
                y_golden.cpu().numpy(),
                rtol=0.01, atol=0.01,
            )
            assert_allclose(
                conv_state.cpu().numpy(),
                conv_state_golden.cpu().numpy(),
                rtol=0.01, atol=0.01,
            )
            print("[PRECISION_PASS]")
        except AssertionError as e:
            print(f"[PRECISION_FAIL] {e}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"Runtime error: {e}", file=sys.stderr)
            sys.exit(2)

    print("  ✓ Passed\n")


def test_decode_p0(device_id=None, run_mode="npu"):
    """Decode_P0: seqlen=1, dim=2048, width=4, batch=1"""
    print("=" * 60)
    print("Test: causal_conv1d Decode_P0")
    print("=" * 60)

    device = f"npu:{device_id}" if (run_mode == "npu" and device_id is not None) else "cpu"

    # 参数
    seqlen = 1
    dim = 2048
    width = 4
    batch = 1
    state_len = width - 1  # 3

    # 测试数据
    torch.manual_seed(42)
    x = torch.randn(batch, seqlen, dim, dtype=torch.float16, device=device)
    weight = torch.randn(width, dim, dtype=torch.float16, device=device)
    conv_state = torch.randn(batch, state_len, dim, dtype=torch.float16, device=device)

    # 复制 conv_state 用于 golden
    conv_state_golden = conv_state.clone()

    # 执行 impl
    y_impl = causal_conv1d_decode_wrapper(x, weight, conv_state, activation="silu", width=width)

    # 执行 golden
    y_golden = causal_conv1d_decode_golden(x, weight, conv_state_golden, activation="silu", width=width)

    # 精度对比
    print(f"  Input shape : {x.shape}")
    print(f"  Output shape: {y_impl.shape}")
    max_diff = np.abs(y_impl.cpu().numpy() - y_golden.cpu().numpy()).max()
    print(f"  Max diff    : {max_diff:.6e}")

    if run_mode == "npu":
        try:
            assert_allclose(
                y_impl.cpu().numpy(),
                y_golden.cpu().numpy(),
                rtol=0.01, atol=0.01,
            )
            print("[PRECISION_PASS]")
        except AssertionError as e:
            print(f"[PRECISION_FAIL] {e}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"Runtime error: {e}", file=sys.stderr)
            sys.exit(2)

    print("  ✓ Passed\n")


def test_decode_full(device_id=None, run_mode="npu"):
    """Decode 完整版本: seqlen=4, dim=2048, batch=1, bias=on, cache_indices=[7]"""
    print("=" * 60)
    print("Test: causal_conv1d Decode full (bias + cache_indices + multi-token)")
    print("=" * 60)

    device = f"npu:{device_id}" if (run_mode == "npu" and device_id is not None) else "cpu"

    seqlen = 4
    dim = 2048
    width = 4
    batch = 1
    state_len = width - 1 + seqlen
    num_cache_lines = 10

    torch.manual_seed(42)
    x = torch.randn(batch, seqlen, dim, dtype=torch.float16, device=device)
    weight = torch.randn(width, dim, dtype=torch.float16, device=device)
    bias = torch.randn(dim, dtype=torch.float16, device=device)
    conv_state = torch.randn(num_cache_lines, state_len, dim, dtype=torch.float16, device=device)
    cache_indices = torch.tensor([7], dtype=torch.int32, device=device)

    conv_state_golden = conv_state.clone()

    y_impl = causal_conv1d_decode_wrapper(
        x, weight, conv_state, activation="silu", width=width, bias=bias, cache_indices=cache_indices
    )
    y_golden = causal_conv1d_decode_golden(
        x, weight, conv_state_golden, activation="silu", width=width, bias=bias, cache_indices=cache_indices
    )

    print(f"  Input shape      : {x.shape}")
    print(f"  Output shape     : {y_impl.shape}")
    print(f"  bias shape       : {bias.shape}")
    print(f"  cache_indices    : {cache_indices.tolist()}")
    max_diff = np.abs(y_impl.cpu().numpy() - y_golden.cpu().numpy()).max()
    print(f"  Max diff         : {max_diff:.6e}")

    if run_mode == "npu":
        try:
            assert_allclose(y_impl.cpu().numpy(), y_golden.cpu().numpy(), rtol=0.01, atol=0.01)
            print("[PRECISION_PASS]")
        except AssertionError as e:
            print(f"[PRECISION_FAIL] {e}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"Runtime error: {e}", file=sys.stderr)
            sys.exit(2)

    print("  ✓ Passed\n")


def test_prefill_with_bias_and_cache(device_id=None, run_mode="npu"):
    """Prefill 完整版本: seqlen=2048, dim=2048, batch=1, bias=on, cache_indices=[5]"""
    print("=" * 60)
    print("Test: causal_conv1d Prefill full (bias + cache_indices)")
    print("=" * 60)

    device = f"npu:{device_id}" if (run_mode == "npu" and device_id is not None) else "cpu"

    seqlen = 2048
    dim = 2048
    width = 4
    batch = 1
    state_len = width - 1
    num_cache_lines = 10

    torch.manual_seed(42)
    x = torch.randn(seqlen, dim, dtype=torch.float16, device=device)
    weight = torch.randn(width, dim, dtype=torch.float16, device=device)
    bias = torch.randn(dim, dtype=torch.float16, device=device)
    conv_state = torch.randn(num_cache_lines, state_len, dim, dtype=torch.float16, device=device)
    cu_seqlens = torch.tensor([0, seqlen], dtype=torch.int32, device=device)
    cache_indices = torch.tensor([5], dtype=torch.int32, device=device)

    conv_state_golden = conv_state.clone()

    y_impl = causal_conv1d_prefill_wrapper(
        x, weight, conv_state, cu_seqlens, activation="silu", width=width, bias=bias, cache_indices=cache_indices
    )
    y_golden = causal_conv1d_prefill_golden(
        x, weight, conv_state_golden, cu_seqlens, activation="silu", width=width, bias=bias, cache_indices=cache_indices
    )

    print(f"  Input shape      : {x.shape}")
    print(f"  Output shape     : {y_impl.shape}")
    print(f"  bias shape       : {bias.shape}")
    print(f"  cache_indices    : {cache_indices.tolist()}")
    y_diff = np.abs(y_impl.cpu().numpy() - y_golden.cpu().numpy()).max()
    print(f"  y max diff       : {y_diff:.6e}")
    state_diff = np.abs(conv_state.cpu().numpy() - conv_state_golden.cpu().numpy()).max()
    print(f"  conv_state diff  : {state_diff:.6e}")

    if run_mode == "npu":
        try:
            assert_allclose(y_impl.cpu().numpy(), y_golden.cpu().numpy(), rtol=0.01, atol=0.01)
            assert_allclose(conv_state.cpu().numpy(), conv_state_golden.cpu().numpy(), rtol=0.01, atol=0.01)
            print("[PRECISION_PASS]")
        except AssertionError as e:
            print(f"[PRECISION_FAIL] {e}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"Runtime error: {e}", file=sys.stderr)
            sys.exit(2)

    print("  ✓ Passed\n")


# ─────────────────────────────────────────────
# CLI 入口
# ─────────────────────────────────────────────

EXAMPLES = {
    "causal_conv1d::test_prefill_tilelang_equiv": {
        "name": "Prefill_TileLang_equiv",
        "description": "seqlen=2048, dim=2048, batch=1, num_cache_lines=804, cache_indices=[0]",
        "function": test_prefill_tilelang_equiv,
    },
    "causal_conv1d::test_prefill_p0": {
        "name": "Prefill_P0",
        "description": "seqlen=2048, dim=2048, batch=1",
        "function": test_prefill_p0,
    },
    "causal_conv1d::test_prefill_with_cache_indices": {
        "name": "Prefill_full",
        "description": "seqlen=2048, dim=2048, batch=1, bias=on, cache_indices=[5]",
        "function": test_prefill_with_bias_and_cache,
    },
    "causal_conv1d::test_decode_p0": {
        "name": "Decode_P0",
        "description": "seqlen=1, dim=2048, batch=1",
        "function": test_decode_p0,
    },
    "causal_conv1d::test_decode_multi_token": {
        "name": "Decode_full",
        "description": "seqlen=4, dim=2048, batch=1, bias=on, cache_indices=[7]",
        "function": test_decode_full,
    },
}


def main():
    parser = argparse.ArgumentParser(
        description="PyPTO causal_conv1d operator test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s causal_conv1d::test_prefill_p0    Run Prefill_P0
  %(prog)s causal_conv1d::test_decode_p0     Run Decode_P0
  %(prog)s --list                            List all cases
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
        import torch_npu
        torch.npu.set_device(device_id)

    # 执行
    try:
        for key, info in to_run:
            print(f"\n▸ Running {key}: {info['name']}")
            info["function"](device_id, args.run_mode)
        print("\n" + "=" * 60)
        print("All tests passed!")
        print("=" * 60)
    except Exception as e:
        print(f"\nError: {e}")
        raise


if __name__ == "__main__":
    main()