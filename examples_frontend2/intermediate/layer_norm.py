#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
Layer Normalization Example for PyPTO

This example demonstrates:
- Standard Layer Normalization (LayerNorm)
- RMS Normalization (RMSNorm)
- Static and dynamic batch size support
- Residual connections

Layer normalization is a key component in transformer architectures.
"""

import os
import sys
import argparse
import pypto
import torch
import torch_npu
import numpy as np
from numpy.testing import assert_allclose
from dataclasses import dataclass
from typing import Literal


def get_device_id():
    """
    Get and validate TILE_FWK_DEVICE_ID from environment variable.

    Returns:
        int: The device ID if valid, None otherwise.
    """
    if 'TILE_FWK_DEVICE_ID' not in os.environ:
        print("ERROR: Environment variable TILE_FWK_DEVICE_ID is not set.")
        print("Please set it before running this example:")
        print("  export TILE_FWK_DEVICE_ID=0")
        return None

    try:
        device_id = int(os.environ['TILE_FWK_DEVICE_ID'])
        return device_id
    except ValueError:
        print(f"ERROR: TILE_FWK_DEVICE_ID must be an integer, got: {os.environ['TILE_FWK_DEVICE_ID']}")
        return None


@dataclass
class NormConfig:
    """Configuration for normalization operations."""
    norm_type: Literal["layernorm", "rmsnorm"] = "layernorm"
    eps: float = 1e-6
    dtype: pypto.DataType = pypto.DT_BF16
    use_dynamic_shape: bool = False


def layernorm_golden(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, eps: float) -> torch.Tensor:
    """PyTorch reference implementation of LayerNorm."""
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    normalized = (x - mean) / torch.sqrt(var + eps)
    return normalized * gamma + beta


def rmsnorm_golden(x: torch.Tensor, gamma: torch.Tensor, eps: float) -> torch.Tensor:
    """PyTorch reference implementation of RMSNorm."""
    rms = torch.sqrt((x ** 2).mean(dim=-1, keepdim=True) + eps)
    return (x / rms) * gamma


@pypto.frontend.jit()
def layer_norm(
    x: pypto.Tensor((32, 128), pypto.DT_BF16),
    gamma: pypto.Tensor((128,), pypto.DT_BF16),
    beta: pypto.Tensor((128,), pypto.DT_BF16),
    eps: float
) -> pypto.Tensor((32, 128), pypto.DT_BF16):
    """Layer Normalization."""
    hidden_size = x.shape[-1]
    pypto.set_vec_tile_shapes(64, 128)

    mean = pypto.sum(x, dim=-1, keepdim=True)
    mean = mean / float(hidden_size)
    centered = x - mean

    squared = centered * centered
    var = pypto.sum(squared, dim=-1, keepdim=True)
    var = var / float(hidden_size)

    var_eps = var + eps
    std = pypto.sqrt(var_eps)
    normalized = centered / std

    scaled = normalized * gamma
    out = scaled + beta
    return out


@pypto.frontend.jit()
def rms_norm(
    x: pypto.Tensor((32, 128), pypto.DT_BF16),
    gamma: pypto.Tensor((128,), pypto.DT_BF16),
    eps: float
) -> pypto.Tensor((32, 128), pypto.DT_BF16):
    """RMS Normalization."""
    hidden_size = x.shape[-1]
    pypto.set_vec_tile_shapes(64, 128)

    # Compute RMS: sqrt(mean(x^2) + eps)
    squared = x * x
    mean_sq = pypto.sum(squared, dim=-1, keepdim=True)
    mean_sq = mean_sq / float(hidden_size)
    rms = pypto.sqrt(mean_sq + eps)

    normalized = x / rms
    out = normalized * gamma
    return out


def test_layer_norm():
    """Test LayerNorm."""
    print("=" * 60)
    print("Test: LayerNorm")
    print("=" * 60)

    device_id = torch.npu.current_device()

    batch_size, hidden_size = 32, 128
    shape = (batch_size, hidden_size)
    x_torch = torch.randn(shape, dtype=torch.bfloat16, device=f'npu:{device_id}')
    gamma_torch = torch.ones(hidden_size, dtype=torch.bfloat16, device=f'npu:{device_id}')
    beta_torch = torch.zeros(hidden_size, dtype=torch.bfloat16, device=f'npu:{device_id}')

    config = NormConfig(norm_type="layernorm", dtype=pypto.DT_BF16)
    out_torch = layer_norm(x_torch, gamma_torch, beta_torch, config.eps)
    pypto.runtime._device_synchronize()

    expected = layernorm_golden(x_torch, gamma_torch, beta_torch, config.eps)
    max_diff = (out_torch - expected).abs().max().item()

    print(f"Input shape: {x_torch.shape}")
    print(f"Output shape: {out_torch.shape}")
    print(f"Max difference: {max_diff:.6f}")
    assert max_diff < 1e-1, "Result mismatch!"
    print("✓ LayerNorm passed")
    print()


def test_rms_norm():
    """Test RMSNorm."""
    print("=" * 60)
    print("Test: RMSNorm")
    print("=" * 60)

    device_id = torch.npu.current_device()

    batch_size, hidden_size = 32, 128
    shape = (batch_size, hidden_size)

    x_torch = torch.randn(shape, dtype=torch.bfloat16, device=f'npu:{device_id}')
    gamma_torch = torch.ones(hidden_size, dtype=torch.bfloat16, device=f'npu:{device_id}')

    config = NormConfig(norm_type="rmsnorm", dtype=pypto.DT_BF16)
    out_torch = rms_norm(x_torch, gamma_torch, config.eps)
    pypto.runtime._device_synchronize()

    expected = rmsnorm_golden(x_torch, gamma_torch, config.eps)
    max_diff = (out_torch - expected).abs().max().item()

    print(f"Input shape: {x_torch.shape}")
    print(f"Output shape: {out_torch.shape}")
    print(f"Max difference: {max_diff:.6f}")
    assert max_diff < 1e-1, "Result mismatch!"
    print("✓ RMSNorm passed")
    print()


def main():
    """Run layer normalization examples.

    Usage:
        python layer_norm.py          # Run all examples
        python layer_norm.py 1         # Run example 1 only
        python layer_norm.py --list   # List all available examples
    """
    parser = argparse.ArgumentParser(
        description="PyPTO Layer Normalization Examples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s              Run all examples
  %(prog)s 1            Run example 1 (LayerNorm)
  %(prog)s --list       List all available examples
        """
    )
    parser.add_argument(
        'example_id',
        type=int,
        nargs='?',
        help='Example ID to run (1-2). If not specified, all examples will run.'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List all available examples and exit'
    )

    args = parser.parse_args()

    examples = {
        1: {
            'name': 'LayerNorm',
            'description': 'Standard Layer Normalization',
            'function': test_layer_norm,
            'requires_npu': True
        },
        2: {
            'name': 'RMSNorm',
            'description': 'RMS Normalization',
            'function': test_rms_norm,
            'requires_npu': True
        }
    }

    if args.list:
        print("\n" + "=" * 60)
        print("Available Examples")
        print("=" * 60 + "\n")
        for ex_id, ex_info in sorted(examples.items()):
            npu_req = " (Requires NPU)" if ex_info['requires_npu'] else " (No NPU required)"
            print(f"  {ex_id}. {ex_info['name']}{npu_req}")
            print(f"     {ex_info['description']}\n")
        return

    if args.example_id is not None:
        if args.example_id not in examples:
            print(f"ERROR: Invalid example ID: {args.example_id}")
            print(f"Valid example IDs are: {', '.join(map(str, sorted(examples.keys())))}")
            print("\nUse --list to see all available examples.")
            sys.exit(1)

    print("\n" + "=" * 60)
    print("PyPTO Layer Normalization Examples")
    print("=" * 60 + "\n")

    device_id = None
    examples_to_run = []

    if args.example_id is not None:
        examples_to_run = [(args.example_id, examples[args.example_id])]
    else:
        examples_to_run = list(examples.items())

    requires_npu = any(ex_info['requires_npu'] for _, ex_info in examples_to_run)

    if requires_npu:
        device_id = get_device_id()
        if device_id is None:
            return
        torch.npu.set_device(device_id)

    try:
        for ex_id, ex_info in examples_to_run:
            if ex_info['requires_npu'] and device_id is None:
                print(f"Skipping example {ex_id} ({ex_info['name']}): NPU device not configured")
                continue

            print(f"Running Example {ex_id}: {ex_info['name']}")
            ex_info['function']()

        if len(examples_to_run) > 1:
            print("=" * 60)
            print("All layer normalization tests passed!")
            print("=" * 60)

    except Exception as e:
        print(f"\nError: {e}")
        raise


if __name__ == "__main__":
    main()

