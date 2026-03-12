#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
Basic Operations Quick-Start for PyPTO

A concise overview of core PyPTO capabilities. Each example demonstrates one
key category from the beginner tutorials:

1. Tensor creation and properties
2. Element-wise operations (add, mul)
3. Matrix multiplication (matmul)
4. Reduction operations (sum)
5. Tiling configuration (vec / cube tile shapes)
6. Transform operations (view + assemble)

For more detailed examples of each category, see the corresponding files:
  - basic/tensor_creation.py, basic/symbolic_scalar.py
  - compute/elementwise_ops.py, compute/matmul_ops.py, compute/reduce_ops.py
  - tiling/tiling_config.py
  - transform/transform_ops.py
"""

import os
import sys
import argparse
import pypto
import torch
import numpy as np
from numpy.testing import assert_allclose


def get_device_id():
    """
    Get and validate TILE_FWK_DEVICE_ID from environment variable.

    Returns:
        int: The device ID if valid, None otherwise.
    """
    if 'TILE_FWK_DEVICE_ID' not in os.environ:
        print("Please set the environment variable TILE_FWK_DEVICE_ID before running:")
        print("  export TILE_FWK_DEVICE_ID=0")
        return None

    try:
        device_id = int(os.environ['TILE_FWK_DEVICE_ID'])
        return device_id
    except ValueError:
        print(f"ERROR: TILE_FWK_DEVICE_ID must be an integer, got: {os.environ['TILE_FWK_DEVICE_ID']}")
        return None


# ============================================================================
# 1. Tensor Creation
# ============================================================================

def test_tensor_creation(device_id=None):
    """Demonstrate tensor creation and property access."""
    print("=" * 60)
    print("Example 1: Tensor Creation")
    print("=" * 60)

    tensor = pypto.Tensor([4, 4], pypto.DT_FP16, "my_tensor")
    print(f"  name={tensor.name}, shape={tensor.shape}, dtype={tensor.dtype}, "
          f"format={tensor.format}, dim={tensor.dim}")
    print("✓ Tensor creation completed successfully\n")


# ============================================================================
# 2. Element-wise Operations
# ============================================================================

@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def elementwise_kernel(
    a: pypto.Tensor([], pypto.DT_FP16),
    b: pypto.Tensor([], pypto.DT_FP16),
    out: pypto.Tensor([], pypto.DT_FP16),
):
    pypto.set_vec_tile_shapes(8, 8)
    out.move(pypto.mul(pypto.add(a, b), 2.0))


def test_elementwise_ops(device_id=None):
    """Element-wise add + scalar mul: out = (a + b) * 2."""
    print("=" * 60)
    print("Example 2: Element-wise Operations")
    print("=" * 60)

    shape = (8, 8)
    device = f'npu:{device_id}'
    a = torch.randn(shape, dtype=torch.float16, device=device)
    b = torch.randn(shape, dtype=torch.float16, device=device)
    out = torch.empty(shape, dtype=torch.float16, device=device)
    elementwise_kernel(a, b, out)

    expected = (a + b) * 2.0
    max_diff = (out - expected).abs().max().item()
    print(f"  Max difference: {max_diff:.6f}")
    assert max_diff < 1e-2, "Result mismatch!"
    print("✓ Element-wise operations completed successfully\n")


# ============================================================================
# 3. Matrix Multiplication
# ============================================================================
@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def matmul_kernel(
    a: pypto.Tensor([], pypto.DT_BF16),
    b: pypto.Tensor([], pypto.DT_BF16),
    out: pypto.Tensor([], pypto.DT_BF16),
):
    pypto.set_cube_tile_shapes([32, 32], [64, 64], [64, 64])
    out.move(pypto.matmul(a, b, a.dtype))

def test_matmul(device_id=None):
    """Basic matrix multiplication: C = A @ B."""
    print("=" * 60)
    print("Example 3: Matrix Multiplication")
    print("=" * 60)

    m, k, n = 64, 128, 64

    device = f'npu:{device_id}'
    a = torch.randn(m, k, dtype=torch.bfloat16, device=device)
    b = torch.randn(k, n, dtype=torch.bfloat16, device=device)
    out = torch.empty((m, n), dtype=torch.bfloat16, device=device)
    matmul_kernel(a, b, out)

    expected = torch.matmul(a, b)
    max_diff = (out - expected).abs().max().item()
    print(f"  Max difference: {max_diff:.6f}")
    assert max_diff < 1e-1, "Result mismatch!"
    print("✓ Matrix multiplication completed successfully\n")


# ============================================================================
# 4. Reduction Operations
# ============================================================================
@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def sum_kernel(
    a: pypto.Tensor([], pypto.DT_FP32),
    out: pypto.Tensor([], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(8, 8)
    out.move(pypto.sum(a, dim=-1, keepdim=False))


def test_reduce_ops(device_id=None):
    """Reduction: sum along last dimension."""
    print("=" * 60)
    print("Example 4: Reduction Operations (sum)")
    print("=" * 60)

    device = f'npu:{device_id}'
    a = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32, device=device)
    out = torch.empty((2,), dtype=torch.float32, device=device)
    sum_kernel(a, out)

    expected = torch.tensor([6, 15], dtype=torch.float32, device=device)
    assert_allclose(out.cpu().numpy(), expected.cpu().numpy(), rtol=1e-3, atol=1e-3)
    print(f"  Input:    {a.tolist()}")
    print(f"  Output:   {out.tolist()}")
    print("✓ Reduction operations completed successfully\n")


# ============================================================================
# 5. Tiling Configuration
# ============================================================================
@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def tiled_add_kernel(
    a: pypto.Tensor(),
    b: pypto.Tensor(),
    out: pypto.Tensor(),
):
    pypto.set_vec_tile_shapes(2, 8)
    out.move(pypto.add(a, b))


def test_tiling_config(device_id=None):
    """Show how to set vec and cube tile shapes."""
    print("=" * 60)
    print("Example 5: Tiling Configuration")
    print("=" * 60)


    device = f'npu:{device_id}'
    a = torch.ones((2, 8), dtype=torch.float32, device=device)
    b = torch.ones((2, 8), dtype=torch.float32, device=device)
    out = torch.empty((2, 8), dtype=torch.float32, device=device)
    tiled_add_kernel(a, b, out)

    expected = a + b
    assert_allclose(out.cpu().numpy(), expected.cpu().numpy(), rtol=1e-3, atol=1e-3)
    print(f"  vec_tile_shapes set to (2, 8)")
    print("✓ Tiling configuration completed successfully\n")


# ============================================================================
# 6. Transform Operations (view + assemble)
# ============================================================================
@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def view_assemble_kernel(
    x: pypto.Tensor(),
    output: pypto.Tensor(),
    tile_h: int,
    tile_w: int,
    height: int,
    width: int,
):
    pypto.set_vec_tile_shapes(tile_h, tile_w)
    h_tiles = height // tile_h
    w_tiles = width // tile_w
    for h_idx in pypto.loop(h_tiles, name="h_loop", idx_name="h_idx"):
        for w_idx in pypto.loop(w_tiles, name="w_loop", idx_name="w_idx"):
            h_off = h_idx * tile_h
            w_off = w_idx * tile_w
            tile = pypto.view(x, [tile_h, tile_w], [h_off, w_off])
            result = pypto.mul(tile, 2.0)
            pypto.assemble(result, [h_off, w_off], output)


def test_transform_ops(device_id=None):
    """Loop-based tiling with view and assemble: out = input * 2."""
    print("=" * 60)
    print("Example 6: Transform Operations (view + assemble)")
    print("=" * 60)

    height, width = 64, 64
    tile_h, tile_w = 32, 32

    device = f'npu:{device_id}'
    x = torch.randn((height, width), dtype=torch.float16, device=device)
    out = torch.empty((height, width), dtype=torch.float16, device=device)
    view_assemble_kernel(x, out, tile_h, tile_w, height, width)

    expected = x * 2.0
    max_diff = (out - expected).abs().max().item()
    print(f"  Max difference: {max_diff:.6f}")
    assert max_diff < 1e-2, "Result mismatch!"
    print("✓ Transform operations completed successfully\n")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="PyPTO Basic Operations Quick-Start",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        'example_id', type=str, nargs='?',
        help='Run a specific case. If omitted, all cases run.'
    )
    parser.add_argument('--list', action='store_true', help='List available examples')
    parser.add_argument(
        '--run_mode',
        type=str,
        nargs='?',
        default='npu',
        choices=["npu"],
        help='Run mode, currently only support npu.'
    )
    args = parser.parse_args()

    examples = {
        "tensor_creation::test_tensor_creation": {
            'name': 'Tensor Creation',
            'function': test_tensor_creation,
        },
        "elementwise_ops::test_elementwise_ops": {
            'name': 'Element-wise Operations',
            'function': test_elementwise_ops,
        },
        "matmul::test_matmul": {
            'name': 'Matrix Multiplication',
            'function': test_matmul,
        },
        "reduce_ops::test_reduce_ops": {
            'name': 'Reduction Operations',
            'function': test_reduce_ops,
        },
        "tiling_config::test_tiling_config": {
            'name': 'Tiling Configuration',
            'function': test_tiling_config,
        },
        "transform_ops::test_transform_ops": {
            'name': 'Transform Operations',
            'function': test_transform_ops,
        },
    }

    if args.list:
        print("\nAvailable Examples:\n")
        for ex_id, ex_info in examples.items():
            print(f"  {ex_id}: {ex_info['name']}")
        return

    if args.example_id is not None:
        if args.example_id not in examples:
            print(f"ERROR: Invalid example ID: {args.example_id}")
            print(f"Valid IDs: {', '.join(examples.keys())}")
            sys.exit(1)

    device_id = get_device_id()
    if device_id is None:
        sys.exit(1)
    import torch_npu
    torch.npu.set_device(device_id)

    examples_to_run = (
        [(args.example_id, examples[args.example_id])]
        if args.example_id else list(examples.items())
    )

    print("\n" + "=" * 60)
    print("PyPTO Basic Operations Quick-Start")
    print("=" * 60 + "\n")

    try:
        for _, ex_info in examples_to_run:
            ex_info['function'](device_id)

        if len(examples_to_run) > 1:
            print("=" * 60)
            print("All examples completed successfully!")
            print("=" * 60)
    except Exception as e:
        print(f"\nError: {e}")
        raise


if __name__ == "__main__":
    main()
============================================================
Testing Static FFN with GELU Activation
============================================================
Input shape: torch.Size([16, 128])
Gate weight shape: torch.Size([128, 1024])
up weight shape: torch.Size([128, 1024])
Down weight shape: torch.Size([1024, 128])
Creating new kmodule for key: ((b't\x00\xa0\x01|\x05j\x02d\x01\x19\x00|\x05j\x02d\x01\x19\x00g\x02|\x05j\x02d\x02\x19\x00|\x05j\x02d\x02\x19\x00g\x02|\x05j\x02d\x03\x19\x00|\x05j\x02d\x03\x19\x00g\x02\xa1\x03\x01\x00t\x00j\x03|\x05j\x04\x8e\x00\x01\x00t\x00\xa0\x05|\x00|\x01|\x05j\x06\xa1\x03}\x06|\x05j\x07d\x04k\x02r\x80|\x05j\x08d\x05k\x02r\x80t\t|\x04|\x00|\x01|\x03|\x05\x83\x05\x01\x00n^|\x05j\x08d\x05k\x02r\x94t\n|\x06\x83\x01}\x07nJ|\x05j\x08d\x06k\x02r\xbat\x00\xa0\x05|\x00|\x02|\x05j\x06\xa1\x03}\x02t\x0b|\x06|\x02\x83\x02}\x07n$|\x05j\x08d\x07k\x02r\xcet\x0c|\x06\x83\x01}\x07n\x10t\rd\x08|\x05j\x08\x9b\x00\x9d\x02\x83\x01\x82\x01|\x05j\x07d\tk\x02\x90\x01r\x10t\x00j\x05|\x07|\x03|\x05j\x06d\td\n\x8d\x04}\x08t\x00\xa0\x0e|\x08d\x01d\x01g\x02|\x04\xa1\x03\x01\x00d\x00S\x00', (None, 0, 1, 2, True, 'gelu', 'swiglu', 'relu', 'Unsupported activation: ', False, ('b_trans',)), ('pypto', 'set_cube_tile_shapes', 'cube_tile_shape', 'set_vec_tile_shapes', 'vec_tile_shape', 'matmul', 'dtype', 'use_dynamic_shape', 'activation', 'dynamic_gelu_activation_core', 'gelu_activation_core', 'swiglu_activation_core', 'relu_activation_core', 'ValueError', 'assemble'), ('hidden_states', 'gate_proj_weight', 'up_proj_weight', 'down_proj_weight', 'output', 'config', 'gate', 'activated', 'result')), (None, None, None, frozenset({('run_mode', '0')}), None, None), frozenset({('get_device_id', '<function get_device_id at 0xffffac667c10>'), ('os', "<module 'os' from '/usr/lib64/python3.9/os.py'>"), ('dataclass', '<function dataclass at 0xffffac49ca60>'), ('gelu_torch', '<function gelu_torch at 0xfffd25789160>'), ('__name__', '__main__'), ('np', "<module 'numpy' from '/usr/lib64/python3.9/site-packages/numpy/__init__.py'>"), ('torch', "<module 'torch' from '/usr/local/lib64/python3.9/site-packages/torch/__init__.py'>"), ('math', "<module 'math' from '/usr/lib64/python3.9/lib-dynload/math.cpython-39-aarch64-linux-gnu.so'>"), ('pytest', "<module 'pytest' from '/data/l00611583/.local/lib/python3.9/site-packages/pytest/__init__.py'>"), ('swiglu_activation_core', '<function swiglu_activation_core at 0xfffd257894c0>'), ('F_NEGA_1', '-1.0'), ('__file__', '/data/l00611583/zjh/pypto_open/examples/02_intermediate/basic_nn/ffn/ffn_module.py'), ('__loader__', '<_frozen_importlib_external.SourceFileLoader object at 0xffffac8a7c10>'), ('__package__', None), ('pypto', "<module 'pypto' from '/data/l00611583/zjh/pypto_open/build_out/pypto/__init__.py'>"), ('ceil_div', '<function ceil_div at 0xfffd25789310>'), ('dynamic_gelu_activation_core', '<function dynamic_gelu_activation_core at 0xfffd25789550>'), ('__builtins__', "<module 'builtins' (built-in)>"), ('argparse', "<module 'argparse' from '/usr/lib64/python3.9/argparse.py'>"), ('sys', "<module 'sys' (built-in)>"), ('__doc__', '\nTest and Example Usage of FFN Module\n\nThis script demonstrates how to use the FFN module with different configurations\nand validates the implementation against PyTorch reference.\n'), ('__spec__', None), ('Literal', 'typing.Literal'), ('F_1', '1.0'), ('GELU_COEFF', '1.702'), ('gelu_activation_core', '<function gelu_activation_core at 0xfffd25789430>'), ('relu_activation_core', '<function relu_activation_core at 0xfffd257893a0>'), ('swiglu_torch', '<function swiglu_torch at 0xfffd25789280>'), ('FFNConfig', "<class '__main__.FFNConfig'>"), ('__annotations__', frozenset()), ('assert_allclose', '<function assert_allclose at 0xfffd2577bb80>'), ('__cached__', None)}), frozenset({('config', "FFNConfig(batch_size=16, hidden_size=128, intermediate_size=1024, activation='gelu', dtype=DataType.DT_BF16, use_dynamic_shape=False, vec_tile_shape=(16, 32), cube_tile_shape=(16, 32, 32), basic_batch=32, run_mode=<RunMode.NPU: 0>)")}))
Output shape: torch.Size([16, 128])
Output range: [-13.5000, 11.1250]
✓ Static FFN with GELU test completed

Running Example ffn_static_swiglu::test_ffn_static_swiglu: Static FFN with SwiGLU
============================================================
Testing Static FFN with SwiGLU Activation
============================================================
Creating new kmodule for key: ((b't\x00\xa0\x01|\x05j\x02d\x01\x19\x00|\x05j\x02d\x01\x19\x00g\x02|\x05j\x02d\x02\x19\x00|\x05j\x02d\x02\x19\x00g\x02|\x05j\x02d\x03\x19\x00|\x05j\x02d\x03\x19\x00g\x02\xa1\x03\x01\x00t\x00j\x03|\x05j\x04\x8e\x00\x01\x00t\x00\xa0\x05|\x00|\x01|\x05j\x06\xa1\x03}\x06|\x05j\x07d\x04k\x02r\x80|\x05j\x08d\x05k\x02r\x80t\t|\x04|\x00|\x01|\x03|\x05\x83\x05\x01\x00n^|\x05j\x08d\x05k\x02r\x94t\n|\x06\x83\x01}\x07nJ|\x05j\x08d\x06k\x02r\xbat\x00\xa0\x05|\x00|\x02|\x05j\x06\xa1\x03}\x02t\x0b|\x06|\x02\x83\x02}\x07n$|\x05j\x08d\x07k\x02r\xcet\x0c|\x06\x83\x01}\x07n\x10t\rd\x08|\x05j\x08\x9b\x00\x9d\x02\x83\x01\x82\x01|\x05j\x07d\tk\x02\x90\x01r\x10t\x00j\x05|\x07|\x03|\x05j\x06d\td\n\x8d\x04}\x08t\x00\xa0\x0e|\x08d\x01d\x01g\x02|\x04\xa1\x03\x01\x00d\x00S\x00', (None, 0, 1, 2, True, 'gelu', 'swiglu', 'relu', 'Unsupported activation: ', False, ('b_trans',)), ('pypto', 'set_cube_tile_shapes', 'cube_tile_shape', 'set_vec_tile_shapes', 'vec_tile_shape', 'matmul', 'dtype', 'use_dynamic_shape', 'activation', 'dynamic_gelu_activation_core', 'gelu_activation_core', 'swiglu_activation_core', 'relu_activation_core', 'ValueError', 'assemble'), ('hidden_states', 'gate_proj_weight', 'up_proj_weight', 'down_proj_weight', 'output', 'config', 'gate', 'activated', 'result')), (None, None, None, frozenset({('run_mode', '0')}), None, None), frozenset({('get_device_id', '<function get_device_id at 0xffffac667c10>'), ('os', "<module 'os' from '/usr/lib64/python3.9/os.py'>"), ('dataclass', '<function dataclass at 0xffffac49ca60>'), ('gelu_torch', '<function gelu_torch at 0xfffd25789160>'), ('__name__', '__main__'), ('np', "<module 'numpy' from '/usr/lib64/python3.9/site-packages/numpy/__init__.py'>"), ('torch', "<module 'torch' from '/usr/local/lib64/python3.9/site-packages/torch/__init__.py'>"), ('math', "<module 'math' from '/usr/lib64/python3.9/lib-dynload/math.cpython-39-aarch64-linux-gnu.so'>"), ('pytest', "<module 'pytest' from '/data/l00611583/.local/lib/python3.9/site-packages/pytest/__init__.py'>"), ('swiglu_activation_core', '<function swiglu_activation_core at 0xfffd257894c0>'), ('F_NEGA_1', '-1.0'), ('__file__', '/data/l00611583/zjh/pypto_open/examples/02_intermediate/basic_nn/ffn/ffn_module.py'), ('__loader__', '<_frozen_importlib_external.SourceFileLoader object at 0xffffac8a7c10>'), ('__package__', None), ('pypto', "<module 'pypto' from '/data/l00611583/zjh/pypto_open/build_out/pypto/__init__.py'>"), ('ceil_div', '<function ceil_div at 0xfffd25789310>'), ('dynamic_gelu_activation_core', '<function dynamic_gelu_activation_core at 0xfffd25789550>'), ('__builtins__', "<module 'builtins' (built-in)>"), ('argparse', "<module 'argparse' from '/usr/lib64/python3.9/argparse.py'>"), ('sys', "<module 'sys' (built-in)>"), ('__doc__', '\nTest and Example Usage of FFN Module\n\nThis script demonstrates how to use the FFN module with different configurations\nand validates the implementation against PyTorch reference.\n'), ('__spec__', None), ('Literal', 'typing.Literal'), ('F_1', '1.0'), ('GELU_COEFF', '1.702'), ('gelu_activation_core', '<function gelu_activation_core at 0xfffd25789430>'), ('relu_activation_core', '<function relu_activation_core at 0xfffd257893a0>'), ('swiglu_torch', '<function swiglu_torch at 0xfffd25789280>'), ('FFNConfig', "<class '__main__.FFNConfig'>"), ('__annotations__', frozenset()), ('assert_allclose', '<function assert_allclose at 0xfffd2577bb80>'), ('__cached__', None)}), frozenset({('config', "FFNConfig(batch_size=16, hidden_size=128, intermediate_size=1024, activation='swiglu', dtype=DataType.DT_BF16, use_dynamic_shape=False, vec_tile_shape=(16, 32), cube_tile_shape=(16, 32, 32), basic_batch=32, run_mode=<RunMode.NPU: 0>)")}))
Input shape: torch.Size([16, 128])
Gate weight shape: torch.Size([128, 1024])
Up weight shape: torch.Size([128, 1024])
Down weight shape: torch.Size([1024, 128])
Output shape: torch.Size([16, 128])
Output range: [-8.6250, 9.3125]
✓ Static FFN with SwiGLU test completed

Running Example ffn_static_relu::test_ffn_static_relu: Static FFN with ReLU
============================================================
Testing Static FFN with ReLU Activation
============================================================
Creating new kmodule for key: ((b't\x00\xa0\x01|\x05j\x02d\x01\x19\x00|\x05j\x02d\x01\x19\x00g\x02|\x05j\x02d\x02\x19\x00|\x05j\x02d\x02\x19\x00g\x02|\x05j\x02d\x03\x19\x00|\x05j\x02d\x03\x19\x00g\x02\xa1\x03\x01\x00t\x00j\x03|\x05j\x04\x8e\x00\x01\x00t\x00\xa0\x05|\x00|\x01|\x05j\x06\xa1\x03}\x06|\x05j\x07d\x04k\x02r\x80|\x05j\x08d\x05k\x02r\x80t\t|\x04|\x00|\x01|\x03|\x05\x83\x05\x01\x00n^|\x05j\x08d\x05k\x02r\x94t\n|\x06\x83\x01}\x07nJ|\x05j\x08d\x06k\x02r\xbat\x00\xa0\x05|\x00|\x02|\x05j\x06\xa1\x03}\x02t\x0b|\x06|\x02\x83\x02}\x07n$|\x05j\x08d\x07k\x02r\xcet\x0c|\x06\x83\x01}\x07n\x10t\rd\x08|\x05j\x08\x9b\x00\x9d\x02\x83\x01\x82\x01|\x05j\x07d\tk\x02\x90\x01r\x10t\x00j\x05|\x07|\x03|\x05j\x06d\td\n\x8d\x04}\x08t\x00\xa0\x0e|\x08d\x01d\x01g\x02|\x04\xa1\x03\x01\x00d\x00S\x00', (None, 0, 1, 2, True, 'gelu', 'swiglu', 'relu', 'Unsupported activation: ', False, ('b_trans',)), ('pypto', 'set_cube_tile_shapes', 'cube_tile_shape', 'set_vec_tile_shapes', 'vec_tile_shape', 'matmul', 'dtype', 'use_dynamic_shape', 'activation', 'dynamic_gelu_activation_core', 'gelu_activation_core', 'swiglu_activation_core', 'relu_activation_core', 'ValueError', 'assemble'), ('hidden_states', 'gate_proj_weight', 'up_proj_weight', 'down_proj_weight', 'output', 'config', 'gate', 'activated', 'result')), (None, None, None, frozenset({('run_mode', '0')}), None, None), frozenset({('get_device_id', '<function get_device_id at 0xffffac667c10>'), ('os', "<module 'os' from '/usr/lib64/python3.9/os.py'>"), ('dataclass', '<function dataclass at 0xffffac49ca60>'), ('gelu_torch', '<function gelu_torch at 0xfffd25789160>'), ('__name__', '__main__'), ('np', "<module 'numpy' from '/usr/lib64/python3.9/site-packages/numpy/__init__.py'>"), ('torch', "<module 'torch' from '/usr/local/lib64/python3.9/site-packages/torch/__init__.py'>"), ('math', "<module 'math' from '/usr/lib64/python3.9/lib-dynload/math.cpython-39-aarch64-linux-gnu.so'>"), ('pytest', "<module 'pytest' from '/data/l00611583/.local/lib/python3.9/site-packages/pytest/__init__.py'>"), ('swiglu_activation_core', '<function swiglu_activation_core at 0xfffd257894c0>'), ('F_NEGA_1', '-1.0'), ('__file__', '/data/l00611583/zjh/pypto_open/examples/02_intermediate/basic_nn/ffn/ffn_module.py'), ('__loader__', '<_frozen_importlib_external.SourceFileLoader object at 0xffffac8a7c10>'), ('__package__', None), ('pypto', "<module 'pypto' from '/data/l00611583/zjh/pypto_open/build_out/pypto/__init__.py'>"), ('ceil_div', '<function ceil_div at 0xfffd25789310>'), ('dynamic_gelu_activation_core', '<function dynamic_gelu_activation_core at 0xfffd25789550>'), ('__builtins__', "<module 'builtins' (built-in)>"), ('argparse', "<module 'argparse' from '/usr/lib64/python3.9/argparse.py'>"), ('sys', "<module 'sys' (built-in)>"), ('__doc__', '\nTest and Example Usage of FFN Module\n\nThis script demonstrates how to use the FFN module with different configurations\nand validates the implementation against PyTorch reference.\n'), ('__spec__', None), ('Literal', 'typing.Literal'), ('F_1', '1.0'), ('GELU_COEFF', '1.702'), ('gelu_activation_core', '<function gelu_activation_core at 0xfffd25789430>'), ('relu_activation_core', '<function relu_activation_core at 0xfffd257893a0>'), ('swiglu_torch', '<function swiglu_torch at 0xfffd25789280>'), ('FFNConfig', "<class '__main__.FFNConfig'>"), ('__annotations__', frozenset()), ('assert_allclose', '<function assert_allclose at 0xfffd2577bb80>'), ('__cached__', None)}), frozenset({('config', "FFNConfig(batch_size=16, hidden_size=128, intermediate_size=1024, activation='relu', dtype=DataType.DT_FP16, use_dynamic_shape=False, vec_tile_shape=(32, 64), cube_tile_shape=(32, 64, 64), basic_batch=32, run_mode=<RunMode.NPU: 0>)")}))
Input shape: torch.Size([16, 128])
Output shape: torch.Size([16, 128])
Output range: [-14.0547, 16.0625]
Max difference: 0.000000
✓ Static FFN with ReLU test completed

Running Example ffn_dynamic_gelu::test_ffn_dynamic_gelu: Dynamic FFN with GELU
============================================================
Testing Dynamic FFN with GELU Activation
============================================================
Input shape: torch.Size([32, 512]) (dynamic batch size: 32)
Basic batch size: 16
Number of iterations: 2
Output shape: torch.Size([32, 512])
Output range: [-8.9375, 9.5000]
Creating new kmodule for key: ((b't\x00\xa0\x01|\x05j\x02d\x01\x19\x00|\x05j\x02d\x01\x19\x00g\x02|\x05j\x02d\x02\x19\x00|\x05j\x02d\x02\x19\x00g\x02|\x05j\x02d\x03\x19\x00|\x05j\x02d\x03\x19\x00g\x02\xa1\x03\x01\x00t\x00j\x03|\x05j\x04\x8e\x00\x01\x00t\x00\xa0\x05|\x00|\x01|\x05j\x06\xa1\x03}\x06|\x05j\x07d\x04k\x02r\x80|\x05j\x08d\x05k\x02r\x80t\t|\x04|\x00|\x01|\x03|\x05\x83\x05\x01\x00n^|\x05j\x08d\x05k\x02r\x94t\n|\x06\x83\x01}\x07nJ|\x05j\x08d\x06k\x02r\xbat\x00\xa0\x05|\x00|\x02|\x05j\x06\xa1\x03}\x02t\x0b|\x06|\x02\x83\x02}\x07n$|\x05j\x08d\x07k\x02r\xcet\x0c|\x06\x83\x01}\x07n\x10t\rd\x08|\x05j\x08\x9b\x00\x9d\x02\x83\x01\x82\x01|\x05j\x07d\tk\x02\x90\x01r\x10t\x00j\x05|\x07|\x03|\x05j\x06d\td\n\x8d\x04}\x08t\x00\xa0\x0e|\x08d\x01d\x01g\x02|\x04\xa1\x03\x01\x00d\x00S\x00', (None, 0, 1, 2, True, 'gelu', 'swiglu', 'relu', 'Unsupported activation: ', False, ('b_trans',)), ('pypto', 'set_cube_tile_shapes', 'cube_tile_shape', 'set_vec_tile_shapes', 'vec_tile_shape', 'matmul', 'dtype', 'use_dynamic_shape', 'activation', 'dynamic_gelu_activation_core', 'gelu_activation_core', 'swiglu_activation_core', 'relu_activation_core', 'ValueError', 'assemble'), ('hidden_states', 'gate_proj_weight', 'up_proj_weight', 'down_proj_weight', 'output', 'config', 'gate', 'activated', 'result')), (None, None, None, frozenset({('run_mode', '0')}), None, None), frozenset({('get_device_id', '<function get_device_id at 0xffffac667c10>'), ('os', "<module 'os' from '/usr/lib64/python3.9/os.py'>"), ('dataclass', '<function dataclass at 0xffffac49ca60>'), ('gelu_torch', '<function gelu_torch at 0xfffd25789160>'), ('__name__', '__main__'), ('np', "<module 'numpy' from '/usr/lib64/python3.9/site-packages/numpy/__init__.py'>"), ('torch', "<module 'torch' from '/usr/local/lib64/python3.9/site-packages/torch/__init__.py'>"), ('math', "<module 'math' from '/usr/lib64/python3.9/lib-dynload/math.cpython-39-aarch64-linux-gnu.so'>"), ('pytest', "<module 'pytest' from '/data/l00611583/.local/lib/python3.9/site-packages/pytest/__init__.py'>"), ('swiglu_activation_core', '<function swiglu_activation_core at 0xfffd257894c0>'), ('F_NEGA_1', '-1.0'), ('__file__', '/data/l00611583/zjh/pypto_open/examples/02_intermediate/basic_nn/ffn/ffn_module.py'), ('__loader__', '<_frozen_importlib_external.SourceFileLoader object at 0xffffac8a7c10>'), ('__package__', None), ('pypto', "<module 'pypto' from '/data/l00611583/zjh/pypto_open/build_out/pypto/__init__.py'>"), ('ceil_div', '<function ceil_div at 0xfffd25789310>'), ('dynamic_gelu_activation_core', '<function dynamic_gelu_activation_core at 0xfffd25789550>'), ('__builtins__', "<module 'builtins' (built-in)>"), ('argparse', "<module 'argparse' from '/usr/lib64/python3.9/argparse.py'>"), ('sys', "<module 'sys' (built-in)>"), ('__doc__', '\nTest and Example Usage of FFN Module\n\nThis script demonstrates how to use the FFN module with different configurations\nand validates the implementation against PyTorch reference.\n'), ('__spec__', None), ('Literal', 'typing.Literal'), ('F_1', '1.0'), ('GELU_COEFF', '1.702'), ('gelu_activation_core', '<function gelu_activation_core at 0xfffd25789430>'), ('relu_activation_core', '<function relu_activation_core at 0xfffd257893a0>'), ('swiglu_torch', '<function swiglu_torch at 0xfffd25789280>'), ('FFNConfig', "<class '__main__.FFNConfig'>"), ('__annotations__', frozenset()), ('assert_allclose', '<function assert_allclose at 0xfffd2577bb80>'), ('__cached__', None)}), frozenset({('config', "FFNConfig(batch_size=32, hidden_size=512, intermediate_size=1024, activation='gelu', dtype=DataType.DT_BF16, use_dynamic_shape=True, vec_tile_shape=(32, 64), cube_tile_shape=(32, 64, 64), basic_batch=16, run_mode=<RunMode.NPU: 0>)")}))
✓ Dynamic FFN with GELU test completed

============================================================
All tests completed!
