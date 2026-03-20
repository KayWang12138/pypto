#!/usr/bin/env python3
# coding: utf-8
"""
npu_swiglu 算子实现
公式: swiglu(x, dim) = swish(A) * B = A * sigmoid(A) * B
其中 A, B 是沿 dim 维度切分的两半
"""

import os
import sys
import argparse
import numpy as np
from numpy.testing import assert_allclose

import torch
import pypto

def get_device_id():
    if 'TILE_FWK_DEVICE_ID' not in os.environ:
        print("export TILE_FWK_DEVICE_ID=0")
        return None
    return int(os.environ['TILE_FWK_DEVICE_ID'])

def swiglu_golden_numpy(x: np.ndarray, dim: int = -1) -> np.ndarray:
    half = x.shape[dim] // 2
    idx = [slice(None)] * x.ndim
    idx[dim] = slice(0, half)
    a = x[tuple(idx)]
    idx[dim] = slice(half, None)
    b = x[tuple(idx)]
    swish_a = a * (1.0 / (1.0 + np.exp(-a)))
    return swish_a * b

@pypto.frontend.jit
def swiglu_kernel(x: pypto.Tensor(), out: pypto.Tensor(), half_dim: int):
    pypto.set_vec_tile_shapes(32, 128)
    a = x[..., :half_dim]
    b = x[..., half_dim:]
    swish_a = a * pypto.sigmoid(a)
    out[:] = swish_a * b

def compare_results(torch_npu_result, golden_result, pypto_result, rtol=1e-3, atol=1e-3):
    results = {
        "torch_npu vs golden": (torch_npu_result, golden_result),
        "pypto vs golden": (pypto_result, golden_result),
        "pypto vs torch_npu": (pypto_result, torch_npu_result),
    }
    all_passed = True
    for name, (a, b) in results.items():
        max_diff = np.max(np.abs(a - b))
        try:
            assert_allclose(a.flatten(), b.flatten(), rtol=rtol, atol=atol)
            status = "✓"
        except:
            status = "✗"
            all_passed = False
        print(f"  {name}: max_diff={max_diff:.6e} {status}")
    return all_passed

def test_basic(device_id=None, run_mode="npu"):
    print("=" * 60)
    print("Test: npu_swiglu")
    print("=" * 60)
    device = f'npu:{device_id}' if (run_mode == "npu" and device_id) else 'cpu'
    shape = (2, 32, 6, 6)
    dim = -1
    
    np.random.seed(42)
    x_np = np.random.randn(*shape).astype(np.float32)
    x_torch = torch.from_numpy(x_np)
    x_npu = x_torch.npu() if run_mode == "npu" else x_torch
    
    print(f"shape: {shape}, dim: {dim}")
    
    torch_npu_result = None
    golden_result = swiglu_golden_numpy(x_np, dim)
    print("[golden ✓]")
    
    if run_mode == "npu":
        import torch_npu
        torch_npu_result = torch_npu.npu_swiglu(x_npu, dim=dim).cpu().numpy()
        print("[torch_npu ✓]")
    
    half_dim = shape[dim] // 2
    out_shape = list(shape)
    out_shape[dim] = half_dim
    pypto_out = torch.empty(out_shape, dtype=torch.float32, device=x_npu.device)
    swiglu_kernel(x_npu, pypto_out, half_dim)
    pypto_result = pypto_out.cpu().numpy()
    print("[pypto ✓]")
    
    if torch_npu_result is not None:
        all_passed = compare_results(torch_npu_result, golden_result, pypto_result)
    else:
        max_diff = np.max(np.abs(golden_result - pypto_result))
        all_passed = max_diff < 1e-2
    
    print("✓ PASS" if all_passed else "✗ FAIL")
    return all_passed

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_mode', default='npu')
    args = parser.parse_args()
    
    device_id = None
    if args.run_mode == "npu":
        device_id = get_device_id()
        if device_id is None: return
        import torch_npu
        torch.npu.set_device(device_id)
    
    all_passed = test_basic(device_id, args.run_mode)
    print("\n" + "=" * 60)
    print("PASS" if all_passed else "FAIL")
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())