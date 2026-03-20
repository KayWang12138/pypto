#!/usr/bin/env python3
# coding: utf-8
"""
npu_max / npu_min 算子实现

数学公式：
    npu_max(x, dim) -> (max_values, max_indices)
    npu_min(x, dim) -> (min_values, min_indices)

实现方式：
1. torch_npu: 使用 torch_npu.npu_max / torch_npu.npu_min
2. golden: 使用 numpy.max / numpy.min 配合 argmax/argmin
3. pypto: 使用 pypto.amax / pypto.amin (仅返回值)
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
        print("Please set the environment variable TILE_FWK_DEVICE_ID before running:")
        print("  export TILE_FWK_DEVICE_ID=0")
        return None
    try:
        return int(os.environ['TILE_FWK_DEVICE_ID'])
    except ValueError:
        print(f"ERROR: TILE_FWK_DEVICE_ID must be an integer")
        return None


# ============================================================================
# Golden 实现
# ============================================================================

def max_golden_numpy(x: np.ndarray, dim: int, keepdim: bool = False):
    max_vals = np.max(x, axis=dim, keepdims=keepdim)
    max_indices = np.argmax(x, axis=dim)
    if keepdim:
        max_indices = np.expand_dims(max_indices, axis=dim)
    return max_vals, max_indices.astype(np.int32)


def min_golden_numpy(x: np.ndarray, dim: int, keepdim: bool = False):
    min_vals = np.min(x, axis=dim, keepdims=keepdim)
    min_indices = np.argmin(x, axis=dim)
    if keepdim:
        min_indices = np.expand_dims(min_indices, axis=dim)
    return min_vals, min_indices.astype(np.int32)


# ============================================================================
# PyPTO 实现
# ============================================================================

@pypto.frontend.jit
def amax_kernel(
    x: pypto.Tensor(),
    out: pypto.Tensor(),
    dim: int,
    keepdim: bool,
):
    ndim = len(x.shape)
    tile_shapes = [32 for _ in range(ndim)]
    pypto.set_vec_tile_shapes(*tile_shapes)
    out[:] = pypto.amax(x, dim=dim, keepdim=keepdim)


@pypto.frontend.jit
def amin_kernel(
    x: pypto.Tensor(),
    out: pypto.Tensor(),
    dim: int,
    keepdim: bool,
):
    ndim = len(x.shape)
    tile_shapes = [32 for _ in range(ndim)]
    pypto.set_vec_tile_shapes(*tile_shapes)
    out[:] = pypto.amin(x, dim=dim, keepdim=keepdim)


# ============================================================================
# 测试与验证
# ============================================================================

def compare_results(torch_npu_result, golden_result, pypto_result, rtol=1e-3, atol=1e-3):
    print("\n精度对比结果:")
    
    results = {
        "torch_npu vs golden": (torch_npu_result, golden_result),
        "pypto vs golden": (pypto_result, golden_result),
        "pypto vs torch_npu": (pypto_result, torch_npu_result),
    }
    
    all_passed = True
    for name, (a, b) in results.items():
        max_diff = np.max(np.abs(a - b))
        mean_diff = np.mean(np.abs(a - b))
        try:
            assert_allclose(a.flatten(), b.flatten(), rtol=rtol, atol=atol)
            status = "✓ 通过"
        except AssertionError:
            status = "✗ 失败"
            all_passed = False
        print(f"  {name}: 最大误差={max_diff:.6e}, 平均误差={mean_diff:.6e}, {status}")
    
    return all_passed


def test_max_basic(device_id=None, run_mode="npu"):
    print("=" * 60)
    print("Test: npu_max 基础功能测试")
    print("=" * 60)
    
    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'
    
    shape = (2, 2, 2, 2)
    dim = 2
    keepdim = False
    
    np.random.seed(42)
    x_np = np.random.randn(*shape).astype(np.float32)
    x_torch = torch.from_numpy(x_np).to(device)
    
    print(f"\n输入 shape: {shape}")
    print(f"dim: {dim}, keepdim: {keepdim}")
    
    torch_npu_vals = None
    golden_vals, golden_indices = max_golden_numpy(x_np, dim, keepdim)
    print("[golden 实现完成]")
    
    if run_mode == "npu":
        import torch_npu
        torch_npu_vals, torch_npu_indices = torch_npu.npu_max(x_torch, dim, keepdim)
        torch_npu_vals = torch_npu_vals.cpu().numpy()
        print("[torch_npu 实现完成]")
    
    out_shape = list(shape)
    out_shape.pop(dim)
    out_shape = tuple(out_shape)
    
    pypto_vals = torch.empty(out_shape, dtype=torch.float32, device=device)
    amax_kernel(x_torch, pypto_vals, dim, keepdim)
    pypto_vals = pypto_vals.cpu().numpy()
    print("[pypto 实现完成] (仅返回值，不返回索引)")
    
    if torch_npu_vals is not None:
        all_passed = compare_results(torch_npu_vals, golden_vals, pypto_vals)
    else:
        max_diff = np.max(np.abs(golden_vals - pypto_vals))
        print(f"\n[golden vs pypto] 最大误差: {max_diff:.6e}")
        all_passed = max_diff < 1e-2
    
    print("\n✓ 测试通过" if all_passed else "\n✗ 测试失败")
    return all_passed


def test_min_basic(device_id=None, run_mode="npu"):
    print("\n" + "=" * 60)
    print("Test: npu_min 基础功能测试")
    print("=" * 60)
    
    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'
    
    shape = (2, 2, 2, 2)
    dim = 2
    keepdim = False
    
    np.random.seed(123)
    x_np = np.random.randn(*shape).astype(np.float32)
    x_torch = torch.from_numpy(x_np).to(device)
    
    print(f"\n输入 shape: {shape}")
    print(f"dim: {dim}, keepdim: {keepdim}")
    
    torch_npu_vals = None
    golden_vals, golden_indices = min_golden_numpy(x_np, dim, keepdim)
    print("[golden 实现完成]")
    
    if run_mode == "npu":
        import torch_npu
        torch_npu_vals, torch_npu_indices = torch_npu.npu_min(x_torch, dim, keepdim)
        torch_npu_vals = torch_npu_vals.cpu().numpy()
        print("[torch_npu 实现完成]")
    
    out_shape = list(shape)
    out_shape.pop(dim)
    out_shape = tuple(out_shape)
    
    pypto_vals = torch.empty(out_shape, dtype=torch.float32, device=device)
    amin_kernel(x_torch, pypto_vals, dim, keepdim)
    pypto_vals = pypto_vals.cpu().numpy()
    print("[pypto 实现完成] (仅返回值，不返回索引)")
    
    if torch_npu_vals is not None:
        all_passed = compare_results(torch_npu_vals, golden_vals, pypto_vals)
    else:
        max_diff = np.max(np.abs(golden_vals - pypto_vals))
        print(f"\n[golden vs pypto] 最大误差: {max_diff:.6e}")
        all_passed = max_diff < 1e-2
    
    print("\n✓ 测试通过" if all_passed else "\n✗ 测试失败")
    return all_passed


def test_max_keepdim(device_id=None, run_mode="npu"):
    print("\n" + "=" * 60)
    print("Test: npu_max keepdim=True 测试")
    print("=" * 60)
    
    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'
    
    shape = (2, 3, 4)
    dim = 1
    keepdim = True
    
    np.random.seed(456)
    x_np = np.random.randn(*shape).astype(np.float32)
    x_torch = torch.from_numpy(x_np).to(device)
    
    print(f"\n输入 shape: {shape}")
    print(f"dim: {dim}, keepdim: {keepdim}")
    
    torch_npu_vals = None
    golden_vals, golden_indices = max_golden_numpy(x_np, dim, keepdim)
    print("[golden 实现完成]")
    
    if run_mode == "npu":
        import torch_npu
        torch_npu_vals, torch_npu_indices = torch_npu.npu_max(x_torch, dim, keepdim)
        torch_npu_vals = torch_npu_vals.cpu().numpy()
        print("[torch_npu 实现完成]")
    
    out_shape = list(shape)
    out_shape[dim] = 1
    out_shape = tuple(out_shape)
    
    pypto_vals = torch.empty(out_shape, dtype=torch.float32, device=device)
    amax_kernel(x_torch, pypto_vals, dim, keepdim)
    pypto_vals = pypto_vals.cpu().numpy()
    print("[pypto 实现完成]")
    
    if torch_npu_vals is not None:
        all_passed = compare_results(torch_npu_vals, golden_vals, pypto_vals)
    else:
        max_diff = np.max(np.abs(golden_vals - pypto_vals))
        print(f"\n[golden vs pypto] 最大误差: {max_diff:.6e}")
        all_passed = max_diff < 1e-2
    
    print("\n✓ 测试通过" if all_passed else "\n✗ 测试失败")
    return all_passed


def main():
    parser = argparse.ArgumentParser(description="npu_max/npu_min 算子测试")
    parser.add_argument('test_case', nargs='?', default='all',
                        choices=['all', 'max', 'min', 'keepdim'])
    parser.add_argument('--run_mode', default='npu', choices=['npu'])
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("npu_max / npu_min 算子测试")
    print("=" * 60)
    
    device_id = None
    if args.run_mode == "npu":
        device_id = get_device_id()
        if device_id is None:
            return
        import torch_npu
        torch.npu.set_device(device_id)
    
    test_cases = {
        'max': test_max_basic,
        'min': test_min_basic,
        'keepdim': test_max_keepdim,
    }
    
    if args.test_case == 'all':
        cases_to_run = list(test_cases.items())
    else:
        cases_to_run = [(args.test_case, test_cases[args.test_case])]
    
    all_passed = True
    for name, test_func in cases_to_run:
        try:
            passed = test_func(device_id, args.run_mode)
            all_passed = all_passed and passed
        except Exception as e:
            print(f"\n测试 {name} 异常: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False
    
    print("\n" + "=" * 60)
    print("所有测试通过!" if all_passed else "存在测试失败!")
    print("=" * 60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())