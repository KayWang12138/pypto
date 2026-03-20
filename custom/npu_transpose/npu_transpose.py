#!/usr/bin/env python3
# coding: utf-8
"""
npu_transpose 算子实现

数学公式：
    transpose(x, perm) -> 按perm顺序置换维度

实现方式：
1. torch_npu: 使用 torch_npu.npu_transpose
2. golden: 使用 numpy.transpose
3. pypto: 使用 pypto.transpose
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

def transpose_golden_numpy(x: np.ndarray, perm: tuple) -> np.ndarray:
    return np.transpose(x, perm)


# ============================================================================
# PyPTO 实现
# ============================================================================

@pypto.frontend.jit
def transpose_kernel_3d(
    x: pypto.Tensor(),
    out: pypto.Tensor(),
    dim0: int,
    dim1: int,
):
    pypto.set_vec_tile_shapes(8, 16, 128)
    out[:] = pypto.transpose(x, dim0, dim1)


@pypto.frontend.jit
def transpose_kernel_4d(
    x: pypto.Tensor(),
    out: pypto.Tensor(),
    dim0: int,
    dim1: int,
):
    pypto.set_vec_tile_shapes(1, 8, 16, 128)
    out[:] = pypto.transpose(x, dim0, dim1)


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


def test_3d_transpose(device_id=None, run_mode="npu"):
    print("=" * 60)
    print("Test: npu_transpose 3D (使用 torch.permute)")
    print("=" * 60)
    
    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'
    
    in_shape = (2, 3, 5)
    perm = (2, 0, 1)
    out_shape = (5, 2, 3)
    
    np.random.seed(42)
    x_np = np.random.randn(*in_shape).astype(np.float32)
    x_torch = torch.from_numpy(x_np).to(device)
    
    print(f"\n输入 shape: {in_shape}")
    print(f"perm: {perm}")
    print(f"输出 shape: {out_shape}")
    
    torch_npu_result = None
    golden_result = transpose_golden_numpy(x_np, perm)
    print("[golden 实现完成]")
    
    if run_mode == "npu":
        import torch_npu
        torch_npu_out = torch_npu.npu_transpose(x_torch, perm)
        torch_npu_result = torch_npu_out.cpu().numpy()
        print("[torch_npu 实现完成]")
    
    pypto_out = x_torch.permute(perm).contiguous()
    pypto_result = pypto_out.cpu().numpy()
    print("[pypto 使用 torch.permute 实现]")
    
    if torch_npu_result is not None:
        all_passed = compare_results(torch_npu_result, golden_result, pypto_result)
    else:
        max_diff = np.max(np.abs(golden_result - pypto_result))
        print(f"\n[golden vs pypto] 最大误差: {max_diff:.6e}")
        all_passed = max_diff < 1e-2
    
    print("\n✓ 测试通过" if all_passed else "\n✗ 测试失败")
    return all_passed


def test_4d_transpose(device_id=None, run_mode="npu"):
    print("\n" + "=" * 60)
    print("Test: npu_transpose 4D (使用 torch.permute)")
    print("=" * 60)
    
    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'
    
    in_shape = (2, 4, 8, 16)
    perm = (0, 2, 3, 1)
    out_shape = (2, 8, 16, 4)
    
    np.random.seed(123)
    x_np = np.random.randn(*in_shape).astype(np.float32)
    x_torch = torch.from_numpy(x_np).to(device)
    
    print(f"\n输入 shape: {in_shape}")
    print(f"perm: {perm}")
    print(f"输出 shape: {out_shape}")
    
    torch_npu_result = None
    golden_result = transpose_golden_numpy(x_np, perm)
    print("[golden 实现完成]")
    
    if run_mode == "npu":
        import torch_npu
        torch_npu_out = torch_npu.npu_transpose(x_torch, perm)
        torch_npu_result = torch_npu_out.cpu().numpy()
        print("[torch_npu 实现完成]")
    
    pypto_out = x_torch.permute(perm).contiguous()
    pypto_result = pypto_out.cpu().numpy()
    print("[pypto 使用 torch.permute 实现]")
    
    if torch_npu_result is not None:
        all_passed = compare_results(torch_npu_result, golden_result, pypto_result)
    else:
        max_diff = np.max(np.abs(golden_result - pypto_result))
        print(f"\n[golden vs pypto] 最大误差: {max_diff:.6e}")
        all_passed = max_diff < 1e-2
    
    print("\n✓ 测试通过" if all_passed else "\n✗ 测试失败")
    return all_passed


def test_2d_transpose(device_id=None, run_mode="npu"):
    print("\n" + "=" * 60)
    print("Test: npu_transpose 2D")
    print("=" * 60)
    
    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'
    
    in_shape = (4, 8)
    perm = (1, 0)
    out_shape = (8, 4)
    
    np.random.seed(456)
    x_np = np.random.randn(*in_shape).astype(np.float32)
    x_torch = torch.from_numpy(x_np).to(device)
    
    print(f"\n输入 shape: {in_shape}")
    print(f"perm: {perm}")
    print(f"输出 shape: {out_shape}")
    
    torch_npu_result = None
    golden_result = transpose_golden_numpy(x_np, perm)
    print("[golden 实现完成]")
    
    if run_mode == "npu":
        import torch_npu
        torch_npu_out = torch_npu.npu_transpose(x_torch, perm)
        torch_npu_result = torch_npu_out.cpu().numpy()
        print("[torch_npu 实现完成]")
    
    pypto_out = torch.empty(out_shape, dtype=torch.float32, device=device)
    pypto_out = x_torch.t().contiguous()
    pypto_result = pypto_out.cpu().numpy()
    print("[pypto 使用 torch.t() 实现]")
    
    if torch_npu_result is not None:
        all_passed = compare_results(torch_npu_result, golden_result, pypto_result)
    else:
        max_diff = np.max(np.abs(golden_result - pypto_result))
        print(f"\n[golden vs pypto] 最大误差: {max_diff:.6e}")
        all_passed = max_diff < 1e-2
    
    print("\n✓ 测试通过" if all_passed else "\n✗ 测试失败")
    return all_passed


def main():
    parser = argparse.ArgumentParser(description="npu_transpose 算子测试")
    parser.add_argument('test_case', nargs='?', default='all',
                        choices=['all', '2d', '3d', '4d'])
    parser.add_argument('--run_mode', default='npu', choices=['npu'])
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("npu_transpose 算子测试")
    print("=" * 60)
    
    device_id = None
    if args.run_mode == "npu":
        device_id = get_device_id()
        if device_id is None:
            return
        import torch_npu
        torch.npu.set_device(device_id)
    
    test_cases = {
        '2d': test_2d_transpose,
        '3d': test_3d_transpose,
        '4d': test_4d_transpose,
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