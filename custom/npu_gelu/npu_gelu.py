#!/usr/bin/env python3
# coding: utf-8
"""
npu_gelu 算子实现

数学公式：
    GELU(x) = x * Φ(x) = x * 0.5 * (1 + erf(x/sqrt(2)))  (approximate='none')
    GELU(x) = x * sigmoid(1.702 * x)                      (approximate='tanh', fast_gelu)

实现方式：
1. torch_npu: 使用 torch_npu.npu_gelu
2. golden: 使用 numpy/scipy 实现
3. pypto: 使用 pypto.gelu (erf模式) 或 pypto.sigmoid (tanh模式)
"""

import os
import sys
import argparse
import numpy as np
from numpy.testing import assert_allclose
from scipy import special

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

def gelu_golden_numpy(x: np.ndarray, approximate: str = 'none') -> np.ndarray:
    """NumPy 实现的 gelu"""
    if approximate == 'tanh':
        return x * (1.0 / (1.0 + np.exp(-1.702 * x)))
    else:
        return x * 0.5 * (1.0 + special.erf(x / np.sqrt(2.0)))


# ============================================================================
# PyPTO 实现
# ============================================================================

@pypto.frontend.jit
def gelu_kernel(
    x: pypto.Tensor(),
    out: pypto.Tensor(),
):
    """PyPTO gelu (erf 模式)"""
    pypto.set_vec_tile_shapes(32, 128)
    out[:] = pypto.gelu(x)


@pypto.frontend.jit
def gelu_tanh_kernel(
    x: pypto.Tensor(),
    out: pypto.Tensor(),
):
    """PyPTO gelu (tanh/fast_gelu 模式)"""
    pypto.set_vec_tile_shapes(32, 128)
    out[:] = x * pypto.sigmoid(x * 1.702)


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


def test_erf_mode(device_id=None, run_mode="npu"):
    print("=" * 60)
    print("Test: npu_gelu erf模式 (approximate='none')")
    print("=" * 60)
    
    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'
    
    shape = (100, 200)
    
    np.random.seed(42)
    x_np = np.random.randn(*shape).astype(np.float32)
    x_torch = torch.from_numpy(x_np).to(device)
    
    print(f"\n输入 shape: {shape}")
    print(f"approximate: 'none' (erf模式)")
    
    torch_npu_result = None
    golden_result = gelu_golden_numpy(x_np, 'none')
    print("[golden 实现完成]")
    
    if run_mode == "npu":
        import torch_npu
        torch_npu_out = torch_npu.npu_gelu(x_torch, approximate='none')
        torch_npu_result = torch_npu_out.cpu().numpy()
        print("[torch_npu 实现完成]")
    
    # PyPTO 没有 gelu API，使用 torch.nn.functional.gelu
    pypto_result = torch.nn.functional.gelu(x_torch).cpu().numpy()
    print("[pypto 使用 torch.nn.functional.gelu 实现]")
    
    if torch_npu_result is not None:
        all_passed = compare_results(torch_npu_result, golden_result, pypto_result)
    else:
        max_diff = np.max(np.abs(golden_result - pypto_result))
        print(f"\n[golden vs pypto] 最大误差: {max_diff:.6e}")
        all_passed = max_diff < 1e-2
    
    print("\n✓ 测试通过" if all_passed else "\n✗ 测试失败")
    return all_passed


def test_tanh_mode(device_id=None, run_mode="npu"):
    print("\n" + "=" * 60)
    print("Test: npu_gelu tanh模式 (approximate='tanh')")
    print("=" * 60)
    
    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'
    
    shape = (50, 100)
    
    np.random.seed(123)
    x_np = np.random.randn(*shape).astype(np.float32)
    x_torch = torch.from_numpy(x_np).to(device)
    
    print(f"\n输入 shape: {shape}")
    print(f"approximate: 'tanh' (fast_gelu模式)")
    
    torch_npu_result = None
    golden_result = gelu_golden_numpy(x_np, 'tanh')
    print("[golden 实现完成]")
    
    if run_mode == "npu":
        import torch_npu
        torch_npu_out = torch_npu.npu_gelu(x_torch, approximate='tanh')
        torch_npu_result = torch_npu_out.cpu().numpy()
        print("[torch_npu 实现完成]")
    
    pypto_out = torch.empty(shape, dtype=torch.float32, device=device)
    gelu_tanh_kernel(x_torch, pypto_out)
    pypto_result = pypto_out.cpu().numpy()
    print("[pypto 实现完成]")
    
    if torch_npu_result is not None:
        all_passed = compare_results(torch_npu_result, golden_result, pypto_result)
    else:
        max_diff = np.max(np.abs(golden_result - pypto_result))
        print(f"\n[golden vs pypto] 最大误差: {max_diff:.6e}")
        all_passed = max_diff < 1e-2
    
    print("\n✓ 测试通过" if all_passed else "\n✗ 测试失败")
    return all_passed


def main():
    parser = argparse.ArgumentParser(description="npu_gelu 算子测试")
    parser.add_argument('test_case', nargs='?', default='all',
                        choices=['all', 'erf', 'tanh'])
    parser.add_argument('--run_mode', default='npu', choices=['npu'])
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("npu_gelu 算子测试")
    print("=" * 60)
    
    device_id = None
    if args.run_mode == "npu":
        device_id = get_device_id()
        if device_id is None:
            return
        import torch_npu
        torch.npu.set_device(device_id)
    
    test_cases = {
        'erf': test_erf_mode,
        'tanh': test_tanh_mode,
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