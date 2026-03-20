#!/usr/bin/env python3
# coding: utf-8
"""
npu_gelu_mul 算子实现

数学公式：
    input[..., :d] = x1, input[..., d:] = x2
    out = GELU(x1) * x2

实现方式：
1. torch_npu: 使用 torch_npu.npu_gelu_mul
2. golden: 使用 numpy 实现
3. pypto: 使用切片和组合实现
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
        return None


def gelu_golden_numpy(x: np.ndarray, approximate: str = 'none') -> np.ndarray:
    if approximate == 'tanh':
        return x * 0.5 * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))
    else:
        return x * 0.5 * (1.0 + special.erf(x / np.sqrt(2.0)))


def gelu_mul_golden_numpy(input_tensor: np.ndarray, approximate: str = 'none') -> np.ndarray:
    d = input_tensor.shape[-1] // 2
    x1 = input_tensor[..., :d]
    x2 = input_tensor[..., d:]
    return gelu_golden_numpy(x1, approximate) * x2


@pypto.frontend.jit
def gelu_mul_tanh_kernel(
    x: pypto.Tensor(),
    out: pypto.Tensor(),
    half_dim: int,
):
    pypto.set_vec_tile_shapes(32, 128)
    x1 = x[..., :half_dim]
    x2 = x[..., half_dim:]
    out[:] = torch.nn.functional.gelu(x1) * x2


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


def test_basic(device_id=None, run_mode="npu"):
    print("=" * 60)
    print("Test: npu_gelu_mul 基础功能测试")
    print("=" * 60)
    
    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'
    
    shape = (100, 400)
    approximate = 'tanh'
    
    np.random.seed(42)
    x_np = np.random.randn(*shape).astype(np.float32)
    x_torch = torch.from_numpy(x_np).to(device)
    
    print(f"\n输入 shape: {shape}")
    print(f"approximate: {approximate}")
    
    torch_npu_result = None
    golden_result = gelu_mul_golden_numpy(x_np, approximate)
    print("[golden 实现完成]")
    
    if run_mode == "npu":
        import torch_npu
        torch_npu_out = torch_npu.npu_gelu_mul(x_torch, approximate=approximate)
        torch_npu_result = torch_npu_out.cpu().numpy()
        print("[torch_npu 实现完成]")
    
    half_dim = shape[-1] // 2
    out_shape = shape[:-1] + (half_dim,)
    pypto_out = torch.empty(out_shape, dtype=torch.float32, device=device)
    gelu_mul_tanh_kernel(x_torch, pypto_out, half_dim)
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


def test_large_shape(device_id=None, run_mode="npu"):
    print("\n" + "=" * 60)
    print("Test: npu_gelu_mul 大shape测试")
    print("=" * 60)
    
    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'
    
    shape = (8, 1024)
    approximate = 'tanh'
    
    np.random.seed(123)
    x_np = np.random.randn(*shape).astype(np.float16)
    x_torch = torch.from_numpy(x_np).to(device)
    
    print(f"\n输入 shape: {shape}")
    print(f"数据类型: float16")
    
    torch_npu_result = None
    golden_result = gelu_mul_golden_numpy(x_np.astype(np.float32), approximate).astype(np.float16)
    print("[golden 实现完成]")
    
    if run_mode == "npu":
        import torch_npu
        torch_npu_out = torch_npu.npu_gelu_mul(x_torch, approximate=approximate)
        torch_npu_result = torch_npu_out.cpu().numpy()
        print("[torch_npu 实现完成]")
    
    half_dim = shape[-1] // 2
    out_shape = shape[:-1] + (half_dim,)
    pypto_out = torch.empty(out_shape, dtype=torch.float16, device=device)
    gelu_mul_tanh_kernel(x_torch, pypto_out, half_dim)
    pypto_result = pypto_out.cpu().numpy()
    print("[pypto 实现完成]")
    
    if torch_npu_result is not None:
        all_passed = compare_results(torch_npu_result, golden_result, pypto_result, rtol=0.02, atol=0.02)
    else:
        max_diff = np.max(np.abs(golden_result - pypto_result))
        print(f"\n[golden vs pypto] 最大误差: {max_diff:.6e}")
        all_passed = max_diff < 1e-1
    
    print("\n✓ 测试通过" if all_passed else "\n✗ 测试失败")
    return all_passed


def main():
    parser = argparse.ArgumentParser(description="npu_gelu_mul 算子测试")
    parser.add_argument('test_case', nargs='?', default='all', choices=['all', 'basic', 'large'])
    parser.add_argument('--run_mode', default='npu', choices=['npu'])
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("npu_gelu_mul 算子测试")
    print("=" * 60)
    
    device_id = None
    if args.run_mode == "npu":
        device_id = get_device_id()
        if device_id is None:
            return
        import torch_npu
        torch.npu.set_device(device_id)
    
    test_cases = {'basic': test_basic, 'large': test_large_shape}
    cases_to_run = list(test_cases.items()) if args.test_case == 'all' else [(args.test_case, test_cases[args.test_case])]
    
    all_passed = True
    for name, test_func in cases_to_run:
        try:
            all_passed = all_passed and test_func(device_id, args.run_mode)
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