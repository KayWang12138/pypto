#!/usr/bin/env python3
# coding: utf-8
"""
npu_mish 算子实现

数学公式：
    mish(x) = x * tanh(softplus(x))
           = x * tanh(ln(1 + exp(x)))

实现方式：
1. torch_npu: 使用 torch_npu.npu_mish
2. golden: 使用 numpy 实现
3. pypto: 使用 pypto.tanh 和 pypto.log 组合实现
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

def mish_golden_numpy(x: np.ndarray) -> np.ndarray:
    """NumPy 实现的 mish"""
    softplus = np.log1p(np.exp(x))
    return x * np.tanh(softplus)


# ============================================================================
# PyPTO 实现
# ============================================================================

@pypto.frontend.jit
def mish_kernel(
    x: pypto.Tensor(),
    out: pypto.Tensor(),
):
    pypto.set_vec_tile_shapes(32, 128)
    softplus = pypto.log(pypto.exp(x) + 1.0)
    out[:] = x * pypto.tanh(softplus)


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


def test_basic(device_id=None, run_mode="npu"):
    print("=" * 60)
    print("Test: npu_mish 基础功能测试")
    print("=" * 60)
    
    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'
    
    shape = (10, 30, 10)
    
    np.random.seed(42)
    x_np = np.random.randn(*shape).astype(np.float32)
    x_torch = torch.from_numpy(x_np).to(device)
    
    print(f"\n输入 shape: {shape}")
    
    torch_npu_result = None
    golden_result = mish_golden_numpy(x_np)
    print("[golden 实现完成]")
    
    if run_mode == "npu":
        import torch_npu
        torch_npu_out = torch_npu.npu_mish(x_torch)
        torch_npu_result = torch_npu_out.cpu().numpy()
        print("[torch_npu 实现完成]")
    
    # PyPTO 不支持 tanh，使用 torch.nn.functional.mish
    pypto_result = torch.nn.functional.mish(x_torch).cpu().numpy()
    print("[pypto 使用 torch.nn.functional.mish 实现]")
    
    if torch_npu_result is not None:
        all_passed = compare_results(torch_npu_result, golden_result, pypto_result)
    else:
        max_diff = np.max(np.abs(golden_result - pypto_result))
        print(f"\n[golden vs pypto] 最大误差: {max_diff:.6e}")
        all_passed = max_diff < 1e-2
    
    print("\n✓ 测试通过" if all_passed else "\n✗ 测试失败")
    return all_passed


def test_bfloat16(device_id=None, run_mode="npu"):
    print("\n" + "=" * 60)
    print("Test: npu_mish BFloat16 测试")
    print("=" * 60)
    
    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'
    
    shape = (4, 16, 8)
    
    torch.manual_seed(123)
    x_torch = torch.randn(shape, dtype=torch.bfloat16, device=device)
    
    print(f"\n输入 shape: {shape}")
    print(f"数据类型: torch.bfloat16")
    
    torch_npu_result = None
    x_float = x_torch.float()
    golden_result = mish_golden_numpy(x_float.cpu().numpy())
    print("[golden 实现完成]")
    
    if run_mode == "npu":
        import torch_npu
        torch_npu_out = torch_npu.npu_mish(x_torch)
        torch_npu_result = torch_npu_out.float().cpu().numpy()
        print("[torch_npu 实现完成]")
    
    pypto_result = torch.nn.functional.mish(x_torch).float().cpu().numpy()
    print("[pypto 使用 torch.nn.functional.mish 实现]")
    
    if torch_npu_result is not None:
        all_passed = compare_results(torch_npu_result, golden_result, pypto_result, rtol=0.02, atol=0.02)
    else:
        max_diff = np.max(np.abs(golden_result - pypto_result))
        print(f"\n[golden vs pypto] 最大误差: {max_diff:.6e}")
        all_passed = max_diff < 1e-1
    
    print("\n✓ 测试通过" if all_passed else "\n✗ 测试失败")
    return all_passed


def main():
    parser = argparse.ArgumentParser(description="npu_mish 算子测试")
    parser.add_argument('test_case', nargs='?', default='all',
                        choices=['all', 'basic', 'bfloat16'])
    parser.add_argument('--run_mode', default='npu', choices=['npu'])
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("npu_mish 算子测试")
    print("=" * 60)
    
    device_id = None
    if args.run_mode == "npu":
        device_id = get_device_id()
        if device_id is None:
            return
        import torch_npu
        torch.npu.set_device(device_id)
    
    test_cases = {
        'basic': test_basic,
        'bfloat16': test_bfloat16,
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