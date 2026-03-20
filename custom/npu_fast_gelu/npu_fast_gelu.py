#!/usr/bin/env python3
# coding: utf-8
"""
npu_fast_gelu 算子实现

数学公式（A3服务器）：
    fast_gelu(x) = x / (1 + e^(-1.702x))
               = x * sigmoid(1.702 * x)

实现方式：
1. torch_npu: 使用 torch_npu.npu_fast_gelu
2. golden: 使用 numpy 实现
3. pypto: 使用 pypto.sigmoid 组合实现
"""

import os
import sys
import argparse
import numpy as np
from numpy.testing import assert_allclose

import torch
import pypto


def get_device_id():
    """获取并验证 TILE_FWK_DEVICE_ID 环境变量"""
    if 'TILE_FWK_DEVICE_ID' not in os.environ:
        print("Please set the environment variable TILE_FWK_DEVICE_ID before running:")
        print("  export TILE_FWK_DEVICE_ID=0")
        return None
    try:
        return int(os.environ['TILE_FWK_DEVICE_ID'])
    except ValueError:
        print(f"ERROR: TILE_FWK_DEVICE_ID must be an integer, got: {os.environ['TILE_FWK_DEVICE_ID']}")
        return None


# ============================================================================
# Golden 实现 (NumPy)
# ============================================================================

def fast_gelu_golden_numpy(x: np.ndarray) -> np.ndarray:
    """
    NumPy 实现的 fast_gelu
    
    公式: fast_gelu(x) = x / (1 + exp(-1.702 * x))
    """
    return x / (1.0 + np.exp(-1.702 * x))


def fast_gelu_golden_torch(x: torch.Tensor) -> torch.Tensor:
    """
    PyTorch 实现的 fast_gelu (用于 golden 对比)
    """
    return x / (1.0 + torch.exp(-1.702 * x))


# ============================================================================
# PyPTO 实现
# ============================================================================

@pypto.frontend.jit
def fast_gelu_kernel(
    x: pypto.Tensor(),
    out: pypto.Tensor(),
):
    """
    PyPTO 实现的 fast_gelu
    
    公式: fast_gelu(x) = x * sigmoid(1.702 * x)
    """
    ndim = len(x.shape)
    if ndim == 4:
        pypto.set_vec_tile_shapes(1, 8, 16, 128)
    elif ndim == 3:
        pypto.set_vec_tile_shapes(8, 16, 128)
    elif ndim == 2:
        pypto.set_vec_tile_shapes(32, 128)
    else:
        pypto.set_vec_tile_shapes(128)
    
    x_scaled = x * 1.702
    out[:] = x * pypto.sigmoid(x_scaled)


# ============================================================================
# 测试与验证
# ============================================================================

def compare_results(
    torch_npu_result: np.ndarray,
    golden_result: np.ndarray,
    pypto_result: np.ndarray,
    rtol: float = 1e-3,
    atol: float = 1e-3,
):
    """
    对比三种实现的结果
    
    Args:
        torch_npu_result: torch_npu 实现结果
        golden_result: golden 实现结果
        pypto_result: pypto 实现结果
        rtol: 相对误差容限
        atol: 绝对误差容限
    """
    print("\n" + "=" * 60)
    print("精度对比结果")
    print("=" * 60)
    
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
            assert_allclose(a, b, rtol=rtol, atol=atol)
            status = "✓ 通过"
        except AssertionError as e:
            status = f"✗ 失败: {str(e)[:50]}"
            all_passed = False
        
        print(f"\n{name}:")
        print(f"  最大误差: {max_diff:.6e}")
        print(f"  平均误差: {mean_diff:.6e}")
        print(f"  状态: {status}")
    
    return all_passed


def test_basic(device_id: int = None, run_mode: str = "npu"):
    """基础功能测试"""
    print("=" * 60)
    print("Test: npu_fast_gelu 基础功能测试")
    print("=" * 60)
    
    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'
    
    shape = (32, 128)
    dtype_np = np.float32
    dtype_torch = torch.float32
    
    np.random.seed(42)
    x_np = np.random.uniform(-2, 2, shape).astype(dtype_np)
    
    x_torch = torch.from_numpy(x_np).to(dtype_torch).to(device)
    
    print(f"\n输入 shape: {shape}")
    print(f"数据类型: {dtype_torch}")
    
    torch_npu_result = None
    golden_result = None
    pypto_result = None
    
    if run_mode == "npu":
        import torch_npu
        
        torch_npu_out = torch_npu.npu_fast_gelu(x_torch)
        torch_npu_result = torch_npu_out.cpu().numpy()
        print("\n[torch_npu 实现完成]")
    
    golden_result = fast_gelu_golden_numpy(x_np)
    print("[golden (numpy) 实现完成]")
    
    pypto_out = torch.empty(shape, dtype=dtype_torch, device=device)
    fast_gelu_kernel(x_torch, pypto_out)
    pypto_result = pypto_out.cpu().numpy()
    print("[pypto 实现完成]")
    
    if torch_npu_result is not None:
        all_passed = compare_results(torch_npu_result, golden_result, pypto_result)
    else:
        print("\n[golden vs pypto 对比]")
        max_diff = np.max(np.abs(golden_result - pypto_result))
        print(f"最大误差: {max_diff:.6e}")
        all_passed = max_diff < 1e-2
    
    if all_passed:
        print("\n✓ 测试通过")
    else:
        print("\n✗ 测试失败")
    
    return all_passed


def test_bfloat16(device_id: int = None, run_mode: str = "npu"):
    """BFloat16 精度测试"""
    print("\n" + "=" * 60)
    print("Test: npu_fast_gelu BFloat16 测试")
    print("=" * 60)
    
    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'
    
    shape = (32, 128)
    dtype_torch = torch.bfloat16
    
    torch.manual_seed(42)
    x_torch = torch.randn(shape, dtype=dtype_torch, device=device)
    
    print(f"\n输入 shape: {shape}")
    print(f"数据类型: {dtype_torch}")
    
    torch_npu_result = None
    golden_result = None
    pypto_result = None
    
    if run_mode == "npu":
        import torch_npu
        
        torch_npu_out = torch_npu.npu_fast_gelu(x_torch)
        torch_npu_result = torch_npu_out.float().cpu().numpy()
        print("\n[torch_npu 实现完成]")
    
    x_float = x_torch.float()
    golden_result = fast_gelu_golden_torch(x_float).cpu().numpy()
    print("[golden (torch) 实现完成]")
    
    pypto_out = torch.empty(shape, dtype=dtype_torch, device=device)
    fast_gelu_kernel(x_torch, pypto_out)
    pypto_result = pypto_out.float().cpu().numpy()
    print("[pypto 实现完成]")
    
    if torch_npu_result is not None:
        all_passed = compare_results(
            torch_npu_result, golden_result, pypto_result,
            rtol=0.02, atol=0.02
        )
    else:
        print("\n[golden vs pypto 对比]")
        max_diff = np.max(np.abs(golden_result - pypto_result))
        print(f"最大误差: {max_diff:.6e}")
        all_passed = max_diff < 1e-1
    
    if all_passed:
        print("\n✓ 测试通过")
    else:
        print("\n✗ 测试失败")
    
    return all_passed


def test_large_shape(device_id: int = None, run_mode: str = "npu"):
    """大 Shape 测试"""
    print("\n" + "=" * 60)
    print("Test: npu_fast_gelu 大 Shape 测试")
    print("=" * 60)
    
    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'
    
    shape = (4, 2048, 16, 128)
    dtype_torch = torch.float32
    
    np.random.seed(123)
    x_np = np.random.uniform(-2, 2, shape).astype(np.float32)
    x_torch = torch.from_numpy(x_np).to(dtype_torch).to(device)
    
    print(f"\n输入 shape: {shape}")
    print(f"数据类型: {dtype_torch}")
    
    torch_npu_result = None
    golden_result = None
    pypto_result = None
    
    if run_mode == "npu":
        import torch_npu
        
        torch_npu_out = torch_npu.npu_fast_gelu(x_torch)
        torch_npu_result = torch_npu_out.cpu().numpy()
        print("\n[torch_npu 实现完成]")
    
    golden_result = fast_gelu_golden_numpy(x_np)
    print("[golden (numpy) 实现完成]")
    
    pypto_out = torch.empty(shape, dtype=dtype_torch, device=device)
    fast_gelu_kernel(x_torch, pypto_out)
    pypto_result = pypto_out.cpu().numpy()
    print("[pypto 实现完成]")
    
    if torch_npu_result is not None:
        all_passed = compare_results(torch_npu_result, golden_result, pypto_result)
    else:
        print("\n[golden vs pypto 对比]")
        max_diff = np.max(np.abs(golden_result - pypto_result))
        print(f"最大误差: {max_diff:.6e}")
        all_passed = max_diff < 1e-2
    
    if all_passed:
        print("\n✓ 测试通过")
    else:
        print("\n✗ 测试失败")
    
    return all_passed


def main():
    parser = argparse.ArgumentParser(
        description="npu_fast_gelu 算子测试",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        'test_case',
        type=str,
        nargs='?',
        default='all',
        choices=['all', 'basic', 'bfloat16', 'large'],
        help='测试用例: all, basic, bfloat16, large'
    )
    parser.add_argument(
        '--run_mode',
        type=str,
        default='npu',
        choices=['npu'],
        help='运行模式'
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("npu_fast_gelu 算子测试")
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
        'large': test_large_shape,
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
    if all_passed:
        print("所有测试通过!")
    else:
        print("存在测试失败!")
    print("=" * 60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())