#!/usr/bin/env python3
# coding: utf-8
"""
npu_linear 算子实现

数学公式：
    output = input @ weight.T + bias

实现方式：
1. torch_npu: 使用 torch_npu.npu_linear
2. golden: 使用 numpy 实现
3. pypto: 使用 pypto.matmul 实现
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

def linear_golden_numpy(x: np.ndarray, weight: np.ndarray, bias: np.ndarray = None):
    """
    NumPy 实现的 linear
    
    公式: output = x @ weight.T + bias
    """
    output = x @ weight.T
    if bias is not None:
        output = output + bias
    return output


def linear_golden_torch(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None):
    """
    PyTorch 实现的 linear
    """
    return torch.nn.functional.linear(x, weight, bias)


# ============================================================================
# PyPTO 实现
# ============================================================================

@pypto.frontend.jit
def linear_kernel(
    x: pypto.Tensor(),
    weight: pypto.Tensor(),
    out: pypto.Tensor(),
):
    """
    PyPTO 实现的 linear (无 bias)
    
    公式: output = x @ weight.T
    """
    pypto.set_cube_tile_shapes([32, 32], [64, 64], [64, 64])
    out[:] = pypto.matmul(x, weight, pypto.DT_FP32, b_trans=True)


@pypto.frontend.jit
def linear_bias_kernel(
    x: pypto.Tensor(),
    weight: pypto.Tensor(),
    bias: pypto.Tensor(),
    out: pypto.Tensor(),
):
    """
    PyPTO 实现的 linear (有 bias)
    
    公式: output = x @ weight.T + bias
    使用分开的 matmul + add 实现
    """
    pypto.set_cube_tile_shapes([32, 32], [64, 64], [64, 64])
    matmul_result = pypto.matmul(x, weight, pypto.DT_FP32, b_trans=True)
    out[:] = matmul_result + bias


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
    """对比三种实现的结果"""
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
            assert_allclose(a, b, rtol=rtol, atol=atol)
            status = "✓ 通过"
        except AssertionError as e:
            status = f"✗ 失败"
            all_passed = False
        
        print(f"  {name}: 最大误差={max_diff:.6e}, 平均误差={mean_diff:.6e}, {status}")
    
    return all_passed


def test_basic(device_id: int = None, run_mode: str = "npu"):
    """基础功能测试 (无 bias)"""
    print("=" * 60)
    print("Test: npu_linear 基础功能测试 (无 bias)")
    print("=" * 60)
    
    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'
    
    batch_size, in_features, out_features = 2, 16, 4
    x_shape = (batch_size, in_features)
    w_shape = (out_features, in_features)
    
    np.random.seed(42)
    x_np = np.random.randn(*x_shape).astype(np.float32)
    w_np = np.random.randn(*w_shape).astype(np.float32)
    
    x_torch = torch.from_numpy(x_np).to(device)
    w_torch = torch.from_numpy(w_np).to(device)
    
    print(f"\n输入 shape: {x_shape}")
    print(f"权重 shape: {w_shape}")
    print(f"输出 shape: ({batch_size}, {out_features})")
    
    torch_npu_result = None
    golden_result = None
    pypto_result = None
    
    if run_mode == "npu":
        import torch_npu
        
        torch_npu_out = torch_npu.npu_linear(x_torch, w_torch)
        torch_npu_result = torch_npu_out.cpu().numpy()
        print("\n[torch_npu 实现完成]")
    
    golden_result = linear_golden_numpy(x_np, w_np)
    print("[golden (numpy) 实现完成]")
    
    pypto_out = torch.empty((batch_size, out_features), dtype=torch.float32, device=device)
    linear_kernel(x_torch, w_torch, pypto_out)
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


def test_with_bias(device_id: int = None, run_mode: str = "npu"):
    """带 bias 的测试"""
    print("\n" + "=" * 60)
    print("Test: npu_linear 带 bias 测试")
    print("=" * 60)
    
    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'
    
    batch_size, in_features, out_features = 2, 16, 4
    x_shape = (batch_size, in_features)
    w_shape = (out_features, in_features)
    b_shape = (out_features,)
    
    np.random.seed(123)
    x_np = np.random.randn(*x_shape).astype(np.float32)
    w_np = np.random.randn(*w_shape).astype(np.float32)
    b_np = np.random.randn(*b_shape).astype(np.float32)
    
    x_torch = torch.from_numpy(x_np).to(device)
    w_torch = torch.from_numpy(w_np).to(device)
    b_torch = torch.from_numpy(b_np).to(device)
    
    print(f"\n输入 shape: {x_shape}")
    print(f"权重 shape: {w_shape}")
    print(f"偏置 shape: {b_shape}")
    
    torch_npu_result = None
    golden_result = None
    pypto_result = None
    
    if run_mode == "npu":
        import torch_npu
        
        torch_npu_out = torch_npu.npu_linear(x_torch, w_torch, b_torch)
        torch_npu_result = torch_npu_out.cpu().numpy()
        print("\n[torch_npu 实现完成]")
    
    golden_result = linear_golden_numpy(x_np, w_np, b_np)
    print("[golden (numpy) 实现完成]")
    
    pypto_out = torch.empty((batch_size, out_features), dtype=torch.float32, device=device)
    linear_bias_kernel(x_torch, w_torch, b_torch, pypto_out)
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
    print("Test: npu_linear BFloat16 测试")
    print("=" * 60)
    
    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'
    
    batch_size, in_features, out_features = 4, 64, 32
    
    torch.manual_seed(42)
    x_torch = torch.randn(batch_size, in_features, dtype=torch.bfloat16, device=device)
    w_torch = torch.randn(out_features, in_features, dtype=torch.bfloat16, device=device)
    b_torch = torch.randn(out_features, dtype=torch.bfloat16, device=device)
    
    print(f"\n输入 shape: ({batch_size}, {in_features})")
    print(f"数据类型: torch.bfloat16")
    
    torch_npu_result = None
    golden_result = None
    pypto_result = None
    
    if run_mode == "npu":
        import torch_npu
        
        torch_npu_out = torch_npu.npu_linear(x_torch, w_torch, b_torch)
        torch_npu_result = torch_npu_out.float().cpu().numpy()
        print("\n[torch_npu 实现完成]")
    
    x_float = x_torch.float()
    w_float = w_torch.float()
    b_float = b_torch.float()
    golden_result = linear_golden_torch(x_float, w_float, b_float).cpu().numpy()
    print("[golden (torch) 实现完成]")
    
    pypto_out = torch.empty((batch_size, out_features), dtype=torch.bfloat16, device=device)
    linear_bias_kernel(x_torch, w_torch, b_torch, pypto_out)
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


def main():
    parser = argparse.ArgumentParser(
        description="npu_linear 算子测试",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        'test_case',
        type=str,
        nargs='?',
        default='all',
        choices=['all', 'basic', 'bias', 'bfloat16'],
        help='测试用例: all, basic, bias, bfloat16'
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
    print("npu_linear 算子测试")
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
        'bias': test_with_bias,
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
    if all_passed:
        print("所有测试通过!")
    else:
        print("存在测试失败!")
    print("=" * 60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())