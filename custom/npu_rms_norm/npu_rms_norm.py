#!/usr/bin/env python3
# coding: utf-8
"""
npu_rms_norm 算子实现

数学公式：
    RmsNorm(x_i) = x_i / Rms(x) * g_i
    Rms(x) = sqrt(1/n * sum(x_i^2) + eps)

返回值：
    - RmsNorm(x): 归一化后的输出
    - rstd: Rms(x)的倒数，用于反向计算

实现方式：
1. torch_npu: 使用 torch_npu.npu_rms_norm
2. golden: 使用 numpy 实现
3. pypto: 使用 pypto.sqrt 和 pypto.sum 组合实现
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

def rms_norm_golden_numpy(x: np.ndarray, gamma: np.ndarray, eps: float = 1e-6):
    """
    NumPy 实现的 rms_norm
    
    返回: (rms_norm_output, rstd)
    """
    variance = np.mean(x ** 2, axis=-1, keepdims=True)
    rms = np.sqrt(variance + eps)
    rstd = 1.0 / rms
    normalized = x * rstd
    output = normalized * gamma
    return output, rstd.squeeze(-1)


def rms_norm_golden_torch(x: torch.Tensor, gamma: torch.Tensor, eps: float = 1e-6):
    """
    PyTorch 实现的 rms_norm
    """
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    rms = torch.sqrt(variance + eps)
    rstd = 1.0 / rms
    normalized = x * rstd
    output = normalized * gamma
    return output, rstd.squeeze(-1)


# ============================================================================
# PyPTO 实现
# ============================================================================

@pypto.frontend.jit
def rms_norm_kernel(
    x: pypto.Tensor(),
    gamma: pypto.Tensor(),
    out_norm: pypto.Tensor(),
    eps: float,
    hidden_size: int,
):
    """
    PyPTO 实现的 rms_norm (仅输出归一化结果)
    
    公式:
        rms = sqrt(mean(x^2) + eps)
        rstd = 1 / rms
        output = x * rstd * gamma
    """
    ndim = len(x.shape)
    if ndim == 3:
        pypto.set_vec_tile_shapes(8, 16, 128)
    elif ndim == 2:
        pypto.set_vec_tile_shapes(64, 128)
    else:
        pypto.set_vec_tile_shapes(128)
    
    squared = x * x
    mean_sq = pypto.sum(squared, dim=-1, keepdim=True)
    mean_sq = mean_sq / hidden_size
    
    rms = pypto.sqrt(mean_sq + eps)
    rstd_val = 1.0 / rms
    
    normalized = x * rstd_val
    result = normalized * gamma
    
    pypto.assemble(result, [0, 0], out_norm)


# ============================================================================
# 测试与验证
# ============================================================================

def compare_results(
    torch_npu_result,
    golden_result,
    pypto_result,
    rtol: float = 1e-3,
    atol: float = 1e-3,
    name: str = ""
):
    """对比三种实现的结果"""
    print(f"\n{name} 精度对比:")
    
    results = {
        "torch_npu vs golden": (torch_npu_result, golden_result),
        "pypto vs golden": (pypto_result, golden_result),
        "pypto vs torch_npu": (pypto_result, torch_npu_result),
    }
    
    all_passed = True
    for cmp_name, (a, b) in results.items():
        max_diff = np.max(np.abs(a - b))
        mean_diff = np.mean(np.abs(a - b))
        try:
            assert_allclose(a, b, rtol=rtol, atol=atol)
            status = "✓ 通过"
        except AssertionError as e:
            status = f"✗ 失败"
            all_passed = False
        
        print(f"  {cmp_name}: 最大误差={max_diff:.6e}, 平均误差={mean_diff:.6e}, {status}")
    
    return all_passed


def test_basic(device_id: int = None, run_mode: str = "npu"):
    """基础功能测试"""
    print("=" * 60)
    print("Test: npu_rms_norm 基础功能测试")
    print("=" * 60)
    
    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'
    
    batch_size, hidden_size = 32, 128
    shape = (batch_size, hidden_size)
    
    np.random.seed(42)
    x_np = np.random.randn(*shape).astype(np.float32)
    gamma_np = np.random.randn(hidden_size).astype(np.float32)
    eps = 1e-6
    
    x_torch = torch.from_numpy(x_np).to(device)
    gamma_torch = torch.from_numpy(gamma_np).to(device)
    
    print(f"\n输入 shape: {shape}")
    print(f"hidden_size: {hidden_size}")
    print(f"epsilon: {eps}")
    
    torch_npu_out_norm = None
    torch_npu_out_rstd = None
    golden_out_norm = None
    golden_out_rstd = None
    pypto_out_norm = None
    pypto_out_rstd = None
    
    if run_mode == "npu":
        import torch_npu
        
        torch_npu_out_norm, torch_npu_out_rstd = torch_npu.npu_rms_norm(x_torch, gamma_torch, epsilon=eps)
        torch_npu_out_norm = torch_npu_out_norm.cpu().numpy()
        torch_npu_out_rstd = torch_npu_out_rstd.cpu().numpy()
        print("\n[torch_npu 实现完成]")
    
    golden_out_norm, golden_out_rstd = rms_norm_golden_numpy(x_np, gamma_np, eps)
    print("[golden (numpy) 实现完成]")
    
    pypto_out_norm = torch.empty(shape, dtype=torch.float32, device=device)
    rms_norm_kernel(x_torch, gamma_torch, pypto_out_norm, eps, hidden_size)
    pypto_out_norm = pypto_out_norm.cpu().numpy()
    pypto_out_rstd = None
    print("[pypto 实现完成]")
    
    all_passed = True
    if torch_npu_out_norm is not None:
        print("\n--- RmsNorm 输出对比 ---")
        passed_norm = compare_results(torch_npu_out_norm, golden_out_norm, pypto_out_norm)
        all_passed = passed_norm
    else:
        print("\n[golden vs pypto 对比]")
        max_diff = np.max(np.abs(golden_out_norm - pypto_out_norm))
        print(f"RmsNorm 最大误差: {max_diff:.6e}")
        all_passed = max_diff < 1e-2
    
    if all_passed:
        print("\n✓ 测试通过")
    else:
        print("\n✗ 测试失败")
    
    return all_passed


def test_bfloat16(device_id: int = None, run_mode: str = "npu"):
    """BFloat16 精度测试"""
    print("\n" + "=" * 60)
    print("Test: npu_rms_norm BFloat16 测试")
    print("=" * 60)
    
    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'
    
    batch_size, hidden_size = 32, 128
    shape = (batch_size, hidden_size)
    
    torch.manual_seed(42)
    x_torch = torch.randn(shape, dtype=torch.bfloat16, device=device)
    gamma_torch = torch.randn(hidden_size, dtype=torch.bfloat16, device=device)
    eps = 1e-6
    
    print(f"\n输入 shape: {shape}")
    print(f"数据类型: torch.bfloat16")
    
    torch_npu_out_norm = None
    torch_npu_out_rstd = None
    golden_out_norm = None
    golden_out_rstd = None
    pypto_out_norm = None
    pypto_out_rstd = None
    
    if run_mode == "npu":
        import torch_npu
        
        torch_npu_out_norm, torch_npu_out_rstd = torch_npu.npu_rms_norm(x_torch, gamma_torch, epsilon=eps)
        torch_npu_out_norm = torch_npu_out_norm.float().cpu().numpy()
        torch_npu_out_rstd = torch_npu_out_rstd.cpu().numpy()
        print("\n[torch_npu 实现完成]")
    
    x_float = x_torch.float()
    gamma_float = gamma_torch.float()
    golden_out_norm, golden_out_rstd = rms_norm_golden_torch(x_float, gamma_float, eps)
    golden_out_norm = golden_out_norm.cpu().numpy()
    golden_out_rstd = golden_out_rstd.cpu().numpy()
    print("[golden (torch) 实现完成]")
    
    pypto_out_norm = torch.empty(shape, dtype=torch.bfloat16, device=device)
    rms_norm_kernel(x_torch, gamma_torch, pypto_out_norm, eps, hidden_size)
    pypto_out_norm = pypto_out_norm.float().cpu().numpy()
    pypto_out_rstd = None
    print("[pypto 实现完成]")
    
    all_passed = True
    if torch_npu_out_norm is not None:
        print("\n--- RmsNorm 输出对比 ---")
        passed_norm = compare_results(
            torch_npu_out_norm, golden_out_norm, pypto_out_norm,
            rtol=0.02, atol=0.02
        )
        all_passed = passed_norm
    else:
        print("\n[golden vs pypto 对比]")
        max_diff = np.max(np.abs(golden_out_norm - pypto_out_norm))
        print(f"RmsNorm 最大误差: {max_diff:.6e}")
        all_passed = max_diff < 1e-1
    
    if all_passed:
        print("\n✓ 测试通过")
    else:
        print("\n✗ 测试失败")
    
    return all_passed


def test_3d_input(device_id: int = None, run_mode: str = "npu"):
    """3D输入测试"""
    print("\n" + "=" * 60)
    print("Test: npu_rms_norm 3D输入测试")
    print("=" * 60)
    
    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'
    
    shape = (24, 1, 128)
    hidden_size = 128
    
    np.random.seed(123)
    x_np = np.random.randn(*shape).astype(np.float32)
    gamma_np = np.random.randn(hidden_size).astype(np.float32)
    eps = 1e-5
    
    x_torch = torch.from_numpy(x_np).to(device)
    gamma_torch = torch.from_numpy(gamma_np).to(device)
    
    print(f"\n输入 shape: {shape}")
    print(f"hidden_size: {hidden_size}")
    
    torch_npu_out_norm = None
    torch_npu_out_rstd = None
    golden_out_norm = None
    golden_out_rstd = None
    pypto_out_norm = None
    pypto_out_rstd = None
    
    if run_mode == "npu":
        import torch_npu
        
        torch_npu_out_norm, torch_npu_out_rstd = torch_npu.npu_rms_norm(x_torch, gamma_torch, epsilon=eps)
        torch_npu_out_norm = torch_npu_out_norm.cpu().numpy()
        torch_npu_out_rstd = torch_npu_out_rstd.cpu().numpy()
        print("\n[torch_npu 实现完成]")
    
    golden_out_norm, golden_out_rstd = rms_norm_golden_numpy(x_np, gamma_np, eps)
    print("[golden (numpy) 实现完成]")
    
    pypto_out_norm = torch.empty(shape, dtype=torch.float32, device=device)
    rms_norm_kernel(x_torch, gamma_torch, pypto_out_norm, eps, hidden_size)
    pypto_out_norm = pypto_out_norm.cpu().numpy()
    pypto_out_rstd = None
    print("[pypto 实现完成]")
    
    all_passed = True
    if torch_npu_out_norm is not None:
        print("\n--- RmsNorm 输出对比 ---")
        passed_norm = compare_results(torch_npu_out_norm, golden_out_norm, pypto_out_norm)
        all_passed = passed_norm
    else:
        print("\n[golden vs pypto 对比]")
        max_diff = np.max(np.abs(golden_out_norm - pypto_out_norm))
        print(f"RmsNorm 最大误差: {max_diff:.6e}")
        all_passed = max_diff < 1e-2
    
    if all_passed:
        print("\n✓ 测试通过")
    else:
        print("\n✗ 测试失败")
    
    return all_passed


def main():
    parser = argparse.ArgumentParser(
        description="npu_rms_norm 算子测试",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        'test_case',
        type=str,
        nargs='?',
        default='all',
        choices=['all', 'basic', 'bfloat16', '3d'],
        help='测试用例: all, basic, bfloat16, 3d'
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
    print("npu_rms_norm 算子测试")
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
        '3d': test_3d_input,
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