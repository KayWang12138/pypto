#!/usr/bin/env python3
# coding: utf-8
"""
npu_softmax_cross_entropy_with_logits 算子实现

数学公式：
    loss = -sum(y_i * log(softmax(x_i)))
    
    其中 softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_i - max(x)))

实现方式：
1. torch_npu: 使用 torch_npu.npu_softmax_cross_entropy_with_logits
2. golden: 使用 numpy 实现
3. pypto: 使用 pypto.softmax 和 pypto.log 组合实现
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

def softmax_cross_entropy_golden_numpy(features: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    NumPy 实现的 softmax_cross_entropy_with_logits
    
    公式: loss = -sum(y_i * log(softmax(x_i)))
    """
    shift_features = features - np.max(features, axis=-1, keepdims=True)
    exp_features = np.exp(shift_features)
    softmax = exp_features / np.sum(exp_features, axis=-1, keepdims=True)
    log_softmax = np.log(softmax + 1e-12)
    loss = -np.sum(labels * log_softmax, axis=-1, keepdims=True)
    return loss


def softmax_cross_entropy_golden_torch(features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    PyTorch 实现的 softmax_cross_entropy_with_logits
    """
    shift_features = features - features.max(dim=-1, keepdim=True)[0]
    exp_features = torch.exp(shift_features)
    softmax = exp_features / exp_features.sum(dim=-1, keepdim=True)
    log_softmax = torch.log(softmax + 1e-12)
    loss = -(labels * log_softmax).sum(dim=-1, keepdim=True)
    return loss


# ============================================================================
# PyPTO 实现
# ============================================================================

@pypto.frontend.jit
def softmax_cross_entropy_kernel(
    features: pypto.Tensor(),
    labels: pypto.Tensor(),
    out: pypto.Tensor(),
):
    """
    PyPTO 实现的 softmax_cross_entropy_with_logits
    
    公式: loss = -sum(y_i * log(softmax(x_i)))
    """
    pypto.set_vec_tile_shapes(64, 128)
    
    row_max = pypto.amax(features, dim=-1, keepdim=True)
    shifted = features - row_max
    exp_vals = pypto.exp(shifted)
    exp_sum = pypto.sum(exp_vals, dim=-1, keepdim=True)
    softmax = exp_vals / exp_sum
    log_softmax = pypto.log(softmax)
    product = labels * log_softmax
    loss = pypto.sum(product, dim=-1, keepdim=True)
    out[:] = -loss


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
    print(f"  torch_npu shape: {torch_npu_result.shape}")
    print(f"  golden shape: {golden_result.shape}")
    print(f"  pypto shape: {pypto_result.shape}")
    
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
        except AssertionError as e:
            status = f"✗ 失败: {str(e)[:50]}"
            all_passed = False
        
        print(f"  {name}: 最大误差={max_diff:.6e}, 平均误差={mean_diff:.6e}, {status}")
    
    return all_passed


def test_basic(device_id: int = None, run_mode: str = "npu"):
    """基础功能测试"""
    print("=" * 60)
    print("Test: npu_softmax_cross_entropy_with_logits 基础功能测试")
    print("=" * 60)
    
    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'
    
    batch_size = 4
    num_classes = 12
    shape = (1, batch_size * num_classes)
    
    np.random.seed(42)
    features_np = np.random.uniform(-1, 1, shape).astype(np.float32)
    labels_np = np.random.uniform(0, 1, shape).astype(np.float32)
    
    features_torch = torch.from_numpy(features_np).to(device)
    labels_torch = torch.from_numpy(labels_np).to(device)
    
    print(f"\n输入 shape: {shape}")
    print(f"batch_size: {batch_size}, num_classes: {num_classes}")
    
    torch_npu_result = None
    golden_result = None
    pypto_result = None
    
    if run_mode == "npu":
        import torch_npu
        
        torch_npu_out = torch_npu.npu_softmax_cross_entropy_with_logits(features_torch, labels_torch)
        torch_npu_result = torch_npu_out.cpu().numpy()
        print("\n[torch_npu 实现完成]")
    
    golden_result = softmax_cross_entropy_golden_numpy(features_np, labels_np)
    print("[golden (numpy) 实现完成]")
    
    pypto_out = torch.empty((1, 1), dtype=torch.float32, device=device)
    softmax_cross_entropy_kernel(features_torch, labels_torch, pypto_out)
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


def test_larger_batch(device_id: int = None, run_mode: str = "npu"):
    """更大 batch 测试"""
    print("\n" + "=" * 60)
    print("Test: npu_softmax_cross_entropy_with_logits 大 Batch 测试")
    print("=" * 60)
    
    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'
    
    batch_size = 32
    num_classes = 64
    shape = (1, batch_size * num_classes)
    
    np.random.seed(123)
    features_np = np.random.uniform(-2, 2, shape).astype(np.float32)
    labels_np = np.random.uniform(0, 1, shape).astype(np.float32)
    
    features_torch = torch.from_numpy(features_np).to(device)
    labels_torch = torch.from_numpy(labels_np).to(device)
    
    print(f"\n输入 shape: {shape}")
    print(f"batch_size: {batch_size}, num_classes: {num_classes}")
    
    torch_npu_result = None
    golden_result = None
    pypto_result = None
    
    if run_mode == "npu":
        import torch_npu
        
        torch_npu_out = torch_npu.npu_softmax_cross_entropy_with_logits(features_torch, labels_torch)
        torch_npu_result = torch_npu_out.cpu().numpy()
        print("\n[torch_npu 实现完成]")
    
    golden_result = softmax_cross_entropy_golden_numpy(features_np, labels_np)
    print("[golden (numpy) 实现完成]")
    
    pypto_out = torch.empty((1, 1), dtype=torch.float32, device=device)
    softmax_cross_entropy_kernel(features_torch, labels_torch, pypto_out)
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
    print("Test: npu_softmax_cross_entropy_with_logits BFloat16 测试")
    print("=" * 60)
    
    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'
    
    batch_size = 8
    num_classes = 32
    shape = (1, batch_size * num_classes)
    
    torch.manual_seed(42)
    features_torch = torch.randn(shape, dtype=torch.bfloat16, device=device)
    labels_torch = torch.randn(shape, dtype=torch.bfloat16, device=device)
    
    print(f"\n输入 shape: {shape}")
    print(f"数据类型: torch.bfloat16")
    
    torch_npu_result = None
    golden_result = None
    pypto_result = None
    
    if run_mode == "npu":
        import torch_npu
        
        torch_npu_out = torch_npu.npu_softmax_cross_entropy_with_logits(features_torch, labels_torch)
        torch_npu_result = torch_npu_out.float().cpu().numpy()
        print("\n[torch_npu 实现完成]")
    
    features_float = features_torch.float()
    labels_float = labels_torch.float()
    golden_result = softmax_cross_entropy_golden_torch(features_float, labels_float).cpu().numpy()
    print("[golden (torch) 实现完成]")
    
    pypto_out = torch.empty((1, 1), dtype=torch.bfloat16, device=device)
    softmax_cross_entropy_kernel(features_torch, labels_torch, pypto_out)
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
        description="npu_softmax_cross_entropy_with_logits 算子测试",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        'test_case',
        type=str,
        nargs='?',
        default='all',
        choices=['all', 'basic', 'large', 'bfloat16'],
        help='测试用例: all, basic, large, bfloat16'
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
    print("npu_softmax_cross_entropy_with_logits 算子测试")
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
        'large': test_larger_batch,
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