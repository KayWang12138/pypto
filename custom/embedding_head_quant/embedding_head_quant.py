import os
import pypto
import torch
import torch_npu
import numpy as np
from numpy.testing import assert_allclose
from typing import Tuple

# ===== Module-level Dynamic Axis Definition =====
# Define dynamic axis N for weight's first dimension
N = pypto.frontend.dynamic("N")

# ===== Golden Function (PyTorch Reference - BF16 version) =====
def embedding_head_quant_golden(weight, scale, eps=1e-4, min_v=-128.0, max_v=127.0):
    """PyTorch reference implementation for embedding head quantization (BF16 I/O, FP32 compute).

    Args:
        weight: Input weight tensor (N, M) in BF16
        scale: Quantization scale tensor (1, 1) scalar in BF16
        eps: Minimum scale threshold (default: 1e-4)
        min_v: Quantization lower bound (default: -128.0)
        max_v: Quantization upper bound (default: 127.0)

    Returns:
        Tuple of (quantized_weight, clamped, protected_scale) all in BF16
    """
    weight_fp32 = weight.float()
    scale_fp32 = scale.float()
    
    eps_tensor = torch.tensor(eps, device=scale_fp32.device, dtype=torch.float32)
    protected_scale = torch.where(scale_fp32 > eps_tensor, scale_fp32, eps_tensor)
    weight_normalized = weight_fp32 / protected_scale
    weight_rounded = (weight_normalized.round() - weight_normalized).detach() + weight_normalized
    clamped = torch.clamp(weight_rounded, min_v, max_v)
    output = clamped * protected_scale
    
    return output.to(torch.bfloat16), clamped.to(torch.bfloat16), protected_scale.to(torch.bfloat16)

# ===== JIT Kernel Implementation =====
def create_embedding_head_quant_kernel(m, eps=1e-4, min_v=-128.0, max_v=127.0, run_mode="npu", unroll_list=None):
    if unroll_list is None:
        unroll_list = [1, 2, 4]
    
    if run_mode == "npu":
        mode = pypto.RunMode.NPU
    elif run_mode == "sim":
        mode = pypto.RunMode.SIM
    else:
        raise ValueError(f"Invalid run_mode: {run_mode}")

    runtime_opts = {"run_mode": mode}

    if run_mode == "npu":
        runtime_opts.update({
            "stitch_function_max_num": 128,
        })

    @pypto.frontend.jit(runtime_options=runtime_opts)
    def embedding_head_quant_kernel(
        weight: pypto.Tensor((N, m), pypto.DT_BF16),
        scale: pypto.Tensor((1, 1), pypto.DT_BF16),
        output_bf16: pypto.Tensor((N, m), pypto.DT_BF16),
        clamped_bf16: pypto.Tensor((N, m), pypto.DT_BF16),
        protected_scale_bf16: pypto.Tensor((N, m), pypto.DT_BF16),
    ):
        n = weight.shape[0]
        
        for n_offset, unroll_length in pypto.loop_unroll(
            0, n, 1, 
            name="LOOP_N_UNROLL", 
            idx_name="n_offset",
            unroll_list=unroll_list
        ):
            tile_n = unroll_length
            
            pypto.set_vec_tile_shapes(tile_n, m)
            
            weight_tile = pypto.view(weight, [tile_n, m], [n_offset, 0])
            weight_fp32 = pypto.cast(weight_tile, pypto.DT_FP32)
            
            scale_fp32 = pypto.cast(scale, pypto.DT_FP32)
            protected_scale = pypto.maximum(scale_fp32, eps)
            
            scale_n = pypto.expand_clone(protected_scale, [tile_n, 1])
            scale_expanded = pypto.expand_clone(scale_n, [tile_n, m])

            normalized = pypto.div(weight_fp32, scale_expanded)
            rounded = pypto.round(normalized, decimals=0)
            clamped = pypto.clip(rounded, min_v, max_v)
            output = pypto.mul(clamped, scale_expanded)

            output_tile = pypto.cast(output, pypto.DT_BF16)
            clamped_tile = pypto.cast(clamped, pypto.DT_BF16)
            scale_tile = pypto.cast(scale_expanded, pypto.DT_BF16)

            pypto.assemble(output_tile, [n_offset, 0], output_bf16)
            pypto.assemble(clamped_tile, [n_offset, 0], clamped_bf16)
            pypto.assemble(scale_tile, [n_offset, 0], protected_scale_bf16)

    return embedding_head_quant_kernel

# ===== Performance Test Cases =====
# Test configurations for different model sizes
PERFORMANCE_TEST_CONFIGS = {
    "3B": {
        "weight_shape": (153376, 2048),
        "description": "3B model embedding head"
    },
    "7B": {
        "weight_shape": (153376, 3072),
        "description": "7B model embedding head"
    },
    "30B": {
        "weight_shape": (75776, 2560),
        "description": "30B model embedding head"
    }
}

def test_embedding_head_quant_3b(device_id=None, run_mode="npu"):
    """Test 3B model embedding head quantization: weight_shape=(153376, 2048)"""
    print("Test: 3B Model - weight_shape=(153376, 2048)")

    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'
    weight_shape = (153376, 2048)
    n, m = weight_shape

    weight_torch = (torch.randn(weight_shape, dtype=torch.float32, device=device) * 50).to(torch.bfloat16)
    scale_torch = torch.full((1, 1), 0.5, dtype=torch.bfloat16, device=device)
    expected_out, _, _ = embedding_head_quant_golden(weight_torch.clone(), scale_torch.clone())

    output_torch = torch.zeros(weight_shape, dtype=torch.bfloat16, device=device)
    clamped_torch = torch.zeros(weight_shape, dtype=torch.bfloat16, device=device)
    protected_scale_torch = torch.zeros(weight_shape, dtype=torch.bfloat16, device=device)

    kernel = create_embedding_head_quant_kernel(m, run_mode=run_mode)
    kernel(weight_torch, scale_torch, output_torch, clamped_torch, protected_scale_torch)

    max_diff = (output_torch - expected_out).abs().max().item()
    print(f"  Max difference: {max_diff:.6f}")
    assert_allclose(output_torch.float().cpu().numpy(), expected_out.float().cpu().numpy(), rtol=3e-3, atol=3e-3)
    print("  ✓ Passed")

def test_embedding_head_quant_7b(device_id=None, run_mode="npu"):
    """Test 7B model embedding head quantization: weight_shape=(153376, 3072)"""
    print("Test: 7B Model - weight_shape=(153376, 3072)")

    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'
    weight_shape = (153376, 3072)
    n, m = weight_shape

    weight_torch = (torch.randn(weight_shape, dtype=torch.float32, device=device) * 50).to(torch.bfloat16)
    scale_torch = torch.full((1, 1), 0.5, dtype=torch.bfloat16, device=device)
    expected_out, _, _ = embedding_head_quant_golden(weight_torch.clone(), scale_torch.clone())

    output_torch = torch.zeros(weight_shape, dtype=torch.bfloat16, device=device)
    clamped_torch = torch.zeros(weight_shape, dtype=torch.bfloat16, device=device)
    protected_scale_torch = torch.zeros(weight_shape, dtype=torch.bfloat16, device=device)

    kernel = create_embedding_head_quant_kernel(m, run_mode=run_mode)
    kernel(weight_torch, scale_torch, output_torch, clamped_torch, protected_scale_torch)

    max_diff = (output_torch - expected_out).abs().max().item()
    print(f"  Max difference: {max_diff:.6f}")
    assert_allclose(output_torch.float().cpu().numpy(), expected_out.float().cpu().numpy(), rtol=3e-3, atol=3e-3)
    print("  ✓ Passed")

def test_embedding_head_quant_30b(device_id=None, run_mode="npu"):
    """Test 30B model embedding head quantization: weight_shape=(75776, 2560)"""
    print("Test: 30B Model - weight_shape=(75776, 2560)")

    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'
    weight_shape = (75776, 2560)
    n, m = weight_shape

    weight_torch = (torch.randn(weight_shape, dtype=torch.float32, device=device) * 50).to(torch.bfloat16)
    scale_torch = torch.full((1, 1), 0.5, dtype=torch.bfloat16, device=device)
    expected_out, _, _ = embedding_head_quant_golden(weight_torch.clone(), scale_torch.clone())

    output_torch = torch.zeros(weight_shape, dtype=torch.bfloat16, device=device)
    clamped_torch = torch.zeros(weight_shape, dtype=torch.bfloat16, device=device)
    protected_scale_torch = torch.zeros(weight_shape, dtype=torch.bfloat16, device=device)

    kernel = create_embedding_head_quant_kernel(m, run_mode=run_mode)
    kernel(weight_torch, scale_torch, output_torch, clamped_torch, protected_scale_torch)

    max_diff = (output_torch - expected_out).abs().max().item()
    print(f"  Max difference: {max_diff:.6f}")
    assert_allclose(output_torch.float().cpu().numpy(), expected_out.float().cpu().numpy(), rtol=3e-3, atol=3e-3)
    print("  ✓ Passed")

# ===== Main Test Runner =====
def main():
    """Run all performance test cases for 3B, 7B, and 30B models."""
    import argparse

    parser = argparse.ArgumentParser(description="Embedding Head Quantization Operator - Performance Tests")
    parser.add_argument('--run_mode', type=str, default="npu", choices=["npu", "sim"],
                        help="Execution mode: npu (hardware) or sim (simulation)")
    parser.add_argument('--model', type=str, default="all", choices=["3b", "7b", "30b", "all"],
                        help="Model size to test: 3b, 7b, 30b, or all")
    args = parser.parse_args()

    device_id = None
    if args.run_mode == "npu":
        if 'TILE_FWK_DEVICE_ID' not in os.environ:
            print("ERROR: TILE_FWK_DEVICE_ID not set. Please set environment variable:")
            print("  export TILE_FWK_DEVICE_ID=0")
            print("\nOr use simulation mode:")
            print("  python3 embedding_head_quant.py --run_mode sim")
            return
        device_id = int(os.environ['TILE_FWK_DEVICE_ID'])
        import torch_npu
        torch.npu.set_device(device_id)
        print(f"Using NPU device: {device_id}")
    else:
        print("Using simulation mode (CPU)")

    print("=" * 70)
    print("Embedding Head Quantization Operator - Performance Tests")
    print("3B: (153376, 2048) | 7B: (153376, 3072) | 30B: (75776, 2560)")
    print("=" * 70)
    print()

    if args.model == "all" or args.model == "3b":
        test_embedding_head_quant_3b(device_id, args.run_mode)
        print()

    if args.model == "all" or args.model == "7b":
        test_embedding_head_quant_7b(device_id, args.run_mode)
        print()

    if args.model == "all" or args.model == "30b":
        test_embedding_head_quant_30b(device_id, args.run_mode)
        print()

    print("=" * 70)
    print("All performance tests passed successfully!")
    print("=" * 70)

if __name__ == "__main__":
    main()
