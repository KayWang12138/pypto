import os
import pypto
import torch
import torch_npu
import numpy as np
from numpy.testing import assert_allclose
from typing import Tuple

# ===== Golden Function (PyTorch Reference - BF16 version) =====
def embedding_head_quant_golden(weight, scale, eps=1e-4, min_v=-128.0, max_v=127.0):
    """PyTorch reference implementation for embedding head quantization (BF16 I/O, FP32 compute).

    Args:
        weight: Input weight tensor (BF16)
        scale: Quantization scale tensor (BF16)
        eps: Minimum scale threshold (default: 1e-4)
        min_v: Quantization lower bound (default: -128.0)
        max_v: Quantization upper bound (default: 127.0)

    Returns:
        Tuple of (quantized_weight, clamped, protected_scale) all in BF16
    """
    # Convert to FP32 for computation
    weight_fp32 = weight.float()
    scale_fp32 = scale.float()
    
    eps_tensor = torch.tensor(eps, device=scale_fp32.device, dtype=torch.float32)
    protected_scale = torch.where(scale_fp32 > eps_tensor, scale_fp32, eps_tensor)
    weight_normalized = weight_fp32 / protected_scale
    weight_rounded = (weight_normalized.round() - weight_normalized).detach() + weight_normalized
    clamped = torch.clamp(weight_rounded, min_v, max_v)
    output = clamped * protected_scale
    
    # Convert back to BF16
    return output.to(torch.bfloat16), clamped.to(torch.bfloat16), protected_scale.to(torch.bfloat16)

# ===== JIT Kernel Implementation =====
def create_embedding_head_quant_kernel(shape, eps=1e-4, min_v=-128.0, max_v=127.0, run_mode="npu"):
    if run_mode == "npu":
        mode = pypto.RunMode.NPU
    elif run_mode == "sim":
        mode = pypto.RunMode.SIM
    else:
        raise ValueError(f"Invalid run_mode: {run_mode}")

    runtime_opts = {"run_mode": mode}

    if run_mode == "npu":
        runtime_opts.update({
            "stitch_function_inner_memory": 512,
            "stitch_function_outcast_memory": 512,
            "stitch_function_num_initial": 128,
            "stitch_function_max_num": 128,
            "stitch_function_num_step": 20
        })

    @pypto.frontend.jit(runtime_options=runtime_opts)
    def embedding_head_quant_kernel(
        weight: pypto.Tensor(shape, pypto.DT_BF16),
        scale: pypto.Tensor(shape, pypto.DT_BF16),
    ) -> (
        pypto.Tensor(shape, pypto.DT_BF16),
        pypto.Tensor(shape, pypto.DT_BF16),
        pypto.Tensor(shape, pypto.DT_BF16),
    ):
        pypto.set_vec_tile_shapes(64, 64)

        # Convert BF16 inputs to FP32 for precision-critical computation
        weight_fp32 = pypto.cast(weight, pypto.DT_FP32)
        scale_fp32 = pypto.cast(scale, pypto.DT_FP32)

        # Quantization computation in FP32
        protected_scale = pypto.maximum(scale_fp32, eps)
        normalized = pypto.div(weight_fp32, protected_scale)
        rounded = pypto.round(normalized, decimals=0)
        clamped = pypto.clip(rounded, min_v, max_v)
        output = pypto.mul(clamped, protected_scale)

        # Convert outputs back to BF16
        output_bf16 = pypto.cast(output, pypto.DT_BF16)
        clamped_bf16 = pypto.cast(clamped, pypto.DT_BF16)
        protected_scale_bf16 = pypto.cast(protected_scale, pypto.DT_BF16)

        return output_bf16, clamped_bf16, protected_scale_bf16

    return embedding_head_quant_kernel

# ===== Test Cases =====
def test_embedding_head_quant_basic(device_id=None, run_mode="npu"):
    """Level 0: Small tensor test (8-16 elements) - Basic functionality verification."""
    print("Test: Basic functionality (small tensor)")

    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'
    shape = (8, 8)

    # Generate test data in BF16
    weight_torch = (torch.randn(shape, dtype=torch.float32, device=device) * 10).to(torch.bfloat16)
    scale_torch = (torch.rand(shape, dtype=torch.float32, device=device) * 2 + 0.1).to(torch.bfloat16)

    # Run PyPTO kernel
    kernel = create_embedding_head_quant_kernel(shape, run_mode=run_mode)
    output_torch, clamped_torch, protected_scale_torch = kernel(weight_torch, scale_torch)

    # Run golden reference
    expected_out, expected_clamped, expected_scale = embedding_head_quant_golden(weight_torch, scale_torch)

    # Verify outputs
    max_diff_out = (output_torch - expected_out).abs().max().item()
    max_diff_clamped = (clamped_torch - expected_clamped).abs().max().item()
    max_diff_scale = (protected_scale_torch - expected_scale).abs().max().item()
    
    print(f"  Output max diff: {max_diff_out:.6f}")
    print(f"  Clamped max diff: {max_diff_clamped:.6f}")
    print(f"  Protected scale max diff: {max_diff_scale:.6f}")
    
    assert_allclose(output_torch.float().cpu().numpy(), expected_out.float().cpu().numpy(), rtol=3e-3, atol=3e-3)
    assert_allclose(clamped_torch.float().cpu().numpy(), expected_clamped.float().cpu().numpy(), rtol=3e-3, atol=3e-3)
    assert_allclose(protected_scale_torch.float().cpu().numpy(), expected_scale.float().cpu().numpy(), rtol=3e-3, atol=3e-3)
    print("  ✓ Passed")

def test_embedding_head_quant_level1(device_id=None, run_mode="npu"):
    """Level 1: Typical size (1K elements) - Realistic use case."""
    print("Test: Typical size (1K elements)")

    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'
    shape = (32, 32)

    weight_torch = (torch.randn(shape, dtype=torch.float32, device=device) * 100).to(torch.bfloat16)
    scale_torch = (torch.rand(shape, dtype=torch.float32, device=device) * 5 + 0.5).to(torch.bfloat16)

    kernel = create_embedding_head_quant_kernel(shape, run_mode=run_mode)
    output_torch, clamped_torch, protected_scale_torch = kernel(weight_torch, scale_torch)
    expected_out, expected_clamped, expected_scale = embedding_head_quant_golden(weight_torch, scale_torch)

    max_diff = (output_torch - expected_out).abs().max().item()
    print(f"  Max difference: {max_diff:.6f}")
    assert_allclose(output_torch.float().cpu().numpy(), expected_out.float().cpu().numpy(), rtol=3e-3, atol=3e-3)
    print("  ✓ Passed")

def test_embedding_head_quant_edge_cases(device_id=None, run_mode="npu"):
    """Level 2: Edge cases (zero, small scale, large values) - Boundary conditions."""
    print("Test: Edge cases")

    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'
    shape = (16, 16)

    # Test case 1: Very small scale (should be protected by eps)
    print("  Test case 1: Very small scale")
    weight_torch = (torch.randn(shape, dtype=torch.float32, device=device) * 10).to(torch.bfloat16)
    scale_torch = torch.full(shape, 1e-6, dtype=torch.bfloat16, device=device)

    kernel = create_embedding_head_quant_kernel(shape, run_mode=run_mode)
    output_torch, _, protected_scale_torch = kernel(weight_torch, scale_torch)
    expected_out, _, expected_scale = embedding_head_quant_golden(weight_torch, scale_torch)

    max_diff = (output_torch - expected_out).abs().max().item()
    print(f"    Max difference: {max_diff:.6f}")
    assert_allclose(output_torch.float().cpu().numpy(), expected_out.float().cpu().numpy(), rtol=3e-3, atol=3e-3)
    print("    ✓ Small scale case passed")

    # Test case 2: Values exceeding quantization range
    print("  Test case 2: Large values exceeding quantization range")
    weight_torch = (torch.randn(shape, dtype=torch.float32, device=device) * 1000).to(torch.bfloat16)
    scale_torch = torch.ones(shape, dtype=torch.bfloat16, device=device)

    output_torch, clamped_torch, _ = kernel(weight_torch, scale_torch)
    expected_out, expected_clamped, _ = embedding_head_quant_golden(weight_torch, scale_torch)

    max_diff = (output_torch - expected_out).abs().max().item()
    print(f"    Max difference: {max_diff:.6f}")
    assert_allclose(output_torch.float().cpu().numpy(), expected_out.float().cpu().numpy(), rtol=3e-3, atol=3e-3)
    print("    ✓ Large value clamping passed")

    # Test case 3: Zero weight
    print("  Test case 3: Zero weight")
    weight_torch = torch.zeros(shape, dtype=torch.bfloat16, device=device)
    scale_torch = (torch.rand(shape, dtype=torch.float32, device=device) * 2 + 0.5).to(torch.bfloat16)

    output_torch, _, _ = kernel(weight_torch, scale_torch)
    expected_out, _, _ = embedding_head_quant_golden(weight_torch, scale_torch)

    max_diff = (output_torch - expected_out).abs().max().item()
    print(f"    Max difference: {max_diff:.6f}")
    assert_allclose(output_torch.float().cpu().numpy(), expected_out.float().cpu().numpy(), rtol=3e-3, atol=3e-3)
    print("    ✓ Zero weight case passed")

    # Test case 4: Uniform scale
    print("  Test case 4: Uniform scale")
    weight_torch = (torch.randn(shape, dtype=torch.float32, device=device) * 50).to(torch.bfloat16)
    scale_torch = torch.full(shape, 2.0, dtype=torch.bfloat16, device=device)

    output_torch, _, _ = kernel(weight_torch, scale_torch)
    expected_out, _, _ = embedding_head_quant_golden(weight_torch, scale_torch)

    max_diff = (output_torch - expected_out).abs().max().item()
    print(f"    Max difference: {max_diff:.6f}")
    assert_allclose(output_torch.float().cpu().numpy(), expected_out.float().cpu().numpy(), rtol=3e-3, atol=3e-3)
    print("    ✓ Uniform scale case passed")

def test_embedding_head_quant_large(device_id=None, run_mode="npu"):
    """Level 3: Large tensor for performance - Verify NPU performance."""
    print("Test: Large tensor (performance test)")

    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'
    shape = (256, 256)

    weight_torch = (torch.randn(shape, dtype=torch.float32, device=device) * 50).to(torch.bfloat16)
    scale_torch = (torch.rand(shape, dtype=torch.float32, device=device) * 3 + 0.5).to(torch.bfloat16)

    kernel = create_embedding_head_quant_kernel(shape, run_mode=run_mode)
    output_torch, clamped_torch, protected_scale_torch = kernel(weight_torch, scale_torch)
    expected_out, expected_clamped, expected_scale = embedding_head_quant_golden(weight_torch, scale_torch)

    max_diff = (output_torch - expected_out).abs().max().item()
    print(f"  Max difference: {max_diff:.6f}")
    assert_allclose(output_torch.float().cpu().numpy(), expected_out.float().cpu().numpy(), rtol=3e-3, atol=3e-3)
    print("  ✓ Passed")

# ===== Main Test Runner =====
def main():
    """Run all test cases."""
    import argparse

    parser = argparse.ArgumentParser(description="Embedding Head Quantization Operator Tests")
    parser.add_argument('--run_mode', type=str, default="npu", choices=["npu", "sim"],
                        help="Execution mode: npu (hardware) or sim (simulation)")
    parser.add_argument('--test_level', type=int, default=-1,
                        help="Test level: 0=basic, 1=typical, 2=edge, 3=large, -1=all")
    args = parser.parse_args()

    # Get device ID
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

    print("=" * 60)
    print("Embedding Head Quantization Operator Tests")
    print("=" * 60)
    print()

    # Run tests based on level
    if args.test_level == -1 or args.test_level == 0:
        test_embedding_head_quant_basic(device_id, args.run_mode)
        print()

    if args.test_level == -1 or args.test_level == 1:
        test_embedding_head_quant_level1(device_id, args.run_mode)
        print()

    if args.test_level == -1 or args.test_level == 2:
        test_embedding_head_quant_edge_cases(device_id, args.run_mode)
        print()

    if args.test_level == -1 or args.test_level == 3:
        test_embedding_head_quant_large(device_id, args.run_mode)
        print()

    print("=" * 60)
    print("All tests passed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()
