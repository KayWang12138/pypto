import os
import pypto
import torch
import torch_npu
import numpy as np
from numpy.testing import assert_allclose

# ===== Golden Function (PyTorch Reference) =====
def embedding_head_quant_golden(weight, scale, eps=1e-4, min_v=-128.0, max_v=127.0):
    """PyTorch reference implementation for embedding head quantization.

    Args:
        weight: Input weight tensor
        scale: Quantization scale tensor
        eps: Minimum scale threshold (default: 1e-4)
        min_v: Quantization lower bound (default: -128.0)
        min_v: Quantization upper bound (default: 127.0)

    Returns:
        Quantized and rescaled weight tensor
    """
    eps_tensor = torch.tensor(eps, device=scale.device).float()
    scale = torch.where(scale > eps_tensor, scale, eps_tensor)
    weight = weight / scale
    weight = (weight.round() - weight).detach() + weight
    weight = torch.clamp(weight, min_v, max_v) * scale
    return weight

# ===== JIT Kernel Implementation =====
def create_embedding_head_quant_kernel(shape, eps=1e-4, min_v=-128.0, max_v=127.0, run_mode="npu"):
    """Create JIT-compiled quantization kernel.

    Args:
        shape: Tensor shape
        eps: Minimum scale threshold (default: 1e-4)
        min_v: Quantization lower bound (default: -128.0)
        max_v: Quantization upper bound (default: 127.0)
        run_mode: Execution mode - "npu" or "sim" (default: "npu")

    Returns:
        JIT-compiled kernel function
    """
    if run_mode == "npu":
        mode = pypto.RunMode.NPU
    elif run_mode == "sim":
        mode = pypto.RunMode.SIM
    else:
        raise ValueError(f"Invalid run_mode: {run_mode}")

    # Optimized tile shapes (tuned on 2026-03-03)
    # Best config: vec_tile=(64, 64) provides +6.84% throughput improvement
    runtime_opts = {"run_mode": mode}

    # Add stitch parameters for better memory pool and task scheduling
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
        weight: pypto.Tensor(shape, pypto.DT_FP32),
        scale: pypto.Tensor(shape, pypto.DT_FP32),
    ) -> pypto.Tensor(shape, pypto.DT_FP32):
        # Set tile shapes for vector operations - optimized to (64, 64)
        pypto.set_vec_tile_shapes(64, 64)

        # Step 1: Scale protection (avoid scale <= eps)
        # Using maximum instead of where for cleaner implementation
        protected_scale = pypto.maximum(scale, eps)

        # Step 2: Normalize weight by scale
        normalized = pypto.div(weight, protected_scale)

        # Step 3: Round to nearest integer (STE forward pass)
        rounded = pypto.round(normalized, decimals=0)

        # Step 4: Clamp to quantization range
        clamped = pypto.clip(rounded, min_v, max_v)

        # Step 5: Rescale back
        output = pypto.mul(clamped, protected_scale)

        return output

    return embedding_head_quant_kernel

# ===== Test Cases =====
def test_embedding_head_quant_basic(device_id=None, run_mode="npu"):
    """Level 0: Small tensor test (8-16 elements) - Basic functionality verification."""
    print("Test: Basic functionality (small tensor)")

    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'
    shape = (8, 8)

    # Generate test data
    weight_torch = torch.randn(shape, dtype=torch.float32, device=device) * 10
    scale_torch = torch.rand(shape, dtype=torch.float32, device=device) * 2 + 0.1

    # Run PyPTO kernel
    kernel = create_embedding_head_quant_kernel(shape, run_mode=run_mode)
    output_torch = kernel(weight_torch, scale_torch)

    # Run golden reference
    expected = embedding_head_quant_golden(weight_torch, scale_torch)

    # Verify
    max_diff = (output_torch - expected).abs().max().item()
    print(f"  Max difference: {max_diff:.6f}")
    assert_allclose(output_torch.cpu().numpy(), expected.cpu().numpy(), rtol=3e-3, atol=3e-3)
    print("  ✓ Passed")

def test_embedding_head_quant_level1(device_id=None, run_mode="npu"):
    """Level 1: Typical size (1K elements) - Realistic use case."""
    print("Test: Typical size (1K elements)")

    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'
    shape = (32, 32)

    weight_torch = torch.randn(shape, dtype=torch.float32, device=device) * 100
    scale_torch = torch.rand(shape, dtype=torch.float32, device=device) * 5 + 0.5

    kernel = create_embedding_head_quant_kernel(shape, run_mode=run_mode)
    output_torch = kernel(weight_torch, scale_torch)
    expected = embedding_head_quant_golden(weight_torch, scale_torch)

    max_diff = (output_torch - expected).abs().max().item()
    print(f"  Max difference: {max_diff:.6f}")
    assert_allclose(output_torch.cpu().numpy(), expected.cpu().numpy(), rtol=3e-3, atol=3e-3)
    print("  ✓ Passed")

def test_embedding_head_quant_edge_cases(device_id=None, run_mode="npu"):
    """Level 2: Edge cases (zero, small scale, large values) - Boundary conditions."""
    print("Test: Edge cases")

    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'
    shape = (16, 16)

    # Test case 1: Very small scale (should be protected by eps)
    print("  Test case 1: Very small scale")
    weight_torch = torch.randn(shape, dtype=torch.float32, device=device) * 10
    scale_torch = torch.full(shape, 1e-6, dtype=torch.float32, device=device)

    kernel = create_embedding_head_quant_kernel(shape, run_mode=run_mode)
    output_torch = kernel(weight_torch, scale_torch)
    expected = embedding_head_quant_golden(weight_torch, scale_torch)

    max_diff = (output_torch - expected).abs().max().item()
    print(f"    Max difference: {max_diff:.6f}")
    assert_allclose(output_torch.cpu().numpy(), expected.cpu().numpy(), rtol=3e-3, atol=3e-3)
    print("    ✓ Small scale case passed")

    # Test case 2: Values exceeding quantization range
    print("  Test case 2: Large values exceeding quantization range")
    weight_torch = torch.randn(shape, dtype=torch.float32, device=device) * 1000
    scale_torch = torch.ones(shape, dtype=torch.float32, device=device)

    output_torch = kernel(weight_torch, scale_torch)
    expected = embedding_head_quant_golden(weight_torch, scale_torch)

    max_diff = (output_torch - expected).abs().max().item()
    print(f"    Max difference: {max_diff:.6f}")
    assert_allclose(output_torch.cpu().numpy(), expected.cpu().numpy(), rtol=3e-3, atol=3e-3)
    print("    ✓ Large value clamping passed")

    # Test case 3: Zero weight
    print("  Test case 3: Zero weight")
    weight_torch = torch.zeros(shape, dtype=torch.float32, device=device)
    scale_torch = torch.rand(shape, dtype=torch.float32, device=device) * 2 + 0.5

    output_torch = kernel(weight_torch, scale_torch)
    expected = embedding_head_quant_golden(weight_torch, scale_torch)

    max_diff = (output_torch - expected).abs().max().item()
    print(f"    Max difference: {max_diff:.6f}")
    assert_allclose(output_torch.cpu().numpy(), expected.cpu().numpy(), rtol=3e-3, atol=3e-3)
    print("    ✓ Zero weight case passed")

    # Test case 4: Uniform scale
    print("  Test case 4: Uniform scale")
    weight_torch = torch.randn(shape, dtype=torch.float32, device=device) * 50
    scale_torch = torch.full(shape, 2.0, dtype=torch.float32, device=device)

    output_torch = kernel(weight_torch, scale_torch)
    expected = embedding_head_quant_golden(weight_torch, scale_torch)

    max_diff = (output_torch - expected).abs().max().item()
    print(f"    Max difference: {max_diff:.6f}")
    assert_allclose(output_torch.cpu().numpy(), expected.cpu().numpy(), rtol=3e-3, atol=3e-3)
    print("    ✓ Uniform scale case passed")

def test_embedding_head_quant_large(device_id=None, run_mode="npu"):
    """Level 3: Large tensor for performance - Verify NPU performance."""
    print("Test: Large tensor (performance test)")

    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'
    shape = (256, 256)

    weight_torch = torch.randn(shape, dtype=torch.float32, device=device) * 50
    scale_torch = torch.rand(shape, dtype=torch.float32, device=device) * 3 + 0.5

    kernel = create_embedding_head_quant_kernel(shape, run_mode=run_mode)
    output_torch = kernel(weight_torch, scale_torch)
    expected = embedding_head_quant_golden(weight_torch, scale_torch)

    max_diff = (output_torch - expected).abs().max().item()
    print(f"  Max difference: {max_diff:.6f}")
    assert_allclose(output_torch.cpu().numpy(), expected.cpu().numpy(), rtol=3e-3, atol=3e-3)
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
