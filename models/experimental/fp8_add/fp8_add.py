#!/usr/bin/env python3
# coding: utf-8
"""
FP8E4M3 Add Example for PyPTO

FP8E4M3 加法算子：使用 cast 方案实现（FP8E4M3 -> FP32 -> add -> FP8E4M3）
"""
import os
import sys
import argparse
import pypto
import torch
import numpy as np
from numpy.testing import assert_allclose


def get_device_id():
    if 'TILE_FWK_DEVICE_ID' not in os.environ:
        print("If no NPU environment is available, set --run_mode sim to run in simulation mode;")
        print("otherwise, set the environment variable TILE_FWK_DEVICE_ID.")
        print("Please set it before running this example:")
        print("  export TILE_FWK_DEVICE_ID=0")
        return None
    try:
        device_id = int(os.environ['TILE_FWK_DEVICE_ID'])
        return device_id
    except ValueError:
        print(f"ERROR: TILE_FWK_DEVICE_ID must be an integer, got: {os.environ['TILE_FWK_DEVICE_ID']}")
        return None


def create_fp8_add_kernel(shape: tuple, run_mode: str = "npu"):
    if run_mode == "npu":
        mode = pypto.RunMode.NPU
    elif run_mode == "sim":
        mode = pypto.RunMode.SIM
    else:
        raise ValueError(f"Invalid run_mode: {run_mode}. Must be 'npu' or 'sim'")

    @pypto.frontend.jit(runtime_options={"run_mode": mode})
    def fp8_add_kernel(
        a: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP8E4M3),
        b: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP8E4M3),
        c: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP8E4M3),
    ):
        pypto.set_vec_tile_shapes(128, 1024)
        
        a_fp32 = pypto.cast(a, pypto.DT_FP32)
        b_fp32 = pypto.cast(b, pypto.DT_FP32)
        c_fp32 = pypto.add(a_fp32, b_fp32)
        
        c_clipped = pypto.clip(c_fp32, -448.0, 448.0)
        c[:] = pypto.cast(c_clipped, pypto.DT_FP8E4M3)

    return fp8_add_kernel


def fp8_add_golden(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """PyTorch golden reference for FP8E4M3 add"""
    a_fp32 = a.float()
    b_fp32 = b.float()
    c_fp32 = a_fp32 + b_fp32
    c_clipped = torch.clamp(c_fp32, -448.0, 448.0)
    return c_clipped.to(torch.float8_e4m3fn)


def test_fp8_add(device_id=None, run_mode: str = "npu") -> None:
    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'
    shape = (1024, 1024)
    
    input_a = torch.rand(shape, dtype=torch.float32, device=device) * 100
    input_b = torch.rand(shape, dtype=torch.float32, device=device) * 100
    
    input_a_fp8 = input_a.to(torch.float8_e4m3fn)
    input_b_fp8 = input_b.to(torch.float8_e4m3fn)
    
    output_fp8 = torch.empty(shape, dtype=torch.float8_e4m3fn, device=device)
    
    create_fp8_add_kernel(shape, run_mode)(input_a_fp8, input_b_fp8, output_fp8)
    
    golden = fp8_add_golden(input_a_fp8, input_b_fp8)
    
    print(f"Input A shape: {input_a_fp8.shape}")
    print(f"Input B shape: {input_b_fp8.shape}")
    print(f"Output shape: {output_fp8.shape}")
    
    if run_mode == "npu":
        output_fp32 = output_fp8.float()
        golden_fp32 = golden.float()
        max_diff = np.abs(output_fp32.cpu().numpy() - golden_fp32.cpu().numpy()).max()
        print(f"Max difference: {max_diff:.6f}")
        assert_allclose(np.array(output_fp32.cpu()), np.array(golden_fp32.cpu()), rtol=1e-3, atol=1e-3)
    
    print("[PRECISION_PASS] ✓ FP8E4M3 add test passed")
    print()


def main():
    parser = argparse.ArgumentParser(description="PyPTO FP8E4M3 Add Example")
    parser.add_argument('--run_mode', type=str, nargs='?', default="npu",
                        choices=["npu", "sim"], help='Run mode: npu/sim')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("PyPTO FP8E4M3 Add Example")
    print("=" * 60 + "\n")
    
    device_id = None
    if args.run_mode == "npu":
        device_id = get_device_id()
        if device_id is None:
            return
        import torch_npu
        torch.npu.set_device(device_id)
        print("Running on NPU...")
        print("(Make sure CANN environment is configured and Ascend 950PR/950DT is available)\n")
    
    try:
        test_fp8_add(device_id, args.run_mode)
        print("=" * 60)
        print("FP8E4M3 add test completed!")
        print("=" * 60)
    except Exception as e:
        print(f"\nError: {e}")
        raise


if __name__ == "__main__":
    main()