from dataclasses import dataclass
import argparse
import os
import sys
import pypto
import torch
import torch_npu
import numpy as np
import math
import time
from numpy.testing import assert_allclose


def mhc_res_golden(x, h):
    """
    Golden function for mhc_res: out[b*N+t,s,d,d] = Σ_r h[r,t] × x[b*N+r,s,d]
    
    Args:
        x: [batch * num_streams, seq, dim]
        h: [num_streams, num_streams]
    
    Returns:
        out: [batch * num_streams, seq, dim]
    """
    batch_n = x.shape[0]
    num_streams = h.shape[0]
    batch = batch_n // num_streams
    seq = x.shape[1]
    dim = x.shape[2]
    
    # Reshape x to [batch, num_streams, seq, dim]
    x_reshaped = x.reshape(batch, num_streams, seq, dim)
    
    # Weighted sum over num_streams (r) dimension
    # h: [num_streams, num_streams] -> [1, num_streams, num_streams, 1, 1] for broadcasting
    h_expanded = h.view(1, num_streams, num_streams, 1, 1)
    
    # x_reshaped: [batch, num_streams, seq, dim] -> [batch, num_streams, 1, seq, dim]
    x_expanded = x_reshaped.unsqueeze(2)
    
    # weighted: [batch, num_streams, num_streams, seq, dim]
    weighted = x_expanded * h_expanded
    
    # Sum over r dimension (dim=1)
    out_expanded = weighted.sum(dim=1)
    
    # Reshape to [batch * num_streams, seq, dim]
    out = out_expanded.reshape(batch * num_streams, seq, dim)
    
    return out


@pypto.jit(
    debug_options={"runtime_debug_mode": 1, "compile_debug_mode": 1},
    runtime_options={"device_sched_mode": 3},
)
def mhc_res_kernel(x: pypto.Tensor, h: pypto.Tensor, out: pypto.Tensor) -> None:
    batch_n = x.shape[0]
    num_streams = h.shape[0]
    batch = batch_n // num_streams
    seq = x.shape[1]
    dim = x.shape[2]
    
    pypto.set_vec_tile_shapes(8, 8, 8, 8)
    
    # Reshape x to [batch, num_streams, seq, dim]
    x_reshaped = pypto.reshape(x, [batch, num_streams, 1, seq * dim])
    
    # Reshape h to [1, num_streams, num_streams, 1] for broadcasting
    h_reshaped = pypto.reshape(h, [1, num_streams, num_streams, 1])

    x_expanded = pypto.expand_clone(x_reshaped, [batch, num_streams, num_streams, seq * dim])

    h_expanded = pypto.expand_clone(h_reshaped, [batch, num_streams, num_streams, 1])

    h_expanded1 = pypto.expand_clone(h_expanded, [batch, num_streams, num_streams, seq * dim])
    
    weighted = pypto.mul(x_expanded, h_expanded1)

    out_expanded = pypto.sum(weighted, 1, True)

    out1 = pypto.reshape(out_expanded, [batch * num_streams, seq, dim])
    
    # Assemble result to output tensor
    pypto.assemble(out1, [0, 0, 0], out)


def test_mhc_res():
    """Test mhc_res operator"""
    print("\n" + "="*60)
    print("Testing mhc_res")
    print("="*60)
    
    # Test configurations
    test_configs = [
        # Level 0: Small scale
        {"batch": 2, "num_streams": 2, "seq": 2, "dim": 2, "dtype": torch.float32},
        # Level 1: Medium scale
        {"batch": 8, "num_streams": 4, "seq": 1024, "dim": 512, "dtype": torch.float32},
        # Level 2: Large scale
        {"batch": 16, "num_streams": 8, "seq": 256, "dim": 1024, "dtype": torch.float32},
    ]
    
    for i, config in enumerate(test_configs):
        print(f"\nTest case {i+1}: {config}")
        
        batch = config["batch"]
        num_streams = config["num_streams"]
        seq = config["seq"]
        dim = config["dim"]
        dtype = config["dtype"]

        # Generate test data on NPU
        x = torch.randn(batch * num_streams, seq, dim, dtype=dtype, device='npu')
        h = torch.randn(num_streams, num_streams, dtype=dtype, device='npu')
        
        # Golden result (on CPU for comparison)
        x_cpu = x.cpu()
        h_cpu = h.cpu()
        expected = mhc_res_golden(x_cpu, h_cpu)
        
        # Map torch dtype to pypto dtype
        dtype_map = {
            torch.float32: pypto.DT_FP32,
            torch.float16: pypto.DT_FP16,
            torch.bfloat16: pypto.DT_BF16,
        }
        pto_dtype = dtype_map[x.dtype]

        out_shape = (batch * num_streams, seq, dim)
        out = torch.zeros(out_shape, dtype=dtype, device='npu')

        input_tensors = {
            x: [],
            h: [],
        }

        output_tensors = {
            out: [],
        }

        pto_inputs = [pypto.from_torch(tensor, dynamic_axis=axis) for tensor, axis in input_tensors.items()]
        pto_outputs = [pypto.from_torch(tensor, dynamic_axis=axis) for tensor, axis in output_tensors.items()]

        # PyPTO result
        mhc_res_kernel(*pto_inputs, *pto_outputs)

        print(f"Expected shape: {expected.shape}")
        print(expected)
        print(f"Output shape: {out.shape}")
        print(out)

        assert_allclose(expected.cpu().numpy(), out.cpu().numpy(), rtol=1e-3, atol=1e-3)
        print(f"Test case {i+1} passed!")


if __name__ == "__main__":
    test_mhc_res()