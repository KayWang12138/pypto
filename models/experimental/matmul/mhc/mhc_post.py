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


def mhc_post_golden(x, h):
    """
    Golden function for mhc_post: out[b*N+n,s,d] = x[b,s,d] × h[n]
    
    Args:
        x: [batch, seq, dim]
        h: [num_streams]
    
    Returns:
        out: [batch * num_streams, seq, dim]
    """
    batch = x.shape[0]
    seq = x.shape[1]
    dim = x.shape[2]
    num_streams = h.shape[0]
    
    # Expand h to [1, num_streams, 1, 1] for broadcasting
    h_expanded = h.view(1, num_streams, 1, 1)
    
    # Expand x to [batch, 1, seq, dim] for broadcasting
    x_expanded = x.unsqueeze(1)
    
    # Compute weighted output
    out_expanded = torch.mul(h_expanded, x_expanded)
    
    # Reshape to [batch * num_streams, seq, dim]
    out = out_expanded.reshape(batch * num_streams, seq, dim)
    
    return out


@pypto.jit(
    debug_options={"runtime_debug_mode": 1, "compile_debug_mode": 1},
    runtime_options={"device_sched_mode": 3},
)
def mhc_post_kernel(x: pypto.Tensor, h: pypto.Tensor, out: pypto.Tensor) -> None:
    batch = x.shape[0]
    seq = x.shape[1]
    dim = x.shape[2]
    num_streams = h.shape[0]
    
    pypto.set_vec_tile_shapes(1, 4, 64, 64)
    
    # Expand x to [batch, 1, seq, dim] for broadcasting
    x_expanded = pypto.unsqueeze(x, dim=1)
    
    # Expand h to [1, num_streams, 1, 1] for broadcasting
    h_expanded = pypto.reshape(h, [1, num_streams, 1, 1])
    
    # Expand x to [batch, num_streams, seq, dim]
    x1 = pypto.expand_clone(x_expanded, [batch, num_streams, seq, dim])
    
    # Expand h to [batch, num_streams, seq, dim]
    h1 = pypto.concat([h_expanded] * batch, dim=0)
    h2 = pypto.concat([h1] * seq, dim=-2)
    h3 = pypto.concat([h2] * dim, dim=-1)

    # Weighted multiplication
    out_expanded = pypto.mul(x1, h3)
    
    # Reshape to [batch * num_streams, seq, dim]
    result = pypto.reshape(out_expanded, [batch * num_streams, seq, dim])
    
    # Assemble result to output tensor
    pypto.assemble(result, [0, 0, 0], out)

    # pypto.set_vec_tile_shapes(1, 4, 64, 64)

    # batch, seq, dim = x.shape
    # num_streams = h.shape[0]

    # tile_b = 1
    # b_loop = batch // tile_b

    # for idx in pypto.loop(b_loop):
    #     b_offset = idx * tile_b
    #     b_offset_end = (idx + 1) * tile_b

    #     x_sub = x[b_offset:b_offset_end, ...]

    #     # Expand x to [batch, 1, seq, dim] for broadcasting
    #     x_expanded = pypto.unsqueeze(x_sub, dim=1)
        
    #     # Expand h to [1, num_streams, 1, 1] for broadcasting
    #     h_expanded = pypto.reshape(h, [1, num_streams, 1, 1])
        
    #     # Expand x to [batch, num_streams, seq, dim]
    #     x1 = pypto.expand_clone(x_expanded, [1, num_streams, seq, dim])
        
    #     # Expand h to [batch, num_streams, seq, dim]
    #     h1 = pypto.concat([h_expanded] * 1, dim=0)
    #     h2 = pypto.concat([h1] * seq, dim=-2)
    #     h3 = pypto.concat([h2] * dim, dim=-1)

    #     out_expanded = pypto.mul(x1, h3)
    #     result = pypto.reshape(out_expanded, [num_streams, seq, dim])

    #     out_b_offset = idx * num_streams

    #     pypto.assemble(result, [out_b_offset, 0, 0], out)


def test_mhc_post():
    """Test mhc_post operator"""
    print("\n" + "="*60)
    print("Testing mhc_post")
    print("="*60)
    
    # Test configurations
    test_configs = [
        # 耗时分析 (4, 512, 256) n=4, AscendC：21μs  torch.einsum: 81μs  PyPTO: 267μs
        # {"batch": 4, "num_streams": 4, "seq": 512, "dim": 256, "dtype": torch.float32},
        # {"batch": 4, "num_streams": 4, "seq": 4, "dim": 4, "dtype": torch.float32},
        # 347μ
        # {"batch": 8, "num_streams": 4, "seq": 1024, "dim": 512, "dtype": torch.float32},
        # 
        # {"batch": 16, "num_streams": 4, "seq": 2048, "dim": 1024, "dtype": torch.float32},
        # {"batch": 32, "num_streams": 4, "seq": 2048, "dim": 1024, "dtype": torch.float32},
        {"batch": 4, "num_streams": 4, "seq": 512, "dim": 1024, "dtype": torch.float32},
        # {"batch": 8, "num_streams": 4, "seq": 256, "dim": 512, "dtype": torch.float32},
        # {"batch": 16, "num_streams": 4, "seq": 1024, "dim": 1024, "dtype": torch.float32},
    ]
    
    for i, config in enumerate(test_configs):
        print(f"\nTest case {i+1}: {config}")
        
        batch = config["batch"]
        num_streams = config["num_streams"]
        seq = config["seq"]
        dim = config["dim"]
        dtype = config["dtype"]

        # Generate test data on NPU
        x = torch.randn(batch, seq, dim, dtype=dtype, device='npu')
        h = torch.randn(num_streams, dtype=dtype, device='npu')
        
        # Golden result (on CPU for comparison)
        x_cpu = x.cpu()
        h_cpu = h.cpu()
        expected = mhc_post_golden(x_cpu, h_cpu)
        
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
        mhc_post_kernel(*pto_inputs, *pto_outputs)

        print(f"Expected shape: {expected.shape}")
        print(f"Output shape: {out.shape}")

        assert_allclose(expected.cpu().numpy(), out.cpu().numpy(), rtol=1e-3, atol=1e-3)
        print(f"Test case {i+1} passed!")


if __name__ == "__main__":
    test_mhc_post()