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


def mhc_pre_golden(x, h):
    """
    Golden function for mhc_pre: out[b,s,d] = Σ_n h[n] × x[b*N+n,s,d]
    
    Args:
        x: [batch * num_streams, seq, dim]
        h: [num_streams]
    
    Returns:
        out: [batch, seq, dim]
    """
    batch_n = x.shape[0]
    num_streams = h.shape[0]
    batch = batch_n // num_streams
    seq = x.shape[1]
    dim = x.shape[2]
    
    # Reshape x to [batch, num_streams, seq, dim]
    x_reshaped = x.reshape(batch, num_streams, seq, dim)
    
    # Weighted sum over num_streams dimension
    # h: [num_streams] -> [1, num_streams, 1, 1] for broadcasting
    h_expanded = h.view(1, num_streams, 1, 1)
    weighted = x_reshaped * h_expanded
    out = weighted.sum(dim=1)  # Sum over num_streams
    
    return out


@dataclass
class ShapeConfig:
    ori_shape: list
    tile_size: int
    m_tile_shape: list
    n_tile_shape: list
    vector_tile_shape: list


@pypto.jit(
    debug_options={"runtime_debug_mode": 1, "compile_debug_mode": 1},
    runtime_options={"device_sched_mode": 3},
)
def mhc_pre_kernel(x: pypto.Tensor, h: pypto.Tensor, out: pypto.Tensor) -> None:
    batch_n = x.shape[0]
    seq = x.shape[1]
    dim = x.shape[2]
    num_streams = h.shape[0]
    batch = batch_n // num_streams
    
    pypto.set_vec_tile_shapes(8, 8, 8, 8)
    
    # Reshape x to [batch, num_streams, seq, dim]
    x_reshaped = pypto.reshape(x, [batch, num_streams, seq, dim])
    
    # Reshape h to [1, num_streams, 1, 1] for broadcasting
    h_expanded = pypto.reshape(h, [1, num_streams, 1, 1])
    
    # Expand h to [batch, num_streams, 1, 1]
    h1 = pypto.concat([h_expanded] * batch, dim=0)
    
    # Expand h to [batch, num_streams, seq, 1]
    h2 = pypto.concat([h1] * seq, dim=-2)
    
    # Expand h to [batch, num_streams, seq, dim]
    h3 = pypto.concat([h2] * dim, dim=-1)
    
    # Weighted multiplication
    weighted = pypto.mul(x_reshaped, h3)
    
    # Sum over num_streams dimension (dim=1)
    result = pypto.sum(weighted, dim=1)
    
    # Assemble result to output tensor
    pypto.assemble(result, [0, 0, 0], out)


def gen_mhc_pre(x, h, tile_config):
    x = x.npu()
    h = h.npu()
    
    batch_n = x.shape[0]
    seq = x.shape[1]
    dim = x.shape[2]
    num_streams = h.shape[0]
    batch = batch_n // num_streams
    
    out_shape = (batch, seq, dim)
    out = torch.zeros(out_shape, dtype=x.dtype, device='npu')
    
    input_tensors = {
        x: [],
        h: [],
    }
    output_tensors = {
        out: [],
    }
    
    pto_inputs = [pypto.from_torch(tensor, dynamic_axis=axis) for tensor, axis in input_tensors.items()]
    pto_outputs = [pypto.from_torch(tensor, dynamic_axis=axis) for tensor, axis in output_tensors.items()]
    
    mhc_pre_kernel(*pto_inputs, *pto_outputs)
    
    return out


def test_mhc_pre(params):
    batch = params["batch"]
    num_streams = params["num_streams"]
    seq = params["seq"]
    dim = params["dim"]
    dtype = params["dtype"]
    tile_size = params["tile_size"]
    m_tile_shape = params["m_tile_shape"]
    n_tile_shape = params["n_tile_shape"]
    vector_tile_shape = params["vector_tile_shape"]
    
    tile_config = ShapeConfig([batch, num_streams, seq, dim], tile_size, m_tile_shape, n_tile_shape, vector_tile_shape)
    
    x = torch.randn(batch * num_streams, seq, dim, dtype=dtype)
    h = torch.randn(num_streams, dtype=dtype)
    
    expected = mhc_pre_golden(x, h)
    
    result = gen_mhc_pre(x, h, tile_config)
    
    if dtype == torch.float32:
        atol, rtol = 1e-5, 1e-5
    elif dtype == torch.float16:
        atol, rtol = 1e-4, 1e-3
    else:  # bfloat16
        atol, rtol = 1e-3, 4.0e-3
    
    print(f"Expected shape: {expected.shape}")
    print(expected)
    print(f"Output shape: {result.shape}")
    print(result)
    assert_allclose(expected.cpu().numpy(), result.cpu().numpy(), rtol=rtol, atol=atol)
    print("success")


def get_params(case_name):
    if case_name == "testcase1":
        params = {
            "batch": 2, "num_streams": 2, "seq": 2, "dim": 2, "dtype": torch.float32,
            "tile_size": 8, "m_tile_shape": [8, 8], "n_tile_shape": [8, 8], "vector_tile_shape": [8, 8, 8]
        }
    elif case_name == "testcase2":
        params = {
            "batch": 2, "num_streams": 3, "seq": 4, "dim": 8, "dtype": torch.float32,
            "tile_size": 8, "m_tile_shape": [8, 8], "n_tile_shape": [8, 8], "vector_tile_shape": [8, 8, 8]
        }
    elif case_name == "testcase3":
        params = {
            "batch": 4, "num_streams": 4, "seq": 16, "dim": 16, "dtype": torch.float32,
            "tile_size": 16, "m_tile_shape": [16, 16], "n_tile_shape": [16, 16], "vector_tile_shape": [16, 16, 16]
        }
    else:
        raise Exception(f"Can't get params, Case({case_name})")
    return params


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Testing mhc_pre")
    print("="*60)
    
    test_cases = ["testcase1", "testcase2", "testcase3"]
    
    for case_name in test_cases:
        print(f"\nRunning {case_name}...")
        params = get_params(case_name)
        test_mhc_pre(params)
    
    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60)
