#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# the CANN Open Software License Agreement Version 2.0 (the "License").
# You should have received a copy of the License along with this program. If not, see
# <http://www.huawei.com/> for a copy of the License.
# -----------------------------------------------------------------------------------------------------------

import os
import sys
import math
import torch
import pypto
import argparse


B = 1
N1 = 16
N2 = 1
D = 64
BLOCK_SIZE = 16


@pypto.frontend.jit
def flash_attention_no_loop_kernel(
    q: pypto.Tensor((B, N1, D), pypto.DT_BF16),
    k: pypto.Tensor((1, N2, BLOCK_SIZE, D), pypto.DT_BF16),
    v: pypto.Tensor((1, N2, BLOCK_SIZE, D), pypto.DT_BF16),
    out: pypto.Tensor((B, N1, D), pypto.DT_BF16),
):
    scale = D ** -0.5

    pypto.set_cube_tile_shapes([64, 64], [64, 64], [64, 64])
    pypto.set_vec_tile_shapes(1, N1, BLOCK_SIZE, D)

    qi_4d = pypto.reshape(q, [B, N1, 1, D])
    
    kj_4d = k[0, 0, 0:BLOCK_SIZE, :]
    kj_4d = pypto.reshape(kj_4d, [1, 1, BLOCK_SIZE, D])
    
    vj_4d = v[0, 0, 0:BLOCK_SIZE, :]
    vj_4d = pypto.reshape(vj_4d, [1, 1, BLOCK_SIZE, D])
    
    kj_t = pypto.transpose(kj_4d, 2, 3)
    
    scores = pypto.matmul(qi_4d, kj_t, out_dtype=pypto.DT_FP32)
    scores_scaled = scores * scale
    
    attn_weights = pypto.softmax(scores_scaled, dim=-1)
    attn_weights_bf16 = pypto.cast(attn_weights, pypto.DT_BF16)
    
    output = pypto.matmul(attn_weights_bf16, vj_4d, out_dtype=pypto.DT_FP32)
    output_bf16 = pypto.cast(output, pypto.DT_BF16)
    
    output_2d = pypto.reshape(output_bf16, [N1, D])
    out[0, 0:N1, :].move(output_2d)


def get_device_id():
    if 'TILE_FWK_DEVICE_ID' not in os.environ:
        print("Please set the environment variable TILE_FWK_DEVICE_ID before running:")
        print("  export TILE_FWK_DEVICE_ID=0")
        return None
    try:
        device_id = int(os.environ['TILE_FWK_DEVICE_ID'])
        return device_id
    except ValueError:
        print(f"ERROR: TILE_FWK_DEVICE_ID must be an integer, got: {os.environ['TILE_FWK_DEVICE_ID']}")
        return None


def test_flash_attention_no_loop(device_id=None, run_mode="npu"):
    print("=" * 60)
    print("Test: Flash Attention No Loop")
    print("=" * 60)

    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'

    b = 1
    n1 = 16
    n2 = 1
    d = 64
    block_size = 16

    torch.manual_seed(42)
    q = torch.randn(b, n1, d, dtype=torch.bfloat16, device=device)
    k = torch.randn(1, n2, block_size, d, dtype=torch.bfloat16, device=device)
    v = torch.randn(1, n2, block_size, d, dtype=torch.bfloat16, device=device)

    out = torch.empty(b, n1, d, dtype=torch.bfloat16, device=device)

    flash_attention_no_loop_kernel(q, k, v, out)

    scale = d ** -0.5
    golden = torch.matmul(q, k[0, 0].transpose(-2, -1)) * scale
    golden = torch.softmax(golden, dim=-1)
    golden = torch.matmul(golden, v[0, 0])

    print(f"Input q shape: {q.shape}")
    print(f"Input k shape: {k.shape}")
    print(f"Input v shape: {v.shape}")
    print(f"Output shape: {out.shape}")

    if run_mode == "npu":
        max_diff = (out - golden).abs().max().item()
        print(f"Max diff: {max_diff:.6f}")
        print(f"Out has nan: {torch.isnan(out).any().item()}")
        print(f"Out has inf: {torch.isinf(out).any().item()}")
        if torch.allclose(out, golden, rtol=1e-2, atol=1e-2):
            print("✓ Test passed!")
            return True
        else:
            print("✗ Test failed!")
            return False
    print()
    return True


def main():
    parser = argparse.ArgumentParser(description="PyPTO Flash Attention Test")
    parser.add_argument('--run_mode', type=str, default='npu', choices=["npu"], help='Run mode')
    args = parser.parse_args()

    print("=" * 60)
    print("PyPTO Flash Attention Test")
    print("=" * 60)
    print()

    device_id = None
    if args.run_mode == "npu":
        device_id = get_device_id()
        if device_id is None:
            return
        import torch_npu
        torch.npu.set_device(device_id)
        print("Running on NPU...")
        print()

    all_passed = test_flash_attention_no_loop(device_id, args.run_mode)

    if all_passed:
        print("=" * 60)
        print("Test passed!")
        print("=" * 60)
    else:
        print("=" * 60)
        print("Test failed!")
        print("=" * 60)


if __name__ == "__main__":
    main()