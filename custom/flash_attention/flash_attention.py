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

from flash_attention_golden import ifa_flash_torch, gen_block_table


B = 1
S1 = 1
N1 = 16
N2 = 1
D = 64
BLOCK_SIZE = 16

BS1 = B * S1
BLOCK_NUM = 1
MAX_BLOCKS = 1
G = N1 // N2
G_TILE = 16


@pypto.frontend.jit
def flash_attention_kernel(
    q: pypto.Tensor((B, G_TILE, 1, D), pypto.DT_BF16),
    k: pypto.Tensor((B, 1, BLOCK_SIZE, D), pypto.DT_BF16),
    v: pypto.Tensor((B, 1, BLOCK_SIZE, D), pypto.DT_BF16),
    out: pypto.Tensor((B, G_TILE, 1, D), pypto.DT_BF16),
):
    scale = D ** -0.5

    pypto.set_cube_tile_shapes([64, 64], [64, 64], [64, 64])
    pypto.set_vec_tile_shapes(1, G_TILE, BLOCK_SIZE, D)

    k_t = pypto.transpose(k, 2, 3)
    scores = pypto.matmul(q, k_t, out_dtype=pypto.DT_FP32)
    scores_scaled = scores * scale
    attn_weights = pypto.softmax(scores_scaled, dim=-1)
    attn_weights_bf16 = pypto.cast(attn_weights, pypto.DT_BF16)
    output = pypto.matmul(attn_weights_bf16, v, out_dtype=pypto.DT_FP32)
    output_bf16 = pypto.cast(output, pypto.DT_BF16)
    out.move(output_bf16)


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


def test_flash_attention_single_block(device_id=None, run_mode="npu"):
    print("=" * 60)
    print("Test: Flash Attention Single Block (s2_loop = 1)")
    print("=" * 60)

    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'

    b = 1
    s1 = 1
    n1 = 16
    n2 = 1
    d = 64
    block_size = 16

    kv_seq_len = block_size
    bs1 = b * s1
    block_num = math.ceil(kv_seq_len / block_size) * b

    torch.manual_seed(42)
    q_orig = torch.randn(bs1, n1, d, dtype=torch.bfloat16, device=device)
    k_orig = torch.randn(block_num, block_size, n2, d, dtype=torch.bfloat16, device=device)
    v_orig = torch.randn(block_num, block_size, n2, d, dtype=torch.bfloat16, device=device)

    q_4d = q_orig.view(b, n1, 1, d)
    k_4d = k_orig[0, :, 0, :].view(b, n2, block_size, d).expand(b, 1, block_size, d)
    v_4d = v_orig[0, :, 0, :].view(b, n2, block_size, d).expand(b, 1, block_size, d)

    out_4d = torch.empty(b, n1, 1, d, dtype=torch.bfloat16, device=device)

    flash_attention_kernel(q_4d, k_4d, v_4d, out_4d)

    out = out_4d.view(bs1, n1, d)

    kv_act_seqs = torch.tensor([kv_seq_len], dtype=torch.int32, device=device)
    max_blocks = block_num
    block_table = gen_block_table(kv_act_seqs, block_size, [b, max_blocks])

    golden_out = torch.empty(bs1, n1, d, dtype=torch.bfloat16, device=device)
    ifa_flash_torch(q_orig.clone(), k_orig, v_orig, block_table, kv_act_seqs, golden_out)

    print(f"Input q shape: {q_orig.shape}")
    print(f"Input k shape: {k_orig.shape}")
    print(f"Input v shape: {v_orig.shape}")
    print(f"Output shape: {out.shape}")
    print(f"kv_seq_len: {kv_seq_len}, s2_loop: 1")
    print(f"N1: {n1}, N2: {n2}, G_TILE: {G_TILE}")

    if run_mode == "npu":
        max_diff = (out - golden_out).abs().max().item()
        print(f"Max diff: {max_diff:.6f}")
        print(f"Out has nan: {torch.isnan(out).any().item()}")
        print(f"Out has inf: {torch.isinf(out).any().item()}")
        if torch.allclose(out, golden_out, rtol=1e-2, atol=1e-2):
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

    all_passed = test_flash_attention_single_block(device_id, args.run_mode)

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