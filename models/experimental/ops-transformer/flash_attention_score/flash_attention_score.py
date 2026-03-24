#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
FlashAttentionScore Operator for PyPTO - True FlashAttention Implementation
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
import pypto


BATCH_SIZE = 2
NUM_HEADS = 8
SEQ_LEN_Q = 64
SEQ_LEN_KV = 64
HEAD_DIM = 128

TILE_S1 = 64
TILE_S2 = 64


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


def flash_attention_score_golden_online_softmax(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale_value: float = 1.0,
) -> tuple:
    B, N, Sq, D = query.shape
    _, _, Skv, _ = key.shape
    
    attention_out = torch.zeros(B, N, Sq, D, dtype=query.dtype, device=query.device)
    softmax_max_out = torch.full((B, N, Sq, 1), float('-inf'), dtype=torch.float32, device=query.device)
    softmax_sum_out = torch.zeros(B, N, Sq, 1, dtype=torch.float32, device=query.device)
    
    num_s1_tiles = (Sq + TILE_S1 - 1) // TILE_S1
    num_s2_tiles = (Skv + TILE_S2 - 1) // TILE_S2
    
    for s1_tile in range(num_s1_tiles):
        s1_start = s1_tile * TILE_S1
        s1_end = min(s1_start + TILE_S1, Sq)
        
        q_block = query[:, :, s1_start:s1_end, :]
        
        m_running = torch.full((B, N, s1_end - s1_start, 1), float('-inf'), dtype=torch.float32, device=query.device)
        l_running = torch.zeros((B, N, s1_end - s1_start, 1), dtype=torch.float32, device=query.device)
        acc_running = torch.zeros((B, N, s1_end - s1_start, D), dtype=torch.float32, device=query.device)
        
        for s2_tile in range(num_s2_tiles):
            s2_start = s2_tile * TILE_S2
            s2_end = min(s2_start + TILE_S2, Skv)
            
            k_block = key[:, :, s2_start:s2_end, :]
            v_block = value[:, :, s2_start:s2_end, :]
            
            scores = torch.matmul(q_block.to(torch.float32), k_block.transpose(-2, -1).to(torch.float32)) * scale_value
            m_new = torch.maximum(m_running, scores.max(dim=-1, keepdim=True)[0])
            exp_scores = torch.exp(scores - m_new)
            
            m_diff = m_running - m_new
            exp_m_diff = torch.exp(m_diff)
            l_running_scaled = l_running * exp_m_diff
            exp_scores_sum = exp_scores.sum(dim=-1, keepdim=True)
            l_new = l_running_scaled + exp_scores_sum
            
            acc_new = acc_running * exp_m_diff + torch.matmul(exp_scores, v_block.to(torch.float32)).to(query.dtype)
            
            m_running = m_new
            l_running = l_new
            acc_running = acc_new
        
        acc_out = acc_running / l_running
        
        attention_out[:, :, s1_start:s1_end, :] = acc_out.to(query.dtype)
        softmax_max_out[:, :, s1_start:s1_end, :] = m_running
        softmax_sum_out[:, :, s1_start:s1_end, :] = l_running
    
    return attention_out, softmax_max_out, softmax_sum_out


@pypto.frontend.jit(debug_options={"runtime_debug_mode": 1})
def flash_attention_score_kernel(
    query: pypto.Tensor((BATCH_SIZE, NUM_HEADS, SEQ_LEN_Q, HEAD_DIM), pypto.DT_BF16),
    key: pypto.Tensor((BATCH_SIZE, NUM_HEADS, SEQ_LEN_KV, HEAD_DIM), pypto.DT_BF16),
    value: pypto.Tensor((BATCH_SIZE, NUM_HEADS, SEQ_LEN_KV, HEAD_DIM), pypto.DT_BF16),
    attention_out: pypto.Tensor((BATCH_SIZE, NUM_HEADS, SEQ_LEN_Q, HEAD_DIM), pypto.DT_BF16),
    softmax_max: pypto.Tensor((BATCH_SIZE, NUM_HEADS, SEQ_LEN_Q, 1), pypto.DT_FP32),
    softmax_sum: pypto.Tensor((BATCH_SIZE, NUM_HEADS, SEQ_LEN_Q, 1), pypto.DT_FP32),
    scale_value: float = 1.0,
):
    pypto.set_cube_tile_shapes([64, 64], [64, 64], [64, 64])
    pypto.set_vec_tile_shapes(1, 8, 16, HEAD_DIM)
    
    s1_loop_count = SEQ_LEN_Q // TILE_S1
    
    for s1_idx in pypto.loop(0, s1_loop_count, 1, name="S1_LOOP"):
        s1_start = s1_idx * TILE_S1
        
        q_block_shape = [BATCH_SIZE, NUM_HEADS, TILE_S1, HEAD_DIM]
        q_block = pypto.view(query, q_block_shape, [0, 0, s1_start, 0])
        
        m_running = pypto.full([BATCH_SIZE, NUM_HEADS, TILE_S1, 1], float('-inf'), pypto.DT_FP32)
        l_running = pypto.zeros(BATCH_SIZE, NUM_HEADS, TILE_S1, 1, dtype=pypto.DT_FP32)
        acc_running = pypto.zeros(BATCH_SIZE, NUM_HEADS, TILE_S1, HEAD_DIM, dtype=pypto.DT_FP32)
        
        s2_loop_count = SEQ_LEN_KV // TILE_S2
        
        for s2_idx in pypto.loop(0, s2_loop_count, 1, name="S2_LOOP"):
            s2_start = s2_idx * TILE_S2
            
            kv_block_shape = [BATCH_SIZE, NUM_HEADS, TILE_S2, HEAD_DIM]
            k_block = pypto.view(key, kv_block_shape, [0, 0, s2_start, 0])
            v_block = pypto.view(value, kv_block_shape, [0, 0, s2_start, 0])
            
            k_t = pypto.transpose(k_block, 2, 3)
            scores = pypto.matmul(q_block, k_t, out_dtype=pypto.DT_FP32)
            scores_scaled = pypto.mul(scores, scale_value)
            
            scores_max = pypto.amax(scores_scaled, dim=-1, keepdim=True)
            m_new = pypto.maximum(m_running, scores_max)
            
            scores_shifted = pypto.sub(scores_scaled, m_new)
            exp_scores = pypto.exp(scores_shifted)
            
            m_diff = pypto.sub(m_running, m_new)
            exp_m_diff = pypto.exp(m_diff)
            l_running_scaled = pypto.mul(l_running, exp_m_diff)
            
            exp_scores_sum = pypto.sum(exp_scores, dim=-1, keepdim=True)
            l_new = pypto.add(l_running_scaled, exp_scores_sum)
            
            acc_running_scaled = pypto.mul(acc_running, exp_m_diff)
            v_block_fp32 = pypto.cast(v_block, pypto.DT_FP32)
            exp_scores_weighted = pypto.matmul(exp_scores, v_block_fp32, out_dtype=pypto.DT_FP32)
            acc_new = pypto.add(acc_running_scaled, exp_scores_weighted)
            
            m_running = m_new
            l_running = l_new
            acc_running = acc_new
        
        acc_out = pypto.div(acc_running, l_running)
        acc_out_bf16 = pypto.cast(acc_out, pypto.DT_BF16)
        
        pypto.assemble(acc_out_bf16, [0, 0, s1_start, 0], attention_out)
        pypto.assemble(m_running, [0, 0, s1_start, 0], softmax_max)
        pypto.assemble(l_running, [0, 0, s1_start, 0], softmax_sum)


def test_flash_attention_score_basic(device_id=None, run_mode: str = "npu") -> None:
    print("=" * 60)
    print("Test: Flash Attention Score (True FlashAttention with Online Softmax)")
    print("=" * 60)
    
    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'
    
    torch.manual_seed(42)
    query_torch = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN_Q, HEAD_DIM, dtype=torch.bfloat16, device='cpu')
    key_torch = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN_KV, HEAD_DIM, dtype=torch.bfloat16, device='cpu')
    value_torch = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN_KV, HEAD_DIM, dtype=torch.bfloat16, device='cpu')
    
    query_torch = query_torch.to(device)
    key_torch = key_torch.to(device)
    value_torch = value_torch.to(device)
    
    scale_value = 1.0 / (HEAD_DIM ** 0.5)
    
    out_torch = torch.empty(BATCH_SIZE, NUM_HEADS, SEQ_LEN_Q, HEAD_DIM, dtype=torch.bfloat16, device=device)
    softmax_max_torch = torch.empty(BATCH_SIZE, NUM_HEADS, SEQ_LEN_Q, 1, dtype=torch.float32, device=device)
    softmax_sum_torch = torch.empty(BATCH_SIZE, NUM_HEADS, SEQ_LEN_Q, 1, dtype=torch.float32, device=device)
    
    flash_attention_score_kernel(
        query_torch, key_torch, value_torch,
        out_torch, softmax_max_torch, softmax_sum_torch,
        scale_value
    )
    
    golden_out, golden_max, golden_sum = flash_attention_score_golden_online_softmax(
        query_torch, key_torch, value_torch,
        scale_value=scale_value
    )
    
    print(f"Input shape: {query_torch.shape}")
    print(f"Output shape: {out_torch.shape}")
    print(f"Softmax max shape: {softmax_max_torch.shape}")
    print(f"Softmax sum shape: {softmax_sum_torch.shape}")
    
    if run_mode == "npu":
        max_diff_out = (out_torch.float() - golden_out.float()).abs().max().item()
        max_diff_max = (softmax_max_torch - golden_max).abs().max().item()
        max_diff_sum = (softmax_sum_torch - golden_sum).abs().max().item()
        
        print(f"Max diff (attention_out): {max_diff_out:.6f}")
        print(f"Max diff (softmax_max): {max_diff_max:.6f}")
        print(f"Max diff (softmax_sum): {max_diff_sum:.6f}")
        
        rtol = 0.01
        atol = 0.01
        assert torch.allclose(out_torch.float(), golden_out.float(), rtol=rtol, atol=atol), \
            f"Attention output mismatch! max_diff={max_diff_out}"
        assert torch.allclose(softmax_max_torch, golden_max, rtol=rtol, atol=atol), \
            f"Softmax max mismatch! max_diff={max_diff_max}"
        assert torch.allclose(softmax_sum_torch, golden_sum, rtol=rtol, atol=atol), \
            f"Softmax sum mismatch! max_diff={max_diff_sum}"
    
    print("✓ Flash Attention Score test passed")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="PyPTO Flash Attention Score Examples (True FlashAttention)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s              Run all tests
  %(prog)s --list       List all available tests
        """
    )
    parser.add_argument(
        'test_id',
        type=str,
        nargs='?',
        help='Test ID to run. If not specified, all tests will run.'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List all available tests and exit'
    )
    parser.add_argument(
        '--run_mode',
        type=str,
        nargs='?',
        default='npu',
        choices=["npu"],
        help='Run mode, currently only support npu.'
    )
    args = parser.parse_args()

    tests = {
        'test_basic': {
            'name': 'Basic Flash Attention Score',
            'description': 'Test FlashAttention with online softmax',
            'function': test_flash_attention_score_basic
        }
    }

    if args.list:
        print("\n" + "=" * 60)
        print("Available Tests")
        print("=" * 60 + "\n")
        for test_id, test_info in sorted(tests.items()):
            print(f"  ID: {test_id}")
            print(f"    name: {test_info['name']}")
            print(f"    description: {test_info['description']}\n")
        return

    if args.test_id is not None:
        if args.test_id not in tests:
            print(f"ERROR: Invalid test ID: {args.test_id}")
            print(f"Valid test IDs are: {', '.join(sorted(tests.keys()))}")
            print("\nUse --list to see all available tests.")
            sys.exit(1)

    print("\n" + "=" * 60)
    print("PyPTO Flash Attention Score Examples (True FlashAttention)")
    print("=" * 60 + "\n")

    device_id = None
    tests_to_run = []

    if args.test_id is not None:
        test = tests.get(args.test_id)
        if test is None:
            raise ValueError(f"Invalid test ID: {args.test_id}")
        tests_to_run = [(args.test_id, test)]
    else:
        tests_to_run = list(tests.items())

    if args.run_mode == "npu":
        device_id = get_device_id()
        if device_id is None:
            return
        import torch_npu
        torch.npu.set_device(device_id)
        print("Running tests that require NPU hardware...")
        print("Make sure CANN environment is configured and NPU is available\n")

    try:
        for test_id, test_info in tests_to_run:
            print(f"Running Test {test_id}: {test_info['name']}")
            test_info['function'](device_id, args.run_mode)

        if len(tests_to_run) > 1:
            print("=" * 60)
            print("All tests passed!")
            print("=" * 60)

    except Exception as e:
        print(f"\nError: {e}")
        raise


if __name__ == "__main__":
    main()
