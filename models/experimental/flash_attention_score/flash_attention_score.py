#!/usr/bin/env python3
# coding: utf-8
"""
FlashAttentionScore - PyPTO Implementation

Training scenario FlashAttention operator with softmax max/sum outputs.
Uses FP32 for all intermediate calculations.
"""

import os
import sys
import argparse
import numpy as np
import torch
import pypto


BATCH_SIZE = 2
NUM_HEADS = 8
SEQ_LEN_Q = 16
SEQ_LEN_KV = 16
HEAD_DIM = 64


def get_device_id():
    if 'TILE_FWK_DEVICE_ID' not in os.environ:
        print("Please set: export TILE_FWK_DEVICE_ID=0")
        return None
    try:
        return int(os.environ['TILE_FWK_DEVICE_ID'])
    except ValueError:
        print(f"ERROR: TILE_FWK_DEVICE_ID must be integer")
        return None


def flash_attention_score_golden(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
) -> tuple:
    """PyTorch reference implementation."""
    dtype = query.dtype
    
    scores = torch.matmul(query.float(), key.float().transpose(-2, -1))
    scores_scaled = scores * scale
    
    softmax_max = torch.amax(scores_scaled, dim=-1, keepdim=True)
    scores_shifted = scores_scaled - softmax_max
    exp_scores = torch.exp(scores_shifted)
    softmax_sum = torch.sum(exp_scores, dim=-1, keepdim=True)
    attn_weights = exp_scores / softmax_sum
    
    attention_out = torch.matmul(attn_weights, value.float())
    
    return attention_out.to(dtype), softmax_max.to(dtype), softmax_sum.to(dtype)


@pypto.frontend.jit
def flash_attention_score_kernel(
    query: pypto.Tensor((BATCH_SIZE, NUM_HEADS, SEQ_LEN_Q, HEAD_DIM), pypto.DT_BF16),
    key: pypto.Tensor((BATCH_SIZE, NUM_HEADS, SEQ_LEN_KV, HEAD_DIM), pypto.DT_BF16),
    value: pypto.Tensor((BATCH_SIZE, NUM_HEADS, SEQ_LEN_KV, HEAD_DIM), pypto.DT_BF16),
    attention_out: pypto.Tensor((BATCH_SIZE, NUM_HEADS, SEQ_LEN_Q, HEAD_DIM), pypto.DT_BF16),
    softmax_max: pypto.Tensor((BATCH_SIZE, NUM_HEADS, SEQ_LEN_Q, 1), pypto.DT_BF16),
    softmax_sum: pypto.Tensor((BATCH_SIZE, NUM_HEADS, SEQ_LEN_Q, 1), pypto.DT_BF16),
    scale: float,
):
    """
    FlashAttentionScore kernel using FP32 for intermediate calculations.
    
    Steps:
    1. scores = Q @ K^T (BF16 -> FP32)
    2. scores_scaled = scores * scale (FP32)
    3. softmax_max = amax(scores_scaled) (FP32)
    4. scores_shifted = scores_scaled - softmax_max (FP32)
    5. exp_scores = exp(scores_shifted) (FP32)
    6. softmax_sum = sum(exp_scores) (FP32)
    7. attn_weights = exp_scores / softmax_sum (FP32)
    8. output = attn_weights @ V (FP32 -> BF16)
    """
    pypto.set_cube_tile_shapes([64, 64], [64, 64], [64, 64])
    pypto.set_vec_tile_shapes(1, 8, 16, SEQ_LEN_KV)
    
    k_t = pypto.transpose(key, 2, 3)
    scores = pypto.matmul(query, k_t, out_dtype=pypto.DT_FP32)
    scores_scaled = pypto.mul(scores, scale)
    
    sm_max = pypto.amax(scores_scaled, dim=-1, keepdim=True)
    
    scores_shifted = pypto.sub(scores_scaled, sm_max)
    exp_scores = pypto.exp(scores_shifted)
    
    sm_sum = pypto.sum(exp_scores, dim=-1, keepdim=True)
    
    attn_weights = pypto.div(exp_scores, sm_sum)
    
    sm_max_bf16 = pypto.cast(sm_max, pypto.DT_BF16)
    softmax_max.move(sm_max_bf16)
    
    sm_sum_bf16 = pypto.cast(sm_sum, pypto.DT_BF16)
    softmax_sum.move(sm_sum_bf16)
    
    attn_weights_bf16 = pypto.cast(attn_weights, pypto.DT_BF16)
    output = pypto.matmul(attn_weights_bf16, value, out_dtype=pypto.DT_BF16)
    attention_out.move(output)


def test_flash_attention_score(device_id=None, run_mode: str = "npu"):
    """Test FlashAttentionScore operator."""
    print("=" * 60)
    print("Test: FlashAttentionScore")
    print("=" * 60)
    
    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'
    
    scale = 1.0 / (HEAD_DIM ** 0.5)
    
    q_torch = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN_Q, HEAD_DIM, dtype=torch.bfloat16, device=device)
    k_torch = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN_KV, HEAD_DIM, dtype=torch.bfloat16, device=device)
    v_torch = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN_KV, HEAD_DIM, dtype=torch.bfloat16, device=device)
    
    out_torch = torch.empty(BATCH_SIZE, NUM_HEADS, SEQ_LEN_Q, HEAD_DIM, dtype=torch.bfloat16, device=device)
    sm_max_torch = torch.empty(BATCH_SIZE, NUM_HEADS, SEQ_LEN_Q, 1, dtype=torch.bfloat16, device=device)
    sm_sum_torch = torch.empty(BATCH_SIZE, NUM_HEADS, SEQ_LEN_Q, 1, dtype=torch.bfloat16, device=device)
    
    flash_attention_score_kernel(q_torch, k_torch, v_torch, out_torch, sm_max_torch, sm_sum_torch, scale)
    
    golden_out, golden_max, golden_sum = flash_attention_score_golden(q_torch, k_torch, v_torch, scale)
    
    print(f"Input shape: Q{q_torch.shape}, K{k_torch.shape}, V{v_torch.shape}")
    print(f"Output shape: {out_torch.shape}")
    
    if run_mode == "npu":
        out_diff = (out_torch - golden_out).abs().max().item()
        max_diff = (sm_max_torch - golden_max).abs().max().item()
        sum_diff = (sm_sum_torch - golden_sum).abs().max().item()
        
        print(f"attention_out max diff: {out_diff:.6f}")
        print(f"softmax_max max diff: {max_diff:.6f}")
        print(f"softmax_sum max diff: {sum_diff:.6f}")
        
        if out_diff < 0.02 and max_diff < 0.1 and sum_diff < 1.0:
            print("✓ All outputs within tolerance")
        else:
            print("✗ Some outputs exceed tolerance")
    
    print()


def main():
    parser = argparse.ArgumentParser(description="FlashAttentionScore PyPTO Test")
    parser.add_argument('--run_mode', type=str, default='npu', choices=["npu"], help='Run mode')
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("FlashAttentionScore PyPTO Test")
    print("=" * 60 + "\n")
    
    device_id = None
    if args.run_mode == "npu":
        device_id = get_device_id()
        if device_id is None:
            return
        import torch_npu
        torch.npu.set_device(device_id)
        print("Running on NPU...")
    
    test_flash_attention_score(device_id, args.run_mode)
    
    print("=" * 60)
    print("FlashAttentionScore test completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()