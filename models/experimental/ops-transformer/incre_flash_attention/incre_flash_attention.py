#!/usr/bin/env python3
# coding: utf-8
"""
IncreFlashAttention - PyPTO Implementation

Incremental inference FlashAttention (query S axis fixed to 1).
Formula: Attention(Q,K,V) = Softmax(QK^T/sqrt(d))V
"""

import os
import sys
import argparse
import torch
import pypto


BATCH_SIZE = 2
NUM_HEADS = 8
SEQ_LEN_Q = 1
SEQ_LEN_KV = 16
HEAD_DIM = 64


def get_device_id():
    if 'TILE_FWK_DEVICE_ID' not in os.environ:
        print("Please set: export TILE_FWK_DEVICE_ID=0")
        return None
    try:
        return int(os.environ['TILE_FWK_DEVICE_ID'])
    except ValueError:
        print("ERROR: TILE_FWK_DEVICE_ID must be integer")
        return None


def incre_flash_attention_golden(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """PyTorch reference implementation of incremental FlashAttention."""
    dtype = query.dtype
    
    scores = torch.matmul(query.float(), key.float().transpose(-2, -1))
    scores_scaled = scores * scale
    
    attn_weights = torch.softmax(scores_scaled, dim=-1)
    
    attention_out = torch.matmul(attn_weights, value.float())
    
    return attention_out.to(dtype)


@pypto.frontend.jit
def incre_flash_attention_kernel(
    query: pypto.Tensor((BATCH_SIZE, NUM_HEADS, SEQ_LEN_Q, HEAD_DIM), pypto.DT_BF16),
    key: pypto.Tensor((BATCH_SIZE, NUM_HEADS, SEQ_LEN_KV, HEAD_DIM), pypto.DT_BF16),
    value: pypto.Tensor((BATCH_SIZE, NUM_HEADS, SEQ_LEN_KV, HEAD_DIM), pypto.DT_BF16),
    attention_out: pypto.Tensor((BATCH_SIZE, NUM_HEADS, SEQ_LEN_Q, HEAD_DIM), pypto.DT_BF16),
    scale: float,
):
    """
    IncreFlashAttention kernel for incremental inference.
    
    Query S axis is fixed to 1 (incremental generation).
    Key and Value have variable sequence lengths (from KV cache).
    """
    pypto.set_cube_tile_shapes([64, 64], [64, 64], [64, 64])
    pypto.set_vec_tile_shapes(1, 8, 16, SEQ_LEN_KV)
    
    k_t = pypto.transpose(key, 2, 3)
    scores = pypto.matmul(query, k_t, out_dtype=pypto.DT_FP32)
    scores_scaled = pypto.mul(scores, scale)
    
    attn_weights = pypto.softmax(scores_scaled, dim=-1)
    
    attn_weights_bf16 = pypto.cast(attn_weights, pypto.DT_BF16)
    output = pypto.matmul(attn_weights_bf16, value, out_dtype=pypto.DT_BF16)
    attention_out.move(output)


def test_incre_flash_attention(device_id=None, run_mode: str = "npu"):
    """Test IncreFlashAttention operator."""
    print("=" * 60)
    print("Test: IncreFlashAttention (Sq=1)")
    print("=" * 60)
    
    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'
    
    scale = 1.0 / (HEAD_DIM ** 0.5)
    
    q_torch = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN_Q, HEAD_DIM, dtype=torch.bfloat16, device=device)
    k_torch = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN_KV, HEAD_DIM, dtype=torch.bfloat16, device=device)
    v_torch = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN_KV, HEAD_DIM, dtype=torch.bfloat16, device=device)
    
    out_torch = torch.empty(BATCH_SIZE, NUM_HEADS, SEQ_LEN_Q, HEAD_DIM, dtype=torch.bfloat16, device=device)
    
    incre_flash_attention_kernel(q_torch, k_torch, v_torch, out_torch, scale)
    
    golden_out = incre_flash_attention_golden(q_torch, k_torch, v_torch, scale)
    
    print(f"Input shape: Q{q_torch.shape}, K{k_torch.shape}, V{v_torch.shape}")
    print(f"Output shape: {out_torch.shape}")
    
    if run_mode == "npu":
        max_diff = (out_torch - golden_out).abs().max().item()
        print(f"max diff: {max_diff:.6f}")
        
        if max_diff < 0.02:
            print(f"✓ Output within tolerance (max_diff={max_diff:.6f} < 0.02)")
        else:
            print(f"✗ Output exceeds tolerance (max_diff={max_diff:.6f} >= 0.02)")
    
    print()
    return max_diff if run_mode == "npu" else None


def main():
    parser = argparse.ArgumentParser(description="IncreFlashAttention PyPTO Test")
    parser.add_argument('--run_mode', type=str, default='npu', choices=["npu"], help='Run mode')
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("IncreFlashAttention PyPTO Test")
    print("Incremental inference FlashAttention (Sq=1)")
    print("=" * 60 + "\n")
    
    device_id = None
    if args.run_mode == "npu":
        device_id = get_device_id()
        if device_id is None:
            return
        import torch_npu
        torch.npu.set_device(device_id)
        print("Running on NPU...")
    
    max_diff = test_incre_flash_attention(device_id, args.run_mode)
    
    print("=" * 60)
    if max_diff is not None and max_diff < 0.02:
        print("IncreFlashAttention test PASSED!")
    else:
        print("IncreFlashAttention test completed.")
    print("=" * 60)


if __name__ == "__main__":
    main()