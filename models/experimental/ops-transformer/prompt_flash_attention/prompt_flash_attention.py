#!/usr/bin/env python3
# coding: utf-8
"""
PromptFlashAttention - PyPTO Implementation

Full inference scenario FlashAttention operator.

Formula:
    Attention(Q, K, V) = Softmax(Q @ K^T / sqrt(d)) @ V
"""

import os
import sys
import argparse
import torch
import pypto


BATCH_SIZE = 1
NUM_HEADS = 8
SEQ_LEN_Q = 128
SEQ_LEN_KV = 128
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


def prompt_flash_attention_golden(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """PyTorch reference implementation for PromptFlashAttention."""
    scores = torch.matmul(query.float(), key.float().transpose(-2, -1))
    scores_scaled = scores * scale
    attn_weights = torch.softmax(scores_scaled, dim=-1)
    attention_out = torch.matmul(attn_weights, value.float())
    return attention_out.to(query.dtype)


@pypto.frontend.jit
def prompt_flash_attention_kernel(
    query: pypto.Tensor((BATCH_SIZE, NUM_HEADS, SEQ_LEN_Q, HEAD_DIM), pypto.DT_BF16),
    key: pypto.Tensor((BATCH_SIZE, NUM_HEADS, SEQ_LEN_KV, HEAD_DIM), pypto.DT_BF16),
    value: pypto.Tensor((BATCH_SIZE, NUM_HEADS, SEQ_LEN_KV, HEAD_DIM), pypto.DT_BF16),
    attention_out: pypto.Tensor((BATCH_SIZE, NUM_HEADS, SEQ_LEN_Q, HEAD_DIM), pypto.DT_BF16),
    scale: float,
):
    """
    PromptFlashAttention kernel for full inference.
    
    Steps:
    1. scores = Q @ K^T
    2. scores_scaled = scores * scale  
    3. attn_weights = softmax(scores_scaled)
    4. output = attn_weights @ V
    """
    pypto.set_cube_tile_shapes([64, 64], [64, 64], [64, 64])
    pypto.set_vec_tile_shapes(1, 8, 16, HEAD_DIM)
    
    k_t = pypto.transpose(key, 2, 3)
    scores = pypto.matmul(query, k_t, out_dtype=pypto.DT_BF16)
    scores_scaled = pypto.mul(scores, scale)
    
    attn_weights = pypto.softmax(scores_scaled, dim=-1)
    
    output = pypto.matmul(attn_weights, value, out_dtype=pypto.DT_BF16)
    attention_out.move(output)


def test_prompt_flash_attention_basic(device_id=None, run_mode: str = "npu"):
    """Test PromptFlashAttention basic scenario: B=1, N=8, Sq=128, Skv=128, D=64."""
    print("=" * 60)
    print("Test: PromptFlashAttention - Basic (B=1, N=8, Sq=128, Skv=128, D=64)")
    print("=" * 60)
    
    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'
    
    scale = 1.0 / (HEAD_DIM ** 0.5)
    
    q_torch = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN_Q, HEAD_DIM, dtype=torch.bfloat16, device=device)
    k_torch = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN_KV, HEAD_DIM, dtype=torch.bfloat16, device=device)
    v_torch = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN_KV, HEAD_DIM, dtype=torch.bfloat16, device=device)
    
    out_torch = torch.empty(BATCH_SIZE, NUM_HEADS, SEQ_LEN_Q, HEAD_DIM, dtype=torch.bfloat16, device=device)
    
    prompt_flash_attention_kernel(q_torch, k_torch, v_torch, out_torch, scale)
    
    golden_out = prompt_flash_attention_golden(q_torch, k_torch, v_torch, scale)
    
    print(f"Input shape: Q{q_torch.shape}, K{k_torch.shape}, V{v_torch.shape}")
    print(f"Output shape: {out_torch.shape}")
    
    if run_mode == "npu":
        out_diff = (out_torch - golden_out).abs().max().item()
        print(f"attention_out max diff: {out_diff:.6f}")
        
        if out_diff < 0.02:
            print("✓ Basic test passed (max diff < 0.02)")
        else:
            print(f"✗ Basic test failed (max diff >= 0.02)")
    
    print()


def test_prompt_flash_attention_large(device_id=None, run_mode: str = "npu"):
    """Test PromptFlashAttention large scenario: B=2, N=16, Sq=256, Skv=256, D=128."""
    global BATCH_SIZE, NUM_HEADS, SEQ_LEN_Q, SEQ_LEN_KV, HEAD_DIM
    
    print("=" * 60)
    print("Test: PromptFlashAttention - Large (B=2, N=16, Sq=256, Skv=256, D=128)")
    print("=" * 60)
    
    BATCH_SIZE = 2
    NUM_HEADS = 16
    SEQ_LEN_Q = 256
    SEQ_LEN_KV = 256
    HEAD_DIM = 128
    
    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'
    scale = 1.0 / (HEAD_DIM ** 0.5)
    
    q_torch = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN_Q, HEAD_DIM, dtype=torch.bfloat16, device=device)
    k_torch = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN_KV, HEAD_DIM, dtype=torch.bfloat16, device=device)
    v_torch = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN_KV, HEAD_DIM, dtype=torch.bfloat16, device=device)
    out_torch = torch.empty(BATCH_SIZE, NUM_HEADS, SEQ_LEN_Q, HEAD_DIM, dtype=torch.bfloat16, device=device)
    
    prompt_flash_attention_kernel(q_torch, k_torch, v_torch, out_torch, scale)
    
    golden_out = prompt_flash_attention_golden(q_torch, k_torch, v_torch, scale)
    
    print(f"Input shape: Q{q_torch.shape}, K{k_torch.shape}, V{v_torch.shape}")
    
    if run_mode == "npu":
        out_diff = (out_torch - golden_out).abs().max().item()
        print(f"attention_out max diff: {out_diff:.6f}")
        if out_diff < 0.02:
            print("✓ Large test passed (max diff < 0.02)")
        else:
            print(f"✗ Large test failed (max diff >= 0.02)")
    print()


def main():
    parser = argparse.ArgumentParser(description="PromptFlashAttention PyPTO Test")
    parser.add_argument('--run_mode', type=str, default='npu', choices=["npu"], help='Run mode')
    parser.add_argument('--test', type=str, default='all', choices=["all", "basic", "large"], help='Test type')
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("PromptFlashAttention PyPTO Test")
    print("=" * 60 + "\n")
    
    device_id = None
    if args.run_mode == "npu":
        device_id = get_device_id()
        if device_id is None:
            return
        import torch_npu
        torch.npu.set_device(device_id)
        print("Running on NPU...\n")
    
    if args.test in ["all", "basic"]:
        test_prompt_flash_attention_basic(device_id, args.run_mode)
    
    if args.test in ["all", "large"]:
        test_prompt_flash_attention_large(device_id, args.run_mode)
    
    print("=" * 60)
    print("PromptFlashAttention tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()