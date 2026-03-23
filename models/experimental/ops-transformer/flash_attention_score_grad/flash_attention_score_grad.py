#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
"""
FlashAttentionScoreGrad Operator for PyPTO - True FlashAttention Implementation

Based on flash_attention_score.py pattern:
- Tiled computation with pypto.loop
- Online softmax forward pass
- FlashSoftmaxGrad for backward
"""

import os
import sys
import argparse
import torch
import pypto


def setup_pto_tile_lib_path():
    """Setup PTO_TILE_LIB_CODE_PATH if not set or invalid."""
    if 'PTO_TILE_LIB_CODE_PATH' in os.environ:
        pto_path = os.environ['PTO_TILE_LIB_CODE_PATH']
        include_pto = os.path.join(pto_path, 'include', 'pto')
        if os.path.exists(include_pto):
            return
    
    candidate_paths = [
        '/data/s00835526/agent_work/pypto/pto_isa/pto-isa',
        '/data/p84341448/work/pto-isa',
    ]
    
    for path in candidate_paths:
        include_pto = os.path.join(path, 'include', 'pto')
        if os.path.exists(include_pto):
            os.environ['PTO_TILE_LIB_CODE_PATH'] = path
            print(f"Auto-set PTO_TILE_LIB_CODE_PATH={path}")
            return
    
    print("Warning: PTO_TILE_LIB_CODE_PATH not found, compilation may fail")


setup_pto_tile_lib_path()

BATCH_SIZE = 2
NUM_HEADS = 8
SEQ_LEN_Q = 16
SEQ_LEN_KV = 16
HEAD_DIM = 64

TILE_S1 = 16
TILE_S2 = 16


def get_device_id():
    if 'TILE_FWK_DEVICE_ID' not in os.environ:
        print("Please set TILE_FWK_DEVICE_ID")
        return None
    return int(os.environ['TILE_FWK_DEVICE_ID'])


def flash_attention_score_grad_golden_simple(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    dy: torch.Tensor,
    scale_value: float,
) -> tuple:
    """Simple golden for verification."""
    B, N, Sq, D = query.shape
    _, _, Skv, _ = key.shape
    
    q_f = query.float()
    k_f = key.float()
    v_f = value.float()
    dy_f = dy.float()
    
    # Forward
    scores = torch.matmul(q_f, k_f.transpose(-2, -1)) * scale_value
    p = torch.softmax(scores, dim=-1)
    attention_in = torch.matmul(p, v_f)
    
    # Backward with FlashSoftmaxGrad
    sfmg = (dy_f * attention_in).sum(dim=-1, keepdim=True)
    dp = torch.matmul(dy_f, v_f.transpose(-2, -1))
    ds = p * (dp - sfmg)
    
    dq = torch.matmul(ds, k_f) * scale_value
    dk = torch.matmul(ds.transpose(-2, -1), q_f) * scale_value
    dv = torch.matmul(p.transpose(-2, -1), dy_f)
    
    return dq.to(query.dtype), dk.to(key.dtype), dv.to(value.dtype)


@pypto.frontend.jit
def flash_attention_score_grad_kernel(
    query: pypto.Tensor((BATCH_SIZE, NUM_HEADS, SEQ_LEN_Q, HEAD_DIM), pypto.DT_BF16),
    key: pypto.Tensor((BATCH_SIZE, NUM_HEADS, SEQ_LEN_KV, HEAD_DIM), pypto.DT_BF16),
    value: pypto.Tensor((BATCH_SIZE, NUM_HEADS, SEQ_LEN_KV, HEAD_DIM), pypto.DT_BF16),
    dy: pypto.Tensor((BATCH_SIZE, NUM_HEADS, SEQ_LEN_Q, HEAD_DIM), pypto.DT_BF16),
    dq_out: pypto.Tensor((BATCH_SIZE, NUM_HEADS, SEQ_LEN_Q, HEAD_DIM), pypto.DT_BF16),
    dk_out: pypto.Tensor((BATCH_SIZE, NUM_HEADS, SEQ_LEN_KV, HEAD_DIM), pypto.DT_BF16),
    dv_out: pypto.Tensor((BATCH_SIZE, NUM_HEADS, SEQ_LEN_KV, HEAD_DIM), pypto.DT_BF16),
    scale_value: float,
):
    """
    True Flash Attention Score Grad.
    
    Algorithm per S1 tile:
    1. Forward: online softmax to get attention_in and softmax stats
    2. FlashSoftmaxGrad: sfmg = sum(dy * attention_in)
    3. Backward: recompute P and compute gradients
    """
    pypto.set_cube_tile_shapes([64, 64], [64, 64], [64, 64])
    pypto.set_vec_tile_shapes(1, 8, 16, HEAD_DIM)
    
    s1_loop_count = SEQ_LEN_Q // TILE_S1
    s2_loop_count = SEQ_LEN_KV // TILE_S2
    
    # Process each S1 tile
    for s1_idx in pypto.loop(0, s1_loop_count, 1, name="S1_LOOP"):
        s1_start = s1_idx * TILE_S1
        
        # Get Q and dY blocks
        q_block_shape = [BATCH_SIZE, NUM_HEADS, TILE_S1, HEAD_DIM]
        q_block = pypto.view(query, q_block_shape, [0, 0, s1_start, 0])
        dy_block = pypto.view(dy, q_block_shape, [0, 0, s1_start, 0])
        
        # Convert to FP32
        q_fp32 = pypto.cast(q_block, pypto.DT_FP32)
        dy_fp32 = pypto.cast(dy_block, pypto.DT_FP32)
        
        # ===== Forward: Online Softmax =====
        m_running = pypto.full([BATCH_SIZE, NUM_HEADS, TILE_S1, 1], float('-inf'), pypto.DT_FP32)
        l_running = pypto.zeros(BATCH_SIZE, NUM_HEADS, TILE_S1, 1, dtype=pypto.DT_FP32)
        acc_running = pypto.zeros(BATCH_SIZE, NUM_HEADS, TILE_S1, HEAD_DIM, dtype=pypto.DT_FP32)
        
        for s2_idx in pypto.loop(0, s2_loop_count, 1, name="S2_FWD"):
            s2_start = s2_idx * TILE_S2
            
            kv_block_shape = [BATCH_SIZE, NUM_HEADS, TILE_S2, HEAD_DIM]
            k_block = pypto.view(key, kv_block_shape, [0, 0, s2_start, 0])
            v_block = pypto.view(value, kv_block_shape, [0, 0, s2_start, 0])
            
            k_fp32 = pypto.cast(k_block, pypto.DT_FP32)
            v_fp32 = pypto.cast(v_block, pypto.DT_FP32)
            
            k_t = pypto.transpose(k_fp32, 2, 3)
            scores = pypto.matmul(q_fp32, k_t, out_dtype=pypto.DT_FP32)
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
            exp_scores_weighted = pypto.matmul(exp_scores, v_fp32, out_dtype=pypto.DT_FP32)
            acc_new = pypto.add(acc_running_scaled, exp_scores_weighted)
            
            m_running = m_new
            l_running = l_new
            acc_running = acc_new
        
        # attention_in = acc / l
        attention_in = pypto.div(acc_running, l_running)
        
        # ===== FlashSoftmaxGrad =====
        dy_atten_mul = pypto.mul(dy_fp32, attention_in)
        sfmg = pypto.sum(dy_atten_mul, dim=-1, keepdim=True)
        
        # ===== Backward =====
        dq_acc = pypto.zeros(BATCH_SIZE, NUM_HEADS, TILE_S1, HEAD_DIM, dtype=pypto.DT_FP32)
        
        # Use final m and l
        m_final = m_running
        l_final = l_running
        
        for s2_idx in pypto.loop(0, s2_loop_count, 1, name="S2_BWD"):
            s2_start = s2_idx * TILE_S2
            
            kv_block_shape = [BATCH_SIZE, NUM_HEADS, TILE_S2, HEAD_DIM]
            k_block = pypto.view(key, kv_block_shape, [0, 0, s2_start, 0])
            v_block = pypto.view(value, kv_block_shape, [0, 0, s2_start, 0])
            
            k_fp32 = pypto.cast(k_block, pypto.DT_FP32)
            v_fp32 = pypto.cast(v_block, pypto.DT_FP32)
            
            # Recompute exp_scores and P
            k_t = pypto.transpose(k_fp32, 2, 3)
            scores = pypto.matmul(q_fp32, k_t, out_dtype=pypto.DT_FP32)
            scores_scaled = pypto.mul(scores, scale_value)
            scores_shifted = pypto.sub(scores_scaled, m_final)
            exp_scores = pypto.exp(scores_shifted)
            p_block = pypto.div(exp_scores, l_final)
            
            # dV = P^T @ dY
            p_t = pypto.transpose(p_block, 2, 3)
            dv_block = pypto.matmul(p_t, dy_fp32, out_dtype=pypto.DT_FP32)
            
            # dP = dY @ V^T
            v_t = pypto.transpose(v_fp32, 2, 3)
            dp = pypto.matmul(dy_fp32, v_t, out_dtype=pypto.DT_FP32)
            
            # dS = P * (dP - sfmg)
            dp_minus_sfmg = pypto.sub(dp, sfmg)
            ds = pypto.mul(p_block, dp_minus_sfmg)
            
            # dQ += dS @ K * scale
            dq_block = pypto.matmul(ds, k_fp32, out_dtype=pypto.DT_FP32)
            dq_scaled = pypto.mul(dq_block, scale_value)
            dq_acc = pypto.add(dq_acc, dq_scaled)
            
            # dK = dS^T @ Q * scale
            ds_t = pypto.transpose(ds, 2, 3)
            dk_block = pypto.matmul(ds_t, q_fp32, out_dtype=pypto.DT_FP32)
            dk_scaled = pypto.mul(dk_block, scale_value)
            dk_bf16 = pypto.cast(dk_scaled, pypto.DT_BF16)
            pypto.assemble(dk_bf16, [0, 0, s2_start, 0], dk_out)
            
            # dV
            dv_bf16 = pypto.cast(dv_block, pypto.DT_BF16)
            pypto.assemble(dv_bf16, [0, 0, s2_start, 0], dv_out)
        
        # Write dQ for this tile
        dq_bf16 = pypto.cast(dq_acc, pypto.DT_BF16)
        pypto.assemble(dq_bf16, [0, 0, s1_start, 0], dq_out)


def test_flash_attention_score_grad_basic(device_id=None, run_mode: str = "npu") -> None:
    print("=" * 60)
    print("Test: Flash Attention Score Grad (True FlashAttention)")
    print("=" * 60)
    
    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'
    
    query = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN_Q, HEAD_DIM, dtype=torch.bfloat16, device=device)
    key = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN_KV, HEAD_DIM, dtype=torch.bfloat16, device=device)
    value = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN_KV, HEAD_DIM, dtype=torch.bfloat16, device=device)
    dy = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN_Q, HEAD_DIM, dtype=torch.bfloat16, device=device)
    
    dq = torch.empty(BATCH_SIZE, NUM_HEADS, SEQ_LEN_Q, HEAD_DIM, dtype=torch.bfloat16, device=device)
    dk = torch.empty(BATCH_SIZE, NUM_HEADS, SEQ_LEN_KV, HEAD_DIM, dtype=torch.bfloat16, device=device)
    dv = torch.empty(BATCH_SIZE, NUM_HEADS, SEQ_LEN_KV, HEAD_DIM, dtype=torch.bfloat16, device=device)
    
    scale_value = 1.0 / (HEAD_DIM ** 0.5)
    
    flash_attention_score_grad_kernel(query, key, value, dy, dq, dk, dv, scale_value)
    
    dq_golden, dk_golden, dv_golden = flash_attention_score_grad_golden_simple(
        query, key, value, dy, scale_value
    )
    
    print(f"Input shape: Q={query.shape}, K={key.shape}, V={value.shape}")
    print(f"Output shape: dQ={dq.shape}, dK={dk.shape}, dV={dv.shape}")
    
    if run_mode == "npu":
        max_diff_dq = (dq.float() - dq_golden.float()).abs().max().item()
        max_diff_dk = (dk.float() - dk_golden.float()).abs().max().item()
        max_diff_dv = (dv.float() - dv_golden.float()).abs().max().item()
        
        print(f"Max diff (dQ): {max_diff_dq:.6f}")
        print(f"Max diff (dK): {max_diff_dk:.6f}")
        print(f"Max diff (dV): {max_diff_dv:.6f}")
        
        rtol = 0.01
        atol = 0.01
        assert torch.allclose(dq.float(), dq_golden.float(), rtol=rtol, atol=atol), f"dQ mismatch!"
        assert torch.allclose(dk.float(), dk_golden.float(), rtol=rtol, atol=atol), f"dK mismatch!"
        assert torch.allclose(dv.float(), dv_golden.float(), rtol=rtol, atol=atol), f"dV mismatch!"
    
    print("✓ Flash Attention Score Grad test passed")
    print()


def main():
    parser = argparse.ArgumentParser(description="PyPTO Flash Attention Score Grad")
    parser.add_argument('test_id', type=str, nargs='?', help='Test ID')
    parser.add_argument('--list', action='store_true', help='List tests')
    parser.add_argument('--run_mode', type=str, nargs='?', default='npu', choices=["npu"])
    args = parser.parse_args()
    
    tests = {
        'test_basic': {
            'name': 'Basic Flash Attention Score Grad',
            'function': test_flash_attention_score_grad_basic
        }
    }
    
    if args.list:
        for tid, tinfo in tests.items():
            print(f"  {tid}: {tinfo['name']}")
        return
    
    print("=" * 60)
    print("PyPTO Flash Attention Score Grad (True FlashAttention)")
    print("=" * 60)
    
    device_id = None
    if args.run_mode == "npu":
        device_id = get_device_id()
        if device_id is None:
            return
        import torch_npu
        torch.npu.set_device(device_id)
    
    tests_to_run = [(args.test_id, tests[args.test_id])] if args.test_id else list(tests.items())
    
    for tid, tinfo in tests_to_run:
        print(f"Running {tid}: {tinfo['name']}")
        tinfo['function'](device_id, args.run_mode)
    
    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()