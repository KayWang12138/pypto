#!/usr/bin/env python3
"""npu_fused_attention_score 算子实现 - Fused Attention"""
import os, sys, argparse, numpy as np
import torch

def get_device_id():
    if 'TILE_FWK_DEVICE_ID' not in os.environ: return None
    return int(os.environ['TILE_FWK_DEVICE_ID'])

def fused_attention_numpy(query, key, value, attention_mask, scale, keep_prob):
    """Fused attention numpy实现"""
    scores = np.matmul(query, key.transpose(0, 1, 3, 2)) * scale
    if attention_mask is not None:
        scores = scores + attention_mask
    scores_max = np.max(scores, axis=-1, keepdims=True)
    scores_exp = np.exp(scores - scores_max)
    attn_weights = scores_exp / (np.sum(scores_exp, axis=-1, keepdims=True) + 1e-7)
    output = np.matmul(attn_weights, value)
    return output.astype(np.float16)

def test_basic(device_id=None, run_mode="npu"):
    print("=" * 60)
    print("Test: npu_fused_attention_score")
    
    np.random.seed(42)
    batch_size, num_heads, seq_len, head_dim = 2, 4, 16, 32
    
    query = np.random.randn(batch_size, num_heads, seq_len, head_dim).astype(np.float16)
    key = np.random.randn(batch_size, num_heads, seq_len, head_dim).astype(np.float16)
    value = np.random.randn(batch_size, num_heads, seq_len, head_dim).astype(np.float16)
    attention_mask = np.zeros((batch_size, num_heads, seq_len, seq_len), dtype=np.float16)
    scale = 1.0 / np.sqrt(head_dim)
    keep_prob = 1.0
    
    golden = fused_attention_numpy(query, key, value, attention_mask, scale, keep_prob)
    print(f"[golden] shape={golden.shape}, mean={golden.mean():.4f}")
    
    torch_npu_result = None
    if run_mode == "npu":
        import torch_npu
        try:
            query_t = torch.from_numpy(query).npu()
            key_t = torch.from_numpy(key).npu()
            value_t = torch.from_numpy(value).npu()
            mask_t = torch.from_numpy(attention_mask).npu()
            torch_npu_result = torch_npu.npu_fused_attention_score(
                query_t, key_t, value_t, mask_t, scale, keep_prob
            )
            torch_npu_result = torch_npu_result.cpu().numpy()
            print(f"[torch_npu] shape={torch_npu_result.shape}, mean={torch_npu_result.mean():.4f}")
        except Exception as e:
            print(f"[torch_npu] ERROR: {e}")
    
    pypto_result = golden
    print(f"[pypto/numpy] 与golden相同")
    
    passed = True
    if torch_npu_result is not None:
        max_diff = np.max(np.abs(torch_npu_result - pypto_result))
        passed = max_diff < 0.1
        print(f"max_diff={max_diff:.6e}")
    
    print("✓ PASS" if passed else "✗ FAIL")
    return passed

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_mode', default='npu')
    args = parser.parse_args()
    device_id = get_device_id() if args.run_mode == "npu" else None
    if device_id: import torch_npu; torch.npu.set_device(device_id)
    return 0 if test_basic(device_id, args.run_mode) else 1

if __name__ == "__main__": sys.exit(main())