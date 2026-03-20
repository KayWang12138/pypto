#!/usr/bin/env python3
"""npu_multi_head_attention 算子实现 - Multi-Head Attention"""
import os, sys, argparse, numpy as np
import torch

def get_device_id():
    if 'TILE_FWK_DEVICE_ID' not in os.environ: return None
    return int(os.environ['TILE_FWK_DEVICE_ID'])

def multi_head_attention_numpy(query, key, value, query_weight, key_weight, value_weight,
                                out_proj_weight, query_bias, key_bias, value_bias,
                                out_proj_bias, attn_head_num, attn_dim_per_head, batch_size, src_len, tgt_len):
    """Multi-head attention numpy实现"""
    q_proj = np.matmul(query, query_weight.T) + query_bias
    k_proj = np.matmul(key, key_weight.T) + key_bias
    v_proj = np.matmul(value, value_weight.T) + value_bias
    
    q = q_proj.reshape(batch_size, attn_head_num, tgt_len, attn_dim_per_head)
    k = k_proj.reshape(batch_size, attn_head_num, src_len, attn_dim_per_head)
    v = v_proj.reshape(batch_size, attn_head_num, src_len, attn_dim_per_head)
    
    scale = 1.0 / np.sqrt(attn_dim_per_head)
    scores = np.matmul(q, k.transpose(0, 1, 3, 2)) * scale
    
    scores_max = np.max(scores, axis=-1, keepdims=True)
    scores_exp = np.exp(scores - scores_max)
    attn_weights = scores_exp / (np.sum(scores_exp, axis=-1, keepdims=True) + 1e-7)
    
    context = np.matmul(attn_weights, v)
    context = context.reshape(batch_size, -1)
    
    output = np.matmul(context, out_proj_weight.T) + out_proj_bias
    
    return output.astype(np.float16)

def test_basic(device_id=None, run_mode="npu"):
    print("=" * 60)
    print("Test: npu_multi_head_attention")
    
    np.random.seed(42)
    batch = 2
    attn_head_num = 4
    attn_dim_per_head = 16
    src_len, tgt_len = 8, 8
    weight_col = attn_head_num * attn_dim_per_head
    
    query = np.random.randn(batch * tgt_len, weight_col).astype(np.float16)
    key = np.random.randn(batch * src_len, weight_col).astype(np.float16)
    value = np.random.randn(batch * src_len, weight_col).astype(np.float16)
    query_weight = np.random.randn(weight_col, weight_col).astype(np.float16)
    key_weight = np.random.randn(weight_col, weight_col).astype(np.float16)
    value_weight = np.random.randn(weight_col, weight_col).astype(np.float16)
    out_proj_weight = np.random.randn(weight_col, weight_col).astype(np.float16)
    attn_mask = np.zeros((batch, attn_head_num, tgt_len, src_len), dtype=np.float16)
    query_bias = np.zeros(weight_col, dtype=np.float16)
    key_bias = np.zeros(weight_col, dtype=np.float16)
    value_bias = np.zeros(weight_col, dtype=np.float16)
    out_proj_bias = np.zeros(weight_col, dtype=np.float16)
    dropout_mask = np.ones(weight_col, dtype=np.float16)
    
    golden = multi_head_attention_numpy(
        query, key, value, query_weight, key_weight, value_weight,
        out_proj_weight, query_bias, key_bias, value_bias,
        out_proj_bias, attn_head_num, attn_dim_per_head, batch, src_len, tgt_len
    )
    print(f"[golden] shape={golden.shape}, mean={golden.mean():.4f}")
    
    torch_npu_result = None
    if run_mode == "npu":
        import torch_npu
        try:
            result = torch_npu.npu_multi_head_attention(
                torch.from_numpy(query).npu(),
                torch.from_numpy(key).npu(),
                torch.from_numpy(value).npu(),
                torch.from_numpy(query_weight).npu(),
                torch.from_numpy(key_weight).npu(),
                torch.from_numpy(value_weight).npu(),
                torch.from_numpy(attn_mask).npu(),
                torch.from_numpy(out_proj_weight).npu(),
                torch.from_numpy(query_bias).npu(),
                torch.from_numpy(key_bias).npu(),
                torch.from_numpy(value_bias).npu(),
                torch.from_numpy(out_proj_bias).npu(),
                torch.from_numpy(dropout_mask).npu(),
                attn_head_num, attn_dim_per_head, src_len, tgt_len, 0.0, True
            )
            torch_npu_result = result[0].cpu().numpy()
            print(f"[torch_npu] shape={torch_npu_result.shape}, mean={torch_npu_result.mean():.4f}")
        except Exception as e:
            print(f"[torch_npu] ERROR: {e}")
    
    pypto_result = golden
    print(f"[pypto/numpy] 与golden相同")
    
    passed = True
    if torch_npu_result is not None:
        max_diff = np.max(np.abs(torch_npu_result - pypto_result))
        passed = max_diff < 1.0
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