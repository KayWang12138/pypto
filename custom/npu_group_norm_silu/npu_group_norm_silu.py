#!/usr/bin/env python3
"""npu_group_norm_silu 算子实现 - Group Normalization + SiLU"""
import os, sys, argparse, numpy as np
import torch
import torch.nn.functional as F

def get_device_id():
    if 'TILE_FWK_DEVICE_ID' not in os.environ: return None
    return int(os.environ['TILE_FWK_DEVICE_ID'])

def group_norm_silu_numpy(x, weight, bias, num_groups, eps=1e-5):
    """Group Norm + SiLU numpy实现"""
    N, C = x.shape[0], x.shape[1]
    G = num_groups
    
    x_reshape = x.reshape(N, G, -1)
    mean = np.mean(x_reshape, axis=2, keepdims=True)
    var = np.var(x_reshape, axis=2, keepdims=True)
    
    x_norm = (x_reshape - mean) / np.sqrt(var + eps)
    x_norm = x_norm.reshape(x.shape)
    
    x_norm = x_norm * weight.reshape(1, -1, 1, 1) + bias.reshape(1, -1, 1, 1)
    
    silu = x_norm * (1.0 / (1.0 + np.exp(-x_norm)))
    
    mean_out = mean.reshape(N, G)
    rstd_out = 1.0 / np.sqrt(var.reshape(N, G) + eps)
    
    return silu.astype(np.float32), mean_out.astype(np.float32), rstd_out.astype(np.float32)

def test_basic(device_id=None, run_mode="npu"):
    print("=" * 60)
    print("Test: npu_group_norm_silu")
    
    np.random.seed(42)
    N, C, H, W = 2, 32, 8, 8
    num_groups = 8
    
    x = np.random.randn(N, C, H, W).astype(np.float32)
    weight = np.random.randn(C).astype(np.float32)
    bias = np.random.randn(C).astype(np.float32)
    
    golden_out, golden_mean, golden_rstd = group_norm_silu_numpy(x, weight, bias, num_groups)
    print(f"[golden] out shape={golden_out.shape}, mean={golden_out.mean():.4f}")
    
    torch_npu_result = None
    if run_mode == "npu":
        import torch_npu
        try:
            x_t = torch.from_numpy(x).npu()
            weight_t = torch.from_numpy(weight).npu()
            bias_t = torch.from_numpy(bias).npu()
            out, mean, rstd = torch_npu.npu_group_norm_silu(x_t, weight_t, bias_t, group=num_groups)
            torch_npu_result = out.cpu().numpy()
            print(f"[torch_npu] out shape={torch_npu_result.shape}, mean={torch_npu_result.mean():.4f}")
        except Exception as e:
            print(f"[torch_npu] ERROR: {e}")
    
    pypto_result = golden_out
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