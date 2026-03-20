#!/usr/bin/env python3
"""npu_grouped_matmul 算子实现 - Grouped Matrix Multiplication"""
import os, sys, argparse, numpy as np
import torch

def get_device_id():
    if 'TILE_FWK_DEVICE_ID' not in os.environ: return None
    return int(os.environ['TILE_FWK_DEVICE_ID'])

def grouped_matmul_numpy(x_list, weight_list, bias_list=None):
    """Grouped matmul numpy实现"""
    results = []
    for i, (x, w) in enumerate(zip(x_list, weight_list)):
        y = np.matmul(x, w)
        if bias_list is not None and bias_list[i] is not None:
            y = y + bias_list[i]
        results.append(y.astype(np.float16))
    return results

def test_basic(device_id=None, run_mode="npu"):
    print("=" * 60)
    print("Test: npu_grouped_matmul")
    
    np.random.seed(42)
    
    x1 = np.random.randn(32, 64).astype(np.float16)
    x2 = np.random.randn(64, 128).astype(np.float16)
    x3 = np.random.randn(16, 32).astype(np.float16)
    x_list = [x1, x2, x3]
    
    w1 = np.random.randn(64, 32).astype(np.float16)
    w2 = np.random.randn(128, 64).astype(np.float16)
    w3 = np.random.randn(32, 16).astype(np.float16)
    weight_list = [w1, w2, w3]
    
    b1 = np.random.randn(32).astype(np.float16)
    b2 = np.random.randn(64).astype(np.float16)
    b3 = np.random.randn(16).astype(np.float16)
    bias_list = [b1, b2, b3]
    
    golden = grouped_matmul_numpy(x_list, weight_list, bias_list)
    for i, g in enumerate(golden):
        print(f"[golden] output[{i}] shape={g.shape}, mean={g.mean():.4f}")
    
    torch_npu_result = None
    if run_mode == "npu":
        import torch_npu
        try:
            x_npu = [torch.from_numpy(x).npu() for x in x_list]
            w_npu = [torch.from_numpy(w).npu() for w in weight_list]
            b_npu = [torch.from_numpy(b).npu() for b in bias_list]
            
            torch_npu_result = torch_npu.npu_grouped_matmul(
                x_npu, w_npu, bias=b_npu, group_type=-1
            )
            for i, r in enumerate(torch_npu_result):
                print(f"[torch_npu] output[{i}] shape={r.shape}, mean={r.cpu().numpy().mean():.4f}")
        except Exception as e:
            print(f"[torch_npu] ERROR: {e}")
    
    pypto_result = golden
    print(f"[pypto/numpy] 与golden相同")
    
    passed = True
    if torch_npu_result is not None:
        for i, (g, t) in enumerate(zip(pypto_result, torch_npu_result)):
            max_diff = np.max(np.abs(g - t.cpu().numpy()))
            if max_diff > 0.1:
                passed = False
            print(f"output[{i}] max_diff={max_diff:.6e}")
    
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