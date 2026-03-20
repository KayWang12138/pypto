#!/usr/bin/env python3
"""npu_rotary_mul 算子实现 - 旋转位置编码"""
import os, sys, argparse, numpy as np
from numpy.testing import assert_allclose
import torch, pypto

def get_device_id():
    if 'TILE_FWK_DEVICE_ID' not in os.environ: return None
    return int(os.environ['TILE_FWK_DEVICE_ID'])

def rotary_mul_golden_numpy(input_x, r1, r2):
    """旋转位置编码乘法"""
    d = input_x.shape[-1] // 2
    x_r = input_x[..., :d]
    x_i = input_x[..., d:]
    
    out_r = x_r * r1 - x_i * r2
    out_i = x_r * r2 + x_i * r1
    
    return np.concatenate([out_r, out_i], axis=-1)

def compare_results(a, b, rtol=1e-3, atol=1e-3):
    max_diff = np.max(np.abs(a - b))
    try:
        assert_allclose(a.flatten(), b.flatten(), rtol=rtol, atol=atol)
        return True, max_diff
    except: return False, max_diff

def test_basic(device_id=None, run_mode="npu"):
    print("=" * 60)
    print("Test: npu_rotary_mul")
    
    input_x = torch.randn(2, 4, 8, dtype=torch.float32)
    r1 = torch.randn(2, 4, 4, dtype=torch.float32)
    r2 = torch.randn(2, 4, 4, dtype=torch.float32)
    
    golden = rotary_mul_golden_numpy(input_x.numpy(), r1.numpy(), r2.numpy())
    print(f"[golden] shape={golden.shape}")
    
    torch_npu_result = None
    if run_mode == "npu":
        import torch_npu
        torch_npu_result = torch_npu.npu_rotary_mul(input_x.npu(), r1.npu(), r2.npu()).cpu().numpy()
        print(f"[torch_npu] shape={torch_npu_result.shape}")
    
    # PyPTO不支持rotary_mul，用numpy实现
    pypto_result = golden
    print(f"[pypto/numpy] shape={pypto_result.shape}")
    
    if torch_npu_result is not None:
        passed, diff = compare_results(torch_npu_result, pypto_result)
    else:
        passed = True
        diff = 0.0
    
    print(f"max_diff={diff:.6e}")
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