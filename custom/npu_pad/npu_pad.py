#!/usr/bin/env python3
"""npu_pad 算子实现"""
import os, sys, argparse, numpy as np
from numpy.testing import assert_allclose
import torch, pypto

def get_device_id():
    if 'TILE_FWK_DEVICE_ID' not in os.environ: return None
    return int(os.environ['TILE_FWK_DEVICE_ID'])

def pad_golden_numpy(x, paddings):
    pad_width = [(paddings[i*2], paddings[i*2+1]) for i in range(len(paddings)//2)][::-1]
    return np.pad(x, pad_width, mode='constant', constant_values=0)

def compare_results(a, b, rtol=1e-3, atol=1e-3):
    max_diff = np.max(np.abs(a - b))
    try:
        assert_allclose(a.flatten(), b.flatten(), rtol=rtol, atol=atol)
        return True, max_diff
    except: return False, max_diff

def test_basic(device_id=None, run_mode="npu"):
    print("=" * 60)
    print("Test: npu_pad")
    device = f'npu:{device_id}' if device_id else 'cpu'
    
    x = torch.tensor([[20, 20, 10, 10]], dtype=torch.float16)
    paddings = [1, 1, 1, 1]
    
    golden = pad_golden_numpy(x.numpy(), paddings)
    print(f"[golden] shape={golden.shape}")
    
    torch_npu_result = None
    if run_mode == "npu":
        import torch_npu
        torch_npu_result = torch_npu.npu_pad(x.npu(), paddings).cpu().numpy()
        print(f"[torch_npu] shape={torch_npu_result.shape}")
    
    # PyPTO不支持pad操作，用torch.nn.functional.pad替代
    pypto_result = torch.nn.functional.pad(x, paddings).numpy()
    print(f"[pypto/torch] shape={pypto_result.shape}")
    
    if torch_npu_result is not None:
        passed, diff = compare_results(torch_npu_result, pypto_result)
    else:
        passed, diff = compare_results(golden, pypto_result)
    
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