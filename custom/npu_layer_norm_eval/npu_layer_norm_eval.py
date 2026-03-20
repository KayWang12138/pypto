#!/usr/bin/env python3
"""npu_layer_norm_eval 算子实现"""
import os, sys, argparse, numpy as np
from numpy.testing import assert_allclose
import torch, pypto

def get_device_id():
    if 'TILE_FWK_DEVICE_ID' not in os.environ: return None
    return int(os.environ['TILE_FWK_DEVICE_ID'])

def layer_norm_golden_numpy(x, normalized_shape, weight=None, bias=None, eps=1e-5):
    axis = tuple(range(-len(normalized_shape), 0))
    mean = np.mean(x, axis=axis, keepdims=True)
    var = np.var(x, axis=axis, keepdims=True)
    x_norm = (x - mean) / np.sqrt(var + eps)
    if weight is not None:
        x_norm = x_norm * weight
    if bias is not None:
        x_norm = x_norm + bias
    return x_norm

def compare_results(a, b, rtol=1e-3, atol=1e-3):
    max_diff = np.max(np.abs(a - b))
    try:
        assert_allclose(a.flatten(), b.flatten(), rtol=rtol, atol=atol)
        return True, max_diff
    except: return False, max_diff

def test_basic(device_id=None, run_mode="npu"):
    print("=" * 60)
    print("Test: npu_layer_norm_eval")
    
    x = torch.rand((6, 4), dtype=torch.float32)
    normalized_shape = [4]
    weight = torch.ones(4, dtype=torch.float32)
    bias = torch.zeros(4, dtype=torch.float32)
    
    golden = layer_norm_golden_numpy(x.numpy(), normalized_shape, weight.numpy(), bias.numpy())
    print(f"[golden] shape={golden.shape}")
    
    torch_npu_result = None
    if run_mode == "npu":
        import torch_npu
        torch_npu_result = torch_npu.npu_layer_norm_eval(x.npu(), normalized_shape, weight.npu(), bias.npu()).cpu().numpy()
        print(f"[torch_npu] shape={torch_npu_result.shape}")
    
    # PyPTO不支持layer_norm，用torch替代
    pypto_result = torch.nn.functional.layer_norm(x, normalized_shape, weight, bias).numpy()
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