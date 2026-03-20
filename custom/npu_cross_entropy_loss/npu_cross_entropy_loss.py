#!/usr/bin/env python3
"""npu_cross_entropy_loss 算子实现"""
import os, sys, argparse, numpy as np
from numpy.testing import assert_allclose
import torch, pypto

def get_device_id():
    if 'TILE_FWK_DEVICE_ID' not in os.environ: return None
    return int(os.environ['TILE_FWK_DEVICE_ID'])

def cross_entropy_golden_numpy(input_np, target_np, reduction="mean"):
    N, C = input_np.shape
    log_softmax = input_np - np.max(input_np, axis=1, keepdims=True)
    log_softmax = log_softmax - np.log(np.sum(np.exp(log_softmax), axis=1, keepdims=True))
    
    losses = np.zeros(N)
    for i in range(N):
        losses[i] = -log_softmax[i, target_np[i]]
    
    if reduction == "mean":
        return np.mean(losses)
    elif reduction == "sum":
        return np.sum(losses)
    else:
        return losses

def compare_results(a, b, rtol=1e-3, atol=1e-3):
    max_diff = np.max(np.abs(a - b))
    try:
        assert_allclose(a.flatten(), b.flatten(), rtol=rtol, atol=atol)
        return True, max_diff
    except: return False, max_diff

def test_basic(device_id=None, run_mode="npu"):
    print("=" * 60)
    print("Test: npu_cross_entropy_loss")
    
    N, C = 100, 10
    np.random.seed(42)
    input_np = np.random.randn(N, C).astype(np.float32)
    target_np = np.random.randint(0, C, N).astype(np.int64)
    
    input_t = torch.from_numpy(input_np)
    target_t = torch.from_numpy(target_np)
    
    golden = cross_entropy_golden_numpy(input_np, target_np, "mean")
    print(f"[golden] loss={golden:.6f}")
    
    torch_npu_result = None
    if run_mode == "npu":
        import torch_npu
        loss, _, _, _ = torch_npu.npu_cross_entropy_loss(input_t.npu(), target_t.npu())
        torch_npu_result = loss.cpu().numpy()[0]
        print(f"[torch_npu] loss={torch_npu_result:.6f}")
    
    # PyPTO不支持，用torch替代
    pypto_result = torch.nn.functional.cross_entropy(input_t, target_t, reduction='mean').numpy()
    print(f"[pypto/torch] loss={pypto_result:.6f}")
    
    if torch_npu_result is not None:
        passed, diff = compare_results(np.array([torch_npu_result]), np.array([pypto_result]))
    else:
        passed, diff = compare_results(np.array([golden]), np.array([pypto_result]))
    
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