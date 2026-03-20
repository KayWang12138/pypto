#!/usr/bin/env python3
"""npu_bmmV2 算子实现: 批量矩阵乘法"""
import os, sys, argparse, numpy as np
from numpy.testing import assert_allclose
import torch, pypto

def get_device_id():
    if 'TILE_FWK_DEVICE_ID' not in os.environ: return None
    return int(os.environ['TILE_FWK_DEVICE_ID'])

def bmm_golden_numpy(a, b):
    return np.matmul(a, b)

@pypto.frontend.jit
def bmm_kernel(x: pypto.Tensor(), w: pypto.Tensor(), out: pypto.Tensor()):
    pypto.set_cube_tile_shapes([32, 32], [64, 64], [64, 64])
    out[:] = pypto.matmul(x, w, pypto.DT_FP32)

def compare_results(a, b, rtol=1e-3, atol=1e-3):
    max_diff = np.max(np.abs(a - b))
    try:
        assert_allclose(a.flatten(), b.flatten(), rtol=rtol, atol=atol)
        return True, max_diff
    except: return False, max_diff

def test_basic(device_id=None, run_mode="npu"):
    print("=" * 60)
    print("Test: npu_bmmV2")
    device = f'npu:{device_id}' if device_id else 'cpu'
    
    shape1, shape2 = (10, 3, 4), (10, 4, 5)
    np.random.seed(42)
    a = np.random.randn(*shape1).astype(np.float32)
    b = np.random.randn(*shape2).astype(np.float32)
    a_t = torch.from_numpy(a)
    b_t = torch.from_numpy(b)
    
    golden = bmm_golden_numpy(a, b)
    print(f"[golden] shape={golden.shape}")
    
    torch_npu_result = None
    if run_mode == "npu":
        import torch_npu
        torch_npu_result = torch_npu.npu_bmmV2(a_t.npu(), b_t.npu(), []).cpu().numpy()
        print(f"[torch_npu] shape={torch_npu_result.shape}")
    
    # PyPTO matmul
    out_shape = (shape1[0], shape1[1], shape2[2])
    pypto_out = torch.empty(out_shape, dtype=torch.float32, device=a_t.npu().device if run_mode=="npu" else 'cpu')
    bmm_kernel(a_t.npu() if run_mode=="npu" else a_t, b_t.npu() if run_mode=="npu" else b_t, pypto_out)
    pypto_result = pypto_out.cpu().numpy() if run_mode=="npu" else pypto_out.numpy()
    print(f"[pypto] shape={pypto_result.shape}")
    
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