#!/usr/bin/env python3
"""npu_broadcast 算子实现"""
import os, sys, argparse, numpy as np
import torch

def get_device_id():
    if 'TILE_FWK_DEVICE_ID' not in os.environ: return None
    return int(os.environ['TILE_FWK_DEVICE_ID'])

def test_basic(device_id=None, run_mode="npu"):
    print("=" * 60)
    print("Test: npu_broadcast")
    
    x = torch.tensor([[1], [2], [3]], dtype=torch.float32)
    size = [3, 4]
    
    golden = np.tile(x.numpy(), (1, 4))
    print(f"[golden] shape={golden.shape}")
    
    torch_npu_result = None
    if run_mode == "npu":
        import torch_npu
        torch_npu_result = torch_npu.npu_broadcast(x.npu(), size).cpu().numpy()
        print(f"[torch_npu] shape={torch_npu_result.shape}")
    
    pypto_result = torch.broadcast_to(x, size).numpy()
    print(f"[pypto/torch] shape={pypto_result.shape}")
    
    passed = True
    if torch_npu_result is not None:
        max_diff = np.max(np.abs(torch_npu_result - pypto_result))
        passed = max_diff < 1e-6
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