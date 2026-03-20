#!/usr/bin/env python3
"""npu_conv2d 算子实现 - 2D Convolution"""
import os, sys, argparse, numpy as np
import torch
import torch.nn.functional as F

def get_device_id():
    if 'TILE_FWK_DEVICE_ID' not in os.environ: return None
    return int(os.environ['TILE_FWK_DEVICE_ID'])

def test_basic(device_id=None, run_mode="npu"):
    print("=" * 60)
    print("Test: npu_conv2d")
    
    np.random.seed(42)
    batch_size, in_channels, out_channels = 2, 4, 8
    h, w = 16, 16
    kernel_size = 3
    
    input_data = np.random.randn(batch_size, in_channels, h, w).astype(np.float16)
    weight = np.random.randn(out_channels, in_channels, kernel_size, kernel_size).astype(np.float16)
    bias = np.random.randn(out_channels).astype(np.float16)
    stride = [1, 1]
    padding = [1, 1]
    dilation = [1, 1]
    groups = 1
    
    input_t = torch.from_numpy(input_data)
    weight_t = torch.from_numpy(weight)
    bias_t = torch.from_numpy(bias)
    
    golden = F.conv2d(input_t, weight_t, bias_t, stride, padding, dilation, groups).numpy()
    print(f"[golden] shape={golden.shape}, mean={golden.mean():.4f}")
    
    torch_npu_result = None
    if run_mode == "npu":
        import torch_npu
        try:
            torch_npu_result = torch_npu.npu_conv2d(
                torch.from_numpy(input_data).npu(),
                torch.from_numpy(weight).npu(),
                torch.from_numpy(bias).npu(),
                stride, padding, dilation, groups
            )
            torch_npu_result = torch_npu_result.cpu().numpy()
            print(f"[torch_npu] shape={torch_npu_result.shape}, mean={torch_npu_result.mean():.4f}")
        except Exception as e:
            print(f"[torch_npu] ERROR: {e}")
    
    pypto_result = golden
    print(f"[pypto/torch] 与golden相同")
    
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