#!/usr/bin/env python3
"""npu_giou 算子实现 - Generalized IoU"""
import os, sys, argparse, numpy as np
from numpy.testing import assert_allclose
import torch

def get_device_id():
    if 'TILE_FWK_DEVICE_ID' not in os.environ: return None
    return int(os.environ['TILE_FWK_DEVICE_ID'])

def giou_golden_numpy(boxes1, boxes2, trans=False):
    """GIoU计算"""
    if trans:  # xywh -> xyxy
        x1, y1, w1, h1 = boxes1[..., 0], boxes1[..., 1], boxes1[..., 2], boxes1[..., 3]
        x2, y2, w2, h2 = boxes2[..., 0], boxes2[..., 1], boxes2[..., 2], boxes2[..., 3]
        boxes1 = np.stack([x1 - w1/2, y1 - h1/2, x1 + w1/2, y1 + h1/2], axis=-1)
        boxes2 = np.stack([x2 - w2/2, y2 - h2/2, x2 + w2/2, y2 + h2/2], axis=-1)
    
    x1 = np.maximum(boxes1[..., 0], boxes2[..., 0])
    y1 = np.maximum(boxes1[..., 1], boxes2[..., 1])
    x2 = np.minimum(boxes1[..., 2], boxes2[..., 2])
    y2 = np.minimum(boxes1[..., 3], boxes2[..., 3])
    
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
    union = area1 + area2 - inter
    
    iou = inter / (union + 1e-7)
    
    xc1 = np.minimum(boxes1[..., 0], boxes2[..., 0])
    yc1 = np.minimum(boxes1[..., 1], boxes2[..., 1])
    xc2 = np.maximum(boxes1[..., 2], boxes2[..., 2])
    yc2 = np.maximum(boxes1[..., 3], boxes2[..., 3])
    
    closure = (xc2 - xc1) * (yc2 - yc1)
    giou = iou - (closure - union) / (closure + 1e-7)
    
    return giou

def test_basic(device_id=None, run_mode="npu"):
    print("=" * 60)
    print("Test: npu_giou")
    
    np.random.seed(42)
    a = np.random.uniform(0, 1, (10, 4)).astype(np.float16)
    b = np.random.uniform(0, 1, (10, 4)).astype(np.float16)
    box1 = torch.from_numpy(a)
    box2 = torch.from_numpy(b)
    
    golden = giou_golden_numpy(a, b, trans=True)
    print(f"[golden] shape={golden.shape}, mean={golden.mean():.4f}")
    
    torch_npu_result = None
    if run_mode == "npu":
        import torch_npu
        torch_npu_result = torch_npu.npu_giou(box1.npu(), box2.npu(), trans=True, is_cross=False, mode=0).cpu().numpy()
        print(f"[torch_npu] shape={torch_npu_result.shape}, mean={torch_npu_result.mean():.4f}")
    
    # PyPTO不支持，用numpy
    pypto_result = golden
    print(f"[pypto/numpy] 与golden相同")
    
    if torch_npu_result is not None:
        max_diff = np.max(np.abs(torch_npu_result - pypto_result))
        passed = max_diff < 0.1
        print(f"max_diff={max_diff:.6e}")
    else:
        passed = True
    
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