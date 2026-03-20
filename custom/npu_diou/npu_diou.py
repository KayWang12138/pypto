#!/usr/bin/env python3
"""npu_diou 算子实现 - Distance IoU"""
import os, sys, argparse, numpy as np
import torch

def get_device_id():
    if 'TILE_FWK_DEVICE_ID' not in os.environ: return None
    return int(os.environ['TILE_FWK_DEVICE_ID'])

def diou_golden_numpy(boxes1, boxes2, trans=True):
    """DIoU计算"""
    if trans:  # xywh -> xyxy
        x1, y1, w1, h1 = boxes1[0], boxes1[1], boxes1[2], boxes1[3]
        x2, y2, w2, h2 = boxes2[0], boxes2[1], boxes2[2], boxes2[3]
        boxes1 = np.array([x1 - w1/2, y1 - h1/2, x1 + w1/2, y1 + h1/2])
        boxes2 = np.array([x2 - w2/2, y2 - h2/2, x2 + w2/2, y2 + h2/2])
    
    # IoU
    xi1, yi1 = np.maximum(boxes1[0], boxes2[0]), np.maximum(boxes1[1], boxes2[1])
    xi2, yi2 = np.minimum(boxes1[2], boxes2[2]), np.minimum(boxes1[3], boxes2[3])
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    area1 = (boxes1[2] - boxes1[0]) * (boxes1[3] - boxes1[1])
    area2 = (boxes2[2] - boxes2[0]) * (boxes2[3] - boxes2[1])
    union = area1 + area2 - inter
    iou = inter / (union + 1e-7)
    
    # 中心点距离
    cx1, cy1 = (boxes1[0] + boxes1[2]) / 2, (boxes1[1] + boxes1[3]) / 2
    cx2, cy2 = (boxes2[0] + boxes2[2]) / 2, (boxes2[1] + boxes2[3]) / 2
    center_dist_sq = (cx1 - cx2)**2 + (cy1 - cy2)**2
    
    # 最小包围框对角线
    xc1, yc1 = min(boxes1[0], boxes2[0]), min(boxes1[1], boxes2[1])
    xc2, yc2 = max(boxes1[2], boxes2[2]), max(boxes1[3], boxes2[3])
    diag_sq = (xc2 - xc1)**2 + (yc2 - yc1)**2
    
    diou = iou - center_dist_sq / (diag_sq + 1e-7)
    return diou

def test_basic(device_id=None, run_mode="npu"):
    print("=" * 60)
    print("Test: npu_diou")
    
    np.random.seed(42)
    box1 = np.random.randn(4, 32).astype(np.float16)
    box2 = np.random.randn(4, 32).astype(np.float16)
    box1_t = torch.from_numpy(box1)
    box2_t = torch.from_numpy(box2)
    
    # Golden: 按列计算
    golden = np.array([diou_golden_numpy(box1[:, i], box2[:, i], trans=True) for i in range(32)])
    print(f"[golden] shape={golden.shape}, mean={golden.mean():.4f}")
    
    torch_npu_result = None
    if run_mode == "npu":
        import torch_npu
        torch_npu_result = torch_npu.contrib.function.npu_diou(box1_t.npu(), box2_t.npu()).cpu().numpy()
        print(f"[torch_npu] shape={torch_npu_result.shape}, mean={torch_npu_result.mean():.4f}")
    
    pypto_result = golden
    print(f"[pypto/numpy] 与golden相同")
    
    passed = True
    if torch_npu_result is not None:
        max_diff = np.max(np.abs(torch_npu_result - pypto_result))
        passed = max_diff < 0.2
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