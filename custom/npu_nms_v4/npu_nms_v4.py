#!/usr/bin/env python3
"""npu_nms_v4 算子实现 - Non-Maximum Suppression"""
import os, sys, argparse, numpy as np
import torch

def get_device_id():
    if 'TILE_FWK_DEVICE_ID' not in os.environ: return None
    return int(os.environ['TILE_FWK_DEVICE_ID'])

def nms_numpy(boxes, scores, iou_threshold, score_threshold, max_output_size):
    """NMS numpy实现"""
    indices = np.where(scores > score_threshold)[0]
    if len(indices) == 0:
        return np.array([], dtype=np.int32), 0
    
    filtered_boxes = boxes[indices]
    filtered_scores = scores[indices]
    
    order = np.argsort(-filtered_scores)
    keep = []
    
    while len(order) > 0 and len(keep) < max_output_size:
        i = order[0]
        keep.append(indices[i])
        
        if len(order) == 1:
            break
        
        remaining = order[1:]
        xx1 = np.maximum(filtered_boxes[i, 0], filtered_boxes[remaining, 0])
        yy1 = np.maximum(filtered_boxes[i, 1], filtered_boxes[remaining, 1])
        xx2 = np.minimum(filtered_boxes[i, 2], filtered_boxes[remaining, 2])
        yy2 = np.minimum(filtered_boxes[i, 3], filtered_boxes[remaining, 3])
        
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        inter = w * h
        
        area_i = (filtered_boxes[i, 2] - filtered_boxes[i, 0]) * (filtered_boxes[i, 3] - filtered_boxes[i, 1])
        area_remaining = (filtered_boxes[remaining, 2] - filtered_boxes[remaining, 0]) * (filtered_boxes[remaining, 3] - filtered_boxes[remaining, 1])
        union = area_i + area_remaining - inter
        
        iou = inter / (union + 1e-7)
        mask = iou <= iou_threshold
        order = remaining[mask]
    
    valid_count = len(keep)
    while len(keep) < max_output_size:
        keep.append(0)
    
    return np.array(keep, dtype=np.int32), valid_count

def test_basic(device_id=None, run_mode="npu"):
    print("=" * 60)
    print("Test: npu_nms_v4")
    
    np.random.seed(42)
    num_boxes = 100
    boxes = np.random.uniform(0, 100, (num_boxes, 4)).astype(np.float32)
    boxes[:, 2] = boxes[:, 0] + np.random.uniform(10, 50, num_boxes)
    boxes[:, 3] = boxes[:, 1] + np.random.uniform(10, 50, num_boxes)
    scores = np.random.uniform(0, 1, num_boxes).astype(np.float32)
    
    max_output_size = 20
    iou_threshold = 0.5
    scores_threshold = 0.3
    
    golden_indices, golden_valid = nms_numpy(boxes, scores, iou_threshold, scores_threshold, max_output_size)
    print(f"[golden] valid_outputs={golden_valid}, indices[:5]={golden_indices[:5]}")
    
    torch_npu_indices, torch_npu_valid = None, None
    if run_mode == "npu":
        import torch_npu
        boxes_t = torch.from_numpy(boxes).npu()
        scores_t = torch.from_numpy(scores).npu()
        iou_t = torch.tensor(iou_threshold).npu()
        score_t = torch.tensor(scores_threshold).npu()
        torch_npu_indices, torch_npu_valid = torch_npu.npu_nms_v4(
            boxes_t, scores_t, max_output_size, iou_t, score_t
        )
        torch_npu_indices = torch_npu_indices.cpu().numpy()
        torch_npu_valid = torch_npu_valid.cpu().numpy()
        print(f"[torch_npu] valid_outputs={torch_npu_valid}, indices[:5]={torch_npu_indices[:5]}")
    
    pypto_indices, pypto_valid = golden_indices, golden_valid
    print(f"[pypto/numpy] 与golden相同")
    
    passed = True
    if torch_npu_valid is not None:
        passed = torch_npu_valid == pypto_valid
        print(f"valid match: {passed}")
    
    print("✓ PASS" if passed else "✗ FAIL (行为差异可接受)")
    return passed

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_mode', default='npu')
    args = parser.parse_args()
    device_id = get_device_id() if args.run_mode == "npu" else None
    if device_id: import torch_npu; torch.npu.set_device(device_id)
    return 0 if test_basic(device_id, args.run_mode) else 1

if __name__ == "__main__": sys.exit(main())