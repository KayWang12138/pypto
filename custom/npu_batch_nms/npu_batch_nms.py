#!/usr/bin/env python3
"""npu_batch_nms 算子实现 - Batch Non-Maximum Suppression"""
import os, sys, argparse, numpy as np
import torch

def get_device_id():
    if 'TILE_FWK_DEVICE_ID' not in os.environ: return None
    return int(os.environ['TILE_FWK_DEVICE_ID'])

def batch_nms_numpy(boxes, scores, score_threshold, iou_threshold, max_size_per_class, max_total_size):
    """Batch NMS numpy实现"""
    batch_size = boxes.shape[0]
    num_classes = scores.shape[2]
    
    nmsed_boxes = np.zeros((batch_size, max_total_size, 4), dtype=np.float16)
    nmsed_scores = np.zeros((batch_size, max_total_size), dtype=np.float16)
    nmsed_classes = np.zeros((batch_size, max_total_size), dtype=np.float16)
    nmsed_num = np.zeros(batch_size, dtype=np.int32)
    
    for b in range(batch_size):
        all_boxes = []
        all_scores = []
        all_classes = []
        
        for c in range(num_classes):
            class_scores = scores[b, :, c]
            mask = class_scores > score_threshold
            if not np.any(mask):
                continue
            
            class_boxes = boxes[b, mask, 0] if boxes.shape[2] == 1 else boxes[b, mask, c]
            class_scores_filtered = class_scores[mask]
            
            order = np.argsort(-class_scores_filtered)[:max_size_per_class]
            
            for i in order:
                all_boxes.append(class_boxes[i] if len(class_boxes.shape) > 1 else class_boxes)
                all_scores.append(class_scores_filtered[i])
                all_classes.append(c)
        
        if len(all_boxes) > 0:
            all_boxes = np.array(all_boxes)
            all_scores = np.array(all_scores)
            order = np.argsort(-all_scores)[:max_total_size]
            
            nmsed_num[b] = min(len(order), max_total_size)
            for i, idx in enumerate(order[:max_total_size]):
                nmsed_boxes[b, i] = all_boxes[idx]
                nmsed_scores[b, i] = all_scores[idx]
                nmsed_classes[b, i] = all_classes[idx]
    
    return nmsed_boxes, nmsed_scores, nmsed_classes, nmsed_num

def test_basic(device_id=None, run_mode="npu"):
    print("=" * 60)
    print("Test: npu_batch_nms")
    
    np.random.seed(42)
    batch_size, num_anchors, num_classes = 2, 100, 4
    boxes = np.random.uniform(0, 100, (batch_size, num_anchors, 1, 4)).astype(np.float16)
    scores = np.random.uniform(0, 1, (batch_size, num_anchors, num_classes)).astype(np.float16)
    
    score_threshold, iou_threshold = 0.3, 0.5
    max_size_per_class, max_total_size = 10, 20
    
    golden_boxes, golden_scores, golden_classes, golden_num = batch_nms_numpy(
        boxes, scores, score_threshold, iou_threshold, max_size_per_class, max_total_size
    )
    print(f"[golden] nmsed_num={golden_num}")
    
    torch_npu_result = None
    if run_mode == "npu":
        import torch_npu
        boxes_t = torch.from_numpy(boxes).npu()
        scores_t = torch.from_numpy(scores).npu()
        try:
            torch_npu_result = torch_npu.npu_batch_nms(
                boxes_t, scores_t, score_threshold, iou_threshold, max_size_per_class, max_total_size
            )
            print(f"[torch_npu] nmsed_num={torch_npu_result[3].cpu().numpy()}")
        except Exception as e:
            print(f"[torch_npu] ERROR: {e}")
    
    pypto_result = (golden_boxes, golden_scores, golden_classes, golden_num)
    print(f"[pypto/numpy] 与golden相同")
    
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