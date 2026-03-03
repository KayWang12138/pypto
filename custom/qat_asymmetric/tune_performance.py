#!/usr/bin/env python3
"""
PyPTO Performance Tuning Loop for QAT asymmetric quantization operator.
Systematic parameter sweep with stop criteria for TileShape optimization.
"""
import os
import time
import pypto
import torch
import torch_npu
import numpy as np
import json

import sys
sys.path.insert(0, os.path.dirname(__file__))
from qat_asymmetric import qat_asymmetric_golden

DEVICE_ID = 0
TEST_SHAPE = (256, 256)
GROUP_SIZE = 128
BIT = 4
WARMUP_ROUNDS = 10
MEASURE_ROUNDS = 100
MIN_GAIN_THRESHOLD = 0.01
REGRESSION_TOLERANCE = 0.015

VEC_TILE_SWEEP = [
    (16, 16),
    (32, 32),
    (64, 64),
    (128, 128),
    (256, 256),
]

def create_tunable_kernel(orig_shape, num_groups, group_size, bit, vec_tile_x, vec_tile_y):
    total_elements = num_groups * group_size
    n_levels = 2 ** (bit - 1)
    shift = 0.5
    neg_clip_val = -0.99
    clip_val = 0.99
    eps = 1e-4

    @pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
    def qat_asymmetric_kernel(
        weight: pypto.Tensor((total_elements,), pypto.DT_FP32),
        scale: pypto.Tensor((num_groups,), pypto.DT_FP32),
        offset: pypto.Tensor((num_groups,), pypto.DT_FP32),
    ) -> pypto.Tensor((total_elements,), pypto.DT_FP32):
        pypto.set_vec_tile_shapes(vec_tile_x, vec_tile_y)
        
        protected_scale = pypto.maximum(scale, eps)
        alpha = pypto.mul(protected_scale, n_levels)
        
        weight_2d = pypto.reshape(weight, [num_groups, group_size])
        
        offset_2d = pypto.reshape(offset, [num_groups, 1])
        offset_expanded = pypto.expand_clone(offset_2d, [num_groups, group_size])
        
        alpha_2d = pypto.reshape(alpha, [num_groups, 1])
        alpha_expanded = pypto.expand_clone(alpha_2d, [num_groups, group_size])
        
        weight_shifted = pypto.sub(weight_2d, offset_expanded)
        
        weight_norm = pypto.div(weight_shifted, alpha_expanded)
        weight_clipped = pypto.clip(weight_norm, neg_clip_val, clip_val)
        
        weight_scaled = pypto.mul(weight_clipped, n_levels)
        weight_shifted2 = pypto.sub(weight_scaled, shift)
        
        weight_rounded = pypto.round(weight_shifted2, decimals=0)
        
        weight_unshifted = pypto.add(weight_rounded, shift)
        weight_denorm = pypto.div(weight_unshifted, n_levels)
        
        weight_rescaled = pypto.mul(weight_denorm, alpha_expanded)
        output_2d = pypto.add(weight_rescaled, offset_expanded)
        
        output = pypto.reshape(output_2d, [total_elements])
        
        return output

    return qat_asymmetric_kernel

def benchmark_config(vec_tile_x, vec_tile_y, device_id):
    print(f"\n{'='*70}")
    print(f"Benchmarking: vec_tile=({vec_tile_x}, {vec_tile_y})")
    print(f"{'='*70}")

    device = f'npu:{device_id}'
    shape = TEST_SHAPE
    total_elements = shape[0] * shape[1]
    num_groups = total_elements // GROUP_SIZE

    kernel = create_tunable_kernel(shape, num_groups, GROUP_SIZE, BIT, vec_tile_x, vec_tile_y)

    torch.manual_seed(42)
    weight_torch = torch.randn(shape, dtype=torch.float32, device=device)
    scale_torch = torch.rand(num_groups, dtype=torch.float32, device=device) * 0.1 + 0.01
    offset_torch = torch.randn(num_groups, dtype=torch.float32, device=device) * 0.1

    weight_flat = weight_torch.view(-1)

    for _ in range(WARMUP_ROUNDS):
        _ = kernel(weight_flat, scale_torch, offset_torch)

    times = []
    for _ in range(MEASURE_ROUNDS):
        start = time.time()
        output_flat = kernel(weight_flat, scale_torch, offset_torch)
        torch.npu.synchronize()
        end = time.time()
        times.append((end - start) * 1000)

    times = np.array(times)
    avg_time_ms = np.mean(times)
    std_time_ms = np.std(times)
    min_time_ms = np.min(times)
    max_time_ms = np.max(times)

    elements = total_elements
    throughput_ops_per_sec = 1000.0 / avg_time_ms
    elements_per_sec = elements * throughput_ops_per_sec / 1e6

    output_torch = output_flat.view(shape)
    expected = qat_asymmetric_golden(weight_torch, scale_torch, offset_torch, GROUP_SIZE, BIT)
    max_diff = (output_torch - expected).abs().max().item()
    correctness = max_diff < 1e-2

    print(f"  Shape: {shape}")
    print(f"  Elements: {elements}")
    print(f"  Latency (ms):")
    print(f"    Mean: {avg_time_ms:.4f}")
    print(f"    Std:  {std_time_ms:.4f}")
    print(f"    Min:  {min_time_ms:.4f}")
    print(f"    Max:  {max_time_ms:.4f}")
    print(f"  Throughput:")
    print(f"    Ops/sec: {throughput_ops_per_sec:.2f}")
    print(f"    Elements/sec: {elements_per_sec:.2f} M/s")
    print(f"  Correctness:")
    print(f"    Max error: {max_diff:.6f}")
    print(f"    Status: {'PASS' if correctness else 'FAIL'}")

    return {
        'vec_tile': (vec_tile_x, vec_tile_y),
        'avg_time_ms': avg_time_ms,
        'std_time_ms': std_time_ms,
        'min_time_ms': min_time_ms,
        'max_time_ms': max_time_ms,
        'throughput_ops_M_per_sec': throughput_ops_per_sec,
        'throughput_elements_M_per_sec': elements_per_sec,
        'max_error': max_diff,
        'correctness': correctness,
        'round_times_ms': times.tolist()
    }

def apply_stop_criteria(results, best_idx, current_idx, best_throughput, current_throughput):
    regression = (best_throughput - current_throughput) / best_throughput
    if regression > REGRESSION_TOLERANCE:
        print(f"\nSTOP: Regression detected ({regression*100:.2f}% > {REGRESSION_TOLERANCE*100:.1f}%)")
        return False, "regression_threshold"

    if current_idx > best_idx:
        improvement = (current_throughput - best_throughput) / best_throughput
        if improvement < MIN_GAIN_THRESHOLD:
            print(f"\nSTOP: No meaningful gain ({improvement*100:.2f}% < {MIN_GAIN_THRESHOLD*100:.1f}%)")
            return False, "no_improve_streak"

    return True, None

def main():
    if 'TILE_FWK_DEVICE_ID' not in os.environ:
        print("ERROR: TILE_FWK_DEVICE_ID not set")
        print("  export TILE_FWK_DEVICE_ID=0")
        return
    device_id = int(os.environ['TILE_FWK_DEVICE_ID'])
    torch.npu.set_device(device_id)

    if 'PTO_TILE_LIB_CODE_PATH' not in os.environ:
        print("ERROR: PTO_TILE_LIB_CODE_PATH not set")
        print("  export PTO_TILE_LIB_CODE_PATH=/path/to/pto-isa")
        return

    print("="*70)
    print("PyPTO Performance Tuning Loop - QAT Asymmetric Quantization")
    print("="*70)
    print(f"Device: npu:{device_id}")
    print(f"Test Shape: {TEST_SHAPE}")
    print(f"Group Size: {GROUP_SIZE}")
    print(f"Bit Width: {BIT}")
    print(f"Warmup: {WARMUP_ROUNDS} rounds")
    print(f"Measure: {MEASURE_ROUNDS} rounds")
    print(f"Min Gain Threshold: {MIN_GAIN_THRESHOLD*100:.1f}%")
    print(f"Regression Tolerance: {REGRESSION_TOLERANCE*100:.1f}%")
    print("="*70)

    results = []
    best_result = None
    best_idx = 0
    stop_reason = None

    for idx, (tile_x, tile_y) in enumerate(VEC_TILE_SWEEP):
        result = benchmark_config(tile_x, tile_y, device_id)
        results.append(result)

        if best_result is None or result['throughput_elements_M_per_sec'] > best_result['throughput_elements_M_per_sec']:
            best_result = result
            best_idx = idx
            print(f"  New best configuration!")
        else:
            should_continue, stop_reason = apply_stop_criteria(
                results, best_idx, idx,
                best_result['throughput_elements_M_per_sec'],
                result['throughput_elements_M_per_sec']
            )
            if not should_continue:
                break

    print("\n" + "="*70)
    print("TUNING SUMMARY")
    print("="*70)

    print(f"\nBaseline (vec_tile={VEC_TILE_SWEEP[1]}):")
    baseline = results[1]
    print(f"  Throughput: {baseline['throughput_elements_M_per_sec']:.2f} M elements/sec")
    print(f"  Latency: {baseline['avg_time_ms']:.4f} ms")

    print(f"\nBest Configuration (vec_tile={best_result['vec_tile']}):")
    print(f"  Throughput: {best_result['throughput_elements_M_per_sec']:.2f} M elements/sec")
    print(f"  Latency: {best_result['avg_time_ms']:.4f} ms")

    if best_result != results[1]:
        gain = (best_result['throughput_elements_M_per_sec'] - baseline['throughput_elements_M_per_sec']) / baseline['throughput_elements_M_per_sec']
        latency_gain = (baseline['avg_time_ms'] - best_result['avg_time_ms']) / baseline['avg_time_ms']
        print(f"\nImprovement:")
        print(f"  Throughput: +{gain*100:.2f}%")
        print(f"  Latency: -{latency_gain*100:.2f}%")
    else:
        print(f"\nImprovement: None (baseline is best)")

    print(f"\nStop Reason: {stop_reason or 'exhausted_search'}")

    if best_idx == 0:
        direction = "decrease"
    elif best_idx == len(results) - 1:
        direction = "increase"
    else:
        direction = "center_optimum"

    print(f"Tuning Direction: {direction}")

    print("\n" + "="*70)
    print("FULL COMPARISON TABLE")
    print("="*70)
    print(f"{'Vec Tile':<15} {'Throughput (M/s)':<20} {'Latency (ms)':<15} {'Error':<10} {'Status':<10}")
    print("-"*70)
    for r in results:
        tile_str = f"({r['vec_tile'][0]}, {r['vec_tile'][1]})"
        marker = "[BEST]" if r == best_result else "      "
        print(f"{marker} {tile_str:<13} {r['throughput_elements_M_per_sec']:<20.2f} "
              f"{r['avg_time_ms']:<15.4f} {r['max_error']:<10.6f} "
              f"{'OK' if r['correctness'] else 'FAIL':<10}")

    output_file = os.path.join(os.path.dirname(__file__), "tuning_results.json")
    with open(output_file, 'w') as f:
        json.dump({
            'operator': 'qat_asymmetric',
            'baseline': results[1],
            'best': best_result,
            'all_results': results,
            'best_idx': best_idx,
            'stop_reason': stop_reason,
            'direction': direction,
            'config': {
                'device_id': device_id,
                'test_shape': TEST_SHAPE,
                'group_size': GROUP_SIZE,
                'bit': BIT,
                'warmup_rounds': WARMUP_ROUNDS,
                'measure_rounds': MEASURE_ROUNDS,
                'min_gain_threshold': MIN_GAIN_THRESHOLD,
                'regression_tolerance': REGRESSION_TOLERANCE
            }
        }, f, indent=2)

    print(f"\nResults saved to: {output_file}")
    print("\n" + "="*70)
    print("Tuning completed!")
    print("="*70)

if __name__ == "__main__":
    main()