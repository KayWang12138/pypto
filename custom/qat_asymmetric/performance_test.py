#!/usr/bin/env python3
"""
Performance test for QAT asymmetric quantization operator.
BF16 I/O with FP32 internal computation.
"""
import os
import time
import pypto
import torch
import torch_npu
import numpy as np
from numpy.testing import assert_allclose

import sys
sys.path.insert(0, os.path.dirname(__file__))
from qat_asymmetric import (
    create_qat_asymmetric_kernel,
    qat_asymmetric_golden
)

def benchmark_operator(orig_shape, group_size, bit, device_id, num_iterations=100):
    device = f'npu:{device_id}'
    total_elements = 1
    for dim in orig_shape:
        total_elements *= dim
    num_groups = total_elements // group_size
    
    print(f"\nBenchmarking shape: {orig_shape}")
    print(f"  Total elements: {total_elements}")
    print(f"  Group size: {group_size}")
    print(f"  Num groups: {num_groups}")
    print(f"  Bit width: {bit}")
    print(f"  Device: {device}")
    print(f"  Iterations: {num_iterations}")

    kernel = create_qat_asymmetric_kernel(
        orig_shape, num_groups, group_size, bit, run_mode="npu"
    )

    # Generate test data in BF16
    weight_torch = (torch.randn(orig_shape, dtype=torch.float32, device=device)).to(torch.bfloat16)
    scale_torch = (torch.rand(num_groups, dtype=torch.float32, device=device) * 0.1 + 0.01).to(torch.bfloat16)
    offset_torch = (torch.randn(num_groups, dtype=torch.float32, device=device) * 0.1).to(torch.bfloat16)

    weight_flat = weight_torch.view(-1)

    for _ in range(10):
        _ = kernel(weight_flat, scale_torch, offset_torch)

    start_time = time.time()
    for _ in range(num_iterations):
        output_flat, weight_denorm_flat, alpha_expanded_flat = kernel(weight_flat, scale_torch, offset_torch)
    end_time = time.time()

    total_time = end_time - start_time
    avg_time = total_time / num_iterations
    throughput = num_iterations / total_time
    elements_per_sec = total_elements * throughput / 1e6

    print(f"  Total time: {total_time:.4f}s")
    print(f"  Average time: {avg_time*1000:.4f}ms")
    print(f"  Throughput: {throughput:.2f} ops/sec")
    print(f"  Element throughput: {elements_per_sec:.2f} M elements/sec")

    output_torch = output_flat.view(orig_shape)
    expected_out, _, _ = qat_asymmetric_golden(weight_torch, scale_torch, offset_torch, group_size, bit)
    max_diff = (output_torch - expected_out).abs().max().item()
    print(f"  Max error: {max_diff:.6f}")

    return {
        'shape': orig_shape,
        'elements': int(total_elements),
        'group_size': group_size,
        'num_groups': num_groups,
        'bit': bit,
        'total_time': total_time,
        'avg_time_ms': avg_time * 1000,
        'throughput_ops_per_sec': throughput,
        'throughput_M_elements_per_sec': elements_per_sec,
        'max_error': max_diff
    }

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

    print("=" * 70)
    print("QAT Asymmetric Quantization - Performance Benchmark (BF16 I/O)")
    print("=" * 70)

    test_configs = [
        ((16, 16), 32, 4),
        ((32, 32), 32, 4),
        ((64, 64), 64, 4),
        ((128, 128), 128, 4),
        ((256, 256), 128, 4),
        ((512, 512), 128, 4),
    ]

    results = []
    for shape, group_size, bit in test_configs:
        result = benchmark_operator(shape, group_size, bit, device_id, num_iterations=100)
        results.append(result)

    print("\n" + "=" * 70)
    print("Performance Summary")
    print("=" * 70)
    print(f"{'Shape':<15} {'Elements':<12} {'Groups':<10} {'Avg (ms)':<12} {'Throughput (M/s)':<18} {'Error':<10}")
    print("-" * 70)
    for r in results:
        shape_str = str(r['shape'])
        print(f"{shape_str:<15} {r['elements']:<12} {r['num_groups']:<10} "
              f"{r['avg_time_ms']:<12.4f} {r['throughput_M_elements_per_sec']:<18.2f} "
              f"{r['max_error']:<10.6f}")

    print("\n" + "=" * 70)
    print("Performance Analysis")
    print("=" * 70)

    best_result = max(results, key=lambda x: x['throughput_M_elements_per_sec'])
    print(f"Best throughput: {best_result['throughput_M_elements_per_sec']:.2f} M elements/sec")
    print(f"  at shape: {best_result['shape']}")

    if len(results) >= 2:
        small_throughput = results[0]['throughput_M_elements_per_sec']
        large_throughput = results[-1]['throughput_M_elements_per_sec']
        scaling_ratio = large_throughput / small_throughput
        print(f"\nThroughput scaling (large/small): {scaling_ratio:.2f}x")
        print(f"  Small shape throughput: {small_throughput:.2f} M elements/sec")
        print(f"  Large shape throughput: {large_throughput:.2f} M elements/sec")

    max_error = max(r['max_error'] for r in results)
    print(f"\nMaximum error across all tests: {max_error:.6f}")
    if max_error < 1e-3:
        print("Accuracy: Excellent (error < 1e-3)")
    elif max_error < 1e-2:
        print("Accuracy: Good (error < 1e-2)")
    else:
        print("Accuracy: Needs improvement")

    print("\n" + "=" * 70)
    print("Benchmark completed successfully!")
    print("=" * 70)

if __name__ == "__main__":
    main()