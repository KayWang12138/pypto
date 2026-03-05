#!/usr/bin/env python3
"""
Performance test for embedding_head_quant operator.
BF16 input/output with FP32 internal computation.
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
from embedding_head_quant import (
    create_embedding_head_quant_kernel,
    embedding_head_quant_golden
)

def benchmark_operator(weight_shape, device_id, num_iterations=5):
    device = f'npu:{device_id}'
    print(f"\nBenchmarking weight_shape: {weight_shape}")
    print(f"Device: {device}")
    print(f"Iterations: {num_iterations}")

    kernel = create_embedding_head_quant_kernel(weight_shape[1], run_mode="npu")

    weight_torch = (torch.randn(weight_shape, dtype=torch.float32, device=device) * 50).to(torch.bfloat16)
    scale_torch = torch.full((1, 1), 0.5, dtype=torch.bfloat16, device=device)

    output_torch = torch.zeros(weight_shape, dtype=torch.bfloat16, device=device)
    clamped_torch = torch.zeros(weight_shape, dtype=torch.bfloat16, device=device)
    protected_scale_torch = torch.zeros(weight_shape, dtype=torch.bfloat16, device=device)

    for _ in range(5):
        kernel(weight_torch, scale_torch, output_torch, clamped_torch, protected_scale_torch)

    start_time = time.time()
    for _ in range(num_iterations):
        kernel(weight_torch, scale_torch, output_torch, clamped_torch, protected_scale_torch)
    end_time = time.time()

    total_time = end_time - start_time
    avg_time = total_time / num_iterations
    throughput = num_iterations / total_time
    elements = np.prod(weight_shape)
    elements_per_sec = elements * throughput / 1e6

    print(f"  Total time: {total_time:.4f}s")
    print(f"  Average time: {avg_time*1000:.4f}ms")
    print(f"  Throughput: {throughput:.2f} ops/sec")
    print(f"  Element throughput: {elements_per_sec:.2f} M elements/sec")

    expected_out, expected_clamped, expected_scale = embedding_head_quant_golden(weight_torch, scale_torch)
    max_diff = (output_torch - expected_out).abs().max().item()
    print(f"  Max error: {max_diff:.6f}")

    return {
        'weight_shape': weight_shape,
        'elements': int(elements),
        'total_time': total_time,
        'avg_time_ms': avg_time * 1000,
        'throughput_ops_per_sec': throughput,
        'throughput_M_elements_per_sec': elements_per_sec,
        'max_error': max_diff
    }

def main():
    if 'TILE_FWK_DEVICE_ID' not in os.environ:
        print("ERROR: TILE_FWK_DEVICE_ID not set")
        return
    device_id = int(os.environ['TILE_FWK_DEVICE_ID'])
    torch.npu.set_device(device_id)

    print("=" * 70)
    print("Embedding Head Quantization - Performance Benchmark (BF16 I/O)")
    print("weight: (N, M), scale: (1, 1)")
    print("=" * 70)

    test_shapes = [
        (1024, 2048),
        (768, 2048),
        (2048, 768),
        (4096, 2048),
    ]

    results = []
    for shape in test_shapes:
        result = benchmark_operator(shape, device_id, num_iterations=5)
        results.append(result)

    print("\n" + "=" * 70)
    print("Performance Summary")
    print("=" * 70)
    print(f"{'Shape':<18} {'Elements':<12} {'Avg (ms)':<12} {'Throughput (M/s)':<20} {'Error':<10}")
    print("-" * 70)
    for r in results:
        shape_str = str(r['weight_shape'])
        print(f"{shape_str:<18} {r['elements']:<12} "
              f"{r['avg_time_ms']:<12.4f} {r['throughput_M_elements_per_sec']:<20.2f} "
              f"{r['max_error']:<10.6f}")

    print("\n" + "=" * 70)
    print("Performance Analysis")
    print("=" * 70)

    best_result = max(results, key=lambda x: x['throughput_M_elements_per_sec'])
    print(f"Best throughput: {best_result['throughput_M_elements_per_sec']:.2f} M elements/sec")
    print(f"  at shape: {best_result['weight_shape']}")

    max_error = max(r['max_error'] for r in results)
    print(f"\nMaximum error: {max_error:.6f}")

    print("\n" + "=" * 70)
    print("Benchmark completed!")
    print("=" * 70)

if __name__ == "__main__":
    main()