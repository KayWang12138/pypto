#!/usr/bin/env python3
import os
import time
import pypto
import torch
import torch_npu
import numpy as np
from numpy.testing import assert_allclose
import json

import sys
sys.path.insert(0, os.path.dirname(__file__))
from embedding_head_quant import (
    create_embedding_head_quant_kernel,
    embedding_head_quant_golden
)

PERFORMANCE_TEST_CONFIGS = {
    "3B": {
        "weight_shape": (153376, 2048),
        "description": "3B model embedding head"
    },
    "7B": {
        "weight_shape": (153376, 3072),
        "description": "7B model embedding head"
    },
    "30B": {
        "weight_shape": (75776, 2560),
        "description": "30B model embedding head"
    }
}

WARMUP_ROUNDS = 5
MEASURE_ROUNDS = 10


def benchmark_pypto_operator(weight_shape, device_id):
    device = f'npu:{device_id}'
    print(f"  Benchmarking PyPTO operator...")
    print(f"    Shape: {weight_shape}")
    print(f"    Warmup: {WARMUP_ROUNDS} rounds")
    print(f"    Measure: {MEASURE_ROUNDS} rounds")

    kernel = create_embedding_head_quant_kernel(weight_shape[1], run_mode="npu")

    torch.manual_seed(42)
    weight_torch = (torch.randn(weight_shape, dtype=torch.float32, device=device) * 50).to(torch.bfloat16)
    scale_torch = torch.full((1, 1), 0.5, dtype=torch.bfloat16, device=device)

    output_torch = torch.zeros(weight_shape, dtype=torch.bfloat16, device=device)
    clamped_torch = torch.zeros(weight_shape, dtype=torch.bfloat16, device=device)
    protected_scale_torch = torch.zeros(weight_shape, dtype=torch.bfloat16, device=device)

    for _ in range(WARMUP_ROUNDS):
        kernel(weight_torch, scale_torch, output_torch, clamped_torch, protected_scale_torch)

    times = []
    for _ in range(MEASURE_ROUNDS):
        start = time.time()
        kernel(weight_torch, scale_torch, output_torch, clamped_torch, protected_scale_torch)
        torch.npu.synchronize()
        end = time.time()
        times.append((end - start) * 1000)

    times = np.array(times)
    avg_time_ms = np.mean(times)
    std_time_ms = np.std(times)
    min_time_ms = np.min(times)
    max_time_ms = np.max(times)

    elements = np.prod(weight_shape)
    throughput_elements_M_per_sec = elements / avg_time_ms / 1000

    expected_out, _, _ = embedding_head_quant_golden(weight_torch, scale_torch)
    max_diff = (output_torch - expected_out).abs().max().item()

    print(f"    Latency (ms): avg={avg_time_ms:.4f}, std={std_time_ms:.4f}, min={min_time_ms:.4f}, max={max_time_ms:.4f}")
    print(f"    Throughput: {throughput_elements_M_per_sec:.2f} M elements/sec")
    print(f"    Max error: {max_diff:.6f}")

    return {
        'avg_time_ms': avg_time_ms,
        'std_time_ms': std_time_ms,
        'min_time_ms': min_time_ms,
        'max_time_ms': max_time_ms,
        'throughput_M_elements_per_sec': throughput_elements_M_per_sec,
        'max_error': max_diff,
        'correctness': max_diff < 3e-3
    }


def benchmark_golden_npu(weight_shape, device_id):
    device = f'npu:{device_id}'
    print(f"  Benchmarking Golden NPU implementation...")
    print(f"    Shape: {weight_shape}")
    print(f"    Warmup: {WARMUP_ROUNDS} rounds")
    print(f"    Measure: {MEASURE_ROUNDS} rounds")

    torch.manual_seed(42)
    weight_torch = (torch.randn(weight_shape, dtype=torch.float32, device=device) * 50).to(torch.bfloat16)
    scale_torch = torch.full((1, 1), 0.5, dtype=torch.bfloat16, device=device)

    for _ in range(WARMUP_ROUNDS):
        _ = embedding_head_quant_golden(weight_torch.clone(), scale_torch.clone())

    times = []
    for _ in range(MEASURE_ROUNDS):
        start = time.time()
        _ = embedding_head_quant_golden(weight_torch.clone(), scale_torch.clone())
        torch.npu.synchronize()
        end = time.time()
        times.append((end - start) * 1000)

    times = np.array(times)
    avg_time_ms = np.mean(times)
    std_time_ms = np.std(times)
    min_time_ms = np.min(times)
    max_time_ms = np.max(times)

    elements = np.prod(weight_shape)
    throughput_elements_M_per_sec = elements / avg_time_ms / 1000

    print(f"    Latency (ms): avg={avg_time_ms:.4f}, std={std_time_ms:.4f}, min={min_time_ms:.4f}, max={max_time_ms:.4f}")
    print(f"    Throughput: {throughput_elements_M_per_sec:.2f} M elements/sec")

    return {
        'avg_time_ms': avg_time_ms,
        'std_time_ms': std_time_ms,
        'min_time_ms': min_time_ms,
        'max_time_ms': max_time_ms,
        'throughput_M_elements_per_sec': throughput_elements_M_per_sec
    }


def run_benchmark(model_name, weight_shape, device_id):
    print(f"\n{'='*70}")
    print(f"Model: {model_name} - Shape: {weight_shape}")
    print(f"{'='*70}")

    pypto_result = benchmark_pypto_operator(weight_shape, device_id)
    golden_result = benchmark_golden_npu(weight_shape, device_id)

    speedup = golden_result['avg_time_ms'] / pypto_result['avg_time_ms']
    throughput_ratio = pypto_result['throughput_M_elements_per_sec'] / golden_result['throughput_M_elements_per_sec']

    print(f"\n  Comparison Summary:")
    print(f"    PyPTO latency:    {pypto_result['avg_time_ms']:.4f} ms")
    print(f"    Golden latency:   {golden_result['avg_time_ms']:.4f} ms")
    print(f"    Speedup:          {speedup:.2f}x")
    print(f"    Throughput ratio: {throughput_ratio:.2f}x")
    print(f"    Correctness:      {'PASS' if pypto_result['correctness'] else 'FAIL'}")

    return {
        'model': model_name,
        'weight_shape': weight_shape,
        'elements': int(np.prod(weight_shape)),
        'pypto': pypto_result,
        'golden': golden_result,
        'speedup': speedup,
        'throughput_ratio': throughput_ratio
    }


def main():
    if 'TILE_FWK_DEVICE_ID' not in os.environ:
        print("ERROR: TILE_FWK_DEVICE_ID not set")
        print("  export TILE_FWK_DEVICE_ID=0")
        return
    device_id = int(os.environ['TILE_FWK_DEVICE_ID'])
    torch.npu.set_device(device_id)

    print("=" * 70)
    print("Embedding Head Quantization - Performance Benchmark")
    print("Comparing PyPTO Operator vs Golden NPU Implementation")
    print("=" * 70)
    print(f"Device: npu:{device_id}")
    print(f"Warmup rounds: {WARMUP_ROUNDS}")
    print(f"Measure rounds: {MEASURE_ROUNDS}")
    print("=" * 70)

    results = []
    for model_name, config in PERFORMANCE_TEST_CONFIGS.items():
        result = run_benchmark(model_name, config['weight_shape'], device_id)
        results.append(result)

    print("\n" + "=" * 70)
    print("PERFORMANCE SUMMARY")
    print("=" * 70)
    print(f"{'Model':<8} {'Shape':<20} {'PyPTO(ms)':<12} {'Golden(ms)':<12} {'Speedup':<10} {'Correctness':<10}")
    print("-" * 70)
    for r in results:
        shape_str = str(r['weight_shape'])
        print(f"{r['model']:<8} {shape_str:<20} {r['pypto']['avg_time_ms']:<12.4f} "
              f"{r['golden']['avg_time_ms']:<12.4f} {r['speedup']:<10.2f} "
              f"{'PASS' if r['pypto']['correctness'] else 'FAIL':<10}")

    print("\n" + "=" * 70)
    print("THROUGHPUT SUMMARY")
    print("=" * 70)
    print(f"{'Model':<8} {'Elements':<15} {'PyPTO(M/s)':<15} {'Golden(M/s)':<15} {'Ratio':<10}")
    print("-" * 70)
    for r in results:
        print(f"{r['model']:<8} {r['elements']:<15} {r['pypto']['throughput_M_elements_per_sec']:<15.2f} "
              f"{r['golden']['throughput_M_elements_per_sec']:<15.2f} {r['throughput_ratio']:<10.2f}")

    output_file = os.path.join(os.path.dirname(__file__), "benchmark_results.json")
    with open(output_file, 'w') as f:
        json.dump({
            'config': {
                'device_id': device_id,
                'warmup_rounds': WARMUP_ROUNDS,
                'measure_rounds': MEASURE_ROUNDS
            },
            'results': results
        }, f, indent=2)
    print(f"\nResults saved to: {output_file}")

    print("\n" + "=" * 70)
    print("Benchmark completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()