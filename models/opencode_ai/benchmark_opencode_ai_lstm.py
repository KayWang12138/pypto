#!/usr/bin/env python3
# coding: utf-8
import os
import sys
import time
import torch
import torch_npu
import pypto

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from opencode_ai_lstm import opencode_ai_lstm, gen_input, LstmConfig
from opencode_ai_lstm_golden import opencode_ai_lstm_compute as golden_compute


def benchmark_opencode_ai_lstm(batch_size=32, hidden_dim=4096, num_iters=100):
    """Benchmark OpenCode AI LSTM kernel."""
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)

    x_dtype = torch.float16
    hidden_dim_4 = hidden_dim * 4

    print(f"=== OpenCode AI LSTM Benchmark ===")
    print(f"Batch Size: {batch_size}")
    print(f"Hidden Dim: {hidden_dim}")
    print(f"Hidden Dim x4: {hidden_dim_4}")
    print(f"Iterations: {num_iters}")
    print(f"Device: NPU {device_id}")
    print()

    # Generate inputs
    states_4d, z4_4d, prev_cell, w_cell, b_cell, w_state, b_state, h_out, c_out = \
        gen_input(batch_size, hidden_dim, x_dtype, device_id)

    # Warmup
    print("Warming up...")
    for _ in range(5):
        opencode_ai_lstm(states_4d, z4_4d, prev_cell, w_cell, b_cell, w_state, b_state, h_out, c_out)
    torch_npu.npu.synchronize()

    # Benchmark
    print("Benchmarking...")
    start_time = time.time()
    for _ in range(num_iters):
        opencode_ai_lstm(states_4d, z4_4d, prev_cell, w_cell, b_cell, w_state, b_state, h_out, c_out)
    torch_npu.npu.synchronize()
    end_time = time.time()

    total_time = end_time - start_time
    avg_time = total_time / num_iters
    throughput = num_iters / total_time

    print(f"\n=== Performance Results ===")
    print(f"Total Time: {total_time:.4f} s")
    print(f"Average Time: {avg_time*1000:.4f} ms")
    print(f"Throughput: {throughput:.2f} iters/s")
    print(f"Tokens/s: {throughput * batch_size:.2f}")

    # Verify correctness
    print("\n=== Verifying Correctness ===")
    golden_h, golden_c = golden_compute(
        states_4d.cpu(), z4_4d.cpu(), prev_cell.cpu(),
        alpha=0.1, eps_cell=1e-6, eps_state=1e-6,
        w_cell=w_cell.cpu(), b_cell=b_cell.cpu(), w_state=w_state.cpu(), b_state=b_state.cpu()
    )

    diff_h = (h_out.cpu() - golden_h).abs().max().item()
    diff_c = (c_out.cpu() - golden_c).abs().max().item()
    print(f"Max Diff Hidden: {diff_h:.6f}")
    print(f"Max Diff Cell:   {diff_c:.6f}")

    if diff_h < 0.01 and diff_c < 0.01:
        print("✓ Correctness Check PASSED")
    else:
        print("✗ Correctness Check FAILED")

    return {
        'batch_size': batch_size,
        'hidden_dim': hidden_dim,
        'num_iters': num_iters,
        'total_time': total_time,
        'avg_time_ms': avg_time * 1000,
        'throughput': throughput,
        'tokens_per_sec': throughput * batch_size,
        'max_diff_h': diff_h,
        'max_diff_c': diff_c
    }


if __name__ == "__main__":
    results = benchmark_opencode_ai_lstm(batch_size=32, hidden_dim=4096, num_iters=100)
