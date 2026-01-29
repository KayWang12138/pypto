#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SumLstm Test Script

- Generates golden data using PyTorch
- Calls aclnn custom operator via C++ executable
- Compares results with precision threshold (双千分之五 = 0.5%)
"""

import os
import subprocess
import sys
import time
import numpy as np

try:
    import torch
except ImportError:
    print("Error: PyTorch not installed. Please install with: pip install torch")
    sys.exit(1)


# ==================== SumLstm PyTorch Implementation ====================

def rms_norm(x: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm: x / sqrt(mean(x^2) + eps)"""
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)


def gelu_fast(x: torch.Tensor) -> torch.Tensor:
    """Fast GELU using tanh approximation"""
    return 0.5 * x * (1.0 + torch.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x.pow(3))))


def sum_lstm_pytorch(states_4d, z4_4d, prev_cell, w_cell=None, b_cell=None,
                     w_state=None, b_state=None, alpha=1.0, eps_cell=1e-6,
                     eps_state=1e-6, use_fast_gelu=True):
    """
    SumLstm forward implementation in PyTorch

    Args:
        states_4d: (..., 4D) - 4-gate states
        z4_4d: (..., 4D) - 4-gate increments
        prev_cell: (..., D) - Previous cell state
        w_cell, b_cell: (D,) - Cell RMSNorm weight/bias (optional)
        w_state, b_state: (D,) - State RMSNorm weight/bias (optional)
        alpha: Scaling coefficient for z
        eps_cell, eps_state: Epsilon for RMSNorm
        use_fast_gelu: Use tanh approximation for GELU

    Returns:
        out_state: (..., D), out_cell: (..., D)
    """
    hidden_dim = prev_cell.shape[-1]

    # Split into 4 gates
    s0, s1, s2, s3 = states_4d.split(hidden_dim, dim=-1)
    z0, z1, z2, z3 = z4_4d.split(hidden_dim, dim=-1)

    # Gate pre-activations
    pre_f = s0 + alpha * z0
    pre_i = s1 + alpha * z1
    pre_o = s2 + alpha * z2
    cpre = s3 + alpha * z3

    # Gate values (sigmoid)
    f = torch.sigmoid(pre_f)
    i = torch.sigmoid(pre_i)
    o = torch.sigmoid(pre_o)

    # Cell state computation
    cpre_norm = rms_norm(cpre, eps_cell)
    if w_cell is not None:
        cpre_norm = cpre_norm * w_cell
    if b_cell is not None:
        cpre_norm = cpre_norm + b_cell
    cact = gelu_fast(cpre_norm)
    out_cell = prev_cell * f + cact * i

    # Output state computation
    cnew_norm = rms_norm(out_cell, eps_state)
    if w_state is not None:
        cnew_norm = cnew_norm * w_state
    if b_state is not None:
        cnew_norm = cnew_norm + b_state
    sact = gelu_fast(cnew_norm)
    out_state = sact * o

    return out_state, out_cell


# ==================== Precision Check ====================

def check_precision(golden, actual, name, threshold=0.005, abs_threshold=1e-3, exceed_ratio_limit=0.02):
    """
    Check precision with relative error threshold.
    双千分之五判定标准:
    - mean_rel_error < 0.5%
    - 超出阈值的元素比例 < 2% (允许少量边缘情况)
    - 或者绝对误差足够小 (< abs_threshold)
    """
    golden_f32 = golden.float().view(-1)
    actual_f32 = actual.float().view(-1)

    abs_diff = torch.abs(golden_f32 - actual_f32)
    # 使用混合误差判定: 相对误差或绝对误差任一满足即可
    rel_diff = abs_diff / torch.clamp(torch.abs(golden_f32), min=1e-6)

    max_abs_err = abs_diff.max().item()
    mean_abs_err = abs_diff.mean().item()
    max_rel_err = rel_diff.max().item()
    mean_rel_err = rel_diff.mean().item()

    # 元素通过判定: 相对误差<阈值 或 绝对误差<绝对阈值
    element_pass = (rel_diff < threshold) | (abs_diff < abs_threshold)
    exceed_count = (~element_pass).sum().item()
    total_count = rel_diff.numel()
    exceed_ratio = exceed_count / total_count

    print(f"\n--- {name} Precision Report ---")
    print(f"  Shape:          {list(golden.shape)}")
    print(f"  Max Abs Error:  {max_abs_err:.6e}")
    print(f"  Mean Abs Error: {mean_abs_err:.6e}")
    print(f"  Max Rel Error:  {max_rel_err:.6e} (threshold: {threshold})")
    print(f"  Mean Rel Error: {mean_rel_err:.6e} (threshold: {threshold})")
    print(f"  Exceed Count:   {exceed_count}/{total_count} ({exceed_ratio * 100:.2f}%)")

    # 通过条件: 平均相对误差<阈值 且 超出比例<限制
    pass_mean = mean_rel_err < threshold
    pass_exceed_ratio = exceed_ratio < exceed_ratio_limit
    passed = pass_mean and pass_exceed_ratio

    if passed:
        print(f"  Result:         PASS")
    else:
        print(f"  Result:         FAIL")
        if not pass_mean:
            print(f"    - mean_rel_err {mean_rel_err:.6e} >= threshold {threshold}")
        if not pass_exceed_ratio:
            print(f"    - exceed_ratio {exceed_ratio*100:.2f}% >= limit {exceed_ratio_limit*100:.1f}%")

    return passed, max_rel_err, mean_rel_err


# ==================== Main ====================

def main():
    print("=" * 60)
    print("SumLstm Custom Operator Test")
    print("Precision threshold: 0.5% (双千分之五)")
    print("=" * 60)

    # Parameters (must match main.cpp)
    batch, seq_len, hidden_dim = 2, 4, 64
    gated_dim = 4 * hidden_dim
    alpha = 1.0
    eps_cell = 1e-6
    eps_state = 1e-6

    # Generate input data
    print("\n[1/4] Generating input data...")
    states_4d = torch.zeros(batch, seq_len, gated_dim, dtype=torch.float16)
    z4_4d = torch.zeros(batch, seq_len, gated_dim, dtype=torch.float16)
    prev_cell = torch.zeros(batch, seq_len, hidden_dim, dtype=torch.float16)

    for i in range(batch * seq_len * gated_dim):
        states_4d.view(-1)[i] = 0.1 * (i % 10 - 5)
        z4_4d.view(-1)[i] = 0.05 * (i % 10 - 5)
    for i in range(batch * seq_len * hidden_dim):
        prev_cell.view(-1)[i] = 0.1 * (i % 10 - 5)

    w_cell = torch.ones(hidden_dim, dtype=torch.float16)
    b_cell = torch.zeros(hidden_dim, dtype=torch.float16)
    w_state = torch.ones(hidden_dim, dtype=torch.float16)
    b_state = torch.zeros(hidden_dim, dtype=torch.float16)

    # Save inputs for C++
    states_4d.contiguous().cpu().numpy().tofile("input_states_4d.bin")
    z4_4d.contiguous().cpu().numpy().tofile("input_z4_4d.bin")
    prev_cell.contiguous().cpu().numpy().tofile("input_prev_cell.bin")
    w_cell.contiguous().cpu().numpy().tofile("input_w_cell.bin")
    b_cell.contiguous().cpu().numpy().tofile("input_b_cell.bin")
    w_state.contiguous().cpu().numpy().tofile("input_w_state.bin")
    b_state.contiguous().cpu().numpy().tofile("input_b_state.bin")
    print("  Input data saved.")

    # Compute golden output using PyTorch (fp16 to match NPU kernel)
    print("\n[2/4] Computing PyTorch golden output (fp16)...")
    warmup_runs = 10
    benchmark_runs = 100

    # Warmup (使用 fp16 计算以匹配 NPU kernel)
    for _ in range(warmup_runs):
        _ = sum_lstm_pytorch(
            states_4d, z4_4d, prev_cell,
            w_cell, b_cell, w_state, b_state,
            alpha, eps_cell, eps_state
        )

    # Benchmark (使用 fp16 计算以匹配 NPU kernel)
    start_time = time.time()
    for _ in range(benchmark_runs):
        golden_state, golden_cell = sum_lstm_pytorch(
            states_4d, z4_4d, prev_cell,
            w_cell, b_cell, w_state, b_state,
            alpha, eps_cell, eps_state
        )
    pytorch_time = (time.time() - start_time) / benchmark_runs * 1000

    golden_state_fp16 = golden_state
    golden_cell_fp16 = golden_cell

    print(f"  PyTorch time: {pytorch_time:.4f} ms (avg over {benchmark_runs} runs)")
    print(f"  Golden out_state first 5: {golden_state_fp16.view(-1)[:5].float().tolist()}")
    print(f"  Golden out_cell first 5:  {golden_cell_fp16.view(-1)[:5].float().tolist()}")

    # Run NPU custom operator
    print("\n[3/4] Running NPU custom operator...")
    exe_path = "./build/execute_sum_lstm"
    if not os.path.exists(exe_path):
        print(f"  ERROR: {exe_path} not found. Please run build first.")
        return 1

    npu_time = None
    try:
        result = subprocess.run([exe_path], capture_output=True, text=True, timeout=60)
        print(result.stdout)
        if result.returncode != 0:
            print(f"  ERROR: NPU execution failed with code {result.returncode}")
            print(result.stderr)
            return 1
        # 解析 NPU 执行时间
        for line in result.stdout.split('\n'):
            if 'NPU Average Time:' in line:
                try:
                    npu_time = float(line.split(':')[1].strip().replace('ms', '').strip())
                except:
                    pass
    except subprocess.TimeoutExpired:
        print("  ERROR: NPU execution timed out")
        return 1
    except Exception as e:
        print(f"  ERROR: {e}")
        return 1

    # Load NPU outputs
    print("\n[4/4] Comparing results...")
    if not os.path.exists("output_state.bin") or not os.path.exists("output_cell.bin"):
        print("  ERROR: NPU output files not found")
        return 1

    actual_state = torch.from_numpy(
        np.fromfile("output_state.bin", dtype=np.float16).reshape(batch, seq_len, hidden_dim)
    )
    actual_cell = torch.from_numpy(
        np.fromfile("output_cell.bin", dtype=np.float16).reshape(batch, seq_len, hidden_dim)
    )

    print(f"  Actual out_state first 5: {actual_state.view(-1)[:5].float().tolist()}")
    print(f"  Actual out_cell first 5:  {actual_cell.view(-1)[:5].float().tolist()}")

    # Precision check (双千分之五 = 0.5%)
    threshold = 0.005
    state_pass, state_max_err, state_mean_err = check_precision(
        golden_state_fp16, actual_state, "out_state", threshold
    )
    cell_pass, cell_max_err, cell_mean_err = check_precision(
        golden_cell_fp16, actual_cell, "out_cell", threshold
    )

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  PyTorch Time:      {pytorch_time:.4f} ms (CPU, fp16)")
    if npu_time is not None:
        print(f"  NPU Time:          {npu_time:.4f} ms (Ascend)")
        if npu_time > 0:
            print(f"  Speedup:           {pytorch_time / npu_time:.2f}x")
    print(f"  Precision Threshold: {threshold * 100:.1f}% rel_error (双千分之五)")
    print(f"  Criteria: mean_rel < {threshold} AND exceed_ratio < 2%")
    print(f"  out_state: {'PASS' if state_pass else 'FAIL'} (max_rel={state_max_err:.4e}, mean_rel={state_mean_err:.4e})")
    print(f"  out_cell:  {'PASS' if cell_pass else 'FAIL'} (max_rel={cell_max_err:.4e}, mean_rel={cell_mean_err:.4e})")

    if state_pass and cell_pass:
        print("\n  *** OVERALL: PASS ***")
        return 0
    else:
        print("\n  *** OVERALL: FAIL ***")
        return 1


if __name__ == "__main__":
    sys.exit(main())
