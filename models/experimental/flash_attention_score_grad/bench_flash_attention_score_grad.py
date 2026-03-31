#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
FlashAttentionScoreGrad 性能测试

测量 NPU 上的 kernel 执行时间，包含预热和多次迭代取平均。
"""

import os
import sys
import time
import logging
import torch
import argparse

# Configure logger for the module
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.propagate = False
formatter = logging.Formatter(
    fmt='%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s',
    datefmt='[%Y-%m-%d %H:%M:%S]'
)
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.handlers.clear()
logger.addHandler(handler)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flash_attention_score_grad_golden import generate_forward_data
from flash_attention_score_grad_impl import flash_attention_score_grad_wrapper


def bench(name, B, N, S, D, device_id, warmup=5, repeat=20):
    device = f"npu:{device_id}"
    q, k, v, dy, sm, ss, ao, scale = generate_forward_data(B, N, S, D, device=device)

    # 预热
    for _ in range(warmup):
        dq, dk, dv = flash_attention_score_grad_wrapper(q, k, v, dy, sm, ss, ao, scale, N, D)
    torch.npu.synchronize()

    # 计时
    times = []
    for _ in range(repeat):
        torch.npu.synchronize()
        t0 = time.perf_counter()
        dq, dk, dv = flash_attention_score_grad_wrapper(q, k, v, dy, sm, ss, ao, scale, N, D)
        torch.npu.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)  # ms

    avg = sum(times) / len(times)
    mn = min(times)
    mx = max(times)

    # 计算 FLOPS
    # matmul FLOPS: Q@K^T = 2*S*S*D, dY@V^T = 2*S*S*D, dS@K = 2*S*D*S, dS^T@Q = 2*S*S*D, P^T@dY = 2*S*S*D
    # 趟1: Q@K^T + dY@V^T + dS@K = 3 matmuls per (s1,s2) pair
    # 趟2: Q@K^T + dY@V^T + dS^T@Q + P^T@dY = 4 matmuls per (s1,s2) pair (P&dS recomputed)
    # Total matmul flops per (b,n): 7 * 2 * S * S * D (approx)
    total_matmul_flops = B * N * 7 * 2 * S * S * D
    tflops = total_matmul_flops / (mn / 1000) / 1e12

    logger.info(f"  {name:30s}  B={B:2d} N={N:2d} S={S:4d} D={D:3d}  |  "
                f"avg={avg:8.3f}ms  min={mn:8.3f}ms  max={mx:8.3f}ms  |  "
                f"~{tflops:.2f} TFLOPS")
    return avg, mn


def main():
    parser = argparse.ArgumentParser(description="FlashAttentionScoreGrad Perf Bench")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    args = parser.parse_args()

    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    import torch_npu
    torch.npu.set_device(device_id)

    logger.info("=" * 100)
    logger.info("FlashAttentionScoreGrad Performance Benchmark")
    logger.info(f"Device: NPU:{device_id}  Warmup: {args.warmup}  Repeat: {args.repeat}")
    logger.info("=" * 100)

    configs = [
        ("S=128",   2, 8, 128,   64),
        ("S=256",   2, 8, 256,   64),
        ("S=512",   2, 8, 512,   64),
        ("S=1024",  2, 8, 1024,  64),
        ("S=2048",  2, 8, 2048,  64),
        ("S=4096",  1, 8, 4096,  64),
        ("S=8192",  1, 8, 8192,  64),
    ]

    # Baseline (S_TILE=64, no pass_options):
    # S=128   avg=0.890ms  min=0.786ms  ~0.30 TFLOPS
    # S=256   avg=1.918ms  min=1.853ms  ~0.51 TFLOPS
    # S=512   avg=6.552ms  min=6.401ms  ~0.59 TFLOPS
    # S=1024  avg=24.12ms  min=23.37ms  ~0.64 TFLOPS
    # S=2048  avg=90.32ms  min=88.58ms  ~0.68 TFLOPS
    # S=4096  avg=180.5ms  min=175.5ms  ~0.69 TFLOPS
    # S=8192  avg=708.7ms  min=662.7ms  ~0.73 TFLOPS

    for name, B, N, S, D in configs:
        try:
            bench(name, B, N, S, D, device_id, args.warmup, args.repeat)
        except Exception as e:
            logger.info(f"  {name:30s}  FAILED: {e}")

    logger.info("=" * 100)


if __name__ == "__main__":
    main()
