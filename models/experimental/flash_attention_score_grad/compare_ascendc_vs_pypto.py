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
AscendC vs PyPTO FlashAttentionScoreGrad 性能对比

使用相同 shape 跑 torch_npu 内置算子 (AscendC) 和 PyPTO 实现，直接对比。
"""

import os
import sys
import time
import math
import logging
import torch
import torch_npu

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flash_attention_score_grad_golden import generate_forward_data
from flash_attention_score_grad_impl import (
    flash_attention_score_grad_wrapper, NUM_HEADS, HEAD_DIM,
)


def run_ascendc(  # noqa: C901,PLR0913
        q, k, v, dy, softmax_max, softmax_sum,
        attention_out, scale, num_heads, warmup=5, repeat=20):
    """用 torch_npu.npu_fusion_attention_grad_v2 跑 AscendC 版本"""
    batch_size, _, seq_len, head_dim = q.shape
    # 先跑前向拿到 AscendC 格式的 softmax_max/sum
    fwd_result = torch_npu.npu_fusion_attention(
        q, k, v, num_heads,
        pse=None,
        padding_mask=None,
        atten_mask=None,
        scale=scale,
        keep_prob=1.0,
        input_layout="BNSD",
        pre_tockens=seq_len,
        next_tockens=seq_len,
        inner_precise=0,
        sparse_mode=0,
    )
    out_fwd = fwd_result[0]
    softmax_max_fwd = fwd_result[1]
    softmax_sum_fwd = fwd_result[2]

    # 预热
    for _ in range(warmup):
        dq, dk, dv, *_ = torch_npu.npu_fusion_attention_grad_v2(
            q, k, v, dy, num_heads,
            pse=None,
            padding_mask=None,
            atten_mask=None,
            softmax_max=softmax_max_fwd,
            softmax_sum=softmax_sum_fwd,
            softmax_in=None,
            attention_in=out_fwd,
            scale_value=scale,
            keep_prob=1.0,
            input_layout="BNSD",
            pre_tokens=seq_len,
            next_tokens=seq_len,
            seed=0,
            offset=0,
            numels=batch_size * num_heads * seq_len * seq_len,
            inner_precise=0,
            sparse_mode=0,
        )
    torch.npu.synchronize()

    # 计时
    times = []
    for _ in range(repeat):
        torch.npu.synchronize()
        t0 = time.perf_counter()
        dq, dk, dv, *_ = torch_npu.npu_fusion_attention_grad_v2(
            q, k, v, dy, num_heads,
            pse=None,
            padding_mask=None,
            atten_mask=None,
            softmax_max=softmax_max_fwd,
            softmax_sum=softmax_sum_fwd,
            softmax_in=None,
            attention_in=out_fwd,
            scale_value=scale,
            keep_prob=1.0,
            input_layout="BNSD",
            pre_tokens=seq_len,
            next_tokens=seq_len,
            seed=0,
            offset=0,
            numels=batch_size * num_heads * seq_len * seq_len,
            inner_precise=0,
            sparse_mode=0,
        )
        torch.npu.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    return times


def run_pypto(  # noqa: C901,PLR0913
        q, k, v, dy, softmax_max, softmax_sum,
        attention_out, scale, num_heads, head_dim, warmup=5, repeat=20):
    """跑 PyPTO 版本"""
    for _ in range(warmup):
        dq, dk, dv = flash_attention_score_grad_wrapper(
            q, k, v, dy, softmax_max, softmax_sum, attention_out,
            scale, num_heads, head_dim)
    torch.npu.synchronize()

    times = []
    for _ in range(repeat):
        torch.npu.synchronize()
        t0 = time.perf_counter()
        dq, dk, dv = flash_attention_score_grad_wrapper(
            q, k, v, dy, softmax_max, softmax_sum, attention_out,
            scale, num_heads, head_dim)
        torch.npu.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    return times


def main():  # noqa: C901,PLR0915
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    torch.npu.set_device(device_id)
    device = f"npu:{device_id}"

    # 使用 PyPTO kernel 的 NUM_HEADS=8, HEAD_DIM=64
    n_heads, head_dim = NUM_HEADS, HEAD_DIM

    configs = [
        ("B=2,S=128",   2, 128),
        ("B=2,S=256",   2, 256),
        ("B=2,S=512",   2, 512),
        ("B=2,S=1024",  2, 1024),
        ("B=2,S=2048",  2, 2048),
        ("B=1,S=4096",  1, 4096),
        ("B=1,S=8192",  1, 8192),
    ]

    logger.info("=" * 120)
    logger.info(f"FlashAttentionScoreGrad: AscendC vs PyPTO  |  N={n_heads}, D={head_dim}, BF16, BNSD layout")
    logger.info("=" * 120)
    header = (f"{'Config':<16s} | {'AscendC min(ms)':>15s} {'avg(ms)':>10s} | "
              f"{'PyPTO min(ms)':>15s} {'avg(ms)':>10s} | "
              f"{'Ratio(AC/PT)':>12s} | {'AscendC TFLOPS':>14s} {'PyPTO TFLOPS':>14s}")
    logger.info(header)
    logger.info("-" * 120)

    for name, batch_size, seq_len in configs:
        total_flops = batch_size * n_heads * 7 * 2 * seq_len * seq_len * head_dim

        try:
            q, k, v, dy, sm, ss, ao, scale = generate_forward_data(
                batch_size, n_heads, seq_len, head_dim, device=device)

            # AscendC
            try:
                ac_times = run_ascendc(
                    q, k, v, dy, sm, ss, ao, scale, n_heads, warmup=3, repeat=10)
                ac_min = min(ac_times)
                ac_avg = sum(ac_times) / len(ac_times)
                ac_tflops = total_flops / (ac_min / 1000) / 1e12
            except Exception as e:
                ac_min = ac_avg = float('inf')
                ac_tflops = 0
                logger.info(f"  AscendC failed for {name}: {e}")

            # PyPTO
            try:
                pt_times = run_pypto(
                    q, k, v, dy, sm, ss, ao, scale, n_heads, head_dim,
                    warmup=3, repeat=10)
                pt_min = min(pt_times)
                pt_avg = sum(pt_times) / len(pt_times)
                pt_tflops = total_flops / (pt_min / 1000) / 1e12
            except Exception as e:
                pt_min = pt_avg = float('inf')
                pt_tflops = 0
                logger.info(f"  PyPTO failed for {name}: {e}")

            ratio = pt_min / ac_min if ac_min > 0 and ac_min != float('inf') else float('inf')
            row = (f"{name:<16s} | {ac_min:>14.3f}ms {ac_avg:>9.3f}ms | "
                   f"{pt_min:>14.3f}ms {pt_avg:>9.3f}ms | "
                   f"{ratio:>11.2f}x | {ac_tflops:>13.2f}T {pt_tflops:>13.2f}T")
            logger.info(row)

        except Exception as e:
            logger.info(f"{name:<16s} | FAILED: {e}")

    logger.info("=" * 120)
    logger.info("Ratio = PyPTO_time / AscendC_time (越小越好，<1 表示 PyPTO 更快)")


if __name__ == "__main__":
    main()
