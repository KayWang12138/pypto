#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
"""
from typing import Callable, List, Literal, Union
import torch
import torch_npu


def do_bench(
    fn: Callable,
    warmup_iters: int = 5,
    benchmark_iters: int = 15,
    aggregation: Literal["mean", "none"] = "mean",
    unit: Literal["ms", "us", "ns"] = "us",
) -> Union[float, List[float]]:
    """
    Benchmark a given function with warmup.

    Args:
        fn: Function to benchmark.
        warmup_iters: Number of warmup runs.
        benchmark_iters: Number of benchmark runs.
        aggregation: Aggregation mode for benchmark times.
        unit: Time unit of the benchmarks.
    Returns:
        Runtime, or list of runtimes, in specified units.
    """
    start_events = [torch.npu.Event(enable_timing=True) for _ in range(benchmark_iters)]
    end_events = [torch.npu.Event(enable_timing=True) for _ in range(benchmark_iters)]

    # Allocate a 256 MB tensor which we write to every iteration to flush L2 cache
    # https://github.com/tile-ai/tilelang/blob/main/tilelang/profiler/bench.py#L103
    cache_size = 256 * 1024 * 1024
    cache = torch.empty((cache_size), dtype=torch.int8).npu()

    for _ in range(warmup_iters):
        # Make sure to sync in-between each call since the output tensor object
        # is reused for all calls. This prevents the deadlock-scenario: when two
        # kernels simultaneously manipulate the same output tensor object.
        torch_npu.npu.synchronize()
        fn()
    torch_npu.npu.synchronize()

    for i in range(benchmark_iters):
        cache.zero_()
        torch_npu.npu.synchronize()
        start_events[i].record()
        fn()
        end_events[i].record()
        torch_npu.npu.synchronize()

    f = {"ms": 1e0, "us": 1e3, "ns": 1e6}[unit]
    times = [f * s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    if aggregation == "mean":
        return sum(times) / len(times)
    return times
