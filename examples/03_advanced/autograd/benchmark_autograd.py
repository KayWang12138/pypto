#!/usr/bin/env python3
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# CANN Open Software License Agreement Version 2.0
"""Autograd performance benchmark."""

import os
import sys
import time
from typing import List

try:
    import pypto
except ImportError:
    print("PyPTO not found. Install with: pip install -e .")
    sys.exit(1)

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def calc_matmul_flops(m: int, k: int, n: int) -> int:
    return 2 * m * k * n


def format_time(s: float) -> str:
    if s < 1e-6:
        return f"{s*1e9:.2f}ns"
    if s < 1e-3:
        return f"{s*1e6:.2f}us"
    if s < 1:
        return f"{s*1e3:.2f}ms"
    return f"{s:.2f}s"


def format_flops(f: float) -> str:
    if f >= 1e12:
        return f"{f/1e12:.2f}TFLOP/s"
    if f >= 1e9:
        return f"{f/1e9:.2f}GFLOP/s"
    return f"{f/1e6:.2f}MFLOP/s"


def run_kernel(kernel, args, warmup: int, iters: int) -> float:
    kernel(*args)
    for _ in range(warmup):
        kernel(*args)
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        kernel(*args)
        times.append(time.perf_counter() - t0)
    return sum(times) / len(times)


class Result:
    def __init__(self, name: str, fwd: float, bwd: float, total: float, batch: int, flops: int):
        self.name, self.fwd, self.bwd, self.total = name, fwd, bwd, total
        self.batch, self.flops = batch, flops

    @property
    def throughput(self) -> float:
        return self.batch / self.total if self.total > 0 else 0

    @property
    def achieved_flops(self) -> float:
        return self.flops / self.total if self.total > 0 else 0

    def __str__(self) -> str:
        return (f"{self.name}: fwd={format_time(self.fwd)}, bwd={format_time(self.bwd)}, "
                f"total={format_time(self.total)}, {self.throughput:.1f}samples/s, "
                f"{format_flops(self.achieved_flops)}")


def benchmark_linear(batch: int = 32, in_dim: int = 512, out_dim: int = 256,
                     warmup: int = 3, iters: int = 10) -> Result:
    if not HAS_TORCH:
        return None

    print(f"\nLinear[{batch}x{in_dim}x{out_dim}]")

    def make_tensors(grad: bool, loss: bool):
        x = pypto.from_torch(torch.randn(batch, in_dim, dtype=torch.float32), "x")
        w = pypto.from_torch(torch.randn(in_dim, out_dim, dtype=torch.float32), "w")
        out = pypto.from_torch(torch.zeros(batch, out_dim, dtype=torch.float32), "out")
        l = pypto.from_torch(torch.zeros(1, dtype=torch.float32), "loss")
        if grad:
            x.requires_grad = w.requires_grad = True
        if loss:
            l.is_loss = True
        return x, w, out, l

    @pypto.jit
    def linear_fwd(x: pypto.Tensor, w: pypto.Tensor, out: pypto.Tensor, loss: pypto.Tensor):
        r = pypto.matmul(x, w)
        out[:] = r
        l = pypto.sum(pypto.sum(r, dim=-1, keepdim=True), dim=0, keepdim=True)
        loss[:] = l

    fwd_avg = run_kernel(linear_fwd, make_tensors(False, False), warmup, iters)
    total_avg = run_kernel(linear_fwd, make_tensors(True, True), warmup, iters)
    bwd_avg = max(total_avg - fwd_avg, 0.0)

    fwd_flops = calc_matmul_flops(batch, in_dim, out_dim)
    total_flops = fwd_flops * 3  # forward + 2 backward matmuls

    result = Result(f"Linear[{batch}x{in_dim}x{out_dim}]", fwd_avg, bwd_avg, total_avg, batch, total_flops)
    print(f"  {result}")
    return result


def benchmark_mlp(batch: int = 32, dims: List[int] = [512, 256, 128],
                  warmup: int = 3, iters: int = 10) -> Result:
    if not HAS_TORCH or len(dims) < 3:
        return None

    print(f"\nMLP[{'-'.join(map(str, dims))}]")

    def make_tensors(grad: bool, loss: bool):
        x = pypto.from_torch(torch.randn(batch, dims[0], dtype=torch.float32), "x")
        w0 = pypto.from_torch(torch.randn(dims[0], dims[1], dtype=torch.float32), "w0")
        w1 = pypto.from_torch(torch.randn(dims[1], dims[2], dtype=torch.float32), "w1")
        out = pypto.from_torch(torch.zeros(batch, dims[2], dtype=torch.float32), "out")
        l = pypto.from_torch(torch.zeros(1, dtype=torch.float32), "loss")
        if grad:
            x.requires_grad = w0.requires_grad = w1.requires_grad = True
        if loss:
            l.is_loss = True
        return x, w0, w1, out, l

    @pypto.jit
    def mlp_fwd(x: pypto.Tensor, w0: pypto.Tensor, w1: pypto.Tensor,
                out: pypto.Tensor, loss: pypto.Tensor):
        h = pypto.matmul(pypto.matmul(x, w0), w1)
        out[:] = h
        l = pypto.sum(pypto.sum(h, dim=-1, keepdim=True), dim=0, keepdim=True)
        loss[:] = l

    fwd_avg = run_kernel(mlp_fwd, make_tensors(False, False), warmup, iters)
    total_avg = run_kernel(mlp_fwd, make_tensors(True, True), warmup, iters)
    bwd_avg = max(total_avg - fwd_avg, 0.0)

    total_flops = sum(calc_matmul_flops(batch, dims[i], dims[i+1]) * 3 for i in range(len(dims)-1))

    result = Result(f"MLP[{'-'.join(map(str, dims))}]", fwd_avg, bwd_avg, total_avg, batch, total_flops)
    print(f"  {result}")
    return result


def main():
    if not HAS_TORCH:
        print("torch required")
        sys.exit(1)

    print("=" * 60)
    print("PyPTO Autograd Benchmark")
    print("=" * 60)
    torch.manual_seed(42)

    results = [
        benchmark_linear(32, 256, 128),
        benchmark_linear(64, 512, 256),
        benchmark_linear(128, 1024, 512),
        benchmark_mlp(32, [512, 256, 128]),
        benchmark_mlp(64, [1024, 512, 256]),
    ]

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for r in results:
        if r:
            print(r)


if __name__ == "__main__":
    if "--build-ci" in sys.argv or os.environ.get("PYPTO_BUILD_CI"):
        os.system(f"python3 build_ci.py -s={os.path.abspath(__file__)}")
    else:
        main()
