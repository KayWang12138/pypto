#!/usr/bin/env python3
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# CANN Open Software License Agreement Version 2.0
"""Simple MLP training example with C++ AutodiffPass."""

import argparse
import os
import sys

import torch
import torch_npu
import pypto


def init_device(device_id: int = 4):
    """Initialize NPU device."""
    torch.npu.set_device(device_id)


def simple_forward(x: pypto.Tensor, w: pypto.Tensor, out: pypto.Tensor, loss: pypto.Tensor):
    pypto.set_cube_tile_shapes([32, 32], [64, 64], [64, 64])
    pypto.set_vec_tile_shapes(32, 32)
    result = pypto.matmul(x, w, pypto.DT_FP32)
    out[:] = result
    loss_row = pypto.sum(out, dim=-1, keepdim=True)
    loss[:] = pypto.sum(loss_row, dim=0, keepdim=True)


def build_train_kernel(compile_only: bool):
    @pypto.jit(host_options={"only_codegen": compile_only})
    def train_kernel(x: pypto.Tensor, w: pypto.Tensor, out: pypto.Tensor, loss: pypto.Tensor,
                     grad_x: pypto.Tensor, grad_w: pypto.Tensor):
        simple_forward(x, w, out, loss)
    return train_kernel


def test_training(compile_only: bool = True):
    print("=" * 60)
    print("Simple MLP Training Test")
    print("=" * 60)

    init_device(4)

    batch, in_dim, out_dim = 4, 8, 4
    torch.manual_seed(42)

    x = pypto.from_torch(torch.randn(batch, in_dim, dtype=torch.float32))
    w = pypto.from_torch(torch.randn(in_dim, out_dim, dtype=torch.float32))
    out = pypto.from_torch(torch.zeros(batch, out_dim, dtype=torch.float32))
    loss = pypto.from_torch(torch.zeros(1, 1, dtype=torch.float32))
    grad_x = pypto.from_torch(torch.zeros(batch, in_dim, dtype=torch.float32))
    grad_w = pypto.from_torch(torch.zeros(in_dim, out_dim, dtype=torch.float32))

    x.requires_grad = True
    w.requires_grad = True
    loss.is_loss = True

    print(f"x: {x.shape}, w: {w.shape}")
    print(f"x.requires_grad={x.requires_grad}, loss.is_loss={loss.is_loss}")

    kernel = build_train_kernel(compile_only)
    kernel(x, w, out, loss, grad_x, grad_w)

    x_grad_info = x.get_gradient_info()
    w_grad_info = w.get_gradient_info()
    print(f"x grad_info: {x_grad_info}")
    print(f"w grad_info: {w_grad_info}")

    status = "[PASS]" if x_grad_info and w_grad_info else "[INFO] Check AutodiffPass"
    print(f"\n{status}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", help="Execute kernel (default: compile-only)")
    args, _ = parser.parse_known_args()

    if "--build-ci" in sys.argv or os.environ.get("PYPTO_BUILD_CI"):
        os.system(f"python3 build_ci.py -s={os.path.abspath(__file__)}")
    else:
        test_training(compile_only=not args.run)
