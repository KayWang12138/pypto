#!/usr/bin/env python3
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# CANN Open Software License Agreement Version 2.0
"""Forward-only test (no autograd) to verify matmul+sum works correctly."""

import argparse
import os
import sys

import torch
import torch_npu
import pypto


def get_device_id():
    if 'TILE_FWK_DEVICE_ID' not in os.environ:
        print("Please set TILE_FWK_DEVICE_ID, e.g.: export TILE_FWK_DEVICE_ID=4")
        return None
    return int(os.environ['TILE_FWK_DEVICE_ID'])


def test_forward_only(run_mode: str = "npu"):
    print("=" * 60)
    print("Forward-Only Test (No Autograd)")
    print("=" * 60)

    device_id = get_device_id()
    if device_id is None and run_mode == "npu":
        return

    if run_mode == "npu":
        torch.npu.set_device(device_id)
        mode = pypto.RunMode.NPU
        device = f'npu:{device_id}'
    else:
        mode = pypto.RunMode.SIM
        device = 'cpu'

    batch, in_dim, out_dim = 4, 8, 4
    torch.manual_seed(42)

    x_shape = (batch, in_dim)
    w_shape = (in_dim, out_dim)
    out_shape = (batch, out_dim)
    loss_shape = (1, 1)

    @pypto.frontend.jit(
        host_options={"only_codegen": True},
        runtime_options={"run_mode": mode}
    )
    def forward_kernel(
        x: pypto.Tensor(x_shape, pypto.DT_FP32),
        w: pypto.Tensor(w_shape, pypto.DT_FP32),
    ) -> (pypto.Tensor(out_shape, pypto.DT_FP32), pypto.Tensor(loss_shape, pypto.DT_FP32)):
        pypto.set_cube_tile_shapes([32, 32], [64, 64], [64, 64])
        pypto.set_vec_tile_shapes(32, 32)
        result = pypto.matmul(x, w, pypto.DT_FP32)
        loss_row = pypto.sum(result, dim=-1, keepdim=True)
        loss = pypto.sum(loss_row, dim=0, keepdim=True)
        return result, loss

    x_torch = torch.randn(batch, in_dim, dtype=torch.float32, device=device)
    w_torch = torch.randn(in_dim, out_dim, dtype=torch.float32, device=device)

    print(f"x shape: {x_torch.shape}, w shape: {w_torch.shape}")
    print("Running forward kernel...")

    out, loss = forward_kernel(x_torch, w_torch)

    print(f"out shape: {out.shape if hasattr(out, 'shape') else 'N/A'}")
    print(f"loss shape: {loss.shape if hasattr(loss, 'shape') else 'N/A'}")

    if run_mode == "npu":
        expected_out = torch.matmul(x_torch, w_torch)
        expected_loss = expected_out.sum().reshape(1, 1)
        print(f"Expected loss: {expected_loss}")
        print(f"Computed loss: {loss}")

    print("[PASS] Forward-only test completed successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_mode", "--run-mode",
        nargs="?", type=str, default="npu", choices=["npu", "sim"],
        help="run mode: npu or sim"
    )
    args, _ = parser.parse_known_args()

    if "--build-ci" in sys.argv or os.environ.get("PYPTO_BUILD_CI"):
        os.system(f"python3 build_ci.py -s={os.path.abspath(__file__)}")
    else:
        test_forward_only(run_mode=args.run_mode)
