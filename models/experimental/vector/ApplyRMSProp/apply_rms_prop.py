#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import os
import sys
import argparse
import numpy as np
import torch
import torch_npu

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from apply_rms_prop_impl import apply_rms_prop_kernel


def apply_rms_prop_golden(var, ms, mom, lr, rho, momentum, epsilon, grad):
    """NumPy reference implementation of ApplyRMSProp."""
    var = var.copy()
    ms = ms.copy()
    mom = mom.copy()

    grad_sq = grad**2
    ms = ms + (grad_sq - ms) * (1.0 - rho)
    mom = mom * momentum + (grad * lr) / np.sqrt(ms + epsilon)
    var = var - mom

    return var, ms, mom


def test_apply_rms_prop_level0(device_id=None, run_mode: str = "npu"):
    """Test ApplyRMSProp with small input (Level 0)."""
    device = (
        f"npu:{device_id}" if (run_mode == "npu" and device_id is not None) else "cpu"
    )

    rows, cols = 8, 8
    shape = (rows, cols)

    var_torch = torch.rand(shape, dtype=torch.float32, device=device) * 1.0
    ms_torch = torch.rand(shape, dtype=torch.float32, device=device) * 1.0
    mom_torch = torch.randn(shape, dtype=torch.float32, device=device) * 2.0 - 1.0
    grad_torch = torch.randn(shape, dtype=torch.float32, device=device) * 2.0 - 1.0

    var_out = torch.empty(shape, dtype=torch.float32, device=device)
    ms_out = torch.empty(shape, dtype=torch.float32, device=device)
    mom_out = torch.empty(shape, dtype=torch.float32, device=device)

    lr = 0.001
    rho = 0.9
    momentum = 0.9
    epsilon = 1e-7

    apply_rms_prop_kernel(
        var_torch,
        ms_torch,
        mom_torch,
        grad_torch,
        var_out,
        ms_out,
        mom_out,
        lr,
        rho,
        momentum,
        epsilon,
    )

    var_np = var_torch.cpu().numpy()
    ms_np = ms_torch.cpu().numpy()
    mom_np = mom_torch.cpu().numpy()
    grad_np = grad_torch.cpu().numpy()

    expected_var, expected_ms, expected_mom = apply_rms_prop_golden(
        var_np, ms_np, mom_np, lr, rho, momentum, epsilon, grad_np
    )

    var_diff = np.abs(var_out.cpu().numpy() - expected_var).max()
    ms_diff = np.abs(ms_out.cpu().numpy() - expected_ms).max()
    mom_diff = np.abs(mom_out.cpu().numpy() - expected_mom).max()

    atol = 1e-5
    rtol = 1e-5
    assert var_diff < atol + rtol * np.abs(expected_var).max(), (
        f"var mismatch: {var_diff}"
    )
    assert ms_diff < atol + rtol * np.abs(expected_ms).max(), f"ms mismatch: {ms_diff}"
    assert mom_diff < atol + rtol * np.abs(expected_mom).max(), (
        f"mom mismatch: {mom_diff}"
    )
    print(
        f"level0: shape={shape}, var_diff={var_diff:.8f}, "
        f"ms_diff={ms_diff:.8f}, mom_diff={mom_diff:.8f}"
    )


def test_apply_rms_prop_level1(device_id=None, run_mode: str = "npu"):
    """Test ApplyRMSProp with typical input (Level 1)."""
    device = (
        f"npu:{device_id}" if (run_mode == "npu" and device_id is not None) else "cpu"
    )

    rows, cols = 1024, 1024
    shape = (rows, cols)

    var_torch = torch.rand(shape, dtype=torch.float32, device=device) * 1.0
    ms_torch = torch.rand(shape, dtype=torch.float32, device=device) * 1.0
    mom_torch = torch.randn(shape, dtype=torch.float32, device=device) * 2.0 - 1.0
    grad_torch = torch.randn(shape, dtype=torch.float32, device=device) * 2.0 - 1.0

    var_out = torch.empty(shape, dtype=torch.float32, device=device)
    ms_out = torch.empty(shape, dtype=torch.float32, device=device)
    mom_out = torch.empty(shape, dtype=torch.float32, device=device)

    lr = 0.001
    rho = 0.9
    momentum = 0.9
    epsilon = 1e-7

    apply_rms_prop_kernel(
        var_torch,
        ms_torch,
        mom_torch,
        grad_torch,
        var_out,
        ms_out,
        mom_out,
        lr,
        rho,
        momentum,
        epsilon,
    )

    var_np = var_torch.cpu().numpy()
    ms_np = ms_torch.cpu().numpy()
    mom_np = mom_torch.cpu().numpy()
    grad_np = grad_torch.cpu().numpy()

    expected_var, expected_ms, expected_mom = apply_rms_prop_golden(
        var_np, ms_np, mom_np, lr, rho, momentum, epsilon, grad_np
    )

    var_diff = np.abs(var_out.cpu().numpy() - expected_var).max()
    ms_diff = np.abs(ms_out.cpu().numpy() - expected_ms).max()
    mom_diff = np.abs(mom_out.cpu().numpy() - expected_mom).max()

    atol = 1e-5
    rtol = 1e-5
    assert var_diff < atol + rtol * np.abs(expected_var).max(), (
        f"var mismatch: {var_diff}"
    )
    assert ms_diff < atol + rtol * np.abs(expected_ms).max(), f"ms mismatch: {ms_diff}"
    assert mom_diff < atol + rtol * np.abs(expected_mom).max(), (
        f"mom mismatch: {mom_diff}"
    )
    print(
        f"level1: shape={shape}, var_diff={var_diff:.8f}, "
        f"ms_diff={ms_diff:.8f}, mom_diff={mom_diff:.8f}"
    )


def test_apply_rms_prop_level2(device_id=None, run_mode: str = "npu"):
    """Test ApplyRMSProp with boundary values (Level 2)."""
    device = (
        f"npu:{device_id}" if (run_mode == "npu" and device_id is not None) else "cpu"
    )

    rows, cols = 16, 16
    shape = (rows, cols)

    var_torch = torch.rand(shape, dtype=torch.float32, device=device) * 1.0
    ms_torch = torch.rand(shape, dtype=torch.float32, device=device) * 1.0
    mom_torch = torch.randn(shape, dtype=torch.float32, device=device) * 2.0 - 1.0
    grad_torch = torch.randn(shape, dtype=torch.float32, device=device) * 2.0 - 1.0

    var_out = torch.empty(shape, dtype=torch.float32, device=device)
    ms_out = torch.empty(shape, dtype=torch.float32, device=device)
    mom_out = torch.empty(shape, dtype=torch.float32, device=device)

    lr = 0.001
    rho = 0.9
    momentum = 0.9
    epsilon = 1e-7

    apply_rms_prop_kernel(
        var_torch,
        ms_torch,
        mom_torch,
        grad_torch,
        var_out,
        ms_out,
        mom_out,
        lr,
        rho,
        momentum,
        epsilon,
    )

    var_np = var_torch.cpu().numpy()
    ms_np = ms_torch.cpu().numpy()
    mom_np = mom_torch.cpu().numpy()
    grad_np = grad_torch.cpu().numpy()

    expected_var, expected_ms, expected_mom = apply_rms_prop_golden(
        var_np, ms_np, mom_np, lr, rho, momentum, epsilon, grad_np
    )

    var_diff = np.abs(var_out.cpu().numpy() - expected_var).max()
    ms_diff = np.abs(ms_out.cpu().numpy() - expected_ms).max()
    mom_diff = np.abs(mom_out.cpu().numpy() - expected_mom).max()

    atol = 1e-5
    rtol = 1e-5
    assert var_diff < atol + rtol * np.abs(expected_var).max(), (
        f"var mismatch: {var_diff}"
    )
    assert ms_diff < atol + rtol * np.abs(expected_ms).max(), f"ms mismatch: {ms_diff}"
    assert mom_diff < atol + rtol * np.abs(expected_mom).max(), (
        f"mom mismatch: {mom_diff}"
    )
    print(
        f"level2: shape={shape}, var_diff={var_diff:.8f}, "
        f"ms_diff={ms_diff:.8f}, mom_diff={mom_diff:.8f}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="PyPTO ApplyRMSProp Tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s              Run all tests
  %(prog)s level0       Run level 0 test only
  %(prog)s --list       List all available tests
        """,
    )
    parser.add_argument(
        "test_id",
        type=str,
        nargs="?",
        help="Test ID to run. If not specified, all tests will run.",
    )
    parser.add_argument(
        "--list", action="store_true", help="List all available tests and exit"
    )
    parser.add_argument(
        "--run_mode",
        type=str,
        nargs="?",
        default="npu",
        choices=["npu", "sim"],
        help="Run mode, such as npu/sim etc.",
    )

    args = parser.parse_args()

    tests = {
        "level0": {
            "name": "Level 0 (8x8)",
            "description": "Small input test",
            "function": test_apply_rms_prop_level0,
        },
        "level1": {
            "name": "Level 1 (1024x1024)",
            "description": "Typical input test",
            "function": test_apply_rms_prop_level1,
        },
        "level2": {
            "name": "Level 2 (boundary)",
            "description": "Boundary values test",
            "function": test_apply_rms_prop_level2,
        },
    }

    if args.list:
        print("\n" + "=" * 60)
        print("Available Tests")
        print("=" * 60 + "\n")
        for test_id, test_info in sorted(tests.items()):
            print(f"  ID: {test_id}")
            print(f"    name: {test_info['name']}")
            print(f"    description: {test_info['description']}\n")
        return

    if args.test_id is not None:
        if args.test_id not in tests:
            print(f"ERROR: Invalid test ID: {args.test_id}")
            print(f"Valid test IDs are: {', '.join(map(str, sorted(tests.keys())))}")
            print("\nUse --list to see all available tests.")
            sys.exit(1)

    device_id = None
    tests_to_run = []

    if args.test_id is not None:
        test = tests.get(args.test_id)
        if test is None:
            raise ValueError(f"Invalid test ID: {args.test_id}")
        tests_to_run = [(args.test_id, test)]
    else:
        tests_to_run = list(tests.items())

    if args.run_mode == "npu":
        if "TILE_FWK_DEVICE_ID" not in os.environ:
            print("ERROR: TILE_FWK_DEVICE_ID not set")
            print("Please set it before running:")
            print("  export TILE_FWK_DEVICE_ID=0")
            sys.exit(1)

        device_id = int(os.environ["TILE_FWK_DEVICE_ID"])
        torch.npu.set_device(device_id)

    try:
        for test_id, test_info in tests_to_run:
            test_info["function"](device_id, args.run_mode)

        if len(tests_to_run) > 1:
            print("All ApplyRMSProp tests passed!")

    except Exception as e:
        print(f"\nError: {e}")
        raise


if __name__ == "__main__":
    main()
