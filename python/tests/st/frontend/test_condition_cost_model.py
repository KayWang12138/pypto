#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025-2026 CANN community contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import os
import sys
import logging
import argparse
import torch
import pypto
import numpy as np
from numpy.testing import assert_allclose

logging.basicConfig(level=logging.INFO, format='%(message)s', force=True)


def get_device_id():
    """
    Get and validate TILE_FWK_DEVICE_ID from environment variable.

    Returns:
        int: The device ID if valid, None otherwise.
    """
    if 'TILE_FWK_DEVICE_ID' not in os.environ:
        logging.error("ERROR: Environment variable TILE_FWK_DEVICE_ID is not set.")
        logging.error("Please set it before running this example:")
        logging.error("  export TILE_FWK_DEVICE_ID=0")
        return None

    try:
        device_id = int(os.environ['TILE_FWK_DEVICE_ID'])
        return device_id
    except ValueError:
        logging.error(f"ERROR: TILE_FWK_DEVICE_ID must be an integer, got: {os.environ['TILE_FWK_DEVICE_ID']}")
        return None


SHAPE = (2, 2)


def nested_loops_with_conditions_kernel(mode: str = "npu"):
    if mode == "npu":
        mode = pypto.RunMode.NPU
    elif mode == "sim":
        mode = pypto.RunMode.SIM
    else:
        raise ValueError(f"Invalid mode: {mode}.")

    @pypto.frontend.jit(runtime_options={"run_mode": mode})
    def kernel(
        a: pypto.Tensor(SHAPE, pypto.DT_FP32),
        b: pypto.Tensor(SHAPE, pypto.DT_FP32),
    ) -> pypto.Tensor(SHAPE, pypto.DT_FP32):
        pypto.set_vec_tile_shapes(2, 8)
        y = pypto.tensor(SHAPE, pypto.DT_FP32)
        for i in pypto.loop(2):
            for j in pypto.loop(2):
                a_view = a[i:i + 1, j:j + 1]
                b_view = b[i:i + 1, j:j + 1]
                if i == 0:
                    y[i:i + 1, j:j + 1] = a_view + b_view
                else:
                    y[i:i + 1, j:j + 1] = a_view - b_view
        return y
    return kernel


def test_nested_loops_with_conditions(device_id=None, run_mode: str = "npu") -> None:
    """Test nested loops with conditional statements"""
    logging.info("=" * 60)
    logging.info("Test: Nested Loops with Conditional Statements")
    logging.info("=" * 60)

    if not device_id:
        device_id = torch.npu.current_device()
    else:
        torch.npu.set_device(device_id)

    if run_mode == "npu":
        import torch_npu
        device = f'npu:{device_id}'
    else:
        device = 'cpu'

    shape = SHAPE
    dtype = torch.float32
    a = torch.rand(shape, dtype=dtype, device=device)
    b = torch.rand(shape, dtype=dtype, device=device)

    if run_mode == 'npu':
        y = nested_loops_with_conditions_kernel(run_mode)(a, b)
    else:
        y = nested_loops_with_conditions_kernel(run_mode)(a, b)

    y = y.cpu()
    golden = torch.zeros(shape, dtype=dtype, device=device)
    golden[0] = a[0] + b[0]
    golden[1] = a[1] - b[1]
    golden = golden.cpu()

    if run_mode == "npu":
        assert_allclose(np.array(y), np.array(golden), rtol=1e-3, atol=1e-3)
        logging.info(f"Output: {y}")
        logging.info(f"Expected: {golden}")
    logging.info("✓ Nested loops with conditional statements completed successfully")


def main():
    """Run condition examples.

    Usage:
        python condition_example.py          # Run all examples
        python condition_example.py 1         # Run example 1 only
        python condition_example.py --list   # List all available examples
    """
    parser = argparse.ArgumentParser(
        description="PyPTO Condition Function Examples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s              Run all examples
  %(prog)s nested_loops_with_conditions::test_nested_loops_with_conditions
            Run example nested_loops_with_conditions::test_nested_loops_with_conditions
  %(prog)s --list       List all available examples
        """
    )
    parser.add_argument(
        'example_id',
        type=str,
        nargs='?',
        help='Example ID to run (1-4). If not specified, all examples will run.'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List all available examples and exit'
    )
    parser.add_argument(
        '--run_mode',
        type=str,
        nargs='?',
        default="npu",
        choices=["npu", "sim"],
        help='Run mode, such as npu/sim etc.'
    )

    args = parser.parse_args()

    # Define available examples
    examples = {
        'nested_loops_with_conditions::test_nested_loops_with_conditions': {
            'name': 'Test nested loops with conditional statements',
            'description': 'Nested loops with conditional statements example',
            'function': test_nested_loops_with_conditions,
            'requires_npu': True
        },
    }

    # List examples if requested
    if args.list:
        logging.info("\n" + "=" * 60)
        logging.info("Available Examples")
        logging.info("=" * 60 + "\n")
        for ex_id, ex_info in sorted(examples.items()):
            logging.info(f"  ID: {ex_id}")
            logging.info(f"     name: {ex_info['name']}")
            logging.info(f"     description: {ex_info['description']}\n")
        return

    # Validate example ID if provided
    if args.example_id is not None:
        if args.example_id not in examples:
            logging.error(f"ERROR: Invalid example ID: {args.example_id}")
            logging.error(f"Valid example IDs are: {', '.join(map(str, sorted(examples.keys())))}")
            logging.error("\nUse --list to see all available examples.")
            sys.exit(1)

    logging.info("\n" + "=" * 60)
    logging.info("PyPTO Condition Function Examples")
    logging.info("=" * 60 + "\n")

    # Get and validate device ID (needed for NPU examples)
    device_id = None
    examples_to_run = []

    if args.example_id is not None:
        # Run single example
        example = examples.get(args.example_id)
        if example is None:
            raise ValueError(f"Invalid example ID: {args.example_id}")
        examples_to_run = [(args.example_id, example)]
    else:
        # Run all examples
        examples_to_run = list(examples.items())

    if args.run_mode == "npu":
        device_id = get_device_id()
        if device_id is None:
            return
        torch.npu.set_device(device_id)
        logging.info("Running examples that require NPU hardware...")
        logging.info("(Make sure CANN environment is configured and NPU is available)\n")

    try:
        for ex_id, ex_info in examples_to_run:
            logging.info(f"Running Example {ex_id}: {ex_info['name']}")
            ex_info['function'](device_id, args.run_mode)

        if len(examples_to_run) > 1:
            logging.info("=" * 60)
            logging.info("All condition tests passed!")
            logging.info("=" * 60)

    except Exception as e:
        logging.error(f"\nError: {e}")
        raise


if __name__ == "__main__":
    main()
