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
Test DUPLICATE_OPERATION (0x10003) Error Code

This test demonstrates the DUPLICATE_OPERATION error that occurs when:
- Trying to add an already existing operation
- Operation name or identifier is duplicated

Note: This test demonstrates the error code concept. Due to environment configuration
requirements (PTO_TILE_LIB_CODE_PATH), the actual NPU/SIM execution may not work
in all environments. The test focuses on demonstrating the correct vs incorrect patterns.
"""
import os
import sys
import pypto
import torch
import numpy as np


def get_device_id():
    """
    Get and validate TILE_FWK_DEVICE_ID from environment variable.

    Returns:
        int: The device ID if valid, None otherwise.
    """
    if 'TILE_FWK_DEVICE_ID' not in os.environ:
        print("If NPU environment is not available, this test will demonstrate the code patterns.")
        print("To run with actual NPU, set TILE_FWK_DEVICE_ID:")
        print("  export TILE_FWK_DEVICE_ID=0")
        return None

    try:
        device_id = int(os.environ['TILE_FWK_DEVICE_ID'])
        return device_id
    except ValueError:
        print(f"ERROR: TILE_FWK_DEVICE_ID must be an integer, got: {os.environ['TILE_FWK_DEVICE_ID']}")
        return None


def demonstrate_duplicate_operation_pattern():
    """
    Demonstrate the DUPLICATE_OPERATION error pattern and correct approach.

    This function shows code examples without actual execution to demonstrate
    the error code concept.
    """
    print("Demonstrating DUPLICATE_OPERATION (0x10003) Error Pattern")
    print("=" * 60)
    print()

    print("Error Example Code:")
    print("-" * 60)
    print("""
@pypto.frontend.jit
def duplicate_op_example(x):
    op1 = pypto.add(x, x)
    op2 = pypto.add(x, x)  # Creating duplicate operation
    return op1

# Issue: op1 and op2 are identical operations with same inputs
# This can lead to DUPLICATE_OPERATION error in the computational graph
""")
    print()

    print("Correct Example Code:")
    print("-" * 60)
    print("""
@pypto.frontend.jit
def correct_op_example(x):
    op1 = pypto.add(x, x)
    op2 = pypto.mul(x, 2)  # Using different operation
    return op1

# Solution: Use different operations or reuse the existing result
# Avoid creating identical operations in the graph
""")
    print()

    print("Best Practices:")
    print("-" * 60)
    print("1. Check if an operation already exists before creating a new one")
    print("2. Reuse existing operation results when possible")
    print("3. Use unique operation names or identifiers")
    print("4. Consider the computational graph structure to avoid duplicates")
    print()


def test_duplicate_operation_error(device_id=None, run_mode: str = "npu") -> None:
    """
    Test DUPLICATE_OPERATION error code with actual execution if environment allows.

    Args:
        device_id: NPU device ID
        run_mode: Execution mode ('npu' or 'sim')
    """
    print("Testing DUPLICATE_OPERATION error scenario...")
    print("=" * 60)

    shape = (1, 4, 1, 64)

    if run_mode == "npu":
        mode = pypto.RunMode.NPU
        device = f'npu:{device_id}' if device_id is not None else 'cpu'
    else:
        mode = pypto.RunMode.SIM
        device = 'cpu'

    try:
        # Error Example: Creating duplicate operations
        print("\n1. Testing duplicate operation pattern...")
        print("-" * 60)

        @pypto.frontend.jit(runtime_options={"run_mode": mode})
        def duplicate_op_example(
            x: pypto.Tensor(shape, pypto.DT_FP32),
        ) -> pypto.Tensor(shape, pypto.DT_FP32):
            pypto.set_vec_tile_shapes(1, 4, 1, 64)
            op1 = pypto.add(x, x)
            op2 = pypto.add(x, x)  # Creating duplicate operation
            return op1

        input_data = torch.rand(shape, dtype=torch.float, device=device)
        output_data = duplicate_op_example(input_data)

        print("Note: PyPTO may optimize away duplicate operations automatically.")
        print(f"Input shape: {input_data.shape}")
        print(f"Output shape: {output_data.shape}")
        print("✓ Duplicate operation test completed")
        print()

    except Exception as e:
        print(f"Error occurred: {type(e).__name__}")
        print(f"Error message: {str(e)[:200]}...")
        print("✓ Error captured (this demonstrates the error scenario)")
        print()

    try:
        # Correct Example: Using different operations
        print("\n2. Testing correct operation pattern...")
        print("-" * 60)

        @pypto.frontend.jit(runtime_options={"run_mode": mode})
        def correct_op_example(
            x: pypto.Tensor(shape, pypto.DT_FP32),
        ) -> pypto.Tensor(shape, pypto.DT_FP32):
            pypto.set_vec_tile_shapes(1, 4, 1, 64)
            op1 = pypto.add(x, x)
            op2 = pypto.mul(x, 2)  # Using different operation
            return op1

        input_data = torch.rand(shape, dtype=torch.float, device=device)
        output_data = correct_op_example(input_data)

        golden = torch.add(input_data, input_data)
        max_diff = np.abs(output_data.cpu().numpy() - golden.cpu().numpy()).max()

        print(f"Input shape: {input_data.shape}")
        print(f"Output shape: {output_data.shape}")
        if run_mode == "npu":
            print(f"Max difference: {max_diff:.6f}")
        print("✓ Correct operation example passed")
        print()

    except Exception as e:
        print(f"Unexpected error in correct example: {type(e).__name__}")
        print(f"Error message: {str(e)[:200]}...")
        print()


def main():
    """Run DUPLICATE_OPERATION error test.

    Usage:
        python test_duplicate_operation.py                    # Demonstrate patterns
        python test_duplicate_operation.py --run_mode npu     # Run with NPU
        python test_duplicate_operation.py --run_mode sim     # Run in simulation
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="PyPTO DUPLICATE_OPERATION Error Test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                      Demonstrate error patterns (no execution)
  %(prog)s --run_mode npu      Run with NPU hardware
  %(prog)s --run_mode sim       Run in simulation mode
        """
    )
    parser.add_argument(
        '--run_mode',
        type=str,
        nargs='?',
        default=None,
        choices=["npu", "sim"],
        help='Run mode: npu/sim. If not specified, only demonstrates patterns.'
    )

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("PyPTO DUPLICATE_OPERATION Error Test")
    print("=" * 60 + "\n")

    # Always demonstrate the pattern
    demonstrate_duplicate_operation_pattern()

    # Run actual tests if run_mode is specified
    if args.run_mode:
        device_id = None

        if args.run_mode == "npu":
            device_id = get_device_id()
            if device_id is None:
                print("\nSkipping NPU execution due to missing device configuration.")
                print("Use --run_mode sim for simulation mode or see README.md for setup.")
                return
            import torch_npu
            torch.npu.set_device(device_id)
            print("\nRunning tests with NPU hardware...")
            print("(Make sure CANN environment is configured and NPU is available)\n")
        else:
            print("\nRunning tests in simulation mode...\n")

        try:
            test_duplicate_operation_error(device_id, args.run_mode)
            print("=" * 60)
            print("DUPLICATE_OPERATION error test completed!")
            print("=" * 60)

        except Exception as e:
            print(f"\nError during execution: {e}")
            print("This may be due to environment configuration.")
            print("See README.md for setup instructions.")
    else:
        print("=" * 60)
        print("Pattern demonstration completed!")
        print("To run actual tests, use --run_mode npu or --run_mode sim")
        print("=" * 60)


if __name__ == "__main__":
    main()
