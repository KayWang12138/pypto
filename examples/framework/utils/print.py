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
set_print_options Example for PyPTO

This example demonstrates how to configure PyPTO runtime debug printing
using `set_print_options`.

Note: `set_print_options` only affects PyPTO runtime and IR-level debug
output, and does not change Python-level printing behavior of tensors.
"""

import os
import sys
import argparse
import pypto
import torch
import numpy as np
from numpy.testing import assert_allclose

# ----------------------------------------------------------------------------
# Device Utilities
# ----------------------------------------------------------------------------

def get_device_id():
    """
    Get and validate TILE_FWK_DEVICE_ID from environment variable.
    
    Returns:
        int: The device ID if valid, None otherwise.
    """
    if 'TILE_FWK_DEVICE_ID' not in os.environ:
        print("If no NPU environment is available, set --run_mode sim to run in simulation mode;")
        print("otherwise, set the environment variable TILE_FWK_DEVICE_ID.")
        print("Please set it before running this example:")
        print("  export TILE_FWK_DEVICE_ID=0")
        return None
    
    try:
        device_id = int(os.environ['TILE_FWK_DEVICE_ID'])
        return device_id
    except ValueError:
        print(f"ERROR: TILE_FWK_DEVICE_ID must be an integer, got: {os.environ['TILE_FWK_DEVICE_ID']}")
        return None

# ============================================================================
# set_print_options Examples
# ============================================================================

def test_set_print_options(device_id=None)->None:
    """
    Test configuring PyPTO runtime print options.

    Note:
    - set_print_options configures PyPTO runtime debug printing.
    - It does NOT affect Python-level printing of torch.Tensor.
    - pypto.Tensor does not expose numeric __repr__ in Python.
    """
    if device_id is None:
        device_id = torch.npu.current_device()
    else:
        torch.npu.set_device(device_id)
        
    precision = 3
    threshold = 10
    linewidth = 80

    print("Configuring PyPTO print options:")
    print(f"  precision  : {precision}")
    print(f"  threshold  : {threshold}")
    print(f"  linewidth  : {linewidth}")

    pypto.set_print_options(
        precision=precision,
        threshold=threshold,
        linewidth=linewidth,
    )

    print("\n✓ PyPTO print options configured successfully")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="PyPTO set print options Example",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s              Run all examples
  %(prog)s --list       List available examples
  %(prog)s print::test_set_print_options
            Run print::test_set_print_options Example
        """
    )
    parser.add_argument(
        "example_id",
        type=str,
        nargs="?",
        help="Example ID to run (1 or 2). If omitted, all examples run."
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available examples and exit"
    )

    args = parser.parse_args()

    examples = {
        'print::test_set_print_options': {
            "name": "Set print options",
            "description": "Test configuring PyPTO runtime print options",
            "function": test_set_print_options,
            "requires_npu": True
        }
    }

    # List examples if requested
    if args.list:
        print("\n" + "=" * 60)
        print("Available Examples")
        print("=" * 60 + "\n")
        for ex_id, ex_info in sorted(examples.items()):
            npu_req = " (Requires NPU)" if ex_info['requires_npu'] else " (No NPU required)"
            print(f"  {ex_id}. {ex_info['name']}{npu_req}")
            print(f"     {ex_info['description']}\n")
        return
    
    # Validate example ID if provided
    if args.example_id is not None:
        if args.example_id not in examples:
            print(f"ERROR: Invalid example ID: {args.example_id}")
            print(f"Valid example IDs are: {', '.join(map(str, sorted(examples.keys())))}")
            print("\nUse --list to see all available examples.")
            sys.exit(1)
    
    print("\n" + "=" * 60)
    print("PyPTO set print options Example")
    print("=" * 60 + "\n")
    
    # Get and validate device ID (needed for NPU examples)
    device_id = None
    examples_to_run = []
    
    if args.example_id is not None:
        # Run single example
        examples_to_run = [(args.example_id, examples[args.example_id])]
    else:
        # Run all examples
        examples_to_run = list(examples.items())
    
    # Check if any example requires NPU
    requires_npu = any(ex_info['requires_npu'] for _, ex_info in examples_to_run)
    
    if requires_npu:
        device_id = get_device_id()
        if device_id is None:
            return
        # Set the device once for all examples
        torch.npu.set_device(device_id)
    
    try:
        for ex_id, ex_info in examples_to_run:
            if ex_info['requires_npu'] and device_id is None:
                print(f"Skipping example {ex_id} ({ex_info['name']}): NPU device not configured")
                continue
            
            print(f"Running Example {ex_id}: {ex_info['name']}")
            ex_info['function']()
        
        if len(examples_to_run) > 1:
            print("=" * 60)
            print("All set print options examples passed!")
            print("=" * 60)
        
    except Exception as e:
        print(f"\nError: {e}")
        raise


if __name__ == "__main__":
    main()
