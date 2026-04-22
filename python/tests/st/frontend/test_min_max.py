#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
Regression tests: verify that Python's builtin min is invoked as-is when all
arguments are non-dynamic (i.e. no SymbolicScalar is involved), and that the
framework intercepts the call only when at least one SymbolicScalar is present.

Dispatch rules under test
--------------------------
* Non-dynamic operands only  →  result is a plain Python object (int / str / …),
                                 Python builtin min is used unchanged.
* At least one SymbolicScalar →  framework intercepts the call and returns a
                                 SymbolicScalar.
"""

import logging
import os

import torch
import torch_npu
import pypto
from numpy.testing import assert_allclose


logging.basicConfig(level=logging.INFO, format="", force=True)
DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))


# ---------------------------------------------------------------------------
# Shared assertion helper – inlined by the frontend JIT into every kernel
# that calls it.
# ---------------------------------------------------------------------------

def _assert_min_result(
    result,
    *,
    expect_symbolic: bool,
    expected_value=None,
    label: str = "",
) -> None:
    """Unified checker for a single min() call site.

    Parameters
    ----------
    result:
        The value returned by min().
    expect_symbolic:
        True  → result must be a pypto.SymbolicScalar (dynamic-axis path).
        False → result must NOT be a SymbolicScalar  (Python-builtin path).
    expected_value:
        When not None, also assert ``result == expected_value``.
    label:
        Human-readable description printed in assertion messages.
    """
    tag = f"[{label}] " if label else ""

    if expect_symbolic:
        assert isinstance(result, pypto.SymbolicScalar), (
            f"{tag}Expected SymbolicScalar, got {type(result).__name__!r}. "
            "min() over a dynamic operand must be intercepted by the framework."
        )
    else:
        assert not isinstance(result, pypto.SymbolicScalar), (
            f"{tag}Expected a plain Python value, got SymbolicScalar. "
            "min() over non-dynamic operands must use the Python builtin."
        )

    if expected_value is not None:
        assert result == expected_value, (
            f"{tag}Value mismatch: expected {expected_value!r}, got {result!r}"
        )


# ---------------------------------------------------------------------------
# Single kernel covering both dispatch paths
# ---------------------------------------------------------------------------

@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def kernel_min_dispatch(
    a: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_FP32),
    out: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_FP32),
):
    """
    axis-0 is DYNAMIC → any min() that touches a.shape[0] must go through the
                         framework and return a SymbolicScalar.
    axis-1 is STATIC  → any min() that touches only a.shape[1] or plain Python
                         objects must fall back to the Python builtin and return
                         a plain value.

    All assertions are centralised in _assert_min_result so that the dispatch
    contract is checked in one place and failures carry a descriptive label.
    """

    # ------------------------------------------------------------------
    # 1. Plain Python objects – must always use the Python builtin
    # ------------------------------------------------------------------

    # 1a. str list with key=len
    _assert_min_result(
        min(["bbb", "a", "cc"], key=len),
        expect_symbolic=False,
        expected_value="a",
        label="str-list key=len",
    )

    # 1b. Plain int iterable
    _assert_min_result(
        min([10, 3, 7]),
        expect_symbolic=False,
        expected_value=3,
        label="int-list",
    )

    # 1c. Two plain int arguments
    _assert_min_result(
        min(8, 16),
        expect_symbolic=False,
        expected_value=8,
        label="int-int",
    )

    # 1d. Empty iterable with int default
    _assert_min_result(
        min([], default=42),
        expect_symbolic=False,
        expected_value=42,
        label="empty-default-int",
    )

    # 1e. Empty iterable with str default and key
    _assert_min_result(
        min([], key=len, default="fallback"),
        expect_symbolic=False,
        expected_value="fallback",
        label="empty-default-str",
    )

    # ------------------------------------------------------------------
    # 2. STATIC axis – resolves to a plain int, must use Python builtin
    # ------------------------------------------------------------------

    # a.shape[1] is STATIC (value == 8 in the test), so min stays native
    _assert_min_result(
        min(a.shape[1], 16),
        expect_symbolic=False,
        expected_value=8,
        label="static-dim vs int",
    )

    _assert_min_result(
        min([a.shape[1], 64, 1]),
        expect_symbolic=False,
        expected_value=1,
        label="static-dim in list",
    )

    # ------------------------------------------------------------------
    # 3. DYNAMIC axis – must be intercepted by the framework
    # ------------------------------------------------------------------

    # a.shape[0] is DYNAMIC → result must be SymbolicScalar
    res = min(a.shape[0], 16)
    _assert_min_result(
        res,
        expect_symbolic=True,
        label="dynamic-dim vs int",
    )

    _assert_min_result(
        min(a.shape[0], 16),
        expect_symbolic=True,
        label="dynamic-dim vs int nested-call-arg",
    )

    # ------------------------------------------------------------------
    # 4. Explicit SymbolicScalar operands – framework intercept path
    # ------------------------------------------------------------------

    # 4a. Two SymbolicScalars
    m = pypto.symbolic_scalar(10)
    n = pypto.symbolic_scalar(20)
    _assert_min_result(
        min(m, n),
        expect_symbolic=False,
        expected_value=10,
        label="SymbolicScalar-SymbolicScalar",
    )

    # 4b. SymbolicScalar vs plain int
    dim0 = pypto.symbolic_scalar(8)
    _assert_min_result(
        min(dim0, 16),
        expect_symbolic=False,
        expected_value=8,
        label="SymbolicScalar vs int",
    )

    # ------------------------------------------------------------------
    # 5. Actual compute
    # ------------------------------------------------------------------
    pypto.set_vec_tile_shapes(16, 16)
    for idx in pypto.loop(a.shape[0], name="LOOP", idx_name="k"):
        temp = a[idx: idx + 1, :]
        out[idx: idx + 1, :] = temp + 1


# ---------------------------------------------------------------------------
# Test entry point
# ---------------------------------------------------------------------------

def test_min_dispatch():
    """
    Drive kernel_min_dispatch with a concrete tensor whose static axis equals 8,
    then verify the numerical output.
    """
    torch.npu.set_device(DEVICE_ID)
    dev = f"npu:{DEVICE_ID}"

    # shape (1, 8): axis-0 dynamic (size 1), axis-1 static (size 8)
    a = torch.ones(1, 8, dtype=torch.float32, device=dev)
    out = torch.zeros(1, 8, dtype=torch.float32, device=dev)

    kernel_min_dispatch(a, out)
    torch_npu.npu.synchronize()

    assert torch.allclose(out.cpu(), (a + 1).cpu()), (
        "Numerical output mismatch: expected all-twos tensor"
    )
    logging.info("test_min_dispatch PASSED")


if __name__ == "__main__":
    test_min_dispatch()
    logging.info("All min frontend dispatch regression tests passed.")
