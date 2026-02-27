# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for IRVerifier."""

import pytest
from pypto import ir
from pypto.ir import builder, DataType
from pypto.ir.pass_manager import passes


def test_verifier_create_default():
    """Test creating default verifier."""
    verifier = passes.IRVerifier.create_default()
    assert verifier is not None


def test_verifier_empty():
    """Test creating empty verifier."""
    verifier = passes.IRVerifier()
    assert verifier is not None


def test_verifier_valid_program():
    """Test verifier on valid SSA program."""
    ib = builder.IRBuilder()

    with ib.function("test_valid") as f:
        a = f.param("a", ir.ScalarType(DataType.INT64))
        b = f.param("b", ir.ScalarType(DataType.INT64))
        f.return_type(ir.ScalarType(DataType.INT64))

        x = ib.let("x", a)
        ib.let("y", b)
        z = ib.let("z", x)

        ib.return_stmt(z)

    func = f.get_result()
    program = ir.Program([func], "test_program", ir.Span.unknown())

    # Create verifier and run verification
    verifier = passes.IRVerifier.create_default()
    diagnostics = verifier.verify(program)

    # Should have no diagnostics
    assert len(diagnostics) == 0


def test_verifier_disable_rule():
    """Test disabling verification rules."""
    verifier = passes.IRVerifier.create_default()
    verifier.disable_rule("SSAVerify")
    assert not verifier.is_rule_enabled("SSAVerify")


def test_verifier_enable_rule():
    """Test enabling a disabled rule."""
    verifier = passes.IRVerifier.create_default()

    # Disable and then re-enable
    verifier.disable_rule("SSAVerify")
    assert not verifier.is_rule_enabled("SSAVerify")

    verifier.enable_rule("SSAVerify")
    assert verifier.is_rule_enabled("SSAVerify")


def test_verifier_or_throw_no_error():
    """Test verify_or_throw on valid program (should not throw)."""
    ib = builder.IRBuilder()

    with ib.function("test_no_throw") as f:
        a = f.param("a", ir.ScalarType(DataType.INT64))
        f.return_type(ir.ScalarType(DataType.INT64))

        x = ib.let("x", a)
        ib.return_stmt(x)

    func = f.get_result()
    program = ir.Program([func], "test_program", ir.Span.unknown())

    verifier = passes.IRVerifier.create_default()
    # Should not raise exception
    verifier.verify_or_throw(program)


def test_verifier_generate_report_empty():
    """Test generating verification report with no diagnostics."""
    report = passes.IRVerifier.generate_report([])
    assert "IR Verification Report" in report
    assert len(report) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
