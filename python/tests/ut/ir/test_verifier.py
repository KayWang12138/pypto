#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
Test cases for IR Verifier Python bindings
"""
from pypto.pypto_impl import ir


def test_verify_status_enum():
    """Test VerifyStatus enum values"""
    assert str(ir.VerifyStatus.PASS) == "VerifyStatus.PASS"
    assert str(ir.VerifyStatus.FAIL) == "VerifyStatus.FAIL"
    assert ir.VerifyStatus.PASS != ir.VerifyStatus.FAIL


def test_verify_result_creation():
    """Test VerifyResult creation and properties"""
    # Create a passing result
    result_pass = ir.VerifyResult(ir.VerifyStatus.PASS, "")
    assert result_pass.status == ir.VerifyStatus.PASS
    assert result_pass.error_msg == ""
    assert result_pass.passed()

    # Create a failing result
    result_fail = ir.VerifyResult(ir.VerifyStatus.FAIL, "Test error message")
    assert result_fail.status == ir.VerifyStatus.FAIL
    assert result_fail.error_msg == "Test error message"
    assert not result_fail.passed()

    # Test __repr__
    assert "PASS" in repr(result_pass)
    assert "FAIL" in repr(result_fail)


def test_verifier_creation():
    """Test Verifier instance creation"""
    verifier = ir.Verifier()
    assert verifier is not None
    assert verifier.is_empty()
    assert verifier.get_rule_count() == 0
    assert "rule_count=0" in repr(verifier)


def test_verifier_register_rule_with_tuple():
    """Test registering rules using tuple return (bool, str)"""
    verifier = ir.Verifier()

    # Register a simple rule that always passes
    def always_pass_rule(tile):
        return (True, "")

    verifier.register_rule("always_pass", always_pass_rule)
    assert verifier.has_rule("always_pass")
    assert verifier.get_rule_count() == 1
    assert not verifier.is_empty()
    assert "always_pass" in verifier.get_rule_names()


def test_verifier_register_rule_with_verify_result():
    """Test registering rules using VerifyResult return"""
    verifier = ir.Verifier()

    # Register a rule that returns VerifyResult
    def verify_result_rule(tile):
        return ir.VerifyResult(ir.VerifyStatus.PASS, "All good")

    verifier.register_rule("result_rule", verify_result_rule)
    assert verifier.has_rule("result_rule")
    assert verifier.get_rule_count() == 1


def test_verifier_register_multiple_rules():
    """Test registering multiple rules"""
    verifier = ir.Verifier()

    def rule1(tile):
        return (True, "")

    def rule2(tile):
        return ir.VerifyResult(ir.VerifyStatus.PASS, "")

    def rule3(tile):
        return (False, "Rule 3 failed")

    verifier.register_rule("rule1", rule1)
    verifier.register_rule("rule2", rule2)
    verifier.register_rule("rule3", rule3)

    assert verifier.get_rule_count() == 3
    rule_names = verifier.get_rule_names()
    assert "rule1" in rule_names
    assert "rule2" in rule_names
    assert "rule3" in rule_names


def test_verifier_remove_rule():
    """Test removing a rule"""
    verifier = ir.Verifier()

    def test_rule(tile):
        return (True, "")

    verifier.register_rule("test_rule", test_rule)
    assert verifier.has_rule("test_rule")
    assert verifier.get_rule_count() == 1

    # Remove the rule
    result = verifier.remove_rule("test_rule")
    assert result is True
    assert not verifier.has_rule("test_rule")
    assert verifier.get_rule_count() == 0

    # Try to remove a non-existent rule
    result = verifier.remove_rule("non_existent")
    assert result is False


def test_verifier_clear_rules():
    """Test clearing all rules"""
    verifier = ir.Verifier()

    def rule1(tile):
        return (True, "")

    def rule2(tile):
        return (True, "")

    verifier.register_rule("rule1", rule1)
    verifier.register_rule("rule2", rule2)
    assert verifier.get_rule_count() == 2

    # Clear all rules
    verifier.clear_rules()
    assert verifier.get_rule_count() == 0
    assert verifier.is_empty()


def test_verifier_verify_rule():
    """Test verifying a tile against a single rule"""
    verifier = ir.Verifier()
    builder = ir.IrBuilder()
    ctx = ir.IrBuilderContext()

    # Create a simple function context
    sig = ir.FunctionSignature()
    func = builder.create_function("test_verify", ir.FunctionKind.DataFlow, sig)
    builder.enter_function(ctx, func)

    # Create a test tile
    tile_shape = [128, 128]
    test_tile = builder.create_tile(ctx, tile_shape, ir.DataType.float32, "test_tile")

    # Register a shape validation rule
    def check_shape_square(tile):
        shape = tile.shape
        if len(shape) == 2 and shape[0] == shape[1]:
            return (True, "")
        return (False, f"Shape is not square: {shape}")

    verifier.register_rule("shape_square", check_shape_square)

    # Verify the tile
    result = verifier.verify_rule("shape_square", test_tile)
    assert result.passed()
    assert result.error_msg == ""

    ctx.pop_scope()


def test_verifier_verify_rule_failure():
    """Test verifying a tile that fails a rule"""
    verifier = ir.Verifier()
    builder = ir.IrBuilder()
    ctx = ir.IrBuilderContext()

    sig = ir.FunctionSignature()
    func = builder.create_function("test_verify_fail", ir.FunctionKind.DataFlow, sig)
    builder.enter_function(ctx, func)

    # Create a non-square tile
    tile_shape = [128, 256]
    test_tile = builder.create_tile(ctx, tile_shape, ir.DataType.float32, "test_tile")

    # Register a shape validation rule that expects square tiles
    def check_shape_square(tile):
        shape = tile.shape
        if len(shape) == 2 and shape[0] == shape[1]:
            return (True, "")
        return (False, f"Shape is not square: {shape}")

    verifier.register_rule("shape_square", check_shape_square)

    # Verify the tile - should fail
    result = verifier.verify_rule("shape_square", test_tile)
    assert not result.passed()
    assert "not square" in result.error_msg

    ctx.pop_scope()


def test_verifier_verify_all_rules():
    """Test verifying a tile against all registered rules"""
    verifier = ir.Verifier()
    builder = ir.IrBuilder()
    ctx = ir.IrBuilderContext()

    sig = ir.FunctionSignature()
    func = builder.create_function("test_verify_all", ir.FunctionKind.DataFlow, sig)
    builder.enter_function(ctx, func)

    # Create a test tile
    tile_shape = [128, 128]
    test_tile = builder.create_tile(ctx, tile_shape, ir.DataType.float32, "test_tile")

    # Register multiple rules
    def check_2d(tile):
        return (len(tile.shape) == 2, "Must be 2D")

    def check_square(tile):
        shape = tile.shape
        return (shape[0] == shape[1], "Must be square")

    def check_size_128(tile):
        shape = tile.shape
        return (shape[0] == 128, "Must have size 128")

    verifier.register_rule("check_2d", check_2d)
    verifier.register_rule("check_square", check_square)
    verifier.register_rule("check_size_128", check_size_128)

    # Verify all rules - should pass
    result = verifier.verify_all_rules(test_tile)
    assert result.passed()

    ctx.pop_scope()


def test_verifier_verify_all_rules_with_failure():
    """Test verifying a tile against all rules when some fail"""
    verifier = ir.Verifier()
    builder = ir.IrBuilder()
    ctx = ir.IrBuilderContext()

    sig = ir.FunctionSignature()
    func = builder.create_function("test_verify_all_fail", ir.FunctionKind.DataFlow, sig)
    builder.enter_function(ctx, func)

    # Create a test tile
    tile_shape = [128, 256]
    test_tile = builder.create_tile(ctx, tile_shape, ir.DataType.float32, "test_tile")

    # Register multiple rules
    def check_2d(tile):
        return (len(tile.shape) == 2, "Must be 2D")

    def check_square(tile):
        shape = tile.shape
        return (shape[0] == shape[1], "Must be square")  # This will fail

    def check_positive_size(tile):
        shape = tile.shape
        return (all(s > 0 for s in shape), "All dimensions must be positive")

    verifier.register_rule("check_2d", check_2d)
    verifier.register_rule("check_square", check_square)
    verifier.register_rule("check_positive_size", check_positive_size)

    # Verify all rules - should fail due to square check
    result = verifier.verify_all_rules(test_tile)
    assert not result.passed()
    assert "check_square" in result.error_msg or "Must be square" in result.error_msg

    ctx.pop_scope()


def test_verifier_verify_unknown_rule():
    """Test verifying against a non-existent rule"""
    verifier = ir.Verifier()
    builder = ir.IrBuilder()
    ctx = ir.IrBuilderContext()

    sig = ir.FunctionSignature()
    func = builder.create_function("test_unknown_rule", ir.FunctionKind.DataFlow, sig)
    builder.enter_function(ctx, func)

    tile_shape = [128, 128]
    test_tile = builder.create_tile(ctx, tile_shape, ir.DataType.float32, "test_tile")

    # Try to verify against a non-existent rule
    result = verifier.verify_rule("non_existent_rule", test_tile)
    assert not result.passed()
    assert "Unknown rule" in result.error_msg

    ctx.pop_scope()


def test_verifier_complex_rule():
    """Test a more complex verification rule"""
    verifier = ir.Verifier()
    builder = ir.IrBuilder()
    ctx = ir.IrBuilderContext()

    sig = ir.FunctionSignature()
    func = builder.create_function("test_complex", ir.FunctionKind.DataFlow, sig)
    builder.enter_function(ctx, func)

    tile_shape = [256, 256]
    test_tile = builder.create_tile(ctx, tile_shape, ir.DataType.float16, "test_tile")

    # Complex rule: check if tile is square and size is power of 2
    def check_power_of_2_square(tile):
        shape = tile.shape
        if len(shape) != 2:
            return ir.VerifyResult(ir.VerifyStatus.FAIL, "Must be 2D tensor")

        if shape[0] != shape[1]:
            return ir.VerifyResult(ir.VerifyStatus.FAIL, "Must be square")

        size = shape[0]
        # Check if power of 2
        if size & (size - 1) != 0:
            return ir.VerifyResult(ir.VerifyStatus.FAIL, f"Size {size} is not a power of 2")

        return ir.VerifyResult(ir.VerifyStatus.PASS, "")

    verifier.register_rule("power_of_2_square", check_power_of_2_square)

    # Verify - should pass (256 is a power of 2)
    result = verifier.verify_rule("power_of_2_square", test_tile)
    assert result.passed()

    ctx.pop_scope()


def test_verifier_empty_rules():
    """Test verifying with no rules registered"""
    verifier = ir.Verifier()
    builder = ir.IrBuilder()
    ctx = ir.IrBuilderContext()

    sig = ir.FunctionSignature()
    func = builder.create_function("test_empty", ir.FunctionKind.DataFlow, sig)
    builder.enter_function(ctx, func)

    tile = builder.create_tile(ctx, [128, 128], ir.DataType.float32, "tile")

    # Verify all rules when no rules are registered
    result = verifier.verify_all_rules(tile)
    assert not result.passed()
    assert "No verification rules" in result.error_msg

    ctx.pop_scope()


if __name__ == "__main__":
    # Run basic tests
    test_verify_status_enum()
    test_verify_result_creation()
    test_verifier_creation()
    test_verifier_register_rule_with_tuple()
    test_verifier_register_rule_with_verify_result()
    test_verifier_register_multiple_rules()
    test_verifier_remove_rule()
    test_verifier_clear_rules()
    test_verifier_verify_rule()
    test_verifier_verify_rule_failure()
    test_verifier_verify_all_rules()
    test_verifier_verify_all_rules_with_failure()
    test_verifier_verify_unknown_rule()
    test_verifier_complex_rule()
    test_verifier_empty_rules()

    print("All Verifier tests passed!")
