# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for BasicMemoryReusePass using IRBuilder with block ops."""

from pypto import ir
from pypto.ir import builder, DataType
from pypto.ir.op import block
from pypto.ir.pass_manager import OptimizationStrategy, PassManager, passes
from pypto.ir.pass_manager import core_ir


def _get_var_type(func, var_name):
    """Extract ShapedType for a variable by name."""
    if not isinstance(func.body, ir.SeqStmts):
        return None
    for stmt in func.body.stmts:
        if isinstance(stmt, ir.AssignStmt) and stmt.var.name == var_name:
            if isinstance(stmt.var.type, core_ir.ShapedType):
                return stmt.var.type
    return None


def _assert_shares_memref(func, var_a, var_b):
    """Assert two variables share the same MemRef object."""
    type_a = _get_var_type(func, var_a)
    type_b = _get_var_type(func, var_b)
    assert type_a is not None, f"{var_a} should have ShapedType"
    assert type_b is not None, f"{var_b} should have ShapedType"
    assert type_a.shares_memref_with(type_b), f"{var_b} should share the same MemRef with {var_a}"


def _assert_not_shares_memref(func, var_a, var_b):
    """Assert two variables do NOT share the same MemRef object."""
    type_a = _get_var_type(func, var_a)
    type_b = _get_var_type(func, var_b)
    assert type_a is not None, f"{var_a} should have ShapedType"
    assert type_b is not None, f"{var_b} should have ShapedType"
    assert not type_a.shares_memref_with(type_b), f"{var_b} should NOT share MemRef with {var_a}"


def _run_memory_reuse(program):
    """Run InitMemRefPass then BasicMemoryReusePass, return the first function."""
    program = passes.init_mem_ref()(program)
    program = passes.basic_memory_reuse()(program)
    return list(program.functions.values())[0]


def _assert_all_have_memrefs(func):
    """Assert all ShapedType variables have memrefs assigned."""
    assert isinstance(func.body, ir.SeqStmts)
    for stmt in func.body.stmts:
        if isinstance(stmt, ir.AssignStmt) and isinstance(stmt.var.type, core_ir.ShapedType):
            assert stmt.var.type.memref is not None, f"{stmt.var.name} should have a memref"


def _build_simple_program():
    """Build the 'simple' test program:
    tile_a = load(input_a), tile_b = load(input_b), tile_c = add(tile_a, tile_b),
    tile_d = mul(tile_c, tile_c), tile_e = add(tile_d, tile_d), result = store(tile_e, output)
    """
    ib = builder.IRBuilder()

    with ib.function("main") as f:
        input_a = f.param("input_a", ir.TensorType([64, 64], DataType.FP32))
        input_b = f.param("input_b", ir.TensorType([64, 64], DataType.FP32))
        output = f.param("output", ir.TensorType([64, 64], DataType.FP32))
        f.return_type(ir.TensorType([64, 64], DataType.FP32))

        tile_a = ib.let("tile_a", block.load(input_a, 0, 0, 64, 64))
        tile_b = ib.let("tile_b", block.load(input_b, 0, 0, 64, 64))
        tile_c = ib.let("tile_c", block.add(tile_a, tile_b))
        tile_d = ib.let("tile_d", block.mul(tile_c, tile_c))
        tile_e = ib.let("tile_e", block.add(tile_d, tile_d))
        result = ib.let("result", block.store(tile_e, 0, 0, 64, 64, output))
        ib.return_stmt(result)

    func = f.get_result()
    return ir.Program([func], "Before", ir.Span.unknown())


def _build_sequential_program():
    """Build the 'sequential' test program:
    tile_a = load(input_a), tile_b = add(tile_a, tile_a), tile_c = add(tile_b, tile_b),
    tile_d = add(tile_c, tile_c), tile_e = add(tile_d, tile_d), result = store(tile_e, output)
    """
    ib = builder.IRBuilder()

    with ib.function("main") as f:
        input_a = f.param("input_a", ir.TensorType([64, 64], DataType.FP32))
        output = f.param("output", ir.TensorType([64, 64], DataType.FP32))
        f.return_type(ir.TensorType([64, 64], DataType.FP32))

        tile_a = ib.let("tile_a", block.load(input_a, 0, 0, 64, 64))
        tile_b = ib.let("tile_b", block.add(tile_a, tile_a))
        tile_c = ib.let("tile_c", block.add(tile_b, tile_b))
        tile_d = ib.let("tile_d", block.add(tile_c, tile_c))
        tile_e = ib.let("tile_e", block.add(tile_d, tile_d))
        result = ib.let("result", block.store(tile_e, 0, 0, 64, 64, output))
        ib.return_stmt(result)

    func = f.get_result()
    return ir.Program([func], "Before", ir.Span.unknown())


def _build_different_sizes_program():
    """Build the 'different_sizes' test program with mixed 64x64 and 32x32 tiles."""
    ib = builder.IRBuilder()

    with ib.function("main") as f:
        input_a = f.param("input_a", ir.TensorType([64, 64], DataType.FP32))
        input_b = f.param("input_b", ir.TensorType([32, 32], DataType.FP32))
        output_a = f.param("output_a", ir.TensorType([64, 64], DataType.FP32))
        output_b = f.param("output_b", ir.TensorType([32, 32], DataType.FP32))
        f.return_type(ir.TensorType([32, 32], DataType.FP32))

        tile_a = ib.let("tile_a", block.load(input_a, 0, 0, 64, 64))
        tile_b = ib.let("tile_b", block.load(input_b, 0, 0, 32, 32))
        tile_c = ib.let("tile_c", block.add(tile_a, tile_a))
        _result_a = ib.let("_result_a", block.store(tile_c, 0, 0, 64, 64, output_a))
        tile_d = ib.let("tile_d", block.add(tile_b, tile_b))
        result_b = ib.let("result_b", block.store(tile_d, 0, 0, 32, 32, output_b))
        ib.return_stmt(result_b)

    func = f.get_result()
    return ir.Program([func], "Before", ir.Span.unknown())


def _build_empty_program():
    """Build program with empty function (just returns output param)."""
    ib = builder.IRBuilder()

    with ib.function("main") as f:
        output = f.param("output", ir.TensorType([64, 64], DataType.FP32))
        f.return_type(ir.TensorType([64, 64], DataType.FP32))
        ib.return_stmt(output)

    func = f.get_result()
    return ir.Program([func], "Before", ir.Span.unknown())


def _build_memref_sharing_program():
    """Build the 'memref_sharing' test program:
    tile_a = load(input_a), tile_b = add(tile_a, tile_a), tile_c = add(tile_b, tile_b),
    tile_d = add(tile_c, tile_c), result = store(tile_d, output)
    """
    ib = builder.IRBuilder()

    with ib.function("main") as f:
        input_a = f.param("input_a", ir.TensorType([64, 64], DataType.FP32))
        output = f.param("output", ir.TensorType([64, 64], DataType.FP32))
        f.return_type(ir.TensorType([64, 64], DataType.FP32))

        tile_a = ib.let("tile_a", block.load(input_a, 0, 0, 64, 64))
        tile_b = ib.let("tile_b", block.add(tile_a, tile_a))
        tile_c = ib.let("tile_c", block.add(tile_b, tile_b))
        tile_d = ib.let("tile_d", block.add(tile_c, tile_c))
        result = ib.let("result", block.store(tile_d, 0, 0, 64, 64, output))
        ib.return_stmt(result)

    func = f.get_result()
    return ir.Program([func], "Before", ir.Span.unknown())


def _build_with_dependencies_program():
    """Build the 'with_dependencies' test program:
    tile_a = load(input_a), tile_b = load(input_b), tile_c = add(tile_a, tile_b),
    tile_d = add(tile_c, tile_c), tile_e = add(tile_d, tile_d), result = store(tile_e, output)
    """
    ib = builder.IRBuilder()

    with ib.function("main") as f:
        input_a = f.param("input_a", ir.TensorType([64, 64], DataType.FP32))
        input_b = f.param("input_b", ir.TensorType([64, 64], DataType.FP32))
        output = f.param("output", ir.TensorType([64, 64], DataType.FP32))
        f.return_type(ir.TensorType([64, 64], DataType.FP32))

        tile_a = ib.let("tile_a", block.load(input_a, 0, 0, 64, 64))
        tile_b = ib.let("tile_b", block.load(input_b, 0, 0, 64, 64))
        tile_c = ib.let("tile_c", block.add(tile_a, tile_b))
        tile_d = ib.let("tile_d", block.add(tile_c, tile_c))
        tile_e = ib.let("tile_e", block.add(tile_d, tile_d))
        result = ib.let("result", block.store(tile_e, 0, 0, 64, 64, output))
        ib.return_stmt(result)

    func = f.get_result()
    return ir.Program([func], "Before", ir.Span.unknown())


def _build_transitive_conflict_program():
    """Build the 'transitive_conflict' test program:
    tile_a = load(input_a), tile_b = add(tile_a, tile_a), tile_c = add(tile_b, tile_b),
    tile_d = add(tile_c, tile_c), tile_e = add(tile_c, tile_d), result = store(tile_e, output)
    """
    ib = builder.IRBuilder()

    with ib.function("main") as f:
        input_a = f.param("input_a", ir.TensorType([64, 64], DataType.FP32))
        output = f.param("output", ir.TensorType([64, 64], DataType.FP32))
        f.return_type(ir.TensorType([64, 64], DataType.FP32))

        tile_a = ib.let("tile_a", block.load(input_a, 0, 0, 64, 64))
        tile_b = ib.let("tile_b", block.add(tile_a, tile_a))
        tile_c = ib.let("tile_c", block.add(tile_b, tile_b))
        tile_d = ib.let("tile_d", block.add(tile_c, tile_c))
        tile_e = ib.let("tile_e", block.add(tile_c, tile_d))
        result = ib.let("result", block.store(tile_e, 0, 0, 64, 64, output))
        ib.return_stmt(result)

    func = f.get_result()
    return ir.Program([func], "Before", ir.Span.unknown())


def _build_multiple_memory_spaces_program():
    """Build the 'multiple_memory_spaces' test program:
    tile_a = load(input_a), tile_b = load(input_b), tile_c = add(tile_a, tile_b),
    _result_a = store(tile_c, output_a), tile_d = add(tile_c, tile_c), result_b = store(tile_d, output_b)
    """
    ib = builder.IRBuilder()

    with ib.function("main") as f:
        input_a = f.param("input_a", ir.TensorType([64, 64], DataType.FP32))
        input_b = f.param("input_b", ir.TensorType([64, 64], DataType.FP32))
        output_a = f.param("output_a", ir.TensorType([64, 64], DataType.FP32))
        output_b = f.param("output_b", ir.TensorType([64, 64], DataType.FP32))
        f.return_type(ir.TensorType([64, 64], DataType.FP32))

        tile_a = ib.let("tile_a", block.load(input_a, 0, 0, 64, 64))
        tile_b = ib.let("tile_b", block.load(input_b, 0, 0, 64, 64))
        tile_c = ib.let("tile_c", block.add(tile_a, tile_b))
        _result_a = ib.let("_result_a", block.store(tile_c, 0, 0, 64, 64, output_a))
        tile_d = ib.let("tile_d", block.add(tile_c, tile_c))
        result_b = ib.let("result_b", block.store(tile_d, 0, 0, 64, 64, output_b))
        ib.return_stmt(result_b)

    func = f.get_result()
    return ir.Program([func], "Before", ir.Span.unknown())


class TestBasicMemoryReuse:
    """Tests for BasicMemoryReusePass with TileType variables."""

    def test_simple(self):
        """tile_d reuses tile_a, tile_e reuses tile_b (transitive conflict prevents both from tile_a).

        Lifetimes: tile_a[0,2], tile_b[1,2], tile_c[2,3], tile_d[3,4], tile_e[4,5]
        """
        program = _build_simple_program()
        func = _run_memory_reuse(program)

        _assert_all_have_memrefs(func)
        _assert_shares_memref(func, "tile_a", "tile_d")
        _assert_shares_memref(func, "tile_b", "tile_e")

    def test_sequential(self):
        """Sequential chain: tile_c reuses tile_a, tile_d reuses tile_b, tile_e reuses tile_c.

        Lifetimes: tile_a[0,1], tile_b[1,2], tile_c[2,3], tile_d[3,4], tile_e[4,5]
        """
        program = _build_sequential_program()
        func = _run_memory_reuse(program)

        _assert_all_have_memrefs(func)
        _assert_shares_memref(func, "tile_a", "tile_c")
        _assert_shares_memref(func, "tile_b", "tile_d")
        _assert_shares_memref(func, "tile_c", "tile_e")

    def test_different_sizes(self):
        """Small tile (32x32) can reuse large tile (64x64) buffer, not vice versa.

        tile_d (32x32) reuses tile_a (64x64) since 64x64 >= 32x32.
        """
        program = _build_different_sizes_program()
        func = _run_memory_reuse(program)

        _assert_all_have_memrefs(func)
        _assert_shares_memref(func, "tile_a", "tile_d")

    def test_empty_function(self):
        """Empty function should not crash."""
        program = _build_empty_program()
        After = passes.basic_memory_reuse()(program)
        func = list(After.functions.values())[0]

        assert func is not None
        assert func.name == "main"

    def test_memref_sharing(self):
        """Chain: tile_c reuses tile_a, tile_d reuses tile_b.

        Lifetimes: tile_a[0,1], tile_b[1,2], tile_c[2,3], tile_d[3,4]
        """
        program = _build_memref_sharing_program()
        func = _run_memory_reuse(program)

        _assert_all_have_memrefs(func)
        _assert_shares_memref(func, "tile_a", "tile_c")
        _assert_shares_memref(func, "tile_b", "tile_d")

    def test_with_dependencies(self):
        """tile_d reuses tile_a, tile_e reuses tile_b (transitive conflict).

        Lifetimes: tile_a[0,2], tile_b[1,2], tile_c[2,3], tile_d[3,4], tile_e[4,5]
        """
        program = _build_with_dependencies_program()
        func = _run_memory_reuse(program)

        _assert_all_have_memrefs(func)
        _assert_shares_memref(func, "tile_a", "tile_d")
        _assert_shares_memref(func, "tile_b", "tile_e")

    def test_transitive_conflict(self):
        """Transitive conflict: tile_c and tile_d must NOT share memory.

        Lifetimes: tile_a[0,1], tile_b[1,2], tile_c[2,4], tile_d[3,4], tile_e[4,5]
        tile_c reuses tile_a, tile_d reuses tile_b (not tile_a, conflict with tile_c).
        """
        program = _build_transitive_conflict_program()
        func = _run_memory_reuse(program)

        _assert_all_have_memrefs(func)
        _assert_shares_memref(func, "tile_a", "tile_c")
        _assert_shares_memref(func, "tile_b", "tile_d")
        _assert_not_shares_memref(func, "tile_c", "tile_d")

    def test_multiple_memory_spaces(self):
        """Memory reuse happens within the same memory space (UB tiles).

        Verifies that variables in DDR don't reuse UB memory and vice versa.
        Parameters are in DDR, tiles are in UB.

        Lifetimes: tile_a[0,2], tile_b[1,2], tile_c[2,4], tile_d[4,5]
        tile_d should reuse tile_a's UB memory.
        """
        program = _build_multiple_memory_spaces_program()
        func = _run_memory_reuse(program)

        _assert_all_have_memrefs(func)
        # tile_d should reuse UB memory from tile_a
        _assert_shares_memref(func, "tile_a", "tile_d")

    def test_with_pass_manager(self):
        """Test using PassManager PTOAS strategy."""
        program = _build_simple_program()

        pm = PassManager.get_strategy(OptimizationStrategy.PTOAS)
        After = pm.run_passes(program)
        func = list(After.functions.values())[0]

        _assert_all_have_memrefs(func)
        _assert_shares_memref(func, "tile_a", "tile_d")
        _assert_shares_memref(func, "tile_b", "tile_e")
