# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for CCECodegen class."""

import pypto_block.language as pl
import pypto_block.language.op.manual as plm
import pytest
import re
from pypto_block import DataType, backend, codegen, ir
from pypto_block.backend import BackendType
from pypto_block.ir.builder import IRBuilder
from pypto_block.ir.op import block
from pypto_block.ir.pass_manager import PassManager


class TestCCECodegenBasics:
    """Test basic CCECodegen functionality."""

    def test_create_cce_codegen(self):
        """Test creating a CCECodegen instance."""
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.CCE)
        generator = codegen.CCECodegen()
        assert generator is not None


def test_manual_fillpad_codegen_uses_destination_pad_value():
    """CCE manual.fillpad should bind a null-pad alias source before TFILLPAD."""
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.CCE)

    @pl.program
    class ManualFillPadCCEProgram:
        @pl.function
        def fillpad_dynamic_tile(
            self,
            input: pl.Tensor[[16, 16], pl.FP32],
            output: pl.Tensor[[16, 16], pl.FP32],
            rows_arg: pl.Scalar[pl.INDEX],
            cols_arg: pl.Scalar[pl.INDEX],
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            src_type = plm.TileType(
                shape=[16, 16],
                dtype=pl.FP32,
                target_memory=pl.MemorySpace.Vec,
                pad=plm.TilePad.zero,
                valid_shape=[-1, -1],
            )
            dst_type = plm.TileType(
                shape=[16, 16],
                dtype=pl.FP32,
                target_memory=pl.MemorySpace.Vec,
                pad=plm.TilePad.zero,
            )
            src = plm.make_tile(src_type, addr=0x0000, size=1024)
            dst = plm.make_tile(dst_type, addr=0x1000, size=1024)
            plm.set_validshape(src, rows_arg, cols_arg)
            plm.fillpad(dst, src)
            return output

    pm = PassManager.get_strategy()
    optimized_program = pm.run_passes(ManualFillPadCCEProgram)

    generator = codegen.CCECodegen()
    files = generator.generate(optimized_program)
    kernel_name = list(optimized_program.functions.values())[0].name
    code = files["kernels/aiv/" + kernel_name + ".cpp"]

    assert "TFILLPAD(" in code
    assert "PadValue::Zero" in code
    assert "Tile<TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero>" in code
    assert "using __manual_fillpad_src_alias_type_" in code
    assert ".GetValidRow(), src.GetValidCol()" in code
    assert "TASSIGN(__manual_fillpad_src_alias_" in code
    assert "TMOV(__manual_fillpad_src_alias_" not in code
    assert "TFILLPAD(dst, __manual_fillpad_src_alias_" in code


def test_manual_fillpad_inplace_codegen_uses_explicit_inplace_lowering():
    """CCE manual.fillpad_inplace should lower without overwriting dst metadata."""
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.CCE)

    @pl.program
    class ManualFillPadInplaceCCEProgram:
        @pl.function
        def fillpad_inplace_dynamic_tile(
            self,
            input: pl.Tensor[[16, 16], pl.FP32],
            output: pl.Tensor[[16, 16], pl.FP32],
            rows_arg: pl.Scalar[pl.INDEX],
            cols_arg: pl.Scalar[pl.INDEX],
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            src_type = plm.TileType(
                shape=[16, 16],
                dtype=pl.FP32,
                target_memory=pl.MemorySpace.Vec,
                pad=plm.TilePad.zero,
                valid_shape=[-1, -1],
            )
            dst_type = plm.TileType(
                shape=[16, 16],
                dtype=pl.FP32,
                target_memory=pl.MemorySpace.Vec,
                pad=plm.TilePad.zero,
            )
            src = plm.make_tile(src_type, addr=0x0000, size=1024)
            dst = plm.make_tile(dst_type, addr=0x0000, size=1024)
            plm.set_validshape(src, rows_arg, cols_arg)
            plm.fillpad_inplace(dst, src)
            return output

    pm = PassManager.get_strategy()
    optimized_program = pm.run_passes(ManualFillPadInplaceCCEProgram)

    generator = codegen.CCECodegen()
    files = generator.generate(optimized_program)
    kernel_name = list(optimized_program.functions.values())[0].name
    code = files["kernels/aiv/" + kernel_name + ".cpp"]

    assert "using __manual_fillpad_src_alias_type_" in code
    assert "auto& dst = src;" not in code
    assert "dst.SetValidShape(16, 16);" not in code
    assert "TFILLPAD_INPLACE(dst, __manual_fillpad_src_alias_" in code


def test_manual_fillpad_expand_codegen_uses_destination_pad_value():
    """CCE manual.fillpad_expand should bind a null-pad alias source before TFILLPAD_EXPAND."""
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.CCE)

    @pl.program
    class ManualFillPadExpandCCEProgram:
        @pl.function
        def fillpad_expand_dynamic_tile(
            self,
            input: pl.Tensor[[16, 16], pl.FP32],
            output: pl.Tensor[[16, 32], pl.FP32],
            rows_arg: pl.Scalar[pl.INDEX],
            cols_arg: pl.Scalar[pl.INDEX],
        ) -> pl.Tensor[[16, 32], pl.FP32]:
            src_type = plm.TileType(
                shape=[16, 16],
                dtype=pl.FP32,
                target_memory=pl.MemorySpace.Vec,
                pad=plm.TilePad.zero,
                valid_shape=[-1, -1],
            )
            dst_type = plm.TileType(
                shape=[16, 32],
                dtype=pl.FP32,
                target_memory=pl.MemorySpace.Vec,
                pad=plm.TilePad.zero,
            )
            src = plm.make_tile(src_type, addr=0x0000, size=1024)
            dst = plm.make_tile(dst_type, addr=0x1000, size=2048)
            plm.set_validshape(src, rows_arg, cols_arg)
            plm.fillpad_expand(dst, src)
            return output

    pm = PassManager.get_strategy()
    optimized_program = pm.run_passes(ManualFillPadExpandCCEProgram)

    generator = codegen.CCECodegen()
    files = generator.generate(optimized_program)
    kernel_name = list(optimized_program.functions.values())[0].name
    code = files["kernels/aiv/" + kernel_name + ".cpp"]

    assert "TFILLPAD_EXPAND(" in code
    assert "PadValue::Zero" in code
    assert "Tile<TileType::Vec, float, 16, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero>" in code
    assert "using __manual_fillpad_src_alias_type_" in code
    assert ".GetValidRow(), src.GetValidCol()" in code
    assert "TASSIGN(__manual_fillpad_src_alias_" in code
    assert "TMOV(__manual_fillpad_src_alias_" not in code
    assert "TFILLPAD_EXPAND(dst, __manual_fillpad_src_alias_" in code


def test_manual_fillpad_expand_codegen_rejects_inplace():
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.CCE)

    @pl.program
    class ManualFillPadExpandInplaceCCEProgram:
        @pl.function
        def fillpad_expand_inplace_dynamic_tile(
            self,
            input: pl.Tensor[[16, 16], pl.FP32],
            output: pl.Tensor[[16, 16], pl.FP32],
            rows_arg: pl.Scalar[pl.INDEX],
            cols_arg: pl.Scalar[pl.INDEX],
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            src_type = plm.TileType(
                shape=[16, 16],
                dtype=pl.FP32,
                target_memory=pl.MemorySpace.Vec,
                pad=plm.TilePad.zero,
                valid_shape=[-1, -1],
            )
            src = plm.make_tile(src_type, addr=0x0000, size=1024)
            plm.set_validshape(src, rows_arg, cols_arg)
            plm.fillpad_expand(src, src)
            return output

    pm = PassManager.get_strategy()
    optimized_program = pm.run_passes(ManualFillPadExpandInplaceCCEProgram)

    generator = codegen.CCECodegen()
    with pytest.raises(ValueError, match="manual.fillpad_expand: inplace is not supported"):
        generator.generate(optimized_program)


def test_manual_store_fp_emits_cce_codegen():
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.CCE)

    @pl.program
    class ManualStoreFpCCEProgram:
        @pl.function
        def store_fp_cce_kernel(
            self,
            output: pl.Tensor[[32, 32], pl.INT8],
        ) -> pl.Tensor[[32, 32], pl.INT8]:
            src_type = plm.TileType(
                shape=[32, 32],
                dtype=pl.INT32,
                target_memory=pl.MemorySpace.Acc,
            )
            fp_type = plm.TileType(
                shape=[1, 16],
                dtype=pl.UINT64,
                target_memory=pl.MemorySpace.Scaling,
            )
            src = plm.make_tile(src_type, addr=0x0000, size=4096)
            fp = plm.make_tile(fp_type, addr=0x1000, size=128)
            plm.store(output, src, [0, 0], fp_tile=fp)
            return output

    pm = PassManager.get_strategy()
    optimized_program = pm.run_passes(ManualStoreFpCCEProgram)

    generator = codegen.CCECodegen()
    files = generator.generate(optimized_program)
    kernel_name = list(optimized_program.functions.values())[0].name
    code = files["kernels/aiv/" + kernel_name + ".cpp"]
    assert "TSTORE_FP(" in code
    assert "TileType::Scaling" in code
    assert "TASSIGN(outputGlobal" in code


def test_manual_store_fp_emits_cce_nz_global_tensor():
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.CCE)

    @pl.program
    class ManualStoreFpNZCCEProgram:
        @pl.function
        def store_fp_nz_cce_kernel(
            self,
            output: pl.Tensor[[32, 32], pl.INT8, pl.NZ],
        ) -> pl.Tensor[[32, 32], pl.INT8, pl.NZ]:
            src_type = plm.TileType(
                shape=[32, 32],
                dtype=pl.INT32,
                target_memory=pl.MemorySpace.Acc,
            )
            fp_type = plm.TileType(
                shape=[1, 16],
                dtype=pl.UINT64,
                target_memory=pl.MemorySpace.Scaling,
            )
            src = plm.make_tile(src_type, addr=0x0000, size=4096)
            fp = plm.make_tile(fp_type, addr=0x1000, size=128)
            plm.store(output, src, [0, 0], fp_tile=fp)
            return output

    pm = PassManager.get_strategy()
    optimized_program = pm.run_passes(ManualStoreFpNZCCEProgram)

    generator = codegen.CCECodegen()
    files = generator.generate(optimized_program)
    kernel_name = list(optimized_program.functions.values())[0].name
    code = files["kernels/aiv/" + kernel_name + ".cpp"]

    assert "TSTORE_FP(" in code
    assert "Layout::NZ" in code
    assert "pto::Shape<1, 1, 2, 16, 32>" in code
    assert "pto::Stride<1024, 1024, 512, 32, 1>" in code


def test_manual_store_fp_emits_cce_nz_global_tensor_in_single_file_codegen():
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.CCE)

    @pl.program
    class ManualStoreFpNZSingleFileCCEProgram:
        @pl.function
        def store_fp_nz_single_file_cce_kernel(
            self,
            output: pl.Tensor[[32, 32], pl.INT8, pl.NZ],
        ) -> pl.Tensor[[32, 32], pl.INT8, pl.NZ]:
            src_type = plm.TileType(
                shape=[32, 32],
                dtype=pl.INT32,
                target_memory=pl.MemorySpace.Acc,
            )
            fp_type = plm.TileType(
                shape=[1, 16],
                dtype=pl.UINT64,
                target_memory=pl.MemorySpace.Scaling,
            )
            src = plm.make_tile(src_type, addr=0x0000, size=4096)
            fp = plm.make_tile(fp_type, addr=0x1000, size=128)
            plm.store(output, src, [0, 0], fp_tile=fp)
            return output

    pm = PassManager.get_strategy()
    optimized_program = pm.run_passes(ManualStoreFpNZSingleFileCCEProgram)

    generator = codegen.CCECodegen()
    code = generator.generate_single(optimized_program, "a5")

    assert "TSTORE_FP(" in code
    assert "Layout::NZ" in code
    assert "pto::Shape<1, 1, 2, 16, 32>" in code
    assert "pto::Stride<1024, 1024, 512, 32, 1>" in code


def test_manual_move_fp_emits_cce_tmov_fp():
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.CCE)

    @pl.program
    class ManualMoveFpCCEProgram:
        @pl.function
        def move_fp_cce_kernel(self):
            src_type = plm.TileType(
                shape=[32, 32],
                dtype=pl.FP32,
                target_memory=pl.MemorySpace.Acc,
            )
            fp_type = plm.TileType(
                shape=[1, 16],
                dtype=pl.UINT64,
                target_memory=pl.MemorySpace.Scaling,
            )
            dst_type = plm.TileType(
                shape=[32, 32],
                dtype=pl.INT8,
                target_memory=pl.MemorySpace.Vec,
            )
            src = plm.make_tile(src_type, addr=0x0000, size=4096)
            fp = plm.make_tile(fp_type, addr=0x1000, size=128)
            dst = plm.make_tile(dst_type, addr=0x2000, size=1024)
            plm.move(dst, src, fp_tile=fp)

    pm = PassManager.get_strategy()
    optimized_program = pm.run_passes(ManualMoveFpCCEProgram)

    generator = codegen.CCECodegen()
    files = generator.generate(optimized_program)
    kernel_name = list(optimized_program.functions.values())[0].name
    kernel_path = next(path for path in files if path.endswith(kernel_name + ".cpp"))
    code = files[kernel_path]

    assert "TMOV_FP(" in code
    assert "TileType::Scaling" in code


def test_manual_move_fp_emits_cce_single_mode_template():
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.CCE)

    @pl.program
    class ManualMoveFpModeCCEProgram:
        @pl.function
        def move_fp_mode_cce_kernel(self):
            src_type = plm.TileType(
                shape=[32, 32],
                dtype=pl.FP32,
                target_memory=pl.MemorySpace.Acc,
            )
            fp_type = plm.TileType(
                shape=[1, 16],
                dtype=pl.UINT64,
                target_memory=pl.MemorySpace.Scaling,
            )
            dst_type = plm.TileType(
                shape=[32, 32],
                dtype=pl.INT8,
                target_memory=pl.MemorySpace.Vec,
            )
            src = plm.make_tile(src_type, addr=0x0000, size=4096)
            fp = plm.make_tile(fp_type, addr=0x1000, size=128)
            dst = plm.make_tile(dst_type, addr=0x2000, size=1024)
            plm.move(dst, src, fp_tile=fp, acc_to_vec_mode="single_vec0", relu_pre_mode="normal_relu")

    pm = PassManager.get_strategy()
    optimized_program = pm.run_passes(ManualMoveFpModeCCEProgram)

    generator = codegen.CCECodegen()
    files = generator.generate(optimized_program)
    kernel_name = list(optimized_program.functions.values())[0].name
    kernel_path = next(path for path in files if path.endswith(kernel_name + ".cpp"))
    code = files[kernel_path]

    assert "TMOV<decltype(" in code
    assert "AccToVecMode::SingleModeVec0" in code
    assert "ReluPreMode::NormalRelu" in code


class TestControlFlowCodegen:
    """Test control flow statement code generation."""

    def test_simple_for_loop(self):
        """Test simple for loop without iter_args."""
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.CCE)
        ib = IRBuilder()

        with ib.function("test_simple_for") as f:
            # Parameters
            input_tensor = f.param("input", ir.TensorType([128, 64], DataType.FP32))
            output_tensor = f.param("output", ir.TensorType([128, 64], DataType.FP32))
            f.return_type(ir.TensorType([128, 64], DataType.FP32))

            # Loop variable
            i = ib.var("i", ir.ScalarType(DataType.INT32))

            # Simple for loop: for i in range(0, 4, 1)
            with ib.for_loop(i, 0, 4, 1):
                # Load tile inside loop
                tile_x = ib.let("tile_x", block.load(input_tensor, [i, 0], [32, 64]))
                # Store tile back
                result = ib.let("result", block.store(tile_x, [i, 0], [32, 64], output_tensor))

            ib.return_stmt(result)

        func = f.get_result()
        program = ir.Program([func], "test_simple_for", ir.Span.unknown())
        generator = codegen.CCECodegen()
        files = generator.generate(program)
        code = files["kernels/aiv/test_simple_for.cpp"]

        # Verify for loop structure
        assert "for (uint64_t i = 0; i < 4; i += 1) {" in code
        assert "TLOAD(tile_x, inputGlobal)" in code
        assert "TSTORE(outputGlobal, tile_x)" in code

    def test_nested_for_loops(self):
        """Test nested for loops."""
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.CCE)
        ib = IRBuilder()

        with ib.function("test_nested_for") as f:
            # Parameters
            input_tensor = f.param("input", ir.TensorType([128, 128], DataType.FP32))
            output_tensor = f.param("output", ir.TensorType([128, 128], DataType.FP32))
            f.return_type(ir.TensorType([128, 128], DataType.FP32))

            # Outer loop variable
            i = ib.var("i", ir.ScalarType(DataType.INT32))
            # Inner loop variable
            j = ib.var("j", ir.ScalarType(DataType.INT32))

            # Nested for loops
            with ib.for_loop(i, 0, 4, 1):
                with ib.for_loop(j, 0, 4, 1):
                    # Load tile inside inner loop
                    tile_x = ib.let("tile_x", block.load(input_tensor, [i, j], [32, 32]))
                    # Store tile back
                    result = ib.let("result", block.store(tile_x, [i, j], [32, 32], output_tensor))

            ib.return_stmt(result)

        func = f.get_result()
        program = ir.Program([func], "test_nested_for", ir.Span.unknown())
        generator = codegen.CCECodegen()
        files = generator.generate(program)
        code = files["kernels/aiv/test_nested_for.cpp"]

        # Verify nested loop structure
        assert "for (uint64_t i = 0; i < 4; i += 1) {" in code
        assert "for (uint64_t j = 0; j < 4; j += 1) {" in code
        # Verify proper nesting (inner loop should appear after outer loop)
        assert code.index("for (uint64_t i") < code.index("for (uint64_t j")

    def test_if_statement_simple(self):
        """Test simple if statement code generation."""
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.CCE)
        span = ir.Span.unknown()

        # Build if statement directly using IR nodes
        condition = ir.ConstBool(True, span)

        # Then body: just an assignment
        x = ir.Var("x", ir.ScalarType(DataType.INT32), span)
        then_assign = ir.AssignStmt(x, ir.ConstInt(5, DataType.INT32, span), span)

        # Create if statement without else
        if_stmt = ir.IfStmt(condition, then_assign, None, [], span)

        # Create a simple function with the if statement
        ret_stmt = ir.ReturnStmt([], span)
        seq = ir.SeqStmts([if_stmt, ret_stmt], span)

        func = ir.Function("test_if", [], [ir.TensorType([1], DataType.FP32)], seq, span)
        program = ir.Program([func], "test_if", ir.Span.unknown())

        generator = codegen.CCECodegen()
        files = generator.generate(program)
        code = files["kernels/aiv/test_if.cpp"]

        # Verify if structure
        assert "if (true) {" in code or "if (1) {" in code
        assert "auto x = 5;" in code

    def test_if_else_statement(self):
        """Test if-else statement code generation."""
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.CCE)
        span = ir.Span.unknown()

        # Build condition
        a = ir.Var("a", ir.ScalarType(DataType.INT32), span)
        b = ir.Var("b", ir.ScalarType(DataType.INT32), span)
        condition = ir.Lt(a, b, DataType.INT32, span)

        # Then body
        x = ir.Var("x", ir.ScalarType(DataType.INT32), span)
        then_assign = ir.AssignStmt(x, ir.ConstInt(1, DataType.INT32, span), span)

        # Else body
        y = ir.Var("y", ir.ScalarType(DataType.INT32), span)
        else_assign = ir.AssignStmt(y, ir.ConstInt(2, DataType.INT32, span), span)

        # Create if-else statement
        if_stmt = ir.IfStmt(condition, then_assign, else_assign, [], span)

        # Create function
        # First assign a and b
        assign_a = ir.AssignStmt(a, ir.ConstInt(5, DataType.INT32, span), span)
        assign_b = ir.AssignStmt(b, ir.ConstInt(10, DataType.INT32, span), span)
        ret_stmt = ir.ReturnStmt([], span)
        seq = ir.SeqStmts([assign_a, assign_b, if_stmt, ret_stmt], span)

        func = ir.Function("test_if_else", [], [ir.TensorType([1], DataType.FP32)], seq, span)
        program = ir.Program([func], "test_if_else", ir.Span.unknown())

        generator = codegen.CCECodegen()
        files = generator.generate(program)
        code = files["kernels/aiv/test_if_else.cpp"]

        # Verify if-else structure
        assert "if ((a < b)) {" in code or "if (a < b) {" in code
        assert "} else {" in code
        assert "auto x = 1;" in code
        assert "auto y = 2;" in code


def test_debug_dump_tensor_dynamic_shape_codegen():
    """CCE debug.dump_tensor should emit a runtime GlobalTensor view for dynamic shapes."""
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.CCE)
    M = pl.DynVar("M")

    @pl.program
    class DebugDumpTensorShapeProgram:
        @pl.function
        def debug_dump_tensor_shape(
            self,
            input: pl.Tensor[[M, 32], pl.FP32],
            output: pl.Tensor[[M, 32], pl.FP32],
        ):
            rows = pl.tensor.dim(input, 0)
            plm.dump_tensor(input, offsets=[0, 0], shapes=[rows, 16])
            return output

    pm = PassManager.get_strategy()
    optimized_program = pm.run_passes(DebugDumpTensorShapeProgram)

    generator = codegen.CCECodegen()
    files = generator.generate(optimized_program)
    code = files["kernels/aiv/debug_dump_tensor_shape.cpp"]

    assert "using __debug_dump_tensor_shape_" in code
    assert "pto::Shape<1, 1, 1, -1, 16>" in code
    assert "GlobalTensor<float" in code
    assert "TPRINT(__debug_dump_tensor_view_" in code


def test_debug_dump_tensor_dynamic_window_codegen():
    """CCE debug.dump_tensor should preserve dynamic offsets in the runtime view."""
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.CCE)
    M = pl.DynVar("M")

    @pl.program
    class DebugDumpTensorWindowProgram:
        @pl.function
        def debug_dump_tensor_window(
            self,
            input: pl.Tensor[[M, 32], pl.FP32],
            row_off: pl.Scalar[pl.INDEX],
            output: pl.Tensor[[M, 32], pl.FP32],
        ):
            rows = pl.tensor.dim(input, 0)
            plm.dump_tensor(input, offsets=[row_off, 0], shapes=[rows, 16])
            return output

    pm = PassManager.get_strategy()
    optimized_program = pm.run_passes(DebugDumpTensorWindowProgram)

    generator = codegen.CCECodegen()
    files = generator.generate(optimized_program)
    code = files["kernels/aiv/debug_dump_tensor_window.cpp"]

    assert "__debug_dump_tensor_view_" in code
    assert "TPRINT(__debug_dump_tensor_view_" in code
    assert " + (" in code or " + " in code


def test_debug_dump_tensor_location_header_codegen():
    """CCE debug.dump_tensor with loc=True should emit a location header before the dump."""
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.CCE)

    @pl.program
    class DebugDumpTensorLocProgram:
        @pl.function
        def debug_dump_tensor_loc(
            self,
            input: pl.Tensor[[32, 32], pl.FP32],
            output: pl.Tensor[[32, 32], pl.FP32],
        ):
            plm.dump_tensor(input, loc=True)
            return output

    pm = PassManager.get_strategy()
    optimized_program = pm.run_passes(DebugDumpTensorLocProgram)

    generator = codegen.CCECodegen()
    files = generator.generate(optimized_program)
    code = files["kernels/aiv/debug_dump_tensor_loc.cpp"]

    assert re.search(r'cce::printf\("\[test_cce_codegen\.py:\d+\] dump_tensor\\n"\);', code)
    assert "TPRINT(" in code


def test_debug_dump_tensor_nz_codegen():
    """CCE debug.dump_tensor should emit an NZ GlobalTensor view for static NZ tensors."""
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.CCE)

    @pl.program
    class DebugDumpTensorNZProgram:
        @pl.function
        def debug_dump_tensor_nz(
            self,
            input: pl.Tensor[[32, 32], pl.FP32, pl.NZ],
            output: pl.Tensor[[32, 32], pl.FP32, pl.NZ],
        ):
            plm.dump_tensor(input)
            return output

    pm = PassManager.get_strategy()
    optimized_program = pm.run_passes(DebugDumpTensorNZProgram)

    generator = codegen.CCECodegen()
    files = generator.generate(optimized_program)
    code = files["kernels/aiv/debug_dump_tensor_nz.cpp"]

    assert "Layout::NZ" in code
    assert "pto::Shape<1, 4, 2, 16, 8>" in code
    assert "pto::Stride<1024, 256, 128, 8, 1>" in code
    assert "->start_offset" in code
    assert "TPRINT(__debug_dump_tensor_view_" in code


def test_debug_dump_tensor_nz_static_window_codegen():
    """CCE debug.dump_tensor should emit aligned static NZ windows with NZ layout metadata."""
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.CCE)

    @pl.program
    class DebugDumpTensorNZWindowProgram:
        @pl.function
        def debug_dump_tensor_nz_window(
            self,
            input: pl.Tensor[[32, 32], pl.FP32, pl.NZ],
            output: pl.Tensor[[32, 32], pl.FP32, pl.NZ],
        ):
            plm.dump_tensor(input, offsets=[16, 8], shapes=[16, 8])
            return output

    pm = PassManager.get_strategy()
    optimized_program = pm.run_passes(DebugDumpTensorNZWindowProgram)

    generator = codegen.CCECodegen()
    files = generator.generate(optimized_program)
    code = files["kernels/aiv/debug_dump_tensor_nz_window.cpp"]

    assert "Layout::NZ" in code
    assert "pto::Shape<1, 1, 1, 16, 8>" in code
    assert "pto::Stride<1024, 256, 128, 8, 1>" in code
    assert "->start_offset" in code
    assert "384" in code
    assert "TPRINT(__debug_dump_tensor_view_" in code


def test_debug_dump_tensor_nz_rejects_unaligned_window():
    """CCE debug.dump_tensor should reject non-aligned static NZ windows."""
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.CCE)

    @pl.program
    class DebugDumpTensorNZBadWindowProgram:
        @pl.function
        def debug_dump_tensor_nz_bad_window(
            self,
            input: pl.Tensor[[32, 32], pl.FP32, pl.NZ],
            output: pl.Tensor[[32, 32], pl.FP32, pl.NZ],
        ):
            plm.dump_tensor(input, offsets=[8, 0], shapes=[16, 8])
            return output

    pm = PassManager.get_strategy()
    optimized_program = pm.run_passes(DebugDumpTensorNZBadWindowProgram)

    generator = codegen.CCECodegen()
    with pytest.raises(ValueError, match="aligned static windows"):
        generator.generate(optimized_program)


def test_debug_dump_tile_dynamic_offset_codegen():
    """CCE dump_tile window lowering should emit runtime clamp logic and direct printing."""
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.CCE)

    @pl.program
    class DebugDumpTileOffsetProgram:
        @pl.function
        def debug_dump_tile_offset(
            self,
            input: pl.Tensor[[32, 32], pl.FP32],
            row_off: pl.Scalar[pl.INDEX],
            output: pl.Tensor[[32, 32], pl.FP32],
        ):
            tile = pl.load(input, offsets=[0, 0], shapes=[16, 16])
            plm.dump_tile(tile, offsets=[row_off, 0], shapes=[8, 16])
            return output

    pm = PassManager.get_strategy()
    optimized_program = pm.run_passes(DebugDumpTileOffsetProgram)

    generator = codegen.CCECodegen()
    files = generator.generate(optimized_program)
    code = files["kernels/aiv/debug_dump_tile_offset.cpp"]

    assert "GetValidRow()" in code
    assert "GetValidCol()" in code
    assert "pto::GetTileOffset" in code
    assert "pto::PrintValue(" in code
    assert 'cce::printf("=== [TPRINT Tile Window]' in code


def test_manual_tile_offset_codegen_emits_offset_tile_in_cce():
    """CCE codegen should support tile[offset] as a manual op tile operand."""
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.CCE)

    @pl.program
    class ManualTileOffsetCCEProgram:
        @pl.function
        def manual_tile_offset_cce(
            self,
            input: pl.Tensor[[4, 64], pl.FP32],
            output: pl.Tensor[[4, 64], pl.FP32],
        ) -> pl.Tensor[[4, 64], pl.FP32]:
            tile_type = plm.TileType(shape=[1, 64], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
            src = plm.make_tile(tile_type, addr=0x0000, size=1024)
            dst = plm.make_tile(tile_type, addr=0x0400, size=1024)
            with pl.section_vector():
                for row in pl.range(0, 4):
                    offset = row * 64
                    plm.load(src[offset], input, [row, 0])
                    plm.add(dst[offset], src[offset], src[offset])
                    plm.store(output, dst[offset], [row, 0])
            return output

    pm = PassManager.get_strategy()
    optimized_program = pm.run_passes(ManualTileOffsetCCEProgram)

    generator = codegen.CCECodegen()
    files = generator.generate(optimized_program)
    code = files["kernels/aiv/manual_tile_offset_cce.cpp"]

    assert "src_eoff_" in code
    assert "dst_eoff_" in code
    assert "TASSIGN(src_eoff_" in code
    assert "TASSIGN(dst_eoff_" in code
    assert re.search(r"TASSIGN\(src_eoff_\d+, .*\+ \(offset\) \* 4\);", code)
    assert re.search(r"TASSIGN\(dst_eoff_\d+, .*\+ \(offset\) \* 4\);", code)
    assert "TADD(" in code


def test_debug_dump_tile_location_header_codegen():
    """CCE dump_tile with loc=True should emit a location header before the dump."""
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.CCE)

    @pl.program
    class DebugDumpTileLocProgram:
        @pl.function
        def debug_dump_tile_loc(
            self,
            input: pl.Tensor[[32, 32], pl.FP32],
            output: pl.Tensor[[32, 32], pl.FP32],
        ):
            tile = pl.load(input, offsets=[0, 0], shapes=[16, 16])
            plm.dump_tile(tile, loc=True)
            return output

    pm = PassManager.get_strategy()
    optimized_program = pm.run_passes(DebugDumpTileLocProgram)

    generator = codegen.CCECodegen()
    files = generator.generate(optimized_program)
    code = files["kernels/aiv/debug_dump_tile_loc.cpp"]

    assert re.search(r'cce::printf\("\[test_cce_codegen\.py:\d+\] dump_tile\\n"\);', code)
    assert "TPRINT(" in code


def test_manual_store_emits_cce_nz_global_tensor():
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.CCE)

    @pl.program
    class ManualStoreNZProgram:
        @pl.function
        def manual_store_nz(
            self,
            output: pl.Tensor[[32, 32], pl.FP32, pl.NZ],
        ) -> pl.Tensor[[32, 32], pl.FP32, pl.NZ]:
            src_type = plm.TileType(
                shape=[32, 32],
                dtype=pl.FP32,
                target_memory=pl.MemorySpace.Acc,
            )
            src = plm.make_tile(src_type, addr=0x0000, size=4096)
            plm.store(output, src, [0, 0])
            return output

    pm = PassManager.get_strategy()
    optimized_program = pm.run_passes(ManualStoreNZProgram)

    generator = codegen.CCECodegen()
    files = generator.generate(optimized_program)
    code = files["kernels/aiv/manual_store_nz.cpp"]

    assert "TSTORE(" in code
    assert "Layout::NZ" in code
    assert "pto::Shape<1, 4, 2, 16, 8>" in code
    assert "pto::Stride<1024, 256, 128, 8, 1>" in code


def test_manual_store_emits_cce_nz_global_tensor_in_single_file_codegen():
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.CCE)

    @pl.program
    class ManualStoreNZSingleFileProgram:
        @pl.function
        def manual_store_nz_single_file(
            self,
            output: pl.Tensor[[32, 32], pl.FP32, pl.NZ],
        ) -> pl.Tensor[[32, 32], pl.FP32, pl.NZ]:
            src_type = plm.TileType(
                shape=[32, 32],
                dtype=pl.FP32,
                target_memory=pl.MemorySpace.Acc,
            )
            src = plm.make_tile(src_type, addr=0x0000, size=4096)
            plm.store(output, src, [0, 0])
            return output

    pm = PassManager.get_strategy()
    optimized_program = pm.run_passes(ManualStoreNZSingleFileProgram)

    generator = codegen.CCECodegen()
    code = generator.generate_single(optimized_program, "a5")

    assert "TSTORE(" in code
    assert "Layout::NZ" in code
    assert "pto::Shape<1, 4, 2, 16, 8>" in code
    assert "pto::Stride<1024, 256, 128, 8, 1>" in code


def test_manual_store_nz_rejects_non_zero_offsets():
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.CCE)

    @pl.program
    class ManualStoreNZOffsetProgram:
        @pl.function
        def manual_store_nz_offset(
            self,
            output: pl.Tensor[[32, 32], pl.FP32, pl.NZ],
        ) -> pl.Tensor[[32, 32], pl.FP32, pl.NZ]:
            src_type = plm.TileType(
                shape=[32, 32],
                dtype=pl.FP32,
                target_memory=pl.MemorySpace.Acc,
            )
            src = plm.make_tile(src_type, addr=0x0000, size=4096)
            plm.store(output, src, [16, 0])
            return output

    pm = PassManager.get_strategy()
    optimized_program = pm.run_passes(ManualStoreNZOffsetProgram)

    generator = codegen.CCECodegen()
    with pytest.raises(ValueError, match=r"offsets=\[0, 0\]"):
        generator.generate(optimized_program)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
