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
from pypto.pypto_impl import ir
from pypto.blockgraph.builder_helper import BlockBuilderHelper


def test_control_flow():
    """
    1:1 ported from `test_control_flow` in `test_ir_binding.py`
    just hide `builder` and `ctx` behind ``BlockBuilderHelper
    """

    # ===== Module =====
    module = ir.module("main")
    builder = ir.IrBuilder()
    ctx = ir.IrBuilderContext()
    block = BlockBuilderHelper(builder, ctx)

    # ===== Signature =====
    sig = ir.FunctionSignature()

    # tensor<[batch, 128], float32>
    # Passing None to Scalar indicates a symbolic/non-immediate value
    batch = ir.Scalar(ir.DataType.int32, None, "batch")
    constant128 = ir.Scalar(ir.DataType.int64, 128, "const_128")
    tensor_shape = [batch, constant128]
    tile_shape = [128, 128]

    input_x = ir.Tensor(tensor_shape, ir.DataType.float,
                        "inputX", ir.Format.ND)
    input_y = ir.Tensor(tensor_shape, ir.DataType.float,
                        "inputY", ir.Format.ND)
    scale1 = ir.Scalar(ir.DataType.float, None, "scale1")
    scale2 = ir.Scalar(ir.DataType.float, None, "scale2")

    result_x = ir.Tensor(tensor_shape, ir.DataType.float,
                         "outputX", ir.Format.ND)
    result_y = ir.Tensor(tensor_shape, ir.DataType.float,
                         "outputY", ir.Format.ND)

    sig.arguments = [input_x, input_y, scale1, scale2, result_x, result_y]
    sig.returns = [ir.Scalar(ir.DataType.int32, None)]

    # ===== Function =====
    func = block.create_function("test_control", ir.FunctionKind.ControlFlow, sig)
    module.add_function(func)
    module.entry = func  # NOTE: now runs until here

    # Enter function body scope
    with block.function_scope(func):

        # for i = 0 to batch step 1
        i = block.Scalar(ir.DataType.int32, "i")
        constant0 = block.Const(0, "const_0")
        constant1 = block.Const(1, "const_1")
        fs = block.ForNode(i, constant0, batch, constant1, unroll=4)
        with block.for_scope(fs):

            res_loop_x = block.Tile(tile_shape, ir.DataType.float, "outputX")
            # Note: create_op used as CreateBinaryOp placeholder
            add_op_x = block.adds(res_loop_x, scale1, out=res_loop_x)

            res_loop_y = block.Tile(tile_shape, ir.DataType.float, "outputY")
            add_op_y = block.adds(res_loop_y, scale2, out=res_loop_y)

            # if i then outputX = mul(outputX, scale1) else outputY = mul(outputY, scale2)
            ifs = block.IfNode(i)

            # --- IF THEN ---
            with block.if_then_scope(ifs):
                res_if_x = block.Tile(tile_shape, ir.DataType.float, "outputX")
                mul_op_x = block.muls(res_loop_x, scale1, out=res_if_x)

                # test compound remove value (Assuming remove_var exists in binding)
                then_comp = ifs.then_stmts()
                assert then_comp.vars()["outputX"] == res_if_x

            # --- IF ELSE ---
            with block.if_else_scope(ifs):
                res_if_y = block.Tile(tile_shape, ir.DataType.float, "outputY")
                mul_op_y = block.muls(res_loop_y, scale2, out=res_if_y)

            block.exit_if(ifs)

            # Check yields
            then_yield = then_comp.stmts()[-1]
            then_yield_set = set(then_yield.values())
            assert then_yield_set == {res_if_x, res_loop_y}

            else_comp = ifs.else_stmts()
            else_yield = else_comp.stmts()[-1]
            else_yield_set = set(else_yield.values())
            assert else_yield_set == {res_loop_x, res_if_y}
            assert else_yield.values()[0] == res_if_y
            assert else_yield.values()[1] == res_loop_x

        # Check for-yield results
        # Accessing the second statement in the for compound
        ifs_in_for = fs.stmts().stmts()[1]
        # assert set(ifs_in_for.results()) == set(fs.yield().values())

        block.create_return([constant0])

    # TODO: assert IR module structure
    print(f"Module: {module}\nEntry: {module.entry}\nFunctions: {module.functions}")


def test_control_flow_closure():
    """
    Rearrange `test_control_flow_rearrange` to a more functional style.
    Further ast transforms will work on `create_function` level, not module level.
    """
    module = ir.module("main")
    builder = ir.IrBuilder()
    ctx = ir.IrBuilderContext()
    block = BlockBuilderHelper(builder, ctx)

    batch = ir.Scalar(ir.DataType.int32, None, "batch")
    constant128 = ir.Scalar(ir.DataType.int64, 128, "const_128")
    tensor_shape = [batch, constant128]
    tile_shape = [128, 128]

    # NOTE: shape parameter `tensor_shape`, `tile_shape`, `batch` are passed via closure
    def create_function(
        block=block,
        name="test_control",
        function_kind=ir.FunctionKind.ControlFlow
    ):
        input_x = ir.Tensor(tensor_shape, ir.DataType.float,
                            "inputX", ir.Format.ND)
        input_y = ir.Tensor(tensor_shape, ir.DataType.float,
                            "inputY", ir.Format.ND)
        scale1 = ir.Scalar(ir.DataType.float, None, "scale1")
        scale2 = ir.Scalar(ir.DataType.float, None, "scale2")
        result_x = ir.Tensor(tensor_shape, ir.DataType.float,
                            "outputX", ir.Format.ND)
        result_y = ir.Tensor(tensor_shape, ir.DataType.float,
                            "outputY", ir.Format.ND)

        sig = ir.FunctionSignature()
        sig.arguments = [input_x, input_y, scale1, scale2, result_x, result_y]
        sig.returns = [ir.Scalar(ir.DataType.int32, None)]

        func = block.create_function(name, function_kind, sig)
        with block.function_scope(func):

            # for i = 0 to batch step 1
            i = block.Scalar(ir.DataType.int32, "i")
            constant0 = block.Const(0, "const_0")
            constant1 = block.Const(1, "const_1")
            fs = block.ForNode(i, constant0, batch, constant1, unroll=4)
            with block.for_scope(fs):
                res_loop_x = block.Tile(tile_shape, ir.DataType.float, "outputX")
                add_op_x = block.adds(res_loop_x, scale1, out=res_loop_x)

                res_loop_y = block.Tile(tile_shape, ir.DataType.float, "outputY")
                add_op_y = block.adds(res_loop_y, scale2, out=res_loop_y)

                ifs = block.IfNode(i)
                with block.if_then_scope(ifs):
                    res_if_x = block.Tile(tile_shape, ir.DataType.float, "outputX")
                    mul_op_x = block.muls(res_loop_x, scale1, out=res_if_x)

                with block.if_else_scope(ifs):
                    res_if_y = block.Tile(tile_shape, ir.DataType.float, "outputY")
                    mul_op_y = block.muls(res_loop_y, scale2, out=res_if_y)

                block.exit_if(ifs)

            block.create_return([constant0])

        return func

    func = create_function(
        block,
        name="test_control",
        function_kind=ir.FunctionKind.ControlFlow
    )
    module.add_function(func)
    module.entry = func

    # TODO: assert IR module structure
    print(f"Module: {module}\nEntry: {module.entry}\nFunctions: {module.functions}")


if __name__ == "__main__":
    test_control_flow()
    test_control_flow_closure()
