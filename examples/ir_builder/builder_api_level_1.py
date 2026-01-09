"""use context manager to simplify builder_api_level_0.py"""

from pypto.pypto_impl import ir
from pypto.ir.context import (
    function_scope, for_scope, if_then_scope, if_else_scope
)

def create_ir_module(name="main"):
    module = ir.module(name)
    builder = ir.IrBuilder(module)
    ctx = ir.IrBuilderContext()

    # ===== Signature =====
    batch = ir.Scalar(ir.DataType.int32, None, "batch")
    constant128 = ir.Scalar(ir.DataType.int64, 128, "const_128")
    tensor_shape = [batch, constant128]
    tile_shape = [128, 128]

    # NOTE: shape parameter `tensor_shape`, `tile_shape`, `batch` are passed via closure
    def create_function(
        builder,
        ctx,
        name="test_control",
        function_kind=ir.FunctionKind.ControlFlow
    ):
        input_x = ir.Tensor(tensor_shape, ir.DataType.float, "inputX", ir.Format.ND)
        input_y = ir.Tensor(tensor_shape, ir.DataType.float, "inputY", ir.Format.ND)
        scale1 = ir.Scalar(ir.DataType.float, None, "scale1")
        scale2 = ir.Scalar(ir.DataType.float, None, "scale2")
        result_x = ir.Tensor(tensor_shape, ir.DataType.float, "outputX", ir.Format.ND)
        result_y = ir.Tensor(tensor_shape, ir.DataType.float, "outputY", ir.Format.ND)

        sig = ir.FunctionSignature()
        sig.arguments = [input_x, input_y, scale1, scale2, result_x, result_y]
        sig.returns = [ir.Scalar(ir.DataType.int32, None)]

        func = builder.create_function(name, function_kind, sig, False)

        with function_scope(builder, ctx, func):
            i = builder.create_scalar(ctx, ir.DataType.int32, "i")
            constant0 = builder.create_const(ctx, 0, "const_0")
            constant1 = builder.create_const(ctx, 1, "const_1")

            # NOTE: type of i, constant0, constant1, batch are all `ir.Scalar`
            fs = builder.create_for(ctx, i, constant0, batch, constant1)
            fs.properties()["unroll"] = "4"

            with for_scope(builder, ctx, fs):
                res_loop_x = builder.create_tile(ctx, tile_shape, ir.DataType.float, "outputX")
                add_op_x = builder.create_binary_scalar_op(ir.Opcode.OP_ADDS, res_loop_x, scale1, res_loop_x)
                builder.emit(ctx, add_op_x)

                res_loop_y = builder.create_tile(ctx, tile_shape, ir.DataType.float, "outputY")
                add_op_y = builder.create_binary_scalar_op(ir.Opcode.OP_ADDS, res_loop_y, scale2, res_loop_y)
                builder.emit(ctx, add_op_y)

                ifs = builder.create_if(ctx, i)

                with if_then_scope(builder, ctx, ifs):
                    res_if_x = builder.create_tile(ctx, tile_shape, ir.DataType.float, "outputX")
                    mul_op_x = builder.create_binary_scalar_op(ir.Opcode.OP_MULS, res_loop_x, scale1, res_if_x)
                    builder.emit(ctx, mul_op_x)

                with if_else_scope(builder, ctx, ifs):
                    res_if_y = builder.create_tile(ctx, tile_shape, ir.DataType.float, "outputY")
                    mul_op_y = builder.create_binary_scalar_op(ir.Opcode.OP_MULS, res_loop_y, scale2, res_if_y)
                    builder.emit(ctx, mul_op_y)

                builder.exit_if(ctx, ifs)

            builder.create_return(ctx, [res_if_y])

        return func

    func = create_function(
        builder,
        ctx,
        name="test_control",
        function_kind=ir.FunctionKind.ControlFlow
    )
    module.entry = func
    return module


if __name__ == "__main__":
    module = create_ir_module()
    print(f"Module: {module}\nEntry: {module.entry}\nFunctions: {module.functions}")
