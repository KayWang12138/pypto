# same as python/tests/ut/ir/test_ir_binding.py

from pypto.pypto_impl import ir


def create_ir_module(name="main"):
    module = ir.module(name)
    builder = ir.IrBuilder(module)
    ctx = ir.IrBuilderContext()

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
    func = builder.create_function(
        "test_control", ir.FunctionKind.ControlFlow, sig, False)
    module.entry = func

    # Enter function body scope
    builder.enter_function(ctx, func)

    # for i = 0 to batch step 1
    i = builder.create_scalar(ctx, ir.DataType.int32, "i")
    constant0 = builder.create_const(ctx, 0, "const_0")
    constant1 = builder.create_const(ctx, 1, "const_1")
    fs = builder.create_for(ctx, i, constant0, batch, constant1)
    fs.properties()["unroll"] = "4"

    builder.enter_for(ctx, fs)

    res_loop_x = builder.create_tile(
        ctx, tile_shape, ir.DataType.float, "outputX")
    add_op_x = builder.create_binary_scalar_op(
        ir.Opcode.OP_ADDS, res_loop_x, scale1, res_loop_x)
    builder.emit(ctx, add_op_x)

    res_loop_y = builder.create_tile(
        ctx, tile_shape, ir.DataType.float, "outputY")
    add_op_y = builder.create_binary_scalar_op(
        ir.Opcode.OP_ADDS, res_loop_y, scale2, res_loop_y)
    builder.emit(ctx, add_op_y)

    ifs = builder.create_if(ctx, i)

    # --- IF THEN ---
    builder.enter_if_then(ctx, ifs)
    res_if_x = builder.create_tile(
        ctx, tile_shape, ir.DataType.float, "outputX")
    mul_op_x = builder.create_binary_scalar_op(
        ir.Opcode.OP_MULS, res_loop_x, scale1, res_if_x)
    builder.emit(ctx, mul_op_x)

    ctx.pop_scope()

    # --- IF ELSE ---
    builder.enter_if_else(ctx, ifs)
    res_if_y = builder.create_tile(
        ctx, tile_shape, ir.DataType.float, "outputY")
    mul_op_y = builder.create_binary_scalar_op(
        ir.Opcode.OP_MULS, res_loop_y, scale2, res_if_y)
    builder.emit(ctx, mul_op_y)

    ctx.pop_scope()

    builder.exit_if(ctx, ifs)

    ctx.pop_scope()  # for-body
    builder.exit_for(ctx, fs)

    builder.create_return(ctx, [constant0])

    ctx.pop_scope()  # function-body
    return module


if __name__ == "__main__":
    module = create_ir_module()

    # TODO: assert module attributes and structure
    print(f"Module: {module}\nEntry: {module.entry}\nFunctions: {module.functions}")
