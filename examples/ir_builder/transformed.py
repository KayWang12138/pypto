"""Auto-generated from `builder_api_level_3.py`, just for record"""

def my_kernel(builder, ctx, name='test_control', function_kind=ir.FunctionKind.ControlFlow):
    input_x = ir.Tensor(tensor_shape, ir.DataType.float, 'inputX', ir.Format.ND)
    input_y = ir.Tensor(tensor_shape, ir.DataType.float, 'inputY', ir.Format.ND)
    scale1 = ir.Scalar(ir.DataType.float, None, 'scale1')
    scale2 = ir.Scalar(ir.DataType.float, None, 'scale2')
    result_x = ir.Tensor(tensor_shape, ir.DataType.float, 'outputX', ir.Format.ND)
    result_y = ir.Tensor(tensor_shape, ir.DataType.float, 'outputY', ir.Format.ND)
    _return_0 = ir.Scalar(ir.DataType.int32, None)
    sig = ir.FunctionSignature()
    func = builder.create_function(name, function_kind, sig, False)
    with function_scope(builder, ctx, func):
        i = builder.create_scalar(ctx, ir.DataType.int32, 'i')
        constant_gen_0 = builder.create_const(ctx, 0, 'const_0')
        constant_gen_1 = builder.create_const(ctx, 1, 'const_1')
        fs_0 = builder.create_for(ctx, i, constant_gen_0, batch, constant_gen_1)
        fs_0.properties()['unroll'] = '4'
        with for_scope(builder, ctx, fs_0):
            res_loop_x = builder.create_tile(ctx, tile_shape, ir.DataType.float, 'outputX')
            op_tmp = builder.create_binary_scalar_op(ir.Opcode.OP_ADDS, res_loop_x, scale1, res_loop_x)
            builder.emit(ctx, op_tmp)
            res_loop_y = builder.create_tile(ctx, tile_shape, ir.DataType.float, 'outputY')
            op_tmp = builder.create_binary_scalar_op(ir.Opcode.OP_ADDS, res_loop_y, scale2, res_loop_y)
            builder.emit(ctx, op_tmp)
            ifs_0 = builder.create_if(ctx, i)
            with if_then_scope(builder, ctx, ifs_0):
                res_if_x = builder.create_tile(ctx, tile_shape, ir.DataType.float, 'outputX')
                op_tmp = builder.create_binary_scalar_op(ir.Opcode.OP_MULS, res_loop_x, scale1, res_if_x)
                builder.emit(ctx, op_tmp)
            with if_else_scope(builder, ctx, ifs_0):
                res_if_y = builder.create_tile(ctx, tile_shape, ir.DataType.float, 'outputY')
                op_tmp = builder.create_binary_scalar_op(ir.Opcode.OP_MULS, res_loop_y, scale2, res_if_y)
                builder.emit(ctx, op_tmp)
            builder.exit_if(ctx, ifs_0)
        builder.create_return(ctx, [res_if_y])
    return func