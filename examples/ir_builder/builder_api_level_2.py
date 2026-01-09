"""use ast parser to simplify `create_function` in builder_api_level_1.py"""

from pypto.pypto_impl import ir
from pypto.ir.context import (
    function_scope, for_scope, if_then_scope, if_else_scope
)
from pypto.ir.ast_transform import transform_and_run


def create_ir_module(name="main"):
    module = ir.module(name)
    builder = ir.IrBuilder(module)
    ctx = ir.IrBuilderContext()

    # ===== Signature =====
    sig = ir.FunctionSignature()
    batch = ir.Scalar(ir.DataType.int32, None, "batch")
    constant128 = ir.Scalar(ir.DataType.int64, 128, "const_128")
    tensor_shape = [batch, constant128]
    tile_shape = [128, 128]

    input_x = ir.Tensor(tensor_shape, ir.DataType.float, "inputX", ir.Format.ND)
    input_y = ir.Tensor(tensor_shape, ir.DataType.float, "inputY", ir.Format.ND)
    scale1 = ir.Scalar(ir.DataType.float, None, "scale1")
    scale2 = ir.Scalar(ir.DataType.float, None, "scale2")
    result_x = ir.Tensor(tensor_shape, ir.DataType.float, "outputX", ir.Format.ND)
    result_y = ir.Tensor(tensor_shape, ir.DataType.float, "outputY", ir.Format.ND)

    sig.arguments = [input_x, input_y, scale1, scale2, result_x, result_y]
    sig.returns = [ir.Scalar(ir.DataType.int32, None)]

    # NOTE: shape parameter `batch`, `tile_shape`, `scale1`, `scale2` are passed via closure
    def create_function(builder, ctx, name="test_control", function_kind=ir.FunctionKind.ControlFlow, sig=None):
        # The transformation will wrap the body in function_scope
        for i in pypto.block.loop(0, batch, step=1, unroll="4"):
            # TODO: can omit shape if IRBuilder supports automatic shape deduction
            # https://gitcode.com/cann/pypto/pull/123?ref=&did=4a5833a7938a8e00cbc6dc56688e79ff7a4f0235
            res_loop_x = pto.block.Tile(tile_shape, ir.DataType.float, "outputX")
            res_loop_x += scale1

            res_loop_y = pto.block.Tile(tile_shape, ir.DataType.float, "outputY")
            res_loop_y += scale2

            if i: # i is an ir.Scalar here
                res_if_x = pto.block.Tile(tile_shape, ir.DataType.float, "outputX")
                res_if_x = res_loop_x * scale1
            else:
                res_if_y = pto.block.Tile(tile_shape, ir.DataType.float, "outputY")
                res_if_y = res_loop_y * scale2

        return res_if_y

    closure_vars = {
        'batch': batch, 'tile_shape': tile_shape, 'scale1': scale1, 'scale2': scale2
        }
    transformed_func = transform_and_run(
        create_function, closure_vars, dump_transformed=True
        )
    func = transformed_func(builder, ctx, sig=sig)

    module.entry = func
    return module


if __name__ == "__main__":
    module = create_ir_module()
    print(f"Module: {module}\nEntry: {module.entry}\nFunctions: {module.functions}")
