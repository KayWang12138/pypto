from pypto.pypto_impl import ir
from pypto.blockgraph.builder_helper import BlockBuilderHelper
from pypto.blockgraph.ast_mutator import ast_to_ir


def ast_to_ir_demo():
    module = ir.module("main")
    builder = ir.IrBuilder()
    ctx = ir.IrBuilderContext()
    block = BlockBuilderHelper(builder, ctx)

    batch = ir.Scalar(ir.DataType.int32, None, "batch")
    constant128 = ir.Scalar(ir.DataType.int64, 128, "const_128")
    tensor_shape = [batch, constant128]
    tile_shape = [128, 128]

    # NOTE: `block` helper and shape parameter `tensor_shape`, `tile_shape`, `batch` are passed via closure
    @ast_to_ir(
        metadata=dict(name="my_kernel", function_kind=ir.FunctionKind.ControlFlow),
        closure_vars=dict(
            tensor_shape=tensor_shape, tile_shape=tile_shape, batch=batch, block=block
        ),
        verbose=True,
    )
    def my_kernel(
        input_x: ir.Tensor(tensor_shape, ir.DataType.float, "inputX", ir.Format.ND),
        input_y: ir.Tensor(tensor_shape, ir.DataType.float, "inputY", ir.Format.ND),
        scale1: ir.Scalar(ir.DataType.float, None, "scale1"),
        scale2: ir.Scalar(ir.DataType.float, None, "scale2"),
        result_x: ir.Tensor(tensor_shape, ir.DataType.float, "outputX", ir.Format.ND),
        result_y: ir.Tensor(tensor_shape, ir.DataType.float, "outputY", ir.Format.ND),
    ) -> (ir.Scalar(ir.DataType.int32, None),):
        # NOTE: original low-level example does not use input_x/y to compute result_x/y
        #       will fix accordingly after the low-level example is fixed
        constant0 = block.Const(0, "const_0")
        constant1 = block.Const(1, "const_1")
        for i in block.loop(constant0, batch, constant1, unroll=4):
            res_loop_x = block.Tile(tile_shape, ir.DataType.float, "outputX")
            block.adds(res_loop_x, scale1, out=res_loop_x)

            res_loop_y = block.Tile(tile_shape, ir.DataType.float, "outputY")
            block.adds(res_loop_y, scale2, out=res_loop_y)

            if i:
                res_if_x = block.Tile(tile_shape, ir.DataType.float, "outputX")
                block.muls(res_loop_x, scale1, out=res_if_x)

            else:
                res_if_y = block.Tile(tile_shape, ir.DataType.float, "outputY")
                block.muls(res_loop_y, scale2, out=res_if_y)

        return (constant0,)

    module.add_function(my_kernel)
    module.entry = my_kernel
    assert isinstance(my_kernel, ir.Function)
    print(my_kernel)


if __name__ == "__main__":
    ast_to_ir_demo()
