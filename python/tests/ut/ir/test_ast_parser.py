import os
from inspect import BlockFinder
from pypto.pypto_impl import ir
from pypto.blockgraph.builder_helper import BlockBuilderHelper
from pypto.blockgraph.ast_mutator import AstMutator

# Global switch to control dumping transformed source code
DUMP_TRANSFORMED_SOURCE = True
DUMP_DIR = "./temp"


def _get_dump_path(test_name: str) -> str:
    """Get dump path for transformed source code."""
    if not DUMP_TRANSFORMED_SOURCE:
        return None
    os.makedirs(DUMP_DIR, exist_ok=True)
    return os.path.join(DUMP_DIR, f"{test_name}_transformed.py")


def test_ast_transform():
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

    # NOTE: `block` helper and shape parameter `tensor_shape`, `tile_shape`, `batch` are passed via closure
    def my_kernel(
        input_x: ir.Tensor(tensor_shape, ir.DataType.float, "inputX", ir.Format.ND),
        input_y: ir.Tensor(tensor_shape, ir.DataType.float, "inputY", ir.Format.ND),
        scale1: ir.Scalar(ir.DataType.float, None, "scale1"),
        scale2: ir.Scalar(ir.DataType.float, None, "scale2"),
        result_x: ir.Tensor(tensor_shape, ir.DataType.float, "outputX", ir.Format.ND),
        result_y: ir.Tensor(tensor_shape, ir.DataType.float, "outputY", ir.Format.ND)
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

    transformed_ast = AstMutator.mutate(my_kernel, dump_source=_get_dump_path("test_ast_transform"))

    metadata=dict(
        name="test_control",
        function_kind=ir.FunctionKind.ControlFlow
    )
    func = transformed_ast(metadata=metadata)  # TODO: properly inject closure vars
    module.add_function(func)
    module.entry = func
    # TODO: assert IR module structure
    print(f"Module: {module}\nEntry: {module.entry}\nFunctions: {module.functions}")


def test_nested_for_loops():
    """Test nested for loops transformation."""
    module = ir.module("main")
    builder = ir.IrBuilder()
    ctx = ir.IrBuilderContext()
    block = BlockBuilderHelper(builder, ctx)

    batch = ir.Scalar(ir.DataType.int32, None, "batch")
    constant128 = ir.Scalar(ir.DataType.int64, 128, "const_128")
    tensor_shape = [batch, constant128]
    tile_shape = [128, 128]

    def my_kernel(
        input_x: ir.Tensor(tensor_shape, ir.DataType.float, "inputX", ir.Format.ND),
    ) -> (ir.Scalar(ir.DataType.int32, None),):
        constant0 = block.Const(0, "const_0")
        constant1 = block.Const(1, "const_1")
        for i in block.loop(constant0, batch, constant1):
            for j in block.loop(constant0, constant128, constant1):
                res = block.Tile(tile_shape, ir.DataType.float, "output")
                block.adds(res, input_x, out=res)
        return (constant0,)

    transformed_ast = AstMutator.mutate(my_kernel, dump_source=_get_dump_path("test_nested_for_loops"))
    metadata = dict(
        name="test_nested_for",
        function_kind=ir.FunctionKind.ControlFlow
    )
    func = transformed_ast(metadata=metadata)
    module.add_function(func)
    module.entry = func
    assert func is not None


def test_nested_if_statements():
    """Test nested if statements transformation."""
    module = ir.module("main")
    builder = ir.IrBuilder()
    ctx = ir.IrBuilderContext()
    block = BlockBuilderHelper(builder, ctx)

    batch = ir.Scalar(ir.DataType.int32, None, "batch")
    constant128 = ir.Scalar(ir.DataType.int64, 128, "const_128")
    tensor_shape = [batch, constant128]
    tile_shape = [128, 128]

    def my_kernel(
        input_x: ir.Tensor(tensor_shape, ir.DataType.float, "inputX", ir.Format.ND),
        cond1: ir.Scalar(ir.DataType.bool, None, "cond1"),
        cond2: ir.Scalar(ir.DataType.bool, None, "cond2"),
    ) -> (ir.Scalar(ir.DataType.int32, None),):
        constant0 = block.Const(0, "const_0")
        if cond1:
            res1 = block.Tile(tile_shape, ir.DataType.float, "output1")
            if cond2:
                res2 = block.Tile(tile_shape, ir.DataType.float, "output2")
                block.adds(res1, res2, out=res1)
            else:
                block.adds(res1, input_x, out=res1)
        else:
            res3 = block.Tile(tile_shape, ir.DataType.float, "output3")
        return (constant0,)

    transformed_ast = AstMutator.mutate(my_kernel, dump_source=_get_dump_path("test_nested_if_statements"))
    metadata = dict(
        name="test_nested_if",
        function_kind=ir.FunctionKind.ControlFlow
    )
    func = transformed_ast(metadata=metadata)
    module.add_function(func)
    module.entry = func
    assert func is not None


def test_for_with_nested_if():
    """Test for loop with nested if statement."""
    module = ir.module("main")
    builder = ir.IrBuilder()
    ctx = ir.IrBuilderContext()
    block = BlockBuilderHelper(builder, ctx)

    batch = ir.Scalar(ir.DataType.int32, None, "batch")
    constant128 = ir.Scalar(ir.DataType.int64, 128, "const_128")
    tensor_shape = [batch, constant128]
    tile_shape = [128, 128]

    def my_kernel(
        input_x: ir.Tensor(tensor_shape, ir.DataType.float, "inputX", ir.Format.ND),
    ) -> (ir.Scalar(ir.DataType.int32, None),):
        constant0 = block.Const(0, "const_0")
        constant1 = block.Const(1, "const_1")
        for i in block.loop(constant0, batch, constant1):
            res = block.Tile(tile_shape, ir.DataType.float, "output")
            if i:
                block.adds(res, input_x, out=res)
            else:
                block.muls(res, input_x, out=res)
        return (constant0,)

    transformed_ast = AstMutator.mutate(my_kernel, dump_source=_get_dump_path("test_for_with_nested_if"))
    metadata = dict(
        name="test_for_nested_if",
        function_kind=ir.FunctionKind.ControlFlow
    )
    func = transformed_ast(metadata=metadata)
    module.add_function(func)
    module.entry = func
    assert func is not None


def test_if_with_nested_for():
    """Test if statement with nested for loop."""
    module = ir.module("main")
    builder = ir.IrBuilder()
    ctx = ir.IrBuilderContext()
    block = BlockBuilderHelper(builder, ctx)

    batch = ir.Scalar(ir.DataType.int32, None, "batch")
    constant128 = ir.Scalar(ir.DataType.int64, 128, "const_128")
    tensor_shape = [batch, constant128]
    tile_shape = [128, 128]

    def my_kernel(
        input_x: ir.Tensor(tensor_shape, ir.DataType.float, "inputX", ir.Format.ND),
        cond: ir.Scalar(ir.DataType.bool, None, "cond"),
    ) -> (ir.Scalar(ir.DataType.int32, None),):
        constant0 = block.Const(0, "const_0")
        constant1 = block.Const(1, "const_1")
        if cond:
            for i in block.loop(constant0, batch, constant1):
                res = block.Tile(tile_shape, ir.DataType.float, "output")
                block.adds(res, input_x, out=res)
        else:
            res = block.Tile(tile_shape, ir.DataType.float, "output")
        return (constant0,)

    transformed_ast = AstMutator.mutate(my_kernel, dump_source=_get_dump_path("test_if_with_nested_for"))
    metadata = dict(
        name="test_if_nested_for",
        function_kind=ir.FunctionKind.ControlFlow
    )
    func = transformed_ast(metadata=metadata)
    module.add_function(func)
    module.entry = func
    assert func is not None


def test_if_without_else():
    """Test if statement without else block."""
    module = ir.module("main")
    builder = ir.IrBuilder()
    ctx = ir.IrBuilderContext()
    block = BlockBuilderHelper(builder, ctx)

    batch = ir.Scalar(ir.DataType.int32, None, "batch")
    constant128 = ir.Scalar(ir.DataType.int64, 128, "const_128")
    tensor_shape = [batch, constant128]
    tile_shape = [128, 128]

    def my_kernel(
        input_x: ir.Tensor(tensor_shape, ir.DataType.float, "inputX", ir.Format.ND),
        cond: ir.Scalar(ir.DataType.bool, None, "cond"),
    ) -> (ir.Scalar(ir.DataType.int32, None),):
        constant0 = block.Const(0, "const_0")
        res = block.Tile(tile_shape, ir.DataType.float, "output")
        if cond:
            block.adds(res, input_x, out=res)
        return (constant0,)

    transformed_ast = AstMutator.mutate(my_kernel, dump_source=_get_dump_path("test_if_without_else"))
    metadata = dict(
        name="test_if_no_else",
        function_kind=ir.FunctionKind.ControlFlow
    )
    func = transformed_ast(metadata=metadata)
    module.add_function(func)
    module.entry = func
    assert func is not None


if __name__ == "__main__":
    test_ast_transform()
    test_nested_for_loops()
    test_nested_if_statements()
    test_for_with_nested_if()
    test_if_with_nested_for()
    test_if_without_else()