import os
import ast
import inspect
from pypto.pypto_impl import ir
from pypto.blockgraph.builder_helper import BlockBuilderHelper
from pypto.blockgraph.ast_mutator import AstMutator


def ast_to_ir(metadata=None, closure_vars=None):
    """
    Decorator that transforms a function's AST and returns an IR function.
    
    Args:
        metadata: Dictionary containing function metadata (name, function_kind, etc.)
        closure_vars: Dictionary of closure variables needed by the function (e.g., tensor_shape, block, etc.)
    
    Returns:
        The IR function (ir.Function) instead of the original Python function.
    """
    # Capture the parameters to avoid closure issues
    provided_metadata = metadata
    provided_closure_vars = closure_vars or {}
    
    def decorator(func):
        # Get transformed AST (without executing)
        transformed_ast = AstMutator.mutate_ast(func)
        module_ast = ast.Module(body=[transformed_ast], type_ignores=[])
        ast.fix_missing_locations(module_ast)
        print("unparsed:\n", ast.unparse(module_ast))

        code = compile(module_ast, filename='<ast>', mode='exec')

        # Get the closure variables from the function
        func_closure_vars = inspect.getclosurevars(func)
        
        # The transformed function needs access to:
        # - Module-level globals (ir, BlockBuilderHelper, etc.) from func.__globals__
        # - Closure variables (explicitly provided + those from function closure)
        exec_namespace = {
            **func.__globals__,  # Include global imports (ir, BlockBuilderHelper, etc.)
            **func_closure_vars.nonlocals,  # Include closure vars from function
            **provided_closure_vars,  # Include explicitly provided closure vars
        }
        
        exec(code, exec_namespace)
        my_kernel_transformed = exec_namespace[func.__name__]
        
        # Use provided metadata or default
        if provided_metadata is None:
            func_metadata = {
                "name": func.__name__,
                "function_kind": ir.FunctionKind.ControlFlow
            }
        else:
            func_metadata = provided_metadata
        
        func_ir = my_kernel_transformed(func_metadata)
        return func_ir
    
    return decorator


def test_ast_to_ir_explicit():
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

    # Get transformed AST (without executing)
    transformed_ast = AstMutator.mutate_ast(my_kernel)
    module_ast = ast.Module(body=[transformed_ast], type_ignores=[])
    ast.fix_missing_locations(module_ast)
    print("unparsed:\n", ast.unparse(module_ast))

    code = compile(module_ast, filename='<ast>', mode='exec')

    # The transformed function needs access to:
    exec_namespace = {
        **globals(),  # Include global imports (ir, BlockBuilderHelper, etc.)
        **locals(),  # Include local vars `tensor_shape`, `tile_shape`, `batch`, `block`
    }
    exec(code, exec_namespace)
    my_kernel_transformed = exec_namespace["my_kernel"]
    metadata = {
        "name": "my_kernel",
        "function_kind": ir.FunctionKind.ControlFlow
    }
    func_ir = my_kernel_transformed(metadata)

    assert isinstance(func_ir, ir.Function)
    print("obtained ir.Function from ast!")
    # TODO: assert more information in `ir.Function` structure


def test_ast_to_ir_decorator():
    """
    Test AST transformation without executing the transformed code.
    Checks that the transformed AST contains expected patterns.
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
    @ast_to_ir(
        metadata=dict(name="my_kernel", function_kind=ir.FunctionKind.ControlFlow),
        closure_vars=dict(tensor_shape=tensor_shape, tile_shape=tile_shape, batch=batch, block=block)
    )
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

    assert isinstance(my_kernel, ir.Function)
    print("obtained ir.Function from ast!")
    # TODO: assert more information in `ir.Function` structure


if __name__ == "__main__":
    test_ast_to_ir_explicit()
    test_ast_to_ir_decorator()
