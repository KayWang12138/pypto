import inspect
import ast
import textwrap


class FunctionAstMutator(ast.NodeTransformer):
    def __init__(self):
        self.const_count = 0

    def _get_const_name(self):
        name = f"constant_gen_{self.const_count}"
        self.const_count += 1
        return name

    def visit_FunctionDef(self, node):
        # 1. Transform the body first
        self.generic_visit(node)

        # 2. Wrap the transformed body in 'with function_scope(builder, ctx, func):'
        # First, add the func creation line at the top
        func_init = ast.parse("func = builder.create_function(name, function_kind, sig, False)").body[0]

        wrapped_body = ast.With(
            items=[ast.withitem(context_expr=ast.Call(
                func=ast.Name(id='function_scope', ctx=ast.Load()),
                args=[ast.Name(id='builder', ctx=ast.Load()),
                      ast.Name(id='ctx', ctx=ast.Load()),
                      ast.Name(id='func', ctx=ast.Load())],
                keywords=[]
            ))],
            body=node.body
        )

        node.body = [func_init, wrapped_body, ast.Return(value=ast.Name(id='func', ctx=ast.Load()))]
        return node

    def visit_For(self, node):
        self.generic_visit(node)
        # Check if it's our DSL loop: for i in pypto.block.loop(...)
        if isinstance(node.iter, ast.Call):
            # i = builder.create_scalar(ctx, ir.DataType.int32, "i")
            target_id = node.target.id
            scalar_init = ast.parse(f"{target_id} = builder.create_scalar(ctx, ir.DataType.int32, '{target_id}')").body[0]

            # Extract loop args: start, stop, step, unroll
            args = node.iter.args
            keywords = {kw.arg: kw.value for kw in node.iter.keywords}

            start_val = args[0].value if isinstance(args[0], ast.Constant) else 0
            step_val = keywords['step'].value if 'step' in keywords else 1
            unroll_val = keywords['unroll'].value if 'unroll' in keywords else "1"

            # Generate constants
            c0_name = self._get_const_name()
            c1_name = self._get_const_name()
            c0_init = ast.parse(f"{c0_name} = builder.create_const(ctx, {start_val}, 'const_{start_val}')").body[0]
            c1_init = ast.parse(f"{c1_name} = builder.create_const(ctx, {step_val}, 'const_{step_val}')").body[0]

            # fs = builder.create_for(...)
            fs_init = ast.parse(f"fs = builder.create_for(ctx, {target_id}, {c0_name}, {args[1].id}, {c1_name})").body[0]
            unroll_attr = ast.parse(f"fs.properties()['unroll'] = '{unroll_val}'").body[0]

            # with for_scope...
            for_with = ast.With(
                items=[ast.withitem(context_expr=ast.Call(
                    func=ast.Name(id='for_scope', ctx=ast.Load()),
                    args=[ast.Name(id='builder', ctx=ast.Load()),
                          ast.Name(id='ctx', ctx=ast.Load()),
                          ast.Name(id='fs', ctx=ast.Load())],
                    keywords=[]
                ))],
                body=node.body
            )
            return [scalar_init, c0_init, c1_init, fs_init, unroll_attr, for_with]
        return node

    def visit_If(self, node):
        self.generic_visit(node)
        # Requirement: only transform if condition is an ir.Scalar (assumed if it's a Name node in this DSL)
        if isinstance(node.test, ast.Name):
            ifs_init = ast.parse(f"ifs = builder.create_if(ctx, {node.test.id})").body[0]

            then_scope = ast.With(
                items=[ast.withitem(context_expr=ast.Call(
                    func=ast.Name(id='if_then_scope', ctx=ast.Load()),
                    args=[ast.Name(id='builder', ctx=ast.Load()),
                          ast.Name(id='ctx', ctx=ast.Load()),
                          ast.Name(id='ifs', ctx=ast.Load())],
                    keywords=[]
                ))],
                body=node.body
            )

            res = [ifs_init, then_scope]

            if node.orelse:
                else_scope = ast.With(
                    items=[ast.withitem(context_expr=ast.Call(
                        func=ast.Name(id='if_else_scope', ctx=ast.Load()),
                        args=[ast.Name(id='builder', ctx=ast.Load()),
                              ast.Name(id='ctx', ctx=ast.Load()),
                              ast.Name(id='ifs', ctx=ast.Load())],
                        keywords=[]
                    ))],
                    body=node.orelse
                )
                res.append(else_scope)

            res.append(ast.parse("builder.exit_if(ctx, ifs)").body[0])
            return res
        return node

    def visit_Assign(self, node):
        # Handle Tile creation and Binary Ops
        if isinstance(node.value, ast.Call) and hasattr(node.value.func, 'attr') and node.value.func.attr == 'Tile':
            # res = pto.block.Tile(...) -> res = builder.create_tile(...)
            node.value.func = ast.Attribute(value=ast.Name(id='builder', ctx=ast.Load()), attr='create_tile', ctx=ast.Load())
            node.value.args.insert(0, ast.Name(id='ctx', ctx=ast.Load()))
            return node

        if isinstance(node.value, ast.BinOp):
            # res = a * b -> mul_op = builder.create_binary_scalar_op(...)
            op_map = {ast.Add: 'OP_ADDS', ast.Mult: 'OP_MULS'}
            opcode = op_map.get(type(node.value.op))
            if opcode:
                target_name = node.targets[0].id
                left = node.value.left.id
                right = node.value.right.id
                new_nodes = ast.parse(
                    f"op_tmp = builder.create_binary_scalar_op(ir.Opcode.{opcode}, {left}, {right}, {target_name})\n"
                    f"builder.emit(ctx, op_tmp)"
                ).body
                return new_nodes
        return node

    def visit_AugAssign(self, node):
        # res += scale -> add_op = builder.create_binary_scalar_op(...)
        op_map = {ast.Add: 'OP_ADDS', ast.Mult: 'OP_MULS'}
        opcode = op_map.get(type(node.op))
        if opcode:
            target_name = node.target.id
            value_name = node.value.id
            return ast.parse(
                f"op_tmp = builder.create_binary_scalar_op(ir.Opcode.{opcode}, {target_name}, {value_name}, {target_name})\n"
                f"builder.emit(ctx, op_tmp)"
            ).body
        return node

    def visit_Return(self, node):
        # return x -> builder.create_return(ctx, [x])
        val = node.value.id if isinstance(node.value, ast.Name) else "None"
        return ast.parse(f"builder.create_return(ctx, [{val}])").body[0]


def transform_and_run(high_level_func, closure_vars):
    from pypto.pypto_impl import ir
    from pypto.ir.context import (
        function_scope, for_scope, if_then_scope, if_else_scope
    )

    raw_source = inspect.getsource(high_level_func)
    source = textwrap.dedent(raw_source)  #  Remove common leading whitespace (Fixes IndentationError)
    tree = ast.parse(source)

    transformer = FunctionAstMutator()
    new_tree = transformer.visit(tree)
    ast.fix_missing_locations(new_tree)

    # Unparse for inspection
    print("=== Transformed Source Code ===")
    print(ast.unparse(new_tree))
    print("===============================\n")

    # Compile and execute in the context of the closure
    code = compile(new_tree, filename="<ast>", mode="exec")
    namespace = {**locals(), **closure_vars}
    exec(code, namespace)

    return namespace['create_function']
