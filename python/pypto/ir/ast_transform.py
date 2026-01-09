import inspect
import ast
import textwrap


class FunctionAstMutator(ast.NodeTransformer):
    def __init__(self):
        self.const_count = 0
        # Track which variables are Tiles (created via pto.block.Tile)
        # NOTE: cannot run `isinstance(var, ir.Tile)` during pure ast parsing phase
        self.tile_vars = set()
        # Counters for unique variable names in nested control flow
        self.ifs_count = 0
        self.fs_count = 0

    def _get_const_name(self):
        name = f"constant_gen_{self.const_count}"
        self.const_count += 1
        return name
    
    def _get_ifs_name(self):
        """Get unique name for if statement variable"""
        name = f"ifs_{self.ifs_count}"
        self.ifs_count += 1
        return name
    
    def _get_fs_name(self):
        """Get unique name for for statement variable"""
        name = f"fs_{self.fs_count}"
        self.fs_count += 1
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
        # Check if it's our DSL loop: for i in pypto.block.loop(...)
        if isinstance(node.iter, ast.Call):
            # Get unique name for for statement variable BEFORE visiting children
            # This ensures outer for loops get lower numbers than inner ones
            fs_name = self._get_fs_name()
            
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

            # Now visit children (which may contain nested if/for statements)
            self.generic_visit(node)
            
            # fs = builder.create_for(...)
            fs_init = ast.parse(f"{fs_name} = builder.create_for(ctx, {target_id}, {c0_name}, {args[1].id}, {c1_name})").body[0]
            unroll_attr = ast.parse(f"{fs_name}.properties()['unroll'] = '{unroll_val}'").body[0]

            # with for_scope...
            for_with = ast.With(
                items=[ast.withitem(context_expr=ast.Call(
                    func=ast.Name(id='for_scope', ctx=ast.Load()),
                    args=[ast.Name(id='builder', ctx=ast.Load()),
                          ast.Name(id='ctx', ctx=ast.Load()),
                          ast.Name(id=fs_name, ctx=ast.Load())],
                    keywords=[]
                ))],
                body=node.body
            )
            return [scalar_init, c0_init, c1_init, fs_init, unroll_attr, for_with]
        # If not transforming, still need to visit children
        self.generic_visit(node)
        return node

    def visit_If(self, node):
        # Requirement: only transform if condition is an ir.Scalar (assumed if it's a Name node in this DSL)
        if isinstance(node.test, ast.Name):
            # Get unique name for if statement variable BEFORE visiting children
            # This ensures outer if statements get lower numbers than inner ones
            ifs_name = self._get_ifs_name()
            
            # Now visit children (which may contain nested if/for statements)
            self.generic_visit(node)
            
            ifs_init = ast.parse(f"{ifs_name} = builder.create_if(ctx, {node.test.id})").body[0]

            then_scope = ast.With(
                items=[ast.withitem(context_expr=ast.Call(
                    func=ast.Name(id='if_then_scope', ctx=ast.Load()),
                    args=[ast.Name(id='builder', ctx=ast.Load()),
                          ast.Name(id='ctx', ctx=ast.Load()),
                          ast.Name(id=ifs_name, ctx=ast.Load())],
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
                              ast.Name(id=ifs_name, ctx=ast.Load())],
                        keywords=[]
                    ))],
                    body=node.orelse
                )
                res.append(else_scope)

            res.append(ast.parse(f"builder.exit_if(ctx, {ifs_name})").body[0])
            return res
        # If not transforming, still need to visit children
        self.generic_visit(node)
        return node

    def visit_Assign(self, node):
        # Handle Tile creation and Binary Ops
        if isinstance(node.value, ast.Call) and hasattr(node.value.func, 'attr') and node.value.func.attr == 'Tile':
            # res = pto.block.Tile(...) -> res = builder.create_tile(...)
            node.value.func = ast.Attribute(value=ast.Name(id='builder', ctx=ast.Load()), attr='create_tile', ctx=ast.Load())
            node.value.args.insert(0, ast.Name(id='ctx', ctx=ast.Load()))
            # Track that this variable is a Tile
            if isinstance(node.targets[0], ast.Name):
                self.tile_vars.add(node.targets[0].id)
            return node

        if isinstance(node.value, ast.BinOp):
            # Map Python binary operators to opcodes
            # Scalar-Tile operations (OP_ADDS, OP_SUBS, OP_MULS, OP_DIVS, OP_MINS, OP_MAXS)
            scalar_tile_op_map = {
                ast.Add: 'OP_ADDS',
                ast.Sub: 'OP_SUBS',
                ast.Mult: 'OP_MULS',
                ast.Div: 'OP_DIVS'
            }
            # Tile-Tile operations (OP_ADD, OP_SUB, OP_MUL, OP_DIV, OP_MIN, OP_MAX)
            tile_tile_op_map = {
                ast.Add: 'OP_ADD',
                ast.Sub: 'OP_SUB',
                ast.Mult: 'OP_MUL',
                ast.Div: 'OP_DIV'
            }
            
            target_name = node.targets[0].id if isinstance(node.targets[0], ast.Name) else None
            left_name = node.value.left.id if isinstance(node.value.left, ast.Name) else None
            right_name = node.value.right.id if isinstance(node.value.right, ast.Name) else None
            
            if target_name and left_name and right_name:
                # Determine if this is a Tile-Tile or Scalar-Tile operation
                # If both operands are Tiles, use Tile-Tile opcodes
                is_tile_tile = (left_name in self.tile_vars and right_name in self.tile_vars)
                
                if is_tile_tile:
                    opcode = tile_tile_op_map.get(type(node.value.op))
                    if opcode:
                        new_nodes = ast.parse(
                            f"op_tmp = builder.create_binary_op(ir.Opcode.{opcode}, {left_name}, {right_name}, {target_name})\n"
                            f"builder.emit(ctx, op_tmp)"
                        ).body
                        # Track that the result is also a Tile
                        self.tile_vars.add(target_name)
                        return new_nodes
                else:
                    # Default to Scalar-Tile operation
                    opcode = scalar_tile_op_map.get(type(node.value.op))
                    if opcode:
                        new_nodes = ast.parse(
                            f"op_tmp = builder.create_binary_scalar_op(ir.Opcode.{opcode}, {left_name}, {right_name}, {target_name})\n"
                            f"builder.emit(ctx, op_tmp)"
                        ).body
                        # If left operand is a Tile, result is also a Tile
                        if left_name in self.tile_vars:
                            self.tile_vars.add(target_name)
                        return new_nodes

        # Handle min() and max() function calls
        if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id in ('min', 'max'):
            if len(node.value.args) == 2:
                target_name = node.targets[0].id if isinstance(node.targets[0], ast.Name) else None
                left_name = node.value.args[0].id if isinstance(node.value.args[0], ast.Name) else None
                right_name = node.value.args[1].id if isinstance(node.value.args[1], ast.Name) else None
                
                if target_name and left_name and right_name:
                    # Determine if this is a Tile-Tile or Scalar-Tile operation
                    is_tile_tile = (left_name in self.tile_vars and right_name in self.tile_vars)
                    
                    if is_tile_tile:
                        opcode = 'OP_MAX' if node.value.func.id == 'max' else 'OP_MIN'
                        new_nodes = ast.parse(
                            f"op_tmp = builder.create_binary_op(ir.Opcode.{opcode}, {left_name}, {right_name}, {target_name})\n"
                            f"builder.emit(ctx, op_tmp)"
                        ).body
                        self.tile_vars.add(target_name)
                        return new_nodes
                    else:
                        opcode = 'OP_MAXS' if node.value.func.id == 'max' else 'OP_MINS'
                        new_nodes = ast.parse(
                            f"op_tmp = builder.create_binary_scalar_op(ir.Opcode.{opcode}, {left_name}, {right_name}, {target_name})\n"
                            f"builder.emit(ctx, op_tmp)"
                        ).body
                        if left_name in self.tile_vars:
                            self.tile_vars.add(target_name)
                        return new_nodes
        return node

    def visit_AugAssign(self, node):
        # res += scale -> add_op = builder.create_binary_scalar_op(...)
        # AugAssign is typically used for in-place operations, usually Scalar-Tile
        scalar_tile_op_map = {
            ast.Add: 'OP_ADDS',
            ast.Sub: 'OP_SUBS',
            ast.Mult: 'OP_MULS',
            ast.Div: 'OP_DIVS'
        }
        opcode = scalar_tile_op_map.get(type(node.op))
        if opcode:
            target_name = node.target.id if isinstance(node.target, ast.Name) else None
            value_name = node.value.id if isinstance(node.value, ast.Name) else None
            if target_name and value_name:
                return ast.parse(
                    f"op_tmp = builder.create_binary_scalar_op(ir.Opcode.{opcode}, {target_name}, {value_name}, {target_name})\n"
                    f"builder.emit(ctx, op_tmp)"
                ).body
        return node


    def visit_Return(self, node):
        # return x -> builder.create_return(ctx, [x])
        val = node.value.id if isinstance(node.value, ast.Name) else "None"
        return ast.parse(f"builder.create_return(ctx, [{val}])").body[0]


def get_common_vars():
    from pypto.pypto_impl import ir
    from pypto.ir.context import (
        function_scope, for_scope, if_then_scope, if_else_scope
    )
    common_vars = {
        "ir": ir,
        "function_scope": function_scope,
        "for_scope": for_scope,
        "if_then_scope": if_then_scope,
        "if_else_scope": if_else_scope
    }
    return common_vars


def transform_and_run(high_level_func, closure_vars, dump_transformed=False):

    raw_source = inspect.getsource(high_level_func)
    source = textwrap.dedent(raw_source)  #  Remove common leading whitespace (Fixes IndentationError)
    tree = ast.parse(source)

    transformer = FunctionAstMutator()
    new_tree = transformer.visit(tree)
    ast.fix_missing_locations(new_tree)

    # Unparse for inspection
    if dump_transformed:
        new_source = ast.unparse(new_tree)
        output_file = "transformed.py"
        print(f"=== Transformed Source Code to f{output_file}===")
        with open(output_file, "w") as f:
            f.write(new_source)

    # Compile and execute in the context of the closure
    code = compile(new_tree, filename="<ast>", mode="exec")
    common_vars = get_common_vars()
    namespace = {**common_vars, **closure_vars}
    exec(code, namespace)

    return namespace['create_function']
