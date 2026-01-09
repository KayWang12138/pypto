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

    def _fix_locations(self, node, reference_node=None):
        """Fix missing locations in AST node, optionally copying from reference node"""
        # First, recursively fix all child nodes
        for child in ast.walk(node):
            if child is not node:  # Don't process the node itself yet
                if not hasattr(child, 'lineno'):
                    child.lineno = 1
                if not hasattr(child, 'col_offset'):
                    child.col_offset = 0

        # Then fix the node itself
        if reference_node is not None:
            ast.copy_location(node, reference_node)
        else:
            # Ensure node has at least default location info
            if not hasattr(node, 'lineno'):
                node.lineno = 1
            if not hasattr(node, 'col_offset'):
                node.col_offset = 0
        return node

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
        # Check if function has type hints (signature extraction mode)
        has_type_hints = False
        signature_init_nodes = []
        arg_names = []
        return_var_names = []

        # Extract signature from type hints if present
        if node.args.args and any(arg.annotation is not None for arg in node.args.args):
            has_type_hints = True

            # Extract arguments from type hints
            for arg in node.args.args:
                if arg.annotation is not None:
                    arg_names.append(arg.arg)
                    # Create assignment: arg_name = <annotation expression>
                    # The annotation AST will be evaluated at runtime in the closure context
                    assign_node = ast.Assign(
                        targets=[ast.Name(id=arg.arg, ctx=ast.Store())],
                        value=arg.annotation
                    )
                    self._fix_locations(assign_node, node)
                    signature_init_nodes.append(assign_node)

            # Extract return type if present
            if node.returns is not None:
                # Handle tuple return: -> (ir.Scalar(...))
                if isinstance(node.returns, ast.Tuple):
                    for i, elt in enumerate(node.returns.elts):
                        return_var = f"_return_{i}"
                        return_var_names.append(return_var)
                        assign_node = ast.Assign(
                            targets=[ast.Name(id=return_var, ctx=ast.Store())],
                            value=elt
                        )
                        self._fix_locations(assign_node, node)
                        signature_init_nodes.append(assign_node)
                else:
                    # Single return value
                    return_var = "_return_0"
                    return_var_names.append(return_var)
                    assign_node = ast.Assign(
                        targets=[ast.Name(id=return_var, ctx=ast.Store())],
                        value=node.returns
                    )
                    self._fix_locations(assign_node, node)
                    signature_init_nodes.append(assign_node)

            # Create signature initialization (only if sig is None)
            if arg_names or return_var_names:
                # Build the signature creation code
                sig_args_str = ', '.join(arg_names) if arg_names else ''
                sig_returns_str = ', '.join(return_var_names) if return_var_names else ''

                sig_check_code = f"""
sig = ir.FunctionSignature()
sig.arguments = [{sig_args_str}]
sig.returns = [{sig_returns_str}]
"""
                sig_conditional = ast.parse(sig_check_code).body[0]
                self._fix_locations(sig_conditional, node)
                signature_init_nodes.append(sig_conditional)

            # Remove arguments that have type hints (they're now created in the function body)
            # Keep only arguments without type hints (if any)
            new_args = []
            kept_arg_indices = []
            for i, arg in enumerate(node.args.args):
                if arg.annotation is None:
                    # This arg didn't have a type hint, keep it
                    new_args.append(arg)
                    kept_arg_indices.append(i)

            # Adjust defaults - only keep defaults for args we're keeping
            # Since defaults are from the right, we need to map them correctly
            old_defaults = list(node.args.defaults) if node.args.defaults else []
            num_old_args = len(node.args.args)
            num_defaults = len(old_defaults)

            # Calculate which defaults to keep
            # Defaults correspond to the rightmost args
            # If we kept args, we need to figure out which defaults correspond to them
            new_defaults = []
            if kept_arg_indices and old_defaults:
                # Map old arg indices to new positions
                # Defaults are for args starting from num_old_args - num_defaults
                default_start_idx = num_old_args - num_defaults
                for i, old_idx in enumerate(kept_arg_indices):
                    if old_idx >= default_start_idx:
                        # This kept arg had a default
                        default_idx = old_idx - default_start_idx
                        if default_idx < len(old_defaults):
                            new_defaults.append(old_defaults[default_idx])

            node.args.args = new_args
            node.args.defaults = new_defaults
            node.returns = None

            # Add builder, ctx, name, function_kind, sig parameters at the beginning
            builder_arg = ast.arg(arg='builder', annotation=None)
            ctx_arg = ast.arg(arg='ctx', annotation=None)
            name_arg = ast.arg(arg='name', annotation=None)
            name_default = ast.Constant(value="test_control")
            function_kind_arg = ast.arg(arg='function_kind', annotation=None)
            function_kind_default = ast.Attribute(
                value=ast.Attribute(value=ast.Name(id='ir', ctx=ast.Load()), attr='FunctionKind', ctx=ast.Load()),
                attr='ControlFlow', ctx=ast.Load()
            )

            # Insert new parameters at the beginning
            node.args.args = [builder_arg, ctx_arg, name_arg, function_kind_arg] + node.args.args
            # Defaults need to match the order of args with defaults (from right to left)
            # Add defaults for the new parameters, then keep any defaults for kept args
            node.args.defaults = [name_default, function_kind_default] + node.args.defaults

        # 1. Transform the body first
        self.generic_visit(node)

        # 2. Wrap the transformed body in 'with function_scope(builder, ctx, func):'
        # First, add the func creation line at the top
        func_init = ast.parse("func = builder.create_function(name, function_kind, sig, False)").body[0]
        self._fix_locations(func_init, node)

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
        self._fix_locations(wrapped_body, node)

        return_node = ast.Return(value=ast.Name(id='func', ctx=ast.Load()))
        self._fix_locations(return_node, node)

        if has_type_hints:
            # Insert signature initialization before func_init
            node.body = signature_init_nodes + [func_init, wrapped_body, return_node]
        else:
            node.body = [func_init, wrapped_body, return_node]
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


def transform_and_run(high_level_func, closure_vars, context_kwargs=None, dump_transformed=False):

    raw_source = inspect.getsource(high_level_func)
    source = textwrap.dedent(raw_source)  #  Remove common leading whitespace (Fixes IndentationError)
    tree = ast.parse(source)

    # Extract function definition to get its name
    func_def = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            func_def = node
            break

    if func_def is None:
        raise ValueError("No function definition found in source")

    transformer = FunctionAstMutator()
    new_tree = transformer.visit(tree)
    ast.fix_missing_locations(new_tree)

    # Unparse for inspection
    if dump_transformed:
        new_source = ast.unparse(new_tree)
        output_file = "transformed.py"
        print(f"=== Transformed Source Code to {output_file}===")
        with open(output_file, "w") as f:
            f.write(new_source)

    # Compile and execute in the context of the closure
    code = compile(new_tree, filename="<ast>", mode="exec")
    common_vars = get_common_vars()
    namespace = {**common_vars, **closure_vars}
    exec(code, namespace)

    # Get the function name (could be different from 'create_function')
    func_name = func_def.name
    transformed_func = namespace[func_name]

    # If context_kwargs provided, create a wrapper that applies them
    if context_kwargs is not None:
        def wrapped_func(*args, **kwargs):
            # Exclude builder and ctx from context_kwargs since they're passed positionally
            # Only use context_kwargs for optional parameters (name, function_kind, sig)
            filtered_context_kwargs = {k: v for k, v in context_kwargs.items()
                                      if k not in ('builder', 'ctx')}
            # Merge filtered context_kwargs with provided kwargs (provided kwargs take precedence)
            merged_kwargs = {**filtered_context_kwargs, **kwargs}
            return transformed_func(*args, **merged_kwargs)
        return wrapped_func

    return transformed_func
