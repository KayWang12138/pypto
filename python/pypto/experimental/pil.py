#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 CANN community contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Build PIL (Python Intermediate Language) for pypto frontend. Compared with the full python ast, 
PIL is a simplified version of python ast, which only contains the necessary information for code generation. 
The main purpose of PIL is to simplify the code generation process and improve the performance of code generation.

Simplify rule:
1.  All expr should be replaced by identifier as much as possible
2.  When assigned, only multiple names with starred, single attribute, single subscript 
    are allowed in the assignment's lhs and for's target

stmt = FunctionDef(identifier name, arguments args,
                    stmt* body, expr* decorator_list)
        | Assign(expr target, expr value, string? type_comment) # target only allow for identifier, attribute and subscript
        | Return(identifier? value)

        | For(identifier target, identifier iter, stmt* body, stmt* orelse, string? type_comment)
        | While(identifier test, stmt* body, stmt* orelse)
        | If(identifier test, stmt* body, stmt* orelse)

        | Assert(identifier test, identifier? msg)

        | Import(alias* names)
        | ImportFrom(identifier? module, alias* names, int? level)

        | Global(identifier* names)
        | Nonlocal(identifier* names)
        | Pass 
        | Break 
        | Continue

        -- col_offset is the byte offset in the utf8 string the parser uses
        attributes (int lineno, int col_offset, int? end_lineno, int? end_col_offset)

expr =  BinOp(identifier left, operator op, identifier right)
        | UnaryOp(unaryop op, identifier operand)
        | Dict(identifier?* keys, identifier* values)
        | Set(expr* elts) # only allow for identifier and starred
        | Compare(identifier left, cmpop ops, identifier comparators)
        | Call(identifier func, identifier* args, keyword* keywords)

        | FormattedValue(identifier value, int conversion, identifier? format_spec)
        | JoinedStr(expr* values)

        | Constant(constant value, string? kind)

        | Name(identifier id)
        | Attribute(identifier value, identifier attr)
        | Subscript(identifier value, expr *slice)

        | Starred(identifier value)
        | List(expr* elts) # Only allow for identifier and starred
        | Tuple(expr* elts) # Only allow for slice and identifier, starred

        | Slice(identifier? lower, identifier? upper, identifier? step)

operator = Add | Sub | Mult | MatMult | Div | Mod | Pow | LShift
                | RShift | BitOr | BitXor | BitAnd | FloorDiv

unaryop = Invert | Not | UAdd | USub

cmpop = Eq | NotEq | Lt | LtE | Gt | GtE | Is | IsNot | In | NotIn
"""

import ast
from collections.abc import Mapping

PILExpr = str | ast.Constant

class PILContext:
    def __init__(self):
        self._continue_stack = []
        self._temp_count = 0

    def create_temp_identifier(self, *_args, **_kwargs) -> str:
        name = f"_pil_{self._temp_count}"
        self._temp_count += 1
        return name

class PILAttr(Mapping):
    ATTR_LIST = [
        'lineno',   
        'col_offset',
        'end_lineno',
        'end_col_offset',
    ]
    def __init__(self, node):
        self._data = {
            attr: 0 if node is None else getattr(node, attr, 0)
            for attr in self.ATTR_LIST
        }

    def __getitem__(self, key):
        return self._data[key]
    
    def __iter__(self):
        return iter(self._data)
    
    def __len__(self):
        return len(self._data)

NOATTR = PILAttr(None)

class PILBuilder(ast.NodeVisitor):
    def __init__(self, ctx: PILContext = None):
        if ctx is None:
            ctx = PILContext()
        self._ctx = ctx

    @property
    def continue_stack(self) -> list[tuple[ast.expr, str] | None]:
        return self._ctx._continue_stack
    
    def create_temp_identifier(self) -> str:
        return self._ctx.create_temp_identifier()
    
    def create_attribute(self, node) -> dict:
        result = {
            'lineno': node.lineno,
            'col_offset': node.col_offset,
            'end_lineno': node.end_lineno,
            'end_col_offset': node.end_col_offset,
        }
        return result
    
    def create_pil_expr(self, value: PILExpr, ctx: ast.expr_context = ast.Load(), node_attr: PILAttr = NOATTR) -> ast.Name | ast.Constant:
        if isinstance(value, ast.Constant):
            return value
        else:
            assert isinstance(value, str)
            return self.create_pil_name(value, ctx)

    def create_pil_maybe_starred(self, expr: PILExpr, starred: bool, ctx=ast.Load(), node_attr: PILAttr = NOATTR) -> ast.expr:
        if starred:
            assert isinstance(expr, str)
            return self.create_pil_starred(expr, ctx)
        else:
            return self.create_pil_expr(expr, ctx)
    
    def create_pil_assign_name(self, targets:str | list[tuple[str, bool]] | tuple[tuple[str, bool]], value: ast.expr, node_attr: PILAttr = NOATTR) -> ast.Assign:
        if isinstance(targets, str):
            result_targets = [ast.Name(id=targets, ctx=ast.Store())]
            return ast.Assign(targets=result_targets, value=value, **node_attr)
        else:
            if isinstance(targets, list):
                result_targets = [self.create_pil_maybe_starred(name, starred, ctx=ast.Store()) for name, starred in targets]
            else:
                result_targets = (self.create_pil_maybe_starred(name, starred, ctx=ast.Store()) for name, starred in targets)
            return ast.Assign(targets=result_targets, value=value, **node_attr)

    def create_pil_assign_name_dup(self, target_name:str, source_expr: PILExpr, node_attr: PILAttr = NOATTR) -> ast.Assign:
        assert isinstance(target_name, str)
        return ast.Assign(
            targets=[ast.Name(id=target_name, ctx=ast.Store())],
            value=self.create_pil_expr(source_expr),
            **node_attr)

    def create_pil_assign_attribute(self, target_name:str, attr_name: str, source_expr: PILExpr, node_attr: PILAttr = NOATTR) -> ast.Assign:
        return ast.Assign(
            targets=[ast.Attribute(value=self.create_pil_name(target_name),
                                   attr=attr_name, ctx=ast.Store())],
            value=self.create_pil_expr(source_expr),
            **node_attr)

    def create_pil_assign_subscript(self, target_name:str, slices: list[tuple[PILExpr | None, PILExpr | None, PILExpr | None]], source_expr: PILExpr, node_attr: PILAttr = NOATTR) -> ast.Assign:
        assert isinstance(target_name, str)
        result_slice_tuple = []
        for slice in slices:
            result_slice_expr = ast.Slice(
                lower=self.create_pil_expr(slice[0]) if slice[0] is not None else None,
                upper=self.create_pil_expr(slice[1]) if slice[1] is not None else None,
                step=self.create_pil_expr(slice[2]) if slice[2] is not None else None)
            result_slice_tuple.append(result_slice_expr)
        if len(result_slice_tuple) == 1:
            result_slice = result_slice_tuple[0]
        else:
            result_slice = ast.Tuple(elts=result_slice_tuple, ctx=ast.Load())
        return ast.Assign(
            targets=[ast.Subscript(value=self.create_pil_name(target_name),
                                   slice=result_slice, ctx=ast.Store())],
            value=self.create_pil_expr(source_expr),
            **node_attr)

    def create_pil_return(self, expr: PILExpr | None, node_attr: PILAttr = NOATTR) -> ast.Return:
        return ast.Return(value=self.create_pil_expr(expr) if expr is not None else None, **node_attr)

    def create_pil_for(self, target_name: str, iter_expr: PILExpr, body: list[ast.stmt], orelse: list[ast.stmt], type_comment: str | None, node_attr: PILAttr = NOATTR) -> ast.For:
        assert isinstance(target_name, str)
        return ast.For(target=ast.Name(id=target_name, ctx=ast.Store()), iter=self.create_pil_expr(iter_expr), body=body, orelse=orelse, type_comment=type_comment, **node_attr)

    def create_pil_while(self, test_expr: PILExpr, body: list[ast.stmt], orelse: list[ast.stmt], node_attr: PILAttr = NOATTR) -> ast.While:
        return ast.While(test=self.create_pil_expr(test_expr), body=body, orelse=orelse, **node_attr)

    def create_pil_if(self, test_expr: PILExpr, body: list[ast.stmt], orelse: list[ast.stmt], node_attr: PILAttr = NOATTR) -> ast.If:
        return ast.If(test=self.create_pil_expr(test_expr), body=body, orelse=orelse, **node_attr)

    def create_pil_assert(self, test_expr: PILExpr, msg_value: PILExpr | None, node_attr: PILAttr = NOATTR) -> ast.Assert:
        return ast.Assert(test=self.create_pil_expr(test_expr), msg=self.create_pil_expr(msg_value) if msg_value is not None else None, **node_attr)

    def create_pil_import(self, names: list[ast.alias], node_attr: PILAttr = NOATTR) -> ast.Import:
        return ast.Import(names=names, **node_attr)

    def create_pil_import_from(self, module: str | None, names: list[ast.alias], level: int | None, node_attr: PILAttr = NOATTR) -> ast.ImportFrom:
        return ast.ImportFrom(module=module, names=names, level=level, **node_attr)

    def create_pil_global(self, names: list[str], node_attr: PILAttr = NOATTR) -> ast.Global:
        return ast.Global(names=names, **node_attr)

    def create_pil_nonlocal(self, names: list[str], node_attr: PILAttr = NOATTR) -> ast.Nonlocal:
        return ast.Nonlocal(names=names, **node_attr)

    def create_pil_pass(self, node_attr: PILAttr = NOATTR) -> ast.Pass:
        return ast.Pass(**node_attr)

    def create_pil_break(self, node_attr: PILAttr = NOATTR) -> ast.Break:
        return ast.Break(**node_attr)

    def create_pil_continue(self, node_attr: PILAttr = NOATTR) -> ast.Continue:
        return ast.Continue(**node_attr)

    def create_pil_bin_op(self, left_expr: PILExpr, op: ast.operator, right_expr: PILExpr, node_attr: PILAttr = NOATTR) -> ast.BinOp:
        return ast.BinOp(left=self.create_pil_expr(left_expr), op=op, right=self.create_pil_expr(right_expr), **node_attr)

    def create_pil_unary_op(self, op: ast.unaryop, operand_expr: PILExpr, node_attr: PILAttr = NOATTR) -> ast.UnaryOp:
        return ast.UnaryOp(op=op, operand=self.create_pil_expr(operand_expr), **node_attr)

    def create_pil_dict(self, keys: list[PILExpr | None], values: list[PILExpr], node_attr: PILAttr = NOATTR) -> ast.Dict:
        return ast.Dict(
            keys=[self.create_pil_expr(key) if key is not None else None for key in keys],
            values=[self.create_pil_expr(value) for value in values],
            **node_attr)

    def create_pil_set(self, elts: list[tuple[PILExpr, bool]], node_attr: PILAttr = NOATTR) -> ast.Set:
        return ast.Set(
            elts=[self.create_pil_maybe_starred(elt[0], elt[1]) for elt in elts],
            **node_attr)

    def create_pil_compare(self, left_expr: PILExpr, op: ast.cmpop, comparator_expr: PILExpr, node_attr: PILAttr = NOATTR) -> ast.Compare:
        return ast.Compare(left=self.create_pil_expr(left_expr), ops=[op], comparators=[self.create_pil_expr(comparator_expr)], **node_attr)

    def create_pil_call(self, func_expr: PILExpr, args: list[PILExpr], keywords: list[ast.keyword], node_attr: PILAttr = NOATTR) -> ast.Call:
        return ast.Call(func=self.create_pil_expr(func_expr), args=[self.create_pil_expr(arg) for arg in args], keywords=keywords, **node_attr)

    def create_pil_constant(self, value: object, kind: str | None, node_attr: PILAttr = NOATTR) -> ast.Constant:
        return ast.Constant(value=value, kind=kind, **node_attr)

    def create_pil_name(self, id: str, ctx: ast.expr_context = ast.Load(), node_attr: PILAttr = NOATTR) -> ast.Name:
        return ast.Name(id=id, ctx=ctx, **node_attr)

    def create_pil_attribute(self, value_expr: PILExpr, attr_name: str, ctx: ast.expr_context = ast.Load(), node_attr: PILAttr = NOATTR) -> ast.Attribute:
        return ast.Attribute(value=self.create_pil_expr(value_expr), attr=attr_name, ctx=ctx, **node_attr)

    def create_pil_subscript(self, value_expr: PILExpr, slices: list[tuple[PILExpr | None, PILExpr | None, PILExpr | None]], ctx: ast.expr_context = ast.Load(), node_attr: PILAttr = NOATTR) -> ast.Subscript:
        result_slice_tuple = []
        for slice in slices:
            result_slice_expr = ast.Slice(
                lower=self.create_pil_expr(slice[0]) if slice[0] is not None else None,
                upper=self.create_pil_expr(slice[1]) if slice[1] is not None else None,
                step=self.create_pil_expr(slice[2]) if slice[2] is not None else None)
            result_slice_tuple.append(result_slice_expr)
        if len(slices) == 1:
            result_slice = result_slice_tuple[0]
        else:
            result_slice = ast.Tuple(elts=result_slice_tuple, ctx=ast.Load())
        return ast.Subscript(value=self.create_pil_expr(value_expr), slice=result_slice, ctx=ast.Load(), **node_attr)

    def create_pil_starred(self, value_expr: PILExpr, ctx: ast.expr_context = ast.Load(), node_attr: PILAttr = NOATTR) -> ast.Starred:
        return ast.Starred(self.create_pil_expr(value_expr), ctx=ctx, **node_attr)

    def create_pil_list(self, elts: list[tuple[PILExpr, bool]], ctx: ast.expr_context = ast.Load(), node_attr: PILAttr = NOATTR) -> ast.List:
        return ast.List(
            elts=[self.create_pil_maybe_starred(elt[0], elt[1]) for elt in elts],
            ctx=ctx,
            **node_attr)

    def create_pil_tuple(self, elts: list[tuple[PILExpr, bool]], ctx: ast.expr_context = ast.Load(), node_attr: PILAttr = NOATTR) -> ast.Tuple:
        return ast.Tuple(
            elts=[self.create_pil_maybe_starred(elt[0], elt[1]) for elt in elts],
            ctx=ctx,
            **node_attr)
    
class PythonParser(PILBuilder, ast.NodeVisitor):

    def __init__(self, ctx: PILContext):
        PILBuilder.__init__(self, ctx)
        ast.NodeVisitor.__init__(self)

    def visit_slice(self, slice: ast.Slice | tuple[ast.Slice]) -> tuple[list[ast.stmt], list[tuple[str | None, str | None, str | None]]]:
        slice_list = slice.elts if isinstance(slice, ast.Tuple) else [slice]
                    
        slice_stmt_list = []
        pil_slice_list = []
        for s in slice_list:
            assert isinstance(s, ast.Slice), "PIL lhs Subscript slice only supports ast.Slice"
            lower_name = upper_name = step_name = None
            if s.lower is not None:
                stmts, lower_name = self.visit(s.lower)
                slice_stmt_list.extend(stmts)
            if s.upper is not None:
                stmts, upper_name = self.visit(s.upper)
                slice_stmt_list.extend(stmts)
            if s.step is not None:
                stmts, step_name = self.visit(s.step)
                slice_stmt_list.extend(stmts)
            pil_slice_list.append((lower_name, upper_name, step_name))
        return slice_stmt_list, pil_slice_list

    def visit_lhs(self, target: ast.expr, source_name: str) -> list[ast.stmt]:
        if isinstance(target, ast.Name):
            return [self.create_pil_assign_name_dup(target.id, source_name)]

        elif isinstance(target, ast.Attribute):
            obj_stmts, obj_name = self.visit(target.value)
            return obj_stmts + [self.create_pil_assign_attribute(obj_name, target.attr, source_name)]

        elif isinstance(target, ast.Subscript):
            obj_stmts, obj_name = self.visit(target.value)
            slice_stmt_list, pil_slice_list = self.visit_slice(target.slice)
            return obj_stmts + slice_stmt_list + [self.create_pil_assign_subscript(obj_name, pil_slice_list, source_name)]

        elif isinstance(target, (ast.Tuple, ast.List)):
            # Step 1: allocate one temp per element, preserving starred-ness
            if isinstance(target, ast.List):
                elt_temps = [(self.create_temp_identifier(), isinstance(elt, ast.Starred))
                            for elt in target.elts]
            else:
                elt_temps = ((self.create_temp_identifier(), isinstance(elt, ast.Starred))
                            for elt in target.elts)

            # Step 2: one-layer unpack — (t0, *t1, t2) = source_name
            unpack_stmt = self.create_pil_assign_name(elt_temps, self.create_pil_name(source_name))
 
            # Step 3: recursively handle each element with its temp
            result_stmts = [unpack_stmt]
            for (temp_name, starred), elt in zip(elt_temps, target.elts):
                actual_elt = elt.value if starred else elt
                result_stmts.extend(self.visit_lhs(actual_elt, temp_name))
            return result_stmts

        raise NotImplementedError(f"LHS target type {type(target).__name__} is not supported")

    def visit_FunctionDef(self, name: str, args: ast.arguments, body: list[ast.stmt], decorator_list: list[ast.expr], returns: ast.expr | None, type_comment: str | None, type_params: list[ast.type_param], node_attr: PILAttr = NOATTR) -> tuple[list[ast.stmt], PILExpr | None]:
        raise NotImplementedError("FunctionDef is not supported")

    def visit_AsyncFunctionDef(self, name: str, args: ast.arguments, body: list[ast.stmt], decorator_list: list[ast.expr], returns: ast.expr | None, type_comment: str | None, type_params: list[ast.type_param], node_attr: PILAttr = NOATTR) -> tuple[list[ast.stmt], PILExpr | None]:
        raise NotImplementedError("AsyncFunctionDef is not supported")

    def visit_ClassDef(self, name: str, bases: list[ast.expr], keywords: list[ast.keyword], body: list[ast.stmt], decorator_list: list[ast.expr], type_params: list[ast.type_param], node_attr: PILAttr = NOATTR) -> tuple[list[ast.stmt], PILExpr | None]:
        raise NotImplementedError("ClassDef is not supported")

    def visit_Return(self, value: ast.expr | None, node_attr: PILAttr = NOATTR) -> tuple[list[ast.stmt], PILExpr | None]:
        if value is not None:
            value_stmt_list, value_name = self.visit(value)
            result_stmt_list = [*value_stmt_list, self.create_pil_return(value_name, node_attr=node_attr)]
        else:
            result_stmt_list = [self.create_pil_return(None, node_attr=node_attr)]
        return result_stmt_list, None

    def visit_Delete(self, targets: list[ast.expr], node_attr: PILAttr = NOATTR) -> tuple[list[ast.stmt], PILExpr | None]:
        raise NotImplementedError("Delete is not supported")

    def visit_Assign(self, targets: list[ast.expr], value: ast.expr, type_comment: str | None, node_attr: PILAttr = NOATTR) -> tuple[list[ast.stmt], PILExpr | None]:
        value_stmt_list, value_name = self.visit(value)
        result_stmt_list = value_stmt_list
        for target in targets:
            target_stmt_list = self.visit_lhs(target, value_name)
            result_stmt_list.extend(target_stmt_list)
        return result_stmt_list, None

    def visit_TypeAlias(self, name: ast.expr, type_params: list[ast.type_param], value: ast.expr, node_attr: PILAttr = NOATTR) -> tuple[list[ast.stmt], PILExpr | None]:
        raise NotImplementedError("TypeAlias is not supported")

    def visit_AugAssign(self, target: ast.expr, op: ast.operator, value: ast.expr, node_attr: PILAttr = NOATTR) -> tuple[list[ast.stmt], PILExpr | None]:
        if isinstance(target, ast.Name):
            # target is a bare name — no side effects, visit value first is fine
            value_stmt_list, value_name = self.visit(value)
            temp_name = self.create_temp_identifier()
            binop_stmt = self.create_pil_assign_name(temp_name, self.create_pil_bin_op(target.id, op, value_name))
            store_stmt = self.create_pil_assign_name_dup(target.id, temp_name)
            return value_stmt_list + [binop_stmt, store_stmt], None

        elif isinstance(target, ast.Attribute):
            # Python evaluates: obj first, then rhs value, then load, then store
            target_stmt_list, target_name = self.visit(target.value)
            value_stmt_list, value_name = self.visit(value)
            load_temp = self.create_temp_identifier()
            load_stmt = self.create_pil_assign_name(load_temp, self.create_pil_attribute(target_name, target.attr))
            temp_name = self.create_temp_identifier()
            binop_stmt = self.create_pil_assign_name(temp_name, self.create_pil_bin_op(load_temp, op, value_name))
            store_stmt = self.create_pil_assign_attribute(target_name, target.attr, temp_name)
            return target_stmt_list + value_stmt_list + [load_stmt, binop_stmt, store_stmt], None

        elif isinstance(target, ast.Subscript):
            # Python evaluates: obj first, then slice, then rhs value, then load, then store
            target_stmt_list, target_name = self.visit(target.value)
            # normalize slice into list of (lower, upper, step) tuples
            slice_stmt_list, pil_slice_list = self.visit_slice(target.slice)            
            value_stmt_list, value_name = self.visit(value)
            load_temp = self.create_temp_identifier()
            load_stmt = self.create_pil_assign_name(load_temp, self.create_pil_subscript(target_name, pil_slice_list))
            temp_name = self.create_temp_identifier()
            binop_stmt = self.create_pil_assign_name(temp_name, self.create_pil_bin_op(load_temp, op, value_name))
            store_stmt = self.create_pil_assign_subscript(target_name, pil_slice_list, temp_name)
            return target_stmt_list + slice_stmt_list + value_stmt_list + [load_stmt, binop_stmt, store_stmt], None

        raise NotImplementedError(f"AugAssign target type {type(target).__name__} is not supported")

    def visit_AnnAssign(self, target: ast.expr, annotation: ast.expr, value: ast.expr | None, simple: int, node_attr: PILAttr = NOATTR) -> tuple[list[ast.stmt], PILExpr | None]:
        raise NotImplementedError("AnnAssign is not supported")

    def visit_For(self, target: ast.expr, iter: ast.expr, body: list[ast.stmt], orelse: list[ast.stmt], type_comment: str | None, node_attr: PILAttr = NOATTR) -> tuple[list[ast.stmt], PILExpr | None]:
        self.continue_stack.append(None)
        iter_stmt_list, iter_name = self.visit(iter)

        target_name = self.create_temp_identifier()
        target_stmt_list = self.visit_lhs(target, target_name)
        body_stmt_list, _ = self.visit_stmts(body)
        result_body_stmt_list = target_stmt_list + body_stmt_list
        orelse_stmt_list, _ = self.visit_stmts(orelse)
        result_stmt_list = iter_stmt_list + [self.create_pil_for(target_name, iter_name, result_body_stmt_list, orelse_stmt_list, type_comment)]
        self.continue_stack.pop()
        return result_stmt_list, None

    def visit_AsyncFor(self, target: ast.expr, iter: ast.expr, body: list[ast.stmt], orelse: list[ast.stmt], type_comment: str | None, node_attr: PILAttr = NOATTR) -> tuple[list[ast.stmt], PILExpr | None]:
        raise NotImplementedError("AsyncFor is not supported")

    def visit_While(self, test: ast.expr, body: list[ast.stmt], orelse: list[ast.stmt], node_attr: PILAttr = NOATTR) -> tuple[list[ast.stmt], PILExpr | None]:
        test_stmt_list, test_name = self.visit(test)
        self.continue_stack.append((test, test_name))
        body_stmt_list, _ = self.visit_stmts(body)
        orelse_stmt_list, _ = self.visit_stmts(orelse)
        # re-evaluate test at the end of each iteration to update test_name for the next check
        reeval_stmt_list, reeval_name = self.visit(test)
        reeval_stmt_list = reeval_stmt_list + [self.create_pil_assign_name_dup(test_name, reeval_name)]
        result_body_stmt_list = body_stmt_list + reeval_stmt_list
        result_stmt_list = test_stmt_list + [self.create_pil_while(test_name, result_body_stmt_list, orelse_stmt_list)]
        self.continue_stack.pop()
        return result_stmt_list, None

    def visit_If(self, test: ast.expr, body: list[ast.stmt], orelse: list[ast.stmt], node_attr: PILAttr = NOATTR) -> tuple[list[ast.stmt], PILExpr | None]:
        test_stmt_list, test_name = self.visit(test)
        body_stmt_list, _ = self.visit_stmts(body)
        orelse_stmt_list, _ = self.visit_stmts(orelse)
        result_stmt_list = test_stmt_list + [self.create_pil_if(test_name, body_stmt_list, orelse_stmt_list)]
        return result_stmt_list, None

    def visit_With(self, items: list[ast.withitem], body: list[ast.stmt], type_comment: str | None, node_attr: PILAttr = NOATTR) -> tuple[list[ast.stmt], PILExpr | None]:
        raise NotImplementedError("With is not supported")

    def visit_AsyncWith(self, items: list[ast.withitem], body: list[ast.stmt], type_comment: str | None, node_attr: PILAttr = NOATTR) -> tuple[list[ast.stmt], PILExpr | None]:
        raise NotImplementedError("AsyncWith is not supported")

    def visit_Match(self, subject: ast.expr, cases: list[ast.match_case], node_attr: PILAttr = NOATTR) -> tuple[list[ast.stmt], PILExpr | None]:
        raise NotImplementedError("Match is not supported")

    def visit_Raise(self, exc: ast.expr | None, cause: ast.expr | None, node_attr: PILAttr = NOATTR) -> tuple[list[ast.stmt], PILExpr | None]:
        raise NotImplementedError("Raise is not supported")

    def visit_Try(self, body: list[ast.stmt], handlers: list[ast.excepthandler], orelse: list[ast.stmt], finalbody: list[ast.stmt], node_attr: PILAttr = NOATTR) -> tuple[list[ast.stmt], PILExpr | None]:
        raise NotImplementedError("Try is not supported")

    def visit_TryStar(self, body: list[ast.stmt], handlers: list[ast.excepthandler], orelse: list[ast.stmt], finalbody: list[ast.stmt], node_attr: PILAttr = NOATTR) -> tuple[list[ast.stmt], PILExpr | None]:
        raise NotImplementedError("TryStar is not supported")

    def visit_Assert(self, test: ast.expr, msg: ast.expr | None, node_attr: PILAttr = NOATTR) -> tuple[list[ast.stmt], PILExpr | None]:
        test_stmt_list, test_name = self.visit(test)
        # not_test = not test_name
        not_test_name = self.create_temp_identifier()
        not_test_stmt = self.create_pil_assign_name(not_test_name, self.create_pil_unary_op(ast.Not(), test_name))
        # msg is only evaluated when the assertion fails
        if msg is not None:
            msg_stmt_list, msg_name = self.visit(msg)
        else:
            msg_stmt_list, msg_name = [], None
        fail_body = msg_stmt_list + [self.create_pil_assert(test_name, msg_name)]
        debug_body = test_stmt_list + [not_test_stmt, self.create_pil_if(not_test_name, fail_body, [])]
        return [self.create_pil_if("__debug__", debug_body, [])], None

    def visit_Import(self, names: list[ast.alias], node_attr: PILAttr = NOATTR) -> tuple[list[ast.stmt], PILExpr | None]:
        return [self.create_pil_import(names)], None

    def visit_ImportFrom(self, module: str | None, names: list[ast.alias], level: int | None, node_attr: PILAttr = NOATTR) -> tuple[list[ast.stmt], PILExpr | None]:
        return [self.create_pil_import_from(module, names, level)], None

    def visit_Global(self, names: list[str], node_attr: PILAttr = NOATTR) -> tuple[list[ast.stmt], PILExpr | None]:
        return [self.create_pil_global(names)], None

    def visit_Nonlocal(self, names: list[str], node_attr: PILAttr = NOATTR) -> tuple[list[ast.stmt], PILExpr | None]:
        return [self.create_pil_nonlocal(names)], None

    def visit_Expr(self, value: ast.expr, node_attr: PILAttr = NOATTR) -> tuple[list[ast.stmt], PILExpr | None]:
        value_stmt_list, value_name = self.visit(value)
        return value_stmt_list, None

    def visit_Pass(self, node_attr: PILAttr = NOATTR) -> tuple[list[ast.stmt], PILExpr | None]:
        return [self.create_pil_pass()], None

    def visit_Break(self, node_attr: PILAttr = NOATTR) -> tuple[list[ast.stmt], PILExpr | None]:
        return [self.create_pil_break()], None

    def visit_Continue(self, node_attr: PILAttr = NOATTR) -> tuple[list[ast.stmt], PILExpr | None]:
        if self.continue_stack[-1] is None:
            result_stmt_list = [self.create_pil_continue()]
        else:
            test_expr, test_name = self.continue_stack[-1]
            reeval_stmt_list, reeval_name = self.visit(test_expr)
            result_stmt_list = reeval_stmt_list + [self.create_pil_assign_name_dup(test_name, reeval_name), self.create_pil_continue()]
        return result_stmt_list, None

    # expr nodes
    def visit_BoolOp(self, op: ast.boolop, values: list[ast.expr], node_attr: PILAttr = NOATTR) -> tuple[list[ast.stmt], PILExpr | None]:
        # Base case: single value, just visit it directly
        if len(values) == 1:
            return self.visit(values[0])

        temp_name = self.create_temp_identifier()
        first_stmt_list, first_name = self.visit(values[0])
        rest_stmt_list, rest_name = self.visit_BoolOp(op=op, values=values[1:])

        if isinstance(op, ast.And):
            # if temp is truthy, evaluate the rest and update temp
            rest_stmt = self.create_pil_if(
                first_name,
                rest_stmt_list + [self.create_pil_assign_name_dup(temp_name, rest_name)],
                [self.create_pil_assign_name_dup(temp_name, first_name)])

            result_stmt_list = first_stmt_list + [rest_stmt]
            return result_stmt_list, temp_name

        elif isinstance(op, ast.Or):
            # if temp is falsy, evaluate the rest and update temp
            rest_stmt = self.create_pil_if(
                first_name,
                [self.create_pil_assign_name_dup(temp_name, first_name)],
                rest_stmt_list + [self.create_pil_assign_name_dup(temp_name, rest_name)])
            
            result_stmt_list = first_stmt_list + [rest_stmt]
            return result_stmt_list, temp_name

        raise NotImplementedError(f"BoolOp {type(op).__name__} is not supported")

    def visit_NamedExpr(self, target: ast.expr, value: ast.expr, node_attr: PILAttr = NOATTR) -> tuple[list[ast.stmt], PILExpr | None]:
        assert isinstance(target, ast.Name), "Python native ast parser should guarantee that the target of NamedExpr is always ast.Name"
        value_stmt_list, value_name = self.visit(value)
        assign_stmt = self.create_pil_assign_name_dup(target.id, value_name)
        result_stmt_list = value_stmt_list + [assign_stmt]
        return result_stmt_list, target.id
    
    def visit_BinOp(self, left: ast.expr, op: ast.operator, right: ast.expr, node_attr: PILAttr = NOATTR) -> tuple[list[ast.stmt], PILExpr | None]:
        left_stmt_list, left_name = self.visit(left)
        right_stmt_list, right_name = self.visit(right)
        
        temp_name = self.create_temp_identifier()
        result_expr = self.create_pil_bin_op(left_name, op, right_name)
        binop_stmt = self.create_pil_assign_name(temp_name, result_expr)
        result_stmt_list = left_stmt_list + right_stmt_list + [binop_stmt]
        return result_stmt_list, temp_name

    def visit_UnaryOp(self, op: ast.unaryop, operand: ast.expr, node_attr: PILAttr = NOATTR) -> tuple[list[ast.stmt], PILExpr | None]:
        operand_stmt_list, operand_name = self.visit(operand)

        temp_name = self.create_temp_identifier()
        result_expr = self.create_pil_unary_op(op, operand_name)
        unaryop_stmt = self.create_pil_assign_name(temp_name, result_expr)
        result_stmt_list = operand_stmt_list + [unaryop_stmt]
        return result_stmt_list, temp_name

    def visit_Lambda(self, args: ast.arguments, body: ast.expr, node_attr: PILAttr = NOATTR) -> tuple[list[ast.stmt], PILExpr | None]:
        raise NotImplementedError("Lambda is not supported")

    def visit_IfExp(self, test: ast.expr, body: ast.expr, orelse: ast.expr, node_attr: PILAttr = NOATTR) -> tuple[list[ast.stmt], PILExpr | None]:
        test_stmt_list, test_name = self.visit(test)
        body_stmt_list, body_name = self.visit(body)
        orelse_stmt_list, orelse_name = self.visit(orelse)

        temp_name = self.create_temp_identifier()
        result_body_stmt_list = body_stmt_list + [self.create_pil_assign_name_dup(temp_name, body_name)]
        result_orelse_stmt_list = orelse_stmt_list + [self.create_pil_assign_name_dup(temp_name, orelse_name)]
        result_if_stmt = self.create_pil_if(test_name, result_body_stmt_list, result_orelse_stmt_list)
        result_stmt_list = test_stmt_list + [result_if_stmt]
        return result_stmt_list, temp_name

    def visit_Dict(self, keys: list[ast.expr | None], values: list[ast.expr], node_attr: PILAttr = NOATTR) -> tuple[list[ast.stmt], PILExpr | None]:
        result_stmt_list = []
        key_name_list = []
        value_name_list = []
        for key, value in zip(keys, values):
            if key is not None:
                key_stmt_list, key_name = self.visit(key)
            else:
                key_stmt_list, key_name = [], None
            value_stmt_list, value_name = self.visit(value)

            result_stmt_list.extend(key_stmt_list)
            result_stmt_list.extend(value_stmt_list)
            key_name_list.append(key_name)
            value_name_list.append(value_name)

        temp_name = self.create_temp_identifier()
        result_expr = self.create_pil_dict(key_name_list, value_name_list)
        result_stmt = self.create_pil_assign_name(temp_name, result_expr)
        result_stmt_list.append(result_stmt)
        return result_stmt_list, temp_name

    def visit_Set(self, elts: list[ast.expr], node_attr: PILAttr = NOATTR) -> tuple[list[ast.stmt], PILExpr | None]:
        result_stmt_list = []

        temp_name = self.create_temp_identifier()
        elt_list = []
        for elt in elts:
            if isinstance(elt, ast.Starred):
                elt_stmt_list, elt_name = self.visit(elt.value)
                elt_list.append((elt_name, True))
            else:
                elt_stmt_list, elt_name = self.visit(elt)
                elt_list.append((elt_name, False))
            result_stmt_list.extend(elt_stmt_list)
        result_expr = self.create_pil_set(elt_list)
        result_stmt = self.create_pil_assign_name(temp_name, result_expr)
        result_stmt_list.append(result_stmt)
        return result_stmt_list, temp_name

    def visit_ListComp(self, elt: ast.expr, generators: list[ast.comprehension], node_attr: PILAttr = NOATTR) -> tuple[list[ast.stmt], PILExpr | None]:
        raise NotImplementedError("ListComp is not supported")

    def visit_SetComp(self, elt: ast.expr, generators: list[ast.comprehension], node_attr: PILAttr = NOATTR) -> tuple[list[ast.stmt], PILExpr | None]:
        raise NotImplementedError("SetComp is not supported")

    def visit_DictComp(self, key: ast.expr, value: ast.expr, generators: list[ast.comprehension], node_attr: PILAttr = NOATTR) -> tuple[list[ast.stmt], PILExpr | None]:
        raise NotImplementedError("DictComp is not supported")

    def visit_GeneratorExp(self, elt: ast.expr, generators: list[ast.comprehension], node_attr: PILAttr = NOATTR) -> tuple[list[ast.stmt], PILExpr | None]:
        raise NotImplementedError("GeneratorExp is not supported")

    def visit_Await(self, value: ast.expr, node_attr: PILAttr = NOATTR) -> tuple[list[ast.stmt], PILExpr | None]:
        raise NotImplementedError("Await is not supported")

    def visit_Yield(self, value: ast.expr | None, node_attr: PILAttr = NOATTR) -> tuple[list[ast.stmt], PILExpr | None]:
        raise NotImplementedError("Yield is not supported")

    def visit_YieldFrom(self, value: ast.expr, node_attr: PILAttr = NOATTR) -> tuple[list[ast.stmt], PILExpr | None]:
        raise NotImplementedError("YieldFrom is not supported")

    def visit_Compare(self, left: ast.expr, ops: list[ast.cmpop], comparators: list[ast.expr], node_attr: PILAttr = NOATTR) -> tuple[list[ast.stmt], PILExpr | None]:
        # Base case: single comparison — visit left and comparator, build compare expr
        left_stmts, left_name = self.visit(left)
        comp_stmts, comp_name = self.visit(comparators[0])
        temp_name = self.create_temp_identifier()
        first_stmt = self.create_pil_assign_name(temp_name, self.create_pil_compare(left_name, ops[0], comp_name))

        if len(ops) == 1:
            return left_stmts + comp_stmts + [first_stmt], temp_name

        # Recursive case: a op0 b op1 c ... => (a op0 b) and (b op1 c ...)
        # comp_name is reused as left of the next comparison (evaluated only once)
        rest_stmt_list, rest_name = self.visit_Compare(
            left=ast.Name(id=comp_name, ctx=ast.Load()),
            ops=ops[1:],
            comparators=comparators[1:])
        rest_stmt = self.create_pil_if(
            temp_name,
            rest_stmt_list + [self.create_pil_assign_name_dup(temp_name, rest_name)],
            [])
        return left_stmts + comp_stmts + [first_stmt, rest_stmt], temp_name

    def visit_Call(self, func: ast.expr, args: list[ast.expr], keywords: list[ast.keyword], node_attr: PILAttr = NOATTR) -> tuple[list[ast.stmt], PILExpr | None]:
        result_stmt_list = []

        # func must be an identifier per PIL spec
        func_stmts, func_name = self.visit(func)
        result_stmt_list.extend(func_stmts)

        # visit positional args
        arg_names = []
        for arg in args:
            arg_stmts, arg_name = self.visit(arg)
            result_stmt_list.extend(arg_stmts)
            arg_names.append(arg_name)

        # visit keyword values, rewrite keyword nodes with resolved names
        pil_keywords = []
        for kw in keywords:
            kw_stmts, kw_name = self.visit(kw.value)
            result_stmt_list.extend(kw_stmts)
            pil_keywords.append(ast.keyword(arg=kw.arg, value=ast.Name(id=kw_name, ctx=ast.Load())))

        temp_name = self.create_temp_identifier()
        result_expr = self.create_pil_call(func_name, arg_names, pil_keywords)
        result_stmt_list.append(self.create_pil_assign_name(temp_name, result_expr))
        return result_stmt_list, temp_name

    def visit_FormattedValue(self, value: ast.expr, conversion: int, format_spec: ast.expr | None, node_attr: PILAttr = NOATTR) -> tuple[list[ast.stmt], PILExpr | None]:
        raise NotImplementedError("FormattedValue is not supported")

    def visit_Interpolation(self, value: ast.expr, str: str, conversion: int, format_spec: ast.expr | None, node_attr: PILAttr = NOATTR) -> tuple[list[ast.stmt], PILExpr | None]:
        raise NotImplementedError("Interpolation is not supported")

    def visit_JoinedStr(self, values: list[ast.expr], node_attr: PILAttr = NOATTR) -> tuple[list[ast.stmt], PILExpr | None]:
        raise NotImplementedError("JoinedStr is not supported")

    def visit_TemplateStr(self, values: list[ast.expr], node_attr: PILAttr = NOATTR) -> tuple[list[ast.stmt], PILExpr | None]:
        raise NotImplementedError("TemplateStr is not supported")

    def visit_Constant(self, value: object, kind: str | None, node_attr: PILAttr = NOATTR) -> tuple[list[ast.stmt], PILExpr | None]:
        return [], self.create_pil_constant(value, kind)

    def visit_Attribute(self, value: ast.expr, attr: str, ctx: ast.expr_context, node_attr: PILAttr = NOATTR) -> tuple[list[ast.stmt], PILExpr | None]:
        assert isinstance(ctx, ast.Load)
        value_stmts, value_name = self.visit(value)
        temp_name = self.create_temp_identifier()
        result_expr = self.create_pil_attribute(value_name, attr)
        result_stmt_list = value_stmts + [self.create_pil_assign_name(temp_name, result_expr)]
        return result_stmt_list, temp_name

    def visit_Subscript(self, value: ast.expr, slice: ast.expr, ctx: ast.expr_context, node_attr: PILAttr = NOATTR) -> tuple[list[ast.stmt], PILExpr | None]:
        assert isinstance(ctx, ast.Load)
        value_stmts, value_name = self.visit(value)

        # Normalize slice into a list of ast.Slice nodes
        slice_stmt_list, pil_slice_list = self.visit_slice(slice)

        temp_name = self.create_temp_identifier()
        result_expr = self.create_pil_subscript(value_name, pil_slice_list)
        result_stmt_list = value_stmts + slice_stmt_list + [self.create_pil_assign_name(temp_name, result_expr)]
        return result_stmt_list, temp_name

    def visit_Starred(self, value: ast.expr, ctx: ast.expr_context, node_attr: PILAttr = NOATTR) -> tuple[list[ast.stmt], PILExpr | None]:
        raise Exception("Starred should not be directly accessed")

    def visit_Name(self, id: str, ctx: ast.expr_context, node_attr: PILAttr = NOATTR) -> tuple[list[ast.stmt], PILExpr | None]:
        assert isinstance(ctx, ast.Load)
        return [], id

    def visit_List(self, elts: list[ast.expr], ctx: ast.expr_context, node_attr: PILAttr = NOATTR) -> tuple[list[ast.stmt], PILExpr | None]:
        assert isinstance(ctx, ast.Load)

        result_stmt_list = []

        temp_name = self.create_temp_identifier()
        elt_list = []
        for elt in elts:
            if isinstance(elt, ast.Starred):
                elt_stmt_list, elt_name = self.visit(elt.value)
                elt_list.append((elt_name, True))
            else:
                elt_stmt_list, elt_name = self.visit(elt)
                elt_list.append((elt_name, False))
            result_stmt_list.extend(elt_stmt_list)
        result_expr = self.create_pil_list(elt_list)
        result_stmt = self.create_pil_assign_name(temp_name, result_expr)
        result_stmt_list.append(result_stmt)
        return result_stmt_list, temp_name

    def visit_Tuple(self, elts: list[ast.expr], ctx: ast.expr_context, node_attr: PILAttr = NOATTR) -> tuple[list[ast.stmt], PILExpr | None]:
        assert isinstance(ctx, ast.Load)

        result_stmt_list = []

        temp_name = self.create_temp_identifier()
        elt_list = []
        for elt in elts:
            if isinstance(elt, ast.Starred):
                elt_stmt_list, elt_name = self.visit(elt.value)
                elt_list.append((elt_name, True))
            else:
                elt_stmt_list, elt_name = self.visit(elt)
                elt_list.append((elt_name, False))
            result_stmt_list.extend(elt_stmt_list)
        result_expr = self.create_pil_tuple(elt_list)
        result_stmt = self.create_pil_assign_name(temp_name, result_expr)
        result_stmt_list.append(result_stmt)
        return result_stmt_list, temp_name

    def visit_Slice(self, lower: ast.expr | None, upper: ast.expr | None, step: ast.expr | None, node_attr: PILAttr = NOATTR) -> tuple[list[ast.stmt], PILExpr | None]:
        raise NotImplementedError("Slice is not supported")

    def visit_stmts(self, stmts: list[ast.stmt]) -> tuple[list[ast.stmt], PILExpr | None]:
        stmt_list = []
        for stmt in stmts:
            result_stmt_list, _ = self.visit(stmt)
            stmt_list.extend(result_stmt_list)
        return stmt_list, None

    def visit(self, node):
        method = 'visit_' + node.__class__.__name__
        visitor = getattr(self, method)
        field_dict = {key: value for key, value in ast.iter_fields(node)}
        node_attr = PILAttr(node)
        return visitor(**field_dict, node_attr = node_attr)
    
    def parse_func(self, func: ast.FunctionDef) -> ast.FunctionDef:
        body_stmt_list, _ = self.visit_stmts(func.body)
        result_func = ast.FunctionDef(
            func.name, func.args, body_stmt_list, func.decorator_list, func.returns, func.type_comment,
            **self.create_attribute(func))
        return result_func

def build_pil(func: ast.FunctionDef) -> ast.FunctionDef:
    ctx = PILContext()
    parser = PythonParser(ctx)
    return parser.parse_func(func)

if __name__ == '__main__':
    def func():
        a = 30
        while a + 30 < 100:
            if a < 20 < 40 < 60:
                a += 10
                continue
        b, [c, *d], *e = f, g, [h, i] = koo()

    src = ast.parse(open(__file__).read())
    c = build_pil(src.body[-1].body[0])
    print(ast.unparse(c))
