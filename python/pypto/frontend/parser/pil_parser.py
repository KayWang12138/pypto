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

import ast
import operator

from irbuilder import IRBuilder, StmtList, AssignStmt
from pil import PILAttr, NOATTR, PIL_DEFAULT_PREFIX
from pil_lvalue import LValuePath, LValueMapping, LValueMissing, LValueTypeMismatch
from pil_runtime import (
    BuiltinOp, TensorValue, ScalarValue, RuntimeValue,
    DynamicLoop, DynamicCond, DynamicFunc,
    eval_unary_op, eval_binary_op,
    is_scalar, is_const
)

UNARY_OP_MAP = {
    ast.UAdd:   BuiltinOp.OP_UADD,
    ast.USub:   BuiltinOp.OP_USUB,
    ast.Not:    BuiltinOp.OP_NOT,
    ast.Invert: BuiltinOp.OP_INVERT,
}

BINARY_OP_MAP = {
    # Binary arithmetic operators (ast.operator)
    ast.Add:      BuiltinOp.OP_ADD,
    ast.Sub:      BuiltinOp.OP_SUB,
    ast.Mult:     BuiltinOp.OP_MUL,
    ast.MatMult:  BuiltinOp.OP_MATMUL,
    ast.Div:      BuiltinOp.OP_DIV,
    ast.Mod:      BuiltinOp.OP_MOD,
    ast.Pow:      BuiltinOp.OP_POW,
    ast.LShift:   BuiltinOp.OP_LSHIFT,
    ast.RShift:   BuiltinOp.OP_RSHIFT,
    ast.BitOr:    BuiltinOp.OP_BITOR,
    ast.BitXor:   BuiltinOp.OP_BITXOR,
    ast.BitAnd:   BuiltinOp.OP_BITAND,
    ast.FloorDiv: BuiltinOp.OP_FLOORDIV,
    # Comparison operators (ast.cmpop)
    ast.Eq:       BuiltinOp.OP_EQ,
    ast.NotEq:    BuiltinOp.OP_NE,
    ast.Lt:       BuiltinOp.OP_LT,
    ast.LtE:      BuiltinOp.OP_LE,
    ast.Gt:       BuiltinOp.OP_GT,
    ast.GtE:      BuiltinOp.OP_GE,
    ast.Is:       BuiltinOp.OP_IS,
    ast.IsNot:    BuiltinOp.OP_IS_NOT,
    ast.In:       BuiltinOp.OP_IN,
    ast.NotIn:    BuiltinOp.OP_NOT_IN,
}

ParseResult = tuple[LValueMapping, StmtList, RuntimeValue]

class ParseError(Exception):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class PILParser:

    def __init__(self, prefix=PIL_DEFAULT_PREFIX):
        self._prefix = prefix
        self.irb = IRBuilder.get_builder()

    @classmethod
    def get_parser(cls):
        return cls.parser

    def is_pil_temp(self, path: LValuePath):
        return path[0].startswith(self._prefix)

    def parse_expr(self, mapping: LValueMapping, expr: ast.expr) -> RuntimeValue:
        if isinstance(expr, ast.Constant):
            if isinstance(expr.value, int):
                return self.irb.create_const_int(expr.value)
            elif isinstance(expr.value, float):
                return self.irb.create_const_float(expr.value)
        elif isinstance(expr, ast.Name):
            return mapping.get_var(expr.id)
        raise Exception(f"Impossible {expr}")

    def parse_subscript(self, mapping: LValueMapping, expr: ast.expr) -> RuntimeValue | list[RuntimeValue | tuple[RuntimeValue, RuntimeValue, RuntimeValue]]:
        def _parse_single(e: ast.expr) -> RuntimeValue | tuple[RuntimeValue, RuntimeValue, RuntimeValue]:
            if isinstance(e, ast.Slice):
                return (
                    self.parse_expr(mapping, e.lower) if e.lower is not None else None,
                    self.parse_expr(mapping, e.upper) if e.upper is not None else None,
                    self.parse_expr(mapping, e.step) if e.step is not None else None,
                )
            else:
                return self.parse_expr(mapping, e)

        if isinstance(expr, ast.Tuple):
            return [_parse_single(e) for e in expr.elts]
        else:
            return _parse_single(expr)

    def eval_native_call(self, func, *args, **kwargs) -> tuple[StmtList, RuntimeValue]:
        self.irb.capture_begin()
        result_value = func(*args, **kwargs)
        result_stmt_list = self.irb.capture_end()
        return result_stmt_list, result_value

    def eval_native_generator_list(self, arg) -> tuple[StmtList, list[RuntimeValue]]:
        self.irb.capture_begin()
        result_value_list = [*arg]
        result_stmt_list = self.irb.capture_end()
        return result_stmt_list, result_value_list

    def eval_native_generator_dict(self, arg) -> tuple[list[AssignStmt], dict[RuntimeValue, RuntimeValue]]:
        self.irb.capture_begin()
        result_value_dict = {**arg}
        result_stmt_list = self.irb.capture_end()
        return result_stmt_list, result_value_dict

    # ------------------------------------------------------------------
    # Statement visitors
    # ------------------------------------------------------------------

    def visit_FunctionDef(self, mapping: LValueMapping, name: str, args: ast.arguments, body: list[ast.stmt], decorator_list: list[ast.expr], returns: ast.expr | None, type_comment: str | None, node_attr: PILAttr = NOATTR, **kwargs) -> ParseResult:
        def handler(*real_args, **real_kwargs):
            pass
        mapping.set_var(name, DynamicFunc(handler))

    def visit_Assign(self, mapping: LValueMapping, targets: list[ast.expr], value: ast.expr, type_comment: str | None, node_attr: PILAttr = NOATTR) -> ParseResult:
        assert len(targets) == 1
        if isinstance(targets[0], ast.Name):
            mapping, value_stmt_list, value_value = self.visit(mapping, value)
            mapping.set_var(targets[0].id, value_value)
            return mapping, value_stmt_list, None
        else:
            value_value = self.parse_expr(mapping, value)
            if isinstance(targets[0], ast.Attribute):
                target_value, name = self.parse_expr(mapping, targets[0].target), targets[0].name
                call_stmt_list, result_value = self.eval_native_call(setattr, target_value, name, value_value)
                return mapping, call_stmt_list, None

            elif isinstance(targets[0], ast.Subscript):
                target_value, slice_value = self.parse_expr(mapping, targets[0].target), self.parse_subscript(mapping, targets[0].slice)
                call_stmt_list, result_value = self.eval_native_call(operator.setitem, target_value, slice_value, value_value)
                return mapping, call_stmt_list, None

            elif isinstance(targets[0], [ast.Tuple, ast.List]):
                gen_stmt_list, gen_value = self.eval_native_generator_list(value_value)

                starred_index = None
                starred_count = 0
                for idx in range(len(targets[0].elts)):
                    if isinstance(targets[0].elts[idx], ast.Starred):
                        starred_index = idx
                        starred_count = len(gen_value) - len(targets[0].elts) + 1

                if starred_index is None:
                    if len(targets[0].elts) != len(gen_value):
                        # raise error
                        raise Exception("assign number mismatch")
                    for idx in range(len(targets[0].elts)):
                        mapping.set_var(targets[0].elts[idx].name, gen_value[idx])
                else:
                    if len(targets[0].elts) > len(gen_value):
                        # raise error
                        raise Exception("assign number mismatch")
                    for idx in range(len(targets[0].elts)):
                        if idx < starred_index:
                            mapping.set_var(targets[0].elts[idx].name, gen_value[idx])
                        elif idx == starred_index:
                            mapping.set_var(targets[0].elts[idx].name, list(gen_value[idx : idx + starred_count]))
                        else:
                            mapping.set_var(targets[0].elts[idx].name, gen_value[idx + starred_count - 1])

                return mapping, gen_stmt_list, None
        raise NotImplementedError("visit_Assign is not implemented")

    def visit_Return(self, mapping: LValueMapping, value: ast.expr | None, node_attr: PILAttr = NOATTR) -> ParseResult:
        raise NotImplementedError("visit_Return is not implemented")

    def visit_Delete(self, mapping: LValueMapping, targets: list[ast.expr], node_attr: PILAttr = NOATTR) -> ParseResult:
        raise NotImplementedError("visit_Delete is not implemented")

    def visit_For(self, mapping: LValueMapping, target: ast.expr, iter: ast.expr, body: list[ast.stmt], orelse: list[ast.stmt], type_comment: str | None, node_attr: PILAttr = NOATTR) -> ParseResult:
        iter_value = self.parse_expr(mapping, iter)
        if not isinstance(iter_value, DynamicLoop):
            mapping_curr = mapping
            result_stmt_list = []
            for k in iter_value:
                mapping_curr.set_var(target.id, k)
                mapping_curr, iter_stmt_list, _ = self.visit_stmts(mapping_curr, body)
                result_stmt_list.extend(iter_stmt_list)
            return mapping_curr, result_stmt_list, None
        else:
            sym_mapping = mapping.clone_symbolic()
            loop_var = ScalarValue.create()
            loop_start, loop_stop, loop_step = iter_value.begin, iter_value.end, iter_value.step
            sym_mapping.set_var(target.id, loop_var)
            body_mapping, body_stmt_list, _ = self.visit_stmts(sym_mapping.clone(), body)
            path_set = sym_mapping.get_path_set() | body_mapping.get_path_set()

            result_value_list, iterarg_value_list, init_value_list, body_yield_value_list = [], [], [], []
            for path in path_set:
                if self.is_pil_temp(path):
                    continue
                print(path)
                sym_value = sym_mapping.get_value(path)
                body_value = body_mapping.get_value(path)
                if sym_value == body_value:
                    continue
                if sym_value is LValueMissing or body_value is LValueMissing:
                    continue
                if sym_value.__class__ != body_value.__class__:
                    result_var = LValueTypeMismatch(sym_value, body_value)
                else:
                    if isinstance(sym_value, ScalarValue):
                        result_var, iterarg_var = ScalarValue.create(), ScalarValue.create()
                    elif isinstance(sym_value, TensorValue):
                        result_var, iterarg_var = TensorValue.create(), TensorValue.create()
                    result_value_list.append(result_var)
                    iterarg_value_list.append(iterarg_var)
                    init_value_list.append(mapping.get_value(path))
                    body_yield_value_list.append(body_value)
                mapping.set_value(path, result_var)
            result_body_stmt_list = body_stmt_list + [
                self.irb.create_yield_stmt(self.get_var_list(body_yield_value_list))
            ]
            result_var_list = self.get_var_list(result_value_list)
            iterarg_var_list = self.get_var_list(iterarg_value_list)
            result_stmt_list = [
                self.irb.create_for_stmt(result_var_list,
                                         iterarg_var_list, init_value_list,
                                         loop_var.value, loop_start, loop_stop, loop_step,
                                         result_body_stmt_list)
            ]
            return mapping, result_stmt_list, None

    def visit_While(self, mapping: LValueMapping, test: ast.expr, body: list[ast.stmt], orelse: list[ast.stmt], node_attr: PILAttr = NOATTR) -> ParseResult:
        raise NotImplementedError("visit_While is not implemented")

    def get_var_list(self, value_list):
        var_list = []
        for value in value_list:
            var_list.extend([value.value, value.token])
        return var_list

    def get_cond_var(self, value):
        if is_scalar(value):
            return value.value
        elif isinstance(value, DynamicCond):
            return value.cond
        raise Exception("cond var")

    def visit_If(self, mapping: LValueMapping, test: ast.expr, body: list[ast.stmt], orelse: list[ast.stmt], node_attr: PILAttr = NOATTR) -> ParseResult:
        test_value = self.parse_expr(mapping, test)
        if is_const(test_value):
            if test_value:
                return self.visit_stmts(mapping, body)
            else:
                return self.visit_stmts(mapping, orelse)
        elif is_scalar(test_value) or isinstance(test_value, DynamicCond):
            body_mapping, body_stmt_list, _ = self.visit_stmts(mapping.clone(), body)
            orelse_mapping, orelse_stmt_list, _ = self.visit_stmts(mapping.clone(), orelse)
            path_set = mapping.get_path_set() | body_mapping.get_path_set() | orelse_mapping.get_path_set()

            result_value_list, then_yield_value_list, else_yield_value_list = [], [], []
            for path in path_set:
                if self.is_pil_temp(path):
                    continue
                curr_value = mapping.get_value(path)
                body_value = body_mapping.get_value(path)
                orelse_value = orelse_mapping.get_value(path)
                if curr_value == body_value == orelse_value:
                    continue
                yield_value_list = [
                    (body_value, orelse_value),
                    (body_value, curr_value),
                    (curr_value, orelse_value)
                ]
                result_var = None
                for then_value_yield, else_value_yield in yield_value_list:
                    if then_value_yield is LValueMissing or else_value_yield is LValueMissing:
                        continue
                    if then_value_yield.__class__ != else_value_yield.__class__:
                        result_var = LValueTypeMismatch(then_value_yield, else_value_yield)
                        break
                    if isinstance(then_value_yield, ScalarValue):
                        result_var = ScalarValue.create()
                    elif isinstance(then_value_yield, TensorValue):
                        result_var = TensorValue.create()
                    else:
                        raise Exception("type mismatch")
                    result_value_list.append(result_var)
                    then_yield_value_list.append(then_value_yield)
                    else_yield_value_list.append(else_value_yield)
                    break
                if result_var is not None:
                    mapping.set_value(path, result_var)

            result_var_list = self.get_var_list(result_value_list)
            result_then_stmt_list = body_stmt_list + [
                self.irb.create_yield_stmt(self.get_var_list(then_yield_value_list))
            ]
            result_else_stmt_list = orelse_stmt_list + [
                self.irb.create_yield_stmt(self.get_var_list(else_yield_value_list))
            ]

            result_stmt = self.irb.create_if_stmt(result_var_list, self.get_cond_var(test_value), result_then_stmt_list, result_else_stmt_list)
            return mapping, [result_stmt], None
        else:
            raise NotImplementedError("visit_If is not implemented")

    def visit_With(self, mapping: LValueMapping, items: list[ast.withitem], body: list[ast.stmt], type_comment: str | None, node_attr: PILAttr = NOATTR) -> ParseResult:
        raise NotImplementedError("visit_With is not implemented")

    def visit_Raise(self, mapping: LValueMapping, exc: ast.expr | None, cause: ast.expr | None, node_attr: PILAttr = NOATTR) -> ParseResult:
        raise NotImplementedError("visit_Raise is not implemented")

    def visit_Try(self, mapping: LValueMapping, body: list[ast.stmt], handlers: list[ast.excepthandler], orelse: list[ast.stmt], finalbody: list[ast.stmt], node_attr: PILAttr = NOATTR) -> ParseResult:
        raise NotImplementedError("visit_Try is not implemented")

    def visit_Assert(self, mapping: LValueMapping, test: ast.expr, msg: ast.expr | None, node_attr: PILAttr = NOATTR) -> ParseResult:
        raise NotImplementedError("visit_Assert is not implemented")

    def visit_Import(self, mapping: LValueMapping, names: list[ast.alias], node_attr: PILAttr = NOATTR) -> ParseResult:
        raise NotImplementedError("visit_Import is not implemented")

    def visit_ImportFrom(self, mapping: LValueMapping, module: str | None, names: list[ast.alias], level: int | None, node_attr: PILAttr = NOATTR) -> ParseResult:
        raise NotImplementedError("visit_ImportFrom is not implemented")

    def visit_Global(self, mapping: LValueMapping, names: list[str], node_attr: PILAttr = NOATTR) -> ParseResult:
        raise NotImplementedError("visit_Global is not implemented")

    def visit_Nonlocal(self, mapping: LValueMapping, names: list[str], node_attr: PILAttr = NOATTR) -> ParseResult:
        raise NotImplementedError("visit_Nonlocal is not implemented")

    def visit_Pass(self, mapping: LValueMapping, node_attr: PILAttr = NOATTR) -> ParseResult:
        raise NotImplementedError("visit_Pass is not implemented")

    def visit_Break(self, mapping: LValueMapping, node_attr: PILAttr = NOATTR) -> ParseResult:
        raise NotImplementedError("visit_Break is not implemented")

    def visit_Continue(self, mapping: LValueMapping, node_attr: PILAttr = NOATTR) -> ParseResult:
        raise NotImplementedError("visit_Continue is not implemented")

    # ------------------------------------------------------------------
    # Expression visitors
    # ------------------------------------------------------------------

    def visit_BinOp(self, mapping: LValueMapping, left: ast.expr, op: ast.operator, right: ast.expr, node_attr: PILAttr = NOATTR) -> ParseResult:
        left_value = self.parse_expr(mapping, left)
        right_value = self.parse_expr(mapping, right)
        stmt_list, result_value = self.eval_native_call(eval_binary_op, BINARY_OP_MAP[op.__class__], left_value, right_value)
        return mapping, stmt_list, result_value

    def visit_UnaryOp(self, mapping: LValueMapping, op: ast.unaryop, operand: ast.expr, node_attr: PILAttr = NOATTR) -> ParseResult:
        operand_value = self.parse_expr(mapping, operand)
        stmt_list, result_value = self.eval_native_call(eval_unary_op, UNARY_OP_MAP[op.__class__], operand_value)
        return mapping, stmt_list, result_value

    def visit_Dict(self, mapping: LValueMapping, keys: list[ast.expr | None], values: list[ast.expr], node_attr: PILAttr = NOATTR) -> ParseResult:
        result_stmt_list = []
        result_value_dict = {}
        for key, value in zip(keys, values):
            key_value = self.parse_expr(mapping, key)
            value_value = self.parse_expr(mapping, value)
            if key_value is None:
                expand_stmt_list, expand_value_dict = self.eval_native_generator_dict(value_value)
                result_stmt_list.extend(expand_stmt_list)
                result_value_dict.update(expand_value_dict)
            else:
                result_value_dict[key_value] = value_value
        return mapping, result_stmt_list, result_value_dict

    def visit_Set(self, mapping: LValueMapping, elts: list[ast.expr], node_attr: PILAttr = NOATTR) -> ParseResult:
        result_stmt_list = []
        result_value_list = []
        for elt in elts:
            if isinstance(elt, ast.Starred):
                expand = self.parse_expr(mapping, elt.value)
                expand_stmt_list, expand_value_list = self.eval_native_generator_list(expand)
                result_stmt_list.extend(expand_stmt_list)
                result_value_list.extend(expand_value_list)
            else:
                elt_value = self.parse_expr(mapping, elt)
                result_value_list.append(elt_value)
        return mapping, result_stmt_list, set(result_value_list)

    def visit_Compare(self, mapping: LValueMapping, left: ast.expr, ops: list[ast.cmpop], comparators: list[ast.expr], node_attr: PILAttr = NOATTR) -> ParseResult:
        assert len(ops) == 1
        left_value = self.parse_expr(mapping, left)
        right_value = self.parse_expr(mapping, comparators[0])
        stmt_list, result_value = self.eval_native_call(eval_binary_op, BINARY_OP_MAP[ops[0].__class__], left_value, right_value)
        return mapping, stmt_list, result_value

    def visit_Call(self, mapping: LValueMapping, func: ast.expr, args: list[ast.expr], keywords: list[ast.keyword], node_attr: PILAttr = NOATTR) -> ParseResult:
        result_stmt_list = []
        func_value = self.parse_expr(mapping, func)
        arg_value_list = []
        keyword_value_dict = {}
        for arg in args:
            if isinstance(arg, ast.Starred):
                expand = self.parse_expr(mapping, arg.value)
                expand_stmt_list, expand_value_list = self.eval_native_generator_list(expand)
                result_stmt_list.extend(expand_stmt_list)
                arg_value_list.extend(expand_value_list)
            else:
                arg_value = self.parse_expr(mapping, arg)
                arg_value_list.append(arg_value)
        for keyword in keywords:
            arg_value = self.parse_expr(mapping, keyword.arg)
            value_value = self.parse_expr(mapping, keyword.value)
            if arg_value is None:
                expand_stmt_list, expand_value_dict = self.eval_native_generator_dict(value_value)
                result_stmt_list.extend(expand_stmt_list)
                keyword_value_dict.update(expand_value_dict)
            else:
                keyword_value_dict[arg_value] = value_value

        if isinstance(func_value, DynamicFunc):
            call_stmt_list, result_value = self.eval_native_call(func_value.handler, *arg_value_list, **keyword_value_dict)
        else:
            call_stmt_list, result_value = self.eval_native_call(func_value, *arg_value_list, **keyword_value_dict)
        result_stmt_list.extend(call_stmt_list)
        return mapping, result_stmt_list, result_value

    def visit_Yield(self, mapping: LValueMapping, value: ast.expr | None, node_attr: PILAttr = NOATTR) -> ParseResult:
        raise NotImplementedError("visit_Yield is not implemented")

    def visit_YieldFrom(self, mapping: LValueMapping, value: ast.expr, node_attr: PILAttr = NOATTR) -> ParseResult:
        raise NotImplementedError("visit_YieldFrom is not implemented")

    def visit_Constant(self, mapping: LValueMapping, value: object, kind: str | None, node_attr: PILAttr = NOATTR) -> ParseResult:
        if isinstance(value, (int, bool)):
            result = self.irb.create_const_int(value)
        elif isinstance(value, float):
            result = self.irb.create_const_float(value)
        else:
            result = value
        return mapping, [], result

    def visit_Name(self, mapping: LValueMapping, id: str, ctx: ast.expr_context, node_attr: PILAttr = NOATTR) -> ParseResult:
        return mapping, [], mapping.get_var(id)

    def visit_Attribute(self, mapping: LValueMapping, value: ast.expr, attr: str, ctx: ast.expr_context, node_attr: PILAttr = NOATTR) -> ParseResult:
        assert isinstance(ctx, ast.Load)
        target_value = self.parse_expr(mapping, value)
        result_stmt_list, result_value = self.eval_native_call(getattr, target_value, attr)
        return mapping, result_stmt_list, result_value

    def visit_Subscript(self, mapping: LValueMapping, value: ast.expr, slice: ast.expr, ctx: ast.expr_context, node_attr: PILAttr = NOATTR) -> ParseResult:
        assert isinstance(ctx, ast.Load)
        target_value = self.parse_expr(mapping, value)
        subscript_value = self.parse_subscript(mapping, slice)
        result_stmt_list, result_value = self.eval_native_call(operator.getitem, target_value, subscript_value)
        return mapping, result_stmt_list, result_value

    def visit_Starred(self, mapping: LValueMapping, value: ast.expr, ctx: ast.expr_context, node_attr: PILAttr = NOATTR) -> ParseResult:
        raise NotImplementedError("visit_Starred is not implemented")

    def visit_List(self, mapping: LValueMapping, elts: list[ast.expr], ctx: ast.expr_context, node_attr: PILAttr = NOATTR) -> ParseResult:
        result_stmt_list = []
        result_value_list = []
        for elt in elts:
            if isinstance(elt, ast.Starred):
                expand = self.parse_expr(mapping, elt.value)
                expand_stmt_list, expand_value_list = self.eval_native_generator_list(expand)
                result_stmt_list.extend(expand_stmt_list)
                result_value_list.extend(expand_value_list)
            else:
                elt_value = self.parse_expr(mapping, elt)
                result_value_list.append(elt_value)
        return mapping, result_stmt_list, result_value_list

    def visit_Tuple(self, mapping: LValueMapping, elts: list[ast.expr], ctx: ast.expr_context, node_attr: PILAttr = NOATTR) -> ParseResult:
        result_stmt_list = []
        result_value_list = []
        for elt in elts:
            if isinstance(elt, ast.Starred):
                expand = self.parse_expr(mapping, elt.value)
                expand_stmt_list, expand_value_list = self.eval_native_generator_list(expand)
                result_stmt_list.extend(expand_stmt_list)
                result_value_list.extend(expand_value_list)
            else:
                elt_value = self.parse_expr(mapping, elt)
                result_value_list.append(elt_value)
        return mapping, result_stmt_list, tuple(result_value_list)

    def visit_Slice(self, mapping: LValueMapping, lower: ast.expr | None, upper: ast.expr | None, step: ast.expr | None, node_attr: PILAttr = NOATTR) -> ParseResult:
        raise NotImplementedError("visit_Slice is not implemented")

    def visit_stmts(self, mapping: LValueMapping, stmts: list[ast.stmt]) -> ParseResult:
        result_stmt_list = []
        result_mapping = mapping
        for stmt in stmts:
            result_mapping, stmt_list, _ = self.visit(result_mapping, stmt)
            result_stmt_list.extend(stmt_list)
        return result_mapping, result_stmt_list, None

    def visit(self, mapping:LValueMapping, node) -> ParseResult:
        method = 'visit_' + node.__class__.__name__
        visitor = getattr(self, method)
        field_dict = {key: value for key, value in ast.iter_fields(node)}
        node_attr = PILAttr(node)
        return visitor(mapping, **field_dict, node_attr = node_attr)
