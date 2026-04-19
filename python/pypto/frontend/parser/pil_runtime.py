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

import operator
from enum import Enum
from dataclasses import dataclass
import ast

from irbuilder import IRBuilder, ScalarVar, TensorVar, ConstInt, ConstFloat


class BuiltinOp(Enum):
    # Binary arithmetic operators (ast.operator)
    OP_ADD          = 'OP_ADD'
    OP_SUB          = 'OP_SUB'
    OP_MUL          = 'OP_MUL'
    OP_MATMUL       = 'OP_MATMUL'
    OP_DIV          = 'OP_DIV'
    OP_MOD          = 'OP_MOD'
    OP_POW          = 'OP_POW'
    OP_LSHIFT       = 'OP_LSHIFT'
    OP_RSHIFT       = 'OP_RSHIFT'
    OP_BITOR        = 'OP_BITOR'
    OP_BITXOR       = 'OP_BITXOR'
    OP_BITAND       = 'OP_BITAND'
    OP_FLOORDIV     = 'OP_FLOORDIV'
    # Unary operators (ast.unaryop)
    OP_INVERT       = 'OP_INVERT'
    OP_NOT          = 'OP_NOT'
    OP_UADD         = 'OP_UADD'
    OP_USUB         = 'OP_USUB'
    # Comparison operators (ast.cmpop)
    OP_EQ           = 'OP_EQ'
    OP_NE           = 'OP_NE'
    OP_LT           = 'OP_LT'
    OP_LE           = 'OP_LE'
    OP_GT           = 'OP_GT'
    OP_GE           = 'OP_GE'
    OP_IS           = 'OP_IS'
    OP_IS_NOT       = 'OP_IS_NOT'
    OP_IN           = 'OP_IN'
    OP_NOT_IN       = 'OP_NOT_IN'

UNARY_OP_MAP = {
    BuiltinOp.OP_INVERT: operator.invert,
    BuiltinOp.OP_NOT:    operator.not_,
    BuiltinOp.OP_UADD:  operator.pos,
    BuiltinOp.OP_USUB:  operator.neg,
}

BINARY_OP_MAP = {
    # Arithmetic (ast.operator)
    BuiltinOp.OP_ADD:      operator.add,
    BuiltinOp.OP_SUB:      operator.sub,
    BuiltinOp.OP_MUL:      operator.mul,
    BuiltinOp.OP_MATMUL:   operator.matmul,
    BuiltinOp.OP_DIV:      operator.truediv,
    BuiltinOp.OP_MOD:      operator.mod,
    BuiltinOp.OP_POW:      operator.pow,
    BuiltinOp.OP_LSHIFT:   operator.lshift,
    BuiltinOp.OP_RSHIFT:   operator.rshift,
    BuiltinOp.OP_BITOR:    operator.or_,
    BuiltinOp.OP_BITXOR:   operator.xor,
    BuiltinOp.OP_BITAND:   operator.and_,
    BuiltinOp.OP_FLOORDIV: operator.floordiv,
    # Comparison (ast.cmpop)
    BuiltinOp.OP_EQ:       operator.eq,
    BuiltinOp.OP_NE:       operator.ne,
    BuiltinOp.OP_LT:       operator.lt,
    BuiltinOp.OP_LE:       operator.le,
    BuiltinOp.OP_GT:       operator.gt,
    BuiltinOp.OP_GE:       operator.ge,
    BuiltinOp.OP_IS:       operator.is_,
    BuiltinOp.OP_IS_NOT:   operator.is_not,
    BuiltinOp.OP_IN:       operator.contains,
    BuiltinOp.OP_NOT_IN:   lambda a, b: not operator.contains(b, a),
}

class RuntimeValue:
    pass

class ScalarValue(RuntimeValue):
    def __init__(self, value_var: ScalarVar, token_var):
        self._value_var = value_var
        self._token_var = token_var

    def __repr__(self):
        return f'<{self.value}, {self.token}>'

    @property
    def value(self):
        return self._value_var

    @property
    def token(self):
        return self._token_var

    @staticmethod
    def create() -> 'ScalarValue':
        irb = IRBuilder.get_builder()
        result_value = irb.create_scalar_var(irb.temp_name())
        result_token = irb.create_token_var(irb.temp_name())
        return ScalarValue(result_value, result_token)


class TensorValue(RuntimeValue):
    def __init__(self, value_var: TensorVar, token_var):
        self._value_var = value_var
        self._token_var = token_var

    def __repr__(self):
        return f'<{self.value}, {self.token}>'

    @property
    def value(self):
        return self._value_var

    @property
    def token(self):
        return self._token_var

    @staticmethod
    def create() -> 'TensorValue':
        irb = IRBuilder.get_builder()
        result_value = irb.create_tensor_var(irb.temp_name())
        result_token = irb.create_token_var(irb.temp_name())
        return TensorValue(result_value, result_token)

@dataclass
class DynamicLoop:
    begin:  ScalarValue | int
    end:    ScalarValue | int
    step:   ScalarValue | int

def loop(*args) -> DynamicLoop:
    irb = IRBuilder.get_builder()
    if len(args) == 1:
        begin, end, step = irb.create_const_int(0), args[0], irb.create_const_int(1)
    elif len(args) == 2:
        begin, end, step = args[0], args[1], irb.create_const_int(1)
    elif len(args) == 3:
        begin, end, step = args[0], args[1], args[2]
    else:
        raise Exception("invalid")
    return DynamicLoop(begin, end, step)

@dataclass
class DynamicCond:
    cond:   ScalarValue

class DynamicFunc:
    def __init__(self, handler):
        self._handler = handler

    @property
    def handler(self):
        return self._handler

    def __call__(self, *args, **kwargs):
        return self._handler(*args, **kwargs)

RuntimeValue = ScalarValue | TensorValue | object | None

def is_tensor(obj : RuntimeValue) -> bool:
    return isinstance(obj, TensorValue)

def is_scalar(obj: RuntimeValue) -> bool:
    return isinstance(obj, ScalarValue)

def is_const(obj: RuntimeValue) -> bool:
    return isinstance(obj, (int, float, bool, ConstInt, ConstFloat))

def eval_unary_op(op: BuiltinOp, value: RuntimeValue) -> RuntimeValue:
    irb = IRBuilder.get_builder()
    if is_scalar(value):
        result_scalar = irb.create_scalar_var(irb.temp_name())
        result_token = irb.create_token_var(irb.temp_name())
        irb.create_scalar_expr_op_assign_stmt([result_scalar], result_token, op.value, [value.value])
        result_value = ScalarValue(result_scalar, result_token)
    else:
        result_value = UNARY_OP_MAP[op](value)
    return result_value

def eval_binary_op(op: BuiltinOp, lhs: RuntimeValue, rhs: RuntimeValue) -> RuntimeValue:
    irb = IRBuilder.get_builder()
    if is_tensor(lhs) or is_tensor(rhs):
        result_tensor = irb.create_tensor_var(irb.temp_name(), None, None, [], [], [], [])
        result_token = irb.create_token_var(irb.temp_name())
        if is_scalar(lhs) or is_const(lhs):
            irb.create_tensor_op_assign_stmt(
                [result_tensor], [result_token], op.value, [rhs.value], [lhs.token], attrs=[
                    irb.create_attr('lhs', lhs.value)
                ])
        elif is_scalar(rhs) or is_const(rhs):
            irb.create_tensor_op_assign_stmt(
                [result_tensor], [result_token], op.value, [lhs.value], [rhs.token], attrs=[
                    irb.create_attr('rhs', rhs.value)
                ])
        else:
            irb.create_tensor_op_assign_stmt(
                [result_tensor], [result_token], op.value, [lhs.value, rhs.value])
        result_value = TensorValue(result_tensor, result_token)
    elif is_scalar(lhs) or is_scalar(rhs):
        result_scalar = irb.create_scalar_var(irb.temp_name())
        result_token = irb.create_token_var(irb.temp_name())
        if is_const(lhs):
            irb.create_scalar_expr_op_assign_stmt([result_scalar], result_token, op.value, [lhs.value, rhs.value])
        elif is_const(rhs):
            irb.create_scalar_expr_op_assign_stmt([result_scalar], result_token, op.value, [lhs.value, rhs.value])
        else:
            irb.create_scalar_expr_op_assign_stmt([result_scalar], result_token, op.value, [lhs.value, rhs.value])
        result_value = ScalarValue(result_scalar, result_token)
    else:
        result_value = BINARY_OP_MAP[op](lhs, rhs)
    return result_value
