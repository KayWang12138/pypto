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
import inspect
import textwrap

#import pypto.frontend.parser.pil as pil
import pil



class Expr:

    trace = []

    @staticmethod
    def clear():
        Expr.trace.clear()

    @staticmethod
    def true(n):
        Expr.trace.append(('true', n))
        return True

    @staticmethod
    def false(n):
        Expr.trace.append(('false', n))
        return False

    @staticmethod
    def str(n):
        Expr.trace.append(('str', n))
        return f'str({n})'

    @staticmethod
    def int(n):
        Expr.trace.append(('int', n))
        return n

    def __init__(self, value):
        self._item_dict = {}
        self._attr_dict = {}
        self._value = value
        Expr.trace.append(('init', self._value))

    def __getitem__(self, item):
        Expr.trace.append(('getitem', self._value, item))
        return self._item_dict[item]

    def __setitem__(self, item, value):
        Expr.trace.append(('setitem', self._value, item, value))
        self._item_dict[item] = value

    def __delitem__(self, item):
        Expr.trace.append(('delitem', self._value, item))
        del self._item_dict[item]

    def __eq__(self, other):
        return self._attr_dict == other._attr_dict and self._item_dict == other._item_dict and self._value == other._value

    def decorate(self, n):
        Expr.trace.append(('decorate', n))
        def wrapper(func):
            Expr.trace.append(('decorate.wrapper', n))
            return func
        return wrapper

    def attr(method_dict, name):
        @property
        def field(self):
            Expr.trace.append(('getattr', self._value, name))
            return self._attr_dict[name]

        @field.setter
        def field(self, value):
            Expr.trace.append(('setattr', self._value, name, value))
            self._attr_dict[name] = value

        @field.deleter
        def field(self):
            Expr.trace.append(('delattr', self._value, name))
            del self._attr_dict[name]

        method_dict[name] = field

    attr(locals(), 'val')

    class ContextManager:
        def __init__(self, enter_n=None, exit_n=None, init_n=None):
            self._enter_n = enter_n
            self._exit_n = exit_n
            if init_n is not None:
                Expr.str(init_n)
        def __enter__(self):
            if self._enter_n is not None:
                Expr.str(self._enter_n)
            return self
        def __exit__(self, *a):
            if self._exit_n is not None:
                Expr.str(self._exit_n)

        def __eq__(self, other):
            return self._enter_n == other._enter_n and self._exit_n == other._exit_n

    class ValueError(Exception):
        def __init__(self, value):
            Expr.trace.append(('error', value))
            self._value = value

        def __eq__(self, other):
            return self._value == other._value

    class TypeA(ValueError):
        pass

    class TypeB(ValueError):
        pass

    class TypeC(ValueError):
        pass

class TestParser:

    target_list = []

    def __init__(self):
        Expr.trace.clear()

        TestParser.target_list = []

    @staticmethod
    def test(target):
        TestParser.target_list.append(target)

    def __enter__(self):
        TestParser.target_list.clear()

    def __exit__(self, exc_type, exc, tb):
        self.run()

    def run_ast(self, stmt_list):
        Expr.clear()
        src = ast.unparse(stmt_list)
        exec_global = {'Expr': Expr}
        try:
            print('-' * 100)
            print(src)
            exec(src, exec_global)
        except:
            print('\n'.join([f'{lineno + 1:3d} | {line}' for lineno, line in enumerate(src.strip().split('\n'))]))
            raise
        run_trace = Expr.trace[:]
        run_vardict = {name: value for name, value in exec_global.items() if name.startswith('var_')}
        return run_trace, run_vardict

    def run(self):
        for target in TestParser.target_list:
            source_lines, _ = inspect.getsourcelines(target)
            source = textwrap.dedent(''.join(source_lines))
            stmt_list = ast.parse(source).body[0].body

            python_trace, python_vardict = self.run_ast(stmt_list)
            pil_trace, pil_vardict = self.run_ast(pil.parse_stmts(stmt_list))

            assert python_trace == pil_trace, f'{target.__name__}: {python_trace=} {pil_trace=}'
            assert python_vardict == pil_vardict, f'{target.__name__}: {python_vardict=} {pil_vardict=}'

def test_pil_parser_boolop():

    with TestParser():

        @TestParser.test
        def true_and_true_and_true():
            if Expr.true(0) and Expr.true(1) and Expr.true(2):
                Expr.str(3)
            else:
                Expr.str(4)

        @TestParser.test
        def true_and_false_and_true():
            if Expr.true(0) and Expr.false(1) and Expr.true(2):
                Expr.str(3)
            else:
                Expr.str(4)

        @TestParser.test
        def false_and_true_and_true():
            if Expr.false(0) and Expr.true(1) and Expr.true(2):
                Expr.str(3)
            else:
                Expr.str(4)

        @TestParser.test
        def false_or_false_or_false():
            if Expr.true(0) and Expr.true(1) and Expr.true(2):
                Expr.str(3)
            else:
                Expr.str(4)

        @TestParser.test
        def false_or_true_or_false():
            if Expr.true(0) and Expr.false(1) and Expr.true(2):
                Expr.str(3)
            else:
                Expr.str(4)

        @TestParser.test
        def true_or_false_or_false():
            if Expr.false(0) and Expr.true(1) and Expr.true(2):
                Expr.str(3)
            else:
                Expr.str(4)

def test_pil_parser_ifexp():

    with TestParser():

        @TestParser.test
        def true_test_body_selected():
            var_x = Expr.str(0) if Expr.true(1) else Expr.str(2)

        @TestParser.test
        def false_test_orelse_selected():
            var_x = Expr.str(0) if Expr.false(1) else Expr.str(2)

        @TestParser.test
        def nested_ifexp_in_body():
            var_x = (Expr.str(0) if Expr.true(1) else Expr.str(2)) if Expr.true(3) else Expr.str(4)

        @TestParser.test
        def nested_ifexp_in_orelse():
            var_x = Expr.str(0) if Expr.false(1) else (Expr.str(2) if Expr.true(3) else Expr.str(4))

        @TestParser.test
        def nested_ifexp_in_orelse_false():
            var_x = Expr.str(0) if Expr.false(1) else (Expr.str(2) if Expr.false(3) else Expr.str(4))

        @TestParser.test
        def ifexp_as_if_test():
            if Expr.str(0) if Expr.true(1) else Expr.str(2):
                Expr.str(3)
            else:
                Expr.str(4)

def test_pil_parser_function_def():

    with TestParser():

        # --- 3-level nesting ---

        @TestParser.test
        def three_level_nesting():
            def outer():
                def middle():
                    def inner():
                        Expr.str(0)
                    inner()
                middle()
            outer()

        @TestParser.test
        def three_level_nesting_with_return_values():
            def outer():
                def middle():
                    def inner():
                        Expr.str(0)
                        Expr.str(1)
                    inner()
                    Expr.str(2)
                middle()
                Expr.str(3)
            outer()

        # --- decorator ---

        @TestParser.test
        def function_with_decorator():
            var_e = Expr(0)
            @var_e.decorate(1)
            def func():
                Expr.str(2)
            func()

        @TestParser.test
        def function_with_multiple_decorators():
            var_e = Expr(0)
            @var_e.decorate(1)
            @var_e.decorate(2)
            def func():
                Expr.str(3)
            func()

        # --- arg with default values ---

        @TestParser.test
        def function_with_default_arg():
            def func(x=Expr.int(0)):
                Expr.str(x)
            func()
            func(Expr.int(1))

        @TestParser.test
        def function_with_multiple_defaults():
            def func(x=Expr.int(0), y=Expr.int(1)):
                Expr.str(x)
                Expr.str(y)
            func()

        # --- combined: 3-level nesting + decorator + default args ---

        @TestParser.test
        def three_level_nesting_with_decorator_and_default():
            var_e = Expr(0)
            @var_e.decorate(1)
            def outer(x=Expr.int(2)):
                @var_e.decorate(3)
                def middle(y=Expr.int(4)):
                    def inner():
                        Expr.str(x)
                        Expr.str(y)
                    inner()
                middle()
            outer()


def test_pil_parser_return():

    with TestParser():

        # --- bare return (value is None) ---

        @TestParser.test
        def return_bare():
            def func():
                Expr.str(0)
                return
            func()

        # --- return identifier ---

        @TestParser.test
        def return_name():
            def func():
                var_x = Expr.int(0)
                return var_x
            var_r = func()
            Expr.str(var_r)

        # --- return call expression (PIL inserts a temp) ---

        @TestParser.test
        def return_call_expr():
            def func():
                return Expr.int(0)
            var_r = func()
            Expr.str(var_r)

        # --- return constant ---

        @TestParser.test
        def return_constant():
            def func():
                return 42
            var_r = func()
            Expr.str(var_r)

        # --- return binop expression ---

        @TestParser.test
        def return_binop():
            def func():
                return Expr.int(0) + Expr.int(1)
            var_r = func()
            Expr.str(var_r)

        # --- return tuple literal ---

        @TestParser.test
        def return_tuple():
            def func():
                return (Expr.int(0), Expr.int(1))
            var_r = func()
            Expr.str(var_r[0])
            Expr.str(var_r[1])

        # --- return tuple with starred ---

        @TestParser.test
        def return_tuple_starred():
            def make():
                return [Expr.int(1), Expr.int(2)]
            def func():
                return (Expr.int(0), *make())
            var_r = func()
            Expr.str(var_r[0])
            Expr.str(var_r[1])

        # --- return constant tuple (all elements are constants) ---

        @TestParser.test
        def return_const_tuple():
            def func():
                return (0, 1, 2)
            var_r = func()
            Expr.str(var_r[0])

        # --- early return: only one branch executes ---

        @TestParser.test
        def return_early():
            def func(flag):
                if flag:
                    return Expr.int(0)
                return Expr.int(1)
            var_a = func(Expr.true(0))
            Expr.str(var_a)
            var_b = func(Expr.false(1))
            Expr.str(var_b)

        # --- nested function: each level has its own return ---

        @TestParser.test
        def return_nested():
            def outer():
                def inner():
                    return Expr.int(0)
                var_x = inner()
                Expr.str(var_x)
                return Expr.int(1)
            var_r = outer()
            Expr.str(var_r)

def test_pil_parser_delete():

    with TestParser():

        # --- delete name ---

        @TestParser.test
        def delete_name():
            var_x = Expr.int(0)
            del var_x

        # --- delete attribute ---

        @TestParser.test
        def delete_attribute():
            var_obj = Expr(0)
            var_obj.val = Expr.str(1)
            del var_obj.val

        # --- delete subscript ---

        @TestParser.test
        def delete_subscript():
            var_obj = Expr(0)
            var_obj[Expr.str(1)] = Expr.str(2)
            del var_obj[Expr.str(1)]

        # --- delete tuple (multiple targets in one del) ---

        @TestParser.test
        def delete_tuple():
            var_a = Expr.int(0)
            var_b = Expr.int(1)
            del var_a, var_b

        # --- delete nested tuple/list syntax ---

        @TestParser.test
        def delete_nested_tuple():
            var_a = Expr.int(0)
            var_b = Expr.int(1)
            del (var_a, var_b)

        @TestParser.test
        def delete_nested_list():
            a = Expr.int(0)
            b = Expr.int(1)
            del [a, b]

        # --- delete mixed: name, attribute, subscript in one statement ---

        @TestParser.test
        def delete_mixed():
            var_obj = Expr(0)
            var_obj.val = Expr.str(1)
            var_obj[Expr.str(2)] = Expr.str(3)
            var_x = Expr.int(4)
            del var_x, var_obj.val, var_obj[Expr.str(2)]

def test_pil_parser_assign():

    with TestParser():

        # --- name target ---

        @TestParser.test
        def assign_name():
            var_x = Expr.int(0)

        @TestParser.test
        def assign_name_rhs_call():
            var_x = Expr.int(0) + Expr.int(1)

        @TestParser.test
        def assign_multi_target():
            # a = b = expr: both names get the same value
            var_x = var_y = Expr.int(0)

        # --- attribute target ---

        @TestParser.test
        def assign_attr():
            var_obj = Expr(0)
            var_obj.val = Expr.int(0)

        @TestParser.test
        def assign_attr_rhs_call():
            var_obj = Expr(0)
            var_obj.val = Expr.int(0) + Expr.int(1)

        @TestParser.test
        def assign_attr_chain():
            # obj.val.val = rhs
            var_obj = Expr(0)
            var_obj.val = Expr(1)
            var_obj.val.val = Expr.int(0)

        # --- subscript target ---

        @TestParser.test
        def assign_subscript_const_index():
            var_obj = Expr(0)
            var_obj[0] = Expr.int(1)

        @TestParser.test
        def assign_subscript_expr_index():
            var_obj = Expr(0)
            var_obj[Expr.int(0)] = Expr.int(1)

        @TestParser.test
        def assign_subscript_attr_index():
            # obj[other.val] = rhs — index is an attribute load
            var_obj = Expr(0)
            var_idx = Expr(1)
            var_idx.val = Expr.int(0)
            var_obj[var_idx.val] = Expr.int(1)

        @TestParser.test
        def assign_subscript_subscript_index():
            # obj[idx[k]] = rhs — index is itself a subscript
            var_obj = Expr(0)
            var_idx = Expr(1)
            var_idx[0] = Expr.int(0)
            var_obj[var_idx[0]] = Expr.int(1)

        @TestParser.test
        def assign_subscript_binop_index():
            # obj[a + b] = rhs — index is a binop
            var_obj = Expr(0)
            var_obj[Expr.int(0) + Expr.int(1)] = Expr.int(2)

        @TestParser.test
        def assign_subscript_slice():
            var_obj = Expr(0)
            var_obj[0:2] = Expr.int(1)

        @TestParser.test
        def assign_subscript_slice_with_step():
            var_obj = Expr(0)
            var_obj[0:4:2] = Expr.int(1)

        @TestParser.test
        def assign_subscript_expr_slice():
            # slice bounds are side-effectful expressions
            var_obj = Expr(0)
            var_obj[Expr.int(0):Expr.int(1)] = Expr.int(2)

        @TestParser.test
        def assign_subscript_attr_slice():
            # obj[a.val:b.val] = rhs — slice bounds are attribute loads
            var_obj = Expr(0)
            var_lo = Expr(1)
            var_lo.val = Expr.int(0)
            var_hi = Expr(2)
            var_hi.val = Expr.int(2)
            var_obj[var_lo.val:var_hi.val] = Expr.int(3)

        @TestParser.test
        def assign_subscript_subscript_slice():
            # obj[lo[0]:hi[0]] = rhs — slice bounds are subscripts
            var_obj = Expr(0)
            var_lo = Expr(1)
            var_lo[0] = Expr.int(0)
            var_hi = Expr(2)
            var_hi[0] = Expr.int(2)
            var_obj[var_lo[0]:var_hi[0]] = Expr.int(3)

        @TestParser.test
        def assign_subscript_binop_slice():
            # obj[a+1 : b*2] = rhs
            var_obj = Expr(0)
            var_obj[Expr.int(0) + 1 : Expr.int(1) * 2] = Expr.int(2)

        # --- nested subscript / attr chains ---

        @TestParser.test
        def assign_attr_subscript():
            # obj.val[k] = rhs
            var_obj = Expr(0)
            var_obj.val = Expr(1)
            var_obj.val[0] = Expr.int(1)

        @TestParser.test
        def assign_subscript_attr():
            # obj[k].val = rhs
            var_obj = Expr(0)
            var_obj[0] = Expr(1)
            var_obj[0].val = Expr.int(1)

        @TestParser.test
        def assign_subscript_attr_subscript_attr():
            # obj[k].val[k].val = rhs — four-level chain
            var_obj = Expr(0)
            var_obj[0] = Expr(1)
            var_obj[0].val = Expr(2)
            var_obj[0].val[0] = Expr(3)
            var_obj[0].val[0].val = Expr.int(1)

        # --- tuple / list unpack ---

        @TestParser.test
        def assign_tuple_unpack():
            var_x, var_y = Expr.int(0), Expr.int(1)

        @TestParser.test
        def assign_list_unpack():
            [var_x, var_y] = [Expr.int(0), Expr.int(1)]

        @TestParser.test
        def assign_starred_unpack():
            var_x, *var_y, var_z = [Expr.int(0), Expr.int(1), Expr.int(2), Expr.int(3)]

        @TestParser.test
        def assign_nested_tuple_unpack():
            (var_x, (var_y, var_z)) = (Expr.int(0), (Expr.int(1), Expr.int(2)))

        @TestParser.test
        def assign_unpack_to_attr_subscript():
            # lhs elements can be attribute / subscript targets
            var_obj = Expr(0)
            var_obj.val = Expr.int(0)
            var_arr = Expr(1)
            var_arr[0] = Expr.int(0)
            var_obj.val, var_arr[0] = Expr.int(1), Expr.int(2)

        # --- chained = ---

        @TestParser.test
        def assign_chain_name_name():
            # x = y = expr: both names bound to same value
            var_x = var_y = Expr.int(0)

        @TestParser.test
        def assign_chain_name_attr():
            # x = obj.val = expr
            var_obj = Expr(0)
            var_obj.val = Expr.int(0)
            var_x = var_obj.val = Expr.int(1)

        @TestParser.test
        def assign_chain_name_subscript():
            # x = obj[k] = expr
            var_obj = Expr(0)
            var_obj[0] = Expr.int(0)
            var_x = var_obj[0] = Expr.int(1)

        @TestParser.test
        def assign_chain_attr_subscript():
            # obj.val = arr[k] = expr
            var_obj = Expr(0)
            var_obj.val = Expr.int(0)
            var_arr = Expr(1)
            var_arr[0] = Expr.int(0)
            var_obj.val = var_arr[0] = Expr.int(1)

        @TestParser.test
        def assign_chain_three():
            # x = obj.val = arr[k] = expr
            var_obj = Expr(0)
            var_obj.val = Expr.int(0)
            var_arr = Expr(1)
            var_arr[0] = Expr.int(0)
            var_x = var_obj.val = var_arr[0] = Expr.int(1)

        @TestParser.test
        def assign_chain_tuple_name():
            # (a, b) = x = expr
            var_x = var_a, var_b = Expr.int(0), Expr.int(1)

        @TestParser.test
        def assign_chain_tuple_tuple():
            # (a, b) = (c, d) = expr — two tuple lhs targets
            var_a, var_b = var_c, var_d = Expr.int(0), Expr.int(1)

        @TestParser.test
        def assign_chain_list_list():
            # [a, b] = [c, d] = expr
            [var_a, var_b] = [var_c, var_d] = [Expr.int(0), Expr.int(1)]

        @TestParser.test
        def assign_chain_tuple_nested_2():
            # (a, (b, c)) = x = expr — 2-level nested tuple on first target
            var_x = var_a, (var_b, var_c) = Expr.int(0), (Expr.int(1), Expr.int(2))

        @TestParser.test
        def assign_chain_tuple_nested_3():
            # x = (a, (b, (c, d))) = expr — 3-level nested tuple
            var_x = var_a, (var_b, (var_c, var_d)) = \
                Expr.int(0), (Expr.int(1), (Expr.int(2), Expr.int(3)))

        @TestParser.test
        def assign_chain_list_nested_3():
            # x = [a, [b, [c, d]]] = expr — 3-level nested list
            var_x = [var_a, [var_b, [var_c, var_d]]] = \
                [Expr.int(0), [Expr.int(1), [Expr.int(2), Expr.int(3)]]]

        @TestParser.test
        def assign_chain_mixed_nested_3():
            # x = (a, [b, (c, d)]) = expr — mixed tuple/list 3-level
            var_x = var_a, [var_b, (var_c, var_d)] = \
                Expr.int(0), [Expr.int(1), (Expr.int(2), Expr.int(3))]

        @TestParser.test
        def assign_chain_starred_nested():
            # (a, *b, c) = x = expr
            var_x = [var_a, var_b, var_c, var_d] = [Expr.int(0), Expr.int(1), Expr.int(2), Expr.int(3)]
            var_a, *var_rest, var_z = var_x = [Expr.int(0), Expr.int(1), Expr.int(2), Expr.int(3)]

        @TestParser.test
        def assign_chain_three_nested():
            # (a, b) = [c, d] = x = expr — three targets, two nested
            var_x = [var_c, var_d] = var_a, var_b = [Expr.int(0), Expr.int(1)]

def test_pil_parser_aug_assign():

    with TestParser():

        @TestParser.test
        def aug_assign_name():
            var_x = Expr.int(0)
            var_x += Expr.int(1)

        @TestParser.test
        def aug_assign_name_rhs_call():
            var_x = Expr.int(0)
            var_x += Expr.int(1) * Expr.int(2)

        @TestParser.test
        def aug_assign_attr():
            var_obj = Expr(0)
            var_obj.val = Expr.int(0)
            var_obj.val += Expr.int(1)

        @TestParser.test
        def aug_assign_attr_rhs_call():
            var_obj = Expr(0)
            var_obj.val = Expr.int(0)
            var_obj.val += Expr.int(1) + Expr.int(2)

        @TestParser.test
        def aug_assign_subscript():
            var_obj = Expr(0)
            var_obj[0] = Expr.int(1)
            var_obj[0] += Expr.int(2)

        @TestParser.test
        def aug_assign_subscript_rhs_call():
            var_obj = Expr(0)
            var_obj[0] = Expr.int(1)
            var_obj[0] += Expr.int(2) * Expr.int(3)

        @TestParser.test
        def aug_assign_subscript_expr_index():
            var_obj = Expr(0)
            var_obj[Expr.int(0)] = Expr.int(1)
            var_obj[Expr.int(0)] += Expr.int(2)

        @TestParser.test
        def aug_assign_nested_attr_subscript():
            var_obj = Expr(0)
            var_obj.val = Expr(1)
            var_obj.val[0] = Expr.int(1)
            var_obj.val[0] += Expr.int(2) + Expr.int(3)

        @TestParser.test
        def aug_assign_subscript_attr_chain():
            # obj[k].val += rhs: subscript then attribute
            var_obj = Expr(0)
            var_obj[0] = Expr(1)
            var_obj[0].val = Expr.int(1)
            var_obj[0].val += Expr.int(2)

        @TestParser.test
        def aug_assign_attr_subscript_attr_subscript():
            # obj.val[k].val[k] += rhs: four-level chain
            var_obj = Expr(0)
            var_obj.val = Expr(1)
            var_obj.val[0] = Expr(2)
            var_obj.val[0].val = Expr(3)
            var_obj.val[0].val[0] = Expr.int(1)
            var_obj.val[0].val[0] += Expr.int(2)

        @TestParser.test
        def aug_assign_subscript_slice():
            # obj[a:b] += rhs
            var_obj = Expr(0)
            var_obj[0:1] = Expr.int(1)
            var_obj[0:1] += Expr.int(2)

        @TestParser.test
        def aug_assign_subscript_slice_with_step():
            # obj[a:b:c] += rhs
            var_obj = Expr(0)
            var_obj[0:4:2] = Expr.int(1)
            var_obj[0:4:2] += Expr.int(2)

def test_pil_parser_ann_assign():

    with TestParser():

        # --- annotation only (no value): annotation is evaluated for side effects ---

        @TestParser.test
        def ann_assign_only_call_annotation():
            var_x: Expr.str(0)

        @TestParser.test
        def ann_assign_only_const_annotation():
            # constant annotation: no side effect, trace stays empty
            var_x: int

        # --- annotation + name target ---

        @TestParser.test
        def ann_assign_name_call_annotation():
            var_x: Expr.str(0) = Expr.int(1)

        @TestParser.test
        def ann_assign_name_const_annotation():
            var_x: int = Expr.int(0)

        # --- annotation + attribute target ---

        @TestParser.test
        def ann_assign_attr_target():
            var_obj = Expr(0)
            var_obj.val: Expr.str(0) = Expr.int(1)

        # --- annotation + subscript target ---

        @TestParser.test
        def ann_assign_subscript_target():
            var_obj = Expr(0)
            var_obj[0]: Expr.str(0) = Expr.int(1)

        # --- annotation + subscript target: object comes from a call ---

        @TestParser.test
        def ann_assign_subscript_target_obj_from_call():
            def make():
                return Expr(0)
            make()[0]: Expr.str(0) = Expr.int(1)

        # --- annotation + subscript target: slice index comes from a call ---

        @TestParser.test
        def ann_assign_subscript_target_slice_from_call():
            var_obj = Expr(0)
            var_obj[Expr.int(0)]: Expr.str(1) = Expr.int(2)

        # --- annotation + subscript target: both object and slice from calls ---

        @TestParser.test
        def ann_assign_subscript_target_obj_and_slice_from_calls():
            def make():
                return Expr(0)
            make()[Expr.int(0)]: Expr.str(1) = Expr.int(2)

        # --- annotation + subscript target: slice is a range (lower:upper from calls) ---

        @TestParser.test
        def ann_assign_subscript_target_slice_range_from_calls():
            var_obj = Expr(0)
            var_obj[Expr.int(0):Expr.int(1)]: Expr.str(2) = Expr.int(3)

        # --- annotation + subscript target: object from call, slice is a range ---

        @TestParser.test
        def ann_assign_subscript_target_obj_call_slice_range():
            def make():
                return Expr(0)
            make()[Expr.int(0):Expr.int(1)]: Expr.str(2) = Expr.int(3)

        # --- annotation expression is a complex call (binop result) ---

        @TestParser.test
        def ann_assign_binop_annotation():
            var_x: Expr.int(0) + Expr.int(1) = Expr.int(2)


def test_pil_parser_for():

    with TestParser():

        # --- simple name target, no orelse ---

        @TestParser.test
        def for_simple_name():
            for var_x in [Expr.int(0), Expr.int(1), Expr.int(2)]:
                Expr.str(var_x)

        # --- with orelse (loop exhausts normally → else fires) ---

        @TestParser.test
        def for_with_orelse():
            for var_x in [Expr.int(0), Expr.int(1)]:
                Expr.str(var_x)
            else:
                Expr.str(99)

        # --- tuple unpack target ---

        @TestParser.test
        def for_tuple_unpack_target():
            for var_x, var_y in [(Expr.int(0), Expr.int(1)), (Expr.int(2), Expr.int(3))]:
                Expr.str(var_x)
                Expr.str(var_y)

        @TestParser.test
        def for_tuple_unpack_target_with_orelse():
            for var_x, var_y in [(Expr.int(0), Expr.int(1))]:
                Expr.str(var_x)
                Expr.str(var_y)
            else:
                Expr.str(99)

        # --- attribute target ---

        @TestParser.test
        def for_attribute_target():
            var_obj = Expr(0)
            var_obj.val = Expr.int(-1)
            for var_obj.val in [Expr.int(0), Expr.int(1)]:
                Expr.str(var_obj.val)

        # --- subscript target ---

        @TestParser.test
        def for_subscript_target():
            var_obj = Expr(0)
            var_obj[0] = Expr.int(-1)
            for var_obj[0] in [Expr.int(0), Expr.int(1)]:
                Expr.str(var_obj[0])

        # --- call iter: iter expression is a function call ---

        @TestParser.test
        def for_call_iter():
            for var_x in range(Expr.int(3)):
                Expr.str(var_x)

        @TestParser.test
        def for_call_iter_with_orelse():
            for var_x in range(Expr.int(2)):
                Expr.str(var_x)
            else:
                Expr.str(99)

        # --- nested for ---

        @TestParser.test
        def for_nested():
            for var_i in [Expr.int(0), Expr.int(1)]:
                for var_j in [Expr.int(0), Expr.int(1)]:
                    Expr.str(var_i)
                    Expr.str(var_j)

        @TestParser.test
        def for_nested_outer_orelse():
            for var_i in [Expr.int(0), Expr.int(1)]:
                for var_j in [Expr.int(0)]:
                    Expr.str(var_i)
                    Expr.str(var_j)
            else:
                Expr.str(99)

        @TestParser.test
        def for_nested_both_orelse():
            for var_i in [Expr.int(0), Expr.int(1)]:
                for var_j in [Expr.int(0)]:
                    Expr.str(var_i)
                    Expr.str(var_j)
                else:
                    Expr.str(88)
            else:
                Expr.str(99)

        # --- 3-level nesting: tuple unpack inner, call iter outer ---

        @TestParser.test
        def for_three_level_nesting():
            for var_i in range(Expr.int(2)):
                for var_j in [Expr.int(0), Expr.int(1)]:
                    for var_x, var_y in [(Expr.int(0), Expr.int(1))]:
                        Expr.str(var_i)
                        Expr.str(var_j)
                        Expr.str(var_x)
                        Expr.str(var_y)


def test_pil_parser_while():

    with TestParser():

        # --- function condition, loop never enters ---

        @TestParser.test
        def while_false_no_body():
            while Expr.false(0):
                Expr.str(1)

        @TestParser.test
        def while_false_with_orelse():
            # orelse fires on natural exit (condition was False from the start)
            while Expr.false(0):
                Expr.str(1)
            else:
                Expr.str(2)

        # --- function condition, body + break (condition always True) ---

        @TestParser.test
        def while_true_break():
            while Expr.true(0):
                Expr.str(1)
                break

        @TestParser.test
        def while_true_break_orelse_not_fired():
            # break suppresses orelse
            while Expr.true(0):
                Expr.str(1)
                break
            else:
                Expr.str(99)

        @TestParser.test
        def while_true_body_then_break():
            while Expr.true(0):
                Expr.str(1)
                Expr.str(2)
                break

        # --- constant condition (variable holding True), exits via break ---
        # Note: bare `while True:` is not used because PIL requires the test
        # expression to produce a string identifier (not a raw Constant node).

        @TestParser.test
        def while_const_var_break():
            while 1:
                Expr.str(0)
                break

        @TestParser.test
        def while_const_var_break_orelse_not_fired():
            while 1:
                Expr.str(0)
                break
            else:
                Expr.str(99)

        @TestParser.test
        def while_const_var_multi_body_break():
            while 1:
                Expr.str(0)
                Expr.str(1)
                Expr.str(2)
                break

        # --- nested: function-cond outer, function-cond inner ---

        @TestParser.test
        def while_nested_func_func():
            while Expr.true(0):
                while Expr.true(1):
                    Expr.str(2)
                    break
                Expr.str(3)
                break

        @TestParser.test
        def while_nested_func_func_inner_orelse():
            while Expr.true(0):
                while Expr.false(1):
                    Expr.str(2)
                else:
                    Expr.str(3)
                break

        # --- nested: constant-cond outer, function-cond inner ---

        @TestParser.test
        def while_nested_const_outer_func_inner():
            while 1:
                while Expr.true(0):
                    Expr.str(1)
                    break
                break

        @TestParser.test
        def while_nested_const_outer_false_inner_orelse():
            while True:
                while Expr.false(0):
                    Expr.str(1)
                else:
                    Expr.str(2)
                break

        # --- 3-level nesting ---

        @TestParser.test
        def while_three_level_nesting():
            while 1:
                while Expr.true(0):
                    while Expr.true(1):
                        Expr.str(2)
                        break
                    Expr.str(3)
                    break
                break

        @TestParser.test
        def while_three_level_mixed_orelse():
            var_outer = True
            while var_outer:
                while Expr.true(0):
                    while Expr.false(1):
                        Expr.str(2)
                    else:
                        Expr.str(3)
                    break
                break


def test_pil_parser_if():

    with TestParser():

        @TestParser.test
        def simple_if_true():
            if Expr.true(0):
                Expr.str(1)

        @TestParser.test
        def simple_if_false():
            if Expr.false(0):
                Expr.str(1)

        @TestParser.test
        def if_else_true():
            if Expr.true(0):
                Expr.str(1)
            else:
                Expr.str(2)

        @TestParser.test
        def if_else_false():
            if Expr.false(0):
                Expr.str(1)
            else:
                Expr.str(2)

        @TestParser.test
        def if_elif_else_first():
            if Expr.true(0):
                Expr.str(1)
            elif Expr.true(2):
                Expr.str(3)
            else:
                Expr.str(4)

        @TestParser.test
        def if_elif_else_second():
            if Expr.false(0):
                Expr.str(1)
            elif Expr.true(2):
                Expr.str(3)
            else:
                Expr.str(4)

        @TestParser.test
        def if_elif_else_third():
            if Expr.false(0):
                Expr.str(1)
            elif Expr.false(2):
                Expr.str(3)
            else:
                Expr.str(4)

        @TestParser.test
        def nested_if_true_true():
            if Expr.true(0):
                if Expr.true(1):
                    Expr.str(2)
                else:
                    Expr.str(3)
            else:
                Expr.str(4)

        @TestParser.test
        def nested_if_true_false():
            if Expr.true(0):
                if Expr.false(1):
                    Expr.str(2)
                else:
                    Expr.str(3)
            else:
                Expr.str(4)

        @TestParser.test
        def nested_if_false():
            if Expr.false(0):
                if Expr.true(1):
                    Expr.str(2)
                else:
                    Expr.str(3)
            else:
                Expr.str(4)

        @TestParser.test
        def if_and_condition_both_true():
            if Expr.true(0) and Expr.true(1):
                Expr.str(2)
            else:
                Expr.str(3)

        @TestParser.test
        def if_and_condition_first_false():
            if Expr.false(0) and True:
                Expr.str(2)
            else:
                Expr.str(3)

        @TestParser.test
        def if_and_condition_second_false():
            if Expr.true(0) and False:
                Expr.str(2)
            else:
                Expr.str(3)

        @TestParser.test
        def if_or_condition_both_false():
            if False or Expr.false(1):
                Expr.str(2)
            else:
                Expr.str(3)

        @TestParser.test
        def if_or_condition_first_true():
            if Expr.true(0) or Expr.false(1):
                Expr.str(2)
            else:
                Expr.str(3)

        @TestParser.test
        def if_or_condition_second_true():
            if Expr.false(0) or Expr.true(1):
                Expr.str(2)
            else:
                Expr.str(3)

        @TestParser.test
        def if_and_or_combined():
            if Expr.true(0) and Expr.false(1) or Expr.true(2):
                Expr.str(3)
            else:
                Expr.str(4)

        @TestParser.test
        def sequential_ifs():
            if Expr.true(0):
                Expr.str(1)
            if Expr.false(2):
                Expr.str(3)
            if Expr.true(4):
                Expr.str(5)


def test_pil_parser_with():

    with TestParser():

        # --- single item, no as-binding ---

        @TestParser.test
        def with_single_no_as():
            with Expr.ContextManager(enter_n=0, exit_n=1):
                Expr.str(2)

        # --- single item, with as-binding (name target) ---

        @TestParser.test
        def with_single_as_name():
            with Expr.ContextManager(enter_n=0, exit_n=1) as var_ctx:
                Expr.str(2)
                Expr.str(var_ctx._enter_n)

        # --- context_expr is a complex call expression ---

        @TestParser.test
        def with_ctx_from_call():
            def make_cm():
                return Expr.ContextManager(init_n=Expr.int(0))
            with make_cm():
                Expr.str(1)

        # --- multiple items, no as-bindings ---

        @TestParser.test
        def with_multiple_no_as():
            with Expr.ContextManager(enter_n=0, exit_n=10), Expr.ContextManager(enter_n=1, exit_n=11):
                Expr.str(2)

        # --- multiple items, both with as-bindings ---

        @TestParser.test
        def with_multiple_as_names():
            with Expr.ContextManager(enter_n=0, exit_n=10) as var_a, Expr.ContextManager(enter_n=1, exit_n=11) as var_b:
                Expr.str(2)
                Expr.str(var_a._enter_n)
                Expr.str(var_b._enter_n)

        # --- multiple items, mixed as and no-as ---

        @TestParser.test
        def with_multiple_mixed_as():
            with Expr.ContextManager(enter_n=0, exit_n=10) as var_a, Expr.ContextManager(enter_n=1, exit_n=11):
                Expr.str(2)
                Expr.str(var_a._enter_n)

        # --- three items ---

        @TestParser.test
        def with_three_items():
            with Expr.ContextManager(enter_n=0, exit_n=10), Expr.ContextManager(enter_n=1, exit_n=11), Expr.ContextManager(enter_n=2, exit_n=12):
                Expr.str(3)

        # --- as-binding to attribute target ---

        @TestParser.test
        def with_as_attr_target():
            var_obj = Expr(2)
            with Expr.ContextManager(enter_n=0, exit_n=1) as var_obj.val:
                Expr.str(3)

        # --- as-binding to subscript target ---

        @TestParser.test
        def with_as_subscript_target():
            var_obj = Expr(2)
            var_obj[0] = None
            with Expr.ContextManager(enter_n=0, exit_n=1) as var_obj[0]:
                Expr.str(3)


def test_pil_parser_raise():
    pass


def test_pil_parser_try():

    with TestParser():

        # --- single typed handler, no binding ---

        @TestParser.test
        def try_single_typed_no_binding():
            try:
                raise Expr.TypeA(0)
            except Expr.TypeA:
                Expr.str(1)

        # --- single typed handler, with binding ---

        @TestParser.test
        def try_single_typed_with_binding():
            try:
                raise Expr.TypeA(0)
            except Expr.TypeA as e:
                Expr.str(e._value)

        # --- multiple typed handlers, dispatch to correct branch ---

        @TestParser.test
        def try_multi_handler_typea():
            try:
                raise Expr.TypeA(0)
            except Expr.TypeA:
                Expr.str(10)
            except Expr.TypeB:
                Expr.str(20)
            except Expr.TypeC:
                Expr.str(30)

        @TestParser.test
        def try_multi_handler_typeb():
            try:
                raise Expr.TypeB(0)
            except Expr.TypeA:
                Expr.str(10)
            except Expr.TypeB:
                Expr.str(20)
            except Expr.TypeC:
                Expr.str(30)

        @TestParser.test
        def try_multi_handler_typec():
            try:
                raise Expr.TypeC(0)
            except Expr.TypeA:
                Expr.str(10)
            except Expr.TypeB:
                Expr.str(20)
            except Expr.TypeC:
                Expr.str(30)

        # --- bare except catches everything ---

        @TestParser.test
        def try_bare_except():
            try:
                raise Expr.TypeA(0)
            except Expr.TypeB:
                Expr.str(10)
            except:
                Expr.str(20)

        # --- else branch fires when no exception ---

        @TestParser.test
        def try_else_no_exception():
            try:
                Expr.str(0)
            except Expr.TypeA:
                Expr.str(10)
            else:
                Expr.str(20)

        @TestParser.test
        def try_else_with_exception():
            try:
                if Expr.true(0):
                    raise Expr.TypeA(1)
            except Expr.TypeA:
                Expr.str(10)
            else:
                Expr.str(20)

        # --- finally always fires ---

        @TestParser.test
        def try_finally_no_exception():
            try:
                Expr.str(0)
            finally:
                Expr.str(1)

        @TestParser.test
        def try_finally_with_exception():
            try:
                raise Expr.TypeA(0)
            except Expr.TypeA:
                Expr.str(10)
            finally:
                Expr.str(20)

        # --- combined: multiple handlers + else + finally ---

        @TestParser.test
        def try_combined_no_exception():
            try:
                Expr.str(0)
            except Expr.TypeA as e:
                Expr.int(e._value)
            except Expr.TypeB as e:
                Expr.str(e._value)
            else:
                Expr.str(30)
            finally:
                Expr.str(40)

        @TestParser.test
        def try_combined_typea():
            try:
                if Expr.true(0):
                    raise Expr.TypeA(1)
            except Expr.TypeA as e:
                Expr.int(e._value)
            except Expr.TypeB as e:
                Expr.str(e._value)
            else:
                Expr.str(30)
            finally:
                Expr.str(40)

        @TestParser.test
        def try_combined_typeb():
            try:
                if Expr.true(0):
                    raise Expr.TypeB(1)
            except Expr.TypeA as e:
                Expr.int(e._value)
            except Expr.TypeB as e:
                Expr.str(e._value)
            else:
                Expr.str(30)
            finally:
                Expr.str(40)

        # --- nested try ---

        @TestParser.test
        def try_nested():
            try:
                try:
                    raise Expr.TypeA(0)
                except Expr.TypeB:
                    Expr.str(10)
            except Expr.TypeA as e:
                Expr.int(e._value)


def test_pil_parser_assert():
    pass


def test_pil_parser_expr():

    with TestParser():

        # --- expr statement: binop (result discarded) ---

        @TestParser.test
        def expr_binop():
            Expr.int(0) + Expr.int(1)

        # --- expr statement: named expr (side-effectful, target assigned) ---

        @TestParser.test
        def expr_named_expr():
            (var_a := Expr.int(0))
            Expr.str(var_a)


def test_pil_parser_break():

    with TestParser():

        # --- break in for, first iteration ---

        @TestParser.test
        def break_for_first():
            for var_x in [Expr.int(0), Expr.int(1), Expr.int(2)]:
                Expr.str(var_x)
                break

        # --- break in for, conditional ---

        @TestParser.test
        def break_for_conditional():
            for var_x in [Expr.int(0), Expr.int(1), Expr.int(2)]:
                Expr.str(var_x)
                if Expr.true(var_x):
                    break

        # --- break in for suppresses orelse ---

        @TestParser.test
        def break_for_suppresses_orelse():
            for var_x in [Expr.int(0), Expr.int(1)]:
                Expr.str(var_x)
                break
            else:
                Expr.str(99)

        # --- break in nested for, inner only ---

        @TestParser.test
        def break_for_nested_inner():
            for var_i in [Expr.int(0), Expr.int(1)]:
                for var_j in [Expr.int(0), Expr.int(1), Expr.int(2)]:
                    Expr.str(var_j)
                    break
                Expr.str(var_i)

        # --- break in nested for, outer ---

        @TestParser.test
        def break_for_nested_outer():
            for var_i in [Expr.int(0), Expr.int(1)]:
                for var_j in [Expr.int(0), Expr.int(1)]:
                    Expr.str(var_j)
                Expr.str(var_i)
                break

        # --- break in while, constant condition ---

        @TestParser.test
        def break_while_const():
            while True:
                Expr.str(0)
                break

        # --- break in while, function condition ---

        @TestParser.test
        def break_while_func_cond():
            while Expr.true(0):
                Expr.str(1)
                break

        # --- break in while suppresses orelse ---

        @TestParser.test
        def break_while_suppresses_orelse():
            while Expr.true(0):
                Expr.str(1)
                break
            else:
                Expr.str(99)

        # --- break in while, condition is named expr ---

        @TestParser.test
        def break_while_named_expr_cond():
            var_items = [Expr.int(0), Expr.int(1), Expr.int(2)]
            var_i = [0]
            while var_n := var_i[0] < len(var_items):
                Expr.str(var_items[var_i[0]])
                var_i[0] = var_i[0] + 1
                if var_i[0] == 2:
                    break
            Expr.str(var_n)

        # --- break in while, named expr cond, suppresses orelse ---

        @TestParser.test
        def break_while_named_expr_cond_suppresses_orelse():
            var_items = [Expr.int(0), Expr.int(1)]
            var_i = [0]
            while var_n := var_i[0] < len(var_items):
                Expr.str(var_items[var_i[0]])
                break
            else:
                Expr.str(99)
            Expr.str(var_n)

        # --- break in nested while ---

        @TestParser.test
        def break_while_nested():
            while Expr.true(0):
                while Expr.true(1):
                    Expr.str(2)
                    break
                Expr.str(3)
                break


def test_pil_parser_continue():

    with TestParser():

        # --- continue in for, skip remaining body ---

        @TestParser.test
        def continue_for_basic():
            for var_x in [Expr.int(0), Expr.int(1), Expr.int(2)]:
                continue
                Expr.str(var_x)

        # --- continue in for, conditional ---

        @TestParser.test
        def continue_for_conditional():
            for var_x in [Expr.int(0), Expr.int(1), Expr.int(2)]:
                if Expr.true(var_x):
                    continue
                Expr.str(var_x)

        # --- continue in for does not suppress orelse ---

        @TestParser.test
        def continue_for_orelse_fires():
            for var_x in [Expr.int(0), Expr.int(1)]:
                Expr.str(var_x)
                continue
            else:
                Expr.str(99)

        # --- continue in nested for, inner only ---

        @TestParser.test
        def continue_for_nested_inner():
            for var_i in [Expr.int(0), Expr.int(1)]:
                Expr.str(var_i)
                for var_j in [Expr.int(0), Expr.int(1), Expr.int(2)]:
                    if Expr.true(var_j):
                        continue
                    Expr.str(var_j)

        # --- continue in nested for, outer ---

        @TestParser.test
        def continue_for_nested_outer():
            for var_i in [Expr.int(0), Expr.int(1), Expr.int(2)]:
                if Expr.true(var_i):
                    continue
                for var_j in [Expr.int(0), Expr.int(1)]:
                    Expr.str(var_j)
                Expr.str(var_i)

        # --- continue in while, constant condition ---

        @TestParser.test
        def continue_while_const():
            var_n = [0]
            while var_n[0] < 3:
                var_n[0] = var_n[0] + 1
                if var_n[0] == 2:
                    continue
                Expr.str(var_n[0])

        # --- continue in while, function condition (PIL re-evaluates test) ---

        @TestParser.test
        def continue_while_func_cond():
            var_i = [0]
            while Expr.true(var_i[0]):
                var_i[0] = var_i[0] + 1
                if var_i[0] < 2:
                    continue
                Expr.str(var_i[0])
                break

        # --- continue in while does not suppress orelse ---

        @TestParser.test
        def continue_while_orelse_fires():
            var_n = [0]
            while var_n[0] < 2:
                var_n[0] = var_n[0] + 1
                continue
            else:
                Expr.str(99)

        # --- continue in while, condition is named expr (PIL re-evaluates) ---

        @TestParser.test
        def continue_while_named_expr_cond():
            var_items = [Expr.int(0), Expr.int(1), Expr.int(2)]
            var_i = [0]
            while var_n := var_i[0] < len(var_items):
                var_cur = var_i[0]
                var_i[0] = var_i[0] + 1
                if var_cur == 1:
                    continue
                Expr.str(var_items[var_cur])
            Expr.str(var_n)

        # --- continue in nested while ---

        @TestParser.test
        def continue_while_nested():
            var_i = [0]
            while Expr.true(var_i[0]):
                var_j = [0]
                while Expr.true(var_j[0]):
                    var_j[0] = var_j[0] + 1
                    if var_j[0] < 2:
                        continue
                    Expr.str(var_j[0])
                    break
                var_i[0] = var_i[0] + 1
                if var_i[0] < 2:
                    continue
                Expr.str(var_i[0])
                break


def test_pil_parser_named_expr():

    with TestParser():

        # --- named expr in binop ---

        @TestParser.test
        def named_expr_binop():
            var_x = (var_a := Expr.int(0)) + (var_b := Expr.int(1))
            Expr.str(var_x)
            Expr.str(var_a)
            Expr.str(var_b)

        # --- named expr in unary op ---

        @TestParser.test
        def named_expr_unary():
            var_x = -(var_a := Expr.int(5))
            Expr.str(var_x)
            Expr.str(var_a)

        # --- named expr as if test ---

        @TestParser.test
        def named_expr_if_true():
            if var_a := Expr.true(0):
                Expr.str(1)
            else:
                Expr.str(2)
            Expr.str(var_a)

        @TestParser.test
        def named_expr_if_false():
            if var_a := Expr.false(0):
                Expr.str(1)
            else:
                Expr.str(2)
            Expr.str(var_a)

        # --- named expr as for iter ---

        @TestParser.test
        def named_expr_for_iter():
            for var_x in (var_it := [Expr.int(0), Expr.int(1)]):
                Expr.str(var_x)
            Expr.str(len(var_it))

        # --- named expr as while test ---

        @TestParser.test
        def named_expr_while():
            var_items = [Expr.int(0), Expr.int(1), Expr.int(2)]
            var_i = [0]
            while var_n := (var_i[0] < len(var_items)):
                Expr.str(var_items[var_i[0]])
                var_i[0] = var_i[0] + 1
            Expr.str(var_n)

        # --- named expr as call positional arg ---

        @TestParser.test
        def named_expr_call_pos_arg():
            Expr.str(var_a := Expr.int(0))
            Expr.str(var_a)

        # --- named expr as call keyword arg ---

        @TestParser.test
        def named_expr_call_kw_arg():
            def func(x):
                Expr.str(x)
            func(x=(var_a := Expr.int(0)))
            Expr.str(var_a)

        # --- named expr in tuple literal ---

        @TestParser.test
        def named_expr_in_tuple():
            var_t = ((var_a := Expr.int(0)), (var_b := Expr.int(1)))
            Expr.str(var_t[0])
            Expr.str(var_t[1])
            Expr.str(var_a)
            Expr.str(var_b)

        # --- named expr in list literal ---

        @TestParser.test
        def named_expr_in_list():
            var_l = [(var_a := Expr.int(0)), (var_b := Expr.int(1))]
            Expr.str(var_l[0])
            Expr.str(var_l[1])
            Expr.str(var_a)
            Expr.str(var_b)

        # --- named expr as dict key and value ---

        @TestParser.test
        def named_expr_dict_key():
            var_d = {(var_k := Expr.int(0)): Expr.int(1)}
            Expr.str(var_d[0])
            Expr.str(var_k)

        @TestParser.test
        def named_expr_dict_value():
            var_d = {Expr.int(0): (var_v := Expr.int(99))}
            Expr.str(var_d[0])
            Expr.str(var_v)

        # --- named expr in set literal ---

        @TestParser.test
        def named_expr_in_set():
            var_s = {(var_a := Expr.int(1)), (var_b := Expr.int(2))}
            Expr.str(1 in var_s)
            Expr.str(var_a)
            Expr.str(var_b)

        # --- named expr as subscript index ---

        @TestParser.test
        def named_expr_subscript_index():
            var_arr = Expr(0)
            var_arr[0] = Expr.int(99)
            var_x = var_arr[(var_i := Expr.int(0))]
            Expr.str(var_x)
            Expr.str(var_i)

        # --- named expr as slice bound ---

        @TestParser.test
        def named_expr_slice_bound():
            var_l = [Expr.int(0), Expr.int(1), Expr.int(2)]
            var_s = var_l[(var_lo := Expr.int(1)):(var_hi := Expr.int(3))]
            Expr.str(var_s[0])
            Expr.str(var_lo)
            Expr.str(var_hi)

        # --- named expr as type annotation value ---

        @TestParser.test
        def named_expr_annotation_value():
            var_x: int = (var_a := Expr.int(0))
            Expr.str(var_x)
            Expr.str(var_a)

        # --- named expr as type annotation's annotation expression ---

        @TestParser.test
        def named_expr_as_annotation():
            var_x: (var_ann := Expr.str(0))
            Expr.str(var_ann)

        @TestParser.test
        def named_expr_as_annotation_with_value():
            var_x: (var_ann := Expr.str(0)) = Expr.int(1)
            Expr.str(var_x)
            Expr.str(var_ann)


def test_pil_parser_bin_op():

    with TestParser():

        @TestParser.test
        def add():
            var_x = Expr.int(0) + Expr.int(1) * Expr.int(2)
            var_y = Expr.int(0) - Expr.int(1) // Expr.int(2)


def test_pil_parser_unary_op():

    with TestParser():

        @TestParser.test
        def add():
            var_x = -Expr.int(0) + Expr.int(1) * Expr.int(2)
            var_y = -Expr.int(0) - Expr.int(1) // Expr.int(2)


def test_pil_parser_lambda():

    with TestParser():

        # --- 无参 lambda, body 是调用 ---

        @TestParser.test
        def lambda_no_args():
            f = lambda: Expr.str(0)
            var_r = f()

        # --- 单参数, body 是调用 ---

        @TestParser.test
        def lambda_single_arg():
            f = lambda x: Expr.str(x)
            var_r = f(Expr.int(0))

        # --- 多参数 ---

        @TestParser.test
        def lambda_multiple_args():
            f = lambda x, y: Expr.str(x)
            var_r = f(Expr.int(0), Expr.int(1))

        # --- 带默认值, default 未被覆盖 ---

        @TestParser.test
        def lambda_default_not_overridden():
            f = lambda x=Expr.int(0): Expr.str(x)
            var_r = f()

        # --- 带默认值, default 被覆盖 ---

        @TestParser.test
        def lambda_default_overridden():
            f = lambda x=Expr.int(0): Expr.str(x)
            var_r = f(Expr.int(1))

        # --- *args 可变参数 ---

        @TestParser.test
        def lambda_vararg():
            f = lambda *args: Expr.str(args[0])
            var_r = f(Expr.int(0), Expr.int(1))

        # --- keyword-only 参数 ---

        @TestParser.test
        def lambda_kwonly():
            f = lambda *, key: Expr.str(key)
            var_r = f(key=Expr.int(0))

        @TestParser.test
        def lambda_kwonly_default_not_overridden():
            f = lambda *, key=0: Expr.str(key)
            var_r = f()

        # --- **kwargs ---

        @TestParser.test
        def lambda_kwargs():
            f = lambda **kw: Expr.str(kw['x'])
            var_r = f(x=Expr.int(0))

        # --- body 是常数 ---

        @TestParser.test
        def lambda_body_const():
            f = lambda: 42
            var_x = f()
            Expr.str(var_x)

        # --- body 是 binop ---

        @TestParser.test
        def lambda_body_binop():
            f = lambda x: x + Expr.int(1)
            var_x = f(Expr.int(0))
            Expr.str(var_x)

        # --- body 是 ifexp ---

        @TestParser.test
        def lambda_body_ifexp():
            f = lambda x: Expr.str(0) if Expr.true(x) else Expr.str(1)
            var_r = f(Expr.int(0))

        # --- body 是嵌套调用 ---

        @TestParser.test
        def lambda_body_nested_call():
            f = lambda x: Expr.str(Expr.int(x))
            var_r = f(0)

        # --- 嵌套 lambda: outer 返回 lambda, inner 不加 var_ ---

        @TestParser.test
        def lambda_nested():
            outer = lambda x: lambda y: Expr.str(x)
            inner = outer(Expr.int(0))
            var_r = inner(Expr.int(1))

        # --- lambda 作为高阶函数参数 ---

        @TestParser.test
        def lambda_as_argument():
            def apply(fn, val):
                return fn(val)
            var_r = apply(lambda x: Expr.int(x), Expr.int(0))
            Expr.str(var_r)

        # --- lambda 在列表推导中作为元素 ---

        @TestParser.test
        def lambda_in_listcomp():
            fs = [lambda x=Expr.int(i): Expr.str(x) for i in range(Expr.int(3))]
            for f in fs:
                var_r = f()


def test_pil_parser_dict():

    with TestParser():

        # --- empty dict ---

        @TestParser.test
        def dict_empty():
            var_d = {}

        # --- single pair: constant key, call value ---

        @TestParser.test
        def dict_const_key_call_value():
            var_d = {0: Expr.int(1)}

        # --- single pair: call key, call value ---

        @TestParser.test
        def dict_call_key_call_value():
            var_d = {Expr.int(0): Expr.int(1)}

        # --- multiple pairs: all call keys and values (eval order: k0,v0,k1,v1,...) ---

        @TestParser.test
        def dict_multiple_pairs():
            var_d = {Expr.int(0): Expr.int(1), Expr.int(2): Expr.int(3)}

        @TestParser.test
        def dict_three_pairs():
            var_d = {Expr.int(0): Expr.int(1), Expr.int(2): Expr.int(3), Expr.int(4): Expr.int(5)}

        # --- spread: **other (key is None in the AST) ---

        @TestParser.test
        def dict_spread_only():
            var_other = {0: Expr.int(0)}
            var_d = {**var_other}

        # --- mixed: normal pair then spread ---

        @TestParser.test
        def dict_normal_then_spread():
            var_other = {2: Expr.int(2)}
            var_d = {Expr.int(0): Expr.int(1), **var_other}

        # --- spread in the middle ---

        @TestParser.test
        def dict_spread_in_middle():
            var_other = {1: Expr.int(2)}
            var_d = {Expr.int(0): Expr.int(0), **var_other, Expr.int(3): Expr.int(4)}

        # --- multiple spreads ---

        @TestParser.test
        def dict_multiple_spreads():
            var_a = {0: Expr.int(0)}
            var_b = {1: Expr.int(1)}
            var_d = {**var_a, **var_b}

        # --- spread where the source is a function call result ---

        @TestParser.test
        def dict_spread_from_func_call():
            def make():
                return {Expr.int(0): Expr.int(1)}
            var_d = {**make()}

        @TestParser.test
        def dict_normal_then_spread_from_call():
            def make():
                return {Expr.int(2): Expr.int(3)}
            var_d = {Expr.int(0): Expr.int(1), **make()}

        @TestParser.test
        def dict_spread_from_call_then_normal():
            def make():
                return {Expr.int(0): Expr.int(1)}
            var_d = {**make(), Expr.int(2): Expr.int(3)}

        @TestParser.test
        def dict_multiple_spreads_from_calls():
            def make_a():
                return {Expr.int(0): Expr.int(1)}
            def make_b():
                return {Expr.int(2): Expr.int(3)}
            var_d = {**make_a(), **make_b()}

        # --- nested: value is itself a dict literal ---

        @TestParser.test
        def dict_nested_value():
            var_d = {Expr.int(0): {Expr.int(1): Expr.int(2)}}


def test_pil_parser_set():

    with TestParser():

        # --- single element: constant ---

        @TestParser.test
        def set_single_const():
            var_s = {0}

        # --- single element: call ---

        @TestParser.test
        def set_single_call():
            var_s = {Expr.int(0)}

        # --- multiple elements: all calls (eval order left-to-right) ---

        @TestParser.test
        def set_multiple_calls():
            var_s = {Expr.int(0), Expr.int(1), Expr.int(2)}

        # --- spread: *other where other is a variable ---

        @TestParser.test
        def set_spread_only():
            var_other = [Expr.int(0), Expr.int(1)]
            var_s = {*var_other}

        # --- spread where source is a function call result ---

        @TestParser.test
        def set_spread_from_func_call():
            def make():
                return [Expr.int(0), Expr.int(1)]
            var_s = {*make()}

        # --- normal element then spread ---

        @TestParser.test
        def set_normal_then_spread():
            def make():
                return [Expr.int(2), Expr.int(3)]
            var_s = {Expr.int(0), Expr.int(1), *make()}

        # --- spread then normal element ---

        @TestParser.test
        def set_spread_then_normal():
            def make():
                return [Expr.int(0), Expr.int(1)]
            var_s = {*make(), Expr.int(2), Expr.int(3)}

        # --- spread in the middle ---

        @TestParser.test
        def set_spread_in_middle():
            def make():
                return [Expr.int(1), Expr.int(2)]
            var_s = {Expr.int(0), *make(), Expr.int(3)}

        # --- multiple spreads from calls ---

        @TestParser.test
        def set_multiple_spreads_from_calls():
            def make_a():
                return [Expr.int(0), Expr.int(1)]
            def make_b():
                return [Expr.int(2), Expr.int(3)]
            var_s = {*make_a(), *make_b()}


def test_pil_parser_list_comp():

    with TestParser():

        # --- 1 for, no if ---

        @TestParser.test
        def listcomp_simple():
            var_l = [Expr.str(var_x) for var_x in [Expr.int(0), Expr.int(1), Expr.int(2)]]

        # --- 1 for, 1 if ---

        @TestParser.test
        def listcomp_one_if():
            var_l = [Expr.str(var_x) for var_x in [Expr.int(0), Expr.int(1), Expr.int(2)] if Expr.true(var_x)]

        # --- 1 for, 2 if ---

        @TestParser.test
        def listcomp_two_ifs():
            var_l = [Expr.str(var_x) for var_x in [Expr.int(0), Expr.int(1), Expr.int(2)]
                     if Expr.true(var_x) if var_x != 2]

        # --- 1 for, if with boolop ---

        @TestParser.test
        def listcomp_if_boolop():
            var_l = [Expr.str(var_x) for var_x in [Expr.int(0), Expr.int(1), Expr.int(2)]
                     if Expr.true(0) and Expr.true(var_x)]

        # --- 1 for, if with ifexp ---

        @TestParser.test
        def listcomp_if_ifexp():
            var_l = [Expr.str(var_x) for var_x in [Expr.int(0), Expr.int(1)]
                     if (Expr.true(var_x) if Expr.true(0) else Expr.false(var_x))]

        # --- for target: tuple unpack ---

        @TestParser.test
        def listcomp_target_tuple():
            var_l = [Expr.str(var_a) for var_a, var_b in [(Expr.int(0), Expr.int(1)), (Expr.int(2), Expr.int(3))]]

        # --- for target: list unpack ---

        @TestParser.test
        def listcomp_target_list():
            var_l = [Expr.str(var_a) for [var_a, var_b] in [[Expr.int(0), Expr.int(1)], [Expr.int(2), Expr.int(3)]]]

        # --- for target: attribute ---

        @TestParser.test
        def listcomp_target_attr():
            var_obj = Expr(0)
            var_obj.val = None
            var_l = [Expr.str(var_obj.val) for var_obj.val in [Expr.int(0), Expr.int(1)]]

        # --- for target: subscript ---

        @TestParser.test
        def listcomp_target_subscript():
            var_arr = Expr(0)
            var_arr[0] = None
            var_l = [Expr.str(var_arr[0]) for var_arr[0] in [Expr.int(0), Expr.int(1)]]

        # --- 2 for (nested) ---

        @TestParser.test
        def listcomp_two_fors():
            var_l = [Expr.str(var_x) for var_x in [Expr.int(0), Expr.int(1)]
                     for var_y in [Expr.int(2), Expr.int(3)]]

        # --- 2 for with if ---

        @TestParser.test
        def listcomp_two_fors_with_if():
            var_l = [Expr.str(var_x) for var_x in [Expr.int(0), Expr.int(1)] if Expr.true(var_x)
                     for var_y in [Expr.int(2), Expr.int(3)] if Expr.true(var_y)]

        # --- 3 for (deep nesting) ---

        @TestParser.test
        def listcomp_three_fors():
            var_l = [Expr.str(var_x) for var_x in [Expr.int(0), Expr.int(1)]
                     for var_y in [Expr.int(0), Expr.int(1)]
                     for var_z in [Expr.int(0), Expr.int(1)]]


def test_pil_parser_set_comp():

    with TestParser():

        # --- 1 for, no if ---

        @TestParser.test
        def setcomp_simple():
            var_s = {Expr.str(var_x) for var_x in [Expr.int(0), Expr.int(1), Expr.int(2)]}

        # --- 1 for, 1 if ---

        @TestParser.test
        def setcomp_one_if():
            var_s = {Expr.str(var_x) for var_x in [Expr.int(0), Expr.int(1), Expr.int(2)] if Expr.true(var_x)}

        # --- 1 for, if with boolop ---

        @TestParser.test
        def setcomp_if_boolop():
            var_s = {Expr.str(var_x) for var_x in [Expr.int(0), Expr.int(1)]
                     if Expr.true(0) and Expr.true(var_x)}

        # --- 1 for, if with ifexp ---

        @TestParser.test
        def setcomp_if_ifexp():
            var_s = {Expr.str(var_x) for var_x in [Expr.int(0), Expr.int(1)]
                     if (Expr.true(var_x) if Expr.true(0) else Expr.false(var_x))}

        # --- for target: tuple unpack ---

        @TestParser.test
        def setcomp_target_tuple():
            var_s = {Expr.str(var_a) for var_a, var_b in [(Expr.int(0), Expr.int(1)), (Expr.int(2), Expr.int(3))]}

        # --- for target: list unpack ---

        @TestParser.test
        def setcomp_target_list():
            var_s = {Expr.str(var_a) for [var_a, var_b] in [[Expr.int(0), Expr.int(1)], [Expr.int(2), Expr.int(3)]]}

        # --- for target: attribute ---

        @TestParser.test
        def setcomp_target_attr():
            var_obj = Expr(0)
            var_obj.val = None
            var_s = {Expr.str(var_obj.val) for var_obj.val in [Expr.int(0), Expr.int(1)]}

        # --- for target: subscript ---

        @TestParser.test
        def setcomp_target_subscript():
            var_arr = Expr(0)
            var_arr[0] = None
            var_s = {Expr.str(var_arr[0]) for var_arr[0] in [Expr.int(0), Expr.int(1)]}

        # --- 2 for (nested) ---

        @TestParser.test
        def setcomp_two_fors():
            var_s = {Expr.str(var_x) for var_x in [Expr.int(0), Expr.int(1)]
                     for var_y in [Expr.int(2), Expr.int(3)]}

        # --- 3 for (deep nesting) ---

        @TestParser.test
        def setcomp_three_fors():
            var_s = {Expr.str(var_x) for var_x in [Expr.int(0), Expr.int(1)]
                     for var_y in [Expr.int(0), Expr.int(1)]
                     for var_z in [Expr.int(0), Expr.int(1)]}


def test_pil_parser_dict_comp():

    with TestParser():

        # --- 1 for, no if ---

        @TestParser.test
        def dictcomp_simple():
            var_d = {Expr.int(var_x): Expr.str(var_x) for var_x in [Expr.int(0), Expr.int(1)]}

        # --- 1 for, 1 if ---

        @TestParser.test
        def dictcomp_one_if():
            var_d = {Expr.int(var_x): Expr.str(var_x) for var_x in [Expr.int(0), Expr.int(1)] if Expr.true(var_x)}

        # --- 1 for, if with boolop ---

        @TestParser.test
        def dictcomp_if_boolop():
            var_d = {Expr.int(var_x): Expr.str(var_x) for var_x in [Expr.int(0), Expr.int(1)]
                     if Expr.true(0) and Expr.true(var_x)}

        # --- 1 for, if with ifexp ---

        @TestParser.test
        def dictcomp_if_ifexp():
            var_d = {Expr.int(var_x): Expr.str(var_x) for var_x in [Expr.int(0), Expr.int(1)]
                     if (Expr.true(var_x) if Expr.true(0) else Expr.false(var_x))}

        # --- for target: tuple unpack ---

        @TestParser.test
        def dictcomp_target_tuple():
            var_d = {Expr.int(var_a): Expr.str(var_b)
                     for var_a, var_b in [(Expr.int(0), Expr.int(1)), (Expr.int(2), Expr.int(3))]}

        # --- for target: list unpack ---

        @TestParser.test
        def dictcomp_target_list():
            var_d = {Expr.int(var_a): Expr.str(var_b)
                     for [var_a, var_b] in [[Expr.int(0), Expr.int(1)], [Expr.int(2), Expr.int(3)]]}

        # --- for target: attribute ---

        @TestParser.test
        def dictcomp_target_attr():
            var_obj = Expr(0)
            var_obj.val = None
            var_d = {Expr.int(var_obj.val): Expr.str(var_obj.val) for var_obj.val in [Expr.int(0), Expr.int(1)]}

        # --- for target: subscript ---

        @TestParser.test
        def dictcomp_target_subscript():
            var_arr = Expr(0)
            var_arr[0] = None
            var_d = {Expr.int(var_arr[0]): Expr.str(var_arr[0]) for var_arr[0] in [Expr.int(0), Expr.int(1)]}

        # --- 2 for (nested) ---

        @TestParser.test
        def dictcomp_two_fors():
            var_d = {Expr.int(var_x): Expr.str(var_y)
                     for var_x in [Expr.int(0), Expr.int(1)]
                     for var_y in [Expr.int(2), Expr.int(3)]}

        # --- 3 for (deep nesting) ---

        @TestParser.test
        def dictcomp_three_fors():
            var_d = {Expr.int(var_x): Expr.str(var_z)
                     for var_x in [Expr.int(0), Expr.int(1)]
                     for var_y in [Expr.int(0), Expr.int(1)]
                     for var_z in [Expr.int(0), Expr.int(1)]}


def test_pil_parser_generator_exp():

    with TestParser():

        # --- 1 for, no if: consume via list() ---

        @TestParser.test
        def genexp_simple():
            g = (Expr.str(var_x) for var_x in [Expr.int(0), Expr.int(1), Expr.int(2)])
            var_l = list(g)

        # --- 1 for, 1 if ---

        @TestParser.test
        def genexp_one_if():
            g = (Expr.str(var_x) for var_x in [Expr.int(0), Expr.int(1), Expr.int(2)] if Expr.true(var_x))
            var_l = list(g)

        # --- 1 for, if with boolop ---

        @TestParser.test
        def genexp_if_boolop():
            g = (Expr.str(var_x) for var_x in [Expr.int(0), Expr.int(1)]
                     if Expr.true(0) and Expr.true(var_x))
            var_l = list(g)

        # --- 1 for, if with ifexp ---

        @TestParser.test
        def genexp_if_ifexp():
            g = (Expr.str(var_x) for var_x in [Expr.int(0), Expr.int(1)]
                     if (Expr.true(var_x) if Expr.true(0) else Expr.false(var_x)))
            var_l = list(g)

        # --- for target: tuple unpack ---

        @TestParser.test
        def genexp_target_tuple():
            g = (Expr.str(var_a) for var_a, var_b in [(Expr.int(0), Expr.int(1)), (Expr.int(2), Expr.int(3))])
            var_l = list(g)

        # --- for target: list unpack ---

        @TestParser.test
        def genexp_target_list():
            g = (Expr.str(var_a) for [var_a, var_b] in [[Expr.int(0), Expr.int(1)], [Expr.int(2), Expr.int(3)]])
            var_l = list(g)

        # --- for target: attribute ---

        @TestParser.test
        def genexp_target_attr():
            var_obj = Expr(0)
            var_obj.val = None
            g = (Expr.str(var_obj.val) for var_obj.val in [Expr.int(0), Expr.int(1)])
            var_l = list(g)

        # --- for target: subscript ---

        @TestParser.test
        def genexp_target_subscript():
            var_arr = Expr(0)
            var_arr[0] = None
            g = (Expr.str(var_arr[0]) for var_arr[0] in [Expr.int(0), Expr.int(1)])
            var_l = list(g)

        # --- 2 for (nested) ---

        @TestParser.test
        def genexp_two_fors():
            g = (Expr.str(var_x) for var_x in [Expr.int(0), Expr.int(1)]
                     for var_y in [Expr.int(2), Expr.int(3)])
            var_l = list(g)

        # --- 3 for (deep nesting) ---

        @TestParser.test
        def genexp_three_fors():
            g = (Expr.str(var_x) for var_x in [Expr.int(0), Expr.int(1)]
                     for var_y in [Expr.int(0), Expr.int(1)]
                     for var_z in [Expr.int(0), Expr.int(1)])
            var_l = list(g)


def test_pil_parser_yield():

    with TestParser():

        # --- bare yield (value is None) ---

        @TestParser.test
        def yield_bare():
            def gen():
                yield
            var_l = list(gen())

        # --- yield name ---

        @TestParser.test
        def yield_name():
            def gen():
                var_x = Expr.int(0)
                yield var_x
            var_l = list(gen())
            Expr.str(var_l[0])

        # --- yield call expr (PIL inserts a temp) ---

        @TestParser.test
        def yield_call_expr():
            def gen():
                yield Expr.int(0)
            var_l = list(gen())
            Expr.str(var_l[0])

        # --- yield constant ---

        @TestParser.test
        def yield_constant():
            def gen():
                yield 42
            var_l = list(gen())
            Expr.str(var_l[0])

        # --- yield binop expression ---

        @TestParser.test
        def yield_binop():
            def gen():
                yield Expr.int(0) + Expr.int(1)
            var_l = list(gen())
            Expr.str(var_l[0])

        # --- yield attribute expression ---

        @TestParser.test
        def yield_attr():
            def gen():
                var_obj = Expr(0)
                var_obj.val = Expr.int(99)
                yield var_obj.val
            var_l = list(gen())
            Expr.str(var_l[0])

        # --- yield subscript expression ---

        @TestParser.test
        def yield_subscript():
            def gen():
                var_arr = [Expr.int(0), Expr.int(1)]
                yield var_arr[0]
            var_l = list(gen())
            Expr.str(var_l[0])

        # --- multiple yields in sequence ---

        @TestParser.test
        def yield_multiple():
            def gen():
                yield Expr.int(0)
                yield Expr.int(1)
                yield Expr.int(2)
            var_l = list(gen())
            Expr.str(var_l[0])
            Expr.str(var_l[1])
            Expr.str(var_l[2])

        # --- yield inside if branch ---

        @TestParser.test
        def yield_in_if():
            def gen(flag):
                if flag:
                    yield Expr.int(0)
                else:
                    yield Expr.int(1)
            var_l0 = list(gen(Expr.true(0)))
            Expr.str(var_l0[0])
            var_l1 = list(gen(Expr.false(1)))
            Expr.str(var_l1[0])

        # --- yield inside for loop ---

        @TestParser.test
        def yield_in_for():
            def gen():
                for var_x in [Expr.int(0), Expr.int(1), Expr.int(2)]:
                    yield var_x
            var_l = list(gen())
            Expr.str(var_l[0])
            Expr.str(var_l[1])
            Expr.str(var_l[2])

        # --- yield from name ---

        @TestParser.test
        def yield_from_name():
            def inner():
                yield Expr.int(0)
                yield Expr.int(1)
            def gen():
                var_g = inner()
                yield from var_g
            var_l = list(gen())
            Expr.str(var_l[0])
            Expr.str(var_l[1])

        # --- yield from call expr ---

        @TestParser.test
        def yield_from_call():
            def inner():
                yield Expr.int(0)
                yield Expr.int(1)
            def gen():
                yield from inner()
            var_l = list(gen())
            Expr.str(var_l[0])
            Expr.str(var_l[1])

        # --- send value captured via yield assignment ---

        @TestParser.test
        def yield_send_value():
            def gen():
                var_sent = yield Expr.int(0)
                Expr.str(var_sent)
            g = gen()
            next(g)
            try:
                g.send(Expr.int(1))
            except StopIteration:
                pass


def test_pil_parser_compare():

    with TestParser():

        # ================================================================
        # Part 1: single-op comparisons with each operator
        # ================================================================

        @TestParser.test
        def cmp_lt_true():
            var_x = Expr.int(1) < Expr.int(2)
            Expr.str(var_x)

        @TestParser.test
        def cmp_lt_false():
            var_x = Expr.int(2) < Expr.int(1)
            Expr.str(var_x)

        @TestParser.test
        def cmp_lte():
            var_x = Expr.int(1) <= Expr.int(1)
            Expr.str(var_x)

        @TestParser.test
        def cmp_gt():
            var_x = Expr.int(2) > Expr.int(1)
            Expr.str(var_x)

        @TestParser.test
        def cmp_gte():
            var_x = Expr.int(2) >= Expr.int(2)
            Expr.str(var_x)

        @TestParser.test
        def cmp_eq_true():
            var_x = Expr.int(1) == Expr.int(1)
            Expr.str(var_x)

        @TestParser.test
        def cmp_eq_false():
            var_x = Expr.int(1) == Expr.int(2)
            Expr.str(var_x)

        @TestParser.test
        def cmp_neq():
            var_x = Expr.int(1) != Expr.int(2)
            Expr.str(var_x)

        @TestParser.test
        def cmp_is():
            var_a = Expr.int(0)
            var_x = var_a is var_a
            Expr.str(var_x)

        @TestParser.test
        def cmp_is_not():
            var_a = Expr.int(0)
            var_b = Expr.int(1)
            var_x = var_a is not var_b
            Expr.str(var_x)

        @TestParser.test
        def cmp_in():
            var_l = [Expr.int(0), Expr.int(1)]
            var_x = Expr.int(0) in var_l
            Expr.str(var_x)

        @TestParser.test
        def cmp_not_in():
            var_l = [Expr.int(0), Expr.int(1)]
            var_x = Expr.int(2) not in var_l
            Expr.str(var_x)

        # ================================================================
        # Part 2: chained comparisons (PIL short-circuits via if)
        # ================================================================

        @TestParser.test
        def cmp_chain_lt_lt_all_true():
            # 1 < 2 < 3 — both sub-comparisons true, b evaluated once
            var_x = Expr.int(1) < Expr.int(2) < Expr.int(3)
            Expr.str(var_x)

        @TestParser.test
        def cmp_chain_lt_lt_first_false():
            # 3 < 2 < 4 — first false, third operand not evaluated
            var_x = Expr.int(3) < Expr.int(2) < Expr.int(4)
            Expr.str(var_x)

        @TestParser.test
        def cmp_chain_lt_eq():
            # 1 < 2 == 2
            var_x = Expr.int(1) < Expr.int(2) == Expr.int(2)
            Expr.str(var_x)

        @TestParser.test
        def cmp_chain_three_ops():
            # 1 < 2 <= 3 < 4
            var_x = Expr.int(1) < Expr.int(2) <= Expr.int(3) < Expr.int(4)
            Expr.str(var_x)

        # ================================================================
        # Part 3: compare result used in various expression contexts
        # ================================================================

        # --- compare result in binop ---

        @TestParser.test
        def cmp_in_binop():
            var_x = (Expr.int(1) < Expr.int(2)) + 0
            Expr.str(var_x)

        # --- compare result in unary op ---

        @TestParser.test
        def cmp_in_unary():
            var_x = not (Expr.int(1) == Expr.int(2))
            Expr.str(var_x)

        # --- compare result as if test ---

        @TestParser.test
        def cmp_if_true():
            if Expr.int(1) < Expr.int(2):
                Expr.str(0)
            else:
                Expr.str(1)

        @TestParser.test
        def cmp_if_false():
            if Expr.int(2) < Expr.int(1):
                Expr.str(0)
            else:
                Expr.str(1)

        # --- compare result as for iter ---

        @TestParser.test
        def cmp_for_iter():
            for var_x in [Expr.int(0) < Expr.int(1), Expr.int(2) < Expr.int(1)]:
                Expr.str(var_x)

        # --- compare result as while test ---

        @TestParser.test
        def cmp_while_test():
            var_n = [0]
            while var_n[0] < 3:
                Expr.str(var_n[0])
                var_n[0] = var_n[0] + 1

        # --- compare result as call positional arg ---

        @TestParser.test
        def cmp_call_pos_arg():
            Expr.str(Expr.int(1) < Expr.int(2))

        # --- compare result as call keyword arg ---

        @TestParser.test
        def cmp_call_kw_arg():
            def func(x):
                Expr.str(x)
            func(x=Expr.int(1) == Expr.int(1))

        # --- compare result in tuple literal ---

        @TestParser.test
        def cmp_in_tuple():
            var_t = (Expr.int(1) < Expr.int(2), Expr.int(3) > Expr.int(4))
            Expr.str(var_t[0])
            Expr.str(var_t[1])

        # --- compare result in list literal ---

        @TestParser.test
        def cmp_in_list():
            var_l = [Expr.int(1) < Expr.int(2), Expr.int(3) > Expr.int(4)]
            Expr.str(var_l[0])
            Expr.str(var_l[1])

        # --- compare result as dict key and value ---

        @TestParser.test
        def cmp_dict_value():
            var_d = {0: Expr.int(1) < Expr.int(2)}
            Expr.str(var_d[0])

        @TestParser.test
        def cmp_dict_key():
            var_d = {Expr.int(1) == Expr.int(1): Expr.int(0)}
            Expr.str(var_d[True])

        # --- compare result in set literal ---

        @TestParser.test
        def cmp_in_set():
            var_s = {Expr.int(1) < Expr.int(2), Expr.int(3) > Expr.int(4)}
            Expr.str(True in var_s)

        # --- compare result as subscript index ---

        @TestParser.test
        def cmp_as_subscript_index():
            var_arr = Expr(0)
            var_arr[True] = Expr.int(99)
            var_x = var_arr[Expr.int(1) == Expr.int(1)]
            Expr.str(var_x)

        # --- compare result as slice bound ---

        @TestParser.test
        def cmp_as_slice_bound():
            var_l = [Expr.int(0), Expr.int(1), Expr.int(2)]
            # True == 1, so slice [True:3] == [1:3]
            var_s = var_l[Expr.int(1) == Expr.int(1):]
            Expr.str(var_s[0])

        # --- compare result as type annotation ---

        @TestParser.test
        def cmp_annotation_with_value():
            var_x: bool = Expr.int(1) < Expr.int(2)
            Expr.str(var_x)


def test_pil_parser_call():

    with TestParser():

        # --- 常数: constant literals as positional call arguments ---

        @TestParser.test
        def call_pos_int_const():
            Expr.str(0)

        @TestParser.test
        def call_pos_str_const():
            Expr.str('hello')

        @TestParser.test
        def call_pos_none_const():
            def func(x):
                Expr.str(x)
            func(None)

        @TestParser.test
        def call_pos_bool_const():
            def func(x):
                Expr.str(x)
            func(True)

        @TestParser.test
        def call_pos_multiple_consts():
            def func(x, y):
                Expr.str(x)
                Expr.str(y)
            func(0, 1)

        @TestParser.test
        def call_pos_mixed_const_and_expr():
            def func(x, y):
                Expr.str(x)
                Expr.str(y)
            func(0, Expr.int(1))

        # --- named arg 传常数: named argument with constant value ---

        @TestParser.test
        def call_named_arg_int_const():
            def func(x):
                Expr.str(x)
            func(x=0)

        @TestParser.test
        def call_named_arg_str_const():
            def func(x):
                Expr.str(x)
            func(x='hello')

        @TestParser.test
        def call_named_arg_none_const():
            def func(x):
                Expr.str(x)
            func(x=None)

        @TestParser.test
        def call_named_arg_multiple_consts():
            def func(x, y):
                Expr.str(x)
                Expr.str(y)
            func(x=0, y=1)

        @TestParser.test
        def call_named_arg_mixed_const_and_expr():
            def func(x, y):
                Expr.str(x)
                Expr.str(y)
            func(x=0, y=Expr.int(1))

        # --- 默认参数: keyword arguments overriding defaults ---

        @TestParser.test
        def call_keyword_override_one_default():
            def func(x=0, y=1):
                Expr.str(x)
                Expr.str(y)
            func(x=Expr.int(0))

        @TestParser.test
        def call_keyword_override_all_defaults():
            def func(x=0, y=0):
                Expr.str(x)
                Expr.str(y)
            func(x=Expr.int(0), y=Expr.int(1))

        @TestParser.test
        def call_keyword_arg_expr_value():
            def func(x):
                Expr.str(x)
            func(x=Expr.int(0))

        # --- 可变参数: *args parameter ---

        @TestParser.test
        def call_vararg_empty():
            def func(*args):
                for var_a in args:
                    Expr.str(var_a)
            func()

        @TestParser.test
        def call_vararg_one():
            def func(*args):
                for var_a in args:
                    Expr.str(var_a)
            func(Expr.int(0))

        @TestParser.test
        def call_vararg_many():
            def func(*args):
                for var_a in args:
                    Expr.str(var_a)
            func(Expr.int(0), Expr.int(1), Expr.int(2))

        @TestParser.test
        def call_pos_and_vararg():
            def func(x, *args):
                Expr.str(x)
                for var_a in args:
                    Expr.str(var_a)
            func(Expr.int(0), Expr.int(1), Expr.int(2))

        # --- name only 参数: keyword-only parameters ---

        @TestParser.test
        def call_kwonly_required():
            def func(*, key):
                Expr.str(key)
            func(key=Expr.int(0))

        @TestParser.test
        def call_kwonly_default_not_overridden():
            def func(*, key=0):
                Expr.str(key)
            func()

        @TestParser.test
        def call_kwonly_default_overridden():
            def func(*, key=0):
                Expr.str(key)
            func(key=Expr.int(1))

        @TestParser.test
        def call_pos_and_kwonly():
            def func(x, *, y):
                Expr.str(x)
                Expr.str(y)
            func(Expr.int(0), y=Expr.int(1))

        @TestParser.test
        def call_vararg_and_kwonly():
            def func(*args, key):
                for var_a in args:
                    Expr.str(var_a)
                Expr.str(key)
            func(Expr.int(0), Expr.int(1), key=Expr.int(2))

        # --- keyword参数: **dict expansion ---

        @TestParser.test
        def call_double_star_expand():
            def func(x, y):
                Expr.str(x)
                Expr.str(y)
            var_d = {'x': Expr.int(0), 'y': Expr.int(1)}
            func(**var_d)

        @TestParser.test
        def call_double_star_from_func():
            def make():
                return {'x': Expr.int(0), 'y': Expr.int(1)}
            def func(x, y):
                Expr.str(x)
                Expr.str(y)
            func(**make())

        @TestParser.test
        def call_pos_and_double_star():
            def func(x, y, z):
                Expr.str(x)
                Expr.str(y)
                Expr.str(z)
            var_d = {'y': Expr.int(1), 'z': Expr.int(2)}
            func(Expr.int(0), **var_d)

        @TestParser.test
        def call_keyword_and_double_star():
            def func(x, y, z):
                Expr.str(x)
                Expr.str(y)
                Expr.str(z)
            var_d = {'z': Expr.int(2)}
            func(Expr.int(0), y=Expr.int(1), **var_d)

        # --- 函数调用嵌套场景: nested function calls as arguments ---

        @TestParser.test
        def call_nested_pos_arg():
            Expr.str(Expr.int(0))

        @TestParser.test
        def call_nested_multiple_pos_args():
            def func(x, y):
                Expr.str(x)
                Expr.str(y)
            func(Expr.int(0), Expr.int(1))

        @TestParser.test
        def call_nested_keyword_arg():
            def func(key):
                Expr.str(key)
            func(key=Expr.int(0))

        @TestParser.test
        def call_nested_two_deep():
            def inner():
                return Expr.int(0)
            Expr.str(inner())

        @TestParser.test
        def call_nested_three_deep():
            def inner():
                return Expr.int(0)
            def middle(x):
                return x
            Expr.str(middle(inner()))


def test_pil_parser_joined_str():

    with TestParser():

        # --- 单个插值, 无 conversion, 无 format_spec ---
        # PIL: single-part path, returns the expr directly (no join)
        # Expr.str returns a str, so format(s, '') == s — values match

        @TestParser.test
        def fstr_single_expr():
            var_x = f"{Expr.str(0)}"

        # --- 多个部分: 字面量前缀 + 插值 ---

        @TestParser.test
        def fstr_prefix_and_expr():
            var_x = f"prefix_{Expr.str(0)}"

        # --- 多个部分: 插值 + 字面量后缀 ---

        @TestParser.test
        def fstr_expr_and_suffix():
            var_x = f"{Expr.str(0)}_suffix"

        # --- 多个部分: 前缀 + 插值 + 后缀 ---

        @TestParser.test
        def fstr_prefix_expr_suffix():
            var_x = f"prefix_{Expr.str(0)}_suffix"

        # --- 多个插值, 无字面量 ---

        @TestParser.test
        def fstr_two_exprs():
            var_x = f"{Expr.str(0)}{Expr.str(1)}"

        # --- 多个插值, 中间有字面量 ---

        @TestParser.test
        def fstr_expr_sep_expr():
            var_x = f"{Expr.str(0)}_{Expr.str(1)}"

        # --- conversion !s ---

        @TestParser.test
        def fstr_conversion_s():
            var_x = f"{Expr.int(0)!s}"

        # --- conversion !r ---

        @TestParser.test
        def fstr_conversion_r():
            var_x = f"{Expr.int(0)!r}"

        # --- conversion !a ---

        @TestParser.test
        def fstr_conversion_a():
            var_x = f"{Expr.int(0)!a}"

        # --- format_spec 是字面量 ---

        @TestParser.test
        def fstr_format_spec_const():
            var_x = f"{Expr.str(0):>10}"

        # --- format_spec 是变量表达式 ---

        @TestParser.test
        def fstr_format_spec_expr():
            var_fmt = '>10'
            var_x = f"{Expr.str(0):{var_fmt}}"

        # --- conversion + format_spec ---

        @TestParser.test
        def fstr_conversion_and_format_spec():
            var_x = f"{Expr.int(0)!r:>10}"

        # --- 插值表达式本身是嵌套调用 ---

        @TestParser.test
        def fstr_nested_call_expr():
            var_x = f"{Expr.str(Expr.int(0))}"


def test_pil_parser_constant():

    with TestParser():

        # --- constant in binop ---

        @TestParser.test
        def const_binop_left():
            var_x = 2 + Expr.int(0)
            Expr.str(var_x)

        @TestParser.test
        def const_binop_right():
            var_x = Expr.int(0) + 3
            Expr.str(var_x)

        @TestParser.test
        def const_binop_both():
            var_x = 2 + 3
            Expr.str(var_x)

        # --- constant in unary op ---

        @TestParser.test
        def const_unary_neg():
            var_x = -1
            Expr.str(var_x)

        @TestParser.test
        def const_unary_not():
            var_x = not False
            Expr.str(var_x)

        # --- constant as if test ---

        @TestParser.test
        def const_if_true():
            if 1:
                Expr.str(0)
            else:
                Expr.str(1)

        @TestParser.test
        def const_if_false():
            if 0:
                Expr.str(0)
            else:
                Expr.str(1)

        # --- constant as for iter ---

        @TestParser.test
        def const_for_iter():
            for var_x in (0, 1, 2):
                Expr.str(var_x)

        # --- constant as while test ---

        @TestParser.test
        def const_while_test():
            while 1:
                Expr.str(0)
                break

        # --- constant as call positional arg ---

        @TestParser.test
        def const_call_pos_arg():
            def func(x):
                Expr.str(x)
            func(42)

        # --- constant as call keyword arg ---

        @TestParser.test
        def const_call_kw_arg():
            def func(x):
                Expr.str(x)
            func(x=42)

        # --- constant in tuple literal ---

        @TestParser.test
        def const_in_tuple():
            var_t = (0, Expr.int(1), 2)
            Expr.str(var_t[0])
            Expr.str(var_t[1])
            Expr.str(var_t[2])

        # --- constant in list literal ---

        @TestParser.test
        def const_in_list():
            var_l = [0, Expr.int(1), 2]
            Expr.str(var_l[0])
            Expr.str(var_l[1])
            Expr.str(var_l[2])

        # --- constant as dict key and value ---

        @TestParser.test
        def const_dict_key():
            var_d = {0: Expr.int(1)}
            Expr.str(var_d[0])

        @TestParser.test
        def const_dict_value():
            var_d = {Expr.int(0): 99}
            Expr.str(var_d[0])

        @TestParser.test
        def const_dict_both():
            var_d = {0: 99}
            Expr.str(var_d[0])

        # --- constant in set literal ---

        @TestParser.test
        def const_in_set():
            var_s = {0, Expr.int(1)}
            Expr.str(0 in var_s)

        # --- constant as subscript index ---

        @TestParser.test
        def const_subscript_index():
            var_obj = Expr(0)
            var_obj[0] = Expr.int(1)
            var_x = var_obj[0]
            Expr.str(var_x)

        # --- constant as slice bounds ---

        @TestParser.test
        def const_slice_bounds():
            var_l = [Expr.int(0), Expr.int(1), Expr.int(2), Expr.int(3)]
            var_s = var_l[1:3]
            Expr.str(var_s[0])

        # --- constant as type annotation (no side-effect) ---

        @TestParser.test
        def const_annotation_no_value():
            var_x: int

        @TestParser.test
        def const_annotation_with_value():
            var_x: int = Expr.int(0)
            Expr.str(var_x)


def test_pil_parser_attribute():

    with TestParser():

        # --- attribute in binop ---

        @TestParser.test
        def attr_binop_left():
            var_obj = Expr(0)
            var_obj.val = Expr.int(2)
            var_x = var_obj.val + Expr.int(1)
            Expr.str(var_x)

        @TestParser.test
        def attr_binop_right():
            var_obj = Expr(0)
            var_obj.val = Expr.int(3)
            var_x = Expr.int(1) + var_obj.val
            Expr.str(var_x)

        @TestParser.test
        def attr_binop_both():
            var_a = Expr(0)
            var_a.val = Expr.int(2)
            var_b = Expr(1)
            var_b.val = Expr.int(3)
            var_x = var_a.val + var_b.val
            Expr.str(var_x)

        # --- attribute in unary op ---

        @TestParser.test
        def attr_unary_neg():
            var_obj = Expr(0)
            var_obj.val = Expr.int(5)
            var_x = -var_obj.val
            Expr.str(var_x)

        @TestParser.test
        def attr_unary_not():
            var_obj = Expr(0)
            var_obj.val = Expr.true(0)
            var_x = not var_obj.val
            Expr.str(var_x)

        # --- attribute as if test ---

        @TestParser.test
        def attr_if_true():
            var_obj = Expr(0)
            var_obj.val = Expr.true(0)
            if var_obj.val:
                Expr.str(1)
            else:
                Expr.str(2)

        @TestParser.test
        def attr_if_false():
            var_obj = Expr(0)
            var_obj.val = Expr.false(0)
            if var_obj.val:
                Expr.str(1)
            else:
                Expr.str(2)

        # --- attribute as for iter ---

        @TestParser.test
        def attr_for_iter():
            var_obj = Expr(0)
            var_obj.val = [Expr.int(0), Expr.int(1), Expr.int(2)]
            for var_x in var_obj.val:
                Expr.str(var_x)

        # --- attribute as while test ---

        @TestParser.test
        def attr_while_test():
            var_obj = Expr(0)
            var_obj.val = Expr.true(0)
            while var_obj.val:
                Expr.str(1)
                var_obj.val = False

        # --- attribute as call positional arg ---

        @TestParser.test
        def attr_call_pos_arg():
            var_obj = Expr(0)
            var_obj.val = Expr.int(1)
            Expr.str(var_obj.val)

        # --- attribute as call keyword arg ---

        @TestParser.test
        def attr_call_kw_arg():
            def func(x):
                Expr.str(x)
            var_obj = Expr(0)
            var_obj.val = Expr.int(1)
            func(x=var_obj.val)

        # --- attribute in tuple literal ---

        @TestParser.test
        def attr_in_tuple():
            var_obj = Expr(0)
            var_obj.val = Expr.int(1)
            var_t = (var_obj.val, Expr.int(2))
            Expr.str(var_t[0])
            Expr.str(var_t[1])

        # --- attribute in list literal ---

        @TestParser.test
        def attr_in_list():
            var_obj = Expr(0)
            var_obj.val = Expr.int(1)
            var_l = [var_obj.val, Expr.int(2)]
            Expr.str(var_l[0])
            Expr.str(var_l[1])

        # --- attribute as dict key and value ---

        @TestParser.test
        def attr_dict_key():
            var_obj = Expr(0)
            var_obj.val = Expr.int(0)
            var_d = {var_obj.val: Expr.int(1)}
            Expr.str(var_d[0])

        @TestParser.test
        def attr_dict_value():
            var_obj = Expr(0)
            var_obj.val = Expr.int(99)
            var_d = {Expr.int(0): var_obj.val}
            Expr.str(var_d[0])

        # --- attribute in set literal ---

        @TestParser.test
        def attr_in_set():
            var_obj = Expr(0)
            var_obj.val = Expr.int(1)
            var_s = {var_obj.val, Expr.int(2)}
            Expr.str(1 in var_s)

        # --- attribute as subscript index ---

        @TestParser.test
        def attr_subscript_index():
            var_obj = Expr(0)
            var_obj.val = Expr.int(0)
            var_arr = Expr(1)
            var_arr[0] = Expr.int(99)
            var_x = var_arr[var_obj.val]
            Expr.str(var_x)

        # --- attribute as slice bounds ---

        @TestParser.test
        def attr_slice_lower():
            var_obj = Expr(0)
            var_obj.val = Expr.int(1)
            var_l = [Expr.int(0), Expr.int(1), Expr.int(2)]
            var_s = var_l[var_obj.val:3]
            Expr.str(var_s[0])

        @TestParser.test
        def attr_slice_upper():
            var_obj = Expr(0)
            var_obj.val = Expr.int(2)
            var_l = [Expr.int(0), Expr.int(1), Expr.int(2)]
            var_s = var_l[0:var_obj.val]
            Expr.str(var_s[0])

        @TestParser.test
        def attr_slice_both():
            var_lo = Expr(0)
            var_lo.val = Expr.int(1)
            var_hi = Expr(1)
            var_hi.val = Expr.int(3)
            var_l = [Expr.int(0), Expr.int(1), Expr.int(2), Expr.int(3)]
            var_s = var_l[var_lo.val:var_hi.val]
            Expr.str(var_s[0])

        # --- attribute as type annotation ---

        @TestParser.test
        def attr_annotation_no_value():
            var_x: Expr.ContextManager

        @TestParser.test
        def attr_annotation_with_value():
            var_x: Expr.ContextManager = Expr.int(0)
            Expr.str(var_x)


def test_pil_parser_subscript():

    with TestParser():

        # ================================================================
        # Part 1: subscript result used in various expression contexts
        # ================================================================

        # --- subscript result in binop ---

        @TestParser.test
        def subscript_binop_left():
            var_l = [Expr.int(2), Expr.int(3)]
            var_x = var_l[0] + Expr.int(1)
            Expr.str(var_x)

        @TestParser.test
        def subscript_binop_right():
            var_l = [Expr.int(3), Expr.int(4)]
            var_x = Expr.int(1) + var_l[0]
            Expr.str(var_x)

        @TestParser.test
        def subscript_binop_both():
            var_l = [Expr.int(2), Expr.int(3)]
            var_x = var_l[0] + var_l[1]
            Expr.str(var_x)

        # --- subscript result in unary op ---

        @TestParser.test
        def subscript_unary_neg():
            var_l = [Expr.int(5)]
            var_x = -var_l[0]
            Expr.str(var_x)

        @TestParser.test
        def subscript_unary_not():
            var_l = [Expr.true(0)]
            var_x = not var_l[0]
            Expr.str(var_x)

        # --- subscript result as if test ---

        @TestParser.test
        def subscript_if_true():
            var_l = [Expr.true(0)]
            if var_l[0]:
                Expr.str(1)
            else:
                Expr.str(2)

        @TestParser.test
        def subscript_if_false():
            var_l = [Expr.false(0)]
            if var_l[0]:
                Expr.str(1)
            else:
                Expr.str(2)

        # --- subscript result as for iter ---

        @TestParser.test
        def subscript_for_iter():
            var_l = [[Expr.int(0), Expr.int(1), Expr.int(2)]]
            for var_x in var_l[0]:
                Expr.str(var_x)

        # --- subscript result as while test ---

        @TestParser.test
        def subscript_while_test():
            var_l = [True]
            while var_l[0]:
                Expr.str(0)
                var_l[0] = False

        # --- subscript result as call positional arg ---

        @TestParser.test
        def subscript_call_pos_arg():
            var_l = [Expr.int(0)]
            Expr.str(var_l[0])

        # --- subscript result as call keyword arg ---

        @TestParser.test
        def subscript_call_kw_arg():
            def func(x):
                Expr.str(x)
            var_l = [Expr.int(0)]
            func(x=var_l[0])

        # --- subscript result in tuple literal ---

        @TestParser.test
        def subscript_in_tuple():
            var_l = [Expr.int(1), Expr.int(2)]
            var_t = (var_l[0], var_l[1])
            Expr.str(var_t[0])
            Expr.str(var_t[1])

        # --- subscript result in list literal ---

        @TestParser.test
        def subscript_in_list():
            var_l = [Expr.int(1), Expr.int(2)]
            var_r = [var_l[0], var_l[1]]
            Expr.str(var_r[0])
            Expr.str(var_r[1])

        # --- subscript result as dict key and value ---

        @TestParser.test
        def subscript_dict_key():
            var_keys = [Expr.int(0)]
            var_d = {var_keys[0]: Expr.int(99)}
            Expr.str(var_d[0])

        @TestParser.test
        def subscript_dict_value():
            var_vals = [Expr.int(99)]
            var_d = {Expr.int(0): var_vals[0]}
            Expr.str(var_d[0])

        # --- subscript result in set literal ---

        @TestParser.test
        def subscript_in_set():
            var_l = [Expr.int(1), Expr.int(2)]
            var_s = {var_l[0], var_l[1]}
            Expr.str(1 in var_s)

        # --- subscript result as subscript index ---

        @TestParser.test
        def subscript_as_index():
            var_idx = [Expr.int(0)]
            var_arr = Expr(0)
            var_arr[0] = Expr.int(99)
            var_x = var_arr[var_idx[0]]
            Expr.str(var_x)

        # --- subscript result as type annotation ---

        @TestParser.test
        def subscript_annotation_no_value():
            var_x: list[int]

        @TestParser.test
        def subscript_annotation_with_value():
            var_x: list[int] = Expr.int(0)
            Expr.str(var_x)

        # ================================================================
        # Part 2: slice expressions of various kinds
        # ================================================================

        # --- slice index: attr ---

        @TestParser.test
        def slice_index_attr():
            var_obj = Expr(0)
            var_obj.val = Expr.int(1)
            var_arr = Expr(1)
            var_arr[1] = Expr.int(99)
            var_x = var_arr[var_obj.val]
            Expr.str(var_x)

        # --- slice index: call ---

        @TestParser.test
        def slice_index_call():
            def idx():
                return Expr.int(0)
            var_arr = Expr(0)
            var_arr[0] = Expr.int(99)
            var_x = var_arr[idx()]
            Expr.str(var_x)

        # --- slice index: binop ---

        @TestParser.test
        def slice_index_binop():
            var_arr = Expr(0)
            var_arr[2] = Expr.int(99)
            var_x = var_arr[Expr.int(1) + Expr.int(1)]
            Expr.str(var_x)

        # --- slice index: unary op ---

        @TestParser.test
        def slice_index_unary():
            var_arr = Expr(0)
            var_arr[-1] = Expr.int(99)
            var_x = var_arr[-Expr.int(1)]
            Expr.str(var_x)

        # --- slice index: subscript ---

        @TestParser.test
        def slice_index_subscript():
            var_idxs = [Expr.int(0)]
            var_arr = Expr(0)
            var_arr[0] = Expr.int(99)
            var_x = var_arr[var_idxs[0]]
            Expr.str(var_x)

        # --- slice index: dict value ---

        @TestParser.test
        def slice_index_dict():
            var_d = {0: Expr.int(1)}
            var_arr = Expr(0)
            var_arr[1] = Expr.int(99)
            var_x = var_arr[var_d[0]]
            Expr.str(var_x)

        # --- slice index: set membership (bool) ---

        @TestParser.test
        def slice_index_set():
            var_s = {0}
            var_arr = Expr(0)
            var_arr[True] = Expr.int(99)
            var_x = var_arr[0 in var_s]
            Expr.str(var_x)

        # --- slice index: named expr ---

        @TestParser.test
        def slice_index_named_expr():
            var_arr = Expr(0)
            var_arr[0] = Expr.int(99)
            var_x = var_arr[(var_k := Expr.int(0))]
            Expr.str(var_x)
            Expr.str(var_k)

        # --- slice range: attr bounds ---

        @TestParser.test
        def slice_range_attr_bounds():
            var_lo = Expr(0)
            var_lo.val = Expr.int(1)
            var_hi = Expr(1)
            var_hi.val = Expr.int(3)
            var_l = [Expr.int(0), Expr.int(1), Expr.int(2), Expr.int(3)]
            var_s = var_l[var_lo.val:var_hi.val]
            Expr.str(var_s[0])

        # --- slice range: call bounds ---

        @TestParser.test
        def slice_range_call_bounds():
            def lo():
                return Expr.int(1)
            def hi():
                return Expr.int(3)
            var_l = [Expr.int(0), Expr.int(1), Expr.int(2), Expr.int(3)]
            var_s = var_l[lo():hi()]
            Expr.str(var_s[0])

        # --- slice range: binop bounds ---

        @TestParser.test
        def slice_range_binop_bounds():
            var_l = [Expr.int(0), Expr.int(1), Expr.int(2), Expr.int(3)]
            var_s = var_l[Expr.int(0) + 1 : Expr.int(1) + 2]
            Expr.str(var_s[0])

        # --- slice range: subscript bounds ---

        @TestParser.test
        def slice_range_subscript_bounds():
            var_bounds = [Expr.int(1), Expr.int(3)]
            var_l = [Expr.int(0), Expr.int(1), Expr.int(2), Expr.int(3)]
            var_s = var_l[var_bounds[0]:var_bounds[1]]
            Expr.str(var_s[0])

        # --- slice range: dict bounds ---

        @TestParser.test
        def slice_range_dict_bounds():
            var_d = {'lo': Expr.int(1), 'hi': Expr.int(3)}
            var_l = [Expr.int(0), Expr.int(1), Expr.int(2), Expr.int(3)]
            var_s = var_l[var_d['lo']:var_d['hi']]
            Expr.str(var_s[0])


def test_pil_parser_list():

    with TestParser():

        # --- empty ---

        @TestParser.test
        def list_empty():
            var_l = []

        # --- single element: constant ---

        @TestParser.test
        def list_single_const():
            var_l = [0]

        # --- single element: call ---

        @TestParser.test
        def list_single_call():
            var_l = [Expr.int(0)]

        # --- multiple elements: all calls (eval order left-to-right) ---

        @TestParser.test
        def list_multiple_calls():
            var_l = [Expr.int(0), Expr.int(1), Expr.int(2)]

        # --- starred: *other where other is a variable ---

        @TestParser.test
        def list_starred_only():
            var_other = [Expr.int(0), Expr.int(1)]
            var_l = [*var_other]

        # --- starred: source is a function call result ---

        @TestParser.test
        def list_starred_from_call():
            def make():
                return [Expr.int(0), Expr.int(1)]
            var_l = [*make()]

        # --- normal then starred ---

        @TestParser.test
        def list_normal_then_starred():
            def make():
                return [Expr.int(2), Expr.int(3)]
            var_l = [Expr.int(0), Expr.int(1), *make()]

        # --- starred then normal ---

        @TestParser.test
        def list_starred_then_normal():
            def make():
                return [Expr.int(0), Expr.int(1)]
            var_l = [*make(), Expr.int(2), Expr.int(3)]

        # --- starred in the middle ---

        @TestParser.test
        def list_starred_in_middle():
            def make():
                return [Expr.int(1), Expr.int(2)]
            var_l = [Expr.int(0), *make(), Expr.int(3)]

        # --- multiple starred from calls ---

        @TestParser.test
        def list_multiple_starred_from_calls():
            def make_a():
                return [Expr.int(0), Expr.int(1)]
            def make_b():
                return [Expr.int(2), Expr.int(3)]
            var_l = [*make_a(), *make_b()]

        # --- nested list literal as element ---

        @TestParser.test
        def list_nested():
            var_l = [Expr.int(0), [Expr.int(1), Expr.int(2)]]


def test_pil_parser_tuple():

    with TestParser():

        # --- single element (trailing comma) ---

        @TestParser.test
        def tuple_single_const():
            var_t = (0,)

        # --- single element: call ---

        @TestParser.test
        def tuple_single_call():
            var_t = (Expr.int(0),)

        # --- multiple elements: all calls (eval order left-to-right) ---

        @TestParser.test
        def tuple_multiple_calls():
            var_t = (Expr.int(0), Expr.int(1), Expr.int(2))

        # --- starred: *other where other is a variable ---

        @TestParser.test
        def tuple_starred_only():
            var_other = [Expr.int(0), Expr.int(1)]
            var_t = (*var_other,)

        # --- starred: source is a function call result ---

        @TestParser.test
        def tuple_starred_from_call():
            def make():
                return [Expr.int(0), Expr.int(1)]
            var_t = (*make(),)

        # --- normal then starred ---

        @TestParser.test
        def tuple_normal_then_starred():
            def make():
                return [Expr.int(2), Expr.int(3)]
            var_t = (Expr.int(0), Expr.int(1), *make())

        # --- starred then normal ---

        @TestParser.test
        def tuple_starred_then_normal():
            def make():
                return [Expr.int(0), Expr.int(1)]
            var_t = (*make(), Expr.int(2), Expr.int(3))

        # --- starred in the middle ---

        @TestParser.test
        def tuple_starred_in_middle():
            def make():
                return [Expr.int(1), Expr.int(2)]
            var_t = (Expr.int(0), *make(), Expr.int(3))

        # --- multiple starred from calls ---

        @TestParser.test
        def tuple_multiple_starred_from_calls():
            def make_a():
                return [Expr.int(0), Expr.int(1)]
            def make_b():
                return [Expr.int(2), Expr.int(3)]
            var_t = (*make_a(), *make_b())

        # --- nested tuple literal as element ---

        @TestParser.test
        def tuple_nested():
            var_t = (Expr.int(0), (Expr.int(1), Expr.int(2)))


test_pil_parser_boolop()
test_pil_parser_ifexp()
test_pil_parser_function_def()
test_pil_parser_return()
test_pil_parser_assign()
test_pil_parser_aug_assign()
test_pil_parser_ann_assign()
test_pil_parser_for()
test_pil_parser_while()
test_pil_parser_if()
test_pil_parser_with()
test_pil_parser_raise()
test_pil_parser_try()
test_pil_parser_assert()
test_pil_parser_expr()
test_pil_parser_break()
test_pil_parser_continue()
test_pil_parser_named_expr()
test_pil_parser_bin_op()
test_pil_parser_unary_op()
test_pil_parser_lambda()
test_pil_parser_dict()
test_pil_parser_set()
test_pil_parser_list_comp()
test_pil_parser_set_comp()
test_pil_parser_dict_comp()
test_pil_parser_generator_exp()
test_pil_parser_yield()
test_pil_parser_compare()
test_pil_parser_call()
test_pil_parser_joined_str()
test_pil_parser_constant()
test_pil_parser_attribute()
test_pil_parser_subscript()
test_pil_parser_list()
test_pil_parser_tuple()