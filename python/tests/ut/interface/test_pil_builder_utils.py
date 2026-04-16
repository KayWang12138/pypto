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

import pypto.frontend.parser.pil as pil


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

    @staticmethod
    def run_test(global_dict):
        for test_name in global_dict:
            if test_name.startswith('test_'):
                global_dict[test_name]()
