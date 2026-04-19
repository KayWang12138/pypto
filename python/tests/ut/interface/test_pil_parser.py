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
import pil_lvalue
import pil_runtime
import pil_parser


class TestParser:
    target_list = []

    def __enter__(self):
        TestParser.target_list.clear()

    def __exit__(self, exc_type, exc, tb):
        self.run()

    @staticmethod
    def test(target):
        TestParser.target_list.append(target)

    def run(self):
        for target in TestParser.target_list:
            source_lines, _ = inspect.getsourcelines(target)
            source = textwrap.dedent(''.join(source_lines))
            stmt_list = ast.parse(source).body[0].body
            pil_stmt_list = pil.parse_stmts(stmt_list, prefix='__pil__')
            for s in pil_stmt_list: print(ast.dump(s, indent=2))

            parser = pil_parser.PILParser(prefix='__pil__')
            mapping = pil_lvalue.LValueMapping()
            mapping.set_var('pil_runtime', pil_runtime)
            for stmt in parser.visit_stmts(mapping, pil_stmt_list)[1]:
                print(stmt)


        pass

def test_scalar():

    with TestParser():

        @TestParser.test
        def test_normal():
            x = pil_runtime.ScalarValue.create()
            y = pil_runtime.ScalarValue.create()
            z = x + y
            t = x * y
            s = z * t

        @TestParser.test
        def test_if():
            x = pil_runtime.ScalarValue.create()
            y = pil_runtime.ScalarValue.create()
            if x < 0:
                z = x + y
                t = x * y
            else:
                t = x + x + x

        @TestParser.test
        def test_for():
            x = pil_runtime.ScalarValue.create()
            y = pil_runtime.ScalarValue.create()
            t = 20
            for k in pil_runtime.loop(10):
                t = t + x + y

test_scalar()
