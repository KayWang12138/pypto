#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
"""
class CodeHelper():
    def __init__(self):
        self._indent: int = 0
        self.result = ''

    def ir(self): # indent right
        self._indent += 4

    def il(self): # indent left
        self._indent -= 4
        if self._indent<0:
            raise ValueError('Indent cannot be less than 0')

    def __call__(self, v: str=''):
        self.result += ' '*self._indent + v + '\n'

    def __str__(self):
        return self.result

