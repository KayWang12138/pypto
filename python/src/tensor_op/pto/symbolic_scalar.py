#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================
from pto import pto_impl


not_less_than = pto_impl.NotLessThan
not_greater_than = pto_impl.NotGreaterThan

_original_init = pto_impl.SymbolicScalar.__init__


def new_init(self, name=None, value=None, value1=None):
    if name is None and value is None and value1 is None:
        _original_init(self)
    elif name is not None and value is None and value1 is None:
        _original_init(self, name)
    elif name is None and value is not None and value1 is None:
        _original_init(self, value)
    elif name is not None and value is not None and value1 is None:
        _original_init(self, name, value)
    elif name is not None and value is None and value1 is not None:
        _original_init(self, name, value1)
    elif name is not None and value is not None and value1 is not None:
        _original_init(self, name, value, value1)
    else:
        raise RuntimeError(f"SymbolicScalar init, input params is invalid")

pto_impl.SymbolicScalar.__init__ = new_init
symbolic_scalar = SymbolicScalar = pto_impl.SymbolicScalar
symbolic_scalar.is_immediate = pto_impl.SymbolicScalar.IsImmediate
symbolic_scalar.is_symbol = pto_impl.SymbolicScalar.IsSymbol
symbolic_scalar.is_expression = pto_impl.SymbolicScalar.IsExpression
symbolic_scalar.is_valid = pto_impl.SymbolicScalar.IsValid
symbolic_scalar.concrete_valid = pto_impl.SymbolicScalar.ConcreteValid
symbolic_scalar.concrete = pto_impl.SymbolicScalar.Concrete
symbolic_scalar.as_intermediate_variable = pto_impl.SymbolicScalar.AsIntermediateVariable
symbolic_scalar.is_intermediate_variable = pto_impl.SymbolicScalar.IsIntermediateVariable
symbolic_scalar.dump = pto_impl.SymbolicScalar.Dump
symbolic_scalar.min = pto_impl.SymbolicScalar.Min
symbolic_scalar.max = pto_impl.SymbolicScalar.Max
