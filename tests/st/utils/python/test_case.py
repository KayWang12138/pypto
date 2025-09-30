#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================
from abc import ABC, abstractmethod
from typing import NoReturn
from test_case_desc import TestCaseDesc


class TestCase(ABC):
    def __init__(
        self,
        case_index: str,
        case_name: str,
        operation: str,
        input_tensors: list,
        output_tensors: list,
        view_shape: tuple,
        tile_shape: tuple,
        params: dict,
        runner,
    ):
        self._case_desc = TestCaseDesc(
            case_index,
            case_name,
            operation,
            input_tensors,
            output_tensors,
            view_shape,
            tile_shape,
            params,
        )
        self._runner = runner

    @abstractmethod
    def golden_func(self, inputs, params: dict) -> list:
        return None

    @abstractmethod
    def golden_func_params(self) -> dict:
        return {}

    @abstractmethod
    def run_in_dyn_func(self, inputs, params: dict) -> dict:
        return None

    def exec(self, binary_compare: bool = False) -> NoReturn:
        self._runner._params = self.golden_func_params()
        self._runner.reg_op_func(self.run_in_dyn_func)
        self._runner.reg_golden_func(self.golden_func, binary_compare)
        self._runner.run()
