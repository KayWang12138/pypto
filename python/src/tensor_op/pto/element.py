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
from pto import pto_impl


class Element:
    def __init__(self, dtype, data):
        if isinstance(data, int):
            self._base = pto_impl.element(dtype, data)
            self._is_int = True
        elif isinstance(data, float):
            self._base = pto_impl.element(dtype, data)
            self._is_int = False
        else:
            raise ValueError(f"Invalid data type {type(data)} for Element")

    @property
    def dtype(self):
        return self._base._get_data_type()

    @property
    def value(self):
        if self._is_int:
            return self._base._get_signed_data()
        else:
            return self._base._get_float_data()

    def base(self):
        return self._base

element = Element