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
from typing import Optional
from pto import pto_impl


get_input_shape = pto_impl.GetInputShape
get_input_data = pto_impl.GetInputData
get_tensor_data = pto_impl.GetTensorData
set_tensor_data = pto_impl.SetTensorData

_original_init = pto_impl.Tensor.__init__


def new_init(self, shape=None, dtype=None, name=None, format=None, data_ptr=None):
    if shape is None and dtype is None and name is None:
        _original_init(self)
    elif shape and dtype:
        if name is None:
            name = ""
        if format is None and data_ptr is None:
            _original_init(self, dtype, shape, name)
        elif format is None:
            _original_init(self, dtype, shape, data_ptr, name)
        elif data_ptr is None:
            _original_init(self, dtype, shape, name, format)
    else:
        raise RuntimeError(f"Tensor init, input params is invalid")

pto_impl.Tensor.__init__ = new_init
tensor = Tensor = pto_impl.Tensor


def _get_shape(self, axis: Optional[int] = None):
    if axis is not None:
        return self.GetShapeAt(axis)
    else:
        return self.GetShape()
    
tensor.get_shape = _get_shape
tensor.get_dtype = pto_impl.Tensor.GetDataType
tensor.assign = pto_impl.Tensor.Assign
tensor.move = pto_impl.Tensor.Move
tensor.has_storage = pto_impl.Tensor.GetStorage
tensor.id = pto_impl.Tensor.Id
tensor.set_cache_policy = pto_impl.Tensor.SetCachePolicy
tensor.get_cache_policy = pto_impl.Tensor.GetCachePolicy
