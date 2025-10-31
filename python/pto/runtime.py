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
"""
"""
from typing import List

import numpy as np
import torch
import pto
from pto import pto_impl

_device_init = pto_impl.DeviceInit
_device_fini = pto_impl.DeviceFini
_device_run_once_data_from_device = pto_impl.OperatorDeviceRunOnceDataFromDevice


def _fill_data_to_target_inplace(output_list, target):
    if not isinstance(output_list, list):
        raise TypeError("input must list")

    input_len = len(output_list)

    if isinstance(target, np.ndarray):
        target_len = np.prod(target.shape)
        if input_len != target_len:
            raise ValueError(f"input data length ({input_len}) not match ({target_len}) ")
        target.flat[:] = output_list

    elif isinstance(target, torch.Tensor):
        target_len = target.numel()
        if input_len != target_len:
            raise ValueError(f"input data length ({input_len}) not match ({target_len}) ")
        target.view(-1)[:] = torch.tensor(output_list)

    elif isinstance(target, list):
        def get_total_elements(lst):
            if isinstance(lst, list):
                return sum(get_total_elements(item) for item in lst)
            else:
                return 1

        target_len = get_total_elements(target)
        if input_len != target_len:
            raise ValueError(f"input data length ({input_len}) not match ({target_len}) ")

        data_iter = iter(output_list)

        def fill_recursive_inplace(lst):
            for index, elem in enumerate(lst):
                if isinstance(elem, list):
                    fill_recursive_inplace(elem)
                else:
                    lst[index] = next(data_iter)

        fill_recursive_inplace(target)

    else:
        raise TypeError("input must be list/numpy.ndarray/torch.Tensor")


def _flatten_to_list(data):
    """
    flatten data to list
    return: List
    """
    if isinstance(data, (list, tuple)):
        flattened = []
        for item in data:
            if isinstance(item, (list, tuple, np.ndarray, torch.Tensor)):
                flattened.extend(_flatten_to_list(item))
            else:
                flattened.append(item)
        return flattened
    elif isinstance(data, torch.Tensor):
        return data.numpy().flatten().tolist()
    elif isinstance(data, np.ndarray):
        return data.flatten().tolist()
    else:
        raise TypeError("input must be list/numpy.ndarray/torch.Tensor")


def _device_run_once_data_from_host(input_list: List, output_list: List):
    input_list = [_flatten_to_list(item) for item in input_list]
    convert_output_list = [_flatten_to_list(item) for item in output_list]

    # onboard run
    pto_impl.DeviceRunOnceDataFromHost(input_list, convert_output_list)

    # fill data to source output_list
    for idx, _ in enumerate(output_list):
        output = output_list[idx]
        src_data = convert_output_list[idx]
        _fill_data_to_target_inplace(src_data, output)


def _torch_to_pto_dtype(dtype: torch.dtype) -> pto.DataType:
    "Converts torch.dtype to pto.DataType"
    if dtype == torch.float16:
        return pto.DT_FP16
    elif dtype == torch.bfloat16:
        return pto.DT_BF16
    elif dtype == torch.float32:
        return pto.DT_FP32
    elif dtype == torch.float64:
        return pto.DT_DOUBLE
    elif dtype == torch.int8:
        return pto.DT_INT8
    elif dtype == torch.uint8:
        return pto.DT_UINT8
    elif dtype == torch.int16:
        return pto.DT_INT16
    elif dtype == torch.int32:
        return pto.DT_INT32
    elif dtype == torch.int64:
        return pto.DT_INT64
    elif dtype == torch.bool:
        return pto.DT_BOOL

    raise ValueError(f"Input torch.dtype is not supported. Got {dtype}")


def _torch_to_pto(t: torch.tensor, name: str) -> pto.Tensor:
    "Converts a `torch.tensor` to `pto.Tensor`."
    pto_dtype = _torch_to_pto_dtype(t.dtype)
    try:
        import torch_npu
    except ImportError as e:
        raise ImportError("pto.runtime._torch_to_pto requires torch_npu Python packages.") from e
    if torch_npu.get_npu_format(t) == 29: # 29: torch_npu.Format.FRACTAL_NZ
        return pto.Tensor(tuple(t.shape), pto_dtype, f"PTO_TENSOR_{name}", pto.TileOpFormat.TILEOP_NZ)
    return pto.Tensor(tuple(t.shape), pto_dtype, f"PTO_TENSOR_{name}")


class jit:
    def __init__(self, dyn_func):
        self.dyn_func = dyn_func
        self._is_function_compiled: bool = False
        self._handler = None

    def __call__(self, *args, **kwargs):
        in_tensors, out_tensors = args[0], args[1]
        if (len(args) < 2):
            raise ValueError("pto.jit required at least two input arguments (input_tensors, output_tensors, ...).")

        for in_tensor in in_tensors:
            if not in_tensor.is_contiguous():
                raise RuntimeError("pto.jit requires that all in_tensors must be contiguous.")

        if not self._is_function_compiled:
            pto_impl.DeviceInit()

            # Convert I/O torch tensors to PTO tensors and run pto.dyn_function
            in_pto_tensors = [
                _torch_to_pto(t, f"IN_{idx}") for idx, t in enumerate(in_tensors)
            ]
            out_pto_tensors = [
                _torch_to_pto(t, f"OUT_{idx}") for idx, t in enumerate(out_tensors)
            ]
            handler = pto_impl.OperatorBegin()
            self.dyn_func(in_pto_tensors, out_pto_tensors, *args[2:], **kwargs)
            pto_impl.OperatorEnd(handler)
            self._handler = handler
            self._is_function_compiled = True

        stream = torch.npu.current_stream()
        pto_impl.OperatorDeviceRunOnceDataFromDevice(
            self._handler,
            [in_tensor.data_ptr() for in_tensor in in_tensors],
            [out_tensor.data_ptr() for out_tensor in out_tensors],
            stream.npu_stream)

    @property
    def handler(self):
        return self._handler


def _device_synchronize():
    stream = torch.npu.current_stream()
    pto_impl.OperatorDeviceSynchronize(stream.npu_stream)
