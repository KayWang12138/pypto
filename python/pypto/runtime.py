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
from typing import List, overload

import torch
import pypto
from . import pto_impl

_device_init = pto_impl.DeviceInit
_device_fini = pto_impl.DeviceFini
_device_run_once_data_from_device = pto_impl.OperatorDeviceRunOnceDataFromDevice


def _torch_to_pto_dtype(dtype: torch.dtype) -> pypto.DataType:
    "Converts torch.dtype to pypto.DataType"
    if dtype == torch.float16:
        return pypto.DT_FP16
    elif dtype == torch.bfloat16:
        return pypto.DT_BF16
    elif dtype == torch.float32:
        return pypto.DT_FP32
    elif dtype == torch.float64:
        return pypto.DT_DOUBLE
    elif dtype == torch.int8:
        return pypto.DT_INT8
    elif dtype == torch.uint8:
        return pypto.DT_UINT8
    elif dtype == torch.int16:
        return pypto.DT_INT16
    elif dtype == torch.int32:
        return pypto.DT_INT32
    elif dtype == torch.int64:
        return pypto.DT_INT64
    elif dtype == torch.bool:
        return pypto.DT_BOOL

    raise ValueError(f"Input torch.dtype is not supported. Got {dtype}")


def _torch_to_pto(t: torch.Tensor, name: str) -> pypto.Tensor:
    "Converts a `torch.tensor` to `pypto.Tensor`."
    dtype = _torch_to_pto_dtype(t.dtype)
    format = pypto.TileOpFormat.TILEOP_ND
    if t.device.type == "npu":
        import torch_npu
        if torch_npu.get_npu_format(t) == 29:
            format = pypto.TileOpFormat.TILEOP_NZ
    return pypto.Tensor(tuple(t.shape), dtype, f"PTO_TENSOR_{name}", format)


def _to_tensor_data(tensors: List[torch.Tensor]):
    datas = []
    for t in tensors:
        data = pto_impl.DeviceTensorData(
            _torch_to_pto_dtype(t.dtype),
            t.data_ptr(),
            list(t.shape),
        )
        datas.append(data)
    return datas


def _device_run_once_data_from_host(inputs: List[torch.Tensor], outputs: List[torch.Tensor]):
    pto_impl.DeviceRunOnceDataFromHost(_to_tensor_data(inputs), _to_tensor_data(outputs))


class JIT:
    def __init__(self, dyn_func, codegen_options=None,
                 host_options=None, pass_options=None, runtime_options=None):
        self.dyn_func = dyn_func
        self._is_function_compiled: bool = False
        self._handler = None
        self.codegen_options = codegen_options
        self.host_options = host_options
        self.pass_options = pass_options
        self.runtime_options = runtime_options

    def __call__(self, *args, **kwargs):
        in_tensors, out_tensors = args[0], args[1]
        if (len(args) < 2):
            raise ValueError("pypto.jit required at least two input arguments (input_tensors, output_tensors, ...).")

        for in_tensor in in_tensors:
            if not in_tensor.is_contiguous():
                raise RuntimeError("pypto.jit requires that all in_tensors must be contiguous.")

        if not self._is_function_compiled:
            pto_impl.DeviceInit()
            self._set_config_option()
            # Convert I/O torch tensors to PyPTO tensors and run pypto.dyn_function
            in_pto_tensors = [
                _torch_to_pto(t, f"IN_{idx}") for idx, t in enumerate(in_tensors)
            ]
            out_pto_tensors = [
                _torch_to_pto(t, f"OUT_{idx}") for idx, t in enumerate(out_tensors)
            ]
            handler = pto_impl.OperatorBegin(
                pypto.Tensor.to_base_list(in_pto_tensors),
                pypto.Tensor.to_base_list(out_pto_tensors))
            self.dyn_func(in_pto_tensors, out_pto_tensors, *args[2:], **kwargs)
            pto_impl.OperatorEnd(handler)
            self._handler = handler
            self._is_function_compiled = True

        stream = torch.npu.current_stream()
        assert self._handler is not None
        pto_impl.OperatorDeviceRunOnceDataFromDevice(
            self._handler,
            _to_tensor_data(in_tensors),
            _to_tensor_data(out_tensors),
            stream.npu_stream)

    @property
    def handler(self):
        return self._handler

    def _set_config_option(self):
        # 添加支持动态的config
        if isinstance(self.codegen_options, dict):
            pypto.set_codegen_options(** self.codegen_options)

        if isinstance(self.host_options, dict):
            pypto.set_host_options(** self.host_options)

        if isinstance(self.pass_options, dict):
            pypto.set_pass_options(** self.pass_options)

        if isinstance(self.runtime_options, dict):
            pypto.set_runtime_options(** self.runtime_options)


@overload
def jit(dyn_func=None):
    ...


@overload
def jit(
        *,
        codegen_options=None,
        host_options=None,
        pass_options=None,
        runtime_options=None
        ):
    ...


def jit(dyn_func=None,
        *,
        codegen_options=None,
        host_options=None,
        pass_options=None,
        runtime_options=None):

    def decorator(func):
        return JIT(func,
                   codegen_options=codegen_options,
                   host_options=host_options,
                   pass_options=pass_options,
                   runtime_options=runtime_options)

    if dyn_func is not None:
        return JIT(dyn_func)
    else:
        return decorator


def _device_synchronize():
    stream = torch.npu.current_stream()
    pto_impl.OperatorDeviceSynchronize(stream.npu_stream)
