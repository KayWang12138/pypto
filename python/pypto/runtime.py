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

__all__ = [
    "_device_init",
    "_device_fini",
    "_device_run_once_data_from_host",
    "_device_synchronize",
    "jit",
    "verify",
    "to_pto",
]

_device_init = pto_impl.DeviceInit
_device_fini = pto_impl.DeviceFini


_dtype_dict = {
    torch.float16: pypto.DT_FP16,
    torch.bfloat16: pypto.DT_BF16,
    torch.float32: pypto.DT_FP32,
    torch.float64: pypto.DT_DOUBLE,
    torch.int8: pypto.DT_INT8,
    torch.uint8: pypto.DT_UINT8,
    torch.int16: pypto.DT_INT16,
    torch.int32: pypto.DT_INT32,
    torch.int64: pypto.DT_INT64,
    torch.bool: pypto.DT_BOOL,
}


def dtype_from(dtype: torch.dtype) -> pypto.DataType:
    return _dtype_dict[dtype]


def set_device(device: int):
    torch.npu.set_device(device)


def current_device() -> int:
    return torch.npu.current_device()


def to_pto(t: torch.Tensor, name: str) -> pypto.Tensor:
    dtype = dtype_from(t.dtype)
    format = pypto.TileOpFormat.TILEOP_ND
    if t.device.type == "npu":
        import torch_npu
        if torch_npu.get_npu_format(t) == 29:
            format = pypto.TileOpFormat.TILEOP_NZ
    if t.dim() == 0:
        return pypto.Tensor(tuple([1]), dtype, name, format)
    return pypto.Tensor(tuple(t.shape), dtype, name, format)


def current_stream():
    return torch.npu.current_stream().npu_stream


def to_tensor_data(tensors: List[torch.Tensor]):
    datas = []
    for t in tensors:
        data = pto_impl.DeviceTensorData(
            dtype_from(t.dtype),
            t.data_ptr(),
            list(t.shape),
        )
        datas.append(data)
    return datas


def _device_run_once_data_from_host(inputs: List[torch.Tensor], outputs: List[torch.Tensor]):
    for in_tensor in inputs:
        if not in_tensor.is_contiguous():
            raise RuntimeError("all input tensor must be contiguous.")
    pto_impl.DeviceRunOnceDataFromHost(
        to_tensor_data(inputs), to_tensor_data(outputs))


class JIT:
    def __init__(self, dyn_func, codegen_options=None,
                 host_options=None, pass_options=None, runtime_options=None):
        self.dyn_func = dyn_func
        self._is_compiled: bool = False
        self._handler = None
        self.codegen_options = codegen_options
        self.host_options = host_options
        self.pass_options = pass_options
        self.runtime_options = runtime_options

    def compile(self, inputs, outputs, *args, **kwargs):
        pto_impl.DeviceInit()
        self._set_config_option()

        inputs = [to_pto(t, f"IN_{idx}")
                  for idx, t in enumerate(inputs)]
        outputs = [to_pto(t, f"OUT_{idx}")
                   for idx, t in enumerate(outputs)]
        handler = pto_impl.OperatorBegin([t.base() for t in inputs],
                                         [t.base() for t in outputs])
        self.dyn_func(inputs, outputs, *args, **kwargs)
        pto_impl.OperatorEnd(handler)

        self._handler = handler
        self._is_compiled = True

    def run(self, inputs, outputs, device_id):
        assert self._handler is not None
        workspace_size = pto_impl.GetWorkSpaceSize(self._handler)
        workspace_tensor = torch.zeros(workspace_size, dtype=torch.uint8, device=device_id)
        pto_impl.OperatorDeviceRunOnceDataFromDevice(
            self._handler,
            to_tensor_data(inputs),
            to_tensor_data(outputs),
            current_stream(),
            workspace_tensor.data_ptr())

    def __call__(self, *args, **kwargs):
        if (len(args) < 2):
            raise ValueError("inputs or outputs missing")
        device = None
        inputs, outputs = args[0], args[1]
        if (len(inputs + outputs) < 1):
            raise ValueError("inputs or outputs missing")
        for t in inputs + outputs:
            if not t.is_contiguous():
                raise RuntimeError("not all tensors are contiguous")
            if device is None:
                device = t.device
            elif device != t.device:
                raise RuntimeError("not all tensors are on the same device")

        if not self._is_compiled:
            self.compile(inputs, outputs, *args[2:], **kwargs)

        ori_device = current_device()
        if device and device.index != ori_device:
            set_device(device.index)
            self.run(inputs, outputs, device.index)
            set_device(ori_device)
        else:
            self.run(inputs, outputs, ori_device)

    @property
    def handler(self):
        return self._handler

    def _set_config_option(self):
        if isinstance(self.codegen_options, dict):
            pypto.set_codegen_options(**self.codegen_options)

        if isinstance(self.host_options, dict):
            pypto.set_host_options(**self.host_options)

        if isinstance(self.pass_options, dict):
            pypto.set_pass_options(**self.pass_options)

        if isinstance(self.runtime_options, dict):
            pypto.set_runtime_options(**self.runtime_options)


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
    pto_impl.OperatorDeviceSynchronize(current_stream())


def verify(func, inputs, outputs, goldens, *args,
           codegen_options=None,
           host_options=None,
           pass_options=None,
           verify_options=None, **kwargs):
    """
    Verify the tensor graph of the function.

    Args:
        func: The function to verify.
        inputs: The input tensors.
        outputs: The output tensors.
        goldens: The golden tensors.
        *args: The extra arguments for func.
        verify_options: dict
            see :func:`set_verify_options`.
        codegen_options: dict
            see :func:`set_codegen_options`.
        host_options: dict
            see :func:`set_host_options`.
        pass_options: dict
            see :func:`set_pass_options`.
        **kwargs: The extra keyword arguments for func.
    Returns:
        None
    """
    pto_impl.DeviceInit()

    if codegen_options is None:
        codegen_options = {"support_dynamic_unaligned": True}
    pypto.set_codegen_options(**codegen_options)

    if host_options is None:
        host_options = {"only_codegen": True}
    pypto.set_host_options(**host_options)

    if pass_options is None:
        pass_options = {}
    pypto.set_pass_options(**pass_options)

    if verify_options is None:
        verify_options = {"verify_tensor_graph": True}
    pypto.set_verify_options(**verify_options)

    pto_impl.SetVerifyData(to_tensor_data(inputs),
                           to_tensor_data(outputs),
                           to_tensor_data(goldens))

    inputs = [to_pto(t, f"IN_{idx}") for idx, t in enumerate(inputs)]
    outputs = [to_pto(t, f"OUT_{idx}") for idx, t in enumerate(outputs)]
    handler = pto_impl.OperatorBegin([t.base() for t in inputs],
                                     [t.base() for t in outputs])
    func(inputs, outputs, *args, **kwargs)
    pto_impl.OperatorEnd(handler)
