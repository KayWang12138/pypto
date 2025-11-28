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
from .converter import _dtype_from, from_torch

__all__ = [
    "_device_init",
    "_device_fini",
    "_device_run_once_data_from_host",
    "_device_synchronize",
    "jit",
    "verify",
]

_device_init = pto_impl.DeviceInit
_device_fini = pto_impl.DeviceFini


def set_device(device: int):
    torch.npu.set_device(device)


def current_device() -> int:
    return torch.npu.current_device()


def current_stream():
    return torch.npu.current_stream().npu_stream


def _torch_to_tensor_data(tensors: List[torch.Tensor]):
    datas = []
    for t in tensors:
        data = pto_impl.DeviceTensorData(
            _dtype_from(t.dtype),
            t.data_ptr(),
            list(t.shape),
        )
        datas.append(data)
    return datas


def _pto_to_tensor_data(tensors: List[pypto.Tensor]) -> List[pto_impl.DeviceTensorData]:
    datas = []
    for t in tensors:
        data = pto_impl.DeviceTensorData(
            t.dtype,
            t.data_ptr,
            list(t.shape),
        )
        datas.append(data)
    return datas


def _device_run_once_data_from_host(inputs: List[torch.Tensor], outputs: List[torch.Tensor]):
    for in_tensor in inputs:
        if not in_tensor.is_contiguous():
            raise RuntimeError("all input tensor must be contiguous.")
    pto_impl.DeviceRunOnceDataFromHost(
        _torch_to_tensor_data(inputs), _torch_to_tensor_data(outputs))


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

        handler = pto_impl.OperatorBegin([t.base() for t in inputs],
                                         [t.base() for t in outputs])
        with pypto.function(self.dyn_func.__name__, inputs, outputs):
            self.dyn_func(inputs, outputs, *args, **kwargs)
        pto_impl.OperatorEnd(handler)

        self._handler = handler
        self._is_compiled = True

    def run(self, in_tensor_data, out_tensor_data, device_id):
        assert self._handler is not None
        workspace_size = pto_impl.GetWorkSpaceSize(self._handler)
        workspace_tensor = torch.zeros(workspace_size, dtype=torch.uint8, device=device_id)
        pto_impl.OperatorDeviceRunOnceDataFromDevice(
            self._handler,
            in_tensor_data,
            out_tensor_data,
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
            if device is None:
                device = t.device
            elif device != t.device:
                raise RuntimeError("not all tensors are on the same device")

        # Convert tensors to tensor data before compile, as compile turns tensor shapes into symbolic scalars.
        in_tensor_data = _pto_to_tensor_data(inputs)
        out_tensor_data = _pto_to_tensor_data(outputs)

        if not self._is_compiled:
            self.compile(inputs, outputs, *args[2:], **kwargs)

        ori_device = current_device()
        if device and device.index != ori_device:
            set_device(device.index)
            self.run(in_tensor_data, out_tensor_data, device.index)
            set_device(ori_device)
        else:
            self.run(in_tensor_data, out_tensor_data, ori_device)

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

    pto_impl.SetVerifyData(_torch_to_tensor_data(inputs),
                           _torch_to_tensor_data(outputs),
                           _torch_to_tensor_data(goldens))

    inputs = [from_torch(t, f"IN_{idx}") for idx, t in enumerate(inputs)]
    outputs = [from_torch(t, f"OUT_{idx}") for idx, t in enumerate(outputs)]
    handler = pto_impl.OperatorBegin([t.base() for t in inputs],
                                     [t.base() for t in outputs])
    func(inputs, outputs, *args, **kwargs)
    pto_impl.OperatorEnd(handler)
