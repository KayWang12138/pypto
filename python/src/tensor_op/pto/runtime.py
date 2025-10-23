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
from pto import pto_impl

device_init = pto_impl.DeviceInit
device_fini = pto_impl.DeviceFini
device_run_once_data_from_device = pto_impl.OperatorDeviceRunOnceDataFromDevice


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
                flattened.extend(flatten_to_list(item))
            else:
                flattened.append(item)
        return flattened
    elif isinstance(data, torch.Tensor):
        return data.numpy().flatten().tolist()
    elif isinstance(data, np.ndarray):
        return data.flatten().tolist()
    else:
        raise TypeError("input must be list/numpy.ndarray/torch.Tensor")


def device_run_once_data_from_host(input_list: List, output_list: List):
    input_list = [_flatten_to_list(item) for item in input_list]
    convert_output_list = [_flatten_to_list(item) for item in output_list]

    # onboard run
    pto_impl.DeviceRunOnceDataFromHost(input_list, convert_output_list)

    # fill data to source output_list
    for idx, _ in enumerate(output_list):
        output = output_list[idx]
        src_data = convert_output_list[idx]
        _fill_data_to_target_inplace(src_data, output)


class PythonOperator:
    def __init__(self, origin_func):
        handler = pto_impl.OperatorBegin()
        origin_func()
        pto_impl.OperatorEnd(handler)
        self._handler = handler

    def __call__(self, input_list, output_list):
        stream = torch.npu.current_stream()
        pto_impl.OperatorDeviceRunOnceDataFromDevice(
            self._handler,
            [input.data_ptr() for input in input_list],
            [output.data_ptr() for output in output_list],
            stream.npu_stream)

    @property
    def handler(self):
        return self._handler


def jit(origin_func):
    pto_impl.DeviceInit()
    op = PythonOperator(origin_func)
    return op


def device_synchronize():
    stream = torch.npu.current_stream()
    pto_impl.OperatorDeviceSynchronize(stream.npu_stream)
