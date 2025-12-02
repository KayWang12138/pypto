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

import torch

from .enum import DataType, TileOpFormat
from .tensor import Tensor


def from_torch(tensor, name: str):
    """
    convert the input into a PTO Tensor

    Parameters
    ----------
    tensor: object
        The input tensor to be converted. Currently, supports PyTorch tensors.
    name: str
        The name of the resulting PTO Tensor.

    Returns
    -------
    Tensor
        A PTO Tensor object containing the following properties:
        - shape: The dimensions of the tensor.
        - name: The specified name of the tensor.
        - data_ptr: The memory address of the tensor data.
        - format: The format of the tensor (e.g., TILEOP_ND or  TILEOP_NZ).

    Examples
    --------
    >>> x = torch.randn(2, 3)
    >>> pto_tensor = pto.from_torch(x, "input_tensor")
    >>> print(pto_tensor.shape)
    [2, 3]
    """
    try:
        if isinstance(tensor, torch.Tensor):
            if not tensor.is_contiguous():
                raise RuntimeError("not all tensors are contiguous")

            tile_op_format = TileOpFormat.TILEOP_ND
            if tensor.device.type == "npu":
                import torch_npu
                if torch_npu.get_npu_format(tensor) == 29:
                    tile_op_format = TileOpFormat.TILEOP_NZ

            dtype = _dtype_from(tensor.dtype)
            if tensor.dim() == 0:
                return Tensor(
                shape=tuple([1]),
                dtype=dtype,
                name=name,
                data_ptr=tensor.data_ptr(),
                format=tile_op_format,
                device=tensor.device)
            return Tensor(
                shape=list(tensor.shape),
                dtype=dtype,
                name=name,
                data_ptr=tensor.data_ptr(),
                format=tile_op_format,
                device=tensor.device)
        else:
            raise TypeError("input type is not currently supported.")
    except Exception as e:
        raise Exception(f"failed to convert the input into a pto.Tensor") from e


_dtype_dict = {
    torch.float16: DataType.DT_FP16,
    torch.bfloat16: DataType.DT_BF16,
    torch.float32: DataType.DT_FP32,
    torch.float64: DataType.DT_DOUBLE,
    torch.int8: DataType.DT_INT8,
    torch.uint8: DataType.DT_UINT8,
    torch.int16: DataType.DT_INT16,
    torch.int32: DataType.DT_INT32,
    torch.int64: DataType.DT_INT64,
    torch.bool: DataType.DT_BOOL,
}


def _dtype_from(dtype: torch.dtype) -> DataType:
    pto_dtype = _dtype_dict.get(dtype)
    if pto_dtype is None:
        raise ValueError(f"Input torch.dtype is not supported. Got {dtype}")
    return pto_dtype
