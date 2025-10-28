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
import typing
from typing import Union, List

import pto
from pto import pto_impl

from .enum import * # noqa
from .pto_utils import to_syms
from .symbolic_scalar import SymbolicScalar


class Tensor:

    def __init__(self, shape=None, dtype: Union[DataType, None] = None,
                 name: str = "", format: TileOpFormat = TileOpFormat.TILEOP_ND):
        if shape is None or dtype is None:
            self._base = pto_impl.Tensor()
        elif all([isinstance(s, int) for s in shape]):
            nshape = typing.cast(List[int], shape)
            self._base = pto_impl.Tensor(dtype, nshape, name, format)
        else:
            sym_shape = to_syms(shape)
            assert isinstance(
                sym_shape, list), "shape must be a list of int or SymbolicScalar"
            self._base = pto_impl.Tensor(dtype, sym_shape, name, format)

    @property
    def dtype(self) -> DataType:
        return self._base.GetDataType()

    @property
    def shape(self) -> Union[List[int], List[SymbolicScalar]]:
        out = []
        for i, n in enumerate(self._base.GetShape()):
            if n == -1:
                out.append(SymbolicScalar.from_base(
                    pto_impl.GetInputShape(self._base, i)))
            else:
                out.append(n)
        return out

    def dim(self) -> int:
        return self._base.Dim()

    def set_cache_policy(self, policy: CachePolicy, value: bool) -> None:
        self._base.SetCachePolicy(policy, value)

    def get_cache_policy(self, policy: CachePolicy) -> bool:
        return self._base.GetCachePolicy(policy)

    @property
    def name(self) -> str:
        return self._base.GetName()

    @name.setter
    def name(self, value: str) -> None:
        self._base.SetName(value)

    def move(self, other: 'Tensor') -> None:
        self._base.Move(other._base)

    def _get_assemble_offset(self, key, shape):
        offsets = []
        for axis, k in enumerate(key):
            if isinstance(k.start, (int, SymbolicScalar)):
                offsets.append(k.start)
            elif isinstance(k.stop, (int, SymbolicScalar)):
                offsets.append(k.stop - shape[axis])
            else:
                raise ValueError("Both start and stop are unknown")
        return offsets

    def _is_empty_slice(self, key):
        if isinstance(key, slice):
            return key.start is None and key.stop is None and key.step is None
        elif isinstance(key, (int, SymbolicScalar)):
            return False
        return all([self._is_empty_slice(k) for k in key])

    def __setitem__(self, key, value):
        """
        Set tensor data by index or slice.

        Args:
            key (Union[int, SymbolicScalar, slice]): Index or slice to set.
            value (Tensor | Element): value to set.

            example:
            >>> a = pto.tensor((16, 16), pto.FLOAT32)
            >>> b = pto.tensor((4, 4), pto.FLOAT32)
            >>> a[0, 0] = 1.0 # SetTensorData
            >>> a[0, 1:] = 2.0 # Not supported now
            >>> a[1:, 1:] = b # Assemb(b, (1, 1), a)
            >>> a[:16, :16] = b # Assemb(b, (16 - 4, 16 - 4), a)
        """
        if self._is_empty_slice(key):
            self.move(value)
        elif isinstance(key, (int, SymbolicScalar, slice)):
            self.__setitem__((key,), value)
        elif isinstance(key, tuple):
            assert self.dim() == len(key), f"rank not match, expect {self.dim()}, but got {len(key)}"
            if all([isinstance(k, (int, SymbolicScalar)) for k in key]):
                pto_impl.SetTensorData(value, to_syms(key), self._base)
            elif all([isinstance(k, slice) for k in key]):
                offsets = self._get_assemble_offset(key, value.shape)
                pto.assemble(value, offsets, self)
            else:
                raise ValueError("tuple key must be int or SymbolicScalar")
        else:
            raise RuntimeError("Invalid key type")

    def _get_view_offset_shape(self, key, shape):
        offsets = []
        shapes = []
        for axis, k in enumerate(key):
            start, stop, step = k.start, k.stop, k.step
            if step != 1 and step is not None:
                raise ValueError("step must be 1 or None")
            if start is None:
                start = 0
            if stop is None:
                stop = shape[axis]
            offsets.append(start)
            shapes.append(int(stop - start)) # shape should be concrete
        return offsets, shapes

    def __getitem__(self, key):
        """
        Get tensor data by index or slice.

        Args:
            key (Union[int, SymbolicScalar, slice]): Index or slice to get.

        Returns:
            Tensor | Element: tensor data.

        example:
        >>> s = pto.tensor((16, 16), pto.DT_FP32)
        >>> a = s[0, 0] # GetTensorData
        0.0
        >>> b = s[:4, :4]
        """
        if self._is_empty_slice(key):
            return self
        if isinstance(key, (int, SymbolicScalar, slice)):
            return self.__getitem__((key,))
        elif isinstance(key, tuple):
            assert self.dim() == len(key), f"rank not match, expect {self.dim()}, but got {len(key)}"
            if all([isinstance(k, (int, SymbolicScalar)) for k in key]):
                return SymbolicScalar.from_base(pto_impl.GetTensorData(self._base, to_syms(key)))
            elif all([isinstance(k, slice) for k in key]):
                offsets, shapes = self._get_view_offset_shape(key, self.shape)
                return pto.view(self, shapes, offsets)
            else:
                raise ValueError("tuple key must be int or SymbolicScalar")
        else:
            raise RuntimeError("Invalid key type")

    def base(self) -> pto_impl.Tensor:
        return self._base

    @classmethod
    def from_base(cls, base: pto_impl.Tensor) -> 'Tensor':
        obj = cls.__new__(cls)
        obj._base = base
        return obj

    def __add__(self, other: 'Tensor | int | float') -> 'Tensor':
        return pto.add(self, other)

    def __radd__(self, other: 'Tensor | int | float') -> 'Tensor':
        return self.__add__(other)

    def __sub__(self, other: 'Tensor | int | float') -> 'Tensor':
        return pto.sub(self, other)

    def __matmul__(self, other: 'Tensor') -> 'Tensor':
        if other.dtype in {pto.DT_FP16, pto.DT_BF16, pto.DT_FP32}:
            out_dype = other.dtype
        elif other.dtype == pto.DT_INT8:
            out_dype = pto.DT_INT32
        else:
            raise RuntimeError("unsupport dtype")
        return pto.matmul(self, other, out_dype)

    def matmul(self, mat2, out_dtype, *, a_trans=False, b_trans=False, c_matrix_nz=False) -> 'Tensor':
        return pto.matmul(self, mat2, out_dtype, a_trans=a_trans, b_trans=b_trans, c_matrix_nz=c_matrix_nz)
    
    def assemble(self, input: 'Tensor', offsets: List[Union[int, SymbolicScalar]]) -> None:
        """
        Assemble a small Tensor into a larger Tensor based on specified offsets.

        Args:
            input (Tensor): The small input tensor to be assembled into the larger tensor.
            offsets (Union[List[int], List[SymbolicScalar]]): Offset for placing the input tensor.

        example:
        >>> s = pto.tensor((16, 16), pto.DT_FP32)
        >>> a = pto.tensor((2, 2), pto.DT_FP32)
        >>> s.assemble(a, [0, 0])
        """
        pto.assemble(input, offsets, self)

    def reshape(self, *shape: List[int]) -> 'Tensor':
        return pto.reshape(self, *shape)
    
    def unsqueeze(self, dim: int) -> 'Tensor':
        return pto.unsqueeze(self, dim)
    
    def view(self, shape: List[int], offsets: List[int]) -> 'Tensor':
        return pto.view(self, shape, offsets)
    
    def sin(self) -> 'Tensor':
        return pto.sin(self)
    
    def cos(self) -> 'Tensor':
        return pto.cos(self)

    def sigmoid(self) -> 'Tensor':
        return pto.sigmoid(self)
    
    def softmax(self, dim: int) -> 'Tensor':
        return pto.softmax(self, dim)   

    def maximum(self, other: 'Tensor') -> 'Tensor':
        return pto.maximum(self, other)

        
tensor = Tensor
