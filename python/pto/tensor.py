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
from typing import Union, List, Optional, Tuple

import pto
from . import pto_impl

from .enum import * # noqa
from .pto_utils import to_syms, to_sym
from .symbolic_scalar import SymbolicScalar, SymInt


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
    def shape(self) -> List[SymInt]:
        out = []
        for i, n in enumerate(self._base.GetShape()):
            if n == -1:
                out.append(SymbolicScalar.from_base(
                    pto_impl.GetInputShape(self._base, i)))
            else:
                out.append(n)
        return out

    @property
    def dim(self) -> int:
        return self._base.Dim()

    @property
    def format(self) -> TileOpFormat:
        return self._base.Format()

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
                offsets.append(0)
        return offsets

    def _is_empty_slice(self, key):
        if isinstance(key, slice):
            return key.start is None and key.stop is None and key.step is None
        elif isinstance(key, (int, SymbolicScalar)):
            return False
        elif key is Ellipsis:
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
        elif isinstance(key, (int, SymbolicScalar)):
            self.__setitem__((key,), value)
        elif isinstance(key, slice):
            if isinstance(key.stop, Tensor):
                assert isinstance(key.start, int)
                return pto.scatter(self, key.start, key.stop, value)
            else:
                self.__setitem__((key,), value)
        elif isinstance(key, tuple):
            assert self.dim == len(
                key), f"rank not match, expect {self.dim}, but got {len(key)}"
            if all([isinstance(k, (int, SymbolicScalar)) for k in key]):
                pto_impl.SetTensorData(to_sym(value), to_syms(key), self._base)
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
            elif isinstance(start, int) and start < 0:
                start = shape[axis] + start
            if stop is None:
                stop = shape[axis]
            elif isinstance(stop, int) and stop <= 0:
                stop = shape[axis] + stop
            offsets.append(start)
            shapes.append(int(stop - start)) # shape should be concrete
        return offsets, shapes

    def __getitem__(self, key, *, valid_shape: Optional[List[Union[int, SymbolicScalar]]] = None):
        """
        Get tensor data by index, supporting integer indices, slices, ellipsis,
        and their combinations to retrieve sub-tensors.

        Args:
            key (Union[int, SymbolicScalar, slice, ellipsis]): Index or slice to get.

        Returns:
            Tensor | Element: tensor data.

        example:
        s = pto.tensor([4, 4], pto.DT_FP32)
        a = s[0, 0] # GetTensorData
        b = s[:2, :2] # All slice
        c = s[1, 1:3] # Index and slice
        d = s[-1, -3:-1] # Negative index
        e = s[..., 1:3] # Ellipsis index

        Input s:[[1, 2, 3, 4],
                 [5, 6, 7, 8],
                 [9, 10, 11, 12],
                 [13, 14, 15, 16]]
        Output a:1
               b:[[1, 2],
                  [5, 6]]
               c:[6, 7]
               d:[14, 15]
               e:[[2, 3],
                  [6, 7],
                  [10, 11],
                  [14, 15]]
        """
        if self._is_empty_slice(key):
            return self
        if isinstance(key, (int, SymbolicScalar)):
            return self.__getitem__((key,))
        elif isinstance(key, slice):
            # Support for slicing operations and gather_element syntactic sugar
            if isinstance(key.stop, Tensor):
                assert isinstance(key.start, int)
                return pto.gather(self, key.start, key.stop)
            else:
                return self.__getitem__((key,))
        elif key is Ellipsis:
            return self

        elif isinstance(key, tuple):
            assert self.dim >= len(
                key), f"rank not match, expect {self.dim}, but got {len(key)}"
            if all([isinstance(k, (int, SymbolicScalar)) for k in key]):
                return SymbolicScalar.from_base(pto_impl.GetTensorData(self._base, to_syms(key)))
            elif all([isinstance(k, slice) for k in key]):
                offsets, shapes = self._get_view_offset_shape(key, self.shape)
                return pto.view(self, shapes, offsets, valid_shape=valid_shape)
            elif all(isinstance(k, (slice, int, SymbolicScalar)) for k in key):
                new_key, bool_shape = self._get_slice_index(key)
                offsets, shapes = self._get_view_offset_shape(tuple(new_key), self.shape)
                res = pto.view(self, shapes, offsets, valid_shape=valid_shape)
                res_shape = [res.shape[d] for d in range(res.dim) if bool_shape[d]]
                return pto.reshape(res, res_shape)
            elif any(k is Ellipsis for k in key):
                ellipsis_count = sum(k is Ellipsis for k in key)
                if ellipsis_count > 1:
                    raise ValueError("Only one ... is supported")
                ellipsis_pos = next(i for i, k in enumerate(key) if k is Ellipsis)
                other_len = len(key) - 1
                colon_count = self.dim - other_len
                if colon_count < 0:
                    raise IndexError(f"Too many indices for tensor with dimension {self.dim}")
                colons = (slice(None),) * colon_count
                return self.__getitem__(key[:ellipsis_pos] + colons + key[ellipsis_pos + 1:])
                        
            else:
                raise ValueError("tuple key must be int or SymbolicScalar")
        else:
            raise RuntimeError("Invalid key type")

    @staticmethod
    def _get_slice_index(key):
        new_key = []
        bool_shape = []
        for axis, k in enumerate(key):
            if isinstance(k, (int, SymbolicScalar)):
                new_key.append(slice(int(k), int(k) + 1))
                bool_shape.append(False)
            else:
                new_key.append(key[axis])
                bool_shape.append(True)
        return new_key, bool_shape

    def base(self) -> pto_impl.Tensor:
        return self._base

    @classmethod
    def from_base(cls, base: pto_impl.Tensor) -> 'Tensor':
        obj = cls.__new__(cls)
        obj._base = base
        return obj

    def add(self, other: 'Tensor | int | float') -> 'Tensor':
        return pto.add(self, other)

    def __add__(self, other: 'Tensor | int | float') -> 'Tensor':
        return self.add(other)

    def __radd__(self, other: 'Tensor | int | float') -> 'Tensor':
        return self.add(other)

    def __iadd__(self, other: 'Tensor | int | float') -> 'Tensor':
        return self.add(other)

    def sub(self, other: 'Tensor | int | float') -> 'Tensor':
        return pto.sub(self, other)

    def __sub__(self, other: 'Tensor | int | float') -> 'Tensor':
        return self.sub(other)

    def __isub__(self, other: 'Tensor | int | float') -> 'Tensor':
        return self.sub(other)

    def mul(self, other: 'Tensor | int | float') -> 'Tensor':
        return pto.mul(self, other)

    def __mul__(self, other: 'Tensor | int | float') -> 'Tensor':
        return self.mul(other)

    def __imul__(self, other: 'Tensor | int | float') -> 'Tensor':
        return self.mul(other)

    def div(self, other: 'Tensor | int | float') -> 'Tensor':

        return pto.div(self, other)

    def __truediv__(self, other: 'Tensor | int | float') -> 'Tensor':
        return self.div(other)

    def __itruediv__(self, other: 'Tensor | int | float') -> 'Tensor':
        return self.div(other)

    def greater(self, other: 'Tensor'):
        return pto.greater(self, other)

    def __gt__(self, other: 'Tensor') -> 'Tensor':
        return self.greater(other)

    def __matmul__(self, other: 'Tensor') -> 'Tensor':
        if other.dtype in {pto.DT_FP16, pto.DT_BF16, pto.DT_FP32}:
            out_dype = other.dtype
        elif other.dtype == pto.DT_INT8:
            out_dype = pto.DT_INT32
        else:
            raise RuntimeError("unsupport dtype")
        return pto.matmul(self, other, out_dype)
    
    @property
    def dynamic_valid_shape(self) -> Union[List[int], List[SymbolicScalar]]:
        out = []
        for _, n in enumerate(self._base.GetDynValShape()):
            out.append(SymbolicScalar.from_base(n))

        return out

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

    def reshape(self, shape: List[int], *, valid_shape: Optional[List[Union[int, SymbolicScalar]]] = None,
                inplace: bool = False) -> 'Tensor':
        if inplace:
            return pto.reshape(self, shape, inplace=inplace)
        else:
            return pto.reshape(self, shape, valid_shape=valid_shape, inplace=inplace)

    def unsqueeze(self, dim: int) -> 'Tensor':
        return pto.unsqueeze(self, dim)

    def view(self, shape: List[int], offsets: List[Union[int, SymbolicScalar]],
             *, valid_shape: Optional[List[Union[int, SymbolicScalar]]] = None) -> 'Tensor':
        return pto.view(self, shape, offsets, valid_shape=valid_shape)

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

    def where(self, condition: 'Tensor', y: Union['Tensor', float]) -> 'Tensor':
        return pto.where(condition, self, y)

    def topk(self, k: int, dim: Optional[int] = None, largest: bool = True) -> Tuple['Tensor', 'Tensor']:
        return pto.topk(self, k, dim, largest)

    def exp(self) -> 'Tensor':
        return pto.exp(self)

    def log(self) -> 'Tensor':
        return pto.log(self)

    def logical_not(self) -> 'Tensor':
        return pto.logical_not(self)

    def amax(self, dim: int = -1, keepdim: bool = False) -> 'Tensor':
        return pto.amax(self, dim, keepdim)

    def amin(self, dim: int = -1, keepdim: bool = False) -> 'Tensor':
        return pto.amin(self, dim, keepdim)

    def sum(self, dim: int = -1, keepdim: bool = False) -> 'Tensor':
        return pto.sum(self, dim, keepdim)

    def rsqrt(self) -> 'Tensor':
        return pto.rsqrt(self)

    def sqrt(self) -> 'Tensor':
        return pto.sqrt(self)

    def transpose(self, dim0: int, dim1: int) -> 'Tensor':
        return pto.transpose(self, dim0, dim1)

    def gather(self, dim: int, index: 'Tensor') -> 'Tensor':
        return pto.gather(self, dim, index)

    def expand_clone(self, shape: List[int], *,
                     valid_shape: Optional[List[Union[int, SymbolicScalar]]] = None) -> 'Tensor':
        if valid_shape is None:
            valid_shape = []
        return pto.expand_clone(self, shape, valid_shape=valid_shape)

    def scatter_update(self, dim: int, index: 'Tensor', src: 'Tensor') -> 'Tensor':
        return pto.scatter_update(self, dim, index, src)

    def scatter_(self, dim: int, index: 'Tensor', src: float) -> 'Tensor':
        return pto.scatter_(self, dim, index, src)

    def scatter(self, dim: int, index: 'Tensor', src: float) -> 'Tensor':
        return pto.scatter(self, dim, index, src)


def mark_dynamic(tensor: 'Tensor', axis: int):
    """
    Mark a tensor axis as dynamic.

    Args:
        tensor (Tensor): The tensor to be marked as dynamic.
        axis (int): The axis to be marked as dynamic.

    Notes:
        The shape acquired before `mark_dynamic` will not be updated. It is
        recommended to call `mark_dynamic` before the shaped is used.
    """
    pto_impl.MarkDynamic(tensor._base, axis)
