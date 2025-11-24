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
import typing
from typing import Union, List, Optional, Tuple

import pypto
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
            start, stop, step = k.start, k.stop, k.step
            if step not in (1, None):
                raise ValueError("step must be 1 or None")
            if start is None and stop is None:
                offsets.append(0)
            elif isinstance(start, (int, SymbolicScalar)):
                offsets.append(start)
            elif isinstance(stop, (int, SymbolicScalar)):
                offsets.append(stop - shape[axis])
        return offsets

    def _is_empty_slice(self, key):
        if isinstance(key, slice):
            return key.start is None and key.stop is None and key.step is None
        elif isinstance(key, (int, SymbolicScalar)):
            return False
        elif key is Ellipsis:
            return False
        return all([self._is_empty_slice(k) for k in key])


    @staticmethod
    def _add_one_dim(key, value_shape):
        slices_cout = sum(1 for k in key if isinstance(k, slice))
        assert slices_cout == len(value_shape), (
            f"The number of slice in key ({slices_cout}) "
            f"must match the length of input Tensor ({len(value_shape)}). "
        )
        new_shape = []
        idx = 0
        for k in key:
            if isinstance(k, slice):
                new_shape.append(value_shape[idx])
                idx += 1
            else:
                new_shape.append(1)
        return new_shape

    def __setitem__(self, key, value):
        """
        Set tensor data by index or slice.

        Args:
            key (Union[int, SymbolicScalar, slice]): Index or slice to set.
            value (Tensor | Element): value to set.

        example:
        # All slice
        a = pypto.tensor((4, 4), pypto.DT_FP32)
        b = pypto.tensor((2, 2), pypto.DT_FP32)
        a[0:, 0:] = b # assemble(b, (0, 0), a)
        Input a:[[0, 0, 0, 0],
                 [0, 0, 0, 0],
                 [0, 0, 0, 0],
                 [0, 0, 0, 0],]
        Input b:[[10, 10]
                 [10, 10]]
        Output a:[[10, 10, 0, 0],
                  [10, 10, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0]]

        # Index and slice
        a = pypto.tensor((4, 4), pypto.DT_FP32)
        b = pypto.tensor((2), pypto.DT_FP32)
        a[0, 1:3] = b # reshape b to (1, 2), assemble(b, (0, 1), a)
        Input a:[[0, 0, 0, 0],
                 [0, 0, 0, 0],
                 [0, 0, 0, 0],
                 [0, 0, 0, 0],]
        Input b:[10, 10]
        Output a:[[0, 10, 10, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0]]

        # Negative index
        a = pypto.tensor((4, 4), pypto.DT_FP32)
        b = pypto.tensor(2), pypto.DT_FP32)
        a[-1, -3:-1] = b # equivalent to a[3, 1:3]

        # Ellipsis index
        a = pypto.tensor((4, 4), pypto.DT_FP32)
        b = pypto.tensor((2, 2), pypto.DT_FP32)
        a[..., 1:3] = b # equivalent to a[0:2, 1:3]

        # single data
        a = pypto.tensor((4, 4), pypto.DT_INT32)
        a[0, 0] = 1 #SetTensorData, supports only DT_INT32 tensors

        """

        if self._is_empty_slice(key):
            self.move(value)
            return

        if isinstance(key, slice) and isinstance(key.stop, Tensor):
            assert isinstance(key.start, int)
            return pypto.scatter(self, key.start, key.stop, value)

        key = self._normalize_key(key)

        if all(isinstance(k, (int, SymbolicScalar)) for k in key):
            pto_impl.SetTensorData(to_sym(value), to_syms(key), self._base)
            return

        if all(isinstance(k, slice) for k in key):
            offsets = self._get_assemble_offset(key, self.shape)
            return pypto.assemble(value, offsets, self)

        if all(isinstance(k, (slice, int, SymbolicScalar)) for k in key):
            new_shape = self._add_one_dim(key, value.shape)
            value_reshaped = pypto.reshape(value, new_shape)
            new_key, _ = self._get_slice_index(key)  # int→slice
            offsets = self._get_assemble_offset(tuple(new_key), self.shape)
            return pypto.assemble(value_reshaped, offsets, self)

        raise ValueError("tuple key must be int, SymbolicScalar or slice")


    def _normalize_key(self, key):
        if self._is_empty_slice(key):
            return key

        if isinstance(key, (int, SymbolicScalar, slice)) or key is Ellipsis:
            key = (key,)

        if not isinstance(key, tuple):
            raise RuntimeError("Invalid key type")

        if any(k is Ellipsis for k in key):
            ellipsis_count = sum(k is Ellipsis for k in key)
            if ellipsis_count > 1:
                raise ValueError("Only one ... is supported")

            ellipsis_pos = next(i for i, k in enumerate(key) if k is Ellipsis)
            other_len = len(key) - 1
            colon_count = self.dim - other_len
            if colon_count < 0:
                raise IndexError(f"Too many indices for tensor with dimension {self.dim}")
            colons = (slice(None),) * colon_count
            key = key[:ellipsis_pos] + colons + key[ellipsis_pos + 1:]

        assert self.dim == len(key), f"rank not match, expect {self.dim}, but got {len(key)}"
        key = self._negative_index_to_positive(key, self.shape)
        return key


    @staticmethod
    def _negative_index_to_positive(key, shape):
        normalized = []
        for axis, k in enumerate(key):
            size = shape[axis]
            if isinstance(k, (int, SymbolicScalar)):
                if isinstance(k, int) and k < 0:
                    k = size + k
                normalized.append(k)
                continue
            start, stop, step = k.start, k.stop, k.step
            if isinstance(start, int) and start < 0:
                start = size + start
            if isinstance(stop, int) and stop < 0:
                stop = size + stop
            normalized.append(slice(start, stop, step))
        return tuple(normalized)


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
        s = pypto.tensor([4, 4], pypto.DT_FP32)
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
                return pypto.gather(self, key.start, key.stop)
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
                return pypto.view(self, shapes, offsets, valid_shape=valid_shape)
            elif all(isinstance(k, (slice, int, SymbolicScalar)) for k in key):
                new_key, bool_shape = self._get_slice_index(key)
                offsets, shapes = self._get_view_offset_shape(tuple(new_key), self.shape)
                res = pypto.view(self, shapes, offsets, valid_shape=valid_shape)
                res_shape = [res.shape[d] for d in range(res.dim) if bool_shape[d]]
                return pypto.reshape(res, res_shape)
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
        for k in key:
            if isinstance(k, (int, SymbolicScalar)):
                new_key.append(slice(k, k + 1))
                bool_shape.append(False)
            else:
                new_key.append(k)
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
        return pypto.add(self, other)

    def __add__(self, other: 'Tensor | int | float') -> 'Tensor':
        return self.add(other)

    def __radd__(self, other: 'Tensor | int | float') -> 'Tensor':
        return self.add(other)

    def __iadd__(self, other: 'Tensor | int | float') -> 'Tensor':
        return self.add(other)

    def sub(self, other: 'Tensor | int | float') -> 'Tensor':
        return pypto.sub(self, other)

    def __sub__(self, other: 'Tensor | int | float') -> 'Tensor':
        return self.sub(other)

    def __isub__(self, other: 'Tensor | int | float') -> 'Tensor':
        return self.sub(other)

    def mul(self, other: 'Tensor | int | float') -> 'Tensor':
        return pypto.mul(self, other)

    def __mul__(self, other: 'Tensor | int | float') -> 'Tensor':
        return self.mul(other)

    def __imul__(self, other: 'Tensor | int | float') -> 'Tensor':
        return self.mul(other)

    def div(self, other: 'Tensor | int | float') -> 'Tensor':

        return pypto.div(self, other)

    def __truediv__(self, other: 'Tensor | int | float') -> 'Tensor':
        return self.div(other)

    def __itruediv__(self, other: 'Tensor | int | float') -> 'Tensor':
        return self.div(other)

    def greater(self, other: 'Tensor'):
        return pypto.greater(self, other)

    def __gt__(self, other: 'Tensor') -> 'Tensor':
        return self.greater(other)

    def __matmul__(self, other: 'Tensor') -> 'Tensor':
        if other.dtype in {pypto.DT_FP16, pypto.DT_BF16, pypto.DT_FP32}:
            out_dype = other.dtype
        elif other.dtype == pypto.DT_INT8:
            out_dype = pypto.DT_INT32
        else:
            raise RuntimeError("unsupport dtype")
        return pypto.matmul(self, other, out_dype)

    def matmul(
        self,
        mat2,
        out_dtype,
        *,
        a_trans=False,
        b_trans=False,
        c_matrix_nz=False,
        extend_params=None
    ) -> "Tensor":
        return pypto.matmul(
            self,
            mat2,
            out_dtype,
            a_trans=a_trans,
            b_trans=b_trans,
            c_matrix_nz=c_matrix_nz,
            extend_params=extend_params
        )

    def assemble(self, input: 'Tensor', offsets: List[Union[int, SymbolicScalar]]) -> None:
        """
        Assemble a small Tensor into a larger Tensor based on specified offsets.

        Args:
            input (Tensor): The small input tensor to be assembled into the larger tensor.
            offsets (Union[List[int], List[SymbolicScalar]]): Offset for placing the input tensor.

        example:
        s = pypto.tensor((16, 16), pypto.DT_FP32)
        a = pypto.tensor((2, 2), pypto.DT_FP32)
        s.assemble(a, [0, 0])
        """
        pypto.assemble(input, offsets, self)

    def reshape(self, shape: List[int], *, valid_shape: Optional[List[Union[int, SymbolicScalar]]] = None,
                inplace: bool = False) -> 'Tensor':
        if inplace:
            return pypto.reshape(self, shape, inplace=inplace)
        else:
            return pypto.reshape(self, shape, valid_shape=valid_shape, inplace=inplace)

    def unsqueeze(self, dim: int) -> 'Tensor':
        return pypto.unsqueeze(self, dim)

    def view(self, shape: List[int], offsets: List[Union[int, SymbolicScalar]],
             *, valid_shape: Optional[List[Union[int, SymbolicScalar]]] = None) -> 'Tensor':
        return pypto.view(self, shape, offsets, valid_shape=valid_shape)

    def sin(self) -> 'Tensor':
        return pypto.sin(self)

    def cos(self) -> 'Tensor':
        return pypto.cos(self)

    def sigmoid(self) -> 'Tensor':
        return pypto.sigmoid(self)

    def softmax(self, dim: int) -> 'Tensor':
        return pypto.softmax(self, dim)

    def maximum(self, other: 'Tensor') -> 'Tensor':
        return pypto.maximum(self, other)

    def where(self, condition: 'Tensor', y: Union['Tensor', float]) -> 'Tensor':
        return pypto.where(condition, self, y)

    def topk(self, k: int, dim: Optional[int] = None, largest: bool = True) -> Tuple['Tensor', 'Tensor']:
        return pypto.topk(self, k, dim, largest)

    def exp(self) -> 'Tensor':
        return pypto.exp(self)

    def log(self) -> 'Tensor':
        return pypto.log(self)

    def logical_not(self) -> 'Tensor':
        return pypto.logical_not(self)

    def amax(self, dim: int = -1, keepdim: bool = False) -> 'Tensor':
        return pypto.amax(self, dim, keepdim)

    def amin(self, dim: int = -1, keepdim: bool = False) -> 'Tensor':
        return pypto.amin(self, dim, keepdim)

    def sum(self, dim: int = -1, keepdim: bool = False) -> 'Tensor':
        return pypto.sum(self, dim, keepdim)

    def rsqrt(self) -> 'Tensor':
        return pypto.rsqrt(self)

    def sqrt(self) -> 'Tensor':
        return pypto.sqrt(self)

    def transpose(self, dim0: int, dim1: int) -> 'Tensor':
        return pypto.transpose(self, dim0, dim1)

    def gather(self, dim: int, index: 'Tensor') -> 'Tensor':
        return pypto.gather(self, dim, index)

    def expand_clone(self, shape: List[int], *,
                     valid_shape: Optional[List[Union[int, SymbolicScalar]]] = None) -> 'Tensor':
        if valid_shape is None:
            valid_shape = []
        return pypto.expand_clone(self, shape, valid_shape=valid_shape)

    def scatter_update(self, dim: int, index: 'Tensor', src: 'Tensor') -> 'Tensor':
        return pypto.scatter_update(self, dim, index, src)

    def scatter_(self, dim: int, index: 'Tensor', src: float) -> 'Tensor':
        return pypto.scatter_(self, dim, index, src)

    def scatter(self, dim: int, index: 'Tensor', src: float) -> 'Tensor':
        return pypto.scatter(self, dim, index, src)


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
