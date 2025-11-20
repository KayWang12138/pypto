# -----------------------------------------------------------------------------------------------------------
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
from .cube import CubeModule
from .vec import VecModule
from .kernel import KernelBase
from .tileop import TileOpModule
from .utils import Var, GMTensor
from typing import Union, Optional
from collections.abc import Callable
import inspect


class vec_func():
    def __init__(self, name: Optional[str]=None):
        self.name = name

    def __call__(self, fn: Callable) -> Callable[..., VecModule]:
        def wrapped_fn(*args):
            vs: list[Var] = []
            tensors: list[GMTensor] = []

            for a in args:
                if isinstance(a, Var):
                    vs.append(a)
                elif isinstance(a, GMTensor):
                    tensors.append(a)
                else:
                    raise TypeError('Kernel function only accepts Var or Tensor as inputs')

            new_vec = VecModule(*vs)
            new_vec.assign_vars_funcmode(*vs)
            if self.name is None:
                new_vec.set_name(fn.__name__)
            else:
                new_vec.set_name(self.name)

            new_args: list[Union[Var, GMTensor]] = []
            for a in args:
                if isinstance(a, Var):
                    if a in new_vec._var_mapping:
                        new_args.append(new_vec._var_mapping[a])
                    else:
                        raise ValueError('Cannot find corresponding variable in cube function entrance')
                elif isinstance(a, GMTensor):
                    new_args.append(a)

            new_vec.forward = fn
            ret = new_vec.inner_forward(*new_args)

            if ret is not None:
                raise ValueError('Kernel function should not return any value')
            return new_vec

        return wrapped_fn


class cube_func():
    def __init__(self, name: Optional[str]=None):
        self.name = name

    def __call__(self, fn: Callable) -> Callable[..., CubeModule]:
        def wrapped_fn(*args):
            vs: list[Var] = []
            tensors: list[GMTensor] = []

            for a in args:
                if isinstance(a, Var):
                    vs.append(a)
                elif isinstance(a, GMTensor):
                    tensors.append(a)
                else:
                    raise TypeError('Kernel function only accepts Var or Tensor as inputs')

            new_cube = CubeModule(*vs)
            new_cube.assign_vars_funcmode(*vs)
            if self.name is None:
                new_cube.set_name(fn.__name__)
            else:
                new_cube.set_name(self.name)

            new_args: list[Union[Var, GMTensor]] = []
            for a in args:
                if isinstance(a, Var):
                    if a in new_cube._var_mapping:
                        new_args.append(new_cube._var_mapping[a])
                    else:
                        raise ValueError('Cannot find corresponding variable in cube function entrance')
                elif isinstance(a, GMTensor):
                    new_args.append(a)

            new_cube.forward = fn
            ret = new_cube.inner_forward(*new_args)

            if ret is not None:
                raise ValueError('Kernel function should not return any value')
            return new_cube

        return wrapped_fn


class kernel_func():
    # def __init__(self):
        # self.name = name

    def __call__(self, fn) -> Callable[..., KernelBase]:
        def wrapped_fn(*args):
            vs: list[Var] = []
            tensors: list[GMTensor] = []

            for a in args:
                if isinstance(a, Var):
                    vs.append(a)
                elif isinstance(a, GMTensor):
                    tensors.append(a)
                else:
                    raise TypeError('Kernel function only accepts Var or Tensor as inputs')

            new_kernel = KernelBase(*vs)
            new_kernel.forward = fn
            ret = new_kernel.inner_forward(*args)

            if ret is not None:
                raise ValueError('Kernel function should not return any value')
            return new_kernel

        return wrapped_fn


class tileop_func():
    def __call__(self, fn) -> Callable[..., TileOpModule]:
        # sig = inspect.signature(fn)
        # for name, param in sig.parameters.items():
        #     print(name, param.kind, param.default)

        def wrapped_fn(*args):
            new_tileop = TileOpModule()
            new_tileop.set_name(fn.__name__)
            new_tileop.forward = fn  # type: ignore

            new_tileop(*args)
            return new_tileop
        return wrapped_fn
