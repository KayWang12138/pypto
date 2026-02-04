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
import functools
import inspect
from typing import Dict, Tuple, Optional, Callable, Any
from dataclasses import dataclass

from . import _jit_functions, _kernel_functions

@dataclass
class KernelConfig:
    """内核配置"""
    compile_options: str


class KernelLauncher:
    """
    This class is used to launch a kernel function.
    Usage:
        ```python
        @ptotile.kernel
        def kernel(arg1, arg2, ...):
            ...

        @ptotile.jit
        def launch_kernel():
            kernel(arg1, arg2, ...).launch()
            # or
            kernel(arg1, arg2, ...)()
        ```
    """

    def __init__(
        self,
        funcBody,
        *func_args,
        **func_kwargs,
    ):
        self.funcBody = funcBody
        self.func_args = func_args
        self.func_kwargs = func_kwargs

        self._name_prefix = func_kwargs.pop("_name_prefix", None)

        self._check_func_args(funcBody, *func_args, **func_kwargs)

    def _check_func_args(self, funcBody, *func_args, **func_kwargs):
        # Get function signature
        sig = inspect.signature(funcBody)

        # func_args and func_kwargs should match funcBody's signature,
        # no extra or missing arguments.
        try:
            sig.bind(*func_args, **func_kwargs)
        except TypeError as e:
            raise RuntimeError(
                f"Failed to bind arguments to function `{funcBody.__name__}` with signature `{sig}`",
                cause=e,
            )

    def launch(self, *args, **kwargs):
        # 通过调用pybind的c++ kernel launch函数来实现
        print("launch kernel")
        return

    def __call__(self, *args, **kwargs):
        return self.launch(*args, **kwargs)


def jit(target: str = None, optimize: bool = True, cache: bool = True, 
    preprocess: bool = True,
    *dargs, **kwargs):
    """
    @pto.jit 装饰器: 标记函数为JIT编译函数
    
    参数:
        func: 要装饰的函数
        target: 编译目标 ('cpu', 'npu')
        optimize: 是否启用优化
        cache: 是否缓存编译结果
    
    示例:
        @pto.jit
        def add(a, b):
            workspace_size = 100
            pto.launch(add_kernel)
            return workspace_size
    """
    
    def decorator(f):
        # 获取函数信息
        name = f.__name__
        signature = inspect.signature(f)
        
        # 获取源代码（去除装饰器行）
        try:
            source_lines = inspect.getsource(f).split('\n')
            # 移除装饰器行
            source_lines = [line for line in source_lines 
                          if '@pto.jit' not in line and '@pto.kernel' not in line]
            source_code = '\n'.join(source_lines).strip()
        except:
            source_code = "<source unavailable>"
        print("In jit decorator, registering JIT function:", name)
        # 存储JIT函数信息
        _jit_functions[name] = {
            'func': f,
            'name': name,
            'signature': signature,
            'source_code': source_code,
            'target': target or 'cpu',
            'optimize': optimize,
            'cache': cache,
            'kwargs': kwargs
        }
        
        @functools.wraps(f)
        def wrapper(*args, **kwargs_):
            # if name in _compiled_cache:
            #     return _compiled_cache[name](*args, **kwargs_)
            
            # 否则回退到Python执行
            print(f"[pto] Warning: JIT function '{name}' not compiled, "
                  f"falling back to Python execution")
            return f(*args, **kwargs_)
        
        return wrapper
    
    if len(dargs) == 1 and callable(dargs[0]):
        return decorator(dargs[0])
    else:
        return decorator


def kernel(options="", *dargs, **kwargs):
    """
    @pto.kernel 装饰器: 标记函数为device上执行的函数,需要映射成ptoas编译器支持的mlir

    参数:
        func: 要装饰的函数    
    示例:
        @pto.kernel()
        def add(x, y, out):
            a_tile = pto.vec(128,128)
            b_tile = pto.vec(128,128)
            c_tile = pto.vec(128,128)
            pto.load(a_tile, x)
            pto.load(b_tile, y)
            pto.add(c_tile, a_tile, b_tile)
            pto.store(out, c_tile)
    """

    def decorator(f):
        name = f.__name__
        signature = inspect.signature(f)

        # 创建内核配置
        config = KernelConfig(
            compile_options=options
        )

        # 存储内核函数信息
        _kernel_functions[name] = {
            'func': f,
            'name': name,
            'signature': signature,
            'config': config,
            'kwargs': kwargs
        }
        print("In kernel decorator, registered kernel:", name)
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            return KernelLauncher(
                    f,
                    *args,
                    **kwargs,
                    _name_prefix=name,
                )
            # TODO: funcBody and args preprocessing
        wrapper.set_name_prefix = name
        return wrapper
    
    if len(dargs) == 1 and callable(dargs[0]):
        return decorator(dargs[0])
    else:
        return decorator
