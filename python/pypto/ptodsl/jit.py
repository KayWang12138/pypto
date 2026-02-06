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
import os
import subprocess
import sys
from pathlib import Path
import ctypes
import functools
import inspect
from typing import Dict, Tuple, Optional, Callable, Any
from dataclasses import dataclass
import torch

from . import _jit_functions, _kernel_functions
from . import ir_builder, register_function
from ptodsl.edit_cpp import convert

@dataclass
class KernelConfig:
    """内核配置"""
    compile_options: Dict


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
        # 先编译

        # 通过调用pybind的c++ kernel launch函数来实现
        print("launch kernel")
        return

    def __call__(self, *args, **kwargs):
        return self.launch(*args, **kwargs)

def _call_meta_and_capture_env(meta_fn):
    """Run meta_fn() and capture its local namespace (for types etc.). Returns (return_value, env dict)."""
    env = {}

    if meta_fn is None:
        return None, env
    def trace(frame, event, arg):
        if event == "return":
            env.clear()
            env.update(frame.f_locals)
        return trace

    old_trace = sys.gettrace()
    sys.settrace(trace)
    try:
        result = meta_fn()
    finally:
        sys.settrace(old_trace)
    return result, env


def pto_meta_data(f):
    """Decorator that marks a function as the meta-data provider (types, config) for jit_compile."""
    return f

def compile_module(module, clean_up=True, timeout=20):
    Path("./build").mkdir(parents=True, exist_ok=True)
    ir_path = "./build/temp.pto"  # TODO: use Python `tempfile` module
    raw_cpp_path = "./build/temp_generated.cpp"
    edited_cpp_path = "./build/temp_edited.cpp"
    lib_path = "./build/temp_lib.so"

    # step 1, Python -> IR
    with open(ir_path, "w") as f:
        f.write(str(module))  # TODO: a direct `module.dump(path)` API?

    # step 2, IR -> CPP
    # TODO: use `ptoas --enable-insert-sync` so no need for explicit sync in frontend
    # need https://github.com/zhangstevenunity/PTOAS/issues/10
    subprocess.run(
        ["ptoas", ir_path, "-o", raw_cpp_path],
        timeout=timeout, stderr=subprocess.DEVNULL
    )

    # Step 3, preprocess cpp source
    # TODO: should extend `ptoas` emitc to largely replace this ad-doc editing
    content = Path(raw_cpp_path).read_text(encoding="utf-8")
    converted = convert(content)
    Path(edited_cpp_path).write_text(converted, encoding="utf-8")

    # Step 4, cpp -> so
    PTO_LIB_PATH = os.environ["PTO_LIB_PATH"]
    ASCEND_HOME_PATH = os.environ.get("ASCEND_HOME_PATH")
    LD_LIB_PATH = ASCEND_HOME_PATH + "/lib64/"
    flags = [
        "-fPIC",
        "-shared",
        "-xcce",
        "--npu-arch=dav-2201",
        "-DMEMORY_BASE",  # here hardcoded for A2A3; TODO: expose this option to jit interface
        "-O2",
        "-std=c++17",
        f"-I{PTO_LIB_PATH}/include",
    ]

    subprocess.run(
        ["bisheng", *flags, edited_cpp_path, "-L", LD_LIB_PATH, "-lruntime", "-o", lib_path],
        timeout=timeout
    )

    if clean_up:
        os.remove(ir_path)
        os.remove(raw_cpp_path)
        os.remove(edited_cpp_path)

    return lib_path


def torch_to_ctypes(tensor):
    return ctypes.c_void_p(tensor.data_ptr())


def load_lib(lib_path, clean_up=True):
    import torch_npu

    lib = ctypes.CDLL(lib_path)

    default_block_dim = 1  # TODO: extend kernel to multi-core

    def func_wrapper(
        x,
        y,
        block_dim=default_block_dim,
        stream=None
    ):
        if stream is None:
            stream = torch.npu.current_stream()
        # TODO (important): matching call signature to arg list information in Python `build_module`
        lib.call_kernel(
            block_dim,
            stream._as_parameter_,
            torch_to_ctypes(x),
            torch_to_ctypes(y)
        )

    if clean_up:
        os.remove(lib_path)

    return func_wrapper

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


def kernel(options=None, meta_data=None, *dargs, **kwargs):
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

    def decorator(kernel_fn):
        name = kernel_fn.__name__
        signature = inspect.signature(kernel_fn)
        compiled_func = None
        # 创建内核配置
        config = KernelConfig(
            compile_options=options
        )

        # 存储内核函数信息
        _kernel_functions[name] = {
            'func': kernel_fn,
            'name': name,
            'signature': signature,
            'config': config,
            'kwargs': kwargs
        }
        print("In kernel decorator, registered kernel:", name)
        @functools.wraps(kernel_fn)
        def wrapper(*args, **kwargs):
            nonlocal compiled_func
            if compiled_func is None:
                with ir_builder() as module:
                    _result, env = _call_meta_and_capture_env(meta_data)
                    kernel_globals = {**kernel_fn.__globals__, **env}
                    kernel_with_env = type(kernel_fn)(
                        kernel_fn.__code__,
                        kernel_globals,
                        kernel_fn.__name__,
                        kernel_fn.__defaults__,
                        kernel_fn.__closure__,
                    )
                    ann = kernel_fn.__annotations__
                    resolved = {
                        k: env[v] if isinstance(v, str) and v in env else v
                        for k, v in ann.items()
                    }
                    kernel_with_env.__annotations__ = resolved
                    register_function(kernel_with_env)
                lib_path = compile_module(module)
                compiled_func = load_lib(lib_path)
            return compiled_func
        wrapper.set_name_prefix = name
        return wrapper
    
    if len(dargs) == 1 and callable(dargs[0]):
        return decorator(dargs[0])
    else:
        return decorator
