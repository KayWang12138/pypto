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
from typing import Sequence, Optional, Tuple, Callable
import os
import sys
import inspect


class CompileOption:
    """
    Base class for compile options.
    """

    option_name = ""  # name of the compile option in the pipeline

    def __init__(self, val):
        self._value = val

    def serialize(self):
        return f"{self.__class__.option_name}={self._value}"

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        self._value = value


class BooleanCompileOption(CompileOption):
    def __init__(self, val: bool = True):
        super().__init__(val)

    def serialize(self):
        return f"{self.__class__.option_name}={'true' if self._value else 'false'}"


class StringCompileOption(CompileOption):
    def serialize(self):
        if self._value:
            self._value = self._value.strip("'")
            return f"{self.__class__.option_name}='{self._value}'"
        return ""


class BooleanBasedFileDumpOption(CompileOption):
    def __init__(self, val: bool = True):
        super().__init__(val)
        self._dump_path = ""

    @property
    def dump_path(self):
        return self._dump_path

    @dump_path.setter
    def dump_path(self, path):
        self._dump_path = path

    def serialize(self):
        if self._value:
            assert self._dump_path, (
                f"Dump path is not set for {self.__class__.__name__}"
            )
            return f"{self.__class__.option_name}='{self._dump_path}'"
        return ""


class EmptyCompileOption(CompileOption):
    def serialize(self):
        return ""


class OptLevel(CompileOption):
    option_name = "opt-level"

    def __init__(self, val: int):
        if val < 0 or val > 3:
            raise RuntimeError(f"Invalid OPT_LEVEL: {val}, valid range is [0, 3]")
        super().__init__(val)

class NPUArch(StringCompileOption):
    option_name = "arch"

    def __init__(self, val):
        super().__init__(val)

    @property
    def value(self) -> bool:
        return self._value

    @value.setter
    def value(self, value: bool):
        self._value = value


class CompileOptions:
    """
    This class encapsulates compilation options to configure the JIT compilation.
    It provides a convenient way to manage and pass compilation options.
    By centralizing these options, it ensures consistent and flexible configuration of
    compilation parameters such as optimization level, debugging control, etc.
    """

    def __init__(self, options=None):
        self.options = {
            # Compilation control options
            OptLevel: OptLevel(3),
            NPUArch: NPUArch(""),
        }

        if options is not None:
            self._update(options)

    def _update(self, options):
        def _validate_and_update_option(option):
            if type(option) not in self.options:
                raise RuntimeError(f"Invalid compile option: {option}")
            self.options[type(option)] = option

        if isinstance(options, tuple):
            for option in options:
                _validate_and_update_option(option)
        else:
            _validate_and_update_option(options)


    @property
    def gpu_arch(self) -> str:
        return self.options[NPUArch].value


    def to_str(self) -> str:
        """
        Generate a string representation of all compilation options
        which will be used in pipeline options.
        """
        flattend_options = ""
        for option in self.options.values():
            flattend_options += option.serialize() + " "

        log().info("`ptotile.compile` CompileOptions: options=" + flattend_options)
        return flattend_options


def _parse_compile_options_from_str(options: str) -> CompileOptions:
    """
    Parse the compile options from a string.
    Deprecated and will be removed in the future.
    """

    def _get_compile_option_from_str(option_str: str):
        mapping = {
            "opt_level": OptLevel,
            "gpu_arch": NPUArch,
        }
        return mapping[option_str]

    import argparse
    import shlex

    parser = argparse.ArgumentParser()
    parser.add_argument("--opt-level", nargs="?", type=int, default=3)
    parser.add_argument("--npu-arch", type=str, default="")
    compile_options = CompileOptions()
    try:
        # Use shlex to properly handle options with spaces
        parsed_options = shlex.split(options) if options else []
        # Avoid parsing the ptxas-options value as a hyphen key
        for i in range(1, len(parsed_options)):
            if parsed_options[i - 1] in ["--ptxas-options"]:
                parsed_options[i] = f"'{parsed_options[i]}'"
        option_dict = vars(parser.parse_args(parsed_options))
        for option, value in option_dict.items():
            option = _get_compile_option_from_str(option)
            compile_options.options[option].value = value
    except SystemExit as e:
        # catch argparse error and raise as RuntimeError
        raise RuntimeError(
            f"Invalid compile options: '{options}'. Please check the option values and format."
        ) from e

    return compile_options

def generate_mlir(funcBody, kwargs, compile_only, no_cahce):
    """
    Placeholder function to generate MLIR from the function.
    In actual implementation, this function will convert the function
    into MLIR representation.
    """
    print("Generating MLIR...")
    
    return None


class CompileCallable:
    def __init__(self, options=None):
        def preprocess_options(option):
            if type(option) is type and issubclass(
                option, (BooleanCompileOption, BooleanBasedFileDumpOption)
            ):
                # Automatically creates a True instance of the option
                return option(True)
            elif isinstance(option, tuple):
                return tuple(preprocess_options(opt) for opt in option)
            return option

        self._compile_options = CompileOptions(preprocess_options(options))

    def __getitem__(self, options):
        """
        Get a new CompileCallable object with the specified options.
        """
        new_callable_with_options = CompileCallable(options)
        return new_callable_with_options

    def __call__(self, *args, **kwargs):
        return self._compile(*args, **kwargs)

    def _compile(self, func, *args, **kwargs):
        """
        This function is used to compile a `ptotile.jit` decorated function.
        It will process the compile options and input parameters, do explicit compilation and return  the jit executor.

        :param func: The function to compile. It can be a regular function, a method or a class instance.
        :param args: The arguments to pass to the function.
        :param kwargs: The keyword arguments to pass to the function. It can contain `options` like
        `opt_level` to control the compilation flags.

        :return: The jit executor.

        :raises: RuntimeError if the function is not decorated with `ptotile.jit` or is not callable.
        """
        if func is None:
            raise RuntimeError("Function is not set or invalid.")

        if not callable(func):
            raise RuntimeError("Object is not callable.")

        kwargs["compile_only"] = True
        kwargs["no_cache"] = True

        if inspect.isfunction(func):
            # regular function
            pass
        elif inspect.ismethod(func):
            # if it's a method, add the instance to the first argument
            args = [func.__self__] + list(args)
            func = func.__func__
        elif (
            inspect.isclass(type(func))
            and hasattr(func, "__call__")
            and hasattr(func.__call__, "__func__")
        ):
            # If it's a class instance, get the class's __call__ method
            args = [func] + list(args)
            # Get the actual function from the class definition
            func = func.__call__.__func__
        else:
            raise RuntimeError(
                "Invalid function type, only function, method and module are supported, but got",
                func,
            )

        func_name_prefix = getattr(func, "_name_prefix", None)
        if func_name_prefix:
            kwargs["_name_prefix"] = func_name_prefix

        # If it's a wrapped function created by decorators, get the original function
        while hasattr(func, "__wrapped__"):
            func = func.__wrapped__

        if not hasattr(func, "_dsl_object"):
            raise RuntimeError(
                f"Function {func} is not decorated with jit decorator."
            )

        # process compile options, extract the options and remove them from the kwargs
        options = kwargs.pop("options", None)
        if isinstance(options, str) and len(options) == 0:
            options = None

        if options is not None and isinstance(options, str):
            compile_options = _parse_compile_options_from_str(options)
        else:
            compile_options = self._compile_options
        func.compile_options = compile_options

        # TODO: Preprocess the function if not already preprocessed

        # TODO: 生成中间IR、并调用ptoas进行编译
        return generate_mlir()