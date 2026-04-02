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
"""PyPTO
"""

# torch/torch_npu may use cxxabi=0 or cxxabi=1, while pypto only support cxxabi=0
# if pypto load first, torch/torch_cpu may crash, force load torch first
try:
    import torch
except ImportError:
    pass


def _load_shared_libs():
    import os
    import ctypes
    from pathlib import Path
    from importlib import metadata
    from typing import List, Any

    cur_dir: Path = Path(__file__).parent
    pkg_dir: Path = Path(str(metadata.distribution("pypto").locate_file("pypto"))).resolve()
    pkg_dir = pkg_dir if pkg_dir == cur_dir else cur_dir  # 适配 edit 模式
    lib_dir: Path = Path(pkg_dir, "lib")
    use_cann: bool = bool(os.environ.get("ASCEND_HOME_PATH"))

    def _load_shared_lib(_desc: List[Any]):
        _name: str = _desc[0]
        _load: bool = _desc[1]
        if not _load:
            return
        _file: Path = Path(lib_dir, _name)
        if not _file.exists():
            return
        ctypes.CDLL(str(_file), mode=ctypes.RTLD_GLOBAL)

    _load_shared_lib(_desc=["libc_sec.so", not use_cann, ])

    # name, load
    desc_lst: List[List[Any]] = [
        ["libtile_fwk_utils.so", True, ],
        ["libtile_fwk_cann_host_runtime.so", True, ],
        ["libtile_fwk_platform.so", True, ],
        ["libtile_fwk_interface.so", True, ],
        ["libtile_fwk_codegen.so", True, ],
        ["libtile_fwk_compiler.so", True, ],
        ["libtile_fwk_runtime.so", use_cann, ],
        ["libtile_fwk_runtime_stub.so", not use_cann, ],
        ["libtile_fwk_simulation.so", True, ],
        ["libtile_fwk_simulation_ca.so", True, ],
        ["libtile_fwk_simulation_pv.so", use_cann, ],
    ]
    for desc in desc_lst:
        _load_shared_lib(_desc=desc)


_load_shared_libs()

from . import experimental

from .config import *  # noqa
from ._controller import *  # noqa
from .converter import from_torch
from .enum import *  # noqa
from .op import *  # noqa
from .operation import *  # noqa
from .operator import *  # noqa
from .pass_config import *  # noqa
from .cost_model import *  # noqa
from ._utils import ceildiv, bytes_of
from .platform import platform
from .runtime import verify, set_verify_golden_data, RunMode
from .symbolic_scalar import SymbolicScalar
from .tensor import Tensor
from .functions import Function, get_last_function, get_current_function
from ._element import Element

# Import frontend after all other imports to avoid circular imports
from . import frontend


tensor = Tensor
element = Element
symbolic_scalar = SymbolicScalar


_atexit_registered = False


def _register_atexit_handler():
    global _atexit_registered
    if _atexit_registered:
        return
    
    import atexit
    from . import pypto_impl
    
    def _cleanup_on_exit():
        try:
            print(f"DeviceDestory called in PID: {os.getpid()}")
            pypto_impl.DeviceDestory()
        except Exception as e:
            print(f"DeviceDestory error in PID {os.getpid()}: {e}")
    
    atexit.register(_cleanup_on_exit)
    _atexit_registered = True
    print(f"atexit handler registered in PID: {os.getpid()}")


def _register_fork_handler():
    try:
        import multiprocessing.util
        
        def _after_fork_in_child():
            print(f"After fork in child PID: {os.getpid()}")
            _register_atexit_handler()
        
        multiprocessing.util.register_after_fork(_register_atexit_handler, _after_fork_in_child)
        print(f"Fork handler registered in PID: {os.getpid()}")
    except Exception as e:
        print(f"Failed to register fork handler in PID {os.getpid()}: {e}")


_register_atexit_handler()
_register_fork_handler()
