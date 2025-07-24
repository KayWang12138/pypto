#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

"""
tile fwk ops compile script
"""
import ctypes
import json
import os
import tbe.common.context.op_context as op_context
from tbe.common.buildcfg import get_current_build_config
from tbe.common.platform.platform_info import get_soc_spec


def load_rt_lib():
    so_lib_path = os.getenv("TILE_FWK_RUNTIME_PATH")
    librt = ctypes.CDLL(so_lib_path)
    return librt


def ascendcpp_compile_op(*args):
    kernel_name = args[-1]
    cur_context = op_context.get_context()
    if cur_context is None:
        return False
    op_infos = cur_context.get_op_info()
    if op_infos is None or len(op_infos) == 0:
        return False
    op_info = op_infos[0]
    if op_info is None:
        return False
    op_type_c = op_info.op_type.encode('utf_8')
    soc_version = get_soc_spec("SOC_VERSION")
    soc_version_c = soc_version.encode('utf_8')
    dump_path = get_current_build_config("kernel_meta_parent_dir") + "/kernel_meta"
    dump_path_c = dump_path.encode('utf_8')
    kernel_name_c = kernel_name.encode('utf_8')
    try:
        librt = load_rt_lib()
        if librt is None:
            return False
        res = librt.TileFwkCompileFatbin(op_type_c, soc_version_c, dump_path_c, kernel_name_c)
    except Exception as e:
        raise RuntimeError("Exception: Fail to call compile func, reason is %s." % str(e)) from e
    if bool(res) is not True:
        return False
    return True
