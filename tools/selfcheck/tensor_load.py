#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Tensor load utilility."""

import torch


def get_valid_file_type_list():
    return ['pt', 'np', 'fp32', 'bf16', 'fp16']

def from_file(fn, file_type='fp32'):
    t = None
    dtype = torch.float32
    dtype_bytes = 4
    mode = 'bin-le'
    if file_type in ['fp32', 'float32', torch.float32]:
        dtype = torch.float32
        mode = 'bin-le'
    elif file_type in ['bf16', 'bfloat16', torch.bfloat16]:
        dtype = torch.bfloat16
        dtype_bytes = 2
        mode = 'bin-le'
    elif file_type in ['fp16', 'float16', torch.float16]:
        dtype = torch.float16
        dtype_bytes = 2
        mode = 'bin-le'
    elif file_type in ['pt']:
        mode = 'pt'
    elif file_type in ['np', 'npy']:
        mode = 'np'
    else:
        assert False, f'not support file type: {file_type}'

    if mode == 'bin-le':
        import os
        f_size = os.path.getsize(fn)
        t_numel, _remainder = divmod(f_size, dtype_bytes)
        assert _remainder == 0, f'invalid file size ({f_size}) for file_type={file_type}, possibly invalid file_type/file_content (file: {fn})'
        t = torch.from_file(fn, shared=False, size=t_numel, dtype=dtype)
    elif mode == 'bin-be':
        assert False, 'not support big-endian binary format: {mode}'
    elif mode == 'pt':
        t = torch.load(fn, map_location='cpu', weights_only=False)
    elif mode == 'np':
        import numpy as np
        t = torch.tensor(np.load(fn, mmap_mode='r'))
    else:
        assert False, 'internal error'

    return t
    