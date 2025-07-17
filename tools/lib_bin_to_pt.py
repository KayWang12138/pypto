#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================
"""
"""
import sys
import torch


def _buf_to_t(buf, dtype, shape=None, byte_order='native'):
    t_storage = torch.UntypedStorage.from_buffer(buf, dtype=dtype, byte_order=byte_order)
    t = torch.tensor(t_storage, dtype=dtype)
    if shape:
        t = t.reshape(shape)
    return t


def file_bin_to_pt(filename_in, dtype, shape=None, filename_out=None, byte_order='native'):
    if byte_order not in ['native', 'big', 'little']:
        assert False
    with open(filename_in, 'rb') as f:
        bin_buf = f.read()
        t = _buf_to_t(bin_buf, dtype, shape, byte_order)
        if filename_out:
            torch.save(t, filename_out)
        return t


if __name__ == '__main__':
    def usage():
        print(f'Usage: {sys.argv[0]} <RAW_DATA.EXT> <PT_FILE.EXT> <dtype> [shape]')
        print(f'    dtype: bfloat16|...|float16|float32|...')
        print(f'    shape: a,b,...,n')

    if len(sys.argv) < 4:
        usage()
        exit()

    filename_in = sys.argv[1]
    filename_out = sys.argv[2]
    dtype = sys.argv[3]    # bfloat16/float16/float32/...

    if dtype in ['bfloat16']:
        out_dtype = torch.bfloat16
    elif dtype in ['float32']:
        out_dtype = torch.float32
    else:
        usage()
        assert False, f'Not implemented for dtype={dtype}'

    if len(sys.argv) > 4:
        shape = sys.argv[4]
        out_shape = [int(i) for i in shape.strip().split(',')]
    else:
        out_shape = None

    t = file_bin_to_pt(filename_in, out_dtype, shape=out_shape, filename_out=filename_out)
    print(t.shape, t)

    tt = torch.load(filename_out)
    print(tt.shape, tt)


