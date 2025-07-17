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
import os
import sys
import torch
_error_file_cnt = 0


def show_statistic(d, brief=''):
    shape_info = ','.join([str(i) for i in d.shape])
    dd = d.flatten()
    dd_len = dd.numel()
    nz_cnt = len(dd.nonzero())
    z_cnt = dd_len - nz_cnt
    z_percent = z_cnt / dd_len
    threshhold = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    if z_percent > threshhold:
        global _error_file_cnt
        _error_file_cnt += 1
        inf_cnt = dd.isinf().sum().item()
        nan_cnt = dd.isnan().sum().item()
        print(f'{brief}[{shape_info}]: inf={inf_cnt} nan={nan_cnt} zero={z_cnt}, {z_percent}')


if __name__ == '__main__':
    if len(sys.argv) > 1:
        dir = sys.argv[1]
        if os.path.isdir(dir):
            for file in os.listdir(dir):
                file_path = f'{dir}/{file}'
                tensor = torch.load(file_path)
                show_statistic(tensor, f'{file}')
            print(f'Toal_file_cnt: {len(os.listdir(dir))}, zero_file_cnt: {_error_file_cnt}')
        elif os.path.isfile(dir):
            tensor = torch.load(dir)
            show_statistic(tensor, f'{dir}')
            print(f'Toal_file_cnt: 1, zero_file_cnt: {_error_file_cnt}')