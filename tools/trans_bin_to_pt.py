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
from dump_data_paraser import pars


def parse_bin_files(root_dir, save_pt_dir):
    # 遍历目录树
    for root, _, files in os.walk(root_dir):
        for file in files:
            # 检查文件扩展名是否为 .bin
            if file.endswith('.bin'):
                # 构建文件的完整路径
                file_path = os.path.join(root, file)
                try:
                    # 打开文件以二进制只读模式
                    pars(file_path, save_pt_dir)
                    print(f"File: {file_path}")
                except Exception as e:
                    print(f"Error parsing {file_path}: {e}")
                    return


def main():
    if len(sys.argv) > 1:
        directory = sys.argv[1]
        save_pt_dir = None
        if len(sys.argv) > 2:
            save_pt_dir = sys.argv[2]
        parse_bin_files(directory, save_pt_dir)

    else:
        print("请提供一个目录名作为参数。")

if __name__ == '__main__':
    main()