#!/usr/bin/env python3
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

import os
import sys
import logging


def setup_logging(level=logging.INFO):
    logging.basicConfig(
        level=level,
        format='%(levelname)s: %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )


def validate_path(path, description="路径"):
    if not os.path.exists(path):
        return False, f"{description}不存在: {path}"
    return True, None


def find_trace_log_dir(log_base_path):
    for root, dirs, files in os.walk(log_base_path):
        for file in files:
            if file.startswith('device') and file.endswith('.log'):
                log_file = os.path.join(root, file)
                try:
                    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        if '#trace' in content:
                            return root
                except Exception:
                    continue
    return None


def get_latest_dyn_topo(output_path):
    if not os.path.exists(output_path):
        return None, "output 目录不存在"

    if not os.path.isdir(output_path):
        return None, "路径不是目录"

    subdirs = []
    for item in os.listdir(output_path):
        item_path = os.path.join(output_path, item)
        if os.path.isdir(item_path) and item.startswith('output_'):
            subdirs.append(item_path)

    if not subdirs:
        return None, "未找到以 'output_' 开头的子文件夹"

    subdirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    latest_dir = subdirs[0]

    dyn_topo_path = os.path.join(latest_dir, 'dyn_topo.txt')
    if not os.path.exists(dyn_topo_path):
        return None, f"dyn_topo.txt 不存在于: {dyn_topo_path}"

    return dyn_topo_path, None
