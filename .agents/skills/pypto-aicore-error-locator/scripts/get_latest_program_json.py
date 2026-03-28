#!/usr/bin/env python3
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

import os
import sys
import logging
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import setup_logging

setup_logging()

logger = logging.getLogger(__name__)


def get_latest_program_json(output_path: str) -> Optional[str]:
    """
    获取 output 目录下最新子文件夹中的 program.json 路径
    
    Args:
        output_path: output 目录的绝对路径
        
    Returns:
        program.json 的完整路径，如果不存在则返回 None
    """
    if not os.path.exists(output_path):
        logger.error("目录不存在: %s", output_path)
        return None
    
    if not os.path.isdir(output_path):
        logger.error("路径不是目录: %s", output_path)
        return None
    
    # 列出所有以 output_ 开头的子文件夹
    subdirs = []
    for item in os.listdir(output_path):
        item_path = os.path.join(output_path, item)
        if os.path.isdir(item_path) and item.startswith('output_'):
            subdirs.append(item_path)
    
    if not subdirs:
        logger.error("未找到以 'output_' 开头的子文件夹")
        return None
    
    # 按修改时间排序，获取最新的文件夹
    subdirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    latest_dir = subdirs[0]
    
    logger.info("找到 %d 个 output 子文件夹", len(subdirs))
    logger.info("最新子文件夹: %s", latest_dir)
    
    # 检查 program.json 是否存在
    program_json_path = os.path.join(latest_dir, 'program.json')
    if not os.path.exists(program_json_path):
        logger.error("program.json 不存在于: %s", program_json_path)
        return None
    
    return program_json_path


def main():
    if len(sys.argv) < 2:
        logger.info("用法: python get_latest_program_json.py <output_path>")
        sys.exit(1)
    
    output_path = sys.argv[1]
    logger.info("Output 路径: %s", output_path)
    
    program_json_path = get_latest_program_json(output_path)
    
    if program_json_path:
        print(program_json_path)
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == '__main__':
    main()
