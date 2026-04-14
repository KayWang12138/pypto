#!/usr/bin/env python3
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

import os
import sys
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import setup_logging, validate_path

setup_logging()

logger = logging.getLogger(__name__)


def get_latest_program_json(output_path):
    if not os.path.exists(output_path):
        logger.error("目录不存在: %s", output_path)
        return None

    if not os.path.isdir(output_path):
        logger.error("路径不是目录: %s", output_path)
        return None

    subdirs = []
    for item in os.listdir(output_path):
        item_path = os.path.join(output_path, item)
        if os.path.isdir(item_path) and item.startswith('output_'):
            subdirs.append(item_path)

    if not subdirs:
        logger.error("未找到以 'output_' 开头的子文件夹")
        return None

    subdirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    latest_dir = subdirs[0]

    logger.info("找到 %d 个 output 子文件夹", len(subdirs))
    logger.info("最新子文件夹: %s", latest_dir)

    program_json_path = os.path.join(latest_dir, 'program.json')
    if not os.path.exists(program_json_path):
        logger.error("program.json 不存在于: %s", program_json_path)
        return None

    return program_json_path


def print_usage():
    logger.info("用法: python3 get_latest_program_json.py <output_path>")
    logger.info("")
    logger.info("参数说明:")
    logger.info("  output_path: output 目录路径")


def main():
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)

    output_path = sys.argv[1]
    output_path = os.path.abspath(output_path)

    valid, error_msg = validate_path(output_path, "output 目录")
    if not valid:
        logger.info(error_msg)
        sys.exit(1)

    logger.info("Output 路径: %s", output_path)

    program_json_path = get_latest_program_json(output_path)

    if program_json_path:
        logger.info(program_json_path)
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == '__main__':
    main()
