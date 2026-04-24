#!/usr/bin/env python3
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

import os
import sys
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (
    find_aicore_entry_h,
    read_file,
    write_file,
    setup_logging,
    validate_path,
    comment_lines_by_range,
    uncomment_lines_by_range
)

setup_logging()

logger = logging.getLogger(__name__)


def find_callsubfunctask_range(lines):
    callsub_idx = -1
    for i, line in enumerate(lines):
        if 'CallSubFuncTask' in line:
            callsub_idx = i
            break

    if callsub_idx == -1:
        logger.info("错误：未找到 CallSubFuncTask")
        return None, None

    end_idx = callsub_idx
    for i in range(callsub_idx, len(lines)):
        if ');' in lines[i]:
            end_idx = i
            break

    start_idx = callsub_idx
    for i in range(callsub_idx, -1, -1):
        if '#if ENABLE_AICORE_PRINT' in lines[i]:
            start_idx = i
            break

    return start_idx, end_idx


def comment_callsubfunctask(file_path):
    lines = read_file(file_path)
    start_idx, end_idx = find_callsubfunctask_range(lines)

    if start_idx is None or end_idx is None:
        return False

    comment_lines_by_range(lines, start_idx, end_idx)
    write_file(file_path, lines)
    logger.info("成功注释 CallSubFuncTask 部分（行 %d-%d）", start_idx + 1, end_idx + 1)
    return True


def uncomment_callsubfunctask(file_path):
    lines = read_file(file_path)
    start_idx, end_idx = find_callsubfunctask_range(lines)

    if start_idx is None or end_idx is None:
        return False

    uncomment_lines_by_range(lines, start_idx, end_idx)
    write_file(file_path, lines)
    logger.info("成功取消注释 CallSubFuncTask 部分（行 %d-%d）", start_idx + 1, end_idx + 1)
    return True


def print_usage():
    logger.info("用法: python3 modify_callsubfunctask.py <action> <pypto_path>")
    logger.info("")
    logger.info("参数说明:")
    logger.info("  action: comment 或 uncomment")
    logger.info("  pypto_path: pypto 项目根目录路径")


def main():
    if len(sys.argv) < 3:
        print_usage()
        sys.exit(1)

    action = sys.argv[1]
    pypto_path = sys.argv[2]
    pypto_path = os.path.abspath(pypto_path)

    valid, error_msg = validate_path(pypto_path, "pypto 路径")
    if not valid:
        logger.info(error_msg)
        sys.exit(1)

    file_path = find_aicore_entry_h(pypto_path, logger)
    if not file_path:
        sys.exit(1)

    logger.info("找到 aicore_entry.h: %s", file_path)

    if action == 'comment':
        success = comment_callsubfunctask(file_path)
    elif action == 'uncomment':
        success = uncomment_callsubfunctask(file_path)
    else:
        logger.info("错误：无效的操作，请使用 comment 或 uncomment")
        sys.exit(1)

    if not success:
        sys.exit(1)


if __name__ == '__main__':
    main()
