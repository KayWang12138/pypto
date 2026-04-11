#!/usr/bin/env python3
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

import os
import sys
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (
    get_commentable_lines,
    comment_lines_by_indices,
    validate_path,
    setup_logging,
    print_error_info,
    backup_and_test
)

setup_logging()

logger = logging.getLogger(__name__)


def test_cce_file(cce_file, test_cmd, run_dir):
    logger.info(f"测试 CCE 文件: {cce_file}")

    def modify_func(cce_lines):
        commentable_lines = get_commentable_lines(cce_lines)
        logger.info(f"可注释的行数: {len(commentable_lines)}")

        if not commentable_lines:
            logger.info("错误：没有可注释的行")
            return None

        logger.info("注释所有可注释的行...")
        return comment_lines_by_indices(cce_lines.copy(), commentable_lines)

    error_exists, output, original_lines = backup_and_test(cce_file, test_cmd, run_dir, modify_func)

    if original_lines is None:
        return False

    if error_exists:
        print_error_info(output, logger)
        logger.info("结果: 注释所有行后仍有 error，此文件可能不是问题文件")
        return False
    else:
        logger.info("结果: 注释所有行后运行成功（无 error），此文件可能是问题文件")
        return True


def print_usage():
    logger.info("用法: python3 test_cce_file.py <cce_file> <test_cmd> <run_dir>")
    logger.info("")
    logger.info("参数说明:")
    logger.info("  cce_file: CCE 文件路径")
    logger.info("  test_cmd: 触发 aicore error 的测试命令")
    logger.info("  run_dir: 运行测试命令的目录路径")


def main():
    if len(sys.argv) < 4:
        print_usage()
        sys.exit(1)

    cce_file = sys.argv[1]
    test_cmd = sys.argv[2]
    run_dir = sys.argv[3]

    cce_file = os.path.abspath(cce_file)
    run_dir = os.path.abspath(run_dir)

    valid, error_msg = validate_path(cce_file, "CCE 文件")
    if not valid:
        logger.info(error_msg)
        sys.exit(1)

    valid, error_msg = validate_path(run_dir, "运行目录")
    if not valid:
        logger.info(error_msg)
        sys.exit(1)

    is_problem_file = test_cce_file(cce_file, test_cmd, run_dir)

    if is_problem_file:
        logger.info("\n此文件可能是问题文件")
        logger.info(cce_file)
    else:
        logger.info("\n此文件不是问题文件")


if __name__ == "__main__":
    main()
