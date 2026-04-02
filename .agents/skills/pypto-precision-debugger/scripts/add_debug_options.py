#!/usr/bin/env python3
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

import os
import sys
import re
import logging
import shutil

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import setup_logging

setup_logging()

logger = logging.getLogger(__name__)


def add_debug_options_to_file(file_path):
    """向测试文件添加 debug_options"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        original_content = content
    
    # 检查是否已经存在 runtime_debug_mode
    if 'runtime_debug_mode' in content:
        logger.info("文件已包含 runtime_debug_mode，无需修改: %s", file_path)
        return False, None
    
    # 模式1: @pypto.frontend.jit 单独一行，后面没有参数
    pattern1 = r'@pypto\.frontend\.jit\s*\n'
    replacement1 = '@pypto.frontend.jit(\n    debug_options={"runtime_debug_mode": 1}\n)\n'
    content, count1 = re.subn(pattern1, replacement1, content)
    
    if count1 > 0:
        logger.info("模式1匹配: 单独的 @pypto.frontend.jit")
    else:
        # 模式2: @pypto.frontend.jit(...) 多行参数
        # 匹配 @pypto.frontend.jit( 后面的内容，直到对应的 )
        pattern2 = r'@pypto\.frontend\.jit\(\s*\n'
        
        def add_debug_option_to_multiline(match):
            # 在多行参数的开头添加 debug_options
            return '@pypto.frontend.jit(\n    debug_options={"runtime_debug_mode": 1},\n'
        
        content, count2 = re.subn(pattern2, add_debug_option_to_multiline, content)
        
        if count2 > 0:
            logger.info("模式2匹配: @pypto.frontend.jit(...) 多行参数")
        else:
            # 模式3: @pypto.frontend.jit() 空参数
            pattern3 = r'@pypto\.frontend\.jit\(\)\s*\n'
            replacement3 = '@pypto.frontend.jit(\n    debug_options={"runtime_debug_mode": 1}\n)\n'
            content, count3 = re.subn(pattern3, replacement3, content)
            
            if count3 > 0:
                logger.info("模式3匹配: @pypto.frontend.jit() 空参数")
            else:
                # 模式4: @pypto.frontend.jit(单行参数)
                pattern4 = r'@pypto\.frontend\.jit\(([^)]+)\)\s*\n'
                
                def add_debug_option_to_inline(match):
                    params = match.group(1).strip()
                    return f'@pypto.frontend.jit(\n    debug_options={{"runtime_debug_mode": 1}},\n    {params}\n)\n'
                
                content, count4 = re.subn(pattern4, add_debug_option_to_inline, content)
                
                if count4 > 0:
                    logger.info("模式4匹配: @pypto.frontend.jit(单行参数)")
    
    if content != original_content:
        # 创建备份（在测试目录，不带时间戳）
        backup_path = f"{file_path}.backup"
        shutil.copy2(file_path, backup_path)
        logger.info("已创建备份: %s", backup_path)
        
        # 写入修改后的内容
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        logger.info("已添加 debug_options 到文件: %s", file_path)
        return True, backup_path
    else:
        logger.warning("未找到需要修改的 @pypto.frontend.jit 装饰器: %s", file_path)
        return False, None


def restore_from_backup(test_file_path):
    """从备份恢复文件并删除备份"""
    backup_path = f"{test_file_path}.backup"
    
    if not os.path.exists(backup_path):
        logger.warning("备份文件不存在: %s", backup_path)
        return False
    
    # 恢复原始文件
    shutil.copy2(backup_path, test_file_path)
    logger.info("已从备份恢复: %s", test_file_path)
    
    # 删除备份文件
    os.remove(backup_path)
    logger.info("已删除备份: %s", backup_path)
    
    return True


def print_usage():
    logger.info("用法:")
    logger.info("  添加 debug_options:")
    logger.info("    python3 add_debug_options.py add <test_file_path>")
    logger.info("")
    logger.info("  从备份恢复并删除备份:")
    logger.info("    python3 add_debug_options.py restore <test_file_path>")
    logger.info("")
    logger.info("参数说明:")
    logger.info("  test_file_path: 测试用例文件路径（如 test_operator.py）")


def main():
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == 'add':
        if len(sys.argv) < 3:
            print_usage()
            sys.exit(1)
        
        test_file_path = os.path.abspath(sys.argv[2])
        
        if not os.path.exists(test_file_path):
            logger.error("测试文件不存在: %s", test_file_path)
            sys.exit(1)
        
        logger.info("测试文件路径: %s", test_file_path)
        
        modified, backup_path = add_debug_options_to_file(test_file_path)
        
        if modified:
            logger.info("")
            logger.info("=" * 60)
            logger.info("修改完成！")
            logger.info("备份文件: %s", backup_path)
            logger.info("如需恢复，请运行:")
            logger.info("  python3 add_debug_options.py restore %s", test_file_path)
            logger.info("=" * 60)
            sys.exit(0)
        else:
            sys.exit(1)
    
    elif command == 'restore':
        if len(sys.argv) < 3:
            print_usage()
            sys.exit(1)
        
        test_file_path = os.path.abspath(sys.argv[2])
        
        if not os.path.exists(test_file_path):
            logger.error("测试文件不存在: %s", test_file_path)
            sys.exit(1)
        
        if restore_from_backup(test_file_path):
            logger.info("")
            logger.info("=" * 60)
            logger.info("恢复成功！已删除备份文件")
            logger.info("=" * 60)
            sys.exit(0)
        else:
            logger.error("恢复失败！")
            sys.exit(1)
    
    else:
        logger.error("未知命令: %s", command)
        print_usage()
        sys.exit(1)


if __name__ == '__main__':
    main()
