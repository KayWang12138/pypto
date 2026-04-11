#!/usr/bin/env python3
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

import os
import sys
import re
import shutil
import logging

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


def write_file(file_path, content):
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)


def add_debug_options_to_file(file_path):
    content = read_file(file_path)
    original_content = content
    
    if 'runtime_debug_mode' in content:
        return False, None
    
    patterns = [
        (r'@pypto\.frontend\.jit\s*\n', 
         '@pypto.frontend.jit(\n    debug_options={"runtime_debug_mode": 1}\n)\n'),
        (r'@pypto\.frontend\.jit\(\)\s*\n', 
         '@pypto.frontend.jit(\n    debug_options={"runtime_debug_mode": 1}\n)\n'),
    ]
    
    for pattern, replacement in patterns:
        content, count = re.subn(pattern, replacement, content)
        if count > 0:
            break
    
    if count == 0:
        pattern = r'@pypto\.frontend\.jit\(([^)]+)\)\s*\n'
        match = re.search(pattern, content)
        if match:
            params = match.group(1).strip()
            if '\n' in match.group(1):
                replacement = (
                    f'@pypto.frontend.jit(\n'
                    f'    debug_options={{"runtime_debug_mode": 1}},\n'
                    f'{match.group(1)})\n'
                )
            else:
                replacement = f'@pypto.frontend.jit(\n    debug_options={{"runtime_debug_mode": 1}},\n    {params}\n)\n'
            content = re.sub(pattern, replacement, content)
            count = 1
    
    if content != original_content:
        backup_path = f"{file_path}.backup"
        shutil.copy2(file_path, backup_path)
        
        write_file(file_path, content)
        return True, backup_path
    
    return False, None


def restore_from_backup(test_file_path):
    backup_path = f"{test_file_path}.backup"
    
    if not os.path.exists(backup_path):
        return False
    
    shutil.copy2(backup_path, test_file_path)
    
    os.remove(backup_path)
    
    return True


def print_usage():
    logger.info(f"Usage: {sys.executable} add_debug_options.py <command> <file_path>")
    logger.info("Commands:")
    logger.info("  add <file_path>     Add debug options to test file")
    logger.info("  restore <file_path> Restore test file from backup")


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
            sys.exit(1)
                
        modified, backup_path = add_debug_options_to_file(test_file_path)
        
        if modified:
            sys.exit(0)
        else:
            sys.exit(1)
    
    elif command == 'restore':
        if len(sys.argv) < 3:
            print_usage()
            sys.exit(1)
        
        test_file_path = os.path.abspath(sys.argv[2])
        
        if not os.path.exists(test_file_path):
            sys.exit(1)
        
        if restore_from_backup(test_file_path):
            sys.exit(0)
        else:
            sys.exit(1)
    
    else:
        print_usage()
        sys.exit(1)


if __name__ == '__main__':
    main()
