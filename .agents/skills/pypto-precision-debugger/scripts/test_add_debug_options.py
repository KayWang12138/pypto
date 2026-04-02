#!/usr/bin/env python3
# Test script for add_debug_options.py

import sys
import os
import tempfile
import shutil

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from add_debug_options import add_debug_options_to_file, restore_from_backup

def test_patterns():
    """测试各种装饰器格式"""
    
    test_cases = [
        # Case 1: 单独的 @pypto.frontend.jit
        (
            "@pypto.frontend.jit\ndef my_kernel():\n    pass",
            "单独的 @pypto.frontend.jit"
        ),
        # Case 2: @pypto.frontend.jit() 空参数
        (
            "@pypto.frontend.jit()\ndef my_kernel():\n    pass",
            "@pypto.frontend.jit() 空参数"
        ),
        # Case 3: @pypto.frontend.jit(单行参数)
        (
            "@pypto.frontend.jit(inline=True)\ndef my_kernel():\n    pass",
            "@pypto.frontend.jit(单行参数)"
        ),
        # Case 4: @pypto.frontend.jit(多行参数)
        (
            "@pypto.frontend.jit(\n    inline=True,\n    debug=False\n)\ndef my_kernel():\n    pass",
            "@pypto.frontend.jit(多行参数)"
        ),
    ]
    
    for i, (content, description) in enumerate(test_cases, 1):
        print(f"\n测试用例 {i}: {description}")
        
        # 创建临时文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(content)
            temp_file = f.name
        
        try:
            # 添加 debug_options
            modified, backup_path = add_debug_options_to_file(temp_file)
            
            if modified:
                with open(temp_file, 'r') as f:
                    result = f.read()
                print("✓ 修改成功")
                print("修改后的内容:")
                print(result)
                
                # 验证恢复
                if restore_from_backup(temp_file):
                    with open(temp_file, 'r') as f:
                        restored = f.read()
                    if restored == content:
                        print("✓ 恢复成功")
                    else:
                        print("✗ 恢复失败：内容不匹配")
                else:
                    print("✗ 恢复失败")
            else:
                print("✗ 修改失败")
        finally:
            # 清理临时文件
            if os.path.exists(temp_file):
                os.remove(temp_file)
            backup_file = temp_file + ".backup"
            if os.path.exists(backup_file):
                os.remove(backup_file)

if __name__ == '__main__':
    test_patterns()
