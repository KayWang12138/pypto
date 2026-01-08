#!/usr/bin/env python3
# coding: utf-8
"""
测试 gtest_reorder.py 脚本的功能
"""

import subprocess
import sys
import tempfile
import os

def create_mock_gtest():
    """创建一个模拟的gtest可执行文件"""
    mock_script = '''#!/usr/bin/env python3
import sys

def list_tests():
    """模拟 --gtest_list_tests 输出"""
    print("TestSuite1.")
    print("  normal_test_1")
    print("  normal_test_2")
    print("DeepSeekIndexerAttentionQuantUTest.")
    print("  4B_mtp_ut")
    print("DynamicQuantLightningIndexerPrologUtest.")
    print("  b4_s1_2_s2_64k")
    print("TestSuite2.")
    print("  another_normal_test")
    print("DynamicOpsTest.")
    print("  Cube")

if __name__ == "__main__":
    if "--gtest_list_tests" in sys.argv:
        list_tests()
    else:
        print("Mock gtest executed with args:", sys.argv[1:])
'''

    # 创建临时文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(mock_script)
        temp_file = f.name

    # 给执行权限
    os.chmod(temp_file, 0o755)

    return temp_file

def test_reorder():
    """测试重排序功能"""
    print("创建模拟的gtest可执行文件...")
    mock_gtest = create_mock_gtest()

    try:
        print("\n1. 原始 gtest 输出:")
        print("-" * 40)
        result1 = subprocess.run([mock_gtest, '--gtest_list_tests'],
                               capture_output=True, text=True)
        print(result1.stdout.strip())

        print("\n2. 使用 gtest_reorder.py 后的输出:")
        print("-" * 40)
        result2 = subprocess.run([sys.executable, 'gtest_reorder.py', mock_gtest, '--gtest_list_tests'],
                               capture_output=True, text=True)
        print(result2.stdout.strip())

        print("\n3. 验证重排序是否正确:")
        print("-" * 40)

        # 检查耗时测试是否排在前面
        output_lines = result2.stdout.strip().split('\n')

        slow_tests_found = []
        in_slow_section = True

        for line in output_lines:
            line = line.strip()
            if not line:
                continue

            if not line.startswith(' '):  # 测试套件名
                if any(slow_test in line for slow_test in [
                    'DeepSeekIndexerAttentionQuantUTest.',
                    'DynamicQuantLightningIndexerPrologUtest.',
                    'DynamicOpsTest.'
                ]):
                    in_slow_section = True
                else:
                    in_slow_section = False
            elif line.startswith('  ') and '[SLOW_TEST]' in line:  # 耗时测试用例
                if in_slow_section:
                    test_name = line.split()[0]  # 取测试名
                    slow_tests_found.append(test_name)

        print(f"找到 {len(slow_tests_found)} 个标记为耗时测试的用例")
        for test in slow_tests_found:
            print(f"  ✓ {test} [SLOW_TEST]")

        if len(slow_tests_found) == 3:
            print("\n✓ 测试通过：耗时测试已正确排在前面并标记")
        else:
            print(f"\n✗ 测试失败：期望3个耗时测试，实际找到{len(slow_tests_found)}个")

    finally:
        # 清理临时文件
        os.unlink(mock_gtest)

if __name__ == "__main__":
    test_reorder()
