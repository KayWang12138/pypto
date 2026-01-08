#!/usr/bin/env python3
# coding: utf-8
"""
测试修改后的args_action.py是否正确重排序耗时测试
"""

import sys
import os
import tempfile
import subprocess

# 添加脚本路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'framework/tests/cmake/scripts/python'))

from utils.args_action import ArgsGTestFilterListAction

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

        print("\n2. 使用 ArgsGTestFilterListAction.parse_all_cases 后的输出:")
        print("-" * 40)
        reordered_cases = ArgsGTestFilterListAction.parse_all_cases(mock_gtest)
        print("重排序后的测试用例列表:")
        for i, case in enumerate(reordered_cases, 1):
            is_slow = case in ArgsGTestFilterListAction.SLOW_TESTS
            marker = " [SLOW_TEST]" if is_slow else ""
            print("2d")

        print("\n3. 验证重排序是否正确:")
        print("-" * 40)

        # 检查前3个是否都是耗时测试
        slow_count = 0
        for case in reordered_cases[:3]:
            if case in ArgsGTestFilterListAction.SLOW_TESTS:
                slow_count += 1

        print(f"前3个测试中有 {slow_count} 个是耗时测试")

        # 检查耗时测试是否都排在前面
        slow_tests_in_order = [case for case in reordered_cases if case in ArgsGTestFilterListAction.SLOW_TESTS]
        expected_order = [case for case in ArgsGTestFilterListAction.SLOW_TESTS if case in reordered_cases]

        if slow_tests_in_order == expected_order:
            print("✓ 耗时测试顺序正确")
        else:
            print("✗ 耗时测试顺序不正确")
            print(f"期望顺序: {expected_order}")
            print(f"实际顺序: {slow_tests_in_order}")

        # 统计信息
        total_cases = len(reordered_cases)
        slow_cases = len(slow_tests_in_order)
        normal_cases = total_cases - slow_cases

        print(f"\n统计信息:")
        print(f"  总测试用例: {total_cases}")
        print(f"  耗时测试: {slow_cases}")
        print(f"  普通测试: {normal_cases}")

        if slow_count >= 3:
            print("\n✓ 测试通过：耗时测试已正确排在前面")
        else:
            print(f"\n✗ 测试失败：前3个测试中只有{slow_count}个耗时测试")

    finally:
        # 清理临时文件
        os.unlink(mock_gtest)

if __name__ == "__main__":
    test_reorder()
