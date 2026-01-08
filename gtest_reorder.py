#!/usr/bin/env python3
# coding: utf-8
"""
gtest 测试用例重排序工具

将指定的耗时测试用例排在 --gtest_list_tests 输出列表的最前面
"""

import subprocess
import sys
from typing import List

# 耗时最长的测试用例列表（按耗时降序排列）
SLOW_TESTS = [
    "DeepSeekIndexerAttentionQuantUTest.4B_mtp_ut",
    "DynamicQuantLightningIndexerPrologUtest.b4_s1_2_s2_64k",
    "DynamicOpsTest.Cube",
    "DecodeIndexerAttentionUtest.utest_decode_indexer_attention",
    "DynamicPATest.dynamic_pa_low_lantency_manual_unroll",
    "FunctionTest.Test_quantMM",
    "DynamicGenGatedScoreUtest.utest_gen_gated_score_dyn",
    "DynamicGenGatedScoreUtest.utest_gen_gated_score_plus_dyn",
    "DynamicLightningIndexerPrologUtest.utest_lightning_indexer_prolog",
    "AttentionPostUTest.b32_s1_nz_fp16_quant",
    "TestLightningIndexerUtest.lightning_indexer_b_4_s1_2_s2_64k_quant",
    "DynamicAttentionUtTest.dynamic_attention_low_nz",
    "GenAttnUtTest.TestDynamicGenAttenTest_FP16_ut",
    "DynamicAttentionUtTest.dynamic_attention_low",
    "DynamicGatherSlcFlashAttnUtest.dsa_gather_slc_attn_bf16_b32_s4_int8",
    "DynamicGatherSlcFlashAttnUtest.dsa_gather_slc_attn_bf16_b32_s4"
]

def get_test_list(test_executable: str) -> List[str]:
    """获取gtest的原始测试列表"""
    try:
        cmd = [test_executable, '--gtest_list_tests']
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

        if result.returncode != 0:
            print(f"错误: 执行gtest_list_tests失败: {result.stderr}", file=sys.stderr)
            return []

        # 解析gtest输出
        lines = result.stdout.strip().split('\n')
        test_suites = []
        current_suite = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if not line.startswith(' '):  # 测试套件名
                current_suite = line.rstrip('.')  # 移除末尾的点
            elif line.startswith('  ') and current_suite:  # 测试用例名
                test_name = line.strip()
                full_name = f"{current_suite}.{test_name}"
                test_suites.append(full_name)

        return test_suites

    except subprocess.TimeoutExpired:
        print("错误: gtest_list_tests 执行超时", file=sys.stderr)
        return []
    except Exception as e:
        print(f"错误: 获取测试列表失败: {e}", file=sys.stderr)
        return []

def reorder_test_list(test_list: List[str]) -> List[str]:
    """重新排序测试列表，将耗时测试放在前面"""
    # 分离耗时测试和普通测试
    slow_tests_in_list = []
    regular_tests = []

    for test in test_list:
        if test in SLOW_TESTS:
            slow_tests_in_list.append(test)
        else:
            regular_tests.append(test)

    # 保持耗时测试的原有顺序（已经是按耗时降序）
    # 并添加标记
    ordered_slow_tests = []
    for test in SLOW_TESTS:
        if test in slow_tests_in_list:
            ordered_slow_tests.append(f"{test}  [SLOW_TEST]")

    # 普通测试保持原顺序
    return ordered_slow_tests + regular_tests

def print_reordered_list(test_list: List[str]):
    """打印重新排序的测试列表（gtest格式）"""
    # 按照gtest的输出格式重新组织
    suite_groups = {}

    for test in test_list:
        # 移除标记（如果有）
        clean_test = test.replace("  [SLOW_TEST]", "")
        if '.' in clean_test:
            suite_name, case_name = clean_test.rsplit('.', 1)
            if suite_name not in suite_groups:
                suite_groups[suite_name] = []
            marker = "  [SLOW_TEST]" if "[SLOW_TEST]" in test else ""
            suite_groups[suite_name].append(f"  {case_name}{marker}")

    # 输出按gtest格式
    for suite_name in sorted(suite_groups.keys()):
        print(f"{suite_name}.")
        for case in suite_groups[suite_name]:
            print(case)

def main():
    if len(sys.argv) < 2:
        print("用法: python gtest_reorder.py <gtest可执行文件> [--gtest_list_tests ...]",
              file=sys.stderr)
        sys.exit(1)

    test_executable = sys.argv[1]

    # 检查是否是list tests命令
    if '--gtest_list_tests' in sys.argv:
        # 获取原始测试列表
        test_list = get_test_list(test_executable)
        if not test_list:
            sys.exit(1)

        # 重新排序
        reordered_list = reorder_test_list(test_list)

        # 输出重新排序的列表
        print_reordered_list(reordered_list)
    else:
        # 执行原始的gtest命令
        cmd = [test_executable] + sys.argv[2:]
        result = subprocess.run(cmd)
        sys.exit(result.returncode)

if __name__ == "__main__":
    main()
