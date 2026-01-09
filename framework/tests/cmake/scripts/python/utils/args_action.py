#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Args处理辅助.
"""
import argparse
import subprocess
from typing import Sequence, Optional, Any, List
from pathlib import Path

class ArgsEnvDictAction(argparse.Action):
    """解析命令行参数传入的环境变量字段(env)
    """

    def __call__(self, parser, namespace, values, option_string=None):
        env_dict = getattr(namespace, self.dest, {}) or {}
        for item in values:
            k, v = item.split('=', 1)
            env_dict[k] = v
        setattr(namespace, self.dest, env_dict)


class ArgsGTestFilterListAction(argparse.Action):
    """解析命令行参数传入的 GTestFilter 字段
    """

    # 耗时最长的测试用例列表（按耗时降序排列）
    SLOW_TESTS = [
        "DeepSeekIndexerAttentionQuantUTest.4B_mtp_ut",                          # 189.66s
 	     "DynamicOpsTest.Cube",                                                  # 162.96s
 	     "DynamicQuantLightningIndexerPrologUtest.b4_s1_2_s2_64k",               # 161.16s
 	     "DecodeIndexerAttentionUtest.utest_decode_indexer_attention",           # 154.81s
 	     "DynamicPATest.dynamic_pa_low_lantency_manual_unroll",                  # 135.72s
 	     "FunctionTest.Test_quantMM",                                            # 93.55s
 	     "DynamicGenGatedScoreUtest.utest_gen_gated_score_dyn",                  # 81.77s
 	     "DynamicGenGatedScoreUtest.utest_gen_gated_score_plus_dyn",             # 41.65s
 	     "DynamicLightningIndexerPrologUtest.utest_lightning_indexer_prolog",    # 38.81s
 	     "AttentionPostUTest.b32_s1_nz_fp16_quant",                              # 36.92s
 	     "TestLightningIndexerUtest.lightning_indexer_b_4_s1_2_s2_64k_quant",    # 34.39s
 	     "DynamicPATest.dynamic_pa_low_lantency_unroll",                        # 33.52s
 	     "GenAttnUtTest.TestDynamicGenAttenTest_FP16_ut",                       # 32.61s
 	     "DynamicGatherSlcFlashAttnUtest.dsa_gather_slc_attn_bf16_b32_s4_int8",  # 32.60s
 	     "DynamicFunctionTest.TestLoopIfWithRank",                              # 32.10s
 	     "DynamicAttentionUtTest.dynamic_attention_low",                         # 31.30s
 	     "DynamicGatherSlcFlashAttnUtest.dsa_gather_slc_attn_bf16_b32_s4",      # 30.82s
 	     "DynamicAttentionUtTest.dynamic_attention_low_nz",                      # 30.72s
 	     "MlaPrologQuantV32UTest.b8_s64k2_pa_nd_bf16_quantB",                   # 27.19s
 	     "CmpAttnTopk.cmp_attn_with_topk_singleop_bf16",                        # 26.97s
 	     "TestCodegenDispatch.TestMoeDispatchMultipyExperts",                  # 23.73s
 	     "MlaPrologUTest.b16_s1_pa_nd_fp16_allquant",                          # 19.09s
 	     "CostModelTest.TestAttentionPostAccuracy3",                           # 18.62s
 	     "MlaPrologUtest.mla_ut_bf16_high_quant_smooth_nz_pa_bsnd",            # 16.95s
 	     "FunctionTest.low",                                                   # 16.18s
 	     "CostModelTest.TestFixedLatencyTasks",                               # 11.98s
 	     "CostModelTest.TestGlobalCalendar",                                 # 11.52s
 	     "TestCodegenDynPa.PaHighThroughputDviewLargeDynamicValidShape",     # 10.30s
 	     "DynamicOpsTest.OpsElementWise",                                    # 9.84s
 	     "FunctionTest.dynamic_page_attention_adds_single_single_out",       # 9.76s
 	     "DynamicFunctionTest.TestLoopWithRank",                             # 9.68s
 	     "CacheManagerUnitTest.test_page_attention",                         # 9.59s
 	     "FunctionTest.dynamic_page_attention_adds",                         # 9.59s
 	     "DyNsa.gateScore_mini_mtp",                                         # 9.17s
 	     "TestCodegenStaticUnderDyn.TestStaticFuncUnderDyn",                 # 8.56s
 	     "DyNsa.GenSlc_b1_s1_fp_6k1",                                        # 7.71s
 	     "DynamicFunctionTest.TestHybridLoopIf",                             # 7.59s
 	     "GraphTest.llama_1_1_1024_128",                                     # 7.17s
 	     "DynamicOpsTest.MatmulFP32FP32",                                   # 7.15s
 	     "CostModelTest.TestAttentionPostSimulationSchedule",               # 7.12s
 	     "CostModelTest.TestCoreMachineDeadlock"                            # 7.00s
 	    ]


    def __init__(self, option_strings: Sequence[str], dest: str, nargs: Optional[int] = None, **kwargs: Any) -> None:
        # 确保 nargs 至少为 1
        if nargs is None:
            nargs = '+'
        super().__init__(option_strings, dest, nargs=nargs, **kwargs)

    @staticmethod
    def parse_all_cases(binary: str) -> List[str]:
        """获取gtest ut测试用例，并将耗时测试排在前面
        """
        result = subprocess.run([binary, '--gtest_list_tests'], capture_output=True, text=True)
        cases = []
        current_suite = ""
        for line in result.stdout.split('\n'):
            line = line.rstrip()
            if not line or "GoogleTestVerification" in line:
                continue
            if line.endswith('.'):
                current_suite = line[:-1]
            elif line.startswith('  '):
                test_name = line.strip()
                cases.append(f"{current_suite}.{test_name}")

        # 重新排序：将耗时测试排在前面
        slow_tests_in_list = []
        regular_tests = []

        for test in cases:
            if test in ArgsGTestFilterListAction.SLOW_TESTS:
                slow_tests_in_list.append(test)
            else:
                regular_tests.append(test)

        # 保持耗时测试的原有顺序（已经是按耗时降序），并添加标记
        ordered_slow_tests = []
        for test in ArgsGTestFilterListAction.SLOW_TESTS:
            if test in slow_tests_in_list:
                ordered_slow_tests.append(test)

        # 返回重排序后的列表：耗时测试在前，普通测试在后
        return ordered_slow_tests + regular_tests

    def __call__(self, parser: argparse.ArgumentParser, namespace: argparse.Namespace, values: List[str],
                 option_string: Optional[str] = None) -> None:
        # 解析每个字符串，按冒号分隔并展平
        case_list = []

        target = getattr(namespace, 'target')
        if (len(values) == 1 and values[0] == "*"):
            case_list = self.parse_all_cases(target[0])
        else:
            for value in values:
                # 分割每个字符串，并过滤空字符串
                cases = [cs.strip() for cs in value.split(':') if cs.strip()]
                case_list.extend(cases)
        # 将结果设置到命名空间
        setattr(namespace, self.dest, case_list)
