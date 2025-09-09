# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

set(_CaseFilterList
        # 整图用例, 绘图
        "GraphTest.llama_1_1_128_128"
        "GraphTest.llama_1_1_256_128"
        "GraphTest.llama_1_1_512_128"
        "GraphTest.llama_1_1_256_128_mix"
        "GraphTest.llama_1_1_1024_128"
        "GraphTest.Test_MlaPrologV2_bfloat16_4_32_1_256_7168_1536"
        "GraphTest.Test_MlaPrologV2_bfloat16_4_32_1_256_7168_1536_splitnz"
        "GraphTest.test_mla_bf16_low_quant_smooth"
        "GraphTest.deepseek_qkvPre"
        "GraphTest.TestAttentionPost"
        "GraphTest.Test_deepseekAttention_s_1"
        "GraphTest.Test_deepseekAttention_pre"
        "GraphTest.test_operation_rope_subgraph_deepseekv3_bf16"
        "GraphTest.test_attention_bf16_4_1024_1024_32_256"
        "GraphTest.TestTranspose_MLA_3D_2_add"
        "GraphTest.TestTranspose_MLA_3D_2_reshape"
        "FunctionTest.dynamic_pa_low_lantency"
        "FunctionTest.low"
        "FunctionTest.low_PAND"
        "FunctionTest.dynamic_prolog_post_low_lantency"
        "FunctionTest.dynamic_page_attention_adds"
        "FunctionTest.dynamic_page_attention_adds_single_single_out"
        # NSA MlaProlog
        "MlaPrologUTest.b16_s1_pa_nd_fp16_allquant"
        # NSA AttentionPost
        "AttentionPostUTest.b32_s1_nz_fp16_quant"
        "AttentionPostUTest.b32_s2_nz_bf16_quant_all"
        # NSA
        "NSAUtest.nsa_b_16_fp16"
        "NSAUtest.nsa_b_16_fp16_debug"
        "SlcAttnUtest.slc_attn_fp16"
        "KvSlcAttnUtest.kv_slc_attn_ut"
        "KvCmpUtest.kv_compress_ut"

        # 前端表达'单 Op'用例
        "FunctionTest.TestAddTensorFunctionDim4"
        "FunctionTest.TestAddTensorFunctionDim2"
        "FunctionTest.TestOperationRopeV2Deepseekv3B32"
        "FunctionTest.test_fa_new"
        "FunctionTest.TestSubTensorFunctionDim2"
        "FunctionTest.TestMulTensorFunctionDim2"
        "FunctionTest.TestDivTensorFunctionDim2"
        "FunctionTest.TestAddScalarFunctionDim2"
        "FunctionTest.TestAddScalarFunctionDim3"
        "FunctionTest.TestSubScalarFunctionDim2"
        "FunctionTest.TestMulScalarFunctionDim2"
        "FunctionTest.TestDivScalarFunctionDim2"
        "FunctionTest.TestExpTensorFunctionDim2"
        "FunctionTest.TestSin"
        "FunctionTest.TestCos"
        "FunctionTest.TestGatherAxis0Indices1"
        "FunctionTest.TestGatherAxis0Indices2"
        "FunctionTest.TestGatherElementAxis1Indices2"
        "FunctionTest.TestGatherElementAxis0Indices2"
        "FunctionTest.TestScatterElement"
        "FunctionTest.TestScatterUpdate2"
        "FunctionTest.testRowSumSingle"
        "FunctionTest.testRowMaxSingle"
        "FunctionTest.testSoftmax"
        "FunctionTest.TestRoPE"
        "FunctionTest.TestRoPEDeepseekV3"
        "FunctionTest.testRmsNormNewMultiDims"
        "FunctionTest.TestConcat"
        # "FunctionTest.TestAttention"
        "FunctionTest.TestAttentionPost"
        "FunctionTest.Test_qkvPre"
        "FunctionTest.Test_qkvPre2"
        "FunctionTest.Test_deepseekAttention_pre"
        "FunctionTest.TestBMMtest"
        "FunctionTest.TestBMMtest2"
        "FunctionTest.Test_deepseekMoEGate"
        "FunctionTest.Test_quant"
        "FunctionTest.Test_ScalarOp"
        "FunctionTest.TestPad"
        "FunctionTest.Test_quantMM"
        "FunctionTest.TestRmsNorm"
        "OperationImplTest.Test_MatmulWithSplitK"
        "OperationImplTest.Test_MatmulWithSplitKWithTrans"
        "OperationImplTest.test_BMMT_NZ_1_128_256_128_Batch"
        "GraphTest.TestTailBlock"
        "MlaPrologUtest.mla_ut_bf16_high_quant_smooth_nz_pa_bsnd"
        "DyNsa.gateScore_mini_mtp"
        "DyNsa.GenTopk_b1_s1_fp_6k1_dyn"
        "DyNsa.GenSlc_b1_s1_fp_6k1"

        # PA
        "DynamicPATest.TestDD"
        "DynamicPATest.dynamic_pa_low_lantency_unroll"
        "DynamicPATest.dynamic_pa_low_lantency_manual_unroll"
        "DynamicPATest.dynamic_pa_high_throughput_only_batch_loop"
        "DynamicPATest.AllocatorCoverage"
        "DynamicPATest.dynamic_pa_low_lantency_pass_unroll"

        #
        "CmpAttnTopk.cmp_attn_with_topk_singleop_bf16"
        "DynamicAttentionUtTest.dynamic_attention_low"
        "DynamicAttentionUtTest.dynamic_attention_low_nz"
        "DynamicKvSlcUtTest.test_kv_slc"
        "GenAttnUtTest.TestDynamicGenAttenTest_FP16_ut"
        "DynamicTestWinAttenUt.TestOnboardWinAttnTest_FP16_Test0"
        "DynamicTestWinAttenFlashUt.TestOnboardWinAttnTest_FP16_Test1"
        "DynamicTestWinAttenDebugUt.TestOnboardWinAttnTest_FP16"
        # Softmax
        "DynamicSoftmaxUtest.utest_softmax_dyn"

)

set(_Target ${PTO_Fwk_UTestNamePrefix}_operator)

file(GLOB_RECURSE _Sources src/*.cpp)

PTO_Fwk_UTest_AddCaseLib(
        TARGET ${_Target}
        SOURCES ${_Sources}
        PUBLIC_LINK_LIBRARIES tile_fwk_passes tile_fwk_interface tile_fwk_operator
        GTEST_FILTER_LIST ${_CaseFilterList}
)
