# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

set(_CaseFilterList
        #===========================================================================
        # mla_prolog cost
        #===========================================================================
        #"MlaPrologV2OnBoardCostTest.test_mla_bf16_low_quant_smooth"
        #"MlaPrologV2OnBoardCostTest.test_mla_bf16_high_quant_smooth"
        #"MlaPrologOnBoardCostTest.test_MlaProlog_float16_32_128_1_4096_7168_1536_quant"
        #"MlaPrologOnBoardCostTest.test_MlaProlog_float16_32_128_1_4096_7168_1536_quant"
        #"MlaPrologOnBoardCostTest.test_MlaProlog_float16_32_128_1_4096_7168_1536_quant"

        # MLA
        #===========================================================================
        # "DyMla.high_PA_NZ"
        "DyMla.low_PA_BSND"

        # NSA MlaProlog
        # "MlaPrologSTest.b32_s1_pa_nz_bf16_quant"
        "MlaPrologSTest.b32_s2_pa_nz_bf16_quant"

        # NSA AttentionPost
        # "AttentionPostSTest.b32_s1_nz_bf16_quant"
        "AttentionPostSTest.b32_s2_nz_bf16_quant"

        # Flash Attention
        #===========================================================================
        "DynamicBasicTest.TestDD"
        "DynamicBasicTest.DynamicRawShape"
        "DynamicBasicTest.TestStaticUnderDynDev"
        "DynamicBasicTest.TestStaticLoop"
        "DynamicBasicTest.TestInnerLoopOrder"
        "DynamicBasicTest.TestDeviceMachineOnModel"
        "DynamicBasicTest.TestDeviceMachineBlockdimOnBoard"
        # "DynamicPATest.dynamic_pa_low_lantency"
        # "DynamicPATest.dynamic_pa_low_lantency_unroll"
        # "DynamicPATest.dynamic_pa_low_lantency_manual_unroll"
        "DynamicPATest.dynamic_pa_high_throughput_dview_large"
        "DynamicPATest.dynamic_pa_high_throughput_dview_large_dyn_unalign"
        "DynamicPATest.dynamic_pa_high_throughput_only_batch_loop"
        "DynamicPATest.dynamic_pa_noflash_unalign"
        #"DynamicPATest.dynamic_pa_noflash"
        "DynamicPATest.dynamic_pa_low_lantency_dyn_valid_shape"
        "DynamicPATest.dynamic_pa_low_lantency_dyn_unalign"
        "DynamicUnalignTest.test_unary_unalign"
        "DynamicUnalignTest.test_mm_unalign"
        "DynamicUnalignTest.test_mm2_unalign"
        "DynamicUnalignTest.test_rowmaxsingle_unalign"
        "DynamicUnalignTest.test_rowsumsingle_unalign"
        "DynamicExpandTest.TestDynamicExpandUnalign"
        "DynamicBinTest.testDynMulsUnalign"
        "DynamicBinTest.testScalarDivsUnalign"
        "DynamicCastTest.testDynCastUnalign"
        "DynamicBinTest.TestDynamicAddUnalign"
        "DynamicTransposeTest.TestDynamicVnchwconv"
        "DynamicGatherTest.TestDynamicGatherDim2"
        "DynamicGatherTest.TestDynamicGatherDim3"
        "DynamicDatamoveTest.TestDynamicDatamove"
        "DynamicBrcTest.TestDynamicMulBrcUnalign"
        "DynamicSlcTest.dynamic_p_slc_bf16"
        "DynamicSlcTest.dynamic_p_slc_fp16"

        #===========================================================================
        # NSA native spare attention
        #===========================================================================
        "TestGenAtten.TestOnboardGenAttenTest_FP16_S1"

        # PaPost
        #===========================================================================
        #"DynamicAttentionPostTest.dynamic_pa_post_new_mm5nz_splitk"
        # "DynamicAttentionPostTest.dynamic_pa_post_new_mm5nz_splitk_low"
        # "DynamicAttentionPostTest.dynamic_pa_papost_bf16_b48"

        # Attention

        # NSA
        "DynamicSATest.slc_attn_fp16"
        "DynamicSATest.slc_attn_bf16"
        "DynamicSATest.slc_attn_mtp_s1_2_fp16"

        #===========================================================================
        # AstApi
        #===========================================================================
        # "OnBoardTestAstApi.test_fa_all2all_ast_api_mode"
        # "OnBoardTestAstApi.test_add_sub_all2all_torchapi_multi_function"

        #===========================================================================
        # DeepSeekV3 'MOE 子图/整图' 精度用例
        #===========================================================================
        # "MoegateOnBoardTest.test_moegate_graph3_case1"
        # "MoegateOnBoardTest.test_moegate_graph3_case2_32_1_7168"
        # "MoegateOnBoardTest.test_moegate_graph3_graph4_case_32_1_7168"
        # "MoEGateOnBoardTest.test_operation_b_4"
        "MoEGateOnBoardTest.test_operation_b_16"
        "MoEGateOnBoardTest.test_operation_b_32"

        #===========================================================================
        # '单 Op'精度用例
        #===========================================================================
        # Scatterupdate
        "ScatterupdateOnBoardTest.test_scatter_update_1_1_16_16"
        "ScatterupdateOnBoardTest.test_scatter_update_1_1_20_20"
        "ScatterupdateOnBoardTest.test_scatter_update_64_7168_64_moe"
        #"ScatterupdateOnBoardTest.test_scatter_update_2_1_512_576"
        #"ScatterupdateOnBoardTest.test_scatter_update_2_1_512_576_multi_row3"
        #"ScatterupdateOnBoardTest.test_scatter_update_2_1_512_576_graphD"
        "ScatterupdateOnBoardTest.test_scatter_update_32_1_512_576_graphD_bf16"
        #"ScatterupdateOnBoardTest.test_scatter_update_1_1_16_16_exp"

        # GatherElement
        #"GatherElementOnBoardTest.test_gather_element_float_16_64_7_32_1"
        #"GatherElementOnBoardTest.test_gather_element_float_16_64_7_32_0"
        #"ScatterElementOnBoardTest.test_scatter_element_float_8_256_8_8_1_moe"
)

set(_Target ${TileFwkSTestNamePrefix}_runtime)

file(GLOB_RECURSE _Sources src/*.cpp)

TileFwk_STest_AddLib(
        TARGET ${_Target}
        SOURCES ${_Sources}
        PUBLIC_LINK_LIBRARIES tile_fwk_runtime tile_fwk_interface
        GTEST_FILTER_LIST ${_CaseFilterList}
        LD_LIBRARIES_EXT tile_fwk_runtime
        GOLDEN_SCRIPT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/src/ops
)
