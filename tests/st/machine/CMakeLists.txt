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
        "DynamicTest.TestPartial"

        # Flash Attention
        #===========================================================================
        "DynamicBasicTest.TestDD"
        "DynamicBasicTest.TestCheckPointRestore"
        "DynamicBasicTest.TestSlotId"
        "DynamicBasicTest.DynamicRawShape"
        "DynamicBasicTest.TestStaticUnderDynDev"
        "DynamicBasicTest.TestStaticLoop"
        "DynamicBasicTest.TestInnerLoopOrder"
        "DynamicBasicTest.TestDeviceMachineOnModel"
        "DynamicBasicTest.TestDeviceMachineBlockdimOnBoard"
        "DynamicBasicTest.TestTensorExtract"
        "DynamicBasicTest.TestGetTensorData"
        "DynamicBasicTest.TestGetTensorDataCrossFunction"
        "DynamicBasicTest.TestGetTensorDataUnalign"
        "DynamicBasicTest.TestGetTensorDataExpr"
        "DynamicBasicTest.TestVectorDup"
        "DynamicBasicTest.TestTensorInsert"
        "DynamicBasicTest.TestSetTensorData"
        "DynamicBasicTest.TestSetTensorDataExpr"
        "DynamicBasicTest.TestGetTensorDataAndDup"
        "DynamicBasicTest.TestGetAndSetTensorDataExpr"
        "DynamicBasicTest.TestLoopIfWithRank456"
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

        "DynamicBindingTest.TestDefaultCompute"
        "DynamicBindingTest.TestDeviceRunDataFromHost"
        "DynamicBindingTest.TestDeviceCompute"

        #===========================================================================
        # NSA native spare attention
        #===========================================================================
        "TestGenAtten.TestDynamicGenAttenTest_B_16_S1_1_FP16"
        "DynamicSlcTest.dynamic_p_slc_bf16"
        "DynamicSlcTest.dynamic_p_slc_fp16"
        "DynamicWinAttenTest.test_DynAttn_nas_win_attn_s1_2_actseqlen_1024_mla_fp16_v1"

        # KV Compress
        "DynKVCmp.KVCmpBatch48float16"
        "DynKVCmp.KVCmpBatch32bf16"

        # NSA gatescore && compress attention part2
        # "DyNsa.gateScore_mini_mtp"
        "DyNsa.GateScore_b32_s2_bf"
        "DyNsa.GateScore_b48_s1_fp"

        # PaPost
        #===========================================================================
        #"DynamicAttentionPostTest.dynamic_pa_post_new_mm5nz_splitk"
        # "DynamicAttentionPostTest.dynamic_pa_post_new_mm5nz_splitk_low"
        # "DynamicAttentionPostTest.dynamic_pa_papost_bf16_b48"


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
        # "MoEGateOnBoardTest.test_operation_b_16"
        # "MoEGateOnBoardTest.test_operation_b_32"

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

        #===========================================================================
        # matmul功能用例
        #===========================================================================
        "DynamicMatmulTest.mm_A_B_ND_bf16"
        "DynamicMatmulTest.mm_A_B_NZ_bf16"
        "DynamicMatmulTest.mm_A_Bt_ND_fp16"
        "DynamicMatmulTest.mm_A_Bt_NZ_fp16"
        "DynamicMatmulTest.mm_A_B_NZ_int8"
        "DynamicMatmulTest.mm_A_Bt_NZ_int8"
        "DynamicMatmulTest.mm_A_B_ND_bf16_tile1"
        "DynamicMatmulTest.mm_A_Bt_ND_fp16_tile2"
        "DynamicMatmulTest.mm_A_B_NZ_int8_tile3"
        "DynamicMatmulTest.mm_A_Bt_NZ_int8_tile4"
        "DynamicMatmulTest.mm_A_ND_B_ND_C_NZ"
        "DynamicMatmulTest.mm_AT_B_ANZ_BND_bf16"
        "DynamicMatmulTest.mm_AT_BT_AND_BND_bf16"
        "DynamicMatmulTest.mm_AT_B_ANZ_BND_fp16_UNALIGN"
        #===========================================================================
        # matmul Interpreter 功能用例
        #===========================================================================
        "DynamicMatmulInterpreterTest.mm_A_B_ND_bf16"
        "DynamicMatmulInterpreterTest.mm_A_B_NZ_bf16"
        "DynamicMatmulInterpreterTest.mm_A_Bt_ND_fp16"
        "DynamicMatmulInterpreterTest.mm_A_Bt_NZ_fp16"
        "DynamicMatmulInterpreterTest.mm_A_Bt_ND_fp16_tile2"
        "DynamicMatmulInterpreterTest.mm_A_Bt_NZ_int8_tile4"
        #===========================================================================
         # batchMatmul功能用例
        #===========================================================================
        "DynamicBatchMatmulTest.test_bmm_A_B_ND_bf16"
        "DynamicBatchMatmulTest.test_bmm_A_Bt_ND_fp16"
        "DynamicBatchMatmulTest.test_bmm_A_B_NZ_bf16"
        "DynamicBatchMatmulTest.test_bmm_A_Bt_NZ_fp16"
        "DynamicBatchMatmulTest.test_bmm_A_B_ND_bf16_tile1"
        "DynamicBatchMatmulTest.test_bmm_At_Bt_ND_fp16"
        "DynamicBatchMatmulTest.test_bmm_At_Bt_ANZ_BND_fp16"
        #===========================================================================
         # batchMatmul Interpreter 功能用例
        #===========================================================================
        "DynamicBatchMatmulInterpreterTest.test_bmm_A_B_ND_bf16"
        "DynamicBatchMatmulInterpreterTest.test_bmm_A_Bt_ND_fp16"
        "DynamicBatchMatmulInterpreterTest.test_bmm_A_B_NZ_bf16"
        "DynamicBatchMatmulInterpreterTest.test_bmm_A_Bt_NZ_fp16"
        "DynamicBatchMatmulInterpreterTest.test_bmm_A_B_ND_bf16_tile1"
        "DynamicBatchMatmulInterpreterTest.bmm4D_A_B_NZ"
        #===========================================================================
        # NSA native spare attention Interpreter
        #===========================================================================
        "DynamicWinAttenInterpreterTest.test_DynAttn_nas_win_attn_s1_2_actseqlen_1024_mla_fp16_inter"

        #===========================================================================
         # Reshape功能用例
        #===========================================================================
        "DynamicReshapeTest.test_only_reshape"
        "DynamicReshapeTest.test_only_reshape2"
        "DynamicReshapeTest.test_dyn_reshape2"
        "DynamicReshapeTest.test_assemble_diff_tile"
        #===========================================================================
        # 非对齐Reshape功能用例
        #===========================================================================
        "DynamicReshapeUnalignTest.test_add_dim"
        "DynamicReshapeUnalignTest.test_merge_dim"
        "DynamicReshapeUnalignTest.test_split_dim"
        #===========================================================================
        # reshapeinplace非对齐场景功能用例
        #===========================================================================
        "DynamicReshapeUnalignImplaceTest.test_op_reshape_op"
        "DynamicReshapeUnalignImplaceTest.merge_two_dynamic_dim"
        "DynamicReshapeUnalignImplaceTest.test_exchange_dim"
        "DynamicReshapeUnalignImplaceTest.test_reshape_special"
        "DynamicReshapeUnalignImplaceTest.test_src_op_dst_op"

)

set(_Target ${PTO_Fwk_STestNamePrefix}_machine)

file(GLOB_RECURSE _Sources src/*.cpp)
file(GLOB_RECURSE _Device_Srcs ${PTO_FWK_SRC_ROOT}/src/machine/device/*.cpp)
list(APPEND _Sources ${_Device_Srcs})

PTO_Fwk_STest_AddLib(
        TARGET ${_Target}
        SOURCES ${_Sources}
        PUBLIC_LINK_LIBRARIES tile_fwk_compiler tile_fwk_runtime tile_fwk_interface
        GTEST_FILTER_LIST ${_CaseFilterList}
        LD_LIBRARIES_EXT tile_fwk_compiler tile_fwk_runtime
        GOLDEN_SCRIPT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/src/ops
)
