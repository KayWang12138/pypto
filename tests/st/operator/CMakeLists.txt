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
        # NSA native spare attention
        #===========================================================================
        #"DynamicNSATest.nsa_b_16_s1_1_s2_8192_h_7168_fp16"
        #"DynamicSATest.slc_attn_fp16"
        "DynamicSATest.slc_attn_mtp_s1_2_fp16"
        "DynamicSATest.slc_attn_bf16_b48_s1_perf"
        "DynamicKvSATest.kv_slc_attn_b48_s1_fp16"
        "DynamicKvSATest.kv_slc_attn_b32_s2_bf16"
        "DynamicKvSATest.kv_slc_attn_b2_s1_fp16"
        "DynamicNSATest.mini"

        # NSA MlaProlog
        # "MlaPrologSTest.b32_s1_pa_nz_bf16_quant"
        # "MlaPrologSTest.b32_s2_pa_nz_bf16_quant"
        "MlaPrologSTest.b32_s2_pa_nd_fp16_quant"

        # NSA AttentionPost
        # "AttentionPostSTest.b32_s1_nz_bf16_quant"
        "AttentionPostSTest.b32_s2_nz_bf16_quant"

        #===========================================================================
        # '单 Op'精度用例
        #===========================================================================
        # Gather
        "GatherOnBoardTest.test_gather_float_64_256_1_64"
        #"GatherOnBoardTest.test_gather_float_1_64_32_1"
        "GatherOnBoardTest.test_gather_float_32_64_1_32"
        "GatherOnBoardTest.test_gather_float_32_65_1_33"
        "GatherOnBoardTest.test_gather_float_64_512_16_64"
        "GatherOnBoardTest.test_gather_float_8_7168_64"
        "GatherOnBoardTest.test_gather_float_8_7169_64"

        # GatherElement
        "GatherElementOnBoardTest.test_gather_element_float_16_64_8_32_1"
        "GatherElementOnBoardTest.test_gather_element_float_16_70_8_40_1"

        # TensorIndex
        #"TensorIndexOnBoardTest.test_tensorindex_float_64_256_1_64"
        "TensorIndexOnBoardTest.test_tensorindex_float_64_512_16_64"
        "TensorIndexOnBoardTest.test_tensorindex_float_8_7168_64_moe"

        # Concat
        #"ConcatOnBoardTest.test_concat_dim4_float32"
        "ConcatOnBoardTest.test_concat_dim2_float32_moe"
        "ConcatOnBoardTest.test_concat_exp_dim4_float32"
        "ConcatOnBoardTest.test_concat_100_inputs_float32"

        # qkvPre
        #"QkvPreOnBoardTest.test_qkvPre_bfloat16_32_2_1_256_256_512"
        #"QkvPreOnBoardTest.test_qkvPre_float16_32_32_1_256_256_1536"
        #"QkvPreOnBoardTest.test_qkvPre_bfloat16_32_32_1_256_7168_1536"

        # MLA prolog
        # "MlaPrologAbOnBoardTest.test_MlaPrologAb_float16_32_32_1_256_1024_1536"
        # "MlaPrologAbOnBoardTest.test_MlaPrologAb_float16_32_32_1_256_7168_1536_quant"
        #"MlaPrologAbOnBoardTest.test_MlaPrologAb_bfloat16_32_32_1_256_7168_1536"
        # "MlaPrologOnBoardTest.test_MlaProlog_float16_32_2_1_256_256_512"
        # "MlaPrologOnBoardTest.test_MlaProlog_bfloat16_4_32_1_256_7168_1536_quant"
        #"MlaPrologOnBoardTest.test_MlaProlog_float16_32_128_1_4096_7168_1536_quant"
        #"MlaPrologOnBoardTest.test_MlaProlog_bfloat16_32_32_1_256_7168_1536"

        # mla_prolog new
        # "MlaPrologV2OnBoardTest.test_MlaPrologV2_bfloat16_4_32_1_256_7168_1536"
        # "MlaPrologV2OnBoardTest.test_MlaPrologV2_bfloat16_32_128_1_4096_7168_1536"
        # "MlaPrologV2OnBoardTest.test_MlaPrologV2_bfloat16_32_32_1_256_7168_1536"
        #"MlaPrologV2OnBoardTest.test_mla_bf16_low_quant_smooth"
        #"MlaPrologV2OnBoardTest.test_mla_bf16_high_quant_smooth"
        # mla_prolog PA_BSND
        #"MlaPrologV2OnBoardTest.test_mla_bf16_high_quant_smooth_nz_pa_bsnd"
        # "MlaPrologV2OnBoardTest.test_mla_bf16_high48_quant_smooth_nz_pa_bsnd"
        # "MlaPrologV2OnBoardTest.attentionV2_bf16_low"
        # CodeGen by Json
        "MlaPrologV2OnBoardTest.test_MlaPrologV2_bfloat16_4_32_1_256_7168_1536_codegen_by_json"

        # attention
        # "MlaPrologOnBoardTest.attention_bf16_4_1024_1024_32_256"

        # Expand
        #"ExpandOnBoardTest.test_expand_32_1_to_32_32"
        #"ExpandOnBoardTest.test_expand_32_8_1_to_32_8_32"
        "ExpandOnBoardTest.test_expand_32_1_to_32_23"
        "ExpandOnBoardTest.test_expand_32_8_1_to_32_8_23"
        "ExpandOnBoardTest.test_expand_for_4_dim"
        "ExpandOnBoardTest.test_expand_1_1_to_1_16384"

        # Abs
        "AbsOnBoardTest.test_abs_8_4608"
        "AbsOnBoardTest.test_abs_8_4609"
        "AbsOnBoardTest.test_abs_1_16384"

        # unalign
        "OnBoardTest.test_operation_tensor_16_16_64_65_tileop_add_unalign"
        "OnBoardTest.test_operation_tensor_16_16_39_65_tileop_add_unalign"
        "OnBoardTest.test_operation_tensor_32_1_tileop_sub_unalign"
        "OnBoardTest.test_operation_tensor_32_1_tileop_mul_unalign"
        "OnBoardTest.test_operation_tensor_32_1_tileop_add_unalign"
        "OnBoardTest.test_operation_add_vs_dim2_unalign"
        "OnBoardTest.test_operation_mul_vs_dim3_unalign"
        "OnBoardTest.test_operation_sub_vs_dim4_unalign"
        "OnBoardTest.test_operation_div_vs_dim1_unalign"

        # ifa-pa
        "OnBoardIFATest.test_32_1_sub_32_1"
        "OnBoardIFATest.test_32_1_mul_32_1"
        "OnBoardIFATest.test_32_1_maximum"
        "OnBoardIFATest.test_32_1_tileop_exp"
        "OnBoardIFATest.test_32_1_reciprocal"
        # "OnBoardIFATest.test_page_attention_static_low_latency"

        #"OnBoardTest.test_operation_tensor_dim2_add"
        #"OnBoardTest.test_operation_scalar_dim2_add_FP16"
        #"OnBoardTest.test_operation_tensor_dim4_add"
        #"OnBoardTest.test_operation_tensor_16_16_64_64_tileop_add"
        #"OnBoardTest.test_operation_tensor_16_16_64_64_tileop_sub"
        #"OnBoardTest.test_operation_tensor_16_16_64_64_tileop_mul"
        #"OnBoardTest.test_operation_tensor_16_16_64_64_tileop_div"
        #"OnBoardTest.test_operation_tensor_8_80_80_tileop_add"
        #"OnBoardTest.test_operation_tensor_8_80_80_tileop_sub"
        #"OnBoardTest.test_operation_tensor_8_80_80_tileop_mul"
        "OnBoardTest.test_operation_tensor_8_80_80_tileop_div"
        "OnBoardTest.test_operation_tensor_64_128_tileop_add"
        "OnBoardTest.test_operation_tensor_1_n_to_m_n_mul"
        #"OnBoardTest.test_operation_tensor_64_128_tileop_sub"
        #"OnBoardTest.test_operation_tensor_64_128_tileop_mul"
        #"OnBoardTest.test_operation_tensor_64_128_tileop_div"
        #"OnBoardTest.test_operation_scalar_dim2_add"
        #"OnBoardTest.test_operation_scalar_dim3_add"
        #"OnBoardTest.test_operation_scalar_dim4_add"
        #"OnBoardTest.test_operation_scalar_dim2_sub"
        #"OnBoardTest.test_operation_scalar_dim3_sub"
        #"OnBoardTest.test_operation_scalar_dim4_sub"
        #"OnBoardTest.test_operation_scalar_dim2_mul"
        #"OnBoardTest.test_operation_scalar_dim3_mul"
        #"OnBoardTest.test_operation_scalar_dim4_mul"
        #"OnBoardTest.test_operation_scalar_dim2_div"
        #"OnBoardTest.test_operation_scalar_dim1_div"
        #"OnBoardTest.test_operation_scalar_dim3_div"
        #"OnBoardTest.test_operation_scalar_dim4_div"
        #"OnBoardTest.test_operation_scalar_dim1_div"
        "OnBoardTest.test_operation_scalar_32_32_1_256_mul"
        "OnBoardTest.test_operation_tensor_2_2_8_8_expand_add"
        "OnBoardTest.test_operation_tensor_4_4_16_16_expand_add"
        # "OnBoardTest.test_operation_tensor_1_1_32_to_16_32_32_expand_add"
        "OnBoardTest.test_operation_tensor_8_16_1_to_8_16_16_expand_add"
        "OnBoardTest.test_operation_tensor_8_8_1_to_8_8_7168_expand_Mul_moe"
        "OnBoardTest.test_operation_tensor_8_1_16_to_8_16_16_expand_add"
        "OnBoardTest.test_operation_tensor_1_16_16_to_8_16_16_expand_add"
        # "OnBoardTest.test_operation_tensor_8_1_1_to_8_16_16_expand_add"
        "OnBoardTest.test_operation_tensor_16_32_32_to_16_32_1_tileop_mul"
        "OnBoardTest.test_operation_tensor_8_8_1_1_to_8_8_1_256_tileop_sub"
        # "OnBoardTest.test_operation_tensor_1_1_1_64_to_1_128_1_64_tileop_mul01"
        #"OnBoardTest.test_operation_tensor_32_32_1_1_to_32_32_1_256_tileop_add"
        "OnBoardTest.test_operation_tensor_32_32_1_1_to_32_32_1_256_tileop_div"
        #"OnBoardTest.test_operation_tensor_8_8_1_to_8_8_7168_expand_mul"
        #"OnBoardTest.test_operation_tensor_8_8_1_to_8_8_7168_expand_sub"
        "OnBoardTest.test_mul_large_row"

        # Cast/Floor/Round
        "CastOnBoard.test_cast_fp32toint32rint_1_4608"
        "CastOnBoard.test_cast_int32tofp16none_1_4608"
        "CastOnBoard.test_cast_fp16toint8trunc_1_4608"
        "CastOnBoard.test_cast_fp16tofp32_unalign"
        "CastOnBoard.test_cast_fp32toint32rint_1_16384"

        # Softmax
        #"SoftmaxOnBoard.test_softmax_max_single"
        #"SoftmaxOnBoard.test_softmax_div"
        #"SoftmaxOnBoard.test_softmax_sum_all"
        #"SoftmaxOnBoard.test_softmax_full_inference"
        "SoftmaxOnBoard.test_softmax_deepseek"
        #"SoftmaxOnBoard.test_softmax_flash_attention"

        # Sin/Cos
        "OnBoardTest.test_cos_dim4_float16"
        "OnBoardTest.test_sin_dim2_float32"

        # attention post
        # "OnBoardTest.test_attention_post_bf16_real_batch4"
        # "OnBoardTest.test_attention_post_bf16_real_n128"

        # RowMaxSingle/RowSumSingle
        #"RowMaxSumSingleOnBoardTest.test_operation_row_max_single"
        #"RowMaxSumSingleOnBoardTest.test_operation_row_sum_single"
        #"RowMaxSumSingleOnBoardTest.test_operation_row_max_single_3dim"
        "RowMaxSumSingleOnBoardTest.test_operation_row_sum_single_3dim_mla_rmsNorm"
        "RowMaxSumSingleOnBoardTest.test_operation_row_max_single_4dim_softmax"
        #"RowMaxSumSingleOnBoardTest.test_operation_row_max_single_4dim_softmax_unalign"
        "RowMaxSumSingleOnBoardTest.test_operation_row_sum_single_4dim_softmax"
        "RowMaxSumSingleOnBoardTest.test_operation_row_sum_single_3dim_moe"
        "RowMaxSumSingleOnBoardTest.test_operation_row_sum_single_2dim_moe"
        "RowMaxSumSingleOnBoardTest.test_operation_row_sum_single_3dim_big_moe"
        "RowMaxSumSingleOnBoardTest.test_row_sum_single_4dim_axis0_unalign"
        "RowMaxSumSingleOnBoardTest.test_row_sum_single_4dim_axis1_unalign"
        "RowMaxSumSingleOnBoardTest.test_row_sum_single_4dim_axis2_unalign"
        "RowMaxSumSingleOnBoardTest.test_row_sum_single_unalign"
        "RowMaxSumSingleOnBoardTest.test_row_sum_single_unalign_4_93"
        "RowMaxSumSingleOnBoardTest.test_row_sum_single_unalign_4d"
        "RowMaxSumSingleOnBoardTest.test_row_max_single_unalign"
        "RowMaxSumSingleOnBoardTest.test_row_max_single_unalign_4_93"
        "RowMaxSumSingleOnBoardTest.test_row_max_single_unalign_4d"

        # rope sub graph
        #"RoPEOnBoardTest.test_operation_rope_reshape_transpose_reshape_muls"
        #"RoPEOnBoardTest.test_operation_rope_tensorIndex_unsqueeze_mul"
        #"RoPEOnBoardTest.test_operation_rope_reshape_view_muls"
        #"RoPEOnBoardTest.test_operation_rope_reshape_view_muls_concat"
        "RoPEOnBoardTest.test_operation_rope_deepseekv3"
        "RoPEOnBoardTest.test_operation_rope_v2_deepseekv3_b32"
        #"RoPESubGraphOnBoardTest.test_operation_rope_subgraph_deepseekv3"
        #"RoPESubGraphOnBoardTest.test_operation_rope_subgraph_deepseekv3_fp16"
        #"RoPESubGraphOnBoardTest.test_operation_rope_subgraph_deepseekv3_fp16_2batch"
        "RoPESubGraphOnBoardTest.test_operation_rope_subgraph_deepseekv3_bf16"
        "RoPESubGraphOnBoardTest.test_operation_rope_subgraph_deepseekv3_bf16_2batch"
        "RoPESubGraphOnBoardTest.test_operation_rope_subgraph_deepseekv3_bf16_32batch"
        "RoPESubGraphOnBoardTest.test_CD_bf16_32batch"
        "RoPESubGraphOnBoardTest.test_CD_bf16_32batch_4k"

        #
        "OnBoardTest.test_unary_operation_32_32_tileop_exp"
        "OnBoardTest.test_unary_operation_16_32_32_tileop_exp"
        "OnBoardTest.test_unary_operation_16_16_64_64_tileop_exp"
        "OnBoardTest.test_unary_operation_32_32_tileop_sqrt"
        "OnBoardTest.test_unary_operation_16_32_32_tileop_sqrt"
        "OnBoardTest.test_unary_operation_16_16_64_64_tileop_sqrt"
        "OnBoardTest.test_unary_operation_16_16_64_70_tileop_sqrt"
        "OnBoardTest.test_unary_operation_32_32_tileop_reciprocal"
        "OnBoardTest.test_unary_operation_16_32_32_tileop_reciprocal"
        "OnBoardTest.test_unary_operation_16_16_64_64_tileop_reciprocal"
        #"LogicalNotOnBoardTest.test_logicalnot_16_32_fp32"
        #"LogicalNotOnBoardTest.test_logicalnot_16_32_32_fp16"

        #
        "TransposeTest.TestTranspose_ROPE_5D"
        "TransposeTest.TestTranspose_MLA_3D_0"
        "TransposeTest.TestTranspose_MLA_3D_1"
        "TransposeTest.TestTranspose_MLA_4D_0"
        "TransposeTest.TestTranspose_MLA_4D_1"
        "TransposeTest.TestTranspose_MLA_4D_3"
        "TransposeTest.TestTranspose_MLA_4D_4"
        "TransposeTest.TestTranspose_MLA_4D_5"
        "TransposeTest.TestTranspose_MLA_4D_50"
        "TransposeTest.TestTranspose_MLA_4D_6"
        "TransposeTest.TestTranspose_MLA_3D_2"
        "TransposeTest.Test_Datamove_Nonalign_Dim4"
        "TransposeTest.Test_Datamove_Nonalign_Dim3"

        # QuantST
        #"QuantTest.Test_ScalarDivS"
        #"QuantTest.Test_ScalarAddS"
        #"QuantTest.Test_ScalarSubS"
        #"QuantTest.Test_ScalarMulS"
        #"QuantTest.Test_ScalarMaxS"
        "QuantTest.Test_ScalarOp"
        "QuantOnBoardTest.test_Quant_32_1_7168"
        "QuantOnBoardTest.test_Quant_32_7168"
        #"QuantOnBoardTest.test_Quant_Smooth_32_4_128"
        "QuantOnBoardTest.test_Quant_Smooth_32_7168"
        "QuantMMOnBoardTest.test_QuantMM_32_16384_times_16384_7168_np"
        "QuantMMOnBoardTest.test_QuantMM_128_32_512_times_128_512_128_torch"

        #
        "OnBoardTest.test_operation_ub_reshape"
        "OnBoardTest.test_operation_gm_reshape"
        "OnBoardTest.test_reshape_matmul_mul"
        "OnBoardTest.test_operation_ub_reshape_3dimto2dim"
        #"OnBoardTest.test_operation_gm_reshape_2dimto3dim"
        "ArgsortOnBoardTest.test_operation_tensor_64__argsort_moe"

        #===========================================================================
        # DeepSeekV3 'MLA 子图/整图' 精度用例
        #===========================================================================
        # "OnBoardTest.test_query_states_fp16_b32_n2"
        # "OnBoardTest.test_query_states_fp16_b32_n32"
        # "OnBoardTest.test_query_states_bf16_b32_n2"
        "OnBoardTest.test_query_states_bf16_b32_n32"

        #===========================================================================
        # DeepSeekV3 'MOE 子图/整图' 精度用例
        #===========================================================================
        # "MoEGatePart1OnBoardTest.test_moe_gate_part1"
        # "MoEGatePart2OnBoardTest.test_operation_b_2"
        # "MoEGatePart2OnBoardTest.test_operation_b_1024"
        # "MoEPart4OnBoardTest.test_operation_b_2"
        "MoeInferOnbroadTest.test_deepseekMoEInfer"
        "MoeInferOnbroadTest.test_deepseekMoEInfer_singleout"
        "DynamicFFNTest.TestOnbroadDynamicFFN"
        # "MoeInferOnbroadTest.test_deepseekMoEInfer_singleout_singlemlp_withquant"

        #===========================================================================
        # Llama Layer 'FA' 精度用例
        #===========================================================================

        #===========================================================================
        # Llama Layer '整图' 精度用例
        #===========================================================================
        #"LLamaLayerTest.llama_1_1_128_128"
        #"LLamaLayerTest.llama_1_1_256_128"
        #"LLamaLayerTest.llama_1_1_512_128"
        #"LLamaLayerTest.llama_1_1_1024_128"
        #"LLamaLayerTest.llama_1_1_4096_128"
        "RmsNormTest.test_32_32_tileop_rmsnorm"
        "RmsNormTest.test_2_32_32_tileop_rmsnorm"
        "RmsNormTest.test_2_2_32_32_tileop_rmsnorm"
        "RmsNormTest.test_32_1536_tileop_rmsnorm_fp16_realCase"
        "RmsNormTest.test_32_1536_tileop_rmsnorm_gamma_fp16_realCase"
        "SigmoidTest.test_32_32_tileop_sigmoid"
        "SigmoidTest.test_32_256_tileop_sigmoid_realcase"
        "TopkOnBoardTest.test_operation_tensor_128_32_32_topk"
        "TopkOnBoardTest.test_operation_tensor_4_32_8_topk"
        "TopkOnBoardTest.test_operation_tensor_2_288_15_topk"
        "TopkOnBoardTest.test_operation_tensor_2_288_15_topk_reverse"
        #"MlpTest.test_256_512_tileop"

        #===========================================================================
        # Argsort: Argsort精度用例
        #===========================================================================
        "ArgsortOnBoardTest.test_operation_tensor_128_32_argsort"
        "ArgsortOnBoardTest.test_operation_tensor_4_32_argsort"
        "ArgsortOnBoardTest.test_operation_tensor_2_16_argsort"
        "ArgsortOnBoardTest.test_operation_tensor_64_argsort"
        "ArgsortOnBoardTest.test_operation_tensor_4_32__argsort"

        #===========================================================================
        # Matmul
        #===========================================================================
        # 1. matmul normal cases
        "MatmulOnBoardTest.test_mm_float32_32_512_576"
        "MatmulOnBoardTest.test_mm_bfloat16_64_128_128"
        "MatmulOnBoardTest.test_mm_bfloat16_f32_64_128_128"
        #"MatmulOnBoardTest.test_mm_float32_128_128_128"
        #"MatmulOnBoardTest.test_mm_float32_32_128_128"

        # 2. matmul with unalign cases
        "MatmulOnBoardTest.test_mm_unalign_float32_2_128_128"
        "MatmulOnBoardTest.test_mm_unalign_float32_16_35_32"
        # 3. batch matmul cases
        "OnBoardTest.test_BMM_Simple"
        "OnBoardTest.test_BMM_3D_Brc"
        "OnBoardTest.test_BMM_3D_Brc_Transpose"
        "OnBoardTest.test_BMM_4D_Brc_Transpose"
        "OnBoardTest.test_BMM_UNALIGN_2_1024_32"
        "OnBoardTest.test_BMM_post"

        # 4. batch matmul with NZ
        "OnBoardTest.test_BMM_NZ_1_32_32_32"
        "OnBoardTest.test_BMM_NZ_1_32_32_320"
        "OnBoardTest.test_BMM_NZ_1_64_32_16"

        # 5. large L1 copyIn tilesize
        "MatmulOnBoardTest.test_mm_float32_32_128_64"
        "MatmulOnBoardTest.test_mm_float32_32_192_64"
        "MatmulOnBoardTest.test_mm_int8_32_128_64"

        # vduplicate
        "VecdupTest.TestVecDup"
)

set(_Target ${TileFwkSTestNamePrefix}_operator)

file(GLOB_RECURSE _Sources src/*.cpp)

TileFwk_STest_AddLib(
        TARGET ${_Target}
        SOURCES ${_Sources}
        PUBLIC_LINK_LIBRARIES tile_fwk_interface tile_fwk_operator tile_fwk_compiler tile_fwk_runtime tile_fwk_simulation tile_fwk_simulation_ca
        GTEST_FILTER_LIST ${_CaseFilterList}
        LD_LIBRARIES_EXT tile_fwk_compiler tile_fwk_simulation tile_fwk_simulation_ca tile_fwk_simulation_pv
        GOLDEN_SCRIPT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src
)
add_custom_command(
        TARGET ${_Target} PRE_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory ${TileFwkSTestExePath}/scripts/
        COMMAND ${CMAKE_COMMAND} -E copy ${TILE_FWK_SRC_ROOT}/src/cost_model/simulation/scripts/print_swim_lane.py ${TileFwkSTestExePath}/scripts/
        COMMAND ${CMAKE_COMMAND} -E copy ${TILE_FWK_SRC_ROOT}/src/cost_model/simulation/scripts/draw_comm_swim_lane_png.py ${TileFwkSTestExePath}/scripts/
        COMMAND ${CMAKE_COMMAND} -E copy ${TILE_FWK_SRC_ROOT}/src/cost_model/simulation/scripts/draw_pipe_swim_lane.py ${TileFwkSTestExePath}/scripts/
        COMMAND ${CMAKE_COMMAND} -E copy ${TILE_FWK_SRC_ROOT}/tools/draw_swim_lane.py ${TileFwkSTestExePath}/scripts/
        COMMAND ${CMAKE_COMMAND} -E copy ${TILE_FWK_SRC_ROOT}/tools/function_json_convert.py ${TileFwkSTestExePath}/scripts/
        COMMENT "Copy ${_Target} dependence scripts file."
)
