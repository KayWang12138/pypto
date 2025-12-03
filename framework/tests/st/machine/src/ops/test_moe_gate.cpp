/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_moe_gate.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include "interface/inner/tilefwk/tilefwk_api.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "tilefwk/data_type.h"
#include "test_common.h"
#include "test_suite_stest_ops.h"
#include "machine/runtime/runtime.h"
#include "machine/runtime/device_runner.h"
#include "machine/runtime/tilefwk_runtime_api.h"

using namespace npu::tile_fwk;

constexpr float F_1 = 1.0;
constexpr float F_0 = 0.0;
constexpr double DF_1E_20 = 1e-20f;

class MoEGateOnBoardTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac {};

struct MoEGateParams {
    int32_t H;
    int32_t nRoutedExperts;
    int32_t nGroup;
    int32_t topkGroup;
    int32_t numExpertsPerTopk;
    int32_t firstKDenseReplace;
    int32_t moeLayerFreq;
    int32_t S;
    int32_t B;
};

void MoEGateOnBoardFunc(MoEGateParams& opsParams) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());
    TileFwkInit("");

    int32_t H = opsParams.H;
    int32_t nRoutedExperts = opsParams.nRoutedExperts;
    int32_t nGroup = opsParams.nGroup;
    int32_t topkGroup = opsParams.topkGroup;
    int32_t numExpertsPerTopk = opsParams.numExpertsPerTopk;
    int32_t S = opsParams.S;
    int32_t B = opsParams.B;

    // set input data
    uint64_t e_score_correction_bias_size = nRoutedExperts;
    uint64_t hidden_states_size = B * S * H;
    uint64_t weight_size = nRoutedExperts * H;
    //read input
    void *inputEScoreCorrectionBiasPtr = readToDev<float>(GetGoldenDir() + "/e_score_correction_bias.bin",
                            e_score_correction_bias_size);
    void *inputHiddenStatePtr = readToDev<float>(GetGoldenDir() + "/hidden_states.bin",
                            hidden_states_size);
    void *inputWeightPtr = readToDev<float>(GetGoldenDir() + "/weight.bin", weight_size);

    assert(inputEScoreCorrectionBiasPtr !=nullptr && inputHiddenStatePtr != nullptr &&
            inputWeightPtr!=nullptr);

    // alloc output
    uint8_t* outputTopkIdxPtr = allocDevAddr(B * S * numExpertsPerTopk * sizeof(float));
    uint8_t* outputTopkWeightPtr = allocDevAddr(B * S * numExpertsPerTopk * sizeof(float));
    assert(outputTopkIdxPtr != nullptr && outputTopkWeightPtr != nullptr);

    TileShape::Current().SetCubeTile({16, 16}, {128, 128}, {64, 64});
    TileShape::Current().SetVecTile({16, 64});
    Tensor input_e_score_bias(DT_FP32, {1, nRoutedExperts},
                            (uint8_t *)inputEScoreCorrectionBiasPtr, "InputEScoresCorrectionBias");
    Tensor input_hidden_state(DT_FP32, {B, S, H}, (uint8_t *)inputHiddenStatePtr, "InputHiddenState");
    Tensor input_weight(DT_FP32, {nRoutedExperts, H}, (uint8_t *)inputWeightPtr, "InputWeight");
    Tensor output_topk_idx(DT_INT32, {B*S, numExpertsPerTopk}, (uint8_t *)outputTopkIdxPtr, "TopkIdx");
    Tensor output_topk_weight(DT_FP32, {B*S, numExpertsPerTopk},
                            (uint8_t *)outputTopkWeightPtr, "TopkWeight");

    TileFwkBeginFunction(
        "MOE_GATE_T", {input_e_score_bias, input_hidden_state, input_weight, output_topk_idx, output_topk_weight});
    {
        //Part1
        Tensor input_hidden_state_reshape = Reshape(input_hidden_state, {B*S, H});
        Tensor tmpC(DT_FP32, {B*S, nRoutedExperts}, "tmp_c");
        tmpC = Mul(tmpC, Element(DataType::DT_FP32, F_0));
        std::vector<Tensor> matmulResult;
        auto kSplit = 2;
        auto kSplitSize = H / kSplit;
        for (int ki = 0; ki < kSplit; ki++) {
            auto inputMk = View(input_hidden_state_reshape, {B*S, kSplitSize}, {0, ki * kSplitSize});
            auto inputKn = View(input_weight, {nRoutedExperts, kSplitSize}, {0, ki * kSplitSize});
            auto tmp = npu::tile_fwk::Matrix::Matmul<false, true>(DT_FP32, inputMk, inputKn, tmpC);    // B * 256
            matmulResult.emplace_back(tmp);
        }
        auto logits = npu::tile_fwk::Reduce(matmulResult, ReduceMode::ATOMIC_ADD);
        // B * 256,不切K
        // auto logits = npu::tile_fwk::Matrix::Matmul<false, true>(DT_FP32, input_hidden_state_reshape, input_weight);

        TileShape::Current().SetVecTile({1, 256});
        auto output_scores = SoftmaxNew(logits);
        auto output_scores_for_choice = Add(output_scores, input_e_score_bias);  // [B, 256]

        // Part2
        Tensor scores_for_choice_reshape;
        scores_for_choice_reshape = Reshape(output_scores_for_choice, {B * nGroup, 32});   // [B*8, 32]
        TileShape::Current().SetVecTile({8, 32});
        auto output_topk2 = TopK(scores_for_choice_reshape, 2, -1);                         // [B*8, 2]
        auto group_scores = Sum(std::get<0>(output_topk2), -1, true);                        // [B*8, 1]
        group_scores = Reshape(group_scores, {B*S, nGroup});                               // [B, 8]
        TileShape::Current().SetVecTile({1, 8});
        auto output_topk4 = TopK(group_scores, topkGroup, -1);                                      // [B, 4]
        auto output_group_idx = std::get<1>(output_topk4);                                       // [B, 4]
        auto output_group_mask = Mul(group_scores, Element(DataType::DT_FP32, F_0));                                 // [B, 8]

        // Part3
        TileShape::Current().SetVecTile({1, nGroup});
        output_group_mask = Scatter_(output_group_mask, output_group_idx,
            Element(DataType::DT_FP32, F_1), 1); // (b*s, nGroup)
        Tensor group_mask_new = Reshape(output_group_mask, {B*S, nGroup, 1}); // (b*s, nGroup, 1)
        TileShape::Current().SetVecTile(1, 8, 32);
        // [b*s,nGroup,1] -> [b*s,nGroup,32]
        Tensor score_mask = Expand(group_mask_new, {B*S, nGroup, nRoutedExperts / nGroup});
        // (b*s,-1) [b*s,256]
        auto score_mask_new = Reshape(score_mask, {B*S, nGroup*nRoutedExperts / nGroup});
        TileShape::Current().SetVecTile(1, 256);
        auto score1 = Mul(output_scores_for_choice, score_mask_new);
        auto score2 = Mul(LogicalNot(score_mask_new), Element(DataType::DT_FP32, F_0));
        auto output_tmp_scores = Add(score1, score2);

        // Part4
        TileShape::Current().SetVecTile({1, 256});
        output_topk_idx = std::get<1>(TopK(output_tmp_scores, numExpertsPerTopk, -1)); // [b*s,256]->[b*s,8]
        auto topk_weight = GatherElements(output_scores, output_topk_idx, 1); // [b*s,8]
        auto topk_weight_sum = Sum(topk_weight, 1, true);      // [b*s,8]->[b*s,1]
        auto denominator = Add(topk_weight_sum, Element(DataType::DT_FP32, DF_1E_20)); // [b*s,1]
        output_topk_weight = Div(topk_weight, denominator); // [b*s,numExpertsPerTopk]
    }
    TileFwkEndFunction();

    /* torch compile */
    void* handle = TileFwkCompile();
    /* torch prepare workspace memory */
    uint8_t* workspaceAddr = nullptr;
    uint64_t workspaceSize = 0;
    TileFwkGetWorkspaceSize(handle, &workspaceSize);
    machine::GetRA()->AllocDevAddr(&workspaceAddr, workspaceSize);
    std::vector<void*> opArgsRun = {inputEScoreCorrectionBiasPtr, inputHiddenStatePtr,
                            inputWeightPtr, outputTopkIdxPtr, outputTopkWeightPtr};
    TileFwkRunAsync(handle, workspaceAddr, machine::GetRA()->GetScheStream(), opArgsRun);
    int rc = rtStreamSynchronize(machine::GetRA()->GetScheStream());
    if (rc < 0) {
        ASSERT(false);
        ALOG_INFO_F("moe gate function aicpu stream sync failed");
    }

    std::vector<int32_t> golden_final_idx(B * S * numExpertsPerTopk);
    std::vector<float> golden_final_weight(B * S * numExpertsPerTopk);
    std::vector<int32_t> dev_final_idx(B * S * numExpertsPerTopk);
    std::vector<float> dev_final_weight(B * S * numExpertsPerTopk);

    printf("outputTopkIdxPtr is %p\n", outputTopkIdxPtr);
    printf("outputTopkWeightPtr is %p\n", outputTopkWeightPtr);

    machine::GetRA()->CopyFromTensor((uint8_t *)dev_final_idx.data(), (uint8_t *)outputTopkIdxPtr,
                            B * S * numExpertsPerTopk * sizeof(float));
    machine::GetRA()->CopyFromTensor((uint8_t *)dev_final_weight.data(), (uint8_t *)outputTopkWeightPtr,
                            B * S * numExpertsPerTopk * sizeof(float));

    // 真值比对
    readInput(GetGoldenDir() + "/topk_idx.bin", golden_final_idx);
    readInput(GetGoldenDir() + "/topk_weight.bin", golden_final_weight);

    // device 结果写出
    writeInput("npu_final_idx.bin", dev_final_idx);
    writeInput("npu_final_weight.bin", dev_final_weight);

    std::cout << "compare topk_idx -------- " << std::endl;
    int ret_final_idx = resultCmp(golden_final_idx, dev_final_idx, 0);
    EXPECT_TRUE(ret_final_idx);

    std::cout << "compare topk_weight -------- " << std::endl;
    int ret_final_weight = resultCmp(golden_final_weight, dev_final_weight, 1.0e-5f);
    EXPECT_TRUE(ret_final_weight);
}

TEST_F(MoEGateOnBoardTest, test_operation_b_4) {
    MoEGateParams params;
    params.H = 7168;
    params.nRoutedExperts = 256;
    params.nGroup = 8;
    params.topkGroup = 4;
    params.numExpertsPerTopk = 8;
    params.firstKDenseReplace = 3;
    params.moeLayerFreq = 1;
    params.S = 1;
    params.B = 4;
    MoEGateOnBoardFunc(params);
}

TEST_F(MoEGateOnBoardTest, test_operation_b_16) {
    MoEGateParams params;
    params.H = 7168;
    params.nRoutedExperts = 256;
    params.nGroup = 8;
    params.topkGroup = 4;
    params.numExpertsPerTopk = 8;
    params.firstKDenseReplace = 3;
    params.moeLayerFreq = 1;
    params.S = 1;
    params.B = 16;
    MoEGateOnBoardFunc(params);
}

TEST_F(MoEGateOnBoardTest, test_operation_b_32) {
    MoEGateParams params;
    params.H = 7168;
    params.nRoutedExperts = 256;
    params.nGroup = 8;
    params.topkGroup = 4;
    params.numExpertsPerTopk = 8;
    params.firstKDenseReplace = 3;
    params.moeLayerFreq = 1;
    params.S = 1;
    params.B = 32;
    MoEGateOnBoardFunc(params);
    uint64_t taskTime = npu::tile_fwk::DeviceRunner::Get().GetTasksTime();
    uint64_t threshold = 3000;
    bool profRet = (taskTime > 0 && taskTime < threshold * 1.05);
    std::cout<<"moe_gate b=32 prof threshold is: "<< threshold <<", cost time is: "<< taskTime << std::endl;
    EXPECT_EQ(profRet, true);
}

TEST_F(MoEGateOnBoardTest, test_operation_b_128) {
    MoEGateParams params;
    params.H = 7168;
    params.nRoutedExperts = 256;
    params.nGroup = 8;
    params.topkGroup = 4;
    params.numExpertsPerTopk = 8;
    params.firstKDenseReplace = 3;
    params.moeLayerFreq = 1;
    params.S = 1;
    params.B = 128;
    MoEGateOnBoardFunc(params);
}
