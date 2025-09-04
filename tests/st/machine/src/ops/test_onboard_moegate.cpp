/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_onboard_moegate.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include "interface/inner/tilefwk/tilefwk_api.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "tilefwk/data_type.h"
#include "test_common.h"
#include "test_suite_stest_ops.h"
#include "runtime.h"
#include "tilefwk_runtime_api.h"

using namespace npu::tile_fwk;

class MoegateOnBoardTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac {};

TEST_F(MoegateOnBoardTest, test_moegate_graph3_case1) {
    int B = 1;
    int S = 4;

    int topkGroup = 4;
    int nGroup = 8;
    int nRoutedExperts = 256;

    std::vector<int64_t> shape_topk_group =  {B*S, topkGroup};
    std::vector<int64_t> shape_group_mask =  {B*S, nGroup};
    std::vector<int64_t> shape_score_for_choice = {B*S, nRoutedExperts};
    std::vector<int64_t> shape_score = {B*S, nRoutedExperts};

    int capacity0 = shape_topk_group[0] * shape_topk_group[1];
    int capacity1 = shape_score_for_choice[0] * shape_score_for_choice[1];
    int capacity2 = shape_group_mask[0] * shape_group_mask[1];
    int capacity3 = shape_score[0] * shape_score[1];

    aclInit(nullptr);
    rtSetDevice(npu::tile_fwk::stubs::DeviceStub::GetCurrentDeviceId());
    TileFwkInit("");

    TileShape::Current().SetVecTile(16, 16);

    void *x_ptr = readToDev(GetGoldenDir() + "/group_idx.bin", capacity0);
    void *y_ptr = readToDev(GetGoldenDir() + "/scores_for_choice.bin", capacity1);
    void *zero_ptr = readToDev(GetGoldenDir() + "/group_mask_zero.bin", capacity2);

    // Tensor group_idx(DataType::DT_FP32, shape_topk_group, (uint8_t *)x_ptr, "group_idx");
    Tensor group_idx(DataType::DT_INT32, shape_topk_group, (uint8_t *)x_ptr, "group_idx");
    Tensor group_mask(DataType::DT_FP32, shape_group_mask, (uint8_t *)zero_ptr, "group_mask");
    Tensor scores_for_choice(DataType::DT_FP32, shape_score_for_choice, (uint8_t *)y_ptr, "scores_for_choice");
    Tensor res_scores(DataType::DT_FP32, shape_score_for_choice, "scores");

    // std::vector<Tensor> opArgs = {group_idx, group_mask, res_scores};
    /* torch capture */
    TileFwkBeginFunction("MOEGATE_GRAPH3", {group_idx, group_mask, scores_for_choice, res_scores});
    {
        auto tmp_group_mask = ScatterElement(group_mask, group_idx, Element(DataType::DT_FP32, 1.0), 1); // (b*s, nGroup)
        Tensor group_mask_new = Reshape(tmp_group_mask, {B*S, nGroup, 1}); // (b*s, nGroup, 1)

        TileShape::Current().SetVecTile(16, 16, 32);
        Tensor score_mask = Expand(group_mask_new, {B*S, nGroup, nRoutedExperts / nGroup}); // [b*s,nGroup,1] -> [b*s,nGroup,32]
        auto score_mask_new = Reshape(score_mask, {B*S, nGroup*nRoutedExperts / nGroup}); // (b*s,-1) [b*s,256]

        TileShape::Current().SetVecTile(64, 64);
        auto score1 = Mul(scores_for_choice, score_mask_new);
        auto score2 = MulS(LogicalNot(score_mask_new), Element(DataType::DT_FP32, -3.4e+38f));
        res_scores = Add(score1, score2);
    }
    TileFwkEndFunction();

    /* torch compile */
    void* handle = TileFwkCompile();

    /* torch prepare workspace memory */
    uint8_t* workspaceAddr = nullptr;
    uint64_t workspaceSize = 0;
    TileFwkGetWorkspaceSize(handle, &workspaceSize);
    machine::GetRA()->AllocDevAddr(&workspaceAddr, workspaceSize);

    /* torch prepare args device memory and run  */
    uint8_t* outTensorAddr = nullptr;
    machine::GetRA()->AllocDevAddr(&outTensorAddr, capacity3 * sizeof(float));

    std::vector<void*> opArgsRun = {x_ptr, zero_ptr, y_ptr, outTensorAddr};
    // std::vector<void*> opArgsRun = {x_ptr, zero_ptr, outTensorAddr};
    TileFwkRunAsync(handle, workspaceAddr, machine::GetRA()->GetStreamAICPU(),  opArgsRun);
    int rc = rtStreamSynchronize(machine::GetRA()->GetStreamAICPU());
    if (rc < 0) {
        ASSERT(false);
        ALOG_INFO_F("FA function aicpu stream sync failed");
    }

    std::cout << "======capacity3 size:" << capacity3 << std::endl;
    std::vector<float> golden(capacity3);
    std::vector<float> dev_res(capacity3);
    machine::GetRA()->CopyFromTensor((uint8_t *)dev_res.data(), (uint8_t *)outTensorAddr, capacity3 * sizeof(float));
    readInput(GetGoldenDir() + "/z_golden.bin", golden);

    int ret = resultCmp(golden, dev_res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(MoegateOnBoardTest, test_moegate_graph3_case2_32_1_7168) {
    int B = 32;
    int S = 1;

    int topkGroup = 4;
    int nGroup = 8;
    int nRoutedExperts = 256;

    std::vector<int64_t> shape_topk_group =  {B*S, topkGroup}; //[32,4]
    std::vector<int64_t> shape_group_mask =  {B*S, nGroup}; // [32,8]
    std::vector<int64_t> shape_score_for_choice = {B*S, nRoutedExperts}; //[32,256]
    std::vector<int64_t> shape_score = {B*S, nRoutedExperts};

    int capacity0 = shape_topk_group[0] * shape_topk_group[1];
    int capacity1 = shape_score_for_choice[0] * shape_score_for_choice[1];
    int capacity2 = shape_group_mask[0] * shape_group_mask[1];
    int capacity3 = shape_score[0] * shape_score[1];

    aclInit(nullptr);
    rtSetDevice(npu::tile_fwk::stubs::DeviceStub::GetCurrentDeviceId());
    TileFwkInit("");

    TileShape::Current().SetVecTile(64, 64);

    void *x_ptr = readToDev(GetGoldenDir() + "/group_idx.bin", capacity0);
    void *y_ptr = readToDev(GetGoldenDir() + "/scores_for_choice.bin", capacity1);
    void *zero_ptr = readToDev(GetGoldenDir() + "/group_mask_zero.bin", capacity2);

    // Tensor group_idx(DataType::DT_FP32, shape_topk_group, (uint8_t *)x_ptr, "group_idx");
    Tensor group_idx(DataType::DT_INT32, shape_topk_group, (uint8_t *)x_ptr, "group_idx");
    Tensor group_mask(DataType::DT_FP32, shape_group_mask, (uint8_t *)zero_ptr, "group_mask");
    Tensor scores_for_choice(DataType::DT_FP32, shape_score_for_choice, (uint8_t *)y_ptr, "scores_for_choice");
    Tensor res_scores(DataType::DT_FP32, shape_score_for_choice, "scores");

    // std::vector<Tensor> opArgs = {group_idx, group_mask, res_scores};
    /* torch capture */
    TileFwkBeginFunction("MOEGATE_GRAPH3", {group_idx, group_mask, scores_for_choice, res_scores});
    {
        auto tmp_group_mask = ScatterElement(group_mask, group_idx, Element(DataType::DT_FP32, 1.0), 1); // (b*s, nGroup)
        Tensor group_mask_new = Reshape(tmp_group_mask, {B*S, nGroup, 1}); // (b*s, nGroup, 1)

        TileShape::Current().SetVecTile(32, 8, 32); // 此处如果用(16 16 32)的tile 会有精度问题
        Tensor score_mask = Expand(group_mask_new, {B*S, nGroup, nRoutedExperts / nGroup}); // [b*s,nGroup,1] -> [b*s,nGroup,32]
        auto score_mask_new = Reshape(score_mask, {B*S, nGroup*nRoutedExperts / nGroup}); // (b*s,-1) [b*s,256]

        TileShape::Current().SetVecTile(64, 64);
        auto score1 = Mul(scores_for_choice, score_mask_new);
        auto score2 = MulS(LogicalNot(score_mask_new), Element(DataType::DT_FP32, -3.4e+38f));
        res_scores = Add(score1, score2);
    }
    TileFwkEndFunction();

    /* torch compile */
    void* handle = TileFwkCompile();

    /* torch prepare workspace memory */
    uint8_t* workspaceAddr = nullptr;
    uint64_t workspaceSize = 0;
    TileFwkGetWorkspaceSize(handle, &workspaceSize);
    machine::GetRA()->AllocDevAddr(&workspaceAddr, workspaceSize);

    /* torch prepare args device memory and run  */
    uint8_t* outTensorAddr = nullptr;
    machine::GetRA()->AllocDevAddr(&outTensorAddr, capacity3 * sizeof(float));

    std::vector<void*> opArgsRun = {x_ptr, zero_ptr, y_ptr, outTensorAddr};
    // std::vector<void*> opArgsRun = {x_ptr, zero_ptr, outTensorAddr};
    TileFwkRunAsync(handle, workspaceAddr, machine::GetRA()->GetStreamAICPU(),  opArgsRun);
    int rc = rtStreamSynchronize(machine::GetRA()->GetStreamAICPU());
    if (rc < 0) {
        ASSERT(false);
        ALOG_INFO_F("FA function aicpu stream sync failed");
    }

    std::cout << "======capacity3 size:" << capacity3 << std::endl;
    std::vector<float> golden(capacity3);
    std::vector<float> dev_res(capacity3);
    machine::GetRA()->CopyFromTensor((uint8_t *)dev_res.data(), (uint8_t *)outTensorAddr, capacity3 * sizeof(float));
    readInput(GetGoldenDir() + "/z_golden.bin", golden);

    int ret = resultCmp(golden, dev_res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(MoegateOnBoardTest, test_moegate_graph3_case2_8_1_7168) {
    int B = 8;
    int S = 1;

    int topkGroup = 4;
    int nGroup = 8;
    int nRoutedExperts = 256;

    std::vector<int64_t> shape_topk_group =  {B*S, topkGroup}; //[8,4]
    std::vector<int64_t> shape_group_mask =  {B*S, nGroup}; // [8,8]
    std::vector<int64_t> shape_score_for_choice = {B*S, nRoutedExperts}; //[8,256]
    std::vector<int64_t> shape_score = {B*S, nRoutedExperts};

    int capacity0 = shape_topk_group[0] * shape_topk_group[1];
    int capacity1 = shape_score_for_choice[0] * shape_score_for_choice[1];
    int capacity2 = shape_group_mask[0] * shape_group_mask[1];
    int capacity3 = shape_score[0] * shape_score[1];

    aclInit(nullptr);
    rtSetDevice(npu::tile_fwk::stubs::DeviceStub::GetCurrentDeviceId());
    TileFwkInit("");

    TileShape::Current().SetVecTile(64, 64);

    void *x_ptr = readToDev(GetGoldenDir() + "/group_idx.bin", capacity0);
    void *y_ptr = readToDev(GetGoldenDir() + "/scores_for_choice.bin", capacity1);
    void *zero_ptr = readToDev(GetGoldenDir() + "/group_mask_zero.bin", capacity2);

    // Tensor group_idx(DataType::DT_FP32, shape_topk_group, (uint8_t *)x_ptr, "group_idx");
    Tensor group_idx(DataType::DT_INT32, shape_topk_group, (uint8_t *)x_ptr, "group_idx");
    Tensor group_mask(DataType::DT_FP32, shape_group_mask, (uint8_t *)zero_ptr, "group_mask");
    Tensor scores_for_choice(DataType::DT_FP32, shape_score_for_choice, (uint8_t *)y_ptr, "scores_for_choice");
    Tensor res_scores(DataType::DT_FP32, shape_score_for_choice, "scores");

    // std::vector<Tensor> opArgs = {group_idx, group_mask, res_scores};
    /* torch capture */
    TileFwkBeginFunction("MOEGATE_GRAPH3", {group_idx, group_mask, scores_for_choice, res_scores});
    {
        auto tmp_group_mask = ScatterElement(group_mask, group_idx, Element(DataType::DT_FP32, 1.0), 1); // (b*s, nGroup)
        Tensor group_mask_new = Reshape(tmp_group_mask, {B*S, nGroup, 1}); // (b*s, nGroup, 1)

        TileShape::Current().SetVecTile(16, 16, 32);
        Tensor score_mask = Expand(group_mask_new, {B*S, nGroup, nRoutedExperts / nGroup}); // [b*s,nGroup,1] -> [b*s,nGroup,32]
        auto score_mask_new = Reshape(score_mask, {B*S, nGroup*nRoutedExperts / nGroup}); // (b*s,-1) [b*s,256]
        TileShape::Current().SetVecTile(64, 64);
        auto score1 = Mul(scores_for_choice, score_mask_new);
        auto score2 = MulS(LogicalNot(score_mask_new), Element(DataType::DT_FP32, -3.4e+38f));
        res_scores = Add(score1, score2);
    }
    TileFwkEndFunction();

    /* torch compile */
    void* handle = TileFwkCompile();

    /* torch prepare workspace memory */
    uint8_t* workspaceAddr = nullptr;
    uint64_t workspaceSize = 0;
    TileFwkGetWorkspaceSize(handle, &workspaceSize);
    machine::GetRA()->AllocDevAddr(&workspaceAddr, workspaceSize);

    /* torch prepare args device memory and run  */
    uint8_t* outTensorAddr = nullptr;
    machine::GetRA()->AllocDevAddr(&outTensorAddr, capacity3 * sizeof(float));

    std::vector<void*> opArgsRun = {x_ptr, zero_ptr, y_ptr, outTensorAddr};
    // std::vector<void*> opArgsRun = {x_ptr, zero_ptr, outTensorAddr};
    TileFwkRunAsync(handle, workspaceAddr, machine::GetRA()->GetStreamAICPU(),  opArgsRun);
    int rc = rtStreamSynchronize(machine::GetRA()->GetStreamAICPU());
    if (rc < 0) {
        ASSERT(false);
        ALOG_INFO_F("FA function aicpu stream sync failed");
    }

    std::cout << "======capacity3 size:" << capacity3 << std::endl;
    std::vector<float> golden(capacity3);
    std::vector<float> dev_res(capacity3);
    machine::GetRA()->CopyFromTensor((uint8_t *)dev_res.data(), (uint8_t *)outTensorAddr, capacity3 * sizeof(float));
    readInput(GetGoldenDir() + "/z_golden.bin", golden);

    int ret = resultCmp(golden, dev_res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(MoegateOnBoardTest, test_moegate_graph3_graph4_case_32_1_7168) {
    int B = 8;
    int S = 1;

    int topkGroup = 4;
    int nGroup = 8;
    int nRoutedExperts = 256;
    int numExpertsPerTopk = 8;

    std::vector<int64_t> shape_topk_group =  {B*S, topkGroup}; //[8,4]
    std::vector<int64_t> shape_group_mask =  {B*S, nGroup}; // [8,8]
    std::vector<int64_t> shape_score_for_choice = {B*S, nRoutedExperts}; //[8,256]
    std::vector<int64_t> shape_score = {B*S, nRoutedExperts};
    std::vector<int64_t> shape_topk_weight = {B*S, numExpertsPerTopk};

    int capacity0 = shape_topk_group[0] * shape_topk_group[1];
    int capacity1 = shape_score_for_choice[0] * shape_score_for_choice[1];
    int capacity2 = shape_group_mask[0] * shape_group_mask[1];
    int capacity3 = shape_score[0] * shape_score[1];
    int capacity4 = shape_topk_weight[0] * shape_topk_weight[1];

    aclInit(nullptr);
    rtSetDevice(npu::tile_fwk::stubs::DeviceStub::GetCurrentDeviceId());
    TileFwkInit("");

    TileShape::Current().SetVecTile(64, 64);

    void *x_ptr = readToDev(GetGoldenDir() + "/group_idx.bin", capacity0);
    void *y_ptr = readToDev(GetGoldenDir() + "/scores_for_choice.bin", capacity1);
    void *zero_ptr = readToDev(GetGoldenDir() + "/group_mask_zero.bin", capacity2);
    void *z_ptr = readToDev(GetGoldenDir() + "/score.bin", capacity3);

    Tensor group_idx(DataType::DT_INT32, shape_topk_group, (uint8_t *)x_ptr, "group_idx");
    Tensor group_mask(DataType::DT_FP32, shape_group_mask, (uint8_t *)zero_ptr, "group_mask");
    Tensor scores_for_choice(DataType::DT_FP32, shape_score_for_choice, (uint8_t *)y_ptr, "scores_for_choice");
    Tensor scores(DataType::DT_FP32, shape_score, (uint8_t *)z_ptr, "scores");

    Tensor outputTensor(DataType::DT_FP32, shape_topk_weight, "out_topk_weight");

    /* torch capture */
    TileFwkBeginFunction("MOEGATE_GRAPH3_GRAPH4", {group_idx, group_mask, scores_for_choice, scores, outputTensor});
    {
        auto tmp_group_mask = ScatterElement(group_mask, group_idx, Element(DataType::DT_FP32, 1.0), 1); // (b*s, nGroup)
        Tensor group_mask_new = Reshape(tmp_group_mask, {B*S, nGroup, 1}); // (b*s, nGroup, 1)

        TileShape::Current().SetVecTile(16, 16, 32); // 此处如果用(16 16 32)的tile 会有精度问题
        Tensor score_mask = Expand(group_mask_new, {B*S, nGroup, nRoutedExperts / nGroup}); // [b*s,nGroup,1] -> [b*s,nGroup,32]
        auto score_mask_new = Reshape(score_mask, {B*S, nGroup*nRoutedExperts / nGroup}); // (b*s,-1) [b*s,256]

        TileShape::Current().SetVecTile(64, 64);
        auto score1 = Mul(scores_for_choice, score_mask_new);
        auto score2 = MulS(LogicalNot(score_mask_new), Element(DataType::DT_FP32, -3.4e+38f));
        auto tmp_scores = Add(score1, score2);

        auto topk_idx = std::get<1>(TopK(tmp_scores, numExpertsPerTopk, -1)); // [b*s,256]->[b*s,8]
        auto topk_weight = GatherElement(scores, topk_idx, 1); // [b*s,8]
        auto topk_weight_sum = RowSumSingle(topk_weight, 1);      // [b*s,8]->[b*s,1]
        auto denominator = AddS(topk_weight_sum, Element(DataType::DT_FP32, 1e-20f)); // [b*s,1]
        outputTensor = Div(topk_weight, denominator); // [b*s,8]
    }
    TileFwkEndFunction();

    /* torch compile */
    void* handle = TileFwkCompile();

    /* torch prepare workspace memory */
    uint8_t* workspaceAddr = nullptr;
    uint64_t workspaceSize = 0;
    TileFwkGetWorkspaceSize(handle, &workspaceSize);
    machine::GetRA()->AllocDevAddr(&workspaceAddr, workspaceSize);

    /* torch prepare args device memory and run  */
    uint8_t* outTensorAddr = nullptr;
    machine::GetRA()->AllocDevAddr(&outTensorAddr, capacity4 * sizeof(float));

    std::vector<void*> opArgsRun = {x_ptr, zero_ptr, y_ptr, z_ptr, outTensorAddr};
    TileFwkRunAsync(handle, workspaceAddr, machine::GetRA()->GetStreamAICPU(),  opArgsRun);
    int rc = rtStreamSynchronize(machine::GetRA()->GetStreamAICPU());
    if (rc < 0) {
        ASSERT(false);
        ALOG_INFO_F("FA function aicpu stream sync failed");
    }

    std::cout << "======capacity4 size:" << capacity4 << std::endl;
    std::vector<float> golden(capacity4);
    std::vector<float> dev_res(capacity4);
    machine::GetRA()->CopyFromTensor((uint8_t *)dev_res.data(), (uint8_t *)outTensorAddr, capacity4 * sizeof(float));
    readInput(GetGoldenDir() + "/z_golden.bin", golden);

    int ret = resultCmp(golden, dev_res, 0.001f);
    EXPECT_EQ(ret, true);
}
