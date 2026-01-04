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
 * \file test_moe_dispatch.cpp
 * \brief
 */

#include "distributed_op_test_common.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_manager.h"
#include "tilefwk/data_type.h"
#include "test_dev_func_runner.h"

namespace npu::tile_fwk::Distributed {

void PrepareMoeDispatchParams(const OpTestParam& testParam, int32_t& batchSize, int32_t& hiddenSize,
    int32_t&totalExpertNum, int32_t&topK, DataType& dType,int32_t& expertNumPerRank, int32_t& expandXRowShape)
{
    constexpr size_t paramsSize = 5;
    auto [bs, h, routedNum, k, dtype_num] = GetParams<paramsSize>(GetGoldenDir() + "/params.bin");
    batchSize = bs;
    hiddenSize = h;
    totalExpertNum = routedNum;
    topK = k;
    dType = GetDataTypeNum(dtype_num);
    expertNumPerRank = totalExpertNum / testParam.rankSize;
    expandXRowShape = std::min(batchSize * topK * testParam.rankSize, batchSize * totalExpertNum);
}

void PrepareMoeDispatchInputAndOutPuts(Tensor& x, Tensor& expertIds, Tensor& expandX, Tensor& expertTokenNums, 
    Tensor& assistInfoForCombine, Tensor& recvCounts, DataType dType, int32_t batchSize, int32_t hiddenSize,
    int32_t topK, int32_t expandXRowShape, int32_t expertNumPerRank)
{
    x = Tensor(dType, {batchSize, hiddenSize}, "x");
    expertIds = Tensor(DataType::DT_INT32, {batchSize, topK}, "expertIds");
    expandX = Tensor(dType, {expandXRowShape, hiddenSize}, "expandX");
    expertTokenNums = Tensor(DataType::DT_INT32, {expertNumPerRank}, "expertTokenNums");
    assistInfoForCombine = Tensor(DataType::DT_INT32, {expandXRowShape, 3}, "assistInfoForCombine");
    recvCounts = Tensor(DataType::DT_INT32, {1}, "recvCounts");
}

void CheckMoeDispatchResult(const OpTestParam& testParam, DataType dType, 
    int64_t expandXEleNum, int64_t expertTokenNumsEleNum, int64_t assistInfoForCombineEleNum)
{
    auto expandXOutPut = ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(CompareWithGolden<uint8_t *>(dType, "/y_rank_", expandXEleNum, expandXOutPut->GetDevPtr(), testParam));
    auto expertTokenNumsOutPut = ProgramData::GetInstance().GetOutputData(1);
    EXPECT_TRUE(CompareWithGolden<uint8_t *>(DataType::DT_INT32, "/valid_count_rank_", expertTokenNumsEleNum, expertTokenNumsOutPut->GetDevPtr(), testParam));
    auto assistInfoForCombineOutPut = ProgramData::GetInstance().GetOutputData(2);
    EXPECT_TRUE(CompareWithGolden<uint8_t *>(DataType::DT_INT32, "/combine_info_rank_", assistInfoForCombineEleNum, assistInfoForCombineOutPut->GetDevPtr(), testParam));
    auto recvCountsOutPut = ProgramData::GetInstance().GetOutputData(3);
    EXPECT_TRUE(CompareWithGolden<uint8_t *>(DataType::DT_INT32, "/recv_counts_rank_", 1, recvCountsOutPut->GetDevPtr(), testParam));
}

void RunMoeDistributedDispatch(const OpTestParam& testParam, int32_t moeExpertNum, int32_t sharedExpertNum, int32_t sharedExpertRankNum, 
    Tensor& x, Tensor& expertIds, Tensor& expandX, Tensor& expertTokenNums, Tensor& assistInfoForCombine, Tensor& recvCounts)
{
    FUNCTION("MoeDispatch", {x, expertIds}, {expandX, expertTokenNums, assistInfoForCombine, recvCounts}) {
        Distributed::MoeDistributedDispatch(x, expertIds, testParam.group, testParam.rankSize, moeExpertNum, sharedExpertNum, 
            sharedExpertRankNum, expandX, assistInfoForCombine, expertTokenNums, recvCounts);
    }
    using T = npu::tile_fwk::bfloat16;
    std::string xPath = GetGoldenDir() + "/x_rank_" + std::to_string(testParam.rankId) + ".bin";
    std::vector<T> xPtr = ReadToVector<T>(xPath, x.GetShape());
    std::string expertIdsPath = GetGoldenDir() + "/expert_ids_rank_" + std::to_string(testParam.rankId) + ".bin";
    std::vector<int32_t> expertIdsPtr = ReadToVector<int32_t>(expertIdsPath, expertIds.GetShape());

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<T>(x, xPtr),
        RawTensorData::CreateTensor<int32_t>(expertIds, expertIdsPtr),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateTensorZero(expandX),
        RawTensorData::CreateTensorZero(expertTokenNums),
        RawTensorData::CreateTensorZero(assistInfoForCombine),
        RawTensorData::CreateTensorZero(recvCounts),
    });

    DeviceLauncherConfig config;
    config.runModel = false;
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction(), config);
}

void TestMoeDistributedDispatch(OpTestParam &testParam)
{
    int32_t batchSize;
    int32_t hiddenSize;
    int32_t totalExpertNum;
    int32_t topK;
    int32_t expertNumPerRank;
    int32_t expandXRowShape;
    DataType dType;
    PrepareMoeDispatchParams(testParam, batchSize, hiddenSize, totalExpertNum, topK, dType, expertNumPerRank, expandXRowShape);
    int32_t moeExpertNum = totalExpertNum;
    int32_t sharedExpertRankNum = 0;
    int32_t sharedExpertNum = 0;
    Tensor x;
    Tensor expertIds;
    Tensor expandX;
    Tensor expertTokenNums;
    Tensor assistInfoForCombine;
    Tensor recvCounts;
    PrepareMoeDispatchInputAndOutPuts(x, expertIds, expandX, expertTokenNums, assistInfoForCombine,
        recvCounts, dType, batchSize, hiddenSize, topK, expandXRowShape, expertNumPerRank);
    RunMoeDistributedDispatch(testParam, moeExpertNum, sharedExpertNum, sharedExpertRankNum, 
        x, expertIds, expandX, expertTokenNums, assistInfoForCombine, recvCounts);
    int64_t expandXEleNum = expandX.GetShape(0) * expandX.GetShape(1);
    int64_t expertTokenNumsEleNum = expertTokenNums.GetShape(0);
    int64_t assistInfoForCombineEleNum = assistInfoForCombine.GetShape(0) * assistInfoForCombine.GetShape(1);
    CheckMoeDispatchResult(testParam, dType, expandXEleNum, expertTokenNumsEleNum, assistInfoForCombineEleNum);
}

} // namespace npu::tile_fwk::Distributed
