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
 * \file test_moe_combine.cpp
 * \brief
 */

#include "distributed_op_test_common.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_manager.h"
#include "tilefwk/data_type.h"
#include "test_dev_func_runner.h"

namespace npu::tile_fwk::Distributed {

struct MoeDistributedCombineTestConfig {
    DataType inDtype;
    DataType outDtype;
    int32_t batchSize;
    int32_t hiddenSize;
    int32_t moeExpertNum;
    int32_t topK;
};

MoeDistributedCombineTestConfig ParseMoeDistributedCombineConfig(const nlohmann::json& testData)
{
    MoeDistributedCombineTestConfig caseInfo;
    auto inTensor = testData["input_tensors"][0];
    auto outTensor = testData["output_tensors"][0];
    caseInfo.inDtype = GetDataTypeNum(GetDtypeNum(inTensor["dtype"].get<std::string>()));
    caseInfo.outDtype = GetDataTypeNum(GetDtypeNum(outTensor["dtype"].get<std::string>()));
    auto params = testData["params"].get<int32_t>();
    caseInfo.batchSize = params["batch_size"].get<int32_t>();
    caseInfo.hiddenSize = params["hidden_size"].get<int32_t>();
    caseInfo.moeExpertNum = params["routed_expert_num"].get<int32_t>();
    caseInfo.topK = params["top_k"].get<int32_t>();
    return caseInfo;
}

template<typename T>
void TestMoeDistributedCombine(OpTestParam& testParam, const nlohmann::json& testData)
{
    std::string goldenDir = GetGoldenDirPath(testData);
    auto caseInfo = ParseMoeDistributedCombineConfig(testData);

    int64_t row = std::min(caseInfo.topK * caseInfo.batchSize * testParam.rankSize, caseInfo.batchSize * caseInfo.moeExpertNum);
    Shape inShape{row, caseInfo.hiddenSize};
    Shape combineInfoShape{row, 3};
    Shape recvCountsShape{1};
    Shape scaleShape{caseInfo.batchSize, caseInfo.topK};
    Shape outShape{caseInfo.batchSize, caseInfo.hiddenSize};

    Tensor expandX(caseInfo.inDtype, inShape, "expandX");
    Tensor assistInfoForCombine(DataType::DT_INT32, combineInfoShape, "assistInfoForCombine");
    Tensor recvCounts(DataType::DT_INT32, recvCountsShape, "recvCounts");
    Tensor expertScales(DataType::DT_FP32, scaleShape, "expertScales");
    Tensor out(caseInfo.outDtype, outShape, "out");

    std::string dispatchPath = goldenDir + "/dispatch";
    std::vector<T> expandXPtr = ReadToVector<T>(
        dispatchPath + "/y_rank_" + std::to_string(testParam.rankId) + ".bin", inShape);
    std::vector<int32_t> assistInfoForCombinePtr = ReadToVector<int32_t>(
        dispatchPath + "/combine_info_rank_" + std::to_string(testParam.rankId) + ".bin", combineInfoShape);
    std::vector<int32_t> recvCountsPtr = ReadToVector<int32_t>(
        dispatchPath + "/recv_counts_rank_" + std::to_string(testParam.rankId) + ".bin", recvCountsShape);
    std::vector<float> expertScalesPtr = ReadToVector<float>(
        dispatchPath + "/scale_rank_" + std::to_string(testParam.rankId) + ".bin", scaleShape);

    FUNCTION("MoeDistributedCombineMain", {expandX, assistInfoForCombine, recvCounts, expertScales}, {out}) {
        MoeDistributedCombine(expandX, assistInfoForCombine, recvCounts, expertScales, testParam.group,
            testParam.rankSize, caseInfo.moeExpertNum, 0, 0, out);
    }

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<T>(expandX, expandXPtr),
        RawTensorData::CreateTensor<int32_t>(assistInfoForCombine, assistInfoForCombinePtr),
        RawTensorData::CreateTensor<int32_t>(recvCounts, recvCountsPtr),
        RawTensorData::CreateTensor<float>(expertScales, expertScalesPtr)
    });
    ProgramData::GetInstance().AppendOutputs({RawTensorData::CreateTensorZero(out)});

    DeviceLauncherConfig config;
    config.runModel = false;
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction(), config);

    int64_t outEleNum = outShape[0] * outShape[1];
    auto outPtr = ProgramData::GetInstance().GetOutputData(0)->GetDevPtr();
    if (caseInfo.batchSize == 256) { // bs=256 暂时不支持零误差一致
        EXPECT_TRUE(CompareWithGolden<uint8_t*>(caseInfo.outDtype, goldenDir + "/out_rank_", outEleNum, outPtr, testParam));
    } else {
        EXPECT_TRUE(CompareWithGolden<uint8_t*>(caseInfo.outDtype, goldenDir + "/out_rank_", outEleNum, outPtr, testParam, 0));
    }
}

template void TestMoeDistributedCombine<int32_t>(OpTestParam& testParam, const nlohmann::json& testData);
template void TestMoeDistributedCombine<float>(OpTestParam& testParam, const nlohmann::json& testData);
template void TestMoeDistributedCombine<float16>(OpTestParam& testParam, const nlohmann::json& testData);
template void TestMoeDistributedCombine<bfloat16>(OpTestParam& testParam, const nlohmann::json& testData);

} // namespace tile_fwk::Distributed