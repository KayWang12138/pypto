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
 * \file test_allreduce.cpp
 * \brief
 */

#include "distributed_op_test_common.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_manager.h"
#include "tilefwk/data_type.h"
#include "test_dev_func_runner.h"

namespace npu::tile_fwk {
namespace Distributed {

struct AllReduceTestConfig {
    DataType inDtype;
    DataType outDtype;
    TileOpFormat inFormat;
    TileOpFormat outFormat;
    Shape inShape;
    Shape outShape;
    Shape tileShape;
    bool useTwoShot;
};

AllReduceTestConfig ParseAllReduceConfig(const nlohmann::json& testData)
{
    AllReduceTestConfig caseInfo;
    auto inTensor = testData["input_tensors"][0];
    auto outTensor = testData["output_tensors"][0];
    caseInfo.inShape = inTensor["shape"].get<Shape>();
    caseInfo.outShape = outTensor["shape"].get<Shape>();
    caseInfo.inDtype = GetDataTypeNum(GetDtypeNum(inTensor["dtype"].get<std::string>()));
    caseInfo.outDtype = GetDataTypeNum(GetDtypeNum(outTensor["dtype"].get<std::string>()));
    caseInfo.inFormat = StringToTileOpFormat(inTensor["format"].get<std::string>());
    caseInfo.outFormat = StringToTileOpFormat(outTensor["format"].get<std::string>());
    caseInfo.tileShape = testData["tile_shape"].get<Shape>();
    auto params = testData["params"];
    caseInfo.useTwoShot = params["use_two_shot"].get<bool>();
    return caseInfo;
}

template<typename T>
void TestAllReduce(OpTestParam &testParam, const nlohmann::json& testData)
{
    std::string goldenDir = GetGoldenDirPath(testData);
    auto caseInfo = ParseAllReduceConfig(testData);

    int32_t row = caseInfo.inShape[0];
    int32_t col = caseInfo.inShape[1];
    int32_t outSize = row * col;

    Tensor in(caseInfo.inDtype, caseInfo.inShape, "in", caseInfo.inFormat);
    Tensor out(caseInfo.outDtype, caseInfo.outShape, "out", caseInfo.outFormat);

    std::vector<T> inPtr = ReadToVector<T>(
        goldenDir + "/input_rank_" + std::to_string(testParam.rankId) + ".bin", caseInfo.inShape);

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<T>(in, inPtr),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateTensorZero(out),
    });
    int32_t rowPerRank = row;
    Shape shmemDataShape{1, rowPerRank, col};
    if (caseInfo.useTwoShot) {
        ASSERT(testParam.rankSize > 0) << "testParam.rankSize must be > 0, but got: " << testParam.rankSize;
        rowPerRank /= testParam.rankSize;
        shmemDataShape = {testParam.rankSize, rowPerRank, col};
    }
    FUNCTION("ALLREDUCE", {in}, {out}) {
        TileShape::Current().SetVecTile(caseInfo.tileShape);
        Tensor shmemData;
        Tensor shmemSignal;
        DataType shmemDataType = in.GetDataType();
        if ((shmemDataType == DT_BF16) || (shmemDataType == DT_FP16)) {
            shmemDataType = DT_FP32;
        }
        LOOP("CreateShmemTensor", FunctionType::DYNAMIC_LOOP, index, LoopRange(1)) {
            (void)index;
            CreateShmemData(testParam.group, testParam.rankSize, shmemDataType, shmemDataShape, shmemData);
            CreateShmemSignal(testParam.group, shmemData, shmemSignal);
        }
        if (caseInfo.useTwoShot) {
            TwoShotAllReduce(in, in, testParam.group, shmemData, shmemSignal, out);
        } else {
            OneShotAllReduce(in, in, testParam.group, shmemData, shmemSignal, out);
        }
    }
    RunTest();
    auto output = ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(CompareWithGolden<uint8_t*>(caseInfo.outDtype, goldenDir + "/output_rank_", outSize, output->GetDevPtr(), testParam));
}
template void TestAllReduce<int32_t>(OpTestParam &testParam, const nlohmann::json& testData);
template void TestAllReduce<float>(OpTestParam &testParam, const nlohmann::json& testData);
template void TestAllReduce<float16>(OpTestParam &testParam, const nlohmann::json& testData);
template void TestAllReduce<bfloat16>(OpTestParam &testParam, const nlohmann::json& testData);
} // namespace Distributed 
} // namespace npu::tile_fwk