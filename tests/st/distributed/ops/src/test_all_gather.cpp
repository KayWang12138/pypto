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
 * \file test_all_gather.cpp
 * \brief
 */

#include "distributed_op_test_suite.h"
#include "distributed_op_test_common.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_manager.h"
#include "tilefwk/data_type.h"
#include "test_dev_func_runner.h"

namespace npu::tile_fwk {
namespace Distributed {

void TestAllGather(OpTestParam &testParam)
{
    constexpr size_t paramsSize = 3;
    auto [M, N, typeNum] = GetParams<paramsSize>(GetGoldenDir() + "/params.bin");

    DataType dType = GetDataTypeNum(typeNum);
    size_t dTypeSize = BytesOf(dType);

    int32_t inSize = M * N;
    int32_t outM = M * testParam.rankSize;
    int32_t outSize = outM * N;
    int32_t outByteSize = dTypeSize * outSize;
    uint8_t *outPtr = allocDevAddr(outByteSize);
    ALOG_INFO_F("before ALL_GATHER [%d, %d], rankSize=%d, outPtr=%p", M, N, testParam.rankSize, outPtr);
    PROGRAM("ALL_GATHER") {
        std::vector<int64_t> inShape = {M, N};
        std::vector<int64_t> outShape = {outM, N};

        void *xPtr = readToDev(GetGoldenDir() + "/input_rank_"+ std::to_string(testParam.rankId) + ".bin",
            inSize * dTypeSize /sizeof(float));

        Tensor in(dType, inShape, (uint8_t *)xPtr, "in");
        Tensor out(dType, outShape, outPtr, "out");

        ConfigManager::Instance();
        FunctionConfig funConfig = {.funcType = FunctionType::STATIC};
        FUNCTION("AllGather", funConfig, {in, out}) {
            Program::GetInstance().GetTileShape().SetDistTileShapes(
                {outM / testParam.rankSize, testParam.rankSize, 0},
                {N / testParam.rankSize, testParam.rankSize, 0},
                {1, testParam.rankSize, 0});
            Program::GetInstance().GetTileShape().SpecifyStaticRankId(testParam.rankId);
            out = AllGather(in, testParam.group);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    EXPECT_TRUE(CompareWithGolden<uint8_t *>(dType, "/output_rank_", outSize, outPtr, testParam));
}

void TestAllGatherEx(OpTestParam &testParam)
{
    constexpr size_t paramsSize = 3;
    auto [M, N, typeNum] = GetParams<paramsSize>(GetGoldenDir() + "/params.bin");

    DataType dType = GetDataTypeNum(typeNum);
    size_t dTypeSize = BytesOf(dType);

    int32_t size = M * N;
    int32_t outByteSize = dTypeSize * size;
    std::vector<uint8_t *> outPtrs;

    PROGRAM("ALL_GATHER_EX") {
        std::vector<int64_t> shape = {M, N};

        void *xPtr = readToDev(GetGoldenDir() + "/input_rank_"+ std::to_string(testParam.rankId) + ".bin",
            outByteSize /sizeof(float));

        Tensor in(dType, shape, (uint8_t *)xPtr, "in");
        std::vector<Tensor> outs;
        for (int32_t i = 0; i < testParam.rankSize; ++i) {
            std::string tensorName = "out" + std::to_string(i);
            uint8_t *outPtr = allocDevAddr(outByteSize);
            outs.push_back(Tensor(dType, shape, (uint8_t *)outPtr, tensorName));
            outPtrs.emplace_back(outPtr);
        }
        std::vector<std::reference_wrapper<Tensor>> tensorParams(outs.begin(), outs.end());
        tensorParams.emplace_back(in);

        ConfigManager::Instance();
        FunctionConfig funConfig = {.funcType = FunctionType::STATIC};
        FUNCTION("AllGather_Ex", funConfig, tensorParams) {
            Program::GetInstance().GetTileShape().SetDistTileShapes(
                {M / 2, 2, 0},
                {N / 2, 2, 0},
                {1, testParam.rankSize, 0});
            Program::GetInstance().GetTileShape().SpecifyStaticRankId(testParam.rankId);
            Distributed::AllGather(in, outs, testParam.group);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    EXPECT_TRUE(outPtrs.size() > 0);
    EXPECT_TRUE(CompareWithGolden<std::vector<uint8_t *>>(dType, "/output_rank_", size * testParam.rankSize, outPtrs,
        testParam));
}

void TestDynAllGather(OpTestParam &testParam)
{
    constexpr size_t paramsSize = 3;
    auto [M, N, typeNum] = GetParams<paramsSize>(GetGoldenDir() + "/params.bin");

    DataType dType = GetDataTypeNum(typeNum);

    int32_t outSize = M * N * testParam.rankSize;

    Shape shape{M, N};
    Tensor in(dType, shape, "in");

    std::vector<int32_t> inPtr = ReadToVector<int32_t>(GetGoldenDir() + "/input_rank_" + std::to_string(testParam.rankId) + ".bin", shape);

    Shape outShape{testParam.rankSize * M, N};
    Tensor out(dType, outShape, "out");

    AllGatherDyn(in, testParam.group, out);

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<int32_t>(in, inPtr)
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateTensorZero(out)
    });

    auto funcOp = Program::GetInstance().GetLastFunction()->GetDyndevAttribute();
    auto hcclContext = GetHcclContext({std::string(testParam.group)});
    FuncRunnerConfig config;
    config.runModel = false;
    config.hcclContext = hcclContext;
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction(), config);

    auto outPut = ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(CompareWithGolden<uint8_t*>(dType, "/output_rank_", outSize, outPut->GetDevPtr(), testParam));
}

} // namespace Distributed
} // namespace npu::tile_fwk
