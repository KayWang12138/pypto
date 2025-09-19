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
 * \file test_reduce_scatter.cpp
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

void TestReduceScatter(OpTestParam &testParam)
{
    constexpr size_t paramsSize = 3;
    auto [M, N, typeNum] = GetParams<paramsSize>(GetGoldenDir() + "/params.bin");

    ASSERT(M % testParam.rankSize == 0);

    DataType dType = GetDataTypeNum(typeNum);

    int32_t inSize = M * N;
    int32_t outM = M / testParam.rankSize;
    int32_t outSize = outM * N;
    size_t dTypeSize = BytesOf(dType);
    int32_t outByteSize = dTypeSize * outSize;
    uint8_t *outPtr = allocDevAddr(outByteSize);
    ALOG_INFO_F("before REDUCESCATTER [%d, %d], rankSize=%d, outPtr=%p", M, N, testParam.rankSize, outPtr);
    PROGRAM("REDUCESCATTER") {
        std::vector<int64_t> inShape = {M, N};
        std::vector<int64_t> outShape = {outM, N};

        void *xPtr = readToDev(GetGoldenDir() + "/input_rank_"+ std::to_string(testParam.rankId) + ".bin",
            inSize * dTypeSize / sizeof(float));

        Tensor in(dType, inShape,  (uint8_t *)xPtr, "in");
        Tensor out(dType, outShape, outPtr, "out");

        ConfigManager::Instance();

        FunctionConfig funConfig(FunctionType::STATIC);
        ;
        FUNCTION("REDUCESCATTER_F", funConfig, {in, out}) {
            TileShape::Current().SetDistTile({M / 2, 2, 0}, {N / 2, 2, 0}, {1, testParam.rankSize, 0});
            TileShape::Current().SetDistRankId(testParam.rankId);
            out = Distributed::ReduceScatter(in, testParam.group,
                npu::tile_fwk::Distributed::DistReduceType::DIST_REDUCE_ADD);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    EXPECT_TRUE(CompareWithGolden<uint8_t *>(dType, "/output_rank_", outSize, outPtr, testParam));
}

void TestReduceScatterEx(OpTestParam &testParam)
{
    constexpr size_t paramsSize = 3;
    auto [M, N, typeNum] = GetParams<paramsSize>(GetGoldenDir() + "/params.bin");

    ASSERT(M % testParam.rankSize == 0);

    DataType dType = GetDataTypeNum(typeNum);
    size_t dTypeSize = BytesOf(dType);

    int32_t outM = M / testParam.rankSize;
    int32_t outSize = outM * N;
    int32_t inSize = outSize;
    int32_t outByteSize = dTypeSize * outSize;
    uint8_t *outPtr = allocDevAddr(outByteSize);
    ALOG_INFO_F("before REDUCESCATTER [%d, %d], rankSize=%d, outPtr=%p", M, N, testParam.rankSize, outPtr);
    PROGRAM("REDUCESCATTEREX") {
        std::vector<int64_t> inShape = {outM, N};
        std::vector<int64_t> outShape = {outM, N};

        void *xPtr = readToDev(GetGoldenDir() + "/input_rank_"+ std::to_string(testParam.rankId) + ".bin",
            M * N * dTypeSize / sizeof(float));
        Tensor out(dType, outShape, outPtr, "out");

        std::vector<Tensor> inVec;
        for (int32_t i = 0; i < testParam.rankSize; i++) {
            std::string inName = "in" + std::to_string(i);
            Tensor inTensor(dType, inShape, (uint8_t *)xPtr + inSize * dTypeSize * i, inName);
            inVec.push_back(inTensor);
        }
        std::vector<std::reference_wrapper<Tensor>> paras(inVec.begin(), inVec.end());
        paras.emplace_back(out);
        ConfigManager::Instance();

        FunctionConfig funConfig(FunctionType::STATIC);
        ;
        FUNCTION("REDUCESCATTER_EX", funConfig, paras) {
            TileShape::Current().SetDistTile({M / 2, 2, 0}, {N / 2, 2, 0}, {1, testParam.rankSize, 0});
            TileShape::Current().SetDistRankId(testParam.rankId);
            out = Distributed::ReduceScatter(inVec, testParam.group,
                npu::tile_fwk::Distributed::DistReduceType::DIST_REDUCE_ADD);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    EXPECT_TRUE(CompareWithGolden<uint8_t *>(dType, "/output_rank_", outSize, outPtr, testParam));
}

void TestShmemReduceScatter(OpTestParam &testParam)
{
    constexpr size_t paramsSize = 3;
    auto [row, col, typeNum] = GetParams<paramsSize>(GetGoldenDir() + "/params.bin");
    int rowOut = row / testParam.rankSize;
    DataType dType = GetDataTypeNum(typeNum);
    using T = float;

    Tensor in(dType, {row, col}, "in");
    Tensor out(dType, {rowOut, col}, "out");

    std::vector<T> inData = ReadToVector<T>(
        GetGoldenDir() + "/input_rank_" + std::to_string(testParam.rankId) + ".bin", {row, col});

    FunctionConfig funcConfig;
    FUNCTION("ShmemReduceScatter", funcConfig, {in}, {out}) {
        LOOP("LOOP", FunctionType::DYNAMIC_LOOP, idx, LoopRange(1)) {
            (void)idx;
            TileShape::Current().SetDistTile({rowOut, 1, 0}, {col, 1, 0}, {1, testParam.rankSize, 0});
            TileShape::Current().SetDistRankId(testParam.rankId);
            out = Distributed::ShmemReduceScatter(in, testParam.group,
                npu::tile_fwk::Distributed::DistReduceType::DIST_REDUCE_ADD);
        }
    }

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<T>(in, inData),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<T>(out, 0),
    });

    auto funcOp = Program::GetInstance().GetLastFunction()->GetDyndevAttribute();
    auto hcclContext = GetHcclContext({std::string(testParam.group)});
    DeviceLauncherConfig config;
    config.runModel = false;
    config.hcclContext = hcclContext;
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction(), config);

    auto outPut = ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(CompareWithGolden<uint8_t*>(dType, "/output_rank_", rowOut * col, outPut->GetDevPtr(), testParam));
}

} // namespace Distributed
} // namespace npu::tile_fwk