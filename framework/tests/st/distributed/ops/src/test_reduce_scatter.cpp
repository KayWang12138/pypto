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
 * \file test_reduce_scatter.cpp
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

template<typename T>
void TestShmemReduceScatter(OpTestParam &testParam)
{
    ASSERT(testParam.rankSize > 0) << "worldSize should be more than 0.";
    constexpr size_t paramsSize = 5;
    auto [row, col, typeNum, tileRow, tileCol] = GetParams<paramsSize>(GetGoldenDir() + "/params.bin");
    int rowOut = row / testParam.rankSize;
    DataType dType = GetDataTypeNum(typeNum);
    Tensor in(dType, {row, col}, "in");
    Tensor out(dType, {rowOut, col}, "out");

    std::vector<T> inData = ReadToVector<T>(
        GetGoldenDir() + "/input_rank_" + std::to_string(testParam.rankId) + ".bin", {row, col});

    FUNCTION("ShmemReduceScatter", {in}, {out}) {
        TileShape::Current().SetVecTile({tileRow, tileCol});
        ReduceScatter(in, in, testParam.group, static_cast<uint32_t>(testParam.rankSize),
            npu::tile_fwk::Distributed::DistReduceType::DIST_REDUCE_ADD, out);
    }

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<T>(in, inData),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<T>(out, 0),
    });

    DeviceLauncherConfig config;
    config.runModel = false;
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction(), config);

    auto outPut = ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(CompareWithGolden<uint8_t*>(dType, "/output_rank_", rowOut * col, outPut->GetDevPtr(), testParam));
}

template<typename T>
void TestShmemReduceScatterParaWithShmem(OpTestParam &testParam)
{
    ASSERT(testParam.rankSize > 0) << "worldSize should be more than 0.";
    constexpr size_t paramsSize = 5;
    auto [row, col, typeNum, tileRow, tileCol] = GetParams<paramsSize>(GetGoldenDir() + "/params.bin");
    ASSERT(row % testParam.rankSize == 0) << "ReduceScatter constraint: row must be divisible by worldSize";
    int rowOut = row / testParam.rankSize;
    DataType dType = GetDataTypeNum(typeNum);
    Shape shmemDataShape = {1, rowOut, col};
    Tensor in(dType, {row, col}, "in");
    Tensor out(dType, {rowOut, col}, "out");
    std::vector<T> inData = ReadToVector<T>(
        GetGoldenDir() + "/input_rank_" + std::to_string(testParam.rankId) + ".bin", {row, col});

    FUNCTION("ShmemReduceScatter", {in}, {out}) {
        TileShape::Current().SetVecTile({tileRow, tileCol});
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
        Tensor predToken(DT_INT32, {1, 1}, "predToken");
        ReduceScatter(in, in, testParam.group, shmemData, shmemSignal, DistReduceType::DIST_REDUCE_ADD, out);
    }
    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<T>(in, inData),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<T>(out, 0),
    });
    DeviceLauncherConfig config;
    config.runModel = false;
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction(), config);
    auto outPut = ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(CompareWithGolden<uint8_t*>(dType, "/output_rank_", rowOut * col, outPut->GetDevPtr(), testParam));
}

template void TestShmemReduceScatter<int32_t>(OpTestParam &testParam);
template void TestShmemReduceScatter<float>(OpTestParam &testParam);
template void TestShmemReduceScatter<float16>(OpTestParam &testParam);
template void TestShmemReduceScatter<bfloat16>(OpTestParam &testParam);
template void TestShmemReduceScatterParaWithShmem<int32_t>(OpTestParam &testParam);
template void TestShmemReduceScatterParaWithShmem<float>(OpTestParam &testParam);
template void TestShmemReduceScatterParaWithShmem<float16>(OpTestParam &testParam);
template void TestShmemReduceScatterParaWithShmem<bfloat16>(OpTestParam &testParam);
} // namespace Distributed
} // namespace npu::tile_fwk