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
 * \brief System tests for AllReduce variants (OneShot base, v10, v10_pipe_ge; TwoShot base).
 *        Each variant runs against golden data for numerical correctness.
 */

#include "distributed_op_test_common.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_manager.h"
#include "tilefwk/data_type.h"
#include "test_dev_func_runner.h"

namespace npu::tile_fwk {
namespace Distributed {

namespace {

void InitAllReduceShmemTensor(const Tensor& in, const OpTestParam& testParam,
    const Shape& shmemDataShape, ShmemTensor& shmemTensor)
{
    DataType shmemDataType = in.GetDataType();
    if ((shmemDataType == DT_BF16) || (shmemDataType == DT_FP16)) {
        shmemDataType = DT_FP32;
    }
    LOOP("CreateShmemTensor", FunctionType::DYNAMIC_LOOP, index, LoopRange(1)) {
        (void)index;
        CreateShmemTensor(testParam.group, testParam.rankSize, shmemDataType, shmemDataShape, shmemTensor);
    }
}

} // namespace

template<typename T>
void TestAllReduce(OpTestParam &testParam, std::string &goldenDir)
{
    constexpr size_t paramsSize = 6;
    auto [row, col, typeNum, tileRow, tileCol, useTwoShot] = GetParams<paramsSize>(goldenDir + "/params.bin");
    DataType dType = GetDataTypeNum(typeNum);

    int32_t outSize = row * col;

    Shape shape{row, col};
    Tensor in(dType, shape, "in");
    Tensor out(dType, shape, "out");

    std::vector<T> inPtr = ReadToVector<T>(
        goldenDir + "/input_rank_" + std::to_string(testParam.rankId) + ".bin", {row, col});

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<T>(in, inPtr),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateTensorZero(out),
    });
    int32_t rowPerRank = row;
    Shape shmemDataShape{1, rowPerRank, col};
    if (useTwoShot) {
        CHECK(testParam.rankSize > 0) << "testParam.rankSize must be > 0, but got: " << testParam.rankSize;
        rowPerRank /= testParam.rankSize;
        shmemDataShape = {testParam.rankSize, rowPerRank, col};
    }
    FUNCTION("ALLREDUCE", {in}, {out}) {
        TileShape::Current().SetVecTile({tileRow, tileCol});
        DataType shmemDataType = in.GetDataType();
        if ((shmemDataType == DT_BF16) || (shmemDataType == DT_FP16)) {
            shmemDataType = DT_FP32;
        }
        ShmemTensor shmemTensor;
        LOOP("CreateShmemTensor", FunctionType::DYNAMIC_LOOP, index, LoopRange(1)) {
            (void)index;
            CreateShmemTensor(testParam.group, testParam.rankSize, shmemDataType, shmemDataShape, shmemTensor);
        }
        if (useTwoShot) {
            TwoShotAllReduce(in, in, shmemTensor, out);
        } else {
            OneShotAllReduce(in, in, shmemTensor, out);
        }
    }
    // DumpAllIR(useTwoShot ? "TwoShot Base" : "OneShot Base");
    if (useTwoShot) {
        VerifyTwoShotAllReduceIR("ALLREDUCE", testParam.rankSize);
    } else {
        VerifyOneShotAllReduceIR("ALLREDUCE", testParam.rankSize);
    }
    RunTest();
    auto output = ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(CompareWithGolden<uint8_t*>(dType, goldenDir + "/output_rank_", outSize, output->GetDevPtr(), testParam));

}

template void TestAllReduce<int32_t>(OpTestParam &testParam, std::string &goldenDir);
template void TestAllReduce<float>(OpTestParam &testParam, std::string &goldenDir);
template void TestAllReduce<float16>(OpTestParam &testParam, std::string& goldenDir);
template void TestAllReduce<bfloat16>(OpTestParam &testParam, std::string& goldenDir);

// v10: SHMEM-only, per-group GE signaling via OneShotCommunicatorV5.
// Sender increments the full-tile counter once per group (ascending order).
// Receiver issues a fresh GE wait per group: threshold = (groupId+1)*worldSize.
// clearSignal=false keeps the counter monotonic so later groups' waits remain valid.
template<typename T>
void TestAllReduce_v10(OpTestParam &testParam, std::string &goldenDir)
{
    constexpr size_t paramsSize = 6;
    auto [row, col, typeNum, tileRow, tileCol, useTwoShot] = GetParams<paramsSize>(goldenDir + "/params.bin");
    (void)useTwoShot;
    DataType dType = GetDataTypeNum(typeNum);

    int32_t outSize = row * col;
    Shape shape{row, col};
    Tensor in(dType, shape, "in");
    Tensor out(dType, shape, "out");

    std::vector<T> inPtr = ReadToVector<T>(
        goldenDir + "/input_rank_" + std::to_string(testParam.rankId) + ".bin", {row, col});

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<T>(in, inPtr),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateTensorZero(out),
    });

    uint32_t chunkCount = static_cast<uint32_t>(row);  // one chunk per row: maximum granularity
    uint32_t chunksPerSignal = 1;                        // one chunk per signal group: maximum granularity

    Shape shmemDataShape{1, row, col};
    FUNCTION("ALLREDUCE_V10", {in}, {out}) {
        TileShape::Current().SetVecTile({tileRow, tileCol});
        ShmemTensor shmemTensor;
        DataType shmemDataType = in.GetDataType();
        if ((shmemDataType == DT_BF16) || (shmemDataType == DT_FP16)) {
            shmemDataType = DT_FP32;
        }
        LOOP("CreateShmemTensor", FunctionType::DYNAMIC_LOOP, index, LoopRange(1)) {
            (void)index;
            CreateShmemTensor(testParam.group, testParam.rankSize, shmemDataType, shmemDataShape, shmemTensor);
        }

        OneShotAllReduce_v10(in, in, shmemTensor, out, chunkCount, chunksPerSignal);
    }
    RunTest();
    auto output = ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(CompareWithGolden<uint8_t*>(dType, goldenDir + "/output_rank_", outSize, output->GetDevPtr(), testParam));
}

template void TestAllReduce_v10<int32_t>(OpTestParam &testParam, std::string &goldenDir);
template void TestAllReduce_v10<float>(OpTestParam &testParam, std::string &goldenDir);
template void TestAllReduce_v10<float16>(OpTestParam &testParam, std::string& goldenDir);
template void TestAllReduce_v10<bfloat16>(OpTestParam &testParam, std::string& goldenDir);

// v10_pipe_ge: SHMEM-only, pipelined GE signaling variant.
// Uses per-group signal with GE thresholds and per-group receive pipelining.
template<typename T>
void TestAllReduce_v10_pipe_ge(OpTestParam &testParam, std::string &goldenDir)
{
    constexpr size_t paramsSize = 6;
    auto [row, col, typeNum, tileRow, tileCol, useTwoShot] = GetParams<paramsSize>(goldenDir + "/params.bin");
    (void)useTwoShot;
    DataType dType = GetDataTypeNum(typeNum);

    int32_t outSize = row * col;
    Shape shape{row, col};
    Tensor in(dType, shape, "in");
    Tensor out(dType, shape, "out");

    std::vector<T> inPtr = ReadToVector<T>(
        goldenDir + "/input_rank_" + std::to_string(testParam.rankId) + ".bin", {row, col});

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<T>(in, inPtr),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateTensorZero(out),
    });

    uint32_t chunkCount = static_cast<uint32_t>(row);
    uint32_t chunksPerSignal = 1;

    Shape shmemDataShape{1, row, col};
    FUNCTION("ALLREDUCE_V10_PIPE_GE", {in}, {out}) {
        TileShape::Current().SetVecTile({tileRow, tileCol});
        ShmemTensor shmemTensor;
        DataType shmemDataType = in.GetDataType();
        if ((shmemDataType == DT_BF16) || (shmemDataType == DT_FP16)) {
            shmemDataType = DT_FP32;
        }
        LOOP("CreateShmemTensor", FunctionType::DYNAMIC_LOOP, index, LoopRange(1)) {
            (void)index;
            CreateShmemTensor(testParam.group, testParam.rankSize, shmemDataType, shmemDataShape, shmemTensor);
        }

        OneShotAllReduce_v10_pipe_ge(in, in, shmemTensor, out, chunkCount, chunksPerSignal);
    }
    RunTest();
    auto output = ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(CompareWithGolden<uint8_t*>(dType, goldenDir + "/output_rank_", outSize, output->GetDevPtr(), testParam));
}

template void TestAllReduce_v10_pipe_ge<int32_t>(OpTestParam &testParam, std::string &goldenDir);
template void TestAllReduce_v10_pipe_ge<float>(OpTestParam &testParam, std::string &goldenDir);
template void TestAllReduce_v10_pipe_ge<float16>(OpTestParam &testParam, std::string& goldenDir);
template void TestAllReduce_v10_pipe_ge<bfloat16>(OpTestParam &testParam, std::string& goldenDir);

} // namespace Distributed
} // namespace npu::tile_fwk
