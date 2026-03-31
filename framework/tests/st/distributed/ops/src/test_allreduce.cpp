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
 * \brief System tests for AllReduce variants (OneShot base, v2–v6; TwoShot base, v2–v5).
 *        Each variant runs against golden data for numerical correctness.
 *        IR equivalence tests verify all variants emit identical SHMEM opcode sequences.
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

// v2–v4: progressive refactorings of the base OneShot, same golden data.
template<typename T>
void TestAllReduce_v2(OpTestParam &testParam, std::string &goldenDir)
{
    constexpr size_t paramsSize = 6;
    auto [row, col, typeNum, tileRow, tileCol, useTwoShot] = GetParams<paramsSize>(goldenDir + "/params.bin");
    (void)useTwoShot; // v2 is always one-shot
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
    Shape shmemDataShape{1, row, col};
    FUNCTION("ALLREDUCE_V2", {in}, {out}) {
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
        OneShotAllReduce_v2(in, in, shmemTensor, out);
    }
    // DumpAllIR("OneShot v2");
    VerifyOneShotAllReduceIR("ALLREDUCE_V2", testParam.rankSize);
    RunTest();
    auto output = ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(CompareWithGolden<uint8_t*>(dType, goldenDir + "/output_rank_", outSize, output->GetDevPtr(), testParam));
}

// v9: SHMEM-only, chunked, tunable signal-to-data ratio, no Communicator class.
template<typename T>
void TestAllReduce_v9(OpTestParam &testParam, std::string &goldenDir)
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
    FUNCTION("ALLREDUCE_V9", {in}, {out}) {
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

        OneShotAllReduce_v9(in, in, shmemTensor, out, chunkCount, chunksPerSignal);
    }
    RunTest();
    auto output = ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(CompareWithGolden<uint8_t*>(dType, goldenDir + "/output_rank_", outSize, output->GetDevPtr(), testParam));
}

template void TestAllReduce_v2<int32_t>(OpTestParam &testParam, std::string &goldenDir);
template void TestAllReduce_v2<float>(OpTestParam &testParam, std::string &goldenDir);
template void TestAllReduce_v2<float16>(OpTestParam &testParam, std::string& goldenDir);
template void TestAllReduce_v2<bfloat16>(OpTestParam &testParam, std::string& goldenDir);
template void TestAllReduce_v9<int32_t>(OpTestParam &testParam, std::string &goldenDir);
template void TestAllReduce_v9<float>(OpTestParam &testParam, std::string &goldenDir);
template void TestAllReduce_v9<float16>(OpTestParam &testParam, std::string& goldenDir);
template void TestAllReduce_v9<bfloat16>(OpTestParam &testParam, std::string& goldenDir);

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

template<typename T>
void TestAllReduce_v4(OpTestParam &testParam, std::string &goldenDir)
{
    constexpr size_t paramsSize = 6;
    auto [row, col, typeNum, tileRow, tileCol, useTwoShot] = GetParams<paramsSize>(goldenDir + "/params.bin");
    (void)useTwoShot; // v4 is always one-shot
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
    Shape shmemDataShape{1, row, col};
    FUNCTION("ALLREDUCE_V4", {in}, {out}) {
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
        OneShotAllReduce_v4(in, in, shmemTensor, out);
    }
    VerifyOneShotAllReduceIR("ALLREDUCE_V4", testParam.rankSize);
    RunTest();
    auto output = ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(CompareWithGolden<uint8_t*>(dType, goldenDir + "/output_rank_", outSize, output->GetDevPtr(), testParam));
}

template void TestAllReduce_v4<int32_t>(OpTestParam &testParam, std::string &goldenDir);
template void TestAllReduce_v4<float>(OpTestParam &testParam, std::string &goldenDir);
template void TestAllReduce_v4<float16>(OpTestParam &testParam, std::string& goldenDir);
template void TestAllReduce_v4<bfloat16>(OpTestParam &testParam, std::string& goldenDir);

// v5: communicator-based API — caller creates the comm and does Pull externally.
template<typename T>
void TestAllReduce_v5(OpTestParam &testParam, std::string &goldenDir)
{
    constexpr size_t paramsSize = 6;
    auto [row, col, typeNum, tileRow, tileCol, useTwoShot] = GetParams<paramsSize>(goldenDir + "/params.bin");
    (void)useTwoShot; // v5 is always one-shot
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
    Shape shmemDataShape{1, row, col};
    FUNCTION("ALLREDUCE_V5", {in}, {out}) {
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
        auto waitToken = OneShotAllReduce_v5(in, in, shmemTensor);
        // Pull happens outside v5 — postprocessing by the caller
        SymbolicScalar thisRank = GetHcclRankId(shmemTensor.group);
        auto shmemDataLocal = ShmemView(shmemTensor, {1, row, col}, std::vector<SymbolicScalar>{0, 0, 0});
        out = ShmemGet(shmemDataLocal, thisRank, waitToken, in.GetDataType());
    }
    VerifyOneShotAllReduceIR("ALLREDUCE_V5", testParam.rankSize);
    RunTest();
    auto output = ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(CompareWithGolden<uint8_t*>(dType, goldenDir + "/output_rank_", outSize, output->GetDevPtr(), testParam));
}

template void TestAllReduce_v5<int32_t>(OpTestParam &testParam, std::string &goldenDir);
template void TestAllReduce_v5<float>(OpTestParam &testParam, std::string &goldenDir);
template void TestAllReduce_v5<float16>(OpTestParam &testParam, std::string& goldenDir);
template void TestAllReduce_v5<bfloat16>(OpTestParam &testParam, std::string& goldenDir);

// v6: scatter-only — the most decomposed form. Caller owns Wait + Pull,
// which is handy when you want to overlap local compute with the scatter.
template<typename T>
void TestAllReduce_v6(OpTestParam &testParam, std::string &goldenDir)
{
    constexpr size_t paramsSize = 6;
    auto [row, col, typeNum, tileRow, tileCol, useTwoShot] = GetParams<paramsSize>(goldenDir + "/params.bin");
    (void)useTwoShot; // v6 is always one-shot
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
    Shape shmemDataShape{1, row, col};
    FUNCTION("ALLREDUCE_V6", {in}, {out}) {
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
        // scatter only — fire off the Puts
        OneShotAllReduce_v6(in, in, shmemTensor);
        // caller-side: block until all signals land, then read back
        SymbolicScalar thisRank = GetHcclRankId(shmemTensor.group);
        auto shmemDataLocal = ShmemView(shmemTensor, {1, row, col}, std::vector<SymbolicScalar>{0, 0, 0});
        auto waitToken = ShmemWaitUntil(shmemDataLocal, thisRank, OpType::EQ,
            static_cast<int32_t>(shmemTensor.worldSize), true, in);
        out = ShmemGet(shmemDataLocal, thisRank, waitToken, in.GetDataType());
    }
    VerifyOneShotAllReduceIR("ALLREDUCE_V6", testParam.rankSize);
    RunTest();
    auto output = ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(CompareWithGolden<uint8_t*>(dType, goldenDir + "/output_rank_", outSize, output->GetDevPtr(), testParam));
}

template void TestAllReduce_v6<int32_t>(OpTestParam &testParam, std::string &goldenDir);
template void TestAllReduce_v6<float>(OpTestParam &testParam, std::string &goldenDir);
template void TestAllReduce_v6<float16>(OpTestParam &testParam, std::string& goldenDir);
template void TestAllReduce_v6<bfloat16>(OpTestParam &testParam, std::string& goldenDir);

// v7: chunked scatter with coarse per-rank signaling (tunable payloadChunkCount).
// Data movement is chunked; signaling uses one coarse signal per target rank and
// receiver-side cached wait token semantics.
template<typename T>
void TestAllReduce_v7(OpTestParam &testParam, std::string &goldenDir)
{
    constexpr size_t paramsSize = 6;
    auto [row, col, typeNum, tileRow, tileCol, useTwoShot] = GetParams<paramsSize>(goldenDir + "/params.bin");
    (void)useTwoShot; // v7 is one-shot only
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

    // Keep a single payloadChunkCount in ST to avoid cross-iteration
    // shmem clear/signal interactions under real NPU runtime.
    uint32_t payloadChunkCount = static_cast<uint32_t>(row >= 7 ? 7 : row);

    Shape shmemDataShape{1, row, col};
    FUNCTION("ALLREDUCE_V7", {in}, {out}) {
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

        OneShotCommunicatorV3 comm(shmemTensor, payloadChunkCount);
        OneShotAllReduce_v7(in, in, comm);
        for (uint32_t chunkId = 0; chunkId < comm.PayloadChunkCount(); ++chunkId) {
            auto waitToken = comm.WaitChunk(in, chunkId);
            auto reducedChunk = comm.PullChunk(waitToken, chunkId, in.GetDataType());
            Assemble(reducedChunk, {comm.ChunkStartRow(chunkId), 0}, out);
        }
    }
    RunTest();
    auto output = ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(CompareWithGolden<uint8_t*>(dType, goldenDir + "/output_rank_", outSize, output->GetDevPtr(), testParam));
}

template void TestAllReduce_v7<int32_t>(OpTestParam &testParam, std::string &goldenDir);
template void TestAllReduce_v7<float>(OpTestParam &testParam, std::string &goldenDir);
template void TestAllReduce_v7<float16>(OpTestParam &testParam, std::string& goldenDir);
template void TestAllReduce_v7<bfloat16>(OpTestParam &testParam, std::string& goldenDir);

// v8: chunked scatter with grouped assembly (tunable chunksPerSignal k).
// Current signaling remains coarse per-rank; grouping controls receiver-side
// chunk assembly cadence and loop structure.
template<typename T>
void TestAllReduce_v8(OpTestParam &testParam, std::string &goldenDir)
{
    constexpr size_t paramsSize = 6;
    auto [row, col, typeNum, tileRow, tileCol, useTwoShot] = GetParams<paramsSize>(goldenDir + "/params.bin");
    (void)useTwoShot; // v8 is one-shot only
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

    // 7 chunks guarantees a tail group when k=4 (7 % 4 != 0).
    uint32_t payloadChunkCount = static_cast<uint32_t>(row >= 7 ? 7 : row);
    ASSERT(payloadChunkCount > 0) << "payloadChunkCount must be > 0";
    // Keep one stable grouping in ST to avoid cross-iteration shmem interactions.
    uint32_t k = static_cast<uint32_t>(payloadChunkCount >= 4 ? 4 : payloadChunkCount);

    Shape shmemDataShape{1, row, col};
    FUNCTION("ALLREDUCE_V8", {in}, {out}) {
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

        OneShotCommunicatorV4 comm(shmemTensor, payloadChunkCount, k);
        OneShotAllReduce_v8(in, in, comm);
        for (uint32_t groupId = 0; groupId < comm.SignalGroupCount(); ++groupId) {
            auto waitToken = comm.WaitGroup(in, groupId);
            uint32_t begin = comm.GroupBeginChunk(groupId);
            uint32_t gSize = comm.GroupSize(groupId);
            for (uint32_t local = 0; local < gSize; ++local) {
                uint32_t chunkId = begin + local;
                auto reducedChunk = comm.PullChunk(waitToken, chunkId, in.GetDataType());
                Assemble(reducedChunk, {comm.ChunkStartRow(chunkId), 0}, out);
            }
        }
    }
    RunTest();
    auto output = ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(CompareWithGolden<uint8_t*>(dType, goldenDir + "/output_rank_", outSize, output->GetDevPtr(), testParam));
}

template void TestAllReduce_v8<int32_t>(OpTestParam &testParam, std::string &goldenDir);
template void TestAllReduce_v8<float>(OpTestParam &testParam, std::string &goldenDir);
template void TestAllReduce_v8<float16>(OpTestParam &testParam, std::string& goldenDir);
template void TestAllReduce_v8<bfloat16>(OpTestParam &testParam, std::string& goldenDir);

// TwoShot v2–v5 system tests — same golden data as TwoShotAllReduce
template<typename T>
void TestAllReduce_TwoShot_v2(OpTestParam &testParam, std::string &goldenDir)
{
    constexpr size_t paramsSize = 6;
    auto [row, col, typeNum, tileRow, tileCol, useTwoShot] = GetParams<paramsSize>(goldenDir + "/params.bin");
    (void)useTwoShot; // always two-shot
    DataType dType = GetDataTypeNum(typeNum);

    int32_t outSize = row * col;
    ASSERT(testParam.rankSize > 0) << "testParam.rankSize must be > 0, but got: " << testParam.rankSize;
    int32_t rowPerRank = row / testParam.rankSize;

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
    Shape shmemDataShape{testParam.rankSize, rowPerRank, col};
    FUNCTION("TWOSHOT_V2", {in}, {out}) {
        TileShape::Current().SetVecTile({tileRow, tileCol});
        ShmemTensor shmemTensor;
        InitAllReduceShmemTensor(in, testParam, shmemDataShape, shmemTensor);
        TwoShotAllReduce_v2(in, in, shmemTensor, out);
    }
    VerifyTwoShotAllReduceIR("TWOSHOT_V2", testParam.rankSize);
    RunTest();
    auto output = ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(CompareWithGolden<uint8_t*>(dType, goldenDir + "/output_rank_", outSize, output->GetDevPtr(), testParam));
}

template void TestAllReduce_TwoShot_v2<int32_t>(OpTestParam &testParam, std::string &goldenDir);
template void TestAllReduce_TwoShot_v2<float>(OpTestParam &testParam, std::string &goldenDir);
template void TestAllReduce_TwoShot_v2<float16>(OpTestParam &testParam, std::string& goldenDir);
template void TestAllReduce_TwoShot_v2<bfloat16>(OpTestParam &testParam, std::string& goldenDir);

template<typename T>
void TestAllReduce_TwoShot_v3(OpTestParam &testParam, std::string &goldenDir)
{
    constexpr size_t paramsSize = 6;
    auto [row, col, typeNum, tileRow, tileCol, useTwoShot] = GetParams<paramsSize>(goldenDir + "/params.bin");
    (void)useTwoShot;
    DataType dType = GetDataTypeNum(typeNum);

    int32_t outSize = row * col;
    ASSERT(testParam.rankSize > 0) << "testParam.rankSize must be > 0, but got: " << testParam.rankSize;
    int32_t rowPerRank = row / testParam.rankSize;

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
    Shape shmemDataShape{testParam.rankSize, rowPerRank, col};
    FUNCTION("TWOSHOT_V3", {in}, {out}) {
        TileShape::Current().SetVecTile({tileRow, tileCol});
        ShmemTensor shmemTensor;
        InitAllReduceShmemTensor(in, testParam, shmemDataShape, shmemTensor);
        TwoShotAllReduce_v3(in, in, shmemTensor, out);
    }
    VerifyTwoShotAllReduceIR("TWOSHOT_V3", testParam.rankSize);
    RunTest();
    auto output = ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(CompareWithGolden<uint8_t*>(dType, goldenDir + "/output_rank_", outSize, output->GetDevPtr(), testParam));
}

template void TestAllReduce_TwoShot_v3<int32_t>(OpTestParam &testParam, std::string &goldenDir);
template void TestAllReduce_TwoShot_v3<float>(OpTestParam &testParam, std::string &goldenDir);
template void TestAllReduce_TwoShot_v3<float16>(OpTestParam &testParam, std::string& goldenDir);
template void TestAllReduce_TwoShot_v3<bfloat16>(OpTestParam &testParam, std::string& goldenDir);

template<typename T>
void TestAllReduce_TwoShot_v4(OpTestParam &testParam, std::string &goldenDir)
{
    constexpr size_t paramsSize = 6;
    auto [row, col, typeNum, tileRow, tileCol, useTwoShot] = GetParams<paramsSize>(goldenDir + "/params.bin");
    (void)useTwoShot;
    DataType dType = GetDataTypeNum(typeNum);

    int32_t outSize = row * col;
    ASSERT(testParam.rankSize > 0) << "testParam.rankSize must be > 0, but got: " << testParam.rankSize;
    int32_t rowPerRank = row / testParam.rankSize;

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
    Shape shmemDataShape{testParam.rankSize, rowPerRank, col};
    FUNCTION("TWOSHOT_V4", {in}, {out}) {
        TileShape::Current().SetVecTile({tileRow, tileCol});
        ShmemTensor shmemTensor;
        InitAllReduceShmemTensor(in, testParam, shmemDataShape, shmemTensor);
        TwoShotAllReduce_v4(in, in, shmemTensor, out);
    }
    VerifyTwoShotAllReduceIR("TWOSHOT_V4", testParam.rankSize);
    RunTest();
    auto output = ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(CompareWithGolden<uint8_t*>(dType, goldenDir + "/output_rank_", outSize, output->GetDevPtr(), testParam));
}

template void TestAllReduce_TwoShot_v4<int32_t>(OpTestParam &testParam, std::string &goldenDir);
template void TestAllReduce_TwoShot_v4<float>(OpTestParam &testParam, std::string &goldenDir);
template void TestAllReduce_TwoShot_v4<float16>(OpTestParam &testParam, std::string& goldenDir);
template void TestAllReduce_TwoShot_v4<bfloat16>(OpTestParam &testParam, std::string& goldenDir);

template<typename T>
void TestAllReduce_TwoShot_v5(OpTestParam &testParam, std::string &goldenDir)
{
    constexpr size_t paramsSize = 6;
    auto [row, col, typeNum, tileRow, tileCol, useTwoShot] = GetParams<paramsSize>(goldenDir + "/params.bin");
    (void)useTwoShot;
    DataType dType = GetDataTypeNum(typeNum);

    int32_t outSize = row * col;
    ASSERT(testParam.rankSize > 0) << "testParam.rankSize must be > 0, but got: " << testParam.rankSize;
    int32_t rowPerRank = row / testParam.rankSize;

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
    Shape shmemDataShape{testParam.rankSize, rowPerRank, col};
    FUNCTION("TWOSHOT_V5", {in}, {out}) {
        TileShape::Current().SetVecTile({tileRow, tileCol});
        ShmemTensor shmemTensor;
        InitAllReduceShmemTensor(in, testParam, shmemDataShape, shmemTensor);
        TwoShotCommunicatorV2 comm(shmemTensor);
        TwoShotAllReduce_v5(in, in, comm, out);
    }
    VerifyTwoShotAllReduceIR("TWOSHOT_V5", testParam.rankSize);
    RunTest();
    auto output = ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(CompareWithGolden<uint8_t*>(dType, goldenDir + "/output_rank_", outSize, output->GetDevPtr(), testParam));
}

template void TestAllReduce_TwoShot_v5<int32_t>(OpTestParam &testParam, std::string &goldenDir);
template void TestAllReduce_TwoShot_v5<float>(OpTestParam &testParam, std::string &goldenDir);
template void TestAllReduce_TwoShot_v5<float16>(OpTestParam &testParam, std::string& goldenDir);
template void TestAllReduce_TwoShot_v5<bfloat16>(OpTestParam &testParam, std::string& goldenDir);

// OneShot IR Equivalence — the "did we break anything?" test.
// Builds base as golden, then checks v2–v6 (+v6_light) all emit the same SHMEM opcodes.
// Graph-only, no hardware execution; just making sure the IR is sane.
template<typename T>
void TestAllReduceIREquivalence(OpTestParam &testParam, std::string &goldenDir)
{
    constexpr size_t paramsSize = 6;
    auto [row, col, typeNum, tileRow, tileCol, useTwoShot] = GetParams<paramsSize>(goldenDir + "/params.bin");
    (void)useTwoShot;
    DataType dType = GetDataTypeNum(typeNum);

    Shape shape{row, col};
    Shape shmemDataShape{1, row, col};

    auto buildShmemTensorSetup = [&](Tensor& in, ShmemTensor& shmemTensor) {
        TileShape::Current().SetVecTile({tileRow, tileCol});
        DataType shmemDataType = in.GetDataType();
        if ((shmemDataType == DT_BF16) || (shmemDataType == DT_FP16)) {
            shmemDataType = DT_FP32;
        }
        LOOP("CreateShmemTensor", FunctionType::DYNAMIC_LOOP, index, LoopRange(1)) {
            (void)index;
            CreateShmemTensor(testParam.group, testParam.rankSize, shmemDataType, shmemDataShape, shmemTensor);
        }
    };

    // ── base (golden reference) ──
    std::vector<Opcode> irBase;
    {
        Tensor in(dType, shape, "in_base");
        Tensor out(dType, shape, "out_base");
        FUNCTION("IR_CHECK_BASE", {in}, {out}) {
            ShmemTensor shmemTensor;
            buildShmemTensorSetup(in, shmemTensor);
            OneShotAllReduce(in, in, shmemTensor, out);
        }
        irBase = ExtractShmemOpcodes("IR_CHECK_BASE");
        VerifyOneShotAllReduceIR("IR_CHECK_BASE", testParam.rankSize);
    }

    // ── v2 ──
    std::vector<Opcode> irV2;
    {
        Program::GetInstance().Reset();
        Tensor in(dType, shape, "in_v2");
        Tensor out(dType, shape, "out_v2");
        FUNCTION("IR_CHECK_V2", {in}, {out}) {
            ShmemTensor shmemTensor;
            buildShmemTensorSetup(in, shmemTensor);
            OneShotAllReduce_v2(in, in, shmemTensor, out);
        }
        irV2 = ExtractShmemOpcodes("IR_CHECK_V2");
        VerifyOneShotAllReduceIR("IR_CHECK_V2", testParam.rankSize);
    }

    // ── v4 ──
    std::vector<Opcode> irV4;
    {
        Program::GetInstance().Reset();
        Tensor in(dType, shape, "in_v4");
        Tensor out(dType, shape, "out_v4");
        FUNCTION("IR_CHECK_V4", {in}, {out}) {
            ShmemTensor shmemTensor;
            buildShmemTensorSetup(in, shmemTensor);
            OneShotAllReduce_v4(in, in, shmemTensor, out);
        }
        irV4 = ExtractShmemOpcodes("IR_CHECK_V4");
        VerifyOneShotAllReduceIR("IR_CHECK_V4", testParam.rankSize);
    }

    // ── v5 + Pull ──
    std::vector<Opcode> irV5;
    {
        Program::GetInstance().Reset();
        Tensor in(dType, shape, "in_v5");
        Tensor out(dType, shape, "out_v5");
        FUNCTION("IR_CHECK_V5", {in}, {out}) {
            ShmemTensor shmemTensor;
            buildShmemTensorSetup(in, shmemTensor);
            auto waitToken = OneShotAllReduce_v5(in, in, shmemTensor);
            SymbolicScalar thisRank = GetHcclRankId(shmemTensor.group);
            auto shmemDataLocal = ShmemView(shmemTensor, {1, row, col}, std::vector<SymbolicScalar>{0, 0, 0});
            out = ShmemGet(shmemDataLocal, thisRank, waitToken, in.GetDataType());
        }
        irV5 = ExtractShmemOpcodes("IR_CHECK_V5");
        VerifyOneShotAllReduceIR("IR_CHECK_V5", testParam.rankSize);
    }

    // ── v6: the fully decomposed path — scatter, Wait, Pull all explicit ──
    std::vector<Opcode> irV6;
    {
        Program::GetInstance().Reset();
        Tensor in(dType, shape, "in_v6");
        Tensor out(dType, shape, "out_v6");
        FUNCTION("IR_CHECK_V6", {in}, {out}) {
            ShmemTensor shmemTensor;
            buildShmemTensorSetup(in, shmemTensor);
            OneShotAllReduce_v6(in, in, shmemTensor);
            SymbolicScalar thisRank = GetHcclRankId(shmemTensor.group);
            auto shmemDataLocal = ShmemView(shmemTensor, {1, row, col}, std::vector<SymbolicScalar>{0, 0, 0});
            auto waitToken = ShmemWaitUntil(shmemDataLocal, thisRank, OpType::EQ,
                static_cast<int32_t>(shmemTensor.worldSize), true, in);
            out = ShmemGet(shmemDataLocal, thisRank, waitToken, in.GetDataType());
        }
        irV6 = ExtractShmemOpcodes("IR_CHECK_V6");
        VerifyOneShotAllReduceIR("IR_CHECK_V6", testParam.rankSize);
    }

    // ── Cross-variant equivalence: all must match the base (golden) ──
    EXPECT_EQ(irBase, irV2) << "OneShot base and v2 SHMEM opcode sequences differ";
    EXPECT_EQ(irBase, irV4) << "OneShot base and v4 SHMEM opcode sequences differ";
    EXPECT_EQ(irBase, irV5) << "OneShot base and v5+Pull SHMEM opcode sequences differ";
    EXPECT_EQ(irBase, irV6) << "OneShot base and v6+Pull SHMEM opcode sequences differ";
}

template void TestAllReduceIREquivalence<int32_t>(OpTestParam &testParam, std::string &goldenDir);
template void TestAllReduceIREquivalence<float>(OpTestParam &testParam, std::string &goldenDir);
template void TestAllReduceIREquivalence<float16>(OpTestParam &testParam, std::string& goldenDir);
template void TestAllReduceIREquivalence<bfloat16>(OpTestParam &testParam, std::string& goldenDir);

// TwoShot IR Equivalence — same idea as OneShot above, for the two-phase path.
// Golden = TwoShotAllReduce base; variants v2–v5 must match.
template<typename T>
void TestTwoShotAllReduceIREquivalence(OpTestParam &testParam, std::string &goldenDir)
{
    constexpr size_t paramsSize = 6;
    auto [row, col, typeNum, tileRow, tileCol, useTwoShot] = GetParams<paramsSize>(goldenDir + "/params.bin");
    (void)useTwoShot;
    DataType dType = GetDataTypeNum(typeNum);

    ASSERT(testParam.rankSize > 0) << "testParam.rankSize must be > 0, but got: " << testParam.rankSize;
    int32_t rowPerRank = row / testParam.rankSize;
    Shape shape{row, col};
    Shape shmemDataShape{testParam.rankSize, rowPerRank, col};

    auto buildShmemSetup = [&](Tensor& in, ShmemTensor& shmemTensor) {
        TileShape::Current().SetVecTile({tileRow, tileCol});
        DataType shmemDataType = in.GetDataType();
        if ((shmemDataType == DT_BF16) || (shmemDataType == DT_FP16)) {
            shmemDataType = DT_FP32;
        }
        LOOP("CreateShmemTensor", FunctionType::DYNAMIC_LOOP, index, LoopRange(1)) {
            (void)index;
            CreateShmemTensor(testParam.group, testParam.rankSize, shmemDataType, shmemDataShape, shmemTensor);
        }
    };

    // ── base (golden reference) ──
    std::vector<Opcode> irBase;
    {
        Tensor in(dType, shape, "in_ts_base");
        Tensor out(dType, shape, "out_ts_base");
        FUNCTION("TS_IR_CHECK_BASE", {in}, {out}) {
            ShmemTensor shmemTensor;
            buildShmemSetup(in, shmemTensor);
            TwoShotAllReduce(in, in, shmemTensor, out);
        }
        irBase = ExtractShmemOpcodes("TS_IR_CHECK_BASE");
        VerifyTwoShotAllReduceIR("TS_IR_CHECK_BASE", testParam.rankSize);
    }

    // ── v2 ──
    std::vector<Opcode> irV2;
    {
        Program::GetInstance().Reset();
        Tensor in(dType, shape, "in_ts_v2");
        Tensor out(dType, shape, "out_ts_v2");
        FUNCTION("TS_IR_CHECK_V2", {in}, {out}) {
            ShmemTensor shmemTensor;
            buildShmemSetup(in, shmemTensor);
            TwoShotAllReduce_v2(in, in, shmemTensor, out);
        }
        irV2 = ExtractShmemOpcodes("TS_IR_CHECK_V2");
        VerifyTwoShotAllReduceIR("TS_IR_CHECK_V2", testParam.rankSize);
    }

    // ── v3 ──
    std::vector<Opcode> irV3;
    {
        Program::GetInstance().Reset();
        Tensor in(dType, shape, "in_ts_v3");
        Tensor out(dType, shape, "out_ts_v3");
        FUNCTION("TS_IR_CHECK_V3", {in}, {out}) {
            ShmemTensor shmemTensor;
            buildShmemSetup(in, shmemTensor);
            TwoShotAllReduce_v3(in, in, shmemTensor, out);
        }
        irV3 = ExtractShmemOpcodes("TS_IR_CHECK_V3");
        VerifyTwoShotAllReduceIR("TS_IR_CHECK_V3", testParam.rankSize);
    }

    // ── v4 ──
    std::vector<Opcode> irV4;
    {
        Program::GetInstance().Reset();
        Tensor in(dType, shape, "in_ts_v4");
        Tensor out(dType, shape, "out_ts_v4");
        FUNCTION("TS_IR_CHECK_V4", {in}, {out}) {
            ShmemTensor shmemTensor;
            buildShmemSetup(in, shmemTensor);
            TwoShotAllReduce_v4(in, in, shmemTensor, out);
        }
        irV4 = ExtractShmemOpcodes("TS_IR_CHECK_V4");
        VerifyTwoShotAllReduceIR("TS_IR_CHECK_V4", testParam.rankSize);
    }

    // ── v5 ──
    std::vector<Opcode> irV5;
    {
        Program::GetInstance().Reset();
        Tensor in(dType, shape, "in_ts_v5");
        Tensor out(dType, shape, "out_ts_v5");
        FUNCTION("TS_IR_CHECK_V5", {in}, {out}) {
            ShmemTensor shmemTensor;
            buildShmemSetup(in, shmemTensor);
            TwoShotCommunicatorV2 comm(shmemTensor);
            TwoShotAllReduce_v5(in, in, comm, out);
        }
        irV5 = ExtractShmemOpcodes("TS_IR_CHECK_V5");
        VerifyTwoShotAllReduceIR("TS_IR_CHECK_V5", testParam.rankSize);
    }

    // ── Cross-variant equivalence: all must match the base (golden) ──
    EXPECT_EQ(irBase, irV2) << "TwoShot base and v2 SHMEM opcode sequences differ";
    EXPECT_EQ(irBase, irV3) << "TwoShot base and v3 SHMEM opcode sequences differ";
    EXPECT_EQ(irBase, irV4) << "TwoShot base and v4 SHMEM opcode sequences differ";
    EXPECT_EQ(irBase, irV5) << "TwoShot base and v5 SHMEM opcode sequences differ";
}

template void TestTwoShotAllReduceIREquivalence<int32_t>(OpTestParam &testParam, std::string &goldenDir);
template void TestTwoShotAllReduceIREquivalence<float>(OpTestParam &testParam, std::string &goldenDir);
template void TestTwoShotAllReduceIREquivalence<float16>(OpTestParam &testParam, std::string& goldenDir);
template void TestTwoShotAllReduceIREquivalence<bfloat16>(OpTestParam &testParam, std::string& goldenDir);

} // namespace Distributed
} // namespace npu::tile_fwk
