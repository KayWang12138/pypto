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
 * \file test_allgather_attn_post_reducescatter.cpp
 * \brief
 */

#include "distributed_op_test_common.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_manager.h"
#include "tilefwk/data_type.h"

namespace npu::tile_fwk {
namespace Distributed {

struct AllGatherAttenPostReducescatterTestConfig {
    DataType inDtype;
    DataType outDtype;
    int32_t b;
    int32_t s;
    int32_t n;
    int32_t kvLoraRank;
    int32_t vHeadDim;
    int32_t h;
};

AllGatherAttenPostReducescatterTestConfig ParseAllGatherAttenPostReducescatterConfig(const nlohmann::json& testData)
{
    AllGatherAttenPostReducescatterTestConfig caseInfo;
    auto inTensor = testData["input_tensors"][0];
    auto outTensor = testData["output_tensors"][0];
    caseInfo.inDtype = GetDataTypeNum(GetDtypeNum(inTensor["dtype"].get<std::string>()));
    caseInfo.outDtype = GetDataTypeNum(GetDtypeNum(outTensor["dtype"].get<std::string>()));
    auto params = testData["params"];
    caseInfo.b = params["batch_size"].get<int32_t>();
    caseInfo.s = params["seq_len"].get<int32_t>();
    caseInfo.n = params["num_heads"].get<int32_t>();
    caseInfo.kvLoraRank = params["kv_lora_rank"].get<int32_t>();
    caseInfo.vHeadDim = params["value_head_dim"].get<int32_t>();
    caseInfo.h = params["output_hidden_size"].get<int32_t>();
    return caseInfo;
}

std::tuple<Tensor, Tensor, Tensor, Tensor> InitializeTestData(OpTestParam &testParam, std::string &goldenDir,
    AllGatherAttenPostReducescatterTestConfig &caseInfo) {
    Shape agInShape = {caseInfo.b * caseInfo.n * caseInfo.s / testParam.rankSize, caseInfo.kvLoraRank};
    Shape wLoraShape = {caseInfo.n, caseInfo.kvLoraRank, caseInfo.vHeadDim};
    Shape wOutShape = {caseInfo.n * caseInfo.vHeadDim, caseInfo.h};
    Shape outShape = {caseInfo.b * caseInfo.s / testParam.rankSize, caseInfo.h};

    Tensor agIn(caseInfo.inDtype, agInShape, "agIn");
    Tensor wLora(caseInfo.inDtype, wLoraShape, "wLora");
    Tensor wOut(caseInfo.inDtype, wOutShape, "wOut");
    Tensor out(caseInfo.outDtype, outShape, "out");

    std::vector<bfloat16> agInPtr =
        ReadToVector<bfloat16>(goldenDir + "/ag_in_rank_" + std::to_string(testParam.rankId) + ".bin", agInShape);
    std::vector<bfloat16> wLoraPtr =
        ReadToVector<bfloat16>(goldenDir + "/w_lora_rank_" + std::to_string(testParam.rankId) + ".bin", wLoraShape);
    std::vector<bfloat16> wOutPtr =
        ReadToVector<bfloat16>(goldenDir + "/w_out_rank_" + std::to_string(testParam.rankId) + ".bin", wOutShape);

    ProgramData::GetInstance().AppendInputs({RawTensorData::CreateTensor<bfloat16>(agIn, agInPtr)});
    ProgramData::GetInstance().AppendInputs({RawTensorData::CreateTensor<bfloat16>(wLora, wLoraPtr)});
    ProgramData::GetInstance().AppendInputs({RawTensorData::CreateTensor<bfloat16>(wOut, wOutPtr)});
    ProgramData::GetInstance().AppendOutputs({RawTensorData::CreateTensorZero(out)});
    return {agIn, wLora, wOut, out};
}

std::tuple<Tensor, Tensor> CreateShmemTensors(OpTestParam& testParam, DataType dtype, const Shape& shape) {
    Tensor shmemData;
    Tensor shmemSignal;
    LOOP("CreateShmemTensor", FunctionType::DYNAMIC_LOOP, index, LoopRange(1)) {
        (void)index;
        CreateShmemData(testParam.group, testParam.rankSize, dtype, shape, shmemData);
        CreateShmemSignal(testParam.group, shmemData, shmemSignal);
    }
    return {shmemData, shmemSignal};
}

void TestAllGatherAttentionPostReducescatter(OpTestParam &testParam, const nlohmann::json& testData) {
    std::string goldenDir = GetGoldenDirPath(testData);
    auto caseInfo = ParseAllGatherAttenPostReducescatterConfig(testData);
    auto [agIn, wLora, wOut, out] = InitializeTestData(testParam, goldenDir, caseInfo);
    ASSERT(testParam.rankSize > 0) << "testParam.rankSize must be > 0, but got: " << testParam.rankSize;
    int32_t outRow = caseInfo.b * caseInfo.s / testParam.rankSize;
    FUNCTION("ALLGATHER_ATTNPOST_REDUCESCATTER", {agIn, wLora, wOut}, {out}) {
        Tensor agOut(caseInfo.inDtype, {caseInfo.b * caseInfo.n * caseInfo.s, caseInfo.kvLoraRank}, "agOut");
        LOOP("ALLGATHER", FunctionType::DYNAMIC_LOOP, unusedDynRankId, LoopRange(1)) {
            (void) unusedDynRankId;
            Shape shmemDataAgShape{testParam.rankSize, caseInfo.b * caseInfo.n * caseInfo.s / testParam.rankSize, caseInfo.kvLoraRank};
            auto [shmemData, shmemSignal] = CreateShmemTensors(testParam, caseInfo.inDtype, shmemDataAgShape);
            TileShape::Current().SetVecTile({64, caseInfo.kvLoraRank});
            AllGather(agIn, agIn, testParam.group, shmemData, shmemSignal, agOut);
        }
        Tensor attnOut(caseInfo.inDtype, {caseInfo.b * caseInfo.s, caseInfo.h}, "attnOut");
        LOOP("ATTNPOST", FunctionType::DYNAMIC_LOOP, batchId, LoopRange(1)) {
            (void) batchId;
            TileShape::Current().SetVecTile({4, 16, 1, caseInfo.kvLoraRank});
            Tensor attnIn = Reshape(agOut, {caseInfo.b, caseInfo.n, caseInfo.s, caseInfo.kvLoraRank});
            TileShape::Current().SetVecTile({4, 16, 1, caseInfo.kvLoraRank});
            Tensor attnRes0 = Transpose(attnIn, {1, 2});
            TileShape::Current().SetVecTile({4, 1, 32, std::min(512, caseInfo.kvLoraRank)});
            Tensor attnRes1 = Reshape(attnRes0, {caseInfo.b * caseInfo.s, caseInfo.n, caseInfo.kvLoraRank});
            TileShape::Current().SetVecTile({4, 16, caseInfo.kvLoraRank});
            Tensor t2Res = Transpose(attnRes1, {0, 1});
            TileShape::Current().SetCubeTile({16, 16}, {256, 256}, {128, 128});
            Tensor fp32Bmm4Res =  Matrix::BatchMatmul(DataType::DT_FP32, t2Res, wLora);
            Tensor bmm4Res = Cast(fp32Bmm4Res, caseInfo.inDtype);
            TileShape::Current().SetVecTile({32, 4, caseInfo.vHeadDim}); // 必须切，但是尾轴不能切
            Tensor t3Res = Transpose(bmm4Res, {0, 1}); // [bs,n,vHeadDim]
            TileShape::Current().SetVecTile({4, 32, caseInfo.vHeadDim});
            Tensor r2Res = Reshape(t3Res, {caseInfo.b * caseInfo.s, caseInfo.n * caseInfo.vHeadDim});
            TileShape::Current().SetCubeTile({16, 16}, {256, 256}, {128, 128});
            attnOut = Matrix::Matmul(caseInfo.inDtype, r2Res, wOut, false, false);
        }
        LOOP("REDUCESCATTER", FunctionType::DYNAMIC_LOOP, unusedIndex, LoopRange(1)) {
            (void) unusedIndex;
            DataType shmemDataType = (attnOut.GetDataType() == DT_BF16 || attnOut.GetDataType() == DT_FP16) 
                ? DT_FP32 : attnOut.GetDataType();
            auto [shmemData, shmemSignal] = CreateShmemTensors(testParam, shmemDataType, {1, outRow, caseInfo.h});
            TileShape::Current().SetVecTile({16, caseInfo.h});
            Distributed::ReduceScatter(attnOut, attnOut, testParam.group, shmemData, shmemSignal,
                DistReduceType::DIST_REDUCE_ADD, out);
        }
    }
    RunTest();
    auto output = ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(CompareWithGolden<uint8_t*>(caseInfo.outDtype, goldenDir + "/rs_out_rank_", outRow * caseInfo.h, output->GetDevPtr(), testParam, 0.1f));
}

} // namespace Distributed
} // namespace npu::tile_fwk