/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_allgather_attn_post_reducescatter.cpp
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

void TestAllGatherAttentionPostReducescatter(OpTestParam &testParam)
{
    int rankSize = testParam.rankSize;
    int rankId = testParam.rankId;
    char* group = testParam.group;

    constexpr size_t paramsSize = 7;
    auto [b, s, n, kvLoraRank, vHeadDim, h, typeNum] = GetParams<paramsSize>(GetGoldenDir() + "/params.bin");
    DataType dtype = GetDataTypeNum(typeNum);

    std::vector<int64_t> agInShape = {b * n * s / rankSize, kvLoraRank};
    std::string agInFile = "/ag_in_rank_" + std::to_string(rankId) + ".bin";
    Tensor agIn = CreateTensorFromFile(agInShape, dtype, agInFile, "agIn");
    std::vector<int64_t> wLoraShape = {n, kvLoraRank, vHeadDim};
    std::string wLoraFile = "/w_lora_rank_" + std::to_string(rankId) + ".bin";
    Tensor wLora = CreateTensorFromFile(wLoraShape, dtype, wLoraFile, "wLora");
    std::vector<int64_t> wOutShape = {n * vHeadDim, h};
    std::string wOutFile = "/w_out_rank_" + std::to_string(rankId) + ".bin";
    Tensor wOut = CreateTensorFromFile(wOutShape, dtype, wOutFile, "wOut");

    std::vector<int64_t> outShape = {b * s / rankSize, h};
    int64_t outEleNum = GetEleNumFromShape(outShape);
    uint64_t outByteSize = outEleNum * BytesOf(dtype);
    uint8_t* outPtr = allocDevAddr(outByteSize);
    Tensor out(dtype, outShape, outPtr, "out");

    FunctionConfig funConfig(FunctionType::STATIC);
    ;
    FUNCTION("Allgather_AttnPost_ReduceScatter", funConfig, {agIn, wLora, wOut, out}) {
        config::SetSemanticLabel("AllGather");
        TileShape::Current().SetDistTile({64, b * n * s / rankSize / 64, 0}, {kvLoraRank, 1, 0}, {1, rankSize, 0});
        TileShape::Current().SetDistRankId(rankId);
        Tensor agOut = AllGather(agIn, group);

        config::SetSemanticLabel("AttnPost");
        TileShape::Current().SetVecTile({4, 16, 1, kvLoraRank});
        Tensor attnIn = Reshape(agOut, {b, s, n, kvLoraRank});
        TileShape::Current().SetVecTile({4, 16, 1, kvLoraRank});
        Tensor attnRes0 = Transpose(attnIn, {1, 2});
        TileShape::Current().SetVecTile({4, 1, 32, std::min(512, kvLoraRank)});
        Tensor attnRes1 = Reshape(attnRes0, {b * s, n, kvLoraRank});
        TileShape::Current().SetVecTile({4, 16, kvLoraRank});
        Tensor t2Res = Transpose(attnRes1, {0, 1});
        TileShape::Current().SetCubeTile({16, 16}, {256, 256}, {128, 128});
        Tensor bmm4Res = Matrix::BatchMatmul(dtype, t2Res, wLora);
        TileShape::Current().SetVecTile({32, 4, vHeadDim}); // 必须切，但是尾轴不能切
        Tensor t3Res = Transpose(bmm4Res, {0, 1}); // [bs,n,vHeadDim]
        TileShape::Current().SetVecTile({4, 32, vHeadDim});
        Tensor r2Res = Reshape(t3Res, {b * s, n * vHeadDim});
        TileShape::Current().SetCubeTile({16, 16}, {256, 256}, {128, 128});
        Tensor attnOut = Matrix::Matmul<false, false>(dtype, r2Res, wOut);

        config::SetSemanticLabel("ReduceScatter");
        TileShape::Current().SetDistTile({16, b * s / rankSize / 16, 0}, {h, 1, 0}, {1, rankSize, 0});
        TileShape::Current().SetDistRankId(rankId);
        out = ReduceScatter(attnOut, group, DistReduceType::DIST_REDUCE_ADD);
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    EXPECT_TRUE(CompareWithGolden<uint8_t *>(dtype, "/rs_out_rank_", outEleNum, outPtr, testParam, 0.1f));
}

} // namespace Distributed
} // namespace npu::tile_fwk