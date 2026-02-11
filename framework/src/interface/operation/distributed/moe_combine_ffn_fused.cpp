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
 * \file moe_combine_ffn_fused.cpp
 * \brief
 */

#include "distributed_common.h"
#include "interface/function/function.h"
#include "interface/inner/tilefwk.h"
#include "interface/operation/operation.h"
#include "interface/program/program.h"
#include "interface/tensor/logical_tensor.h"
#include "interface/utils/common.h"
#include "interface/utils/log.h"
#include "tilefwk/data_type.h"
#include "tilefwk/platform.h"
#include "tilefwk/symbolic_distributed.h"
#include "tilefwk/tensor.h"
#include "tilefwk/tilefwk.h"
#include "tilefwk/tilefwk_op.h"

#include <algorithm>
#include <functional>
#include <limits>
#include <numeric>

namespace npu::tile_fwk::Distributed {
void MoeDistributedCombineValidate(const Tensor& expandX, const Tensor& assistInfoForCombine, const Tensor& recvCounts,
    const Tensor& expertScales, const char* group, uint32_t epWorldSize, uint32_t moeExpertNum,
    uint32_t sharedExpertNum, uint32_t sharedExpertRankNum, Tensor& out);
void CreateShmemTensor(Tensor& shmemTensor, int32_t rankSize, int32_t hcclGroupIndex, DataType dataType,
    const Shape& shape, uint64_t memType);
Tensor MoeDistributedCombineSend(const Tensor& in, const Tensor& assistInfoForCombine, const Tensor& recvCounts,
    const Tensor& shmemData, const Tensor& shmemSignal, int32_t topK);
Tensor MoeDistributedCombineReceive(
    const Tensor& predToken,
    const Tensor& expertScales,
    const Tensor& recvCounts,
    const Tensor& shmemData,
    const Tensor& shmemSignal);

namespace {
int32_t GetFfnIntermediateSizeFromShape(const Shape& shape, int32_t hiddenSize)
{
    ASSERT(hiddenSize > 0) << "hiddenSize must be positive, but got " << hiddenSize;
    int64_t weightElements = std::accumulate(shape.begin(), shape.end(), int64_t{1}, std::multiplies<int64_t>());
    int64_t denom = static_cast<int64_t>(hiddenSize) * 3;
    ASSERT(weightElements > 0) << "ffnWeight cannot be empty.";
    ASSERT(weightElements % denom == 0) << "ffnWeight element count must be divisible by 3 * hiddenSize, got "
        << weightElements << " and hiddenSize=" << hiddenSize;
    int64_t intermediateSize = weightElements / denom;
    ASSERT(intermediateSize > 0) << "ffnWeight intermediateSize must be positive, but got " << intermediateSize;
    ASSERT(intermediateSize <= static_cast<int64_t>(std::numeric_limits<uint16_t>::max()))
        << "ffnWeight intermediateSize exceeds uint16_t: " << intermediateSize;
    return static_cast<int32_t>(intermediateSize);
}

int32_t GetFfnIntermediateSize(const Tensor& ffnWeight, int32_t hiddenSize)
{
    return GetFfnIntermediateSizeFromShape(ffnWeight.GetShape(), hiddenSize);
}
} // namespace

void TiledMoeFfnFused(
    Function& function,
    const TileShape& tileShape,
    const std::vector<std::shared_ptr<LogicalTensor>>& iOperand,
    const std::vector<std::shared_ptr<LogicalTensor>>& oOperand,
    const Operation& op)
{
    ASSERT(iOperand.size() == 2UL) << "TiledMoeFfnFused iOperand size is not equal to 2";
    ASSERT(oOperand.size() == 2UL) << "TiledMoeFfnFused oOperand size is not equal to 2";
    auto combineOut = iOperand[0];
    auto ffnWeight = iOperand[1];
    auto out = oOperand[0];
    auto workspace = oOperand[1];

    int64_t hiddenSize = out->shape[1];
    int64_t intermediateSize = 0;
    int64_t workspaceRowsPerTile = 0;
    if (!op.GetAttr("intermediateSize", intermediateSize)) {
        intermediateSize = GetFfnIntermediateSizeFromShape(ffnWeight->shape, static_cast<int32_t>(hiddenSize));
    }

    int64_t workspaceCol = hiddenSize + 3LL * intermediateSize;
    ASSERT(op.GetAttr("workspaceRowsPerTile", workspaceRowsPerTile))
        << "workspaceRowsPerTile attr is required for OP_MOE_FFN_FUSED";
    ASSERT(workspace->shape[1] == workspaceCol) << "Workspace shape mismatch, expected "
        << workspaceCol << " but got " << workspace->shape[1];

    int64_t dataByteSize = BytesOf(out->Datatype());
    ASSERT(dataByteSize != 0);
    int64_t paddedColShape = AlignUp(dataByteSize * hiddenSize, COPY_BLOCK_BYTE_SIZE) / dataByteSize;

    int64_t floatByteSize = BytesOf(DataType::DT_FP32);
    ASSERT(floatByteSize != 0);
    int64_t floatEleNum = AlignUp(floatByteSize * paddedColShape, REPEAT_BYTE) / floatByteSize;
    // SiLU tiled kernel with tileSize=1024 requires at least 4096 float slots in UB scratch.
    floatEleNum = std::max<int64_t>(floatEleNum, 4096);

    DistOpAttr distOpAttr;
    distOpAttr.extraTemplateParam = std::to_string(intermediateSize);

    CreateTileOp(tileShape,
        [&](int32_t tileIndex, int32_t rowOffset, int32_t colOffset, int32_t rowShape, int32_t colShape) {
            (void)tileIndex;

            auto combineTile = combineOut->View(function, {rowShape, colShape}, {rowOffset, colOffset});
            auto outTile = out->View(function, {rowShape, colShape}, {rowOffset, colOffset});
            auto workspaceTile = workspace->View(function, {workspaceRowsPerTile, workspaceCol},
                {workspaceRowsPerTile * tileIndex, 0});
            auto ubBuffer = std::make_shared<LogicalTensor>(function, DT_FP32, Shape{floatEleNum});

            auto& tileOp = function.AddOperation(Opcode::OP_MOE_FFN_FUSED,
                {combineTile, ffnWeight},
                {outTile, workspaceTile, ubBuffer});

            distOpAttr.paddedColShape = paddedColShape;
            distOpAttr.rowShape = rowShape;
            tileOp.SetAttr(OpAttributeKey::distOpAttr, distOpAttr);
            tileOp.SetAttr(OpAttributeKey::excludeBufferReuse, true);
            tileOp.SetAttribute(OpAttributeKey::isCube, true);
        });
}

Tensor MoeFfnFused(
    const Tensor& combineOut,
    const Tensor& ffnWeight,
    int32_t intermediateSize,
    int32_t tileNum)
{
    auto& function = *Program::GetInstance().GetCurrentFunction();
    int32_t batchSize = combineOut.GetShape(0);
    int32_t hiddenSize = combineOut.GetShape(1);

    constexpr int64_t cubeBlockM = 16;
    int64_t maxRowsPerTile = std::max<int64_t>(1, (batchSize + tileNum - 1) / tileNum);
    int64_t workspaceRowsPerTile = AlignUp(maxRowsPerTile, cubeBlockM);
    int64_t workspaceRows = workspaceRowsPerTile * tileNum;

    auto out = std::make_shared<LogicalTensor>(function, combineOut.GetDataType(), Shape{batchSize, hiddenSize});
    auto workspace = std::make_shared<LogicalTensor>(
        function, combineOut.GetDataType(), Shape{workspaceRows, static_cast<int64_t>(hiddenSize) + 3LL * intermediateSize});

    auto& op = function.AddOperation(
        Opcode::OP_MOE_FFN_FUSED,
        {combineOut.GetStorage(), ffnWeight.GetStorage()},
        {out, workspace});
    op.SetAttr("intermediateSize", static_cast<int64_t>(intermediateSize));
    op.SetAttr("workspaceRowsPerTile", workspaceRowsPerTile);
    return out;
}

void MoeDistributedCombineFfnFused(const Tensor& expandX, const Tensor& assistInfoForCombine, const Tensor& recvCounts,
    const Tensor& expertScales, const Tensor& ffnWeight, const char* group, uint32_t epWorldSize,
    uint32_t moeExpertNum, uint32_t sharedExpertNum, uint32_t sharedExpertRankNum, Tensor& out)
{
    MoeDistributedCombineValidate(expandX, assistInfoForCombine, recvCounts, expertScales, group, epWorldSize,
        moeExpertNum, sharedExpertNum, sharedExpertRankNum, out);
    ASSERT(ffnWeight.Format() == npu::tile_fwk::TileOpFormat::TILEOP_ND) << "The format of \"ffnWeight\" only "
        << "supports ND, but got NZ";
    ASSERT(ffnWeight.GetDataType() == expandX.GetDataType()) << "The data type of \"ffnWeight\" must be consistent "
        << "with that of \"expandX\", but got " << DataType2String(ffnWeight.GetDataType()) << " and "
        << DataType2String(expandX.GetDataType());

    int32_t batchSize = expertScales.GetShape(0);
    int32_t topK = expertScales.GetShape(1);
    int32_t hiddenSize = expandX.GetShape(1);
    int32_t intermediateSize = GetFfnIntermediateSize(ffnWeight, hiddenSize);

    int32_t shmemDataRow = topK * batchSize;
    Shape shmemDataShape = {1, shmemDataRow, hiddenSize};
    int32_t shmemSignalCol = SAME_ADDR_BYTE_SIZE / BytesOf(DataType::DT_FP32);
    Shape shmemSignalShape = {1, batchSize, shmemSignalCol};

    Tensor shmemData;
    Tensor shmemSignal;
    int32_t hcclGroupIndex = static_cast<int>(CommGroupRecorder::GetInstance().Input(std::string(group)));
    LOOP("CreateShmemTensor", FunctionType::DYNAMIC_LOOP, index, LoopRange(1)) {
        (void)index;
        CreateShmemTensor(shmemData, epWorldSize, hcclGroupIndex, expandX.GetDataType(), shmemDataShape, 0);
        CreateShmemTensor(shmemSignal, epWorldSize, hcclGroupIndex, DT_INT32, shmemSignalShape, 1);
        (void)ShmemDataSet(recvCounts, shmemSignal);
    }
    Tensor combineOut;
    LOOP("MoeDistributedCombineOnly", FunctionType::DYNAMIC_LOOP, index, LoopRange(1)) {
        (void)index;

        int32_t expandXRow = expandX.GetShape(0);
        int32_t aivNum = AIV_NUM;
        TileShape::Current().SetDistTile({expandXRow / aivNum, aivNum, expandXRow % aivNum}, {hiddenSize, 1, 0},
            {0, 0, 0});
        auto sendOut = MoeDistributedCombineSend(
            expandX,
            assistInfoForCombine,
            recvCounts,
            shmemData,
            shmemSignal,
            topK);

        SymbolicScalar thisRank = GetHcclRankId(group);
        auto shmemDataThisRank = View(shmemData, {1, 1, shmemDataRow, hiddenSize},
            std::vector<SymbolicScalar>{thisRank, 0, 0, 0});
        auto shmemSignalThisRank = View(shmemSignal, {1, 1, batchSize, shmemSignalCol},
            std::vector<SymbolicScalar>{thisRank, 0, 0, 0});

        TileShape::Current().SetDistTile(
            {batchSize / aivNum, aivNum, batchSize % aivNum}, {hiddenSize, 1, 0}, {0, 0, 0});
        combineOut = MoeDistributedCombineReceive(sendOut, expertScales, recvCounts, shmemDataThisRank,
            shmemSignalThisRank);
    }

    LOOP("MoeFfnOnly", FunctionType::DYNAMIC_LOOP, index, LoopRange(1)) {
        (void)index;
        int32_t aivNum = AIV_NUM;
        TileShape::Current().SetDistTile(
            {batchSize / aivNum, aivNum, batchSize % aivNum}, {hiddenSize, 1, 0}, {0, 0, 0});
        out = MoeFfnFused(combineOut, ffnWeight, intermediateSize, aivNum);
    }
}
} // namespace npu::tile_fwk::Distributed
