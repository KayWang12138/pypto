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
 * \file moe_dispatch.cpp
 * \brief
 */
#include <functional>
#include <memory>
#include <vector>
#include "interface/operation/operation.h"
#include "tilefwk/tensor.h"
#include "interface/tensor/logical_tensor.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"
#include "interface/utils/common.h"
#include "distributed_common.h"
#include "tilefwk/symbolic_distributed.h"
#include "interface/function/function.h"

namespace npu::tile_fwk {
namespace Distributed {
namespace {
constexpr int32_t GLM_V1_BATCH_SIZE = 8;
constexpr int32_t GLM_V1_HIDDEN_SIZE = 5120;
constexpr int32_t GLM_V1_TOPK = 8;
constexpr int32_t GLM_V1_MOE_EXPERT_NUM = 160;

constexpr int32_t AIGCODE_CHUNK_BATCH_SIZE_V1 = 1024;
constexpr int32_t AIGCODE_HIDDEN_SIZE = 4096;
constexpr int32_t AIGCODE_TOPK = 4;
constexpr int32_t AIGCODE_MOE_EXPERT_NUM = 16;
constexpr int32_t AIGCODE_EP_WORLD_SIZE = 4;

constexpr int32_t QWEN3_NEXT_CHUNK_BATCH_SIZE_V1 = 1024;
constexpr int32_t QWEN3_NEXT_CHUNK_BATCH_SIZE_V2 = 2048;
constexpr int32_t QWEN3_NEXT_HIDDEN_SIZE = 4096;
constexpr int32_t QWEN3_NEXT_TOPK = 2;
constexpr int32_t QWEN3_NEXT_MOE_EXPERT_NUM = 8;
constexpr int32_t QWEN3_NEXT_EP_WORLD_SIZE = 8;

constexpr uint64_t MOE_V2_MAX_WIN_SIZE = 1024ULL * 1024ULL * 200ULL;

bool IsQwen3NextChunkBatchSizeSupported(int32_t batchSize)
{
    return batchSize == QWEN3_NEXT_CHUNK_BATCH_SIZE_V1 || batchSize == QWEN3_NEXT_CHUNK_BATCH_SIZE_V2;
}

bool IsSupportedMoeDispatchV1Case(int32_t batchSize, int32_t hiddenSize, int32_t topK, const MoeConfig& moeConfig)
{
    const bool isGlmCase =
        batchSize == GLM_V1_BATCH_SIZE && hiddenSize == GLM_V1_HIDDEN_SIZE && topK == GLM_V1_TOPK &&
        moeConfig.routedExpertNum == GLM_V1_MOE_EXPERT_NUM && (moeConfig.rankNum == 4 || moeConfig.rankNum == 8);
    const bool isAigcodeCase =
        batchSize == AIGCODE_CHUNK_BATCH_SIZE_V1 && hiddenSize == AIGCODE_HIDDEN_SIZE && topK == AIGCODE_TOPK &&
        moeConfig.routedExpertNum == AIGCODE_MOE_EXPERT_NUM && moeConfig.rankNum == AIGCODE_EP_WORLD_SIZE;
    const bool isQwen3NextCase =
        IsQwen3NextChunkBatchSizeSupported(batchSize) && hiddenSize == QWEN3_NEXT_HIDDEN_SIZE &&
        topK == QWEN3_NEXT_TOPK && moeConfig.routedExpertNum == QWEN3_NEXT_MOE_EXPERT_NUM &&
        moeConfig.rankNum == QWEN3_NEXT_EP_WORLD_SIZE;
    return isGlmCase || isAigcodeCase || isQwen3NextCase;
}

int32_t InferSupportedMoeTopK(
    int32_t batchSize, int32_t hiddenSize, int32_t epWorldSize, int32_t moeExpertNum, int32_t sharedExpertNum,
    int32_t sharedExpertRankNum)
{
    const bool isGlmCase =
        batchSize == GLM_V1_BATCH_SIZE && hiddenSize == GLM_V1_HIDDEN_SIZE && moeExpertNum == GLM_V1_MOE_EXPERT_NUM &&
        sharedExpertNum == 0 && sharedExpertRankNum == 0 && (epWorldSize == 4 || epWorldSize == 8);
    if (isGlmCase) {
        return GLM_V1_TOPK;
    }

    const bool isAigcodeCase =
        batchSize == AIGCODE_CHUNK_BATCH_SIZE_V1 && hiddenSize == AIGCODE_HIDDEN_SIZE &&
        epWorldSize == AIGCODE_EP_WORLD_SIZE && moeExpertNum == AIGCODE_MOE_EXPERT_NUM && sharedExpertNum == 0 &&
        sharedExpertRankNum == 0;
    if (isAigcodeCase) {
        return AIGCODE_TOPK;
    }

    const bool isQwen3NextCase =
        IsQwen3NextChunkBatchSizeSupported(batchSize) && hiddenSize == QWEN3_NEXT_HIDDEN_SIZE &&
        epWorldSize == QWEN3_NEXT_EP_WORLD_SIZE && moeExpertNum == QWEN3_NEXT_MOE_EXPERT_NUM &&
        sharedExpertNum == 0 && sharedExpertRankNum == 0;
    if (isQwen3NextCase) {
        return QWEN3_NEXT_TOPK;
    }

    return 0;
}

std::set<int> GetMoeDispatchBackwardUnrollList(int32_t batchSize, int32_t hiddenSize, int32_t topK, int32_t epWorldSize)
{
    if (batchSize == AIGCODE_CHUNK_BATCH_SIZE_V1 && hiddenSize == AIGCODE_HIDDEN_SIZE && topK == AIGCODE_TOPK &&
        epWorldSize == AIGCODE_EP_WORLD_SIZE) {
        return {8, 4, 2, 1};
    }
    if (batchSize == QWEN3_NEXT_CHUNK_BATCH_SIZE_V1 && hiddenSize == QWEN3_NEXT_HIDDEN_SIZE &&
        topK == QWEN3_NEXT_TOPK && epWorldSize == QWEN3_NEXT_EP_WORLD_SIZE) {
        return {8, 4, 2, 1};
    }
    if (batchSize == QWEN3_NEXT_CHUNK_BATCH_SIZE_V2 && hiddenSize == QWEN3_NEXT_HIDDEN_SIZE &&
        topK == QWEN3_NEXT_TOPK && epWorldSize == QWEN3_NEXT_EP_WORLD_SIZE) {
        return {8, 4, 2, 1};
    }
    return {64, 32, 16, 8, 4, 2, 1};
}

} // namespace

void TiledDispatchFFNSched(
    Function& function, const TileShape& tileShape, const std::vector<std::shared_ptr<LogicalTensor>>& iOperand,
    const std::vector<std::shared_ptr<LogicalTensor>>& oOperand, const Operation& op)
{
    auto syncTensor = iOperand[DIST_INDEX_ZERO];
    auto shmemFlag = iOperand[DIST_INDEX_ONE];
    auto recvTokenCntOut = oOperand[DIST_INDEX_ZERO];
    int flagColSize = shmemFlag->GetShape()[3];
    std::string hcclGroupIndex;
    std::vector<int64_t> bufferShape;
    int32_t sharedExpertNum = 0;
    int64_t expertNumPerRank;
    op.GetAttr("hcclGroupIndex", hcclGroupIndex);
    op.GetAttr("dispatchBufferSize", bufferShape);
    op.GetAttr("expertNumPerRank", expertNumPerRank);

    const auto& tileRank = tileShape.GetDistTileRank();
    int32_t totalTileNum = GetTotalTileNum(tileRank) * static_cast<int32_t>(expertNumPerRank);
    const int32_t tileRankShape = tileRank[DIST_HEAD_SHAPE];
    const int32_t tileRankCnt = tileRank[DIST_HEAD_COUNT] + (tileRank[DIST_TAIL_SHAPE] == 0 ? 0 : 1);
    const int32_t tailRankShape = tileRank[DIST_TAIL_SHAPE];
    int32_t tileIndex = 0;
    for (int expertIndex = 0; expertIndex < expertNumPerRank; ++expertIndex) {
        for (int rankIndex = 0; rankIndex < tileRankCnt; ++rankIndex) {
            int32_t rankShape = ((tileRank[2] != 0) && (rankIndex == tileRankCnt - 1) ? tailRankShape : tileRankShape);
            int32_t rankOffset = rankIndex * tileRankShape;
            auto bufferTensor = std::make_shared<LogicalTensor>(function, DataType::DT_INT32, bufferShape);
            auto shmemFlagTile =
                shmemFlag->View(function, {1, 1, rankShape, flagColSize}, {0, expertIndex, rankOffset, 0});
            auto& opr = function.AddOperation(
                Opcode::OP_FFN_SCHED, {syncTensor, shmemFlagTile}, {recvTokenCntOut, bufferTensor});
            std::string extraParam = std::to_string(tileIndex) + ", " + hcclGroupIndex + ", " +
                                     std::to_string(sharedExpertNum) + ", " + std::to_string(totalTileNum) + ", " +
                                     std::to_string(rankShape) + ", " + std::to_string(expertNumPerRank);
            MoeDispatchAttr distOpAttr;
            distOpAttr.extraTemplateParam = extraParam;
            opr.SetAttr(OpAttributeKey::distOpAttr, distOpAttr);
            tileIndex++;
        }
    }
}

void TiledDispatchFFNCombineInfo(
    Function& function, const TileShape& tileShape, const std::vector<std::shared_ptr<LogicalTensor>>& iOperand,
    const std::vector<std::shared_ptr<LogicalTensor>>& oOperand, const Operation& op)
{
    auto recvTokenCntOut = iOperand[DIST_INDEX_ZERO];
    auto shmemData = iOperand[DIST_INDEX_ONE];
    auto shmemFlag = iOperand[DIST_INDEX_TWO];
    auto combineInfo = oOperand[DIST_INDEX_ZERO];

    int32_t shmemDataLength = shmemData->GetShape()[3];
    Shape combineInfoBufferShape = {combineInfo->GetShape()[0] + 32};
    std::string hcclGroupIndex;
    std::vector<int64_t> bufferShape;
    std::string axisH;
    std::string batchSize;
    int32_t sharedExpertNum = 0;
    int64_t expertNumPerRank;
    op.GetAttr("expertNumPerRank", expertNumPerRank);
    op.GetAttr("hcclGroupIndex", hcclGroupIndex);
    op.GetAttr("dispatchBufferSize", bufferShape);
    op.GetAttr("hiddenSize", axisH);
    op.GetAttr("tokenBatchSize", batchSize);

    const auto& tileRank = tileShape.GetDistTileRank();
    int32_t totalTileNum = GetTotalTileNum(tileRank) * static_cast<int32_t>(expertNumPerRank);
    const int32_t tileRankShape = tileRank[DIST_HEAD_SHAPE];
    const int32_t tileRankCnt = tileRank[DIST_HEAD_COUNT] + (tileRank[DIST_TAIL_SHAPE] == 0 ? 0 : 1);
    const int32_t tailRankShape = tileRank[DIST_TAIL_SHAPE];

    int32_t tileIndex = 0;
    for (int expertIndex = 0; expertIndex < expertNumPerRank; ++expertIndex) {
        for (int rankIndex = 0; rankIndex < tileRankCnt; ++rankIndex) {
            int32_t rankShape = ((tileRank[2] != 0) && (rankIndex == tileRankCnt - 1) ? tailRankShape : tileRankShape);
            int32_t rankOffset = rankIndex * tileRankShape;
            auto bufferCombineInfo = std::make_shared<LogicalTensor>(function, DataType::DT_INT32, bufferShape);
            auto shmemDataTile =
                shmemData->View(function, {1, rankShape, 1, shmemDataLength}, {0, rankOffset, expertIndex, 0});
            auto& opr = function.AddOperation(
                Opcode::OP_FFN_COMBINEINFO, {shmemDataTile, shmemFlag, recvTokenCntOut},
                {combineInfo, bufferCombineInfo});
            std::string extraParam = std::to_string(tileIndex) + ", " + hcclGroupIndex + ", " +
                                     std::to_string(sharedExpertNum) + ", " + std::to_string(totalTileNum) + ", " +
                                     std::to_string(rankShape) + ", " + axisH + ", " + batchSize + ", " +
                                     std::to_string(combineInfo->GetShape()[0]);
            MoeDispatchAttr distOpAttr;
            distOpAttr.extraTemplateParam = extraParam;
            opr.SetAttr(OpAttributeKey::distOpAttr, distOpAttr);
            tileIndex++;
        }
    }
}

void TiledDispatchFFNBatching(
    Function& function, const TileShape& tileShape, const std::vector<std::shared_ptr<LogicalTensor>>& iOperand,
    const std::vector<std::shared_ptr<LogicalTensor>>& oOperand, const Operation& op)
{
    auto recvTokenCntOut = iOperand[DIST_INDEX_ZERO];
    auto shmemData = iOperand[DIST_INDEX_ONE];
    auto shmemFlag = iOperand[DIST_INDEX_TWO];
    auto expandX = oOperand[DIST_INDEX_ZERO];
    auto validCnt = oOperand[DIST_INDEX_ONE];

    int32_t shmemDataLength = shmemData->GetShape()[3];
    std::string groupIndex;
    std::vector<int64_t> bufferShape;
    std::string axisH;
    std::string batchSize;
    int32_t sharedExpertNum = 0;
    int64_t expertNumPerRank;
    op.GetAttr("expertNumPerRank", expertNumPerRank);
    op.GetAttr("hcclGroupIndex", groupIndex);
    op.GetAttr("dispatchBufferSize", bufferShape);
    op.GetAttr("hiddenSize", axisH);
    op.GetAttr("tokenBatchSize", batchSize);

    const auto& tileRank = tileShape.GetDistTileRank();
    int32_t totalTileNum = GetTotalTileNum(tileRank) * static_cast<int32_t>(expertNumPerRank);
    const int32_t tileRankShape = tileRank[DIST_HEAD_SHAPE];
    const int32_t tileRankCnt = tileRank[DIST_HEAD_COUNT] + (tileRank[DIST_TAIL_SHAPE] == 0 ? 0 : 1);
    const int32_t tailRankShape = tileRank[DIST_TAIL_SHAPE];

    int32_t tileIndex = 0;
    for (int expertIndex = 0; expertIndex < expertNumPerRank; ++expertIndex) {
        for (int rankIndex = 0; rankIndex < tileRankCnt; ++rankIndex) {
            int32_t rankShape = ((tileRank[2] != 0) && (rankIndex == tileRankCnt - 1) ? tailRankShape : tileRankShape);
            int32_t rankOffset = rankIndex * tileRankShape;
            auto bufferTensor = std::make_shared<LogicalTensor>(function, DataType::DT_INT32, bufferShape);
            auto shmemDataTile =
                shmemData->View(function, {1, rankShape, 1, shmemDataLength}, {0, rankOffset, expertIndex, 0});
            auto& opr = function.AddOperation(
                Opcode::OP_FFN_BATCHING, {shmemDataTile, shmemFlag, recvTokenCntOut},
                {expandX, validCnt, bufferTensor});
            std::string extraParam = std::to_string(tileIndex) + ", " + groupIndex + ", " +
                                     std::to_string(sharedExpertNum) + ", " + std::to_string(totalTileNum) + ", " +
                                     std::to_string(rankShape) + ", " + axisH + ", " + batchSize + ", " +
                                     std::to_string(expandX->GetShape()[0]);
            MoeDispatchAttr distOpAttr;
            distOpAttr.extraTemplateParam = extraParam;
            opr.SetAttr(OpAttributeKey::distOpAttr, distOpAttr);
            tileIndex++;
        }
    }
}

void TiledDispatchFFNValidCnt(
    Function& function, const TileShape& tileShape, const std::vector<std::shared_ptr<LogicalTensor>>& iOperand,
    const std::vector<std::shared_ptr<LogicalTensor>>& oOperand, const Operation& op)
{
    (void)op;
    auto recvTokenCntOut = iOperand[DIST_INDEX_ZERO];
    auto shmemFlag = iOperand[DIST_INDEX_ONE];
    auto validCnt = oOperand[DIST_INDEX_ZERO];

    int32_t flagColSize = shmemFlag->GetShape()[3];
    int32_t rankSize = shmemFlag->GetShape()[0];

    const auto& tileExpert = tileShape.GetDistTileRank();
    int32_t tileExpertShape = tileExpert[0];
    int32_t expertCount = tileExpert[1] + (tileExpert[2] == 0 ? 0 : 1);
    Shape bufferShape{shmemFlag->shape[0] * expertCount};

    for (int32_t expertIndex = 0; expertIndex < expertCount; ++expertIndex) {
        int32_t expertShape =
            ((tileExpert[2] != 0) && (expertIndex == expertCount - 1)) ? tileExpert[2] : tileExpert[0];
        int32_t expertOffset = expertIndex * tileExpertShape;
        auto validCntBuffer = std::make_shared<LogicalTensor>(function, DataType::DT_INT32, bufferShape);
        auto shmemFlagTile =
            shmemFlag->View(function, {1, expertShape, rankSize, flagColSize}, {0, expertOffset, 0, 0});
        auto& tileop = function.AddOperation(
            Opcode::OP_FFN_VALIDCNT, {recvTokenCntOut, shmemFlagTile}, {validCnt, validCntBuffer});
        std::string extraParam = std::to_string(expertShape);
        MoeDispatchAttr distOpAttr;
        distOpAttr.extraTemplateParam = extraParam;
        tileop.SetAttr(OpAttributeKey::distOpAttr, distOpAttr);
    }
}

Tensor DispatchFFNValidCnt(const Tensor& recvTokenCntOut, const Tensor& shmemFlag, const MoeConfig& moeConfig)
{
    auto& function = *Program::GetInstance().GetCurrentFunction();
    Shape validCntShape = {moeConfig.expertNumPerRank};
    auto validCntPtr = std::make_shared<LogicalTensor>(function, DataType::DT_INT32, validCntShape);
    auto& oper = function.AddOperation(
        Opcode::OP_FFN_VALIDCNT, {recvTokenCntOut.GetStorage(), shmemFlag.GetStorage()}, {validCntPtr});
    (void)oper;
    return validCntPtr;
}

Tensor DispatchFFNCombineInfo(
    const char* group, const Tensor& tokenTensor, const Tensor& recvTokenCntOut, const Tensor& shmemData,
    const Tensor& shmemFlag, int32_t expandXRow, int32_t ffnTileNum, const MoeConfig& moeConfig)
{
    auto& function = *Program::GetInstance().GetCurrentFunction();
    Shape combineInfoShape = {expandXRow, 3};
    auto combineInfoPtr = std::make_shared<LogicalTensor>(function, DataType::DT_INT32, combineInfoShape);
    auto& oper = function.AddOperation(
        Opcode::OP_FFN_COMBINEINFO, {recvTokenCntOut.GetStorage(), shmemData.GetStorage(), shmemFlag.GetStorage()},
        {combineInfoPtr});
    int tempBufSize = AlignUp(moeConfig.expertNumPerRank * ffnTileNum * 32, 256) + 256 +
                      AlignUp(moeConfig.expertNumPerRank * ffnTileNum * 4, 32) +
                      512; // tempBufSize = recvTokenCnt数 + 存储mask + recvTokenCnt的int数 + 数据搬运
    std::string hcclGroupIndex = std::to_string(CommGroupRecorder::GetInstance().Input(std::string(group)));
    const std::vector<int64_t> bufferShape{tempBufSize};
    oper.SetAttr("hcclGroupIndex", hcclGroupIndex);
    oper.SetAttr("dispatchBufferSize", bufferShape);
    oper.SetAttr("hiddenSize", std::to_string(tokenTensor.GetShape()[1]));
    oper.SetAttr("tokenBatchSize", std::to_string(tokenTensor.GetShape()[0]));
    oper.SetAttr("expertNumPerRank", static_cast<int64_t>(moeConfig.expertNumPerRank));
    return combineInfoPtr;
}

Tensor DispatchFFNBatching(
    const char* group, const Tensor& tokenTensor, const Tensor& recvTokenCntOut, const Tensor& shmemData,
    const Tensor& shmemFlag, int32_t expandXRow, int32_t ffnTileNum, const MoeConfig& moeConfig)
{
    auto& function = *Program::GetInstance().GetCurrentFunction();
    Shape validCntShape = {moeConfig.expertNumPerRank};
    auto validCntPtr = std::make_shared<LogicalTensor>(function, DataType::DT_INT32, validCntShape);
    Shape expandXShape = {expandXRow, tokenTensor.GetShape()[1]};
    auto expandXPtr = std::make_shared<LogicalTensor>(function, tokenTensor.GetDataType(), expandXShape);
    auto& oper = function.AddOperation(
        Opcode::OP_FFN_BATCHING, {recvTokenCntOut.GetStorage(), shmemData.GetStorage(), shmemFlag.GetStorage()},
        {expandXPtr, validCntPtr});
    int cumSumBuffer = AlignUp(moeConfig.expertNumPerRank * ffnTileNum * 32, 256) + 256 +
                       AlignUp(moeConfig.expertNumPerRank * ffnTileNum * 4, 32) +
                       512; // tempBufSize = recvTokenCnt数 + 存储mask + recvTokenCnt的int数 + 数据搬运
    int tokenCopyBuffer = tokenTensor.GetShape(1);
    int tempBufSize = (cumSumBuffer < tokenCopyBuffer) ? tokenCopyBuffer : cumSumBuffer;
    std::string hcclGroupIndex = std::to_string(CommGroupRecorder::GetInstance().Input(std::string(group)));
    const std::vector<int64_t> bufferShape{tempBufSize};
    oper.SetAttr("hcclGroupIndex", hcclGroupIndex);
    oper.SetAttr("dispatchBufferSize", bufferShape);
    oper.SetAttr("hiddenSize", std::to_string(tokenTensor.GetShape()[1]));
    oper.SetAttr("tokenBatchSize", std::to_string(tokenTensor.GetShape()[0]));
    oper.SetAttr("expertNumPerRank", static_cast<int64_t>(moeConfig.expertNumPerRank));
    return expandXPtr;
}

Tensor DispatchFFNSched(
    const char* group, const Tensor& flagDummy, Tensor& shmemFlag, const MoeConfig& moeConfig, int32_t ffnTileCnt)
{
    auto& function = *Program::GetInstance().GetCurrentFunction();
    int32_t totalTileNum = moeConfig.routedExpertNum * ffnTileCnt;
    Shape shape = {totalTileNum, 512}; // 每个flag_count预留512个int存储
    auto recvTokenCntOutPtr = std::make_shared<LogicalTensor>(function, DataType::DT_INT32, shape);
    auto& oper = function.AddOperation(
        Opcode::OP_FFN_SCHED, {flagDummy.GetStorage(), shmemFlag.GetStorage()}, {recvTokenCntOutPtr});
    int32_t moeOpProcessRankSize = ffnTileCnt;
    int32_t maxProcessRankSize = moeOpProcessRankSize;
    int tempBufSize = maxProcessRankSize * 32 + 256 + AlignUp(maxProcessRankSize * 4, 256);
    std::string hcclGroupIndex = std::to_string(CommGroupRecorder::GetInstance().Input(std::string(group)));
    oper.SetAttr("hcclGroupIndex", hcclGroupIndex);
    const std::vector<int64_t> bufferShape{tempBufSize / 8, 8};
    oper.SetAttr("dispatchBufferSize", bufferShape);
    oper.SetAttr("expertNumPerRank", static_cast<int64_t>(moeConfig.expertNumPerRank));
    return recvTokenCntOutPtr;
}

std::vector<int64_t> GetCommBufferSize(const std::shared_ptr<LogicalTensor>& tokenTensor)
{
    // The tileop dispatch kernels use element-count based layout:
    //   AlignUp(hiddenSize, 512) + 512
    // for both shmem slots and the temporary token buffer. Keep the
    // LogicalTensor shape in the same unit to avoid UB overrun in
    // SendToRoutingExpert when the data type is BF16.
    const int64_t hiddenSize = tokenTensor->shape[1];
    constexpr int64_t combineInfoPad = 512;
    return {1, AlignUp(hiddenSize, static_cast<int64_t>(512)) + combineInfoPad};
}

void TiledSendToRoutingExpert(
    Function& function, const TileShape& tileShape, const std::vector<std::shared_ptr<LogicalTensor>>& iOperand,
    const std::vector<std::shared_ptr<LogicalTensor>>& oOperand, const Operation& op)
{
    auto shmemData = iOperand[DIST_INDEX_ZERO];
    auto tokenTensor = iOperand[DIST_INDEX_ONE];
    auto expertTable = iOperand[DIST_INDEX_TWO];
    auto syncTensor = oOperand[DIST_INDEX_ZERO];
    std::string hcclGroupIndex;
    int64_t expertNumPerRank;
    op.GetAttr("expertNumPerRank", expertNumPerRank);
    op.GetAttr("hcclGroupIndex", hcclGroupIndex);
    CreateTileOp(
        tileShape, [&](int32_t tileIndex, int32_t rowOffset, int32_t colOffset, int32_t rowShape, int32_t colShape) {
            (void)tileIndex;
            auto expertTableTile = expertTable->View(function, {rowShape, colShape}, {rowOffset, colOffset});
            auto expertBufferUb = std::make_shared<LogicalTensor>(
                function, expertTable->Datatype(),
                std::vector<int64_t>{1, expertTable->shape[0] * expertTable->shape[1]});
            auto expertBuffer = std::make_shared<LogicalTensor>(
                function, expertTable->Datatype(),
                std::vector<int64_t>{
                    1, expertTable->shape[0] * expertTable->shape[1] * (static_cast<int64_t>(sizeof(int32_t)) + 1)});
            auto tokenBuffer =
                std::make_shared<LogicalTensor>(function, tokenTensor->Datatype(), GetCommBufferSize(tokenTensor));
            auto& tileop = function.AddOperation(
                Opcode::OP_SEND_TO_ROUTING_EXPERT, {tokenTensor, shmemData, expertTableTile},
                {syncTensor, tokenBuffer, expertBufferUb, expertBuffer});
            std::string extraParam = std::to_string(tokenTensor->shape[1]) + ", " + std::to_string(rowOffset) + ", " +
                                     std::to_string(colOffset) + ", " + std::to_string(rowShape) + ", " +
                                     std::to_string(colShape) + ", " + hcclGroupIndex;
            MoeDispatchAttr distOpAttr;
            distOpAttr.extraTemplateParam = extraParam;
            tileop.SetAttr(OpAttributeKey::distOpAttr, distOpAttr);
        });
}

void TiledSendToSharedExpert(
    Function& function, const TileShape& tileShape, const std::vector<std::shared_ptr<LogicalTensor>>& iOperand,
    const std::vector<std::shared_ptr<LogicalTensor>>& oOperand, const Operation& op)
{
    auto shmemData = iOperand[DIST_INDEX_ZERO];
    auto tokenTensor = iOperand[DIST_INDEX_ONE];
    auto syncTensor = oOperand[DIST_INDEX_ZERO];
    (void)oOperand;
    std::string hcclGroupIndex;
    op.GetAttr("hcclGroupIndex", hcclGroupIndex);
    CreateTileOp(
        tileShape, [&](int32_t tileIndex, int32_t rowOffset, int32_t colOffset, int32_t rowShape, int32_t colShape) {
            (void)tileIndex;
            Shape shape = {rowShape, colShape};
            auto tokenTensorTile = tokenTensor->View(function, {rowShape, colShape}, {rowOffset, colOffset});
            auto tokenBuffer =
                std::make_shared<LogicalTensor>(function, tokenTensor->Datatype(), GetCommBufferSize(tokenTensor));
            auto& tileop = function.AddOperation(
                Opcode::OP_SEND_TO_SHARED_EXPERT, {tokenTensorTile, shmemData}, {syncTensor, tokenBuffer});
            std::string extraParam = std::to_string(tokenTensor->shape[0]) + ", " +
                                     std::to_string(tokenTensor->shape[1]) + ", " + std::to_string(rowShape) + ", " +
                                     hcclGroupIndex;
            MoeDispatchAttr distOpAttr;
            distOpAttr.extraTemplateParam = extraParam;
            tileop.SetAttr(OpAttributeKey::distOpAttr, distOpAttr);
        });
}

void TiledCopyToLocalExpert(
    Function& function, const TileShape& tileShape, const std::vector<std::shared_ptr<LogicalTensor>>& iOperand,
    const std::vector<std::shared_ptr<LogicalTensor>>& oOperand, const Operation& op)
{
    auto tokenTensor = iOperand[DIST_INDEX_ZERO];
    auto expandX = oOperand[DIST_INDEX_ZERO];
    auto syncTensor = oOperand[DIST_INDEX_ONE];
    (void)op;
    CreateTileOp(
        tileShape, [&](int32_t tileIndex, int32_t rowOffset, int32_t colOffset, int32_t rowShape, int32_t colShape) {
            (void)tileIndex;
            auto tokenTensorTile = tokenTensor->View(function, {rowShape, colShape}, {rowOffset, colOffset});
            auto tokenBuffer =
                std::make_shared<LogicalTensor>(function, tokenTensor->Datatype(), GetCommBufferSize(tokenTensor));
            auto& tileop = function.AddOperation(
                Opcode::OP_COPY_TO_LOCAL_EXPERT, {tokenTensorTile}, {expandX, syncTensor, tokenBuffer});
            std::string extraParam = std::to_string(tokenTensor->shape[0]) + ", " +
                                     std::to_string(tokenTensor->shape[1]) + ", " + std::to_string(rowShape);
            MoeDispatchAttr distOpAttr;
            distOpAttr.extraTemplateParam = extraParam;
            tileop.SetAttr(OpAttributeKey::distOpAttr, distOpAttr);
        });
}

void TiledDispatchSetFlag(
    Function& function, const TileShape& tileShape, const std::vector<std::shared_ptr<LogicalTensor>>& iOperand,
    const std::vector<std::shared_ptr<LogicalTensor>>& oOperand, const Operation& op)
{
    auto shmemFlag = iOperand[DIST_INDEX_ZERO];
    auto syncTensor = iOperand[DIST_INDEX_ONE];
    auto tokenExpertTable = iOperand[DIST_INDEX_TWO];
    auto syncDummy = oOperand[DIST_INDEX_ZERO];
    int flagColSize = shmemFlag->GetShape()[3];
    std::string hcclGroupIndex;
    op.GetAttr("hcclGroupIndex", hcclGroupIndex);
    int64_t expertNumPerRank;
    op.GetAttr("expertNumPerRank", expertNumPerRank);

    const auto& tileExpert = tileShape.GetDistTileRank();
    const auto& tileRank = tileShape.GetDistTileCol();
    int32_t tileRankShape = tileRank[0];
    int32_t tileExpertShape = tileExpert[0];
    int32_t rankCount = tileRank[1] + (tileRank[2] == 0 ? 0 : 1);
    int32_t expertCount = tileExpert[1] + (tileExpert[2] == 0 ? 0 : 1);

    for (int32_t rankIndex = 0; rankIndex < rankCount; ++rankIndex) {
        int32_t rankShape = ((tileRank[2] != 0) && (rankIndex == rankCount - 1)) ? tileRank[2] : tileRank[0];
        for (int32_t expertIndex = 0; expertIndex < expertCount; ++expertIndex) {
            int32_t expertShape =
                ((tileExpert[2] != 0) && (expertIndex == expertCount - 1)) ? tileExpert[2] : tileExpert[0];
            int32_t rankOffset = rankIndex * tileRankShape;
            int32_t expertOffset = expertIndex * tileExpertShape;
            auto statusTensor = std::make_shared<LogicalTensor>(
                function, tokenExpertTable->Datatype(),
                std::vector<int64_t>{
                    1, expertNumPerRank * 16 + 32}); // 每个expert预留16B缓存flag跟count,最后一个expert后预留32位
            auto expertBufferUb = std::make_shared<LogicalTensor>(
                function, tokenExpertTable->Datatype(),
                std::vector<int64_t>{1, tokenExpertTable->shape[0] * tokenExpertTable->shape[1]});
            auto expertBuffer = std::make_shared<LogicalTensor>(
                function, tokenExpertTable->Datatype(),
                std::vector<int64_t>{
                    1, tokenExpertTable->shape[0] * tokenExpertTable->shape[1] *
                           (static_cast<int64_t>(sizeof(int32_t)) + 1)});
            auto shmemFlagTile =
                shmemFlag->View(function, {rankShape, expertShape, 1, flagColSize}, {rankOffset, expertOffset, 0, 0});
            auto& tileop = function.AddOperation(
                Opcode::OP_DISPATCH_SET_FLAG, {tokenExpertTable, shmemFlagTile, syncTensor},
                {syncDummy, statusTensor, expertBufferUb, expertBuffer});
            std::string extraParam = std::to_string(tokenExpertTable->shape[0]) + ", " +
                                     std::to_string(tokenExpertTable->shape[1]) + ", " + hcclGroupIndex + ", " +
                                     std::to_string(expertShape) + ", " + std::to_string(rankShape);
            MoeDispatchAttr distOpAttr;
            distOpAttr.extraTemplateParam = extraParam;
            tileop.SetAttr(OpAttributeKey::distOpAttr, distOpAttr);
        }
    }
}

Tensor SendToRoutingExpert(
    const Tensor& shmemData, const Tensor& tokenTensor, const Tensor& tokenExpertTable, const char* group,
    const MoeConfig& moeConfig)
{
    Shape shape{1, 1};
    auto& function = *Program::GetInstance().GetCurrentFunction();
    auto syncTensor = std::make_shared<LogicalTensor>(function, DataType::DT_INT32, shape);
    auto& oper = function.AddOperation(
        Opcode::OP_SEND_TO_ROUTING_EXPERT,
        {shmemData.GetStorage(), tokenTensor.GetStorage(), tokenExpertTable.GetStorage()}, {syncTensor});
    std::string hcclGroupIndex = std::to_string(CommGroupRecorder::GetInstance().Input(std::string(group)));
    oper.SetAttr("hcclGroupIndex", hcclGroupIndex);
    oper.SetAttr("expertNumPerRank", static_cast<int64_t>(moeConfig.expertNumPerRank));
    return syncTensor;
}

void SendToSharedExpert(const Tensor& shmemData, const Tensor& tokenTensor, const Tensor& syncTensor, const char* group)
{
    auto& function = *Program::GetInstance().GetCurrentFunction();
    auto& oper = function.AddOperation(
        Opcode::OP_SEND_TO_SHARED_EXPERT, {shmemData.GetStorage(), tokenTensor.GetStorage()},
        {syncTensor.GetStorage()});
    std::string hcclGroupIndex = std::to_string(CommGroupRecorder::GetInstance().Input(std::string(group)));
    oper.SetAttr("hcclGroupIndex", hcclGroupIndex);
}

Tensor DispatchSetFlag(
    Tensor& shmemFlag, const Tensor& tokenExpertTable, const Tensor& syncTensor, const char* group,
    const MoeConfig& moeConfig)
{
    Shape shape = {1, 1};
    auto& function = *Program::GetInstance().GetCurrentFunction();
    auto syncDummy = std::make_shared<LogicalTensor>(function, DataType::DT_INT32, shape);
    auto& oper = function.AddOperation(
        Opcode::OP_DISPATCH_SET_FLAG, {shmemFlag.GetStorage(), syncTensor.GetStorage(), tokenExpertTable.GetStorage()},
        {syncDummy});
    std::string hcclGroupIndex = std::to_string(CommGroupRecorder::GetInstance().Input(std::string(group)));
    oper.SetAttr("hcclGroupIndex", hcclGroupIndex);
    oper.SetAttr("expertNumPerRank", static_cast<int64_t>(moeConfig.expertNumPerRank));
    return syncDummy;
}

Tensor CopyToLocalExpert(const Tensor& tokenTensor, const Tensor& syncTensor, const MoeConfig& moeConfig)
{
    auto& function = *Program::GetInstance().GetCurrentFunction();
    Shape expandXShape = {tokenTensor.GetShape()[0] * moeConfig.routedExpertNum, tokenTensor.GetShape()[1]};
    auto expandXPtr = std::make_shared<LogicalTensor>(function, tokenTensor.GetDataType(), expandXShape);
    auto& oper = function.AddOperation(
        Opcode::OP_COPY_TO_LOCAL_EXPERT, {tokenTensor.GetStorage()}, {expandXPtr, syncTensor.GetStorage()});
    (void)oper;
    return expandXPtr;
}

std::tuple<int32_t, int32_t, int32_t> GetFFNTileParam(const MoeConfig& moeConfig)
{
    int32_t tileRankCnt = moeConfig.rankNum > FFN_TILE_SIZE ? FFN_TILE_SIZE : moeConfig.rankNum;
    int32_t tileNum = tileRankCnt == FFN_TILE_SIZE ? moeConfig.rankNum / FFN_TILE_SIZE : 1;
    int32_t tailNum = tileNum == 1 ? 0 : (moeConfig.rankNum % FFN_TILE_SIZE == 0 ? 0 : 1);
    return {tileRankCnt, tileNum, tailNum};
}

void MoeDispatchValidateV1(
    const Tensor& tokenTensor, const Tensor& tokenExpertTable, Tensor& expandX, Tensor& validCnt, Tensor& combineInfo,
    const char* group, const MoeConfig& moeConfig)
{
    std::string assertResult;
    CHECK(checkValidConfig(moeConfig, assertResult)) << assertResult;
    CHECK(group != nullptr) << "MoeDispatch constraint violated: group name can't be nullptr.";
    CHECK(group[0] != '\0') << "MoeDispatch constraint violated: group name is not valid.";
    CHECK(strnlen(group, 128) < 128) << "MoeDispatch constraint violated: group name max size must be 128.";
    int32_t batchSize = tokenTensor.GetShape(0);
    int32_t hiddenSize = tokenTensor.GetShape(1);
    int32_t topK = tokenExpertTable.GetShape(1);
    CHECK(
        IsSupportedMoeDispatchV1Case(batchSize, hiddenSize, topK, moeConfig))
        << "MoeDispatch constraint violated: only GLM V1 "
           "(batch=8, hidden=5120, topK=8, routedExpertNum=160, rankNum=4/8) or Aigcode V1 "
           "(batch=1024, hidden=4096, topK=4, routedExpertNum=16, rankNum=4) or Qwen3 Next Aigcode V1 "
           "(batch=1024/2048, hidden=4096, topK=2, routedExpertNum=8, rankNum=8) are supported.";
    CHECK(checkValidInput(tokenTensor, 2, DataType::DT_BF16, batchSize, hiddenSize, assertResult)) << assertResult;
    int32_t expandXRow = std::min(batchSize * topK * moeConfig.rankNum, batchSize * moeConfig.routedExpertNum);
    CHECK(checkValidInput(tokenExpertTable, 2, DataType::DT_INT32, batchSize, topK, assertResult)) << assertResult;
    CHECK(checkValidInput(validCnt, 1, DataType::DT_INT32, moeConfig.expertNumPerRank, 1, assertResult)) << assertResult;
    CHECK(checkValidInput(expandX, 2, DataType::DT_BF16, expandXRow, hiddenSize, assertResult)) << assertResult;
    CHECK(checkValidInput(combineInfo, 2, DataType::DT_INT32, expandXRow, 3, assertResult)) << assertResult;

    int32_t shmemDataLength = AlignUp(hiddenSize, 512) + 512;
    uint64_t shmemSize = static_cast<uint64_t>(moeConfig.rankNum) * static_cast<uint64_t>(moeConfig.expertNumPerRank) *
                             static_cast<uint64_t>(shmemDataLength) * static_cast<uint64_t>(batchSize) *
                             BytesOf(tokenTensor.GetDataType()) +
                         static_cast<uint64_t>(moeConfig.expertNumPerRank) * static_cast<uint64_t>(moeConfig.rankNum) *
                             128ULL * BytesOf(DataType::DT_INT32);
    CHECK(shmemSize < MOE_V2_MAX_WIN_SIZE)
        << "MoeDispatch constraint violated: shmem window exceeds the limit. Maximum allowed: " << MOE_V2_MAX_WIN_SIZE
        << ", got: " << shmemSize;
}

void CreateShmemData(
    const char* group, int64_t worldSize, DataType dataType, const Shape& shape, Tensor& shmemTensor, uint64_t memType)
{
    auto& function = *Program::GetInstance().GetCurrentFunction();
    int32_t hcclGroupIndex = static_cast<int>(CommGroupRecorder::GetInstance().Input(std::string(group)));
    Shape shmemShape{worldSize};
    shmemShape.insert(shmemShape.end(), shape.begin(), shape.end());
    auto shmemTensorInner = std::make_shared<LogicalTensor>(function, dataType, shmemShape);
    shmemTensor = shmemTensorInner;
    Program::GetInstance().GetTensorSlotManager()->TensorWrite(shmemTensor, SlotProperty::SHMEM_TENSOR);
    auto& op = function.AddOperation(Opcode::OP_BIND_TENSOR, {}, {shmemTensorInner});
    op.SetAttribute(
        OpAttributeKey::bindTensor,
        BindTensor(
            hcclGroupIndex, memType,
            AlignUp(
                BytesOf(dataType) * std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int64_t>()), 512)));
}

void CreateShmemDispatchLoop(
    Tensor& shmemData, Tensor& shmemFlag, const char* group, const MoeConfig& moeConfig, int32_t shmemDataCol,
    int32_t flagCol, DataType tokenTendorDtype)
{
    LOOP("CreateShmemTensor", FunctionType::DYNAMIC_LOOP, index, LoopRange(1))
    {
        (void)index;
        Shape shmemDataShape = {moeConfig.rankNum, moeConfig.expertNumPerRank, shmemDataCol};
        Shape shmemFlagShape = {moeConfig.expertNumPerRank, moeConfig.rankNum, flagCol};
        CreateShmemData(group, moeConfig.rankNum, tokenTendorDtype, shmemDataShape, shmemData, 0);
        CreateShmemData(group, moeConfig.rankNum, DT_INT32, shmemFlagShape, shmemFlag, 0);
    }
}

void MoeDistributedDispatch(
    const Tensor& tokenTensor, const Tensor& tokenExpertTable, Tensor& expandX, Tensor& validCnt, Tensor& combineInfo,
    const char* group, const MoeConfig& moeConfig)
{
    MoeDispatchValidateV1(tokenTensor, tokenExpertTable, expandX, validCnt, combineInfo, group, moeConfig);
    SymbolicScalar thisRank = GetHcclRankId(group);
    int batchSize = tokenTensor.GetShape(0);
    int hiddenSize = tokenTensor.GetShape(1);
    int topK = tokenExpertTable.GetShape(1);
    int shmemDataLength = AlignUp(hiddenSize, 512) + 512;
    int flagRow = 1;
    int flagCol = 128;
    int shmemDataCol = shmemDataLength * batchSize;
    Tensor shmemData;
    Tensor shmemFlag;
    CreateShmemDispatchLoop(shmemData, shmemFlag, group, moeConfig, shmemDataCol, flagCol, tokenTensor.GetDataType());
    LOOP("L0", FunctionType::DYNAMIC_LOOP, index, LoopRange(1))
    {
        (void)index;
        TileShape::Current().SetDistTile({1, batchSize, 0}, {topK, 1, 0}, {moeConfig.rankNum, 1, 0});
        Tensor syncTensor = SendToRoutingExpert(shmemData, tokenTensor, tokenExpertTable, group, moeConfig);
        TileShape::Current().SetDistTile(
            {flagRow, 1, 0}, {moeConfig.rankNum, 1, 0}, {1, moeConfig.expertNumPerRank, 0});
        auto localShmemFlag =
            View(shmemFlag, {moeConfig.rankNum, moeConfig.expertNumPerRank, 1, flagCol}, {0, 0, thisRank, 0});
        Tensor flagDummy = DispatchSetFlag(localShmemFlag, tokenExpertTable, syncTensor, group, moeConfig);
        auto [ffnTileCnt, ffnTileNum, ffnTailNum] = GetFFNTileParam(moeConfig);
        TileShape::Current().SetDistTile({batchSize, 1, 0}, {hiddenSize, 1, 0}, {ffnTileCnt, ffnTileNum, ffnTailNum});
        auto shmemFlagSched =
            View(shmemFlag, {1, moeConfig.expertNumPerRank, moeConfig.rankNum, flagCol}, {thisRank, 0, 0, 0});
        auto recvTokenCntOut = DispatchFFNSched(group, flagDummy, shmemFlagSched, moeConfig, ffnTileCnt);
        auto shmemDataBatching =
            View(shmemData, {1, moeConfig.rankNum, moeConfig.expertNumPerRank, shmemDataLength}, {thisRank, 0, 0, 0});
        auto expandXPtr = DispatchFFNBatching(
            group, tokenTensor, recvTokenCntOut, shmemDataBatching, localShmemFlag, expandX.GetShape(0),
            ffnTileNum + ffnTailNum, moeConfig);
        auto combineInfoPtr = DispatchFFNCombineInfo(
            group, tokenTensor, recvTokenCntOut, shmemDataBatching, localShmemFlag, expandX.GetShape(0),
            ffnTileNum + ffnTailNum, moeConfig);
        int32_t validCntTileShape = std::max(1, moeConfig.expertNumPerRank / 10);
        int32_t validCntTileNum = moeConfig.expertNumPerRank / validCntTileShape;
        int32_t validCntTileTail = moeConfig.expertNumPerRank % validCntTileShape;
        TileShape::Current().SetDistTileRank({validCntTileShape, validCntTileNum, validCntTileTail});
        auto shmemFlagValidCnt =
            View(shmemFlag, {1, moeConfig.expertNumPerRank, moeConfig.rankNum, flagCol}, {thisRank, 0, 0, 0});
        auto validCntPtr = DispatchFFNValidCnt(recvTokenCntOut, shmemFlagValidCnt, moeConfig);
        expandX = expandXPtr;
        validCnt = validCntPtr;
        combineInfo = combineInfoPtr;
    }
}

void MoeDispatchValidateV2(
    const Tensor& x, const Tensor& expertIds, const char* group, uint32_t epWorldSize, uint32_t moeExpertNum,
    uint32_t sharedExpertNum, uint32_t sharedExpertRankNum, Tensor& expandX, Tensor& expertTokenNums,
    Tensor& assistInfoForCombine, Tensor& recvCounts)
{
    std::string assertResult;
    CHECK(group != nullptr) << "MoeDispatch constraint violated: group name can't be nullptr.";
    CHECK(group[0] != '\0') << "MoeDispatch constraint violated: group name must be valid, but got '\0'";
    CHECK(strnlen(group, 128) < 128) << "MoeDispatch constraint violated: group name max size must be 128, but got "
                                     << strnlen(group, 128);
    CHECK(epWorldSize > 0) << "MoeDispatch constraint violated: epWorldSize must be > 0, but got " << epWorldSize;
    CHECK(moeExpertNum == 160) << "MoeDispatch constraint violated: moeExpertNum must 160, but got " << moeExpertNum;
    CHECK(sharedExpertNum == 0) << "MoeDispatch constraint violated: sharedExpertNum must 0, but got "
                                << sharedExpertNum;
    CHECK(sharedExpertRankNum == 0) << "MoeDispatch constraint violated: sharedExpertRankNum must 0, but got "
                                    << sharedExpertRankNum;
    int32_t routedExpertNum = moeExpertNum - sharedExpertNum;
    int32_t expertNumPerRank = routedExpertNum / epWorldSize;
    CHECK(checkValidInput(x, 2, DataType::DT_BF16, 8, 5120, assertResult)) << assertResult;
    CHECK(checkValidInput(expertIds, 2, DataType::DT_INT32, 8, 8, assertResult)) << assertResult;
    CHECK(checkValidInput(expertTokenNums, 1, DataType::DT_INT32, expertNumPerRank, 1, assertResult)) << assertResult;
    CHECK(checkValidInput(recvCounts, 1, DataType::DT_INT32, 1, 0, assertResult)) << assertResult;
    int batchSize = x.GetShape(0);
    int topK = expertIds.GetShape(1);
    int32_t expandXRow = std::min(
        static_cast<int32_t>(batchSize) * static_cast<int32_t>(topK) * static_cast<int32_t>(epWorldSize),
        static_cast<int32_t>(batchSize) * routedExpertNum);
    CHECK(checkValidInput(expandX, 2, DataType::DT_BF16, expandXRow, 5120, assertResult)) << assertResult;
    CHECK(checkValidInput(assistInfoForCombine, 2, DataType::DT_INT32, expandXRow, 3, assertResult)) << assertResult;
    uint64_t shmemSize =
        moeExpertNum * x.GetShape(0) * x.GetShape(1) * BytesOf(x.GetDataType()) +
        moeExpertNum * x.GetShape(0) * assistInfoForCombine.GetShape(1) * BytesOf(assistInfoForCombine.GetDataType()) +
        AlignUp(routedExpertNum, 256) * 8 * BytesOf(DataType::DT_INT32) +
        moeExpertNum * 128 * BytesOf(DataType::DT_INT32) + 128 * BytesOf(DataType::DT_INT32);
    CHECK(shmemSize < MOE_V2_MAX_WIN_SIZE)
        << "Exceeds winSize limit. Maximum allowed: " << MOE_V2_MAX_WIN_SIZE << ", got: " << shmemSize;
}

Tensor Nop(const std::vector<Tensor>& inTensors)
{
    auto& function = *Program::GetInstance().GetCurrentFunction();
    auto out = std::make_shared<LogicalTensor>(function, DT_INT32, Shape{1, 1});
    LogicalTensors iOperands;
    for (const Tensor& inTensor : inTensors) {
        iOperands.emplace_back(inTensor.GetStorage());
    }
    function.AddOperation(Opcode::OP_NOP, iOperands, {out});
    return out;
}

Tensor DispatchCalcOccurrences(Tensor& expertIds, SymbolicScalar expertId, int32_t calcIndex)
{
    Tensor expertIdsDup = Full(expertId, DT_INT32, {1, expertIds.GetShape(1)});
    Tensor subResult = Sub(expertIdsDup, expertIds);
    Tensor subResultFp32 = Cast(subResult, DT_FP32, CAST_TRUNC);
    Tensor absSubResult = Abs(subResultFp32);
    Tensor subResultInt32 = Cast(absSubResult, DT_INT32, CAST_TRUNC);
    Tensor countOfEquals = Clip(subResultInt32, Element(DT_INT32, 0), Element(DT_INT32, 1));
    Tensor cumSumOffset = CumSum(countOfEquals, 1);
    Tensor cumSumOffsetInt32 = Cast(cumSumOffset, DT_INT32, CAST_TRUNC);
    Tensor expertOffsetResult = ScalarSubS(cumSumOffsetInt32, Element(DT_INT32, calcIndex));
    Tensor expertOffsetResultFp32 = Cast(expertOffsetResult, DT_FP32, CAST_TRUNC);
    Tensor expertOffsetAbsFp32 = Abs(expertOffsetResultFp32);
    Tensor expertOffset = Cast(expertOffsetAbsFp32, DT_INT32, CAST_TRUNC);
    return expertOffset;
}

void MoeDistributedDispatchV2(
    const Tensor& x, const Tensor& expertIds, const char* group, uint32_t epWorldSize, uint32_t moeExpertNum,
    uint32_t sharedExpertNum, uint32_t sharedExpertRankNum, Tensor& expandX, Tensor& assistInfoForCombine,
    Tensor& expertTokenNums, Tensor& recvCounts)
{
    MoeDispatchValidateV2(
        x, expertIds, group, epWorldSize, moeExpertNum, sharedExpertNum, sharedExpertRankNum, expandX, expertTokenNums,
        assistInfoForCombine, recvCounts);
    int32_t routedExpertNum = moeExpertNum - sharedExpertNum;
    CHECK(epWorldSize > 0) << "MoeDispatch constraint violated: epWorldSize must be > 0, but got " << epWorldSize;
    int32_t expertNumPerRank = routedExpertNum / epWorldSize;
    int32_t batchSize = x.GetShape(0);
    int32_t hiddenSize = x.GetShape(1);
    int32_t topK = expertIds.GetShape(1);
    CHECK(topK > 0) << "MoeDispatch constraint violated: topK must be > 0, but got " << topK;
    CHECK(expertNumPerRank > 0) << "MoeDispatch constraint violated: expertNumPerRank must be > 0, but got "
                                << expertNumPerRank;
    int32_t infoSize = AlignUp(assistInfoForCombine.GetShape(1), 8);
    int32_t countSize = 8;
    int32_t signalCol = 128;
    int32_t cumSumRowShape = AlignUp(routedExpertNum, 256);
    SymbolicScalar thisRank = GetHcclRankId(group);

    Shape shmemDataShape = {expertNumPerRank * epWorldSize, batchSize, hiddenSize};
    auto shmemData = CreateShmemTensor(group, epWorldSize, x.GetDataType(), shmemDataShape);

    Shape shmemInfoShape = {expertNumPerRank * epWorldSize, batchSize, infoSize};
    auto shmemInfo = CreateShmemTensor(group, epWorldSize, DT_INT32, shmemInfoShape);

    Shape shmemCountShape = {1, cumSumRowShape, countSize};
    auto shmemCount = CreateShmemTensor(group, epWorldSize, DT_INT32, shmemCountShape);

    Shape shmemCountSignalShape = {moeExpertNum, 1, signalCol};
    auto shmemCountSignal = CreateShmemTensor(group, epWorldSize, DT_INT32, shmemCountSignalShape);

    Shape shmemDataSignalgShape = {1, 1, signalCol};
    auto shmemDataSignal = CreateShmemTensor(group, epWorldSize, DT_INT32, shmemDataSignalgShape);

    Tensor offsetTable(DataType::DT_INT32, {batchSize, topK}, "offsetTable");
    Tensor expertOffsetCounter = Full(Element(DT_INT32, 0), DT_INT32, {moeExpertNum});
    LOOP("MoeDistributedDispatchPrepare", FunctionType::DYNAMIC_LOOP, i, LoopRange(1))
    {
        (void)i;
        for (int index = 0; index < batchSize * topK; ++index) {
            int32_t rowIndex = index / topK;
            int32_t colIndex = index % topK;
            SymbolicScalar remoteExpertId = GetTensorData(expertIds, {rowIndex, colIndex});
            SymbolicScalar tokenOffset = GetTensorData(expertOffsetCounter, {remoteExpertId});
            SetTensorData(tokenOffset, {rowIndex, colIndex}, offsetTable);
            SetTensorData(tokenOffset + 1, {remoteExpertId}, expertOffsetCounter);
        }
    }

    LOOP("MoeDistributedDispatchSendData", FunctionType::DYNAMIC_LOOP, index, LoopRange(topK * batchSize))
    {
        Tensor moeInfo(DataType::DT_INT32, {1, infoSize}, "moeInfo");
        SymbolicScalar rowIndex = index / topK;
        SymbolicScalar colIndex = index % topK;
        Tensor tensorTile = View(x, {1, hiddenSize}, {rowIndex, 0});
        SetTensorData(thisRank, {0, 0}, moeInfo);
        SetTensorData(rowIndex, {0, 1}, moeInfo);
        SetTensorData(colIndex, {0, 2}, moeInfo);
        SymbolicScalar remoteExpertId = GetTensorData(expertIds, {rowIndex, colIndex});
        SymbolicScalar remoteExpertOffset = remoteExpertId % expertNumPerRank;
        SymbolicScalar remoteRankId = remoteExpertId / expertNumPerRank;
        SymbolicScalar tokenOffset = GetTensorData(offsetTable, {rowIndex, colIndex});
        auto shmemDataTile = ShmemView(
            shmemData, {1, 1, hiddenSize},
            std::vector<SymbolicScalar>{remoteExpertOffset * epWorldSize + thisRank, tokenOffset, 0});
        TileShape::Current().SetVecTile({1, hiddenSize});
        Tensor shmemDataPutOut = ShmemPut(tensorTile, shmemDataTile, remoteRankId, AtomicType::SET, offsetTable);
        auto shmemInfoTile = ShmemView(
            shmemInfo, {1, 1, infoSize},
            std::vector<SymbolicScalar>{remoteExpertOffset * epWorldSize + thisRank, tokenOffset, 0});
        TileShape::Current().SetVecTile({1, infoSize});
        Tensor shmemInfoPutOut = ShmemPut(moeInfo, shmemInfoTile, remoteRankId, AtomicType::SET, offsetTable);
        Tensor sendOut = Nop({shmemDataPutOut, shmemInfoPutOut});
        TileShape::Current().SetVecTile({1, signalCol});
        auto shmemDataSignalTile = ShmemView(shmemDataSignal, {1, 1, signalCol}, {0, 0, 0});
        ShmemSignalAll(shmemDataSignalTile, 0, 1, AtomicType::ADD, sendOut);
    }

    Tensor shmemCountOut(DT_INT32, {1, 1}, "shmemCountOut");
    LOOP("MoeDistributedDispatchSendCount", FunctionType::DYNAMIC_LOOP, expertId, LoopRange(moeExpertNum))
    {
        TileShape::Current().SetVecTile({1, 1});
        SymbolicScalar remoteRankId = expertId / expertNumPerRank;
        SymbolicScalar remoteExpertOffset = expertId % expertNumPerRank;
        auto shmemCountTile = ShmemView(shmemCount, {1, 1, 1}, {0, remoteExpertOffset * epWorldSize + thisRank + 1, 0});
        SymbolicScalar expertCount = GetTensorData(expertOffsetCounter, {expertId});
        Tensor totalOffsetTile = Full(expertCount, DT_INT32, {1, 1});
        Tensor shmemPutOut = ShmemPut(totalOffsetTile, shmemCountTile, remoteRankId, AtomicType::SET, totalOffsetTile);
        TileShape::Current().SetVecTile({1, signalCol});
        auto shmemCountSignalTile = ShmemView(shmemCountSignal, {1, 1, signalCol}, {0, 0, 0});
        shmemCountOut = ShmemSignal(shmemCountSignalTile, 0, remoteRankId, 1, AtomicType::ADD, shmemPutOut);
    }

    Tensor cumSumResult(DT_INT32, {cumSumRowShape, countSize}, "cumSumResult");
    Tensor localExpertRecvCount(DT_INT32, {cumSumRowShape, countSize}, "localExpertRecvCount");
    LOOP("MoeDistributedDispatchCumSum", FunctionType::DYNAMIC_LOOP, i, LoopRange(1))
    {
        (void)i;
        TileShape::Current().SetVecTile({1, signalCol});
        auto shmemDataSignalLocalTile = ShmemView(shmemDataSignal, {1, 1, signalCol}, {0, 0, 0});
        Tensor waitUntilOut1 =
            ShmemWaitUntil(shmemDataSignalLocalTile, 0, OpType::EQ, batchSize * topK * epWorldSize, true, cumSumResult);
        TileShape::Current().SetVecTile({1, signalCol});
        auto shmemCountSignalLocalTile = ShmemView(shmemCountSignal, {1, 1, signalCol}, {0, 0, 0});
        Tensor waitUntilOut =
            ShmemWaitUntil(shmemCountSignalLocalTile, 0, OpType::EQ, moeExpertNum, true, cumSumResult);
        Tensor waitOut = Nop({waitUntilOut1, waitUntilOut});

        TileShape::Current().SetVecTile({cumSumRowShape, countSize});
        auto shmemReceiveCountTile = ShmemView(shmemCount, {1, cumSumRowShape, countSize}, {0, 0, 0});
        localExpertRecvCount = ShmemGet(shmemReceiveCountTile, thisRank, waitOut);
        TileShape::Current().SetVecTile({cumSumRowShape, countSize});
        auto shmemCountTile = ShmemView(shmemCount, {1, cumSumRowShape, countSize}, {0, 0, 0});
        Tensor shmemGetOut = ShmemGet(shmemCountTile, thisRank, waitOut);
        Tensor cumSumCurrent = CumSum(shmemGetOut, 0);
        cumSumResult = Cast(cumSumCurrent, DT_INT32, CAST_TRUNC);

        SymbolicScalar recvCountResult = GetTensorData(cumSumResult, {expertNumPerRank * epWorldSize, 0});
        SetTensorData(recvCountResult, {0}, recvCounts);

        for (int32_t expertId = 0; expertId < expertNumPerRank; ++expertId) {
            Tensor expertValidCnt = View(shmemGetOut, {epWorldSize, countSize}, {expertId * epWorldSize + 1, 0});
            Tensor expertValidCumSum = CumSum(expertValidCnt, 0);
            Tensor expertCumSumResult = Cast(expertValidCumSum, DT_INT32, CAST_TRUNC);
            SymbolicScalar recvValidResult = GetTensorData(expertCumSumResult, {epWorldSize - 1, 0});
            SetTensorData(recvValidResult, {expertId}, expertTokenNums);
        }
    }

    LOOP("MoeDistributedDispatchReceive", FunctionType::DYNAMIC_LOOP, i, LoopRange(1))
    {
        (void)i;
        for (uint32_t index = 0; index < expertNumPerRank * epWorldSize; ++index) {
            SymbolicScalar curCount = GetTensorData(localExpertRecvCount, {index + 1, 0});
            SymbolicScalar offset = GetTensorData(cumSumResult, {index, 0});
            auto curShmemDataTile = ShmemView(
                shmemData, {1, batchSize, hiddenSize}, std::vector<SymbolicScalar>{1, curCount, hiddenSize},
                {index, 0, 0});
            TileShape::Current().SetVecTile({batchSize, hiddenSize});
            Tensor localDataRecvCount = ShmemLoad(curShmemDataTile, thisRank, cumSumResult);
            Assemble(localDataRecvCount, std::vector<SymbolicScalar>{offset, 0}, expandX);
            auto curShmemInfoTile = ShmemView(
                shmemInfo, {1, batchSize, assistInfoForCombine.GetShape(1)},
                std::vector<SymbolicScalar>{1, curCount, assistInfoForCombine.GetShape(1)}, {index, 0, 0});
            TileShape::Current().SetVecTile({batchSize, assistInfoForCombine.GetShape(1)});
            Tensor localInfoRecvCount = ShmemLoad(curShmemInfoTile, thisRank, cumSumResult);
            Assemble(localInfoRecvCount, std::vector<SymbolicScalar>{offset, 0}, assistInfoForCombine);
        }
    }
}

void MoeDistributedDispatchBackwardValidateV2(
    const Tensor& gradExpandX, const Tensor& assistInfoForCombine, const Tensor& recvCounts, const char* group,
    uint32_t epWorldSize, uint32_t moeExpertNum, uint32_t sharedExpertNum, uint32_t sharedExpertRankNum, Tensor& gradX)
{
    std::string assertResult;
    CHECK(group != nullptr) << "MoeDispatchBackwardV2 constraint violated: group name can't be nullptr.";
    CHECK(group[0] != '\0') << "MoeDispatchBackwardV2 constraint violated: group name must be valid, but got '\\0'";
    CHECK(strnlen(group, 128) < 128)
        << "MoeDispatchBackwardV2 constraint violated: group name max size must be 128, but got " << strnlen(group, 128);
    CHECK(epWorldSize > 0) << "MoeDispatchBackwardV2 constraint violated: epWorldSize must be > 0, but got "
                           << epWorldSize;
    CHECK(sharedExpertNum == 0) << "MoeDispatchBackwardV2 constraint violated: sharedExpertNum must be 0, but got "
                                << sharedExpertNum;
    CHECK(sharedExpertRankNum == 0)
        << "MoeDispatchBackwardV2 constraint violated: sharedExpertRankNum must be 0, but got " << sharedExpertRankNum;

    int32_t batchSize = gradX.GetShape(0);
    int32_t hiddenSize = gradX.GetShape(1);
    int32_t topK = InferSupportedMoeTopK(
        batchSize, hiddenSize, static_cast<int32_t>(epWorldSize), static_cast<int32_t>(moeExpertNum),
        static_cast<int32_t>(sharedExpertNum), static_cast<int32_t>(sharedExpertRankNum));
    CHECK(topK > 0) << "MoeDispatchBackwardV2 constraint violated: unsupported shape configuration.";

    int32_t expandXRow = std::min(
        static_cast<int32_t>(batchSize) * topK * static_cast<int32_t>(epWorldSize),
        static_cast<int32_t>(batchSize) * static_cast<int32_t>(moeExpertNum));
    CHECK(checkValidInput(gradExpandX, 2, DataType::DT_BF16, expandXRow, hiddenSize, assertResult)) << assertResult;
    CHECK(checkValidInput(assistInfoForCombine, 2, DataType::DT_INT32, expandXRow, 3, assertResult)) << assertResult;
    CHECK(checkValidInput(recvCounts, 1, DataType::DT_INT32, 1, 0, assertResult)) << assertResult;
    CHECK(checkValidInput(gradX, 2, DataType::DT_BF16, batchSize, hiddenSize, assertResult)) << assertResult;

    constexpr uint64_t signalCol = 128;
    uint64_t shmemSize =
        static_cast<uint64_t>(batchSize) * static_cast<uint64_t>(topK) * static_cast<uint64_t>(hiddenSize) *
            BytesOf(gradExpandX.GetDataType()) +
        static_cast<uint64_t>(batchSize) * signalCol * BytesOf(DataType::DT_INT32);
    CHECK(shmemSize < MOE_V2_MAX_WIN_SIZE)
        << "MoeDispatchBackwardV2 constraint violated: shmem window exceeds the limit. Maximum allowed: "
        << MOE_V2_MAX_WIN_SIZE << ", got: " << shmemSize;
}

void MoeDistributedDispatchBackwardV2(
    const Tensor& gradExpandX, const Tensor& assistInfoForCombine, const Tensor& recvCounts, const char* group,
    uint32_t epWorldSize, uint32_t moeExpertNum, uint32_t sharedExpertNum, uint32_t sharedExpertRankNum, Tensor& gradX)
{
    MoeDistributedDispatchBackwardValidateV2(
        gradExpandX, assistInfoForCombine, recvCounts, group, epWorldSize, moeExpertNum, sharedExpertNum,
        sharedExpertRankNum, gradX);

    int32_t batchSize = gradX.GetShape(0);
    int32_t hiddenSize = gradX.GetShape(1);
    int32_t topK = InferSupportedMoeTopK(
        batchSize, hiddenSize, static_cast<int32_t>(epWorldSize), static_cast<int32_t>(moeExpertNum),
        static_cast<int32_t>(sharedExpertNum), static_cast<int32_t>(sharedExpertRankNum));
    constexpr int32_t signalCol = 128;

    auto shmemDataTensor =
        CreateShmemTensor(group, epWorldSize, gradExpandX.GetDataType(), {1, batchSize * topK, hiddenSize});
    auto shmemSignalTensor = CreateShmemTensor(group, epWorldSize, DT_INT32, {1, batchSize, signalCol});

    SymbolicScalar recvCountsScalar = GetTensorData(recvCounts, {0});
    std::set<int> unrollList =
        GetMoeDispatchBackwardUnrollList(batchSize, hiddenSize, topK, static_cast<int32_t>(epWorldSize));
    LOOP(
        "MoeDistributedDispatchBackwardSend", FunctionType::DYNAMIC_LOOP, rowIndex, LoopRange(recvCountsScalar),
        unrollList)
    {
        SymbolicScalar rankId = GetTensorData(assistInfoForCombine, {rowIndex, 0});
        SymbolicScalar tokenId = GetTensorData(assistInfoForCombine, {rowIndex, 1});
        SymbolicScalar kOffset = GetTensorData(assistInfoForCombine, {rowIndex, 2});

        Tensor gradExpandXTile = View(gradExpandX, {1, hiddenSize}, {rowIndex, 0});
        auto shmemDataTile = ShmemView(shmemDataTensor, {1, 1, hiddenSize}, {0, topK * tokenId + kOffset, 0});
        TileShape::Current().SetVecTile({1, hiddenSize});
        Tensor predToken(DT_INT32, {1, 1}, "dispatchBackwardSendPredToken");
        Tensor shmemPutOut = ShmemPut(gradExpandXTile, shmemDataTile, rankId, AtomicType::SET, predToken);

        auto shmemSignalTile = ShmemView(shmemSignalTensor, {1, 1, signalCol}, {0, tokenId, 0});
        TileShape::Current().SetVecTile({1, signalCol});
        ShmemSignal(shmemSignalTile, rankId, rankId, 1, AtomicType::ADD, shmemPutOut);
    }

    SymbolicScalar thisRank = GetHcclRankId(group);
    LOOP("MoeDistributedDispatchBackwardReceive", FunctionType::DYNAMIC_LOOP, tokenId, LoopRange(batchSize))
    {
        auto shmemSignalTile = ShmemView(shmemSignalTensor, {1, 1, signalCol}, {0, tokenId, 0});
        TileShape::Current().SetVecTile({1, signalCol});
        Tensor predToken(DT_INT32, {1, 1}, "dispatchBackwardReceivePredToken");
        Tensor waitUntilOut = ShmemWaitUntil(shmemSignalTile, thisRank, OpType::EQ, topK, true, predToken);

        TileShape::Current().SetVecTile({topK, hiddenSize});
        auto shmemDataTile = ShmemView(shmemDataTensor, {1, topK, hiddenSize}, {0, topK * tokenId, 0});
        Tensor shmemGetOut = ShmemGet(shmemDataTile, thisRank, waitUntilOut);

        Tensor gradXTileFp32;
        if (topK == QWEN3_NEXT_TOPK) {
            TileShape::Current().SetVecTile({1, hiddenSize});
            Tensor token0Fp32 = Cast(View(shmemGetOut, {1, hiddenSize}, {0, 0}), DT_FP32);
            Tensor token1Fp32 = Cast(View(shmemGetOut, {1, hiddenSize}, {1, 0}), DT_FP32);
            gradXTileFp32 = Add(token0Fp32, token1Fp32);
        } else {
            Tensor shmemGetOutFp32 = Cast(shmemGetOut, DT_FP32);
            TileShape::Current().SetVecTile({topK, hiddenSize});
            gradXTileFp32 = Sum(shmemGetOutFp32, 0, true);
        }

        Tensor gradXTile = Cast(gradXTileFp32, DT_BF16);
        Assemble(gradXTile, {tokenId, 0}, gradX);
    }
}
} // namespace Distributed
} // namespace npu::tile_fwk
