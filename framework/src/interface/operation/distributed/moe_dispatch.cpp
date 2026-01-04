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
#include "interface/utils/log.h"
#include "distributed_common.h"
#include "tilefwk/symbolic_distributed.h"

namespace npu::tile_fwk {
namespace Distributed {
void TiledDispatchWaitRecvFlags(Function& function, const TileShape& tileShape,
    const std::vector<std::shared_ptr<LogicalTensor>>& iOperand,
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

    const auto &tileRank = tileShape.GetDistTileRank();
    int32_t totalTileNum = GetTotalTileNum(tileRank) * static_cast<int32_t>(expertNumPerRank);
    const int32_t tileRankShape = tileRank[DIST_HEAD_SHAPE];
    const int32_t tileRankCnt = tileRank[DIST_HEAD_COUNT] + (tileRank[DIST_TAIL_SHAPE] == 0 ? 0 : 1);
    const int32_t tailRankShape = tileRank[DIST_TAIL_SHAPE];
    int32_t tileIndex = 0;
    for (int expertIndex = 0; expertIndex < expertNumPerRank; ++expertIndex) {
        for (int rankIndex = 0; rankIndex < tileRankCnt; ++rankIndex) {
            int32_t rankShape = ((tileRank[2] != 0) && (rankIndex == tileRankCnt - 1) ? tailRankShape :tileRankShape);
            int32_t rankOffset = rankIndex * tileRankShape;
            auto bufferTensor = std::make_shared<LogicalTensor>(function, DataType::DT_INT32, bufferShape);
            auto shmemFlagTile = shmemFlag->View(function, {1, 1, rankShape, flagColSize}, 
                {0, expertIndex, rankOffset, 0});
            auto &opr = function.AddOperation(Opcode::OP_MOE_DISPATCH_WAIT_FLAG, {syncTensor, shmemFlagTile}, 
                {recvTokenCntOut, bufferTensor});
            std::string extraParam = std::to_string(tileIndex) + ", " + hcclGroupIndex + ", " + 
                std::to_string(sharedExpertNum) + ", " + std::to_string(totalTileNum) + ", " + 
                std::to_string(rankShape) + ", " + std::to_string(expertNumPerRank);
            DistOpAttr distOpAttr;
            distOpAttr.extraTemplateParam = extraParam;
            opr.SetAttr(OpAttributeKey::distOpAttr, distOpAttr);
            tileIndex++;
        }
    }
}

void TiledDispatchAssembleCombineInfo(Function& function, const TileShape& tileShape,
    const std::vector<std::shared_ptr<LogicalTensor>>& iOperand,
    const std::vector<std::shared_ptr<LogicalTensor>>& oOperand, const Operation& op)
{
    auto recvTokenCntOut = iOperand[DIST_INDEX_ZERO];
    auto shmemData = iOperand[DIST_INDEX_ONE];
    auto shmemFlag = iOperand[DIST_INDEX_TWO];
    auto assistInfoForCombine = oOperand[DIST_INDEX_ZERO];

    int32_t shmemDataLength = shmemData->GetShape()[3];
    Shape assistInfoForCombineBufferShape = {assistInfoForCombine->GetShape()[0] + 32};
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

    const auto &tileRank = tileShape.GetDistTileRank();
    int32_t totalTileNum = GetTotalTileNum(tileRank) * static_cast<int32_t>(expertNumPerRank);
    const int32_t tileRankShape = tileRank[DIST_HEAD_SHAPE];
    const int32_t tileRankCnt = tileRank[DIST_HEAD_COUNT] + (tileRank[DIST_TAIL_SHAPE] == 0 ? 0 : 1);
    const int32_t tailRankShape = tileRank[DIST_TAIL_SHAPE];

    int32_t tileIndex = 0;
    for (int expertIndex = 0; expertIndex < expertNumPerRank; ++expertIndex) {
        for (int rankIndex = 0; rankIndex < tileRankCnt; ++rankIndex) {
            int32_t rankShape = ((tileRank[2] != 0) && (rankIndex == tileRankCnt - 1) ? tailRankShape :tileRankShape);
            int32_t rankOffset = rankIndex * tileRankShape;
            auto bufferCombineInfo = std::make_shared<LogicalTensor>(function, DataType::DT_INT32, bufferShape);
            auto shmemDataTile = shmemData->View(function, {1, rankShape, 1, shmemDataLength}, 
                {0, rankOffset, expertIndex, 0});
            auto &opr = function.AddOperation(Opcode::OP_MOE_DISPATCH_ASSEMBLE_COMBINEINFO, {shmemDataTile, shmemFlag, recvTokenCntOut}, 
                {assistInfoForCombine, bufferCombineInfo});
            std::string extraParam = std::to_string(tileIndex) + ", " + hcclGroupIndex + ", " +
                std::to_string(sharedExpertNum) + ", " + std::to_string(totalTileNum) + ", " +
                std::to_string(rankShape) + ", " + axisH + ", " + batchSize + ", " +
                std::to_string(assistInfoForCombine->GetShape()[0]);
            DistOpAttr distOpAttr;
            distOpAttr.extraTemplateParam = extraParam;
            opr.SetAttr(OpAttributeKey::distOpAttr, distOpAttr);
            tileIndex++;
        }
    }
}

void TiledDispatchAssembleExpandX(Function& function, const TileShape& tileShape,
    const std::vector<std::shared_ptr<LogicalTensor>>& iOperand,
    const std::vector<std::shared_ptr<LogicalTensor>>& oOperand, const Operation& op)
{
    auto recvTokenCntOut = iOperand[DIST_INDEX_ZERO];
    auto shmemData = iOperand[DIST_INDEX_ONE];
    auto shmemFlag = iOperand[DIST_INDEX_TWO];
    auto expandX = oOperand[DIST_INDEX_ZERO];
    auto recvCounts = oOperand[DIST_INDEX_ONE];

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

    const auto &tileRank = tileShape.GetDistTileRank();
    int32_t totalTileNum = GetTotalTileNum(tileRank) * static_cast<int32_t>(expertNumPerRank);
    const int32_t tileRankShape = tileRank[DIST_HEAD_SHAPE];
    const int32_t tileRankCnt = tileRank[DIST_HEAD_COUNT] + (tileRank[DIST_TAIL_SHAPE] == 0 ? 0 : 1);
    const int32_t tailRankShape = tileRank[DIST_TAIL_SHAPE];

    int32_t tileIndex = 0;
    for (int expertIndex = 0; expertIndex < expertNumPerRank; ++expertIndex) {
        for (int rankIndex = 0; rankIndex < tileRankCnt; ++rankIndex) {
            int32_t rankShape = ((tileRank[2] != 0) && (rankIndex == tileRankCnt - 1) ? tailRankShape :tileRankShape);
            int32_t rankOffset = rankIndex * tileRankShape;
            auto bufferTensor = std::make_shared<LogicalTensor>(function, DataType::DT_INT32, bufferShape);
            auto shmemDataTile = shmemData->View(function, {1, rankShape, 1, shmemDataLength}, 
                {0, rankOffset, expertIndex, 0});
            auto &opr = function.AddOperation(Opcode::OP_MOE_DISPATCH_ASSEMBLE_EXPANDX, {shmemDataTile, shmemFlag, recvTokenCntOut}, 
                {expandX, recvCounts, bufferTensor});
            std::string extraParam = std::to_string(tileIndex) + ", " + groupIndex + ", " +
                std::to_string(sharedExpertNum) + ", " + std::to_string(totalTileNum) + ", " +
                std::to_string(rankShape) + ", " + axisH + ", " + batchSize + ", " +
                std::to_string(expandX->GetShape()[0]);
            DistOpAttr distOpAttr;
            distOpAttr.extraTemplateParam = extraParam;
            opr.SetAttr(OpAttributeKey::distOpAttr, distOpAttr);
            tileIndex++;
        }
    }
}

void TiledDispatchBuildExpertTokenNum(Function& function, const TileShape& tileShape,
    const std::vector<std::shared_ptr<LogicalTensor>>& iOperand,
    const std::vector<std::shared_ptr<LogicalTensor>>& oOperand, const Operation& op)
{
    (void) op;
    auto recvTokenCntOut = iOperand[DIST_INDEX_ZERO];
    auto shmemFlag = iOperand[DIST_INDEX_ONE];
    auto expertTokenNums = oOperand[DIST_INDEX_ZERO];

    int32_t flagColSize = shmemFlag->GetShape()[3];
    int32_t rankSize = shmemFlag->GetShape()[0];

    const auto& tileExpert = tileShape.GetDistTileRank();
    int32_t tileExpertShape = tileExpert[0];
    int32_t expertCount = tileExpert[1] + (tileExpert[2] == 0 ? 0 : 1);
    Shape bufferShape {shmemFlag->shape[0] * expertCount};

    for (int32_t expertIndex = 0; expertIndex < expertCount; ++expertIndex) {
        int32_t expertShape = ((tileExpert[2] != 0) && (expertIndex == expertCount - 1)) ? tileExpert[2] : tileExpert[0];
        int32_t expertOffset = expertIndex * tileExpertShape;
        auto expertTokenNumsBuffer = std::make_shared<LogicalTensor>(function, DataType::DT_INT32, bufferShape);
        auto shmemFlagTile = shmemFlag->View(function, {1, expertShape, rankSize, flagColSize},
            {0, expertOffset, 0, 0});
        auto &tileop = function.AddOperation(Opcode::OP_MOE_DISPATCH_EXPERT_TOKEN_NUM, {recvTokenCntOut, shmemFlagTile},
            {expertTokenNums, expertTokenNumsBuffer});
        std::string extraParam = std::to_string(expertShape);
        DistOpAttr distOpAttr;
        distOpAttr.extraTemplateParam = extraParam;
        tileop.SetAttr(OpAttributeKey::distOpAttr, distOpAttr);
    }
}

Tensor DispatchBuildExpertTokenNum(const Tensor& recvTokenCntOut, const Tensor& shmemFlag, uint32_t expertNumPerRank)
{
    auto &function = *Program::GetInstance().GetCurrentFunction();
    Shape expertTokenNumsShape = {expertNumPerRank, 1};
    auto expertTokenNumsPtr = std::make_shared<LogicalTensor>(function, DataType::DT_INT32, expertTokenNumsShape);
    auto &oper = function.AddOperation(Opcode::OP_MOE_DISPATCH_EXPERT_TOKEN_NUM, {recvTokenCntOut.GetStorage(), shmemFlag.GetStorage()}, {expertTokenNumsPtr});
    (void)oper;
    return expertTokenNumsPtr;
}

Tensor DispatchAssembleCombineInfo(const char *group, const Tensor &x,
    const Tensor &recvTokenCntOut, const Tensor &shmemData, const Tensor &shmemFlag,
    int32_t expandXRow, int32_t ffnTileNum, uint32_t expertNumPerRank)
{
    auto &function = *Program::GetInstance().GetCurrentFunction();
    Shape assistInfoForCombineShape = {expandXRow, 3};
    auto assistInfoForCombinePtr = std::make_shared<LogicalTensor>(function, DataType::DT_INT32, assistInfoForCombineShape);
    auto &oper = function.AddOperation(Opcode::OP_MOE_DISPATCH_ASSEMBLE_COMBINEINFO, {recvTokenCntOut.GetStorage(), shmemData.GetStorage(),
        shmemFlag.GetStorage()}, {assistInfoForCombinePtr});
    int tempBufSize = AlignUp(expertNumPerRank * ffnTileNum * 32, 256) + 256 +
        AlignUp(expertNumPerRank * ffnTileNum * 4, 32) + 512; // tempBufSize = recvTokenCnt数 + 存储mask + recvTokenCnt的int数 + 数据搬运
    std::string hcclGroupIndex = std::to_string(CommGroupRecorder::GetInstance().Input(std::string(group)));
    const std::vector<int64_t> bufferShape {tempBufSize};
    oper.SetAttr("hcclGroupIndex", hcclGroupIndex);
    oper.SetAttr("dispatchBufferSize", bufferShape);
    oper.SetAttr("hiddenSize", std::to_string(x.GetShape()[1]));
    oper.SetAttr("tokenBatchSize", std::to_string(x.GetShape()[0]));
    oper.SetAttr("expertNumPerRank", static_cast<int64_t>(expertNumPerRank));
    return assistInfoForCombinePtr;
}

std::tuple<Tensor, Tensor> DispatchAssembleExpandX(const char *group, const Tensor &x, 
    const Tensor &recvTokenCntOut, const Tensor &shmemData, const Tensor &shmemFlag, 
    int32_t expandXRow, int32_t ffnTileNum, uint32_t expertNumPerRank)
{
    auto &function = *Program::GetInstance().GetCurrentFunction();
    Shape recvCountsShape = {1};
    auto recvCountsPtr = std::make_shared<LogicalTensor>(function, DataType::DT_INT32, recvCountsShape);
    Shape expandXShape = {expandXRow, x.GetShape()[1]};
    auto expandXPtr = std::make_shared<LogicalTensor>(function, x.GetDataType(), expandXShape);
    auto &oper = function.AddOperation(Opcode::OP_MOE_DISPATCH_ASSEMBLE_EXPANDX, {recvTokenCntOut.GetStorage(), shmemData.GetStorage(),
        shmemFlag.GetStorage()}, {expandXPtr, recvCountsPtr});
    int cumSumBuffer = AlignUp(expertNumPerRank * ffnTileNum * 32, 256)
        + 256 + AlignUp(expertNumPerRank * ffnTileNum * 4, 32) + 512; // tempBufSize = recvTokenCnt数 + 存储mask + recvTokenCnt的int数 + 数据搬运
    int tokenCopyBuffer = x.GetShape(1);
    int tempBufSize = (cumSumBuffer < tokenCopyBuffer) ? tokenCopyBuffer : cumSumBuffer;
    std::string hcclGroupIndex = std::to_string(CommGroupRecorder::GetInstance().Input(std::string(group)));
    const std::vector<int64_t> bufferShape {tempBufSize};
    oper.SetAttr("hcclGroupIndex", hcclGroupIndex);
    oper.SetAttr("dispatchBufferSize", bufferShape);
    oper.SetAttr("hiddenSize", std::to_string(x.GetShape()[1]));
    oper.SetAttr("tokenBatchSize", std::to_string(x.GetShape()[0]));
    oper.SetAttr("expertNumPerRank", static_cast<int64_t>(expertNumPerRank));
    return {expandXPtr, recvCountsPtr};
}

Tensor DispatchWaitRecvFlags(const char *group, const Tensor &flagDummy, Tensor &shmemFlag, uint32_t expertNumPerRank, int32_t ffnTileNum)
{
    auto &function = *Program::GetInstance().GetCurrentFunction();
    int32_t totalTileNum = expertNumPerRank * ffnTileNum;
    Shape shape = {totalTileNum, 128}; // 每个flag_count预留512个int存储
    auto recvTokenCntOutPtr = std::make_shared<LogicalTensor>(function, DataType::DT_INT32, shape);
    auto &oper = function.AddOperation(Opcode::OP_MOE_DISPATCH_WAIT_FLAG, {flagDummy.GetStorage(), shmemFlag.GetStorage()},
        {recvTokenCntOutPtr});
    int32_t moeOpProcessRankSize = ffnTileNum;
    int32_t maxProcessRankSize = moeOpProcessRankSize;
    int tempBufSize = maxProcessRankSize * 32 + 256 + AlignUp(maxProcessRankSize * 4, 256);
    std::string hcclGroupIndex = std::to_string(CommGroupRecorder::GetInstance().Input(std::string(group)));
    oper.SetAttr("hcclGroupIndex", hcclGroupIndex);
    const std::vector<int64_t> bufferShape {tempBufSize / 8, 8};
    oper.SetAttr("dispatchBufferSize", bufferShape);
    oper.SetAttr("expertNumPerRank", static_cast<int64_t>(expertNumPerRank));
    return recvTokenCntOutPtr;
}

std::vector<int64_t> GetCommBufferSize(const std::shared_ptr<LogicalTensor> &x)
{
    const int64_t hOutSize = x->shape[1] * BytesOf(x->Datatype());
    constexpr int64_t scaleParamPad = 512;
    const int64_t hCommuSize = AlignUp(hOutSize, 512) + scaleParamPad;
    return {1, static_cast<int64_t>(hCommuSize / BytesOf(x->Datatype()))};
}

void TiledDispatchSendToRoutingExperts(Function& function, const TileShape& tileShape,
    const std::vector<std::shared_ptr<LogicalTensor>>& iOperand,
    const std::vector<std::shared_ptr<LogicalTensor>>& oOperand, const Operation& op)
{
    auto shmemData = iOperand[DIST_INDEX_ZERO];
    auto x = iOperand[DIST_INDEX_ONE];
    auto expertTable = iOperand[DIST_INDEX_TWO];
    auto syncTensor = oOperand[DIST_INDEX_ZERO];
    std::string hcclGroupIndex;
    int64_t expertNumPerRank;
    op.GetAttr("expertNumPerRank", expertNumPerRank);
    op.GetAttr("hcclGroupIndex", hcclGroupIndex);
    CreateTileOp(tileShape,
        [&](int32_t tileIndex, int32_t rowOffset, int32_t colOffset, int32_t rowShape, int32_t colShape) {
            (void) tileIndex;
            auto expertTableTile = expertTable->View(function, {rowShape, colShape}, {rowOffset, colOffset});
            auto expertBufferUb = std::make_shared<LogicalTensor>(function, expertTable->Datatype(),
                std::vector<int64_t>{1, expertTable->shape[0] * expertTable->shape[1]});
            auto expertBuffer = std::make_shared<LogicalTensor>(function, expertTable->Datatype(),
                std::vector<int64_t>{1, expertTable->shape[0] * expertTable->shape[1] *
                (static_cast<int64_t>(sizeof(int32_t)) + 1)});
            auto tokenBuffer = std::make_shared<LogicalTensor>(function, x->Datatype(), 
                GetCommBufferSize(x));
            auto &tileop = function.AddOperation(Opcode::OP_MOE_DISPATCH_SEND_ROUTED, {x, shmemData, 
                expertTableTile}, {syncTensor, tokenBuffer, expertBufferUb, expertBuffer});
            std::string extraParam = std::to_string(x->shape[1]) + ", " + std::to_string(rowOffset) +
                ", " + std::to_string(colOffset) + ", " + std::to_string(rowShape) +
                ", " + std::to_string(colShape) + ", " + hcclGroupIndex;
            DistOpAttr distOpAttr;
            distOpAttr.extraTemplateParam = extraParam;
            tileop.SetAttr(OpAttributeKey::distOpAttr, distOpAttr);
        });
}

void TiledSendToSharedExpert(Function& function, const TileShape& tileShape,
    const std::vector<std::shared_ptr<LogicalTensor>>& iOperand,
    const std::vector<std::shared_ptr<LogicalTensor>>& oOperand, const Operation& op)
{
    auto shmemData = iOperand[DIST_INDEX_ZERO];
    auto x = iOperand[DIST_INDEX_ONE];
    auto syncTensor = oOperand[DIST_INDEX_ZERO];
    (void) oOperand;
    std::string hcclGroupIndex;
    op.GetAttr("hcclGroupIndex", hcclGroupIndex);
    CreateTileOp(tileShape,
        [&](int32_t tileIndex, int32_t rowOffset, int32_t colOffset, int32_t rowShape, int32_t colShape) {
            (void) tileIndex;
            Shape shape = {rowShape, colShape};
            auto xTile = x->View(function, {rowShape, colShape}, {rowOffset, colOffset});
            auto tokenBuffer = std::make_shared<LogicalTensor>(function, x->Datatype(), 
                GetCommBufferSize(x));
            auto &tileop = function.AddOperation(Opcode::OP_MOE_DISPATCH_SEND_SHARED, {xTile, shmemData},
                {syncTensor, tokenBuffer});
            std::string extraParam = std::to_string(x->shape[0]) + ", " +
                std::to_string(x->shape[1]) + ", " + std::to_string(rowShape) + ", " + hcclGroupIndex;
            DistOpAttr distOpAttr;
            distOpAttr.extraTemplateParam = extraParam;
            tileop.SetAttr(OpAttributeKey::distOpAttr, distOpAttr);
        });
}

void TiledCopyToLocalExpert(Function& function, const TileShape& tileShape,
    const std::vector<std::shared_ptr<LogicalTensor>>& iOperand,
    const std::vector<std::shared_ptr<LogicalTensor>>& oOperand, const Operation& op)
{
    auto x = iOperand[DIST_INDEX_ZERO];
    auto expandX = oOperand[DIST_INDEX_ZERO];
    auto syncTensor = oOperand[DIST_INDEX_ONE];
    (void) op;
    CreateTileOp(tileShape,
        [&](int32_t tileIndex, int32_t rowOffset, int32_t colOffset, int32_t rowShape, int32_t colShape) {
            (void) tileIndex;
            auto xTile = x->View(function, {rowShape, colShape}, {rowOffset, colOffset});
            auto tokenBuffer = std::make_shared<LogicalTensor>(function, x->Datatype(), 
                GetCommBufferSize(x));
            auto &tileop = function.AddOperation(Opcode::OP_MOE_DISPATCH_LOCAL_COPY_OUT, {xTile},
                {expandX, syncTensor, tokenBuffer});
            std::string extraParam = std::to_string(x->shape[0]) + ", " +
                std::to_string(x->shape[1]) + ", " + std::to_string(rowShape);
            DistOpAttr distOpAttr;
            distOpAttr.extraTemplateParam = extraParam;
            tileop.SetAttr(OpAttributeKey::distOpAttr, distOpAttr);
        });
}

void TiledDispatchSetRecvFlags(Function& function, const TileShape& tileShape,
    const std::vector<std::shared_ptr<LogicalTensor>>& iOperand,
    const std::vector<std::shared_ptr<LogicalTensor>>& oOperand, const Operation& op)
{
    auto shmemFlag = iOperand[DIST_INDEX_ZERO];
    auto syncTensor = iOperand[DIST_INDEX_ONE];
    auto expertIds = iOperand[DIST_INDEX_TWO];
    auto syncDummy = oOperand[DIST_INDEX_ZERO];
    int flagColSize = shmemFlag->GetShape()[3];
    std::string hcclGroupIndex;
    op.GetAttr("hcclGroupIndex", hcclGroupIndex);
    int64_t expertNumPerRank;
    op.GetAttr("expertNumPerRank", expertNumPerRank);

    const auto &tileExpert = tileShape.GetDistTileRank();
    const auto &tileRank = tileShape.GetDistTileCol();
    int32_t tileRankShape = tileRank[0];
    int32_t tileExpertShape = tileExpert[0];
    int32_t rankCount = tileRank[1] + (tileRank[2] == 0 ? 0 : 1);
    int32_t expertCount = tileExpert[1] + (tileExpert[2] == 0 ? 0 : 1);

    for (int32_t rankIndex = 0; rankIndex < rankCount; ++rankIndex) {
        int32_t rankShape = ((tileRank[2] != 0) && (rankIndex == rankCount - 1)) ? tileRank[2] : tileRank[0];
        for (int32_t expertIndex = 0; expertIndex < expertCount; ++expertIndex) {
            int32_t expertShape = ((tileExpert[2] != 0) && (expertIndex == expertCount - 1)) ?
                tileExpert[2] : tileExpert[0];
            int32_t rankOffset = rankIndex * tileRankShape;
            int32_t expertOffset = expertIndex * tileExpertShape;
            auto statusTensor = std::make_shared<LogicalTensor>(function, expertIds->Datatype(),
                std::vector<int64_t>{1, expertNumPerRank * 16 + 32}); // 每个expert预留16B缓存flag跟count,最后一个expert后预留32位
            auto expertBufferUb = std::make_shared<LogicalTensor>(function, expertIds->Datatype(),
                std::vector<int64_t>{1, expertIds->shape[0] * expertIds->shape[1]});
            auto expertBuffer = std::make_shared<LogicalTensor>(function, expertIds->Datatype(),
                std::vector<int64_t>{1, expertIds->shape[0] * expertIds->shape[1] *
                (static_cast<int64_t>(sizeof(int32_t)) + 1)});
            auto shmemFlagTile = shmemFlag->View(function, {rankShape, expertShape, 1, flagColSize}, 
                {rankOffset, expertOffset, 0, 0});
            auto &tileop = function.AddOperation(Opcode::OP_MOE_DISPATCH_SET_FLAG, {expertIds, shmemFlagTile, 
                syncTensor}, {syncDummy, statusTensor, expertBufferUb, expertBuffer}); 
            std::string extraParam = std::to_string(expertIds->shape[0]) + ", " +
                std::to_string(expertIds->shape[1]) + ", " + hcclGroupIndex + ", " +
                std::to_string(expertShape) + ", " + std::to_string(rankShape);
            DistOpAttr distOpAttr;
            distOpAttr.extraTemplateParam = extraParam;
            tileop.SetAttr(OpAttributeKey::distOpAttr, distOpAttr);
        }
    }
}

Tensor DispatchSendToRoutingExperts(const Tensor &shmemData, const Tensor &x,
    const Tensor &expertIds, const char *group, uint32_t expertNumPerRank)
{
    Shape shape{1, 1};
    auto &function = *Program::GetInstance().GetCurrentFunction();
    auto syncTensor = std::make_shared<LogicalTensor>(function, DataType::DT_INT32, shape);
    auto &oper = function.AddOperation(Opcode::OP_MOE_DISPATCH_SEND_ROUTED, {shmemData.GetStorage(),
        x.GetStorage(), expertIds.GetStorage()}, {syncTensor});
    std::string hcclGroupIndex = std::to_string(CommGroupRecorder::GetInstance().Input(std::string(group)));    
    oper.SetAttr("hcclGroupIndex", hcclGroupIndex);
    oper.SetAttr("expertNumPerRank", static_cast<int64_t>(expertNumPerRank));
    return syncTensor;
}

void SendToSharedExpert(const Tensor &shmemData, const Tensor &x, 
    const Tensor &syncTensor, const char *group)
{
    auto &function = *Program::GetInstance().GetCurrentFunction();
    auto &oper = function.AddOperation(Opcode::OP_MOE_DISPATCH_SEND_SHARED, {shmemData.GetStorage(), 
        x.GetStorage()},
        {syncTensor.GetStorage()});
    std::string hcclGroupIndex = std::to_string(CommGroupRecorder::GetInstance().Input(std::string(group)));
    oper.SetAttr("hcclGroupIndex", hcclGroupIndex);
}

Tensor DispatchSetRecvFlags(Tensor &shmemFlag, const Tensor &expertIds, const Tensor &syncTensor,
    const char *group, uint32_t expertNumPerRank)
{
    Shape shape = {1, 1};
    auto &function = *Program::GetInstance().GetCurrentFunction();
    auto syncDummy = std::make_shared<LogicalTensor>(function, DataType::DT_INT32, shape);
    auto &oper = function.AddOperation(Opcode::OP_MOE_DISPATCH_SET_FLAG, {shmemFlag.GetStorage(), syncTensor.GetStorage(), 
        expertIds.GetStorage()}, {syncDummy});
    std::string hcclGroupIndex = std::to_string(CommGroupRecorder::GetInstance().Input(std::string(group)));
    oper.SetAttr("hcclGroupIndex", hcclGroupIndex);
    oper.SetAttr("expertNumPerRank", static_cast<int64_t>(expertNumPerRank));
    return syncDummy;
}

Tensor CopyToLocalExpert(const Tensor &x, const Tensor &syncTensor, uint32_t routedExpertNum)
{
    auto &function = *Program::GetInstance().GetCurrentFunction();
    Shape expandXShape = {x.GetShape()[0] * routedExpertNum, x.GetShape()[1]};
    auto expandXPtr = std::make_shared<LogicalTensor>(function, x.GetDataType(), expandXShape);
    auto &oper = function.AddOperation(Opcode::OP_MOE_DISPATCH_LOCAL_COPY_OUT, {x.GetStorage()}, 
        {expandXPtr, syncTensor.GetStorage()});
    (void) oper;
    return expandXPtr;
}

void CreateShmem(Tensor& shmemTensor, int32_t rankSize, int32_t hcclGroupIndex, DataType dataType,
    const Shape& shape)
{
    auto &function = *Program::GetInstance().GetCurrentFunction();
    Shape shmemShape{rankSize};
    shmemShape.insert(shmemShape.end(), shape.begin(), shape.end());
    auto shmemTensorInner = std::make_shared<LogicalTensor>(function, dataType, shmemShape);
    shmemTensor = shmemTensorInner;
    Program::GetInstance().GetTensorSlotManager()->TensorWrite(shmemTensor, SlotProperty::SHMEM_TENSOR);
    auto &op = function.AddOperation(Opcode::OP_BIND_TENSOR, {}, {shmemTensorInner});
    op.SetAttribute(OpAttributeKey::bindTensor, BindTensor(hcclGroupIndex, 0,
        BytesOf(dataType) * std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int64_t>())));
}

std::tuple<int32_t, int32_t, int32_t> GetFFNTileParam(uint32_t epWorldSize)
{
    int32_t tileRankCnt = epWorldSize > FFN_TILE_SIZE ? FFN_TILE_SIZE : epWorldSize;
    int32_t tileNum = tileRankCnt == FFN_TILE_SIZE ? epWorldSize / FFN_TILE_SIZE : 1;
    int32_t tailNum = tileNum == 1 ? 0 : (epWorldSize % FFN_TILE_SIZE == 0 ? 0 : 1);
    return {tileRankCnt, tileNum, tailNum};
}

void MoeDistributedDispatch(const Tensor &x, const Tensor &expertIds, const char *group, 
    uint32_t epWorldSize, uint32_t moeExpertNum, uint32_t sharedExpertNum, uint32_t sharedExpertRankNum, Tensor &expandX,
    Tensor &expertTokenNums, Tensor &assistInfoForCombine, Tensor& recvCounts)
{
    std::string assertResult;
    ASSERT(checkValidConfig(epWorldSize, moeExpertNum, sharedExpertNum, sharedExpertRankNum, assertResult)) << assertResult;
    ASSERT(group != nullptr) << "MoeDispatch constraint violated: group name can't be nullptr.";
    ASSERT(group[0] != '\0') << "MoeDispatch constraint violated: group name is not valid.";
    ASSERT(strnlen(group, 128) < 128) << "MoeDispatch constraint violated: group name max size must be 128.";
    int32_t routedExpertNum =  moeExpertNum - sharedExpertNum;
    int32_t expertNumPerRank = routedExpertNum / epWorldSize;
    ASSERT(checkValidInput(x, 2, DataType::DT_BF16, 8, 5120, assertResult)) << assertResult; // 当前仅支持shape:8,5120
    ASSERT(checkValidInput(expertIds, 2, DataType::DT_INT32, 8, 8, assertResult)) << assertResult; // 当前仅支持shape:8,8
    ASSERT(checkValidInput(expertTokenNums, 1, DataType::DT_INT32, expertNumPerRank, 1, assertResult)) << assertResult;
    ASSERT(checkValidInput(recvCounts, 1, DataType::DT_INT32, 1, 1, assertResult)) << assertResult;

    int hcclGroupIndex = static_cast<int32_t>(CommGroupRecorder::GetInstance().Input(std::string(group)));
    SymbolicScalar thisRank = GetHcclRankId(hcclGroupIndex);
    int batchSize = x.GetShape(0);
    int hiddenSize = x.GetShape(1);
    int topK = expertIds.GetShape(1);
    int shmemDataLength = AlignUp(hiddenSize, 512) + 512;
    int32_t expandXRow = std::min(static_cast<int32_t>(batchSize) *
        static_cast<int32_t>(topK) * static_cast<int32_t>(epWorldSize), static_cast<int32_t>(batchSize) * routedExpertNum);

    ASSERT(checkValidInput(expandX, 2, DataType::DT_BF16, expandXRow, 5120, assertResult)) << assertResult; // 当前仅支持hiddenSize:5120
    ASSERT(checkValidInput(assistInfoForCombine, 2, DataType::DT_INT32, expandXRow, 3, assertResult)) << assertResult; // comBineInfo固定hiddenSize:3

    int flagRow = 1;
    int flagCol = 128;
    Tensor shmemData;
    Tensor shmemFlag;
    LOOP("CreateShmemTensor", FunctionType::DYNAMIC_LOOP, index, LoopRange(1)) {
        (void) index;
        int32_t shmemDataCol = shmemDataLength * batchSize;
        Shape shmemDataShape = {epWorldSize, expertNumPerRank, shmemDataCol};
        Shape shmemFlagShape = {expertNumPerRank, epWorldSize, flagCol};
        CreateShmem(shmemData, epWorldSize, hcclGroupIndex, x.GetDataType(), shmemDataShape);
        CreateShmem(shmemFlag, epWorldSize, hcclGroupIndex, DT_INT32, shmemFlagShape);
    }
    LOOP("L0", FunctionType::DYNAMIC_LOOP, index, LoopRange(1)) {
        (void) index;
        TileShape::Current().SetDistTile(
            {1, batchSize, 0},
            {topK, 1, 0},
            {static_cast<int32_t>(epWorldSize), 1, 0});
        Tensor syncTensor = DispatchSendToRoutingExperts(shmemData, x, expertIds, group, expertNumPerRank);
        TileShape::Current().SetDistTile(
            {flagRow, 1, 0},
            {static_cast<int32_t>(epWorldSize), 1, 0},
            {1, expertNumPerRank, 0});
        auto localShmemFlag = View(shmemFlag, {epWorldSize, expertNumPerRank, 1, flagCol}, 
            {0, 0, thisRank, 0});
        Tensor flagDummy = DispatchSetRecvFlags(localShmemFlag, expertIds, syncTensor, group, expertNumPerRank);
        auto [ffnTileCnt, ffnTileNum, ffnTailNum] = GetFFNTileParam(epWorldSize);
        TileShape::Current().SetDistTile(
            {batchSize, 1, 0},
            {hiddenSize, 1, 0},
            {ffnTileCnt, ffnTileNum, ffnTailNum});
        auto shmemFlagSched = View(shmemFlag, {1, expertNumPerRank, epWorldSize, flagCol},
            {thisRank, 0, 0, 0});
        auto recvTokenCntOut = DispatchWaitRecvFlags(group, flagDummy, shmemFlagSched, expertNumPerRank, ffnTileNum);
        auto shmemDataBatching = View(shmemData, {1, epWorldSize, expertNumPerRank, shmemDataLength},
            {thisRank, 0 ,0 ,0});
        auto [expandXPtr, recvCountsPtr] = DispatchAssembleExpandX(group, x, recvTokenCntOut, shmemDataBatching,
            localShmemFlag, expandX.GetShape(0), ffnTileNum + ffnTailNum, expertNumPerRank);
        auto assistInfoForCombinePtr = DispatchAssembleCombineInfo(group, x, recvTokenCntOut, shmemDataBatching,
            localShmemFlag, expandX.GetShape(0), ffnTileNum + ffnTailNum, expertNumPerRank);
        TileShape::Current().SetDistTileRank({expertNumPerRank / 10, 10, 0});
        auto shmemFlagValidCnt = View(shmemFlag, {1, expertNumPerRank, epWorldSize, flagCol},
            {thisRank, 0, 0, 0});
        auto expertTokenNumsPtr = DispatchBuildExpertTokenNum(recvTokenCntOut, shmemFlagValidCnt, expertNumPerRank);
        expandX = expandXPtr;
        recvCounts = recvCountsPtr;
        expertTokenNums = expertTokenNumsPtr;
        assistInfoForCombine = assistInfoForCombinePtr;
    }
}
}
}