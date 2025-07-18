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
#include "comm_barrier_manager.h"

namespace npu::tile_fwk {
namespace Distributed {
template <typename T>
struct DispatchTileArgs {
    Function &function;
    const std::vector<std::shared_ptr<LogicalTensor>> &in;
    const std::vector<std::shared_ptr<LogicalTensor>> &out;
    T tilingInfo{};
    const CommGroupInfo groupInfo;
    const DistTensorTilingInfo tensorTileInfo;
    const std::string tilingSymbol;
};

template <typename T>
void ExpandTensorTiles(const std::function<void(T &)> &dealFunc, T &args)
{
    auto &tilingInfo = args.tilingInfo;
    tilingInfo.groupIndex = args.groupInfo.groupIndex;
    auto& tileArray = args.tensorTileInfo[0];
    tilingInfo.totalTileNum = GetTotalTileNum(tileArray);
    for (tilingInfo.tileIndex = 0; tilingInfo.tileIndex < tilingInfo.totalTileNum; tilingInfo.tileIndex++) {
        tilingInfo.shape = (tilingInfo.tileIndex < tileArray[DIST_HEAD_COUNT]) ?
            tileArray[DIST_HEAD_SHAPE] : tileArray[DIST_TAIL_SHAPE];
        dealFunc(args);
        tilingInfo.offset += tilingInfo.shape;
    }
}

void TileProcess(const std::function<void(DispatchTileArgs<TilingInfo> &)> &dealFunc,
    DispatchTileArgs<TilingInfo> &args)
{
    const auto &rankTileInfo = args.groupInfo.rank.value();
    const int32_t tileRankShape = rankTileInfo[DIST_HEAD_SHAPE];
    const int32_t tileRankCnt = rankTileInfo[DIST_HEAD_COUNT];
    const int32_t tailRankShape = rankTileInfo[DIST_TAIL_SHAPE];

    auto &tilingInfo = args.tilingInfo;
    tilingInfo.tileIndex = 0; // tileIndex 初始化
    tilingInfo.rankOffset = 0; // rankOffset 初始化
    tilingInfo.groupIndex = args.groupInfo.groupIndex;
    tilingInfo.totalTileNum = GetTotalTileNum(rankTileInfo); // tileOp 个数
    tilingInfo.shareRankCnt = Distributed::SHARED_EXPERT_NUM; // 共享专家卡数，moe 卡数可以计算出来

    while (tilingInfo.tileIndex < tilingInfo.totalTileNum) {
        tilingInfo.rankShape = (tilingInfo.tileIndex < tileRankCnt) ? tileRankShape : tailRankShape;
        dealFunc(args);
        tilingInfo.rankOffset += tilingInfo.rankShape;
        tilingInfo.tileIndex++;
    }
}

void FFNSchedOpCallback(DispatchTileArgs<TilingInfo> &args)
{
    std::shared_ptr<LogicalTensor> syncTensor = args.in[DIST_INDEX_ZERO];
    std::shared_ptr<LogicalTensor> tilingTensor = args.in[DIST_INDEX_ONE];
    std::shared_ptr<LogicalTensor> flagTensor = args.in[DIST_INDEX_TWO];
    std::shared_ptr<LogicalTensor> recvTokenCntOut = args.out[DIST_INDEX_ZERO];

    auto &tilingInfo = args.tilingInfo;
    tilingInfo.rowShape = args.tensorTileInfo[0][0]; // 没切 x，所以 rowTile colTile 就是 x 的 shape
    tilingInfo.colShape = args.tensorTileInfo[1][0];

    OpArgs<TilingInfo> opArgs = {"TILE_FFN_SCHED", {syncTensor, flagTensor}, {recvTokenCntOut},
        tilingTensor, args.tilingSymbol, std::make_optional(args.tilingInfo), std::nullopt};
    auto &op = AddOperation(args.function, opArgs);
    if (!IsRoutingExpert(args.groupInfo.rankId.value())) { // share rank 传入模板参数 true
        op.SetAttr("extraTemplateParam", std::string("true"));
    }
}

void FFNBatchingOpCallback(DispatchTileArgs<TilingInfo> &args)
{
    std::shared_ptr<LogicalTensor> recvTokenCntOut = args.in[DIST_INDEX_ZERO];
    std::shared_ptr<LogicalTensor> tilingTensor = args.in[DIST_INDEX_ONE];
    std::shared_ptr<LogicalTensor> flagTensor = args.in[DIST_INDEX_TWO];
    std::shared_ptr<LogicalTensor> expandX = args.out[DIST_INDEX_ZERO];
    std::shared_ptr<LogicalTensor> validCnt = args.out[DIST_INDEX_ONE];

    auto &tilingInfo = args.tilingInfo;
    tilingInfo.rowShape = args.tensorTileInfo[0][0];
    tilingInfo.colShape = args.tensorTileInfo[1][0];

    OpArgs<TilingInfo> opArgs = {"TILE_FFN_BATCHING", {recvTokenCntOut, flagTensor}, {expandX, validCnt},
        tilingTensor, args.tilingSymbol, std::make_optional(args.tilingInfo), std::nullopt};
    auto &op = AddOperation(args.function, opArgs);
    if (!IsRoutingExpert(args.groupInfo.rankId.value())) { // share rank 传入模板参数 true
        op.SetAttr("extraTemplateParam", std::string("true"));
    }
}

void TiledDispatchFFNBatching(Function &function, const TileShape &tileShape,
    const std::vector<std::shared_ptr<LogicalTensor>> &iOperand,
    const std::vector<std::shared_ptr<LogicalTensor>> &oOperand, const Operation &op)
{
    CommGroupInfo groupInfo;
    op.GetAttr(OpAttributeKey::commGroupInfo, groupInfo);
    std::string tilingSymbol;
    op.GetAttr("tiling_tensor_symbol", tilingSymbol);
    TilingInfo tilingInfo;
    constexpr size_t tensorTileDim = 2;
    DistTensorTilingInfo tensorTileInfo(tileShape, tensorTileDim);
    DispatchTileArgs<TilingInfo> args = {function, iOperand, oOperand, tilingInfo, groupInfo, tensorTileInfo,
        tilingSymbol};
    ALOG_INFO_F("Distributed opinfo: row=[%d %d %d], col=[%d %d %d]", args.tensorTileInfo[0][DIST_HEAD_SHAPE],
        args.tensorTileInfo[0][DIST_HEAD_COUNT], args.tensorTileInfo[0][DIST_TAIL_SHAPE],
        args.tensorTileInfo[1][DIST_HEAD_SHAPE], args.tensorTileInfo[1][DIST_HEAD_COUNT],
        args.tensorTileInfo[1][DIST_TAIL_SHAPE]);
    TileProcess(FFNBatchingOpCallback, args);
}
 
 
void TiledDispatchFFNSched(Function &function, const TileShape &tileShape,
    const std::vector<std::shared_ptr<LogicalTensor>> &iOperand,
    const std::vector<std::shared_ptr<LogicalTensor>> &oOperand, const Operation &op)
{
    CommGroupInfo groupInfo;
    op.GetAttr(OpAttributeKey::commGroupInfo, groupInfo);
    std::string tilingSymbol;
    op.GetAttr("tiling_tensor_symbol", tilingSymbol);
    TilingInfo tilingInfo;
    constexpr size_t tensorTileDim = 2;
    DistTensorTilingInfo tensorTileInfo(tileShape, tensorTileDim);
    DispatchTileArgs<TilingInfo> args = {function, iOperand, oOperand, tilingInfo, groupInfo, tensorTileInfo,
        tilingSymbol};
    ALOG_INFO_F("Distributed opinfo: row=[%d %d %d], col=[%d %d %d]", args.tensorTileInfo[0][DIST_HEAD_SHAPE],
        args.tensorTileInfo[0][DIST_HEAD_COUNT], args.tensorTileInfo[0][DIST_TAIL_SHAPE],
        args.tensorTileInfo[1][DIST_HEAD_SHAPE], args.tensorTileInfo[1][DIST_HEAD_COUNT],
        args.tensorTileInfo[1][DIST_TAIL_SHAPE]);
    TileProcess(FFNSchedOpCallback, args);
}

void TensorGraphAddOp(const std::string &opName, const std::vector<std::shared_ptr<LogicalTensor>> &iOperands,
    const std::vector<std::shared_ptr<LogicalTensor>> &oOperands, const char *group)
{
    auto &function = *Program::GetInstance().GetCurrentFunction();
    auto &oper = function.AddOperation(opName, iOperands, oOperands);
    std::shared_ptr<LogicalTensor> tilingTensor = iOperands[1];

    int32_t groupIndex = static_cast<int32_t>(
        Program::GetInstance().GetCommGroupRecorder().Input(std::string(group)));
    CommGroupInfo groupInfo;
    const TileShape &tileShape = Program::GetInstance().GetTileShape();
    CheckAndGetGroupInfo(groupIndex, tileShape, groupInfo);
    oper.SetAttr(OpAttributeKey::commGroupInfo, groupInfo);
    oper.SetAttr("tiling_tensor_symbol", tilingTensor->Symbol());
}

void DispatchFFNBatching(std::vector<std::shared_ptr<LogicalTensor>> &iOperands,
    std::vector<std::shared_ptr<LogicalTensor>> &oOperands, const char *group, const Tensor &tokenTensor)
{
    int tempSize1 = (Distributed::AIV_NUM * 32 + 255) / 256 * 256 +
                     256 +
                    (Distributed::AIV_NUM * 4 + 31) / 32 * 32;
    int tempSize2 = tokenTensor->shape[1] * BytesOf(tokenTensor.GetDataType());
    int tempBufSize = (tempSize1 < tempSize2) ? tempSize2 : tempSize1;

    std::vector<int> flagShape {tempBufSize / 8, 8}; // 肯定能除尽
    // std::vector<int> flagShape {Distributed::TOTAL_EXPERT_NUM, 128}; // 需要一个计算公式
    auto &function = *Program::GetInstance().GetCurrentFunction();
    auto flagTensor = std::make_shared<LogicalTensor>(function, DataType::DT_INT32, flagShape);

    iOperands.push_back(flagTensor);
    TensorGraphAddOp("FFN_BATCHING", iOperands, oOperands, group);
}

void DispatchFFNSched(std::vector<std::shared_ptr<LogicalTensor>> &iOperands,
    std::vector<std::shared_ptr<LogicalTensor>> &oOperands, const char *group)
{
    uint32_t moeOpProcessRankSize = Distributed::TOTAL_EXPERT_NUM / Distributed::AIV_NUM;
    uint32_t shareOpProcessRankSize = Distributed::ROUTING_EXPERT_NUM / Distributed::SHARED_EXPERT_NUM;
    uint32_t maxProcessRankSize = (moeOpProcessRankSize < shareOpProcessRankSize) ?
        shareOpProcessRankSize : moeOpProcessRankSize;
    int tempBufSize = maxProcessRankSize * 32 +
                      32 +
                      (maxProcessRankSize * 4 + 31) / 32 * 32;

    std::vector<int> flagShape {tempBufSize / 8, 8}; // 肯定能除尽
    // std::vector<int> flagShape {Distributed::TOTAL_EXPERT_NUM, 128}; // 需要一个计算公式
    auto &function = *Program::GetInstance().GetCurrentFunction();
    auto flagTensor = std::make_shared<LogicalTensor>(function, DataType::DT_INT32, flagShape);

    iOperands.push_back(flagTensor);
    TensorGraphAddOp("FFN_SCHED", iOperands, oOperands, group);
}

std::vector<int32_t> GetCommBufferSize(const std::shared_ptr<LogicalTensor> &tokenTensor)
{
    const int32_t hOutSize = tokenTensor->shape[1] * BytesOf(tokenTensor->Datatype());
    constexpr int32_t scaleParamPad = 128;
    const int32_t hCommuSize = hOutSize + scaleParamPad;
    return {1, static_cast<int32_t>(hCommuSize / BytesOf(tokenTensor->Datatype()))};
}

void DealSendToRoutingExpertTile(DispatchTileArgs<DispatchTilingInfo> &args)
{
    std::shared_ptr<LogicalTensor> tokenTensor = args.in[DIST_INDEX_ZERO];
    std::shared_ptr<LogicalTensor> tokenExpertTable = args.in[DIST_INDEX_ONE];
    std::shared_ptr<LogicalTensor> tilingTensor = args.in[DIST_INDEX_TWO];
    std::shared_ptr<LogicalTensor> syncTensor = args.out[DIST_INDEX_ZERO];
    auto &tilingInfo = args.tilingInfo;
    auto tokenBuffer = std::make_shared<LogicalTensor>(args.function, tokenTensor->Datatype(),
        GetCommBufferSize(tokenTensor));
    auto expertBufferUb = std::make_shared<LogicalTensor>(args.function, tokenExpertTable->Datatype(),
        std::vector<int32_t>{1, tokenExpertTable->shape[0] * tokenExpertTable->shape[1]});
    auto expertBuffer = std::make_shared<LogicalTensor>(args.function, tokenExpertTable->Datatype(),
        std::vector<int32_t>{1, tokenExpertTable->shape[0] * tokenExpertTable->shape[1] * 2});
    OpArgs<DispatchTilingInfo> opArgs = {"TILE_SEND_TO_ROUTING_EXPERT",
        {tokenTensor, tokenExpertTable, tokenBuffer, expertBufferUb, expertBuffer}, {syncTensor},
        tilingTensor, args.tilingSymbol, std::make_optional(tilingInfo), std::nullopt};
    auto& op = AddOperation(args.function, opArgs);
    std::string extraParam = std::to_string(tokenTensor->shape[0]) + ", " +
        std::to_string(tokenTensor->shape[1]) + ", " + std::to_string(tokenExpertTable->shape[1]);
    op.SetAttr("extraTemplateParam", extraParam);
}

void TiledSendToRoutingExpert(Function &function, const TileShape &tileShape,
    const std::vector<std::shared_ptr<LogicalTensor>> &iOperand,
    const std::vector<std::shared_ptr<LogicalTensor>> &oOperand, const Operation &op)
{
    (void)tileShape;
    CommGroupInfo groupInfo;
    op.GetAttr(OpAttributeKey::commGroupInfo, groupInfo);
    std::string tilingSymbol;
    op.GetAttr("tiling_tensor_symbol", tilingSymbol);
    DistTensorTilingInfo tensorTileInfo;
    op.GetAttr("DistTensorTilingInfo", tensorTileInfo);

    DispatchTileArgs<DispatchTilingInfo> args = {function, iOperand, oOperand, {}, groupInfo, tensorTileInfo,
        tilingSymbol};
    ExpandTensorTiles<decltype(args)>(DealSendToRoutingExpertTile, args);
}

void DealSendToSharedExpertTile(DispatchTileArgs<DispatchTilingInfo> &args)
{
    std::shared_ptr<LogicalTensor> tokenTensor = args.in[0];
    std::shared_ptr<LogicalTensor> tilingTensor = args.in[1];
    std::shared_ptr<LogicalTensor> syncTensor = args.out[0];
    auto &tilingInfo = args.tilingInfo;
    auto tokenBuffer = std::make_shared<LogicalTensor>(args.function, tokenTensor->Datatype(),
        GetCommBufferSize(tokenTensor));
    OpArgs<DispatchTilingInfo> opArgs = {"TILE_SEND_TO_SHARED_EXPERT", {tokenTensor, tokenBuffer}, {syncTensor},
        tilingTensor, args.tilingSymbol, std::make_optional(tilingInfo), std::nullopt};
    auto &op = AddOperation(args.function, opArgs);
    std::string extraParam = std::to_string(tokenTensor->shape[0]) + ", " +
        std::to_string(tokenTensor->shape[1]);
    op.SetAttr("extraTemplateParam", extraParam);
}

void DealCopyToLocalExpertTile(DispatchTileArgs<DispatchTilingInfo> &args)
{
    std::shared_ptr<LogicalTensor> tokenTensor = args.in[DIST_INDEX_ZERO];
    std::shared_ptr<LogicalTensor> tilingTensor = args.in[DIST_INDEX_ONE];
    std::shared_ptr<LogicalTensor> expandX = args.out[DIST_INDEX_ZERO];
    auto &tilingInfo = args.tilingInfo;
    auto tokenBuffer = std::make_shared<LogicalTensor>(args.function, tokenTensor->Datatype(),
        GetCommBufferSize(tokenTensor));
    OpArgs<DispatchTilingInfo> opArgs = {"TILE_COPY_TO_LOCAL_EXPERT", {tokenTensor, tokenBuffer}, {expandX},
        tilingTensor, args.tilingSymbol, std::make_optional(tilingInfo), std::nullopt};
    auto &op = AddOperation(args.function, opArgs);
    std::string extraParam = std::to_string(tokenTensor->shape[0]) + ", " +
        std::to_string(tokenTensor->shape[1]);
    op.SetAttr("extraTemplateParam", extraParam);
}

void TiledSendToSharedExpert(Function &function, const TileShape &tileShape,
    const std::vector<std::shared_ptr<LogicalTensor>> &iOperand,
    const std::vector<std::shared_ptr<LogicalTensor>> &oOperand, const Operation &op)
{
    (void)tileShape;
    CommGroupInfo groupInfo;
    op.GetAttr(OpAttributeKey::commGroupInfo, groupInfo);
    std::string tilingSymbol;
    op.GetAttr("tiling_tensor_symbol", tilingSymbol);
    DistTensorTilingInfo tensorTileInfo;
    op.GetAttr("DistTensorTilingInfo", tensorTileInfo);

    DispatchTileArgs<DispatchTilingInfo> args = {function, iOperand, oOperand, {}, groupInfo, tensorTileInfo,
        tilingSymbol};
    ExpandTensorTiles<decltype(args)>(DealSendToSharedExpertTile, args);
}

void TiledCopyToLocalExpert(Function &function, const TileShape &tileShape,
    const std::vector<std::shared_ptr<LogicalTensor>> &iOperand,
    const std::vector<std::shared_ptr<LogicalTensor>> &oOperand, const Operation &op)
{
    (void)tileShape;
    std::string tilingSymbol;
    op.GetAttr("tiling_tensor_symbol", tilingSymbol);
    DistTensorTilingInfo tensorTileInfo;
    op.GetAttr("DistTensorTilingInfo", tensorTileInfo);

    DispatchTileArgs<DispatchTilingInfo> args = {function, iOperand, oOperand, {}, {}, tensorTileInfo, tilingSymbol};
    ExpandTensorTiles<decltype(args)>(DealCopyToLocalExpertTile, args);
}

void DealDispatchSetFlagTile(DispatchTileArgs<DispatchTilingInfo> &args)
{
    std::shared_ptr<LogicalTensor> syncTensor = args.in[DIST_INDEX_ZERO];
    std::shared_ptr<LogicalTensor> tokenExpertTable = args.in[DIST_INDEX_ONE];
    std::shared_ptr<LogicalTensor> tilingTensor = args.in[DIST_INDEX_TWO];
    std::shared_ptr<LogicalTensor> dummy = args.out[DIST_INDEX_ZERO];
    auto statusTensor = std::make_shared<LogicalTensor>(args.function, DataType::DT_INT32,
        std::vector<int32_t>{1, TOTAL_EXPERT_NUM * 8});
    auto expertBufferUb = std::make_shared<LogicalTensor>(args.function, tokenExpertTable->Datatype(),
        std::vector<int32_t>{1, tokenExpertTable->shape[0] * tokenExpertTable->shape[1]});
    auto expertBuffer = std::make_shared<LogicalTensor>(args.function, tokenExpertTable->Datatype(),
        std::vector<int32_t>{1, tokenExpertTable->shape[0] * tokenExpertTable->shape[1] * 2});
    OpArgs<DispatchTilingInfo> opArgs = {"TILE_DISPATCH_SET_FLAG",
        {syncTensor, tokenExpertTable, statusTensor, expertBufferUb, expertBuffer}, {dummy},
        tilingTensor, args.tilingSymbol, std::make_optional(args.tilingInfo), std::nullopt};
    auto& op = AddOperation(args.function, opArgs);
    std::string extraParam = std::to_string(tokenExpertTable->shape[0]) + ", " +
        std::to_string(tokenExpertTable->shape[1]);
    op.SetAttr("extraTemplateParam", extraParam);
}

void TiledDispatchSetFlag(Function &function, const TileShape &tileShape,
    const std::vector<std::shared_ptr<LogicalTensor>> &iOperand,
    const std::vector<std::shared_ptr<LogicalTensor>> &oOperand, const Operation &op)
{
    (void)tileShape;
    CommGroupInfo groupInfo;
    op.GetAttr(OpAttributeKey::commGroupInfo, groupInfo);
    std::string tilingSymbol;
    op.GetAttr("tiling_tensor_symbol", tilingSymbol);
    DistTensorTilingInfo tensorTileInfo;
    op.GetAttr("DistTensorTilingInfo", tensorTileInfo);

    DispatchTileArgs<DispatchTilingInfo> args = {function, iOperand, oOperand, {}, groupInfo, tensorTileInfo,
        tilingSymbol};
    ExpandTensorTiles<decltype(args)>(DealDispatchSetFlagTile, args);
}

void SendToRoutingExpert(const Tensor &tokenTensor, const Tensor &tokenExpertTable, const Tensor &tilingTensor,
    const Tensor &syncTensor, const char *group)
{
    auto &function = *Program::GetInstance().GetCurrentFunction();
    auto &oper = function.AddOperation("SEND_TO_ROUTING_EXPERT", {tokenTensor.GetStorage(),
        tokenExpertTable.GetStorage(), tilingTensor.GetStorage()}, {syncTensor.GetStorage()});
    oper.SetAttr("tiling_tensor_symbol", tilingTensor.GetStorage()->Symbol());
    const TileShape &tileShape = Program::GetInstance().GetTileShape();
    CommGroupInfo groupInfo(group, tileShape);
    DistTensorTilingInfo tileInfo(tileShape, 1);
    ASSERT(tileInfo.Check({tokenExpertTable.GetShape()[0] * tokenExpertTable.GetShape()[1]}));
    oper.SetAttr(OpAttributeKey::commGroupInfo, groupInfo);
    oper.SetAttr("DistTensorTilingInfo", tileInfo);
}

void SendToSharedExpert(const Tensor &tokenTensor, const Tensor &tilingTensor, const Tensor &syncTensor,
    const char *group)
{
    auto &function = *Program::GetInstance().GetCurrentFunction();
    auto &oper = function.AddOperation("SEND_TO_SHARED_EXPERT", {tokenTensor.GetStorage(), tilingTensor.GetStorage()},
        {syncTensor.GetStorage()});
    oper.SetAttr("tiling_tensor_symbol", tilingTensor.GetStorage()->Symbol());
    const TileShape &tileShape = Program::GetInstance().GetTileShape();
    CommGroupInfo groupInfo(group, tileShape);
    DistTensorTilingInfo tileInfo(tileShape, 1);
    ASSERT(tileInfo.Check({tokenTensor.GetShape()[0]}));
    oper.SetAttr(OpAttributeKey::commGroupInfo, groupInfo);
    oper.SetAttr("DistTensorTilingInfo", tileInfo);
}

Tensor DispatchSetFlag(const Tensor &tokenExpertTable, const Tensor &syncTensor, const Tensor &tilingTensor,
    const char *group)
{
    Tensor dummyTensor(DataType::DT_INT32, {1, 1}, "dummyTensor");
    auto &function = *Program::GetInstance().GetCurrentFunction();
    auto &oper = function.AddOperation("DISPATCH_SET_FLAG", {syncTensor.GetStorage(), tokenExpertTable.GetStorage(),
        tilingTensor.GetStorage()}, {dummyTensor.GetStorage()});
    oper.SetAttr("tiling_tensor_symbol", tilingTensor.GetStorage()->Symbol());
    const TileShape &tileShape = Program::GetInstance().GetTileShape();
    CommGroupInfo groupInfo(group, tileShape);
    DistTensorTilingInfo tileInfo(tileShape, 1);
    ASSERT(tileInfo.Check({Distributed::TOTAL_EXPERT_NUM}) && groupInfo.CheckAndUpdate(Distributed::TOTAL_EXPERT_NUM));
    oper.SetAttr(OpAttributeKey::commGroupInfo, groupInfo);
    oper.SetAttr("DistTensorTilingInfo", tileInfo);
    return dummyTensor;
}

void CopyToLocalExpert(const Tensor &tokenTensor, const Tensor &tilingTensor, const Tensor &expandX)
{
    auto &function = *Program::GetInstance().GetCurrentFunction();
    auto &oper = function.AddOperation("COPY_TO_LOCAL_EXPERT", {tokenTensor.GetStorage(), tilingTensor.GetStorage()},
        {expandX.GetStorage()});
    oper.SetAttr("tiling_tensor_symbol", tilingTensor.GetStorage()->Symbol());
    const TileShape &tileShape = Program::GetInstance().GetTileShape();
    DistTensorTilingInfo tileInfo(tileShape, 1);
    ASSERT(tileInfo.Check({tokenTensor.GetShape()[0]}));
    oper.SetAttr("DistTensorTilingInfo", tileInfo);
}

Tensor MoeDispatch(const Tensor &tokenTensor, const Tensor &tokenExpertTable, Tensor &validCnt, const char *group)
{
    OperatorChecker checker;

    auto &function = *Program::GetInstance().GetCurrentFunction();
    std::vector<int32_t> tilingShape = {1, 8192};
    const std::string tilingSymbol = function.GetDistTilingManager()->CreateTilingStorage("dispatch", tilingShape[1]);
    Tensor tilingTensor(DataType::DT_INT32, tilingShape, tilingSymbol);

    const int32_t tableSize = tokenExpertTable.GetShape()[0] * tokenExpertTable.GetShape()[1];
    Tensor expandX(tokenTensor.GetDataType(), {(Distributed::TOTAL_EXPERT_NUM) * tokenTensor.GetShape()[0],
        tokenTensor.GetShape()[1]}, "expandX");
    Tensor syncTensor(DataType::DT_INT32, {1, 1}, "syncTensor");
    Program::GetInstance().GetTileShape().SetDistTileShapes({1, tableSize, 0});
    Distributed::SendToRoutingExpert(tokenTensor, tokenExpertTable, tilingTensor, syncTensor, group);

    // 发送的专家号是固定的，通过本卡的rankId确定；这里对token的M轴进行切分，为了方便在Rank上操作
    Program::GetInstance().GetTileShape().SetDistTileShapes({1, tokenTensor.GetShape()[0], 0});
    if (IsRoutingExpert(Program::GetInstance().GetTileShape().GetDistRankId())) {
        Distributed::SendToSharedExpert(tokenTensor, tilingTensor, syncTensor, group);
    } else {
        Distributed::CopyToLocalExpert(tokenTensor, tilingTensor, expandX);
    }

    Program::GetInstance().GetTileShape().SetDistTileShapes({1, Distributed::TOTAL_EXPERT_NUM, 0});
    auto dummy = Distributed::DispatchSetFlag(tokenExpertTable, syncTensor, tilingTensor,  group);

    // 48 * 48 * 512B
    Tensor recvTokenCntOut(DataType::DT_INT32, {Distributed::AIV_NUM * Distributed::AIV_NUM, 128},
        "recvTokenCntOut");
 
    Program::GetInstance().GetTileShape().SetDistTileShapes(
        {tokenTensor->shape[0], 1, 0},
        {tokenTensor->shape[1], 1, 0}, // 不切 x
        {Distributed::TOTAL_EXPERT_NUM / Distributed::AIV_NUM, Distributed::AIV_NUM, 0}); // 暂不处理不整除的场景
 
    std::vector<std::shared_ptr<LogicalTensor>> schedInOperands {dummy.GetStorage(), tilingTensor.GetStorage()};
    std::vector<std::shared_ptr<LogicalTensor>> schedOutOperands {recvTokenCntOut.GetStorage()};
    Distributed::DispatchFFNSched(schedInOperands, schedOutOperands, group);
 
    std::vector<std::shared_ptr<LogicalTensor>> iOperands {recvTokenCntOut.GetStorage(), tilingTensor.GetStorage()};
    std::vector<std::shared_ptr<LogicalTensor>> oOperands {expandX.GetStorage(), validCnt.GetStorage()};
    Distributed::DispatchFFNBatching(iOperands, oOperands, group, tokenTensor);
 
    return expandX;
}
}
}