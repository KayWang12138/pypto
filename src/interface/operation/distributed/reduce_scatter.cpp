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
 * \file reduce_scatter.cpp
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
void RedeceByRankView(
    const std::shared_ptr<LogicalTensor> &inTile, const std::shared_ptr<LogicalTensor> &outTile,
    const std::shared_ptr<LogicalTensor> &dummyOut, TileArgs &args)
{
    // 此rank view仅包含localRankId时，不需要做remote reduce，否则需要remote reduce
    const bool onlyLocalInView =
        (args.groupInfo.rankId.value() == args.tilingInfo.rankOffset) && (args.tilingInfo.rankShape == 1);
    if (onlyLocalInView) {
        return;
    }

    const bool aicpuWaitFlagEnable = ConfigManager::Instance().GetDistConfig(KEY_AICPU_WAIT_FLAG_ENABLE, true);
    std::vector<int> flagShape = {1, FLAG_TENSOR_SIZE}; // 256 byte
    // aicpu wait flag方案中，flagTensor作为控制边
    auto flagTensor = std::make_shared<LogicalTensor>(args.function, DataType::DT_INT32, flagShape);
    if (aicpuWaitFlagEnable) {
        std::vector<int> opAttr = {args.tilingInfo.tileIndex, args.tilingInfo.groupIndex, args.tilingInfo.rankShape,
            args.tilingInfo.rankOffset};
        OpArgs<TilingInfo> opArgs = {"COMM_WAIT_FLAG", {inTile}, {flagTensor}, args.tilingTensor, args.tilingSymbol,
            std::nullopt, std::make_optional(opAttr)};
        (void)AddOperation(args.function, opArgs);
    }

    // remoteTmp is used as data transfer station, dummyOut is control edge between local reduce and remote reduce
    auto remoteTmp = std::make_shared<LogicalTensor>(args.function, inTile->Datatype(), inTile->shape);
    OpArgs<TilingInfo> opArgs = {"REMOTE_REDUCE", {dummyOut, flagTensor}, {outTile, remoteTmp}, args.tilingTensor,
        args.tilingSymbol, std::make_optional(args.tilingInfo), std::nullopt};
    auto &op = AddOperation(args.function, opArgs);
    if (!aicpuWaitFlagEnable) {
        op.SetAttr("extraTemplateParam", std::string("true"));
    }
    // output of local reduce is the same tensor with input of remote reduce (dummyOut)
    op.SetAttr(OpAttributeKey::sameInOut, Opcode::OP_LOCAL_COPY_OUT);
}

void LocalReduce(
    const std::shared_ptr<LogicalTensor> &inTile, const std::shared_ptr<LogicalTensor> &outTile, TileArgs &args)
{
    const auto &tileRank = args.groupInfo.rank.value();
    // LocalReduce kernel 侧现在不会使用 rank 相关信息，此处暂时无用
    if (args.groupInfo.rankId.value() < tileRank[0] * tileRank[1]) {
        args.tilingInfo.rankShape = tileRank[0];
        args.tilingInfo.rankOffset = args.groupInfo.rankId.value() / tileRank[0] * tileRank[0];
    } else {
        args.tilingInfo.rankShape = tileRank[DIST_TAIL_SHAPE];
        args.tilingInfo.rankOffset = tileRank[DIST_HEAD_COUNT] * tileRank[DIST_HEAD_SHAPE];
    }

    OpArgs<TilingInfo> opArgs = {"LOCAL_COPY_OUT", {inTile}, {outTile}, args.tilingTensor, args.tilingSymbol,
        std::make_optional(args.tilingInfo), std::nullopt};
    auto &op = AddOperation(args.function, opArgs);
    op.SetAttr("extraTemplateParam", std::string("true"));
}

void DealTileBelongLocal(TileArgs &args)
{
    const auto &tileRank = args.groupInfo.rank.value();
    std::vector<int> shape = {args.tilingInfo.rowShape, args.tilingInfo.colShape};
    std::vector<int> offset = {args.tilingInfo.rowOffset, args.tilingInfo.colOffset};
    auto inTile = args.in->View(args.function, shape, offset);
    auto outTile = args.out->View(args.function, shape, offset);

    auto dummyOut = std::make_shared<LogicalTensor>(args.function, outTile->Datatype(), outTile->shape);

    LocalReduce(inTile, dummyOut, args);

    for (int rankIndex = 0; rankIndex < tileRank[1]; rankIndex++) {
        args.tilingInfo.rankShape = tileRank[0];
        args.tilingInfo.rankOffset = rankIndex * tileRank[0];
        RedeceByRankView(inTile, outTile, dummyOut, args);
    }

    if (tileRank[DIST_TAIL_SHAPE] != 0)
    {
        args.tilingInfo.rankShape = tileRank[DIST_TAIL_SHAPE];
        args.tilingInfo.rankOffset = tileRank[DIST_HEAD_COUNT] * tileRank[DIST_HEAD_SHAPE];
        RedeceByRankView(inTile, outTile, dummyOut, args);
    }
}

void DealTileBelongRemote(TileArgs &args)
{
    auto inTile = args.in->View(args.function, {args.tilingInfo.rowShape, args.tilingInfo.colShape},
        {args.tilingInfo.rowOffset, args.tilingInfo.colOffset});
    std::vector<int> flagShape = {1, FLAG_TENSOR_SIZE}; // 256 byte
    auto flagTensor = std::make_shared<LogicalTensor>(args.function, DataType::DT_INT32, flagShape);
    OpArgs<TilingInfo> opArgs = {"WRITE_REMOTE", {inTile}, {flagTensor}, args.tilingTensor, args.tilingSymbol,
        std::make_optional(args.tilingInfo), std::nullopt};
    (void)AddOperation(args.function, opArgs);
}

inline std::vector<int> GetRsOutShape(const Tensor &in, int rankSize)
{
    return {in->shape[0] / rankSize, in->shape[1]};
}

inline std::vector<int> GetRsOutShape(const std::vector<Tensor> &in, int rankSize)
{
    (void)rankSize;
    return in[0]->GetShape();
}

inline Tensor GetInTensorView(const std::vector<Tensor> &in, const std::vector<int>& outShape, int rankIndex)
{
    (void)outShape;
    return in[rankIndex];
}

inline Tensor GetInTensorView(const Tensor &in, const std::vector<int>& outShape, int rankIndex)
{
    std::vector<int> offset = {rankIndex * outShape[0], 0};
    return View(in, outShape, offset);
}

inline auto GetDataType(const Tensor &in)
{
    return in.GetDataType();
}

inline auto GetDataType(const std::vector<Tensor> &in)
{
    return in[0].GetDataType();
}

template <typename T>
Tensor TensorReduceScatter(const T &in, const Tensor &tilingTensor, const CommGroupInfo &groupInfo,
    DistReduceType reduceType)
{
    (void)reduceType;
    auto outShape = GetRsOutShape(in, groupInfo.rankSize.value());
    Tensor out(GetDataType(in), outShape, "out", NodeType::OUTCAST);
    auto &function = *Program::GetInstance().GetCurrentFunction();
    TilingInfo tilingInfo;
    tilingInfo.groupIndex = groupInfo.groupIndex;
    tilingInfo.rowPerRank = outShape[0];
    tilingInfo.colPerRank = outShape[1];
    for (int rankIndex = 0; rankIndex < groupInfo.rankSize.value(); rankIndex++) {
        auto inTile = GetInTensorView(in, outShape, rankIndex);
        if (rankIndex != groupInfo.rankId.value()) {
            tilingInfo.rankShape = 1;
            tilingInfo.rankOffset = rankIndex;
            auto &oper = function.AddOperation("DIST_SCATTER", {inTile.GetStorage(), tilingTensor.GetStorage()}, {});
            oper.SetAttr(OpAttributeKey::commGroupInfo, groupInfo);
            oper.SetAttr(OpAttributeKey::distTilingInfo, tilingInfo);
            oper.SetAttr("tiling_tensor_symbol", tilingTensor.GetStorage()->Symbol());
        } else {
            auto &oper = function.AddOperation("DIST_REDUCE", {inTile.GetStorage(), tilingTensor.GetStorage()},
                {out.GetStorage()});
            oper.SetAttr(OpAttributeKey::commGroupInfo, groupInfo);
            oper.SetAttr(OpAttributeKey::distTilingInfo, tilingInfo);
            oper.SetAttr("tiling_tensor_symbol", tilingTensor.GetStorage()->Symbol());
        }
    }
    return out;
}

template <typename T>
Tensor ReduceScatterImpl(const T &in, const char *group, DistReduceType reduceType)
{
    static_assert((std::is_same_v<T, const npu::tile_fwk::Tensor &>) ||
        (std::is_same_v<T, const std::vector<npu::tile_fwk::Tensor>&>), "T must be Tensor or std::vector<Tensor>");
    int groupIndex = static_cast<int>(Program::GetInstance().GetCommGroupRecorder().Input(std::string(group)));
    CommGroupInfo groupInfo;
    const TileShape &tileShape = Program::GetInstance().GetTileShape();
    CheckAndGetGroupInfo(groupIndex, tileShape, groupInfo);

    auto outShape = GetRsOutShape(in, groupInfo.rankSize.value());
    int rowPerRank = outShape[0];
    int colPerRank = outShape[1];
    TensorTileInfo tileInfo;
    CheckAndGetTileInfo(rowPerRank, colPerRank, tileShape, tileInfo);
    auto &function = *Program::GetInstance().GetCurrentFunction();
    int tilingTensorSize = GetTilingTensorSize(tileInfo, groupInfo);
    std::vector<int> tilingShape = {1, tilingTensorSize};
    const std::string tilingSymbol = function.GetDistTilingManager()->CreateTilingStorage("reducescatter",
        tilingShape[1]);
    Tensor tilingTensor(DataType::DT_INT32, tilingShape, tilingSymbol);
    return TensorReduceScatter<decltype(in)>(in, tilingTensor, groupInfo, reduceType);
}

Tensor ReduceScatter(const std::vector<Tensor> &in, const char *group, DistReduceType reduceType)
{
    return ReduceScatterImpl<decltype(in)>(in, group, reduceType);
}

Tensor ReduceScatter(const Tensor &in, const char *group, DistReduceType reduceType)
{
    return ReduceScatterImpl<decltype(in)>(in, group, reduceType);
}

void TiledDistReduce(Function &function, const TileShape &tileShape,
    const std::vector<std::shared_ptr<LogicalTensor>> &iOperand,
    const std::vector<std::shared_ptr<LogicalTensor>> &oOperand, const Operation &op)
{
    if ((iOperand.size() != 2UL) || (oOperand.size() != 1UL)) {
        ALOG_ERROR_F("TiledDistReduce iOperand size=%lu, oOperand size=%lu", iOperand.size(), oOperand.size());
        return;
    }
    std::shared_ptr<LogicalTensor> in = iOperand[0];
    std::shared_ptr<LogicalTensor> tilingTensor = iOperand[1];
    std::shared_ptr<LogicalTensor> out = oOperand[0];
    npu::tile_fwk::Distributed::CommGroupInfo groupInfo;
    op.GetAttr(OpAttributeKey::commGroupInfo, groupInfo);
    npu::tile_fwk::Distributed::TilingInfo tilingInfo;
    op.GetAttr(OpAttributeKey::distTilingInfo, tilingInfo);
    std::string tilingSymbol;
    op.GetAttr("tiling_tensor_symbol", tilingSymbol);

    TensorTileInfo tileInfo;
    CheckAndGetTileInfo(tilingInfo.rowPerRank, tilingInfo.colPerRank, tileShape, tileInfo);

    TileArgs args = {function, in, tilingTensor, out, tilingInfo, tileInfo, groupInfo, tilingSymbol};
    TileColAndRowProcess<TileArgs>(DealTileBelongLocal, args);
}

void TiledDistScatter(Function &function, const TileShape &tileShape,
        const std::vector<std::shared_ptr<LogicalTensor>> &iOperand, const Operation &op)
{
    if (iOperand.size() != 2UL) {
        ALOG_ERROR_F("TiledDistScatter iOperand size=%lu", iOperand.size());
        return;
    }
    std::shared_ptr<LogicalTensor> in = iOperand[0];
    std::shared_ptr<LogicalTensor> tilingTensor = iOperand[1];
    npu::tile_fwk::Distributed::CommGroupInfo groupInfo;
    op.GetAttr(OpAttributeKey::commGroupInfo, groupInfo);
    npu::tile_fwk::Distributed::TilingInfo tilingInfo;
    op.GetAttr(OpAttributeKey::distTilingInfo, tilingInfo);
    std::string tilingSymbol;
    op.GetAttr("tiling_tensor_symbol", tilingSymbol);

    TensorTileInfo tileInfo;
    CheckAndGetTileInfo(tilingInfo.rowPerRank, tilingInfo.colPerRank, tileShape, tileInfo);

    TileArgs args = {function, in, tilingTensor, nullptr, tilingInfo, tileInfo, groupInfo, tilingSymbol};
    TileColAndRowProcess<TileArgs>(DealTileBelongRemote, args);
}
} // namespace Distributed
} // namespace npu::tile_fwk
