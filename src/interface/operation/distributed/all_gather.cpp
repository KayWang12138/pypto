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
 * \file all_gather.cpp
 * \brief
 */

#include <type_traits>
#include "distributed_common.h"
#include "interface/operation/operation.h"
#include "interface/function/function.h"
#include "tilefwk/tensor.h"
#include "interface/tensor/logical_tensor.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"
#include "interface/utils/common.h"
#include "interface/utils/log.h"

namespace npu::tile_fwk {
namespace Distributed {
void WriteRemoteProcess(const TileArgs &args)
{
    auto inTile = args.in->View(args.function, {args.tilingInfo.rowShape, args.tilingInfo.colShape},
        {args.tilingInfo.rowOffset, args.tilingInfo.colOffset});
    std::vector<int64_t> flagShape = {1, FLAG_TENSOR_SIZE}; // 256B
    auto flagTensor = std::make_shared<LogicalTensor>(args.function, DT_INT32, flagShape);
    OpArgs<TilingInfo> opArgs = {"WRITE_REMOTE", {inTile}, {flagTensor}, args.tilingTensor, args.tilingSymbol,
        std::make_optional(args.tilingInfo), std::nullopt};
    (void)AddOperation(args.function, opArgs);
}

void WaitFlagAndRemoteGatherProcess(const TileArgs &args)
{
    std::vector<int64_t> shape = {args.tilingInfo.rowShape, args.tilingInfo.colShape};
    std::vector<int64_t> offset = {args.tilingInfo.rowOffset, args.tilingInfo.colOffset};
    auto outTile = args.out->View(args.function, shape, offset);

    const bool aicpuWaitFlagEnable = ConfigManager::Instance().GetDistConfig(KEY_AICPU_WAIT_FLAG_ENABLE, true);
    std::vector<int64_t> flagShape = {1, FLAG_TENSOR_SIZE}; // 256B
    auto inTensor = std::make_shared<LogicalTensor>(args.function, args.in->Datatype(), shape);
    auto flagTensor = std::make_shared<LogicalTensor>(args.function, DataType::DT_INT32, flagShape);

    if (aicpuWaitFlagEnable) {
        std::vector<int64_t> opAttr = {args.tilingInfo.tileIndex, args.tilingInfo.groupIndex, args.tilingInfo.rankShape,
            args.tilingInfo.rankOffset};
        auto inTile = args.in->View(args.function, shape, offset);
        OpArgs<TilingInfo> opArgs = {"COMM_WAIT_FLAG", {inTile}, {flagTensor}, args.tilingTensor, args.tilingSymbol,
            std::nullopt, std::make_optional(opAttr)};
        AddOperation(args.function, opArgs);
    }

    // flagTensor is control edge between remote gather and comm_wait_flag
    OpArgs<TilingInfo> opArgs = {"REMOTE_GATHER", {flagTensor}, {outTile, inTensor}, args.tilingTensor, args.tilingSymbol,
        std::make_optional(args.tilingInfo), std::nullopt};
    auto &op = AddOperation(args.function, opArgs);
    if (!aicpuWaitFlagEnable) {
        op.SetAttr("extraTemplateParam", std::string("true"));
    }
}

void LocalCopyOutProcess(const TileArgs &args)
{
    std::vector<int64_t> shape = {args.tilingInfo.rowShape, args.tilingInfo.colShape};
    std::vector<int64_t> offset = {args.tilingInfo.rowOffset, args.tilingInfo.colOffset};
    auto inTile = args.in->View(args.function, shape, offset);
    auto outTile = args.out->View(args.function, shape, offset);

    OpArgs<TilingInfo> opArgs = {"LOCAL_COPY_OUT", {inTile}, {outTile}, args.tilingTensor, args.tilingSymbol,
        std::make_optional(args.tilingInfo), std::nullopt};
    (void)AddOperation(args.function, opArgs);
}

inline LogicalTensorPtr GetTensorView(Function &function, const std::vector<Tensor> &tensor,
    const std::vector<int64_t> &shape, const int rankIndex)
{
    (void)shape;
    (void)function;
    return tensor[rankIndex].GetStorage();
}

inline LogicalTensorPtr GetTensorView(Function &function, const Tensor &tensor, const std::vector<int64_t> &shape,
    const int rankIndex)
{
    const std::vector<int64_t> offset = {rankIndex * shape[0], 0};
    return tensor.GetStorage()->View(function, shape, offset);
}

void RemoteTileOpCallback(const TileArgs &args)
{
    WriteRemoteProcess(args);
    WaitFlagAndRemoteGatherProcess(args);
}

void TileRankProcess(TileArgs &args, const int32_t rankIndex)
{
    args.tilingInfo.rankShape = 1;
    args.tilingInfo.rankOffset = rankIndex;
}

void AllGatherRemoteTileProcess(TileArgs &args, const int32_t rankIndex)
{
    TileRankProcess(args, rankIndex);
    TileColAndRowProcess<TileArgs>(RemoteTileOpCallback, args);
}

void LocalTileOpCallback(TileArgs &args)
{
    LocalCopyOutProcess(args);
}

void AllGatherLocalTileProcess(TileArgs &args, const int32_t rankIndex)
{
    TileRankProcess(args, rankIndex);
    TileColAndRowProcess<TileArgs>(LocalTileOpCallback, args);
}

void DistGatherTileOpCallback(const TileArgs &args)
{
    WaitFlagAndRemoteGatherProcess(args);
}

void DistLocalTileOpCallback(const TileArgs &args)
{
    LocalCopyOutProcess(args);
}

void DistBroadCastTileOpCallback(const TileArgs &args)
{
    WriteRemoteProcess(args);
}

void DistGatherTileProcess(TileArgs &args, const int32_t rankIndex)
{
    TileRankProcess(args, rankIndex);
    if (rankIndex != args.groupInfo.rankId.value()) {
        TileColAndRowProcess<TileArgs>(DistGatherTileOpCallback, args);
    } else {
        TileColAndRowProcess<TileArgs>(DistLocalTileOpCallback, args);
    }
}

void DistBroadCastTileProcess(TileArgs &args, const int32_t rankIndex)
{
    TileRankProcess(args, rankIndex);
    TileColAndRowProcess<TileArgs>(DistBroadCastTileOpCallback, args);
}

template <typename T>
void TensorAllGatherTenor(const LogicalTensorPtr &in, const LogicalTensorPtr &tilingTensor,
    const CommGroupInfo &groupInfo, T &out)
{
    auto &function = *Program::GetInstance().GetCurrentFunction();
    TilingInfo tilingInfo;
    tilingInfo.groupIndex = groupInfo.groupIndex;
    tilingInfo.rowPerRank = in->shape[0];
    tilingInfo.colPerRank = in->shape[1];
    T tmpOut(out->Datatype(), out->GetShape());
    for (int rankIndex = 0; rankIndex < groupInfo.rankSize.value(); rankIndex++) {
        tilingInfo.rankShape = 1;
        tilingInfo.rankOffset = rankIndex;
        auto outPerRank = GetTensorView(function, tmpOut, in->GetShape(), rankIndex);
        if (rankIndex != groupInfo.rankId.value()) {
            // 远卡的时候，把本卡数据 broadcast 到其他卡
            auto &oper = function.AddOperation("DIST_BROADCAST", {in, tilingTensor},
                {outPerRank});
            oper.SetAttr(OpAttributeKey::commGroupInfo, groupInfo);
            oper.SetAttr(OpAttributeKey::distTilingInfo, tilingInfo);
            oper.SetAttr("tiling_tensor_symbol", tilingTensor->Symbol());
        }
        // gather 本卡及其他卡数据写到 out
        auto &oper = function.AddOperation("DIST_GATHER", {in, tilingTensor},
            {outPerRank});
        oper.SetAttr(OpAttributeKey::commGroupInfo, groupInfo);
        oper.SetAttr(OpAttributeKey::distTilingInfo, tilingInfo);
        oper.SetAttr("tiling_tensor_symbol", tilingTensor->Symbol());
    }
    if constexpr (std::is_same_v<T, Tensor>) {
        Operation &op = function.AddOperation(Opcode::OP_ASSEMBLE, {tmpOut.GetStorage()}, {out.GetStorage()});
        op.SetOpAttribute(std::make_shared<AssembleOpAttribute>(std::vector<int64_t>(out->GetOffset().size())));
    }
}

template <typename T>
void TensorAllGatherVector(const LogicalTensorPtr &in, const LogicalTensorPtr &tilingTensor,
    const CommGroupInfo &groupInfo, T &out)
{
    auto &function = *Program::GetInstance().GetCurrentFunction();
    TilingInfo tilingInfo;
    tilingInfo.groupIndex = groupInfo.groupIndex;
    tilingInfo.rowPerRank = in->shape[0];
    tilingInfo.colPerRank = in->shape[1];
    for (int rankIndex = 0; rankIndex < groupInfo.rankSize.value(); rankIndex++) {
        tilingInfo.rankShape = 1;
        tilingInfo.rankOffset = rankIndex;
        auto outPerRank = GetTensorView(function, out, in->GetShape(), rankIndex);
        if (rankIndex != groupInfo.rankId.value()) {
            // 远卡的时候，把本卡数据 broadcast 到其他卡
            auto &oper = function.AddOperation("DIST_BROADCAST", {in, tilingTensor},
                {outPerRank});
            oper.SetAttr(OpAttributeKey::commGroupInfo, groupInfo);
            oper.SetAttr(OpAttributeKey::distTilingInfo, tilingInfo);
            oper.SetAttr("tiling_tensor_symbol", tilingTensor->Symbol());
        }
        // gather 本卡及其他卡数据写到 out
        auto &oper = function.AddOperation("DIST_GATHER", {in, tilingTensor},
            {outPerRank});
        oper.SetAttr(OpAttributeKey::commGroupInfo, groupInfo);
        oper.SetAttr(OpAttributeKey::distTilingInfo, tilingInfo);
        oper.SetAttr("tiling_tensor_symbol", tilingTensor->Symbol());
    }
}

void TiledDistGather(Function &function, const TileShape &tileShape,
    const std::vector<std::shared_ptr<LogicalTensor>> &iOperand,
    const std::vector<std::shared_ptr<LogicalTensor>> &oOperand, const Operation &op)
{
    if (iOperand.size() != 2UL || oOperand.size() != 1UL) {
        ALOG_ERROR_F("TiledDistGather iOperand size=%lu, oOperand size=%lu", iOperand.size(), oOperand.size());
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
    DistGatherTileProcess(args, tilingInfo.rankOffset);
}

void TiledDistBroadCast(Function &function, const TileShape &tileShape,
    const std::vector<std::shared_ptr<LogicalTensor>> &iOperand,
    const std::vector<std::shared_ptr<LogicalTensor>> &oOperand, const Operation &op)
{
    if (iOperand.size() != 2UL || oOperand.size() != 1UL) {
        ALOG_ERROR_F("TiledDistBroadCast iOperand size=%lu, oOperand size=%lu", iOperand.size(), oOperand.size());
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
    DistBroadCastTileProcess(args, tilingInfo.rankOffset);
}

template <typename T>
void AllGatherImpl(const Tensor &in, T &out, const char *group)
{
    int groupIndex = static_cast<int>(Program::GetInstance().GetCommGroupRecorder().Input(std::string(group)));
    const TileShape &tileShape = TileShape::Current();
    CommGroupInfo groupInfo;
    CheckAndGetGroupInfo(groupIndex, tileShape, groupInfo);

    TensorTileInfo tileInfo;
    CheckAndGetTileInfo(in->shape[0], in->shape[1], tileShape, tileInfo);

    auto &function = *Program::GetInstance().GetCurrentFunction();
    int32_t tilingTensorSize = GetTilingTensorSize(tileInfo, groupInfo);
    std::vector<int64_t> tilingShape = {1, tilingTensorSize};
    const std::string tilingSymbol = function.GetDistTilingManager()->CreateTilingStorage("allgather", tilingShape[1]);
    Tensor tilingTensor(DataType::DT_INT32, tilingShape, tilingSymbol);
    if constexpr (std::is_same_v<T, Tensor>) {
        TensorAllGatherTenor(in.GetStorage(), tilingTensor.GetStorage(), groupInfo, out);
    } else {
        TensorAllGatherVector(in.GetStorage(), tilingTensor.GetStorage(), groupInfo, out);
    }
}

void AllGather(const Tensor &in, std::vector<Tensor> &out, const char *group)
{
    AllGatherImpl<std::vector<Tensor>>(in, out, group);
}

inline std::vector<int64_t> GetOutShape(const Tensor &in)
{
    const TileShape &tileShape = TileShape::Current();
    auto rankShape = tileShape.GetDistTileRank();
    ASSERT((rankShape[DIST_HEAD_SHAPE] >= 0) && (rankShape[DIST_HEAD_COUNT] >= 0) && (rankShape[DIST_TAIL_SHAPE] >= 0));
    int32_t rankSize = rankShape[DIST_HEAD_SHAPE] * rankShape[DIST_HEAD_COUNT] + rankShape[DIST_TAIL_SHAPE];
    ASSERT(rankSize > 0);

    return {in->shape[0] * rankSize, in->shape[1]};
}

Tensor AllGather(const Tensor &in, const char *group)
{
    auto outShape = GetOutShape(in);
    Tensor out(in.GetDataType(), outShape, "out");
    AllGatherImpl<Tensor>(in, out, group);
    return out;
}

} // namespace Distributed
} // namespace npu::tile_fwk
