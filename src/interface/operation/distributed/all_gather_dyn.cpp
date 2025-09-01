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
 * \file all_gather_dyn.cpp
 * \brief
 */

#include <type_traits>
#include "distributed_common.h"
#include "interface/operation/operation.h"
#include "interface/function/function.h"
#include "tilefwk/distributed.h"
#include "tilefwk/tensor.h"
#include "interface/tensor/logical_tensor.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"
#include "interface/utils/common.h"
#include "interface/utils/log.h"
#include "tilefwk/shmem_tensor_manager.h"

namespace npu::tile_fwk {
namespace Distributed {
Tensor AddShmemPut(const Tensor &in, const Tensor &shmemDataTile, const CommGroupInfo &groupInfo, const TilingInfo &tilingInfo)
{
    auto &function = *Program::GetInstance().GetCurrentFunction();
    Shape shape{1, 1};
    auto dummy = std::make_shared<LogicalTensor>(function, DT_INT32, shape);
    auto &op= function.AddOperation("SHMEM_PUT", {in.GetStorage(), shmemDataTile.GetStorage()}, {dummy});
    op.SetAttr(OpAttributeKey::commGroupInfo, groupInfo);
    op.SetAttr(OpAttributeKey::distTilingInfo, tilingInfo);
    op.SetAttr("AtomicType", std::string("TileOp::Distributed::AtomicType::SET"));
    return dummy;
}

void AddShmemSignal(const Tensor &dummy, const Tensor &shmemSignalTile, const CommGroupInfo &groupInfo, const TilingInfo &tilingInfo)
{
    auto &function = *Program::GetInstance().GetCurrentFunction();
    auto &op= function.AddOperation("SHMEM_SIGNAL", {dummy.GetStorage()}, {shmemSignalTile.GetStorage()});
    op.SetAttr(OpAttributeKey::commGroupInfo, groupInfo);
    op.SetAttr(OpAttributeKey::distTilingInfo, tilingInfo);
    std::string value = "1";
    op.SetAttr("Value", value);
    op.SetAttr("AtomicType", std::string("TileOp::Distributed::AtomicType::SET"));
}

Tensor AddShmemGet(const Tensor &dummy, const Tensor &shmemDataTile, const CommGroupInfo &groupInfo, const TilingInfo &tilingInfo)
{
    auto &function = *Program::GetInstance().GetCurrentFunction();
    Shape shape = {shmemDataTile.GetShape()[2], shmemDataTile.GetShape()[3]};
    auto tempOutTile = std::make_shared<LogicalTensor>(function, shmemDataTile.GetDataType(), shape);
    auto &op= function.AddOperation("SHMEM_GET", {dummy.GetStorage(), shmemDataTile.GetStorage()}, {tempOutTile});
    op.SetAttr(OpAttributeKey::commGroupInfo, groupInfo);
    op.SetAttr(OpAttributeKey::distTilingInfo, tilingInfo);
    op.SetAttr("AtomicType", std::string("TileOp::Distributed::AtomicType::SET"));
    return tempOutTile;
}

Tensor AddWaitUntil(const Tensor &in, const Tensor &shmemSignalTile, const CommGroupInfo &groupInfo, const TilingInfo &tilingInfo)
{
    auto &function = *Program::GetInstance().GetCurrentFunction();
    Shape shape {1, 1};
    auto dummy = std::make_shared<LogicalTensor>(function, DT_INT32, shape);
    auto &op= function.AddOperation("SHMEM_WAIT_UNTIL", {in.GetStorage(), shmemSignalTile.GetStorage()}, {dummy});
    op.SetAttr(OpAttributeKey::commGroupInfo, groupInfo);
    op.SetAttr(OpAttributeKey::distTilingInfo, tilingInfo);
    std::string value = "1";
    std::string stride = "32";
    op.SetAttr("Value", value);
    op.SetAttr("Stride", stride);
    return dummy;
}

void AllGatherDyn(const Tensor &in, const char *group, Tensor &out)
{
    FunctionConfig funConfig;
    FUNCTION("ALLGATHER", funConfig, {in}, {out}) {
        Program::GetInstance().GetTileShape().SetDistTileShapes(
            {32, 1, 0},
            {256, 1, 0},
            {1, 4, 0});
        int groupIndex = static_cast<int>(Program::GetInstance().GetCommGroupRecorder().Input(std::string(group)));
        const TileShape &tileShape = Program::GetInstance().GetTileShape();
        CommGroupInfo groupInfo;
        groupInfo.groupIndex = groupIndex;

        auto rankShape = tileShape.GetDistTileRank();
        int rankSize = rankShape[0] * rankShape[1] + rankShape[2];
        groupInfo.rankSize = std::make_optional(rankSize);

        Shape shape = {in->shape[0] * rankSize, in->shape[1]};
        TilingInfo tilingInfo;
        tilingInfo.groupIndex = groupInfo.groupIndex;
        tilingInfo.rowPerRank = in.GetStorage()->shape[0];
        tilingInfo.colPerRank = in.GetStorage()->shape[1];

        SymbolicScalar thisRank = GetHcclRankId(groupInfo.groupIndex);

        Shape shape1 = {tilingInfo.rowPerRank, tilingInfo.colPerRank};
        auto shmemData = ShmemTensorMgr::GetInstance().CreateTensor(groupInfo.rankSize.value(), groupInfo.groupIndex, in.GetStorage()->Datatype(), shape1);
        Shape signalShape = {1, 8};
        auto shmemSignal = ShmemTensorMgr::GetInstance().CreateTensor(groupInfo.rankSize.value(), groupInfo.groupIndex, DT_INT32, signalShape);
        LOOP("L0", FunctionType::DYNAMIC_LOOP, dynRankId, LoopRange(0, groupInfo.rankSize.value(), 1)) {
            auto shmemDataTile = ShmemTensorMgr::GetInstance().GetView(shmemData, {1, 1, tilingInfo.rowPerRank, tilingInfo.colPerRank}, std::vector<SymbolicScalar>{dynRankId, thisRank, 0, 0});
            auto shmemSignalTile = ShmemTensorMgr::GetInstance().GetView(shmemSignal, {1, 1, 1, 8}, std::vector<SymbolicScalar>{dynRankId, thisRank, 0, 0});
            auto dummy = AddShmemPut(in, shmemDataTile, groupInfo, tilingInfo);
            AddShmemSignal(dummy, shmemSignalTile, groupInfo, tilingInfo);

            auto shmemDataLocal = ShmemTensorMgr::GetInstance().GetView(shmemData, {1, 1, tilingInfo.rowPerRank, tilingInfo.colPerRank}, std::vector<SymbolicScalar>{thisRank, dynRankId, 0, 0});
            auto shmemSignalLocal = ShmemTensorMgr::GetInstance().GetView(shmemSignal, {1, 1, 1, 8}, std::vector<SymbolicScalar>{thisRank, dynRankId, 0, 0});
            auto dummyLocal = AddWaitUntil(in, shmemSignalLocal, groupInfo, tilingInfo);
            auto tempOutTile = AddShmemGet(dummyLocal, shmemDataLocal, groupInfo, tilingInfo);
            Assemble(tempOutTile, {dynRankId * tempOutTile.GetShape()[0] , 0}, out);
        }
    }
}

} // namespace Distributed
} // namespace npu::tile_fwk
