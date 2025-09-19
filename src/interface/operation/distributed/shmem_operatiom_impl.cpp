/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file shmem_operation_impl.cpp
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

namespace npu::tile_fwk::Distributed {

Tensor AddShmemPut(const Tensor &in, const Tensor &shmemDataTile, const int tileCount)
{
    auto &function = *Program::GetInstance().GetCurrentFunction();
    Shape shape{tileCount, 1};
    auto dummy = std::make_shared<LogicalTensor>(function, DT_INT32, shape);
    auto &op= function.AddOperation("SHMEM_PUT", {in.GetStorage(), shmemDataTile.GetStorage()}, {dummy});
    op.SetAttr("AtomicType", std::string("TileOp::Distributed::AtomicType::SET"));
    return dummy;
}

void AddShmemSignal(const Tensor &dummy, const Tensor &shmemSignalTile, std::string atomicType)
{
    auto &function = *Program::GetInstance().GetCurrentFunction();
    auto &op= function.AddOperation("SHMEM_SIGNAL", {dummy.GetStorage()}, {shmemSignalTile.GetStorage()});
    std::string value = "1";
    op.SetAttr("Value", value);
    op.SetAttr("AtomicType", atomicType);
}

Tensor AddShmemGet(const Tensor &dummy, const Tensor &shmemDataTile)
{
    auto &function = *Program::GetInstance().GetCurrentFunction();
    Shape shape = {shmemDataTile.GetShape()[2], shmemDataTile.GetShape()[3]};
    auto tempOutTile = std::make_shared<LogicalTensor>(function, shmemDataTile.GetDataType(), shape);
    auto &op= function.AddOperation("SHMEM_GET", {dummy.GetStorage(), shmemDataTile.GetStorage()}, {tempOutTile});
    op.SetAttr("AtomicType", std::string("TileOp::Distributed::AtomicType::SET"));
    return tempOutTile;
}

Tensor AddWaitUntil(const Tensor &in, const Tensor &shmemSignalTile, const int tileCount, const int value)
{
    auto &function = *Program::GetInstance().GetCurrentFunction();
    Shape shape {tileCount, 1};
    auto dummy = std::make_shared<LogicalTensor>(function, DT_INT32, shape);
    auto &op= function.AddOperation("SHMEM_WAIT_UNTIL", {in.GetStorage(), shmemSignalTile.GetStorage()}, {dummy});
    std::string stride = "32";
    op.SetAttr("Value", std::to_string(value));
    op.SetAttr("Stride", stride);
    return dummy;
}

void AddShmemReduce(const Tensor &in, const Tensor &shmData, const Tensor &dummy, const Tensor &out)
{
    auto &function = *Program::GetInstance().GetCurrentFunction();
    auto &op = function.AddOperation("SHMEM_REDUCE", {in.GetStorage(), shmData.GetStorage(), dummy.GetStorage()},
        {out.GetStorage()});
    // fp16 和 bf16 做reduce计算，默认转化为fp32
    if ((in.GetDataType() == DT_BF16) || (in.GetDataType() == DT_BF16)) {
        op.SetAttr("FP32Mode", true);
    } else {
        op.SetAttr("FP32Mode", false);
    }
}

void AllGatherDyn(const Tensor &in, const char *group, Tensor &out)
{
    FunctionConfig funConfig;
    FUNCTION("ALLGATHER", funConfig, {in}, {out}) {
        TileShape::Current().SetDistTile(
            {128, 1, 0},
            {256, 1, 0},
            {1, 4, 0});
        int groupIndex = static_cast<int>(Program::GetInstance().GetCommGroupRecorder().Input(std::string(group)));
        const TileShape &tileShape = TileShape::Current();
        auto rankShape = tileShape.GetDistTileRank();
        int rankSize = rankShape[0] * rankShape[1] + rankShape[2];

        auto tileRow = tileShape.GetDistTileRow();
        auto tileCol = tileShape.GetDistTileCol();
        int rowCount = tileRow[1] + (tileRow[2] != 0? 1: 0);
        int colCount = tileCol[1] + (tileCol[2] != 0? 1: 0);
        int tileCount = rowCount * colCount;
        int row = in.GetShape(0);
        int col = in.GetShape(1);

        Shape shmDataShape = {row, col};
        Shape shmSignalShape = {tileCount, 8};
        Shape outShape = {row * rankSize, col};

        SymbolicScalar thisRank = GetHcclRankId(groupIndex);

        auto shmemData = ShmemTensorMgr::GetInstance().CreateTensor(
            rankSize, groupIndex, in.GetDataType(), shmDataShape);
        auto shmemSignal = ShmemTensorMgr::GetInstance().CreateTensor(rankSize, groupIndex, DT_INT32, shmSignalShape);
        std::string atomicType = "TileOp::Distributed::AtomicType::SET";
        LOOP("L0", FunctionType::DYNAMIC_LOOP, dynRankId, LoopRange(0, rankSize, 1)) {
            auto shmemDataTile = ShmemTensorMgr::GetInstance().GetView(shmemData,
                {1, 1, row, col}, std::vector<SymbolicScalar>{dynRankId, thisRank, 0, 0});
            auto shmemSignalTile = ShmemTensorMgr::GetInstance().GetView(shmemSignal,
                {1, 1, tileCount, 8}, std::vector<SymbolicScalar>{dynRankId, thisRank, 0, 0});
            auto dummy = AddShmemPut(in, shmemDataTile, tileCount);
            AddShmemSignal(dummy, shmemSignalTile, atomicType);

            auto shmemDataLocal = ShmemTensorMgr::GetInstance().GetView(shmemData,
                {1, 1, row, col}, std::vector<SymbolicScalar>{thisRank, dynRankId, 0, 0});
            auto shmemSignalLocal = ShmemTensorMgr::GetInstance().GetView(shmemSignal,
                {1, 1, tileCount, 8}, std::vector<SymbolicScalar>{thisRank, dynRankId, 0, 0});
            auto dummyLocal = AddWaitUntil(in, shmemSignalLocal, tileCount, 1);
            auto tempOutTile = AddShmemGet(dummyLocal, shmemDataLocal);
            Assemble(tempOutTile, {dynRankId * row , 0}, out);
        }
    }
}

Tensor ShmemReduceScatter(Tensor &in, const char* group, DistReduceType reduceType)
{
    (void)reduceType;
    int hcclGroupIndex = static_cast<int>(Program::GetInstance().GetCommGroupRecorder().Input(std::string(group)));
    const TileShape &tileShape = TileShape::Current();
    auto rankShape = tileShape.GetDistTileRank();
    int rankSize = rankShape[0] * rankShape[1] + rankShape[2];

    auto tileRow = tileShape.GetDistTileRow();
    auto tileCol = tileShape.GetDistTileCol();
    int rowCount = tileRow[1] + (tileRow[2] != 0? 1: 0);
    int colCount = tileCol[1] + (tileCol[2] != 0? 1: 0);
    int tileCount = rowCount * colCount;
    int row = in.GetShape(0);
    int col = in.GetShape(1);
    ASSERT((row % rankSize) == 0);
    const int rowOut = row / rankSize;

    SymbolicScalar thisRank = GetHcclRankId(hcclGroupIndex);

    Shape outShape = {rowOut, col};
    Shape signalShape = {tileCount, 8};
    Tensor out(in.GetDataType(), outShape, "out");
    auto shmemData = ShmemTensorMgr::GetInstance().CreateTensor(rankSize, hcclGroupIndex, in.GetDataType(), outShape);
    auto shmemSignal = ShmemTensorMgr::GetInstance().CreateTensor(rankSize, hcclGroupIndex, DT_INT32, signalShape);

    std::string atomicType = "TileOp::Distributed::AtomicType::ADD";
    for (int i = 1; i < rankSize; i++) {
        SymbolicScalar otherRank = (thisRank + i) % rankSize;
        auto shmDataRank = ShmemTensorMgr::GetInstance().GetView(shmemData,
            {1, 1, rowOut, col}, std::vector<SymbolicScalar>{otherRank, thisRank, 0, 0});
        auto shmSignalRank = ShmemTensorMgr::GetInstance().GetView(shmemSignal,
            {1, 1, tileCount, 8}, std::vector<SymbolicScalar>{otherRank, 0, 0, 0});
        auto inRank = View(in, {rowOut, col}, std::vector<SymbolicScalar>{otherRank * rowOut, 0});
        auto dummy = AddShmemPut(inRank, shmDataRank, tileCount);
        AddShmemSignal(dummy, shmSignalRank, atomicType);
    }

    auto shmDataLocal = ShmemTensorMgr::GetInstance().GetView(shmemData,
        {1, rankSize, rowOut, col}, std::vector<SymbolicScalar>{thisRank, 0, 0, 0});
    auto shmSignalLocal = ShmemTensorMgr::GetInstance().GetView(shmemSignal,
        {1, 1, tileCount, 8}, std::vector<SymbolicScalar>{thisRank, 0, 0, 0});
    auto inLocal = View(in, {rowOut, col}, std::vector<SymbolicScalar>{thisRank * rowOut, 0});
    auto dummyLocal = AddWaitUntil(inLocal, shmSignalLocal, tileCount, rankSize - 1);
    AddShmemReduce(inLocal, shmDataLocal, dummyLocal, out);
    return out;
}

}   // namespace npu::tile_fwk::Distributed