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
 * \file shmem_operation_impl.cpp
 * \brief
*/

#include <type_traits>
#include "distributed_common.h"
#include "interface/operation/operation.h"
#include "interface/function/function.h"
#include "tilefwk/symbolic_distributed.h"
#include "tilefwk/tensor.h"
#include "interface/tensor/logical_tensor.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"
#include "interface/utils/common.h"
#include "interface/utils/log.h"

namespace npu::tile_fwk::Distributed {
std::pair<int, int> GetRankSizeAndTileCount()
{
    const TileShape& tileShape = TileShape::Current();

    auto rankShape = tileShape.GetDistTileRank();
    int rankSize = rankShape[0] * rankShape[1] + rankShape[2];

    auto tileRow = tileShape.GetDistTileRow();
    auto tileCol = tileShape.GetDistTileCol();
    int rowCount = tileRow[1] + (tileRow[2] != 0 ? 1 : 0);
    int colCount = tileCol[1] + (tileCol[2] != 0 ? 1 : 0);
    int tileCount = rowCount * colCount;

    return {rankSize, tileCount};
}

Tensor AddShmemPut(const Tensor &in, const Tensor &shmemDataTile, const Tensor &barrierDummy, int tileCount,
    AtomicType atomicType = AtomicType::SET)
{
    auto &function = *Program::GetInstance().GetCurrentFunction();
    Shape shape{tileCount, 1};
    auto dummy = std::make_shared<LogicalTensor>(function, DT_INT32, shape);
    auto &op = function.AddOperation("SHMEM_PUT",
        {in.GetStorage(), shmemDataTile.GetStorage(), barrierDummy.GetStorage()}, {dummy});
    op.SetAttr("AtomicType", atomicType);
    return dummy;
}

Tensor AddShmemSignal(const Tensor &dummy, const Tensor &shmemSignalTile, AtomicType atomicType)
{
    auto &function = *Program::GetInstance().GetCurrentFunction();
    auto dummyOut = std::make_shared<LogicalTensor>(function, DT_INT32, dummy.GetShape());
    auto &op = function.AddOperation("SHMEM_SIGNAL", {dummy.GetStorage(), shmemSignalTile.GetStorage()}, {dummyOut});
    int64_t value = 1;
    op.SetAttr("Value", value);
    op.SetAttr("AtomicType", atomicType);
    op.SetAttr(OpAttributeKey::dontTouch, true);
    return dummyOut;
}

Tensor AddShmemGet(const Tensor &dummy, const Tensor &shmemDataTile, AtomicType atomicType = AtomicType::SET)
{
    auto &function = *Program::GetInstance().GetCurrentFunction();
    Shape shape = {shmemDataTile.GetShape()[2], shmemDataTile.GetShape()[3]};
    auto tempOutTile = std::make_shared<LogicalTensor>(function, shmemDataTile.GetDataType(), shape);
    auto &op = function.AddOperation("SHMEM_GET", {dummy.GetStorage(), shmemDataTile.GetStorage()}, {tempOutTile});
    op.SetAttr("AtomicType", atomicType);
    return tempOutTile;
}

Tensor AddWaitUntil(const Tensor &dummyIn, const Tensor &shmemSignalTile, int tileCount, int hcclGroupIndex,
    int expectedSum)
{
    auto &function = *Program::GetInstance().GetCurrentFunction();
    Shape shape{tileCount, 1};
    auto dummy = std::make_shared<LogicalTensor>(function, DT_INT32, shape);
    auto &op = function.AddOperation("SHMEM_WAIT_UNTIL", {dummyIn.GetStorage(), shmemSignalTile.GetStorage()}, {dummy});
    std::vector<int64_t> param = {static_cast<int64_t>(hcclGroupIndex), static_cast<int64_t>(expectedSum)};
    op.SetAttr("AicpuOpParams", param);
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

Tensor AddShmemClearSignal(const Tensor &in, const Tensor &shmemSignalTile, const int tileCount)
{
    auto &function = *Program::GetInstance().GetCurrentFunction();
    Shape shape{tileCount, 1};
    auto dummy = std::make_shared<LogicalTensor>(function, DT_INT32, shape);
    function.AddOperation("SHMEM_CLEAR_SIGNAL", {in.GetStorage(), shmemSignalTile.GetStorage()}, {dummy});
    return dummy;
}

Tensor CreateShmemTensor(int32_t rankSize, int32_t hcclGroupIndex, DataType dataType, const Shape &shape)
{
    auto &function = *Program::GetInstance().GetCurrentFunction();
    Shape shmemShape{rankSize, rankSize};
    shmemShape.insert(shmemShape.end(), shape.begin(), shape.end());
    auto shmemTensor = std::make_shared<LogicalTensor>(function, dataType, shmemShape);
    auto &op = function.AddOperation("BIND_TENSOR", {}, {shmemTensor});
    op.SetAttribute(OpAttributeKey::bindTensor, BindTensor(hcclGroupIndex, 0,
        BytesOf(dataType) * rankSize * std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int64_t>())));
    return shmemTensor;
}

Tensor Barrier(const Tensor &in, const char *group)
{
    int hcclGroupIndex = static_cast<int>(CommGroupRecorder::GetInstance().Input(std::string(group)));
    auto [rankSize, tileCount] = GetRankSizeAndTileCount();

    Shape shmSignalShape = {tileCount, 8};

    SymbolicScalar thisRank = GetHcclRankId(hcclGroupIndex);

    Tensor shmemSignal;
    Tensor shmemBarrierSignal;
    Tensor barrierDummy(DT_INT32, {1, 1}, "barrierDummy");
    LOOP("CreateShmemTensor", FunctionType::DYNAMIC_LOOP, index, LoopRange(1)) {
        (void)index;

        shmemSignal = CreateShmemTensor(rankSize, hcclGroupIndex, DT_INT32, shmSignalShape);

        // 虽然创建出来的 tensor 的 shape 是 {rankSize, rankSize, 1, 8}，但其实只需要用到最前面的 {rankSize, 1, 1, 8}
        shmemBarrierSignal = CreateShmemTensor(rankSize, hcclGroupIndex, DT_INT32, Shape{1, 8});
    }
    LOOP("Barrier", FunctionType::DYNAMIC_LOOP, index, LoopRange(1)) {
        (void)index;
        auto shmemSignalTile =
            View(shmemSignal, {1, 1, tileCount, 8}, std::vector<SymbolicScalar>{rankSize, rankSize, 0, 0});
        auto clearSignalDummy = AddShmemClearSignal(in, shmemSignalTile, tileCount);

        for (int32_t rank = 0; rank < rankSize; rank++) {
            auto shmemBarrierSignalTile =
                View(shmemBarrierSignal, {1, 1, 1, 8}, std::vector<SymbolicScalar>{rank, 0, 0, 0});
            AddShmemSignal(clearSignalDummy, shmemBarrierSignalTile, AtomicType::ADD);
        }
        auto shmemBarrierSignalLocal =
            View(shmemBarrierSignal, {1, 1, 1, 8}, std::vector<SymbolicScalar>{thisRank, 0, 0, 0});
        barrierDummy = AddWaitUntil(clearSignalDummy, shmemBarrierSignalLocal, 1, hcclGroupIndex, rankSize);
    }
    return barrierDummy;
}

void ShmemAllGather(const Tensor &in, const Tensor &barrierDummy, const char *group, Tensor &out)
{
    int hcclGroupIndex = static_cast<int>(CommGroupRecorder::GetInstance().Input(std::string(group)));
    auto [rankSize, tileCount] = GetRankSizeAndTileCount();
    int row = in.GetShape(0);
    int col = in.GetShape(1);

    Shape shmDataShape = {row, col};
    Shape shmSignalShape = {tileCount, 8};
    Shape outShape = {row * rankSize, col};

    SymbolicScalar thisRank = GetHcclRankId(hcclGroupIndex);

    Tensor shmemData;
    Tensor shmemSignal;
    LOOP("CreateShmemTensor", FunctionType::DYNAMIC_LOOP, index, LoopRange(1)) {
        (void)index;
        shmemData = CreateShmemTensor(rankSize, hcclGroupIndex, in.GetDataType(), shmDataShape);
        shmemSignal = CreateShmemTensor(rankSize, hcclGroupIndex, DT_INT32, shmSignalShape);
    }
    LOOP("L0", FunctionType::DYNAMIC_LOOP, dynRankId, LoopRange(0, rankSize, 1)) {
        auto shmemDataTile = View(shmemData, {1, 1, row, col}, std::vector<SymbolicScalar>{dynRankId, thisRank, 0, 0});
        auto shmemSignalTile =
            View(shmemSignal, {1, 1, tileCount, 8}, std::vector<SymbolicScalar>{dynRankId, thisRank, 0, 0});

        auto dummy = AddShmemPut(in, shmemDataTile, barrierDummy, tileCount);
        auto dummySignal = AddShmemSignal(dummy, shmemSignalTile, AtomicType::SET);

        auto shmemDataLocal = View(shmemData, {1, 1, row, col}, std::vector<SymbolicScalar>{thisRank, dynRankId, 0, 0});
        auto shmemSignalLocal =
            View(shmemSignal, {1, 1, tileCount, 8}, std::vector<SymbolicScalar>{thisRank, dynRankId, 0, 0});
        auto dummyLocal = AddWaitUntil(dummySignal, shmemSignalLocal, tileCount, hcclGroupIndex, 1);
        auto tempOutTile = AddShmemGet(dummyLocal, shmemDataLocal);
        Assemble(tempOutTile, {dynRankId * row, 0}, out);
    }
}

void ShmemReduceScatter(Tensor &in, const char* group, DistReduceType reduceType, Tensor &out)
{
    (void)reduceType;
    int hcclGroupIndex = static_cast<int>(CommGroupRecorder::GetInstance().Input(std::string(group)));
    auto [rankSize, tileCount] = GetRankSizeAndTileCount();
    int row = in.GetShape(0);
    int col = in.GetShape(1);
    ASSERT((row % rankSize) == 0);
    const int rowOut = row / rankSize;

    SymbolicScalar thisRank = GetHcclRankId(hcclGroupIndex);

    Shape outShape = {rowOut, col};
    Shape signalShape = {tileCount, 8};
    Tensor shmemData;
    Tensor shmemSignal;
    LOOP("CreateShmemTensor", FunctionType::DYNAMIC_LOOP, index, LoopRange(1)) {
        (void)index;
        shmemData = CreateShmemTensor(rankSize, hcclGroupIndex, in.GetDataType(), outShape);
        shmemSignal = CreateShmemTensor(rankSize, hcclGroupIndex, DT_INT32, signalShape);
    }
    Tensor barrierDummy(DT_INT32, {1, 1}, "barrierDummy");
    LOOP("RS", FunctionType::DYNAMIC_LOOP, dynRankId, LoopRange(0, rankSize, 1)) {
        auto shmemDataTile = View(shmemData, {1, 1, rowOut, col}, std::vector<SymbolicScalar>{dynRankId, 0, 0, 0});
        auto shmemSignalTile =
            View(shmemSignal, {1, 1, tileCount, 8}, std::vector<SymbolicScalar>{dynRankId, 0, 0, 0});
        auto inTile = View(in, {rowOut, col}, std::vector<SymbolicScalar>{dynRankId * rowOut, 0});
        auto dummy = AddShmemPut(inTile, shmemDataTile, barrierDummy, tileCount, AtomicType::ADD);
        auto dummySignal = AddShmemSignal(dummy, shmemSignalTile, AtomicType::ADD);

        IF (dynRankId == thisRank) {
            auto shmemDataLocal = View(shmemData, {1, 1, rowOut, col}, std::vector<SymbolicScalar>{thisRank, 0, 0, 0});
            auto shmemSignalLocal =
                View(shmemSignal, {1, 1, tileCount, 8}, std::vector<SymbolicScalar>{thisRank, 0, 0, 0});
            auto dummyLocal = AddWaitUntil(dummySignal, shmemSignalLocal, tileCount, hcclGroupIndex, rankSize);
            out = AddShmemGet(dummyLocal, shmemDataLocal);
        }
    }
}

}   // namespace npu::tile_fwk::Distributed
