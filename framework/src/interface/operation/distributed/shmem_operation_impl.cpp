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
    ASSERT(tileRow[0] > 0 && tileCol[0] > 0) << "Invalid Tiling rules";
    int rowCount = tileRow[1] + (tileRow[2] != 0 ? 1 : 0);
    int colCount = tileCol[1] + (tileCol[2] != 0 ? 1 : 0);
    int tileCount = rowCount * colCount;

    return {rankSize, tileCount};
}

Tensor ShmemPut(const Tensor &in, const Tensor &shmemDataTile, const Tensor &barrierDummy, int tileCount,
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

Tensor ShmemPutUb2Gm(const Tensor &in, const Tensor &shmemDataTile, const Tensor &barrierDummy, int tileCount,
    AtomicType atomicType = AtomicType::SET)
{
    auto &function = *Program::GetInstance().GetCurrentFunction();
    Shape shape{tileCount, 1};
    auto dummy = std::make_shared<LogicalTensor>(function, DT_INT32, shape);
    auto &op = function.AddOperation("SHMEM_PUT_UB2GM",
        {in.GetStorage(), shmemDataTile.GetStorage(), barrierDummy.GetStorage()}, {dummy});
    op.SetAttr("AtomicType", atomicType);
    return dummy;
}

Tensor ShmemSignal(const Tensor &dummy, const Tensor &shmemSignalTile, AtomicType atomicType)
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

Tensor ShmemGet(const Tensor &dummy, const Tensor &shmemDataTile, DataType nonShmemDataType = DataType::DT_BOTTOM,
    AtomicType atomicType = AtomicType::SET)
{
    if (nonShmemDataType == DT_BOTTOM) {
        nonShmemDataType = shmemDataTile.GetDataType();
    }
    auto &function = *Program::GetInstance().GetCurrentFunction();
    Shape shape = {shmemDataTile.GetShape()[2], shmemDataTile.GetShape()[3]};
    auto tempOutTile = std::make_shared<LogicalTensor>(function, nonShmemDataType, shape);
    auto &op = function.AddOperation("SHMEM_GET", {dummy.GetStorage(), shmemDataTile.GetStorage()}, {tempOutTile});
    op.SetAttr("AtomicType", atomicType);
    return tempOutTile;
}

Tensor ShmemGetGm2Ub(const Tensor &dummy, const Tensor &shmemDataTile, DataType nonShmemDataType = DataType::DT_BOTTOM,
    AtomicType atomicType = AtomicType::SET)
{
    if (nonShmemDataType == DT_BOTTOM) {
        nonShmemDataType = shmemDataTile.GetDataType();
    }
    auto &function = *Program::GetInstance().GetCurrentFunction();
    Shape shape = {shmemDataTile.GetShape()[2], shmemDataTile.GetShape()[3]};
    auto tempOutTile = std::make_shared<LogicalTensor>(function, nonShmemDataType, shape);
    auto &op = function.AddOperation("SHMEM_GET_GM2UB", {dummy.GetStorage(), shmemDataTile.GetStorage()}, {tempOutTile});
    op.SetAttr("AtomicType", atomicType);
    return tempOutTile;
}

Tensor WaitUntil(const Tensor &dummyIn, const Tensor &shmemSignalTile, int tileCount, int hcclGroupIndex,
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

void ShmemReduce(const Tensor &in, const Tensor &shmData, const Tensor &dummy, const Tensor &out)
{
    auto &function = *Program::GetInstance().GetCurrentFunction();
    auto &op = function.AddOperation("SHMEM_REDUCE", {in.GetStorage(), shmData.GetStorage(), dummy.GetStorage()},
        {out.GetStorage()});
    // fp16 和 bf16 做reduce计算，默认转化为fp32
    if ((in.GetDataType() == DT_FP16) || (in.GetDataType() == DT_BF16)) {
        op.SetAttr("FP32Mode", true);
    } else {
        op.SetAttr("FP32Mode", false);
    }
}

Tensor ShmemClearSignal(const Tensor &in, const Tensor &shmemSignalTile, const int tileCount)
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
        auto clearSignalDummy = ShmemClearSignal(in, shmemSignalTile, tileCount);

        for (int32_t rank = 0; rank < rankSize; rank++) {
            auto shmemBarrierSignalTile =
                View(shmemBarrierSignal, {1, 1, 1, 8}, std::vector<SymbolicScalar>{rank, 0, 0, 0});
            ShmemSignal(clearSignalDummy, shmemBarrierSignalTile, AtomicType::ADD);
        }
        auto shmemBarrierSignalLocal =
            View(shmemBarrierSignal, {1, 1, 1, 8}, std::vector<SymbolicScalar>{thisRank, 0, 0, 0});
        barrierDummy = WaitUntil(clearSignalDummy, shmemBarrierSignalLocal, 1, hcclGroupIndex, rankSize);
    }
    return barrierDummy;
}

void ShmemAllGather(const Tensor &in, const Tensor &barrierDummy, const char *group, Tensor &out)
{
    int hcclGroupIndex = static_cast<int>(CommGroupRecorder::GetInstance().Input(std::string(group)));
    int row = in.GetShape(0);
    int col = in.GetShape(1);
    ASSERT(row > 0 && col > 0) << "Invalid shape: row and col must be > 0, but got row=" << row << ", col=" << col;
    auto [rankSize, tileCount] = GetRankSizeAndTileCount();

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

        auto dummy = ShmemPut(in, shmemDataTile, barrierDummy, tileCount);
        auto dummySignal = ShmemSignal(dummy, shmemSignalTile, AtomicType::SET);

        auto shmemDataLocal = View(shmemData, {1, 1, row, col}, std::vector<SymbolicScalar>{thisRank, dynRankId, 0, 0});
        auto shmemSignalLocal =
            View(shmemSignal, {1, 1, tileCount, 8}, std::vector<SymbolicScalar>{thisRank, dynRankId, 0, 0});
        auto dummyLocal = WaitUntil(dummySignal, shmemSignalLocal, tileCount, hcclGroupIndex, 1);
        auto tempOutTile = ShmemGet(dummyLocal, shmemDataLocal);
        Assemble(tempOutTile, {dynRankId * row, 0}, out);
    }
}

void ShmemReduceScatter(Tensor &in, const char* group, DistReduceType reduceType, Tensor &out)
{
    (void)reduceType;
    int hcclGroupIndex = static_cast<int>(CommGroupRecorder::GetInstance().Input(std::string(group)));
    int row = in.GetShape(0);
    int col = in.GetShape(1);
    ASSERT(row > 0 && col > 0) << "Invalid shape: row and col must be > 0, but got row=" << row << ", col=" << col;
    auto [rankSize, tileCount] = GetRankSizeAndTileCount();
    ASSERT((row % rankSize) == 0);
    const int rowOut = row / rankSize;

    SymbolicScalar thisRank = GetHcclRankId(hcclGroupIndex);

    Shape outShape = {rowOut, col};
    Shape signalShape = {tileCount, 8};
    Tensor shmemData;
    Tensor shmemSignal;
    DataType shmemDataType = in.GetDataType();
    if ((shmemDataType == DT_BF16) || (shmemDataType == DT_FP16)) {
        shmemDataType = DT_FP32;
    }
    LOOP("CreateShmemTensor", FunctionType::DYNAMIC_LOOP, index, LoopRange(1)) {
        (void)index;
        shmemData = CreateShmemTensor(rankSize, hcclGroupIndex, shmemDataType, outShape);
        shmemSignal = CreateShmemTensor(rankSize, hcclGroupIndex, DT_INT32, signalShape);
    }
    Tensor barrierDummy(DT_INT32, {1, 1}, "barrierDummy");
    LOOP("RS", FunctionType::DYNAMIC_LOOP, dynRankId, LoopRange(0, rankSize, 1)) {
        auto shmemDataTile = View(shmemData, {1, 1, rowOut, col}, std::vector<SymbolicScalar>{dynRankId, 0, 0, 0});
        auto shmemSignalTile =
            View(shmemSignal, {1, 1, tileCount, 8}, std::vector<SymbolicScalar>{dynRankId, 0, 0, 0});
        auto inTile = View(in, {rowOut, col}, std::vector<SymbolicScalar>{dynRankId * rowOut, 0});
        auto dummy = ShmemPut(inTile, shmemDataTile, barrierDummy, tileCount, AtomicType::ADD);
        auto dummySignal = ShmemSignal(dummy, shmemSignalTile, AtomicType::ADD);

        IF (dynRankId == thisRank) {
            auto shmemDataLocal = View(shmemData, {1, 1, rowOut, col}, std::vector<SymbolicScalar>{thisRank, 0, 0, 0});
            auto shmemSignalLocal =
                View(shmemSignal, {1, 1, tileCount, 8}, std::vector<SymbolicScalar>{thisRank, 0, 0, 0});
            auto dummyLocal = WaitUntil(dummySignal, shmemSignalLocal, tileCount, hcclGroupIndex, rankSize);
            out = ShmemGet(dummyLocal, shmemDataLocal, in.GetDataType());
        }
    }
}

void ShmemAddAllReduce(Tensor &in, const char* group, Tensor &out)
{
    int32_t hcclGroupIndex = static_cast<int32_t>(CommGroupRecorder::GetInstance().Input(std::string(group)));
    int32_t row = in.GetShape(0);
    int32_t col = in.GetShape(1);
    ASSERT(row > 0 && col > 0) << "Invalid shape: row and col must be > 0, but got row=" << row << ", col=" << col;
    auto [rankSize, tileCount] = GetRankSizeAndTileCount();
    ASSERT((row % rankSize) == 0);
    const int32_t rowPerRank = row / rankSize;
    SymbolicScalar thisRank = GetHcclRankId(hcclGroupIndex);

    Shape shmDataShape = {rowPerRank, col};
    Shape shmSignalShape = {tileCount, 8};
    Tensor shmemData;
    Tensor shmemSignal;
    LOOP("CreateShmemTensor", FunctionType::DYNAMIC_LOOP, index, LoopRange(1)) {
        (void)index;
        shmemData = CreateShmemTensor(rankSize, hcclGroupIndex, in.GetDataType(), shmDataShape);
        shmemSignal = CreateShmemTensor(rankSize, hcclGroupIndex, DT_INT32, shmSignalShape);
    }

    Tensor dummySingal;
    Tensor dummyDarrier(DT_INT32, {1, 1}, "dummyDarrier");
    LOOP("LOCALMOVE", FunctionType::DYNAMIC_LOOP, index, LoopRange(1)) {
        (void)index;
        auto res = in; // 实际网络中使用计算OP替代
        auto shmemDataLocal = View(shmemData, 
            {1, rankSize, rowPerRank, col},  std::vector<SymbolicScalar>{thisRank, 0, 0, 0});
        auto dummy = ShmemPutUb2Gm(res, shmemDataLocal, dummyDarrier, tileCount, AtomicType::ADD);
        for (int i = 0; i < rankSize; ++i) {
            auto shmemSignalLocal = View(shmemSignal, 
                {1, 1, tileCount, 8}, std::vector<SymbolicScalar>{thisRank, i, 0, 0});
            dummySingal = ShmemSignal(dummy, shmemSignalLocal, AtomicType::ADD);
        }
    }

    LOOP("L0", FunctionType::DYNAMIC_LOOP, dynRankId, LoopRange(rankSize)) {
        IF (thisRank == dynRankId) {
            auto shmemDataLocal = View(shmemData, 
                {1, 1, rowPerRank, col},  std::vector<SymbolicScalar>{thisRank, thisRank, 0, 0});
            auto shmemSignalLocal = View(shmemSignal, 
                {1, 1, tileCount, 8}, std::vector<SymbolicScalar>{thisRank, thisRank, 0, 0});
            auto dummyLocal = WaitUntil(dummySingal, shmemSignalLocal, tileCount, hcclGroupIndex, rankSize);
            auto tmp = ShmemGetGm2Ub(dummyLocal, shmemDataLocal, in.GetDataType());
            TileShape::Current().SetVecTile(rowPerRank, col);
            Assemble(tmp, {thisRank * rowPerRank, 0}, out); // 实际网络中此处插入计算OP
        } ELSE {
            auto shmemDataLocal = View(shmemData,
                {1, 1, rowPerRank, col},  std::vector<SymbolicScalar>{thisRank, dynRankId, 0, 0});
            auto shmemDataRemote = View(shmemData,
                {1, 1, rowPerRank, col},  std::vector<SymbolicScalar>{dynRankId, dynRankId, 0, 0});
            auto shmemSignalRemote = View(shmemSignal,
                {1, 1, tileCount, 8}, std::vector<SymbolicScalar>{dynRankId, dynRankId, 0, 0});
            auto dummyPut = ShmemPut(shmemDataLocal, shmemDataRemote, dummySingal, tileCount, AtomicType::ADD);
            auto dummySignalRemote = ShmemSignal(dummyPut, shmemSignalRemote, AtomicType::ADD);
            auto dummtWait = WaitUntil(dummySignalRemote, shmemSignalRemote, tileCount, hcclGroupIndex, rankSize);
            auto tmp = ShmemGetGm2Ub(dummtWait, shmemDataRemote, in.GetDataType());
            TileShape::Current().SetVecTile(rowPerRank, col);
            Assemble(tmp, {dynRankId * rowPerRank, 0}, out); // 实际网络中此处插入计算OP
        }
    }
}
}   // namespace npu::tile_fwk::Distributed
