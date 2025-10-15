/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file shmem_expand_funcion.cpp
 * \brief
 */

#include "distributed_expand.h"
#include "distributed_common.h"

namespace npu::tile_fwk::Distributed {
constexpr uint16_t UB_BUFFER_BYTE_SIZE = 256;

void TiledShmemPut(Function &function, const TileShape &tileShape,
    const std::vector<std::shared_ptr<LogicalTensor>> &iOperand,
    const std::vector<std::shared_ptr<LogicalTensor>> &oOperand, const Operation &op)
{
    ASSERT(iOperand.size() == 3UL) << "TiledShmemPut iOperand size is not equal to 3";
    ASSERT(oOperand.size() == 1UL) << "TiledShmemSignal oOperand size is not equal to 1";
    std::shared_ptr<LogicalTensor> in = iOperand[0];
    std::shared_ptr<LogicalTensor> shmData = iOperand[1];
    std::shared_ptr<LogicalTensor> barrierDummy = iOperand[2]; // operand 2
    std::shared_ptr<LogicalTensor> dummy = oOperand[0];
    ASSERT(in->shape.size() == 2UL);
    ASSERT(shmData->shape.size() == 4UL);
    const int64_t oriRow = in->shape[0];
    const int64_t oriCol = in->shape[1];
    const auto tileRow = tileShape.GetDistTileRow();
    const auto tileCol = tileShape.GetDistTileCol();
    const int64_t rowStep = tileRow[0];
    const int64_t colStep = tileCol[0];
    AtomicType atomicType;
    op.GetAttr("AtomicType", atomicType);
    int tileIndex = 0;
    for (int64_t rowIdx = 0; rowIdx < oriRow; rowIdx += rowStep) {
        auto rowSize = std::min(oriRow - rowIdx, rowStep);
        for (int64_t colIdx = 0; colIdx < oriCol; colIdx += colStep) {
            auto colSize = std::min(oriCol - colIdx, colStep);
            ASSERT(rowSize > 0 && colSize > 0) << "tileShape is not valid";
            Shape shape = {rowSize, colSize};
            auto inTile = in->View(function, shape, {rowIdx, colIdx});
            auto shmDataTile = shmData->View(function, {1, 1, rowSize, colSize}, {0, 0, rowIdx, colIdx});
            auto dummyTile = dummy->View(function, {1, 1}, {tileIndex, 0});

            const uint16_t copyNum = UB_BUFFER_BYTE_SIZE / sizeof(in->Datatype());
            Shape bufferShape;
            if (copyNum >= rowSize * colSize) {
                bufferShape = {rowSize, colSize};
            } else if (copyNum >= colSize) {
                bufferShape = {(copyNum + colSize - 1) / colSize, colSize};
            } else {
                bufferShape = {1, copyNum};
            }
            auto ubTensor = std::make_shared<LogicalTensor>(function, in->Datatype(), bufferShape);

            auto& tileop = function.AddOperation("SHMEM_PUT", {inTile, shmDataTile, barrierDummy},
                {dummyTile, ubTensor});
            tileop.SetAttr("AtomicType", atomicType);
            tileIndex++;
        }
    }
}

void TiledShmemSignal(Function &function, const TileShape &tileShape,
    const std::vector<std::shared_ptr<LogicalTensor>> &iOperand,
    const std::vector<std::shared_ptr<LogicalTensor>> &oOperand, const Operation &op)
{
    ASSERT(iOperand.size() == 2UL) << "TiledShmemSignal iOperand size is not equal to 2";
    ASSERT(oOperand.size() == 1UL) << "TiledShmemSignal oOperand size is not equal to 1";
    std::shared_ptr<LogicalTensor> dummy = iOperand[0];
    std::shared_ptr<LogicalTensor> shmSignal = iOperand[1];
    std::shared_ptr<LogicalTensor> dummyOut = oOperand[0];
    ASSERT(shmSignal->shape.size() == 4UL);
    const auto tileRow = tileShape.GetDistTileRow();
    const auto tileCol = tileShape.GetDistTileCol();
    AtomicType atomicType;
    int64_t value;
    op.GetAttr("AtomicType", atomicType);
    op.GetAttr("Value", value);
    const int64_t tileLen = shmSignal->shape[3];
    const int64_t rowCount = tileRow[1] + (tileRow[2] == 0 ? 0 : 1);
    const int64_t colCount = tileCol[1] + (tileCol[2] == 0 ? 0 : 1);
    int tileIndex = 0;
    for (int64_t rowIdx = 0; rowIdx < rowCount; rowIdx++) {
        for (int64_t colIdx = 0; colIdx < colCount; colIdx++) {
            auto dummyTile = dummy->View(function, {1, 1}, {tileIndex, 0});
            auto shmSignalTile = shmSignal->View(function, {1, 1, 1, tileLen}, {0, 0, tileIndex, 0});
            auto dummyOutTile = dummyOut->View(function, {1, 1}, {tileIndex, 0});
            Shape ubShape = {tileLen};
            auto ubTensor = std::make_shared<LogicalTensor>(function, shmSignal->Datatype(), ubShape);

            auto& tileop = function.AddOperation("SHMEM_SIGNAL", {dummyTile, shmSignalTile}, {dummyOutTile, ubTensor});
            tileop.SetAttr(OpAttributeKey::dontTouch, true);
            tileop.SetAttr("Value", value);
            tileop.SetAttr("AtomicType", atomicType);
            tileIndex++;
        }
    }
}

void TiledShmemWaitUntil(Function &function, const TileShape &tileShape,
    const std::vector<std::shared_ptr<LogicalTensor>> &iOperand,
    const std::vector<std::shared_ptr<LogicalTensor>> &oOperand, const Operation &op)
{
    ASSERT(iOperand.size() == 2UL) << "TiledShmemWaitUntil iOperand size is not equal to 2";
    ASSERT(oOperand.size() == 1UL) << "TiledShmemWaitUntil oOperand size is not equal to 1";
    std::shared_ptr<LogicalTensor> dummyIn = iOperand[0];
    std::shared_ptr<LogicalTensor> shmSignal = iOperand[1];
    std::shared_ptr<LogicalTensor> dummy = oOperand[0];
    ASSERT(shmSignal->shape.size() == 4UL);
    const int64_t tileLen = shmSignal->shape[3];
    const auto tileRow = tileShape.GetDistTileRow();
    const auto tileCol = tileShape.GetDistTileCol();
    const int64_t rowCount = tileRow[1] + (tileRow[2] == 0 ? 0 : 1); // 1和2分别表示头快和尾块数
    const int64_t colCount = tileCol[1] + (tileCol[2] == 0 ? 0 : 1); // 1和2分别表示头快和尾块数
    int64_t value;
    std::vector<int64_t> extraAttrs;
    op.GetAttr("Value", value);
    op.GetAttr("AicpuOpParams", extraAttrs);
    int tileIndex = 0;
    for (int64_t rowIdx = 0; rowIdx < rowCount; rowIdx++) {
        for (int64_t colIdx = 0; colIdx < colCount; colIdx++) {
            auto dummyInTile = dummyIn->View(function, {1, 1}, {tileIndex, 0});
            auto shmSignalTile = shmSignal->View(function, {1, 1, 1, tileLen}, {0, 0, tileIndex, 0});
            auto dummyTile = dummy->View(function, {1, 1}, {tileIndex, 0});

            auto& tileop = function.AddOperation("SHMEM_WAIT_UNTIL", {dummyInTile, shmSignalTile}, {dummyTile});
            tileop.SetAttr("Value", value);
            tileop.SetAttr("AicpuOpParams", extraAttrs);
            tileIndex++;
        }
    }
}

void TiledShmemGet(Function &function, const TileShape &tileShape,
    const std::vector<std::shared_ptr<LogicalTensor>> &iOperand,
    const std::vector<std::shared_ptr<LogicalTensor>> &oOperand, const Operation &op)
{
    ASSERT(iOperand.size() == 2UL) << "TiledShmemGet iOperand size is not equal to 2";
    ASSERT(oOperand.size() == 1UL) << "TiledShmemGet oOperand size is not equal to 1";
    std::shared_ptr<LogicalTensor> dummy = iOperand[0];
    std::shared_ptr<LogicalTensor> shmData = iOperand[1];
    std::shared_ptr<LogicalTensor> out = oOperand[0];
    ASSERT(shmData->shape.size() == 4UL);
    ASSERT(out->shape.size() == 2UL);
    const int64_t oriRow = out->shape[0];
    const int64_t oriCol = out->shape[1];
    const auto tileRow = tileShape.GetDistTileRow();
    const auto tileCol = tileShape.GetDistTileCol();
    const int64_t rowStep = tileRow[0];
    const int64_t colStep = tileCol[0];
    AtomicType atomicType;
    op.GetAttr("AtomicType", atomicType);
    int tileIndex = 0;
    for (int64_t rowIdx = 0; rowIdx < oriRow; rowIdx += rowStep) {
        auto rowSize = std::min(oriRow - rowIdx, rowStep);
        for (int64_t colIdx = 0; colIdx < oriCol; colIdx += colStep) {
            auto colSize = std::min(oriCol - colIdx, colStep);
            ASSERT(rowSize > 0 && colSize > 0) << "tileShape is not valid";
            auto dummyTile = dummy->View(function, {1, 1}, {tileIndex, 0});
            auto shmDataTile = shmData->View(function, {1, 1, rowSize, colSize}, {0, 0, rowIdx, colIdx});
            Shape shape = {rowSize, colSize};
            auto outTile = out->View(function, shape, {rowIdx, colIdx});

            const uint16_t copyNum = UB_BUFFER_BYTE_SIZE / sizeof(out->Datatype());
            Shape bufferShape;
            if (copyNum >= rowSize * colSize) {
                bufferShape = {rowSize, colSize};
            } else if (copyNum >= colSize) {
                bufferShape = {(copyNum + colSize - 1) / colSize, colSize};
            } else {
                bufferShape = {1, copyNum};
            }
            auto ubTensor = std::make_shared<LogicalTensor>(function, out->Datatype(), bufferShape);
    
            auto& tileop = function.AddOperation("SHMEM_GET", {dummyTile, shmDataTile}, {outTile, ubTensor});
            tileop.SetAttr("AtomicType", atomicType);
            tileIndex++;
        }
    }
}

void TiledShmemClearSignal(Function &function, const TileShape &tileShape,
    const std::vector<std::shared_ptr<LogicalTensor>> &iOperand,
    const std::vector<std::shared_ptr<LogicalTensor>> &oOperand, const Operation &op)
{
    (void)op;
    ASSERT(iOperand.size() == 2UL) << "TiledShmemGet iOperand size is not equal to 2";
    ASSERT(oOperand.size() == 1UL) << "TiledShmemGet oOperand size is not equal to 1";
    std::shared_ptr<LogicalTensor> in = iOperand[0];
    std::shared_ptr<LogicalTensor> signal = iOperand[1];
    std::shared_ptr<LogicalTensor> dummy = oOperand[0];
    ASSERT(signal->shape.size() == 4UL);
    const int64_t oriRow = in->shape[0];
    const int64_t oriCol = in->shape[1];
    const int64_t tileLen = signal->shape[3];
    const auto tileRow = tileShape.GetDistTileRow();
    const auto tileCol = tileShape.GetDistTileCol();
    const int64_t rowStep = tileRow[0];
    const int64_t colStep = tileCol[0];
    int tileIndex = 0;
    for (int64_t rowIdx = 0; rowIdx < oriRow; rowIdx += rowStep) {
        auto rowSize = std::min(oriRow - rowIdx, rowStep);
        for (int64_t colIdx = 0; colIdx < oriCol; colIdx += colStep) {
            auto colSize = std::min(oriCol - colIdx, colStep);
            auto signalTile = signal->View(function, {1, 1, 1, tileLen}, {0, 0, tileIndex, 0});
            auto inTile = in->View(function, {rowSize, colSize}, {rowIdx, colIdx});
            Shape ubShape = {tileLen};
            auto ubTensor = std::make_shared<LogicalTensor>(function, signal->Datatype(), ubShape);
            function.AddOperation("SHMEM_CLEAR_SIGNAL", {signal, inTile}, {dummy, ubTensor});
            tileIndex++;
        }
    }
}

Shape GetReduceUbShape(int64_t rowSize, int64_t colSize, DataType dType, bool fp32Mode)
{
    Shape ubShape;
    if (fp32Mode) {
        ubShape = {rowSize * colSize +                                              // copy需要的ub大小
            rowSize * colSize * (int64_t)(BytesOf(DT_FP32) / BytesOf(dType)) +      // 存放fp32计算结果的ub大小
            (int64_t)(256 / BytesOf(dType))};                                       // fp32计算需要的额外
    } else {
        ubShape = {2 * rowSize * colSize};  // copy 和 sum 需要的ub大小
    }
    return ubShape;
}

void TiledShmemReduce(Function &function, const TileShape &tileShape,
    const std::vector<std::shared_ptr<LogicalTensor>> &iOperand,
    const std::vector<std::shared_ptr<LogicalTensor>> &oOperand, const Operation &op)
{
    (void)op;
    ASSERT(iOperand.size() == 3UL) << "TiledShmemGet iOperand size is not equal to 3";
    ASSERT(oOperand.size() == 1UL) << "TiledShmemGet oOperand size is not equal to 1";
    const auto in = iOperand[0];
    const auto shmData = iOperand[1];
    const auto dummy = iOperand[2];
    const auto out = oOperand[0];
    ASSERT(in->shape.size() == 2UL);
    ASSERT(shmData->shape.size() == 4UL);
    ASSERT(out->shape.size() == 2UL);
    const int64_t oriRow = out->shape[0];
    const int64_t oriCol = out->shape[1];
    const auto tileRow = tileShape.GetDistTileRow();
    const auto tileCol = tileShape.GetDistTileCol();
    const int64_t rowStep = tileRow[0];
    const int64_t colStep = tileCol[0];
    bool fp32Mode;
    op.GetAttr("FP32Mode", fp32Mode);
    std::string extraTemplateParam = fp32Mode ? "true" : "false";
    int tileIndex = 0;
    for (int64_t rowIdx = 0; rowIdx < oriRow; rowIdx += rowStep) {
        auto rowSize = std::min(oriRow - rowIdx, rowStep);
        for (int64_t colIdx = 0; colIdx < oriCol; colIdx += colStep) {
            auto colSize = std::min(oriCol - colIdx, colStep);
            auto inTile = in->View(function, {rowSize, colSize}, {rowIdx, colIdx});
            auto dummyTile = dummy->View(function, {1, 1}, {tileIndex, 0});
            auto outTile = out->View(function, {rowSize, colSize}, {rowIdx, colIdx});
            Shape ubShape = GetReduceUbShape(rowSize, colSize, out->Datatype(), fp32Mode);
            auto ubTensor = std::make_shared<LogicalTensor>(function, out->Datatype(), ubShape);
            auto& tileop = function.AddOperation("SHMEM_REDUCE", {inTile, shmData, dummyTile}, {outTile, ubTensor});
            tileop.SetAttr("extraTemplateParam", extraTemplateParam);
            tileIndex++;
        }
    }
}

void TiledShmemBindTensor(Function &function, const TileShape &tileShape,
    const std::vector<std::shared_ptr<LogicalTensor>> &iOperand,
    const std::vector<std::shared_ptr<LogicalTensor>> &oOperand, const Operation &op)
{
    (void)iOperand;
    (void)tileShape;
    auto &oper = function.AddOperation("BIND_TENSOR", {}, oOperand);
    SymbolicScalar bindTensor;
    if (op.HasAttr(OpAttributeKey::bindTensor)) {
        bindTensor = op.GetSymbolicScalarAttribute(OpAttributeKey::bindTensor);
        oper.SetAttribute(OpAttributeKey::bindTensor, bindTensor);
    }
}

}   // namespace npu::tile_fwk::Distributed
