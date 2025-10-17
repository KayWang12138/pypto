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
namespace {
constexpr uint16_t UB_BUFFER_BYTE_SIZE = 256;

void CreateTileOp(const TileShape& tileShape,
    const std::function<void(int32_t, int32_t, int32_t, int32_t, int32_t)>& callback)
{
    const auto& tileRow = tileShape.GetDistTileRow();
    const auto& tileCol = tileShape.GetDistTileCol();
    int32_t rowCount = tileRow[1] + (tileRow[2] == 0 ? 0 : 1);
    int32_t colCount = tileCol[1] + (tileCol[2] == 0 ? 0 : 1);

    int32_t tileIndex = 0;
    for (int32_t rowIndex = 0; rowIndex < rowCount; rowIndex++) {
        int32_t rowShape = ((tileRow[2] != 0) && (rowIndex == rowCount - 1)) ? tileRow[2] : tileRow[0];
        for (int32_t colIndex = 0; colIndex < colCount; colIndex++) {
            int32_t colShape = ((tileCol[2] != 0) && (colIndex == colCount - 1)) ? tileCol[2] : tileCol[0];
            callback(tileIndex, rowIndex * tileRow[0], colIndex * tileCol[0], rowShape, colShape);
            tileIndex++;
        }
    }
}

LogicalTensorPtr CreateAdaptiveUbTensor(Function& function, const Shape& shape, DataType dataType)
{
    uint16_t copyNum = UB_BUFFER_BYTE_SIZE / sizeof(dataType);
    Shape bufferShape;
    if (copyNum >= shape[0] * shape[1]) {
        bufferShape = {shape[0], shape[1]};
    } else if (copyNum >= shape[1]) {
        bufferShape = {(copyNum + shape[1] - 1) / shape[1], shape[1]};
    } else {
        bufferShape = {1, copyNum};
    }
    return std::make_shared<LogicalTensor>(function, dataType, bufferShape);
}
} // namespace

void TiledShmemPut(Function& function, const TileShape& tileShape,
    const std::vector<std::shared_ptr<LogicalTensor>>& iOperand,
    const std::vector<std::shared_ptr<LogicalTensor>>& oOperand, const Operation& op)
{
    ASSERT(iOperand.size() == 3UL) << "TiledShmemPut iOperand size is not equal to 3";
    ASSERT(oOperand.size() == 1UL) << "TiledShmemPut oOperand size is not equal to 1";
    auto in = iOperand[0];
    auto shmData = iOperand[1];
    auto barrierDummy = iOperand[2]; // operand 2
    auto dummy = oOperand[0];

    AtomicType atomicType;
    op.GetAttr("AtomicType", atomicType);

    CreateTileOp(tileShape,
        [&](int32_t tileIndex, int32_t rowOffset, int32_t colOffset, int32_t rowShape, int32_t colShape) {
            Shape shape = {rowShape, colShape};
            auto inTile = in->View(function, shape, {rowOffset, colOffset});
            auto shmDataTile = shmData->View(function, {1, 1, rowShape, colShape}, {0, 0, rowOffset, colOffset});
            auto dummyTile = dummy->View(function, {1, 1}, {tileIndex, 0});
            auto ubTensor = CreateAdaptiveUbTensor(function, shape, in->Datatype());

            auto& tileop = function.AddOperation("SHMEM_PUT", {inTile, shmDataTile, barrierDummy},
                {dummyTile, ubTensor});
            tileop.SetAttr("AtomicType", atomicType);
        });
}

void TiledShmemSignal(Function& function, const TileShape& tileShape,
    const std::vector<std::shared_ptr<LogicalTensor>>& iOperand,
    const std::vector<std::shared_ptr<LogicalTensor>>& oOperand, const Operation& op)
{
    ASSERT(iOperand.size() == 2UL) << "TiledShmemSignal iOperand size is not equal to 2";
    ASSERT(oOperand.size() == 1UL) << "TiledShmemSignal oOperand size is not equal to 1";
    auto dummy = iOperand[0];
    auto shmSignal = iOperand[1];
    auto dummyOut = oOperand[0];

    ASSERT(shmSignal->shape.size() == 4UL);
    int64_t tileLen = shmSignal->shape[3];

    AtomicType atomicType;
    int64_t value;
    op.GetAttr("AtomicType", atomicType);
    op.GetAttr("Value", value);

    CreateTileOp(tileShape,
        [&](int32_t tileIndex, int32_t rowOffset, int32_t colOffset, int32_t rowShape, int32_t colShape) {
            (void)rowOffset;
            (void)colOffset;
            (void)rowShape;
            (void)colShape;

            auto dummyTile = dummy->View(function, {1, 1}, {tileIndex, 0});
            auto shmSignalTile = shmSignal->View(function, {1, 1, 1, tileLen}, {0, 0, tileIndex, 0});
            auto dummyOutTile = dummyOut->View(function, {1, 1}, {tileIndex, 0});
            auto ubTensor = std::make_shared<LogicalTensor>(function, shmSignal->Datatype(), Shape{tileLen});

            auto& tileop = function.AddOperation("SHMEM_SIGNAL", {dummyTile, shmSignalTile}, {dummyOutTile, ubTensor});
            tileop.SetAttr(OpAttributeKey::dontTouch, true);
            tileop.SetAttr("Value", value);
            tileop.SetAttr("AtomicType", atomicType);
        });
}

void TiledShmemWaitUntil(Function& function, const TileShape& tileShape,
    const std::vector<std::shared_ptr<LogicalTensor>>& iOperand,
    const std::vector<std::shared_ptr<LogicalTensor>>& oOperand, const Operation& op)
{
    ASSERT(iOperand.size() == 2UL) << "TiledShmemWaitUntil iOperand size is not equal to 2";
    ASSERT(oOperand.size() == 1UL) << "TiledShmemWaitUntil oOperand size is not equal to 1";
    auto dummyIn = iOperand[0];
    auto shmSignal = iOperand[1];
    auto dummy = oOperand[0];

    ASSERT(shmSignal->shape.size() == 4UL);
    int64_t tileLen = shmSignal->shape[3];

    int64_t value;
    std::vector<int64_t> extraAttrs;
    op.GetAttr("Value", value);
    op.GetAttr("AicpuOpParams", extraAttrs);

    CreateTileOp(tileShape,
        [&](int32_t tileIndex, int32_t rowOffset, int32_t colOffset, int32_t rowShape, int32_t colShape) {
            (void)rowOffset;
            (void)colOffset;
            (void)rowShape;
            (void)colShape;

            auto dummyInTile = dummyIn->View(function, {1, 1}, {tileIndex, 0});
            auto shmSignalTile = shmSignal->View(function, {1, 1, 1, tileLen}, {0, 0, tileIndex, 0});
            auto dummyTile = dummy->View(function, {1, 1}, {tileIndex, 0});

            auto& tileop = function.AddOperation("SHMEM_WAIT_UNTIL", {dummyInTile, shmSignalTile}, {dummyTile});
            tileop.SetAttr("Value", value);
            tileop.SetAttr("AicpuOpParams", extraAttrs);
        });
}

void TiledShmemGet(Function& function, const TileShape& tileShape,
    const std::vector<std::shared_ptr<LogicalTensor>>& iOperand,
    const std::vector<std::shared_ptr<LogicalTensor>>& oOperand, const Operation& op)
{
    ASSERT(iOperand.size() == 2UL) << "TiledShmemGet iOperand size is not equal to 2";
    ASSERT(oOperand.size() == 1UL) << "TiledShmemGet oOperand size is not equal to 1";
    auto dummy = iOperand[0];
    auto shmData = iOperand[1];
    auto out = oOperand[0];

    AtomicType atomicType;
    op.GetAttr("AtomicType", atomicType);

    CreateTileOp(tileShape,
        [&](int32_t tileIndex, int32_t rowOffset, int32_t colOffset, int32_t rowShape, int32_t colShape) {
            Shape shape = {rowShape, colShape};
            auto dummyTile = dummy->View(function, {1, 1}, {tileIndex, 0});
            auto shmDataTile = shmData->View(function, {1, 1, rowShape, colShape}, {0, 0, rowOffset, colOffset});
            auto outTile = out->View(function, shape, {rowOffset, colOffset});
            auto ubTensor = CreateAdaptiveUbTensor(function, shape, out->Datatype());

            auto& tileop = function.AddOperation("SHMEM_GET", {dummyTile, shmDataTile}, {outTile, ubTensor});
            tileop.SetAttr("AtomicType", atomicType);
        });
}

void TiledShmemClearSignal(Function& function, const TileShape& tileShape,
    const std::vector<std::shared_ptr<LogicalTensor>>& iOperand,
    const std::vector<std::shared_ptr<LogicalTensor>>& oOperand, const Operation& op)
{
    (void)op;

    ASSERT(iOperand.size() == 2UL) << "TiledShmemClearSignal iOperand size is not equal to 2";
    ASSERT(oOperand.size() == 1UL) << "TiledShmemClearSignal oOperand size is not equal to 1";
    auto in = iOperand[0];
    auto signal = iOperand[1];
    auto dummy = oOperand[0];

    ASSERT(signal->shape.size() == 4UL);
    int64_t tileLen = signal->shape[3];

    CreateTileOp(tileShape,
        [&](int32_t tileIndex, int32_t rowOffset, int32_t colOffset, int32_t rowShape, int32_t colShape) {
            auto signalTile = signal->View(function, {1, 1, 1, tileLen}, {0, 0, tileIndex, 0});
            auto inTile = in->View(function, {rowShape, colShape}, {rowOffset, colOffset});
            auto ubTensor = std::make_shared<LogicalTensor>(function, signal->Datatype(), Shape{tileLen});
            function.AddOperation("SHMEM_CLEAR_SIGNAL", {signalTile, inTile}, {dummy, ubTensor});
        });
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

void TiledShmemReduce(Function& function, const TileShape& tileShape,
    const std::vector<std::shared_ptr<LogicalTensor>>& iOperand,
    const std::vector<std::shared_ptr<LogicalTensor>>& oOperand, const Operation& op)
{
    ASSERT(iOperand.size() == 3UL) << "TiledShmemReduce iOperand size is not equal to 3";
    ASSERT(oOperand.size() == 1UL) << "TiledShmemReduce oOperand size is not equal to 1";
    auto in = iOperand[0];
    auto shmData = iOperand[1];
    auto dummy = iOperand[2];
    auto out = oOperand[0];

    bool fp32Mode;
    op.GetAttr("FP32Mode", fp32Mode);
    std::string extraTemplateParam = fp32Mode ? "true" : "false";

    CreateTileOp(tileShape,
        [&](int32_t tileIndex, int32_t rowOffset, int32_t colOffset, int32_t rowShape, int32_t colShape) {
            auto inTile = in->View(function, {rowShape, colShape}, {rowOffset, colOffset});
            auto dummyTile = dummy->View(function, {1, 1}, {tileIndex, 0});
            auto outTile = out->View(function, {rowShape, colShape}, {rowOffset, colOffset});
            Shape ubShape = GetReduceUbShape(rowShape, colShape, out->Datatype(), fp32Mode);
            auto ubTensor = std::make_shared<LogicalTensor>(function, out->Datatype(), ubShape);

            auto& tileop = function.AddOperation("SHMEM_REDUCE", {inTile, shmData, dummyTile}, {outTile, ubTensor});
            tileop.SetAttr("extraTemplateParam", extraTemplateParam);
        });
}

void TiledShmemBindTensor(Function& function, const TileShape& tileShape,
    const std::vector<std::shared_ptr<LogicalTensor>>& iOperand,
    const std::vector<std::shared_ptr<LogicalTensor>>& oOperand, const Operation& op)
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
