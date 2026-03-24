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
#include "tilefwk/distributed_communicator.h"
#include "interface/tensor/logical_tensor.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"
#include "interface/utils/common.h"
#include "interface/utils/log.h"

namespace npu::tile_fwk::Distributed {

void ValidateGroup(const char* group)
{
    ASSERT(group != nullptr) << "\"group\" cannot be nullptr";
    int32_t groupLen = std::strlen(group);
    ASSERT((groupLen >= 1) && (groupLen < 128)) << "The length of \"group\" only supports [1, 128), but got "
        << groupLen;
}

void ValidateTypeAndShape(const Tensor& tensor, const DataType expectedType, const Shape expectedShape)
{
    ASSERT(tensor.GetDataType() == expectedType);
    ASSERT(tensor.GetShape() == expectedShape);
}

void ValidateTilingSize(const VecTile& vecTile, const Tensor& in, int32_t worldSize)
{
    int32_t expectedTileSize = in.GetShape().size();
    ASSERT(expectedTileSize == static_cast<int32_t>(vecTile.size())) <<
        "Invalid dim of tile shape: dim of tile shape must be equal to " << std::to_string(expectedTileSize) << ".";
    ASSERT(std::all_of(vecTile.tile.begin(), vecTile.tile.begin() + expectedTileSize, [](int64_t v) { return v > 0;}))
        << "Invalid vecTile set: each element of the tileSize must be > 0";
    ASSERT([&](){
        for (int32_t i = 0; i < expectedTileSize; ++i) {
            if (vecTile[i] > in.GetShape(i)) {
                return false;
            }
        }
        return true;
    }()) << "Invalid vecTile set: tile size must be <= input shape for each dimension";
    int32_t tileRowShape = vecTile[0];
    int32_t tileColShape = vecTile[1];
    int32_t tileRowNum = in.GetShape(0) / tileRowShape + (in.GetShape(0) % tileRowShape == 0 ? 0 : 1);
    int32_t tileColNum = in.GetShape(1) / tileColShape + (in.GetShape(1) % tileColShape == 0 ? 0 : 1);
    ASSERT(worldSize > 0) << "WorldSize is invalid, worldSize should be less than 0, but got " << worldSize;
    ASSERT(tileRowNum * tileColNum <= MAX_TILE_NUM / worldSize) <<
        "TotalTileNum is invalid, totalTileNum shoule be less than " << MAX_TILE_NUM / worldSize << ", but got " << tileRowNum * tileColNum;
}

void ValidateParams(const Tensor& predToken, const Tensor& in, const Tensor& out, Shape shmemDataShape, DataType shmemDataType,
    bool checkShapeMatch = false, bool validateType = false, const std::unordered_set<DataType>& allowedTypes = {}) 
{
    ASSERT(predToken.GetShape().size() == 2UL) << "Invalid dimensional: PredToken dimensional must be 2, but got dimensional=" << predToken.GetShape().size(); 
    int32_t predRow = predToken.GetShape(0);
    int32_t predCol = predToken.GetShape(1);
    ASSERT(predRow > 0 && predCol > 0) << "PredToken parameter error - the 'row' and 'col' dimensional of the input tensor must be greater than 0, "
        << "but got row=" << predRow << ", col=" << predCol;
    ASSERT(in.GetShape().size() == 2UL) << "Invalid dimensional: Input dimensional must be 2, but got dimensional=" << in.GetShape().size();
    ASSERT(out.GetShape().size() == 2UL) << "Invalid dimensional: Output dimensional must be 2, but got dimensional=" << out.GetShape().size();
    ASSERT(out.GetDataType() == in.GetDataType()) << "The data type of \"out\" must be consistent with that of \"in\", "
        << "but the data type of \"out\" is "<< DataType2String(out.GetDataType()) << " and the data type of \"in\" is "
        << DataType2String(in.GetDataType()) << ".";
    ASSERT(in.Format() == out.Format()) << "Output tensor format dose not match input tensor fromat. "
        << "in format: " << std::to_string(in.Format())
        << "out format: " << std::to_string(out.Format()) << ".";
    int32_t inRow = in.GetShape(0);
    int32_t inCol = in.GetShape(1);
    int32_t outRow = out.GetShape(0);
    int32_t outCol = out.GetShape(1);
    ASSERT((in.Format() != TileOpFormat::TILEOP_NZ) && (out.Format() != TileOpFormat::TILEOP_NZ)) << "NZ not supported.";
    ASSERT(inRow > 0 && inCol > 0) << "Input parameter error - the 'row' and 'col' dimensional of the input tensor must be greater than 0, "
        << "but got row=" << inRow << ", col=" << inCol;
    if (checkShapeMatch) {
        ASSERT((inRow == outRow) && (inCol == outCol)) << "Shape mismatch: Input and output dimensions must be the same, but got "
        << "Input shape: (" << inRow << "," << inCol << "), Output shape: (" << outRow << "," << outCol << ").";
    }
    if (validateType) {
        std::ostringstream oss;
        oss << "[";
        bool first = true;
        for (const auto& dtype : allowedTypes) {
            if (!first) {
                oss << ", ";
            }
            oss << DataType2CCEStr(dtype);
            first = false;
        }
        oss << "]";
        ASSERT(allowedTypes.count(in.GetDataType())) << "Invalid data type for input tensor. Expected: " << oss.str() <<
        ", but got: " << DataType2CCEStr(in.GetDataType());
    }
    int64_t shmemDataEleNum =
        std::accumulate(shmemDataShape.begin() + 1, shmemDataShape.end(), 1, std::multiplies<int64_t>());
    int64_t shmemSignalEleNum = shmemDataShape[0] * MAX_TILE_NUM * SHMEM_SIGNAL_STRIDE;
    uint64_t shmemSize = shmemDataEleNum * BytesOf(shmemDataType) + shmemSignalEleNum * BytesOf(DT_INT32);
    const uint64_t winSize = 1024 * 1024 * 200;
    ASSERT(shmemSize < winSize) << "Exceeds winSize limit. Maximum allowed: " << winSize << ", got: " << shmemSize;
}

template <typename CommunicatorT, typename PutChunkFn>
void OneShotAllReduceGroupedScatterImpl(
    const Tensor& predToken, const Tensor& in, Tensor& shmemData, CommunicatorT& comm, PutChunkFn&& putChunkFn)
{
    int32_t col = in.GetShape(1);
    for (uint32_t dynRankId = 0; dynRankId < comm.WorldSize(); ++dynRankId) {
        for (uint32_t chunkId = 0; chunkId < comm.PayloadChunkCount(); ++chunkId) {
            int32_t chunkRow = comm.ChunkStartRow(chunkId);
            int32_t chunkRows = comm.ChunkRows(chunkId);
            auto inChunk = View(in, {chunkRows, col}, std::vector<SymbolicScalar>{chunkRow, 0});
            auto dynRankView = View(shmemData, {1, 1, chunkRows, col},
                std::vector<SymbolicScalar>{dynRankId, 0, chunkRow, 0});
            putChunkFn(predToken, inChunk, dynRankView, dynRankId, chunkId);
        }
    }
}

void OneShotAllReduceCoarseScatterImpl(
    const Tensor& predToken, const Tensor& in, Tensor& shmemData, OneShotCommunicatorV2& comm)
{
    int32_t row = in.GetShape(0);
    int32_t col = in.GetShape(1);

    for (uint32_t dynRankId = 0; dynRankId < comm.WorldSize(); ++dynRankId) {
        auto dynRankView = View(shmemData, {1, 1, row, col},
            std::vector<SymbolicScalar>{dynRankId, 0, 0, 0});
        comm.Put(predToken, in, dynRankView, dynRankId, AtomicType::ADD);
    }
}

Tensor ShmemPut(const Tensor& predToken, const Tensor& in, const Tensor& shmemData, AtomicType atomicType)
{
    auto &function = *Program::GetInstance().GetCurrentFunction();
    auto out = std::make_shared<LogicalTensor>(function, DT_INT32, predToken.GetShape());
    auto &op = function.AddOperation(Opcode::OP_SHMEM_PUT,
        {predToken.GetStorage(), in.GetStorage(), shmemData.GetStorage()}, {out});
    DistOpAttr distOpAttr;
    distOpAttr.atomicType = atomicType;
    op.SetAttr(OpAttributeKey::distOpAttr, distOpAttr);
    return out;
}

Tensor ShmemPutUb2Gm(const Tensor &in, const Tensor &shmemDataTile, const Tensor &barrierDummy,
 	AtomicType atomicType)
{
    auto &function = *Program::GetInstance().GetCurrentFunction();
    auto dummy = std::make_shared<LogicalTensor>(function, DT_INT32, barrierDummy.GetShape());
    auto &op = function.AddOperation(Opcode::OP_SHMEM_PUT_UB2GM,
        {in.GetStorage(), shmemDataTile.GetStorage(), barrierDummy.GetStorage()}, {dummy});
    DistOpAttr distOpAttr;
    distOpAttr.atomicType = atomicType;
    op.SetAttr(OpAttributeKey::distOpAttr, distOpAttr);
    return dummy;
}

Tensor ShmemSignal(const Tensor& predToken, const Tensor& shmemSignal, AtomicType atomicType,
    const std::vector<int64_t>* effectiveSignalTrailingDims)
{
    auto &function = *Program::GetInstance().GetCurrentFunction();
    auto out = std::make_shared<LogicalTensor>(function, DT_INT32, predToken.GetShape());
    auto &op = function.AddOperation(Opcode::OP_SHMEM_SIGNAL, {predToken.GetStorage(), shmemSignal.GetStorage()},
        {out});
    DistOpAttr distOpAttr;
    distOpAttr.signalValue = 1;
    distOpAttr.atomicType = atomicType;
    distOpAttr.signalStride = SHMEM_SIGNAL_STRIDE;
    if (effectiveSignalTrailingDims && effectiveSignalTrailingDims->size() >= 2) {
        distOpAttr.effectiveSignalTrailingDims = *effectiveSignalTrailingDims;
    }
    op.SetAttr(OpAttributeKey::distOpAttr, distOpAttr);
    return out;
}

Tensor ShmemGet(const Tensor& predToken, const Tensor& shmemData, DataType nonShmemDataType, AtomicType atomicType)
{
    if (nonShmemDataType == DT_BOTTOM) {
        nonShmemDataType = shmemData.GetDataType();
    }
    auto &function = *Program::GetInstance().GetCurrentFunction();
    const auto& s = shmemData.GetShape();
    Shape shape = (s.size() == 3)
        ? Shape{s[1], s[2]} : Shape{s[2], s[3]};
    auto out = std::make_shared<LogicalTensor>(function, nonShmemDataType, shape, shmemData.Format());
    auto &op = function.AddOperation(Opcode::OP_SHMEM_GET, {predToken.GetStorage(), shmemData.GetStorage()},
        {out});
    DistOpAttr distOpAttr;
    distOpAttr.atomicType = atomicType;
    op.SetAttr(OpAttributeKey::distOpAttr, distOpAttr);
    return out;
}

Tensor ShmemGetGm2Ub(const Tensor &dummy, const Tensor &shmemDataTile, DataType nonShmemDataType, AtomicType atomicType)
{
    if (nonShmemDataType == DT_BOTTOM) {
        nonShmemDataType = shmemDataTile.GetDataType();
    }
    auto &function = *Program::GetInstance().GetCurrentFunction();
    Shape shape = {shmemDataTile.GetShape()[2], shmemDataTile.GetShape()[3]};
    auto tempOutTile = std::make_shared<LogicalTensor>(function, nonShmemDataType, shape);
    auto &op = function.AddOperation(Opcode::OP_SHMEM_GET_GM2UB, {dummy.GetStorage(), shmemDataTile.GetStorage()},
        {tempOutTile});
    op.SetOpAttribute(std::make_shared<CopyOpAttribute>(OpImmediate::Specified({0, 0}), MEM_UB,
        OpImmediate::Specified({shmemDataTile.GetShape()[2], shmemDataTile.GetShape()[3]}), 
        OpImmediate::Specified({tempOutTile->shape[0], tempOutTile->shape[1]}),
        OpImmediate::Specified(std::vector<SymbolicScalar>{shmemDataTile.GetValidShape()[2], shmemDataTile.GetValidShape()[3]})));
    function.UpdateTensorDataUsage(op);
    DistOpAttr distOpAttr;
    distOpAttr.atomicType = atomicType;
    op.SetAttr(OpAttributeKey::distOpAttr, distOpAttr);
    return tempOutTile;
}

Tensor WaitUntil(const Tensor& predToken, const Tensor& shmemSignal, int32_t expectedSum, bool resetSignal,
    const std::vector<int64_t>* effectiveSignalTrailingDims)
{
    auto &function = *Program::GetInstance().GetCurrentFunction();
    auto out = std::make_shared<LogicalTensor>(function, DT_INT32, predToken.GetShape());
    auto &op = function.AddOperation(Opcode::OP_SHMEM_WAIT_UNTIL, {predToken.GetStorage(), shmemSignal.GetStorage()},
        {out});
    std::vector<int64_t> param = {static_cast<int64_t>(expectedSum), static_cast<int64_t>(SHMEM_SIGNAL_STRIDE), static_cast<int64_t>(resetSignal)};
    DistOpAttr distOpAttr;
    distOpAttr.aicpuOpParams = param;
    if (effectiveSignalTrailingDims && effectiveSignalTrailingDims->size() >= 2) {
        distOpAttr.effectiveSignalTrailingDims = *effectiveSignalTrailingDims;
    }
    op.SetAttr(OpAttributeKey::distOpAttr, distOpAttr);
    return out;
}

void ShmemReduce(const Tensor& in, const Tensor& shmData, const Tensor& dummy, const Tensor& out)
{
    auto &function = *Program::GetInstance().GetCurrentFunction();
    auto &op = function.AddOperation(Opcode::OP_SHMEM_REDUCE,
        {in.GetStorage(), shmData.GetStorage(), dummy.GetStorage()}, {out.GetStorage()});
    DistOpAttr distOpAttr;
    // fp16 和 bf16 做reduce计算，默认转化为fp32
    if ((in.GetDataType() == DT_FP16) || (in.GetDataType() == DT_BF16)) {
        distOpAttr.fp32Mode = true;
    } else {
        distOpAttr.fp32Mode = false;
    }
    op.SetAttr(OpAttributeKey::distOpAttr, distOpAttr);
}

void CreateShmemData(const char* group, int64_t worldSize, DataType dataType,
    const Shape &shape, Tensor &shmemTensor, uint64_t memType)
{
    auto &function = *Program::GetInstance().GetCurrentFunction();
    int32_t hcclGroupIndex = static_cast<int>(CommGroupRecorder::GetInstance().Input(std::string(group)));
    Shape shmemShape{worldSize};
    shmemShape.insert(shmemShape.end(), shape.begin(), shape.end());
    auto shmemTensorInner = std::make_shared<LogicalTensor>(function, dataType, shmemShape);
    shmemTensor = shmemTensorInner;
    Program::GetInstance().GetTensorSlotManager()->TensorWrite(shmemTensor, SlotProperty::SHMEM_TENSOR);
    auto &op = function.AddOperation(Opcode::OP_BIND_TENSOR, {}, {shmemTensorInner});
    op.SetAttribute(OpAttributeKey::bindTensor, BindTensor(hcclGroupIndex, memType,
        AlignUp(BytesOf(dataType) * std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int64_t>()), 512)));
}

void CreateShmemSignal(const char* group, Tensor& shmemData, Tensor& shmemSignal)
{
    auto &function = *Program::GetInstance().GetCurrentFunction();
    int32_t hcclGroupIndex = static_cast<int>(CommGroupRecorder::GetInstance().Input(std::string(group)));
    int64_t worldSize = shmemData.GetShape(0);
    Shape shmemShape{worldSize, worldSize};
    Shape shmemDataShape;
    shmemDataShape.assign(shmemData.GetShape().begin() + 1, shmemData.GetShape().end());
    shmemShape.insert(shmemShape.end(), shmemDataShape.begin(), shmemDataShape.end());
    auto shmemTensorInner = std::make_shared<LogicalTensor>(function, DataType::DT_INT32, shmemShape);
    shmemSignal = shmemTensorInner;
    Program::GetInstance().GetTensorSlotManager()->TensorWrite(shmemSignal, SlotProperty::SHMEM_TENSOR);
    auto &op = function.AddOperation(Opcode::OP_BIND_TENSOR, {}, {shmemTensorInner});
    op.SetAttribute(OpAttributeKey::bindTensor, BindTensor(hcclGroupIndex, 0,
        BytesOf(DataType::DT_INT32) * worldSize * SHMEM_SIGNAL_STRIDE * MAX_TILE_NUM));
}

// Compact coarse signal domain: shape {W,W,1,1,1}. Per proposal Section 14.5,
// logical bytes = 4*W*W; BindTensor reservation kept at 32768*W for compatibility.
void CreateShmemSignalLight(const char* group, int64_t worldSize, Tensor& shmemSignal)
{
    auto &function = *Program::GetInstance().GetCurrentFunction();
    int32_t hcclGroupIndex = static_cast<int>(CommGroupRecorder::GetInstance().Input(std::string(group)));
    Shape shmemShape{worldSize, worldSize, 1, 1, 1};
    auto shmemTensorInner = std::make_shared<LogicalTensor>(function, DataType::DT_INT32, shmemShape);
    shmemSignal = shmemTensorInner;
    Program::GetInstance().GetTensorSlotManager()->TensorWrite(shmemSignal, SlotProperty::SHMEM_TENSOR);
    auto &op = function.AddOperation(Opcode::OP_BIND_TENSOR, {}, {shmemTensorInner});
    op.SetAttribute(OpAttributeKey::bindTensor, BindTensor(hcclGroupIndex, 0,
        BytesOf(DataType::DT_INT32) * worldSize * SHMEM_SIGNAL_STRIDE * MAX_TILE_NUM));
}

// Compact grouped signal domain: shape {W,W,1,S,1}. Per proposal Section 14,
// S = ceil(C_payload/k); logical bytes = 4*W*W*S.
void CreateShmemSignalGroupedLight(const char* group, int64_t worldSize, int64_t signalGroupCount,
    Tensor& shmemSignal)
{
    ASSERT(signalGroupCount > 0) << "signalGroupCount must be > 0";
    auto &function = *Program::GetInstance().GetCurrentFunction();
    int32_t hcclGroupIndex = static_cast<int>(CommGroupRecorder::GetInstance().Input(std::string(group)));
    Shape shmemShape{worldSize, worldSize, 1, signalGroupCount, 1};
    auto shmemTensorInner = std::make_shared<LogicalTensor>(function, DataType::DT_INT32, shmemShape);
    shmemSignal = shmemTensorInner;
    Program::GetInstance().GetTensorSlotManager()->TensorWrite(shmemSignal, SlotProperty::SHMEM_TENSOR);
    auto &op = function.AddOperation(Opcode::OP_BIND_TENSOR, {}, {shmemTensorInner});
    op.SetAttribute(OpAttributeKey::bindTensor, BindTensor(hcclGroupIndex, 0,
        BytesOf(DataType::DT_INT32) * worldSize * SHMEM_SIGNAL_STRIDE * MAX_TILE_NUM));
}

Tensor ShmemBarrier(const Tensor& predToken, Tensor& shmemSignal, const char* group, uint32_t worldSize)
{
    ValidateGroup(group);
    SymbolicScalar thisRank = GetHcclRankId(group);
    auto shmemSignalTile = View(shmemSignal, {worldSize, 1, 1, shmemSignal.GetShape(3), shmemSignal.GetShape(4)},
        std::vector<SymbolicScalar>{0, 0, 0, 0, 0});
    auto shmemSignalOut = ShmemSignal(predToken, shmemSignalTile, AtomicType::ADD);
    auto shmemSignalLocal = View(shmemSignal, {1, 1, 1, shmemSignal.GetShape(3), shmemSignal.GetShape(4)},
        std::vector<SymbolicScalar>{thisRank, 0, 0, 0, 0});
    return WaitUntil(shmemSignalOut, shmemSignalLocal, worldSize, true);
}

Tensor ShmemDataSet(const Tensor& predToken, const Tensor& shmemData)
{
    constexpr int32_t supportedDim = 4;
    ASSERT(shmemData.GetShape().size() == supportedDim) << "The dim of \"shmemData\" only supports " << supportedDim
        << ", but got " << shmemData.GetShape().size();

    auto& function = *Program::GetInstance().GetCurrentFunction();
    auto out = std::make_shared<LogicalTensor>(function, DT_INT32, Shape{1, 1});
    auto& op = function.AddOperation(Opcode::OP_SHMEM_SET, {predToken.GetStorage(), shmemData.GetStorage()}, {out});
    DistOpAttr distOpAttr;
    distOpAttr.setType = 0;
    op.SetAttr(OpAttributeKey::distOpAttr, distOpAttr);
    return out;
}

Tensor ShmemSignalSet(const Tensor& predToken, const Tensor& shmemSignal)
{
    constexpr int32_t supportedDim = 5;
    ASSERT(shmemSignal.GetShape().size() == supportedDim) << "The dim of \"shmemSignal\" only supports " << supportedDim
        << ", but got " << shmemSignal.GetShape().size();

    auto& function = *Program::GetInstance().GetCurrentFunction();
    auto out = std::make_shared<LogicalTensor>(function, DT_INT32, Shape{1, 1});
    auto& op = function.AddOperation(Opcode::OP_SHMEM_SET, {predToken.GetStorage(), shmemSignal.GetStorage()}, {out});
    DistOpAttr distOpAttr;
    distOpAttr.setType = 1;
    op.SetAttr(OpAttributeKey::distOpAttr, distOpAttr);
    return out;
}

void AllGather(const Tensor& predToken, const Tensor& in, const char* group, Tensor& shmemData, Tensor& shmemSignal,
    Tensor& out)
{
    uint32_t worldSize = shmemData.GetShape()[0];
    ASSERT(worldSize > 0) << "worldSize should be more than 0.";
    int32_t row = in.GetShape(0);
    int32_t col = in.GetShape(1);
    SymbolicScalar thisRank = GetHcclRankId(group);
    const TileShape& tileShape = TileShape::Current();
    ValidateGroup(group);
    ValidateTilingSize(tileShape.GetVecTile(), in, worldSize);
    ValidateParams(predToken, in, out, shmemData.GetShape(), in.GetDataType());
    ValidateTypeAndShape(shmemData, out.GetDataType(), {worldSize, worldSize, row, col});
    ValidateTypeAndShape(shmemSignal, DataType::DT_INT32, {worldSize, worldSize, worldSize, row, col});
    ValidateTypeAndShape(out, in.GetDataType(), {row * worldSize, col});
    for (uint32_t dynRankId = 0; dynRankId < worldSize; ++dynRankId) {
        auto shmemDataTile = View(shmemData, {1, 1, row, col}, std::vector<SymbolicScalar>{dynRankId, thisRank, 0, 0});
        auto shmemSignalTile = View(shmemSignal, {1, 1, 1, row, col}, 
            std::vector<SymbolicScalar>{dynRankId, dynRankId, thisRank, 0, 0});
        auto shmemPutOut = ShmemPut(predToken, in, shmemDataTile);
        auto shmemSignalOut = ShmemSignal(shmemPutOut, shmemSignalTile, AtomicType::SET);
        auto shmemDataLocal = View(shmemData, {1, 1, row, col}, std::vector<SymbolicScalar>{thisRank, dynRankId, 0, 0});
        auto shmemSignalLocal = View(shmemSignal, {1, 1, 1, row, col}, 
            std::vector<SymbolicScalar>{thisRank, thisRank, dynRankId, 0, 0});
        auto waitUntilOut= WaitUntil(shmemSignalOut, shmemSignalLocal, 1);
        auto shmemGetOut= ShmemGet(waitUntilOut, shmemDataLocal);
        Assemble(shmemGetOut, {dynRankId * row, 0}, out);
    }
}

void ReduceScatter(const Tensor& predToken, const Tensor& in, const char* group, Tensor& shmemData, Tensor& shmemSignal,
    DistReduceType reduceType, Tensor& out)
{
    (void)reduceType;
    int32_t row = in.GetShape(0);
    int32_t col = in.GetShape(1);
    uint32_t worldSize = shmemData.GetShape()[0];
    ASSERT(worldSize > 0) << "worldSize should be more than 0.";
    ASSERT((row % worldSize) == 0) << "ReduceScatter constraint: row must be divisible by worldSize, but row: " << row
        << ", worldSize: " << worldSize; 
    const int32_t rowOut = row / worldSize;
    SymbolicScalar thisRank = GetHcclRankId(group);
    const TileShape& tileShape = TileShape::Current();
    ValidateGroup(group);
    ValidateTilingSize(tileShape.GetVecTile(), in, worldSize);
    ValidateParams(predToken, in, out, shmemData.GetShape(), shmemData.GetDataType(),
        false, true, {DT_INT32, DT_FP32, DT_FP16, DT_BF16});
    ValidateTypeAndShape(shmemData, ((in.GetDataType() == DT_BF16) || (in.GetDataType() == DT_FP16) ? DT_FP32 :
        out.GetDataType()), {worldSize, 1, rowOut, col});
    ValidateTypeAndShape(shmemSignal, DataType::DT_INT32, {worldSize, worldSize, 1, rowOut, col});
    ValidateTypeAndShape(out, in.GetDataType(), {rowOut, col});
    for (uint32_t dynRankId = 0; dynRankId < worldSize; ++dynRankId) {
        auto shmemDataTile = View(shmemData, {1, 1, rowOut, col}, std::vector<SymbolicScalar>{dynRankId, 0, 0, 0});
        auto shmemSignalTile =
            View(shmemSignal, {1, 1, 1, rowOut, col}, std::vector<SymbolicScalar>{dynRankId, dynRankId, 0, 0, 0});
        auto inTile = View(in, {rowOut, col}, std::vector<SymbolicScalar>{dynRankId * rowOut, 0});
        auto shmemPutOut = ShmemPut(predToken, inTile, shmemDataTile, AtomicType::ADD);
        ShmemSignal(shmemPutOut, shmemSignalTile, AtomicType::ADD);
    }
    auto shmemDataLocal = View(shmemData, {1, 1, rowOut, col}, std::vector<SymbolicScalar>{thisRank, 0, 0, 0});
    auto shmemSignalLocal =
        View(shmemSignal, {1, 1, 1, rowOut, col}, std::vector<SymbolicScalar>{thisRank, thisRank, 0, 0, 0});
    auto waitUntilOut = WaitUntil(in, shmemSignalLocal, worldSize);
    out = ShmemGet(waitUntilOut, shmemDataLocal, in.GetDataType());
}

void AllReduceValidate(const Tensor& predToken, const Tensor& in, const Tensor& shmemData, const char* group,
    const Tensor& out)
{
    ValidateGroup(group);
    ValidateParams(predToken, in, out, shmemData.GetShape(), shmemData.GetDataType(), true, true,
        {DT_INT32, DT_FP32, DT_FP16, DT_BF16});
    const TileShape& tileShape = TileShape::Current();
    ValidateTilingSize(tileShape.GetVecTile(), in, shmemData.GetShape(0));
}

void OneShotAllReduce(const Tensor& predToken, const Tensor& in, const char* group, Tensor& shmemData,
    Tensor& shmemSignal, Tensor& out)
{
    int32_t row = in.GetShape(0);
    int32_t col = in.GetShape(1);
    SymbolicScalar thisRank = GetHcclRankId(group);
    uint32_t worldSize = shmemData.GetShape()[0];
    AllReduceValidate(predToken, in, shmemData, group, out);
    ValidateTypeAndShape(shmemData, ((in.GetDataType() == DT_BF16) || (in.GetDataType() == DT_FP16) ? DT_FP32 :
        out.GetDataType()), {worldSize, 1, row, col});
    ValidateTypeAndShape(shmemSignal, DataType::DT_INT32, {worldSize, worldSize, 1, row, col});
    ASSERT(worldSize > 0) << "worldSize should be more than 0.";
    for (uint32_t dynRankId = 0; dynRankId < worldSize; ++dynRankId) {
        auto shmemDataTile = View(shmemData, {1, 1, row, col}, std::vector<SymbolicScalar>{dynRankId, 0, 0, 0});
        auto shmemSignalTile = View(shmemSignal, {1, 1, 1, row, col},
            std::vector<SymbolicScalar>{dynRankId, dynRankId, 0, 0, 0});
        auto shmemPutOut = ShmemPut(predToken, in, shmemDataTile, AtomicType::ADD);
        ShmemSignal(shmemPutOut, shmemSignalTile, AtomicType::ADD);
    }
    auto shmemDataTile = View(shmemData, {1, 1, row, col}, std::vector<SymbolicScalar>{thisRank, 0, 0, 0});
    auto shmemSignalTile = View(shmemSignal, {1, 1, 1, row, col}, std::vector<SymbolicScalar>{thisRank, thisRank, 0, 0, 0});
    auto waitUntilout = WaitUntil(in, shmemSignalTile, worldSize);
    out = ShmemGet(waitUntilout, shmemDataTile, in.GetDataType());
}

// PUT + signal in one call. Data and signal Views are passed inline by the caller.
static void PutAndSignal(const Tensor& pred, const Tensor& input,
    const Tensor& dataTile, const Tensor& signalTile)
{
    auto putOut = ShmemPut(pred, input, dataTile, AtomicType::ADD);
    ShmemSignal(putOut, signalTile, AtomicType::ADD);
}

// Wait for all contributions + read the reduced result. Views passed inline by the caller.
// Output dtype taken from pred (callers pass the input data tensor here).
static Tensor WaitAndGet(const Tensor& pred, const Tensor& dataTile, const Tensor& signalTile,
    uint32_t worldSize)
{
    auto waitOut = WaitUntil(pred, signalTile, worldSize);
    return ShmemGet(waitOut, dataTile, pred.GetDataType());
}

// OneShotAllReduce_v2: Same algorithm and signature as OneShotAllReduce.
//
// Reduces repetition via:
//   - viewData(rank)/viewSignal(rank) lambdas for the fixed shmem layout
//   - PutAndSignal / WaitAndGet --> paired operations
void OneShotAllReduce_v2(const Tensor& predToken, const Tensor& in, const char* group, Tensor& shmemData,
    Tensor& shmemSignal, Tensor& out)
{
    int32_t row = in.GetShape(0);
    int32_t col = in.GetShape(1);
    SymbolicScalar thisRank = GetHcclRankId(group);
    uint32_t worldSize = shmemData.GetShape()[0];

    // Phase 1: Scatter — every rank puts its full input to all targets
    for (uint32_t dynRankId = 0; dynRankId < worldSize; ++dynRankId) {
        PutAndSignal(predToken, in,
            View(shmemData, {1, 1, row, col}, std::vector<SymbolicScalar>{dynRankId, 0, 0, 0}),
            View(shmemSignal, {1, 1, 1, row, col}, std::vector<SymbolicScalar>{dynRankId, dynRankId, 0, 0, 0}));
    }

    // Phase 2: Wait & Gather — read this rank's fully-reduced result
    out = WaitAndGet(in,
        View(shmemData, {1, 1, row, col}, std::vector<SymbolicScalar>{thisRank, 0, 0, 0}),
        View(shmemSignal, {1, 1, 1, row, col}, std::vector<SymbolicScalar>{thisRank, thisRank, 0, 0, 0}),
        worldSize);
}

// OneShotAllReduce_v3: Uses an OneShotCommunicator to hide shmem layout details.
//
// Work rank IDs instead of raw View() calls.
// Emitted IR is identical to OneShotAllReduce / OneShotAllReduce_v2.
void OneShotAllReduce_v3(const Tensor& predToken, const Tensor& in, const char* group,
    Tensor& shmemData, Tensor& shmemSignal, Tensor& out)
{
    OneShotCommunicator comm(group, shmemData, shmemSignal);

    // Phase 1: Scatter to all ranks with atomic ADD
    for (uint32_t dynRankId = 0; dynRankId < comm.WorldSize(); ++dynRankId) {
        comm.Put(predToken, in, dynRankId, AtomicType::ADD);
    }

    // Phase 2: Gather — wait for all contributions, read reduced result
    out = comm.WaitAndGet(in);
}

// OneShotAllReduce_v4: like v3 but with OneShotCommunicatorV2 and explicit data Views.
// Same IR as v2/v3.
void OneShotAllReduce_v4(const Tensor& predToken, const Tensor& in, const char* group,
    Tensor& shmemData, Tensor& shmemSignal, Tensor& out)
{
    int32_t row = in.GetShape(0);
    int32_t col = in.GetShape(1);
    OneShotCommunicatorV2 comm(group, static_cast<uint32_t>(shmemData.GetShape()[0]), shmemSignal);

    // Phase 1: Scatter to all ranks with atomic ADD
    for (uint32_t dynRankId = 0; dynRankId < comm.WorldSize(); ++dynRankId) {
        auto dynRankView = View(shmemData, {1, 1, row, col},
            std::vector<SymbolicScalar>{dynRankId, 0, 0, 0});
        comm.Put(predToken, in, dynRankView, dynRankId, AtomicType::ADD);
    }

    // Phase 2: Gather — wait for all contributions, read reduced result
    auto dataLocal = View(shmemData, {1, 1, row, col},
        std::vector<SymbolicScalar>{comm.ThisRank(), 0, 0, 0});
    out = comm.WaitAndGet(in, dataLocal);
}

// OneShotAllReduce_v5: like v4 but with external OneShotCommunicatorV2.
Tensor OneShotAllReduce_v5(const Tensor& predToken, const Tensor& in, Tensor& shmemData, OneShotCommunicatorV2& comm)
{
    int32_t row = in.GetShape(0);
    int32_t col = in.GetShape(1);

    // Phase 1: Scatter — broadcast input to all ranks with atomic ADD
    for (uint32_t dynRankId = 0; dynRankId < comm.WorldSize(); ++dynRankId) {
        // API: where: View, who: rank
        auto dynRankView = View(shmemData, {1, 1, row, col},
            std::vector<SymbolicScalar>{dynRankId, 0, 0, 0});
        comm.Put(predToken, in, dynRankView, dynRankId, AtomicType::ADD);
    }

    // Phase 2: Wait — use predToken (not the Put output) to match base IR ordering
    return comm.Wait(predToken);
}

void TwoShotAllReduce(const Tensor& predToken, const Tensor& in, const char* group, Tensor& shmemData,
    Tensor& shmemSignal, Tensor& out)
{
    int32_t row = in.GetShape(0);
    int32_t col = in.GetShape(1);
    uint32_t worldSize = shmemData.GetShape()[0];
    ASSERT(worldSize > 0) << "AllReduce worldSize should be more than 0.";
    ASSERT(row % worldSize == 0) << "Two_Shot_AllReduce constraint: row must be divisible by worldSize but row: " << row
        << ", worldSize: " << worldSize;  
    int32_t rowPerRank = row / worldSize;
    AllReduceValidate(predToken, in, shmemData, group, out);
    ValidateTypeAndShape(shmemData, ((in.GetDataType() == DT_BF16) || (in.GetDataType() == DT_FP16) ? DT_FP32 :
        out.GetDataType()), {worldSize, worldSize, rowPerRank, col});
    ValidateTypeAndShape(shmemSignal, DataType::DT_INT32, {worldSize, worldSize, worldSize, rowPerRank, col});
    SymbolicScalar thisRank = GetHcclRankId(group);
    for (uint32_t dynRankId = 0; dynRankId < worldSize; ++dynRankId) {
        auto shmemDataTile = View(shmemData, {1, 1, rowPerRank, col}, std::vector<SymbolicScalar>{dynRankId, dynRankId, 0, 0});
        auto shmemSignalTile = View(shmemSignal, {worldSize, 1, 1, rowPerRank, col}, std::vector<SymbolicScalar>{0, dynRankId, dynRankId, 0, 0});
        auto inTile = View(in, {rowPerRank, col}, std::vector<SymbolicScalar>{rowPerRank * dynRankId, 0});
        auto shmemPutOut = ShmemPut(predToken, inTile, shmemDataTile, AtomicType::ADD);
        auto shmemSignalOut = ShmemSignal(shmemPutOut, shmemSignalTile, AtomicType::ADD);
        auto waitSignalTile = View(shmemSignal, {1, 1, 1, rowPerRank, col}, std::vector<SymbolicScalar>{thisRank, dynRankId, dynRankId, 0, 0});
        auto waitUntilOut = WaitUntil(shmemSignalOut, waitSignalTile, worldSize);
        auto tmp = ShmemGet(waitUntilOut, shmemDataTile, in.GetDataType());
        Assemble(tmp, {rowPerRank * dynRankId, 0}, out);
    }
}
// TwoShotAllReduce_v2: Same algorithm and signature as TwoShotAllReduce.
//
// Reduces repetition via views for the TwoShot shmem layout:
//   - viewData(chunkId) / viewPutSignal(chunkId) / viewWaitSignal(chunkId)
//   - inputChunk(chunkId) for input tiling
void TwoShotAllReduce_v2(const Tensor& predToken, const Tensor& in, const char* group,
    Tensor& shmemData, Tensor& shmemSignal, Tensor& out)
{
    int32_t row = in.GetShape(0);
    int32_t col = in.GetShape(1);
    uint32_t worldSize = shmemData.GetShape()[0];
    int32_t rowPerRank = row / worldSize;
    SymbolicScalar thisRank = GetHcclRankId(group);

    auto viewData = [&](uint32_t dynRankId) {
        return View(shmemData, {1, 1, rowPerRank, col},
            std::vector<SymbolicScalar>{dynRankId, dynRankId, 0, 0});
    };
    auto viewPutSignal = [&](uint32_t dynRankId) {
        return View(shmemSignal, {static_cast<int64_t>(worldSize), 1, 1, rowPerRank, col},
            std::vector<SymbolicScalar>{0, dynRankId, dynRankId, 0, 0});
    };
    auto viewWaitSignal = [&](uint32_t dynRankId) {
        return View(shmemSignal, {1, 1, 1, rowPerRank, col},
            std::vector<SymbolicScalar>{thisRank, dynRankId, dynRankId, 0, 0});
    };
    auto inputChunk = [&](uint32_t dynRankId) {
        return View(in, {rowPerRank, col},
            std::vector<SymbolicScalar>{rowPerRank * dynRankId, 0});
    };

    for (uint32_t dynRankId = 0; dynRankId < worldSize; ++dynRankId) {
        auto putOut = ShmemPut(predToken, inputChunk(dynRankId), viewData(dynRankId), AtomicType::ADD);
        auto signalOut = ShmemSignal(putOut, viewPutSignal(dynRankId), AtomicType::ADD);
        auto waitOut = WaitUntil(signalOut, viewWaitSignal(dynRankId), worldSize);
        auto reduced = ShmemGet(waitOut, viewData(dynRankId), in.GetDataType());
        Assemble(reduced, {rowPerRank * dynRankId, 0}, out);
    }
}

// TwoShotAllReduce_v3: Uses a TwoShotCommunicator to hide shmem layout details.
//
// Works with chunk IDs instead of raw View() calls
// for shmem data and signal buffers. Input chunking and Assemble remain
// in the algorithm.
void TwoShotAllReduce_v3(const Tensor& predToken, const Tensor& in, const char* group,
    Tensor& shmemData, Tensor& shmemSignal, Tensor& out)
{
    TwoShotCommunicator comm(group, shmemData, shmemSignal);
    int32_t rowPerRank = in.GetShape(0) / comm.WorldSize();
    int32_t col = in.GetShape(1);

    for (uint32_t dynRankId = 0; dynRankId < comm.WorldSize(); ++dynRankId) {
        auto inChunk = View(in, {rowPerRank, col},
            std::vector<SymbolicScalar>{rowPerRank * dynRankId, 0});
        auto signalOut = comm.Put(predToken, inChunk, dynRankId, AtomicType::ADD);
        auto reduced = comm.WaitAndGet(signalOut, dynRankId, in.GetDataType());
        Assemble(reduced, {rowPerRank * dynRankId, 0}, out);
    }
}

// TwoShotAllReduce_v4: TwoShotCommunicatorV2 with explicit data Views.
//
// Data placement via explicit View() calls.
// TwoShotCommunicatorV2 handles signal coordination and group metadata.
// Uses combined WaitAndGet() per chunk (convenience method).
void TwoShotAllReduce_v4(const Tensor& predToken, const Tensor& in, const char* group,
    Tensor& shmemData, Tensor& shmemSignal, Tensor& out)
{
    int32_t row = in.GetShape(0);
    int32_t col = in.GetShape(1);
    int32_t rowPerRank = row / static_cast<int32_t>(shmemData.GetShape()[0]);
    TwoShotCommunicatorV2 comm(group, static_cast<uint32_t>(shmemData.GetShape()[0]), shmemSignal);

    for (uint32_t dynRankId = 0; dynRankId < comm.WorldSize(); ++dynRankId) {
        auto inChunk = View(in, {rowPerRank, col},
            std::vector<SymbolicScalar>{rowPerRank * dynRankId, 0});
        auto dataView = View(shmemData, {1, 1, rowPerRank, col},
            std::vector<SymbolicScalar>{dynRankId, dynRankId, 0, 0});
        auto signalOut = comm.Put(predToken, inChunk, dataView, dynRankId, AtomicType::ADD);
        auto reduced = comm.WaitAndGet(signalOut, dataView, dynRankId);
        Assemble(reduced, {rowPerRank * dynRankId, 0}, out);
    }
}

// TwoShotAllReduce_v5: per-chunk Put/Wait/Pull with external TwoShotCommunicatorV2.
void TwoShotAllReduce_v5(const Tensor& predToken, const Tensor& in,
    Tensor& shmemData, TwoShotCommunicatorV2& comm, Tensor& out)
{
    int32_t row = in.GetShape(0);
    int32_t col = in.GetShape(1);
    int32_t rowPerRank = row / comm.WorldSize();

    for (uint32_t dynRankId = 0; dynRankId < comm.WorldSize(); ++dynRankId) {
        auto inChunk = View(in, {rowPerRank, col},
            std::vector<SymbolicScalar>{rowPerRank * dynRankId, 0});
        auto dataView = View(shmemData, {1, 1, rowPerRank, col},
            std::vector<SymbolicScalar>{dynRankId, dynRankId, 0, 0});

        // Phase 1: Put — write chunk to shmem slot + signal all ranks
        auto signalOut = comm.Put(predToken, inChunk, dataView, dynRankId, AtomicType::ADD);

        // Phase 2: Wait — block until all contributions for this chunk have arrived
        auto waitOut = comm.Wait(signalOut, dynRankId);

        // Phase 3: Pull — read reduced chunk from shmem
        auto reduced = comm.Pull(waitOut, dataView);
        Assemble(reduced, {rowPerRank * dynRankId, 0}, out);
    }
}

// OneShotAllReduce_v6: scatter-only — Put to all ranks, no Wait/Pull.
//
// predToken is used for Put() (scheduling the ShmemPut ops).
// Wait and Pull are the caller's responsibility, enabling overlap
// between scatter and local compute.
void OneShotAllReduce_v6(const Tensor& predToken, const Tensor& in, Tensor& shmemData, OneShotCommunicatorV2& comm)
{
    OneShotAllReduceCoarseScatterImpl(predToken, in, shmemData, comm);
}

// OneShotAllReduce_v6_light: same coarse scatter cadence as v6.
// Uses compact signal domains via CreateShmemSignalLight at setup time.
void OneShotAllReduce_v6_light(const Tensor& predToken, const Tensor& in, Tensor& shmemData, OneShotCommunicatorV2& comm)
{
    OneShotAllReduceCoarseScatterImpl(predToken, in, shmemData, comm);
}

// OneShotAllReduce_v7: scatter-only with grouped signaling.
// Communicator controls chunking and chunk->signal-group mapping.
void OneShotAllReduce_v7(const Tensor& predToken, const Tensor& in, Tensor& shmemData, OneShotCommunicatorV3& comm)
{
    OneShotAllReduceGroupedScatterImpl(
        predToken, in, shmemData, comm,
        [&](const Tensor& pred, const Tensor& inChunk, const Tensor& dataView, uint32_t targetRank, uint32_t chunkId) {
            comm.Put(pred, inChunk, dataView, targetRank, chunkId, AtomicType::ADD);
        });
}

// OneShotAllReduce_v7_light: same grouped signaling cadence as v7,
// but intended for compact grouped signal layouts.
void OneShotAllReduce_v7_light(const Tensor& predToken, const Tensor& in, Tensor& shmemData, OneShotCommunicatorV3Light& comm)
{
    OneShotAllReduceGroupedScatterImpl(
        predToken, in, shmemData, comm,
        [&](const Tensor& pred, const Tensor& inChunk, const Tensor& dataView, uint32_t targetRank, uint32_t chunkId) {
            comm.Put(pred, inChunk, dataView, targetRank, chunkId, AtomicType::ADD);
        });
}

// OneShotAllReduce_v8: scatter-only with policy-driven signal plan (v8).
// The communicator owns chunk->group mapping and thresholds.
void OneShotAllReduce_v8(const Tensor& predToken, const Tensor& in, Tensor& shmemData, OneShotCommunicatorV4& comm)
{
    OneShotAllReduceGroupedScatterImpl(
        predToken, in, shmemData, comm,
        [&](const Tensor& pred, const Tensor& inChunk, const Tensor& dataView, uint32_t targetRank, uint32_t chunkId) {
            comm.PutWithSignal(pred, inChunk, dataView, targetRank, chunkId, AtomicType::ADD);
        });
}

// OneShotAllReduce_v8_light: same policy-driven grouped signaling cadence as v8,
// but intended for compact grouped signal layouts.
void OneShotAllReduce_v8_light(const Tensor& predToken, const Tensor& in, Tensor& shmemData, OneShotCommunicatorV4Light& comm)
{
    OneShotAllReduceGroupedScatterImpl(
        predToken, in, shmemData, comm,
        [&](const Tensor& pred, const Tensor& inChunk, const Tensor& dataView, uint32_t targetRank, uint32_t chunkId) {
            comm.PutWithSignal(pred, inChunk, dataView, targetRank, chunkId, AtomicType::ADD);
        });
}

}   // namespace npu::tile_fwk::Distributed
