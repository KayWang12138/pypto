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

void ValidateGroup(const char* group)
{
    CHECK(group != nullptr) << "\"group\" cannot be nullptr";
    int32_t groupLen = std::strlen(group);
    CHECK((groupLen >= 1) && (groupLen < 128)) << "The length of \"group\" only supports [1, 128), but got "
        << groupLen;
}

void ValidateTypeAndShape(const Tensor& tensor, const DataType expectedType, const Shape expectedShape)
{
    CHECK(tensor.GetDataType() == expectedType) << "Tensor dtype not supported";
    CHECK(tensor.GetShape() == expectedShape) << "Tensor shape not supported";
}

void ValidateTilingSize(const Opcode& opCode, const VecTile& vecTile, const int32_t supportDim)
{
    CHECK(vecTile.valid()) << OpcodeManager::Inst().GetOpcodeStr(opCode) << ": vecTile must contains exactly " <<
        supportDim << " elements and both must be non-zero";
    CHECK(supportDim == static_cast<int32_t>(vecTile.size())) << OpcodeManager::Inst().GetOpcodeStr(opCode) <<
        " has invalid dim of tile shape: dim of tile shape must be equal to " << std::to_string(supportDim) <<
        ", but got " << static_cast<int32_t>(vecTile.size());
}

template <typename Container, typename F>
std::string ToString(const Container& c, F func)
{
    std::ostringstream oss;
    oss << "[";
    bool first = true;
    for (const auto& item : c) {
        if (!first) {
            oss << ", ";
        }
        oss << func(item);
        first = false;
    }
    oss << "]";
    return oss.str();
}

void ValidateTiling(const Opcode& opCode, const Tensor &target, const std::string& tensorDesc)
{
    const auto vecTile = TileShape::Current().GetVecTile();
    CHECK(vecTile.valid()) << OpcodeManager::Inst().GetOpcodeStr(opCode) <<
        ": vecTile every shape and dim should > 0, but got:" << ToString(vecTile.tile, [](auto v){ return std::to_string(v);});
    CHECK(target.Dim() == vecTile.size()) << OpcodeManager::Inst().GetOpcodeStr(opCode) <<
        " dim of vectile shape must be equal to " << std::to_string(target.Dim()) <<
        ", which is same as " << tensorDesc << ", but got " << vecTile.size();
}

void ValidateDataType(const Tensor& tensor, const std::string& tensorDesc, const std::unordered_set<DataType>& allowedTypes)
{
    auto dataType = tensor.GetDataType();
    CHECK(allowedTypes.empty() || allowedTypes.count(dataType)) << "Invalid data type: " << tensorDesc <<
        " data type only support " << ToString(allowedTypes, DataType2String) << ", but got:" <<
        DataType2String(dataType);
}

void ValidateDim(const Tensor& tensor, const std::string& tensorDesc, const std::unordered_set<size_t>& allowedDims) {
    const auto& shape = tensor.GetShape();
    CHECK(allowedDims.empty() || allowedDims.count(shape.size())) << "Invalid dimensional: " << tensorDesc <<
        " dimensional must be " << ToString(allowedDims, [](auto v){ return std::to_string(v);}) << ", but got dimensional=" << shape.size();
    for (size_t i = 0; i < shape.size(); ++i) {
        CHECK(shape[i] > 0) << "Invaild dimemsion value: " << tensorDesc << " dimension " << i
            << " must be greater than 0, but got " << shape[i];
    }
}

void ValidateShape(const Tensor& tensor, const std::string& tensorDesc, uint32_t supportedDim) {
    const auto& shape = tensor.GetShape();
    CHECK(shape.size() == supportedDim) << "Invalid dimensional: " << tensorDesc << " dimensional must be "
        << supportedDim << ", but got dimensional=" << shape.size();
    for (size_t i = 0; i < shape.size(); ++i) {
        CHECK(shape[i] > 0) << "Invaild dimemsion value: " << tensorDesc << " dimension " << i
            << " must be greater than 0, but got " << shape[i];
    }
}

void ValidateFormat(const Tensor& tensor, const std::string& tensorDesc,
    const std::unordered_set<TileOpFormat>& allowedFormats = {TileOpFormat::TILEOP_ND}) {
    CHECK(allowedFormats.empty() || allowedFormats.count(tensor.Format())) << "Invalid format: " << tensorDesc
        << " only support ND format, but got format: " << std::to_string(tensor.Format());
}

void ValidateTensor(const Tensor& tensor, const std::string& tensorDesc,
    const std::unordered_set<size_t>& allowedDims = {},
    const std::unordered_set<DataType>& allowedTypes = {},
    const std::unordered_set<TileOpFormat>& allowedFormats = {}) {
    ValidateDim(tensor, tensorDesc, allowedDims);
    ValidateDataType(tensor, tensorDesc, allowedTypes);
    ValidateFormat(tensor, tensorDesc, allowedFormats);
}

void ValidateWorldSize(const char* group, int64_t worldSize) {
    static std::unordered_map<std::string, int64_t> groupWorldSizeMap;
    auto groupWorldSize = groupWorldSizeMap.find(group);
    if (groupWorldSize== groupWorldSizeMap.end()) {
        CHECK(worldSize > 0) << "Invalid world size for group " << group << ": world size must be greather than 0"
            << ", but got " << worldSize;
        groupWorldSizeMap.emplace(group, worldSize);
    } else {
        CHECK(worldSize == groupWorldSize->second) << "WorldSize mismatch for group " << group
            << ": expected " << groupWorldSize->second << ", but got " << worldSize;
    }
}

void ValidateParams(const Tensor& predToken, const Tensor& in, const Tensor& out, Shape shmemDataShape,
    DataType shmemDataType, bool checkShapeMatch = false, bool validateType = false,
    const std::unordered_set<DataType>& allowedTypes = {})
{
    ValidateShape(predToken, "PredToken", 2);
    int32_t predRow = predToken.GetShape(0);
    int32_t predCol = predToken.GetShape(1);
    CHECK(predRow > 0 && predCol > 0) << "PredToken parameter error - the 'row' and 'col' dimensional of the input tensor must be greater than 0, "
        << "but got row=" << predRow << ", col=" << predCol;
    ValidateShape(in, "Input tensor", 2);
    ValidateShape(out, "Output tensor", 2);
    CHECK(out.GetDataType() == in.GetDataType()) << "The data type of \"out\" must be consistent with that of \"in\", "
        << "but the data type of \"out\" is "<< DataType2String(out.GetDataType()) << " and the data type of \"in\" is "
        << DataType2String(in.GetDataType()) << ".";
    ValidateFormat(in, "Input tensor");
    ValidateFormat(out, "Output tensor");
    int32_t inRow = in.GetShape(0);
    int32_t inCol = in.GetShape(1);
    int32_t outRow = out.GetShape(0);
    int32_t outCol = out.GetShape(1);
    if (checkShapeMatch) {
        CHECK((inRow == outRow) && (inCol == outCol)) <<
        "Shape mismatch: Input and output dimensions must be the same, but got "
        << "Input shape: (" << inRow << "," << inCol << "), Output shape: (" << outRow << "," << outCol << ").";
    }
    if (validateType) {
       ValidateDataType(in, "Input tensor", allowedTypes);
    }
    int64_t shmemDataEleNum =
        std::accumulate(shmemDataShape.begin() + 1, shmemDataShape.end(), 1, std::multiplies<int64_t>());
    int64_t shmemSignalEleNum = shmemDataShape[0] * MAX_TILE_NUM * SHMEM_SIGNAL_STRIDE;
    uint64_t shmemSize = shmemDataEleNum * BytesOf(shmemDataType) + shmemSignalEleNum * BytesOf(DT_INT32);
    const uint64_t winSize = 1024 * 1024 * 200;
    CHECK(shmemSize < winSize) << "Exceeds winSize limit. Maximum allowed: " << winSize << ", got: " << shmemSize;
}

ShmemTensor CreateShmemData(const char* group, int64_t worldSize, DataType dataType, const Shape& shape)
{
    ValidateGroup(group);
    ValidateWorldSize(group, worldSize);
    ShmemTensor t;
    t.group = std::string(group);
    t.worldSize = worldSize;
    // static uint64_t s_index = 0;
    // LOOP("CreateShmemData" + std::to_string(s_index++), FunctionType::DYNAMIC_LOOP, index, LoopRange(1)) {
    //     (void)index;
        auto &function = *Program::GetInstance().GetCurrentFunction();
        int32_t hcclGroupIndex = static_cast<int>(CommGroupRecorder::GetInstance().Input(std::string(group)));
        Shape dataShape = shape;
        auto dataInner = std::make_shared<LogicalTensor>(function, dataType, dataShape);
        t.data = dataInner;
        Program::GetInstance().GetTensorSlotManager()->TensorWrite(t.data, SlotProperty::SHMEM_TENSOR);
        auto &dataOp = function.AddOperation(Opcode::OP_BIND_TENSOR, {}, {dataInner});
        dataOp.SetAttribute(OpAttributeKey::bindTensor, BindTensor(hcclGroupIndex, 0,
            AlignUp(BytesOf(dataType) * std::accumulate(dataShape.begin(), dataShape.end(), 1, std::multiplies<int64_t>()), 512)));

        Shape signalShape{worldSize};
        signalShape.insert(signalShape.end(), shape.begin(), shape.end());
        auto signalInner = std::make_shared<LogicalTensor>(function, DataType::DT_INT32, signalShape);
        t.signal = signalInner;
        Program::GetInstance().GetTensorSlotManager()->TensorWrite(t.signal, SlotProperty::SHMEM_TENSOR);
        auto &signalOp = function.AddOperation(Opcode::OP_BIND_TENSOR, {}, {signalInner});
        signalOp.SetAttribute(OpAttributeKey::bindTensor, BindTensor(hcclGroupIndex, 1,
            BytesOf(DataType::DT_INT32) * worldSize * SHMEM_SIGNAL_STRIDE * MAX_TILE_NUM));
    // }
    return t;
}

void CreateShmemData(const char* group, int64_t worldSize, DataType dataType, const Shape& shape, ShmemTensor &t)
{
    ValidateGroup(group);
    ValidateWorldSize(group, worldSize);
    // ShmemTensor t;
    t.group = std::string(group);
    t.worldSize = worldSize;
    // static uint64_t s_index = 0;
    // LOOP("CreateShmemData" + std::to_string(s_index++), FunctionType::DYNAMIC_LOOP, index, LoopRange(1)) {
    //     (void)index;
        auto &function = *Program::GetInstance().GetCurrentFunction();
        int32_t hcclGroupIndex = static_cast<int>(CommGroupRecorder::GetInstance().Input(std::string(group)));
        Shape dataShape = shape;
        auto dataInner = std::make_shared<LogicalTensor>(function, dataType, dataShape);
        t.data = dataInner;
        Program::GetInstance().GetTensorSlotManager()->TensorWrite(t.data, SlotProperty::SHMEM_TENSOR);
        auto &dataOp = function.AddOperation(Opcode::OP_BIND_TENSOR, {}, {dataInner});
        dataOp.SetAttribute(OpAttributeKey::bindTensor, BindTensor(hcclGroupIndex, 0,
            AlignUp(BytesOf(dataType) * std::accumulate(dataShape.begin(), dataShape.end(), 1, std::multiplies<int64_t>()), 512)));

        Shape signalShape{worldSize};
        signalShape.insert(signalShape.end(), shape.begin(), shape.end());
        auto signalInner = std::make_shared<LogicalTensor>(function, DataType::DT_INT32, signalShape);
        t.signal = signalInner;
        Program::GetInstance().GetTensorSlotManager()->TensorWrite(t.signal, SlotProperty::SHMEM_TENSOR);
        auto &signalOp = function.AddOperation(Opcode::OP_BIND_TENSOR, {}, {signalInner});
        signalOp.SetAttribute(OpAttributeKey::bindTensor, BindTensor(hcclGroupIndex, 1,
            BytesOf(DataType::DT_INT32) * worldSize * SHMEM_SIGNAL_STRIDE * MAX_TILE_NUM));
    // }
    // return t;
}

ShmemTensor CreateShmemSignal(const char* group, int64_t worldSize)
{
    ValidateGroup(group);
    ValidateWorldSize(group, worldSize);
    ShmemTensor t;
    t.group = std::string(group);
    t.worldSize = worldSize;
    static uint64_t s_index = 0;
    LOOP("CreateShmemSignal" + std::to_string(s_index++), FunctionType::DYNAMIC_LOOP, index, LoopRange(1)) {
        (void)index;
        auto &function = *Program::GetInstance().GetCurrentFunction();
        int32_t hcclGroupIndex = static_cast<int>(CommGroupRecorder::GetInstance().Input(std::string(group)));
        Shape signalShape{worldSize, 1, 1, 8};
        auto signalInner = std::make_shared<LogicalTensor>(function, DataType::DT_INT32, signalShape);
        t.signal = signalInner;
        Program::GetInstance().GetTensorSlotManager()->TensorWrite(t.signal, SlotProperty::SHMEM_TENSOR);
        auto &signalOp = function.AddOperation(Opcode::OP_BIND_TENSOR, {}, {signalInner});
        signalOp.SetAttribute(OpAttributeKey::bindTensor, BindTensor(hcclGroupIndex, 1,
            BytesOf(DataType::DT_INT32) * worldSize * SHMEM_SIGNAL_STRIDE * MAX_TILE_NUM));
    }
    return t;
}

void CreateShmemSignal(const char* group, int64_t worldSize, ShmemTensor &t)
{
    ValidateGroup(group);
    ValidateWorldSize(group, worldSize);
    // ShmemTensor t;
    t.group = std::string(group);
    t.worldSize = worldSize;
    // static uint64_t s_index = 0;
    // LOOP("CreateShmemSignal" + std::to_string(s_index++), FunctionType::DYNAMIC_LOOP, index, LoopRange(1)) {
    //     (void)index;
        auto &function = *Program::GetInstance().GetCurrentFunction();
        int32_t hcclGroupIndex = static_cast<int>(CommGroupRecorder::GetInstance().Input(std::string(group)));
        Shape signalShape{worldSize, 1, 1, 8};
        auto signalInner = std::make_shared<LogicalTensor>(function, DataType::DT_INT32, signalShape);
        t.signal = signalInner;
        Program::GetInstance().GetTensorSlotManager()->TensorWrite(t.signal, SlotProperty::SHMEM_TENSOR);
        auto &signalOp = function.AddOperation(Opcode::OP_BIND_TENSOR, {}, {signalInner});
        signalOp.SetAttribute(OpAttributeKey::bindTensor, BindTensor(hcclGroupIndex, 1,
            BytesOf(DataType::DT_INT32) * worldSize * SHMEM_SIGNAL_STRIDE * MAX_TILE_NUM));
    // }
    // return t;
}

template<typename OffsetType>
ShmemTensor ShmemView(const ShmemTensor &operand, const std::vector<int64_t> &shapes, const std::vector<OffsetType> &offsets)
{
    auto data = View(operand.data, shapes, offsets);
    Shape signalShape = operand.signal.GetShape();
    std::copy(shapes.begin(), shapes.end(), signalShape.end() - shapes.size());
    std::vector<OffsetType> signalOffset(operand.signal.GetShape().size() - offsets.size(), 0);
    signalOffset.insert(signalOffset.end(), offsets.begin(), offsets.end());
    auto signal = View(operand.signal, signalShape, signalOffset);
    return ShmemTensor{operand.group, operand.worldSize, data, signal};
}

ShmemTensor ShmemView(const ShmemTensor &operand, const std::vector<int64_t> &shapes, const std::vector<int64_t> &offsets)
{
    return ShmemView<int64_t>(operand, shapes, offsets);
}

ShmemTensor ShmemView(const ShmemTensor &operand, const std::vector<int64_t> &shapes, const std::vector<SymbolicScalar> &offsets)
{
    return ShmemView<SymbolicScalar>(operand, shapes, offsets);
}

ShmemTensor ShmemView(const ShmemTensor &operand, const std::vector<int64_t> &shapes, const std::vector<SymbolicScalar> &newValidShapes, const std::vector<SymbolicScalar> &newOffsets)
{
    auto data = View(operand.data, shapes, newValidShapes, newOffsets);
    Shape signalShape = operand.signal.GetShape();
    std::copy(shapes.begin(), shapes.end(), signalShape.end() - shapes.size());
    std::vector<SymbolicScalar> signalValidShape;
    if (operand.signal.GetValidShape().size() != 0) {
        signalValidShape = operand.signal.GetValidShape();
    } else {
        signalValidShape.insert(signalValidShape.end(), operand.signal.GetShape().begin(), operand.signal.GetShape().end());
    }
    std::copy(newValidShapes.begin(), newValidShapes.end(), signalValidShape.end() - newValidShapes.size());
    std::vector<SymbolicScalar> signalOffset(operand.signal.GetShape().size() - newOffsets.size(), 0);
    signalOffset.insert(signalOffset.end(), newOffsets.begin(), newOffsets.end());
    auto signal = View(operand.signal, signalShape, signalValidShape, signalOffset);
    return ShmemTensor{operand.group, operand.worldSize, data, signal};
}

ShmemTensor ShmemView(const ShmemTensor &operand, const std::vector<int64_t> &shapes, const std::initializer_list<SymbolicScalar> &newOffsets)
{
    return ShmemView(operand, shapes, std::vector<SymbolicScalar>(newOffsets));
}

Tensor ShmemPut(const Tensor &src, const ShmemTensor &dst, const SymbolicScalar &dstRank, AtomicType putOp, const Tensor& pred)
{
    std::unordered_set<DataType> allowedTypes = {DT_INT32, DT_FP32, DT_FP16, DT_BF16};
    ValidateTensor(src, "src", {2}, allowedTypes, {TileOpFormat::TILEOP_ND});
    ValidateTensor(dst.data, "dst", {3}, allowedTypes, {TileOpFormat::TILEOP_ND});
    ValidateTensor(pred, "pred", {2});
    ValidateTiling(Opcode::OP_SHMEM_PUT, src, "src");
    auto &function = *Program::GetInstance().GetCurrentFunction();
    auto out = std::make_shared<LogicalTensor>(function, DT_INT32, pred.GetShape());
    auto &op = function.AddOperation(Opcode::OP_SHMEM_PUT,
        {pred.GetStorage(), src.GetStorage(), dst.data.GetStorage()}, {out});
    ShmemPutAttr distOpAttr;
    distOpAttr.atomicType = putOp;
    distOpAttr.targetRank = dstRank;
    op.SetAttr(OpAttributeKey::distOpAttr, distOpAttr);
    return out;
}

Tensor ShmemStore(const Tensor &src, const ShmemTensor &dst, const SymbolicScalar &dstRank, AtomicType putOp, const Tensor &pred)
{
    auto &function = *Program::GetInstance().GetCurrentFunction();
    auto dummy = std::make_shared<LogicalTensor>(function, DT_INT32, pred.GetShape());
    auto &op = function.AddOperation(Opcode::OP_SHMEM_PUT_UB2GM,
        {src.GetStorage(), dst.data.GetStorage(), pred.GetStorage()}, {dummy});
    ShmemPutAttr distOpAttr;
    distOpAttr.atomicType = putOp;
    distOpAttr.targetRank = dstRank;
    op.SetAttr(OpAttributeKey::distOpAttr, distOpAttr);
    return dummy;
}

Tensor ShmemGet(const ShmemTensor &src, const SymbolicScalar &srcRank, const Tensor& pred, DataType targetDataType)
{
    ValidateTensor(src.data, "src", {3}, {DT_INT32, DT_FP32, DT_FP16, DT_BF16}, {TileOpFormat::TILEOP_ND});
    ValidateTensor(pred, "pred", {2});
    ValidateTiling(Opcode::OP_SHMEM_GET, pred, "pred");
    if (targetDataType == DT_BOTTOM) {
        targetDataType = src.data.GetDataType();
    }
    auto &function = *Program::GetInstance().GetCurrentFunction();
    Shape shape = {src.data.GetShape(1), src.data.GetShape(2)};
    auto out = std::make_shared<LogicalTensor>(function, targetDataType, shape, src.data.Format());
    auto &op = function.AddOperation(Opcode::OP_SHMEM_GET, {pred.GetStorage(), src.data.GetStorage()},
        {out});
    ShmemGetAttr distOpAttr;
    distOpAttr.targetRank = srcRank;
    op.SetAttr(OpAttributeKey::distOpAttr, distOpAttr);
    return out;
}

Tensor ShmemLoad(const ShmemTensor &src, const SymbolicScalar &srcRank, const Tensor& pred, DataType nonShmemDataType)
{
    ValidateTiling(Opcode::OP_SHMEM_GET_GM2UB, pred, "pred");
    if (nonShmemDataType == DT_BOTTOM) {
        nonShmemDataType = src.data.GetDataType();
    }
    Shape shape = {src.data.GetShape(1), src.data.GetShape(2)};
    auto &function = *Program::GetInstance().GetCurrentFunction();
    auto out = std::make_shared<LogicalTensor>(function, nonShmemDataType, shape);
    auto &op = function.AddOperation(Opcode::OP_SHMEM_GET_GM2UB, {pred.GetStorage(), src.data.GetStorage()},
        {out});
    op.SetOpAttribute(std::make_shared<CopyOpAttribute>(OpImmediate::Specified({0, 0}), MEM_UB,
        OpImmediate::Specified({src.data.GetShape(1), src.data.GetShape(2)}),
        OpImmediate::Specified({out->shape[0], out->shape[1]}),
        OpImmediate::Specified(std::vector<SymbolicScalar>{src.data.GetValidShape()[1], src.data.GetValidShape()[2]})));
    function.UpdateTensorDataUsage(op);
    ShmemGetAttr distOpAttr;
    distOpAttr.targetRank = srcRank;
    op.SetAttr(OpAttributeKey::distOpAttr, distOpAttr);
    return out;
}

Tensor ShmemSignal(const ShmemTensor& dst, const SymbolicScalar &dstRank, const SymbolicScalar &notifyRank, int32_t signal, AtomicType sigOp, const Tensor& pred)
{
    ValidateTensor(pred, "pred", {2});
    ValidateTensor(dst.signal, "dst", {4});
    ValidateTiling(Opcode::OP_SHMEM_SIGNAL, pred, "pred");
    auto &function = *Program::GetInstance().GetCurrentFunction();
    Shape signalShape = dst.signal.GetShape();
    signalShape[0] = 1;
    std::vector<SymbolicScalar> signalOffset(signalShape.size(), 0);
    signalOffset[0] = dstRank;
    auto signalTensor = View(dst.signal, signalShape, signalOffset);
    auto out = std::make_shared<LogicalTensor>(function, DT_INT32, pred.GetShape());
    auto &op = function.AddOperation(Opcode::OP_SHMEM_SIGNAL, {pred.GetStorage(), signalTensor.GetStorage()}, {out});
    ShmemSignalAttr distOpAttr;
    distOpAttr.signalValue = signal;
    distOpAttr.atomicType = sigOp;
    distOpAttr.signalStride = SHMEM_SIGNAL_STRIDE;
    distOpAttr.targetRank = notifyRank;
    distOpAttr.worldSize = dst.worldSize;
    op.SetAttr(OpAttributeKey::distOpAttr, distOpAttr);
    return out;
}

Tensor ShmemSignal(const ShmemTensor& dst, const SymbolicScalar &dstRank, int32_t signal, AtomicType sigOp, const Tensor& pred)
{
    return ShmemSignal(dst, dstRank, dstRank, signal, sigOp, pred);
}

Tensor ShmemSignalAll(const ShmemTensor& dst, const SymbolicScalar &dstRank, int32_t signal, AtomicType sigOp, const Tensor& pred)
{
    ValidateTensor(pred, "pred", {2});
    ValidateTensor(dst.signal, "dst", {4});
    ValidateTiling(Opcode::OP_SHMEM_SIGNAL, pred, "pred");
    auto &function = *Program::GetInstance().GetCurrentFunction();
    Shape signalShape = dst.signal.GetShape();
    signalShape[0] = 1;
    std::vector<SymbolicScalar> signalOffset(signalShape.size(), 0);
    signalOffset[0] = dstRank;
    auto signalTensor = View(dst.signal, signalShape, signalOffset);
    auto out = std::make_shared<LogicalTensor>(function, DT_INT32, pred.GetShape());
    auto &op = function.AddOperation(Opcode::OP_SHMEM_SIGNAL, {pred.GetStorage(), signalTensor.GetStorage()},
        {out});
    ShmemSignalAttr distOpAttr;
    distOpAttr.signalValue = signal;
    distOpAttr.atomicType = sigOp;
    distOpAttr.signalStride = SHMEM_SIGNAL_STRIDE;
    distOpAttr.notifyAll = true;
    distOpAttr.worldSize = dst.worldSize;
    distOpAttr.targetRank = 0;
    op.SetAttr(OpAttributeKey::distOpAttr, distOpAttr);
    return out;
}

Tensor ShmemWaitUntil(const ShmemTensor& src, const SymbolicScalar &srcRank, OpType cmp, int32_t cmpValue, bool clearSignal, const Tensor &pred)
{
    ValidateTensor(pred, "pred", {2});
    ValidateTensor(src.signal, "src.signal", {4});
    ValidateTiling(Opcode::OP_SHMEM_WAIT_UNTIL, pred, "pred");
    (void)cmp;
    auto &function = *Program::GetInstance().GetCurrentFunction();
    Shape signalShape = src.signal.GetShape();
    signalShape[0] = 1;
    std::vector<SymbolicScalar> signalOffset(signalShape.size(), 0);
    signalOffset[0] = srcRank;
    auto signalTensor = View(src.signal, signalShape, signalOffset);
    auto out = std::make_shared<LogicalTensor>(function, DT_INT32, pred.GetShape());
    auto &op = function.AddOperation(Opcode::OP_SHMEM_WAIT_UNTIL, {pred.GetStorage(), signalTensor.GetStorage()},
        {out});
    std::vector<int64_t> param = {static_cast<int64_t>(cmpValue),
        static_cast<int64_t>(SHMEM_SIGNAL_STRIDE), static_cast<int64_t>(clearSignal)};
    ShmemWaitUntilAttr distOpAttr;
    distOpAttr.expectedSum = cmpValue;
    distOpAttr.signalStride = SHMEM_SIGNAL_STRIDE;
    distOpAttr.resetSignal = clearSignal;
    distOpAttr.targetRank = GetHcclRankId(src.group);
    op.SetAttr(OpAttributeKey::distOpAttr, distOpAttr);
    return out;
}

Tensor ShmemClearData(const ShmemTensor& src, Tensor &pred)
{
    ValidateTensor(src.data, "src", {3});
    auto& function = *Program::GetInstance().GetCurrentFunction();
    auto out = std::make_shared<LogicalTensor>(function, DT_INT32, Shape{1, 1});
    auto& op = function.AddOperation(Opcode::OP_SHMEM_SET, {pred.GetStorage(), src.data.GetStorage()}, {out});
    ShmemSetAttr distOpAttr;
    distOpAttr.setType = 0;
    distOpAttr.targetRank = GetHcclRankId(src.group);
    op.SetAttr(OpAttributeKey::distOpAttr, distOpAttr);
    return out;
}

Tensor ShmemClearSignal(const ShmemTensor& src, Tensor &pred)
{
    ValidateTensor(src.signal, "src", {4});
    auto& function = *Program::GetInstance().GetCurrentFunction();
    auto out = std::make_shared<LogicalTensor>(function, DT_INT32, Shape{1, 1});
    auto& op = function.AddOperation(Opcode::OP_SHMEM_SET, {pred.GetStorage(), src.signal.GetStorage()}, {out});
    ShmemSetAttr distOpAttr;
    distOpAttr.setType = 1;
    distOpAttr.targetRank = GetHcclRankId(src.group);
    op.SetAttr(OpAttributeKey::distOpAttr, distOpAttr);
    return out;
}

Tensor ShmemBarrier(const ShmemTensor &src, const Tensor &pred)
{
    ShmemSignalAll(src, 0, 1, AtomicType::ADD, pred);
    return ShmemWaitUntil(src, 0, OpType::EQ, src.worldSize, true, pred);
}

void AllGather(const Tensor& predToken, const Tensor& in, ShmemTensor &shmemTensor, Tensor& out)
{
    uint32_t worldSize = shmemTensor.worldSize;
    CHECK(worldSize > 0) << "worldSize should be more than 0.";
    int32_t row = in.GetShape(0);
    int32_t col = in.GetShape(1);
    SymbolicScalar thisRank = GetHcclRankId(shmemTensor.group);
    ValidateParams(predToken, in, out, shmemTensor.data.GetShape(), in.GetDataType());
    ValidateTypeAndShape(shmemTensor.data, out.GetDataType(), {worldSize, row, col});
    ValidateTypeAndShape(shmemTensor.signal, DataType::DT_INT32, {worldSize, worldSize, row, col});
    ValidateTypeAndShape(out, in.GetDataType(), {row * worldSize, col});
    for (uint32_t dynRankId = 0; dynRankId < worldSize; ++dynRankId) {
        auto shmemDataTile = ShmemView(shmemTensor, {1, row, col}, std::vector<SymbolicScalar>{thisRank, 0, 0});
        auto shmemPutOut = ShmemPut(in, shmemDataTile, dynRankId, AtomicType::SET, predToken);
        auto shmemSignalOut = ShmemSignal(shmemDataTile, dynRankId, 1, AtomicType::SET, shmemPutOut);
        auto shmemDataLocal = ShmemView(shmemTensor, {1, row, col}, std::vector<SymbolicScalar>{dynRankId, 0, 0});
        auto waitUntilOut= ShmemWaitUntil(shmemDataLocal, thisRank, OpType::EQ, 1, true, shmemSignalOut);
        auto shmemGetOut= ShmemGet(shmemDataLocal, thisRank, waitUntilOut);
        Assemble(shmemGetOut, {dynRankId * row, 0}, out);
    }
}

void ReduceScatter(const Tensor& predToken, const Tensor& in, ShmemTensor& shmemTensor,
    DistReduceType reduceType, Tensor& out)
{
    (void)reduceType;
    uint32_t worldSize = shmemTensor.worldSize;
    CHECK(worldSize > 0) << "worldSize should be more than 0.";
    int32_t row = in.GetShape(0);
    int32_t col = in.GetShape(1);
    int32_t rowOut = row / worldSize;
    SymbolicScalar thisRank = GetHcclRankId(shmemTensor.group);
    ValidateParams(predToken, in, out, shmemTensor.data.GetShape(), shmemTensor.data.GetDataType(),
        false, true, {DT_INT32, DT_FP32, DT_FP16, DT_BF16});
    ValidateTypeAndShape(shmemTensor.data, ((in.GetDataType() == DT_BF16) || (in.GetDataType() == DT_FP16) ? DT_FP32 :
        out.GetDataType()), {1, rowOut, col});
    ValidateTypeAndShape(shmemTensor.signal, DataType::DT_INT32, {worldSize, 1, rowOut, col});
    ValidateTypeAndShape(out, in.GetDataType(), {rowOut, col});
    for (uint32_t dynRankId = 0; dynRankId < worldSize; ++dynRankId) {
        auto shmemDataTile = ShmemView(shmemTensor, {1, rowOut, col}, std::vector<SymbolicScalar>{0, 0, 0});
        auto inTile = View(in, {rowOut, col}, std::vector<SymbolicScalar>{dynRankId * rowOut, 0});
        auto shmemPutOut = ShmemPut(inTile, shmemDataTile, dynRankId, AtomicType::ADD, predToken);
        ShmemSignal(shmemDataTile, dynRankId, 1, AtomicType::ADD, shmemPutOut);
    }
    auto shmemDataLocal = ShmemView(shmemTensor, {1, rowOut, col}, std::vector<SymbolicScalar>{0, 0, 0});
    auto waitUntilOut = ShmemWaitUntil(shmemDataLocal, thisRank, OpType::EQ, worldSize, true, in);
    out = ShmemGet(shmemDataLocal, thisRank, waitUntilOut, in.GetDataType());
}

void OneShotAllReduce(const Tensor& predToken, const Tensor &in, ShmemTensor& shmemTensor, Tensor& out)
{
    uint32_t worldSize = shmemTensor.worldSize;
    CHECK(worldSize > 0) << "worldSize should be more than 0.";
    int32_t row = in.GetShape(0);
    int32_t col = in.GetShape(1);
    SymbolicScalar thisRank = GetHcclRankId(shmemTensor.group);
    ValidateParams(predToken, in, out, shmemTensor.data.GetShape(), shmemTensor.data.GetDataType(), true, true,
        {DT_INT32, DT_FP32, DT_FP16, DT_BF16});
    ValidateTypeAndShape(shmemTensor.data, ((in.GetDataType() == DT_BF16) || (in.GetDataType() == DT_FP16) ? DT_FP32 :
        out.GetDataType()), {1, row, col});
    ValidateTypeAndShape(shmemTensor.signal, DataType::DT_INT32, {worldSize, 1, row, col});
    for (uint32_t dynRankId = 0; dynRankId < worldSize; ++dynRankId) {
        auto shmemDataTile = ShmemView(shmemTensor, {1, row, col}, std::vector<SymbolicScalar>{0, 0, 0});
        auto shmemPutOut = ShmemPut(in, shmemDataTile, dynRankId, AtomicType::ADD, predToken);
        ShmemSignal(shmemDataTile, dynRankId, 1, AtomicType::ADD, shmemPutOut);
    }
    auto shmemDataLocal = ShmemView(shmemTensor, {1, row, col}, std::vector<SymbolicScalar>{0, 0, 0});
    auto waitUntilOut = ShmemWaitUntil(shmemDataLocal, thisRank, OpType::EQ, worldSize, true, in);
    out = ShmemGet(shmemDataLocal, thisRank, waitUntilOut, in.GetDataType());
}

void TwoShotAllReduce(const Tensor& predToken, const Tensor &in, ShmemTensor& shmemTensor, Tensor& out)
{
    uint32_t worldSize = shmemTensor.worldSize;
    CHECK(worldSize > 0) << "worldSize should be more than 0.";
    int32_t row = in.GetShape(0);
    int32_t col = in.GetShape(1);
    int32_t rowPerRank = row / worldSize;
    SymbolicScalar thisRank = GetHcclRankId(shmemTensor.group);
    ValidateParams(predToken, in, out, shmemTensor.data.GetShape(), shmemTensor.data.GetDataType(), true, true,
        {DT_INT32, DT_FP32, DT_FP16, DT_BF16});
    ValidateTypeAndShape(shmemTensor.data, ((in.GetDataType() == DT_BF16) || (in.GetDataType() == DT_FP16) ? DT_FP32 :
        out.GetDataType()), {worldSize, rowPerRank, col});
    ValidateTypeAndShape(shmemTensor.data, DataType::DT_INT32, {worldSize, worldSize, rowPerRank, col});
    for (uint32_t dynRankId = 0; dynRankId < worldSize; ++dynRankId) {
        auto shmemDataTile = ShmemView(shmemTensor, {1, rowPerRank, col}, std::vector<SymbolicScalar>{0, 0, 0});
        auto inTile = View(in, {rowPerRank, col}, std::vector<SymbolicScalar>{dynRankId * rowPerRank, 0});
        auto shmemPutOut = ShmemPut(inTile, shmemDataTile, dynRankId, AtomicType::ADD, predToken);
        ShmemSignalAll(shmemDataTile, dynRankId, 1, AtomicType::ADD, shmemPutOut);
        auto waitUntilOut = ShmemWaitUntil(shmemDataTile, dynRankId, OpType::EQ, worldSize, true, in);
        auto tmp = ShmemGet(shmemDataTile, dynRankId, waitUntilOut, in.GetDataType());
        Assemble(tmp, {rowPerRank * dynRankId, 0}, out);
    }
}

void CreateShmemData(const char* group, int64_t worldSize, DataType dataType,
    const Shape &shape, Tensor &shmemTensor, uint64_t memType)
{
    ValidateGroup(group);
    ValidateWorldSize(group, worldSize);
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
    ValidateGroup(group);
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
    op.SetAttribute(OpAttributeKey::bindTensor, BindTensor(hcclGroupIndex, 1,
        BytesOf(DataType::DT_INT32) * worldSize * SHMEM_SIGNAL_STRIDE * MAX_TILE_NUM));
}
}   // namespace npu::tile_fwk::Distributed
