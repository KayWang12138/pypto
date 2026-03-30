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
#include "interface/utils/distributed_error.h"

namespace npu::tile_fwk::Distributed {
void ValidateTensor(const Tensor& tensor, const std::string& tensorDesc,
    const std::unordered_set<size_t>& allowedDims = {},
    const std::unordered_set<DataType>& allowedTypes = {},
    const std::unordered_set<TileOpFormat>& allowedFormats = {},
    const Shape& expectShape = {});

void ValidateShmemTensor(const ShmemTensor& t, bool hasData, bool hasSignal);

// OneShotAllReduce_v9: SHMEM-only, chunked, tunable signal-to-data ratio, no Communicator class.
// Arguments:
//   predToken: dependency token
//   in: input tensor (row x col)
//   shmemTensor: shared memory tensor (1, row, col)
//   out: output tensor (row x col)
//   payloadChunkCount: number of chunks to split input rows
//   chunksPerSignal: number of chunks per signal (signal-to-data ratio)
void OneShotAllReduce_v9(const Tensor& predToken, const Tensor& in, ShmemTensor& shmemTensor, Tensor& out,
                         uint32_t payloadChunkCount, uint32_t chunksPerSignal)
{
    ValidateShmemTensor(shmemTensor, true, true);
    ValidateTensor(predToken, "predToken", {2});
    ValidateTensor(in, "in", {predToken.Dim()});
    ASSERT(payloadChunkCount > 0) << "payloadChunkCount must be > 0";
    ASSERT(chunksPerSignal > 0) << "chunksPerSignal must be > 0";
    int32_t row = in.GetShape(0);
    int32_t col = in.GetShape(1);
    ValidateTensor(shmemTensor.data, "shmemTensor.data", {}, {}, {in.Format()}, {1, row, col});
    ValidateTensor(out, "out", {}, {in.GetDataType()}, {in.Format()}, in.GetShape());
    ASSERT(static_cast<int64_t>(payloadChunkCount) <= row)
        << "payloadChunkCount must be <= row dimension (" << row << ")";
    ASSERT(chunksPerSignal <= payloadChunkCount)
        << "chunksPerSignal must be <= payloadChunkCount, but got "
        << chunksPerSignal << " > " << payloadChunkCount;

    OneShotCommunicatorV4 comm(shmemTensor, payloadChunkCount, chunksPerSignal);
    OneShotAllReduce_v8(predToken, in, comm);

    for (uint32_t groupId = 0; groupId < comm.SignalGroupCount(); ++groupId) {
        auto waitToken = comm.WaitGroup(in, groupId);
        uint32_t begin = comm.GroupBeginChunk(groupId);
        uint32_t gSize = comm.GroupSize(groupId);
        for (uint32_t local = 0; local < gSize; ++local) {
            uint32_t chunkId = begin + local;
            auto reducedChunk = comm.PullChunk(waitToken, chunkId, in.GetDataType());
            Assemble(reducedChunk, {comm.ChunkStartRow(chunkId), 0}, out);
        }
    }
}

void ValidateGroup(const char* group)
{
    ASSERT(DistributedErrorCode::INVALID_GROUP_NAME, group != nullptr) << "\"group\" cannot be nullptr";
    auto groupLen = std::string(group).size();
    ASSERT(DistributedErrorCode::INVALID_GROUP_NAME, (groupLen >= 1) && (groupLen < 128)) << "The length of \"group\" only supports [1, 128), but got "
        << groupLen;
}

void ValidateTiling(const Opcode& opCode, const Tensor& target, const std::string& tensorDesc)
{
    const auto vecTile = TileShape::Current().GetVecTile();
    ASSERT(DistributedErrorCode::INVALID_TILE_SHAPE, vecTile.valid()) << ToString(opCode) <<
        ": vecTile every shape and dim should > 0, but got:" << ToString(vecTile.tile);
    ASSERT(DistributedErrorCode::INVALID_TILE_DIM, target.Dim() == vecTile.size()) <<
        ToString(opCode) << " dim of vectile shape must be equal to " << std::to_string(target.Dim()) <<
        ", which is same as " << tensorDesc << ", but got " << vecTile.size();
}

void ValidateDataType(const Tensor& tensor, const std::string& tensorDesc, const std::unordered_set<DataType>& allowedTypes)
{
    auto dataType = tensor.GetDataType();
    ASSERT(DistributedErrorCode::INVALID_TENSOR_DTYPE, allowedTypes.empty() || allowedTypes.count(dataType)) <<
        "Invalid data type: " << tensorDesc << " data type only support " << ToString(allowedTypes) <<
        ", but got:" << ToString(dataType);
}

void ValidateDim(const Tensor& tensor, const std::string& tensorDesc, const std::unordered_set<size_t>& allowedDims)
{
    const auto& shape = tensor.GetShape();
    ASSERT(DistributedErrorCode::INVALID_TENSOR_DIM, allowedDims.empty() || allowedDims.count(shape.size())) << "Invalid dimensional: " << tensorDesc <<
        " dimensional must be " << ToString(allowedDims) << ", but got dimensional=" << shape.size();
}

void ValidateFormat(const Tensor& tensor, const std::string& tensorDesc,
    const std::unordered_set<TileOpFormat>& allowedFormats = {TileOpFormat::TILEOP_ND})
{
    ASSERT(DistributedErrorCode::INVALID_TENSOR_FORMAT, allowedFormats.empty() || allowedFormats.count(tensor.Format())) <<
        "Invalid format: " << tensorDesc << " only support ND format, but got NZ format";
}

void ValidateShape(const Tensor& tensor, const std::string& tensorDesc, const Shape& expectShape)
{
    const auto& shape = tensor.GetShape();
    ASSERT(DistributedErrorCode::INVALID_TENSOR_SHAPE,
        std::all_of(shape.begin(), shape.end(), [](int64_t val){ return val > 0; })) << "Invaild shape value: " <<
        tensorDesc << ", all shape must be greater than 0, but got " << ToString(shape);
    ASSERT(DistributedErrorCode::INVALID_TENSOR_SHAPE, expectShape.empty() || expectShape == shape) <<
        "Invalid shape: " << tensorDesc << " expect:" << ToString(expectShape) << ", but got: " << ToString(shape);
}

void ValidateTensor(const Tensor& tensor, const std::string& tensorDesc,
    const std::unordered_set<size_t>& allowedDims,
    const std::unordered_set<DataType>& allowedTypes,
    const std::unordered_set<TileOpFormat>& allowedFormats,
    const Shape& expectShape)
{
    ValidateDim(tensor, tensorDesc, allowedDims);
    ValidateDataType(tensor, tensorDesc, allowedTypes);
    ValidateFormat(tensor, tensorDesc, allowedFormats);
    ValidateShape(tensor, tensorDesc, expectShape);
}

void ValidateOpType(OpType cmp, const std::unordered_set<OpType>& allowedOpTypes)
{
    ASSERT(DistributedErrorCode::INVALID_OP_TYPE, allowedOpTypes.empty() || allowedOpTypes.count(cmp)) <<
        "Invaild OP type, only support:" << ToString(allowedOpTypes) << ", but got:" << ToString(cmp);
}

void ValidateShmemTensor(const ShmemTensor& t, bool hasData = false, bool hasSignal = false) {
    static std::unordered_map<std::string, int64_t> groupWorldSizeMap;
    ValidateGroup(t.group.c_str());
    auto groupWorldSize = groupWorldSizeMap.find(t.group);
    if (groupWorldSize == groupWorldSizeMap.end()) {
        ASSERT(DistributedErrorCode::INVALID_WORLD_SIZE, t.worldSize > 0) << "Invalid world size for group " <<
            t.group << ": world size must be greather than 0" << ", but got " << t.worldSize;
        groupWorldSizeMap.emplace(t.group, t.worldSize);
    } else {
        ASSERT(DistributedErrorCode::INVALID_WORLD_SIZE, t.worldSize == groupWorldSize->second) << "WorldSize mismatch for group " << t.group
            << ": expected " << groupWorldSize->second << ", but got " << t.worldSize;
    }
    if (hasData) {
        ASSERT(DistributedErrorCode::INVALID_SHMEM_TENSOR, t.data.GetStorage() != nullptr) <<
            "shmem tensor's data should not be empty";
    }
    if (hasSignal) {
        ASSERT(DistributedErrorCode::INVALID_SHMEM_TENSOR, t.signal.GetStorage() != nullptr) <<
            "shmem tensor's signal should not be empty";
    }
}

ShmemTensor CreateShmemTensor(const char* group, int64_t worldSize, DataType dataType, const Shape& shape)
{
    ShmemTensor t;
    static uint64_t s_index = 0;
    LOOP("CreateShmemTensor" + std::to_string(s_index++), FunctionType::DYNAMIC_LOOP, index, LoopRange(1)) {
        (void)index;
        CreateShmemTensor(group, worldSize, dataType, shape, t);
    }
    return t;
}

void CreateShmemTensor(const char* group, int64_t worldSize, DataType dataType, const Shape& shape, ShmemTensor& t)
{
    ValidateGroup(group);

    t.group = std::string(group);
    t.worldSize = worldSize;
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

    ValidateShmemTensor(t, true, true);
}

ShmemTensor CreateShmemSignal(const char* group, int64_t worldSize)
{
    ShmemTensor t;
    static uint64_t s_index = 0;
    LOOP("CreateShmemSignal" + std::to_string(s_index++), FunctionType::DYNAMIC_LOOP, index, LoopRange(1)) {
        (void)index;
        CreateShmemSignal(group, worldSize, t);
    }
    return t;
}

void CreateShmemSignal(const char* group, int64_t worldSize, ShmemTensor& t)
{
    ValidateGroup(group);
    t.group = std::string(group);
    t.worldSize = worldSize;
    auto &function = *Program::GetInstance().GetCurrentFunction();
    int32_t hcclGroupIndex = static_cast<int>(CommGroupRecorder::GetInstance().Input(std::string(group)));
    Shape signalShape{worldSize, 1, 1, 8};
    auto signalInner = std::make_shared<LogicalTensor>(function, DataType::DT_INT32, signalShape);
    t.signal = signalInner;
    Program::GetInstance().GetTensorSlotManager()->TensorWrite(t.signal, SlotProperty::SHMEM_TENSOR);
    auto &signalOp = function.AddOperation(Opcode::OP_BIND_TENSOR, {}, {signalInner});
    signalOp.SetAttribute(OpAttributeKey::bindTensor, BindTensor(hcclGroupIndex, 1,
        BytesOf(DataType::DT_INT32) * worldSize * SHMEM_SIGNAL_STRIDE * MAX_TILE_NUM));
    ValidateShmemTensor(t, false, true);
}



template<typename OffsetType, bool HasValidShape = false>
ShmemTensor ShmemViewImpl(const ShmemTensor& operand, const std::vector<int64_t>& shapes,
    const std::vector<OffsetType>& offsets, const std::vector<SymbolicScalar>& validShapes = {})
{
    auto data = [&]() {
        if constexpr (HasValidShape) {
            return View(operand.data, shapes, validShapes, offsets);
        } else {
            return View(operand.data, shapes, offsets);
        }
    }();

    Shape signalShape = operand.signal.GetShape();
    ASSERT(DistributedErrorCode::INVALID_SHMEM_VIEW_PARAM, signalShape.size() > shapes.size()) <<
        "input shape dim should smaller than signal dim, input shape dim:" <<
        signalShape.size() << ", signal dim:" << shapes.size();
    ASSERT(DistributedErrorCode::INVALID_SHMEM_VIEW_PARAM, signalShape.size() > offsets.size()) <<
        "input offsets dim should smaller than signal dim, input shape dim:" <<
        offsets.size() << ", signal dim:" << shapes.size();
    std::copy(shapes.begin(), shapes.end(), signalShape.end() - shapes.size());
    std::vector<OffsetType> signalOffset(operand.signal.GetShape().size(), 0);
    std::copy(offsets.begin(), offsets.end(), signalOffset.end() - offsets.size());
    auto signal = View(operand.signal, signalShape, signalOffset);
    return ShmemTensor{operand.group, operand.worldSize, data, signal};
}

ShmemTensor ShmemView(const ShmemTensor &operand, const std::vector<int64_t> &shapes, const std::vector<int64_t> &offsets)
{
    return ShmemViewImpl<int64_t>(operand, shapes, offsets);
}

ShmemTensor ShmemView(const ShmemTensor &operand, const std::vector<int64_t> &shapes, const std::vector<SymbolicScalar> &offsets)
{
    return ShmemViewImpl<SymbolicScalar>(operand, shapes, offsets);
}

ShmemTensor ShmemView(const ShmemTensor &operand, const std::vector<int64_t> &shapes,
    const std::vector<SymbolicScalar> &newValidShapes, const std::vector<SymbolicScalar> &newOffsets)
{
    return ShmemViewImpl<SymbolicScalar, true>(operand, shapes, newOffsets, newValidShapes);
}

ShmemTensor ShmemView(const ShmemTensor &operand, const std::vector<int64_t> &shapes,
    const std::initializer_list<SymbolicScalar> &newOffsets)
{
    return ShmemView(operand, shapes, std::vector<SymbolicScalar>(newOffsets));
}

static Tensor ShmemPutImpl(const Tensor& src, const ShmemTensor& dst, const SymbolicScalar& dstRank,
    AtomicType putOp, const Tensor& pred, bool isUb2Gm)
{
    ValidateShmemTensor(dst, true);
    std::unordered_set<DataType> allowedTypes = {DT_INT32, DT_FP32, DT_FP16, DT_BF16};
    ValidateTensor(src, "src", {2}, allowedTypes, {TileOpFormat::TILEOP_ND});
    auto shmemDataType = ((putOp == AtomicType::ADD) &&
        ((src.GetDataType() == DT_BF16) || (src.GetDataType() == DT_FP16))) ? DT_FP32 : src.GetDataType();
    ValidateTensor(dst.data, "dst", {3}, {shmemDataType}, {TileOpFormat::TILEOP_ND}, {1, src.GetShape(0), src.GetShape(1)});
    ValidateTensor(pred, "pred", {2});
    ValidateTiling(isUb2Gm ? Opcode::OP_SHMEM_PUT_UB2GM : Opcode::OP_SHMEM_PUT, src, "src");
    auto& function = *Program::GetInstance().GetCurrentFunction();
    auto out = std::make_shared<LogicalTensor>(function, DT_INT32, pred.GetShape());
    auto& op = isUb2Gm ?
        function.AddOperation(Opcode::OP_SHMEM_PUT_UB2GM, {src.GetStorage(), dst.data.GetStorage(), pred.GetStorage()}, {out}) :
        function.AddOperation(Opcode::OP_SHMEM_PUT, {pred.GetStorage(), src.GetStorage(), dst.data.GetStorage()}, {out});
    ShmemPutAttr distOpAttr;
    distOpAttr.atomicType = putOp;
    distOpAttr.ownerRank = dstRank;
    op.SetAttr(OpAttributeKey::distOpAttr, distOpAttr);
    return out;
}

Tensor ShmemPut(const Tensor &src, const ShmemTensor &dst, const SymbolicScalar &dstRank, AtomicType putOp,
    const Tensor& pred)
{
    return ShmemPutImpl(src, dst, dstRank, putOp, pred, false);
}

Tensor ShmemStore(const Tensor& src, const ShmemTensor& dst, const SymbolicScalar& dstRank, AtomicType putOp, const Tensor& pred)
{
    return ShmemPutImpl(src, dst, dstRank, putOp, pred, true);
}

Tensor ShmemGet(const ShmemTensor& src, const SymbolicScalar& srcRank, const Tensor& pred, DataType targetDataType)
{
    ValidateShmemTensor(src, true);
    ValidateTensor(src.data, "src.data", {3}, {DT_INT32, DT_FP32, DT_FP16, DT_BF16}, {TileOpFormat::TILEOP_ND},
        {1, src.data.GetShape(1), src.data.GetShape(2)});
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
    distOpAttr.ownerRank = srcRank;
    op.SetAttr(OpAttributeKey::distOpAttr, distOpAttr);
    return out;
}

Tensor ShmemLoad(const ShmemTensor& src, const SymbolicScalar& srcRank, const Tensor& pred, DataType nonShmemDataType)
{
    ValidateShmemTensor(src, true);
    ValidateTensor(src.data, "src.data", {3}, {DT_INT32, DT_FP32, DT_FP16, DT_BF16}, {TileOpFormat::TILEOP_ND},
        {1, src.data.GetShape(1), src.data.GetShape(2)});
    ValidateTensor(pred, "pred", {2});
    ValidateTiling(Opcode::OP_SHMEM_GET_GM2UB, pred, "pred");
    if (nonShmemDataType == DT_BOTTOM) {
        nonShmemDataType = src.data.GetDataType();
    }
    Shape shape = {src.data.GetShape(1), src.data.GetShape(2)};
    auto &function = *Program::GetInstance().GetCurrentFunction();
    auto out = std::make_shared<LogicalTensor>(function, nonShmemDataType, shape);
    auto &op = function.AddOperation(Opcode::OP_SHMEM_GET_GM2UB, {pred.GetStorage(), src.data.GetStorage()},
        {out});
    if (src.data.GetValidShape().size() != 0) {
        op.SetOpAttribute(std::make_shared<CopyOpAttribute>(OpImmediate::Specified({0, 0}), MEM_UB,
            OpImmediate::Specified({src.data.GetShape(1), src.data.GetShape(2)}),
            OpImmediate::Specified({out->shape[0], out->shape[1]}),
            OpImmediate::Specified(std::vector<SymbolicScalar>{src.data.GetValidShape()[1], src.data.GetValidShape()[2]})));
        function.UpdateTensorDataUsage(op);
    }
    ShmemGetAttr distOpAttr;
    distOpAttr.ownerRank = srcRank;
    op.SetAttr(OpAttributeKey::distOpAttr, distOpAttr);
    return out;
}

static Tensor ShmemSignalImpl(const ShmemTensor& src, const SymbolicScalar &srcRank, const SymbolicScalar &targetRank,
    int32_t signal, AtomicType sigOp, const Tensor& pred, bool notifyAll = false)
{
    ValidateShmemTensor(src, false, true);
    ValidateTensor(pred, "pred", {2});
    ValidateTensor(src.signal, "dst", {4});
    ValidateTiling(Opcode::OP_SHMEM_SIGNAL, pred, "pred");
    auto &function = *Program::GetInstance().GetCurrentFunction();
    Shape signalShape = src.signal.GetShape();
    signalShape[0] = 1;
    std::vector<SymbolicScalar> signalOffset(signalShape.size(), 0);
    signalOffset[0] = srcRank;
    auto signalTensor = View(src.signal, signalShape, signalOffset);
    auto out = std::make_shared<LogicalTensor>(function, DT_INT32, pred.GetShape());
    auto &op = function.AddOperation(Opcode::OP_SHMEM_SIGNAL, {pred.GetStorage(), signalTensor.GetStorage()}, {out});
    ShmemSignalAttr distOpAttr;
    distOpAttr.signalValue = signal;
    distOpAttr.atomicType = sigOp;
    distOpAttr.signalStride = SHMEM_SIGNAL_STRIDE;
    distOpAttr.notifyAll = notifyAll;
    distOpAttr.worldSize = src.worldSize;
    distOpAttr.ownerRank = targetRank;
    op.SetAttr(OpAttributeKey::distOpAttr, distOpAttr);
    return out;
}

Tensor ShmemSignal(const ShmemTensor& src, const SymbolicScalar &srcRank, const SymbolicScalar &targetRank, int32_t signal, AtomicType sigOp, const Tensor& pred)
{
    return ShmemSignalImpl(src, srcRank, targetRank, signal, sigOp, pred);
}

Tensor ShmemSignalAll(const ShmemTensor& src, const SymbolicScalar &srcRank, int32_t signal, AtomicType sigOp, const Tensor& pred)
{
    return ShmemSignalImpl(src, srcRank, 0, signal, sigOp, pred, true);
}

Tensor ShmemWaitUntil(const ShmemTensor& src, const SymbolicScalar &srcRank, OpType cmp, int32_t cmpValue, bool clearSignal, const Tensor &pred)
{
    ValidateOpType(cmp, {OpType::EQ});
    ValidateShmemTensor(src, false, true);
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
    ShmemWaitUntilAttr distOpAttr;
    distOpAttr.expectedSum = cmpValue;
    distOpAttr.signalStride = SHMEM_SIGNAL_STRIDE;
    distOpAttr.resetSignal = clearSignal;
    distOpAttr.ownerRank = GetHcclRankId(src.group);
    op.SetAttr(OpAttributeKey::distOpAttr, distOpAttr);
    return out;
}

static Tensor ShmemClearImpl(const ShmemTensor& src, Tensor &pred, bool clearData)
{
    if (clearData) {
        ValidateShmemTensor(src, true);
        ValidateTensor(src.data, "src", {3});
    } else {
        ValidateShmemTensor(src, false, true);
        ValidateTensor(src.signal, "src", {4});
    }
    auto& function = *Program::GetInstance().GetCurrentFunction();
    auto out = std::make_shared<LogicalTensor>(function, DT_INT32, Shape{1, 1});
    auto& op = function.AddOperation(Opcode::OP_SHMEM_SET,
        {pred.GetStorage(), clearData ? src.data.GetStorage() : src.signal.GetStorage()}, {out});
    ShmemSetAttr distOpAttr;
    distOpAttr.setType = clearData ? 0 : 1;
    distOpAttr.ownerRank = GetHcclRankId(src.group);
    op.SetAttr(OpAttributeKey::distOpAttr, distOpAttr);
    return out;
}

Tensor ShmemClearData(const ShmemTensor& src, Tensor &pred)
{
    return ShmemClearImpl(src, pred, true);
}

Tensor ShmemClearSignal(const ShmemTensor& src, Tensor &pred)
{
    return ShmemClearImpl(src, pred, false);
}

Tensor ShmemBarrier(const ShmemTensor &src, const Tensor &pred)
{
    ShmemSignalAll(src, 0, 1, AtomicType::ADD, pred);
    return ShmemWaitUntil(src, 0, OpType::EQ, src.worldSize, true, pred);
}

void AllGather(const Tensor& predToken, const Tensor& in, ShmemTensor &shmemTensor, Tensor& out)
{
    ValidateShmemTensor(shmemTensor, true, true);
    ValidateTensor(predToken, "predToken", {2});
    ValidateTensor(in, "in", {predToken.Dim()});
    uint32_t worldSize = shmemTensor.worldSize;
    int32_t row = in.GetShape(0);
    int32_t col = in.GetShape(1);
    ValidateTensor(shmemTensor.data, "shmemTensor.data", {}, {in.GetDataType()}, {in.Format()}, {worldSize, row, col});
    ValidateTensor(out, "out", {}, {in.GetDataType()}, {in.Format()}, {row * worldSize, col});
    SymbolicScalar thisRank = GetHcclRankId(shmemTensor.group);
    for (uint32_t dynRankId = 0; dynRankId < worldSize; ++dynRankId) {
        auto shmemDataTile = ShmemView(shmemTensor, {1, row, col}, std::vector<SymbolicScalar>{thisRank, 0, 0});
        auto shmemPutOut = ShmemPut(in, shmemDataTile, dynRankId, AtomicType::SET, predToken);
        auto shmemSignalOut = ShmemSignal(shmemDataTile, dynRankId, dynRankId, 1, AtomicType::SET, shmemPutOut);
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
    ValidateShmemTensor(shmemTensor, true, true);
    ValidateTensor(predToken, "predToken", {2});
    ValidateTensor(in, "in", {predToken.Dim()});
    uint32_t worldSize = shmemTensor.worldSize;
    int32_t row = in.GetShape(0);
    int32_t col = in.GetShape(1);
    int32_t rowOut = row / worldSize;
    SymbolicScalar thisRank = GetHcclRankId(shmemTensor.group);
    ValidateTensor(shmemTensor.data, "shmemTensor.data", {}, {}, {in.Format()}, {1, rowOut, col});
    ValidateTensor(out, "out", {}, {in.GetDataType()}, {in.Format()}, {rowOut, col});
    for (uint32_t dynRankId = 0; dynRankId < worldSize; ++dynRankId) {
        auto shmemDataTile = ShmemView(shmemTensor, {1, rowOut, col}, std::vector<SymbolicScalar>{0, 0, 0});
        auto inTile = View(in, {rowOut, col}, std::vector<SymbolicScalar>{dynRankId * rowOut, 0});
        auto shmemPutOut = ShmemPut(inTile, shmemDataTile, dynRankId, AtomicType::ADD, predToken);
        ShmemSignal(shmemDataTile, dynRankId, dynRankId, 1, AtomicType::ADD, shmemPutOut);
    }
    auto shmemDataLocal = ShmemView(shmemTensor, {1, rowOut, col}, std::vector<SymbolicScalar>{0, 0, 0});
    auto waitUntilOut = ShmemWaitUntil(shmemDataLocal, thisRank, OpType::EQ, worldSize, true, in);
    out = ShmemGet(shmemDataLocal, thisRank, waitUntilOut, in.GetDataType());
}

void OneShotAllReduce(const Tensor& predToken, const Tensor &in, ShmemTensor& shmemTensor, Tensor& out)
{
    ValidateShmemTensor(shmemTensor, true, true);
    ValidateTensor(predToken, "predToken", {2});
    ValidateTensor(in, "in", {predToken.Dim()});
    uint32_t worldSize = shmemTensor.worldSize;
    int32_t row = in.GetShape(0);
    int32_t col = in.GetShape(1);
    SymbolicScalar thisRank = GetHcclRankId(shmemTensor.group);
    ValidateTensor(shmemTensor.data, "shmemTensor.data", {}, {}, {in.Format()}, {1, row, col});
    ValidateTensor(out, "out", {}, {in.GetDataType()}, {in.Format()}, in.GetShape());
    for (uint32_t dynRankId = 0; dynRankId < worldSize; ++dynRankId) {
        auto shmemDataTile = ShmemView(shmemTensor, {1, row, col}, std::vector<SymbolicScalar>{0, 0, 0});
        auto shmemPutOut = ShmemPut(in, shmemDataTile, dynRankId, AtomicType::ADD, predToken);
        ShmemSignal(shmemDataTile, dynRankId, dynRankId, 1, AtomicType::ADD, shmemPutOut);
    }
    auto shmemDataLocal = ShmemView(shmemTensor, {1, row, col}, std::vector<SymbolicScalar>{0, 0, 0});
    auto waitUntilOut = ShmemWaitUntil(shmemDataLocal, thisRank, OpType::EQ, worldSize, true, in);
    out = ShmemGet(shmemDataLocal, thisRank, waitUntilOut, in.GetDataType());
}

void TwoShotAllReduce(const Tensor& predToken, const Tensor &in, ShmemTensor& shmemTensor, Tensor& out)
{
    ValidateShmemTensor(shmemTensor, true, true);
    ValidateTensor(predToken, "predToken", {2});
    ValidateTensor(in, "in", {predToken.Dim()});
    uint32_t worldSize = shmemTensor.worldSize;
    int32_t row = in.GetShape(0);
    int32_t col = in.GetShape(1);
    int32_t rowPerRank = row / worldSize;
    ValidateTensor(shmemTensor.data, "shmemTensor.data", {}, {}, {in.Format()}, {worldSize, rowPerRank, col});
    ValidateTensor(out, "out", {}, {in.GetDataType()}, {in.Format()}, in.GetShape());
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

// OneShotAllReduce_v2: Same algorithm and signature as OneShotAllReduce.
//
// Mirrors the original's ShmemView/ShmemPut/ShmemSignal/ShmemWaitUntil/ShmemGet
// pattern but names the shared tile view as a variable to reduce repetition.
void OneShotAllReduce_v2(const Tensor& predToken, const Tensor& in, ShmemTensor& shmemTensor, Tensor& out)
{
    ValidateShmemTensor(shmemTensor, true, true);
    ValidateTensor(predToken, "predToken", {2});
    ValidateTensor(in, "in", {predToken.Dim()});
    uint32_t worldSize = shmemTensor.worldSize;
    int32_t row = in.GetShape(0);
    int32_t col = in.GetShape(1);
    SymbolicScalar thisRank = GetHcclRankId(shmemTensor.group);
    ValidateTensor(shmemTensor.data, "shmemTensor.data", {}, {}, {in.Format()}, {1, row, col});
    ValidateTensor(out, "out", {}, {in.GetDataType()}, {in.Format()}, in.GetShape());
    auto shmemDataTile = ShmemView(shmemTensor, {1, row, col}, std::vector<SymbolicScalar>{0, 0, 0});

    // Phase 1: Scatter — every rank puts its full input to all targets
    for (uint32_t dynRankId = 0; dynRankId < worldSize; ++dynRankId) {
        auto putOut = ShmemPut(in, shmemDataTile, dynRankId, AtomicType::ADD, predToken);
        ShmemSignal(shmemDataTile, dynRankId, dynRankId, 1, AtomicType::ADD, putOut);
    }

    // Phase 2: Wait & Gather — read this rank's fully-reduced result
    auto waitUntilOut = ShmemWaitUntil(shmemDataTile, thisRank, OpType::EQ, worldSize, true, in);
    out = ShmemGet(shmemDataTile, thisRank, waitUntilOut, in.GetDataType());
}

// OneShotAllReduce_v3: Uses an OneShotCommunicator to hide shmem layout details.
//
// Works with rank IDs instead of raw ShmemView() calls.
// Emitted IR is identical to OneShotAllReduce / OneShotAllReduce_v2.
void OneShotAllReduce_v3(const Tensor& predToken, const Tensor& in, ShmemTensor& shmemTensor, Tensor& out)
{
    ValidateShmemTensor(shmemTensor, true, true);
    ValidateTensor(predToken, "predToken", {2});
    ValidateTensor(in, "in", {predToken.Dim()});
    ValidateTensor(shmemTensor.data, "shmemTensor.data", {}, {}, {in.Format()}, {1, in.GetShape(0), in.GetShape(1)});
    ValidateTensor(out, "out", {}, {in.GetDataType()}, {in.Format()}, in.GetShape());
    OneShotCommunicator comm(shmemTensor);

    // Phase 1: Scatter to all ranks with atomic ADD
    for (uint32_t dynRankId = 0; dynRankId < comm.WorldSize(); ++dynRankId) {
        comm.Put(predToken, in, dynRankId, AtomicType::ADD);
    }

    // Phase 2: Gather — wait for all contributions, read reduced result
    out = comm.WaitAndGet(in);
}

// OneShotAllReduce_v4: like v3 but with OneShotCommunicatorV2 (three-phase API).
// Same IR as v2/v3.
void OneShotAllReduce_v4(const Tensor& predToken, const Tensor& in, ShmemTensor& shmemTensor, Tensor& out)
{
    ValidateShmemTensor(shmemTensor, true, true);
    ValidateTensor(predToken, "predToken", {2});
    ValidateTensor(in, "in", {predToken.Dim()});
    ValidateTensor(shmemTensor.data, "shmemTensor.data", {}, {}, {in.Format()}, {1, in.GetShape(0), in.GetShape(1)});
    ValidateTensor(out, "out", {}, {in.GetDataType()}, {in.Format()}, in.GetShape());
    OneShotCommunicatorV2 comm(shmemTensor);

    // Phase 1: Scatter to all ranks with atomic ADD
    for (uint32_t dynRankId = 0; dynRankId < comm.WorldSize(); ++dynRankId) {
        comm.Put(predToken, in, dynRankId, AtomicType::ADD);
    }

    // Phase 2: Gather — wait for all contributions, read reduced result
    out = comm.WaitAndGet(in);
}

// OneShotAllReduce_v5: scatter + wait; Pull is the caller's responsibility.
// Enables overlap between scatter and local compute.
Tensor OneShotAllReduce_v5(const Tensor& predToken, const Tensor& in, ShmemTensor& shmemTensor)
{
    ValidateShmemTensor(shmemTensor, true, true);
    ValidateTensor(predToken, "predToken", {2});
    ValidateTensor(in, "in", {predToken.Dim()});
    uint32_t worldSize = shmemTensor.worldSize;
    int32_t row = in.GetShape(0);
    int32_t col = in.GetShape(1);
    SymbolicScalar thisRank = GetHcclRankId(shmemTensor.group);
    ValidateTensor(shmemTensor.data, "shmemTensor.data", {}, {}, {in.Format()}, {1, row, col});
    auto shmemDataTile = ShmemView(shmemTensor, {1, row, col}, std::vector<SymbolicScalar>{0, 0, 0});

    // Phase 1: Scatter — broadcast input to all ranks with atomic ADD
    for (uint32_t dynRankId = 0; dynRankId < worldSize; ++dynRankId) {
        auto putOut = ShmemPut(in, shmemDataTile, dynRankId, AtomicType::ADD, predToken);
        ShmemSignal(shmemDataTile, dynRankId, dynRankId, 1, AtomicType::ADD, putOut);
    }

    // Phase 2: Wait — use predToken (not the Put output) to match base IR ordering
    return ShmemWaitUntil(shmemDataTile, thisRank, OpType::EQ, worldSize, true, predToken);
}

// TwoShotAllReduce_v2: Same AllReduce semantics as TwoShotAllReduce(ShmemTensor&).
//
// Inline implementation of the symmetric-memory two-shot protocol using the
// canonical ShmemTensor API. Reduces repetition with a ShmemView helper:
//   - dataTile view extracted once per iteration
//   - inputChunk sliced directly from the input tensor
void TwoShotAllReduce_v2(const Tensor& predToken, const Tensor& in,
    ShmemTensor& shmemTensor, Tensor& out)
{
    ValidateShmemTensor(shmemTensor, true, true);
    ValidateTensor(predToken, "predToken", {2});
    ValidateTensor(in, "in", {predToken.Dim()});
    uint32_t worldSize = shmemTensor.worldSize;
    int32_t row = in.GetShape(0);
    int32_t col = in.GetShape(1);
    int32_t rowPerRank = row / static_cast<int32_t>(worldSize);
    ValidateTensor(shmemTensor.data, "shmemTensor.data", {}, {}, {in.Format()},
        {static_cast<int64_t>(worldSize), rowPerRank, col});
    ValidateTensor(out, "out", {}, {in.GetDataType()}, {in.Format()}, in.GetShape());

    for (uint32_t dynRankId = 0; dynRankId < worldSize; ++dynRankId) {
        auto dataTile = ShmemView(shmemTensor, {1, rowPerRank, col},
            std::vector<SymbolicScalar>{0, 0, 0});
        auto inChunk = View(in, {rowPerRank, col},
            std::vector<SymbolicScalar>{static_cast<int32_t>(dynRankId) * rowPerRank, 0});
        auto putOut = ShmemPut(inChunk, dataTile, dynRankId, AtomicType::ADD, predToken);
        ShmemSignalAll(dataTile, dynRankId, 1, AtomicType::ADD, putOut);
        auto waitOut = ShmemWaitUntil(dataTile, dynRankId, OpType::EQ,
            static_cast<int32_t>(worldSize), true, in);
        auto reduced = ShmemGet(dataTile, dynRankId, waitOut, in.GetDataType());
        Assemble(reduced, {static_cast<int32_t>(dynRankId) * rowPerRank, 0}, out);
    }
}

// TwoShotAllReduce_v3: Uses TwoShotCommunicator to hide shmem layout details.
//
// Works with chunk IDs instead of raw ShmemView() calls.
// Input chunking and Assemble remain in the algorithm.
void TwoShotAllReduce_v3(const Tensor& predToken, const Tensor& in,
    ShmemTensor& shmemTensor, Tensor& out)
{
    ValidateShmemTensor(shmemTensor, true, true);
    ValidateTensor(predToken, "predToken", {2});
    ValidateTensor(in, "in", {predToken.Dim()});
    ValidateTensor(out, "out", {}, {in.GetDataType()}, {in.Format()}, in.GetShape());
    TwoShotCommunicator comm(shmemTensor);
    int32_t rowPerRank = in.GetShape(0) / static_cast<int32_t>(comm.WorldSize());
    int32_t col = in.GetShape(1);

    for (uint32_t dynRankId = 0; dynRankId < comm.WorldSize(); ++dynRankId) {
        auto inChunk = View(in, {rowPerRank, col},
            std::vector<SymbolicScalar>{rowPerRank * dynRankId, 0});
        comm.Put(predToken, inChunk, dynRankId, AtomicType::ADD);
        auto reduced = comm.WaitAndGet(in, dynRankId, in.GetDataType());
        Assemble(reduced, {rowPerRank * dynRankId, 0}, out);
    }
}

// TwoShotAllReduce_v4: TwoShotCommunicatorV2 with combined WaitAndGet per chunk.
//
// Uses the three-phase communicator but executes Wait+Pull as a single step
// (WaitAndGet). Enables per-chunk dtype latching via Put().
void TwoShotAllReduce_v4(const Tensor& predToken, const Tensor& in,
    ShmemTensor& shmemTensor, Tensor& out)
{
    ValidateShmemTensor(shmemTensor, true, true);
    ValidateTensor(predToken, "predToken", {2});
    ValidateTensor(in, "in", {predToken.Dim()});
    ValidateTensor(out, "out", {}, {in.GetDataType()}, {in.Format()}, in.GetShape());
    TwoShotCommunicatorV2 comm(shmemTensor);
    int32_t rowPerRank = in.GetShape(0) / static_cast<int32_t>(comm.WorldSize());
    int32_t col = in.GetShape(1);

    for (uint32_t dynRankId = 0; dynRankId < comm.WorldSize(); ++dynRankId) {
        auto inChunk = View(in, {rowPerRank, col},
            std::vector<SymbolicScalar>{rowPerRank * dynRankId, 0});
        comm.Put(predToken, inChunk, dynRankId, AtomicType::ADD);
        auto reduced = comm.WaitAndGet(in, dynRankId);
        Assemble(reduced, {rowPerRank * dynRankId, 0}, out);
    }
}

// TwoShotAllReduce_v5: per-chunk Put/Wait/Pull with external TwoShotCommunicatorV2.
void TwoShotAllReduce_v5(const Tensor& predToken, const Tensor& in,
    TwoShotCommunicatorV2& comm, Tensor& out)
{
    ValidateTensor(predToken, "predToken", {2});
    ValidateTensor(in, "in", {predToken.Dim()});
    ValidateTensor(out, "out", {}, {in.GetDataType()}, {in.Format()}, in.GetShape());
    int32_t rowPerRank = in.GetShape(0) / static_cast<int32_t>(comm.WorldSize());
    int32_t col = in.GetShape(1);

    for (uint32_t dynRankId = 0; dynRankId < comm.WorldSize(); ++dynRankId) {
        auto inChunk = View(in, {rowPerRank, col},
            std::vector<SymbolicScalar>{rowPerRank * dynRankId, 0});

        // Phase 1: Put — write chunk to shmem slot + signal all ranks
        comm.Put(predToken, inChunk, dynRankId, AtomicType::ADD);

        // Phase 2: Wait — dep is `in` (not signalOut) to match v2 blocked IR ordering
        auto waitOut = comm.Wait(in, dynRankId);

        // Phase 3: Pull — read reduced chunk from shmem
        auto reduced = comm.Pull(waitOut, dynRankId);
        Assemble(reduced, {rowPerRank * dynRankId, 0}, out);
    }
}

// OneShotAllReduce_v6: scatter-only — Put to all ranks, no Wait/Pull.
//
// predToken is used for Put() (scheduling the ShmemPut ops).
// Wait and Pull are the caller's responsibility, enabling overlap
// between scatter and local compute.
void OneShotAllReduce_v6(const Tensor& predToken, const Tensor& in, ShmemTensor& shmemTensor)
{
    ValidateShmemTensor(shmemTensor, true, true);
    ValidateTensor(predToken, "predToken", {2});
    ValidateTensor(in, "in", {predToken.Dim()});
    uint32_t worldSize = shmemTensor.worldSize;
    int32_t row = in.GetShape(0);
    int32_t col = in.GetShape(1);
    ValidateTensor(shmemTensor.data, "shmemTensor.data", {}, {}, {in.Format()}, {1, row, col});
    auto shmemDataTile = ShmemView(shmemTensor, {1, row, col}, std::vector<SymbolicScalar>{0, 0, 0});

    for (uint32_t dynRankId = 0; dynRankId < worldSize; ++dynRankId) {
        auto putOut = ShmemPut(in, shmemDataTile, dynRankId, AtomicType::ADD, predToken);
        ShmemSignal(shmemDataTile, dynRankId, dynRankId, 1, AtomicType::ADD, putOut);
    }
}

// OneShotAllReduce_v6_light: same coarse scatter cadence as v6.
// Signal compactness is controlled at ShmemTensor construction time.
void OneShotAllReduce_v6_light(const Tensor& predToken, const Tensor& in, ShmemTensor& shmemTensor)
{
    ValidateShmemTensor(shmemTensor, true, true);
    ValidateTensor(predToken, "predToken", {2});
    ValidateTensor(in, "in", {predToken.Dim()});
    uint32_t worldSize = shmemTensor.worldSize;
    int32_t row = in.GetShape(0);
    int32_t col = in.GetShape(1);
    ValidateTensor(shmemTensor.data, "shmemTensor.data", {}, {}, {in.Format()}, {1, row, col});
    auto shmemDataTile = ShmemView(shmemTensor, {1, row, col}, std::vector<SymbolicScalar>{0, 0, 0});

    for (uint32_t dynRankId = 0; dynRankId < worldSize; ++dynRankId) {
        auto putOut = ShmemPut(in, shmemDataTile, dynRankId, AtomicType::ADD, predToken);
        ShmemSignal(shmemDataTile, dynRankId, dynRankId, 1, AtomicType::ADD, putOut);
    }
}

// OneShotAllReduce_v7: chunked scatter with coarse per-rank signaling.
//
// Splits the input into payloadChunkCount chunks (OneShotCommunicatorV3
// controls chunk geometry) and scatters each chunk to every rank with atomic
// ADD.  A SINGLE ShmemSignal is fired on the FULL tile view after all chunks
// for each rank are put, keeping the signal counter at the hardware-supported
// full-tile address (sub-tile row offsets all resolve to the same tile_id and
// cannot serve as independent per-chunk counters with the current framework).
//
// Comparison with v6: v7 exposes a chunked Put/Signal API that enables the
// caller to observe chunk geometry and pipeline output assembly, while v6's
// ShmemClearAndSignal API is monolithic.
void OneShotAllReduce_v7(const Tensor& predToken, const Tensor& in, OneShotCommunicatorV3& comm)
{
    int32_t col = in.GetShape(1);
    // Phase 1: Scatter — put every chunk to every rank, then signal once per rank.
    for (uint32_t dynRankId = 0; dynRankId < comm.WorldSize(); ++dynRankId) {
        Tensor putOut = predToken;
        for (uint32_t chunkId = 0; chunkId < comm.PayloadChunkCount(); ++chunkId) {
            int32_t chunkRow = comm.ChunkStartRow(chunkId);
            int32_t chunkRows = comm.ChunkRows(chunkId);
            auto inChunk = View(in, {chunkRows, col},
                std::vector<SymbolicScalar>{chunkRow, 0});
            putOut = comm.Put(putOut, inChunk, dynRankId, chunkId, AtomicType::ADD);
        }
        comm.Signal(putOut, dynRankId, AtomicType::ADD);
    }
}

// OneShotAllReduce_v8: grouped-scatter with coarse per-rank signaling.
//
// Uses OneShotCommunicatorV4, which organises payloadChunkCount chunks into
// groups of chunksPerSignal (k).  The scatter loop mirrors v7 but is
// structured in group-major order, making the group boundary explicit in the
// IR for analysis and future per-group signal support.  A SINGLE ShmemSignal
// is fired on the FULL tile view once all groups for each rank are put
// (same coarse-wait constraint as v7: EQ-only ShmemWaitUntil cannot safely
// track per-group thresholds without GE semantics).
//
// The key V4 value-add vs V7 is on the receiver side: WaitGroup exposes
// group-level readiness tokens that map naturally to k-chunk output assembly
// units, enabling tighter loop structures when k aligns to compute tile size.
void OneShotAllReduce_v8(const Tensor& predToken, const Tensor& in, OneShotCommunicatorV4& comm)
{
    int32_t col = in.GetShape(1);
    // Phase 1: Scatter — group-major, put every chunk in every group to every rank,
    // then fire ONE coarse signal after all groups for each rank.
    for (uint32_t dynRankId = 0; dynRankId < comm.WorldSize(); ++dynRankId) {
        Tensor putOut = predToken;
        for (uint32_t groupId = 0; groupId < comm.SignalGroupCount(); ++groupId) {
            uint32_t begin = comm.GroupBeginChunk(groupId);
            uint32_t gSize = comm.GroupSize(groupId);
            for (uint32_t local = 0; local < gSize; ++local) {
                uint32_t chunkId = begin + local;
                int32_t chunkRow = comm.ChunkStartRow(chunkId);
                int32_t chunkRows = comm.ChunkRows(chunkId);
                auto inChunk = View(in, {chunkRows, col},
                    std::vector<SymbolicScalar>{chunkRow, 0});
                putOut = comm.Put(putOut, inChunk, dynRankId, chunkId, AtomicType::ADD);
            }
        }
        comm.Signal(putOut, dynRankId, AtomicType::ADD);
    }
}

}   // namespace npu::tile_fwk::Distributed
