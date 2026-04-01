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

// File-scope cache used by ValidateShmemTensor. Exposed here so unit tests can
// call ResetShmemTensorGroupCache() in SetUp/TearDown for test isolation.
static std::unordered_map<std::string, int64_t> s_groupWorldSizeMap;

void ResetShmemTensorGroupCache()
{
    s_groupWorldSizeMap.clear();
    CommGroupRecorder::GetInstance().Reset();
}

// OneShotAllReduce_v10: SHMEM-only, chunked, per-group GE-semantics signaling.
//
// Per-group independent readiness using two key changes vs coarse-signal variants:
//   1. Sender fires ONE ShmemSignal per group per target rank (inside the group loop).
//      Signals accumulate monotonically in the full-tile counter.
//   2. Receiver issues a fresh ShmemWaitUntil per group using GE semantics:
//      WaitGroup(G) waits until counter >= (G+1)*worldSize.
//      By ordered-signaling induction, this guarantees all W senders have
//      completed at least group G before the receiver pulls group G's chunks.
//
// Correctness: since each sender signals after each of its groups in order,
// the sum of all senders' contributions to a rank's counter reaching K*W means
// all W senders have each sent at least K groups (pigeonhole, each contributes
// ≤ K). The counter is NOT reset between groups (clearSignal=false).
//
// Constraint: the communicator should be used once per kernel invocation.
//             Counter must be cleared (ShmemClearSignal) before reuse.
void OneShotAllReduce_v10(const Tensor& predToken, const Tensor& in, ShmemTensor& shmemTensor, Tensor& out,
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

    OneShotCommunicatorV5 comm(shmemTensor, payloadChunkCount, chunksPerSignal);

    // Phase 1: Scatter — group-major puts; signal AFTER each group per rank.
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
            // Signal after this group — increments the monotonic counter by 1.
            putOut = comm.SignalGroup(putOut, dynRankId, groupId, AtomicType::ADD);
        }
    }

    // Phase 2: Receive — independent GE wait and pull per group, no caching.
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

// OneShotAllReduce_v10_pipe_ge: SHMEM-only, chunked, pipelined GE variant.
//
// This variant keeps v10's GE wait semantics but changes loop ordering from
// two-phase (all scatter, then all receive) to per-group pipelining:
//   - scatter group g to all ranks,
//   - wait GE threshold for group g,
//   - pull/assemble group g,
// then continue with group g+1.
//
// Goal: improve communication/computation overlap and reduce time-to-first-
// consumable group on the receiver.
void OneShotAllReduce_v10_pipe_ge(const Tensor& predToken, const Tensor& in, ShmemTensor& shmemTensor, Tensor& out,
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

    OneShotCommunicatorV5 comm(shmemTensor, payloadChunkCount, chunksPerSignal);

    for (uint32_t groupId = 0; groupId < comm.SignalGroupCount(); ++groupId) {
        uint32_t begin = comm.GroupBeginChunk(groupId);
        uint32_t gSize = comm.GroupSize(groupId);

        // Scatter this group to every rank, signaling once per rank.
        Tensor groupDep = predToken;
        for (uint32_t dynRankId = 0; dynRankId < comm.WorldSize(); ++dynRankId) {
            Tensor putOut = groupDep;
            for (uint32_t local = 0; local < gSize; ++local) {
                uint32_t chunkId = begin + local;
                int32_t chunkRow = comm.ChunkStartRow(chunkId);
                int32_t chunkRows = comm.ChunkRows(chunkId);
                auto inChunk = View(in, {chunkRows, col},
                    std::vector<SymbolicScalar>{chunkRow, 0});
                putOut = comm.Put(putOut, inChunk, dynRankId, chunkId, AtomicType::ADD);
            }
            groupDep = comm.SignalGroup(putOut, dynRankId, groupId, AtomicType::ADD);
        }

        // Wait GE threshold for this group, then consume it immediately.
        auto waitToken = comm.WaitGroup(groupDep, groupId);
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
    ValidateGroup(t.group.c_str());
    auto groupWorldSize = s_groupWorldSizeMap.find(t.group);
    if (groupWorldSize == s_groupWorldSizeMap.end()) {
        ASSERT(DistributedErrorCode::INVALID_WORLD_SIZE, t.worldSize > 0) << "Invalid world size for group " <<
            t.group << ": world size must be greather than 0" << ", but got " << t.worldSize;
        s_groupWorldSizeMap.emplace(t.group, t.worldSize);
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
    ValidateOpType(cmp, {OpType::EQ, OpType::GE});
    ASSERT(cmp != OpType::GE || !clearSignal)
        << "ShmemWaitUntil: clearSignal must be false when using GE semantics";
    ValidateShmemTensor(src, false, true);
    ValidateTensor(pred, "pred", {2});
    ValidateTensor(src.signal, "src.signal", {4});
    ValidateTiling(Opcode::OP_SHMEM_WAIT_UNTIL, pred, "pred");
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
    distOpAttr.cmpType = cmp;
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

}   // namespace npu::tile_fwk::Distributed
