/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file distributed_communicator.h
 * \brief Communicator classes for OneShot distributed ops.
 */

#pragma once

#include <string>
#include <cstdint>
#include <utility>
#include <vector>
#include "tilefwk/tilefwk_op.h"
#include "tilefwk/symbolic_distributed.h"
#include "tilefwk/error.h"

namespace npu::tile_fwk {
namespace Distributed {

// Clear the group→worldSize validation cache and CommGroupRecorder registry.
// Call from test fixture SetUp/TearDown to isolate tests that use different
// world sizes with the same or different group names.
void ResetShmemTensorGroupCache();

// Shared state/helpers for OneShot communicators.
class OneShotCommunicatorBase {
public:
    explicit OneShotCommunicatorBase(ShmemTensor& shmemTensor)
        : shmemTensor_(shmemTensor)
        , worldSize_(static_cast<uint32_t>(shmemTensor.worldSize))
        , thisRank_(GetHcclRankId(shmemTensor.group))
        , row_(shmemTensor.data.GetShape()[1])
        , col_(shmemTensor.data.GetShape()[2])
    {}

    uint32_t WorldSize() const { return worldSize_; }
    SymbolicScalar ThisRank() const { return thisRank_; }

protected:
    ShmemTensor& shmemTensor_;
    uint32_t worldSize_;
    SymbolicScalar thisRank_;
    int32_t row_;
    int32_t col_;
};

// Shared chunk geometry for chunked OneShot communicators (v3/v4/v5).
class OneShotChunkedCommunicatorBase : public OneShotCommunicatorBase {
public:
    OneShotChunkedCommunicatorBase(ShmemTensor& shmemTensor, uint32_t payloadChunkCount)
        : OneShotCommunicatorBase(shmemTensor)
        , payloadChunkCount_(payloadChunkCount)
    {
        ASSERT(payloadChunkCount_ > 0) << "payloadChunkCount must be > 0";
        ASSERT(static_cast<int64_t>(payloadChunkCount_) <= row_)
            << "payloadChunkCount must be <= row dimension (" << row_ << ")";
    }

    uint32_t PayloadChunkCount() const { return payloadChunkCount_; }

    int32_t ChunkStartRow(uint32_t chunkId) const
    {
        ASSERT(chunkId < payloadChunkCount_) << "chunkId out of range: " << chunkId;
        return static_cast<int32_t>((static_cast<int64_t>(chunkId) * row_) / payloadChunkCount_);
    }

    int32_t ChunkRows(uint32_t chunkId) const
    {
        ASSERT(chunkId < payloadChunkCount_) << "chunkId out of range: " << chunkId;
        int32_t start = ChunkStartRow(chunkId);
        int32_t end = static_cast<int32_t>((static_cast<int64_t>(chunkId + 1) * row_) / payloadChunkCount_);
        ASSERT(end > start) << "Empty chunk detected for chunkId " << chunkId;
        return end - start;
    }

protected:
    uint32_t payloadChunkCount_;
};

// Shared group geometry for grouped OneShot communicators (v4/v5).
class OneShotGroupedCommunicatorBase : public OneShotChunkedCommunicatorBase {
public:
    OneShotGroupedCommunicatorBase(ShmemTensor& shmemTensor, uint32_t payloadChunkCount,
        uint32_t chunksPerSignal)
        : OneShotChunkedCommunicatorBase(shmemTensor, payloadChunkCount)
        , chunksPerSignal_(chunksPerSignal)
    {
        ASSERT(chunksPerSignal_ > 0) << "chunksPerSignal must be > 0";
    }

    uint32_t ChunksPerSignal() const { return chunksPerSignal_; }
    uint32_t SignalGroupCount() const
    {
        return (payloadChunkCount_ + chunksPerSignal_ - 1) / chunksPerSignal_;
    }

    uint32_t GroupBeginChunk(uint32_t groupId) const { return groupId * chunksPerSignal_; }
    uint32_t GroupSize(uint32_t groupId) const
    {
        uint32_t begin = GroupBeginChunk(groupId);
        uint32_t remaining = payloadChunkCount_ - begin;
        return remaining < chunksPerSignal_ ? remaining : chunksPerSignal_;
    }

protected:
    uint32_t chunksPerSignal_;
};

// OneShotCommunicatorV5: per-group GE-semantics communicator for v10.
//
// Unlike earlier communicators (one coarse Signal after all groups then EQ wait), V5 calls
// SignalGroup() once per group per target rank (in ascending groupId order).
// Each SignalGroup() atomically increments the target's full-tile counter by 1.
// WaitGroup(G) issues a fresh ShmemWaitUntil with OpType::GE and threshold
// (G+1)*worldSize every call — no caching is needed or performed.
//
// Correctness argument (pigeonhole): ordered per-group signaling ensures that
// once the counter cross >= (G+1)*W the receiver knows all W senders have each
// completed at least G+1 SignalGroup() calls, which are issued strictly after
// the corresponding Put() calls for group G. clearSignal=false keeps the
// counter monotonically increasing so later groups' waits remain valid.
class OneShotCommunicatorV5 : public OneShotGroupedCommunicatorBase {
public:
    OneShotCommunicatorV5(ShmemTensor& shmemTensor, uint32_t payloadChunkCount,
        uint32_t chunksPerSignal)
        : OneShotGroupedCommunicatorBase(shmemTensor, payloadChunkCount, chunksPerSignal)
    {}

    OneShotCommunicatorV5(const OneShotCommunicatorV5&) = delete;
    OneShotCommunicatorV5& operator=(const OneShotCommunicatorV5&) = delete;
    // Put one chunk's data to targetRank's shmem slot (does NOT signal).
    Tensor Put(const Tensor& pred, const Tensor& inChunk, uint32_t targetRank,
        uint32_t chunkId, AtomicType atomicType) const
    {
        auto chunkView = ShmemView(shmemTensor_, {1, ChunkRows(chunkId), col_},
            std::vector<SymbolicScalar>{0, ChunkStartRow(chunkId), 0});
        return ShmemPut(inChunk, chunkView, targetRank, atomicType, pred);
    }

    // Signal targetRank after completing all Put()s for groupId.
    // Atomically increments targetRank's full-tile counter by 1.
    // Must be called in ascending groupId order: 0, 1, ..., SignalGroupCount()-1.
    Tensor SignalGroup(const Tensor& pred, uint32_t targetRank,
        uint32_t /*groupId*/, AtomicType atomicType) const
    {
        auto fullView = ShmemView(shmemTensor_, {1, row_, col_},
            std::vector<SymbolicScalar>{0, 0, 0});
        return ShmemSignal(fullView, targetRank, targetRank, 1, atomicType, pred);
    }

    // Wait until the local counter has reached (groupId+1)*worldSize.
    // Issues a fresh ShmemWaitUntil every call (no caching).
    // clearSignal=false: counter is monotonically increasing; later groups' waits remain valid.
    Tensor WaitGroup(const Tensor& dep, uint32_t groupId) const
    {
        ASSERT(groupId < SignalGroupCount()) << "groupId out of range: " << groupId;
        auto fullView = ShmemView(shmemTensor_, {1, row_, col_},
            std::vector<SymbolicScalar>{0, 0, 0});
        int32_t threshold = static_cast<int32_t>((groupId + 1) * worldSize_);
        return ShmemWaitUntil(fullView, thisRank_, OpType::GE, threshold, false, dep);
    }

    // Read one reduced chunk after WaitGroup.
    Tensor PullChunk(const Tensor& waitToken, uint32_t chunkId, DataType dtype) const
    {
        auto chunkView = ShmemView(shmemTensor_, {1, ChunkRows(chunkId), col_},
            std::vector<SymbolicScalar>{0, ChunkStartRow(chunkId), 0});
        return ShmemGet(chunkView, thisRank_, waitToken, dtype);
    }

private:
};

} // namespace Distributed
} // namespace npu::tile_fwk
