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
 * \brief Communicator classes for OneShot and TwoShot distributed ops.
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

// OneShotCommunicator: wraps a ShmemTensor, exposes Put/WaitAndGet by rank ID.
class OneShotCommunicator {
public:
    explicit OneShotCommunicator(ShmemTensor& shmemTensor)
        : shmemTensor_(shmemTensor)
        , worldSize_(static_cast<uint32_t>(shmemTensor.worldSize))
        , thisRank_(GetHcclRankId(shmemTensor.group))
        , row_(shmemTensor.data.GetShape()[1])
        , col_(shmemTensor.data.GetShape()[2])
    {}

    OneShotCommunicator(const OneShotCommunicator&) = delete;
    OneShotCommunicator& operator=(const OneShotCommunicator&) = delete;

    uint32_t WorldSize() const { return worldSize_; }
    SymbolicScalar ThisRank() const { return thisRank_; }

    // Put + Signal: write data to targetRank's shmem slot, then signal it.
    void Put(const Tensor& pred, const Tensor& input, uint32_t targetRank,
        AtomicType atomicType) const
    {
        auto dataTile = ShmemView(shmemTensor_, {1, row_, col_}, std::vector<SymbolicScalar>{0, 0, 0});
        auto putOut = ShmemPut(input, dataTile, targetRank, atomicType, pred);
        ShmemSignal(dataTile, targetRank, targetRank, 1, atomicType, putOut);
    }

    // Fused WaitUntil + ShmemGet: block until all contributions arrive,
    // then read the reduced result. Output dtype taken from input.
    Tensor WaitAndGet(const Tensor& input) const
    {
        auto dataTile = ShmemView(shmemTensor_, {1, row_, col_}, std::vector<SymbolicScalar>{0, 0, 0});
        auto waitOut = ShmemWaitUntil(dataTile, thisRank_, OpType::EQ,
            static_cast<int32_t>(worldSize_), true, input);
        return ShmemGet(dataTile, thisRank_, waitOut, input.GetDataType());
    }

private:
    ShmemTensor& shmemTensor_;
    uint32_t worldSize_;
    SymbolicScalar thisRank_;
    int32_t row_;
    int32_t col_;
};

// OneShotCommunicatorV2: wraps a ShmemTensor, three-phase: Put() -> Wait() -> Pull().
// WaitAndGet() combines Wait+Pull.
class OneShotCommunicatorV2 {
public:
    explicit OneShotCommunicatorV2(ShmemTensor& shmemTensor)
        : shmemTensor_(shmemTensor)
        , worldSize_(static_cast<uint32_t>(shmemTensor.worldSize))
        , thisRank_(GetHcclRankId(shmemTensor.group))
        , row_(shmemTensor.data.GetShape()[1])
        , col_(shmemTensor.data.GetShape()[2])
    {}

    OneShotCommunicatorV2(const OneShotCommunicatorV2&) = delete;
    OneShotCommunicatorV2& operator=(const OneShotCommunicatorV2&) = delete;

    uint32_t WorldSize() const { return worldSize_; }
    SymbolicScalar ThisRank() const { return thisRank_; }

    // Put data to targetRank's shmem slot + signal.
    void Put(const Tensor& pred, const Tensor& input, uint32_t targetRank, AtomicType atomicType)
    {
        auto dataTile = ShmemView(shmemTensor_, {1, row_, col_}, std::vector<SymbolicScalar>{0, 0, 0});
        auto putOut = ShmemPut(input, dataTile, targetRank, atomicType, pred);
        ShmemSignal(dataTile, targetRank, targetRank, 1, atomicType, putOut);
    }

    // Wait: block until all contributions for this rank have arrived.
    Tensor Wait(const Tensor& depToken) const
    {
        auto dataTile = ShmemView(shmemTensor_, {1, row_, col_}, std::vector<SymbolicScalar>{0, 0, 0});
        return ShmemWaitUntil(dataTile, thisRank_, OpType::EQ,
            static_cast<int32_t>(worldSize_), true, depToken);
    }

    // Pull: read the reduced result from this rank's shmem slice.
    Tensor Pull(const Tensor& waitToken, DataType dtype) const
    {
        auto dataTile = ShmemView(shmemTensor_, {1, row_, col_}, std::vector<SymbolicScalar>{0, 0, 0});
        return ShmemGet(dataTile, thisRank_, waitToken, dtype);
    }

    // Wait + Pull in one shot.
    Tensor WaitAndGet(const Tensor& input) const
    {
        auto waitOut = Wait(input);
        return Pull(waitOut, input.GetDataType());
    }

private:
    ShmemTensor& shmemTensor_;
    uint32_t worldSize_;
    SymbolicScalar thisRank_;
    int32_t row_;
    int32_t col_;
};

// OneShotCommunicatorV3: chunked-scatter communicator wrapping ShmemTensor.
//
// Payload is split into payloadChunkCount chunks.  The sender scatters each
// chunk to every rank via ShmemPut, then fires ONE coarse ShmemSignal (over
// the full data tile) after all chunks for each target rank are sent.  This
// keeps signaling on the hardware-supported full-tile counter — sub-tile row
// offsets all map to the same tile_id and therefore cannot be used as
// independent per-chunk counters with the current framework.
//
// On the receiver side WaitChunk issues a single ShmemWaitUntil (on the full
// view) for the first call and caches the resulting token; subsequent calls
// for chunkId > 0 return the cached token without issuing another wait.
// PullChunk reads the requested chunk's row range via ShmemGet.
//
// Three-phase API:
//   1) Put(pred, inChunk, targetRank, chunkId, atomicType)
//        — ShmemPut on the chunk's row range; returns the put output token
//   2) Signal(putToken, targetRank, atomicType)
//        — ShmemSignal on the FULL tile view (call once after all chunks for
//          a given targetRank)
//   3) WaitChunk(dep, chunkId)
//        — issues ShmemWaitUntil for chunkId==0; returns cached token otherwise
//   4) PullChunk(waitToken, chunkId, dtype)
//        — ShmemGet on the chunk's row range
class OneShotCommunicatorV3 {
public:
    OneShotCommunicatorV3(ShmemTensor& shmemTensor, uint32_t payloadChunkCount)
        : shmemTensor_(shmemTensor)
        , worldSize_(static_cast<uint32_t>(shmemTensor.worldSize))
        , thisRank_(GetHcclRankId(shmemTensor.group))
        , row_(shmemTensor.data.GetShape()[1])
        , col_(shmemTensor.data.GetShape()[2])
        , payloadChunkCount_(payloadChunkCount)
        , waitTokenSet_(false)
    {
        ASSERT(payloadChunkCount_ > 0) << "payloadChunkCount must be > 0";
        ASSERT(static_cast<int64_t>(payloadChunkCount_) <= row_)
            << "payloadChunkCount must be <= row dimension (" << row_ << ")";
    }

    OneShotCommunicatorV3(const OneShotCommunicatorV3&) = delete;
    OneShotCommunicatorV3& operator=(const OneShotCommunicatorV3&) = delete;

    uint32_t WorldSize() const { return worldSize_; }
    SymbolicScalar ThisRank() const { return thisRank_; }
    uint32_t PayloadChunkCount() const { return payloadChunkCount_; }

    // Reset per-iteration mutable state so the communicator can be reused
    // across multiple iterations within the same FUNCTION body.
    void Reset() const
    {
        waitTokenSet_ = false;
        lastSignalTokenSet_ = false;
    }

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

    // Put one chunk's data to targetRank's shmem slot (does NOT signal).
    // Returns the ShmemPut output token; pass to Signal() after all chunks
    // for this targetRank have been put.
    Tensor Put(const Tensor& pred, const Tensor& inChunk, uint32_t targetRank,
        uint32_t chunkId, AtomicType atomicType) const
    {
        auto chunkView = ShmemView(shmemTensor_, {1, ChunkRows(chunkId), col_},
            std::vector<SymbolicScalar>{0, ChunkStartRow(chunkId), 0});
        return ShmemPut(inChunk, chunkView, targetRank, atomicType, pred);
    }

    // Fire ONE coarse signal (over the full data tile) to targetRank.
    // Call once per targetRank, after all payloadChunkCount Put() calls for
    // that rank, using the last Put's output token as pred.
    // Returns and caches the signal token so WaitChunk() can depend on it.
    Tensor Signal(const Tensor& pred, uint32_t targetRank, AtomicType atomicType) const
    {
        auto fullView = ShmemView(shmemTensor_, {1, row_, col_},
            std::vector<SymbolicScalar>{0, 0, 0});
        lastSignalToken_ = ShmemSignal(fullView, targetRank, targetRank, 1, atomicType, pred);
        lastSignalTokenSet_ = true;
        return lastSignalToken_;
    }

    // Wait for all worldSize ranks to complete their scatter to thisRank.
    // For chunkId == 0: issues ShmemWaitUntil on the full tile and caches the
    // result token.  For chunkId > 0: returns the cached token immediately.
    // This ensures every PullChunk call has a valid data-ready dependency while
    // issuing exactly one hardware wait per communicator instance.
    Tensor WaitChunk(const Tensor& dep, uint32_t chunkId) const
    {
        (void)chunkId;
        if (!waitTokenSet_) {
            auto fullView = ShmemView(shmemTensor_, {1, row_, col_},
                std::vector<SymbolicScalar>{0, 0, 0});
            const Tensor& actualDep = lastSignalTokenSet_ ? lastSignalToken_ : dep;
            waitToken_ = ShmemWaitUntil(fullView, thisRank_, OpType::EQ,
                static_cast<int32_t>(worldSize_), true, actualDep);
            waitTokenSet_ = true;
        }
        return waitToken_;
    }

    // Read one reduced chunk after WaitChunk.
    Tensor PullChunk(const Tensor& waitToken, uint32_t chunkId, DataType dtype) const
    {
        auto chunkView = ShmemView(shmemTensor_, {1, ChunkRows(chunkId), col_},
            std::vector<SymbolicScalar>{0, ChunkStartRow(chunkId), 0});
        return ShmemGet(chunkView, thisRank_, waitToken, dtype);
    }

private:
    ShmemTensor& shmemTensor_;
    uint32_t worldSize_;
    SymbolicScalar thisRank_;
    int32_t row_;
    int32_t col_;
    uint32_t payloadChunkCount_;
    mutable Tensor waitToken_;         // cached token from first WaitChunk call
    mutable bool waitTokenSet_;         // true once WaitChunk(0) has been called
    mutable Tensor lastSignalToken_;    // token returned by the last Signal() call
    mutable bool lastSignalTokenSet_ = false;  // true once Signal() has been called
};

// OneShotCommunicatorV4: grouped-scatter communicator wrapping ShmemTensor.
//
// Payload is split into payloadChunkCount chunks, logically organised into
// contiguous groups of chunksPerSignal (k) chunks.  The sender puts chunks
// group-by-group: Put() all k chunks, then Signal() once on the full tile
// view for that target rank after each group.  The receiver calls WaitGroup()
// which issues one ShmemWaitUntil on the first call (groupId == 0) and returns
// a cached token for all subsequent groups.  Because all groups share the same
// hardware counter (full-tile tile_id), coarse single-wait semantics apply —
// independent per-group overlap requires a future per-group signal counter.
//
// Four-phase API:
//   1) Put(pred, inChunk, targetRank, chunkId, atomicType)
//        — ShmemPut on the chunk's row range; returns the put output token
//   2) Signal(putToken, targetRank, atomicType)
//        — ShmemSignal on the FULL tile view; call once after the last chunk
//          of each group for a given targetRank
//   3) WaitGroup(dep, groupId)
//        — issues ShmemWaitUntil for groupId==0; returns cached token otherwise
//   4) PullChunk(waitToken, chunkId, dtype)
//        — ShmemGet on the chunk's row range
class OneShotCommunicatorV4 {
public:
    OneShotCommunicatorV4(ShmemTensor& shmemTensor, uint32_t payloadChunkCount,
        uint32_t chunksPerSignal)
        : shmemTensor_(shmemTensor)
        , worldSize_(static_cast<uint32_t>(shmemTensor.worldSize))
        , thisRank_(GetHcclRankId(shmemTensor.group))
        , row_(shmemTensor.data.GetShape()[1])
        , col_(shmemTensor.data.GetShape()[2])
        , payloadChunkCount_(payloadChunkCount)
        , chunksPerSignal_(chunksPerSignal)
        , waitTokenSet_(false)
    {
        ASSERT(payloadChunkCount_ > 0) << "payloadChunkCount must be > 0";
        ASSERT(chunksPerSignal_ > 0) << "chunksPerSignal must be > 0";
        ASSERT(static_cast<int64_t>(payloadChunkCount_) <= row_)
            << "payloadChunkCount must be <= row dimension (" << row_ << ")";
    }

    OneShotCommunicatorV4(const OneShotCommunicatorV4&) = delete;
    OneShotCommunicatorV4& operator=(const OneShotCommunicatorV4&) = delete;

    uint32_t WorldSize() const { return worldSize_; }
    SymbolicScalar ThisRank() const { return thisRank_; }
    uint32_t PayloadChunkCount() const { return payloadChunkCount_; }
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

    // Put one chunk's data to targetRank's shmem slot (does NOT signal).
    // Returns the ShmemPut output token.
    Tensor Put(const Tensor& pred, const Tensor& inChunk, uint32_t targetRank,
        uint32_t chunkId, AtomicType atomicType) const
    {
        auto chunkView = ShmemView(shmemTensor_, {1, ChunkRows(chunkId), col_},
            std::vector<SymbolicScalar>{0, ChunkStartRow(chunkId), 0});
        return ShmemPut(inChunk, chunkView, targetRank, atomicType, pred);
    }

    // Fire ONE coarse signal (over the full data tile) to targetRank.
    // Call once per group, after all Put() calls for that group and targetRank,
    // using the last Put's output token as pred.
    // Returns and caches the signal token so WaitGroup() can depend on it.
    Tensor Signal(const Tensor& pred, uint32_t targetRank, AtomicType atomicType) const
    {
        auto fullView = ShmemView(shmemTensor_, {1, row_, col_},
            std::vector<SymbolicScalar>{0, 0, 0});
        lastSignalToken_ = ShmemSignal(fullView, targetRank, targetRank, 1, atomicType, pred);
        lastSignalTokenSet_ = true;
        return lastSignalToken_;
    }

    // Wait for all worldSize ranks to complete their scatter to thisRank.
    // For groupId == 0: issues ShmemWaitUntil on the full tile and caches the
    // result token.  For groupId > 0: returns the cached token immediately.
    Tensor WaitGroup(const Tensor& dep, uint32_t groupId) const
    {
        ASSERT(groupId < SignalGroupCount()) << "groupId out of range: " << groupId;
        if (!waitTokenSet_) {
            auto fullView = ShmemView(shmemTensor_, {1, row_, col_},
                std::vector<SymbolicScalar>{0, 0, 0});
            const Tensor& actualDep = lastSignalTokenSet_ ? lastSignalToken_ : dep;
            waitToken_ = ShmemWaitUntil(fullView, thisRank_, OpType::EQ,
                static_cast<int32_t>(worldSize_), true, actualDep);
            waitTokenSet_ = true;
        }
        return waitToken_;
    }

    // Read one reduced chunk after WaitGroup.
    Tensor PullChunk(const Tensor& waitToken, uint32_t chunkId, DataType dtype) const
    {
        auto chunkView = ShmemView(shmemTensor_, {1, ChunkRows(chunkId), col_},
            std::vector<SymbolicScalar>{0, ChunkStartRow(chunkId), 0});
        return ShmemGet(chunkView, thisRank_, waitToken, dtype);
    }

private:
    ShmemTensor& shmemTensor_;
    uint32_t worldSize_;
    SymbolicScalar thisRank_;
    int32_t row_;
    int32_t col_;
    uint32_t payloadChunkCount_;
    uint32_t chunksPerSignal_;
    mutable Tensor waitToken_;         // cached token from first WaitGroup call
    mutable bool waitTokenSet_;         // true once WaitGroup(0) has been called
    mutable Tensor lastSignalToken_;    // token returned by the last Signal() call
    mutable bool lastSignalTokenSet_ = false;  // true once Signal() has been called
};

// TwoShotCommunicator: ShmemTensor-based engine for TwoShot AllReduce.
//
// Data slot layout  : {worldSize, rowPerRank, col}  (3-D, canonical)
// Signal slot layout: embedded in shmemTensor.signal ({W, worldSize, rowPerRank, col})
//
// Per-chunk protocol:
//   1. Put()       — atomic-add write of local chunk to each target rank's slot + SignalAll
//   2. WaitAndGet()— wait until this slot has accumulated worldSize contributions, then read
class TwoShotCommunicator {
public:
    explicit TwoShotCommunicator(ShmemTensor& shmemTensor)
        : shmemTensor_(shmemTensor)
        , worldSize_(static_cast<uint32_t>(shmemTensor.worldSize))
        , thisRank_(GetHcclRankId(shmemTensor.group))
        , rowPerRank_(shmemTensor.data.GetShape()[1])
        , col_(shmemTensor.data.GetShape()[2])
    {}

    TwoShotCommunicator(const TwoShotCommunicator&) = delete;
    TwoShotCommunicator& operator=(const TwoShotCommunicator&) = delete;

    uint32_t WorldSize() const { return worldSize_; }
    SymbolicScalar ThisRank() const { return thisRank_; }

    // Write input to the shared slot for chunkId and signal all ranks.
    // Returns signal dependency token.
    Tensor Put(const Tensor& pred, const Tensor& input, uint32_t chunkId,
        AtomicType atomicType) const
    {
        auto dataTile = ShmemView(shmemTensor_, {1, rowPerRank_, col_},
            std::vector<SymbolicScalar>{0, 0, 0});
        auto putOut = ShmemPut(input, dataTile, chunkId, atomicType, pred);
        return ShmemSignalAll(dataTile, chunkId, 1, atomicType, putOut);
    }

    // Wait for worldSize contributions to chunkId's slot, then read reduced result.
    Tensor WaitAndGet(const Tensor& dep, uint32_t chunkId, DataType dtype) const
    {
        auto dataTile = ShmemView(shmemTensor_, {1, rowPerRank_, col_},
            std::vector<SymbolicScalar>{0, 0, 0});
        auto waitOut = ShmemWaitUntil(dataTile, chunkId, OpType::EQ,
            static_cast<int32_t>(worldSize_), true, dep);
        return ShmemGet(dataTile, chunkId, waitOut, dtype);
    }

private:
    ShmemTensor& shmemTensor_;
    uint32_t worldSize_;
    SymbolicScalar thisRank_;
    int32_t rowPerRank_;
    int32_t col_;
};

// TwoShotCommunicatorV2: Three-phase TwoShot engine with latched dtype.
//
// Extends TwoShotCommunicator with separate Wait() / Pull() phases so callers
// can overlap computation between the write and the read.
//
// Protocol per chunk:
//   1. Put()       — same as TwoShotCommunicator::Put(); also latches input dtype
//   2. Wait()      — ShmemWaitUntil on chunkId's signal
//   3. Pull()      — ShmemGet using latched dtype
//   OR WaitAndGet()— convenience wrapper combining Wait + Pull
class TwoShotCommunicatorV2 {
public:
    explicit TwoShotCommunicatorV2(ShmemTensor& shmemTensor)
        : shmemTensor_(shmemTensor)
        , worldSize_(static_cast<uint32_t>(shmemTensor.worldSize))
        , thisRank_(GetHcclRankId(shmemTensor.group))
        , rowPerRank_(shmemTensor.data.GetShape()[1])
        , col_(shmemTensor.data.GetShape()[2])
        , inputDtype_(DataType::DT_BOTTOM)
    {}

    TwoShotCommunicatorV2(const TwoShotCommunicatorV2&) = delete;
    TwoShotCommunicatorV2& operator=(const TwoShotCommunicatorV2&) = delete;

    uint32_t WorldSize() const { return worldSize_; }
    SymbolicScalar ThisRank() const { return thisRank_; }

    // Write input to chunkId's slot + signal all ranks.
    // Latches input dtype on first call; all subsequent calls must match.
    Tensor Put(const Tensor& pred, const Tensor& input, uint32_t chunkId,
        AtomicType atomicType)
    {
        if (inputDtype_ == DataType::DT_BOTTOM) {
            inputDtype_ = input.GetDataType();
        } else {
            ASSERT(inputDtype_ == input.GetDataType())
                << "All Put() calls must use the same dtype, expected " << inputDtype_
                << " but got " << input.GetDataType();
        }
        auto dataTile = ShmemView(shmemTensor_, {1, rowPerRank_, col_},
            std::vector<SymbolicScalar>{0, 0, 0});
        auto putOut = ShmemPut(input, dataTile, chunkId, atomicType, pred);
        return ShmemSignalAll(dataTile, chunkId, 1, atomicType, putOut);
    }

    // Wait for worldSize contributions to chunkId's slot.
    Tensor Wait(const Tensor& depToken, uint32_t chunkId) const
    {
        auto dataTile = ShmemView(shmemTensor_, {1, rowPerRank_, col_},
            std::vector<SymbolicScalar>{0, 0, 0});
        return ShmemWaitUntil(dataTile, chunkId, OpType::EQ,
            static_cast<int32_t>(worldSize_), true, depToken);
    }

    // Read reduced result from chunkId's slot using latched dtype.
    Tensor Pull(const Tensor& waitToken, uint32_t chunkId) const
    {
        ASSERT(inputDtype_ != DataType::DT_BOTTOM)
            << "Pull() requires at least one prior Put() call";
        auto dataTile = ShmemView(shmemTensor_, {1, rowPerRank_, col_},
            std::vector<SymbolicScalar>{0, 0, 0});
        return ShmemGet(dataTile, chunkId, waitToken, inputDtype_);
    }

    // Convenience: Wait + Pull for one chunk.
    Tensor WaitAndGet(const Tensor& dep, uint32_t chunkId) const
    {
        auto waitOut = Wait(dep, chunkId);
        return Pull(waitOut, chunkId);
    }

private:
    ShmemTensor& shmemTensor_;
    uint32_t worldSize_;
    SymbolicScalar thisRank_;
    int32_t rowPerRank_;
    int32_t col_;
    DataType inputDtype_;
};

} // namespace Distributed
} // namespace npu::tile_fwk
