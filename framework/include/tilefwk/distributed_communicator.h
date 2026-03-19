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

// CommunicatorBase: common state (group, worldSize, thisRank, signal)
// for all communicators.
class CommunicatorBase {
protected:
    CommunicatorBase(const std::string& group, uint32_t worldSize, Tensor& shmemSignal)
        : group_(group)
        , worldSize_(worldSize)
        , thisRank_(GetHcclRankId(group))
        , shmemSignal_(shmemSignal)
    {}

    CommunicatorBase(const CommunicatorBase&) = delete;
    CommunicatorBase& operator=(const CommunicatorBase&) = delete;

public:
    uint32_t WorldSize() const { return worldSize_; }
    SymbolicScalar ThisRank() const { return thisRank_; }

protected:
    std::string group_;
    uint32_t worldSize_;
    SymbolicScalar thisRank_;
    Tensor& shmemSignal_;
};

// OneShotCommunicator: owns shmem data+signal, exposes Put/Signal/WaitAndGet
// by rank ID.
class OneShotCommunicator : public CommunicatorBase {
public:
    OneShotCommunicator(const std::string& group, Tensor& shmemData, Tensor& shmemSignal)
        : CommunicatorBase(group, static_cast<uint32_t>(shmemData.GetShape()[0]), shmemSignal)
        , shmemData_(shmemData)
        , row_(shmemData.GetShape()[2])
        , col_(shmemData.GetShape()[3])
    {}

    // Put + Signal: write data to targetRank's shmem slot, then signal it.
    void Put(const Tensor& pred, const Tensor& input, uint32_t targetRank,
        AtomicType atomicType) const
    {
        auto dataTile = View(shmemData_, {1, 1, row_, col_},
            std::vector<SymbolicScalar>{targetRank, 0, 0, 0});
        auto signalTile = View(shmemSignal_, {1, 1, 1, row_, col_},
            std::vector<SymbolicScalar>{targetRank, targetRank, 0, 0, 0});
        auto putOut = ShmemPut(pred, input, dataTile, atomicType);
        ShmemSignal(putOut, signalTile, AtomicType::ADD);
    }

    // Fused WaitUntil + ShmemGet: block until all contributions arrive,
    // then read the reduced result. Output dtype taken from input.
    Tensor WaitAndGet(const Tensor& input) const
    {
        auto dataTile = View(shmemData_, {1, 1, row_, col_},
            std::vector<SymbolicScalar>{thisRank_, 0, 0, 0});
        auto signalTile = View(shmemSignal_, {1, 1, 1, row_, col_},
            std::vector<SymbolicScalar>{thisRank_, thisRank_, 0, 0, 0});
        auto waitOut = WaitUntil(input, signalTile, static_cast<int32_t>(worldSize_));
        return ShmemGet(waitOut, dataTile, input.GetDataType());
    }

private:
    Tensor& shmemData_;
    int32_t row_;
    int32_t col_;
};

// OneShotCommunicatorV2: holds signal only, data views passed by caller.
// Three-phase: Put() -> Wait() -> Pull().
// WaitAndGet() combines Wait+Pull. Put() latches dtype for Pull().
class OneShotCommunicatorV2 : public CommunicatorBase {
public:
    OneShotCommunicatorV2(const std::string& group, uint32_t worldSize, Tensor& shmemSignal)
        : CommunicatorBase(group, worldSize, shmemSignal)
        , row_(shmemSignal.GetShape()[3])
        , col_(shmemSignal.GetShape()[4])
        , inputDtype_(DT_BOTTOM)
    {}

    // Put data to targetRank's shmem + signal. Latches dtype on first call.
    Tensor Put(const Tensor& pred, const Tensor& input,
        const Tensor& dataView, uint32_t targetRank, AtomicType atomicType)
    {
        if (inputDtype_ == DT_BOTTOM) {
            inputDtype_ = input.GetDataType();
        } else {
            ASSERT(inputDtype_ == input.GetDataType())
                << "All Put() calls must use the same dtype, expected " << inputDtype_
                << " but got " << input.GetDataType();
        }
        auto signalView = View(shmemSignal_, {1, 1, 1, row_, col_},
            std::vector<SymbolicScalar>{targetRank, targetRank, 0, 0, 0});
        auto putOut = ShmemPut(pred, input, dataView, atomicType);
        return ShmemSignal(putOut, signalView, AtomicType::ADD);
    }

    // Wait: synchronization barrier on this rank's signal slot.
    // Ensures all contributions have arrived before any read.
    // Takes a pure dep token (not the input tensor).
    // Returns a dep token for Pull().
    Tensor Wait(const Tensor& depToken) const
    {
        auto signalView = View(shmemSignal_, {1, 1, 1, row_, col_},
            std::vector<SymbolicScalar>{thisRank_, thisRank_, 0, 0, 0});
        return WaitUntil(depToken, signalView, static_cast<int32_t>(worldSize_));
    }

    // Pull: postprocessing — read the reduced result from this rank's shmem
    // data slot into regular device memory (GM). + dtype cast.
    // Called after Put() (which captures dtype) and Wait().
    Tensor Pull(const Tensor& waitToken, const Tensor& shmemData) const
    {
        int32_t dataRow = shmemData.GetShape()[2];
        int32_t dataCol = shmemData.GetShape()[3];
        auto dataLocal = View(shmemData, {1, 1, dataRow, dataCol},
            std::vector<SymbolicScalar>{thisRank_, 0, 0, 0});
        return ShmemGet(waitToken, dataLocal, inputDtype_);
    }

    // Wait + Pull in one shot.
    Tensor WaitAndGet(const Tensor& input, const Tensor& dataView) const
    {
        auto signalView = View(shmemSignal_, {1, 1, 1, row_, col_},
            std::vector<SymbolicScalar>{thisRank_, thisRank_, 0, 0, 0});
        auto waitOut = WaitUntil(input, signalView, static_cast<int32_t>(worldSize_));
        return ShmemGet(waitOut, dataView, input.GetDataType());
    }

private:
    int32_t row_;
    int32_t col_;
    DataType inputDtype_;
};

// OneShotCommunicatorV3: grouped-signal communicator with tunable chunk-to-signal ratio.
// Three-phase API:
//   1) Put(..., chunkId)  -> write one chunk and signal its group counter
//   2) WaitGroup(groupId) -> wait until all contributors complete this group
//   3) PullChunk(chunkId) -> read one reduced chunk after wait
class OneShotCommunicatorV3 : public CommunicatorBase {
public:
    OneShotCommunicatorV3(const std::string& group, uint32_t worldSize, Tensor& shmemSignal,
        uint32_t payloadChunkCount, uint32_t chunksPerSignal)
        : CommunicatorBase(group, worldSize, shmemSignal)
        , row_(shmemSignal.GetShape()[3])
        , col_(shmemSignal.GetShape()[4])
        , payloadChunkCount_(payloadChunkCount)
        , chunksPerSignal_(chunksPerSignal)
        , inputDtype_(DT_BOTTOM)
    {
        ASSERT(payloadChunkCount_ > 0) << "payloadChunkCount must be > 0";
        ASSERT(chunksPerSignal_ > 0) << "chunksPerSignal must be > 0";
        ASSERT(static_cast<int64_t>(payloadChunkCount_) <= row_)
            << "payloadChunkCount must be <= row dimension (" << row_ << "), but got " << payloadChunkCount_;
        ASSERT(static_cast<int64_t>(SignalGroupCount()) <= row_)
            << "SignalGroupCount must be <= row dimension (" << row_ << "), but got " << SignalGroupCount();
    }

    uint32_t PayloadChunkCount() const { return payloadChunkCount_; }
    uint32_t ChunksPerSignal() const { return chunksPerSignal_; }
    uint32_t SignalGroupCount() const { return (payloadChunkCount_ + chunksPerSignal_ - 1) / chunksPerSignal_; }

    uint32_t GroupIndex(uint32_t chunkId) const
    {
        ASSERT(chunkId < payloadChunkCount_) << "chunkId out of range: " << chunkId;
        return chunkId / chunksPerSignal_;
    }

    uint32_t GroupBeginChunk(uint32_t groupId) const
    {
        ASSERT(groupId < SignalGroupCount()) << "groupId out of range: " << groupId;
        return groupId * chunksPerSignal_;
    }

    uint32_t GroupSize(uint32_t groupId) const
    {
        uint32_t begin = GroupBeginChunk(groupId);
        uint32_t remaining = payloadChunkCount_ - begin;
        return remaining < chunksPerSignal_ ? remaining : chunksPerSignal_;
    }

    int32_t ExpectedCount(uint32_t groupId) const
    {
        return static_cast<int32_t>(worldSize_ * GroupSize(groupId));
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

    Tensor Put(const Tensor& pred, const Tensor& input, const Tensor& dataView,
        uint32_t targetRank, uint32_t chunkId, AtomicType atomicType)
    {
        if (inputDtype_ == DT_BOTTOM) {
            inputDtype_ = input.GetDataType();
        } else {
            ASSERT(inputDtype_ == input.GetDataType())
                << "All Put() calls must use the same dtype, expected " << inputDtype_
                << " but got " << input.GetDataType();
        }
        uint32_t groupId = GroupIndex(chunkId);
        auto signalView = View(shmemSignal_, {1, 1, 1, 1, 1},
            std::vector<SymbolicScalar>{targetRank, targetRank, 0, static_cast<int64_t>(groupId), 0});
        auto putOut = ShmemPut(pred, input, dataView, atomicType);
        return ShmemSignal(putOut, signalView, AtomicType::ADD);
    }

    Tensor WaitGroup(const Tensor& depToken, uint32_t groupId) const
    {
        ASSERT(groupId < SignalGroupCount()) << "groupId out of range: " << groupId;
        auto signalView = View(shmemSignal_, {1, 1, 1, 1, 1},
            std::vector<SymbolicScalar>{thisRank_, thisRank_, 0, static_cast<int64_t>(groupId), 0});
        return WaitUntil(depToken, signalView, ExpectedCount(groupId));
    }

    Tensor PullChunk(const Tensor& waitToken, const Tensor& shmemData, uint32_t chunkId) const
    {
        ASSERT(inputDtype_ != DT_BOTTOM) << "PullChunk() requires at least one prior Put() call";
        int32_t chunkRow = ChunkStartRow(chunkId);
        int32_t chunkRows = ChunkRows(chunkId);
        auto dataLocal = View(shmemData, {1, 1, chunkRows, col_},
            std::vector<SymbolicScalar>{thisRank_, 0, chunkRow, 0});
        return ShmemGet(waitToken, dataLocal, inputDtype_);
    }

private:
    int32_t row_;
    int32_t col_;
    uint32_t payloadChunkCount_;
    uint32_t chunksPerSignal_;
    DataType inputDtype_;
};

enum class SignalGroupingMode : uint8_t {
    CONTIGUOUS = 0,
    INTERLEAVED = 1
};

// OneShotSignalPlan: shape-agnostic mapping from payload chunks to signal groups.
// Supports contiguous and interleaved mappings while preserving threshold rule:
//   T_g = worldSize * |Omega_g|
class OneShotSignalPlan {
public:
    OneShotSignalPlan(uint32_t payloadChunkCount, uint32_t chunksPerSignal,
        SignalGroupingMode mode = SignalGroupingMode::CONTIGUOUS)
        : payloadChunkCount_(payloadChunkCount)
        , chunksPerSignal_(chunksPerSignal)
        , mode_(mode)
    {
        ASSERT(payloadChunkCount_ > 0) << "payloadChunkCount must be > 0";
        ASSERT(chunksPerSignal_ > 0) << "chunksPerSignal must be > 0";
    }

    uint32_t PayloadChunkCount() const { return payloadChunkCount_; }
    uint32_t ChunksPerSignal() const { return chunksPerSignal_; }
    SignalGroupingMode Mode() const { return mode_; }
    uint32_t SignalGroupCount() const { return (payloadChunkCount_ + chunksPerSignal_ - 1) / chunksPerSignal_; }

    uint32_t GroupIndex(uint32_t chunkId) const
    {
        ASSERT(chunkId < payloadChunkCount_) << "chunkId out of range: " << chunkId;
        if (mode_ == SignalGroupingMode::CONTIGUOUS) {
            return chunkId / chunksPerSignal_;
        }
        return chunkId % SignalGroupCount();
    }

    uint32_t GroupSize(uint32_t groupId) const
    {
        ASSERT(groupId < SignalGroupCount()) << "groupId out of range: " << groupId;
        if (mode_ == SignalGroupingMode::CONTIGUOUS) {
            uint32_t begin = groupId * chunksPerSignal_;
            uint32_t remaining = payloadChunkCount_ - begin;
            return remaining < chunksPerSignal_ ? remaining : chunksPerSignal_;
        }
        uint32_t q = payloadChunkCount_ / SignalGroupCount();
        uint32_t r = payloadChunkCount_ % SignalGroupCount();
        return q + (groupId < r ? 1u : 0u);
    }

    // Returns chunk id at local index inside a group.
    uint32_t GroupChunkAt(uint32_t groupId, uint32_t localIndex) const
    {
        ASSERT(groupId < SignalGroupCount()) << "groupId out of range: " << groupId;
        ASSERT(localIndex < GroupSize(groupId)) << "localIndex out of range: " << localIndex;
        if (mode_ == SignalGroupingMode::CONTIGUOUS) {
            return groupId * chunksPerSignal_ + localIndex;
        }
        return groupId + localIndex * SignalGroupCount();
    }

    // Lightweight default policy selector used by v8 callers.
    // Heuristic:
    //  - if no overlap potential, prefer coarse signaling (k = C_payload)
    //  - otherwise use moderate grouping (k = 4 when available)
    static OneShotSignalPlan AutoSelect(uint32_t payloadChunkCount,
        bool overlapPotential = true, bool preferInterleaved = false)
    {
        uint32_t k = payloadChunkCount;
        if (overlapPotential) {
            k = payloadChunkCount >= 4 ? 4u : payloadChunkCount;
        }
        return OneShotSignalPlan(payloadChunkCount, k,
            preferInterleaved ? SignalGroupingMode::INTERLEAVED : SignalGroupingMode::CONTIGUOUS);
    }

private:
    uint32_t payloadChunkCount_;
    uint32_t chunksPerSignal_;
    SignalGroupingMode mode_;
};

// OneShotCommunicatorV4: v8 communicator with policy-driven signal plan.
// API is compatible with v7 usage while exposing shape-agnostic signal grouping.
class OneShotCommunicatorV4 : public CommunicatorBase {
public:
    OneShotCommunicatorV4(const std::string& group, uint32_t worldSize, Tensor& shmemSignal,
        OneShotSignalPlan signalPlan)
        : CommunicatorBase(group, worldSize, shmemSignal)
        , row_(shmemSignal.GetShape()[3])
        , col_(shmemSignal.GetShape()[4])
        , signalPlan_(std::move(signalPlan))
        , inputDtype_(DT_BOTTOM)
    {
        ASSERT(static_cast<int64_t>(signalPlan_.PayloadChunkCount()) <= row_)
            << "payloadChunkCount must be <= row dimension (" << row_
            << "), but got " << signalPlan_.PayloadChunkCount();
        ASSERT(static_cast<int64_t>(signalPlan_.SignalGroupCount()) <= row_)
            << "SignalGroupCount must be <= row dimension (" << row_
            << "), but got " << signalPlan_.SignalGroupCount();
    }

    uint32_t PayloadChunkCount() const { return signalPlan_.PayloadChunkCount(); }
    uint32_t ChunksPerSignal() const { return signalPlan_.ChunksPerSignal(); }
    uint32_t SignalGroupCount() const { return signalPlan_.SignalGroupCount(); }
    SignalGroupingMode GroupingMode() const { return signalPlan_.Mode(); }
    uint32_t GroupIndex(uint32_t chunkId) const { return signalPlan_.GroupIndex(chunkId); }
    uint32_t GroupSize(uint32_t groupId) const { return signalPlan_.GroupSize(groupId); }
    uint32_t GroupChunkAt(uint32_t groupId, uint32_t localIndex) const
    {
        return signalPlan_.GroupChunkAt(groupId, localIndex);
    }

    int32_t ExpectedCount(uint32_t groupId) const
    {
        return static_cast<int32_t>(worldSize_ * GroupSize(groupId));
    }

    int32_t ChunkStartRow(uint32_t chunkId) const
    {
        ASSERT(chunkId < PayloadChunkCount()) << "chunkId out of range: " << chunkId;
        return static_cast<int32_t>((static_cast<int64_t>(chunkId) * row_) / PayloadChunkCount());
    }

    int32_t ChunkRows(uint32_t chunkId) const
    {
        ASSERT(chunkId < PayloadChunkCount()) << "chunkId out of range: " << chunkId;
        int32_t start = ChunkStartRow(chunkId);
        int32_t end = static_cast<int32_t>((static_cast<int64_t>(chunkId + 1) * row_) / PayloadChunkCount());
        ASSERT(end > start) << "Empty chunk detected for chunkId " << chunkId;
        return end - start;
    }

    Tensor Put(const Tensor& pred, const Tensor& input, const Tensor& dataView,
        uint32_t targetRank, uint32_t chunkId, AtomicType atomicType)
    {
        if (inputDtype_ == DT_BOTTOM) {
            inputDtype_ = input.GetDataType();
        } else {
            ASSERT(inputDtype_ == input.GetDataType())
                << "All Put() calls must use the same dtype, expected " << inputDtype_
                << " but got " << input.GetDataType();
        }
        uint32_t groupId = GroupIndex(chunkId);
        auto signalView = View(shmemSignal_, {1, 1, 1, 1, 1},
            std::vector<SymbolicScalar>{targetRank, targetRank, 0, static_cast<int64_t>(groupId), 0});
        auto putOut = ShmemPut(pred, input, dataView, atomicType);
        return ShmemSignal(putOut, signalView, AtomicType::ADD);
    }

    // API-level fused call site for v8.
    // Currently lowers to the same Put->Signal op pair; kept as a distinct API
    // so backend fusion can be introduced without changing caller code.
    Tensor PutWithSignal(const Tensor& pred, const Tensor& input, const Tensor& dataView,
        uint32_t targetRank, uint32_t chunkId, AtomicType atomicType)
    {
        return Put(pred, input, dataView, targetRank, chunkId, atomicType);
    }

    Tensor WaitGroup(const Tensor& depToken, uint32_t groupId) const
    {
        ASSERT(groupId < SignalGroupCount()) << "groupId out of range: " << groupId;
        auto signalView = View(shmemSignal_, {1, 1, 1, 1, 1},
            std::vector<SymbolicScalar>{thisRank_, thisRank_, 0, static_cast<int64_t>(groupId), 0});
        return WaitUntil(depToken, signalView, ExpectedCount(groupId));
    }

    Tensor PullChunk(const Tensor& waitToken, const Tensor& shmemData, uint32_t chunkId) const
    {
        ASSERT(inputDtype_ != DT_BOTTOM) << "PullChunk() requires at least one prior Put() call";
        int32_t chunkRow = ChunkStartRow(chunkId);
        int32_t chunkRows = ChunkRows(chunkId);
        auto dataLocal = View(shmemData, {1, 1, chunkRows, col_},
            std::vector<SymbolicScalar>{thisRank_, 0, chunkRow, 0});
        return ShmemGet(waitToken, dataLocal, inputDtype_);
    }

private:
    int32_t row_;
    int32_t col_;
    OneShotSignalPlan signalPlan_;
    DataType inputDtype_;
};

// TwoShotCommunicator: Encapsulates shmem data/signal buffers for TwoShot.
//
// Hides the TwoShot symmetric memory layout so that it works
// with chunk IDs instead of raw multi-dimensional View() indexing.
// Both data and signal Views are derived internally from chunk IDs.
//
// (drop) VS OneShotCommunicator:
//   - Put() returns a signal token (needed as per-chunk dependency for Wait)
//   - WaitAndGet() takes a chunk ID, dependency token, and explicit dtype
//   - shmemData layout: {worldSize, worldSize, rowPerRank, col}
//   - shmemSignal layout: {worldSize, worldSize, worldSize, rowPerRank, col}
class TwoShotCommunicator : public CommunicatorBase {
public:
    TwoShotCommunicator(const std::string& group, Tensor& shmemData, Tensor& shmemSignal)
        : CommunicatorBase(group, static_cast<uint32_t>(shmemData.GetShape()[0]), shmemSignal)
        , shmemData_(shmemData)
        , rowPerRank_(shmemData.GetShape()[2])
        , col_(shmemData.GetShape()[3])
    {}

    // Put + Signal: write input chunk to chunkId's shmem slot, then signal all ranks.
    // Returns signal dependency token (needed as dep for per-chunk WaitAndGet).
    Tensor Put(const Tensor& pred, const Tensor& input, uint32_t chunkId,
        AtomicType atomicType) const
    {
        auto dataTile = View(shmemData_, {1, 1, rowPerRank_, col_},
            std::vector<SymbolicScalar>{chunkId, chunkId, 0, 0});
        auto signalTile = View(shmemSignal_,
            {static_cast<int64_t>(worldSize_), 1, 1, rowPerRank_, col_},
            std::vector<SymbolicScalar>{0, chunkId, chunkId, 0, 0});
        auto putOut = ShmemPut(pred, input, dataTile, atomicType);
        return ShmemSignal(putOut, signalTile, AtomicType::ADD);
    }

    // Wait for all contributions to chunkId + read the reduced result.
    Tensor WaitAndGet(const Tensor& dep, uint32_t chunkId, DataType dtype) const
    {
        auto dataTile = View(shmemData_, {1, 1, rowPerRank_, col_},
            std::vector<SymbolicScalar>{chunkId, chunkId, 0, 0});
        auto waitSignalTile = View(shmemSignal_, {1, 1, 1, rowPerRank_, col_},
            std::vector<SymbolicScalar>{thisRank_, chunkId, chunkId, 0, 0});
        auto waitOut = WaitUntil(dep, waitSignalTile, static_cast<int32_t>(worldSize_));
        return ShmemGet(waitOut, dataTile, dtype);
    }

private:
    Tensor& shmemData_;
    int32_t rowPerRank_;
    int32_t col_;
};

// TwoShotCommunicatorV2: Communication engine for TwoShot with explicit data views.
//   - Data layout (explicit View calls at call site)
// Three-phase API (per chunk):
//   1. Put()  — write chunk to shmem slot + signal all ranks
//   2. Wait() — synchronization barrier on this rank's signal for a given chunk
//   3. Pull() — read reduced result from shmem (uses latched dtype)
class TwoShotCommunicatorV2 : public CommunicatorBase {
public:
    TwoShotCommunicatorV2(const std::string& group, uint32_t worldSize, Tensor& shmemSignal)
        : CommunicatorBase(group, worldSize, shmemSignal)
        , rowPerRank_(shmemSignal.GetShape()[3])
        , col_(shmemSignal.GetShape()[4])
        , inputDtype_(DT_BOTTOM)
    {}

    // Write input chunk to chunkId's shmem slot + signal all ranks.
    // Caller provides the data View; signal View is built internally.
    // Latches input dtype on first call (used by Pull()/WaitAndGet()).
    // Returns signal dependency token.
    Tensor Put(const Tensor& pred, const Tensor& input,
        const Tensor& dataView, uint32_t chunkId, AtomicType atomicType)
    {
        if (inputDtype_ == DT_BOTTOM) {
            inputDtype_ = input.GetDataType();
        } else {
            ASSERT(inputDtype_ == input.GetDataType())
                << "All Put() calls must use the same dtype, expected " << inputDtype_
                << " but got " << input.GetDataType();
        }
        auto signalView = View(shmemSignal_,
            {static_cast<int64_t>(worldSize_), 1, 1, rowPerRank_, col_},
            std::vector<SymbolicScalar>{0, chunkId, chunkId, 0, 0});
        auto putOut = ShmemPut(pred, input, dataView, atomicType);
        return ShmemSignal(putOut, signalView, AtomicType::ADD);
    }

    // Wait on chunkId's signal slot. Returns dependency token for Pull().
    Tensor Wait(const Tensor& depToken, uint32_t chunkId) const
    {
        auto signalView = View(shmemSignal_, {1, 1, 1, rowPerRank_, col_},
            std::vector<SymbolicScalar>{thisRank_, chunkId, chunkId, 0, 0});
        return WaitUntil(depToken, signalView, static_cast<int32_t>(worldSize_));
    }

    // Read the reduced result from shmem. Uses latched dtype from Put().
    Tensor Pull(const Tensor& waitToken, const Tensor& dataView) const
    {
        ASSERT(inputDtype_ != DT_BOTTOM) << "Pull() requires at least one prior Put() call";
        return ShmemGet(waitToken, dataView, inputDtype_);
    }

    // Convenience: Wait + Pull for one chunk. Uses latched dtype from Put().
    Tensor WaitAndGet(const Tensor& dep, const Tensor& dataView, uint32_t chunkId) const
    {
        ASSERT(inputDtype_ != DT_BOTTOM) << "WaitAndGet() requires at least one prior Put() call";
        auto signalView = View(shmemSignal_, {1, 1, 1, rowPerRank_, col_},
            std::vector<SymbolicScalar>{thisRank_, chunkId, chunkId, 0, 0});
        auto waitOut = WaitUntil(dep, signalView, static_cast<int32_t>(worldSize_));
        return ShmemGet(waitOut, dataView, inputDtype_);
    }

private:
    int32_t rowPerRank_;
    int32_t col_;
    DataType inputDtype_;
};

} // namespace Distributed
} // namespace npu::tile_fwk
