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
 * \brief Reusable communication abstractions for distributed algorithms.
 */

#pragma once

#include <string>
#include "tilefwk/tilefwk_op.h"
#include "tilefwk/symbolic_distributed.h"
#include "tilefwk/error.h"

namespace npu::tile_fwk {
namespace Distributed {

// =============================================================================
// Communicator: Encapsulates shmem data/signal buffers and group metadata.
//
// Hides the symmetric memory layout so that algorithm authors work with
// rank IDs instead of raw multi-dimensional View() indexing.
// Both data and signal Views are derived internally from rank IDs.
// =============================================================================
class Communicator {
public:
    Communicator(const std::string& group, Tensor& shmemData, Tensor& shmemSignal)
        : group_(group)
        , shmemData_(shmemData)
        , shmemSignal_(shmemSignal)
        , worldSize_(static_cast<uint32_t>(shmemData.GetShape()[0]))
        , thisRank_(GetHcclRankId(group))
        , row_(shmemData.GetShape()[2])
        , col_(shmemData.GetShape()[3])
    {}

    Communicator(const Communicator&) = delete;
    Communicator& operator=(const Communicator&) = delete;

    uint32_t WorldSize() const { return worldSize_; }
    SymbolicScalar ThisRank() const { return thisRank_; }  // unused; kept for symmetry with CommunicatorV2

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
    std::string group_;  // unused for now; kept for debug
    Tensor& shmemData_;
    Tensor& shmemSignal_;
    uint32_t worldSize_;
    SymbolicScalar thisRank_;
    int32_t row_;
    int32_t col_;
};

// =============================================================================
// CommunicatorV2: Communication engine with group metadata and explicit data views.
//
// Designed for algorithm authors writing distributed collectives on top of the
// SHMEM primitives (ShmemPut, ShmemSignal, WaitUntil, ShmemGet).
//
// Separation of concerns:
//   - Algorithm author: data layout (explicit View calls at call site)
//   - CommunicatorV2:  group metadata + signal coordination
//
// Three-phase API:
//   1. Put()  — scatter data to target rank's shmem slot + signal
//   2. Wait() — synchronization barrier, ensures all contributions arrived
//   3. Pull() — postprocessing, read reduced result from SHMEM to GM
//
// Put() captures the input dtype so that Pull() can produce the correct
// output type without requiring the caller to pass it again.
// WaitAndGet() is retained as a convenience for the combined Wait+Pull pattern.
// =============================================================================
class CommunicatorV2 {
public:
    CommunicatorV2(const std::string& group, uint32_t worldSize, Tensor& shmemSignal)
        : group_(group)
        , worldSize_(worldSize)
        , thisRank_(GetHcclRankId(group))
        , shmemSignal_(shmemSignal)
        , row_(shmemSignal.GetShape()[3])
        , col_(shmemSignal.GetShape()[4])
        , inputDtype_(DT_BOTTOM)
    {}

    CommunicatorV2(const CommunicatorV2&) = delete;
    CommunicatorV2& operator=(const CommunicatorV2&) = delete;

    uint32_t GetWorldSize() const { return worldSize_; }
    SymbolicScalar GetThisRank() const { return thisRank_; }

    // Write data to targetRank's shmem slot + signal it.
    // Caller provides the data View; signal View is built internally.
    // Latches input dtype on first call (used by Pull()); all Puts must match dtype.
    // Returns signal dependency token.
    Tensor Put(const Tensor& pred, const Tensor& input,
        const Tensor& dataView, uint32_t targetRank, AtomicType atomicType)
    {
        if (inputDtype_ == DT_BOTTOM) {
            inputDtype_ = input.GetDataType();
        } else {
            // Guard against mixed dtypes across Put() calls within the same collective.
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
    // Takes a pure dependency token (not the input tensor).
    // Returns a dependency token for Pull().
    Tensor Wait(const Tensor& depToken) const
    {
        auto signalView = View(shmemSignal_, {1, 1, 1, row_, col_},
            std::vector<SymbolicScalar>{thisRank_, thisRank_, 0, 0, 0});
        return WaitUntil(depToken, signalView, static_cast<int32_t>(worldSize_));
    }

    // Pull: postprocessing — read the reduced result from this rank's shmem
    // data slot into regular device memory (GM). Pure data movement + dtype cast.
    // Must be called after Put() (which captures dtype) and Wait().
    Tensor Pull(const Tensor& waitToken, const Tensor& shmemData) const
    {
        int32_t dataRow = shmemData.GetShape()[2];
        int32_t dataCol = shmemData.GetShape()[3];
        auto dataLocal = View(shmemData, {1, 1, dataRow, dataCol},
            std::vector<SymbolicScalar>{thisRank_, 0, 0, 0});
        return ShmemGet(waitToken, dataLocal, inputDtype_);
    }

    // Convenience: Wait + Pull in one shot.
    // input doubles as dependency token and dtype source.
    Tensor WaitAndGet(const Tensor& input, const Tensor& dataView) const
    {
        auto signalView = View(shmemSignal_, {1, 1, 1, row_, col_},
            std::vector<SymbolicScalar>{thisRank_, thisRank_, 0, 0, 0});
        auto waitOut = WaitUntil(input, signalView, static_cast<int32_t>(worldSize_));
        return ShmemGet(waitOut, dataView, input.GetDataType());
    }

private:
    std::string group_;  // unused for now; kept for debug
    uint32_t worldSize_;
    SymbolicScalar thisRank_;
    Tensor& shmemSignal_;
    int32_t row_;
    int32_t col_;
    DataType inputDtype_;
};

} // namespace Distributed
} // namespace npu::tile_fwk
