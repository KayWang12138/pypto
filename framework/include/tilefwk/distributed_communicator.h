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

#include "tilefwk/tilefwk_op.h"
#include "tilefwk/symbolic_distributed.h"

namespace npu::tile_fwk {
namespace Distributed {

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
    CommunicatorV2(const char* group, uint32_t worldSize, Tensor& shmemSignal)
        : group_(group)
        , worldSize_(worldSize)
        , thisRank_(GetHcclRankId(group))
        , shmemSignal_(shmemSignal)
        , row_(shmemSignal.GetShape()[3])
        , col_(shmemSignal.GetShape()[4])
        , inputDtype_(DT_BOTTOM)
    {}

    uint32_t GetWorldSize() const { return worldSize_; }
    SymbolicScalar GetThisRank() const { return thisRank_; }

    // Put data to an explicit shmem data slot + signal targetRank.
    // The caller provides the data View; the signal View is derived internally.
    // Captures input dtype on the first call for subsequent Pull().
    // Returns the signal dependency token.
    Tensor Put(const Tensor& pred, const Tensor& input,
        const Tensor& dataView, uint32_t targetRank, AtomicType atomicType)
    {
        if (inputDtype_ == DT_BOTTOM) {
            inputDtype_ = input.GetDataType();
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
        return WaitUntil(depToken, signalView, worldSize_);
    }

    // Pull: postprocessing — read the reduced result from this rank's shmem
    // data slot into regular device memory (GM). Pure data movement + dtype cast.
    // Must be called after Put() (which captures dtype) and Wait().
    Tensor Pull(const Tensor& waitToken, Tensor& shmemData) const
    {
        int32_t dataRow = shmemData.GetShape()[2];
        int32_t dataCol = shmemData.GetShape()[3];
        auto dataLocal = View(shmemData, {1, 1, dataRow, dataCol},
            std::vector<SymbolicScalar>{thisRank_, 0, 0, 0});
        return ShmemGet(waitToken, dataLocal, inputDtype_);
    }

    // WaitAndGet: convenience method combining Wait + Pull in one call.
    // Retained for cases where the three-phase split is not needed.
    Tensor WaitAndGet(const Tensor& pred, const Tensor& dataView) const
    {
        auto signalView = View(shmemSignal_, {1, 1, 1, row_, col_},
            std::vector<SymbolicScalar>{thisRank_, thisRank_, 0, 0, 0});
        auto waitOut = WaitUntil(pred, signalView, worldSize_);
        return ShmemGet(waitOut, dataView, pred.GetDataType());
    }

private:
    const char* group_;
    uint32_t worldSize_;
    SymbolicScalar thisRank_;
    Tensor& shmemSignal_;
    int32_t row_;
    int32_t col_;
    DataType inputDtype_;
};

} // namespace Distributed
} // namespace npu::tile_fwk
