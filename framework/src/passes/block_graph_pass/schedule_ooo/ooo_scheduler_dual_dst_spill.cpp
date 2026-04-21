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
 * \file ooo_scheduler_dual_dst_spill.cpp
 * \brief DualDst feature spill/reload implementation
 */

#include "ooo_scheduler.h"
#include "passes/pass_log/pass_log.h"
#include "interface/configs/config_manager.h"

#ifndef MODULE_NAME
#define MODULE_NAME "OoOScheduler.DualDst.Spill"
#endif

namespace npu::tile_fwk {

// Select spill buffers considering dual_dst requirements
// Must ensure AIV0/AIV1 have matching free space for dual_dst tensors
Status OoOScheduler::SelectSpillBuffersDualDst(
    LocalBufferPtr allocBuffer,
    Operation* allocOp,
    std::vector<int>& spillGroup,
    bool isGenSpill)
{
    if (!enableDualDst_ || allocBuffer == nullptr || allocOp == nullptr) {
        return SelectSpillBuffers(allocBuffer, allocOp, spillGroup, isGenSpill);
    }

    APASS_LOG_DEBUG_F(Elements::Operation,
        "SelectSpillBuffersDualDst: allocOp=%p, memId=%d", allocOp, allocBuffer->GetMemId());

    // Check if this is a dual_dst allocation
    int memId = allocBuffer->GetMemId();
    bool isDualDst = (dualDstMemIds_.count(memId) > 0);

    if (!isDualDst) {
        // Not a dual_dst buffer, use standard spill selection
        return SelectSpillBuffers(allocBuffer, allocOp, spillGroup, isGenSpill);
    }

    // For dual_dst, we need to ensure both AIV0 and AIV1 have matching space
    // after spilling

    // Check if AIV0 and AIV1 buffer managers exist
    MemoryType memType = allocBuffer->GetMemoryType();
    if (bufferManagerMap.find(CoreLocationType::AIV0) == bufferManagerMap.end() ||
        bufferManagerMap.find(CoreLocationType::AIV1) == bufferManagerMap.end()) {
        return SelectSpillBuffers(allocBuffer, allocOp, spillGroup, isGenSpill);
    }

    auto& bufferManagerAIV0 = bufferManagerMap[CoreLocationType::AIV0][memType];
    auto& bufferManagerAIV1 = bufferManagerMap[CoreLocationType::AIV1][memType];

    // Get candidate buffers for spill
    std::vector<int> candidateGroup;
    for (int candidateMemId : bufferManagerAIV0.GetOccupiedMemIds()) {
        if (candidateMemId == memId) {
            continue;  // Skip the current allocation
        }

        // Check if this memId is used in both AIV0 and AIV1
        if (!bufferManagerAIV1.IsOccupied(candidateMemId)) {
            continue;  // Not a dual_dst buffer
        }

        candidateGroup.push_back(candidateMemId);
    }

    if (candidateGroup.empty()) {
        APASS_LOG_DEBUG_F(Elements::Operation,
            "No dual_dst candidate buffers found for spill");
        return FAILED;
    }

    // Calculate next use order for candidates
    std::unordered_map<int, size_t> nextUseTimeCache;
    std::vector<int> groupNextUseTime;
    if (GetGroupNextUseOrder(candidateGroup, allocOp, groupNextUseTime,
            nextUseTimeCache, isGenSpill) != SUCCESS) {
        return FAILED;
    }

    // Select the buffer with farthest next use time
    int bestIdx = 0;
    int bestNextUse = -1;
    for (size_t i = 0; i < candidateGroup.size(); i++) {
        if (groupNextUseTime[i] > bestNextUse) {
            bestNextUse = groupNextUseTime[i];
            bestIdx = i;
        }
    }

    spillGroup.push_back(candidateGroup[bestIdx]);

    APASS_LOG_INFO_F(Elements::Operation,
        "Selected dual_dst spill buffer: memId=%d, nextUse=%d",
        candidateGroup[bestIdx], bestNextUse);

    return SUCCESS;
}

// Execute spill for dual_dst buffer
// Frees buffers in both AIV0 and AIV1
Status OoOScheduler::ExecuteDualDstSpill(Operation* allocOp, int memId)
{
    if (!enableDualDst_ || allocOp == nullptr) {
        return FAILED;
    }

    APASS_LOG_DEBUG_F(Elements::Operation,
        "ExecuteDualDstSpill: allocOp=%p, memId=%d", allocOp, memId);

    // Get memory type from the allocation
    MemoryType memType = MEM_INVALID;
    for (size_t i = 0; i < allocOp->GetOOperands().size(); i++) {
        auto* tensor = allocOp->GetOOperand(i);
        if (tensor != nullptr && tensor->GetMemId() == memId) {
            memType = tensor->GetMemoryTypeOriginal();
            break;
        }
    }

    if (memType == MEM_INVALID) {
        APASS_LOG_ERROR_F(Elements::Operation,
            "Cannot determine memory type for dual_dst spill: memId=%d", memId);
        return FAILED;
    }

    // Check if AIV0 and AIV1 buffer managers exist
    if (bufferManagerMap.find(CoreLocationType::AIV0) == bufferManagerMap.end() ||
        bufferManagerMap.find(CoreLocationType::AIV1) == bufferManagerMap.end()) {
        APASS_LOG_ERROR_F(Elements::Operation,
            "Buffer managers not initialized for AIV0/AIV1");
        return FAILED;
    }

    auto& bufferManagerAIV0 = bufferManagerMap[CoreLocationType::AIV0][memType];
    auto& bufferManagerAIV1 = bufferManagerMap[CoreLocationType::AIV1][memType];

    // Free in AIV0
    if (!bufferManagerAIV0.Free(memId)) {
        APASS_LOG_ERROR_F(Elements::Operation,
            "Failed to free AIV0 buffer: memId=%d", memId);
        return FAILED;
    }

    // Free in AIV1
    if (!bufferManagerAIV1.Free(memId)) {
        APASS_LOG_ERROR_F(Elements::Operation,
            "Failed to free AIV1 buffer: memId=%d", memId);
        // Try to rollback AIV0
        bufferManagerAIV0.AllocateAt(memId, bufferManagerAIV0.GetBlockSize() *
            bufferManagerAIV0.GetBlockCount(memId));
        return FAILED;
    }

    APASS_LOG_INFO_F(Elements::Operation,
        "Freed dual_dst buffer in both AIV0 and AIV1: memId=%d", memId);

    return SUCCESS;
}

// Execute reload for dual_dst buffer
// Reloads from DDR to both AIV0 and AIV1
Status OoOScheduler::ExecuteDualDstReload(Operation* allocOp, int memId)
{
    if (!enableDualDst_ || allocOp == nullptr) {
        return FAILED;
    }

    APASS_LOG_DEBUG_F(Elements::Operation,
        "ExecuteDualDstReload: allocOp=%p, memId=%d", allocOp, memId);

    // Get memory type from the allocation
    MemoryType memType = MEM_INVALID;
    for (size_t i = 0; i < allocOp->GetOOperands().size(); i++) {
        auto* tensor = allocOp->GetOOperand(i);
        if (tensor != nullptr && tensor->GetMemId() == memId) {
            memType = tensor->GetMemoryTypeOriginal();
            break;
        }
    }

    if (memType == MEM_INVALID) {
        APASS_LOG_ERROR_F(Elements::Operation,
            "Cannot determine memory type for dual_dst reload: memId=%d", memId);
        return FAILED;
    }

    // Check if AIV0 and AIV1 buffer managers exist
    if (bufferManagerMap.find(CoreLocationType::AIV0) == bufferManagerMap.end() ||
        bufferManagerMap.find(CoreLocationType::AIV1) == bufferManagerMap.end()) {
        APASS_LOG_ERROR_F(Elements::Operation,
            "Buffer managers not initialized for AIV0/AIV1");
        return FAILED;
    }

    auto& bufferManagerAIV0 = bufferManagerMap[CoreLocationType::AIV0][memType];
    auto& bufferManagerAIV1 = bufferManagerMap[CoreLocationType::AIV1][memType];

    // Allocate in AIV0
    if (!bufferManagerAIV0.AllocateAt(memId, bufferManagerAIV0.GetBlockSize() *
            bufferManagerAIV0.GetBlockCount(memId))) {
        APASS_LOG_ERROR_F(Elements::Operation,
            "Failed to allocate AIV0 buffer: memId=%d", memId);
        return FAILED;
    }

    // Allocate in AIV1
    if (!bufferManagerAIV1.AllocateAt(memId, bufferManagerAIV1.GetBlockSize() *
            bufferManagerAIV1.GetBlockCount(memId))) {
        APASS_LOG_ERROR_F(Elements::Operation,
            "Failed to allocate AIV1 buffer: memId=%d", memId);
        // Try to rollback AIV0
        bufferManagerAIV0.Free(memId);
        return FAILED;
    }

    APASS_LOG_INFO_F(Elements::Operation,
        "Reloaded dual_dst buffer in both AIV0 and AIV1: memId=%d", memId);

    return SUCCESS;
}

// Update refCount for dual_dst buffer
// Synchronizes refCount across AIV0 and AIV1 cores
Status OoOScheduler::UpdateDualDstRefCount(Operation* op, int memId)
{
    if (!enableDualDst_ || op == nullptr) {
        return FAILED;
    }

    // Check if this is a dual_dst buffer
    bool isDualDst = (dualDstMemIds_.count(memId) > 0);
    if (!isDualDst) {
        return SUCCESS;  // Not a dual_dst buffer, nothing to sync
    }

    APASS_LOG_DEBUG_F(Elements::Operation,
        "UpdateDualDstRefCount: op=%p, memId=%d", op, memId);

    // For dual_dst buffers, we need to track refCount for both cores
    // The buffer is only freed when both cores are done with it

    // Get the refCount from the operation
    int refCount = op->GetAttribute<int>("refCount", 0);

    // Update tensor refCounts
    for (size_t i = 0; i < op->GetIOperands().size(); i++) {
        auto* tensor = op->GetInputOperand(i);
        if (tensor != nullptr && tensor->GetMemId() == memId) {
            int tensorRefCount = tensor->GetAttribute<int>("refCount", 1);
            tensorRefCount--;
            tensor->SetAttribute("refCount", tensorRefCount);

            // Mark the corresponding AIV1 tensor as well
            tensor->SetAttribute("dual_dst_refCount_aiv0", tensorRefCount);
            tensor->SetAttribute("dual_dst_refCount_aiv1", tensorRefCount);

            APASS_LOG_DEBUG_F(Elements::Operation,
                "Updated dual_dst tensor refCount: tensor=%p, refCount=%d",
                tensor, tensorRefCount);
        }
    }

    return SUCCESS;
}

} // namespace npu::tile_fwk
