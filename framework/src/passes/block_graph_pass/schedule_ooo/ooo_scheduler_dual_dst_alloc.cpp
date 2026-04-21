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
 * \file ooo_scheduler_dual_dst_alloc.cpp
 * \brief DualDst feature memory allocation implementation
 */

#include "ooo_scheduler.h"
#include "passes/pass_log/pass_log.h"
#include "interface/configs/config_manager.h"

#ifndef MODULE_NAME
#define MODULE_NAME "OoOScheduler.DualDst.Alloc"
#endif

namespace npu::tile_fwk {

// Check if tensors can be allocated with dual_dst requirement
// Requires matching address space in both AIV0 and AIV1
bool OoOScheduler::CanAllocateDualDst(std::vector<LocalBufferPtr> tensors, MemoryType memType)
{
    if (!enableDualDst_) {
        return false;
    }

    if (tensors.empty()) {
        return false;
    }

    // Check if AIV0 and AIV1 buffer managers exist
    if (bufferManagerMap.find(CoreLocationType::AIV0) == bufferManagerMap.end() ||
        bufferManagerMap.find(CoreLocationType::AIV1) == bufferManagerMap.end()) {
        return false;
    }

    auto& bufferManagerAIV0 = bufferManagerMap[CoreLocationType::AIV0][memType];
    auto& bufferManagerAIV1 = bufferManagerMap[CoreLocationType::AIV1][memType];

    // Calculate total size needed
    size_t totalSize = 0;
    for (auto& tensor : tensors) {
        if (tensor == nullptr) {
            continue;
        }
        totalSize += tensor->GetSize();
    }

    if (totalSize == 0) {
        return false;
    }

    // Check if both AIV0 and AIV1 have enough contiguous space
    // at the same memId (address)
    bool canAllocate = false;

    // Try to find a memId that has enough space in both cores
    for (int memId = 0; memId < bufferManagerAIV0.GetMaxMemId(); memId++) {
        if (bufferManagerAIV0.IsFree(memId, totalSize) &&
            bufferManagerAIV1.IsFree(memId, totalSize)) {
            canAllocate = true;
            break;
        }
    }

    APASS_LOG_DEBUG_F(Elements::Operation,
        "CanAllocateDualDst: memType=%d, totalSize=%zu, result=%d",
        static_cast<int>(memType), totalSize, canAllocate);

    return canAllocate;
}

// Allocate memory for dual_dst tensor
// Requires same memId (address) in both AIV0 and AIV1
Status OoOScheduler::AllocateDualDst(Operation* allocOp, MemoryType memType)
{
    if (!enableDualDst_ || allocOp == nullptr) {
        return FAILED;
    }

    APASS_LOG_DEBUG_F(Elements::Operation,
        "AllocateDualDst: allocOp=%p, memType=%d", allocOp, static_cast<int>(memType));

    // Check if AIV0 and AIV1 buffer managers exist
    if (bufferManagerMap.find(CoreLocationType::AIV0) == bufferManagerMap.end() ||
        bufferManagerMap.find(CoreLocationType::AIV1) == bufferManagerMap.end()) {
        APASS_LOG_ERROR_F(Elements::Operation,
            "Buffer managers not initialized for AIV0/AIV1");
        return FAILED;
    }

    auto& bufferManagerAIV0 = bufferManagerMap[CoreLocationType::AIV0][memType];
    auto& bufferManagerAIV1 = bufferManagerMap[CoreLocationType::AIV1][memType];

    // Get tensors to allocate from the alloc op
    std::vector<LocalBufferPtr> tensors;
    for (size_t i = 0; i < allocOp->GetOOperands().size(); i++) {
        auto* tensor = allocOp->GetOOperand(i);
        if (tensor != nullptr && tensor->GetMemoryTypeOriginal() == memType) {
            tensors.push_back(tensor);
        }
    }

    if (tensors.empty()) {
        APASS_LOG_DEBUG_F(Elements::Operation,
            "No tensors of memType=%d found in allocOp=%p",
            static_cast<int>(memType), allocOp);
        return SUCCESS;
    }

    // Calculate total size needed
    size_t totalSize = 0;
    for (auto& tensor : tensors) {
        if (tensor != nullptr) {
            totalSize += tensor->GetSize();
        }
    }

    if (totalSize == 0) {
        return FAILED;
    }

    // Find a memId with enough space in both AIV0 and AIV1
    int allocatedMemId = -1;
    for (int memId = 0; memId < bufferManagerAIV0.GetMaxMemId(); memId++) {
        if (bufferManagerAIV0.IsFree(memId, totalSize) &&
            bufferManagerAIV1.IsFree(memId, totalSize)) {
            allocatedMemId = memId;
            break;
        }
    }

    if (allocatedMemId < 0) {
        APASS_LOG_ERROR_F(Elements::Operation,
            "Failed to find matching free space in AIV0/AIV1 for dual_dst allocation: size=%zu",
            totalSize);
        return FAILED;
    }

    // Allocate in AIV0
    auto bufferAIV0 = bufferManagerAIV0.Allocate(tensors[0], allocatedMemId);
    if (bufferAIV0 == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation,
            "Failed to allocate in AIV0 for dual_dst: memId=%d", allocatedMemId);
        return FAILED;
    }

    // Allocate matching space in AIV1
    // Create a shadow tensor for AIV1 if needed
    auto bufferAIV1 = bufferManagerAIV1.Allocate(tensors[0], allocatedMemId);
    if (bufferAIV1 == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation,
            "Failed to allocate in AIV1 for dual_dst: memId=%d", allocatedMemId);
        // Rollback AIV0 allocation
        bufferManagerAIV0.Free(bufferAIV0);
        return FAILED;
    }

    // Mark the memId as dual_dst
    dualDstMemIds_[allocatedMemId] = true;

    // Set tensor attributes
    for (auto& tensor : tensors) {
        if (tensor != nullptr) {
            tensor->SetMemId(allocatedMemId);
            tensor->SetAttribute("is_dual_dst", true);
            tensor->SetAttribute("dual_dst_aiv0_memId", allocatedMemId);
            tensor->SetAttribute("dual_dst_aiv1_memId", allocatedMemId);
        }
    }

    APASS_LOG_INFO_F(Elements::Operation,
        "Allocated dual_dst tensor at memId=%d in both AIV0 and AIV1, size=%zu",
        allocatedMemId, totalSize);

    return SUCCESS;
}

} // namespace npu::tile_fwk
