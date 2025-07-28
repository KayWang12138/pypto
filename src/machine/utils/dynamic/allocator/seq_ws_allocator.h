/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file seq_ws_allocator.h
 * \brief
 */

#pragma once

#include "ws_allocator_counter.h"

#include "machine/utils/dynamic/sheet_formatter.h"
#include "machine/utils/device_switch.h"

#include <map>

namespace npu::tile_fwk::dynamic {

// Sequential workspace allocator
class SeqWsAllocator {
    using uintdevptr_t = uint64_t;

public:
    void InitAicpuCoherent(uintdevptr_t workspaceAddr, uint64_t workspaceSize) {
        InternalInit(workspaceAddr, workspaceSize, WsAllocatorProperty::AICPU_COHERENT);
    }

    void InitAicoreLocal(uintdevptr_t workspaceAddr, uint64_t workspaceSize) {
        InternalInit(workspaceAddr, workspaceSize, WsAllocatorProperty::AICORE_LOCAL);
    }

    bool CanAllocate(uint64_t memReq) const {
        return allocated_ + memReq <= workspaceSize_;
    }

    template <typename T>
    WsAllocation Allocate(uint64_t count, WsMemCategory category = WsMemCategory::UNCLASSIFIED) {
        return Malloc(count * sizeof(T), category);
    }

    WsAllocation Malloc(uint64_t memReq, WsMemCategory category = WsMemCategory::UNCLASSIFIED) {
        DEV_DEBUG_ASSERT(CanAllocate(memReq));

        WsAllocation allocation;
        allocation.ptr = workspaceAddr_ + allocated_;
        allocation.node_ = reinterpret_cast<void *>(0xDEADBEEFDEADBEEF); // Random value, to make allocation valid(void *)0xDEADBEEFDEADBEEF; // Random value, to make allocation valid
#if DEBUG_MEM_DUMP_LEVEL >= DEBUG_MEM_DUMP_FULL
        dfx_.categories[category].totalAllocNum++;
        dfx_.categories[category].totalMemReq += memReq;
        allocation.rawMemReq_ = memReq;
        allocation.category_ = category;
        dfx_.memCounter.LogMalloc(allocation);
#endif // DEBUG_MEM_DUMP_LEVEL >= DEBUG_MEM_DUMP_FULL

#if DEBUG_MEM_DUMP_LEVEL >= DEBUG_MEM_DUMP_LIGHT
        dfx_.totalMemReq += memReq;
        dfx_.allocNum++;
#endif // DEBUG_MEM_DUMP_LEVEL >= DEBUG_MEM_DUMP_LIGHT

        allocated_ += memReq;
        (void)category;
        return allocation;
    }

    void Deallocate(WsAllocation) {}

    void ResetPool() {
        allocated_ = 0;
        resetTimes_++;
    }

    uint32_t ResetTimes() const {
        return resetTimes_;
    }

    // Call me after initialization memory allocations
    void ResetCounter() {
#if DEBUG_MEM_DUMP_LEVEL >= DEBUG_MEM_DUMP_FULL
        dfx_.memCounter.Reset();
#endif // DEBUG_MEM_DUMP_LEVEL >= DEBUG_MEM_DUMP_FULL
    }

    void DelayedDumpAndResetCounter(DelayedDumper &dumper) {
#if DEBUG_MEM_DUMP_LEVEL >= DEBUG_MEM_DUMP_FULL
        dfx_.memCounter.DelayedDumpAsAicpuCounterAndReset(dumper);
#endif // DEBUG_MEM_DUMP_LEVEL >= DEBUG_MEM_DUMP_FULL
        (void)dumper;
    }

    WsAllocatorCounter *GetCounter() {
#if DEBUG_MEM_DUMP_LEVEL >= DEBUG_MEM_DUMP_FULL
        return &dfx_.memCounter;
#else
        return nullptr;
#endif // DEBUG_MEM_DUMP_LEVEL >= DEBUG_MEM_DUMP_FULL
    }

    void DumpMemoryUsage(const char *hint, const char *title) const {
#if DEBUG_MEM_DUMP_LEVEL >= DEBUG_MEM_DUMP_LIGHT
        DEV_MEM_DUMP("%s memory usage (%s)\n", title, hint);
        DEV_MEM_DUMP("            Memory pool size: %10lu bytes\n", workspaceSize_);
        DEV_MEM_DUMP("    Total memory requirement: %10zu bytes\n", dfx_.totalMemReq);
        DEV_MEM_DUMP("      Total allocation count: %10zu\n", dfx_.allocNum);
#endif // DEBUG_MEM_DUMP_LEVEL >= DEBUG_MEM_DUMP_LIGHT
        (void)title;
        (void)hint;
    }

    bool IsAicoreLocal() const { return property_ == WsAllocatorProperty::AICORE_LOCAL; }
    bool IsAicpuCoherent() const { return property_ == WsAllocatorProperty::AICPU_COHERENT; }

    uint64_t MemBaseAddr() const { return workspaceAddr_; }
    uint64_t AllocatedSize() const { return allocated_; }
    uint64_t FreeMemorySize() { return  workspaceSize_ - allocated_; }

private:
    void InternalInit(uintdevptr_t workspaceAddr, uint64_t workspaceSize, WsAllocatorProperty property) {
        property_ = property;
        workspaceAddr_ = workspaceAddr;
        workspaceSize_ = workspaceSize;
        allocated_ = 0;
        resetTimes_ = 0;
    }

private:
    WsAllocatorProperty property_;

    uintdevptr_t workspaceAddr_{0};
    uint64_t workspaceSize_{0};
    uint64_t allocated_{0};
    uint32_t resetTimes_{0};

#if DEBUG_MEM_DUMP_LEVEL >= DEBUG_MEM_DUMP_LIGHT
    struct DfxInfo {
#if DEBUG_MEM_DUMP_LEVEL >= DEBUG_MEM_DUMP_FULL
        struct ClassifiedDfxInfo {
            size_t totalAllocNum{0};
            size_t totalMemReq{0};
        };
        std::map<WsMemCategory, ClassifiedDfxInfo> categories;
        WsAllocatorCounter memCounter;
#endif // DEBUG_MEM_DUMP_LEVEL >= DEBUG_MEM_DUMP_FULL

        size_t totalMemReq{0};
        size_t allocNum{0};
    } dfx_;
#endif // DEBUG_MEM_DUMP_LEVEL >= DEBUG_MEM_DUMP_LIGHT
};

} // namespace npu::tile_fwk::dynamic