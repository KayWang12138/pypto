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
 * \file ws_rt_allocator.h
 * \brief
 */

#pragma once

#include "ws_allocator.h"
#include "ws_allocator_counter.h"

#include "runtime/utils/dynamic/sheet_formatter.h"
#include "runtime/utils/device_switch.h"

#include <map>

#define SIMPLE_SEQUENTIAL_NORECYCLE_ALLOCATION_STRAT 0

namespace npu::tile_fwk::dynamic {

class WsRtAllocator {
    using uintdevptr_t = uint64_t;

private:
    struct BlockNode {
        BlockNode *next{nullptr};
        uint32_t refCnt{0};
        size_t size{0};
        size_t allocated{0};
        uint8_t content[];

        WsAllocation SelfAlloc(WsAllocator &allocator) const {
            return allocator.BuildAicpuCoherentWsAllocation(reinterpret_cast<uintdevptr_t>(this));
        }
    };

    static constexpr size_t DEFAULT_BLOCKSIZE = 32 * 1024 - sizeof(BlockNode) - sizeof(WsAllocator::BlockHeader);
    static constexpr size_t LARGE_NODE_THRESHOLD = 16 * (32 * 1024);

public:
    WsRtAllocator() = default;

    WsRtAllocator(const WsRtAllocator &) = delete;
    WsRtAllocator(WsRtAllocator &&oth) = delete;
    void operator==(const WsRtAllocator &) = delete;
    void operator==(WsRtAllocator &&) = delete;

    ~WsRtAllocator() = default;

    void InitAicpuCoherent(uintdevptr_t workspaceAddr, uint64_t workspaceSize) {
#if !SIMPLE_SEQUENTIAL_NORECYCLE_ALLOCATION_STRAT
        allocator_.InitAicpuCoherent(workspaceAddr, workspaceSize);
#else // SIMPLE_SEQUENTIAL_NORECYCLE_ALLOCATION_STRAT: ^^^ 0 / 1 vvv
        workspaceAddr_ = workspaceAddr;
        workspaceSize_ = workspaceSize;
        allocated_ = 0;
#endif // !SIMPLE_SEQUENTIAL_NORECYCLE_ALLOCATION_STRAT
    }

    template <typename T>
    WsAllocation Allocate(uint64_t count, WsMemCategory category = WsMemCategory::UNCLASSIFIED) {
        return Malloc(count * sizeof(T), category);
    }

    WsAllocation Malloc(size_t memReq, WsMemCategory category = WsMemCategory::UNCLASSIFIED) {
        AutoScopedPerf asp(PERF_EVT_WSRTALLOC_CPU_A);

#if DEBUG_SWITCH
        DEV_ASSERT(std::this_thread::get_id() == owner_);
#endif // DEBUG_SWITCH

        size_t aligned = AlignedPow2(memReq, 8);
#if !SIMPLE_SEQUENTIAL_NORECYCLE_ALLOCATION_STRAT
        while (active_ && active_->allocated + aligned > active_->size) {
            BlockNode *node = active_;
            active_ = active_->next;
            node->next = nullptr;
            if (node->refCnt == 0) {
                DeallocateNode(node);
            }
        }

        if (!active_) {
            AllocateNewNode(std::max(DEFAULT_BLOCKSIZE, aligned));
        }

        WsAllocation allocation;
        allocation.ptr = reinterpret_cast<uintdevptr_t>(active_->content) + active_->allocated;
        allocation.node_ = active_;

        active_->allocated += aligned;
        active_->refCnt++;
#else // SIMPLE_SEQUENTIAL_NORECYCLE_ALLOCATION_STRAT: ^^^ 0 / 1 vvv
        DEV_ASSERT(allocated_ + aligned <= workspaceSize_);

        WsAllocation allocation;
        allocation.ptr = workspaceAddr_ + allocated_;
        allocation.node_ = reinterpret_cast<void *>(0xDEADBEEFDEADBEEF);
        allocated_ += aligned;
#endif // !SIMPLE_SEQUENTIAL_NORECYCLE_ALLOCATION_STRAT

#if DEBUG_MEM_DUMP_LEVEL >= DEBUG_MEM_DUMP_LIGHT
        dfx_.totalMemReq += memReq;
        dfx_.allocNum++;
#endif // DEBUG_MEM_DUMP_LEVEL >= DEBUG_MEM_DUMP_LIGHT

#if DEBUG_MEM_DUMP_LEVEL >= DEBUG_MEM_DUMP_FULL
        dfx_.categories[category].totalAllocNum++;
        dfx_.categories[category].totalMemReq += memReq;
        allocation.rawMemReq_ = memReq;
        allocation.category_ = category;
        dfx_.memCounter.LogMalloc(allocation);
#endif // DEBUG_MEM_DUMP_LEVEL >= DEBUG_MEM_DUMP_FULL
        (void)category;
        return allocation;
    }

    void Deallocate(WsAllocation allocation) {
#if !SIMPLE_SEQUENTIAL_NORECYCLE_ALLOCATION_STRAT
        AutoScopedPerf asp(PERF_EVT_WSRTALLOC_CPU_D);

#if DEBUG_SWITCH
        DEV_ASSERT(std::this_thread::get_id() == owner_);
#endif // DEBUG_SWITCH

#if DEBUG_MEM_DUMP_LEVEL >= DEBUG_MEM_DUMP_FULL
        dfx_.categories[allocation.category_].totalAllocNum--;
        dfx_.categories[allocation.category_].totalMemReq -= allocation.rawMemReq_;
        dfx_.memCounter.LogDealloc(allocation);
#endif // DEBUG_MEM_DUMP_LEVEL >= DEBUG_MEM_DUMP_FULL

        BlockNode *node = reinterpret_cast<BlockNode *>(allocation.node_);
        node->refCnt--;
        if (node->refCnt == 0) {
            node->allocated = 0;
            if (active_ == node) {
                return;
            }

            InsertNode(node);
        }
#endif // !SIMPLE_SEQUENTIAL_NORECYCLE_ALLOCATION_STRAT
    }

    void DumpMemoryUsage(const char *hint) const {
#if DEBUG_MEM_DUMP_LEVEL >= DEBUG_MEM_DUMP_LIGHT
        DEV_MEM_DUMP("Metadata memory usage (%s)\n", hint);

#if DEBUG_MEM_DUMP_LEVEL >= DEBUG_MEM_DUMP_FULL
        SheetFormatter sheet({"Mem Category", "Alloc Num", "Total Mem Req"});
        for (auto &&[key, value] : dfx_.categories) {
            if (!value.totalAllocNum || !value.totalMemReq) {
                DEV_ASSERT(!value.totalAllocNum && !value.totalMemReq);
                continue;
            }
            sheet.AddRow(
                GetCategoryName(key),
                value.totalAllocNum,
                value.totalMemReq
            );
        }
        auto lines = sheet.DumpLines();
        for (auto &&line : lines) {
            DEV_MEM_DUMP("%s\n", line.c_str());
            (void)line;
        }
#endif // DEBUG_MEM_DUMP_LEVEL >= DEBUG_MEM_DUMP_FULL

        DEV_MEM_DUMP("  Total memory requirement: %zu bytes\n", dfx_.totalMemReq);
        DEV_MEM_DUMP("  Total allocation count: %zu\n", dfx_.allocNum);
#endif // DEBUG_MEM_DUMP_LEVEL >= DEBUG_MEM_DUMP_LIGHT
        (void)hint;
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

    // Call me after initialization memory allocations
    void ResetCounter() {
#if DEBUG_MEM_DUMP_LEVEL >= DEBUG_MEM_DUMP_FULL
        dfx_.memCounter.Reset();
#endif // DEBUG_MEM_DUMP_LEVEL >= DEBUG_MEM_DUMP_FULL
    }

private:
    void Release() {
        tail_ = nullptr;
        active_ = nullptr;
    }

    static constexpr uint64_t AlignedPow2(uint64_t value, uint64_t alignment) {
        return (value + alignment - 1) & ~(alignment - 1);
    }

    static constexpr uint64_t RoundupNextPow2(uint64_t value) {
        // value cannot be 1 in this case otherwise it should be
        //   value == 1 ? UINT64_C(1) : UINT64_C(1) << (64 - __builtin_clzl(value - 1))
        return UINT64_C(1) << (64 - __builtin_clzl(value - 1)); // 64 is bits per uint64
    }

#if !SIMPLE_SEQUENTIAL_NORECYCLE_ALLOCATION_STRAT
    void AllocateNewNode(size_t size) {
        size_t alignedMemReq = RoundupNextPow2(size + sizeof(BlockNode) + sizeof(WsAllocator::BlockHeader))
            - sizeof(WsAllocator::BlockHeader);
        WsAllocation alloc = allocator_.Malloc(alignedMemReq, WsMemCategory::WS_RT_ALLOCATOR_MEM_BLOCK);
        BlockNode *newNode = reinterpret_cast<BlockNode *>(alloc.ptr);
        newNode->next = nullptr;
        newNode->refCnt = 0;
        newNode->size = allocator_.AvailableMemory(alloc) - sizeof(BlockNode);
        newNode->allocated = 0;
        InsertNode(newNode);
    }

    void DeallocateNode(BlockNode *node) {
        allocator_.Deallocate(node->SelfAlloc(allocator_));
    }

    void InsertNode(BlockNode *node) {
        if (tail_) {
            tail_->next = node;
        }
        tail_ = node;

        if (active_ == nullptr) {
            active_ = node;
        }
    }
#endif // !SIMPLE_SEQUENTIAL_NORECYCLE_ALLOCATION_STRAT

private:
    BlockNode *tail_{nullptr};
    BlockNode *active_{nullptr};

#if !SIMPLE_SEQUENTIAL_NORECYCLE_ALLOCATION_STRAT
    WsAllocator allocator_;
#else // SIMPLE_SEQUENTIAL_NORECYCLE_ALLOCATION_STRAT: ^^^ 0 / 1 vvv
    uintdevptr_t workspaceAddr_{0};
    size_t workspaceSize_{0};
    size_t allocated_{0};
#endif // SIMPLE_SEQUENTIAL_NORECYCLE_ALLOCATION_STRAT

#if DEBUG_SWITCH
    std::thread::id owner_{std::this_thread::get_id()};
#endif // DEBUG_SWITCH

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