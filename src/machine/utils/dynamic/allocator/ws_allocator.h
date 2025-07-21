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
 * \file ws_allocator.h
 * \brief
 */

#pragma once

#include <cstdint>
#include <cinttypes>
#include <utility>
#include <thread>

#include "ws_allocator_basics.h"
#include "machine/utils/device_switch.h"
#include "machine/utils/device_log.h"
#include "machine/device/dynamic/device_utils.h"

namespace npu::tile_fwk::dynamic {

#define BLOCK_DEBUG(header, fmt, args...) \
    DEV_DEBUG("[WsAllocator (%" PRIu64 ")] " fmt " | @%p, ptr offset: %" PRIu64 " , block size: %" PRIu64 "\n", \
        workspaceSize_, ##args, header, (uintdevptr_t)header->ptr - workspaceAddr_, (uint64_t)header->size)

// Workspace suballocator (aligned to 8 bytes)
class WsAllocator {
    static constexpr uint64_t MIN_BLOCK_SIZE = 512;

public:
    using uintdevptr_t = uint64_t;

    struct BlockHeader {
        uintdevptr_t ptr;
        struct {
            uint64_t size   : 63;
            uint64_t isBusy : 1;
        };

        // ordered linked list
        BlockHeader *listPrev;
        BlockHeader *listNext;

        // linked list from freeListHead_, no order requirement
        BlockHeader *freeListPrev;
        BlockHeader *freeListNext;

        void Init() {
            memset_s(this, sizeof(BlockHeader), 0, sizeof(BlockHeader));
        }

        static bool CheckNeighbors(BlockHeader *prev, BlockHeader *next) {
            return prev->ptr + prev->size == next->ptr;
        }
    };

public:
    WsAllocator() = default;
    ~WsAllocator() = default;

    WsAllocator(const WsAllocator &) = delete;
    WsAllocator(WsAllocator &&) = delete;
    void operator==(const WsAllocator &) = delete;
    void operator==(WsAllocator &&) = delete;

    bool QuickReject(uint64_t memReq) const {
        return availablePoolSize_ < memReq;
    }

    // aicore local init, header memory must be included in workspace memory (maxHeaderCount * sizeof(BlockHeader))
    void InitAicoreLocal(uintdevptr_t workspaceAddr, uint64_t workspaceSize, size_t maxHeaderCount) {
        InternalInit(workspaceAddr, workspaceSize, WsAllocatorProperty::AICORE_LOCAL, maxHeaderCount);
    }

    // aicpu coherent init, headers will be dynamically attached to allocated block (isolated to users)
    void InitAicpuCoherent(uintdevptr_t workspaceAddr, uint64_t workspaceSize) {
        InternalInit(workspaceAddr, workspaceSize, WsAllocatorProperty::AICPU_COHERENT, 0);
    }

    template <typename T>
    WsAllocation Allocate(uint64_t count, WsMemCategory category = WsMemCategory::UNCLASSIFIED) {
        return Malloc(count * sizeof(T), category);
    }

    WsAllocation TryMalloc(uint64_t memReq, WsMemCategory category = WsMemCategory::UNCLASSIFIED) {
        AutoScopedPerf asp(IsAicpuCoherent() ? PERF_EVT_WSALLOC_CPU_A : PERF_EVT_WSALLOC_CORE_A);

#if DEBUG_SWITCH
        DEV_ASSERT(std::this_thread::get_id() == owner_);
#endif // DEBUG_SWITCH

        if (memReq == 0) {
            DEV_DEBUG("[WsAllocator (%" PRIu64 ")] Failed to allocate %" PRIu64 "\n", workspaceSize_, memReq);
            return WsAllocation{};
        }

        uint64_t unalignedMemReq = memReq;
        if (IsAicpuCoherent()) {
            unalignedMemReq += sizeof(BlockHeader);
        }
        uint64_t alignedMemReq = Aligned(unalignedMemReq);
        for (BlockHeader *header = freeListHead_; header;) {
            DEV_ASSERT_MSG(!header->isBusy,
                "[WsAllocator (%" PRIu64 ")] ptr offset: %" PRIu64 ", block size: %" PRIu64 "\n",
                workspaceSize_, reinterpret_cast<uintdevptr_t>(header->ptr) - workspaceAddr_, header->size);
            if (header->size >= alignedMemReq) {
                if (category == WsMemCategory::TENSOR_ROOTFUNC_INTERNAL) {
                    auto res = TrySplitIntoTwo(header, header->size - alignedMemReq);
                    if (res.second == true) {
                        header = res.first;
                    }
                } else {
                    (void)TrySplitIntoTwo(header, alignedMemReq);
                }

                RemoveFromFreeList(header);
                header->isBusy = true;

                WsAllocation allocation;
                allocation.ptr = header->ptr;
                if (IsAicpuCoherent()) {
                    allocation.ptr += sizeof(BlockHeader);
                }
                allocation.node_ = header;
#if DEBUG_MEM_DUMP_LEVEL >= DEBUG_MEM_DUMP_FULL
                allocation.rawMemReq_ = memReq;
                allocation.category_ = category;
#endif // DEBUG_MEM_DUMP_LEVEL >= DEBUG_MEM_DUMP_FULL
                availablePoolSize_ -= header->size;

#if DEBUG_MEM_DUMP_LEVEL >= DEBUG_MEM_DUMP_LIGHT
                dfx_.totalMemReq += memReq;
                dfx_.allocNum++;
#endif // DEBUG_MEM_DUMP_LEVEL >= DEBUG_MEM_DUMP_LIGHT

                return allocation;
            }

            header = header->freeListNext;
        }

        for (BlockHeader *header = freeListHead_; header; header = header->freeListNext) {
            BLOCK_DEBUG(header, "Free list");
        }
        DEV_DEBUG("[WsAllocator (%" PRIu64 ")] Failed to allocate %" PRIu64 "\n", workspaceSize_, memReq);
        return WsAllocation{}; // failed
    }

    WsAllocation Malloc(uint64_t memReq, WsMemCategory category = WsMemCategory::UNCLASSIFIED) {
        DEV_ASSERT(memReq != 0);

        WsAllocation allocation = TryMalloc(memReq, category);
        DEV_ASSERT(allocation.ptr);
        return allocation;
    }

    void Deallocate(WsAllocation allocation) {
        AutoScopedPerf asp(IsAicpuCoherent() ? PERF_EVT_WSALLOC_CPU_D : PERF_EVT_WSALLOC_CORE_D);

#if DEBUG_SWITCH
        DEV_ASSERT(std::this_thread::get_id() == owner_);
#endif // DEBUG_SWITCH

        DEV_ASSERT(allocation);

        uintdevptr_t ptr = allocation.ptr;
        if (IsAicpuCoherent()) {
            ptr -= sizeof(BlockHeader);
        }
        BlockHeader *header = ((BlockHeader *)allocation.node_);
        availablePoolSize_ += header->size;

        DEV_ASSERT(ptr == header->ptr);
        DEV_ASSERT(workspaceAddr_ <= ptr && ptr < workspaceAddr_ + workspaceSize_);

        header->isBusy = false;

        if (auto res = TryMergeBlockWithPrev(header); res.second) {
            header = res.first;
        }

        (void)TryMergeBlockWithNext(header);

#if DEBUG_SWITCH
        CheckContinuity(header);
#endif // DEBUG_SWITCH

        InsertIntoFreeList(header);
    }

    size_t AvailableMemory(WsAllocation allocation) const {
        if (!allocation) {
            return 0;
        }
        size_t available = ((BlockHeader *)allocation.node_)->size;
        if (IsAicpuCoherent()) {
            available -= sizeof(BlockHeader);
        }
        return available;
    }

    WsAllocation BuildAicpuCoherentWsAllocation(uintdevptr_t ptr) {
        DEV_ASSERT(IsAicpuCoherent() && workspaceAddr_ <= ptr && ptr < workspaceAddr_ + workspaceSize_);
        WsAllocation alloc;
        alloc.ptr = ptr;
        alloc.node_ = reinterpret_cast<void *>(ptr - sizeof(BlockHeader));
        return alloc;
    }

    void DumpMemoryUsage(const char *hint) const {
#if DEBUG_MEM_DUMP_LEVEL >= DEBUG_MEM_DUMP_LIGHT
        DEV_MEM_DUMP("%s memory usage (%s)\n", IsAicoreLocal() ? "Tensor" : "Metadata", hint);
        DEV_MEM_DUMP("  Total memory requirement: %zu bytes\n", dfx_.totalMemReq);
        DEV_MEM_DUMP("  Total allocation count: %zu\n", dfx_.allocNum);
#endif // DEBUG_MEM_DUMP_LEVEL >= DEBUG_MEM_DUMP_LIGHT
        (void)hint;
    }

private:
    void InternalInit(uintdevptr_t workspaceAddr, uint64_t workspaceSize,
                      WsAllocatorProperty property, size_t maxHeaderCount) {
        workspaceAddr_ = workspaceAddr;
        workspaceSize_ = workspaceSize;
        availablePoolSize_ = workspaceSize;
        property_ = property;

        listHead_ = nullptr;
        freeListHead_ = nullptr;
        headerPool_ = nullptr;

        uint64_t availableSize = workspaceSize_;
        if (IsAicoreLocal()) {
            DEV_ASSERT(maxHeaderCount != 0);
            DEV_ASSERT(workspaceSize_ > maxHeaderCount * sizeof(BlockHeader));
            availableSize = workspaceSize_ - maxHeaderCount * sizeof(BlockHeader);
            BlockHeader *headers = reinterpret_cast<BlockHeader *>(workspaceAddr_ + availableSize);
            for (size_t i = 0; i < maxHeaderCount; i++) {
                AddAvailableHeader(headers + i);
            }
            availablePoolSize_ = availableSize;
        }

        BlockHeader *header = CreateHeader(workspaceAddr_, availableSize);
        DEV_ASSERT(header);
        InsertIntoList(header, nullptr);
        InsertIntoFreeList(header);
    }

    // Pooling headers in the future
    BlockHeader *CreateHeader(uintdevptr_t ptr, uint64_t size) {
        if (IsAicpuCoherent()) {
            BlockHeader *header = reinterpret_cast<BlockHeader *>(ptr);
            header->Init();
            header->ptr = ptr;
            header->size = size;
            return header;
        }

        // WsAllocatorProperty::AICORE_LOCAL
        if (!headerPool_) {
            return nullptr;
        }

        BlockHeader *header = headerPool_;
        headerPool_ = headerPool_->listNext;

        header->Init();
        header->ptr = ptr;
        header->size = size;
        return header;
    }

    void AddAvailableHeader(BlockHeader *header) {
        // Must be removed from list first

        // single directional linked-list
        header->listNext = headerPool_;
        headerPool_ = header;
    }

    std::pair<BlockHeader *, bool> TryMergeBlockWithPrev(BlockHeader *header) {
        BlockHeader *prev = header->listPrev;
        if (!prev || prev->isBusy) {
            return std::make_pair(nullptr, false);
        }

        RemoveFromFreeList(prev);

        prev->size += header->size;

        RemoveFromList(header);
        AddAvailableHeader(header);

        return std::make_pair(prev, true);
    }

    bool TryMergeBlockWithNext(BlockHeader *header) {
        BlockHeader *next = header->listNext;
        if (!next || next->isBusy) {
            return false;
        }

        RemoveFromFreeList(next);

        header->size += next->size;

        RemoveFromList(next);
        AddAvailableHeader(next);

        return true;
    }

    std::pair<BlockHeader *, bool> TrySplitIntoTwo(BlockHeader *header, uint64_t firstBlockSize) {
        if (firstBlockSize < MIN_BLOCK_SIZE || header->size < firstBlockSize + MIN_BLOCK_SIZE) {
            // We don't keep too small block in free list
            return std::make_pair(nullptr, false);
        }

        uint64_t secondBlockSize = header->size - firstBlockSize;

        BlockHeader *next = CreateHeader(header->ptr + firstBlockSize, secondBlockSize);
        if (!next) {
            return std::make_pair(nullptr, false);
        }
        header->size = firstBlockSize;

        InsertIntoList(next, header);
        InsertIntoFreeList(next);

#if DEBUG_SWITCH
        CheckContinuity(next);
#endif // DEBUG_SWITCH

        return std::make_pair(next, true);
    }

    void InsertIntoList(BlockHeader *header, BlockHeader *prev) {
        if (!prev) {
            header->listNext = listHead_;
            if (listHead_) {
                listHead_->listPrev = header;
            }
            listHead_ = header;
        } else {
            header->listNext = prev->listNext;
            if (prev->listNext) {
                prev->listNext->listPrev = header;
            }
            header->listPrev = prev;
            prev->listNext = header;
        }
    }

    void RemoveFromList(BlockHeader *header) {
        if (header->listPrev) {
            header->listPrev->listNext = header->listNext;
        } else {
            listHead_ = header->listNext;
        }

        if (header->listNext) {
            header->listNext->listPrev = header->listPrev;
        }

        header->listPrev = nullptr;
        header->listNext = nullptr;
    }

    void InsertIntoFreeList(BlockHeader *header) {
        header->freeListNext = freeListHead_;
        if (freeListHead_) {
            freeListHead_->freeListPrev = header;
        }
        freeListHead_ = header;
    }

    void RemoveFromFreeList(BlockHeader *header) {
        if (header->freeListPrev) {
            header->freeListPrev->freeListNext = header->freeListNext;
        } else {
            freeListHead_ = header->freeListNext;
        }

        if (header->freeListNext) {
            header->freeListNext->freeListPrev = header->freeListPrev;
        }

        header->freeListPrev = nullptr;
        header->freeListNext = nullptr;
    }

    void CheckContinuity(BlockHeader *header) const {
        if (header->listPrev) {
            if (!BlockHeader::CheckNeighbors(header->listPrev, header)) {
                BLOCK_DEBUG(header->listPrev, "listPrev");
                BLOCK_DEBUG(header, "header");
                DEV_ASSERT(false);
            }
        }
        if (header->listNext) {
            if (!BlockHeader::CheckNeighbors(header, header->listNext)) {
                BLOCK_DEBUG(header, "header");
                BLOCK_DEBUG(header->listNext, "listNext");
                DEV_ASSERT(false);
            }
        }
    }

    bool IsAicoreLocal() const { return property_ == WsAllocatorProperty::AICORE_LOCAL; }
    bool IsAicpuCoherent() const { return property_ == WsAllocatorProperty::AICPU_COHERENT; }

    static constexpr uint64_t AlignedPow2(uint64_t value, uint64_t alignment) {
        return (value + alignment - 1) & ~(alignment - 1);
    }

    static constexpr uint64_t Lowbit(uint64_t x) { return x & (~x + 1); }

    // (0, MAX_ALIGNED_TO_POW2]    ==>  round-up to power of 2
    // (MAX_ALIGNED_TO_POW2, ...)  ==>  round-up by times of MAX_ALIGNED_TO_POW2
    static constexpr uint64_t Aligned(uint64_t size) {
        constexpr uint64_t MAX_ALIGNED_TO_POW2 = 1024;

        if (size < MIN_BLOCK_SIZE) {
            return MIN_BLOCK_SIZE;
        }
        if (size >= MAX_ALIGNED_TO_POW2) {
            return AlignedPow2(size, MAX_ALIGNED_TO_POW2);
        }
        // size cannot be 1 in this case otherwise it should be
        //   size == 1 ? UINT64_C(1) : UINT64_C(1) << (64 - __builtin_clzl(size - 1))
        return UINT64_C(1) << (64 - __builtin_clzl(size - 1)); // 64 is bits per uint64
    }

private:
#if DEBUG_SWITCH
    std::thread::id owner_{std::this_thread::get_id()};
#endif // DEBUG_SWITCH

    uintdevptr_t workspaceAddr_{0};
    uint64_t workspaceSize_{0};
    uint64_t availablePoolSize_{0};

    BlockHeader *listHead_{nullptr};
    BlockHeader *freeListHead_{nullptr};

    // single directional linked-list
    BlockHeader *headerPool_{nullptr};

    WsAllocatorProperty property_;

#if DEBUG_MEM_DUMP_LEVEL >= DEBUG_MEM_DUMP_LIGHT
    struct DfxInfo {
        size_t totalMemReq{0};
        size_t allocNum{0};
    } dfx_;
#endif // DEBUG_MEM_DUMP_LEVEL >= DEBUG_MEM_DUMP_LIGHT
};

#undef BLOCK_DEBUG

} // namespace npu::tile_fwk::dynamic
