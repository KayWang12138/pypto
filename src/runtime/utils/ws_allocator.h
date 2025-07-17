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
#include <utility>

#include "rt_allocator.h"

namespace npu::tile_fwk {
namespace dynamic {
// Workspace suballocator (aligned to 8 bytes)
class WsAllocator {
    static constexpr uint64_t MIN_BLOCK_SIZE = 128;

public:
    using uintdevptr_t = uint64_t;

    struct BlockHeader;
    struct BlockFooter;

    struct BlockHeader {
        uintdevptr_t ptr{0};
        struct {
            uint64_t size   : 63;
            uint64_t isBusy : 1;
        };

        // ordered linked list
        BlockHeader *listPrev{nullptr};
        BlockHeader *listNext{nullptr};

        // linked list from freeListHead_, no order requirement
        BlockHeader *freeListPrev{nullptr};
        BlockHeader *freeListNext{nullptr};
    };

    struct WsAllocation {
        friend class WsAllocator;
        using uintdevptr_t = uint64_t;

        uintdevptr_t ptr{0};

        operator bool() const { return header != nullptr; }

        WsAllocation() = default;
        explicit WsAllocation(uintdevptr_t tptr) : ptr(tptr) {}

    private:
        WsAllocator::BlockHeader *header{nullptr};
    };

public:
    WsAllocator() = default;
    ~WsAllocator() = default;

    void Init(uintdevptr_t workspaceAddr, uint64_t workspaceSize);

    template <typename T>
    WsAllocation Allocate(uint64_t count) {
        return Malloc(count * sizeof(T));
    }

    WsAllocation TryMalloc(uint64_t memReq);
    WsAllocation Malloc(uint64_t memReq);

    void Deallocate(WsAllocation allocation);

private:
    // Pooling headers in the future
    BlockHeader *CreateHeader(uintdevptr_t ptr, uint64_t size);
    void DeleteHeader(BlockHeader *header);

    std::pair<BlockHeader *, bool> TryMergeBlockWithPrev(BlockHeader *header);
    bool TryMergeBlockWithNext(BlockHeader *header);

    bool TrySplitIntoTwo(BlockHeader *header, uint64_t firstBlockSize);

    void InsertIntoList(BlockHeader *header, BlockHeader *prev);
    void RemoveFromList(BlockHeader *header);

    void InsertIntoFreeList(BlockHeader *header);
    void RemoveFromFreeList(BlockHeader *header);

    static constexpr uint64_t AlignedPow2(uint64_t value, uint64_t alignment) {
        return (value + alignment - 1) & ~(alignment - 1);
    }

    static constexpr uint64_t Lowbit(uint64_t x) { return x & (~x + 1); }

    // (0, maxAlignedToPoW2]    ==>  round-up to power of 2
    // (maxAlignedToPoW2, ...)  ==>  round-up by times of maxAlignedToPoW2
    static constexpr uint64_t Aligned(uint64_t size) {
        constexpr uint64_t maxAlignedToPoW2 = 1024;

        if (size >= maxAlignedToPoW2) {
            return AlignedPow2(size, maxAlignedToPoW2);
        }
        // size cannot be 1 in this case otherwise it should be
        return UINT64_C(1) << (64 - __builtin_clzl(size - 1)); // 64 is bits per uint64
    }

private:
    uintdevptr_t workspaceAddr_{0};
    uint64_t workspaceSize_{0};

    BlockHeader *listHead_{nullptr};
    BlockHeader *freeListHead_{nullptr};

    // single directional linked-list
    BlockHeader *headerPool_{nullptr};

    RtAllocator headerAllocator_;
};

using WsAllocation = WsAllocator::WsAllocation;
} // namespace dynamic
} // namespace npu::tile_fwk
