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
 * \file rt_allocator.h
 * \brief
 */

#pragma once

#include <cstdlib>
#include <cstdint>
#include <cstddef>
#include <algorithm>

namespace npu::tile_fwk {
namespace dynamic {
class RtAllocator {
private:
    struct BlockNode;

public:
    struct RtAllocation {
        void *ptr{nullptr};

        operator bool() const { return node_ != nullptr; }

        void Invalidate() {
            ptr = nullptr;
            node_ = nullptr;
        }

    private:
        friend class RtAllocator;
        BlockNode *node_{nullptr};
    };

public:
    RtAllocator() = default;

    ~RtAllocator() { Reset(); }

    void SetBlockSizeHint(size_t blockSizeHint) { blockSizeHint_ = blockSizeHint; }

    template <typename T>
    RtAllocation Allocate(uint64_t count) {
        return Malloc(count * sizeof(T));
    }

    RtAllocation Malloc(size_t size) {
        size_t aligned = Aligned(size);

        while (active_ && active_->allocated + aligned > active_->size) {
            BlockNode *node = active_;
            active_ = active_->next;
            if (node->refCnt == 0) {
                DetachNode(node);
                InternalFree(node);
            }
        }

        if (!active_) {
            AllocateNewNode(std::max(blockSizeHint_, aligned));
        }

        RtAllocation allocation;
        allocation.ptr = active_->content + active_->allocated;
        allocation.node_ = active_;

        active_->allocated += aligned;
        active_->refCnt++;

        return allocation;
    }

    void Deallocate(RtAllocation allocation) {
        BlockNode *node = allocation.node_;
        node->refCnt--;
        if (node->refCnt == 0) {
            if (active_ == node) {
                active_ = node->next;
            }

            DetachNode(node);
            node->allocated = 0;
            InsertNode(node);
        }
    }

    void Reset() {
        for (BlockNode *node = tail_; node;) {
            BlockNode *curr = node;
            node = node->prev;
            InternalFree(curr);
        }
        tail_ = nullptr;
        active_ = nullptr;
    }

private:
    void *InternalMalloc(size_t size) {
        return (void *)malloc(size);
    }

    void InternalFree(BlockNode *node) {
        free(node);
    }

private:
    static constexpr uint64_t AlignedPow2(uint64_t value, uint64_t alignment) {
        return (value + alignment - 1) & ~(alignment - 1);
    }

    // (0, MAX_ALIGNED_TO_POW2]    ==>  round-up to power of 2
    // (MAX_ALIGNED_TO_POW2, ...)  ==>  round-up by times of MAX_ALIGNED_TO_POW2
    static constexpr uint64_t Aligned(uint64_t size) {
        constexpr uint64_t MAX_ALIGNED_TO_POW2 = 1024;

        if (size >= MAX_ALIGNED_TO_POW2) {
            return AlignedPow2(size, MAX_ALIGNED_TO_POW2);
        }
        // size cannot be 1 in this case otherwise it should be
        return UINT64_C(1) << (64 - __builtin_clzl(size - 1)); // 64 is bits per uint64
    }

    void AllocateNewNode(size_t size) {
        BlockNode *newNode = (BlockNode *)InternalMalloc(sizeof(BlockNode) + size);
        if (newNode != nullptr) {
            newNode->prev = nullptr;
            newNode->next = nullptr;
            newNode->refCnt = 0;
            newNode->size = size;
            newNode->allocated = 0;
            InsertNode(newNode);
        }
    }

    void InsertNode(BlockNode *node) {
        if (tail_) {
            tail_->next = node;
        }
        node->prev = tail_;
        tail_ = node;

        if (active_ == nullptr) {
            active_ = node;
        }
    }

    void DetachNode(BlockNode *node) {
        if (node->prev) {
            node->prev->next = node->next;
        }
        if (node->next) {
            node->next->prev = node->prev;
        } else {
            tail_ = node->prev;
        }
        node->prev = nullptr;
        node->next = nullptr;
    }

private:
    struct BlockNode {
        BlockNode *prev{nullptr};
        BlockNode *next{nullptr};
        uint32_t refCnt{0};
        size_t size;
        size_t allocated{0};
        std::byte content[];
    };

    BlockNode *tail_{nullptr};
    BlockNode *active_{nullptr};

    static constexpr size_t DEFAULT_BLOCKSIZE = 16 * 1024;
    size_t blockSizeHint_{DEFAULT_BLOCKSIZE};
};

using RtAllocation = RtAllocator::RtAllocation;

} // namespace dynamic
} // namespace npu::tile_fwk
