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
 * \file ws_allocator.cpp
 * \brief
 */

#include "ws_allocator.h"
#include "device_log.h"

namespace npu::tile_fwk {
namespace dynamic {
using BlockHeader = WsAllocator::BlockHeader;
using uintdevptr_t = uint64_t;

constexpr int DEFAULT_BLOCK_NUM = 64;

namespace {

void Init(BlockHeader *header) {
    header->ptr = 0;
    header->size = 0;
    header->isBusy = false;
    header->freeListPrev = nullptr;
    header->freeListNext = nullptr;
    header->listPrev = nullptr;
    header->listNext = nullptr;
}

} // anonymous namespace

void WsAllocator::Init(uintdevptr_t workspaceAddr, uint64_t workspaceSize) {
    workspaceAddr_ = workspaceAddr;
    workspaceSize_ = workspaceSize;
    listHead_ = nullptr;
    freeListHead_ = nullptr;
    headerPool_ = nullptr;
    headerAllocator_.Reset();
    headerAllocator_.SetBlockSizeHint(sizeof(BlockHeader) * DEFAULT_BLOCK_NUM);

    BlockHeader *header = CreateHeader(workspaceAddr_, workspaceSize_);
    InsertIntoList(header, nullptr);
    InsertIntoFreeList(header);
}

WsAllocation WsAllocator::TryMalloc(uint64_t memReq) {
    if (memReq == 0) {
        return WsAllocation{};
    }

    uint64_t alignedMemReq = Aligned(memReq);
    for (BlockHeader *header = freeListHead_; header;) {
        DEV_ASSERT_MSG(
            !header->isBusy, "size: %lu, offset to addr: %lu\n", header->size, reinterpret_cast<uintdevptr_t>(header) - workspaceAddr_);
        if (header->size >= alignedMemReq) {
            RemoveFromFreeList(header);

            (void)TrySplitIntoTwo(header, alignedMemReq);

            header->isBusy = true;

            WsAllocation allocation;
            allocation.ptr = static_cast<uintdevptr_t>(header->ptr);
            allocation.header = header;
            return allocation;
        }

        header = header->freeListNext;
    }

    return WsAllocation{}; // failed
}

WsAllocation WsAllocator::Malloc(uint64_t memReq) {
    DEV_ASSERT(memReq != 0);

    WsAllocation allocation = TryMalloc(memReq);
    if (!allocation.ptr) {
        for (BlockHeader *header = freeListHead_; header; header = header->freeListNext) {
            DEV_INFO("[WsAllocator (%lu)] Free List | ptr offset: %lu , block size: %lu", workspaceSize_,
                static_cast<uintdevptr_t>(header->ptr) - workspaceAddr_, static_cast<uint64_t>(header->size));
        }
        DEV_ASSERT_MSG(false, "Failed to allocate ");
    }
    return allocation;
}

void WsAllocator::Deallocate(WsAllocation allocation) {
    if (!allocation) {
        return;
    }

    uintdevptr_t ptr = allocation.ptr;
    DEV_ASSERT(ptr == allocation.header->ptr);
    DEV_ASSERT(workspaceAddr_ <= ptr && ptr < workspaceAddr_ + workspaceSize_);

    BlockHeader *header = allocation.header;
    header->isBusy = false;

    while (true) {
        auto res = TryMergeBlockWithPrev(header);
        if (!res.second) {
            break;
        }
        header = res.first;
    }

    while (TryMergeBlockWithNext(header)) {
    }

    InsertIntoFreeList(header);
}

BlockHeader *WsAllocator::CreateHeader(uintdevptr_t ptr, uint64_t size) {
    if (headerPool_ == nullptr) {
        headerPool_ = (BlockHeader *)headerAllocator_.Allocate(sizeof(BlockHeader)).ptr;
        headerPool_->listNext = nullptr; // in-pool header only listNext is meaningful
    }

    BlockHeader *header = headerPool_;
    headerPool_ = headerPool_->listNext;

    ::npu::tile_fwk::dynamic::Init(header);
    header->ptr = ptr;
    header->size = size;
    return header;
}

void WsAllocator::DeleteHeader(BlockHeader *header) {
    // Must be removed from list first

    // single directional linked-list
    header->listNext = headerPool_;
    headerPool_ = header;
}

std::pair<BlockHeader *, bool> WsAllocator::TryMergeBlockWithPrev(BlockHeader *header) {
    BlockHeader *prev = header->listPrev;
    if (!prev || prev->isBusy) {
        return std::make_pair(nullptr, false);
    }

    RemoveFromFreeList(prev);

    prev->size += header->size;

    RemoveFromList(header);
    DeleteHeader(header);

    return std::make_pair(prev, true);
}

bool WsAllocator::TryMergeBlockWithNext(BlockHeader *header) {
    BlockHeader *next = header->listNext;
    if (!next || next->isBusy) {
        return false;
    }

    RemoveFromFreeList(next);

    header->size += next->size;

    RemoveFromList(next);
    DeleteHeader(next);

    return true;
}

bool WsAllocator::TrySplitIntoTwo(BlockHeader *header, uint64_t firstBlockSize) {
    if (header->size <= firstBlockSize + MIN_BLOCK_SIZE) {
        // We don't keep too small block in free list
        return false;
    }

    uint64_t secondBlockSize = header->size - firstBlockSize;

    BlockHeader *next = CreateHeader(header->ptr + firstBlockSize, secondBlockSize);
    header->size = firstBlockSize;

    InsertIntoList(next, header);
    InsertIntoFreeList(next);

    return true;
}

void WsAllocator::InsertIntoList(BlockHeader *header, BlockHeader *prev) {
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

void WsAllocator::RemoveFromList(BlockHeader *header) {
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

void WsAllocator::InsertIntoFreeList(BlockHeader *header) {
    header->freeListNext = freeListHead_;
    if (freeListHead_) {
        freeListHead_->freeListPrev = header;
    }
    freeListHead_ = header;
}

void WsAllocator::RemoveFromFreeList(BlockHeader *header) {
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
} // namespace dynamic
} // namespace npu::tile_fwk
