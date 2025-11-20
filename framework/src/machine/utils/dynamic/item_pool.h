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
 * \file item_pool.h
 * \brief
 */

#pragma once

#include "machine/utils/dynamic/allocator/allocators.h"

namespace npu::tile_fwk::dynamic {

template <typename T, WsMemCategory category = WsMemCategory::UNCLASSIFIED_ITEMPOOL,
    typename WsAllocator_T = WsMetadataAllocator>
class ItemPool {
public:
    struct ItemBlock {
        char buf[sizeof(T)];
        bool isFree{false};
        ItemBlock *freeListNext{nullptr};
    };

public:
    ItemPool() = default;
    ItemPool(WsAllocator_T &allocator, size_t count) {
        Init(allocator, count);
    }

    ~ItemPool() {
        if (allocation_) {
            // Call destructor on alive items
            ItemBlock *arr = allocation_.As<ItemBlock>();
            for (size_t i = 0; i < count_; i++) {
                if (!arr[i].isFree) {
                    ((T *)(arr + i))->~T();
                }
            }

            DEV_ASSERT(allocator_);
            allocator_->Deallocate(allocation_);
        }
    }

    void Init(WsAllocator_T &allocator, size_t count) {
        DEV_ASSERT(!allocator_);
        allocator_ = &allocator;
        count_ = count;
        allocation_ = allocator_->template Allocate<ItemBlock>(count_, category);
        ItemBlock *arr = allocation_.As<ItemBlock>();
        for (size_t i = 0; i < count_; i++) {
            InsertFreeList(arr + i);
        }
    }

    template <typename ...Args>
    T *Make(Args &&...args) {
        DEV_ASSERT(freeListHead_ != nullptr);
        freeListHead_->isFree = false;
        T *newItem = (T *)freeListHead_->buf;
        freeListHead_ = freeListHead_->freeListNext;
        new(newItem) T(std::forward<Args>(args)...);
        return newItem;
    }

    void Destroy(T *item) {
        item->~T();
        ItemBlock *block = (ItemBlock *)item;
        block->isFree = true;
        InsertFreeList(block);
    }

private:
    inline void InsertFreeList(ItemBlock *block) {
        block->freeListNext = freeListHead_;
        freeListHead_ = block;
    }

private:
    WsAllocator_T *allocator_{nullptr};
    WsAllocation allocation_;
    size_t count_;
    ItemBlock *freeListHead_{nullptr};
};

} // namespace npu::tile_fwk::dynamic