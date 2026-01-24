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
 * \file memory_pool.h
 * \brief
 */

#pragma once

#include <unordered_map>
#include <unordered_set>
#include <list>
#include <mutex>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include "interface/utils/log.h"


namespace npu::tile_fwk {
inline constexpr int RTMALLOC_SUCCESS = 0;
inline constexpr uint32_t ONG_GB_HUGE_PAGE_FLAGS = RT_MEMORY_HBM | RT_MEMORY_POLICY_HUGE1G_PAGE_ONLY;
inline constexpr size_t ONT_GB_SIZE = 1024 * 1024 * 1024;
inline constexpr uint32_t TWO_MB_HUGE_PAGE_FLAGS = RT_MEMORY_HBM | RT_MEMORY_POLICY_HUGE_PAGE_FIRST;

inline uint64_t MemSizeAlign(const uint64_t bytes, const uint32_t aligns = 512U) {
    const uint64_t alignSize = (aligns == 0U) ? sizeof(uintptr_t) : aligns;
    return (((bytes + alignSize) - 1U) / alignSize) * alignSize;
}

inline constexpr size_t MIN_FREE_NODE_SIZE = 4 * 1024;;

// 内存块描述结构
struct MemoryBlock {
    void* base_addr;   // 块的基地址
    size_t block_size; // 块大小
    size_t used_size;  // 已使用大小
    bool is_huge_1g;   // 是否为1GB大页
    bool is_allocated; // 是否已分配
    
    // Only 1G page has FreeNode list
    struct FreeNode {
        FreeNode* next; // 指向下一个空闲节点
        size_t size;    // 空闲区域大小
    }* free_list;       // 空闲列表头
    
    MemoryBlock(void* addr, size_t size, bool is_huge_1g) 
        : base_addr(addr), block_size(size), used_size(0), 
          is_huge_1g(is_huge_1g), is_allocated(false), free_list(nullptr) {
            Init();
          }
    
    //初始化
    void Init() {
        // 1G page has FreeNode list
        free_list = reinterpret_cast<FreeNode*>(base_addr);
        free_list->next = nullptr;
        free_list->size = block_size;
    }

    //block块内分配内存
    void* Allocate(uint64_t alignSize) {
        // 兜底：最小分配粒度不能小于 FreeNode，否则还回来时写不下
        if (alignSize < sizeof(FreeNode)) {
            alignSize = sizeof(FreeNode);
        }

        FreeNode* prev = nullptr;
        FreeNode* curr = free_list;

        while (curr != nullptr) {
            if (curr->size >= alignSize) {
                void* use_ptr = curr;
                size_t remaining = curr->size - alignSize;

                //切分
                if (remaining >= sizeof(FreeNode)) {
                    auto new_node = reinterpret_cast<FreeNode*>(static_cast<char*>(use_ptr) + alignSize);
                    new_node->size = remaining;
                    new_node->next = curr->next;
                    
                    if (prev == nullptr) {
                        free_list = new_node;
                    } else {
                        prev->next = new_node;
                    }
                } else {
                    // 剩余太小，全给用户，摘除节点
                    if (prev == nullptr) {
                        free_list = curr->next;
                    } else {
                        prev->next = curr->next;
                    }
                    // 修正实际分配大小
                    alignSize = curr->size;
                }
                used_size += alignSize;
                return use_ptr;
            }
            prev = curr;
            curr = curr->next;
        }
        return nullptr;
    }

    // free
    void Free(void* ptr, size_t size) {
        auto new_node = reinterpret_cast<FreeNode*>(ptr);
        new_node->size = size;

        //插入链表
        FreeNode* prev = nullptr;
        FreeNode* curr = free_list;
        while (curr != nullptr && curr < new_node) {
            prev = curr;
            curr = curr->next;
        }
        
        new_node->next = curr;
        if (prev == nullptr) {
            free_list = new_node;
        } else {
            prev->next = new_node;
        }
        
        used_size -= size;
        Merge();
    }

    //block 内部合并
    void Merge() {
        FreeNode* curr = free_list;
        while (curr && curr->next) {
            char* curr_end = reinterpret_cast<char*>(curr) + curr->size;
            if (curr_end == reinterpret_cast<char*>(curr->next)) {
                curr->size += curr->next->size;
                curr->next = curr->next->next;
            } else {
                curr = curr->next;
            }
        }
    }
    
};

class DevMemoryPool {
public:
    DevMemoryPool();
    ~DevMemoryPool();

    bool AllocDevAddr(uint8_t **devAddr, uint64_t size) {
        if (devAddr == nullptr || size == 0) {
            return false;
        }
        *devAddr = nullptr;
        auto alignSize = MemSizeAlign(size);
        ALOG_INFO_F("MemoryPool::Allocate size[%lu] with align size[%lu].", size, alignSize);
        std::lock_guard<std::mutex> lock(mutex_);

        // 1. 尝试从现有内存块分配
        for (auto& block : memoryBlocks_) {
            void* ptr = block->Allocate(alignSize);
            if (ptr) {
                *devAddr = static_cast<uint8_t*>(ptr);
                RecordAllocation(ptr, block.get(), alignSize);
                return true;
            }
        }
        
        // 2. 现有不够用，尝试申请新的大块
        MemoryBlock* newBlock = CreateNewBlock(alignSize);
        if (newBlock) {
            void* ptr = newBlock->Allocate(alignSize);
            if (ptr) {
                *devAddr = static_cast<uint8_t*>(ptr);
                RecordAllocation(ptr, newBlock, alignSize);
                return true;
            }
        }
        
        ALOG_ERROR_F("Allocate failed size %lu", size);
        return false;
    }
    
    
    void FreeDevAddr(void* ptr) {
        if (!ptr) return;
        std::lock_guard<std::mutex> lock(mutex_);

        auto it = addrToBlock_.find(ptr);
        if (it == addrToBlock_.end()) {
            ALOG_ERROR_F("Freeing unknown pointer: %p", ptr);
            return;
        }

        MemoryBlock* block = it->second;
        size_t size = allocSizes_[ptr];

        block->Free(ptr, size);
        addrToBlock_.erase(it);
        allocSizes_.erase(ptr);
        ALOG_INFO_F("Freed ptr %p back to block %p", ptr, block->base_addr);
    }
    // 获取内存池状态（调试用）
    
    void PrintPoolStatus() {
        //...
    }
    
    void FreeMemBlock(MemoryBlock* block) {
        if (block && block->base_addr) {
            ALOG_INFO_F("FreeMemBlock %p with size %lu", block->base_addr, block->block_size);
            rtFree(block->base_addr, block->block_size);
            block->base_addr = nullptr;
        }
    }


    // 动态回收：只管大块能不能退，不管块内细节
    void DynamicRecycle() {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = memoryBlocks_.begin();
        while (it != memoryBlocks_.end()) {
            if ((*it)->used_size == 0) {
                ALOG_INFO_F("Recycling block %p", (*it)->base_addr);
                rtFree((*it)->base_addr, (*it)->block_size);
                it = memoryBlocks_.erase(it);
            } else {
                ++it;
            }
        }
    }

    // 销毁内存池（程序退出时调用，回收所有内存）
    void DestroyPool() {
        std::lock_guard<std::mutex> lock(mutex_);

        // 释放所有已使用的内存
        for (auto& block : memoryBlocks_) {
            if (block->base_addr) rtFree(block->base_addr, block->block_size);
        }
        
        memoryBlocks_.clear();
        addrToBlock_.clear();
        allocSizes_.clear();
        ALOG_INFO_F("MemPool destroyed, all memory freed\n");
    }
 private:
    void RecordAllocation(void* ptr, MemoryBlock* block, size_t size) {
        addrToBlock_[ptr] = block;
        allocSizes_[ptr] = size;
    }

    MemoryBlock* CreateNewBlock(uint64_t alignSize) {
        uint8_t *devAddr = nullptr;
        //优先1g
        size_t size1G = ((alignSize - 1) / ONT_GB_SIZE + 1) * ONT_GB_SIZE;
        if (rtMalloc((void**)&devAddr, size1G, ONG_GB_HUGE_PAGE_FLAGS, 0) == RTMALLOC_SUCCESS) {
            auto block = std::make_unique<MemoryBlock>(devAddr, size1G, true);
            MemoryBlock* ptr = block.get();
            memoryBlocks_.push_back(std::move(block));
            return ptr;
        }

        //降级2m
        if (rtMalloc((void**)&devAddr, alignSize, TWO_MB_HUGE_PAGE_FLAGS, 0) == RT_ERROR_NONE) {
            auto block = std::make_unique<MemoryBlock>(devAddr, alignSize, false);
            MemoryBlock* ptr = block.get();
            memoryBlocks_.push_back(std::move(block));
            return ptr;
        }

        ALOG_ERROR_F("All memory alloc strategies failed");
        return nullptr;
    }
 private:
    mutable std::mutex mutex_;
    // 内存块列表
    std::vector<std::unique_ptr<MemoryBlock>> memoryBlocks_;
    // 内存块查找表（用于快速定位内存所属的块）
    std::unordered_map<void*, MemoryBlock*> addrToBlock_;
    std::unordered_map<void*, size_t> allocSizes_;
};
} // namespace npu::tile_fwk