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

#ifdef BUILD_WITH_CANN
#include "acl/acl.h"
#include "runtime/rt.h"
#include "runtime/rt_preload_task.h"
#endif

namespace npu::tile_fwk {
#ifdef BUILD_WITH_CANN
inline constexpr int RTMALLOC_SUCCESS = 0;
inline constexpr uint32_t ONG_GB_HUGE_PAGE_FLAGS = RT_MEMORY_HBM | RT_MEMORY_POLICY_HUGE1G_PAGE_ONLY;
inline constexpr size_t ONT_GB_SIZE = 1024 * 1024 * 1024;
inline constexpr uint32_t TWO_MB_HUGE_PAGE_FLAGS = RT_MEMORY_HBM | RT_MEMORY_POLICY_HUGE_PAGE_FIRST;

inline uint64_t MemSizeAlign(const uint64_t bytes, const uint32_t aligns = 512U) {
    const uint64_t alignSize = (aligns == 0U) ? sizeof(uintptr_t) : aligns;
    return (((bytes + alignSize) - 1U) / alignSize) * alignSize;
}

inline constexpr size_t MIN_FREE_NODE_SIZE = 4 * 1024;

struct MemoryBlock {
    void* base_addr;
    size_t block_size;
    size_t used_size;
    bool is_huge_1g;
    
    // Only 1G page has FreeNode list
    struct FreeNode {
        FreeNode* next;
        size_t size;
    }* free_list;
    
    MemoryBlock(void* addr, size_t size, bool in_is_huge_1g) 
        : base_addr(addr), block_size(size), used_size(0), 
          is_huge_1g(in_is_huge_1g), free_list(nullptr) {
            Init();
          }
    
    void Init() {
        if (is_huge_1g) {
            free_list = reinterpret_cast<FreeNode*>(base_addr);
            free_list->next = nullptr;
            free_list->size = block_size;
        } else {
            free_list = nullptr;
        }
    }

    void* Allocate(uint64_t alignSize) {
        if (!is_huge_1g) {
            if (used_size == 0 && block_size >= alignSize) {
                used_size = block_size;
                return base_addr;
            }
            return nullptr;
        }

        if (alignSize < sizeof(FreeNode)) {
            alignSize = sizeof(FreeNode);
        }

        FreeNode* prev = nullptr;
        FreeNode* curr = free_list;

        while (curr != nullptr) {
            if (curr->size >= alignSize) {
                void* use_ptr = curr;
                size_t remaining = curr->size - alignSize;

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
                    if (prev == nullptr) {
                        free_list = curr->next;
                    } else {
                        prev->next = curr->next;
                    }
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

    void Free(void* ptr, size_t size) {
        if (!is_huge_1g) {
            ALOG_ERROR_F("Logic Error: 2MB block should not call Free()");
            return;
        }

        auto new_node = reinterpret_cast<FreeNode*>(ptr);
        new_node->size = size;

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

    void Merge() {
        if (!is_huge_1g) {
            return;
        }

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
    DevMemoryPool() {}
    ~DevMemoryPool() { DestroyPool(); }

    bool AllocDevAddr(uint8_t **devAddr, uint64_t size) {
        if (devAddr == nullptr || size == 0) {
            return false;
        }
        *devAddr = nullptr;
        auto alignSize = MemSizeAlign(size);
        ALOG_INFO_F("MemoryPool::Allocate size[%lu] with align size[%lu].", size, alignSize);
        std::lock_guard<std::mutex> lock(mutex_);

        for (auto& block : memoryBlocks_) {
            void* ptr = block->Allocate(alignSize);
            if (ptr) {
                *devAddr = static_cast<uint8_t*>(ptr);
                RecordAllocation(ptr, block.get(), alignSize);
                return true;
            }
        }
        
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

        if (block->is_huge_1g) {
            block->Free(ptr, size);
            ALOG_INFO_F("Recycled ptr %p to 1GB block pool", ptr);
        } else {
            ALOG_INFO_F("Direzctly freeing 2MB block addr %p", block->base_addr);

            if (block->base_addr) {
                FreeMemBlock(block);
            }

            for (auto vec_it = memoryBlocks_.begin(); vec_it != memoryBlocks_.end(); ++vec_it) {
                if (vec_it->get() == block) {
                    memoryBlocks_.erase(vec_it);
                    break; 
                }
            }
        }

        addrToBlock_.erase(it);
        allocSizes_.erase(ptr);
    }
        
    void DynamicRecycle() {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = memoryBlocks_.begin();
        while (it != memoryBlocks_.end()) {
            if ((*it)->used_size == 0) {
                ALOG_INFO_F("Recycling block %p", (*it)->base_addr);
                FreeMemBlock(it->get());
                it = memoryBlocks_.erase(it);
            } else {
                ++it;
            }
        }
    }

    void DestroyPool() {
        std::lock_guard<std::mutex> lock(mutex_);

        for (auto& block : memoryBlocks_) {
            if (block->base_addr) {
                FreeMemBlock(block.get());
            }
        }

        memoryBlocks_.clear();
        addrToBlock_.clear();
        allocSizes_.clear();
        ALOG_INFO_F("MemPool destroyed, all memory freed\n");
    }

    void PrintPoolStatus() {
        std::lock_guard<std::mutex> lock(mutex_);
        
        size_t total_1g_count = 0;
        size_t total_2m_count = 0;
        size_t total_mem_bytes = 0;
        size_t total_used_bytes = 0;
        ALOG_INFO_F("========== [Memory Pool Status] ==========");
        
        for (size_t i = 0; i < memoryBlocks_.size(); ++i) {
            MemoryBlock* block = memoryBlocks_[i].get();
            
            if (block->is_huge_1g) {
                total_1g_count++;
            } else {
                total_2m_count++;
            }
            
            total_mem_bytes += block->block_size;
            total_used_bytes += block->used_size;
            
            double usage_rate = 0.0;
            if (block->block_size > 0) {
                usage_rate = (static_cast<double>(block->used_size) / block->block_size) * 100.0;
            }

            ALOG_INFO_F("Block[%lu] %s | Addr: %p | Size: %lu MB | Used: %lu MB (%.2f%%)", 
                        i,
                        block->is_huge_1g ? "[1GB POOL]" : "[2MB PAGE]",
                        block->base_addr,
                        block->block_size / 1024 / 1024,
                        block->used_size / 1024 / 1024,
                        usage_rate);
        }

        size_t total_free_bytes = total_mem_bytes - total_used_bytes;
        
        ALOG_INFO_F("---------------- Summary -----------------");
        ALOG_INFO_F("Block Counts : 1GB Huge x %lu, 2MB Page x %lu", total_1g_count, total_2m_count);
        ALOG_INFO_F("Total Memory : %lu MB", total_mem_bytes / 1024 / 1024);
        ALOG_INFO_F("Total Used   : %lu MB", total_used_bytes / 1024 / 1024);
        ALOG_INFO_F("Total Free   : %lu MB", total_free_bytes / 1024 / 1024);
        ALOG_INFO_F("==========================================");
    }

 private:
    void FreeMemBlock(MemoryBlock* block) {
        if (block && block->base_addr) {
            ALOG_INFO_F("Releasing physical memory: %p (size %lu)", block->base_addr, block->block_size);
            rtFree(block->base_addr);
            block->base_addr = nullptr;
        }
    }

    void RecordAllocation(void* ptr, MemoryBlock* block, size_t size) {
        addrToBlock_[ptr] = block;
        allocSizes_[ptr] = size;
    }

    MemoryBlock* CreateNewBlock(uint64_t alignSize) {
        uint8_t *devAddr = nullptr;
        size_t size1G = ((alignSize - 1) / ONT_GB_SIZE + 1) * ONT_GB_SIZE;
        
        if (rtMalloc((void**)&devAddr, size1G, ONG_GB_HUGE_PAGE_FLAGS, 0) == RTMALLOC_SUCCESS) {
            auto block = std::make_unique<MemoryBlock>(devAddr, size1G, true);
            MemoryBlock* ptr = block.get();
            memoryBlocks_.push_back(std::move(block));
            return ptr;
        }

        if (rtMalloc((void**)&devAddr, alignSize, TWO_MB_HUGE_PAGE_FLAGS, 0) == RTMALLOC_SUCCESS) {
            auto block = std::make_unique<MemoryBlock>(devAddr, alignSize, false);
            MemoryBlock* ptr = block.get();
            memoryBlocks_.push_back(std::move(block));
            return ptr;
        }

        ALOG_ERROR_F("All memory alloc strategies failed");
        return nullptr;
    }

    mutable std::mutex mutex_;
    std::vector<std::unique_ptr<MemoryBlock>> memoryBlocks_;
    std::unordered_map<void*, MemoryBlock*> addrToBlock_;
    std::unordered_map<void*, size_t> allocSizes_;
};
#endif
} // namespace npu::tile_fwk