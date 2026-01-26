/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
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

#include <map>
#include <unordered_map>
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

struct MemoryBlock {
    void* base_addr;
    size_t block_size;
    size_t used_size;
    bool is_huge_1g;
    
    std::map<uintptr_t, size_t> free_map;
    
    MemoryBlock(void* addr, size_t size, bool is_huge) 
        : base_addr(addr), block_size(size), used_size(0), 
          is_huge_1g(is_huge) {
        Init();
    }
    
    void Init() {
        if (is_huge_1g) {
            free_map[reinterpret_cast<uintptr_t>(base_addr)] = block_size;
        } else {
            free_map.clear();
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

        for (auto it = free_map.begin(); it != free_map.end(); ++it) {
            uintptr_t chunk_addr = it->first;
            size_t chunk_size = it->second;

            if (chunk_size >= alignSize) {
                void* use_ptr = reinterpret_cast<void*>(chunk_addr);
                size_t remaining = chunk_size - alignSize;

                free_map.erase(it);

                if (remaining > 0) {
                    free_map[chunk_addr + alignSize] = remaining;
                }

                used_size += alignSize;
                return use_ptr;
            }
        }
        return nullptr;
    }

    void Free(void* ptr, size_t size) {
        if (!is_huge_1g) {
            ALOG_ERROR_F("Logic Error: 2MB block should not call Free()");
            return;
        }

        uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
        
        free_map[addr] = size;
        used_size -= size;

        auto it = free_map.find(addr);
        if (it == free_map.end()) return; // Should not happen

        auto next_it = std::next(it);
        if (next_it != free_map.end()) {
            if (it->first + it->second == next_it->first) {
                it->second += next_it->second;
                free_map.erase(next_it);
            }
        }

        if (it != free_map.begin()) {
            auto prev_it = std::prev(it);
            if (prev_it->first + prev_it->second == it->first) {
                prev_it->second += it->second;
                free_map.erase(it);
            }
        }
    }
};

class DevMemoryPool {
public:
    DevMemoryPool() {}
    ~DevMemoryPool() { DestroyPool(); }

    bool AllocDevAddr(uint8_t **devAddr, uint64_t size) {
        if (!devAddr || size == 0) return false;
        *devAddr = nullptr;
        
        auto alignSize = MemSizeAlign(size);
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
        } else {
            ALOG_INFO_F("Directly freeing 2MB block addr %p", block->base_addr);
            FreeMemBlock(block); 
            // 移除块逻辑
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
                ALOG_INFO_F("Recycling empty block %p", (*it)->base_addr);
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
            FreeMemBlock(block.get());
        }
        memoryBlocks_.clear();
        addrToBlock_.clear();
        allocSizes_.clear();
        ALOG_INFO_F("MemPool destroyed, all memory freed");
    }

    void PrintPoolStatus() {
        std::lock_guard<std::mutex> lock(mutex_);
        size_t cnt_1g = 0, cnt_2m = 0;
        size_t total = 0, used = 0;
        
        ALOG_INFO_F("========== [Memory Pool Status] ==========");
        for (size_t i = 0; i < memoryBlocks_.size(); ++i) {
            auto* blk = memoryBlocks_[i].get();
            if (blk->is_huge_1g) cnt_1g++; else cnt_2m++;
            total += blk->block_size;
            used += blk->used_size;
            
            double rate = blk->block_size ? (double)blk->used_size * 100.0 / blk->block_size : 0;
            ALOG_INFO_F("Block[%lu] %s | Addr: %p | Used: %.1f%% | Fragments: %lu", 
                i, blk->is_huge_1g ? "1G" : "2M", blk->base_addr, rate, blk->free_map.size());
        }
        ALOG_INFO_F("Summary: 1G x %lu, 2M x %lu | Used/Total: %lu/%lu MB", 
            cnt_1g, cnt_2m, used>>20, total>>20);
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