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
          is_huge_1g(is_huge_1g), is_allocated(false), free_list(nullptr) {}
};

class DevMemoryPool {
public:
    DevMemoryPool();
    ~DevMemoryPool();

    bool AllocDevAddr(uint8_t **devAddr, uint64_t size) {
        if (devAddr == nullptr) {
            return false;
        }
        *devAddr = nullptr;
        if (size == 0) {
            return false;
        }
        auto alignSize = MemSizeAlign(size);
        ALOG_INFO_F("MemoryPool::Allocate size[%lu] with align size[%lu].", size, alignSize);
        std::lock_guard<std::mutex> lock(mutex_);
        // 1. 尝试从现有内存块分配
        void* ptr = TryAllocateFromExistingBlocks(alignSize);
        if (ptr != nullptr) {
            return ptr;
        }
        
        // 2. 尝试分配1GB大页
        MemoryBlock* block = Allocate1GBBlock(alignSize);
        if (block != nullptr) {
            ptr = AllocateFromBlock(block, alignSize);
            if (ptr != nullptr) {
                return ptr;
            }
        }
        
        // 3. 尝试分配2MB大页
        block = Allocate2MBBlock(alignSize);
        if (block != nullptr) {
            ptr = AllocateFromBlock(block, alignSize);
            if (ptr != nullptr) {
                return ptr;
            }
        }
        
        ALOG_ERROR_F("MemoryPool::Allocate failed for size %lu", size);
        return nullptr;
    }
    
    void FreeDevAddr(void* ptr) {

    }
       // 获取内存池状态（调试用）
    void PrintPoolStatus() {
        std::lock_guard<std::mutex> lock(poolMutex);
        uint64_t freeTotal = 0;
        uint64_t usedTotal = 0;

        for (const auto& pair : freeBlocks) {
            for (const auto& block : pair.second) {
                freeTotal += block.size;
            }
        }

        for (const auto& pair : usedBlocks) {
            usedTotal += pair.second.size;
        }

        ALOG_INFO_F("MemPool status: free %lu bytes, used %lu bytes\n", freeTotal, usedTotal);
    }

    // 销毁内存池（程序退出时调用，回收所有内存）
    void DestroyPool() {
        std::lock_guard<std::mutex> lock(poolMutex);

        // 释放所有已使用的内存
        for (const auto& pair : usedBlocks) {
            FreeMemBlock(pair.second);
        }
        usedBlocks.clear();

        // 释放所有空闲的内存
        for (const auto& pair : freeBlocks) {
            for (const auto& block : pair.second) {
                FreeMemBlock(block);
            }
        }
        freeBlocks.clear();

        ALOG_INFO_F("MemPool destroyed, all memory freed\n");
    }
 private:
    void* AllocateFromBlock(MemoryBlock* block, uint64_t alignSize) {
        typename MemoryBlock::FreeNode* prev = nullptr;
        typename MemoryBlock::FreeNode* curr = block->free_list;
        
        // 查找足够大的空闲块
        while (curr != nullptr) {
            if (curr->size >= alignSize) {
                // 找到足够大的块
                void* ptr = curr + 1; // 跳过FreeNode头部
                
                // 如果剩余空间足够大，分割成两个块
                size_t remaining_size = curr->size - align_size;
                if (remaining_size >= MIN_FREE_NODE_SIZE) {
                    // 分割块
                    typename MemoryBlock::FreeNode* new_node = reinterpret_cast<typename MemoryBlock::FreeNode*>(
                        static_cast<char*>(ptr) + align_size);
                    new_node->size = remaining_size - sizeof(typename MemoryBlock::FreeNode);
                    new_node->next = curr->next;
                    
                    if (prev == nullptr) {
                        block->free_list = new_node;
                    } else {
                        prev->next = new_node;
                    }
                } else {
                    // 整个块都分配出去
                    if (prev == nullptr) {
                        block->free_list = curr->next;
                    } else {
                        prev->next = curr->next;
                    }
                }
                
                // 更新块的使用情况
                block->used_size += align_size;
                block->is_allocated = true;
                
                // 记录地址到块的映射
                addr_to_block_[ptr] = block;
                
                ALOG_INFO_F("Allocate from block %p, ptr %p, size %lu", block->base_addr, ptr, align_size);
                return ptr;
            }
            
            prev = curr;
            curr = curr->next;
        }
        
        return nullptr; // 没有找到足够大的块
    }

    void* TryAllocateFromExistingBlocks(uint64_t alignSize) {
        // 遍历所有内存块，尝试分配
        for (auto blockPtr : memoryBlocks_) {
            void* ptr = AllocateFromBlock(blockPtr.get(), alignSize);
            if (ptr != nullptr) {
                return ptr;
            }
        }
        return nullptr;
    }

            size_t allocSize = ((alignSize - 1) / ONT_GB_SIZE + 1) * ONT_GB_SIZE;
        int res = rtMalloc((void **)devAddr, allocSize, ONG_GB_HUGE_PAGE_FLAGS, 0);
        if (res != 0) {
            ALOG_WARN_F("1G page mem alloc failed, turn to 2M page.\n");
            res = rtMalloc((void **)devAddr, alignSize, TWO_MB_HUGE_PAGE_FLAGS, 0);
            if (res != 0) {
                ALOG_ERROR_F("RuntimeAgent::AllocDevAddr failed for size %lu", size);
                return;
            }
            allocatedDevAddr.emplace_back(*devAddr);
            ALOG_INFO_F("AllocDevAddr %p size is %lu", *devAddr, size);
            return;
        }
        allocatedDevAddr.emplace_back(*devAddr);
        hugePageVec.emplace_back(HugePageDesc(*devAddr, allocSize));
        if (!TryGetHugePageMem(devAddr, alignSize)) {
            ALOG_ERROR_F("RuntimeAgent::AllocDevAddr failed for size %lu", size);
            return;
        }

    MemoryBlock* AllocateNewBlock(uint64_t alignSize) {
        uint8_t *devAddr = nullptr;
        // 1. Try alloc 1G page memory
        size_t allocSize = ((alignSize - 1) / ONT_GB_SIZE + 1) * ONT_GB_SIZE;
        int res = rtMalloc((void **)&devAddr, allocSize, ONG_GB_HUGE_PAGE_FLAGS, 0);
        if (res == 0) {
            auto 1gMemBlk = std::make_unique<MemoryBlock>(devAddr, allocSize, true));
            InitFreeList(block);
            memoryBlocks_.push_back(1gMemBlk);
            return 1gMemBlk.get();
        }
        
        ALOG_WARN_F("1G page mem alloc failed, turn to 2M page.");
        // 尝试使用rtMalloc分配2MB大页
        int res = rtMalloc((void**)&devAddr, alignSize, TWO_MB_HUGE_PAGE_FLAGS, 0);
        if (res == 0) {
            auto 2mMemBlk = std::make_unique<MemoryBlock>(devAddr, alignSize, false));
            InitFreeList(block);
            memoryBlocks_.push_back(2mMemBlk);
            return 2mMemBlk.get();
        }
        
        ALOG_ERROR_F("2M page mem alloc failed");
        return nullptr;
    }

 private:
    mutable std::mutex mutex_;
    // 内存块列表
    std::vector<std::unique_ptr<MemoryBlock>> memoryBlocks_;
    // 内存块查找表（用于快速定位内存所属的块）
    std::unordered_map<void*, MemoryBlock*> addrToBlock_;
}