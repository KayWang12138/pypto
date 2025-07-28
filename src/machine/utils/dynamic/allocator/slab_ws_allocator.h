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
 * \file seq_ws_allocator.h
 * \brief
 */

#pragma once
#include <cstddef>
#include <cstdint>
#include <algorithm>

#include "machine/utils/device_log.h"
#include "ws_allocator_basics.h"

namespace npu::tile_fwk::dynamic {

enum class WsAicpuSlabMemType : uint8_t {
    DUPPED_FUNC_DATA = 0,
    DYN_FUNC_DATA,
    VEC_STITCHED_LIST,
    DEV_DYN_TASK,
    READY_QUE,
    COHERENT_SLAB_MEM_TYPE_BUTT,

    DUPPED_STITCH, // stitch pool memory
    SLAB_MEM_TYPE_BUTT
};
constexpr int SLAB_ALLOCATOR_MAX_CACHES = 16;

struct StageAllocInfo {
    void* heads[SLAB_ALLOCATOR_MAX_CACHES];
    void* tails[SLAB_ALLOCATOR_MAX_CACHES];
};

class SlabWsAllocator {
private:
    struct SlabHeader;
    struct SlabCache {
        uint32_t objSize{0};
        void* freeList{nullptr};
        SlabHeader* activeSlab{nullptr};
        SlabWsAllocator* allocator{nullptr};

        // Per-cache allocation tracking
        void* stageAllocHead{nullptr};
        void* stageAllocTail{nullptr};

        SlabCache() {}
        SlabCache(uint32_t size, SlabWsAllocator* alloc) 
            : objSize(size), freeList(nullptr), activeSlab(nullptr), 
              allocator(alloc), stageAllocHead(nullptr), stageAllocTail(nullptr) {}
    };

    struct SlabHeader {
        SlabCache* cache; // extend
        uint16_t allocatedCount;
        uint16_t totalCount;
    };
public:
    SlabWsAllocator() = default;
    
    void Init(void* baseAddr, uint32_t totalSize, uint32_t alignSize) {
        memBaseaddr_ = static_cast<uint8_t*>(baseAddr);
        totalMemSize_ = totalSize;
        slabAlignSize_ = (((alignSize) + (sizeof(uint64_t)) - 1) & ~((sizeof(uint64_t)) - 1));
        nextFreeSlabAddr_ = memBaseaddr_;
        freeSlabList_ = nullptr;
        numCaches_ = 0;

        DEV_DEBUG("[SlabWsAllocator]Init SlabWsAllocator: base=%p, size=%u, align=%u\n",
                  memBaseaddr_, totalMemSize_, slabAlignSize_);
    }

    bool RegistCache(uint32_t type, uint32_t objSize) {
        if (type >= SLAB_ALLOCATOR_MAX_CACHES || objSize == 0) {
            return false;
        }
        
        if (caches_[type].objSize != 0) {
            if (caches_[type].objSize >= objSize) {
                DEV_DEBUG("[SlabWsAllocator]Slab cache exists : objsize = %u, cacheType = %u .\n", objSize, type);
                return true;
            }
            DEV_ERROR("[SlabWsAllocator]Add cache failed type = %u, objsize = %u", type, objSize);
            return false;
        }
        uint32_t realObjSize = (((objSize) + (sizeof(uint64_t)) - 1) & ~((sizeof(uint64_t)) - 1));
        caches_[type] = SlabCache(realObjSize, this);
        numCaches_++;
        DEV_DEBUG("[SlabWsAllocator]Add slab cache : objsize = %u, realobjsize = %u, type = %u .\n", objSize, realObjSize, type);
        return true;
    }

    bool ExistCache(uint32_t cacheType, uint32_t objSize) {
        if (cacheType >= SLAB_ALLOCATOR_MAX_CACHES) {
            return false;
        }
        if (caches_[cacheType].objSize >= objSize) {
            return true;
        }
        return false;
    }

    void* Alloc(uint32_t cacheType) {
        if (cacheType >= SLAB_ALLOCATOR_MAX_CACHES) {
            DEV_DEBUG_ASSERT_MSG(false, "[SlabWsAllocator]cache type invalid %u.\n", cacheType);
            return nullptr;
        }
        
        SlabCache& cache = caches_[cacheType];
        uint32_t objSize = cache.objSize;
        if (objSize == 0) {
            DEV_DEBUG_ASSERT_MSG(false, "[SlabWsAllocator]cache type not regist %u.\n", cacheType);
            return nullptr;
        }

        void* obj = nullptr;
        if (cache.freeList) {
            obj = cache.freeList;
            cache.freeList = *static_cast<void**>(obj);
            DEV_DEBUG("[SlabWsAllocator]Alloc from slab free list: objsize = %u.\n", objSize);
        } else if (cache.activeSlab && cache.activeSlab->allocatedCount < cache.activeSlab->totalCount) {
            SlabHeader* slab = cache.activeSlab;
            obj = static_cast<uint8_t*>(static_cast<void*>(slab)) + 
                   sizeof(SlabHeader) + slab->allocatedCount * (sizeof(void*) + objSize);
            slab->allocatedCount++;
            DEV_DEBUG("[SlabWsAllocator]Alloc from active slab: slab = %p, objsize = %u, allocCnt=%u.\n",
                slab, objSize, slab->allocatedCount);
        } else {
            void* slabMem = get_free_slab();
            if (!slabMem) {
                DEV_DEBUG_ASSERT_MSG(false, "[SlabWsAllocator]Alloc memory not enough : objsize = %u.\n", objSize);
                return nullptr; // memory not enough
            }

            SlabHeader* header = new (slabMem) SlabHeader();
            header->cache = &cache;
            header->allocatedCount = 1;
            header->totalCount = (slabAlignSize_ - sizeof(SlabHeader)) / (sizeof(void*) + objSize);
            cache.activeSlab = header;
            obj = static_cast<uint8_t*>(slabMem) + sizeof(SlabHeader);
            DEV_DEBUG("[SlabWsAllocator]Alloc from new slab: slab = %p, objsize = %u, totalCnt=%u.\n",
                header, objSize, header->totalCount);
        }

        *static_cast<void**>(obj) = nullptr;
        
        if (!cache.stageAllocHead) {
            cache.stageAllocHead = obj;
        } else {
            *static_cast<void**>(cache.stageAllocTail) = obj;
        }
        cache.stageAllocTail = obj;
        DEV_DEBUG("[SlabWsAllocator]Alloc sucess obj = %p cacheType = %u size = %u.\n", obj, cacheType, objSize);
        return static_cast<uint8_t*>(obj) + sizeof(void*);
    }

    StageAllocInfo PopStageAllocMem() {
        StageAllocInfo info;
        
        for (int i = 0; i < SLAB_ALLOCATOR_MAX_CACHES; i++) {
            info.heads[i] = caches_[i].stageAllocHead;
            info.tails[i] = caches_[i].stageAllocTail;
            
            // Reset cache tracking
            caches_[i].stageAllocHead = nullptr;
            caches_[i].stageAllocTail = nullptr;
        }
        
        return info;
    }

    void FreeStageAllocMem(const StageAllocInfo& info) {
        for (int i = 0; i < SLAB_ALLOCATOR_MAX_CACHES; i++) {
            if (!info.heads[i]) continue;
            SlabCache& cache = caches_[i];

#if defined(DEBUG_SWITCH) && DEBUG_SWITCH
            void* temp = info.heads[i];
            while (temp) {
                DEV_DEBUG("[SlabWsAllocator]recycle sucess obj = %p cacheType = %d size = %u.\n",
                    temp, i, cache.objSize);
                temp = *static_cast<void**>(temp);
            }
#endif
            if (cache.freeList) {
                *static_cast<void**>(info.tails[i]) = cache.freeList;
                cache.freeList = info.heads[i];
            } else {
                cache.freeList = info.heads[i];
            }
        }
    }

    void DumpMemoryUsage(const char *hint, const char *title) const {
#if DEBUG_MEM_DUMP_LEVEL >= DEBUG_MEM_DUMP_LIGHT
     // todo
#endif // DEBUG_MEM_DUMP_LEVEL >= DEBUG_MEM_DUMP_LIGHT
        (void)title;
        (void)hint;
    }

private:
    void* get_free_slab() {
        if (freeSlabList_) {
            void* slab = freeSlabList_;
            freeSlabList_ = *static_cast<void**>(freeSlabList_);
            return slab;
        }

        if (nextFreeSlabAddr_ + slabAlignSize_ <= memBaseaddr_ + totalMemSize_) {
            void* slab = nextFreeSlabAddr_;
            nextFreeSlabAddr_ += slabAlignSize_;
            return slab;
        }

        return nullptr;
    }

private:
    uint8_t* memBaseaddr_{nullptr};
    uint8_t* nextFreeSlabAddr_{nullptr};
    uint32_t totalMemSize_{0};
    uint32_t slabAlignSize_{0};
    void* freeSlabList_{nullptr};

    SlabCache caches_[SLAB_ALLOCATOR_MAX_CACHES];
    int numCaches_{0};
};
}