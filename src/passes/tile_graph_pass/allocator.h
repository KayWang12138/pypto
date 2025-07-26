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
 * \file allocator.h
 * \brief
 */

#ifndef TILE_FWK_ALLOCATOR_H
#define TILE_FWK_ALLOCATOR_H

#include <cstddef>
#include <set>
#include <string>
#include <vector>
#include <memory>
#include <algorithm>
#include <iostream>
#include "tilefwk/data_type.h"   // Assuming this is where MemoryType is defined
#include "interface/configs/config_storage.h" // Assuming this is where the configuration manager is defined
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/tensor/tensormap.h" // Assuming this is where LogicalTensor and tensormap are defined
#include "interface/tensor/logical_tensor.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h" // Include this if Function is defined in a separate file
#include "interface/program/program.h"
#include "passes/pass_config/pass_config_manager.h"


namespace npu::tile_fwk {

enum class TiFwkAllocatorStrategy { ALLOC_MINFIT, ALLOC_FAST };
enum class ModelID { ASCEND_910B };

// Standalone functions
bool IsOverlap(const TileRange &a, const TileRange &b);
bool LessThan(const TileRange &a, const TileRange &b);

// Define a comparator for std::set
struct TiFwkRangeComparator {
    bool operator()(const TileRange &a, const TileRange &b) const { return LessThan(a, b); }
};

class TiFwkAllocator {
public:
    std::string allocatorName;
    size_t totalSize;
    size_t totalInUse;
    std::set<TileRange, TiFwkRangeComparator> freepool;

    // Constructor
    TiFwkAllocator(const std::string &name, const TileRange &freerange)
        : allocatorName(name), totalSize(freerange.end - freerange.start + 1), totalInUse(0) {
        freepool.insert(freerange);
    }

    // Check if a block of the given size can be allocated
    bool CanAllocate(size_t blocksize) const {
        bool result = false;
        for (const auto &range : freepool) {
            if (range.Size() >= blocksize) {
                result = true;
                break;
            }
        }

        return result;
    }

    bool IsInUse() const { return !CanAllocate(totalSize); }
    // Allocate a range of the requested size using the specified strategy
    TileRange Allocate(size_t reqsize, TiFwkAllocatorStrategy strategy = TiFwkAllocatorStrategy::ALLOC_MINFIT) {
        auto bestFit = freepool.end();
        for (auto it = freepool.begin(); it != freepool.end(); ++it) {
            if (it->Size() >= reqsize) {
                if (strategy == TiFwkAllocatorStrategy::ALLOC_FAST) {
                    bestFit = it;
                    break;
                }
                if (bestFit == freepool.end() || it->Size() < bestFit->Size()) {
                    bestFit = it;
                }
            }
        }

        if (bestFit == freepool.end()) {
            return TileRange();
        }

        TileRange allocatedRange(bestFit->start, bestFit->start + reqsize);
        TileRange remainingRange(bestFit->start + reqsize, bestFit->end);

        freepool.erase(bestFit);
        if (!remainingRange.IsEmpty()) {
            freepool.insert(remainingRange);
        }

        totalInUse += reqsize;

        return allocatedRange;
    }

    // Free a range and merge with contiguous ranges in the freepool
    void Free(TileRange reqblock) {
        totalInUse -= reqblock.Size();

        auto it = freepool.lower_bound(reqblock);
        if (it != freepool.begin() && std::prev(it)->end == reqblock.start) {
            --it;
            reqblock.start = it->start;
            freepool.erase(it);
        }

        while (it != freepool.end() && it->start == reqblock.end) {
            reqblock.end = it->end;
            it = freepool.erase(it);
        }

        freepool.insert(reqblock);
    }

    // Additional member functions can be defined here
};

class TiFWKJointAllocator {
public:
    std::vector<TiFwkAllocator> allocators;
    bool isSplitVectorCube;
    // Constructor
    explicit TiFWKJointAllocator(ModelID modelId) { Reset(modelId); }
    void Reset(ModelID modelId) {
        for (size_t i = static_cast<size_t>(MemoryType::MEM_UB); i <= static_cast<size_t>(MemoryType::MEM_FAR2); ++i) {
            MemoryType memType = static_cast<MemoryType>(i);
            size_t sizeLimit = PassConfigManager::Instance().GetPlatformConfig().GetMemoryLimit(memType);
            TileRange initialRange(0, sizeLimit);
            allocators.emplace_back(MemoryTypeToString(memType), initialRange);
        }
        switch (modelId) {
            case ModelID::ASCEND_910B: isSplitVectorCube = true; break;
            default: isSplitVectorCube = true;
        }
    }

    // Check if all tensors in conflictermagic can be allocated
    bool CanAllocate(const std::vector<std::shared_ptr<LogicalTensor>> &toAlloc, int subGraphID,
        std::shared_ptr<LogicalTensor> &failedTensor);

    // Allocate a range for the given tensor using the specified strategy
    bool Allocate(bool isRealAlloc, std::shared_ptr<LogicalTensor> tensor, int subGraphID,
        TiFwkAllocatorStrategy strategy = TiFwkAllocatorStrategy::ALLOC_MINFIT) {
        MemoryType memType = tensor->GetMemoryTypeOriginal();
        size_t requiredSize = tensor->MemorySize();
        if (requiredSize == 0) { // dummy tensor
            tensor->memorymap.insert(std::make_pair(subGraphID, TileRange()));
            return true;
        }
        if (tensor->memorymap.find(subGraphID) != tensor->memorymap.end()) {
            return true; // no need to allocate again
        }
        ASSERT(memType != MemoryType::MEM_UNKNOWN);
        // Find the corresponding allocator for the memory type
        auto it = std::find_if(allocators.begin(), allocators.end(), [memType](const TiFwkAllocator &allocator) {
            return allocator.allocatorName == MemoryTypeToString(memType);
        });
        if (it != allocators.end()) {
            TileRange allocatedRange = it->Allocate(requiredSize, strategy);
            if (!allocatedRange.IsEmpty()) {
                if (isRealAlloc) {
                    tensor->memorymap.insert(std::make_pair(subGraphID, allocatedRange));
                }
                return true;
            } else {
                return false;
            }
        }
        return false; // Return false if no suitable allocator is found or allocation fails
    }

    // Free a range for the given tensor
    bool Free(bool removefrommemorymap, std::shared_ptr<LogicalTensor> tensor, int subGraphID) {
        if (tensor->memorymap.find(subGraphID) == tensor->memorymap.end()) {
            return true;
        }
        MemoryType memType = tensor->GetMemoryTypeOriginal();
        // Find the corresponding allocator for the memory type
        auto it = std::find_if(allocators.begin(), allocators.end(), [memType](const TiFwkAllocator &allocator) {
            return allocator.allocatorName == MemoryTypeToString(memType);
        });
        if (it != allocators.end()) {
            ASSERT(tensor->memorymap.find(subGraphID) != tensor->memorymap.end());
            it->Free(tensor->memorymap[subGraphID]);
            if (removefrommemorymap) {
                // if this was fake alloc, need to erase the item from memorymap, to remove trace
                tensor->memorymap.erase(subGraphID);
            }
            return true;
        }
        return false; // Return false if no suitable allocator is found
    }

    // Additional member functions can be defined here
    void DumpFreePool(MemoryType type) const {
        auto &pool = allocators[static_cast<size_t>(type)];
        std::cout << "  Total size: " << pool.totalSize << " bytes" << std::endl;
        std::cout << "  Free blocks: ";
        for (const auto &block : pool.freepool) {
            std::cout << "[offset=" << block.start << ", size=" << block.Size() << "] ";
        }
        std::cout << std::endl;
    }
};

} // namespace npu::tile_fwk

#endif // TILE_FWK_ALLOCATOR_H