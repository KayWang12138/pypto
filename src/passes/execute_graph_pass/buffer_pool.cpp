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
 * \file buffer_pool.cpp
 * \brief
 */

#include "passes/execute_graph_pass/buffer_pool.h"
namespace npu::tile_fwk {
constexpr size_t START_ADDR_IDX = 2;

std::map<uint64_t, std::map<uint64_t, uint64_t>> BufferPool::FindFreeIntervals() {
    // 收集可用的offset + size
    std::map<uint64_t, uint64_t> occupiedSpace;
    std::map<uint64_t, std::map<uint64_t, uint64_t>> freeIntervalsMap;
    for (auto slice : bufferSlices) {
        // 当前slice被占用着
        auto tensorEnd = slice.second.offset + slice.second.size;
        occupiedSpace[slice.second.offset] = tensorEnd;
    }
    std::map<uint64_t, uint64_t> freeIntervals;
    if (occupiedSpace.empty()) {
        freeIntervalsMap[memSize_].insert({0, memSize_});
        return freeIntervalsMap;
    }
    // 检查起始点是否为空闲
    if (occupiedSpace.begin()->first > 0) {
        freeIntervals.insert({0, occupiedSpace.begin()->first});
    }
    // 遍历所有已占用的片段，查找相邻片段之间的空闲区域
    auto prevIt = occupiedSpace.begin();
    for (auto it = occupiedSpace.begin(); it != occupiedSpace.end(); ++it) {
        if (prevIt != it && prevIt->second < it->first) {
            freeIntervals.insert({prevIt->second, it->first});
        }
        prevIt = it;
    }
    // 检查末尾是否为空闲
    if (prevIt->second < memSize_) {
        freeIntervals.insert({prevIt->second, memSize_});
    }
    for (auto freeInterval : freeIntervals) {
        freeIntervalsMap[freeInterval.second - freeInterval.first].insert(freeInterval);
    }
    return freeIntervalsMap;
}

std::vector<std::vector<int>> BufferPool::GetSpillGroup(size_t sizeNeedSpill) {
    std::vector<std::tuple<int, size_t, size_t>> allocatedBufs;
    for (auto &[memId, bufferSlice] : bufferSlices) {
        allocatedBufs.push_back(std::make_tuple(memId, bufferSlice.offset, bufferSlice.offset + bufferSlice.size));
    }
    std::sort(allocatedBufs.begin(), allocatedBufs.end(),
        [&](std::tuple<int, size_t, size_t> &a, std::tuple<int, size_t, size_t> &b) {
            return std::get<1>(a) < std::get<1>(b);
        });

    std::vector<std::vector<int>> canSpillGroups;
    size_t i = 0;
    while (i < allocatedBufs.size()) {
        size_t startAddr = std::get<1>(allocatedBufs[i]);
        if (i == 0) {
            startAddr = 0;
        } else {
            startAddr = std::get<START_ADDR_IDX>(allocatedBufs[i - 1]);
        }

        if ((memSize_ - startAddr) < sizeNeedSpill) {
            break;
        }

        size_t j = i;
        while (j < allocatedBufs.size() && (std::get<1>(allocatedBufs[j]) - startAddr) < sizeNeedSpill) {
            j += 1;
        }

        size_t endAddr = memSize_;
        if (j < allocatedBufs.size()) {
            endAddr = std::get<1>(allocatedBufs[j]);
        }

        while (i < (j-1) && (endAddr - std::get<START_ADDR_IDX>(allocatedBufs[i])) >= sizeNeedSpill) {
            i += 1;
        }
        ASSERT(i != j);

        std::vector<int> group;
        for (size_t k = i; k < j; k++) {
            group.push_back(std::get<0>(allocatedBufs[k]));
        }

        canSpillGroups.push_back(group);
        i += 1;
    }
    return canSpillGroups;
}

Status BufferPool::Allocate(LocalBufferPtr tensor) {
    std::map<uint64_t, std::map<uint64_t, uint64_t>> freeIntervals = FindFreeIntervals();
    // 创建新的bufferSlice
    for (auto &interval : freeIntervals) {
        if (interval.first < tensor->size) {
            continue;
        }
        for (auto &freeSpace : interval.second) {
            BufferSlice newSlice;
            newSlice.size = tensor->size;
            newSlice.offset = freeSpace.first;
            if (bufferSlices.find(tensor->id) != bufferSlices.end()) { ALOG_ERROR_F("Tensor[%u] already alloc in bufferSlices", tensor->id); return FAILED; }
            bufferSlices[tensor->id] = newSlice;
            tensor->start = newSlice.offset;
            tensor->end = newSlice.offset + newSlice.size;
            ALOG_DEBUG_F("    Allocate Tensor[%u], range [%lu, %lu].",
                tensor->id, newSlice.offset, newSlice.size + newSlice.offset);
            return SUCCESS;
        }
    }
    ALOG_ERROR_F("Buffer doesnot have enough memory to allocate Tensor[%u]", tensor->id);
    return FAILED;
}

Status BufferPool::Free(const uint32_t tensorId) {
    if (bufferSlices.find(tensorId) == bufferSlices.end()) { ALOG_ERROR_F("Tensor[%d] not in bufferSlices", tensorId); return FAILED; }
    ALOG_DEBUG_F("    Free tensor[%u], range:[%lu, %lu]", tensorId,
        bufferSlices[tensorId].offset, bufferSlices[tensorId].size + bufferSlices[tensorId].offset);
    bufferSlices.erase(tensorId);
    return SUCCESS;
}

bool BufferPool::IsFull(const LocalBufferPtr tensor) {
    auto freeSpace = FindFreeIntervals();
    for (auto inter : freeSpace) {
        if (inter.first >= tensor->size) {
            return false;
        }
    }
    return true;
}
}  // namespace npu::tile_fwk