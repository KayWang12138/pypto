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
 * \file buffer_pool.h
 * \brief
 */

#ifndef PASS_BUFFER_POOL_H_
#define PASS_BUFFER_POOL_H_
#include "tilefwk/data_type.h"
#include "interface/utils/log.h"
#include "interface/utils/common.h"
#include "passes/pass_utils/pass_utils.h"

namespace npu::tile_fwk {
struct LocalBuffer {
    uint32_t id{0};
    uint64_t retireCycle{0};
    uint64_t startCycle{0};
    size_t start{0};
    size_t end{0};
    size_t size{0};
    MemoryType memType{MemoryType::MEM_UNKNOWN};
    bool operator<(const LocalBuffer& other) const {
        if (size < other.size) {
            return true;
        }
        if (size == other.size) {
            return id < other.id;
        }
        return false;
    }

    LocalBuffer(uint32_t tensorId, uint64_t shapeSize, MemoryType type) {
       id = tensorId;
       size = shapeSize; 
       memType = type;
    }
};

using LocalBufferPtr = std::shared_ptr<LocalBuffer>;
// BufferSlice 一个时刻只能给一个tensor使用，允许生命周期不重叠的多个tensor分时复用
struct BufferSlice {
    uint64_t size{0};
    uint64_t offset{0};
};

class BufferPool {
  public:
    BufferPool(const MemoryType mem, const uint64_t memSize)
        : memType_(mem), memSize_(memSize) {}
    BufferPool() {}
    ~BufferPool() = default;
    // 返回tensorid 到 bufferblock的映射关系，value是bufferblock的index不是bufferblock的magic
    // 在已有的block中分配tensor空间
    Status Allocate(LocalBufferPtr tensor);
    std::map<uint64_t, uint64_t> GenFreeIntervals(const std::map<uint64_t, uint64_t> &occupiedSpace);
    std::map<uint64_t, std::map<uint64_t, uint64_t>> FindFreeIntervals();
    bool IsFull(const LocalBufferPtr tensor);
    Status Free(const uint32_t tensorId);
    uint64_t GetMemSize();

    size_t ObtainStartAddr(size_t i, const std::vector<std::tuple<int, size_t, size_t>> &allocatedBufs);
    size_t UpdateIdx(size_t &i, size_t sizeNeedSpill, size_t startAddr, const std::vector<std::tuple<int, size_t, size_t>> &allocatedBufs);
    Status GetSpillGroup(size_t sizeNeedSpill, std::vector<std::vector<int>> &canSpillGroups);
    std::vector<uint32_t> GetBufferSlices();
    std::vector<int> GetAddrSortedBufs();
    bool isAllocate(const uint32_t tensorId);
  private:
    MemoryType memType_{MemoryType::MEM_UNKNOWN};
    uint64_t memSize_{0};
    std::map<uint32_t, BufferSlice> bufferSlices;
    std::unordered_map<uint32_t, uint32_t> tensorIdToBuffer_;
};
}  // namespace npu::tile_fwk

#endif