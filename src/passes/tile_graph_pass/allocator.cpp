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
 * \file allocator.cpp
 * \brief
 */

#include "passes/tile_graph_pass/allocator.h"
#include "interface/function/function.h"

namespace npu::tile_fwk {

bool IsOverlap(const TileRange &a, const TileRange &b) {
    // Two ranges overlap if one starts before the other ends
    // and the other starts before the first ends
    return (a.start < b.end) && (b.start < a.end);
}

bool LessThan(const TileRange &a, const TileRange &b) {
    return a.start < b.start;
}

// Check if all tensors in conflictermagic can be allocated
bool TiFWKJointAllocator::CanAllocate(const std::vector<std::shared_ptr<LogicalTensor>> &toAlloc, int subGraphID,
    std::shared_ptr<LogicalTensor> &failedTensor) {
    bool success = true;
    std::vector<std::shared_ptr<LogicalTensor>> tmpSet;

    for (auto &&toAllocTensor : toAlloc) {
        // already allocated tensor shall not add to tmpSet to free
        if (toAllocTensor->memorymap.find(subGraphID) != toAllocTensor->memorymap.end()) {
            continue;
        }
        bool rv = Allocate(true, toAllocTensor, subGraphID);
        if (rv) {
            tmpSet.push_back(toAllocTensor);
        } else {
            success = false;
            failedTensor = toAllocTensor;
            break;
        }
    }

    if (!success) {
        for (auto &&allocedTensor : tmpSet) {
            // free all the tensors that were allocated and remove from memorymap
            Free(true, allocedTensor, subGraphID);
        }
    }

    return success;
}

} // namespace npu::tile_fwk