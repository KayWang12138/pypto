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
 * \file allocators.h
 * \brief
 */

#pragma once

#include "slab_ws_allocator.h"
#include "ws_allocator_basics.h"
#include "ws_slot_allocator.h"
#include "seq_ws_allocator.h"
#include "ws_allocator_counter.h"
#include "ws_metadata_allocator.h"
#include "tilefwk/aikernel_data.h"

namespace npu::tile_fwk::dynamic {
struct MetadataAllocator {
    WsMetadataAllocator general; // aicpu coherent for small suballocation, not support recycle
    SlabWsAllocator generalSlab; // aicpu meta memory, support reclamation
    SlabWsAllocator stitchSlab;  // aicpu stitched data support reclamation
};

struct TensorAllocator {
    uint32_t curParallelWsId{0}; // different parallel for loop use different workspace range
    uint32_t parallelism{1};
    SeqWsAllocator rootInner[SCH_DEVTASK_MAX_PARALLELISM];
    SeqWsAllocator devTaskInnerExclusiveOutcasts[SCH_DEVTASK_MAX_PARALLELISM];
    WsSlotAllocator devTaskBoundaryOutcasts[SCH_DEVTASK_MAX_PARALLELISM];

    bool RootInnerCanAllocate(uint64_t rootInnerMemReq) {
        return rootInner[curParallelWsId].CanAllocate(rootInnerMemReq);
    }

    WsAllocation RootInnerMalloc(uint64_t memReq, WsMemCategory category = WsMemCategory::UNCLASSIFIED) {
        return rootInner[curParallelWsId].Malloc(memReq, category);
    }

    void RootInnerResetPool() {
        return rootInner[curParallelWsId].ResetPool();
    }

    uint32_t RootInnerResetTimes() const {
        return rootInner[curParallelWsId].ResetTimes();
    }

    bool DevTaskInnerExOutCastCanAllocate(uint64_t rootInnerMemReq) {
        return devTaskInnerExclusiveOutcasts[curParallelWsId].CanAllocate(rootInnerMemReq);
    }

    WsAllocation DevTaskInnerExOutCastMalloc(uint64_t memReq, WsMemCategory category = WsMemCategory::UNCLASSIFIED) {
        return devTaskInnerExclusiveOutcasts[curParallelWsId].Malloc(memReq, category);
    }

    void DevTaskInnerExOutCastResetPool() {
        devTaskInnerExclusiveOutcasts[curParallelWsId].ResetPool();
    }

    size_t DevTaskBoundaryOutcastsAvailableSlots() const {
        return devTaskBoundaryOutcasts[curParallelWsId].AvailableSlots();;
    }

    bool DevTaskBoundaryOutcastsIsValidSlotMemReq(uint64_t memReq) const {
        return devTaskBoundaryOutcasts[curParallelWsId].IsValidSlotMemRequirement(memReq);
    }

    WsAllocation DevTaskBoundaryOutcastsAllocate() {
        WsAllocation allocation = devTaskBoundaryOutcasts[curParallelWsId].Allocate();
        allocation.parallelWsId = curParallelWsId;
        return allocation;
    }

    uint64_t DevTaskBoundaryOutcastsSlotByteSize() const {
        return devTaskBoundaryOutcasts[curParallelWsId].SlotByteSize();
    }

    void DevTaskBoundaryOutcastsDeallocate(WsAllocation allocation) {
        return devTaskBoundaryOutcasts[allocation.parallelWsId].Deallocate(allocation.ptr);
    }

    void DumpMemoryUsage(const char* hint) const {
        for (uint32_t i = 0; i < parallelism; i++) {
            DEV_MEM_DUMP("Parallel workspace %u.", i);
            rootInner[i].DumpMemoryUsage(hint, "Tensor (root inner) workspace");
            devTaskInnerExclusiveOutcasts[i].DumpMemoryUsage(hint, "Tensor (DeviceTask inner outcasts) workspace");
            devTaskBoundaryOutcasts[i].DumpMemoryUsage(hint);
        }
    }
};
struct RuntimeReuseInfo {
    uint32_t poolResetTimes;
};
}
