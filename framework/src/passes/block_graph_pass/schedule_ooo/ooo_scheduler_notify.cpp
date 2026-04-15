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
 * \file ooo_scheduler_notify.cpp
 * \brief Observer notification helpers — all event-construction logic lives here
 *        so scheduler main flows stay focused on scheduling.
 */

#include <numeric>
#include "ooo_scheduler.h"

namespace npu::tile_fwk {

void OoOScheduler::NotifyPipeIssued(PipeType pipeType, int latency)
{
    if (observers_.empty()) return;
    Notify(PipeIssuedEvent{pipeType, latency, clock});
}

void OoOScheduler::NotifyBufferAllocated(MemoryType memType, int memId)
{
    if (observers_.empty()) return;
    Notify(BufferAllocEvent{memType, memId, localBufferMap_[memId]->size, clock});
}

void OoOScheduler::NotifyBufferFreed(MemoryType memType, int memId)
{
    if (observers_.empty()) return;
    Notify(BufferFreeEvent{memType, memId, localBufferMap_[memId]->size, clock});
}

void OoOScheduler::NotifySpill(const SpillInfo& info, LocalBufferPtr allocBuffer)
{
    if (observers_.empty()) return;

    bool needCopyOut = info.spillOp_->GetOpcodeStr().find("COPY_IN") == std::string::npos;
    int allocOccupied = 0;
    for (const auto& pair : tensorOccupyMap[allocBuffer->memType]) {
        if (opIsAllocMap[pair.second]) {
            allocOccupied += localBufferMap_[pair.first]->size;
        }
    }
    int spillCopyoutSize = 0;
    if (needCopyOut) {
        auto dtype = info.ddrTensor_->tensor->datatype;
        spillCopyoutSize = std::accumulate(
            info.ddrTensor_->shape.begin(), info.ddrTensor_->shape.end(),
            1, std::multiplies<int64_t>()) * BytesOf(dtype);
    }
    Notify(SpillEvent{allocBuffer->memType, info.spillMemId_,
        localBufferMap_[info.spillMemId_]->size, allocBuffer->size,
        info.ddrTensor_->GetMagic(), allocOccupied, spillCopyoutSize, clock});
}

void OoOScheduler::NotifyScheduleEnd(bool success)
{
    if (observers_.empty()) return;
    Notify(ScheduleEndEvent{clock, workspaceOffset, success});
}

} // namespace npu::tile_fwk
