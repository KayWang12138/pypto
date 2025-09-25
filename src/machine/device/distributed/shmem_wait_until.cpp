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
 * \file shmem_wait_until.cpp
 * \brief
 */

#include "shmem_wait_until.h"

#include <cstddef>
#include <type_traits>
#include <vector>
#include <cstring>
#include <cstdint>

#include "securec.h"

#include "tileop/hccl_context.h"
#include "machine/utils/device_log.h"
#include "machine/utils/dynamic/dev_workspace.h"
#include "neon_stub.h"


namespace npu::tile_fwk::Distributed {
void SignalTileOp::Init(uint64_t taskId, int32_t* addr, uint32_t count, uint32_t stride, int32_t expectedSum)
{
    taskId_ = taskId;
    addr_ = addr;
    endOffset_ = count * stride;
    stride_ = stride;
    expectedSum_ = expectedSum;
}

bool SignalTileOp::PollCompleted(std::vector<uint64_t> &completed)
{
    int32_t sum = 0;
    for (uint32_t offset = 0; offset < endOffset_; offset += stride_) {
        sum += addr_[offset];
    }
    if (sum == expectedSum_) {
        completed.emplace_back(taskId_);
        return true;
    }
    return false;
}

void ShmemWaitUntil::Init(npu::tile_fwk::dynamic::DynDeviceTask* dynDeviceTask)
{
    (void)dynDeviceTask;
}

void ShmemWaitUntil::EnqueueOp(uint64_t taskId, TensorInfo& info)
{
    const uint32_t offset1 = info.offset[1]; // offset 1
    const uint32_t offset2 = info.offset[2]; // offset 2
    const uint32_t offset3 = info.offset[3]; // offset 3
    const uint32_t shape2 = info.shape[2]; // shape 2
    const uint32_t shape3 = info.shape[3]; // shape 3
    const uint32_t rawShape2 = info.rawShape[2]; // raw shape 2
    const uint32_t rawShape3 = info.rawShape[3]; // raw shape 3
    const int32_t expectedSum = 1; // todo
    DEV_DEBUG("ShmemWaitUntil::EnqueueOp offset1=%u, offset2=%u, offset3=%u, shape2=%u, shape3=%u, rawShape2=%u, rawShape3=%u", offset1, offset2, offset3, shape2, shape3, rawShape2, rawShape3);

    int32_t* addr = reinterpret_cast<int32_t*>(info.rawAddr) + offset1 * rawShape2 * rawShape3 + offset2 * rawShape3 + offset3;

    if (signalTileOpCount_ == signalTileOp_.size()) {
        signalTileOp_.resize(signalTileOpCount_ * 2); // 扩容到原本的 2 倍
        done_.resize(signalTileOpCount_ * 2); // 扩容到原本的 2 倍
    }
    signalTileOp_[signalTileOpCount_].Init(taskId, addr, shape2, shape3, expectedSum);
    ++signalTileOpCount_;
}

void ShmemWaitUntil::PollCompleted(std::vector<uint64_t> &completed)
{
    for (uint32_t i = 0; i < signalTileOpCount_; ++i) {
        if (done_[i]) {
            continue;
        }
        if (signalTileOp_[i].PollCompleted(completed)) {
            done_[i] = true;
        }
    }
}

} // namespace npu::tile_fwk::Distributed
