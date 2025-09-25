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
 * \file shmem_barrier_all.cpp
 * \brief
 */

#include "shmem_barrier_all.h"

#include <cstdint>
#include <vector>

#include "machine/utils/device_log.h"
#include "machine/utils/dynamic/dev_workspace.h"
#include "tilefwk/aicore_data.h"
#include "tileop/hccl_context.h"

namespace npu::tile_fwk::Distributed {

void ShmemBarrierAll::Init(npu::tile_fwk::dynamic::DynDeviceTask* deviceTask)
{
    DEV_ASSERT(deviceTask != nullptr);
    npu::tile_fwk::DynFuncHeader* header = deviceTask->dynFuncData;
    DEV_ASSERT(header != nullptr);
    auto* data = reinterpret_cast<npu::tile_fwk::DynFuncData*>(header + 1);
    DEV_ASSERT((data != nullptr) && (data->hcclContext != nullptr));
    hcclContext_ = data->hcclContext;
}

void ShmemBarrierAll::EnqueueOp(uint64_t taskId, TensorInfo& info)
{
    (void)info;

    constexpr uint32_t groupIndex = 0;
    TileOp::HcclCombinOpParam* hcclOpParam = reinterpret_cast<TileOp::HcclCombinOpParam*>(hcclContext_[groupIndex]);
    rankSize_ = hcclOpParam->rankNum;
    for (uint32_t rankId = 0; rankId < rankSize_; rankId++) {
        uint64_t* winExp = reinterpret_cast<uint64_t*>(hcclOpParam->windowsExp[rankId]);
        winExp[winExpIndex_]++;
    }

    if (tileOpCount_ == barrierInfo_.size()) {
        barrierInfo_.resize(tileOpCount_ * 2); // 扩容到原本的 2 倍
        done_.resize(tileOpCount_ * 2); // 扩容到原本的 2 倍
    }
    barrierInfo_[tileOpCount_] = {taskId, hcclOpParam->windowsExp, winExpIndex_, round_ * rankSize_};

    winExpIndex_++;
    if (winExpIndex_ == TOTAL_WIN_EXP_SIZE) {
        winExpIndex_ = 0;
        round_++;
    }
    tileOpCount_++;
}

bool ShmemBarrierAll::Ready(BarrierInfo& info)
{
    for (uint32_t rankId = 0; rankId < rankSize_; rankId++) {
        uint64_t* winExp = reinterpret_cast<uint64_t*>(info.winExp[rankId]);
        if (winExp[info.winExpIndex] != info.expected) {
            return false;
        }
    }
    return true;
}

void ShmemBarrierAll::PollCompleted(std::vector<uint64_t>& completed)
{
    for (uint32_t tileOpIndex = 0; tileOpIndex < tileOpCount_; tileOpIndex++) {
        if (done_[tileOpIndex]) {
            continue;
        }
        BarrierInfo& info = barrierInfo_[tileOpIndex];
        if (Ready(info)) {
            completed.emplace_back(info.taskId);
            done_[tileOpIndex] = true;
        }
    }
}

} // namespace npu::tile_fwk::Distributed
