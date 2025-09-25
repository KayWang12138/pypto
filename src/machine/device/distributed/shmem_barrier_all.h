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
 * \file shmem_barrier_all.h
 * \brief
 */

#pragma once

#include <cstdint>
#include <vector>

#include "common.h"
#include "machine/utils/dynamic/dev_workspace.h"

namespace npu::tile_fwk::Distributed {

constexpr uint16_t TOTAL_WIN_EXP_SIZE = 1024;

struct BarrierInfo {
    uint64_t taskId;
    uint64_t* winExp;
    uint32_t winExpIndex;
    uint64_t expected;
};

class ShmemBarrierAll {
public:
    void Init(npu::tile_fwk::dynamic::DynDeviceTask* deviceTask);
    void EnqueueOp(uint64_t taskId, TensorInfo& info);
    void PollCompleted(std::vector<uint64_t>& completed);

private:
    std::vector<BarrierInfo> barrierInfo_{VECTOR_PRE_SIZE};
    std::vector<bool> done_ = std::vector<bool>(VECTOR_PRE_SIZE, false);
    uint64_t* hcclContext_{nullptr};
    uint64_t tileOpCount_{0};
    uint16_t winExpIndex_{0};
    uint64_t round_{1};
    uint32_t rankSize_{0};

    bool Ready(BarrierInfo& info);
};

} // namespace npu::tile_fwk::Distributed
