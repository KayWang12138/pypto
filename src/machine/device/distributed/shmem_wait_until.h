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
 * \file shmem_wait_until.h
 * \brief
 */

#ifndef SHMEM_WAIT_UNTIL_H
#define SHMEM_WAIT_UNTIL_H

#include <vector>

#include "common.h"
#include "interface/cache/core_func_data.h"

namespace npu::tile_fwk::Distributed {
constexpr uint32_t VECTOR_PRE_SIZE = 1024;

class SignalTileOp {
public:
    void Init(uint64_t taskId, int32_t* addr, uint32_t endOffset, uint32_t stride, int32_t expectedSum);
    bool PollCompleted(std::vector<uint64_t> &completed);

private:
    uint64_t taskId_;
    int32_t* addr_;
    uint32_t endOffset_;
    uint32_t stride_;
    int32_t expectedSum_;
};

class ShmemWaitUntil {
public:
    void Init(DeviceTask *deviceTask);
    void EnqueueOp(uint64_t taskId, TensorInfo& info);
    void PollCompleted(std::vector<uint64_t> &completed);

private:
    std::vector<SignalTileOp> signalTileOp_{VECTOR_PRE_SIZE};
    std::vector<bool> done_ = std::vector<bool>(VECTOR_PRE_SIZE, false);
    uint32_t signalTileOpCount_{0};
};

} // namespace npu::tile_fwk::Distributed
#endif // SHMEM_WAIT_UNTIL_H
