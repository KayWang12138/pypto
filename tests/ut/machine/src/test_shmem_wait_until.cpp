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
 * \file test_shmem_wait_until.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include "machine/device/distributed/shmem_wait_until.h"
#include "interface/cache/core_func_data.h"
#include "tileop/hccl_context.h"

namespace {
TEST(ShmemWaitUntilTest, ShmemWaitUntil_EnqueueOp_and_PollCompleted_success) {
    constexpr uint64_t taskId = 123;
    npu::tile_fwk::Distributed::TensorInfo info;
    info.offset = {0, 1, 0, 0};
    info.shape = {1, 1, 1, 8};
    constexpr uint32_t rawShape1 = 4;
    constexpr uint32_t rawShape2 = 4;
    constexpr uint32_t rawShape3 = 8;
    info.rawShape = {4, rawShape1, rawShape2, rawShape3};
    int32_t rawAddr[rawShape1 * rawShape2 * rawShape3] = {0};
    info.rawAddr = reinterpret_cast<uint64_t>(rawAddr);
    constexpr int32_t value = 1;

    npu::tile_fwk::Distributed::ShmemWaitUntil shmemWaitUntil;
    npu::tile_fwk::DeviceTask deviceTask;
    shmemWaitUntil.Init(&deviceTask);

    shmemWaitUntil.EnqueueOp(taskId, info);

    int32_t* addr = rawAddr + info.offset[1] * info.rawShape[2] * info.rawShape[3] + info.offset[2] * info.rawShape[3] + info.offset[3];
    for (uint32_t offset = 0; offset < info.shape[2] * info.shape[3]; offset += info.shape[3]) {
        addr[offset] = value;
    }
    std::vector<uint64_t> completed;
    shmemWaitUntil.PollCompleted(completed);
    ASSERT_EQ(completed.size(), 1);
    ASSERT_EQ(completed[0], taskId);
}
} // namespace