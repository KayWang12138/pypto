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
#include "tileop/a2a3/hccl_context.h"

namespace {
TEST(ShmemWaitUntilTest, ShmemWaitUntil_EnqueueOp_and_PollCompleted_success) {
    constexpr uint64_t taskId = 123;
    constexpr uint32_t offset0 = 0;
    constexpr uint32_t offset1 = 0;
    constexpr uint32_t offset2 = 0;
    constexpr uint32_t offset3 = 0;
    constexpr uint32_t shape0 = 1;
    constexpr uint32_t shape1 = 1;
    constexpr uint32_t shape2 = 4;
    constexpr uint32_t shape3 = 8;
    constexpr uint32_t rawShape0 = 4;
    constexpr uint32_t rawShape1 = 4;
    constexpr uint32_t rawShape2 = 4;
    constexpr uint32_t rawShape3 = 8;
    constexpr int32_t value = 2;
    constexpr int32_t expectedSum = shape2 * value;

    npu::tile_fwk::Distributed::ShmemWaitUntil shmemWaitUntil;
    npu::tile_fwk::DeviceTask deviceTask;
    shmemWaitUntil.Init(&deviceTask);

    uint64_t attr[13] = {offset0, offset1, offset2, offset3, shape0, shape1, shape2, shape3, rawShape0,
        rawShape1, rawShape2, rawShape3, expectedSum};
    int32_t rawAddr[rawShape1 * rawShape2 * rawShape3] = {0};
    npu::tile_fwk::Distributed::SignalTensorInfo info = {attr, reinterpret_cast<uint64_t>(rawAddr)};

    shmemWaitUntil.EnqueueOp(taskId, info);

    int32_t* addr = rawAddr + offset1 * rawShape2 * rawShape3 + offset2 * rawShape3 + offset3;
    for (uint32_t offset = 0; offset < shape2 * shape3; offset += shape3) {
        addr[offset] = value;
    }
    std::vector<uint64_t> completed;
    shmemWaitUntil.PollCompleted(completed);
    ASSERT_EQ(completed.size(), 1);
    ASSERT_EQ(completed[0], taskId);
}
} // namespace