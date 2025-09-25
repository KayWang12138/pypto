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

#include "machine/device/distributed/common.h"
#include "machine/device/distributed/shmem_wait_until.h"
#include "machine/utils/dynamic/dev_workspace.h"
#include "tileop/hccl_context.h"

namespace {

void TestShmemWaitUntil(const uint32_t tileOpCount) {
    npu::tile_fwk::Distributed::TensorInfo info;
    info.offset = {0, 1, 0, 0};
    info.shape = {1, 1, 1, 8};
    constexpr uint32_t rankSize = 4;
    constexpr uint32_t rawShape0 = rankSize;
    constexpr uint32_t rawShape1 = rankSize;
    constexpr uint32_t rawShape2 = 4;
    constexpr uint32_t rawShape3 = 8;
    info.rawShape = {rawShape0, rawShape1, rawShape2, rawShape3};
    int32_t rawAddr[rawShape1 * rawShape2 * rawShape3] = {0};
    info.rawAddr = reinterpret_cast<uint64_t>(rawAddr);
    constexpr int32_t value = 1;

    int32_t* addr = rawAddr + info.offset[1] * info.rawShape[2] * info.rawShape[3] + info.offset[2] * info.rawShape[3] + info.offset[3];
    for (uint32_t offset = 0; offset < info.shape[2] * info.shape[3]; offset += info.shape[3]) {
        addr[offset] = value;
    }

    npu::tile_fwk::Distributed::ShmemWaitUntil shmemWaitUntil;
    npu::tile_fwk::dynamic::DeviceWorkspaceAllocator allocator;
    npu::tile_fwk::dynamic::DynDeviceTask dynDeviceTask(allocator);
    shmemWaitUntil.Init(&dynDeviceTask);

    for (uint32_t tileOpIndex = 0; tileOpIndex < tileOpCount; ++tileOpIndex) {
        shmemWaitUntil.EnqueueOp(tileOpIndex, info);
        std::vector<uint64_t> completed;
        shmemWaitUntil.PollCompleted(completed);
        ASSERT_EQ(completed.size(), 1);
        ASSERT_EQ(completed[0], tileOpIndex);
    }
}

TEST(ShmemWaitUntilTest, BasicFunctionality) {
    constexpr int32_t tileOpCount = 1;
    TestShmemWaitUntil(tileOpCount);
}

TEST(ShmemWaitUntilTest, VectorResize) {
    constexpr int32_t tileOpCount = npu::tile_fwk::Distributed::VECTOR_PRE_SIZE + 1;
    TestShmemWaitUntil(tileOpCount);
}

} // namespace