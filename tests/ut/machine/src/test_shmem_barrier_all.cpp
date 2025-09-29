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
 * \file test_shmem_barrier_all.cpp
 * \brief
 */

#include <memory>
#include <vector>

#include <gtest/gtest.h>

#include "machine/device/distributed/common.h"
#include "machine/device/distributed/shmem_barrier_all.h"
#include "machine/utils/dynamic/dev_workspace.h"
#include "tilefwk/aicore_data.h"
#include "tilefwk/core_func_data.h"
#include "tileop/hccl_context.h"

namespace {

struct MockHcclCombinOpParam : TileOp::HcclCombinOpParam {
    MockHcclCombinOpParam(const uint32_t rankSize)
    {
        rankNum = rankSize;
        for (uint32_t rankIndex = 0; rankIndex < rankSize; ++rankIndex) {
            windowsExp[rankIndex] = reinterpret_cast<uint64_t>(
                new uint64_t[npu::tile_fwk::Distributed::TOTAL_WIN_EXP_SIZE]{0});
        }
    }
    ~MockHcclCombinOpParam()
    {
        for (uint32_t rankIndex = 0; rankIndex < rankNum; ++rankIndex) {
            delete[] reinterpret_cast<uint64_t*>(windowsExp[rankIndex]);
        }
    }
};

struct MockDeviceTask : npu::tile_fwk::dynamic::DynDeviceTask {
    std::unique_ptr<uint8_t[]> buffer;
    MockDeviceTask(npu::tile_fwk::dynamic::DeviceWorkspaceAllocator& allocator, const MockHcclCombinOpParam& mockParam)
    : npu::tile_fwk::dynamic::DynDeviceTask(allocator) {
        size_t size = sizeof(npu::tile_fwk::DynFuncHeader) + sizeof(npu::tile_fwk::DynFuncData);
        this->buffer = std::make_unique<uint8_t[]>(size);
        memset(this->buffer.get(), 0, size);
        auto* header = reinterpret_cast<npu::tile_fwk::DynFuncHeader*>(this->buffer.get());
        auto* data = reinterpret_cast<npu::tile_fwk::DynFuncData*>(header + 1);
        constexpr uint32_t groupIndex = 0;
        data->hcclContext[groupIndex] = reinterpret_cast<uint64_t>(&mockParam);
        this->dynFuncData = header;
    }
    ~MockDeviceTask() {
        this->dynFuncData = nullptr;
    }
};

struct MockData {
    npu::tile_fwk::dynamic::DeviceWorkspaceAllocator allocator;
    MockHcclCombinOpParam mockParam;
    std::vector<std::unique_ptr<MockDeviceTask>> deviceTasks;
    std::vector<npu::tile_fwk::Distributed::ShmemBarrierAll> barriers;
    MockData(const uint32_t rankSize) : mockParam(rankSize)
    {
        this->deviceTasks.reserve(rankSize);
        this->barriers.reserve(rankSize);
        for (uint32_t rankIndex = 0; rankIndex < rankSize; ++rankIndex) {
            this->deviceTasks.emplace_back(std::make_unique<MockDeviceTask>(this->allocator, this->mockParam));
            this->barriers.emplace_back();
            this->barriers.back().Init(this->deviceTasks.back().get());
        }
    }
};

void TestShmemBarrierAll(const uint32_t rankSize, const uint32_t tileOpCount)
{
    constexpr int codeSize = 9;
    auto data = std::make_unique<int32_t[]>(codeSize);
    int32_t initData[codeSize] = {153, 2, 0, 0, 2, 0, 0, 1, 0};
    std::copy(initData, initData + codeSize, data.get());
    npu::tile_fwk::dynamic::DevRelocVector<int32_t> aicpuCode(codeSize, data.get());
    MockData mockData(rankSize);
    std::vector<std::vector<uint64_t>> completedTasks(rankSize);
    for (uint32_t tileOpIndex = 0; tileOpIndex < tileOpCount; ++tileOpIndex) {
        for (uint32_t rankIndex = 0; rankIndex < rankSize; ++rankIndex) {
            mockData.barriers[rankIndex].EnqueueOp(tileOpIndex, aicpuCode);
        }
        for (uint32_t rankIndex = 0; rankIndex < rankSize; ++rankIndex) {
            mockData.barriers[rankIndex].PollCompleted(completedTasks[rankIndex]);
        }
    }

    for (auto& completedTask : completedTasks) {
        ASSERT_EQ(completedTask.size(), tileOpCount);
        for (uint32_t tileOpIndex = 0; tileOpIndex < tileOpCount; ++tileOpIndex) {
            ASSERT_EQ(completedTask[tileOpIndex], tileOpIndex);
        }
    }
}

TEST(ShmemBarrierAllTest, BasicFunctionality)
{
    constexpr int32_t rankSize = 4;
    constexpr int32_t tileOpCount = 1;
    TestShmemBarrierAll(rankSize, tileOpCount);
}

TEST(ShmemBarrierAllTest, VectorResize)
{
    constexpr int32_t rankSize = 4;
    constexpr int32_t tileOpCount = npu::tile_fwk::Distributed::VECTOR_PRE_SIZE + 1;
    TestShmemBarrierAll(rankSize, tileOpCount);
}

TEST(ShmemBarrierAllTest, WinExpReuse)
{
    constexpr int32_t rankSize = 4;
    constexpr int32_t tileOpCount = npu::tile_fwk::Distributed::TOTAL_WIN_EXP_SIZE * 3;
    TestShmemBarrierAll(rankSize, tileOpCount);
}

} // namespace
