/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <vector>
#include <string>
#include <memory>

#include "machine/device/distributed/shmem_wait_until.h"
#include "machine/device/dynamic/device_sche.h"
#include "machine/utils/dynamic/dev_workspace.h"

namespace npu::tile_fwk {

class TestDevErrorLog : public testing::Test {
public:
    void SetUp() override {
        unsetenv("ASCEND_GLOBAL_LOG_LEVEL");
        g_isLogEnableError = true;
    }

    void TearDown() override {
        unsetenv("ASCEND_GLOBAL_LOG_LEVEL");
    }
};

TEST_F(TestDevErrorLog, TaskDispatch_EnqueueOp_InvalidTaskId) {
    auto allocator = std::make_unique<dynamic::DeviceWorkspaceAllocator>();
    auto task = std::make_unique<dynamic::DynDeviceTask>(*allocator);
    auto shmemWaitUntil = std::make_unique<Distributed::ShmemWaitUntil>();

    shmemWaitUntil->Init(task.get());

    uint64_t invalidTaskId = 9999;
    TaskStat* taskStat = nullptr;

    int32_t result = shmemWaitUntil->EnqueueOp(invalidTaskId, taskStat);
    ASSERT_EQ(result, dynamic::DEVICE_MACHINE_ERROR);
}

TEST_F(TestDevErrorLog, TaskDispatch_HashMap_TaskArrayFull) {
    auto hashMap = std::make_unique<Distributed::HashMap>();
    hashMap->Init();

    int32_t dummyAddr[4] = {0};
    
    for (uint32_t i = 0; i <= Distributed::AICPU_TASK_ARRAY_SIZE; i++) {
        Distributed::SignalTileOp* task = hashMap->CreateTaskData(i, dummyAddr, 0, false);
        if (i == Distributed::AICPU_TASK_ARRAY_SIZE) {
            ASSERT_EQ(task, nullptr);
        }
    }
}

TEST_F(TestDevErrorLog, TaskDispatch_HashMap_InsertTask_Failed) {
    auto hashMap = std::make_unique<Distributed::HashMap>();
    hashMap->Init();

    int32_t dummyAddr[4] = {0};
    
    for (uint32_t i = 0; i < Distributed::AICPU_TASK_ARRAY_SIZE; i++) {
        (void)hashMap->CreateTaskData(i, dummyAddr, 0, false);
    }
    
    int32_t result = hashMap->InsertTask(Distributed::AICPU_TASK_ARRAY_SIZE, dummyAddr, 0, false);
    ASSERT_EQ(result, dynamic::DEVICE_MACHINE_ERROR);
}

TEST_F(TestDevErrorLog, TaskDispatch_CircularQueue_DequeueEmpty) {
    auto queue = std::make_unique<Distributed::CircularQueue>();

    int32_t result = queue->Dequeue();
    ASSERT_EQ(result, dynamic::DEVICE_MACHINE_ERROR);
}

TEST_F(TestDevErrorLog, TaskDispatch_CircularQueue_EnqueueFull) {
    auto queue = std::make_unique<Distributed::CircularQueue>();
    auto hashMap = std::make_unique<Distributed::HashMap>();
    hashMap->Init();
    
    int32_t dummyAddr[4] = {0};
    std::vector<Distributed::SignalTileOp*> tasks;
    
    for (uint32_t i = 0; i < Distributed::AICPU_TASK_ARRAY_SIZE - 1; i++) {
        Distributed::SignalTileOp* task = hashMap->CreateTaskData(i, dummyAddr, 0, false);
        tasks.push_back(task);
        int32_t result = queue->Enqueue(task);
        ASSERT_EQ(result, dynamic::DEVICE_MACHINE_OK);
    }
    
    Distributed::SignalTileOp* lastTask = hashMap->CreateTaskData(Distributed::AICPU_TASK_ARRAY_SIZE, dummyAddr, 0, false);
    int32_t result = queue->Enqueue(lastTask);
    ASSERT_EQ(result, dynamic::DEVICE_MACHINE_ERROR);
}

TEST_F(TestDevErrorLog, AllDevErrorTags_FormatIsCorrect) {
    std::vector<std::string> allTags = {
        "[sync_timeout]",
        "[except_signal]",
        "[except_reset]",
        "[data_valid]",
        "[init_resource]",
        "[task_dispatch]",
        "[init_params]",
        "[kernel_exec]",
        "[kernel_load]",
        "[mem_alloc]",
        "[perf_trace]"
    };
    
    for (const auto& tag : allTags) {
        ASSERT_EQ(tag[0], '[');
        ASSERT_EQ(tag[tag.size() - 1], ']');
        ASSERT_GT(tag.size(), 2u);
        ASSERT_EQ(tag.find(' '), std::string::npos);
        ASSERT_EQ(tag.find('-'), std::string::npos);
        
        bool hasUppercase = false;
        for (char c : tag) {
            if (c >= 'A' && c <= 'Z') {
                hasUppercase = true;
                break;
            }
        }
        ASSERT_FALSE(hasUppercase);
    }
}

} // namespace npu::tile_fwk
