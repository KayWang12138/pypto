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
#include <cstdlib>
#include <memory>
#include <cstdint>
#include <cstring>
#include <sstream>
#include <iostream>
#include <cstdio>
#include <functional>
#include <array>

#ifdef _WIN32
#include <io.h>
#define pipe(x) _pipe(x, 8192, _O_BINARY)
#define dup(x) _dup(x)
#define dup2(x, y) _dup2(x, y)
#define close(x) _close(x)
#define read(x, y, z) _read(x, y, z)
#define STDOUT_FILENO 1
#define STDERR_FILENO 2
#else
#include <unistd.h>
#include <fcntl.h>
#endif

#include "machine/device/distributed/shmem_wait_until.h"
#include "machine/device/dynamic/device_sche.h"
#include "machine/device/dynamic/aicore_manager.h"
#include "machine/device/dynamic/aicore_hal.h"
#include "machine/device/dynamic/device_utils.h"
#include "machine/device/dynamic/aicpu_task_manager.h"
#include "machine/device/dynamic/context/device_execute_context.h"
#include "machine/device/dynamic/context/device_task_context.h"
#include "machine/device/machine_interface/pypto_aicpu_interface.h"
#include "machine/utils/device_log.h"
#include "machine/utils/dynamic/dev_workspace.h"
#include "machine/utils/dynamic/small_array.h"
#include "machine/utils/dynamic/dev_encode_types.h"
#include "utils/host_log/log_manager.h"

namespace npu::tile_fwk {

struct CapturedOutput {
    std::string stdout;
    std::string stderr;
};

CapturedOutput CaptureAllOutput(std::function<void()> func) {
    CapturedOutput result;
    
    int pipe_stdout[2];
    int pipe_stderr[2];
    
    if (pipe(pipe_stdout) != 0 || pipe(pipe_stderr) != 0) {
        return result;
    }
    
    int old_stdout = dup(STDOUT_FILENO);
    int old_stderr = dup(STDERR_FILENO);
    
    if (old_stdout == -1 || old_stderr == -1) {
        close(pipe_stdout[0]);
        close(pipe_stdout[1]);
        close(pipe_stderr[0]);
        close(pipe_stderr[1]);
        return result;
    }
    
    if (dup2(pipe_stdout[1], STDOUT_FILENO) == -1 || 
        dup2(pipe_stderr[1], STDERR_FILENO) == -1) {
        close(pipe_stdout[0]);
        close(pipe_stdout[1]);
        close(pipe_stderr[0]);
        close(pipe_stderr[1]);
        close(old_stdout);
        close(old_stderr);
        return result;
    }
    
    close(pipe_stdout[1]);
    close(pipe_stderr[1]);
    
    func();
    fflush(stdout);
    fflush(stderr);
    
    dup2(old_stdout, STDOUT_FILENO);
    dup2(old_stderr, STDERR_FILENO);
    close(old_stdout);
    close(old_stderr);
    
    std::array<char, 8192> buffer_stdout;
    std::array<char, 8192> buffer_stderr;
    
    ssize_t len_stdout = read(pipe_stdout[0], buffer_stdout.data(), buffer_stdout.size() - 1);
    ssize_t len_stderr = read(pipe_stderr[0], buffer_stderr.data(), buffer_stderr.size() - 1);
    
    close(pipe_stdout[0]);
    close(pipe_stderr[0]);
    
    if (len_stdout > 0) {
        result.stdout.assign(buffer_stdout.data(), len_stdout);
    }
    if (len_stderr > 0) {
        result.stderr.assign(buffer_stderr.data(), len_stderr);
    }
    
    return result;
}

bool ContainsTag(const std::string& output, const std::string& tag) {
    return output.find(tag) != std::string::npos;
}

void VerifyTagFormat(const std::string& tag) {
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

    CapturedOutput output = CaptureAllOutput([&]() {
        int32_t result = shmemWaitUntil->EnqueueOp(invalidTaskId, taskStat);
        ASSERT_EQ(result, dynamic::DEVICE_MACHINE_ERROR);
    });

    ASSERT_TRUE(ContainsTag(output.stdout, "[task_dispatch]")) 
        << "stdout should contain [task_dispatch] tag. stdout: " << output.stdout;
    
    ASSERT_TRUE(output.stdout.find("EnqueueOp failed") != std::string::npos)
        << "stdout should contain 'EnqueueOp failed' message";
}

TEST_F(TestDevErrorLog, TaskDispatch_HashMap_TaskArrayFull) {
    auto hashMap = std::make_unique<Distributed::HashMap>();
    hashMap->Init();

    int32_t dummyAddr[4] = {0};
    
    CapturedOutput output = CaptureAllOutput([&]() {
        for (uint32_t i = 0; i <= Distributed::AICPU_TASK_ARRAY_SIZE; i++) {
            Distributed::SignalTileOp* task = hashMap->CreateTaskData(i, dummyAddr, 0, false);
            if (i == Distributed::AICPU_TASK_ARRAY_SIZE) {
                ASSERT_EQ(task, nullptr);
            }
        }
    });

    ASSERT_TRUE(ContainsTag(output.stdout, "[task_dispatch]")) 
        << "stdout should contain [task_dispatch] tag. stdout: " << output.stdout;
    
    ASSERT_TRUE(output.stdout.find("Task array is full") != std::string::npos)
        << "stdout should contain 'Task array is full' message";
}

TEST_F(TestDevErrorLog, TaskDispatch_CircularQueue_DequeueEmpty) {
    auto queue = std::make_unique<Distributed::CircularQueue>();

    CapturedOutput output = CaptureAllOutput([&]() {
        int32_t result = queue->Dequeue();
        ASSERT_EQ(result, dynamic::DEVICE_MACHINE_ERROR);
    });

    ASSERT_TRUE(ContainsTag(output.stdout, "[task_dispatch]")) 
        << "stdout should contain [task_dispatch] tag. stdout: " << output.stdout;
    
    ASSERT_TRUE(output.stdout.find("Dequeue failed: queue is empty") != std::string::npos)
        << "stdout should contain 'Dequeue failed' message";
}

TEST_F(TestDevErrorLog, DataValid_FixedArray_ResizeExceedLimit) {
    FixedArray<int, 10> arr;
    
    CapturedOutput output = CaptureAllOutput([&]() {
        arr.resize(11);
    });

    ASSERT_TRUE(ContainsTag(output.stdout, "[data_valid]")) 
        << "stdout should contain [data_valid] tag. stdout: " << output.stdout;
    
    ASSERT_TRUE(output.stdout.find("resize failed: size") != std::string::npos)
        << "stdout should contain 'resize failed' message";
}

TEST_F(TestDevErrorLog, DataValid_DevRelocVector_IndexOutOfBounds) {
    using namespace npu::tile_fwk::dynamic;
    int data[5] = {1, 2, 3, 4, 5};
    DevRelocVector<int> vec(5, data);
    
    CapturedOutput output = CaptureAllOutput([&]() {
        int val = vec[10];
        (void)val;
    });

    ASSERT_TRUE(ContainsTag(output.stdout, "[data_valid]")) 
        << "stdout should contain [data_valid] tag. stdout: " << output.stdout;
    
    ASSERT_TRUE(output.stdout.find("Index out of bounds") != std::string::npos)
        << "stdout should contain 'Index out of bounds' message";
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
        VerifyTagFormat(tag);
    }
}

TEST_F(TestDevErrorLog, CaptureAllOutput_WorksCorrectly) {
    std::string testStdoutMsg = "This goes to stdout";
    std::string testStderrMsg = "This goes to stderr";
    
    CapturedOutput output = CaptureAllOutput([&]() {
        std::cout << testStdoutMsg << std::endl;
        std::cerr << testStderrMsg << std::endl;
    });
    
    ASSERT_TRUE(output.stdout.find(testStdoutMsg) != std::string::npos)
        << "stdout capture failed. Captured: " << output.stdout;
    ASSERT_TRUE(output.stderr.find(testStderrMsg) != std::string::npos)
        << "stderr capture failed. Captured: " << output.stderr;
}

} // namespace npu::tile_fwk
