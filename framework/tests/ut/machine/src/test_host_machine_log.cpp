/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_host_machine_log.cpp
 * \brief Unit tests for host_machine.cpp covering MACHINE_LOG calls at lines 195, 274, 377
 */

#include <gtest/gtest.h>
#include <sys/wait.h>
#include <unistd.h>
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_manager.h"
#include "tilefwk/tilefwk_log.h"

#define private public
#include "host_machine.h"
#undef private

using namespace npu::tile_fwk;

class TestHostMachineLog : public testing::Test {
public:
    void SetUp() override {
        auto &hm = HostMachine::GetInstance();
        // Ensure clean state: API mode, no threads, no pending task
        if (!hm.initialized_.load()) {
            hm.Init(HostMachineMode::API);
        }
        hm.curTask = nullptr;
    }

    void TearDown() override {
        auto &hm = HostMachine::GetInstance();
        // Clean up any MachineTask allocated during the test
        if (hm.curTask != nullptr) {
            delete hm.curTask;
            hm.curTask = nullptr;
        }
        hm.curTaskId_ = 0;
    }
};

// ===================================================================
// Covers Line 377: MACHINE_LOGE("Backend platform symbol GetPlatformInfo not found.")
// In UT environment, libtile_fwk_compiler.so is not loaded,
// so Backend::GetBackend().platform is nullptr
// ===================================================================
TEST_F(TestHostMachineLog, GetPlatformInfo_BackendNotFound) {
    auto &hm = HostMachine::GetInstance();
    std::string result = hm.GetPlatformInfo();
    // backend.platform is nullptr in UT -> line 377 MACHINE_LOGE is hit -> returns ""
    EXPECT_TRUE(result.empty());
}

// ===================================================================
// Covers Line 195: MACHINE_LOGW("CurTask is already running.")
// In API mode, calling SubTask when curTask is already set
// ===================================================================
TEST_F(TestHostMachineLog, SubTask_CurTaskAlreadyRunning) {
    auto &hm = HostMachine::GetInstance();
    hm.mode_ = HostMachineMode::API;

    // First call: sets curTask to a new MachineTask
    hm.SubTask(nullptr);
    EXPECT_NE(hm.curTask, nullptr);
    MachineTask *firstTask = hm.curTask;

    // Second call: curTask != nullptr -> line 195 MACHINE_LOGW is hit
    // Then a new MachineTask is created, overwriting curTask
    hm.SubTask(nullptr);
    EXPECT_NE(hm.curTask, nullptr);
    EXPECT_NE(hm.curTask, firstTask);

    // Clean up leaked first task
    delete firstTask;
    // curTask (second task) will be cleaned up in TearDown
}

// ===================================================================
// Covers Line 274: MACHINE_LOGW("Compile task is null.")
// Compile(nullptr) when curTask is also nullptr
// MACHINE_ASSERT is no-op (MACHINE_DEBUG not 1), so compileTask becomes nullptr
// and the code eventually crashes on nullptr dereference.
// Use manual fork to avoid EXPECT_DEATH macro (which triggers -Wswitch-default).
// ===================================================================
TEST_F(TestHostMachineLog, Compile_NullTaskWhenCurTaskNull) {
    pid_t pid = fork();
    if (pid == 0) {
        // Child process: trigger line 274 MACHINE_LOGW, then crash on nullptr deref
        auto &hm = HostMachine::GetInstance();
        hm.mode_ = HostMachineMode::API;
        hm.curTask = nullptr;
        hm.Compile(nullptr);
        _exit(0);  // Should not reach here
    }
    ASSERT_GT(pid, 0);
    int status = 0;
    waitpid(pid, &status, 0);
    // Expect child to crash (signal), not exit normally with 0
    EXPECT_TRUE(WIFSIGNALED(status) || (WIFEXITED(status) && WEXITSTATUS(status) != 0));
}
