/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_compiler_monitor.cpp
 * \brief
 */

#include <gtest/gtest.h>

#include "interface/inner/tilefwk.h"
#include "interface/interpreter/raw_tensor_data.h"
#include "interface/compiler_monitor/monitor_manager.h"
#include "interface/compiler_monitor/monitor_impl.h"
#include "interface/compiler_monitor/monitor_stage_scope.h"

namespace npu::tile_fwk {
class CompilerMonitor : public testing::Test {
public:
    static void TearDownTestCase() {}

    static void SetUpTestCase() {}

    void SetUp() override {}

    void TearDown() override {}
};

TEST_F(CompilerMonitor, CompilerMonitorInitial)
{
    MonitorManager::Instance().Initialize(true, 2, 4, 5);
    MonitorManager::Instance().SetStageTimeoutFlag("Pass");
    MonitorManager::Instance().GetStageTimeoutFlag("Pass");
    MonitorManager::Instance().GetStageStartTime();
    MonitorManager::Instance().GetStageElapsedTotals();
}

TEST_F(CompilerMonitor, CompilerMonitorImpl)
{
    MonitorImpl* impl_ = new MonitorImpl(&(MonitorManager::Instance()));
    MonitorManager::Instance().Initialize(true, 2, 4, 5);
    MonitorManager::Instance().SetTotalFunctionCount(5);
    MonitorManager::Instance().SetCurrentFunctionIndex(3);
    impl_->Start();
    impl_->StartMonitoring();
    sleep(10);
    impl_->StopMonitoring();
    impl_->Stop();

    delete impl_;
}

TEST_F(CompilerMonitor, CompilerMonitorTestPrint)
{
    MonitorImpl* impl_ = new MonitorImpl(&(MonitorManager::Instance()));
    MonitorManager::Instance().Initialize(true, 2, 4, 5);
    MonitorManager::Instance().SetTotalFunctionCount(5);
    MonitorManager::Instance().SetCurrentFunctionIndex(3);
    MonitorManager::Instance().StartStage("Pass");
    sleep(1);
    MonitorManager::Instance().EndStage("Pass");
    impl_->Start();
    impl_->StartMonitoring();
    sleep(1);
    MonitorManager::Instance().NotifyCompilationFinished();
    impl_->StopMonitoring();
    impl_->Stop();

    delete impl_;
}

TEST_F(CompilerMonitor, CompilerMonitorRootFuncBasic)
{
    MonitorManager::Instance().Initialize(true, 2, 4, 5);
    MonitorManager::Instance().SetTotalFunctionCount(3);
    MonitorManager::Instance().SetRootFuncCount(4);

    EXPECT_EQ(MonitorManager::Instance().GetRootFuncCount(), 4);
    EXPECT_EQ(MonitorManager::Instance().GetCurrentRootFuncIndex(), 0);
    EXPECT_EQ(MonitorManager::Instance().GetCurrentRootFuncName(), "");

    int idx1 = MonitorManager::Instance().PrepareNextRootFunc("func_A");
    EXPECT_EQ(idx1, 1);
    EXPECT_EQ(MonitorManager::Instance().GetCurrentRootFuncIndex(), 1);
    EXPECT_EQ(MonitorManager::Instance().GetCurrentRootFuncName(), "func_A");

    int idx2 = MonitorManager::Instance().PrepareNextRootFunc("func_B");
    EXPECT_EQ(idx2, 2);
    EXPECT_EQ(MonitorManager::Instance().GetCurrentRootFuncIndex(), 2);
    EXPECT_EQ(MonitorManager::Instance().GetCurrentRootFuncName(), "func_B");

    MonitorManager::Instance().NotifyCompilationFinished();
}

TEST_F(CompilerMonitor, CompilerMonitorFuncToBinStage)
{
    MonitorManager::Instance().Initialize(true, 2, 4, 5);
    MonitorManager::Instance().SetTotalFunctionCount(3);
    MonitorManager::Instance().SetRootFuncCount(2);
    MonitorManager::Instance().SetCurrentFunctionIndex(1);
    MonitorManager::Instance().SetCurrentFunctionName("leaf_func_1");

    int rootFuncIdx = MonitorManager::Instance().PrepareNextRootFunc("root_func_1");
    MonitorManager::Instance().StartStage(STAGE_FUNC_TO_BIN);
    sleep(1);
    MonitorManager::Instance().EndStage(STAGE_FUNC_TO_BIN, rootFuncIdx, "root_func_1");

    int rootFuncIdx2 = MonitorManager::Instance().PrepareNextRootFunc("root_func_2");
    MonitorManager::Instance().StartStage(STAGE_FUNC_TO_BIN);
    MonitorManager::Instance().EndStage(STAGE_FUNC_TO_BIN, rootFuncIdx2, "root_func_2");

    MonitorManager::Instance().NotifyCompilationFinished();
}

TEST_F(CompilerMonitor, CompilerMonitorStageScopeRootFunc)
{
    MonitorManager::Instance().Initialize(true, 2, 4, 5);
    MonitorManager::Instance().SetTotalFunctionCount(2);
    MonitorManager::Instance().SetRootFuncCount(2);

    {
        MonitorStageScope scope("Pass");
        sleep(1);
    }

    {
        MonitorStageScope scope(STAGE_FUNC_TO_BIN, 1, "root_func_A");
        sleep(1);
    }

    MonitorManager::Instance().NotifyCompilationFinished();
}

} // namespace npu::tile_fwk
