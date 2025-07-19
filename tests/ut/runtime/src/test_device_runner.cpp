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
 * \file test_device_runner.cpp
 * \brief
 */

#include <regex>
#include <gtest/gtest.h>
#include <iostream>
#include <cstdlib>
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "runtime/host/device_runner.h"
#include "runtime/utils/machine_ws_intf.h"
class TestDeviceRunner : public testing::Test {
public:
    static void SetUpTestCase() {
    }

    static void TearDownTestCase() {}

    void SetUp() override {}

    void TearDown() override {}
};

TEST_F(TestDeviceRunner, test_device_runner_get_task_time) {
    // auto runner = npu::tile_fwk::DeviceRunner::Get();
    npu::tile_fwk::DeviceRunner runner;
    std::uint64_t tastWastTime = 0;
    runner.args_.taskWastTime = (uint64_t)&tastWastTime;
    runner.GetTasksTime();
}

TEST_F(TestDeviceRunner, test_set_pmu_event) {
    // auto runner = npu::tile_fwk::DeviceRunner::Get();
    for (int i = 0; i < 9; i++) {
        setenv("PROF_PMU_EVENT_TYPE", std::to_string(i).c_str(), 1);
        npu::tile_fwk::DeviceRunner runner;
        runner.GetPmuEventType();
    }
}

TEST_F(TestDeviceRunner, test_ini_device_runner) {
    npu::tile_fwk::DeviceRunner runner;
    runner.Init();
}