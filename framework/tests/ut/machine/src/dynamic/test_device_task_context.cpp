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
 * \file test_device_task_context.cpp
 * \brief
 */

#include <gtest/gtest.h>

#include "machine/device/dynamic/context/device_task_context.h"
#include "machine/utils/dynamic/device_task.h"
#include "machine/utils/dynamic/dev_encode_program.h"
#include "machine/utils/dynamic/dev_workspace.h"

using namespace npu::tile_fwk::dynamic;

class TestDeviceTaskContext : public testing::Test {
public:
    void SetUp() override {}
    void TearDown() override {}
};

// Cover InitReadyQueues path where coreFunctionCnt exceeds stitchFunctionsize.
TEST_F(TestDeviceTaskContext, InitReadyQueues_ExceedsStitchFunctionsize) {
    // Use heap allocation to avoid huge stack frame.
    auto devProg = std::make_unique<DevAscendProgram>();
    devProg->stitchFunctionsize = 1;

    auto workspace = std::make_unique<DeviceWorkspaceAllocator>();
    auto dyntask = std::make_unique<DynDeviceTask>(*workspace);
    dyntask->devTask.coreFunctionCnt = 2;  // > stitchFunctionsize, trigger error path

    npu::tile_fwk::ReadyCoreFunctionQueue* queues[READY_QUEUE_SIZE] = {nullptr};

    DeviceTaskContext ctx;
    int ret = ctx.InitReadyQueues(dyntask.get(), devProg.get(), queues);
    EXPECT_EQ(ret, DEVICE_MACHINE_ERROR);
}

