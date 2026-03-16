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
 * \file test_device_context_stats.cpp
 * \brief
 */

#include "gtest/gtest.h"

#include "machine/device/dynamic/context/device_execute_context.h"
#include "machine/device/dynamic/context/device_task_context.h"
#include "interface/tensor/symbol_handler.h"

using namespace npu::tile_fwk::dynamic;

namespace {

// Minimal fake DevStartArgs to satisfy DeviceExecuteContext constructor.
struct FakeDevStartArgs : public DevStartArgs {
    FakeDevStartArgs()
    {
        devProg = nullptr;
        controlFlowEntry = nullptr;
    }
};

} // namespace

TEST(DeviceContextStatsTest, SymbolHandlerIdInvalidLogsAndReturnsNull)
{
    // Use an out-of-range value to trigger the default branch.
    auto invalidId = static_cast<npu::tile_fwk::SymbolHandlerId>(static_cast<uint64_t>(-1));
    void *handler = DeviceExecuteContext::SymbolHandlerIdToHandler(invalidId);

    // Expect nullptr when id is invalid (and DEV_ERROR/DEV_ASSERT have been executed).
    EXPECT_EQ(handler, nullptr);
}

TEST(DeviceContextStatsTest, DeviceTaskContextShowStatsDoesNotCrash)
{
    DeviceTaskContext ctx;

    // Bump the internal counters to non-zero so logs contain meaningful values.
    ctx.UpdateReadyTaskNum(5);

    // Just ensure ShowStats can be called safely; DEV_ERROR logs go to framework logger.
    ASSERT_NO_THROW(ctx.ShowStats());
}

