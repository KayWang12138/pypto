/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "test_machine_common.h"
#include "machine/device/dynamic/device_sche_context.h"

using namespace npu::tile_fwk;
using namespace npu::tile_fwk::dynamic;

struct FirstBatchSelfDispatchUT : UnitTestBase {};

TEST_F(FirstBatchSelfDispatchUT, FirstBatchOwnerInitUnclaimed)
{
    SchDeviceTaskContext ctx;
    ctx.Init();
    EXPECT_EQ(
        ctx.firstBatchOwner.load(std::memory_order_relaxed),
        static_cast<uint8_t>(FirstBatchDispatchOwnerState::FIRST_BATCH_UNCLAIMED));
}

TEST_F(FirstBatchSelfDispatchUT, RuntimeOptionEncodeToMachineConfig)
{
    config::SetRuntimeOption<uint8_t>(DEVICE_SCHED_MODE, static_cast<uint8_t>(MachineScheduleConfig::DEFAULT_SCH));
    config::SetRuntimeOption<bool>(ENABLE_AICORE_FIRST_BATCH_SELF_DISPATCH, true);
    TileShape::Current().SetVecTile(16, 16);

    Tensor a(DT_INT32, {16, 16}, "A");
    Tensor b(DT_INT32, {16, 16}, "B");
    Tensor o(DT_INT32, {16, 16}, "O");

    FUNCTION("first_batch_self_dispatch_ut", {a, b}, {o}) { o = Add(a, b); }

    auto* func = Program::GetInstance().GetLastFunction();
    ASSERT_NE(func, nullptr);
    EXPECT_NE(
        func->paramConfigs_.machineConfig_ &
            static_cast<uint8_t>(MachineScheduleConfig::AICORE_FIRST_BATCH_SELF_DISPATCH),
        0);
}

TEST_F(FirstBatchSelfDispatchUT, FirstBatchOwnerClaimByAicoreOnlyOnce)
{
    SchDeviceTaskContext ctx;
    ctx.Init();

    uint8_t expected0 = static_cast<uint8_t>(FirstBatchDispatchOwnerState::FIRST_BATCH_UNCLAIMED);
    bool firstClaim = ctx.firstBatchOwner.compare_exchange_strong(
        expected0, static_cast<uint8_t>(FirstBatchDispatchOwnerState::FIRST_BATCH_CLAIMED_BY_AICORE),
        std::memory_order_acq_rel, std::memory_order_acquire);

    uint8_t expected1 = static_cast<uint8_t>(FirstBatchDispatchOwnerState::FIRST_BATCH_UNCLAIMED);
    bool secondClaim = ctx.firstBatchOwner.compare_exchange_strong(
        expected1, static_cast<uint8_t>(FirstBatchDispatchOwnerState::FIRST_BATCH_CLAIMED_BY_SCHE),
        std::memory_order_acq_rel, std::memory_order_acquire);

    EXPECT_TRUE(firstClaim);
    EXPECT_FALSE(secondClaim);
    EXPECT_EQ(
        ctx.firstBatchOwner.load(std::memory_order_acquire),
        static_cast<uint8_t>(FirstBatchDispatchOwnerState::FIRST_BATCH_CLAIMED_BY_AICORE));
}

TEST_F(FirstBatchSelfDispatchUT, FirstBatchOwnerDoneByAicoreTransition)
{
    SchDeviceTaskContext ctx;
    ctx.Init();
    ctx.firstBatchOwner.store(
        static_cast<uint8_t>(FirstBatchDispatchOwnerState::FIRST_BATCH_CLAIMED_BY_AICORE), std::memory_order_release);

    uint8_t owner = ctx.firstBatchOwner.load(std::memory_order_acquire);
    if (owner == static_cast<uint8_t>(FirstBatchDispatchOwnerState::FIRST_BATCH_CLAIMED_BY_AICORE)) {
        ctx.firstBatchOwner.store(
            static_cast<uint8_t>(FirstBatchDispatchOwnerState::FIRST_BATCH_DONE_BY_AICORE), std::memory_order_release);
    }

    EXPECT_EQ(
        ctx.firstBatchOwner.load(std::memory_order_acquire),
        static_cast<uint8_t>(FirstBatchDispatchOwnerState::FIRST_BATCH_DONE_BY_AICORE));
}

TEST_F(FirstBatchSelfDispatchUT, FirstBatchOwnerFallbackClaimBySche)
{
    SchDeviceTaskContext ctx;
    ctx.Init();

    uint8_t expected = static_cast<uint8_t>(FirstBatchDispatchOwnerState::FIRST_BATCH_UNCLAIMED);
    bool claimBySche = ctx.firstBatchOwner.compare_exchange_strong(
        expected, static_cast<uint8_t>(FirstBatchDispatchOwnerState::FIRST_BATCH_CLAIMED_BY_SCHE),
        std::memory_order_acq_rel, std::memory_order_acquire);

    EXPECT_TRUE(claimBySche);
    EXPECT_EQ(
        ctx.firstBatchOwner.load(std::memory_order_acquire),
        static_cast<uint8_t>(FirstBatchDispatchOwnerState::FIRST_BATCH_CLAIMED_BY_SCHE));
}

TEST_F(FirstBatchSelfDispatchUT, FirstBatchOwnerDoneStateRejectsReClaim)
{
    SchDeviceTaskContext ctx;
    ctx.Init();
    ctx.firstBatchOwner.store(
        static_cast<uint8_t>(FirstBatchDispatchOwnerState::FIRST_BATCH_DONE_BY_AICORE), std::memory_order_release);

    uint8_t expected = static_cast<uint8_t>(FirstBatchDispatchOwnerState::FIRST_BATCH_UNCLAIMED);
    bool reClaim = ctx.firstBatchOwner.compare_exchange_strong(
        expected, static_cast<uint8_t>(FirstBatchDispatchOwnerState::FIRST_BATCH_CLAIMED_BY_SCHE),
        std::memory_order_acq_rel, std::memory_order_acquire);

    EXPECT_FALSE(reClaim);
    EXPECT_EQ(expected, static_cast<uint8_t>(FirstBatchDispatchOwnerState::FIRST_BATCH_DONE_BY_AICORE));
    EXPECT_EQ(
        ctx.firstBatchOwner.load(std::memory_order_acquire),
        static_cast<uint8_t>(FirstBatchDispatchOwnerState::FIRST_BATCH_DONE_BY_AICORE));
}

