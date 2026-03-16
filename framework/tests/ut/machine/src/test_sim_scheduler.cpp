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
 * \file test_sim_scheduler.cpp
 * \brief UTs for cost_model simulation Scheduler
 */

#include <gtest/gtest.h>

#include "cost_model/simulation/machine/Scheduler.h"

using namespace CostModel;

class TestSimScheduler : public testing::Test {
public:
    void SetUp() override {}
    void TearDown() override {}
};

// Verify that SortTile safely handles empty inputs (early-return branch).
TEST_F(TestSimScheduler, SortTile_EmptyTilesAndOps) {
    Scheduler scheduler;
    std::unordered_map<int, TilePtr> tiles;
    std::unordered_map<int, TileOpPtr> tileOps;
    std::vector<std::vector<int>> tileAllocSequence(4);

    // Should just return without crash or accessing sim.
    scheduler.SortTile(tiles, tileOps, tileAllocSequence);
}

