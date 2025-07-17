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
 * \file test_calendar_scheduler.cpp
 * \brief Unit test for calendar scheduler.
 */

#include <gtest/gtest.h>
#include <fstream>
#include <cstdlib>
#include "passes/execute_graph_pass/calendar_scheduling/calendar_scheduler.h"

class CalendarSchedulerTest : public testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {}

    void TearDown() override {}
};

TEST_F(CalendarSchedulerTest, TestFullWorkflowWithReassign) {
    std::vector<std::string> args = {
        "test_calendar_scheduler", "--reassign", "--weight", "0.5",
        "--swim", "./config/pass/json/tilefwk_prof_data.json",
        "--topo", "./config/pass/json/topo.json"
    };
    int result = runCalendarScheduler(args);

    EXPECT_EQ(result, 0);

    std::ifstream outputFile("startTimeWeight_0.5.json");
    EXPECT_TRUE(outputFile.good());
    outputFile.close();
}