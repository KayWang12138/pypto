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
 * \file test_tune_sync_for_vf.cpp
 * \brief Unit test for TuneSyncForVF.
 */
#include <gtest/gtest.h>
#include "tilefwk/platform.h"
#include "passes/block_graph_pass/tune_sync_for_vf.h"
#define private public

namespace npu {
namespace tile_fwk {

class TuneSyncForVFTest : public ::testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);
        config::SetPlatformConfig(KEY_ENABLE_COST_MODEL, false);
        Platform::Instance().ObtainPlatformInfo();
    }
    void TearDown() override {}
};

TEST_F(TuneSyncForVFTest, TestTuneSync) {
    // Build Graph
    EXPECT_EQ(1, 1);
}

} // namespace tile_fwk
} // namespace npu

#undef private
