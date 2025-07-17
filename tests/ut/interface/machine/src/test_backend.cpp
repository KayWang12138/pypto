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
 * \file test_backend.cpp
 * \brief
 */

#include "gtest/gtest.h"
#include "tilefwk/tilefwk.h"
#include "tilefwk.h"
#include "interface/program/program.h"

using namespace npu::tile_fwk;

class TestSuite_Backend : public testing::Test {};

TEST_F(TestSuite_Backend, AihacBackend_Err1) {
    config::SetPlatformConfig(KEY_ENABLE_AIHAC_BACKEND, true);
    try {
        Program::GetInstance().GetConfig().Reset();
    } catch (std::runtime_error&) {
    }
}

TEST_F(TestSuite_Backend, SimulationBackend_Err1) {
    config::SetPlatformConfig(KEY_ENABLE_AIHAC_BACKEND, false);
    config::SetPlatformConfig(KEY_ENABLE_COST_MODEL, true);
    try {
        Program::GetInstance().GetConfig().Reset();
    } catch (std::runtime_error&) {
    }
}
