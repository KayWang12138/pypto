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
 * \file test_device_runner.cpp
 * \brief
 */

#include <regex>
#include <gtest/gtest.h>
#include <iostream>
#include <cstdlib>

#include <tracr/tracr.hpp>
#include "machine/device/aicore_manager.h"


class TestAiCoreManager : public testing::Test {
public:
    static void SetUpTestCase() {
    }

    static void TearDownTestCase() {}

    void SetUp() override {}

    void TearDown() override {}
};


TEST_F(TestAiCoreManager, test_tracr) {
    std::unique_ptr<npu::tile_fwk::AicpuTaskManager> aicpuTaskPtr = std::make_unique<npu::tile_fwk::AicpuTaskManager>();
    std::unique_ptr<npu::tile_fwk::AiCoreManager> AiCoreManagerPtr = std::make_unique<npu::tile_fwk::AiCoreManager>(*aicpuTaskPtr);

    const npu::tile_fwk::CoreType type = npu::tile_fwk::CoreType::AIC;
    AiCoreManagerPtr->WaitAllAicoreFinish(0, 0, type);

    const uint16_t tracrIdx = AiCoreManagerPtr->coreIdx2tracrIdx(0, type);
    (void)tracrIdx;

    INSTRUMENTATION_START("");

    INSTRUMENTATION_MARK_SET(tracrIdx, 0, 0);
    INSTRUMENTATION_MARK_RESET(tracrIdx);

    INSTRUMENTATION_END();
}