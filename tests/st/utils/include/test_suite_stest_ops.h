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
 * \file test_suite_stest_ops.h
 * \brief Ops STest TestSuite.
 */

#pragma once

#include <gtest/gtest.h>
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "test_common.h"
#include "common/data_type.h"
#include "interface/interpreter/raw_tensor_data.h"

namespace npu::tile_fwk::stest {
template<typename TestBase>
class TestSuiteBase : public TestBase {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {
        // 使能 Aihac 后端
        oriEnableAihacBackend = config::GetPlatformConfig(KEY_ENABLE_AIHAC_BACKEND, oriEnableAihacBackend);
        config::SetPlatformConfig(KEY_ENABLE_AIHAC_BACKEND, true);
#ifdef ENABLE_TESTS_STEST_BINARY_CACHE
        // BinaryCache
        oriEnableBinaryCache = config::GetHostConfig(KEY_ENABLE_BINARY_CACHE, oriEnableBinaryCache);
        config::SetHostConfig(KEY_ENABLE_BINARY_CACHE, true);
#endif
#ifdef ENABLE_TESTS_STEST_DUMP_JSsON
        oriEnableDumpJson = config::GetPassConfig(KEY_PRINT_FUNCTION, oriEnableDumpJson);
        config::GetPassConfig(KEY_PRINT_FUNCTION, true);
#endif
        if (config::GetPlatformConfig("ENABLE_PV_DATA", false)) {
            CostModel::PvData::Instance().Enable();
        }
        if (config::GetPlatformConfig("ENABLE_SOFT_MEMORY", false)) {
            CostModel::SoftMemory::Instance().Enable();
        }
        // Reset Program

        Program::GetInstance().Reset();
        ProgramData::GetInstance().Reset();
    }

    void TearDown() override {
        config::SetPlatformConfig(KEY_ENABLE_AIHAC_BACKEND, oriEnableAihacBackend);
#ifdef ENABLE_TESTS_STEST_BINARY_CACHE
        config::SetHostConfig(KEY_ENABLE_BINARY_CACHE, oriEnableBinaryCache);
#endif
#ifdef ENABLE_TESTS_STEST_DUMO_JSON
        config::SetHostConfig(KEY_PRINT_FUNCTION, oriEnablePrintJson);
#endif
    }

protected:
    bool oriEnableAihacBackend = false;
#ifdef ENABLE_TESTS_STEST_BINARY_CACHE
    bool oriEnableBinaryCache = false;
#endif
#ifdef ENABLE_TESTS_STEST_DUMO_JSON
    bool oriEnableDumpJson = false;
#endif
};

template<typename T>
class TestSuite_STest_Ops_Aihac_param : public TestSuiteBase<testing::TestWithParam<T>> {};

class TestSuite_STest_Ops_Aihac : public TestSuiteBase<testing::Test> {};

} // namespace npu::tile_fwk::stest
