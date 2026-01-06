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
 * \file test_suite_utest.h
 * \brief UTest TestSuite.
 */

#pragma once

#include <gtest/gtest.h>

namespace npu::tile_fwk::utest {

template<typename TestBase>
class UTestSuiteBase : public TestBase {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {}

    void TearDown() override {}
};

template<typename T>
class UTestSuite_WithParam : public UTestSuiteBase<testing::TestWithParam<T>> {};

class UTestSuite : public UTestSuiteBase<testing::Test> {};

} // namespace npu::tile_fwk::utest
