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
 * \file test_execute_calc.cpp
 * \brief
 */

#include <gtest/gtest.h>

#include "interface/utils/log.h"
#include "interface/interpreter/raw_tensor_data.h"
#include "interface/tensor/float.h"
#include "interface/interpreter/function.h"
#include "interface/interpreter/operation.h"
#include "interface/inner/tilefwk.h"

namespace npu::tile_fwk {
class CalcCommonTest : public testing::Test {
public:
    static void TearDownTestCase() {}

    static void SetUpTestCase() {}

    void SetUp() override {
        Program::GetInstance().Reset();
        config::Reset();
    }

    void TearDown() override {}
};

//測試帶有髒數據的Reshape操作
} // namespace npu::tile_fwk
