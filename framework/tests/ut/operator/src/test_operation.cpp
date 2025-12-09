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
 * \file test_operation.cpp
 * \brief
 */

#include "gtest/gtest.h"

#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/function/function.h"
#include "interface/operation/operation.h"
#include "passes/pass_utils/pass_utils.h"

using namespace npu::tile_fwk;

class OperationTest : public testing::Test {
public:
    static void TearDownTestCase() {}

    static void SetUpTestCase() {}

    void SetUp() override {
        Program::GetInstance().Reset();
        config::Reset();
    }

    void TearDown() override {}
};

TEST_F(OperationTest, TestOperationAttr) {
    std::vector<int64_t> shape{64, 64};
    FUNCTION("func_exp") {
        Tensor a(DT_FP32, shape, "a");
        a = Exp(a);
    }
    Function *func = Program::GetInstance().GetFunctionByRawName("func_exp");
    ASSERT_NE(func, nullptr);
    ASSERT_NE(func->Operations().size(), 0);
    Operation &oper = func->Operations().at(0);
    bool is_data = true;
    oper.SetAttribute("is_data_op", is_data);
    EXPECT_EQ(oper.HasAttr("is_data_op"), true);
    EXPECT_EQ(oper.HasAttr("data_op"), false);
    EXPECT_EQ(oper.GetBoolAttribute("is_data_op"), true);

    int input_index = 7;
    oper.SetAttribute("input_index", input_index);
    EXPECT_EQ(oper.HasAttr("input_index"), true);
    EXPECT_EQ(oper.GetIntAttribute("input_index"), input_index);

    std::string parent_op_type = "Sum";
    oper.SetAttribute("parent_op_type", parent_op_type);
    EXPECT_EQ(oper.HasAttr("parent_op_type"), true);
    EXPECT_EQ(oper.GetStringAttribute("parent_op_type"), parent_op_type);

    Json op_json = oper.DumpJson();
    std::cout << op_json << std::endl;
}