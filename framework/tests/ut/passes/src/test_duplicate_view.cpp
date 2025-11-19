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
 * \file test_duplicate_view.cpp
 * \brief Unit test for DuplicateView pass.
 */

#include "gtest/gtest.h"
#include "tilefwk/tilefwk_op.h"
#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "passes/pass_mgr/pass_manager.h"
#include "interface/configs/config_manager.h"
#include <fstream>
#include <vector>
#include <string>
#include "passes/tile_graph_pass/graph_optimization/duplicate_op.h"

using namespace npu::tile_fwk;

void PrintGraphInfoDuplicateView(Function* func) {
    std::cout << "func->Operations().size() = "  << func->Operations().size() << std::endl;
    for (auto &op : func->Operations()) {
        std::cout << "Op:" << op.GetOpMagic() << " " <<  op.GetOpcodeStr() << std::endl;
        std::cout << "input operation:";
        for (const std::shared_ptr<LogicalTensor> &input_tensor : op.GetIOperands()) {
            for (const auto &item_op : input_tensor->GetProducers()) {
                std::cout << "(" << item_op->opmagic << ", " << item_op->GetOpcodeStr() << ") ";
            }
        }
        std::cout << std::endl << "output operation:";
        for (const std::shared_ptr<LogicalTensor> &output_tensor : op.GetOOperands()) {
            for (const auto &item_op : output_tensor->GetConsumers()) {
                std::cout << "(" << item_op->opmagic << ", " << item_op->GetOpcodeStr() << ") ";
            }
        }
        std::cout << std::endl;
    }
}

class DuplicateViewTest : public testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);
        config::SetHostConfig(KEY_STRATEGY, "DuplicateViewTestStrategy");
        config::SetPlatformConfig("ENABLE_COST_MODEL", false);
    }
    void TearDown() override {}
};

TEST_F(DuplicateViewTest, TestThreeConsumersAfterView) {
    config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);

    //Define the shape of the Tensors
    std::vector<int64_t> shape1{2, 1, 64};
    std::vector<int64_t> shape2{2, 1, 1, 64};

    //Initialize PassManager
    PassManager &passManager = PassManager::Instance();
    passManager.RegisterStrategy("DuplicateViewTestStrategy", {
    {            "DuplicateOp",            "DuplicateOp"},
    });
    ConfigManager::Instance();

    //Create and configure the function
    Function* originFunction = nullptr;
    std::vector<int> originOpmagic;

    //Create Tensor
    Tensor in_tensor(DT_FP32, shape1, "in_tensor");
    Tensor out_tensor1(DT_FP32, shape2, "out_tensor1");
    Tensor out_tensor2(DT_FP32, shape2, "out_tensor1");
    Tensor out_tensor3(DT_FP32, shape2, "out_tensor1");

    config::SetBuildStatic(true);
    FUNCTION("DuplicateViewFunction") {
        out_tensor1 = Reshape(in_tensor, shape2);
        out_tensor2 = Reshape(in_tensor, shape2);
        out_tensor3 = Reshape(in_tensor, shape2);
        originFunction = Program::GetInstance().GetCurrentFunction();
        ASSERT_NE(originFunction, nullptr) << "当前函数指针为空";
        auto operations = originFunction->Operations();
        for (const auto &op : operations) {
            originOpmagic.emplace_back(op.opmagic);
        }
    }

    std::cout << Program::GetInstance().Dump() << std::endl;

    //Print and draw the graph
    Function* func = Program::GetInstance().GetFunctionByRawName("TENSOR_DuplicateViewFunction");
    PrintGraphInfoDuplicateView(func);
    // ================== Verify the effect of the Pass ==================
    auto updated_operations = func->Operations();
    int opSize = 9;

    EXPECT_EQ(updated_operations.size(), opSize) << "After the Pass, there should be 10 operations, two new VIEW be genearted";
    EXPECT_EQ(updated_operations[0].GetOpcode(), Opcode::OP_VIEW) << "The first operation should be VIEW";
    EXPECT_EQ(updated_operations[1].GetOpcode(), Opcode::OP_VIEW) << "The second operation should be newly generated VIEW";
    EXPECT_EQ(updated_operations[2].GetOpcode(), Opcode::OP_VIEW) << "The first operation should be newly generated VIEW";
}

TEST_F(DuplicateViewTest, TestOneConsumersAfterView) {
    config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);

    //Define the shape of the Tensors
    std::vector<int64_t> shape1{2, 1, 64};
    std::vector<int64_t> shape2{2, 1, 1, 64};

    //Initialize PassManager
    PassManager &passManager = PassManager::Instance();
    passManager.RegisterStrategy("DuplicateViewTestStrategy", {
    {            "DuplicateOp",            "DuplicateOp"},
    });
    ConfigManager::Instance();

    //Create and configure the function
    Function* originFunction = nullptr;
    std::vector<int> originOpmagic;

    //Create Tensor
    Tensor in_tensor(DT_FP32, shape1, "in_tensor");
    Tensor out_tensor1(DT_FP32, shape2, "out_tensor1");

    config::SetBuildStatic(true);
    FUNCTION("DuplicateViewFunction") {
        out_tensor1 = Reshape(in_tensor, shape2);

        originFunction = Program::GetInstance().GetCurrentFunction();
        ASSERT_NE(originFunction, nullptr) << "当前函数指针为空";
        auto operations = originFunction->Operations();
        for (const auto &op : operations) {
            originOpmagic.emplace_back(op.opmagic);
        }
    }

    std::cout << Program::GetInstance().Dump() << std::endl;

    //Print and draw the graph
    Function* func = Program::GetInstance().GetFunctionByRawName("TENSOR_DuplicateViewFunction");
    PrintGraphInfoDuplicateView(func);
    // ================== Verify the effect of the Pass ==================
    auto updated_operations = func->Operations();
    int opSize = 3;

    EXPECT_EQ(updated_operations.size(), opSize) << "After the Pass, there should be 3 operations, no new VIEW be genearted";
    EXPECT_EQ(updated_operations[0].GetOpcode(), Opcode::OP_VIEW) << "The first operation should be VIEW";
    EXPECT_NE(updated_operations[1].GetOpcode(), Opcode::OP_VIEW) << "The second operation should not be VIEW";
}
