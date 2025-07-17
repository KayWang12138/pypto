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
 * \file test_graph_init.cpp
 * \brief Unit test for GraphInit pass.
 */

#include "gtest/gtest.h"
#include "operation/tilefwk_op.h"
#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"
#include "passes/pass_manager.h"
#include "interface/configs/config_manager.h"
#include "ut_json/ut_json_tool.h"
#include "passes/tile_graph_pass/graph_init.h"
#include <fstream>
#include <vector>
#include <string>

using namespace npu::tile_fwk;

void PrintGraphInfoGraphInit(Function* func) {
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

class GraphInitTest : public testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {
        Program::GetInstance().Reset();
        Program::GetInstance().GetConfig().Reset();
        config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);
        config::SetHostConfig(KEY_STRATEGY, "GraphInitTestStrategy");
        config::SetPlatformConfig("ENABLE_COST_MODEL", false);
    }
    void TearDown() override {}
};

TEST_F(GraphInitTest, TestIsCubeOrNot) {
    config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);

        //Define the shape of the Tensors
    std::vector<int> shape1{2, 1, 64};
    std::vector<int> shape2{2, 1, 1, 64};
    std::vector<int> shape3{1, 2, 1, 64};
    std::vector<int> shape4{1, 2, 64};
    std::vector<int> shape5{2, 64};
    std::vector<int> shape6{64, 2};
    std::vector<int> shape7{2, 2};

    //Initialize PassManager
    PassManager &passManager = PassManager::Instance();
    passManager.RegisterStrategy("GraphInitTestStrategy", {
    {             "GraphInitPass",           "GraphInitPass",    PassType::TYPE_TILE_GRAPH},
    });
    ConfigManager::Instance();

    //Create and configure the function
    Function* originFunction = nullptr;
    std::vector<int> originOpmagic;

    //Create Tensor
    Tensor in_tensor(DT_FP32, shape1, "in_tensor");
    Tensor out_tensor(DT_FP16, shape7, "in_tensor");

    FUNCTION("PartitionVCFunction") {
        auto out_tensor_1_A = Reshape(in_tensor, shape2);
        auto out_tensor_1_B = Reshape(in_tensor, shape2);
        auto out_tensor_2_A = Transpose(out_tensor_1_A, {0, 1});
        auto out_tensor_2_B = Transpose(out_tensor_1_B, {0, 1});
        auto out_tensor_3_A = Reshape(out_tensor_2_A, shape1);
        auto out_tensor_3_B = Reshape(out_tensor_2_B, shape4);
        auto out_tensor_4_A = Cast(out_tensor_3_A, DT_FP16);
        auto out_tensor_4_B = Cast(out_tensor_3_B, DT_FP16);
        auto out_tensor_5_A = Reshape(out_tensor_4_A, shape5);
        auto out_tensor_5_B = Reshape(out_tensor_4_A, shape6);
        out_tensor = npu::tile_fwk::Matrix::Matmul<false, false>(DataType::DT_FP32, out_tensor_5_A, out_tensor_5_B);

        originFunction = Program::GetInstance().GetCurrentFunction();
        ASSERT_NE(originFunction, nullptr) << "当前函数指针为空";
        auto operations = originFunction->Operations();
        for (const auto &op : operations) {
            originOpmagic.emplace_back(op.opmagic);
        }
    }
    std::string jsonFilePath = "./config/pass/json/graph_init_is_cube_or_not.json";
    bool dumpJsonFlag = false;
    if (dumpJsonFlag) {
        auto programJson = Program::GetInstance().DumpJson();
        DumpJsonFile(programJson, jsonFilePath);
    }
    // Json readData = LoadJsonFile(jsonFilePath);
    // Program::GetInstance().LoadJson(readData);


    Function* func = Program::GetInstance().GetFunctionByRawName("TENSOR_PartitionVCFunction");
    Program testProgram(HostMachineMode::SERVER);
    GraphInitPass graphInitPass;
    graphInitPass.PreCheck(*func);
    graphInitPass.RunOnFunction(*func);
    graphInitPass.PostCheck(*func);

    PrintGraphInfoGraphInit(func);

    // ================== Verify the effect of the Pass ==================
    auto updated_operations = func->Operations();
    int opSize = 17;
    // Verify if the RemoveRedundentReshape Pass removes redundant Reshape operation
    EXPECT_EQ(updated_operations.size(), opSize) << "After the Pass, there should be 13 operations";
    EXPECT_EQ(updated_operations[13].GetOpcode(), Opcode::OP_A_MUL_B) << "The 11th operation should be A_MUL_B";
    EXPECT_EQ(updated_operations[13].GetBoolAttribute(OpAttributeKey::isCube), true) << "Operaton A_MUL_B is attribute shoule be true";
    EXPECT_EQ(updated_operations[0].GetOpcode(), Opcode::OP_VIEW) << "The first operation should be VIEW";
    EXPECT_EQ(updated_operations[0].GetBoolAttribute(OpAttributeKey::isCube), false) << "Operaton VIEW is attribute shoule be false";
    EXPECT_EQ(updated_operations[1].GetOpcode(), Opcode::OP_RESHAPE) << "The second operation should be RESHAPE";
    EXPECT_EQ(updated_operations[1].GetBoolAttribute(OpAttributeKey::isCube), false) << "Operaton RESHAPE is attribute shoule be false";
}
