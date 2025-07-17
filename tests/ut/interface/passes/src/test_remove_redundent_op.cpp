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
 * \file test_remove_redundent_op.cpp
 * \brief Unit test for RemoveRedundentOp pass.
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
#include "passes/tile_graph_pass/remove_redundent_op.h"
#include <fstream>
#include <vector>
#include <string>

using namespace npu::tile_fwk;

void PrintGraphInfoRemoveRedundentOp(Function* func) {
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

class RemoveRedundentOpTest : public testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {
        Program::GetInstance().Reset();
        Program::GetInstance().GetConfig().Reset();
        config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);
        config::SetHostConfig(KEY_STRATEGY, "RemoveRedundentOpTestStrategy");
        config::SetPlatformConfig("ENABLE_COST_MODEL", false);
    }
    void TearDown() override {}
};

TEST_F(RemoveRedundentOpTest, TestInShapeEqualOutShape) {
    config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);

    std::vector<int> shape1{2, 1};
    std::vector<int> shape2{1, 2};

    PassManager &passManager = PassManager::Instance();
    passManager.RegisterStrategy("RemoveRedundentOpTestStrategy", {
    {         "AssignMemoryType",         "AssignMemoryType",    PassType::TYPE_TILE_GRAPH},
    {          "InsertConvertOp",          "InsertConvertOp",    PassType::TYPE_TILE_GRAPH},
    {   "SplitLargeFanoutTensor",   "SplitLargeFanoutTensor",    PassType::TYPE_TILE_GRAPH},

    });
    ConfigManager::Instance();

    Function* originFunction = nullptr;
    std::vector<int> originOpmagic;

    Tensor in_tensor(DT_FP32, shape1, "in_tensor");
    Tensor out_tensor(DT_FP32, shape2, "out_tensor");
    Tensor in_tensor1(DT_FP32, shape1, "in_tensor1");
    Tensor out_tensor1(DT_FP32, shape2, "out_tensor1");

    FUNCTION("RemoveRedundentOpFunction") {
        auto a = View(in_tensor, shape1, {0,0});
        auto b = View(in_tensor1, shape1, {0,0});
        out_tensor = Reshape(a, shape2);
        out_tensor1 = Reshape(b, shape2);
        originFunction = Program::GetInstance().GetCurrentFunction();
        ASSERT_NE(originFunction, nullptr) << "当前函数指针为空";
        auto operations = originFunction->Operations();
        for (const auto &op : operations) {
            originOpmagic.emplace_back(op.opmagic);
        }
    }
    std::string jsonFilePath = "./config/pass/json/remove_redundent_op_inshape_equal_outshape.json";
    bool dumpJsonFlag = true;
    if (dumpJsonFlag) {
        auto programJson = Program::GetInstance().DumpJson();
        DumpJsonFile(programJson, jsonFilePath);
    }
    Json readData = LoadJsonFile(jsonFilePath);
    Program::GetInstance().LoadJson(readData);

    // Call the pass
    Function* func = Program::GetInstance().GetFunctionByRawName("TENSOR_RemoveRedundentOpFunction");
    npu::tile_fwk::RemoveRedundentOp removeRedundentOp;
    removeRedundentOp.PreCheck(*func);
    removeRedundentOp.RunOnFunction(*func);
    removeRedundentOp.PostCheck(*func);

    PrintGraphInfoRemoveRedundentOp(func);
    // ================== Verify the effect of the Pass ==================
    auto updated_operations = func->Operations();
    int opSize = 2;
    // Verify if the RemoveRedundentReshape Pass removes redundant Reshape operation
    EXPECT_EQ(updated_operations.size(), opSize) << "After the Pass, there should be 3 operations, one VIEW be deleted";
    EXPECT_NE(updated_operations[0].GetOpcode(), Opcode::OP_VIEW) << "The first operation should not be VIEW";
    EXPECT_NE(updated_operations[1].GetOpcode(), Opcode::OP_VIEW) << "The second operation should not be VIEW";
}

TEST_F(RemoveRedundentOpTest, TestInShapeNotEqualOutShape) {
    config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);

    std::vector<int> shape1{100,100};
    std::vector<int> shape2{10, 1000};

    PassManager &passManager = PassManager::Instance();
    passManager.RegisterStrategy("RemoveRedundentOpTestStrategy", {
    {         "AssignMemoryType",         "AssignMemoryType",    PassType::TYPE_TILE_GRAPH},
    {          "InsertConvertOp",          "InsertConvertOp",    PassType::TYPE_TILE_GRAPH},
    {   "SplitLargeFanoutTensor",   "SplitLargeFanoutTensor",    PassType::TYPE_TILE_GRAPH},

    });
    ConfigManager::Instance();

    Function* originFunction = nullptr;
    std::vector<int> originOpmagic;

    Tensor in_tensor(DT_FP32, shape1, "in_tensor");
    Tensor out_tensor(DT_FP32, shape2, "out_tensor");
    Tensor in_tensor1(DT_FP32, shape1, "in_tensor1");
    Tensor out_tensor1(DT_FP32, {8,800}, "out_tensor1");

    FUNCTION("RemoveRedundentOpFunction") {
        auto a = View(in_tensor, shape1, {0,0});
        auto b = View(in_tensor1, {80,80}, {10,10});
        out_tensor = Reshape(a, shape2);
        out_tensor1 = Reshape(b, {8,800});
        originFunction = Program::GetInstance().GetCurrentFunction();
        ASSERT_NE(originFunction, nullptr) << "当前函数指针为空";
        auto operations = originFunction->Operations();
        for (const auto &op : operations) {
            originOpmagic.emplace_back(op.opmagic);
        }
    }
    std::string jsonFilePath = "./config/pass/json/remove_redundent_op_inshape_not_equal_outshape.json";
    bool dumpJsonFlag = true;
    if (dumpJsonFlag) {
        auto programJson = Program::GetInstance().DumpJson();
        DumpJsonFile(programJson, jsonFilePath);
    }
    Json readData = LoadJsonFile(jsonFilePath);
    Program::GetInstance().LoadJson(readData);

    // Call the pass
    Function* func = Program::GetInstance().GetFunctionByRawName("TENSOR_RemoveRedundentOpFunction");
    npu::tile_fwk::RemoveRedundentOp removeRedundentOp;
    removeRedundentOp.PreCheck(*func);
    removeRedundentOp.RunOnFunction(*func);
    removeRedundentOp.PostCheck(*func);
    PrintGraphInfoRemoveRedundentOp(func);
    // ================== Verify the effect of the Pass ==================
    auto updated_operations = func->Operations();
    int opSize = 4;
    // Verify if the RemoveRedundentReshape Pass removes redundant Reshape operation
    EXPECT_EQ(updated_operations.size(), opSize) << "After the Pass, there should be 3 operations, no VIEW be deleted";
    EXPECT_EQ(updated_operations[0].GetOpcode(), Opcode::OP_VIEW) << "The first operation should be VIEW";
    EXPECT_NE(updated_operations[1].GetOpcode(), Opcode::OP_VIEW) << "The second operation should not be VIEW";
}

TEST_F(RemoveRedundentOpTest, TestIntermediateOutcast) {
    config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);
    int bs = 1;
    int n = 32;
    int d = 128;
    std::vector<int> shape{bs, n, d};
    std::vector<int> resShape{bs, n, d};
    PassManager &passManager = PassManager::Instance();
    passManager.RegisterStrategy("RemoveRedundentOpTestStrategy", {
    {   "RemoveRedundentReshape",   "RemoveRedundentReshape",  PassType::TYPE_TENSOR_GRAPH},
    {           "ExpandFunction",           "ExpandFunction",  PassType::TYPE_TENSOR_GRAPH},
    {            "DuplicateView",            "DuplicateView",    PassType::TYPE_TILE_GRAPH},
    {        "MergeViewAssemble",        "MergeViewAssemble",    PassType::TYPE_TILE_GRAPH},
    {         "AssignMemoryType",         "AssignMemoryType",    PassType::TYPE_TILE_GRAPH},
    {       "InsertConvertOp_01",          "InsertConvertOp",    PassType::TYPE_TILE_GRAPH},
    {   "SplitLargeFanoutTensor",   "SplitLargeFanoutTensor",    PassType::TYPE_TILE_GRAPH},
    {       "SplitReshapeOpPVC2",       "SplitReshapeOpPVC2",    PassType::TYPE_TILE_GRAPH},
    });
    ConfigManager::Instance();

    Tensor input(DataType::DT_FP32, shape, "input");
    Tensor output(DataType::DT_FP32, resShape, "res");
    Tensor output_add(DataType::DT_FP32, resShape, "res_add");
    FUNCTION("RemoveRedundentOpFunction", FunctionType::STATIC, {input, output, output_add}) {
        Program::GetInstance().GetTileShape().SetVecTileShapes(1, 32, 128);
        output = Transpose(input, {0, 1});
        Program::GetInstance().GetTileShape().SetVecTileShapes(8, 1, 128);
        output_add = AddS(output, Element(DataType::DT_FP32, 0.0));
    }

    Function* func = Program::GetInstance().GetFunctionByRawName("TENSOR_RemoveRedundentOpFunction");
    npu::tile_fwk::RemoveRedundentOp removeRedundentOp;
    auto oriOpList = func->Operations(true);
    EXPECT_EQ(oriOpList.size(), 15) << "Before the Pass, there should be 15 operations";
    int ori_view_count = 0;
    int ori_assemble_count = 0;
    for (auto &op : oriOpList) {
        if (op.GetOpcode() == Opcode::OP_VIEW) {
            ori_view_count += 1;
        } else if (op.GetOpcode() == Opcode::OP_ASSEMBLE) {
            ori_assemble_count += 1;
        }
    }
    EXPECT_EQ(ori_view_count, 5) << "There shoule be 5 VIEW op before RemoveRedundentOp";
    EXPECT_EQ(ori_assemble_count, 5) << "There shoule be 5 ASSEMBLE op before RemoveRedundentOp";
    removeRedundentOp.PreCheck(*func);
    removeRedundentOp.RunOnFunction(*func);
    removeRedundentOp.PostCheck(*func);
    PrintGraphInfoRemoveRedundentOp(func);
    // ================== Verify the effect of the Pass ==================
    auto updated_operations = func->Operations(true);
    int opSize = 14;
    EXPECT_EQ(updated_operations.size(), opSize) << "After the Pass, there should be 14 operations, no VIEW be deleted";
    EXPECT_EQ(updated_operations[0].GetOpcode(), Opcode::OP_VIEW) << "The first operation should be VIEW";
    int view_count = 0;
    int assemble_count = 0;

    for (auto &op : updated_operations) {
        if (op.GetOpcode() == Opcode::OP_VIEW) {
            view_count += 1;
        } else if (op.GetOpcode() == Opcode::OP_ASSEMBLE) {
            assemble_count += 1;
        }
    }
    EXPECT_EQ(view_count, 5) << "There shoule be 5 ASSEMBLE op after RemoveRedundentOp";
    EXPECT_EQ(assemble_count, 4) << "There shoule be 5 ASSEMBLE op after RemoveRedundentOp";
}

TEST_F(RemoveRedundentOpTest, TestInternalAssembleView) {
    config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);
    int bs = 4;
    int n = 32;
    int d = 128;
    std::vector<int> shape{bs, n, d};
    std::vector<int> resShape{bs, n, d};
    PassManager &passManager = PassManager::Instance();
    passManager.RegisterStrategy("RemoveRedundentOpTestStrategy", {
    {   "RemoveRedundentReshape",   "RemoveRedundentReshape",  PassType::TYPE_TENSOR_GRAPH},
    {           "ExpandFunction",           "ExpandFunction",  PassType::TYPE_TENSOR_GRAPH},
    {            "DuplicateView",            "DuplicateView",    PassType::TYPE_TILE_GRAPH},
    {        "MergeViewAssemble",        "MergeViewAssemble",    PassType::TYPE_TILE_GRAPH},
    {         "AssignMemoryType",         "AssignMemoryType",    PassType::TYPE_TILE_GRAPH},
    {       "InsertConvertOp_01",          "InsertConvertOp",    PassType::TYPE_TILE_GRAPH},
    {   "SplitLargeFanoutTensor",   "SplitLargeFanoutTensor",    PassType::TYPE_TILE_GRAPH},
    {       "SplitReshapeOpPVC2",       "SplitReshapeOpPVC2",    PassType::TYPE_TILE_GRAPH},
    });
    ConfigManager::Instance();

    Tensor input(DataType::DT_FP32, shape, "input");
    Tensor output(DataType::DT_FP32, resShape, "res");
    FUNCTION("RemoveRedundentOpFunction", FunctionType::STATIC, {input, output}) {
        Program::GetInstance().GetTileShape().SetVecTileShapes(1, 32, 128);
        auto tmp = Transpose(input, {0, 1}); // [32, 4, 128]
        Program::GetInstance().GetTileShape().SetVecTileShapes(8, 1, 64);
        output = AddS(tmp, Element(DataType::DT_FP32, 3.0));
    }

    Function* func = Program::GetInstance().GetFunctionByRawName("TENSOR_RemoveRedundentOpFunction");
    npu::tile_fwk::RemoveRedundentOp removeRedundentOp;
    auto oriOpList = func->Operations(true);
    int ori_view_count = 0;
    int ori_assemble_count = 0;
    for (auto &op : oriOpList) {
        if (op.GetOpcode() == Opcode::OP_VIEW) {
            ori_view_count += 1;
        } else if (op.GetOpcode() == Opcode::OP_ASSEMBLE) {
            ori_assemble_count += 1;
        }
    }
    removeRedundentOp.PreCheck(*func);
    removeRedundentOp.RunOnFunction(*func);
    removeRedundentOp.PostCheck(*func);
    PrintGraphInfoRemoveRedundentOp(func);
    // ================== Verify the effect of the Pass ==================
    auto updated_operations = func->Operations(true);
    int view_count = 0;
    int assemble_count = 0;

    for (auto &op : updated_operations) {
        if (op.GetOpcode() == Opcode::OP_VIEW) {
            view_count += 1;
        } else if (op.GetOpcode() == Opcode::OP_ASSEMBLE) {
            assemble_count += 1;
        }
    }
    EXPECT_EQ(updated_operations.size(), oriOpList.size()) << "No op should be removed in RemoveRedundentOp";
    EXPECT_EQ(view_count, ori_view_count) << "No VIEW op should be removed in RemoveRedundentOp";
    EXPECT_EQ(assemble_count, ori_assemble_count) << "No ASSEMBLE op should be removed in RemoveRedundentOp";
}
