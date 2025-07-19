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
 * \file test_insert_convert_op.cpp
 * \brief Unit test for InsertConvertOpPass pass.
 */

#include <gtest/gtest.h>
#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "passes/pass_manager.h"
#include "interface/configs/config_manager.h"
#include "ut_json/ut_json_tool.h"
#include "passes/tile_graph_pass/insert_convert_op.h"
#include <vector>

using namespace npu::tile_fwk;

class InsertConvertOpPassTest : public ::testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {
        Program::GetInstance().Reset();
        Program::GetInstance().GetConfig().Reset();
        config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);
        config::SetHostConfig(KEY_STRATEGY, "InsertConvertOpPassTestStrategy");
        config::SetPlatformConfig("ENABLE_COST_MODEL", false);
    }
    void TearDown() override {}
};

TEST_F(InsertConvertOpPassTest, L1ToUB) {
    PROGRAM("InsertConvertOpPassTest") {
        std::vector<int> shape1{256, 256};
        std::vector<int> shape2{128, 128};
        Program::GetInstance().GetTileShape().SetVecTileShapes({128, 128});
        Tensor input_a(DT_FP32, shape1, "input_a");
        Tensor input_b(DT_FP32, shape1, "input_b");
        Tensor output(DT_FP32, shape2, "output");
        PassManager &passManager = PassManager::Instance();
        passManager.RegisterStrategy("InsertConvertOpPassTestStrategy", {
            {   "RemoveRedundentReshape",   "RemoveRedundentReshape",  PassType::TYPE_TENSOR_GRAPH},
            {           "ExpandFunction",           "ExpandFunction",  PassType::TYPE_TENSOR_GRAPH},
            {            "DuplicateView",            "DuplicateView",    PassType::TYPE_TILE_GRAPH},
            {        "MergeViewAssemble",        "MergeViewAssemble",    PassType::TYPE_TILE_GRAPH},
            {         "AssignMemoryType",         "AssignMemoryType",    PassType::TYPE_TILE_GRAPH},

        });
        ConfigManager::Instance();

        Function* originFunction = nullptr;
        std::vector<int> originOpmagic;
        FUNCTION("ADD", FunctionType::STATIC, {input_a, input_b, output}) {
            config::SetPassStrategy("InsertConvertOpPassTestStrategy");

            auto tmp_a_0 = View(input_a, shape2, {0,0});
            tmp_a_0->SetMemoryTypeBoth(MEM_L1, true);
            auto tmp_b_1 = View(input_b, shape2, {0,0});
            tmp_b_1->SetMemoryTypeBoth(MEM_L1, true);

            output = Add(tmp_a_0, tmp_b_1);
        }
        originFunction = Program::GetInstance().GetFunctionByRawName("TENSOR_ADD");;
        ASSERT_NE(originFunction, nullptr) << "当前函数指针为空";
        auto operations = originFunction->Operations();
        for (const auto &op : operations) {
            originOpmagic.emplace_back(op.opmagic);
        }

        std::string jsonFilePath = "./config/pass/json/insert_convert_op_l1_to_ub.json";
        bool dumpJsonFlag = true;
        if (dumpJsonFlag) {
            auto programJson = Program::GetInstance().DumpJson();
            DumpJsonFile(programJson, jsonFilePath);
        }
        Json readData = LoadJsonFile(jsonFilePath);
        Program::GetInstance().LoadJson(readData);

        Function* func = Program::GetInstance().GetCurrentFunction();

        Program testProgram(HostMachineMode::SERVER);
        npu::tile_fwk::InsertConvertOp insertConvertOp;
        insertConvertOp.PreCheck(*func);
        insertConvertOp.RunOnFunction(*func);
        insertConvertOp.PostCheck(*func);

        // ================== Verify Pass Effect ==================
        auto updatedOperations = Program::GetInstance().GetFunctionByRawName("TENSOR_ADD")->Operations();
        constexpr int expectedOperationsSize = 8;
        EXPECT_EQ(updatedOperations.size(), expectedOperationsSize) << "8 operations should remain View + Convert + Add + Assemble";
        int convertNum = 0;
        for (const auto &updatedOperation : updatedOperations) {
            if(updatedOperation.GetOpcode() == Opcode::OP_CONVERT)
                convertNum++;
        }
        constexpr int expextedConvertNum = 4;
        EXPECT_EQ(convertNum, expextedConvertNum) << "4 operations should be Convert";
    }
}

TEST_F(InsertConvertOpPassTest, UBToDDR) {
    PROGRAM("InsertConvertOpPassTest") {
        std::vector<int> shape1{256, 256};
        std::vector<int> shape2{128, 128};
        Program::GetInstance().GetTileShape().SetVecTileShapes({128, 128});
        Tensor input_a(DT_FP32, shape1, "input_a");
        Tensor input_b(DT_FP32, shape1, "input_b");
        Tensor output(DT_FP32, shape2, "output");
        PassManager &passManager = PassManager::Instance();
        passManager.RegisterStrategy("InsertConvertOpPassTestStrategy", {
            {   "RemoveRedundentReshape",   "RemoveRedundentReshape",  PassType::TYPE_TENSOR_GRAPH},
            {           "ExpandFunction",           "ExpandFunction",  PassType::TYPE_TENSOR_GRAPH},
            {            "DuplicateView",            "DuplicateView",    PassType::TYPE_TILE_GRAPH},
            {        "MergeViewAssemble",        "MergeViewAssemble",    PassType::TYPE_TILE_GRAPH},
            {         "AssignMemoryType",         "AssignMemoryType",    PassType::TYPE_TILE_GRAPH},

        });
        ConfigManager::Instance();

        Function* originFunction = nullptr;
        std::vector<int> originOpmagic;
        FUNCTION("ADD", FunctionType::STATIC, {input_a, input_b, output}) {
            config::SetPassStrategy("InsertConvertOpPassTestStrategy");

            auto tmp_a_0 = View(input_a, shape2, {0,0});
            tmp_a_0->SetMemoryTypeBoth(MEM_DEVICE_DDR, true);
            auto tmp_b_1 = View(input_b, shape2, {0,0});

            output = Add(tmp_a_0, tmp_b_1);
        }
        originFunction = Program::GetInstance().GetFunctionByRawName("TENSOR_ADD");
        ASSERT_NE(originFunction, nullptr) << "当前函数指针为空";
        auto operations = originFunction->Operations();
        for (const auto &op : operations) {
            originOpmagic.emplace_back(op.opmagic);
        }
        std::string jsonFilePathBefore = "./config/pass/json/before_insert_convert_op_ub_to_ddr.json";
        DumpJsonFile(Program::GetInstance().DumpJson(), jsonFilePathBefore);
        Json readData = LoadJsonFile(jsonFilePathBefore);
        Program::GetInstance().LoadJson(readData);

        Function* func = Program::GetInstance().GetFunctionByRawName("TENSOR_ADD");

        Program testProgram(HostMachineMode::SERVER);
        npu::tile_fwk::InsertConvertOp insertConvertOp;
        insertConvertOp.PreCheck(*func);
        insertConvertOp.RunOnFunction(*func);
        insertConvertOp.PostCheck(*func);

        std::string jsonFilePathAfter = "./config/pass/json/after_insert_convert_op_ub_to_ddr.json";
        DumpJsonFile(Program::GetInstance().DumpJson(), jsonFilePathAfter);
        // ================== Verify Pass Effect ==================
        auto updatedOperations = func->Operations();
        constexpr int expectedOperationsSize = 5;
        EXPECT_EQ(updatedOperations.size(), expectedOperationsSize) << "5 operations should remain View + Convert + Add + Assemble";
        int convertNum = 0;
        for (const auto &updatedOperation : updatedOperations) {
            if(updatedOperation.GetOpcode() == Opcode::OP_CONVERT)
                convertNum++;
        }
        constexpr int expextedConvertNum = 1;
        EXPECT_EQ(convertNum, expextedConvertNum) << "1 operation should be Convert";
    }
}

TEST_F(InsertConvertOpPassTest, L1ToDDR) {
    PROGRAM("InsertConvertOpPassTest") {
        std::vector<int> shape1{256, 256};
        std::vector<int> shape2{128, 128};
        Program::GetInstance().GetTileShape().SetVecTileShapes({128, 128});
        Program::GetInstance().GetTileShape().SetCubeTileShapes({128, 128}, {128, 128}, {128, 128});
        Tensor input_a(DT_FP32, shape1, "input_a");
        Tensor input_b(DT_FP32, shape1, "input_b");
        Tensor output(DT_FP32, shape2, "output");

        PassManager &passManager = PassManager::Instance();
        passManager.RegisterStrategy("InsertConvertOpPassTestStrategy", {
            {   "RemoveRedundentReshape",   "RemoveRedundentReshape",  PassType::TYPE_TENSOR_GRAPH},
            {           "ExpandFunction",           "ExpandFunction",  PassType::TYPE_TENSOR_GRAPH},
            {            "DuplicateView",            "DuplicateView",    PassType::TYPE_TILE_GRAPH},
            {        "MergeViewAssemble",        "MergeViewAssemble",    PassType::TYPE_TILE_GRAPH},
            {         "AssignMemoryType",         "AssignMemoryType",    PassType::TYPE_TILE_GRAPH},

        });
        ConfigManager::Instance();

        Function* originFunction = nullptr;
        std::vector<int> originOpmagic;
        FUNCTION("ADD", FunctionType::STATIC, {input_a, input_b, output}) {
            config::SetPassStrategy("InsertConvertOpPassTestStrategy");

            auto tmp_a_0 = View(input_a, shape2, {0,0});
            tmp_a_0->SetMemoryTypeBoth(MEM_L1, true);
            auto tmp_b_1 = View(input_b, shape2, {0,0});

            Tensor tmp_output(DT_FP32, shape2, "C");
            tmp_output->SetMemoryTypeBoth(MEM_DEVICE_DDR, true);
            tmp_output = View(tmp_a_0, shape2, {0,0});
            tmp_output->SetMemoryTypeBoth(MEM_L1, true);

            auto tmp_b_2 = View(tmp_b_1, shape2, {0,0});
            tmp_b_2->SetMemoryTypeBoth(MEM_L1, true);
            output = Add(tmp_b_2, tmp_output);
        }
        originFunction = Program::GetInstance().GetFunctionByRawName("TENSOR_ADD");
        ASSERT_NE(originFunction, nullptr) << "当前函数指针为空";
        auto operations = originFunction->Operations();
        for (const auto &op : operations) {
            originOpmagic.emplace_back(op.opmagic);
        }
        std::string jsonFilePathBefore = "./config/pass/json/before_insert_convert_op_l1_to_ddr.json";
        DumpJsonFile(Program::GetInstance().DumpJson(), jsonFilePathBefore);
        Json readData = LoadJsonFile(jsonFilePathBefore);
        Program::GetInstance().LoadJson(readData);

        Function* func = Program::GetInstance().GetCurrentFunction();

        Program testProgram(HostMachineMode::SERVER);
        npu::tile_fwk::InsertConvertOp insertConvertOp;
        insertConvertOp.PreCheck(*func);
        insertConvertOp.RunOnFunction(*func);
        insertConvertOp.PostCheck(*func);

        std::string jsonFilePathAfter = "./config/pass/json/after_insert_convert_op_l1_to_ddr.json";
        DumpJsonFile(Program::GetInstance().DumpJson(), jsonFilePathAfter);

        // ================== Verify Pass Effect ==================
        auto updatedOperations = Program::GetInstance().GetFunctionByRawName("TENSOR_ADD")->Operations();
        constexpr int expectedOperationsSize = 8;
        EXPECT_EQ(updatedOperations.size(), expectedOperationsSize) << "8 operations should remain View + Convert + Add + Assemble";
        int convertNum = 0;
        for (const auto &updatedOperation : updatedOperations) {
            if (updatedOperation.GetOpcode() == Opcode::OP_CONVERT)
                convertNum++;
        }
        constexpr int expextedConvertNum = 4;
        EXPECT_EQ(convertNum, expextedConvertNum) << "4 operations should be Convert";
    }
}

TEST_F(InsertConvertOpPassTest, L0CToDDR) {
    PROGRAM("InsertConvertOpPassTest") {
        std::vector<int> shape1{256, 256};
        std::vector<int> shape2{128, 128};
        Program::GetInstance().GetTileShape().SetVecTileShapes({128, 128});
        Program::GetInstance().GetTileShape().SetCubeTileShapes({128, 128}, {128, 128}, {128, 128});
        Tensor input_a(DT_FP32, shape1, "A");
        Tensor input_b(DT_FP32, shape1, "B");
        Tensor output(DT_FP32, shape2, "C");

        PassManager &passManager = PassManager::Instance();
        passManager.RegisterStrategy("InsertConvertOpPassTestStrategy", {
            {   "RemoveRedundentReshape",   "RemoveRedundentReshape",  PassType::TYPE_TENSOR_GRAPH},
            {           "ExpandFunction",           "ExpandFunction",  PassType::TYPE_TENSOR_GRAPH},
            {            "DuplicateView",            "DuplicateView",    PassType::TYPE_TILE_GRAPH},
            {        "MergeViewAssemble",        "MergeViewAssemble",    PassType::TYPE_TILE_GRAPH},
            {         "AssignMemoryType",         "AssignMemoryType",    PassType::TYPE_TILE_GRAPH},

        });
        ConfigManager::Instance();

        Function* originFunction = nullptr;
        std::vector<int> originOpmagic;
        FUNCTION("A_MUL_Bt", FunctionType::STATIC, {input_a, input_b, output}) {
            config::SetPassStrategy("InsertConvertOpPassTestStrategy");

            auto tmp_a_0 = View(input_a, shape2, {0,0});
            auto tmp_b_1 = View(input_b, shape2, {0,0});

            Tensor tmp_output(DT_FP32, shape2, "C");
            tmp_output->SetMemoryTypeBoth(MEM_DEVICE_DDR, true);
            tmp_output = Matrix::Matmul<false, true>(DataType::DT_FP32, tmp_a_0, tmp_b_1);

            output = Add(tmp_a_0, tmp_output);
        }
        originFunction = Program::GetInstance().GetFunctionByRawName("TENSOR_A_MUL_Bt");
        ASSERT_NE(originFunction, nullptr) << "当前函数指针为空";
        auto operations = originFunction->Operations();
        for (const auto &op : operations) {
            originOpmagic.emplace_back(op.opmagic);
        }

        std::string jsonFilePath = "./config/pass/json/insert_convert_op_l0c_to_ddr.json";
        bool dumpJsonFlag = true;
        if (dumpJsonFlag) {
            auto programJson = Program::GetInstance().DumpJson();
            DumpJsonFile(programJson, jsonFilePath);
        }
        Json readData = LoadJsonFile(jsonFilePath);
        Program::GetInstance().LoadJson(readData);

        Function* func = Program::GetInstance().GetCurrentFunction();

        Program testProgram(HostMachineMode::SERVER);
        npu::tile_fwk::InsertConvertOp insertConvertOp;
        insertConvertOp.PreCheck(*func);
        insertConvertOp.RunOnFunction(*func);
        insertConvertOp.PostCheck(*func);
        // ================== Verify Pass Effect ==================
        auto updatedOperations = Program::GetInstance().GetFunctionByRawName("TENSOR_A_MUL_Bt")->Operations();
        constexpr int expectedOperationsSize = 8;
        EXPECT_EQ(updatedOperations.size(), expectedOperationsSize) << "8 operations should remain View + Convert + Add + Assemble";
        int convertNum = 0;
        for (const auto &updatedOperation : updatedOperations) {
            if (updatedOperation.GetOpcode() == Opcode::OP_CONVERT)
                convertNum++;
        }
        constexpr int expextedConvertNum = 2;
        EXPECT_EQ(convertNum, expextedConvertNum) << "2 operations should be Convert";
    }
}
