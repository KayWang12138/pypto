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
 * \file test_assign_memory_type.cpp
 * \brief Unit test for assign_memory_type pass.
 */

#include <gtest/gtest.h>
#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_manager.h"
#include "passes/tile_graph_pass/assign_memory_type.h"
#include "passes/pass_manager.h"
#include "passes/pass_config/pass_config_manager.h"
#include <fstream>
#include <vector>

using namespace npu::tile_fwk;

namespace npu{
namespace tile_fwk {
const int NUM_32 = 32;
const int NUM_64 = 64;
const int NUM_128 = 128;
constexpr float F_1 = 1.0;
constexpr float F_3 = 3.0;

class AssignMemoryTypeTest : public testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {
        Program::GetInstance().Reset();
        Program::GetInstance().GetConfig().Reset();
        config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);
        config::SetPlatformConfig("ENABLE_COST_MODEL", false);
        config::SetPlatformConfig("TEST_IS_TIG", true);
        PassConfigManager::Instance().Initialize(DPlatform::ASCEND_910B2);
    }
    void TearDown() override {}

    void SetHalfwayStrategy() {
        PassManager &passManager = PassManager::Instance();
        passManager.RegisterStrategy("AssignMemoryTypeTestStrategy", {
            {   "RemoveRedundantReshape",   "RemoveRedundantReshape",  PassType::TYPE_TENSOR_GRAPH},
            {           "ExpandFunction",           "ExpandFunction",  PassType::TYPE_TENSOR_GRAPH},
            {            "DuplicateView",            "DuplicateView",    PassType::TYPE_TILE_GRAPH},
            {        "MergeViewAssemble",        "MergeViewAssemble",    PassType::TYPE_TILE_GRAPH},
        });
        ConfigManager::Instance();
    }

    void CheckConvertOp(const Operation &op, bool verbose = false) {
        /*
        1. 单输入单输出
        2. 输入/输出的tensor mem类型唯一
        3. 输入和输出的tensor的mem类型不同
        */
        EXPECT_EQ(op.GetIOperands().size(), 1) << "OP_CONVERT should have ONLY ONE input.";
        EXPECT_EQ(op.GetOOperands().size(), 1) << "OP_CONVERT should have ONLY ONE input.";
        auto input = op.GetIOperands().front();
        ASSERT_NE(input, nullptr) << "OP_CONVERT input is nullptr";
        auto output = op.GetOOperands().front();
        ASSERT_NE(output, nullptr) << "OP_CONVERT output is nullptr";
        auto inputMemOri = input->GetMemoryTypeOriginal();
        auto inputMemTobe = input->GetMemoryTypeToBe();
        auto outputMemOri = output->GetMemoryTypeOriginal();
        auto outputMemTobe = output->GetMemoryTypeToBe();
        if (verbose) {
            std::cout << "\t|--- iOperand " << input->magic;
            std::cout << ", mem ori: " << BriefMemoryTypeToString(inputMemOri);
            std::cout << ", tobe: " << BriefMemoryTypeToString(inputMemTobe) << std::endl;
            std::cout << "\t|--- oOperand " << output->magic;
            std::cout << ", mem ori: " << BriefMemoryTypeToString(outputMemOri);
            std::cout << ", tobe: " << BriefMemoryTypeToString(outputMemTobe) << std::endl;
        }
        EXPECT_EQ(inputMemOri, inputMemTobe) << "OP_CONVERT input Memory Ori should be the same as Memory Tobe.";
        EXPECT_EQ(outputMemOri, outputMemTobe) << "OP_CONVERT output Memory Ori should be the same as Memory Tobe.";
        EXPECT_NE(inputMemOri, outputMemOri) << "OP_CONVERT input should have different memory type from output.";
    }
};

TEST_F(AssignMemoryTypeTest, AddReshape) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TILE_AddReshape", "TILE_AddReshape", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    Program::GetInstance().InsertFuncToFunctionMap("TILE_AddReshape", currFunctionPtr);

    constexpr int opMagic0 = 1001;
    constexpr int opMagic1 = 1002;
    constexpr int opMagic2 = 1003;
    constexpr int opMagic3 = 1004;
    constexpr int opMagic4 = 1005;

    constexpr int tensorMagic0 = 1;
    constexpr int tensorMagic1 = 2;
    constexpr int tensorMagic2 = 3;
    constexpr int tensorMagic3 = 4;
    constexpr int tensorMagic4 = 5;
    constexpr int tensorMagic5 = 6;
    constexpr int tensorMagic6 = 7;
    // Prepare the graph
    std::vector<int> shape = {16, 32};
    std::vector<int> shape1 = {32,16};
    std::shared_ptr<LogicalTensor> input_tensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    input_tensor1->SetMemoryTypeBoth(MEM_DEVICE_DDR);
    input_tensor1->SetMagic(tensorMagic0);

    std::shared_ptr<LogicalTensor> input_tensor2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    input_tensor2->SetMemoryTypeBoth(MEM_UNKNOWN);
    input_tensor2->SetMagic(tensorMagic1);

    std::shared_ptr<LogicalTensor> view_output1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    view_output1->SetMemoryTypeBoth(MEM_UNKNOWN);
    view_output1->SetMagic(tensorMagic6);

    std::shared_ptr<LogicalTensor> view_output2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    view_output2->SetMemoryTypeBoth(MEM_UNKNOWN);
    view_output2->SetMagic(tensorMagic2);

    std::shared_ptr<LogicalTensor> add_output = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    add_output->SetMemoryTypeBoth(MEM_UNKNOWN);
    add_output->SetMagic(tensorMagic3);

    std::shared_ptr<LogicalTensor> reshape_output = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    reshape_output->SetMemoryTypeBoth(MEM_UNKNOWN);
    reshape_output->SetMagic(tensorMagic4);

    std::shared_ptr<LogicalTensor> assemble_output = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    assemble_output->SetMemoryTypeBoth(MEM_UNKNOWN);
    assemble_output->SetMagic(tensorMagic5);

    auto &view_op1 = currFunctionPtr->AddRawOperation(Opcode::OP_VIEW, {input_tensor1}, {view_output1});
    view_op1.SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int>{0, 0}));
    view_op1.opmagic = opMagic0;

    auto &view_op2 = currFunctionPtr->AddRawOperation(Opcode::OP_VIEW, {input_tensor2}, {view_output2});
    view_op2.SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int>{0, 0}));
    view_op2.opmagic = opMagic3;

    auto &add_op = currFunctionPtr->AddRawOperation(Opcode::OP_ADD, {view_output1, view_output2}, {add_output});
    add_op.opmagic = opMagic1;

    auto &reshape_op = currFunctionPtr->AddRawOperation(Opcode::OP_RESHAPE, {add_output}, {reshape_output});
    reshape_op.opmagic = opMagic4;

    auto &assemble_op = currFunctionPtr->AddRawOperation(Opcode::OP_ASSEMBLE, {reshape_output}, {assemble_output});
    assemble_op.SetOpAttribute(std::make_shared<AssembleOpAttribute>(std::vector<int>{0, 0}));
    assemble_op.opmagic = opMagic2;

    currFunctionPtr->inCasts_.push_back(input_tensor1);
    currFunctionPtr->inCasts_.push_back(input_tensor2);
    currFunctionPtr->outCasts_.push_back(assemble_output);

    std::stringstream ssBefore;
    ssBefore << "Before_AssignMemoryType";

    // Call the pass
    AssignMemoryType assignMemoryType;
    assignMemoryType.PreCheck(*currFunctionPtr);
    assignMemoryType.RunOnFunction(*currFunctionPtr);
    assignMemoryType.PostCheck(*currFunctionPtr);

    std::stringstream ss;
    ss << "After_AssignMemoryType";

    // Validate the results, 所有op的输入输出memory类型唯一
    std::cout << "========== op size: " << currFunctionPtr->Operations().size() << std::endl;
    int convertNum = 0;
    for (auto &op : currFunctionPtr->Operations()) {
        std::cout << op.GetOpcodeStr() << " " << op.GetOpMagic() << std::endl;
        for (auto &input : op.GetIOperands()) {
            auto memOri = input->GetMemoryTypeOriginal();
            auto memTobe = input->GetMemoryTypeToBe();
            std::cout << "\t|--- iOperand " << input->magic;
            std::cout << ", mem ori: " << BriefMemoryTypeToString(memOri);
            std::cout << ", tobe: " << BriefMemoryTypeToString(memTobe) << std::endl;
            EXPECT_EQ(memOri, memTobe) << " input Memory Ori should be the same as Memory Tobe";
        }
        for (auto &output : op.GetOOperands()) {
            auto memOri = output->GetMemoryTypeOriginal();
            auto memTobe = output->GetMemoryTypeToBe();
            std::cout << "\t|--- oOperand " << output->magic;
            std::cout << ", mem ori: " << BriefMemoryTypeToString(memOri);
            std::cout << ", tobe: " << BriefMemoryTypeToString(memTobe) << std::endl;
            EXPECT_EQ(memOri, memTobe) << " output Memory Ori should be the same as Memory Tobe";
        }
        if (op.GetOpcode() == Opcode::OP_CONVERT) {
            convertNum++;
            CheckConvertOp(op);
        }
    }
    constexpr int expextedConvertNum = 1;
    EXPECT_EQ(convertNum, expextedConvertNum) << "ONLY ONE OP_CONVERT.";
}

TEST_F(AssignMemoryTypeTest, TestVecToCube) {
    config::SetHostConfig(KEY_STRATEGY, "PVC2_OOO");
    std::vector<int> shape0 = {256, 128};
    std::vector<int> shape1 = {128, 64};
    std::vector<int> shape2 = {256, 64};
    PROGRAM("AssignMemoryTest") {
        Tensor input1(DataType::DT_FP32, shape0, "A");
        Tensor input2(DataType::DT_FP32, shape0, "B");
        Tensor weight(DataType::DT_FP32, shape1, "weight");
        Tensor out(DataType::DT_FP32, shape2, "output");
        FUNCTION("TestVecToCube", FunctionType::STATIC, {input1, input2, weight, out}) {
            Program::GetInstance().GetTileShape().SetVecTileShapes(NUM_128, NUM_128);
            Tensor addRes = Add(input1, input2); // 256 * 128
            Program::GetInstance().GetTileShape().SetCubeTileShapes({NUM_32, NUM_32}, {NUM_128, NUM_128}, {NUM_64, NUM_64});
            Tensor mmRes = Matrix::Matmul(out.GetDataType(), addRes, weight); // (256 * 128) @ (128 * 64) = (256 * 64)
            Program::GetInstance().GetTileShape().SetVecTileShapes(NUM_128, NUM_128);
            Tensor sumRes = RowSumSingle(addRes, 1);
            Program::GetInstance().GetTileShape().SetVecTileShapes(NUM_64, NUM_64);
            out = Add(mmRes, sumRes);
        }
    }
}

TEST_F(AssignMemoryTypeTest, TestVecToCubeV2) {
    config::SetHostConfig(KEY_STRATEGY, "AssignMemoryTypeTestStrategy");
    std::vector<int> shape0 = {256, 128};
    std::vector<int> shape1 = {128, 64};
    std::vector<int> shape2 = {256, 64};
    PROGRAM("AssignMemoryTest") {
        Tensor input1(DataType::DT_FP32, shape0, "A");
        Tensor input2(DataType::DT_FP32, shape0, "B");
        Tensor weight(DataType::DT_FP32, shape1, "weight");
        Tensor out(DataType::DT_FP32, shape2, "output");
        SetHalfwayStrategy();
        Function* originFunction = nullptr;

        FUNCTION("TestVecToCubeV2", FunctionType::STATIC, {input1, input2, weight, out}) {
            config::SetPassStrategy("AssignMemoryTypeTestStrategy");
            Program::GetInstance().GetTileShape().SetVecTileShapes(NUM_128, NUM_128);
            Tensor addRes = Add(input1, input2); // 256 * 128
            Program::GetInstance().GetTileShape().SetCubeTileShapes({NUM_32, NUM_32}, {NUM_128, NUM_128}, {NUM_64, NUM_64});
            Tensor mmRes = Matrix::Matmul(out.GetDataType(), addRes, weight); // (256 * 128) @ (128 * 64) = (256 * 64)
            Program::GetInstance().GetTileShape().SetVecTileShapes(NUM_128, NUM_128);
            Tensor sumRes = RowSumSingle(addRes, 1);
            Program::GetInstance().GetTileShape().SetVecTileShapes(NUM_64, NUM_64);
            out = Add(mmRes, sumRes);
        }

        originFunction = Program::GetInstance().GetFunctionByRawName("TENSOR_TestVecToCubeV2"); // Tensor_{Function名字}
        ASSERT_NE(originFunction, nullptr) << "当前函数指针为空";
        // Call the pass
        AssignMemoryType assignMemoryType;
        assignMemoryType.PreCheck(*originFunction);
        assignMemoryType.RunOnFunction(*originFunction);
        assignMemoryType.PostCheck(*originFunction);
        // ================== Verify Pass Effect ==================
        auto updatedOperations = originFunction->Operations();
        int convertNum = 0;
        for (const auto &op : updatedOperations) {
            if (op.GetOpcode() == Opcode::OP_CONVERT) {
                convertNum++;
                std::cout << op.GetOpcodeStr() << " " << op.GetOpMagic() << std::endl;
                CheckConvertOp(op, true);
            }
        }
        constexpr int expextedConvertNum = 6;
        EXPECT_EQ(convertNum, expextedConvertNum) << "6 operations should be Convert";
    }
}

TEST_F(AssignMemoryTypeTest, TestCubeToCube) {
    config::SetHostConfig(KEY_STRATEGY, "PVC2_OOO");
    std::vector<int> shape0 = {256, 128};
    std::vector<int> shape1 = {128, 64};
    std::vector<int> shape2 = {256, 256};
    PROGRAM("AssignMemoryTest") {
        Tensor inputQ(DataType::DT_FP32, shape0, "Q");
        Tensor inputK(DataType::DT_FP32, shape0, "K");
        Tensor weight(DataType::DT_FP32, shape1, "weight");
        Tensor out(DataType::DT_FP32, shape2, "output");
        FUNCTION("TestCubeToCube", FunctionType::STATIC, {inputQ, inputK, weight, out}) {
            Program::GetInstance().GetTileShape().SetCubeTileShapes({NUM_128, NUM_128}, {NUM_128, NUM_128}, {NUM_64, NUM_64});
            Tensor qUpdate = Matrix::Matmul(out.GetDataType(), inputQ, weight); // (256 * 128) @ (128 * 64) = (256 * 64)
            Program::GetInstance().GetTileShape().SetCubeTileShapes({NUM_128, NUM_128}, {NUM_128, NUM_128}, {NUM_64, NUM_64});
            Tensor kUpdate = Matrix::Matmul(out.GetDataType(), inputK, weight); // (256 * 128) @ (128 * 64) = (256 * 64)
            Program::GetInstance().GetTileShape().SetCubeTileShapes({NUM_128, NUM_128}, {NUM_64, NUM_64}, {NUM_128, NUM_128});
            Tensor QKT = Matrix::Matmul<false, true>(out.GetDataType(), qUpdate, kUpdate); // (256 * 64) @ (64 * 256) = (256 * 256)
            Program::GetInstance().GetTileShape().SetVecTileShapes(NUM_64, NUM_64);
            out = SubS(QKT, Element(DataType::DT_FP32, F_3));
        }
    }
}

TEST_F(AssignMemoryTypeTest, TestCubeToCubeV2) {
    config::SetHostConfig(KEY_STRATEGY, "AssignMemoryTypeTestStrategy");
    std::vector<int> shape0 = {256, 128};
    std::vector<int> shape1 = {128, 64};
    std::vector<int> shape2 = {256, 256};
    PROGRAM("AssignMemoryTest") {
        Tensor inputQ(DataType::DT_FP32, shape0, "Q");
        Tensor inputK(DataType::DT_FP32, shape0, "K");
        Tensor weight(DataType::DT_FP32, shape1, "weight");
        Tensor out(DataType::DT_FP32, shape2, "output");
        SetHalfwayStrategy();
        Function* originFunction = nullptr;

        FUNCTION("TestCubeToCubeV2", FunctionType::STATIC, {inputQ, inputK, weight, out}) {
            Program::GetInstance().GetTileShape().SetCubeTileShapes({NUM_128, NUM_128}, {NUM_128, NUM_128}, {NUM_64, NUM_64});
            Tensor qUpdate = Matrix::Matmul(out.GetDataType(), inputQ, weight); // (256 * 128) @ (128 * 64) = (256 * 64)
            Program::GetInstance().GetTileShape().SetCubeTileShapes({NUM_128, NUM_128}, {NUM_128, NUM_128}, {NUM_64, NUM_64});
            Tensor kUpdate = Matrix::Matmul(out.GetDataType(), inputK, weight); // (256 * 128) @ (128 * 64) = (256 * 64)
            Program::GetInstance().GetTileShape().SetCubeTileShapes({NUM_128, NUM_128}, {NUM_64, NUM_64}, {NUM_128, NUM_128});
            Tensor QKT = Matrix::Matmul<false, true>(out.GetDataType(), qUpdate, kUpdate); // (256 * 64) @ (64 * 256) = (256 * 256)
            Program::GetInstance().GetTileShape().SetVecTileShapes(NUM_64, NUM_64);
            out = AddS(QKT, Element(DataType::DT_FP32, F_1));
        }

        originFunction = Program::GetInstance().GetFunctionByRawName("TENSOR_TestCubeToCubeV2"); // Tensor_{Function名字}
        ASSERT_NE(originFunction, nullptr) << "当前函数指针为空";
        // Call the pass
        AssignMemoryType assignMemoryType;
        assignMemoryType.PreCheck(*originFunction);
        assignMemoryType.RunOnFunction(*originFunction);
        assignMemoryType.PostCheck(*originFunction);
        // ================== Verify Pass Effect ==================
        auto opList = originFunction->Operations();
        int convertNum = 0;
        for (const auto &op : opList) {
            if (op.GetOpcode() != Opcode::OP_CONVERT) {
                continue;
            }
            std::cout << op.GetOpcodeStr() << " " << op.GetOpMagic() << std::endl;
            CheckConvertOp(op, true);
            convertNum++;
        }
        constexpr int expextedConvertNum = 12;
        EXPECT_EQ(convertNum, expextedConvertNum) << "12 operations should be Convert";
    }
}

void GetInvalidPatternGraph(std::shared_ptr<Function> &currFunctionPtr) {
    constexpr int opMagic0 = 1001;
    constexpr int opMagic1 = 1002;
    constexpr int opMagic2 = 1003;
    constexpr int opMagic3 = 1004;
    constexpr int opMagic4 = 1005;
    constexpr int opMagic5 = 1006;
    constexpr int opMagic6 = 1007;
    constexpr int opMagic7 = 1008;

    constexpr int tensorMagic0 = 1;
    constexpr int tensorMagic1 = 2;
    constexpr int tensorMagic2 = 3;
    constexpr int tensorMagic3 = 4;
    constexpr int tensorMagic4 = 5;
    constexpr int tensorMagic5 = 6;
    constexpr int tensorMagic6 = 7;
    constexpr int tensorMagic7 = 8;
    // Prepare the graph
    std::vector<int> shape = {16, 32};
    std::vector<int> shape1 = {32,16};
    std::vector<int> shape2 = {8,32};
    std::vector<int> shape3 = {32,8};
    std::shared_ptr<LogicalTensor> input_cast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    input_cast->SetMagic(tensorMagic0);

    std::shared_ptr<LogicalTensor> input_tensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    input_tensor1->SetMagic(tensorMagic1);

    std::shared_ptr<LogicalTensor> view_output1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    view_output1->SetMagic(tensorMagic2);

    std::shared_ptr<LogicalTensor> view_output2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    view_output2->SetMagic(tensorMagic3);

    std::shared_ptr<LogicalTensor> reshape_output1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape3);
    reshape_output1->SetMagic(tensorMagic4);

    std::shared_ptr<LogicalTensor> reshape_output2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape3);
    reshape_output2->SetMagic(tensorMagic5);

    std::shared_ptr<LogicalTensor> assemble_output = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    assemble_output->SetMagic(tensorMagic6);

    std::shared_ptr<LogicalTensor> output_cast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    output_cast->SetMagic(tensorMagic7);

    auto &reshape_op0 = currFunctionPtr->AddRawOperation(Opcode::OP_RESHAPE, {input_cast}, {input_tensor1});
    reshape_op0.opmagic = opMagic0;

    auto &view_op1 = currFunctionPtr->AddRawOperation(Opcode::OP_VIEW, {input_tensor1}, {view_output1});
    view_op1.SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int>{0, 0}));
    view_op1.opmagic = opMagic1;

    auto &view_op2 = currFunctionPtr->AddRawOperation(Opcode::OP_VIEW, {input_tensor1}, {view_output2});
    view_op2.SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int>{8, 0}));
    view_op2.opmagic = opMagic2;

    auto &reshape_op1 = currFunctionPtr->AddRawOperation(Opcode::OP_RESHAPE, {view_output1}, {reshape_output1});
    reshape_op1.opmagic = opMagic3;

    auto &reshape_op2 = currFunctionPtr->AddRawOperation(Opcode::OP_RESHAPE, {view_output2}, {reshape_output2});
    reshape_op2.opmagic = opMagic4;

    auto &assemble_op1 = currFunctionPtr->AddRawOperation(Opcode::OP_ASSEMBLE, {reshape_output1}, {assemble_output});
    assemble_op1.SetOpAttribute(std::make_shared<AssembleOpAttribute>(std::vector<int>{0, 0}));
    assemble_op1.opmagic = opMagic5;

    auto &assemble_op2 = currFunctionPtr->AddRawOperation(Opcode::OP_ASSEMBLE, {reshape_output2}, {assemble_output});
    assemble_op2.SetOpAttribute(std::make_shared<AssembleOpAttribute>(std::vector<int>{8, 0}));
    assemble_op2.opmagic = opMagic6;

    auto &view_op3 = currFunctionPtr->AddRawOperation(Opcode::OP_VIEW, {assemble_output}, {output_cast});
    view_op3.SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int>{0, 0}));
    view_op3.opmagic = opMagic7;

    currFunctionPtr->inCasts_.push_back(input_cast);
    currFunctionPtr->outCasts_.push_back(output_cast);
}
TEST_F(AssignMemoryTypeTest, InValidOpPattern) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "InValidOpPattern", "InValidOpPattern", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    Program::GetInstance().InsertFuncToFunctionMap("InValidOpPattern", currFunctionPtr);
    
    GetInvalidPatternGraph(currFunctionPtr);

    std::stringstream ssBefore;
    ssBefore << "Before_AssignMemoryType";

    // Call the pass
    AssignMemoryType assignMemoryType;
    assignMemoryType.PreCheck(*currFunctionPtr);
    assignMemoryType.RunOnFunction(*currFunctionPtr);
    assignMemoryType.PostCheck(*currFunctionPtr);

    std::stringstream ss;
    ss << "After_AssignMemoryType";

    std::string josnFilePath = "/config/pass/json/assign_mem_type_invalidpattern.json";
    currFunctionPtr->DumpJsonFile(josnFilePath);

    // Validate the results
    std::cout << "========== op size: " << currFunctionPtr->Operations().size() << std::endl;
    for (auto &op : currFunctionPtr->Operations()) {
        std::cout << op.GetOpcodeStr() << " " << op.GetOpMagic() << std::endl;
        for (auto &input : op.GetIOperands()) {
            std::cout << "\t|--- iOperand " << input->magic;
            EXPECT_EQ(input->GetMemoryTypeOriginal(), MemoryType::MEM_DEVICE_DDR) << " Unexpected memory type.";
            EXPECT_EQ(input->GetMemoryTypeOriginal(), input->GetMemoryTypeToBe()) << " iOperand has two memory type.";
        }
        for (auto &output : op.GetOOperands()) {
            std::cout << "\t|--- oOperand " << output->magic << std::endl;
            EXPECT_EQ(output->GetMemoryTypeOriginal(), MemoryType::MEM_DEVICE_DDR) << " Unexpected memory type.";
            EXPECT_EQ(output->GetMemoryTypeOriginal(), output->GetMemoryTypeToBe()) << " oOperand has two memory type.";
        }
    }
}

void GetViewReshapeGraph (std::shared_ptr<Function> &currFunctionPtr) {
    constexpr int opMagic0 = 1001;
    constexpr int opMagic1 = 1002;
    constexpr int opMagic2 = 1003;
    constexpr int opMagic3 = 1004;
    constexpr int opMagic4 = 1005;

    constexpr int tensorMagic0 = 1;
    constexpr int tensorMagic1 = 2;
    constexpr int tensorMagic2 = 3;
    constexpr int tensorMagic3 = 4;
    constexpr int tensorMagic4 = 5;
    constexpr int tensorMagic5 = 6;

    // Prepare the graph
    std::vector<int> shape = {16, 32};
    std::vector<int> shape1 = {32, 16};
    std::vector<int> shape2 = {1, 32};
    std::vector<int> shape3 = {8, 32};
    std::shared_ptr<LogicalTensor> input_cast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    input_cast->SetMagic(tensorMagic0);

    std::shared_ptr<LogicalTensor> transpose_out = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    transpose_out->SetMagic(tensorMagic1);

    std::shared_ptr<LogicalTensor> view_output1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    view_output1->SetMagic(tensorMagic2);

    std::shared_ptr<LogicalTensor> reshape_output = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    reshape_output->SetMagic(tensorMagic3);

    std::shared_ptr<LogicalTensor> view_output2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    view_output2->SetMagic(tensorMagic4);

    std::shared_ptr<LogicalTensor> output_cast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape3);
    output_cast->SetMagic(tensorMagic5);

    auto &transpose_op = currFunctionPtr->AddRawOperation(Opcode::OP_TRANSPOSE_VNCHWCONV, {input_cast}, {transpose_out});
    transpose_op.opmagic = opMagic0;

    auto &view_op1 = currFunctionPtr->AddRawOperation(Opcode::OP_VIEW, {transpose_out}, {view_output1});
    view_op1.SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int>{0, 0}));
    view_op1.opmagic = opMagic1;

    auto &reshape_op = currFunctionPtr->AddRawOperation(Opcode::OP_RESHAPE, {view_output1}, {reshape_output});
    reshape_op.opmagic = opMagic2;

    auto &view_op2 = currFunctionPtr->AddRawOperation(Opcode::OP_VIEW, {reshape_output}, {view_output2});
    view_op2.SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int>{0, 0}));
    view_op2.opmagic = opMagic3; 

    auto &expand_op = currFunctionPtr->AddRawOperation(Opcode::OP_EXPAND, {view_output2}, {output_cast});
    expand_op.opmagic = opMagic4;
}
TEST_F(AssignMemoryTypeTest, ViewReshape) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "ViewReshape", "ViewReshape", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    Program::GetInstance().InsertFuncToFunctionMap("ViewReshape", currFunctionPtr);
    
    GetViewReshapeGraph(currFunctionPtr);

    std::stringstream ssBefore;
    ssBefore << "Before_AssignMemoryType";

    // Call the pass
    AssignMemoryType assignMemoryType;
    assignMemoryType.PreCheck(*currFunctionPtr);
    assignMemoryType.RunOnFunction(*currFunctionPtr);
    assignMemoryType.PostCheck(*currFunctionPtr);

    std::stringstream ss;
    ss << "After_AssignMemoryType";

    std::string josnFilePath = "/config/pass/json/assign_mem_type_invalidpattern.json";
    currFunctionPtr->DumpJsonFile(josnFilePath);

    // Validate the results
    std::cout << "========== op size: " << currFunctionPtr->Operations().size() << std::endl;
    for (auto &op : currFunctionPtr->Operations()) {
        std::cout << op.GetOpcodeStr() << " " << op.GetOpMagic() << std::endl;
        for (auto &input : op.GetIOperands()) {
            std::cout << "\t|--- iOperand " << input->magic;
            EXPECT_EQ(input->GetMemoryTypeOriginal(), MemoryType::MEM_UB) << " Unexpected memory type.";
            EXPECT_EQ(input->GetMemoryTypeOriginal(), input->GetMemoryTypeToBe()) << " iOperand has two memory type.";
        }
        for (auto &output : op.GetOOperands()) {
            std::cout << "\t|--- oOperand " << output->magic << std::endl;
            EXPECT_EQ(output->GetMemoryTypeOriginal(), MemoryType::MEM_UB) << " Unexpected memory type.";
            EXPECT_EQ(output->GetMemoryTypeOriginal(), output->GetMemoryTypeToBe()) << " oOperand has two memory type.";
        }
    }
}

}
} // namespace npu::tile_fwk