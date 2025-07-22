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
 * \file test_generate_move_op.cpp
 * \brief Unit test for Generate Move Op pass.
 */

#include <gtest/gtest.h>
#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "passes/pass_manager.h"
#include "ut_json/ut_json_tool.h"
#include "passes/tile_graph_pass/generate_move_op.h"
#include "interface/configs/config_manager.h"
#include <fstream>
#include <vector>
#include <string>

using namespace npu::tile_fwk;

namespace npu{
namespace tile_fwk {
class GenerateMoveOpPassTest : public ::testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {
        Program::GetInstance().Reset();
        Program::GetInstance().GetConfig().Reset();
        config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);
        config::SetHostConfig(KEY_STRATEGY, "GenerateMoveOpPassTestStrategy");
        config::SetPlatformConfig("ENABLE_COST_MODEL", false);
    }
    void TearDown() override {}
};
void GetCopyInCopyOutGraph(std::shared_ptr<Function> &currFunctionPtr){
    constexpr int opMagic0 = 1001;
    constexpr int opMagic1 = 1002;
    constexpr int opMagic2 = 1003;
    constexpr int opMagic3 = 1004;
    constexpr int opMagic4 = 1005;
    constexpr int opMagic5 = 1006;

    constexpr int tensorMagic0 = 1;
    constexpr int tensorMagic1 = 2;
    constexpr int tensorMagic2 = 3;
    constexpr int tensorMagic3 = 4;
    constexpr int tensorMagic4 = 5;
    constexpr int tensorMagic5 = 6;

    // Prepare the graph
    std::vector<int> shape = {16, 32};
    std::vector<int> shape1 = {8,32};
    std::shared_ptr<LogicalTensor> start_tensor = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    start_tensor->SetMagic(tensorMagic0);

    std::shared_ptr<LogicalTensor> tmp_tensor = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    tmp_tensor->SetMagic(tensorMagic1);

    std::shared_ptr<LogicalTensor> end_tensor = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    end_tensor->SetMagic(tensorMagic2);

    std::shared_ptr<LogicalTensor> output_tensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    output_tensor1->SetMagic(tensorMagic3);

    std::shared_ptr<LogicalTensor> output_tensor2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    output_tensor2->SetMagic(tensorMagic4);

    std::shared_ptr<LogicalTensor> assemble_output = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    assemble_output->SetMagic(tensorMagic5);

    auto &copy_in = currFunctionPtr->AddRawOperation(Opcode::OP_COPY_IN, {start_tensor}, {tmp_tensor});
    copy_in.opmagic = opMagic0;

    auto &copy_out = currFunctionPtr->AddRawOperation(Opcode::OP_COPY_OUT, {tmp_tensor}, {end_tensor});
    copy_out.opmagic = opMagic1;

    auto &view_op1 = currFunctionPtr->AddRawOperation(Opcode::OP_VIEW, {end_tensor}, {output_tensor1});
    view_op1.SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int>{0, 0}));
    view_op1.opmagic = opMagic2;

    auto &view_op2 = currFunctionPtr->AddRawOperation(Opcode::OP_VIEW, {end_tensor}, {output_tensor2});
    view_op2.SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int>{8, 0}));
    view_op2.opmagic = opMagic3;

    auto &assemble_op1 = currFunctionPtr->AddRawOperation(Opcode::OP_ASSEMBLE, {output_tensor1}, {assemble_output});
    assemble_op1.SetOpAttribute(std::make_shared<AssembleOpAttribute>(std::vector<int>{0, 0}));
    assemble_op1.opmagic = opMagic4;

    auto &assemble_op2 = currFunctionPtr->AddRawOperation(Opcode::OP_ASSEMBLE, {output_tensor2}, {assemble_output});
    assemble_op2.SetOpAttribute(std::make_shared<AssembleOpAttribute>(std::vector<int>{8, 0}));
    assemble_op2.opmagic = opMagic5;

    currFunctionPtr->inCasts_.push_back(start_tensor);
    currFunctionPtr->outCasts_.push_back(assemble_output);
}
TEST_F(GenerateMoveOpPassTest, MergeCopyInCopyOut) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "MergeCopyInCopyOut", "MergeCopyInCopyOut", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    Program::GetInstance().InsertFuncToFunctionMap("MergeCopyInCopyOut", currFunctionPtr);
    
    GetCopyInCopyOutGraph(currFunctionPtr);

    std::stringstream ssBefore;
    ssBefore << "Before_GenerateMoveOp";

    // Call the pass
    GenerateMoveOp generateMoveOp;
    generateMoveOp.PreCheck(*currFunctionPtr);
    generateMoveOp.RunOnFunction(*currFunctionPtr);
    generateMoveOp.PostCheck(*currFunctionPtr);

    std::stringstream ss;
    ss << "After_GenerateMoveOp";

    // Validate the results
    std::cout << "========== op size: " << currFunctionPtr->Operations().size() << std::endl;
    int view_num = 0;
    int copy_in_num = 0;
    int copy_out_num = 0;
    for (auto &op : currFunctionPtr->Operations()) {
        std::cout << op.GetOpcodeStr() << " " << op.GetOpMagic() << std::endl;
        for (auto &input : op.GetIOperands()) {
            std::cout << "\t|--- iOperand " << input->magic;
        }
        for (auto &output : op.GetOOperands()) {
            std::cout << "\t|--- oOperand " << output->magic;
        }
        if(op.GetOpcode()==Opcode::OP_VIEW){
            view_num++;
        }else if(op.GetOpcode()==Opcode::OP_COPY_IN){
            copy_in_num++;
        }else if(op.GetOpcode()==Opcode::OP_COPY_OUT) {
            copy_out_num++;
        }
    }
    constexpr int expectedView =2;
    constexpr int expectedCopyIn =0;
    constexpr int expectedCopyOut =0;
    EXPECT_EQ(view_num,expectedView) << "2 operations shoulde be OP_VIEW.";
    EXPECT_EQ(copy_in_num,expectedCopyIn) << "0 operations shoulde be OP_COPY_IN.";
    EXPECT_EQ(copy_out_num,expectedCopyOut) << "0 operations shoulde be OP_COPY_OUT.";
}

TEST_F(GenerateMoveOpPassTest, AssembleViewToCopy) {
    PROGRAM("GenerateMoveOpPassTest") {
        std::vector<int> shape1{256, 256};
        std::vector<int> shape2{128, 128};
        Program::GetInstance().GetTileShape().SetVecTileShapes({128, 128});
        Tensor input_a(DT_FP32, shape1, "input_a");
        Tensor input_b(DT_FP32, shape1, "input_b");
        Tensor output(DT_FP32, shape2, "output");
        PassManager &passManager = PassManager::Instance();
        passManager.RegisterStrategy("GenerateMoveOpPassTestStrategy", {
            {   "RemoveRedundentReshape",   "RemoveRedundentReshape",  PassType::TYPE_TENSOR_GRAPH},
            {           "ExpandFunction",           "ExpandFunction",  PassType::TYPE_TENSOR_GRAPH},
            {            "DuplicateView",            "DuplicateView",    PassType::TYPE_TILE_GRAPH},
            {        "MergeViewAssemble",        "MergeViewAssemble",    PassType::TYPE_TILE_GRAPH},
            {         "AssignMemoryType",         "AssignMemoryType",    PassType::TYPE_TILE_GRAPH},
            {          "InsertConvertOp",          "InsertConvertOp",    PassType::TYPE_TILE_GRAPH},
            {   "SplitLargeFanoutTensor",   "SplitLargeFanoutTensor",    PassType::TYPE_TILE_GRAPH},
            {       "SplitReshapeOpPVC2",       "SplitReshapeOpPVC2",    PassType::TYPE_TILE_GRAPH},
            {        "RemoveRedundentOp",        "RemoveRedundentOp",    PassType::TYPE_TILE_GRAPH},

        });
        ConfigManager::Instance();

        Function* originFunction = nullptr;
        std::vector<int> originOpmagic;
        FUNCTION("ADD", FunctionType::STATIC, {input_a, input_b, output}) {
            config::SetPassStrategy("GenerateMoveOpPassTestStrategy");

            auto tmp_a_0 = View(input_a, shape2, {0,0});
            auto tmp_b_1 = View(input_b, shape2, {0,0});

            output = Add(tmp_a_0, tmp_b_1);
        }

        std::string jsonFilePath = "./config/pass/json/generate_move_op_assemble_view_to_copy.json";
        bool dumpJsonFlag = true;
        if (dumpJsonFlag) {
            auto programJson = Program::GetInstance().DumpJson();
            DumpJsonFile(programJson, jsonFilePath);
        }
        Json readData = LoadJsonFile(jsonFilePath);
        Program::GetInstance().LoadJson(readData);

        originFunction = Program::GetInstance().GetFunctionByRawName("TENSOR_ADD");

        ASSERT_NE(originFunction, nullptr) << "当前函数指针为空";
        auto operations = originFunction->Operations();
        for (const auto &op : operations) {
            std::cout << "opmagic: " << op.opmagic << "op type " << op.GetOpcodeStr() << std::endl;
            originOpmagic.emplace_back(op.opmagic);
        }
        Program testProgram(HostMachineMode::SERVER);
        GenerateMoveOp generateMoveOp;
        generateMoveOp.RunOnFunction(*originFunction);

        // ================== Verify Pass Effect ==================
        auto updatedOperations = Program::GetInstance().GetFunctionByRawName("TENSOR_ADD")->Operations();
        constexpr int expectedOperations = 4;
        EXPECT_EQ(updatedOperations.size(), expectedOperations) << "4 operations should remain View + Convert + Add + Assemble";
        int assemble_num = 0;
        int view_num = 0;
        int copy_in_num = 0;
        int copy_out_num = 0;
        for (const auto &updatedOperation : updatedOperations) {
            switch (updatedOperation.GetOpcode()){
                case Opcode::OP_ASSEMBLE: {
                    assemble_num++;
                    break;
                }
                case Opcode::OP_VIEW: {
                    view_num++;
                    break;
                }
                case Opcode::OP_COPY_IN: {
                    copy_in_num++;
                    break;
                }
                case Opcode::OP_COPY_OUT: {
                    copy_out_num++;
                    break;
                }
                default: break;
            }
        }
        constexpr int expectedAssemble = 0;
        constexpr int expectedView = 0;
        constexpr int expectedCopyIn = 2;
        constexpr int expectedCopyOut = 1;
        EXPECT_EQ(assemble_num, expectedAssemble) << "0 operations should be OP_ASSEMBLE";
        EXPECT_EQ(view_num, expectedView) << "0 operations should be OP_VIEW";
        EXPECT_EQ(copy_in_num, expectedCopyIn) << "2 operations should be OP_COPY_IN";
        EXPECT_EQ(copy_out_num, expectedCopyOut) << "1 operations should be OP_COPY_OUT";
    }
}

TEST_F(GenerateMoveOpPassTest, ConvertToCopy) {
    PROGRAM("GenerateMoveOpPassTest") {
        std::vector<int> shape1{256, 256};
        std::vector<int> shape2{128, 128};
        Program::GetInstance().GetTileShape().SetVecTileShapes({128, 128});
        Tensor input_a(DT_FP32, shape1, "input_a");
        Tensor input_b(DT_FP32, shape1, "input_b");
        Tensor output(DT_FP32, shape2, "output");
        PassManager &passManager = PassManager::Instance();
        passManager.RegisterStrategy("GenerateMoveOpPassTestStrategy", {
            {   "RemoveRedundentReshape",   "RemoveRedundentReshape",  PassType::TYPE_TENSOR_GRAPH},
            {           "ExpandFunction",           "ExpandFunction",  PassType::TYPE_TENSOR_GRAPH},
            {            "DuplicateView",            "DuplicateView",    PassType::TYPE_TILE_GRAPH},
            {        "MergeViewAssemble",        "MergeViewAssemble",    PassType::TYPE_TILE_GRAPH},
            {         "AssignMemoryType",         "AssignMemoryType",    PassType::TYPE_TILE_GRAPH},
            {          "InsertConvertOp",          "InsertConvertOp",    PassType::TYPE_TILE_GRAPH},
            {   "SplitLargeFanoutTensor",   "SplitLargeFanoutTensor",    PassType::TYPE_TILE_GRAPH},
            {       "SplitReshapeOpPVC2",       "SplitReshapeOpPVC2",    PassType::TYPE_TILE_GRAPH},
            {        "RemoveRedundentOp",        "RemoveRedundentOp",    PassType::TYPE_TILE_GRAPH},
            {           "GenerateMoveOp",           "GenerateMoveOp",    PassType::TYPE_TILE_GRAPH},
        });
        ConfigManager::Instance();

        std::vector<int> originOpmagic;
        FUNCTION("ADD", FunctionType::STATIC, {input_a, input_b, output}) {
            config::SetPassStrategy("GenerateMoveOpPassTestStrategy");

            auto tmp_a_0 = View(input_a, shape2, {0,0});
            tmp_a_0->SetMemoryTypeBoth(MEM_L1, true);
            auto tmp_b_1 = View(input_b, shape2, {0,0});
            tmp_b_1->SetMemoryTypeBoth(MEM_L1, true);

            output = Add(tmp_a_0, tmp_b_1);
        }

        // ================== Verify Pass Effect ==================
        auto updatedOperations = Program::GetInstance().GetFunctionByRawName("TENSOR_ADD")->Operations();
        constexpr int expectedOperations = 8;
        EXPECT_EQ(updatedOperations.size(), expectedOperations) << "8 operations should remain View + Convert + Add + Assemble";
        int assemble_num = 0;
        int view_num = 0;
        int copy_in_num = 0;
        int copy_out_num = 0;
        for (const auto &updatedOperation : updatedOperations) {
            switch (updatedOperation.GetOpcode()){
                case Opcode::OP_ASSEMBLE: {
                    assemble_num++;
                    break;
                }
                case Opcode::OP_VIEW: {
                    view_num++;
                    break;
                }
                case Opcode::OP_COPY_IN: {
                    copy_in_num++;
                    break;
                }
                case Opcode::OP_COPY_OUT: {
                    copy_out_num++;
                    break;
                }
                default: break;
            }
        }
        constexpr int expectedAssemble = 0;
        constexpr int expectedView = 0;
        constexpr int expectedCopyIn = 4;
        constexpr int expectedCopyOut = 3;
        EXPECT_EQ(assemble_num, expectedAssemble) << "0 operations should be OP_ASSEMBLE";
        EXPECT_EQ(view_num, expectedView) << "0 operations should be OP_VIEW";
        EXPECT_EQ(copy_in_num, expectedCopyIn) << "4 operations should be OP_COPY_IN";
        EXPECT_EQ(copy_out_num, expectedCopyOut) << "3 operations should be OP_COPY_OUT";
    }
}

TEST_F(GenerateMoveOpPassTest, Transpose) {
    PROGRAM("GenerateMoveOpPassTest") {
        std::vector<int> shape{1, 32, 32, 2};
        Tensor a(DT_FP32, shape, "a");
        Tensor a_trans(DT_FP32, shape, "a_trans");

        constexpr int dim0 = 1, dim1 = 16, dim2 = 16, dim3 = 2;
        Program::GetInstance().GetTileShape().SetVecTileShapes(dim0, dim1, dim2, dim3);

        PassManager &passManager = PassManager::Instance();
        passManager.RegisterStrategy("GenerateMoveOpPassTestStrategy", {
            {   "RemoveRedundentReshape",   "RemoveRedundentReshape",  PassType::TYPE_TENSOR_GRAPH},
            {           "ExpandFunction",           "ExpandFunction",  PassType::TYPE_TENSOR_GRAPH},
            {            "DuplicateView",            "DuplicateView",    PassType::TYPE_TILE_GRAPH},
            {        "MergeViewAssemble",        "MergeViewAssemble",    PassType::TYPE_TILE_GRAPH},
            {         "AssignMemoryType",         "AssignMemoryType",    PassType::TYPE_TILE_GRAPH},
            {          "InsertConvertOp",          "InsertConvertOp",    PassType::TYPE_TILE_GRAPH},
            {   "SplitLargeFanoutTensor",   "SplitLargeFanoutTensor",    PassType::TYPE_TILE_GRAPH},
            {       "SplitReshapeOpPVC2",       "SplitReshapeOpPVC2",    PassType::TYPE_TILE_GRAPH},
            {        "RemoveRedundentOp",        "RemoveRedundentOp",    PassType::TYPE_TILE_GRAPH},

        });
        ConfigManager::Instance();

        FUNCTION("Tranpose") {
            a_trans = Transpose(a, {1, 2});
        }
        std::string jsonFilePath = "./config/pass/json/generate_move_op_transpose.json";

        bool dumpJsonFlag = true;
        if (dumpJsonFlag) {
            auto programJson = Program::GetInstance().DumpJson();
            DumpJsonFile(programJson, jsonFilePath);
        }
        Json readData = LoadJsonFile(jsonFilePath);
        Program::GetInstance().LoadJson(readData);

        Function* originFunction = Program::GetInstance().GetCurrentFunction();
        Program testProgram(HostMachineMode::SERVER);
        GenerateMoveOp generateMoveOp;
        generateMoveOp.RunOnFunction(*originFunction);

        // ================== Verify Pass Effect ==================
        auto updatedOperations = Program::GetInstance().GetFunctionByRawName("TENSOR_Tranpose")->Operations();
        constexpr int expectedOperations = 12;
        EXPECT_EQ(updatedOperations.size(), expectedOperations) << "total 12 operations";
        int assemble_num = 0;
        int view_num = 0;
        int copy_in_num = 0;
        int copy_out_num = 0;
        int transpose_datamove_num = 0;
        for (const auto &updatedOperation : updatedOperations) {
            switch (updatedOperation.GetOpcode()){
                case Opcode::OP_ASSEMBLE: {
                    assemble_num++;
                    break;
                }
                case Opcode::OP_VIEW: {
                    view_num++;
                    break;
                }
                case Opcode::OP_COPY_IN: {
                    copy_in_num++;
                    break;
                }
                case Opcode::OP_COPY_OUT: {
                    copy_out_num++;
                    break;
                }
                case Opcode::OP_TRANSPOSE_DATAMOVE: {
                    transpose_datamove_num++;
                    break;
                }
                default: break;
            }
        }
        constexpr int expectedAssemble = 4;
        constexpr int expectedView = 0;
        constexpr int expectedCopyIn = 4;
        constexpr int expectedCopyOut = 0;
        EXPECT_EQ(assemble_num, expectedAssemble) << "4 operations should be OP_ASSEMBLE";
        EXPECT_EQ(assemble_num, transpose_datamove_num) << "num of OP_ASSEMBLE and OP_TRANSPOSE_DATAMOVE should be equal";
        EXPECT_EQ(view_num, expectedView) << "0 operations should be OP_VIEW";
        EXPECT_EQ(copy_in_num, expectedCopyIn) << "4 operations should be OP_COPY_IN";
        EXPECT_EQ(copy_out_num, expectedCopyOut) << "0 operations should be OP_COPY_OUT";
    }
}

TEST_F(GenerateMoveOpPassTest, ScatterUpdate) {
    PROGRAM("GenerateMoveOpPassTest") {
        int row = 64, col = 32;
        Program::GetInstance().GetTileShape().SetVecTileShapes(row, col);

        PassManager &passManager = PassManager::Instance();
        passManager.RegisterStrategy("GenerateMoveOpPassTestStrategy", {
            {   "RemoveRedundentReshape",   "RemoveRedundentReshape",  PassType::TYPE_TENSOR_GRAPH},
            {           "ExpandFunction",           "ExpandFunction",  PassType::TYPE_TENSOR_GRAPH},
            {            "DuplicateView",            "DuplicateView",    PassType::TYPE_TILE_GRAPH},
            {        "MergeViewAssemble",        "MergeViewAssemble",    PassType::TYPE_TILE_GRAPH},
            {         "AssignMemoryType",         "AssignMemoryType",    PassType::TYPE_TILE_GRAPH},
            {          "InsertConvertOp",          "InsertConvertOp",    PassType::TYPE_TILE_GRAPH},
            {   "SplitLargeFanoutTensor",   "SplitLargeFanoutTensor",    PassType::TYPE_TILE_GRAPH},
            {       "SplitReshapeOpPVC2",       "SplitReshapeOpPVC2",    PassType::TYPE_TILE_GRAPH},
            {        "RemoveRedundentOp",        "RemoveRedundentOp",    PassType::TYPE_TILE_GRAPH},

        });
        ConfigManager::Instance();

        int b = 2, s = 64, numExpertsPerTok = 2, h = 128, minus_two = -2;
        Tensor output(DT_FP32, {b*s*numExpertsPerTok, h}, "output");
        Tensor idxs(DT_FP32, {1, b*s*numExpertsPerTok}, "idxs");
        Tensor key_states(DT_FP32, {b*s*numExpertsPerTok, h}, "key_states");
        FUNCTION("ScatterUpdate") {
            output = ScatterUpdate(output, {idxs}, key_states, minus_two);
        }
        std::string jsonFilePath = "./config/pass/json/generate_move_op_scatter_update.json";
        bool dumpJsonFlag = false;
        if (dumpJsonFlag) {
            auto programJson = Program::GetInstance().DumpJson();
            DumpJsonFile(programJson, jsonFilePath);
        }

        Function* originFunction = Program::GetInstance().GetFunctionByRawName("TENSOR_ScatterUpdate");
        Program testProgram(HostMachineMode::SERVER);
        GenerateMoveOp generateMoveOp;
        generateMoveOp.RunOnFunction(*originFunction);

        // ================== Verify Pass Effect ==================
        auto updatedOperations = Program::GetInstance().GetFunctionByRawName("TENSOR_ScatterUpdate")->Operations();
        constexpr int expectedOperations = 20;
        EXPECT_EQ(updatedOperations.size(), expectedOperations) << "total 16 operations";
        int assemble_num = 0;
        int view_num = 0;
        int copy_in_num = 0;
        int copy_out_num = 0;
        int index_outcast_num = 0;
        for (const auto &updatedOperation : updatedOperations) {
            switch (updatedOperation.GetOpcode()){
                case Opcode::OP_ASSEMBLE: {
                    assemble_num++;
                    break;
                }
                case Opcode::OP_VIEW: {
                    view_num++;
                    break;
                }
                case Opcode::OP_COPY_IN: {
                    copy_in_num++;
                    break;
                }
                case Opcode::OP_COPY_OUT: {
                    copy_out_num++;
                    break;
                }
                case Opcode::OP_INDEX_OUTCAST: {
                    index_outcast_num++;
                    break;
                }
                default: break;
            }
        }
        constexpr int expectedAssemble = 4;
        constexpr int expectedView = 4;
        constexpr int expectedCopyIn = 8;
        constexpr int expectedCopyOut = 0;
        EXPECT_EQ(assemble_num, expectedAssemble) << "4 operations should be OP_ASSEMBLE";
        EXPECT_EQ(assemble_num, index_outcast_num) << "num of OP_ASSEMBLE and OP_INDEX_OUTCAST should be equal";
        EXPECT_EQ(view_num, expectedView) << "0 operations should be OP_VIEW";
        EXPECT_EQ(copy_in_num, expectedCopyIn) << "8 operations should be OP_COPY_IN";
        EXPECT_EQ(copy_out_num, expectedCopyOut) << "0 operations should be OP_COPY_OUT";
    }
}

}
} // namespace npu::tile_fwk