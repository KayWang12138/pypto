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
 * \file test_infer_memory_conflict.cpp
 * \brief Unit test for InferMemoryConflict pass.
 */

#include <gtest/gtest.h>
#include <vector>
#include <string>
#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "ut_json/ut_json_tool.h"
#include "passes/pass_mgr/pass_manager.h"
#include "interface/configs/config_manager.h"

#include "interface/operation/operation.h"
#include "passes/tensor_graph_pass/infer_memory_conflict.h"

using namespace npu::tile_fwk;

namespace npu {
namespace tile_fwk {

const int NUM_ZERO = 0;
const int NUM_ONE = 1;
const int NUM_2 = 2;
const int NUM_8 = 8;
const int NUM_32 = 32;
const int NUM_64 = 64;
const int NUM_128 = 128;
const int NUM_256 = 256;
const int NUM_512 = 512;
const int NUM_576 = 512 + 64;

class InferMemoryConflictTest : public ::testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);
        config::SetHostConfig(KEY_STRATEGY, "ReshapeTestStrategy");
        config::SetPlatformConfig("ENABLE_COST_MODEL", false);
        TileShape::Current().SetVecTile({64, 64});
    }
    void TearDown() override {}

    int CountCopyOp(Function &func) {
        int result = 0;
        for (auto &op : func.Operations()) {
            std::cout << op.GetOpcodeStr() << " " << op.GetOpMagic() << std::endl;
            if (op.GetOpcode() == Opcode::OP_REGISTER_COPY) {
                result++;
            }
        }
        return result;
    }
};

TEST_F(InferMemoryConflictTest, InferMemoryConflictUTest_View_Assemble) {
    auto currFunctionPtr = std::make_shared<Function>(
        Program::GetInstance(), "TestInferMemoryConflict", "TestInferMemoryConflict", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int64_t> shape1 = {NUM_32, NUM_128};
    std::vector<int64_t> shape2 = {NUM_32, NUM_128};
    auto inCast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    inCast->tensor->symbol = std::string("input");
    auto tensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    auto outCast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    outCast->tensor->symbol = std::string("output");

    auto view_attr = std::make_shared<ViewOpAttribute>(std::vector<int64_t>{NUM_ZERO, NUM_ZERO});
    auto assemble_attr = std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{NUM_ZERO, NUM_ZERO});
    auto &view_op = currFunctionPtr->AddOperation(Opcode::OP_VIEW, {inCast}, {tensor1});
    TileShape::Current().SetVecTile(shape1);
    view_op.UpdateTileShape(TileShape::Current());
    auto &assemble_op = currFunctionPtr->AddOperation(Opcode::OP_ASSEMBLE, {tensor1}, {outCast});
    view_op.SetOpAttribute(view_attr);
    assemble_op.SetOpAttribute(assemble_attr);
    assemble_op.UpdateTileShape(TileShape::Current());

    currFunctionPtr->inCasts_.push_back(inCast);
    currFunctionPtr->outCasts_.push_back(outCast);
    currFunctionPtr->SetGraphType(GraphType::TENSOR_GRAPH);

    // 确认构图完毕
    constexpr int opNumBefore = 2;
    EXPECT_EQ(currFunctionPtr->Operations().size(), opNumBefore) << opNumBefore << " operations before pass";
    std::cout << "Build Graph Done." << std::endl;

    InferMemoryConflict inferMemoryConflictPass;
    auto status = inferMemoryConflictPass.RunOnFunction(*currFunctionPtr);
    EXPECT_EQ(status, SUCCESS);

    constexpr int opNumAfter = 3;
    constexpr int copyOpNumExpect = 1;
    int copyOpNum = CountCopyOp(*currFunctionPtr);
    EXPECT_EQ(currFunctionPtr->Operations().size(), opNumAfter) << opNumAfter << " operations after pass";
    EXPECT_EQ(copyOpNum, copyOpNumExpect) << copyOpNumExpect << " copy op should be inserted after pass";
}

TEST_F(InferMemoryConflictTest, InferMemoryConflictUTest_View_Assemble_V2) {
    auto currFunctionPtr = std::make_shared<Function>(
        Program::GetInstance(), "TestInferMemoryConflict", "TestInferMemoryConflict", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int64_t> shape1 = {NUM_2, NUM_ONE, NUM_512, NUM_512};
    std::vector<int64_t> shape2 = {NUM_2, NUM_ONE, NUM_512, NUM_64};
    std::vector<int64_t> shape3 = {NUM_2, NUM_ONE, NUM_512, NUM_576};
    auto inCast1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    inCast1->tensor->symbol = std::string("input1");
    auto tensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    auto inCast2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    inCast2->tensor->symbol = std::string("input2");
    auto tensor2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    auto outCast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape3);
    outCast->tensor->symbol = std::string("output");

    auto view_attr1 = std::make_shared<ViewOpAttribute>(std::vector<int64_t>{NUM_ZERO, NUM_ZERO, NUM_ZERO, NUM_ZERO});
    auto view_attr2 = std::make_shared<ViewOpAttribute>(std::vector<int64_t>{NUM_ZERO, NUM_ZERO, NUM_ZERO, NUM_ZERO});
    auto assemble_attr1 =
        std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{NUM_ZERO, NUM_ZERO, NUM_ZERO, NUM_ZERO});
    auto assemble_attr2 =
        std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{NUM_ZERO, NUM_ZERO, NUM_ZERO, NUM_512});
    auto &view_op1 = currFunctionPtr->AddOperation(Opcode::OP_VIEW, {inCast1}, {tensor1});
    view_op1.SetOpAttribute(view_attr1);
    TileShape::Current().SetVecTile({2, 1, 1, NUM_512});
    view_op1.UpdateTileShape(TileShape::Current());
    auto &view_op2 = currFunctionPtr->AddOperation(Opcode::OP_VIEW, {inCast2}, {tensor2});
    view_op2.SetOpAttribute(view_attr2);
    TileShape::Current().SetVecTile({2, 1, 1, NUM_64});
    view_op2.UpdateTileShape(TileShape::Current());
    auto &assemble_op1 = currFunctionPtr->AddOperation(Opcode::OP_ASSEMBLE, {tensor1}, {outCast});
    assemble_op1.SetOpAttribute(assemble_attr1);
    TileShape::Current().SetVecTile({2, 1, 1, NUM_512});
    assemble_op1.UpdateTileShape(TileShape::Current());
    auto &assemble_op2 = currFunctionPtr->AddOperation(Opcode::OP_ASSEMBLE, {tensor2}, {outCast});
    assemble_op2.SetOpAttribute(assemble_attr2);
    TileShape::Current().SetVecTile({2, 1, 1, NUM_64});
    assemble_op1.UpdateTileShape(TileShape::Current());

    currFunctionPtr->inCasts_.push_back(inCast1);
    currFunctionPtr->inCasts_.push_back(inCast2);
    currFunctionPtr->outCasts_.push_back(outCast);
    currFunctionPtr->SetGraphType(GraphType::TENSOR_GRAPH);

    // 确认构图完毕
    constexpr int opNumBefore = 4;
    EXPECT_EQ(currFunctionPtr->Operations().size(), opNumBefore) << opNumBefore << " operations before pass";
    std::cout << "Build Graph Done." << std::endl;

    InferMemoryConflict inferMemoryConflictPass;
    auto status = inferMemoryConflictPass.RunOnFunction(*currFunctionPtr);
    EXPECT_EQ(status, SUCCESS);

    constexpr int opNumAfter = 6;
    constexpr int copyOpNumExpect = 2;
    int copyOpNum = CountCopyOp(*currFunctionPtr);
    EXPECT_EQ(currFunctionPtr->Operations().size(), opNumAfter) << opNumAfter << " operations after pass";
    EXPECT_EQ(copyOpNum, copyOpNumExpect) << copyOpNumExpect << " copy op should be inserted after pass";
}

TEST_F(InferMemoryConflictTest, InferMemoryConflictUTest_Reshape) {
    auto currFunctionPtr = std::make_shared<Function>(
        Program::GetInstance(), "TestInferMemoryConflict", "TestInferMemoryConflict", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int64_t> shape1 = {NUM_32, NUM_128};
    std::vector<int64_t> shape2 = {NUM_8, NUM_512};
    auto inCast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    inCast->tensor->symbol = std::string("input");
    auto tensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    auto tensor2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    auto outCast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    outCast->tensor->symbol = std::string("output");

    auto view_attr = std::make_shared<ViewOpAttribute>(std::vector<int64_t>{NUM_ZERO, NUM_ZERO});
    auto assemble_attr = std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{NUM_ZERO, NUM_ZERO});
    auto &view_op = currFunctionPtr->AddOperation(Opcode::OP_VIEW, {inCast}, {tensor1});
    auto &reshape_op = currFunctionPtr->AddOperation(Opcode::OP_RESHAPE, {tensor1}, {tensor2});
    TileShape::Current().SetVecTile({1, NUM_512});
    reshape_op.UpdateTileShape(TileShape::Current());
    auto &assemble_op = currFunctionPtr->AddOperation(Opcode::OP_ASSEMBLE, {tensor2}, {outCast});
    view_op.SetOpAttribute(view_attr);
    assemble_op.SetOpAttribute(assemble_attr);

    currFunctionPtr->inCasts_.push_back(inCast);
    currFunctionPtr->outCasts_.push_back(outCast);
    currFunctionPtr->SetGraphType(GraphType::TENSOR_GRAPH);

    // 确认构图完毕
    constexpr int opNumBefore = 3;
    EXPECT_EQ(currFunctionPtr->Operations().size(), opNumBefore) << opNumBefore << " operations before pass";
    std::cout << "Build Graph Done." << std::endl;

    InferMemoryConflict inferMemoryConflictPass;
    auto status = inferMemoryConflictPass.RunOnFunction(*currFunctionPtr);
    EXPECT_EQ(status, SUCCESS);

    constexpr int opNumAfter = 4;
    constexpr int copyOpNumExpect = 1;
    int copyOpNum = CountCopyOp(*currFunctionPtr);
    EXPECT_EQ(currFunctionPtr->Operations().size(), opNumAfter) << opNumAfter << " operations after pass";
    EXPECT_EQ(copyOpNum, copyOpNumExpect) << copyOpNumExpect << " copy op should be inserted after pass";
}

TEST_F(InferMemoryConflictTest, InferMemoryConflictUTest_ScatterUpdate) {
    auto currFunctionPtr = std::make_shared<Function>(
        Program::GetInstance(), "TestInferMemoryConflict", "TestInferMemoryConflict", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int64_t> shape1 = {NUM_2, NUM_ONE, NUM_512, NUM_576};
    std::vector<int64_t> shape2 = {NUM_2, NUM_ONE, NUM_ONE, NUM_576};
    std::vector<int64_t> shape3 = {NUM_2, NUM_ONE};
    auto inCast1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    inCast1->tensor->symbol = std::string("kv_cache");
    auto inCast2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    inCast2->tensor->symbol = std::string("update_tensor");
    auto inCast3 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape3);
    inCast3->tensor->symbol = std::string("index");
    auto tensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    auto tensor2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    auto tensor3 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape3);
    auto tensor4 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    auto outCast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    outCast->tensor->symbol = std::string("kv_cache");

    auto view_attr1 = std::make_shared<ViewOpAttribute>(std::vector<int64_t>{NUM_ZERO, NUM_ZERO, NUM_ZERO, NUM_ZERO});
    auto view_attr2 = std::make_shared<ViewOpAttribute>(std::vector<int64_t>{NUM_ZERO, NUM_ZERO, NUM_ZERO, NUM_ZERO});
    auto view_attr3 = std::make_shared<ViewOpAttribute>(std::vector<int64_t>{NUM_ZERO, NUM_ZERO});
    auto assemble_attr =
        std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{NUM_ZERO, NUM_ZERO, NUM_ZERO, NUM_ZERO});
    auto &view_op1 = currFunctionPtr->AddOperation(Opcode::OP_VIEW, {inCast1}, {tensor1});
    view_op1.SetOpAttribute(view_attr1);
    auto &view_op2 = currFunctionPtr->AddOperation(Opcode::OP_VIEW, {inCast2}, {tensor2});
    view_op2.SetOpAttribute(view_attr2);
    auto &view_op3 = currFunctionPtr->AddOperation(Opcode::OP_VIEW, {inCast3}, {tensor3});
    view_op3.SetOpAttribute(view_attr3);
    currFunctionPtr->AddOperation(Opcode::OP_INDEX_OUTCAST, {tensor2, tensor3, tensor1}, {tensor4});
    auto &assemble_op = currFunctionPtr->AddOperation(Opcode::OP_ASSEMBLE, {tensor4}, {outCast});
    assemble_op.SetOpAttribute(assemble_attr);

    currFunctionPtr->inCasts_.push_back(inCast1);
    currFunctionPtr->inCasts_.push_back(inCast2);
    currFunctionPtr->inCasts_.push_back(inCast3);
    currFunctionPtr->outCasts_.push_back(outCast);
    currFunctionPtr->SetGraphType(GraphType::TENSOR_GRAPH);

    // 确认构图完毕
    constexpr int opNumBefore = 5;
    EXPECT_EQ(currFunctionPtr->Operations().size(), opNumBefore) << opNumBefore << " operations before pass";
    std::cout << "Build Graph Done." << std::endl;

    InferMemoryConflict inferMemoryConflictPass;
    auto status = inferMemoryConflictPass.RunOnFunction(*currFunctionPtr);
    EXPECT_EQ(status, SUCCESS);

    constexpr int opNumAfter = 5;
    constexpr int copyOpNumExpect = 0;
    int copyOpNum = CountCopyOp(*currFunctionPtr);
    EXPECT_EQ(currFunctionPtr->Operations().size(), opNumAfter) << opNumAfter << " operations after pass";
    EXPECT_EQ(copyOpNum, copyOpNumExpect) << copyOpNumExpect << " copy op should be inserted after pass";
}

TEST_F(InferMemoryConflictTest, InferMemoryConflictUTest_ScatterUpdate_Reshape) {
    auto currFunctionPtr = std::make_shared<Function>(
        Program::GetInstance(), "TestInferMemoryConflict", "TestInferMemoryConflict", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int64_t> shape1 = {NUM_2, NUM_ONE, NUM_512, NUM_576};
    std::vector<int64_t> shape2 = {NUM_2, NUM_ONE, NUM_ONE, NUM_576};
    std::vector<int64_t> shape3 = {NUM_2, NUM_ONE};
    std::vector<int64_t> shape4 = {NUM_2 * NUM_ONE * NUM_512, NUM_576};
    auto inCast1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    inCast1->tensor->symbol = std::string("kv_cache");
    auto inCast2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    inCast2->tensor->symbol = std::string("update_tensor");
    auto inCast3 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape3);
    inCast3->tensor->symbol = std::string("index");
    auto tensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    auto tensor2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    auto tensor3 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape3);
    auto tensor4 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    auto tensor5 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape4);
    auto outCast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape4);
    outCast->tensor->symbol = std::string("output");

    auto view_attr1 = std::make_shared<ViewOpAttribute>(std::vector<int64_t>{NUM_ZERO, NUM_ZERO, NUM_ZERO, NUM_ZERO});
    auto view_attr2 = std::make_shared<ViewOpAttribute>(std::vector<int64_t>{NUM_ZERO, NUM_ZERO, NUM_ZERO, NUM_ZERO});
    auto view_attr3 = std::make_shared<ViewOpAttribute>(std::vector<int64_t>{NUM_ZERO, NUM_ZERO});
    auto assemble_attr = std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{NUM_ZERO, NUM_ZERO});
    auto &view_op1 = currFunctionPtr->AddOperation(Opcode::OP_VIEW, {inCast1}, {tensor1});
    view_op1.SetOpAttribute(view_attr1);
    auto &view_op2 = currFunctionPtr->AddOperation(Opcode::OP_VIEW, {inCast2}, {tensor2});
    view_op2.SetOpAttribute(view_attr2);
    auto &view_op3 = currFunctionPtr->AddOperation(Opcode::OP_VIEW, {inCast3}, {tensor3});
    view_op3.SetOpAttribute(view_attr3);
    currFunctionPtr->AddOperation(Opcode::OP_INDEX_OUTCAST, {tensor2, tensor3, tensor1}, {tensor4});
    auto &reshape_op = currFunctionPtr->AddOperation(Opcode::OP_RESHAPE, {tensor4}, {tensor5});
    TileShape::Current().SetVecTile({NUM_2, NUM_512});
    reshape_op.UpdateTileShape(TileShape::Current());
    auto &assemble_op = currFunctionPtr->AddOperation(Opcode::OP_ASSEMBLE, {tensor5}, {outCast});
    assemble_op.SetOpAttribute(assemble_attr);

    currFunctionPtr->inCasts_.push_back(inCast1);
    currFunctionPtr->inCasts_.push_back(inCast2);
    currFunctionPtr->inCasts_.push_back(inCast3);
    currFunctionPtr->outCasts_.push_back(outCast);
    currFunctionPtr->SetGraphType(GraphType::TENSOR_GRAPH);

    // 确认构图完毕
    constexpr int opNumBefore = 6;
    EXPECT_EQ(currFunctionPtr->Operations().size(), opNumBefore) << opNumBefore << " operations before pass";
    std::cout << "Build Graph Done." << std::endl;

    InferMemoryConflict inferMemoryConflictPass;
    auto status = inferMemoryConflictPass.RunOnFunction(*currFunctionPtr);
    EXPECT_EQ(status, SUCCESS);

    constexpr int opNumAfter = 7;
    constexpr int copyOpNumExpect = 1;
    int copyOpNum = CountCopyOp(*currFunctionPtr);
    EXPECT_EQ(currFunctionPtr->Operations().size(), opNumAfter) << opNumAfter << " operations after pass";
    EXPECT_EQ(copyOpNum, copyOpNumExpect) << copyOpNumExpect << " copy op should be inserted after pass";
}

TEST_F(InferMemoryConflictTest, InferMemoryConflictUTest_Reshape_ScatterUpdate_Reshape) {
    auto currFunctionPtr = std::make_shared<Function>(
        Program::GetInstance(), "TestInferMemoryConflict", "TestInferMemoryConflict", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int64_t> shape1 = {NUM_2, NUM_ONE, NUM_512, NUM_576}; // [2, 1, 512, 576]
    std::vector<int64_t> shape2 = {NUM_2, NUM_ONE, NUM_ONE, NUM_576}; // [2, 1, 1, 576]
    std::vector<int64_t> shape3 = {NUM_2, NUM_ONE};
    std::vector<int64_t> shape4 = {NUM_2 * NUM_ONE * NUM_512, NUM_576}; // [1024, 576]
    std::vector<int64_t> shape5 = {NUM_2 * NUM_ONE * NUM_ONE, NUM_576}; // [2, 576]
    auto inCast1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape4);
    inCast1->tensor->symbol = std::string("kv_cache");
    auto inCast2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape5);
    inCast2->tensor->symbol = std::string("update_tensor");
    auto inCast3 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape3);
    inCast3->tensor->symbol = std::string("index");
    auto tensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape4);
    auto tensor1_reshape = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    auto tensor2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape5);
    auto tensor2_reshape = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    auto tensor3 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape3);
    auto tensor4 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    auto tensor5 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape4);
    auto outCast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape4);
    outCast->tensor->symbol = std::string("output");

    auto view_attr1 = std::make_shared<ViewOpAttribute>(std::vector<int64_t>{NUM_ZERO, NUM_ZERO});
    auto view_attr2 = std::make_shared<ViewOpAttribute>(std::vector<int64_t>{NUM_ZERO, NUM_ZERO});
    auto view_attr3 = std::make_shared<ViewOpAttribute>(std::vector<int64_t>{NUM_ZERO, NUM_ZERO});
    auto assemble_attr = std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{NUM_ZERO, NUM_ZERO});
    auto &view_op1 = currFunctionPtr->AddOperation(Opcode::OP_VIEW, {inCast1}, {tensor1});
    view_op1.SetOpAttribute(view_attr1);
    currFunctionPtr->AddOperation(Opcode::OP_RESHAPE, {tensor1}, {tensor1_reshape});
    auto &view_op2 = currFunctionPtr->AddOperation(Opcode::OP_VIEW, {inCast2}, {tensor2});
    view_op2.SetOpAttribute(view_attr2);
    currFunctionPtr->AddOperation(Opcode::OP_RESHAPE, {tensor2}, {tensor2_reshape});
    auto &view_op3 = currFunctionPtr->AddOperation(Opcode::OP_VIEW, {inCast3}, {tensor3});
    view_op3.SetOpAttribute(view_attr3);
    currFunctionPtr->AddOperation(Opcode::OP_INDEX_OUTCAST, {tensor2_reshape, tensor3, tensor1_reshape}, {tensor4});
    auto &reshape_op = currFunctionPtr->AddOperation(Opcode::OP_RESHAPE, {tensor4}, {tensor5});
    TileShape::Current().SetVecTile({NUM_2, NUM_512});
    reshape_op.UpdateTileShape(TileShape::Current());
    auto &assemble_op = currFunctionPtr->AddOperation(Opcode::OP_ASSEMBLE, {tensor5}, {outCast});
    assemble_op.SetOpAttribute(assemble_attr);

    currFunctionPtr->inCasts_.push_back(inCast1);
    currFunctionPtr->inCasts_.push_back(inCast2);
    currFunctionPtr->inCasts_.push_back(inCast3);
    currFunctionPtr->outCasts_.push_back(outCast);
    currFunctionPtr->SetGraphType(GraphType::TENSOR_GRAPH);

    // 确认构图完毕
    constexpr int opNumBefore = 8;
    EXPECT_EQ(currFunctionPtr->Operations().size(), opNumBefore) << opNumBefore << " operations before pass";
    std::cout << "Build Graph Done." << std::endl;

    InferMemoryConflict inferMemoryConflictPass;
    auto status = inferMemoryConflictPass.RunOnFunction(*currFunctionPtr);
    EXPECT_EQ(status, SUCCESS);

    constexpr int opNumAfter = 9;
    constexpr int copyOpNumExpect = 1;
    int copyOpNum = CountCopyOp(*currFunctionPtr);
    EXPECT_EQ(currFunctionPtr->Operations().size(), opNumAfter) << opNumAfter << " operations after pass";
    EXPECT_EQ(copyOpNum, copyOpNumExpect) << copyOpNumExpect << " copy op should be inserted after pass";
}
} // namespace tile_fwk
} // namespace npu
