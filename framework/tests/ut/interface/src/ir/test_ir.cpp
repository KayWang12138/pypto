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
 * \file test_ir.cpp
 * \brief
 */

#include "gtest/gtest.h"
#include "ir/utils_defop.h"
#include "ir/opcode.h"
#include "ir/builder/ir_builder.h"
#include "ir/builder/ir_context.h"
#include "ir/statement.h"
#include "ir/tile_graph.h"

using namespace pto;

class IRTest : public testing::Test {
public:
    void SetUp() override {
    }

    void TearDown() override {
    }
};

static_assert(MAP_SIZE(a, b, c) == 3, "Invalid MAP_SIZE 3");
static_assert(MAP_SIZE(a, b, c, d, e, f, g, h, a, b, c, d, e, f, g, h,
                       a, b, c, d, e, f, g, h, a, b, c, d, e, f, g, h) == 32, "Invalid MAP_SIZE 32");

TEST_F(IRTest, TestUtils) {
    EXPECT_EQ(std::vector<int>({2, 3, 4}), std::vector<int>({
#define ADD_1(n) (n) + 1,
        MAP(ADD_1, 1, 2, 3)
    }));
}

TEST_F(IRTest, TestOpcode) {
    EXPECT_EQ("OP_SCALAR_NEG", GetOpcodeName(Opcode::OP_SCALAR_NEG));
    EXPECT_EQ("", GetOpcodeName(Opcode::OP_INVALID));
}

TEST_F(IRTest, TestClass) {
    ScalarValuePtr lhs = std::make_shared<ScalarValue>(int64_t{2});
    ScalarValuePtr rhs = std::make_shared<ScalarValue>(int64_t{4});
    ScalarValuePtr out = std::make_shared<ScalarValue>(DataType::INT64, "aaa");
    BinaryScalarOpPtr op = std::make_shared<BinaryScalarOp>(Opcode::OP_SCALAR_ADD, rhs, lhs, out);
    EXPECT_EQ(2, op->GetNumInputOperand());
    EXPECT_EQ(1, op->GetNumOutputOperand());
}

TEST_F(IRTest, TestIRBuilder) {
    auto module = std::make_shared<ProgramModule>("main");
    IRBuilder builder;
    IRBuilderContext ctx;

    auto func = builder.CreateFunction("bbb", FunctionKind::Block, FunctionSignature());
    module->AddFunction(func);
    module->SetProgramEntry(func);
    builder.EnterFunctionBody(ctx, func);

    ScalarValuePtr lhs = builder.CreateConst(ctx, int64_t{2});
    ScalarValuePtr rhs = builder.CreateConst(ctx, int64_t{4});
    ScalarValuePtr out = builder.CreateScalar(ctx, DataType::INT64, "aaa");

    std::vector<ScalarValuePtr> dataList;
    BinaryScalarOpPtr op = builder.CreateBinaryScalarOp(Opcode::OP_SCALAR_ADD, lhs, rhs, out);
    builder.Emit(ctx, op);
    EXPECT_EQ(2, op->GetNumInputOperand());
    EXPECT_EQ(1, op->GetNumOutputOperand());

    ctx.PopScope();
}

TEST_F(IRTest, TestAttributeSet) {
    // Test attribute registration and setting with type checking
    auto module = std::make_shared<ProgramModule>("test_module");

    // Test setting valid attributes on ProgramModule
    EXPECT_TRUE(module->SetAttr("arch", std::string("ascend910")));
    EXPECT_TRUE(module->SetAttr("tile_default", std::string("16x16")));
    EXPECT_TRUE(module->SetAttr("enable_debug", std::string("true")));
    EXPECT_TRUE(module->SetAttr("test_type", std::string("unit_test")));

    // Test reading back the attributes
    EXPECT_TRUE(module->HasAttr("arch"));
    EXPECT_EQ(std::get<std::string>(module->GetAttr("arch")), "ascend910");
    EXPECT_EQ(std::get<std::string>(module->GetAttr("tile_default")), "16x16");

    // Test setting unregistered attribute (should fail)
    EXPECT_FALSE(module->SetAttr("invalid_key", std::string("value")));

    // Test type mismatch (should fail)
    EXPECT_FALSE(module->SetAttr("arch", int64_t(123)));

    // Test ForStatement attribute with int64_t type
    IRBuilder builder;
    IRBuilderContext ctx;
    auto func = builder.CreateFunction("test_func", FunctionKind::Block, FunctionSignature());
    builder.EnterFunctionBody(ctx, func);

    auto start = builder.CreateConst(ctx, int64_t{0});
    auto end = builder.CreateConst(ctx, int64_t{10});
    auto step = builder.CreateConst(ctx, int64_t{1});
    auto loopVar = builder.CreateScalar(ctx, DataType::INT64, "i");

    auto forStmt = builder.CreateForStmt(ctx, loopVar, start, end, step);

    // Test setting unroll attribute (int64_t type)
    EXPECT_TRUE(forStmt->SetAttr("unroll", int64_t(4)));
    EXPECT_TRUE(forStmt->HasAttr("unroll"));
    EXPECT_EQ(std::get<int64_t>(forStmt->GetAttr("unroll")), 4);

    // Test type mismatch for ForStatement (should fail)
    EXPECT_FALSE(forStmt->SetAttr("unroll", std::string("4")));

    // Test Value attributes (string type)
    ScalarValuePtr scalarVal = builder.CreateScalar(ctx, DataType::INT64, "test_scalar");
    EXPECT_TRUE(scalarVal->SetAttr("io", std::string("input")));
    EXPECT_TRUE(scalarVal->HasAttr("io"));
    EXPECT_EQ(std::get<std::string>(scalarVal->GetAttr("io")), "input");

    ctx.PopScope();
}

TEST_F(IRTest, TestPropertyRead) {
    // Test reading operation properties (INPUTS_MEM, OUTPUTS_MEM, PIPE, etc.)
    IRBuilder builder;
    IRBuilderContext ctx;

    auto func = builder.CreateFunction("test_func", FunctionKind::Block, FunctionSignature());
    builder.EnterFunctionBody(ctx, func);

    // Create UnaryOp which has PROPERTY defined
    auto input = builder.CreateTile(ctx, {16, 16}, DataType::FP16, "input");
    auto output = builder.CreateTile(ctx, {16, 16}, DataType::FP16, "output");
    auto unaryOp = builder.CreateUnaryOp(Opcode::OP_ABS, input, output);

    // Get operation property
    OpClassInfo classInfo = GetOpClassInfo(unaryOp);

    // Verify INPUTS_MEM(UB) - 1 input
    EXPECT_EQ(classInfo.inputsMemType_.size(), 1);
    EXPECT_EQ(classInfo.inputsMemType_[0], MemSpaceKind::UB);

    // Verify OUTPUTS_MEM(UB) - 1 output
    EXPECT_EQ(classInfo.outputsMemType_.size(), 1);
    EXPECT_EQ(classInfo.outputsMemType_[0], MemSpaceKind::UB);

    // Verify PIPE_V, PIPE_V (start and end pipe)
    EXPECT_EQ(classInfo.pipeIdStart_, PIPE_V);
    EXPECT_EQ(classInfo.pipeIdEnd_, PIPE_V);

    // Verify CoreType AIV
    EXPECT_EQ(classInfo.coreType_, CoreType::AIV);

    // Verify CalcType ELMWISE
    EXPECT_EQ(classInfo.calcType_, OpCalcType::ELMWISE);

    // Test CompareOp which has 2 inputs and 2 outputs
    auto input1 = builder.CreateTile(ctx, {16, 16}, DataType::FP16, "input1");
    auto input2 = builder.CreateTile(ctx, {16, 16}, DataType::FP16, "input2");
    auto cmpOutput = builder.CreateTile(ctx, {16, 16}, DataType::BOOL, "cmp_output");
    auto tempTensor = builder.CreateTile(ctx, {16, 16}, DataType::FP16, "temp");

    // Create CompareOp manually since it has attributes
    auto compareOp = std::make_shared<CompareOp>(
        Opcode::OP_CMP,
        input1, input2, cmpOutput, tempTensor,
        static_cast<int>(CmpOperationType::EQ),
        static_cast<int>(CmpModeType::BOOL)
    );

    OpClassInfo cmpClassInfo = GetOpClassInfo(compareOp);

    // Verify INPUTS_MEM(UB, UB) - 2 inputs
    EXPECT_EQ(cmpClassInfo.inputsMemType_.size(), 2);
    EXPECT_EQ(cmpClassInfo.inputsMemType_[0], MemSpaceKind::UB);
    EXPECT_EQ(cmpClassInfo.inputsMemType_[1], MemSpaceKind::UB);

    // Verify OUTPUTS_MEM(UB, UB) - 2 outputs
    EXPECT_EQ(cmpClassInfo.outputsMemType_.size(), 2);
    EXPECT_EQ(cmpClassInfo.outputsMemType_[0], MemSpaceKind::UB);
    EXPECT_EQ(cmpClassInfo.outputsMemType_[1], MemSpaceKind::UB);

    // Verify other properties
    EXPECT_EQ(cmpClassInfo.pipeIdStart_, PIPE_S);
    EXPECT_EQ(cmpClassInfo.pipeIdEnd_, PIPE_S);
    EXPECT_EQ(cmpClassInfo.coreType_, CoreType::AIV);
    EXPECT_EQ(cmpClassInfo.calcType_, OpCalcType::ELMWISE);

    ctx.PopScope();
}