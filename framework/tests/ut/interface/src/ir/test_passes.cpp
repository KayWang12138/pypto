/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_passes.cpp
 * \brief Unit tests for IR pass framework
 */

#include "gtest/gtest.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/dtype.h"
#include "core/error.h"
#include "ir/expr.h"
#include "ir/function.h"
#include "ir/program.h"
#include "ir/scalar_expr.h"
#include "ir/stmt.h"
#include "ir/transform/passes.h"
#include "ir/type.h"

namespace pypto {
namespace ir {

// Helper to create a simple program
static ProgramPtr MakeTestProgram(const std::string &funcName = "main") {
    auto bodyVal = std::make_shared<ConstInt>(0, DataType::INT32, Span::Unknown());
    auto body = std::make_shared<EvalStmt>(bodyVal, Span::Unknown());
    std::vector<VarPtr> params;
    std::vector<TypePtr> returnTypes;
    auto func = std::make_shared<Function>(funcName, params, returnTypes, body, Span::Unknown());
    std::vector<FunctionPtr> funcs = {func};
    return std::make_shared<Program>(funcs, "test", Span::Unknown());
}

class IRPassTest : public testing::Test {};

// ============================================================================
// CreateProgramPass Tests
// ============================================================================

TEST_F(IRPassTest, TestCreateProgramPassIdentity) {
    auto p = pass::CreateProgramPass([](const ProgramPtr &prog) -> ProgramPtr { return prog; }, "identity_pass");

    auto program = MakeTestProgram();
    auto result = p(program);
    ASSERT_NE(result, nullptr);
    ASSERT_EQ(result->name_, "test");
}

TEST_F(IRPassTest, TestCreateProgramPassTransform) {
    auto p = pass::CreateProgramPass(
        [](const ProgramPtr & /*prog*/) -> ProgramPtr {
            // Create a new program with different name
            auto bodyVal = std::make_shared<ConstInt>(1, DataType::INT32, Span::Unknown());
            auto body = std::make_shared<EvalStmt>(bodyVal, Span::Unknown());
            auto func = std::make_shared<Function>(
                "transformed", std::vector<VarPtr>{}, std::vector<TypePtr>{}, body, Span::Unknown());
            return std::make_shared<Program>(std::vector<FunctionPtr>{func}, "transformed_prog", Span::Unknown());
        },
        "transform_pass");

    auto program = MakeTestProgram();
    auto result = p(program);
    ASSERT_NE(result, nullptr);
    ASSERT_EQ(result->name_, "transformed_prog");
}

// ============================================================================
// CreateFunctionPass Tests
// ============================================================================

TEST_F(IRPassTest, TestCreateFunctionPassIdentity) {
    auto p = pass::CreateFunctionPass([](const FunctionPtr &func) -> FunctionPtr { return func; }, "func_identity");

    auto program = MakeTestProgram();
    auto result = p(program);
    ASSERT_NE(result, nullptr);
    ASSERT_EQ(result->functions_.size(), 1);
}

// ============================================================================
// Pass Operator Tests
// ============================================================================

TEST_F(IRPassTest, TestPassCallOperator) {
    auto p = pass::CreateProgramPass([](const ProgramPtr &prog) -> ProgramPtr { return prog; }, "test_pass");

    auto program = MakeTestProgram();
    auto result = p(program);
    ASSERT_NE(result, nullptr);
}

// ============================================================================
// Pass::run Tests
// ============================================================================

TEST_F(IRPassTest, TestPassRunMethod) {
    auto p = pass::CreateProgramPass([](const ProgramPtr &prog) -> ProgramPtr { return prog; }, "run_test");

    auto program = MakeTestProgram();
    auto result = p.run(program);
    ASSERT_NE(result, nullptr);
    ASSERT_EQ(result->name_, "test");
}

// ============================================================================
// Pass Copy/Move Tests
// ============================================================================

TEST_F(IRPassTest, TestPassCopy) {
    auto p1 = pass::CreateProgramPass([](const ProgramPtr &prog) -> ProgramPtr { return prog; }, "copy_test");

    auto p2 = p1; // Copy
    auto program = MakeTestProgram();
    auto result = p2(program);
    ASSERT_NE(result, nullptr);
}

TEST_F(IRPassTest, TestPassMove) {
    auto p1 = pass::CreateProgramPass([](const ProgramPtr &prog) -> ProgramPtr { return prog; }, "move_test");

    auto p2 = std::move(p1);
    auto program = MakeTestProgram();
    auto result = p2(program);
    ASSERT_NE(result, nullptr);
}

// ============================================================================
// FunctionPass Transform Tests
// ============================================================================

TEST_F(IRPassTest, TestFunctionPassTransform) {
    auto p = pass::CreateFunctionPass(
        [](const FunctionPtr &func) -> FunctionPtr {
            // Transform: create a new function with modified body
            auto newBodyVal = std::make_shared<ConstInt>(99, DataType::INT32, Span::Unknown());
            auto newBody = std::make_shared<EvalStmt>(newBodyVal, Span::Unknown());
            return std::make_shared<Function>(func->name_, func->params_, func->returnTypes_, newBody, func->span_);
        },
        "func_transform");

    auto program = MakeTestProgram();
    auto result = p(program);
    ASSERT_NE(result, nullptr);
    ASSERT_EQ(result->functions_.size(), 1);
}

TEST_F(IRPassTest, TestFunctionPassMultipleFunctions) {
    // Create program with two functions
    auto body1Val = std::make_shared<ConstInt>(1, DataType::INT32, Span::Unknown());
    auto body1 = std::make_shared<EvalStmt>(body1Val, Span::Unknown());
    auto func1 =
        std::make_shared<Function>("func1", std::vector<VarPtr>{}, std::vector<TypePtr>{}, body1, Span::Unknown());

    auto body2Val = std::make_shared<ConstInt>(2, DataType::INT32, Span::Unknown());
    auto body2 = std::make_shared<EvalStmt>(body2Val, Span::Unknown());
    auto func2 =
        std::make_shared<Function>("func2", std::vector<VarPtr>{}, std::vector<TypePtr>{}, body2, Span::Unknown());

    auto program = std::make_shared<Program>(std::vector<FunctionPtr>{func1, func2}, "multi", Span::Unknown());

    auto p = pass::CreateFunctionPass([](const FunctionPtr &func) -> FunctionPtr { return func; }, "multi_func");

    auto result = p(program);
    ASSERT_NE(result, nullptr);
    ASSERT_EQ(result->functions_.size(), 2);
}

// ============================================================================
// Pass with Empty Name Tests
// ============================================================================

TEST_F(IRPassTest, TestProgramPassEmptyName) {
    auto p = pass::CreateProgramPass([](const ProgramPtr &prog) -> ProgramPtr { return prog; }, "");

    auto program = MakeTestProgram();
    auto result = p(program);
    ASSERT_NE(result, nullptr);
}

TEST_F(IRPassTest, TestFunctionPassEmptyName) {
    auto p = pass::CreateFunctionPass([](const FunctionPtr &func) -> FunctionPtr { return func; }, "");

    auto program = MakeTestProgram();
    auto result = p(program);
    ASSERT_NE(result, nullptr);
}

// ============================================================================
// Default Pass Tests
// ============================================================================

TEST_F(IRPassTest, TestDefaultPassNullImpl) {
    Pass p; // Default constructor - null impl
    auto program = MakeTestProgram();
    ASSERT_THROW(p(program), InternalError);
}

TEST_F(IRPassTest, TestPassCopyAssignment) {
    auto p1 = pass::CreateProgramPass([](const ProgramPtr &prog) -> ProgramPtr { return prog; }, "assign_test");

    Pass p2;
    p2 = p1; // Copy assignment
    auto program = MakeTestProgram();
    auto result = p2(program);
    ASSERT_NE(result, nullptr);
}

TEST_F(IRPassTest, TestPassMoveAssignment) {
    auto p1 = pass::CreateProgramPass([](const ProgramPtr &prog) -> ProgramPtr { return prog; }, "move_assign");

    Pass p2;
    p2 = std::move(p1);
    auto program = MakeTestProgram();
    auto result = p2(program);
    ASSERT_NE(result, nullptr);
}

} // namespace ir
} // namespace pypto
