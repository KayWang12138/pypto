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
#include <vector>

#include "core/dtype.h"
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
static ProgramPtr MakeTestProgram(const std::string& func_name = "main") {
  auto body_val = std::make_shared<ConstInt>(0, DataType::INT32, Span::unknown());
  auto body = std::make_shared<EvalStmt>(body_val, Span::unknown());
  std::vector<VarPtr> params;
  std::vector<TypePtr> return_types;
  auto func = std::make_shared<Function>(func_name, params, return_types, body, Span::unknown());
  std::vector<FunctionPtr> funcs = {func};
  return std::make_shared<Program>(funcs, "test", Span::unknown());
}

class IRPassTest : public testing::Test {};

// ============================================================================
// CreateProgramPass Tests
// ============================================================================

TEST_F(IRPassTest, TestCreateProgramPassIdentity) {
  auto p = pass::CreateProgramPass([](const ProgramPtr& prog) -> ProgramPtr { return prog; }, "identity_pass");

  auto program = MakeTestProgram();
  auto result = p(program);
  ASSERT_NE(result, nullptr);
  ASSERT_EQ(result->name_, "test");
}

TEST_F(IRPassTest, TestCreateProgramPassTransform) {
  auto p = pass::CreateProgramPass(
      [](const ProgramPtr& /*prog*/) -> ProgramPtr {
        // Create a new program with different name
        auto body_val = std::make_shared<ConstInt>(1, DataType::INT32, Span::unknown());
        auto body = std::make_shared<EvalStmt>(body_val, Span::unknown());
        auto func = std::make_shared<Function>("transformed", std::vector<VarPtr>{}, std::vector<TypePtr>{},
                                               body, Span::unknown());
        return std::make_shared<Program>(std::vector<FunctionPtr>{func}, "transformed_prog", Span::unknown());
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
  auto p = pass::CreateFunctionPass([](const FunctionPtr& func) -> FunctionPtr { return func; }, "func_identity");

  auto program = MakeTestProgram();
  auto result = p(program);
  ASSERT_NE(result, nullptr);
  ASSERT_EQ(result->functions_.size(), 1);
}

// ============================================================================
// Pass Operator Tests
// ============================================================================

TEST_F(IRPassTest, TestPassCallOperator) {
  auto p = pass::CreateProgramPass([](const ProgramPtr& prog) -> ProgramPtr { return prog; }, "test_pass");

  auto program = MakeTestProgram();
  auto result = p(program);
  ASSERT_NE(result, nullptr);
}

}  // namespace ir
}  // namespace pypto
