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
 * \file test_builder.cpp
 * \brief Unit tests for IR builder
 */

#include "gtest/gtest.h"

#include <memory>
#include <string>
#include <vector>

#include "core/dtype.h"
#include "ir/builder.h"
#include "ir/expr.h"
#include "ir/scalar_expr.h"
#include "ir/type.h"

namespace pypto {
namespace ir {

class IRBuilderTest : public testing::Test {};

// ============================================================================
// IRBuilder Context Tests
// ============================================================================

TEST_F(IRBuilderTest, TestInitialState) {
  IRBuilder builder;
  ASSERT_FALSE(builder.InFunction());
  ASSERT_FALSE(builder.InLoop());
  ASSERT_FALSE(builder.InIf());
  ASSERT_FALSE(builder.InProgram());
}

TEST_F(IRBuilderTest, TestBeginEndFunction) {
  IRBuilder builder;
  auto span = Span::unknown();

  builder.BeginFunction("test_func", span);
  ASSERT_TRUE(builder.InFunction());

  builder.ReturnType(std::make_shared<ScalarType>(DataType::INT32));
  auto val = std::make_shared<ConstInt>(0, DataType::INT32, span);
  builder.Return({val}, span);

  auto func = builder.EndFunction(span);
  ASSERT_NE(func, nullptr);
  ASSERT_EQ(func->name_, "test_func");
  ASSERT_FALSE(builder.InFunction());
}

TEST_F(IRBuilderTest, TestBeginEndProgram) {
  IRBuilder builder;
  auto span = Span::unknown();

  builder.BeginProgram("test_prog", span);
  ASSERT_TRUE(builder.InProgram());

  builder.BeginFunction("main", span);
  builder.ReturnType(std::make_shared<ScalarType>(DataType::INT32));
  auto val = std::make_shared<ConstInt>(0, DataType::INT32, span);
  builder.Return({val}, span);
  builder.EndFunction(span);

  auto program = builder.EndProgram(span);
  ASSERT_NE(program, nullptr);
  ASSERT_EQ(program->name_, "test_prog");
  ASSERT_FALSE(builder.InProgram());
}

// ============================================================================
// IRBuilder Variable Tests
// ============================================================================

TEST_F(IRBuilderTest, TestCreateVar) {
  IRBuilder builder;
  auto span = Span::unknown();
  auto int_type = std::make_shared<ScalarType>(DataType::INT32);
  auto var = builder.Var("x", int_type, span);

  ASSERT_NE(var, nullptr);
  ASSERT_EQ(var->name_, "x");
}

// ============================================================================
// IRBuilder Statement Tests
// ============================================================================

TEST_F(IRBuilderTest, TestAssign) {
  IRBuilder builder;
  auto span = Span::unknown();
  auto int_type = std::make_shared<ScalarType>(DataType::INT32);

  builder.BeginFunction("test", span);
  builder.ReturnType(int_type);

  auto var = builder.Var("x", int_type, span);
  auto val = std::make_shared<ConstInt>(42, DataType::INT32, span);
  builder.Assign(var, val, span);

  auto ret_val = std::make_shared<ConstInt>(0, DataType::INT32, span);
  builder.Return({ret_val}, span);

  auto func = builder.EndFunction(span);
  ASSERT_NE(func, nullptr);
  ASSERT_NE(func->body_, nullptr);
}

// ============================================================================
// IRBuilder ForLoop Tests
// ============================================================================

TEST_F(IRBuilderTest, TestBeginEndForLoop) {
  IRBuilder builder;
  auto span = Span::unknown();
  auto int_type = std::make_shared<ScalarType>(DataType::INT32);

  builder.BeginFunction("test", span);
  builder.ReturnType(int_type);

  auto loop_var = builder.Var("i", int_type, span);
  auto start = std::make_shared<ConstInt>(0, DataType::INT32, span);
  auto stop = std::make_shared<ConstInt>(10, DataType::INT32, span);
  auto step = std::make_shared<ConstInt>(1, DataType::INT32, span);

  builder.BeginForLoop(loop_var, start, stop, step, span);
  ASSERT_TRUE(builder.InLoop());

  builder.EndForLoop(span);
  ASSERT_FALSE(builder.InLoop());

  auto ret_val = std::make_shared<ConstInt>(0, DataType::INT32, span);
  builder.Return({ret_val}, span);

  auto func = builder.EndFunction(span);
  ASSERT_NE(func, nullptr);
}

// ============================================================================
// IRBuilder If Tests
// ============================================================================

TEST_F(IRBuilderTest, TestBeginEndIf) {
  IRBuilder builder;
  auto span = Span::unknown();
  auto int_type = std::make_shared<ScalarType>(DataType::INT32);

  builder.BeginFunction("test", span);
  builder.ReturnType(int_type);

  auto cond = std::make_shared<ConstBool>(true, span);
  builder.BeginIf(cond, span);
  ASSERT_TRUE(builder.InIf());

  builder.EndIf(span);
  ASSERT_FALSE(builder.InIf());

  auto ret_val = std::make_shared<ConstInt>(0, DataType::INT32, span);
  builder.Return({ret_val}, span);

  auto func = builder.EndFunction(span);
  ASSERT_NE(func, nullptr);
}

TEST_F(IRBuilderTest, TestBeginIfElseEndIf) {
  IRBuilder builder;
  auto span = Span::unknown();
  auto int_type = std::make_shared<ScalarType>(DataType::INT32);

  builder.BeginFunction("test", span);
  builder.ReturnType(int_type);

  auto cond = std::make_shared<ConstBool>(true, span);
  builder.BeginIf(cond, span);
  builder.BeginElse(span);
  builder.EndIf(span);

  auto ret_val = std::make_shared<ConstInt>(0, DataType::INT32, span);
  builder.Return({ret_val}, span);

  auto func = builder.EndFunction(span);
  ASSERT_NE(func, nullptr);
}

}  // namespace ir
}  // namespace pypto
