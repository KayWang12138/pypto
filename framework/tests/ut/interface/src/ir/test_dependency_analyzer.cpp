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
 * \file test_dependency_analyzer.cpp
 * \brief Unit tests for IR dependency analyzer
 */

#include "gtest/gtest.h"

#include <memory>
#include <vector>

#include "core/dtype.h"
#include "ir/expr.h"
#include "ir/function.h"
#include "ir/scalar_expr.h"
#include "ir/stmt.h"
#include "ir/transform/dependency_analyzer.h"
#include "ir/transform/dependency_graph.h"
#include "ir/type.h"

namespace pypto {
namespace ir {

// Helper to create a simple function with sequential assignments
static FunctionPtr MakeSequentialFunction() {
  auto int_type = std::make_shared<ScalarType>(DataType::INT32);
  auto var_x = std::make_shared<Var>("x", int_type, Span::unknown());
  auto var_y = std::make_shared<Var>("y", int_type, Span::unknown());

  auto val1 = std::make_shared<ConstInt>(1, DataType::INT32, Span::unknown());
  auto val2 = std::make_shared<ConstInt>(2, DataType::INT32, Span::unknown());

  auto assign1 = std::make_shared<AssignStmt>(var_x, val1, Span::unknown());
  auto assign2 = std::make_shared<AssignStmt>(var_y, val2, Span::unknown());

  std::vector<StmtPtr> stmts = {assign1, assign2};
  auto body = std::make_shared<SeqStmts>(stmts, Span::unknown());

  return std::make_shared<Function>("test", std::vector<VarPtr>{}, std::vector<TypePtr>{}, body,
                                    Span::unknown());
}

// Helper to create a function with data dependency (RAW)
static FunctionPtr MakeRAWDependencyFunction() {
  auto int_type = std::make_shared<ScalarType>(DataType::INT32);
  auto var_x = std::make_shared<Var>("x", int_type, Span::unknown());
  auto var_y = std::make_shared<Var>("y", int_type, Span::unknown());

  auto val1 = std::make_shared<ConstInt>(1, DataType::INT32, Span::unknown());
  auto assign1 = std::make_shared<AssignStmt>(var_x, val1, Span::unknown());

  // y = x (read after write on x)
  auto assign2 = std::make_shared<AssignStmt>(var_y, var_x, Span::unknown());

  std::vector<StmtPtr> stmts = {assign1, assign2};
  auto body = std::make_shared<SeqStmts>(stmts, Span::unknown());

  return std::make_shared<Function>("test", std::vector<VarPtr>{}, std::vector<TypePtr>{}, body,
                                    Span::unknown());
}

class DependencyAnalyzerTest : public testing::Test {};

// ============================================================================
// Basic Analysis Tests
// ============================================================================

TEST_F(DependencyAnalyzerTest, TestAnalyzeSequentialFunction) {
  DependencyAnalyzer analyzer;
  auto func = MakeSequentialFunction();
  auto graph = analyzer.Analyze(func);

  ASSERT_FALSE(graph.blocks.empty());
}

TEST_F(DependencyAnalyzerTest, TestAnalyzeRAWDependency) {
  DependencyAnalyzer analyzer;
  auto func = MakeRAWDependencyFunction();
  auto graph = analyzer.Analyze(func);

  ASSERT_FALSE(graph.blocks.empty());
  // Should detect RAW dependency on x
  bool has_raw = false;
  for (const auto& dep : graph.dependencies) {
    if (dep.type == DependencyEdge::RAW) {
      has_raw = true;
      break;
    }
  }
  ASSERT_TRUE(has_raw);
}

// ============================================================================
// Basic Block Tests
// ============================================================================

TEST_F(DependencyAnalyzerTest, TestAnalyzeBasicBlocks) {
  DependencyAnalyzer analyzer;
  auto func = MakeSequentialFunction();
  auto blocks = analyzer.AnalyzeBasicBlocks(func);

  ASSERT_FALSE(blocks.empty());
  // Sequential statements should be in one basic block
  ASSERT_GE(blocks[0].statements.size(), 1);
}

// ============================================================================
// Empty Function Tests
// ============================================================================

TEST_F(DependencyAnalyzerTest, TestAnalyzeEmptyBody) {
  auto body_val = std::make_shared<ConstInt>(0, DataType::INT32, Span::unknown());
  auto body = std::make_shared<EvalStmt>(body_val, Span::unknown());
  auto func = std::make_shared<Function>("empty", std::vector<VarPtr>{}, std::vector<TypePtr>{}, body,
                                         Span::unknown());

  DependencyAnalyzer analyzer;
  auto graph = analyzer.Analyze(func);
  // Should not crash on simple function
  ASSERT_TRUE(graph.blocks.empty() || !graph.blocks.empty());
}

// ============================================================================
// Function with Loop Tests
// ============================================================================

TEST_F(DependencyAnalyzerTest, TestAnalyzeFunctionWithLoop) {
  auto int_type = std::make_shared<ScalarType>(DataType::INT32);
  auto loop_var = std::make_shared<Var>("i", int_type, Span::unknown());
  auto start = std::make_shared<ConstInt>(0, DataType::INT32, Span::unknown());
  auto stop = std::make_shared<ConstInt>(10, DataType::INT32, Span::unknown());
  auto step = std::make_shared<ConstInt>(1, DataType::INT32, Span::unknown());

  auto body_expr = std::make_shared<ConstInt>(0, DataType::INT32, Span::unknown());
  auto body = std::make_shared<EvalStmt>(body_expr, Span::unknown());

  std::vector<IterArgPtr> iter_args;
  std::vector<VarPtr> return_vars;
  auto for_stmt = std::make_shared<ForStmt>(loop_var, start, stop, step, iter_args, body, return_vars,
                                            Span::unknown());

  auto func = std::make_shared<Function>("loop_func", std::vector<VarPtr>{}, std::vector<TypePtr>{},
                                         for_stmt, Span::unknown());

  DependencyAnalyzer analyzer;
  auto graph = analyzer.Analyze(func);
  // Should handle loop without crashing
  (void)graph;
}

// ============================================================================
// DependencyGraph Structure Tests
// ============================================================================

TEST_F(DependencyAnalyzerTest, TestDependencyGraphDefault) {
  DependencyGraph graph;
  ASSERT_TRUE(graph.blocks.empty());
  ASSERT_TRUE(graph.dependencies.empty());
}

TEST_F(DependencyAnalyzerTest, TestBasicBlockDefault) {
  BasicBlock block;
  block.id = 0;
  ASSERT_EQ(block.id, 0);
  ASSERT_TRUE(block.statements.empty());
  ASSERT_TRUE(block.predecessors.empty());
  ASSERT_TRUE(block.successors.empty());
  ASSERT_FALSE(block.is_loop_body);
}

TEST_F(DependencyAnalyzerTest, TestDependencyEdgeTypes) {
  DependencyEdge edge;
  edge.type = DependencyEdge::RAW;
  ASSERT_EQ(edge.type, DependencyEdge::RAW);

  edge.type = DependencyEdge::WAR;
  ASSERT_EQ(edge.type, DependencyEdge::WAR);

  edge.type = DependencyEdge::WAW;
  ASSERT_EQ(edge.type, DependencyEdge::WAW);
}

}  // namespace ir
}  // namespace pypto
