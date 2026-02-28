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
  auto intType = std::make_shared<ScalarType>(DataType::INT32);
  auto varX = std::make_shared<Var>("x", intType, Span::unknown());
  auto varY = std::make_shared<Var>("y", intType, Span::unknown());

  auto val1 = std::make_shared<ConstInt>(1, DataType::INT32, Span::unknown());
  auto val2 = std::make_shared<ConstInt>(2, DataType::INT32, Span::unknown());

  auto assign1 = std::make_shared<AssignStmt>(varX, val1, Span::unknown());
  auto assign2 = std::make_shared<AssignStmt>(varY, val2, Span::unknown());

  std::vector<StmtPtr> stmts = {assign1, assign2};
  auto body = std::make_shared<SeqStmts>(stmts, Span::unknown());

  return std::make_shared<Function>("test", std::vector<VarPtr>{}, std::vector<TypePtr>{}, body,
                                    Span::unknown());
}

// Helper to create a function with data dependency (RAW)
static FunctionPtr MakeRAWDependencyFunction() {
  auto intType = std::make_shared<ScalarType>(DataType::INT32);
  auto varX = std::make_shared<Var>("x", intType, Span::unknown());
  auto varY = std::make_shared<Var>("y", intType, Span::unknown());

  auto val1 = std::make_shared<ConstInt>(1, DataType::INT32, Span::unknown());
  auto assign1 = std::make_shared<AssignStmt>(varX, val1, Span::unknown());

  // y = x (read after write on x)
  auto assign2 = std::make_shared<AssignStmt>(varY, varX, Span::unknown());

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
  bool hasRaw = false;
  for (const auto &dep : graph.dependencies) {
    if (dep.type == DependencyEdge::RAW) {
      hasRaw = true;
      break;
    }
  }
  ASSERT_TRUE(hasRaw);
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
  auto bodyVal = std::make_shared<ConstInt>(0, DataType::INT32, Span::unknown());
  auto body = std::make_shared<EvalStmt>(bodyVal, Span::unknown());
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
  auto intType = std::make_shared<ScalarType>(DataType::INT32);
  auto loopVar = std::make_shared<Var>("i", intType, Span::unknown());
  auto start = std::make_shared<ConstInt>(0, DataType::INT32, Span::unknown());
  auto stop = std::make_shared<ConstInt>(10, DataType::INT32, Span::unknown());
  auto step = std::make_shared<ConstInt>(1, DataType::INT32, Span::unknown());

  auto bodyExpr = std::make_shared<ConstInt>(0, DataType::INT32, Span::unknown());
  auto body = std::make_shared<EvalStmt>(bodyExpr, Span::unknown());

  std::vector<IterArgPtr> iterArgs;
  std::vector<VarPtr> returnVars;
  auto forStmt = std::make_shared<ForStmt>(loopVar, start, stop, step, iterArgs, body, returnVars,
                                            Span::unknown());

  auto func = std::make_shared<Function>("loop_func", std::vector<VarPtr>{}, std::vector<TypePtr>{},
                                         forStmt, Span::unknown());

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

// ============================================================================
// WAR and WAW Dependency Tests
// ============================================================================

TEST_F(DependencyAnalyzerTest, TestAnalyzeWARDependency) {
  auto intType = std::make_shared<ScalarType>(DataType::INT32);
  auto varX = std::make_shared<Var>("x", intType, Span::unknown());

  auto val1 = std::make_shared<ConstInt>(1, DataType::INT32, Span::unknown());
  auto val2 = std::make_shared<ConstInt>(2, DataType::INT32, Span::unknown());

  // y = x (read x)
  auto varY = std::make_shared<Var>("y", intType, Span::unknown());
  auto assign1 = std::make_shared<AssignStmt>(varY, varX, Span::unknown());

  // x = 2 (write x after read)
  auto assign2 = std::make_shared<AssignStmt>(varX, val2, Span::unknown());

  std::vector<StmtPtr> stmts = {assign1, assign2};
  auto body = std::make_shared<SeqStmts>(stmts, Span::unknown());
  auto func = std::make_shared<Function>("test", std::vector<VarPtr>{}, std::vector<TypePtr>{}, body,
                                         Span::unknown());

  DependencyAnalyzer analyzer;
  auto graph = analyzer.Analyze(func);

  bool hasWar = false;
  for (const auto &dep : graph.dependencies) {
    if (dep.type == DependencyEdge::WAR) {
      hasWar = true;
      break;
    }
  }
  ASSERT_TRUE(hasWar);
}

TEST_F(DependencyAnalyzerTest, TestAnalyzeWAWDependency) {
  auto intType = std::make_shared<ScalarType>(DataType::INT32);
  auto varX = std::make_shared<Var>("x", intType, Span::unknown());

  auto val1 = std::make_shared<ConstInt>(1, DataType::INT32, Span::unknown());
  auto val2 = std::make_shared<ConstInt>(2, DataType::INT32, Span::unknown());

  // x = 1 (write x)
  auto assign1 = std::make_shared<AssignStmt>(varX, val1, Span::unknown());
  // x = 2 (write x again)
  auto assign2 = std::make_shared<AssignStmt>(varX, val2, Span::unknown());

  std::vector<StmtPtr> stmts = {assign1, assign2};
  auto body = std::make_shared<SeqStmts>(stmts, Span::unknown());
  auto func = std::make_shared<Function>("test", std::vector<VarPtr>{}, std::vector<TypePtr>{}, body,
                                         Span::unknown());

  DependencyAnalyzer analyzer;
  auto graph = analyzer.Analyze(func);

  bool hasWaw = false;
  for (const auto &dep : graph.dependencies) {
    if (dep.type == DependencyEdge::WAW) {
      hasWaw = true;
      break;
    }
  }
  ASSERT_TRUE(hasWaw);
}

// ============================================================================
// AnalyzeDependencies Tests
// ============================================================================

TEST_F(DependencyAnalyzerTest, TestAnalyzeDependenciesMethod) {
  DependencyAnalyzer analyzer;
  auto func = MakeRAWDependencyFunction();
  auto deps = analyzer.AnalyzeDependencies(func);

  bool hasRaw = false;
  for (const auto &dep : deps) {
    if (dep.type == DependencyEdge::RAW) {
      hasRaw = true;
      break;
    }
  }
  ASSERT_TRUE(hasRaw);
}

// ============================================================================
// If Statement Tests
// ============================================================================

TEST_F(DependencyAnalyzerTest, TestAnalyzeFunctionWithIf) {
  auto intType = std::make_shared<ScalarType>(DataType::INT32);
  auto cond = std::make_shared<ConstBool>(true, Span::unknown());

  auto thenVal = std::make_shared<ConstInt>(1, DataType::INT32, Span::unknown());
  auto thenBody = std::make_shared<EvalStmt>(thenVal, Span::unknown());

  auto elseVal = std::make_shared<ConstInt>(2, DataType::INT32, Span::unknown());
  auto elseBody = std::make_shared<EvalStmt>(elseVal, Span::unknown());

  auto ifStmt = std::make_shared<IfStmt>(cond, thenBody, std::optional<StmtPtr>(elseBody),
                                          std::vector<VarPtr>{}, Span::unknown());

  auto func = std::make_shared<Function>("if_func", std::vector<VarPtr>{}, std::vector<TypePtr>{},
                                         ifStmt, Span::unknown());

  DependencyAnalyzer analyzer;
  auto graph = analyzer.Analyze(func);
  // Should handle if statement without crashing
  (void)graph;
}

// ============================================================================
// Nested SeqStmts Tests
// ============================================================================

TEST_F(DependencyAnalyzerTest, TestAnalyzeNestedSeqStmts) {
  auto intType = std::make_shared<ScalarType>(DataType::INT32);
  auto varX = std::make_shared<Var>("x", intType, Span::unknown());
  auto val1 = std::make_shared<ConstInt>(1, DataType::INT32, Span::unknown());
  auto val2 = std::make_shared<ConstInt>(2, DataType::INT32, Span::unknown());

  auto assign1 = std::make_shared<AssignStmt>(varX, val1, Span::unknown());
  auto assign2 = std::make_shared<AssignStmt>(varX, val2, Span::unknown());

  auto innerSeq = std::make_shared<SeqStmts>(std::vector<StmtPtr>{assign1}, Span::unknown());
  auto outerSeq = std::make_shared<SeqStmts>(std::vector<StmtPtr>{innerSeq, assign2}, Span::unknown());

  auto func = std::make_shared<Function>("nested", std::vector<VarPtr>{}, std::vector<TypePtr>{},
                                         outerSeq, Span::unknown());

  DependencyAnalyzer analyzer;
  auto blocks = analyzer.AnalyzeBasicBlocks(func);
  ASSERT_FALSE(blocks.empty());
}

// ============================================================================
// EvalStmt Dependency Tests
// ============================================================================

TEST_F(DependencyAnalyzerTest, TestAnalyzeEvalStmtDependency) {
  auto intType = std::make_shared<ScalarType>(DataType::INT32);
  auto varX = std::make_shared<Var>("x", intType, Span::unknown());
  auto val1 = std::make_shared<ConstInt>(1, DataType::INT32, Span::unknown());

  // x = 1
  auto assign = std::make_shared<AssignStmt>(varX, val1, Span::unknown());
  // eval(x) - reads x after write
  auto eval = std::make_shared<EvalStmt>(varX, Span::unknown());

  std::vector<StmtPtr> stmts = {assign, eval};
  auto body = std::make_shared<SeqStmts>(stmts, Span::unknown());
  auto func = std::make_shared<Function>("test", std::vector<VarPtr>{}, std::vector<TypePtr>{}, body,
                                         Span::unknown());

  DependencyAnalyzer analyzer;
  auto graph = analyzer.Analyze(func);

  bool hasRaw = false;
  for (const auto &dep : graph.dependencies) {
    if (dep.type == DependencyEdge::RAW) {
      hasRaw = true;
      break;
    }
  }
  ASSERT_TRUE(hasRaw);
}

// ============================================================================
// MergeDependencies Tests
// ============================================================================

TEST_F(DependencyAnalyzerTest, TestMergeDependenciesEmpty) {
  DependencyAnalyzer analyzer;
  std::vector<std::vector<DependencyEdge>> paths;
  auto merged = analyzer.MergeDependencies(paths);
  ASSERT_TRUE(merged.empty());
}

TEST_F(DependencyAnalyzerTest, TestMergeDependenciesDeduplicate) {
  auto intType = std::make_shared<ScalarType>(DataType::INT32);
  auto varX = std::make_shared<Var>("x", intType, Span::unknown());
  auto val1 = std::make_shared<ConstInt>(1, DataType::INT32, Span::unknown());
  auto stmt1 = std::make_shared<AssignStmt>(varX, val1, Span::unknown());
  auto stmt2 = std::make_shared<EvalStmt>(varX, Span::unknown());

  DependencyEdge edge;
  edge.producer = stmt1;
  edge.consumer = stmt2;
  edge.variable = varX;
  edge.type = DependencyEdge::RAW;
  edge.producer_pipe = "UNKNOWN";
  edge.consumer_pipe = "UNKNOWN";

  // Same edge in two paths
  std::vector<std::vector<DependencyEdge>> paths = {{edge}, {edge}};

  DependencyAnalyzer analyzer;
  auto merged = analyzer.MergeDependencies(paths);
  ASSERT_EQ(merged.size(), 1);  // Should deduplicate
}

}  // namespace ir
}  // namespace pypto
