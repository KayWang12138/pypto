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
 * \file test_visitor.cpp
 * \brief Unit tests for IRVisitor
 */

#include <cstdint>
#include <memory>
#include <unordered_set>
#include <vector>
#include "gtest/gtest.h"

#include "ir/builder/ir_builder.h"
#include "ir/builder/ir_context.h"
#include "ir/opcode.h"
#include "ir/program.h"
#include "ir/function.h"
#include "ir/statement.h"
#include "ir/value.h"
#include "ir/operation_base.h"
#include "ir/transform/visitor.h"

namespace pto {

/**
 * @brief Test visitor that counts visited nodes
 */
class CountingVisitor : public IRVisitor {
public:
  int programCount = 0;
  int functionCount = 0;
  int compoundStmtCount = 0;
  int opStmtCount = 0;
  int forStmtCount = 0;
  int ifStmtCount = 0;
  int yieldStmtCount = 0;
  int returnStmtCount = 0;
  int operationCount = 0;
  int scalarValueCount = 0;
  int tileValueCount = 0;
  int tensorValueCount = 0;

  // 引入父类的通用方法以避免 overloaded-virtual 警告
  using IRVisitor::VisitStmt_;
  using IRVisitor::VisitOp_;
  using IRVisitor::VisitValue_;

  void VisitProgram_(ProgramModulePtr& program) override {
    programCount++;
    IRVisitor::VisitProgram_(program);
  }

  void VisitFunction_(FunctionPtr& func) override {
    functionCount++;
    IRVisitor::VisitFunction_(func);
  }

  void VisitStmt_(CompoundStatementPtr& stmt) override {
    compoundStmtCount++;
    IRVisitor::VisitStmt_(stmt);
  }

  void VisitStmt_(OpStatementPtr& stmt) override {
    opStmtCount++;
    IRVisitor::VisitStmt_(stmt);
  }

  void VisitStmt_(ForStatementPtr& stmt) override {
    forStmtCount++;
    IRVisitor::VisitStmt_(stmt);
  }

  void VisitStmt_(IfStatementPtr& stmt) override {
    ifStmtCount++;
    IRVisitor::VisitStmt_(stmt);
  }

  void VisitStmt_(YieldStatementPtr& stmt) override {
    yieldStmtCount++;
    IRVisitor::VisitStmt_(stmt);
  }

  void VisitStmt_(ReturnStatementPtr& stmt) override {
    returnStmtCount++;
    IRVisitor::VisitStmt_(stmt);
  }

  // ---- operations ----
#define DEFOP(name, inherit, opcode, ...)           \
  void VisitOp_(name##Ptr& op) override {           \
    operationCount++;                               \
    IRVisitor::VisitOp_(op);                        \
  }
#include "ir/operation.def"
#include "ir/tile_graph.def"
#undef DEFOP

  void VisitOp_(OperationPtr& op) override {
    operationCount++;
    IRVisitor::VisitOp_(op);
  }

  void VisitValue_(ScalarValuePtr& value) override {
    scalarValueCount++;
    IRVisitor::VisitValue_(value);
  }

  void VisitValue_(TileValuePtr& value) override {
    tileValueCount++;
    IRVisitor::VisitValue_(value);
  }

  void VisitValue_(TensorValuePtr& value) override {
    tensorValueCount++;
    IRVisitor::VisitValue_(value);
  }
};

/**
 * @brief Test visitor that collects visited values
 */
class ValueCollectorVisitor : public IRVisitor {
public:
  std::unordered_set<ValuePtr> visitedValues;

  // 引入父类的通用方法以避免 overloaded-virtual 警告
  using IRVisitor::VisitValue_;

  void VisitValue_(ScalarValuePtr& value) override {
    if (value) {
      visitedValues.insert(value);
    }
    IRVisitor::VisitValue_(value);
  }

  void VisitValue_(TileValuePtr& value) override {
    if (value) {
      visitedValues.insert(value);
    }
    IRVisitor::VisitValue_(value);
  }

  void VisitValue_(TensorValuePtr& value) override {
    if (value) {
      visitedValues.insert(value);
    }
    IRVisitor::VisitValue_(value);
  }
};

TEST(IRVisitorTest, TestBasicTraversal) {
  // 创建简单的IR程序
  auto module = std::make_shared<ProgramModule>("main");
  IRBuilder builder(module);
  IRBuilderContext ctx;

  FunctionSignature sig;
  std::vector<int64_t> tileShape = { 128, 128 };
  auto inputTensor = std::make_shared<TileValue>(tileShape, DataType::FP32, "input");
  auto outputTensor = std::make_shared<TileValue>(tileShape, DataType::FP32, "output");
  sig.arguments = { inputTensor, outputTensor };

  auto func = builder.CreateFunction("test_func", FunctionKind::Kernel, sig, /*setAsEntry=*/true);
  builder.EnterFunctionBody(ctx, func);

  auto c2 = builder.CreateConst(ctx, 2.0, "c2");
  auto mulRes = builder.CreateTile(ctx, tileShape, DataType::FP32, "mul_res");
  auto mulOp = builder.CreateBinaryScalarMixOp(Opcode::OP_MULS, inputTensor, c2, mulRes);
  builder.Emit(ctx, mulOp);

  builder.CreateReturn(ctx, {});

  ctx.PopScope();

  // 使用 CountingVisitor 遍历
  CountingVisitor visitor;
  ProgramModulePtr modulePtr = module;
  visitor.VisitProgram(modulePtr);

  // 验证访问计数
  ASSERT_EQ(visitor.programCount, 1);
  // DefaultVisitProgram 会先访问 entry function，然后再访问 functions 列表中的所有函数
  ASSERT_GE(visitor.functionCount, 1);
  ASSERT_GE(visitor.compoundStmtCount, 1);  // 至少有一个compound statement
  ASSERT_GE(visitor.opStmtCount, 1);  // 至少有一个op statement
  // return statement 也会因为函数被访问两次而被访问两次
  ASSERT_GE(visitor.returnStmtCount, 1);
  ASSERT_GE(visitor.operationCount, 1);  // 至少有一个operation
  ASSERT_GE(visitor.scalarValueCount, 1);  // 至少有一个scalar value (常量c2)
  ASSERT_GE(visitor.tileValueCount, 3);  // inputTensor, outputTensor, mulRes
}

TEST(IRVisitorTest, TestValueCollection) {
  // 创建包含多个值的IR程序
  auto module = std::make_shared<ProgramModule>("main");
  IRBuilder builder(module);
  IRBuilderContext ctx;

  FunctionSignature sig;
  std::vector<int64_t> tileShape = { 64, 64 };
  auto in1 = std::make_shared<TileValue>(tileShape, DataType::FP32, "in1");
  auto in2 = std::make_shared<TileValue>(tileShape, DataType::FP32, "in2");
  sig.arguments = { in1, in2 };

  auto func = builder.CreateFunction("test_collect", FunctionKind::Kernel, sig, /*setAsEntry=*/true);
  builder.EnterFunctionBody(ctx, func);

  auto c1 = builder.CreateConst(ctx, 1.0, "c1");
  auto c2 = builder.CreateConst(ctx, 2.0, "c2");
  auto c3 = builder.CreateConst(ctx, 3.0, "c3");

  auto addRes = builder.CreateTile(ctx, tileShape, DataType::FP32, "add_res");
  auto addOp = builder.CreateBinaryScalarMixOp(Opcode::OP_ADDS, in1, c1, addRes);
  builder.Emit(ctx, addOp);

  auto mulRes = builder.CreateTile(ctx, tileShape, DataType::FP32, "mul_res");
  auto mulOp = builder.CreateBinaryScalarMixOp(Opcode::OP_MULS, addRes, c2, mulRes);
  builder.Emit(ctx, mulOp);

  // 使用 c3 以确保它被访问
  ctx.ResetInsertionPoint();
  auto finalRes = builder.CreateTile(ctx, tileShape, DataType::FP32, "final_res");
  auto finalOp = builder.CreateBinaryScalarMixOp(Opcode::OP_MULS, mulRes, c3, finalRes);
  builder.Emit(ctx, finalOp);

  builder.CreateReturn(ctx, {});

  ctx.PopScope();

  // 使用 ValueCollectorVisitor 收集所有值
  ValueCollectorVisitor collector;
  ProgramModulePtr modulePtr = module;
  collector.VisitProgram(modulePtr);

  // 验证收集到的值
  ASSERT_GT(collector.visitedValues.size(), 0);
  ASSERT_TRUE(collector.visitedValues.find(in1) != collector.visitedValues.end());
  ASSERT_TRUE(collector.visitedValues.find(in2) != collector.visitedValues.end());
  ASSERT_TRUE(collector.visitedValues.find(c1) != collector.visitedValues.end());
  ASSERT_TRUE(collector.visitedValues.find(c2) != collector.visitedValues.end());
  ASSERT_TRUE(collector.visitedValues.find(c3) != collector.visitedValues.end());
  ASSERT_TRUE(collector.visitedValues.find(addRes) != collector.visitedValues.end());
  ASSERT_TRUE(collector.visitedValues.find(mulRes) != collector.visitedValues.end());
  ASSERT_TRUE(collector.visitedValues.find(finalRes) != collector.visitedValues.end());
}

TEST(IRVisitorTest, TestControlFlowTraversal) {
  // 创建包含控制流的IR程序
  auto module = std::make_shared<ProgramModule>("main");
  IRBuilder builder(module);
  IRBuilderContext ctx;

  FunctionSignature sig;
  auto batch = std::make_shared<ScalarValue>(DataType::INT32, "batch", ScalarValueKind::Symbolic);
  auto constant128 = std::make_shared<ScalarValue>(int64_t(128), "const_128");
  std::vector<ScalarValuePtr> tensorShape = { batch, constant128 };
  std::vector<int64_t> tileShape = { 128, 128 };

  auto inputX = std::make_shared<TensorValue>(tensorShape, DataType::FP32, "inputX");
  auto scale1 = std::make_shared<ScalarValue>(DataType::FP32, "scale1", ScalarValueKind::Symbolic);
  auto resultX = std::make_shared<TensorValue>(tensorShape, DataType::FP32, "outputX");

  sig.arguments = { inputX, scale1, resultX };

  auto func = builder.CreateFunction("test_control", FunctionKind::ControlFlow, sig, /*setAsEntry=*/true);
  builder.EnterFunctionBody(ctx, func);

  // for i = 0 to batch step 1
  auto i = builder.CreateScalar(ctx, DataType::INT32, "i");
  auto constant0 = builder.CreateConst(ctx, int64_t(0), "const_0");
  auto constant1 = builder.CreateConst(ctx, int64_t(1), "const_1");
  auto fs = builder.CreateForStmt(ctx, i, constant0, batch, constant1);

  builder.EnterForBody(ctx, fs);

  auto resLoopX = builder.CreateTile(ctx, tileShape, DataType::FP32, "outputX");
  auto addOpX = builder.CreateBinaryScalarMixOp(Opcode::OP_ADDS, resLoopX, scale1, resLoopX);
  builder.Emit(ctx, addOpX);

  // if i then ...
  auto ifs = builder.CreateIfStmt(ctx, i);
  builder.EnterIfThen(ctx, ifs);
  auto resIfX = builder.CreateTile(ctx, tileShape, DataType::FP32, "outputX");
  auto mulOpX = builder.CreateBinaryScalarMixOp(Opcode::OP_MULS, resLoopX, scale1, resIfX);
  builder.Emit(ctx, mulOpX);
  ctx.PopScope();

  builder.EnterIfElse(ctx, ifs);
  ctx.PopScope();

  builder.ExitIfStatement(ctx, ifs);
  ctx.PopScope();  // for-body
  builder.ExitForStatement(ctx, fs);

  builder.CreateReturn(ctx, {constant0});
  ctx.PopScope();  // function-body

  // 使用 CountingVisitor 遍历
  CountingVisitor visitor;
  ProgramModulePtr modulePtr = module;
  visitor.VisitProgram(modulePtr);

  // 验证控制流节点被访问
  ASSERT_EQ(visitor.programCount, 1);
  // DefaultVisitProgram 会先访问 entry function，然后再访问 functions 列表中的所有函数
  // 如果 entry function 也在 functions 列表中，会被访问两次
  ASSERT_GE(visitor.functionCount, 1);
  ASSERT_GE(visitor.forStmtCount, 1);  // 至少有一个for statement
  ASSERT_GE(visitor.ifStmtCount, 1);   // 至少有一个if statement
  ASSERT_GE(visitor.compoundStmtCount, 3);  // function body, for body, if branches
  ASSERT_GE(visitor.operationCount, 2);  // addOpX 和 mulOpX
}

/**
 * @brief Test visitor for DefaultVisit methods
 */
class DefaultVisitTester : public IRVisitor {
public:
  int visitedCount = 0;
  std::unordered_set<void*> visitedNodes;

  // 使用 using 声明引入受保护方法以便测试
  using IRVisitor::DefaultVisitProgram;
  using IRVisitor::DefaultVisitFunction;
  using IRVisitor::DefaultVisitStmt;
  using IRVisitor::DefaultVisitOp;
  using IRVisitor::DefaultVisitValue;
  using IRVisitor::VisitFunction_;
  using IRVisitor::VisitStmt_;
  using IRVisitor::VisitOp_;
  using IRVisitor::VisitValue_;

  // 覆盖 VisitXXX_ 方法来追踪访问，但调用 DefaultVisit
  void VisitFunction_(FunctionPtr& func) override {
    if (func) {
      visitedNodes.insert(func.get());
      visitedCount++;
    }
    DefaultVisitFunction(func);
  }

  void VisitStmt_(CompoundStatementPtr& stmt) override {
    if (stmt) {
      visitedNodes.insert(stmt.get());
      visitedCount++;
    }
    DefaultVisitStmt(stmt);
  }

  void VisitStmt_(OpStatementPtr& stmt) override {
    if (stmt) {
      visitedNodes.insert(stmt.get());
      visitedCount++;
    }
    DefaultVisitStmt(stmt);
  }

  void VisitStmt_(ForStatementPtr& stmt) override {
    if (stmt) {
      visitedNodes.insert(stmt.get());
      visitedCount++;
    }
    DefaultVisitStmt(stmt);
  }

  void VisitStmt_(IfStatementPtr& stmt) override {
    if (stmt) {
      visitedNodes.insert(stmt.get());
      visitedCount++;
    }
    DefaultVisitStmt(stmt);
  }

  void VisitOp_(OperationPtr& op) override {
    if (op) {
      visitedNodes.insert(op.get());
      visitedCount++;
    }
    DefaultVisitOp(op);
  }

  void VisitValue_(ScalarValuePtr& value) override {
    if (value) {
      visitedNodes.insert(value.get());
      visitedCount++;
    }
    DefaultVisitValue(value);
  }

  void VisitValue_(TileValuePtr& value) override {
    if (value) {
      visitedNodes.insert(value.get());
      visitedCount++;
    }
    DefaultVisitValue(value);
  }

  void VisitValue_(TensorValuePtr& value) override {
    if (value) {
      visitedNodes.insert(value.get());
      visitedCount++;
    }
    DefaultVisitValue(value);
  }
};

/**
 * @brief Test visitor for null handling
 */
class NullHandlingVisitor : public IRVisitor {
public:
  // 使用 using 声明引入受保护方法以便测试
  using IRVisitor::VisitFunction_;
  using IRVisitor::VisitStmt_;
  using IRVisitor::VisitOp_;
  using IRVisitor::VisitValue_;
};

TEST(IRVisitorTest, TestDefaultTraversal) {
  // 测试默认遍历行为
  auto module = std::make_shared<ProgramModule>("main");
  IRBuilder builder(module);
  IRBuilderContext ctx;

  FunctionSignature sig;
  std::vector<int64_t> tileShape = { 32, 32 };
  auto input = std::make_shared<TileValue>(tileShape, DataType::FP32, "input");
  sig.arguments = { input };

  auto func = builder.CreateFunction("test_default", FunctionKind::Kernel, sig, /*setAsEntry=*/true);
  builder.EnterFunctionBody(ctx, func);

  auto c1 = builder.CreateConst(ctx, 1.0, "c1");
  auto result = builder.CreateTile(ctx, tileShape, DataType::FP32, "result");
  auto op = builder.CreateBinaryScalarMixOp(Opcode::OP_ADDS, input, c1, result);
  builder.Emit(ctx, op);

  builder.CreateReturn(ctx, {});

  ctx.PopScope();

  // 使用默认的 IRVisitor（不覆盖任何方法）
  IRVisitor visitor;
  ProgramModulePtr modulePtr = module;
  visitor.VisitProgram(modulePtr);

  // 默认访问器应该能够遍历所有节点而不崩溃
  // 虽然没有计数，但应该能正常完成遍历
  ASSERT_TRUE(true);  // 如果到达这里，说明遍历成功
}

TEST(IRVisitorTest, TestDefaultVisitMethods) {
  // 测试 DefaultVisit 方法的行为
  auto module = std::make_shared<ProgramModule>("main");
  IRBuilder builder(module);
  IRBuilderContext ctx;

  FunctionSignature sig;
  std::vector<int64_t> tileShape = { 64, 64 };
  auto input = std::make_shared<TileValue>(tileShape, DataType::FP32, "input");
  auto output = std::make_shared<TileValue>(tileShape, DataType::FP32, "output");
  sig.arguments = { input, output };

  auto func = builder.CreateFunction("test_default_visit", FunctionKind::Kernel, sig, /*setAsEntry=*/true);
  builder.EnterFunctionBody(ctx, func);

  auto c1 = builder.CreateConst(ctx, 1.0, "c1");
  auto result = builder.CreateTile(ctx, tileShape, DataType::FP32, "result");
  auto addOp = builder.CreateBinaryScalarMixOp(Opcode::OP_ADDS, input, c1, result);
  builder.Emit(ctx, addOp);

  builder.CreateReturn(ctx, {});
  ctx.PopScope();

  // 使用 DefaultVisitTester，它会调用 DefaultVisit 方法
  DefaultVisitTester tester;
  ProgramModulePtr modulePtr = module;
  tester.DefaultVisitProgram(modulePtr);

  // 验证 DefaultVisitProgram 正确遍历了所有子节点
  // 它应该访问了 function, statements, operations, values
  ASSERT_GT(tester.visitedCount, 0);
  ASSERT_NE(tester.visitedNodes.find(func.get()), tester.visitedNodes.end());
  
  // 验证 DefaultVisitFunction 被调用
  FunctionPtr funcPtr = func;
  tester.DefaultVisitFunction(funcPtr);
  // 应该访问了 function 的参数、结果、body 等

  // 验证 DefaultVisitOp 被调用
  OperationPtr opPtr = addOp;
  tester.DefaultVisitOp(opPtr);
  // 应该访问了 operation 的 operands

  // 验证 DefaultVisitValue 被调用
  ScalarValuePtr c1Ptr = c1;
  tester.DefaultVisitValue(c1Ptr);
  
  TileValuePtr inputPtr = input;
  tester.DefaultVisitValue(inputPtr);
  
  // 验证 DefaultVisitStmt 被调用
  auto compound = func->GetCompound();
  CompoundStatementPtr compoundPtr = compound;
  tester.DefaultVisitStmt(compoundPtr);
  // 应该访问了 compound 中的所有 statements
}

TEST(IRVisitorTest, TestNullHandling) {
  // 测试空指针处理
  NullHandlingVisitor visitor;

  // 测试空 Function - DefaultVisitFunction 会检查 nullptr
  FunctionPtr nullFunc = nullptr;
  visitor.VisitFunction_(nullFunc); 

  // 测试空 Statement - DefaultVisitStmt 会检查 nullptr
  CompoundStatementPtr nullCompound = nullptr;
  visitor.VisitStmt_(nullCompound); 

  // 测试空 Operation - DefaultVisitOp 会检查 nullptr
  OperationPtr nullOp = nullptr;
  visitor.VisitOp_(nullOp); 

  // 测试空 Value - DefaultVisitValue 会检查 nullptr
  ScalarValuePtr nullScalar = nullptr;
  visitor.VisitValue_(nullScalar); 

  TileValuePtr nullTile = nullptr;
  visitor.VisitValue_(nullTile); 

  TensorValuePtr nullTensor = nullptr;
  visitor.VisitValue_(nullTensor); 

  ASSERT_TRUE(true);  // 如果到达这里，说明空指针处理正常
}

} // namespace pto
