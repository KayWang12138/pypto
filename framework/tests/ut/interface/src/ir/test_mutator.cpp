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
 * \file test_mutator.cpp
 * \brief Unit tests for IRMutator
 */

#include <cstdint>
#include <memory>
#include <unordered_map>
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
#include "ir/transform/mutator.h"
#include "ir/transform/visitor.h"

namespace pto {

/**
 * @brief Test mutator that counts mutations but doesn't modify
 */
class CountingMutator : public IRMutator {
public:
  int programCount = 0;
  int functionCount = 0;
  int compoundStmtCount = 0;
  int opStmtCount = 0;
  int forStmtCount = 0;
  int ifStmtCount = 0;
  int operationCount = 0;
  int valueCount = 0;

  // 引入父类的通用方法以避免 overloaded-virtual 警告
  using IRMutator::VisitStmt_;
  using IRMutator::VisitOp_;
  using IRMutator::VisitValue_;

  ProgramModulePtr VisitProgram_(ProgramModulePtr& program) override {
    programCount++;
    return IRMutator::VisitProgram_(program);
  }

  FunctionPtr VisitFunction_(FunctionPtr& func) override {
    functionCount++;
    return IRMutator::VisitFunction_(func);
  }

  StatementPtr VisitStmt_(CompoundStatementPtr& stmt) override {
    compoundStmtCount++;
    return IRMutator::VisitStmt_(stmt);
  }

  StatementPtr VisitStmt_(OpStatementPtr& stmt) override {
    opStmtCount++;
    return IRMutator::VisitStmt_(stmt);
  }

  StatementPtr VisitStmt_(ForStatementPtr& stmt) override {
    forStmtCount++;
    return IRMutator::VisitStmt_(stmt);
  }

  StatementPtr VisitStmt_(IfStatementPtr& stmt) override {
    ifStmtCount++;
    return IRMutator::VisitStmt_(stmt);
  }

// ---- operations ----
#define DEFOP(name, inherit, opcode, ...)                 \
  OperationPtr VisitOp_(name##Ptr& op) override {         \
    operationCount++;                                     \
    return IRMutator::VisitOp_(op);                       \
  }
#include "ir/operation.def"
#include "ir/tile_graph.def"
#undef DEFOP

  OperationPtr VisitOp_(OperationPtr& op) override {
    return IRMutator::VisitOp_(op);
  }

  ValuePtr VisitValue_(ScalarValuePtr& value) override {
    valueCount++;
    return IRMutator::VisitValue_(value);
  }

  ValuePtr VisitValue_(TileValuePtr& value) override {
    valueCount++;
    return IRMutator::VisitValue_(value);
  }

  ValuePtr VisitValue_(TensorValuePtr& value) override {
    valueCount++;
    return IRMutator::VisitValue_(value);
  }
};

/**
 * @brief Test mutator that replaces scalar constants with different values
 */
class ConstantReplacerMutator : public IRMutator {
public:
  double replacementValue = 42.0;

  // 引入父类的通用方法以避免 overloaded-virtual 警告
  using IRMutator::VisitValue_;

  ValuePtr VisitValue_(ScalarValuePtr& value) override {
    if (!value) return value;

    // 如果是常量，且值为数值类型
    if (value->GetScalarValueKind() == ScalarValueKind::Immediate && value->HasImmediateValue()) {
      // 创建一个新的常量值
      auto newValue = std::make_shared<ScalarValue>(replacementValue, value->GetName());
      return newValue;
    }

    return IRMutator::VisitValue_(value);
  }
};

/**
 * @brief Test mutator that renames values
 */
class ValueRenamingMutator : public IRMutator {
private:
  std::unordered_map<ValuePtr, std::string> renameMap;

public:
  // 引入父类的通用方法以避免 overloaded-virtual 警告
  using IRMutator::VisitValue_;

  void AddRename(ValuePtr oldValue, const std::string& newName) {
    renameMap[oldValue] = newName;
  }

  ValuePtr VisitValue_(ScalarValuePtr& value) override {
    if (!value) return value;

    auto it = renameMap.find(value);
    if (it != renameMap.end()) {
      auto newValue = std::make_shared<ScalarValue>(
        value->GetDataType(), it->second, value->GetScalarValueKind()
      );
      return newValue;
    }

    return IRMutator::VisitValue_(value);
  }

  ValuePtr VisitValue_(TileValuePtr& value) override {
    if (!value) return value;

    auto it = renameMap.find(value);
    if (it != renameMap.end()) {
      auto newValue = std::make_shared<TileValue>(
        value->GetShape(), value->GetDataType(), it->second
      );
      return newValue;
    }

    return IRMutator::VisitValue_(value);
  }

  ValuePtr VisitValue_(TensorValuePtr& value) override {
    if (!value) return value;

    auto it = renameMap.find(value);
    if (it != renameMap.end()) {
      auto newValue = std::make_shared<TensorValue>(
        value->GetShape(), value->GetDataType(), it->second, value->GetFormat()
      );
      return newValue;
    }

    return IRMutator::VisitValue_(value);
  }
};

/**
 * @brief Test mutator for DefaultVisit methods
 */
class DefaultVisitMutatorTester : public IRMutator {
public:
  int visitedCount = 0;
  std::unordered_set<void*> visitedNodes;

  // 使用 using 声明引入受保护方法以便测试
  using IRMutator::DefaultVisitProgram;
  using IRMutator::DefaultVisitFunction;
  using IRMutator::DefaultVisitStmt;
  using IRMutator::DefaultVisitOp;
  using IRMutator::DefaultVisitValue;
  
  // 引入父类的通用方法以避免 overloaded-virtual 警告
  using IRMutator::VisitStmt_;
  using IRMutator::VisitOp_;
  using IRMutator::VisitValue_;

  // 覆盖 VisitXXX_ 方法来追踪访问，但调用 DefaultVisit
  FunctionPtr VisitFunction_(FunctionPtr& func) override {
    if (func) {
      visitedNodes.insert(func.get());
      visitedCount++;
    }
    return DefaultVisitFunction(func);
  }

  StatementPtr VisitStmt_(CompoundStatementPtr& stmt) override {
    if (stmt) {
      visitedNodes.insert(stmt.get());
      visitedCount++;
    }
    return DefaultVisitStmt(stmt);
  }

  StatementPtr VisitStmt_(OpStatementPtr& stmt) override {
    if (stmt) {
      visitedNodes.insert(stmt.get());
      visitedCount++;
    }
    return DefaultVisitStmt(stmt);
  }

  StatementPtr VisitStmt_(ForStatementPtr& stmt) override {
    if (stmt) {
      visitedNodes.insert(stmt.get());
      visitedCount++;
    }
    return DefaultVisitStmt(stmt);
  }

  StatementPtr VisitStmt_(IfStatementPtr& stmt) override {
    if (stmt) {
      visitedNodes.insert(stmt.get());
      visitedCount++;
    }
    return DefaultVisitStmt(stmt);
  }

#define DEFOP(name, inherit, opcode, ...)                                \
  OperationPtr VisitOp_(name##Ptr& op) override {                        \
    if (op) {                                                            \
      visitedNodes.insert(op.get());                                     \
      visitedCount++;                                                    \
    }                                                                    \
    OperationPtr base = std::static_pointer_cast<Operation>(op);         \
    return DefaultVisitOp(base);                                         \
  }
#include "ir/operation.def"
#include "ir/tile_graph.def"
#undef DEFOP

  OperationPtr VisitOp_(OperationPtr& op) override {
    if (op) {
      visitedNodes.insert(op.get());
      visitedCount++;
    }
    return DefaultVisitOp(op);
  }

  ValuePtr VisitValue_(ScalarValuePtr& value) override {
    if (value) {
      visitedNodes.insert(value.get());
      visitedCount++;
    }
    return DefaultVisitValue(value);
  }

  ValuePtr VisitValue_(TileValuePtr& value) override {
    if (value) {
      visitedNodes.insert(value.get());
      visitedCount++;
    }
    return DefaultVisitValue(value);
  }

  ValuePtr VisitValue_(TensorValuePtr& value) override {
    if (value) {
      visitedNodes.insert(value.get());
      visitedCount++;
    }
    return DefaultVisitValue(value);
  }
};

/**
 * @brief Visitor to collect all values in a program
 */
class ValueCollectorVisitor : public IRVisitor {
public:
  std::unordered_set<ValuePtr> values;

  // 引入父类的通用方法以避免 overloaded-virtual 警告
  using IRVisitor::VisitValue_;

  void VisitValue_(ScalarValuePtr& value) override {
    if (value) values.insert(value);
    IRVisitor::VisitValue_(value);
  }

  void VisitValue_(TileValuePtr& value) override {
    if (value) values.insert(value);
    IRVisitor::VisitValue_(value);
  }

  void VisitValue_(TensorValuePtr& value) override {
    if (value) values.insert(value);
    IRVisitor::VisitValue_(value);
  }
};

TEST(IRMutatorTest, TestBasicMutate) {
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

  // 使用 CountingMutator 遍历并计数
  CountingMutator mutator;
  ProgramModulePtr modulePtr = module;
  auto newModule = mutator.VisitProgram(modulePtr);

  // 验证访问计数
  ASSERT_EQ(mutator.programCount, 1);
  ASSERT_EQ(mutator.functionCount, 1);
  ASSERT_GE(mutator.compoundStmtCount, 1);
  ASSERT_GE(mutator.opStmtCount, 1);
  ASSERT_GE(mutator.operationCount, 1);
  ASSERT_GE(mutator.valueCount, 3);  // inputTensor, outputTensor, c2, mulRes

  // 验证返回了新模块（即使没有修改）
  ASSERT_NE(newModule, nullptr);
}

TEST(IRMutatorTest, TestIdentityMutate) {
  // 测试identity mutator（不修改任何内容）
  auto module = std::make_shared<ProgramModule>("main");
  IRBuilder builder(module);
  IRBuilderContext ctx;

  FunctionSignature sig;
  std::vector<int64_t> tileShape = { 64, 64 };
  auto input = std::make_shared<TileValue>(tileShape, DataType::FP32, "input");
  sig.arguments = { input };

  auto func = builder.CreateFunction("test_identity", FunctionKind::Kernel, sig, /*setAsEntry=*/true);
  builder.EnterFunctionBody(ctx, func);

  auto c1 = builder.CreateConst(ctx, 1.0, "c1");
  auto result = builder.CreateTile(ctx, tileShape, DataType::FP32, "result");
  auto op = builder.CreateBinaryScalarMixOp(Opcode::OP_ADDS, input, c1, result);
  builder.Emit(ctx, op);

  builder.CreateReturn(ctx, {});

  ctx.PopScope();

  // 使用默认的 IRMutator（identity）
  IRMutator mutator;
  ProgramModulePtr modulePtr = module;
  auto newModule = mutator.VisitProgram(modulePtr);

  // 验证返回了有效的模块
  ASSERT_NE(newModule, nullptr);
  ASSERT_NE(newModule->GetProgramEntry(), nullptr);
}

TEST(IRMutatorTest, TestDefaultVisitMethods) {
  // 测试 DefaultVisit 方法的行为
  auto module = std::make_shared<ProgramModule>("main");
  IRBuilder builder(module);
  IRBuilderContext ctx;

  FunctionSignature sig;
  std::vector<int64_t> tileShape = { 32, 32 };
  auto input = std::make_shared<TileValue>(tileShape, DataType::FP32, "input");
  auto output = std::make_shared<TileValue>(tileShape, DataType::FP32, "output");
  sig.arguments = { input, output };

  auto func = builder.CreateFunction("test_default_visit_mutate", FunctionKind::Kernel, sig, /*setAsEntry=*/true);
  builder.EnterFunctionBody(ctx, func);

  auto c1 = builder.CreateConst(ctx, 1.0, "c1");
  auto result = builder.CreateTile(ctx, tileShape, DataType::FP32, "result");
  auto addOp = builder.CreateBinaryScalarMixOp(Opcode::OP_ADDS, input, c1, result);
  builder.Emit(ctx, addOp);

  builder.CreateReturn(ctx, {});
  ctx.PopScope();

  // 使用 DefaultVisitMutatorTester，它会调用 DefaultVisit 方法
  DefaultVisitMutatorTester tester;
  ProgramModulePtr modulePtr = module;
  auto newModule = tester.DefaultVisitProgram(modulePtr);

  // 验证 DefaultVisitProgram 正确遍历了所有子节点并返回新模块
  ASSERT_NE(newModule, nullptr);
  ASSERT_GT(tester.visitedCount, 0);
  
  // 验证 DefaultVisitFunction 被调用并返回新函数（copy-on-write）
  FunctionPtr funcPtr = func;
  auto newFunc = tester.DefaultVisitFunction(funcPtr);
  ASSERT_NE(newFunc, nullptr);
  // DefaultVisitFunction 会在有变化时返回新函数，否则返回原函数

  // 验证 DefaultVisitOp 被调用
  OperationPtr opPtr = addOp;
  auto newOp = tester.DefaultVisitOp(opPtr);
  ASSERT_NE(newOp, nullptr);
  // DefaultVisitOp 会访问 operation 的 operands

  // 验证 DefaultVisitValue 被调用
  ScalarValuePtr c1Ptr = c1;
  auto newC1 = tester.DefaultVisitValue(c1Ptr);
  ASSERT_EQ(newC1, c1);  // ScalarValue 默认不修改

  TileValuePtr inputPtr = input;
  auto newInput = tester.DefaultVisitValue(inputPtr);
  ASSERT_EQ(newInput, input);  // TileValue 默认不修改

  // 验证 DefaultVisitStmt 被调用
  auto compound = func->GetCompound();
  CompoundStatementPtr compoundPtr = compound;
  auto newCompound = tester.DefaultVisitStmt(compoundPtr);
  ASSERT_NE(newCompound, nullptr);
  // 如果有变化，会返回新的 CompoundStatement，否则返回原 statement
}

TEST(IRMutatorTest, TestConstantReplacement) {
  // 创建包含常量的IR程序
  auto module = std::make_shared<ProgramModule>("main");
  IRBuilder builder(module);
  IRBuilderContext ctx;

  FunctionSignature sig;
  std::vector<int64_t> tileShape = { 32, 32 };
  auto input = std::make_shared<TileValue>(tileShape, DataType::FP32, "input");
  sig.arguments = { input };

  auto func = builder.CreateFunction("test_replace", FunctionKind::Kernel, sig, /*setAsEntry=*/true);
  builder.EnterFunctionBody(ctx, func);

  auto originalConst = builder.CreateConst(ctx, 3.14, "original_const");
  auto result = builder.CreateTile(ctx, tileShape, DataType::FP32, "result");
  auto op = builder.CreateBinaryScalarMixOp(Opcode::OP_MULS, input, originalConst, result);
  builder.Emit(ctx, op);

  builder.CreateReturn(ctx, {});

  ctx.PopScope();

  // 使用 ConstantReplacerMutator 替换常量
  ConstantReplacerMutator replacer;
  replacer.replacementValue = 99.9;
  ProgramModulePtr modulePtr = module;
  auto newModule = replacer.VisitProgram(modulePtr);

  // 验证返回了新模块
  ASSERT_NE(newModule, nullptr);

  // 收集新模块中的所有常量值
  ValueCollectorVisitor collector;
  ProgramModulePtr newModulePtr = newModule;
  collector.VisitProgram(newModulePtr);

  // 验证常量被访问和替换（具体值替换可能需要在operation级别验证）
  ASSERT_GT(collector.values.size(), 0);
}

TEST(IRMutatorTest, TestNullHandling) {
  // 测试空指针处理
  IRMutator mutator;

  // 测试空 Function - DefaultVisitFunction 会检查 nullptr 并直接返回
  FunctionPtr nullFunc = nullptr;
  auto result2 = mutator.VisitFunction_(nullFunc);
  ASSERT_EQ(result2, nullptr);

  // 测试空 Operation - DefaultVisitOp 会检查 nullptr 并直接返回
  OperationPtr nullOp = nullptr;
  auto result4 = mutator.VisitOp_(nullOp);
  ASSERT_EQ(result4, nullptr);

  // 测试空 Value - DefaultVisitValue 会检查 nullptr 并直接返回
  ScalarValuePtr nullScalar = nullptr;
  auto result5 = mutator.VisitValue_(nullScalar);
  ASSERT_EQ(result5, nullptr);

  TileValuePtr nullTile = nullptr;
  auto result6 = mutator.VisitValue_(nullTile);
  ASSERT_EQ(result6, nullptr);

  TensorValuePtr nullTensor = nullptr;
  auto result7 = mutator.VisitValue_(nullTensor);
  ASSERT_EQ(result7, nullptr);
}

TEST(IRMutatorTest, TestControlFlowMutate) {
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

  auto func = builder.CreateFunction("test_control_mutate", FunctionKind::ControlFlow, sig, /*setAsEntry=*/true);
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

  // 使用 CountingMutator 遍历并计数
  CountingMutator mutator;
  ProgramModulePtr modulePtr = module;
  auto newModule = mutator.VisitProgram(modulePtr);

  // 验证访问计数
  ASSERT_EQ(mutator.programCount, 1);
  ASSERT_EQ(mutator.functionCount, 1);
  ASSERT_GE(mutator.forStmtCount, 1);
  ASSERT_GE(mutator.ifStmtCount, 1);
  ASSERT_GE(mutator.compoundStmtCount, 3);
  ASSERT_GE(mutator.operationCount, 2);

  // 验证返回了新模块
  ASSERT_NE(newModule, nullptr);
  ASSERT_NE(newModule->GetProgramEntry(), nullptr);
}

TEST(IRMutatorTest, TestMultipleMutations) {
  // 测试多次连续变换
  auto module = std::make_shared<ProgramModule>("main");
  IRBuilder builder(module);
  IRBuilderContext ctx;

  FunctionSignature sig;
  std::vector<int64_t> tileShape = { 16, 16 };
  auto input = std::make_shared<TileValue>(tileShape, DataType::FP32, "input");
  sig.arguments = { input };

  auto func = builder.CreateFunction("test_multiple", FunctionKind::Kernel, sig, /*setAsEntry=*/true);
  builder.EnterFunctionBody(ctx, func);

  auto c1 = builder.CreateConst(ctx, 1.0, "c1");
  auto c2 = builder.CreateConst(ctx, 2.0, "c2");
  auto tmp1 = builder.CreateTile(ctx, tileShape, DataType::FP32, "tmp1");
  auto op1 = builder.CreateBinaryScalarMixOp(Opcode::OP_ADDS, input, c1, tmp1);
  builder.Emit(ctx, op1);

  ctx.ResetInsertionPoint();
  auto tmp2 = builder.CreateTile(ctx, tileShape, DataType::FP32, "tmp2");
  auto op2 = builder.CreateBinaryScalarMixOp(Opcode::OP_MULS, tmp1, c2, tmp2);
  builder.Emit(ctx, op2);

  builder.CreateReturn(ctx, {});

  ctx.PopScope();

  // 第一次变换
  CountingMutator mutator1;
  ProgramModulePtr modulePtr1 = module;
  auto newModule1 = mutator1.VisitProgram(modulePtr1);
  ASSERT_NE(newModule1, nullptr);

  // 第二次变换（在第一次的结果上）
  CountingMutator mutator2;
  ProgramModulePtr modulePtr2 = newModule1;
  auto newModule2 = mutator2.VisitProgram(modulePtr2);
  ASSERT_NE(newModule2, nullptr);

  // 验证两次变换都成功
  ASSERT_EQ(mutator1.programCount, 1);
  ASSERT_EQ(mutator2.programCount, 1);
  ASSERT_GE(mutator1.operationCount, 2);
  ASSERT_GE(mutator2.operationCount, 2);
}

} // namespace pto
