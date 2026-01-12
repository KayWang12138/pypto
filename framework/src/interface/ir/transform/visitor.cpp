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
 * \file visitor.cpp
 * \brief Implementation of IRVisitor methods
 */

#include "ir/transform/visitor.h"

namespace pto {

// ----------------- Public Visit* entrypoints (default to DefaultVisit*) -----------------

void IRVisitor::VisitProgram_(ProgramModulePtr& program) {
  DefaultVisitProgram(program);
}

void IRVisitor::VisitFunction_(FunctionPtr& func) {
  DefaultVisitFunction(func);
}

void IRVisitor::VisitStmt_(CompoundStatementPtr& stmt) {
  DefaultVisitStmt(stmt);
}

void IRVisitor::VisitStmt_(OpStatementPtr& stmt) {
  DefaultVisitStmt(stmt);
}

void IRVisitor::VisitStmt_(ForStatementPtr& stmt) {
  DefaultVisitStmt(stmt);
}

void IRVisitor::VisitStmt_(IfStatementPtr& stmt) {
  DefaultVisitStmt(stmt);
}

void IRVisitor::VisitStmt_(YieldStatementPtr& stmt) {
  DefaultVisitStmt(stmt);
}

void IRVisitor::VisitStmt_(ReturnStatementPtr& stmt) {
  DefaultVisitStmt(stmt);
}

void IRVisitor::VisitStmt_(StatementPtr& stmt) {
  DefaultVisitStmt(stmt);
}

// ---- Concrete ops (auto-generated from *.def) ----
#define DEFOP(name, inherit, opcode, ...)            \
  void IRVisitor::VisitOp_(name##Ptr& op) {          \
    OperationPtr opPtr = std::static_pointer_cast<Operation>(op); \
    DefaultVisitOp(opPtr);                           \
  }
#include "ir/operation.def"
#include "ir/tile_graph.def"
#undef DEFOP

void IRVisitor::VisitOp_(OperationPtr& op) {
  DefaultVisitOp(op);
}

void IRVisitor::VisitValue_(ScalarValuePtr& value) {
  DefaultVisitValue(value);
}

void IRVisitor::VisitValue_(TileValuePtr& value) {
  DefaultVisitValue(value);
}

void IRVisitor::VisitValue_(TensorValuePtr& value) {
  DefaultVisitValue(value);
}

void IRVisitor::VisitValue_(ValuePtr& value) {
  DefaultVisitValue(value);
}

// ----------------- Default traversal implementations -----------------

void IRVisitor::DefaultVisitProgram(ProgramModulePtr& program) {
  if (!program) return;

  // Visit program entry
  if (auto entry = program->GetProgramEntry()) {
    FunctionPtr entryPtr = entry;
    VisitFunction(entryPtr);
  }

  // Visit all functions
  for (auto func : program->GetFunctions()) {
    FunctionPtr funcPtr = func;
    VisitFunction(funcPtr);
  }
}

void IRVisitor::DefaultVisitFunction(FunctionPtr& func) {
  if (!func) return;

  // Visit signature arguments
  const auto& sig = func->GetSignature();
  for (const auto& arg : sig.arguments) {
    ValuePtr argPtr = arg;
    VisitValue(argPtr);
  }

  // Visit signature results
  for (const auto& result : sig.results) {
    ValuePtr resultPtr = result;
    VisitValue(resultPtr);
  }

  // Visit input compound
  if (auto inputCompound = func->GetInputCompound()) {
    StatementPtr inputPtr = std::static_pointer_cast<Statement>(inputCompound);
    VisitStmt(inputPtr);
  }

  // Visit function body compound
  if (auto compound = func->GetCompound()) {
    StatementPtr compoundPtr = std::static_pointer_cast<Statement>(compound);
    VisitStmt(compoundPtr);
  }
}

void IRVisitor::DefaultVisitStmt(CompoundStatementPtr& stmt) {
  if (!stmt) return;

  for (size_t i = 0; i < stmt->GetStatementsNum(); ++i) {
    StatementPtr stmtPtr = stmt->GetStatement(i);
    VisitStmt(stmtPtr);
  }
}

void IRVisitor::DefaultVisitStmt(OpStatementPtr& stmt) {
  if (!stmt) return;

  const auto& ops = stmt->Operations();
  for (const auto& op : ops) {
    OperationPtr opPtr = op;
    VisitOp(opPtr);
  }
}

void IRVisitor::DefaultVisitStmt(ForStatementPtr& stmt) {
  if (!stmt) return;

  // Visit iteration variable
  if (auto iterVar = stmt->GetIterationVar()) {
    ValuePtr iterVarPtr = std::static_pointer_cast<Value>(iterVar);
    VisitValue(iterVarPtr);
  }

  // Visit range values
  if (auto range = stmt->GetRange()) {
    if (auto start = range->GetStart()) {
      ValuePtr startPtr = std::static_pointer_cast<Value>(start);
      VisitValue(startPtr);
    }
    if (auto end = range->GetEnd()) {
      ValuePtr endPtr = std::static_pointer_cast<Value>(end);
      VisitValue(endPtr);
    }
    if (auto step = range->GetStep()) {
      ValuePtr stepPtr = std::static_pointer_cast<Value>(step);
      VisitValue(stepPtr);
    }
  }

  // Visit iter args
  const auto& iterArgs = stmt->IterArgs();
  for (const auto& iterArg : iterArgs) {
    if (iterArg.initValue) {
      ValuePtr initPtr = iterArg.initValue;
      VisitValue(initPtr);
    }
    if (iterArg.value) {
      ValuePtr valuePtr = iterArg.value;
      VisitValue(valuePtr);
    }
  }

  // Visit loop body compound
  if (auto compound = stmt->GetCompound()) {
    StatementPtr compoundPtr = std::static_pointer_cast<Statement>(compound);
    VisitStmt(compoundPtr);
  }

  // Visit results
  const auto& results = stmt->Results();
  for (const auto& result : results) {
    ValuePtr resultPtr = result;
    VisitValue(resultPtr);
  }
}

void IRVisitor::DefaultVisitStmt(IfStatementPtr& stmt) {
  if (!stmt) return;

  // Visit condition
  if (auto condition = stmt->GetCondition()) {
    ValuePtr conditionPtr = std::static_pointer_cast<Value>(condition);
    VisitValue(conditionPtr);
  }

  // Visit then branch
  if (auto thenCompound = stmt->GetThenCompound()) {
    StatementPtr thenPtr = std::static_pointer_cast<Statement>(thenCompound);
    VisitStmt(thenPtr);
  }

  // Visit else branch
  if (auto elseCompound = stmt->GetElseCompound()) {
    StatementPtr elsePtr = std::static_pointer_cast<Statement>(elseCompound);
    VisitStmt(elsePtr);
  }

  // Visit results
  const auto& results = stmt->Results();
  for (const auto& result : results) {
    ValuePtr resultPtr = result;
    VisitValue(resultPtr);
  }
}

void IRVisitor::DefaultVisitStmt(YieldStatementPtr& stmt) {
  if (!stmt) return;

  const auto& values = stmt->Values();
  for (const auto& value : values) {
    ValuePtr valuePtr = value;
    VisitValue(valuePtr);
  }
}

void IRVisitor::DefaultVisitStmt(ReturnStatementPtr& stmt) {
  if (!stmt) return;

  const auto& values = stmt->Values();
  for (const auto& value : values) {
    ValuePtr valuePtr = value;
    VisitValue(valuePtr);
  }
}

void IRVisitor::DefaultVisitStmt(StatementPtr& stmt) {
  // Fallback for unknown statement types: no children to visit.
  (void)stmt;
}

void IRVisitor::DefaultVisitOp(OperationPtr& op) {
  if (!op) return;

  // Visit input operands
  for (size_t i = 0; i < op->GetNumInputOperand(); ++i) {
    ValuePtr operandPtr = op->GetInputOperand(i);
    VisitValue(operandPtr);
  }

  // Visit output operands
  for (size_t i = 0; i < op->GetNumOutputOperand(); ++i) {
    ValuePtr operandPtr = op->GetOutputOperand(i);
    VisitValue(operandPtr);
  }
}

void IRVisitor::DefaultVisitValue(ScalarValuePtr& value) {
  // ScalarValue is a leaf node
  (void)value;
}

void IRVisitor::DefaultVisitValue(TileValuePtr& value) {
  // TileValue treated as leaf in this IR
  (void)value;
}

void IRVisitor::DefaultVisitValue(TensorValuePtr& value) {
  if (!value) return;

  const auto& shape = value->GetShape();
  for (const auto& shapeElem : shape) {
    ValuePtr shapePtr = std::static_pointer_cast<Value>(shapeElem);
    VisitValue(shapePtr);
  }
}

void IRVisitor::DefaultVisitValue(ValuePtr& value) {
  // Fallback for unknown value types
  (void)value;
}

} // namespace pto
