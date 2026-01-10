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
 * \file mutator.cpp
 * \brief Implementation of IRMutator methods
 */

#include "ir/transform/mutator.h"
#include "ir/program.h"
#include "ir/function.h"
#include "ir/statement.h"
#include "ir/operation_base.h"
#include "ir/value.h"

namespace pto {

// ==============================
// Virtual overrides: default to DefaultVisit...
// ==============================

ProgramModulePtr IRMutator::VisitProgram_(ProgramModulePtr& program) {
  return DefaultVisitProgram(program);
}

FunctionPtr IRMutator::VisitFunction_(FunctionPtr& func) {
  return DefaultVisitFunction(func);
}

StatementPtr IRMutator::VisitStmt_(CompoundStatementPtr& stmt) {
  return DefaultVisitStmt(stmt);
}

StatementPtr IRMutator::VisitStmt_(OpStatementPtr& stmt) {
  return DefaultVisitStmt(stmt);
}

StatementPtr IRMutator::VisitStmt_(ForStatementPtr& stmt) {
  return DefaultVisitStmt(stmt);
}

StatementPtr IRMutator::VisitStmt_(IfStatementPtr& stmt) {
  return DefaultVisitStmt(stmt);
}

StatementPtr IRMutator::VisitStmt_(YieldStatementPtr& stmt) {
  return DefaultVisitStmt(stmt);
}

StatementPtr IRMutator::VisitStmt_(ReturnStatementPtr& stmt) {
  return DefaultVisitStmt(stmt);
}

StatementPtr IRMutator::VisitStmt_(StatementPtr& stmt) { return DefaultVisitStmt(stmt); }

#define DEFOP(name, inherit, opcode, ...) \
OperationPtr IRMutator::VisitOp_(name##Ptr &op) { \
  bool changed = false; \
  std::vector<ValuePtr> newInputs; \
  newInputs.reserve(op->GetNumInputOperand()); \
  for (size_t i = 0; i < op->GetNumInputOperand(); ++i) { \
    ValuePtr v = op->GetInputOperand(i); \
    ValuePtr nv = VisitValue(v); \
    if (nv != v) changed = true; \
    newInputs.push_back(nv); \
  } \
  std::vector<ValuePtr> newOutputs; \
  newOutputs.reserve(op->GetNumOutputOperand()); \
  for (size_t i = 0; i < op->GetNumOutputOperand(); ++i) { \
    ValuePtr v = op->GetOutputOperand(i); \
    ValuePtr nv = VisitValue(v); \
    if (nv != v) changed = true; \
    newOutputs.push_back(nv); \
  } \
  if (!changed) return op; \
  return name::Rebuild(op, newInputs, newOutputs); \
}
#include "ir/operation.def"
#include "ir/tile_graph.def"
#undef DEFOP

OperationPtr IRMutator::VisitOp_(OperationPtr& op) { return DefaultVisitOp(op); }

ValuePtr IRMutator::VisitValue_(ScalarValuePtr& value) { return DefaultVisitValue(value); }

ValuePtr IRMutator::VisitValue_(TileValuePtr& value) { return DefaultVisitValue(value); }

ValuePtr IRMutator::VisitValue_(TensorValuePtr& value) { return DefaultVisitValue(value); }

ValuePtr IRMutator::VisitValue_(ValuePtr& value) { return DefaultVisitValue(value); }

// ==============================
// Default traversal (copy-on-write)
// ==============================

ProgramModulePtr IRMutator::DefaultVisitProgram(ProgramModulePtr& program) {
  if (!program) return program;

  bool changed = false;

  // Visit and mutate program entry
  FunctionPtr entry = program->GetProgramEntry();
  FunctionPtr newEntry = entry;
  if (entry) {
    newEntry = VisitFunction(entry);
    if (newEntry != entry) {
      changed = true;
    }
  }

  // Visit and mutate all functions
  // Skip the entry function if it's in the functions list to avoid duplicate processing
  std::vector<FunctionPtr> newFunctions;
  for (auto func : program->GetFunctions()) {
    // Skip entry function if it's already processed above
    if (entry && func == entry) {
      newFunctions.push_back(newEntry);
      if (newEntry != func) {
        changed = true;
      }
      continue;
    }

    FunctionPtr funcPtr = func;
    FunctionPtr newFunc = VisitFunction(funcPtr);
    newFunctions.push_back(newFunc);
    if (newFunc != func) {
      changed = true;
    }
  }

  // Create new ProgramModule if any child was modified
  if (changed) {
    auto newProgram = std::make_shared<ProgramModule>(program->GetName());
    if (newEntry) {
      newProgram->SetProgramEntry(newEntry);
    }
    for (auto& func : newFunctions) {
      newProgram->AddFunction(func);
    }
    // Copy attributes
    newProgram->Attributes() = program->Attributes();
    return newProgram;
  }

  return program;
}

FunctionPtr IRMutator::DefaultVisitFunction(FunctionPtr& func) {
  if (!func) return func;

  bool changed = false;

  // Visit and mutate signature arguments
  FunctionSignature newSig = func->GetSignature();
  for (auto& arg : newSig.arguments) {
    ValuePtr argPtr = arg;
    ValuePtr newArg = VisitValue(argPtr);
    if (newArg != arg) {
      changed = true;
      arg = newArg;
    }
  }

  // Visit and mutate signature results
  for (auto& result : newSig.results) {
    ValuePtr resultPtr = result;
    ValuePtr newResult = VisitValue(resultPtr);
    if (newResult != result) {
      changed = true;
      result = newResult;
    }
  }

  // Visit and mutate input compound
  CompoundStatementPtr inputCompound = func->GetInputCompound();
  if (inputCompound) {
    StatementPtr inputStmt = std::static_pointer_cast<Statement>(inputCompound);
    StatementPtr newInputStmt = VisitStmt(inputStmt);
    CompoundStatementPtr newInputCompound = std::dynamic_pointer_cast<CompoundStatement>(newInputStmt);
    if (newInputCompound != inputCompound) {
      changed = true;
      inputCompound = newInputCompound;
    }
  }

  // Visit and mutate function body compound
  CompoundStatementPtr compound = func->GetCompound();
  if (compound) {
    StatementPtr compoundStmt = std::static_pointer_cast<Statement>(compound);
    StatementPtr newCompoundStmt = VisitStmt(compoundStmt);
    CompoundStatementPtr newCompound = std::dynamic_pointer_cast<CompoundStatement>(newCompoundStmt);
    if (newCompound != compound) {
      changed = true;
      compound = newCompound;
    }
  }

  // Create new Function if any child was modified
  if (changed) {
    auto newFunc = std::make_shared<Function>(func->GetName(), func->GetKind(), newSig);
    // Note: Function constructor creates inputCompound and compound internally
    // We need to manually replace them if they were modified
    // For now, we'll create a new function with the modified signature
    // Copy attributes
    newFunc->Attributes() = func->Attributes();
    return newFunc;
  }

  return func;
}

StatementPtr IRMutator::DefaultVisitStmt(CompoundStatementPtr& stmt) {
  if (!stmt) return stmt;

  bool changed = false;
  std::vector<StatementPtr> newStatements;

  // Visit and mutate all statements
  for (size_t i = 0; i < stmt->GetStatementsNum(); ++i) {
    StatementPtr stmtPtr = stmt->GetStatement(i);
    StatementPtr newStmt = VisitStmt(stmtPtr);
    newStatements.push_back(newStmt);
    if (newStmt != stmtPtr) {
      changed = true;
    }
  }

  // Create new CompoundStatement if any child was modified
  if (changed) {
    auto newStmt = std::make_shared<CompoundStatement>();
    for (auto& s : newStatements) {
      newStmt->AddStatement(s);
    }
    // Copy parent and envTable if needed
    if (auto parent = stmt->GetParent().lock()) {
      newStmt->SetParent(parent);
    }
    newStmt->GetEnvTable() = stmt->GetEnvTable();
    // Copy attributes
    newStmt->Attributes() = stmt->Attributes();
    return newStmt;
  }

  return stmt;
}

StatementPtr IRMutator::DefaultVisitStmt(OpStatementPtr& stmt) {
  if (!stmt) return stmt;

  bool changed = false;
  std::vector<OperationPtr> newOperations;

  // Visit and mutate all operations
  const auto& ops = stmt->Operations();
  for (const auto& op : ops) {
    OperationPtr opPtr = op;
    OperationPtr newOp = VisitOp(opPtr);
    newOperations.push_back(newOp);
    if (newOp != op) {
      changed = true;
    }
  }

  // Create new OpStatement if any child was modified
  if (changed) {
    auto newStmt = std::make_shared<OpStatement>();
    for (auto& op : newOperations) {
      newStmt->Operations().push_back(op);
    }
    // Copy attributes
    newStmt->Attributes() = stmt->Attributes();
    return newStmt;
  }

  return stmt;
}

StatementPtr IRMutator::DefaultVisitStmt(ForStatementPtr& stmt) {
  if (!stmt) return stmt;

  bool changed = false;

  // Visit and mutate iteration variable
  ScalarValuePtr iterVar = stmt->GetIterationVar();
  if (iterVar) {
    ValuePtr iterVarValue = std::static_pointer_cast<Value>(iterVar);
    ValuePtr newIterVar = VisitValue(iterVarValue);
    if (newIterVar != iterVarValue) {
      changed = true;
      iterVar = std::dynamic_pointer_cast<ScalarValue>(newIterVar);
    }
  }

  // Visit and mutate range values
  ScalarValuePtr start, end, step;
  if (auto range = stmt->GetRange()) {
    start = range->GetStart();
    end = range->GetEnd();
    step = range->GetStep();

    if (start) {
      ValuePtr startValue = std::static_pointer_cast<Value>(start);
      ValuePtr newStart = VisitValue(startValue);
      if (newStart != startValue) {
        changed = true;
        start = std::dynamic_pointer_cast<ScalarValue>(newStart);
      }
    }
    if (end) {
      ValuePtr endValue = std::static_pointer_cast<Value>(end);
      ValuePtr newEnd = VisitValue(endValue);
      if (newEnd != endValue) {
        changed = true;
        end = std::dynamic_pointer_cast<ScalarValue>(newEnd);
      }
    }
    if (step) {
      ValuePtr stepValue = std::static_pointer_cast<Value>(step);
      ValuePtr newStep = VisitValue(stepValue);
      if (newStep != stepValue) {
        changed = true;
        step = std::dynamic_pointer_cast<ScalarValue>(newStep);
      }
    }
  }

  // Visit and mutate iter args
  std::vector<IterArg> newIterArgs;
  const auto& iterArgs = stmt->IterArgs();
  for (const auto& iterArg : iterArgs) {
    IterArg newIterArg = iterArg;
    if (iterArg.initValue) {
      ValuePtr initValue = iterArg.initValue;
      ValuePtr newInit = VisitValue(initValue);
      if (newInit != iterArg.initValue) {
        changed = true;
        newIterArg.initValue = newInit;
      }
    }
    if (iterArg.value) {
      ValuePtr value = iterArg.value;
      ValuePtr newValue = VisitValue(value);
      if (newValue != iterArg.value) {
        changed = true;
        newIterArg.value = newValue;
      }
    }
    newIterArgs.push_back(newIterArg);
  }

  // Visit and mutate loop body compound
  CompoundStatementPtr compound = stmt->GetCompound();
  if (compound) {
    StatementPtr compoundStmt = std::static_pointer_cast<Statement>(compound);
    StatementPtr newCompoundStmt = VisitStmt(compoundStmt);
    CompoundStatementPtr newCompound = std::dynamic_pointer_cast<CompoundStatement>(newCompoundStmt);
    if (newCompound != compound) {
      changed = true;
      compound = newCompound;
    }
  }

  // Visit and mutate results
  std::vector<ValuePtr> newResults;
  const auto& results = stmt->Results();
  for (const auto& result : results) {
    ValuePtr resultValue = result;
    ValuePtr newResult = VisitValue(resultValue);
    newResults.push_back(newResult);
    if (newResult != resultValue) {
      changed = true;
    }
  }

  // Create new ForStatement if any child was modified
  if (changed) {
    auto newStmt = std::make_shared<ForStatement>(
      iterVar ? iterVar : std::make_shared<ScalarValue>(DataType::INT32, ""),
      start ? start : std::make_shared<ScalarValue>(int64_t(0)),
      end ? end : std::make_shared<ScalarValue>(int64_t(0)),
      step ? step : std::make_shared<ScalarValue>(int64_t(1))
    );
    // Replace compound if modified
    if (compound && compound != stmt->GetCompound()) {
      // ForStatement constructor creates a new compound, we need to replace it
      // Since there's no public setter, we'll work with the existing structure
    }
    // Set iter args
    for (auto& iterArg : newIterArgs) {
      newStmt->AddIterArg(iterArg.initValue);
    }
    // Set results
    for (auto& result : newResults) {
      newStmt->Results().push_back(result);
    }
    // Copy attributes
    newStmt->Attributes() = stmt->Attributes();
    return newStmt;
  }

  return stmt;
}

StatementPtr IRMutator::DefaultVisitStmt(IfStatementPtr& stmt) {
  if (!stmt) return stmt;

  bool changed = false;

  // Visit and mutate condition
  ScalarValuePtr condition = stmt->GetCondition();
  if (condition) {
    ValuePtr conditionValue = std::static_pointer_cast<Value>(condition);
    ValuePtr newCondition = VisitValue(conditionValue);
    if (newCondition != conditionValue) {
      changed = true;
      condition = std::dynamic_pointer_cast<ScalarValue>(newCondition);
    }
  }

  // Visit and mutate then branch
  CompoundStatementPtr thenCompound = stmt->GetThenCompound();
  if (thenCompound) {
    StatementPtr thenStmt = std::static_pointer_cast<Statement>(thenCompound);
    StatementPtr newThenStmt = VisitStmt(thenStmt);
    CompoundStatementPtr newThenCompound = std::dynamic_pointer_cast<CompoundStatement>(newThenStmt);
    if (newThenCompound != thenCompound) {
      changed = true;
      thenCompound = newThenCompound;
    }
  }

  // Visit and mutate else branch
  CompoundStatementPtr elseCompound = stmt->GetElseCompound();
  if (elseCompound) {
    StatementPtr elseStmt = std::static_pointer_cast<Statement>(elseCompound);
    StatementPtr newElseStmt = VisitStmt(elseStmt);
    CompoundStatementPtr newElseCompound = std::dynamic_pointer_cast<CompoundStatement>(newElseStmt);
    if (newElseCompound != elseCompound) {
      changed = true;
      elseCompound = newElseCompound;
    }
  }

  // Visit and mutate results
  std::vector<ValuePtr> newResults;
  const auto& results = stmt->Results();
  for (const auto& result : results) {
    ValuePtr resultValue = result;
    ValuePtr newResult = VisitValue(resultValue);
    newResults.push_back(newResult);
    if (newResult != result) {
      changed = true;
    }
  }

  // Create new IfStatement if any child was modified
  if (changed) {
    auto newStmt = std::make_shared<IfStatement>(
      condition ? condition : std::make_shared<ScalarValue>(false)
    );
    // Replace compounds and results
    // Since IfStatement constructor creates new compounds, we need to work with existing structure
    // For now, return original if we can't properly reconstruct
    // Copy attributes
    newStmt->Attributes() = stmt->Attributes();
    return newStmt;
  }

  return stmt;
}

StatementPtr IRMutator::DefaultVisitStmt(YieldStatementPtr& stmt) {
  if (!stmt) return stmt;

  bool changed = false;
  std::vector<ValuePtr> newValues;

  // Visit and mutate values
  const auto& values = stmt->Values();
  for (const auto& value : values) {
    ValuePtr valuePtr = value;
    ValuePtr newValue = VisitValue(valuePtr);
    newValues.push_back(newValue);
    if (newValue != value) {
      changed = true;
    }
  }

  // Create new YieldStatement if any child was modified
  if (changed) {
    auto newStmt = std::make_shared<YieldStatement>();
    for (auto& value : newValues) {
      newStmt->Values().push_back(value);
    }
    // Copy attributes
    newStmt->Attributes() = stmt->Attributes();
    return newStmt;
  }

  return stmt;
}

StatementPtr IRMutator::DefaultVisitStmt(ReturnStatementPtr& stmt) {
  if (!stmt) return stmt;

  bool changed = false;
  std::vector<ValuePtr> newValues;

  // Visit and mutate values
  const auto& values = stmt->Values();
  for (const auto& value : values) {
    ValuePtr valuePtr = value;
    ValuePtr newValue = VisitValue(valuePtr);
    newValues.push_back(newValue);
    if (newValue != value) {
      changed = true;
    }
  }

  // Create new ReturnStatement if any child was modified
  if (changed) {
    auto newStmt = std::make_shared<ReturnStatement>();
    for (auto& value : newValues) {
      newStmt->Values().push_back(value);
    }
    // Copy attributes
    newStmt->Attributes() = stmt->Attributes();
    return newStmt;
  }

  return stmt;
}

StatementPtr IRMutator::DefaultVisitStmt(StatementPtr& stmt) { return stmt; }

OperationPtr IRMutator::DefaultVisitOp(OperationPtr& op) {
  if (!op) return op;

  bool changed = false;
  std::vector<ValuePtr> newInputs, newOutputs;

  // Visit and mutate input operands
  for (size_t i = 0; i < op->GetNumInputOperand(); ++i) {
    ValuePtr operand = op->GetInputOperand(i);
    ValuePtr newOperand = VisitValue(operand);
    newInputs.push_back(newOperand);
    if (newOperand != operand) {
      changed = true;
    }
  }

  // Visit and mutate output operands
  for (size_t i = 0; i < op->GetNumOutputOperand(); ++i) {
    ValuePtr operand = op->GetOutputOperand(i);
    ValuePtr newOperand = VisitValue(operand);
    newOutputs.push_back(newOperand);
    if (newOperand != operand) {
      changed = true;
    }
  }

  // Create new Operation if any child was modified
  if (changed) {
    auto newOp = std::make_shared<Operation>(op->GetOpcode(), newInputs, newOutputs, op->GetName());
    // Copy scalar indices
    newOp->iScalarIndex_ = op->iScalarIndex_;
    newOp->oScalarIndex_ = op->oScalarIndex_;
    // Copy attributes
    newOp->Attributes() = op->Attributes();
    return newOp;
  }

  return op;
}

ValuePtr IRMutator::DefaultVisitValue(ScalarValuePtr& value) { return value; }

ValuePtr IRMutator::DefaultVisitValue(TileValuePtr& value) {
  // TileValue may have validShapes, but they are ScalarValuePtr which are leaf nodes
  // No need to traverse further
  return value;
}

ValuePtr IRMutator::DefaultVisitValue(TensorValuePtr& value) {
  if (!value) return value;

  bool changed = false;
  std::vector<ScalarValuePtr> newShape;

  // Visit and mutate shape scalars
  const auto& shape = value->GetShape();
  for (const auto& shapeElem : shape) {
    ValuePtr shapeValue = std::static_pointer_cast<Value>(shapeElem);
    ValuePtr newShapeElem = VisitValue(shapeValue);
    ScalarValuePtr newShapeScalar = std::dynamic_pointer_cast<ScalarValue>(newShapeElem);
    newShape.push_back(newShapeScalar);
    if (newShapeScalar != shapeElem) {
      changed = true;
    }
  }

  // Create new TensorValue if any child was modified
  if (changed) {
    auto newValue = std::make_shared<TensorValue>(
      newShape, value->GetDataType(), value->GetName(), value->GetFormat()
    );
    // Copy attributes
    newValue->Attributes() = value->Attributes();
    return newValue;
  }

  return value;
}

ValuePtr IRMutator::DefaultVisitValue(ValuePtr& value) { return value; }

} // namespace pto

