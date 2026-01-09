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
 * \file unroll_static_for.cpp
 * \brief Implementation of UnrollStaticFor pass
 */

#include "ir/transform/unroll_static_for.h"
#include "ir/transform/merge_op_stmt.h"
#include "ir/statement.h"
#include "ir/value.h"
#include <unordered_map>

namespace pto {

FunctionPtr UnrollStaticFor::VisitFunction_(FunctionPtr& func) {
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
  FunctionPtr resultFunc = func;
  if (changed) {
    auto newFunc = std::make_shared<Function>(func->GetName(), func->GetKind(), newSig);
    // Note: Function constructor creates inputCompound and compound internally
    // We need to manually replace them if they were modified
    // For now, we'll create a new function with the modified signature
    // Copy attributes
    newFunc->Attributes() = func->Attributes();
    
    // Replace compounds if modified
    if (inputCompound && inputCompound != func->GetInputCompound()) {
      // Copy statements from new input compound
      for (size_t i = 0; i < inputCompound->GetStatementsNum(); ++i) {
        newFunc->GetInputCompound()->AddStatement(inputCompound->GetStatement(i));
      }
    }
    
    if (compound && compound != func->GetCompound()) {
      // Copy statements from new compound to function body
      for (size_t i = 0; i < compound->GetStatementsNum(); ++i) {
        StatementPtr stmt = compound->GetStatement(i);
        newFunc->GetCompound()->AddStatement(stmt);
      }
    } else {
      // If compound wasn't modified, copy original statements
      for (size_t i = 0; i < func->GetCompound()->GetStatementsNum(); ++i) {
        StatementPtr stmt = func->GetCompound()->GetStatement(i);
        StatementPtr newStmt = VisitStmt(stmt);
        newFunc->GetCompound()->AddStatement(newStmt);
      }
    }
    
    resultFunc = newFunc;
  }
  
  return resultFunc;
}

StatementPtr UnrollStaticFor::VisitStmt_(CompoundStatementPtr& stmt) {
  if (!stmt) return stmt;
  
  // Set current compound for accessing envTable in nested statements
  CompoundStatementPtr prevCompound = currentCompound_;
  currentCompound_ = stmt;
  
  bool changed = false;
  std::vector<StatementPtr> newStatements;
  
  // Visit and mutate all statements
  for (size_t i = 0; i < stmt->GetStatementsNum(); ++i) {
    StatementPtr stmtPtr = stmt->GetStatement(i);
    StatementPtr newStmt = VisitStmt(stmtPtr);
    
    newStatements.push_back(newStmt);
    if (newStmt != stmtPtr) {
      changed = true;
      
      // If a for loop was unrolled, update environment variables immediately
      // so that subsequent statements can use the updated values
      if (auto forStmt = std::dynamic_pointer_cast<ForStatement>(stmtPtr)) {
        if (auto opStmt = std::dynamic_pointer_cast<OpStatement>(newStmt)) {
          // This for loop was unrolled, update environment variables
          UpdateEnvVarsAfterUnroll(stmt, forStmt, opStmt);
        }
      }
    }
  }
  
  // Restore previous compound
  currentCompound_ = prevCompound;
  
  // Create new CompoundStatement if any child was modified
  CompoundStatementPtr resultStmt = stmt;
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
    resultStmt = newStmt;
  }
  
  // Apply MergeOpStmt pass to merge consecutive OpStatements in this compound
  // This handles nested CompoundStatements as well
  MergeOpStmt mergePass;
  StatementPtr stmtPtr = std::static_pointer_cast<Statement>(resultStmt);
  StatementPtr mergedStmt = mergePass.VisitStmt(stmtPtr);
  CompoundStatementPtr mergedCompound = std::dynamic_pointer_cast<CompoundStatement>(mergedStmt);
  
  return mergedCompound ? mergedCompound : resultStmt;
}

StatementPtr UnrollStaticFor::VisitStmt_(OpStatementPtr& stmt) {
  if (!stmt) return stmt;
  
  bool changed = false;
  std::vector<OperationPtr> newOperations;
  
  // Visit and mutate all operations
  for (size_t i = 0; i < stmt->Operations().size(); ++i) {
    OperationPtr op = stmt->Operations()[i];
    if (!op) {
      newOperations.push_back(op);
      continue;
    }
    
    // Visit input operands
    std::vector<ValuePtr> newInputs;
    for (size_t j = 0; j < op->GetNumInputOperand(); ++j) {
      ValuePtr input = op->GetInputOperand(j);
      ValuePtr newInput = VisitValue(input);
      
      // Check if this input value's name exists in envTable and should be replaced
      if (currentCompound_ && input) {
        std::string inputName = input->GetName();
        if (!inputName.empty()) {
          ValuePtr envValue = currentCompound_->GetEnvVar(inputName);
          if (envValue && envValue != input) {
            newInput = envValue;
            changed = true;
          }
        }
      }
      
      newInputs.push_back(newInput);
      if (newInput != input) {
        changed = true;
      }
    }
    
    // Visit output operands
    std::vector<ValuePtr> newOutputs;
    for (size_t j = 0; j < op->GetNumOutputOperand(); ++j) {
      ValuePtr output = op->GetOutputOperand(j);
      ValuePtr newOutput = VisitValue(output);
      newOutputs.push_back(newOutput);
      if (newOutput != output) {
        changed = true;
      }
    }
    
    if (changed) {
      // Create new operation with updated operands
      auto newOp = std::make_shared<Operation>(
        op->GetOpcode(),
        newInputs,
        newOutputs,
        op->GetName()
      );
      newOp->iScalarIndex_ = op->iScalarIndex_;
      newOp->oScalarIndex_ = op->oScalarIndex_;
      newOp->Attributes() = op->Attributes();
      newOperations.push_back(newOp);
    } else {
      newOperations.push_back(op);
    }
  }
  
  if (changed) {
    auto newStmt = std::make_shared<OpStatement>();
    for (auto& op : newOperations) {
      newStmt->Operations().push_back(op);
    }
    newStmt->Attributes() = stmt->Attributes();
    return newStmt;
  }
  
  return stmt;
}

StatementPtr UnrollStaticFor::VisitStmt_(ForStatementPtr& stmt) {
  if (!stmt) return stmt;

  // Check if this is a static for loop that should be unrolled
  // We check this BEFORE visiting the loop body to avoid modifying it unnecessarily
  bool isStatic = IsStaticForLoop(stmt);
  
  if (isStatic) {
    int64_t iterationCount = GetIterationCount(stmt);
    
    if (iterationCount > 0 && iterationCount <= 100) { // Limit unrolling to reasonable sizes
      // Unroll the loop BEFORE visiting the loop body
      // This ensures we use the original loop body for unrolling
      StatementPtr unrolled = UnrollLoop(stmt, iterationCount);
      if (unrolled) {
        return unrolled;
      }
    }
  }

  // If not unrolled, recursively visit the loop body
  bool changed = false;

  // Visit and mutate loop body
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

  // If not unrolled, return the (possibly modified) original statement
  if (changed) {
    auto newForStmt = std::make_shared<ForStatement>(
      stmt->GetIterationVar(),
      stmt->GetStart(),
      stmt->GetEnd(),
      stmt->GetStep()
    );
    // Copy iter args
    for (const auto& iterArg : stmt->IterArgs()) {
      newForStmt->AddIterArg(iterArg.initValue);
    }
    // Set compound
    if (compound) {
      for (size_t i = 0; i < compound->GetStatementsNum(); ++i) {
        newForStmt->GetCompound()->AddStatement(compound->GetStatement(i));
      }
    }
    // Copy attributes
    newForStmt->Attributes() = stmt->Attributes();
    return newForStmt;
  }

  return stmt;
}

bool UnrollStaticFor::IsStaticForLoop(ForStatementPtr& stmt) {
  if (!stmt) {
    return false;
  }

  auto start = stmt->GetStart();
  auto end = stmt->GetEnd();
  auto step = stmt->GetStep();

  if (!start || !end || !step) {
    return false;
  }

  // Check if all bounds are immediate values (constants)
  // ScalarValue with ScalarValueKind::Immediate indicates a constant
  bool startIsImm = start->GetScalarValueKind() == ScalarValueKind::Immediate;
  bool endIsImm = end->GetScalarValueKind() == ScalarValueKind::Immediate;
  bool stepIsImm = step->GetScalarValueKind() == ScalarValueKind::Immediate;

  if (startIsImm && endIsImm && stepIsImm) {
    return true;
  }

  return false;
}

int64_t UnrollStaticFor::GetIterationCount(ForStatementPtr& stmt) {
  if (!IsStaticForLoop(stmt)) return -1;

  auto start = stmt->GetStart();
  auto end = stmt->GetEnd();
  auto step = stmt->GetStep();

  if (!start || !end || !step) return -1;

  // Extract immediate values using GetInt64Value
  int64_t startVal = 0, endVal = 0, stepVal = 1;

  if (start->HasImmediateValue()) {
    startVal = start->GetInt64Value();
  } else {
    return -1;
  }

  if (end->HasImmediateValue()) {
    endVal = end->GetInt64Value();
  } else {
    return -1;
  }

  if (step->HasImmediateValue()) {
    stepVal = step->GetInt64Value();
  } else {
    return -1;
  }

  if (stepVal == 0) return -1; // Invalid step

  // Calculate iteration count: ceil((end - start) / step)
  int64_t diff = endVal - startVal;
  if (stepVal > 0 && diff <= 0) return 0; // No iterations
  if (stepVal < 0 && diff >= 0) return 0; // No iterations

  int64_t count = (diff + stepVal - 1) / stepVal; // Ceiling division
  if (count < 0) count = -count;
  
  return count;
}

StatementPtr UnrollStaticFor::UnrollLoop(ForStatementPtr& stmt, int64_t iterationCount) {
  if (!stmt || iterationCount <= 0) {
    return nullptr;
  }

  // Get the loop body
  auto loopBody = stmt->GetCompound();
  if (!loopBody) {
    return nullptr;
  }

  // Create a new OpStatement to hold all unrolled operations
  auto unrolledOpStmt = std::make_shared<OpStatement>();

  // Collect all operations from the loop body
  std::vector<OperationPtr> allOps;
  
  // Traverse loop body to find all OpStatements and extract their operations
  for (size_t i = 0; i < loopBody->GetStatementsNum(); ++i) {
    StatementPtr bodyStmt = loopBody->GetStatement(i);
    if (!bodyStmt) {
      continue;
    }

    // Check if it's an OpStatement
    if (auto opStmt = std::dynamic_pointer_cast<OpStatement>(bodyStmt)) {
      const auto& ops = opStmt->Operations();
      
      // Extract all operations from this OpStatement
      for (const auto& op : ops) {
        if (op) {
          allOps.push_back(op);
        }
      }
    }
    // Skip YieldStatement and other non-op statements for now
    // (We only unroll the operations)
  }

  // If no operations found, return nullptr
  if (allOps.empty()) {
    return nullptr;
  }

  // Build mapping from iter_arg.value to iter_arg.initValue
  // This ensures that the first iteration uses initValue instead of value
  std::unordered_map<ValuePtr, ValuePtr> iterArgValueToInitValue;
  for (const auto& iterArg : stmt->IterArgs()) {
    if (iterArg.value && iterArg.initValue) {
      iterArgValueToInitValue[iterArg.value] = iterArg.initValue;
    }
  }

  // Map to track value updates across iterations
  // Maps original value -> current iteration's value
  // This ensures that loop-carried variables use the value from the previous iteration
  // Also track by name for loop-carried variables (same name, different values)
  std::unordered_map<ValuePtr, ValuePtr> valueMap;
  std::unordered_map<std::string, ValuePtr> nameToValueMap;  // Track loop-carried variables by name
  
  // For each iteration, copy all operations with updated values
  for (int64_t iter = 0; iter < iterationCount; ++iter) {
    for (const auto& op : allOps) {
      if (!op) continue;

      // Create a new operation with updated input operands
      std::vector<ValuePtr> inputOperands;
      for (size_t i = 0; i < op->GetNumInputOperand(); ++i) {
        ValuePtr originalInput = op->GetInputOperand(i);
        ValuePtr mappedInput = originalInput;
        
        // First, check if this is an iter_arg.value that should use initValue in first iteration
        if (iter == 0) {
          auto iterArgIt = iterArgValueToInitValue.find(originalInput);
          if (iterArgIt != iterArgValueToInitValue.end()) {
            mappedInput = iterArgIt->second;
            inputOperands.push_back(mappedInput);
            continue;
          }
        }
        
        // Check if this value is directly mapped (same value object)
        auto it = valueMap.find(originalInput);
        if (it != valueMap.end()) {
          mappedInput = it->second;
        } else {
          // Check if this is a loop-carried variable by name
          // If an output with the same name was created in a previous iteration, use it
          std::string inputName = originalInput->GetName();
          auto nameIt = nameToValueMap.find(inputName);
          if (nameIt != nameToValueMap.end()) {
            mappedInput = nameIt->second;
          }
        }
        
        inputOperands.push_back(mappedInput);
      }
      
      // Create new output operands for this iteration
      std::vector<ValuePtr> outputOperands;
      for (size_t i = 0; i < op->GetNumOutputOperand(); ++i) {
        ValuePtr originalOutput = op->GetOutputOperand(i);
        
        // Create a new value with a unique name for this iteration
        ValuePtr newOutput = CreateNewValue(originalOutput, iter, i);
        outputOperands.push_back(newOutput);
        
        // Update value map: next iteration will use this new value for this original value
        valueMap[originalOutput] = newOutput;
        
        // Also track by name for loop-carried variables
        // If output has the same name as a previous output, update the name map
        std::string outputName = originalOutput->GetName();
        if (!outputName.empty()) {
          nameToValueMap[outputName] = newOutput;
        }
      }

      // Create a new operation with updated operands
      auto newOp = std::make_shared<Operation>(
        op->GetOpcode(),
        inputOperands,
        outputOperands,
        op->GetName() + "_iter" + std::to_string(iter)
      );
      
      // Copy scalar indices
      newOp->iScalarIndex_ = op->iScalarIndex_;
      newOp->oScalarIndex_ = op->oScalarIndex_;
      
      // Copy attributes
      newOp->Attributes() = op->Attributes();
      
      unrolledOpStmt->Operations().push_back(newOp);
    }
  }

  // Copy attributes
  unrolledOpStmt->Attributes() = stmt->Attributes();

  return unrolledOpStmt;
}

ValuePtr UnrollStaticFor::CreateNewValue(ValuePtr originalValue, int64_t iteration, size_t outputIndex) {
  if (!originalValue) return nullptr;

  std::string newName = originalValue->GetName();
  if (newName.empty()) {
    newName = "unrolled_" + std::to_string(iteration) + "_" + std::to_string(outputIndex);
  } else {
    newName = newName + "_iter" + std::to_string(iteration);
  }

  ValueKind kind = originalValue->GetValueKind();
  DataType dataType = originalValue->GetDataType();

  if (kind == ValueKind::Tensor) {
    auto tensor = std::dynamic_pointer_cast<TensorValue>(originalValue);
    if (tensor) {
      // Create new tensor with same shape and type
      auto newTensor = std::make_shared<TensorValue>(
        tensor->GetShape(),
        dataType,
        newName,
        tensor->GetFormat()
      );
      return newTensor;
    }
  } else if (kind == ValueKind::Scalar) {
    auto scalar = std::dynamic_pointer_cast<ScalarValue>(originalValue);
    if (scalar) {
      // Create new scalar with same type
      auto newScalar = std::make_shared<ScalarValue>(
        dataType,
        newName,
        scalar->GetScalarValueKind()
      );
      return newScalar;
    }
  } else if (kind == ValueKind::Tile) {
    auto tile = std::dynamic_pointer_cast<TileValue>(originalValue);
    if (tile) {
      // Create new tile with same shape and type
      auto newTile = std::make_shared<TileValue>(
        tile->GetShape(),
        dataType,
        tile->GetValidShape(),
        newName
      );
      return newTile;
    }
  }

  // Fallback: return original value if we can't create a new one
  return originalValue;
}

void UnrollStaticFor::UpdateEnvVarsAfterUnroll(CompoundStatementPtr compound, ForStatementPtr originalForStmt, OpStatementPtr unrolledOpStmt) {
  if (!compound || !originalForStmt || !unrolledOpStmt) {
    return;
  }
  
  // Update environment variables in the compound with final values from unrolled loop
  // Track the final values of loop-carried variables by name and iteration
  // Map: variable_name -> (iteration_number, value)
  std::unordered_map<std::string, std::pair<int64_t, ValuePtr>> finalValues;
  
  // Iterate through all operations to find the last iteration's output for each loop-carried variable
  for (const auto& op : unrolledOpStmt->Operations()) {
    if (!op) continue;
    
    // Check outputs - these are the final values for loop-carried variables
    for (size_t i = 0; i < op->GetNumOutputOperand(); ++i) {
      ValuePtr output = op->GetOutputOperand(i);
      if (output) {
        std::string outputName = output->GetName();
        int64_t iterNum = -1;
        
        // Extract iteration number from name (e.g., "buf_iter4" -> 4)
        size_t iterPos = outputName.find("_iter");
        if (iterPos != std::string::npos) {
          std::string iterStr = outputName.substr(iterPos + 5); // "_iter" is 5 chars
          try {
            iterNum = std::stoll(iterStr);
          } catch (...) {
            // If parsing fails, skip
            continue;
          }
          outputName = outputName.substr(0, iterPos);
        }
        
        // Check if this is a loop-carried variable
        for (const auto& iterArg : originalForStmt->IterArgs()) {
          if (iterArg.initValue && iterArg.initValue->GetName() == outputName) {
            // Update if this is a later iteration
            auto it = finalValues.find(outputName);
            if (it == finalValues.end() || iterNum > it->second.first) {
              finalValues[outputName] = std::make_pair(iterNum, output);
            }
            break;
          }
        }
      }
    }
  }
  
  // Update compound environment with final values
  for (const auto& [varName, iterValue] : finalValues) {
    compound->SetEnvVar(varName, iterValue.second);
  }
}

} // namespace pto

