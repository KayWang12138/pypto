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
 * \file static_for_unroll.cpp
 * \brief Implementation of static for loop unrolling transformation
 */

#include "static_for_unroll.h"
#include "passes/pass_log/pass_log.h"
#include "interface/utils/common.h"
#include <vector>
#include <memory>

#define MODULE_NAME "StaticForUnroll"

namespace npu {
namespace tile_fwk {

Status StaticForUnrollTransform::Apply(const std::shared_ptr<pto::ProgramModule>& module) {
    if (!module) {
        APASS_LOG_ERROR_F(Elements::Function, "Null module provided to StaticForUnrollTransform");
        return FAILED;
    }

    // Process all functions in the module
    auto functions = module->GetFunctions();
    for (const auto& func : functions) {
        if (func->GetKind() == pto::FunctionKind::ControlFlow) {
            if (ProcessFunction(func) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Function, "Failed to process function: %s", 
                                  func->GetName().c_str());
                return FAILED;
            }
        }
    }

    // Also process entry function if it exists
    auto entry = module->GetProgramEntry();
    if (entry && entry->GetKind() == pto::FunctionKind::ControlFlow) {
        if (ProcessFunction(entry) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Function, "Failed to process entry function: %s", 
                              entry->GetName().c_str());
            return FAILED;
        }
    }

    return SUCCESS;
}

Status StaticForUnrollTransform::ProcessFunction(const std::shared_ptr<pto::Function>& func) {
    if (!func) {
        return FAILED;
    }

    APASS_LOG_INFO_F(Elements::Function, "Processing function: %s", func->GetName().c_str());

    auto compound = func->GetCompound();
    if (!compound) {
        APASS_LOG_ERROR_F(Elements::Function, "Function %s has no compound statement", 
                          func->GetName().c_str());
        return FAILED;
    }

    return ProcessCompound(compound);
}

Status StaticForUnrollTransform::ProcessCompound(const pto::CompoundStatementPtr& compound) {
    if (!compound) {
        return FAILED;
    }

    size_t numStmts = compound->GetStatementsNum();
    
    // Process statements, handling unrolling which may change the statement count
    for (size_t i = 0; i < compound->GetStatementsNum(); ) {
        auto stmt = compound->GetStatement(i);
        
        if (!stmt) {
            i++;
            continue;
        }

        // Check for IfStatement - not allowed
        if (stmt->GetKind() == pto::StatementKind::If) {
            APASS_LOG_ERROR_F(Elements::Operation, 
                              "IfStatement found at index %zu, only static for loops are allowed", i);
            return FAILED;
        }

        // Process ForStatement
        if (stmt->GetKind() == pto::StatementKind::For) {
            auto forStmt = std::dynamic_pointer_cast<pto::ForStatement>(stmt);
            if (!forStmt) {
                APASS_LOG_ERROR_F(Elements::Operation, "Failed to cast to ForStatement at index %zu", i);
                return FAILED;
            }

            // Check if it's a static for loop
            int64_t start, end, step;
            if (!IsStaticImmediate(forStmt->GetStart(), start)) {
                APASS_LOG_ERROR_F(Elements::Operation, 
                                  "ForStatement start is not static immediate at index %zu", i);
                return FAILED;
            }
            if (!IsStaticImmediate(forStmt->GetEnd(), end)) {
                APASS_LOG_ERROR_F(Elements::Operation, 
                                  "ForStatement end is not static immediate at index %zu", i);
                return FAILED;
            }
            if (!IsStaticImmediate(forStmt->GetStep(), step)) {
                APASS_LOG_ERROR_F(Elements::Operation, 
                                  "ForStatement step is not static immediate at index %zu", i);
                return FAILED;
            }

            if (step <= 0) {
                APASS_LOG_ERROR_F(Elements::Operation, 
                                  "ForStatement step must be positive, got %ld at index %zu", step, i);
                return FAILED;
            }

            // Recursively process the loop body first to handle nested loops
            if (ProcessCompound(forStmt->GetCompound()) != SUCCESS) {
                return FAILED;
            }

            // Unroll the loop
            if (UnrollForLoop(forStmt, compound, i) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Operation, 
                                  "Failed to unroll ForStatement at index %zu", i);
                return FAILED;
            }

            // After unrolling, the ForStatement is replaced with unrolled statements
            // Don't increment i, check the same index again
            continue;
        }

        i++;
    }

    return SUCCESS;
}

bool StaticForUnrollTransform::IsStaticImmediate(const pto::ScalarValuePtr& value, int64_t& result) {
    if (!value) {
        return false;
    }

    // Check if it's an immediate constant
    if (value->GetScalarValueKind() == pto::ScalarValueKind::Immediate) {
        result = value->GetInt64Value();
        return true;
    }

    return false;
}

Status StaticForUnrollTransform::UnrollForLoop(const pto::ForStatementPtr& forStmt,
                                                const pto::CompoundStatementPtr& parentCompound,
                                                size_t stmtIndex) {
    if (!forStmt || !parentCompound) {
        return FAILED;
    }

    int64_t start, end, step;
    if (!IsStaticImmediate(forStmt->GetStart(), start) ||
        !IsStaticImmediate(forStmt->GetEnd(), end) ||
        !IsStaticImmediate(forStmt->GetStep(), step)) {
        return FAILED;
    }

    auto iterVar = forStmt->GetIterationVar();
    auto loopBody = forStmt->GetCompound();

    APASS_LOG_DEBUG_F(Elements::Operation, 
                      "Unrolling loop: start=%ld, end=%ld, step=%ld", start, end, step);

    // Calculate iteration count
    int64_t iterCount = 0;
    for (int64_t i = start; i < end; i += step) {
        iterCount++;
    }

    APASS_LOG_DEBUG_F(Elements::Operation, "Loop iteration count: %ld", iterCount);

    // Collect cloned statements for all iterations
    std::vector<pto::StatementPtr> unrolledStmts;

    int64_t iterIndex = 0;
    for (int64_t i = start; i < end; i += step) {
        // Create a constant value for this iteration
        auto iterValue = std::make_shared<pto::ScalarValue>(i, "iter_" + std::to_string(i));

        // Clone each statement in the loop body
        size_t bodyStmtCount = loopBody->GetStatementsNum();
        for (size_t j = 0; j < bodyStmtCount; ++j) {
            auto bodyStmt = loopBody->GetStatement(j);
            
            // Skip yield statements as they're only for loop-carried values
            if (bodyStmt->GetKind() == pto::StatementKind::Yield) {
                continue;
            }

            auto clonedStmt = CloneStatement(bodyStmt, iterVar, iterValue, iterIndex);
            if (clonedStmt) {
                unrolledStmts.push_back(clonedStmt);
            }
        }
        iterIndex++;
    }

    // Replace the ForStatement with unrolled statements in the parent compound
    // Build new statement list
    std::vector<pto::StatementPtr> newStmts;
    for (size_t i = 0; i < parentCompound->GetStatementsNum(); ++i) {
        if (i == stmtIndex) {
            // Insert unrolled statements here
            for (const auto& stmt : unrolledStmts) {
                newStmts.push_back(stmt);
            }
        } else {
            newStmts.push_back(parentCompound->GetStatement(i));
        }
    }

    // Update the parent compound's statements by overwriting them
    for (size_t i = 0; i < newStmts.size(); ++i) {
        if (i < parentCompound->GetStatementsNum()) {
            parentCompound->SetStatement(i, newStmts[i]);
        } else {
            parentCompound->AddStatement(newStmts[i]);
        }
    }

    APASS_LOG_INFO_F(Elements::Operation, "Successfully unrolled loop with %ld iterations", iterCount);
    return SUCCESS;
}

pto::StatementPtr StaticForUnrollTransform::CloneStatement(const pto::StatementPtr& stmt,
                                                           const pto::ScalarValuePtr& iterVar,
                                                           const pto::ScalarValuePtr& iterValue,
                                                           int64_t iterIndex) {
    if (!stmt) {
        return nullptr;
    }

    // Currently only handle OpStatement
    // ForStatement should already be processed recursively
    if (stmt->GetKind() == pto::StatementKind::Op) {
        auto opStmt = std::dynamic_pointer_cast<pto::OpStatement>(stmt);
        if (!opStmt) {
            return nullptr;
        }

        // Create a new OpStatement to hold the cloned operations
        auto clonedOpStmt = std::make_shared<pto::OpStatement>();
        
        // Track value mapping from old to new (for replacing references)
        std::unordered_map<pto::ValuePtr, pto::ValuePtr> valueMap;
        
        // Map the iteration variable to its constant value
        valueMap[iterVar] = iterValue;
        
        // Clone each operation in the OpStatement
        for (const auto& op : opStmt->Operations()) {
            if (!op) {
                continue;
            }
            
            // Clone input operands, replacing iteration variable references
            std::vector<pto::ValuePtr> clonedInputs;
            for (size_t i = 0; i < op->GetNumInputOperand(); ++i) {
                auto inputOperand = op->GetInputOperand(i);
                // Check if this input has been mapped (from previous operations in this iteration)
                auto it = valueMap.find(inputOperand);
                if (it != valueMap.end()) {
                    // Use the mapped value (either iter var replacement or output from previous op)
                    clonedInputs.push_back(it->second);
                } else {
                    // This is an external value (function parameter or constant), use as-is
                    clonedInputs.push_back(inputOperand);
                }
            }
            
            // Clone output operands, creating new values with iteration-specific names
            std::vector<pto::ValuePtr> clonedOutputs;
            for (size_t i = 0; i < op->GetNumOutputOperand(); ++i) {
                auto outputOperand = op->GetOutputOperand(i);
                auto clonedOutput = CloneValue(outputOperand, valueMap, iterIndex);
                clonedOutputs.push_back(clonedOutput);
                
                // Remember this mapping for future operations that may use this value
                valueMap[outputOperand] = clonedOutput;
            }
            
            // Create a new operation with cloned operands
            auto clonedOp = std::make_shared<pto::Operation>(
                op->GetOpcode(), 
                clonedInputs, 
                clonedOutputs,
                op->GetName()
            );
            
            // Add the cloned operation to the new OpStatement
            clonedOpStmt->Operations().push_back(clonedOp);
        }
        
        return clonedOpStmt;
    }

    return stmt;
}

pto::ValuePtr StaticForUnrollTransform::CloneValue(const pto::ValuePtr& value,
                                                   std::unordered_map<pto::ValuePtr, pto::ValuePtr>& valueMap,
                                                   int64_t iterIndex) {
    if (!value) {
        return nullptr;
    }

    // Check if this value has already been cloned in this iteration
    auto it = valueMap.find(value);
    if (it != valueMap.end()) {
        return it->second;
    }

    // Clone the value based on its type
    pto::ValuePtr clonedValue;
    
    if (value->GetValueKind() == pto::ValueKind::Scalar) {
        auto scalarValue = std::dynamic_pointer_cast<pto::ScalarValue>(value);
        if (!scalarValue) {
            return value;
        }
        
        // If it's an immediate value, just return it as-is (constants are shared)
        if (scalarValue->GetScalarValueKind() == pto::ScalarValueKind::Immediate) {
            return value;
        }
        
        // For symbolic scalars, create a new one with iteration-specific name
        std::string newName = scalarValue->GetName();
        if (!newName.empty()) {
            newName += "_unroll_" + std::to_string(iterIndex);
        }
        
        clonedValue = std::make_shared<pto::ScalarValue>(
            scalarValue->GetDataType(),
            newName,
            scalarValue->GetScalarValueKind()
        );
        
    } else if (value->GetValueKind() == pto::ValueKind::Tensor) {
        auto tensorValue = std::dynamic_pointer_cast<pto::TensorValue>(value);
        if (!tensorValue) {
            return value;
        }
        
        // Create a new tensor with iteration-specific name
        std::string newName = tensorValue->GetName();
        if (!newName.empty()) {
            newName += "_unroll_" + std::to_string(iterIndex);
        }
        
        // Clone the shape (may contain symbolic scalars)
        std::vector<pto::ScalarValuePtr> clonedShape;
        for (const auto& dim : tensorValue->GetShape()) {
            auto clonedDim = std::dynamic_pointer_cast<pto::ScalarValue>(
                CloneValue(dim, valueMap, iterIndex)
            );
            clonedShape.push_back(clonedDim);
        }
        
        clonedValue = std::make_shared<pto::TensorValue>(
            clonedShape,
            tensorValue->GetDataType(),
            newName,
            tensorValue->GetFormat()
        );
        
    } else if (value->GetValueKind() == pto::ValueKind::Tile) {
        auto tileValue = std::dynamic_pointer_cast<pto::TileValue>(value);
        if (!tileValue) {
            return value;
        }
        
        // Create a new tile with iteration-specific name
        std::string newName = tileValue->GetName();
        if (!newName.empty()) {
            newName += "_unroll_" + std::to_string(iterIndex);
        }
        
        // Clone the valid shape
        std::vector<pto::ScalarValuePtr> clonedValidShape;
        for (const auto& dim : tileValue->GetValidShape()) {
            auto clonedDim = std::dynamic_pointer_cast<pto::ScalarValue>(
                CloneValue(dim, valueMap, iterIndex)
            );
            clonedValidShape.push_back(clonedDim);
        }
        
        clonedValue = std::make_shared<pto::TileValue>(
            tileValue->GetShape(),
            tileValue->GetDataType(),
            clonedValidShape,
            newName
        );
        
    } else {
        // Unknown value kind, return as-is
        return value;
    }
    
    // Cache the cloned value
    valueMap[value] = clonedValue;
    
    return clonedValue;
}

} // namespace tile_fwk
} // namespace npu
