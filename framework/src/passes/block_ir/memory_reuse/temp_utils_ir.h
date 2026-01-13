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
 * \file temp_utils_ir.h
 * \brief Utility functions for IR operations
 */

#pragma once

#include <vector>
#include <unordered_set>
#include "ir/function.h"
#include "ir/operation.h"
#include "ir/statement.h"

namespace npu::tile_fwk {

// Helper function to collect all operations from a CompoundStatement recursively
inline void CollectOperationsFromCompound(const pto::CompoundStatementPtr& compound,
                                          std::vector<pto::OperationPtr>& operations) {
    if (!compound) {
        return;
    }
    
    for (size_t i = 0; i < compound->GetStatementsNum(); ++i) {
        auto stmt = compound->GetStatement(i);
        if (!stmt) {
            continue;
        }
        
        switch (stmt->GetKind()) {
            case pto::StatementKind::Op: {
                auto opStmt = std::static_pointer_cast<pto::OpStatement>(stmt);
                if (opStmt) {
                    for (const auto& op : opStmt->Operations()) {
                        if (op) {
                            operations.push_back(op);
                        }
                    }
                }
                break;
            }
            case pto::StatementKind::For: {
                auto forStmt = std::static_pointer_cast<pto::ForStatement>(stmt);
                if (forStmt && forStmt->GetCompound()) {
                    CollectOperationsFromCompound(forStmt->GetCompound(), operations);
                }
                break;
            }
            case pto::StatementKind::If: {
                auto ifStmt = std::static_pointer_cast<pto::IfStatement>(stmt);
                if (ifStmt) {
                    if (ifStmt->GetThenCompound()) {
                        CollectOperationsFromCompound(ifStmt->GetThenCompound(), operations);
                    }
                    if (ifStmt->GetElseCompound()) {
                        CollectOperationsFromCompound(ifStmt->GetElseCompound(), operations);
                    }
                }
                break;
            }
            default:
                // Other statement types (Compound, Yield, Call, Return) don't contain operations
                break;
        }
    }
}

// Helper function to get all operations from a pto::Function
inline std::vector<pto::OperationPtr> GetAllOperations(pto::Function *func) {
    std::vector<pto::OperationPtr> operations;
    if (!func) {
        return operations;
    }
    
    auto compound = func->GetCompound();
    CollectOperationsFromCompound(compound, operations);
    
    return operations;
}

// Helper function to get producer operations for a given operation
// In IR, we need to find operations that produce the input values of this operation
inline std::unordered_set<pto::Operation *> GetProducerOps(
    const pto::Operation &op,
    const std::vector<pto::OperationPtr> &allOps) {
    std::unordered_set<pto::Operation *> producers;
    
    // For each input operand, find the operation that produces it
    for (size_t i = 0; i < op.GetNumInputOperand(); ++i) {
        auto inputValue = op.GetInputOperand(i);
        if (!inputValue) {
            continue;
        }
        
        // Find the operation that produces this value
        for (const auto &candidateOp : allOps) {
            if (!candidateOp) {
                continue;
            }
            
            // Check if this operation produces the input value
            for (size_t j = 0; j < candidateOp->GetNumOutputOperand(); ++j) {
                if (candidateOp->GetOutputOperand(j) == inputValue) {
                    producers.insert(candidateOp.get());
                    break;
                }
            }
        }
    }
    
    return producers;
}

} // namespace npu::tile_fwk

