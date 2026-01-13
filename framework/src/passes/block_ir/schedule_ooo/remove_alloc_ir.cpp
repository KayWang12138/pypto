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
 * \file remove_alloc.cpp
 * \brief
 */

#include "remove_alloc_ir.h"
#include "ir/opcode.h"
#include "ir/statement.h"
#include <algorithm>

namespace npu::tile_fwk {

// Helper function to remove ALLOC operations from a compound statement recursively
static void RemoveAllocFromCompound(const pto::CompoundStatementPtr& compound) {
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
                    auto& operations = opStmt->Operations();
                    operations.erase(
                        std::remove_if(operations.begin(), operations.end(),
                            [](const pto::OperationPtr& op) {
                                if (!op) {
                                    return false;
                                }
                                std::string opcodeName = pto::GetOpcodeName(op->GetOpcode());
                                return opcodeName.find("ALLOC") != std::string::npos;
                            }),
                        operations.end());
                }
                break;
            }
            case pto::StatementKind::For: {
                auto forStmt = std::static_pointer_cast<pto::ForStatement>(stmt);
                if (forStmt && forStmt->GetCompound()) {
                    RemoveAllocFromCompound(forStmt->GetCompound());
                }
                break;
            }
            case pto::StatementKind::If: {
                auto ifStmt = std::static_pointer_cast<pto::IfStatement>(stmt);
                if (ifStmt) {
                    if (ifStmt->GetThenCompound()) {
                        RemoveAllocFromCompound(ifStmt->GetThenCompound());
                    }
                    if (ifStmt->GetElseCompound()) {
                        RemoveAllocFromCompound(ifStmt->GetElseCompound());
                    }
                }
                break;
            }
            default:
                // Other statement types don't contain operations
                break;
        }
    }
}

void RemoveAllocIR::RemoveAllocCall(pto::Function &function) const {
    auto compound = function.GetCompound();
    RemoveAllocFromCompound(compound);
}
} // namespace npu::tile_fwk
