/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ir/verifier/ssa_verify.h"

#include <map>
#include <string>
#include <vector>

namespace pto {

void TileValueSSAVisitor::VisitOp_(OperationPtr &op) {
    if (!op)
        return;

    // Count how many times each TileValue is used as input in this operation
    for (size_t i = 0; i < op->GetNumInputOperand(); ++i) {
        ValuePtr input = op->GetInputOperand(i);
        if (input) {
            auto tileInput = std::dynamic_pointer_cast<TileValue>(input);
            if (tileInput) {
                // Use the raw pointer as key to identify unique TileValue instances
                const TileValue *tilePtr = tileInput.get();
                (*inputCountMap_)[tilePtr]++;
            }
        }
    }
}

// ---- Concrete ops (auto-generated from *.def) ----
#define DEFOP(name, inherit, opcode, ...)                             \
    void TileValueSSAVisitor::VisitOp_(name##Ptr &op) {               \
        OperationPtr opPtr = std::static_pointer_cast<Operation>(op); \
        VisitOp_(opPtr);                                              \
    }
#include "ir/operation.def"
#include "ir/tile_graph.def"
#undef DEFOP

VerifyResult RuleVerifySSASingleInput(ProgramModulePtr program) {
    if (!program) {
        return {false, "ProgramModule is null, cannot verify SSA single input"};
    }

    std::map<const TileValue *, size_t> inputCountMap;
    TileValueSSAVisitor visitor(&inputCountMap);
    ProgramModulePtr programPtr = program;
    visitor.VisitProgram(programPtr);
    std::vector<std::string> violations;

    // Check each TileValue
    for (const auto &[tilePtr, count] : inputCountMap) {
        if (count != 1) {
            std::string tileInfo = "TileValue";
            if (tilePtr) {
                std::string ssaName = tilePtr->GetSSAName();
                if (!ssaName.empty()) {
                    tileInfo += " '" + ssaName + "'";
                }
                tileInfo += " (ID: " + std::to_string(tilePtr->GetID()) + ")";
            }
            tileInfo += " has " + std::to_string(count) + " input(s), expected exactly 1";
            violations.push_back(tileInfo);
        }
    }

    if (!violations.empty()) {
        std::string errorMsg = "SSA semantics violation - " + std::to_string(violations.size()) +
                               " TileValue(s) have incorrect input count:\n";
        for (size_t i = 0; i < violations.size(); ++i) {
            errorMsg += "  " + std::to_string(i + 1) + ". " + violations[i];
            if (i + 1 < violations.size()) {
                errorMsg += "\n";
            }
        }
        return {false, errorMsg};
    }

    return {true, ""};
}

} // namespace pto
