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

std::string GetValueInfoString(const Value *valuePtr) {
    std::string valueInfo;
    if (valuePtr) {
        // Determine the type name based on ValueKind
        switch (valuePtr->GetValueKind()) {
            case ValueKind::Tile: valueInfo = "TileValue"; break;
            case ValueKind::Scalar: valueInfo = "ScalarValue"; break;
            case ValueKind::Tensor: valueInfo = "TensorValue"; break;
            default: valueInfo = "Value"; break;
        }

        std::string ssaName = valuePtr->GetSSAName();
        if (!ssaName.empty()) {
            valueInfo += " '" + ssaName + "'";
        }
        valueInfo += " (ID: " + std::to_string(valuePtr->GetID()) + ")";
    } else {
        valueInfo = "Value";
    }
    return valueInfo;
}

void ValueSSAVisitor::VisitImplOp(OperationPtr &op) {
    if (!op)
        return;

    // Count the number of times each Value is used as output operand of this operation
    for (size_t i = 0; i < op->GetNumOutputOperand(); ++i) {
        ValuePtr outputOperand = op->GetOutputOperand(i);
        if (outputOperand && valueCountMap_) {
            (*valueCountMap_)[outputOperand.get()]++;
        }
    }
}

// ---- Concrete ops (auto-generated from *.def) ----
#define DEFOP(name, inherit, opcode, ...)                             \
    void ValueSSAVisitor::VisitImplOp(name##Ptr &op) {                \
        OperationPtr opPtr = std::static_pointer_cast<Operation>(op); \
        VisitImplOp(opPtr);                                           \
    }
#include "ir/operation.def"
#include "ir/tile_graph.def"
#undef DEFOP

VerifyResult VerifySSA(ProgramModulePtr program) {
    if (!program) {
        return {VerifyStatus::FAIL, "ProgramModule is null, cannot verify SSA."};
    }

    std::map<const Value *, size_t> valueSSACountMap;
    ValueSSAVisitor visitor(&valueSSACountMap);
    ProgramModulePtr programPtr = program;
    visitor.VisitProgram(programPtr);
    std::vector<std::string> violations;

    // Check each Value that has more than one definition
    for (const auto &[valuePtr, count] : valueSSACountMap) {
        if (count > 1) {
            std::string valueInfo = GetValueInfoString(valuePtr);
            valueInfo += " has " + std::to_string(count) + " definitions, expected at most 1.";
            violations.push_back(valueInfo);
        }
    }

    if (!violations.empty()) {
        std::string errorMsg = "SSA semantics violation - " + std::to_string(violations.size()) +
                               " Value(s) have more than one definition:\n";
        for (size_t i = 0; i < violations.size(); ++i) {
            errorMsg += "  " + std::to_string(i + 1) + ". " + violations[i];
            if (i + 1 < violations.size()) {
                errorMsg += "\n";
            }
        }
        return {VerifyStatus::FAIL, errorMsg};
    }

    return {VerifyStatus::PASS, ""};
}

} // namespace pto
