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
 * \file pass_utils.cpp
 * \brief
 */

#include "block_pass_utils.h"
#include "ir/opcode.h"
#include "ir/statement.h"

namespace npu::tile_fwk {

std::vector<pto::OperationPtr> BlockPassUtils::GetBlockFunctionOperations(pto::Function &function) {
    for (size_t index = 0; index < function.BodyStmtsNum(); index++) {
        if (function.GetBodyStatement(index)->GetKind() == pto::StatementKind::Op) {
            auto opStatement = std::dynamic_pointer_cast<pto::OpStatement>(function.GetBodyStatement(index));
            return opStatement->Operations();
        }
    }
    return {};
}

bool BlockPassUtils::IsCopyInOrOut(const pto::Opcode &opcode) {
    return IsCopyIn(opcode) || IsCopyOut(opcode);
}

bool BlockPassUtils::IsCopyIn(const pto::Opcode &opcode) {
    return opcode == pto::Opcode::OP_L1_COPY_IN || 
           opcode == pto::Opcode::OP_L1_TO_FIX_QUANT_PRE ||
           opcode == pto::Opcode::OP_L1_TO_BT;
}

bool BlockPassUtils::IsCopyOut(const pto::Opcode &opcode) {
    return opcode == pto::Opcode::OP_L0C_COPY_OUT;
}

// Helper function to get producer operations for a given operation
// In IR, we need to find operations that produce the input values of this operation
std::unordered_set<pto::Operation *> BlockPassUtils::GetProducerOps(
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

// Helper function to get consumer operations for a given operation
// In IR, we need to find operations that consume the output values of this operation
std::unordered_set<pto::Operation *> BlockPassUtils::GetConsumerOps(
    const pto::Operation &op,
    const std::vector<pto::OperationPtr> &allOps) {
    std::unordered_set<pto::Operation *> consumers;
    // For each output operand of the given op, find operations that use it as input
    for (size_t i = 0; i < op.GetNumOutputOperand(); ++i) {
        auto outputValue = op.GetOutputOperand(i);
        if (!outputValue) {
            continue;
        }
        for (const auto &candidateOp : allOps) {
            if (!candidateOp) {
                continue;
            }
            for (size_t j = 0; j < candidateOp->GetNumInputOperand(); ++j) {
                if (candidateOp->GetInputOperand(j) == outputValue) {
                    consumers.insert(candidateOp.get());
                    break;
                }
            }
        }
    }
    return consumers;
}

} // namespace npu::tile_fwk
