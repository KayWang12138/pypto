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
 * \file pairsum_replacement.cpp
 * \brief Implementation of PairSumReplacementPass
 */

#include "pairsum_replacement.h"
#include "interface/operation/operation.h"
#include "interface/tensor/logical_tensor.h"
#include "passes/pass_log/pass_log.h"

#define MODULE_NAME "PairSumReplacementPass"

namespace npu {
namespace tile_fwk {

Status PairSumReplacementPass::RunOnFunction(Function &function) {
    APASS_LOG_INFO_F(Elements::Function, "===> Start PairSumReplacementPass");
    
    int replacedCount = 0;
    
    // Iterate through all operations in the function
    for (auto &op : function.Operations(false)) {
        // Check if this is an OP_PAIRSUM operation
        if (op.GetOpcode() != Opcode::OP_PAIRSUM) {
            continue;
        }
        
        // Check if replacement should occur
        if (ShouldReplacePairSum(op)) {
            ReplacePairSumWithAdd(op);
            replacedCount++;
        }
    }
    
    APASS_LOG_INFO_F(Elements::Function, "===> End PairSumReplacementPass, replaced %d operations", replacedCount);
    return SUCCESS;
}

bool PairSumReplacementPass::ShouldReplacePairSum(const Operation &op) const {
    // Precondition checks
    const auto &inputs = op.GetIOperands();
    const auto &outputs = op.GetOOperands();
    
    // Verify input and output counts
    if (inputs.size() != 2) {
        APASS_LOG_DEBUG_F(Elements::Operation, "OP_PAIRSUM[%d] not replaced: input count is %zu, expected 2", 
            op.GetOpMagic(), inputs.size());
        return false;
    }
    
    if (outputs.size() != 1) {
        APASS_LOG_DEBUG_F(Elements::Operation, "OP_PAIRSUM[%d] not replaced: output count is %zu, expected 1", 
            op.GetOpMagic(), outputs.size());
        return false;
    }
    
    // Check for null pointers
    if (inputs[0] == nullptr || inputs[1] == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation, "OP_PAIRSUM[%d] has null input tensor.%s", 
            op.GetOpMagic(), GetFormatBacktrace(op).c_str());
        return false;
    }
    
    if (outputs[0] == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation, "OP_PAIRSUM[%d] has null output tensor.%s", 
            op.GetOpMagic(), GetFormatBacktrace(op).c_str());
        return false;
    }
    
    // Get the two input tensors
    const auto &input0 = inputs[0];
    const auto &input1 = inputs[1];
    
    // Get valid shapes
    const auto &validShape0 = input0->GetDynValidShape();
    const auto &validShape1 = input1->GetDynValidShape();
    
    // Case 1: Both tensors have no valid_shape (static shape scenario)
    if (validShape0.empty() && validShape1.empty()) {
        APASS_LOG_DEBUG_F(Elements::Operation, "OP_PAIRSUM[%d] can be replaced: both inputs have static shape", 
            op.GetOpMagic());
        return true;
    }
    
    // Case 2: Both tensors have valid_shape and they match
    if (!validShape0.empty() && !validShape1.empty()) {
        // Check dimension count
        if (validShape0.size() != validShape1.size()) {
            APASS_LOG_DEBUG_F(Elements::Operation, 
                "OP_PAIRSUM[%d] not replaced: valid_shape dimension mismatch (%zu vs %zu)", 
                op.GetOpMagic(), validShape0.size(), validShape1.size());
            return false;
        }
        
        // Check if valid_shapes match
        if (CheckValidShapeMatch(input0, input1)) {
            APASS_LOG_DEBUG_F(Elements::Operation, "OP_PAIRSUM[%d] can be replaced: valid_shape matches", 
                op.GetOpMagic());
            return true;
        } else {
            APASS_LOG_DEBUG_F(Elements::Operation, "OP_PAIRSUM[%d] not replaced: valid_shape does not match", 
                op.GetOpMagic());
            return false;
        }
    }
    
    // Case 3: One has valid_shape, one doesn't (mixed scenario)
    APASS_LOG_DEBUG_F(Elements::Operation, 
        "OP_PAIRSUM[%d] not replaced: mixed valid_shape scenario (one has valid_shape, one doesn't)", 
        op.GetOpMagic());
    return false;
}

bool PairSumReplacementPass::CheckValidShapeMatch(const LogicalTensorPtr &tensor1, 
                                                    const LogicalTensorPtr &tensor2) const {
    const auto &validShape1 = tensor1->GetDynValidShape();
    const auto &validShape2 = tensor2->GetDynValidShape();
    
    // Check dimension count
    if (validShape1.size() != validShape2.size()) {
        return false;
    }
    
    // Compare each dimension
    for (size_t i = 0; i < validShape1.size(); ++i) {
        const auto &dim1 = validShape1[i];
        const auto &dim2 = validShape2[i];
        
        // Case 1: Both are concrete values
        if (dim1.ConcreteValid() && dim2.ConcreteValid()) {
            if (dim1.Concrete() != dim2.Concrete()) {
                APASS_LOG_DEBUG_F(Elements::Operation, 
                    "valid_shape mismatch at dimension %zu: %lld vs %lld", 
                    i, dim1.Concrete(), dim2.Concrete());
                return false;
            }
        }
        // Case 2: Both are symbolic expressions
        else if (!dim1.ConcreteValid() && !dim2.ConcreteValid()) {
            // Compare symbolic expressions using Dump() for string comparison
            if (dim1.Dump() != dim2.Dump()) {
                APASS_LOG_DEBUG_F(Elements::Operation, 
                    "valid_shape mismatch at dimension %zu: symbolic expressions differ (%s vs %s)", 
                    i, dim1.Dump().c_str(), dim2.Dump().c_str());
                return false;
            }
        }
        // Case 3: One concrete, one symbolic - not equal
        else {
            APASS_LOG_DEBUG_F(Elements::Operation, 
                "valid_shape mismatch at dimension %zu: one concrete, one symbolic", i);
            return false;
        }
    }
    
    return true;
}

void PairSumReplacementPass::ReplacePairSumWithAdd(Operation &op) {
    // Record information for logging
    int opMagic = op.GetOpMagic();
    auto inputs = op.GetIOperands();
    auto outputs = op.GetOOperands();
    
    APASS_LOG_INFO_F(Elements::Operation, 
        "Replace OP_PAIRSUM[%d] with OP_ADD, inputs: tensor[%d], tensor[%d], output: tensor[%d]",
        opMagic, 
        inputs[0]->magic, 
        inputs[1]->magic, 
        outputs[0]->magic);
    
    // Change opcode from OP_PAIRSUM to OP_ADD
    op.SetOpCode(Opcode::OP_ADD);
    
    // Remove excludeBufferReuse attribute if it exists
    // This is a static attribute of OP_PAIRSUM that should not be present on OP_ADD
    if (op.HasAttribute(OpAttributeKey::excludeBufferReuse)) {
        op.RemoveAttr(OpAttributeKey::excludeBufferReuse);
        APASS_LOG_DEBUG_F(Elements::Operation, 
            "Removed excludeBufferReuse attribute from operation %d", opMagic);
    }
    
    // Other attributes (like inputCombineAxis) are automatically preserved through AttrHolder
    // No need to explicitly copy them
    
    APASS_LOG_DEBUG_F(Elements::Operation, 
        "Successfully replaced operation %d from OP_PAIRSUM to OP_ADD", opMagic);
}

} // namespace tile_fwk
} // namespace npu
