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
 * \file vjp_reduction.cpp
 * \brief VJP rules for reduction operations.
 *
 * Corresponds to python/pypto/autograd/rules/reduction.py
 */

#include "../vjp_registry.h"
#include "tilefwk/tilefwk_op.h"

namespace npu::tile_fwk {

namespace {

// Helper: Create tensor
LogicalTensorPtr CreateTensor(VJPContext& ctx, const Shape& shape, DataType dtype) {
    if (ctx.function == nullptr) {
        return nullptr;
    }
    return std::make_shared<LogicalTensor>(*ctx.function, dtype, shape);
}

// Helper: Create comparison operation (input == other)
LogicalTensorPtr CreateCmpEqOp(VJPContext& ctx, LogicalTensorPtr input, LogicalTensorPtr other) {
    if (input == nullptr || other == nullptr || ctx.function == nullptr) {
        return nullptr;
    }
    auto output = CreateTensor(ctx, input->GetShape(), DataType::DT_BOOL);
    if (output == nullptr) {
        return nullptr;
    }
    Operation& cmpOp = ctx.function->AddRawOperation(Opcode::OP_CMP, {input, other}, {output});
    cmpOp.SetAttr(OP_ATTR_PREFIX + "cmp_operation", static_cast<int64_t>(OpType::EQ));
    cmpOp.SetAttr(OP_ATTR_PREFIX + "cmp_mode", static_cast<int64_t>(OutType::BOOL));
    return output;
}

// Helper: Create cast operation
LogicalTensorPtr CreateCastOp(VJPContext& ctx, LogicalTensorPtr input, DataType targetDtype) {
    if (input == nullptr || ctx.function == nullptr) {
        return nullptr;
    }
    if (input->Datatype() == targetDtype) {
        return input;
    }
    auto output = CreateTensor(ctx, input->GetShape(), targetDtype);
    if (output == nullptr) {
        return nullptr;
    }
    Operation& castOp = ctx.function->AddRawOperation(Opcode::OP_CAST, {input}, {output});
    castOp.SetAttribute(OP_ATTR_PREFIX + "mode", CastMode::CAST_NONE);
    return output;
}

// Helper: Create mul operation
LogicalTensorPtr CreateMulOp(VJPContext& ctx, LogicalTensorPtr a, LogicalTensorPtr b) {
    if (a == nullptr || b == nullptr || ctx.function == nullptr) {
        return nullptr;
    }
    auto output = CreateTensor(ctx, a->GetShape(), a->Datatype());
    if (output == nullptr) {
        return nullptr;
    }
    ctx.function->AddRawOperation(Opcode::OP_MUL, {a, b}, {output});
    return output;
}

// Helper: Create expand/broadcast operation
LogicalTensorPtr CreateExpandOp(VJPContext& ctx, LogicalTensorPtr input, const Shape& targetShape) {
    if (input == nullptr || ctx.function == nullptr) {
        return nullptr;
    }

    const Shape& inputShape = input->GetShape();

    // Check if shapes already match
    if (inputShape == targetShape) {
        return input;
    }

    auto output = CreateTensor(ctx, targetShape, input->Datatype());
    Operation& expandOp = ctx.function->AddRawOperation(Opcode::OP_EXPAND, {input}, {output});

    // Set required EXPANDDIM attribute
    // Find which dimensions are expanded (from 1 to larger)
    std::vector<int64_t> expandDims;
    size_t inputIdx = inputShape.size() - 1;
    for (size_t i = targetShape.size(); i > 0; --i) {
        size_t targetIdx = i - 1;
        if (inputIdx < inputShape.size()) {
            if (inputShape[inputIdx] == 1 && targetShape[targetIdx] > 1) {
                expandDims.push_back(static_cast<int64_t>(targetIdx));
            }
            if (inputIdx > 0) --inputIdx;
        } else {
            // Leading dimensions that were added
            expandDims.push_back(static_cast<int64_t>(targetIdx));
        }
    }
    // Reverse to get dimensions in ascending order
    std::reverse(expandDims.begin(), expandDims.end());
    // If no specific expand dims found, use last dim as default
    if (expandDims.empty() && !targetShape.empty()) {
        expandDims.push_back(static_cast<int64_t>(targetShape.size() - 1));
    }
    expandOp.SetAttr(OP_ATTR_PREFIX + "EXPANDDIM", expandDims);

    return output;
}

// Helper: Create unsqueeze operation (add dimension at axis)
LogicalTensorPtr CreateUnsqueezeOp(VJPContext& ctx, LogicalTensorPtr input, int64_t axis) {
    if (input == nullptr || ctx.function == nullptr) {
        return nullptr;
    }

    const Shape& inputShape = input->GetShape();
    Shape newShape;

    // Handle negative axis
    int64_t ndim = static_cast<int64_t>(inputShape.size());
    if (axis < 0) {
        axis = ndim + 1 + axis;
    }

    for (size_t i = 0; i < inputShape.size(); ++i) {
        if (static_cast<int64_t>(i) == axis) {
            newShape.push_back(1);
        }
        newShape.push_back(inputShape[i]);
    }
    if (static_cast<size_t>(axis) >= inputShape.size()) {
        newShape.push_back(1);
    }

    auto output = CreateTensor(ctx, newShape, input->Datatype());
    ctx.function->AddRawOperation(Opcode::OP_RESHAPE, {input}, {output});
    return output;
}

/**
 * VJP for RowSum: d_input = broadcast(d_out, input_shape)
 *
 * RowSum reduces along the last dimension.
 */
VJPResult vjp_rowsum(VJPContext& ctx) {
    auto dOut = ctx.GetOutputGrad("output");
    if (dOut == nullptr) {
        return {{"input", nullptr}};
    }

    auto inputShape = ctx.GetInputShape("input");
    if (inputShape.empty()) {
        return {{"input", nullptr}};
    }

    // RowSum reduces along last axis (axis=-1)
    int64_t dim = static_cast<int64_t>(inputShape.size()) - 1;
    bool keepdim = ctx.GetAttr<bool>("keepdim", true);

    LogicalTensorPtr result = dOut;

    // If keepdim is false, unsqueeze
    if (!keepdim) {
        result = CreateUnsqueezeOp(ctx, result, dim);
    }

    // Expand to input shape
    Shape targetShape(inputShape.begin(), inputShape.end());
    result = CreateExpandOp(ctx, result, targetShape);

    return {{"input", result}};
}

/**
 * VJP for RowMax: gradient only flows to the max element
 *
 * d_input = d_out * (input == max(input))
 *
 * Algorithm:
 * 1. expandedOutput = expand(output, input_shape)
 * 2. expandedGrad = expand(d_out, input_shape)
 * 3. mask = (input == expandedOutput)
 * 4. maskFloat = cast(mask, input.dtype)
 * 5. d_input = expandedGrad * maskFloat
 */
VJPResult vjp_rowmax(VJPContext& ctx) {
    auto dOut = ctx.GetOutputGrad("output");
    if (dOut == nullptr) {
        return {{"input", nullptr}};
    }

    auto inputTensor = ctx.GetSaved("input");
    auto outputTensor = ctx.GetSaved("output");
    if (inputTensor == nullptr || outputTensor == nullptr) {
        return {{"input", nullptr}};
    }

    auto inputShape = ctx.GetInputShape("input");
    if (inputShape.empty()) {
        return {{"input", nullptr}};
    }

    Shape targetShape(inputShape.begin(), inputShape.end());
    DataType dtype = inputTensor->Datatype();

    // Expand output and gradient to input shape
    auto expandedOutput = CreateExpandOp(ctx, outputTensor, targetShape);
    auto expandedGrad = CreateExpandOp(ctx, dOut, targetShape);
    if (expandedOutput == nullptr || expandedGrad == nullptr) {
        return {{"input", nullptr}};
    }

    // Create mask: input == expandedOutput
    auto mask = CreateCmpEqOp(ctx, inputTensor, expandedOutput);
    if (mask == nullptr) {
        return {{"input", nullptr}};
    }

    // Cast mask to float type
    auto maskFloat = CreateCastOp(ctx, mask, dtype);
    if (maskFloat == nullptr) {
        return {{"input", nullptr}};
    }

    // d_input = expandedGrad * maskFloat
    auto dInput = CreateMulOp(ctx, expandedGrad, maskFloat);
    return {{"input", dInput}};
}

/**
 * VJP for RowMin: gradient only flows to the min element
 *
 * Same algorithm as RowMax, just using min output instead.
 */
VJPResult vjp_rowmin(VJPContext& ctx) {
    auto dOut = ctx.GetOutputGrad("output");
    if (dOut == nullptr) {
        return {{"input", nullptr}};
    }

    auto inputTensor = ctx.GetSaved("input");
    auto outputTensor = ctx.GetSaved("output");
    if (inputTensor == nullptr || outputTensor == nullptr) {
        return {{"input", nullptr}};
    }

    auto inputShape = ctx.GetInputShape("input");
    if (inputShape.empty()) {
        return {{"input", nullptr}};
    }

    Shape targetShape(inputShape.begin(), inputShape.end());
    DataType dtype = inputTensor->Datatype();

    // Expand output and gradient to input shape
    auto expandedOutput = CreateExpandOp(ctx, outputTensor, targetShape);
    auto expandedGrad = CreateExpandOp(ctx, dOut, targetShape);
    if (expandedOutput == nullptr || expandedGrad == nullptr) {
        return {{"input", nullptr}};
    }

    // Create mask: input == expandedOutput
    auto mask = CreateCmpEqOp(ctx, inputTensor, expandedOutput);
    if (mask == nullptr) {
        return {{"input", nullptr}};
    }

    // Cast mask to float type
    auto maskFloat = CreateCastOp(ctx, mask, dtype);
    if (maskFloat == nullptr) {
        return {{"input", nullptr}};
    }

    // d_input = expandedGrad * maskFloat
    auto dInput = CreateMulOp(ctx, expandedGrad, maskFloat);
    return {{"input", dInput}};
}

/**
 * VJP for PairSum: sum pairs of elements
 */
VJPResult vjp_pairsum(VJPContext& ctx) {
    auto dOut = ctx.GetOutputGrad("output");
    if (dOut == nullptr) {
        return {{"input", nullptr}};
    }

    auto inputShape = ctx.GetInputShape("input");
    if (inputShape.empty()) {
        return {{"input", nullptr}};
    }

    // PairSum typically reduces dimension by 2
    // d_input needs to be expanded back
    Shape targetShape(inputShape.begin(), inputShape.end());
    auto result = CreateExpandOp(ctx, dOut, targetShape);

    return {{"input", result}};
}

} // anonymous namespace

// Register reduction VJP rules
REG_VJP(OP_ROWSUM, vjp_rowsum);
REG_VJP(OP_ROWSUM_SINGLE, vjp_rowsum);
REG_VJP(OP_ROWSUMLINE, vjp_rowsum);
REG_VJP(OP_ROWSUM_COMBINE_AXIS_SINGLE, vjp_rowsum);
REG_VJP(OP_ROWMAX, vjp_rowmax);
REG_VJP(OP_ROWMAX_SINGLE, vjp_rowmax);
REG_VJP(OP_ROWMAXLINE, vjp_rowmax);
REG_VJP(OP_ROWMAX_COMBINE_AXIS_SINGLE, vjp_rowmax);
REG_VJP(OP_ROWMINLINE, vjp_rowmin);
REG_VJP(OP_PAIRSUM, vjp_pairsum);

} // namespace npu::tile_fwk
