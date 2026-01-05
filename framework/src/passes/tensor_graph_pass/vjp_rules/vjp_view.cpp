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
 * \file vjp_view.cpp
 * \brief VJP rules for view/reshape operations.
 *
 * Corresponds to python/pypto/autograd/rules/view.py
 */

#include "../vjp_registry.h"

#include "interface/inner/element.h"
#include "interface/operation/attribute.h"
#include "interface/tensor/symbolic_scalar.h"

namespace npu::tile_fwk {

namespace {

// Helper: Create tensor
LogicalTensorPtr CreateTensor(VJPContext& ctx, const Shape& shape, DataType dtype) {
    if (ctx.function == nullptr) {
        return nullptr;
    }
    return std::make_shared<LogicalTensor>(*ctx.function, dtype, shape);
}

LogicalTensorPtr CreateZerosTensor(VJPContext& ctx, const Shape& shape, DataType dtype,
                                   const std::vector<SymbolicScalar>& dynValidShape = {}) {
    auto output = CreateTensor(ctx, shape, dtype);
    if (output == nullptr || ctx.function == nullptr) {
        return nullptr;
    }

    if (!dynValidShape.empty()) {
        output->UpdateDynValidShape(dynValidShape);
    }

    Operation& dupOp = ctx.function->AddRawOperation(Opcode::OP_VEC_DUP, {}, {output});
    Element scalarValue(dtype, 0.0);
    dupOp.SetAttr(OpAttributeKey::scalar, scalarValue);
    std::vector<int64_t> shapeVec(shape.begin(), shape.end());
    dupOp.SetAttr(OP_ATTR_PREFIX + "shape", shapeVec);
    return output;
}

LogicalTensorPtr CreateCopyOp(VJPContext& ctx, LogicalTensorPtr input) {
    if (input == nullptr || ctx.function == nullptr) {
        return nullptr;
    }
    auto output = CreateTensor(ctx, input->GetShape(), input->Datatype());
    if (output == nullptr) {
        return nullptr;
    }
    output->UpdateDynValidShape(input->GetDynValidShape());
    ctx.function->AddRawOperation(Opcode::OP_REGISTER_COPY, {input}, {output});
    return output;
}

LogicalTensorPtr CreateViewOp(VJPContext& ctx, LogicalTensorPtr input, const Shape& targetShape,
                              const Offset& offsets, const std::vector<SymbolicScalar>& dynOffsets) {
    if (input == nullptr || ctx.function == nullptr) {
        return nullptr;
    }
    auto output = CreateTensor(ctx, targetShape, input->Datatype());
    if (output == nullptr) {
        return nullptr;
    }
    auto validShape = GetViewValidShape(input->GetDynValidShape(), offsets, dynOffsets, targetShape);
    if (!validShape.empty()) {
        output->UpdateDynValidShape(validShape);
    }
    Operation& viewOp = ctx.function->AddRawOperation(Opcode::OP_VIEW, {input}, {output});
    viewOp.SetOpAttribute(std::make_shared<ViewOpAttribute>(offsets, dynOffsets, validShape));
    return output;
}

LogicalTensorPtr CreateAssembleOp(VJPContext& ctx, LogicalTensorPtr value, LogicalTensorPtr base,
                                  const Offset& offsets, const std::vector<SymbolicScalar>& dynOffsets) {
    if (value == nullptr || base == nullptr || ctx.function == nullptr) {
        return nullptr;
    }
    auto output = CreateTensor(ctx, base->GetShape(), base->Datatype());
    if (output == nullptr) {
        return nullptr;
    }
    output->UpdateDynValidShape(base->GetDynValidShape());
    auto view = CreateViewOp(ctx, base, value->GetShape(), offsets, dynOffsets);
    if (view == nullptr) {
        return nullptr;
    }
    Operation& op = ctx.function->AddRawOperation(Opcode::OP_ASSEMBLE_SSA, {value, view}, {output});
    op.SetAssembleOpAttribute(offsets, dynOffsets);
    op.SetAttribute(OpAttributeKey::inplaceIdx, 1);
    return output;
}

// Helper: Create reshape operation
LogicalTensorPtr CreateReshapeOp(VJPContext& ctx, LogicalTensorPtr input, const Shape& targetShape) {
    if (input == nullptr || ctx.function == nullptr) {
        return nullptr;
    }

    const Shape& inputShape = input->GetShape();

    // Check if shapes already match
    if (inputShape == targetShape) {
        return input;
    }

    auto output = CreateTensor(ctx, targetShape, input->Datatype());
    ctx.function->AddRawOperation(Opcode::OP_RESHAPE, {input}, {output});
    return output;
}

/**
 * VJP for Reshape: d_input = reshape(d_out, input_shape)
 */
VJPResult vjp_reshape(VJPContext& ctx) {
    auto dOut = ctx.GetOutputGrad("output");
    if (dOut == nullptr) {
        return {{"input", nullptr}};
    }

    auto inputShape = ctx.GetInputShape("input");
    if (inputShape.empty()) {
        return {{"input", nullptr}};
    }

    Shape targetShape(inputShape.begin(), inputShape.end());
    auto dInput = CreateReshapeOp(ctx, dOut, targetShape);

    return {{"input", dInput}};
}

/**
 * VJP for View: d_input = view(d_out, input_shape)
 *
 * View is essentially a reshape that may share memory.
 * For gradient computation, we treat it the same as reshape.
 */
VJPResult vjp_view(VJPContext& ctx) {
    auto dOut = ctx.GetOutputGrad("output");
    if (dOut == nullptr) {
        return {{"input", nullptr}};
    }

    auto inputShape = ctx.GetInputShape("input");
    if (inputShape.empty()) {
        return {{"input", nullptr}};
    }

    auto viewAttr = ctx.node ? std::dynamic_pointer_cast<ViewOpAttribute>(ctx.node->GetOpAttribute()) : nullptr;
    if (viewAttr == nullptr) {
        Shape targetShape(inputShape.begin(), inputShape.end());
        auto dInput = CreateReshapeOp(ctx, dOut, targetShape);
        return {{"input", dInput}};
    }

    Shape targetShape(inputShape.begin(), inputShape.end());
    auto inputTensor = ctx.GetSaved("input");
    const auto& dynValidShape = inputTensor ? inputTensor->GetDynValidShape() : std::vector<SymbolicScalar>();
    auto base = CreateZerosTensor(ctx, targetShape, dOut->Datatype(), dynValidShape);
    auto dInput = CreateAssembleOp(ctx, dOut, base, viewAttr->GetFromOffset(), viewAttr->GetFromDynOffset());

    return {{"input", dInput}};
}

/**
 * VJP for Assemble: gradient flows to all inputs
 *
 * Assemble combines multiple tensors - gradient needs to be split.
 */
VJPResult vjp_assemble(VJPContext& ctx) {
    auto dOut = ctx.GetOutputGrad("output");
    if (dOut == nullptr) {
        return {};
    }

    auto assembleAttr = ctx.node ? std::dynamic_pointer_cast<AssembleOpAttribute>(ctx.node->GetOpAttribute()) : nullptr;
    if (assembleAttr == nullptr) {
        return {{"input", nullptr}, {"other", nullptr}};
    }

    auto inputShape = ctx.GetInputShape("input");
    if (inputShape.empty()) {
        return {{"input", nullptr}, {"other", nullptr}};
    }

    Shape targetShape(inputShape.begin(), inputShape.end());
    auto dInput = CreateViewOp(ctx, dOut, targetShape, assembleAttr->GetToOffset(), assembleAttr->GetToDynOffset());

    LogicalTensorPtr dOther = nullptr;
    if (ctx.GetSaved("other") != nullptr) {
        auto dOutOld = CreateCopyOp(ctx, dOut);
        auto inputTensor = ctx.GetSaved("input");
        const auto& dynValidShape = inputTensor ? inputTensor->GetDynValidShape() : std::vector<SymbolicScalar>();
        auto zeros = CreateZerosTensor(ctx, targetShape, dOut->Datatype(), dynValidShape);
        dOther = CreateAssembleOp(ctx, zeros, dOutOld, assembleAttr->GetToOffset(), assembleAttr->GetToDynOffset());
    }

    return {{"input", dInput}, {"other", dOther}};
}

/**
 * VJP for Duplicate: d_input = sum of all uses
 *
 * When a tensor is duplicated, gradients from all uses are summed.
 */
VJPResult vjp_duplicate(VJPContext& ctx) {
    auto dOut = ctx.GetOutputGrad("output");
    if (dOut == nullptr) {
        return {{"input", nullptr}};
    }

    // Gradient flows directly through duplicate
    return {{"input", dOut}};
}

/**
 * VJP for Transpose: d_input = transpose(d_out, inverse_perm)
 *
 * For simple 2D transpose (swap last two axes), the inverse is another transpose.
 * For general permutation, we need the inverse permutation.
 */
VJPResult vjp_transpose(VJPContext& ctx) {
    auto dOut = ctx.GetOutputGrad("output");
    if (dOut == nullptr) {
        return {{"input", nullptr}};
    }

    auto inputShape = ctx.GetInputShape("input");
    if (inputShape.empty()) {
        return {{"input", nullptr}};
    }

    std::vector<int64_t> axes = ctx.GetAttr<std::vector<int64_t>>("shape", {});
    if (axes.size() != 2 && inputShape.size() >= 2) {
        axes = {
            static_cast<int64_t>(inputShape.size() - 2),
            static_cast<int64_t>(inputShape.size() - 1),
        };
    }

    if (axes.size() == 2 && ctx.function != nullptr) {
        Shape targetShape(inputShape.begin(), inputShape.end());
        auto dInput = CreateTensor(ctx, targetShape, dOut->Datatype());
        if (dInput == nullptr) {
            return {{"input", nullptr}};
        }
        Opcode opcode = ctx.node ? ctx.node->GetOpcode() : Opcode::OP_TRANSPOSE_MOVEIN;
        Operation& transposeOp = ctx.function->AddRawOperation(opcode, {dOut}, {dInput});
        transposeOp.SetAttr(OP_ATTR_PREFIX + "shape", axes);
        return {{"input", dInput}};
    }

    // Fallback: reshape to input shape if element count matches
    Shape targetShape(inputShape.begin(), inputShape.end());
    auto dInput = CreateReshapeOp(ctx, dOut, targetShape);

    return {{"input", dInput}};
}

/**
 * VJP for Expand: d_input = sum(d_out) along expanded dimensions
 *
 * Expand broadcasts a tensor to a larger shape. The gradient needs to be
 * summed along the dimensions that were expanded (broadcast).
 */
VJPResult vjp_expand(VJPContext& ctx) {
    auto dOut = ctx.GetOutputGrad("output");
    if (dOut == nullptr) {
        return {{"input", nullptr}};
    }

    auto inputShape = ctx.GetInputShape("input");
    if (inputShape.empty()) {
        return {{"input", nullptr}};
    }

    // Use Unbroadcast to sum along expanded dimensions
    auto dInput = ctx.Unbroadcast(dOut, inputShape);
    return {{"input", dInput}};
}

/**
 * VJP for ReshapeCopyIn: gradient flows through as reshape
 */
VJPResult vjp_reshape_copy_in(VJPContext& ctx) {
    return vjp_reshape(ctx);
}

/**
 * VJP for ReshapeCopyOut: gradient flows through as reshape
 */
VJPResult vjp_reshape_copy_out(VJPContext& ctx) {
    return vjp_reshape(ctx);
}

} // anonymous namespace

// Register view VJP rules
REG_VJP(OP_RESHAPE, vjp_reshape);
REG_VJP(OP_RESHAPE_COPY_IN, vjp_reshape_copy_in);
REG_VJP(OP_RESHAPE_COPY_OUT, vjp_reshape_copy_out);
REG_VJP(OP_VIEW, vjp_view);
REG_VJP(OP_VIEW_TYPE, vjp_view);
REG_VJP(OP_ASSEMBLE, vjp_assemble);
REG_VJP(OP_ASSEMBLE_SSA, vjp_assemble);
REG_VJP(OP_DUPLICATE, vjp_duplicate);
REG_VJP(OP_EXPAND, vjp_expand);
REG_VJP(OP_TRANSPOSE_MOVEIN, vjp_transpose);
REG_VJP(OP_TRANSPOSE_MOVEOUT, vjp_transpose);

} // namespace npu::tile_fwk
