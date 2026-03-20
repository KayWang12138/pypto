/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/dtype.h"
#include "core/logging.h"
#include "ir/core.h"
#include "ir/expr.h"
#include "ir/kind_traits.h"
#include "ir/op_registry.h"
#include "ir/op_utils.h"
#include "ir/scalar_expr.h"
#include "ir/type.h"

namespace pypto {
namespace ir {

// ============================================================================
// Type Inference Functions
// ============================================================================

TypePtr DeduceTensorReshapeType(
    const std::vector<ExprPtr> &args, const std::vector<std::pair<std::string, std::any>> & /*kwargs*/,
    const Span &span) {
    // tensor.reshape requires exactly 2 arguments: input tensor and shape tuple
    INTERNAL_CHECK(args.size() == 2) << "tensor.reshape requires exactly 2 arguments (input, shape), but got "
                                     << args.size() << " at " << span.ToString();

    // First argument must be TensorType
    auto tensorType = As<TensorType>(args[0]->GetType());
    INTERNAL_CHECK(tensorType) << "tensor.reshape requires first argument to be a TensorType, but got "
                               << args[0]->GetType()->TypeName() << " at " << span.ToString();

    // Second argument must be TupleType (shape)
    auto shapeTupleType = As<TupleType>(args[1]->GetType());
    INTERNAL_CHECK(shapeTupleType) << "tensor.reshape requires shape to be TupleType, but got "
                                   << args[1]->GetType()->TypeName() << " at " << span.ToString();

    // Validate all shape elements are ScalarType(INT64 or UINT64)
    for (size_t i = 0; i < shapeTupleType->types_.size(); ++i) {
        auto scalarType = As<ScalarType>(shapeTupleType->types_[i]);
        INTERNAL_CHECK(scalarType) << "tensor.reshape shape tuple element " << i << " must be ScalarType, but got "
                                   << shapeTupleType->types_[i]->TypeName() << " at " << span.ToString();
        INTERNAL_CHECK(scalarType->dtype_ == DataType::INT64 || scalarType->dtype_ == DataType::UINT64)
            << "tensor.reshape shape tuple element " << i << " must have dtype INT64 or UINT64, but got "
            << scalarType->dtype_.ToString() << " at " << span.ToString();
    }

    // Extract new shape dimensions
    // If args[1] is MakeTuple, extract elements directly to preserve constants
    // Otherwise use TupleGetItemExpr for runtime tuples
    std::vector<ExprPtr> newShape;
    newShape.reserve(shapeTupleType->types_.size());

    if (auto make_tuple = As<MakeTuple>(args[1])) {
        // MakeTuple: extract elements directly to preserve ConstInt
        newShape = make_tuple->elements_;
    } else {
        // Runtime tuple: use TupleGetItemExpr
        for (size_t i = 0; i < shapeTupleType->types_.size(); ++i) {
            newShape.emplace_back(std::make_shared<TupleGetItemExpr>(args[1], static_cast<int>(i), args[1]->span_));
        }
    }

    // For static shapes, verify that the total number of elements matches
    int64_t oldProduct = ComputeShapeProduct(tensorType->shape_);
    int64_t newProduct = ComputeShapeProduct(newShape);

    if (oldProduct > 0 && newProduct > 0) {
        INTERNAL_CHECK(oldProduct == newProduct)
            << "tensor.reshape: cannot reshape tensor of size " << oldProduct << " into shape with size " << newProduct
            << " at " << span.ToString();
    }

    // Return new TensorType with reshaped dimensions and same dtype
    return std::make_shared<TensorType>(newShape, tensorType->dtype_);
}

TypePtr DeduceTensorTransposeType(
    const std::vector<ExprPtr> &args, const std::vector<std::pair<std::string, std::any>> & /*kwargs*/,
    const Span &span) {
    // tensor.transpose requires exactly 3 arguments: input tensor, axis1, axis2
    INTERNAL_CHECK(args.size() == 3) << "tensor.transpose requires exactly 3 arguments (input, axis1, axis2), but got "
                                     << args.size() << " at " << span.ToString();

    // First argument must be TensorType
    auto tensorType = As<TensorType>(args[0]->GetType());
    INTERNAL_CHECK(tensorType) << "tensor.transpose requires first argument to be a TensorType, but got "
                               << args[0]->GetType()->TypeName() << " at " << span.ToString();

    const auto &inputShape = tensorType->shape_;
    size_t ndim = inputShape.size();

    INTERNAL_CHECK(ndim >= 2) << "tensor.transpose requires at least 2 dimensions, but got " << ndim
                              << " at " << span.ToString();

    // Second argument is axis1 (ConstInt)
    auto axis1Const = As<ConstInt>(args[1]);
    INTERNAL_CHECK(axis1Const) << "tensor.transpose requires second argument (axis1) to be a ConstInt"
                               << " at " << span.ToString();

    // Third argument is axis2 (ConstInt)
    auto axis2Const = As<ConstInt>(args[2]);
    INTERNAL_CHECK(axis2Const) << "tensor.transpose requires third argument (axis2) to be a ConstInt"
                               << " at " << span.ToString();

    // Normalize axes (handle negative indexing)
    int axis1 = NormalizeAxis(static_cast<int>(axis1Const->value_), ndim, span);
    int axis2 = NormalizeAxis(static_cast<int>(axis2Const->value_), ndim, span);

    INTERNAL_CHECK(axis1 != axis2) << "tensor.transpose: axis1 and axis2 must be different, but got axis1=" << axis1
                                   << ", axis2=" << axis2 << " at " << span.ToString();

    // Create new shape by swapping the specified dimensions
    std::vector<ExprPtr> newShape = inputShape;
    std::swap(newShape[axis1], newShape[axis2]);

    // Return new TensorType with transposed shape and same dtype
    return std::make_shared<TensorType>(newShape, tensorType->dtype_);
}

// ============================================================================
// Registration Function for Tensor Transform Operations
// ============================================================================

REGISTER_OP("tensor.reshape")
    .SetOpCategory("TensorOp")
    .SetDescription("Reshape tensor to new shape")
    .AddArgument("input", "Input tensor (TensorType)")
    .AddArgument("shape", "New shape dimensions (TupleType of ScalarType(UINT64))")
    .SetDeduceType([](const std::vector<ExprPtr> &args, const std::vector<std::pair<std::string, std::any>> &kwargs,
        const Span &span) {
        return DeduceTensorReshapeType(args, kwargs, span);
    });

REGISTER_OP("tensor.transpose")
    .SetOpCategory("TensorOp")
    .SetDescription("Transpose tensor by swapping two axes")
    .AddArgument("input", "Input tensor (TensorType)")
    .AddArgument("axis1", "First axis to swap (ConstInt)")
    .AddArgument("axis2", "Second axis to swap (ConstInt)")
    .SetDeduceType([](const std::vector<ExprPtr> &args, const std::vector<std::pair<std::string, std::any>> &kwargs,
        const Span &span) {
        return DeduceTensorTransposeType(args, kwargs, span);
    });

} // namespace ir
} // namespace pypto
