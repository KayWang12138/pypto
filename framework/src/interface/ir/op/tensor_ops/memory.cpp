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
#include <vector>

#include "core/any_cast.h"
#include "core/dtype.h"
#include "core/logging.h"
#include "ir/expr.h"
#include "ir/kind_traits.h"
#include "ir/op_registry.h"
#include "ir/scalar_expr.h"
#include "ir/type.h"

namespace pypto {
namespace ir {

TypePtr DeduceTensorCreateType(
    const std::vector<ExprPtr> &args, const std::vector<std::pair<std::string, std::any>> &kwargs) {
    // tensor.create: shape is a single TupleType argument
    // dtype comes from kwargs
    INTERNAL_CHECK(args.size() == 1) << "tensor.create requires exactly 1 argument (shape tuple), but got "
                                     << args.size();

    // Extract dtype from kwargs
    bool foundDtype = false;
    DataType dtype;
    for (const auto &[key, value] : kwargs) {
        if (key == "dtype") {
            dtype = AnyCast<DataType>(value, "kwarg key: dtype");
            foundDtype = true;
            break;
        }
    }
    INTERNAL_CHECK(foundDtype) << "tensor.create requires 'dtype' kwarg";

    // First argument must be TupleType (shape)
    auto shapeTupleType = As<TupleType>(args[0]->GetType());
    INTERNAL_CHECK(shapeTupleType) << "tensor.create requires shape to be TupleType, but got "
                                   << args[0]->GetType()->TypeName();

    // Validate all shape elements are ScalarType(INT64 or UINT64)
    for (size_t i = 0; i < shapeTupleType->types_.size(); ++i) {
        auto scalarType = As<ScalarType>(shapeTupleType->types_[i]);
        INTERNAL_CHECK(scalarType) << "tensor.create shape tuple element " << i << " must be ScalarType, but got "
                                   << shapeTupleType->types_[i]->TypeName();
        INTERNAL_CHECK(scalarType->dtype_ == DataType::INT64 || scalarType->dtype_ == DataType::UINT64)
            << "tensor.create shape tuple element " << i << " must have dtype INT64 or UINT64, but got "
            << scalarType->dtype_.ToString();
    }

    // Extract shape dimensions
    // If args[0] is MakeTuple, extract elements directly to preserve constants
    // Otherwise use TupleGetItemExpr for runtime tuples
    std::vector<ExprPtr> shape;
    shape.reserve(shapeTupleType->types_.size());

    if (auto make_tuple = As<MakeTuple>(args[0])) {
        // MakeTuple: extract elements directly to preserve ConstInt
        shape = make_tuple->elements_;
    } else {
        // Runtime tuple: use TupleGetItemExpr
        for (size_t i = 0; i < shapeTupleType->types_.size(); ++i) {
            shape.emplace_back(std::make_shared<TupleGetItemExpr>(args[0], static_cast<int>(i), args[0]->span_));
        }
    }

    return std::make_shared<TensorType>(shape, dtype);
}

TypePtr DeduceTensorViewType(
    const std::vector<ExprPtr> &args, const std::vector<std::pair<std::string, std::any>> & /*kwargs*/) {
    // tensor.view requires exactly 3 arguments: input tensor, shape tuple, and offset tuple
    INTERNAL_CHECK(args.size() == 3) << "tensor.view requires exactly 3 arguments (input, shape, offset), but got "
                                     << args.size();

    // First argument must be TensorType
    auto tensorType = As<TensorType>(args[0]->GetType());
    INTERNAL_CHECK(tensorType) << "tensor.view requires first argument to be a TensorType, but got "
                               << args[0]->GetType()->TypeName();

    // Second argument must be TupleType (shape)
    auto shapeTupleType = As<TupleType>(args[1]->GetType());
    INTERNAL_CHECK(shapeTupleType) << "tensor.view requires shape to be TupleType, but got "
                                   << args[1]->GetType()->TypeName();

    // Validate all shape elements are ScalarType(INT64 or UINT64)
    for (size_t i = 0; i < shapeTupleType->types_.size(); ++i) {
        auto scalarType = As<ScalarType>(shapeTupleType->types_[i]);
        INTERNAL_CHECK(scalarType) << "tensor.view shape tuple element " << i << " must be ScalarType, but got "
                                   << shapeTupleType->types_[i]->TypeName();
        INTERNAL_CHECK(scalarType->dtype_ == DataType::INT64 || scalarType->dtype_ == DataType::UINT64)
            << "tensor.view shape tuple element " << i << " must have dtype INT64 or UINT64, but got "
            << scalarType->dtype_.ToString();
    }

    // Third argument must be TupleType (offset)
    auto offsetTupleType = As<TupleType>(args[2]->GetType());
    INTERNAL_CHECK(offsetTupleType) << "tensor.view requires offset to be TupleType, but got "
                                    << args[2]->GetType()->TypeName();

    // Validate all offset elements are ScalarType(INT64 or UINT64)
    for (size_t i = 0; i < offsetTupleType->types_.size(); ++i) {
        auto scalarType = As<ScalarType>(offsetTupleType->types_[i]);
        INTERNAL_CHECK(scalarType) << "tensor.view offset tuple element " << i << " must be ScalarType, but got "
                                   << offsetTupleType->types_[i]->TypeName();
        INTERNAL_CHECK(scalarType->dtype_ == DataType::INT64 || scalarType->dtype_ == DataType::UINT64)
            << "tensor.view offset tuple element " << i << " must have dtype INT64 or UINT64, but got "
            << scalarType->dtype_.ToString();
    }

    // Extract shape dimensions
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

    // View preserves dtype but has new shape (which can have different rank than input)
    return std::make_shared<TensorType>(newShape, tensorType->dtype_);
}

TypePtr DeduceTensorAssembleType(
    const std::vector<ExprPtr> &args, const std::vector<std::pair<std::string, std::any>> & /*kwargs*/) {
    // tensor.assemble requires exactly 3 arguments: target, source, and offset tuple
    INTERNAL_CHECK(args.size() == 3)
        << "tensor.assemble requires exactly 3 arguments (target, source, offset), but got " << args.size();

    // First argument (target) must be TensorType
    auto targetType = As<TensorType>(args[0]->GetType());
    INTERNAL_CHECK(targetType) << "tensor.assemble requires first argument to be a TensorType, but got "
                               << args[0]->GetType()->TypeName();

    // Second argument (source) must be TensorType
    auto sourceType = As<TensorType>(args[1]->GetType());
    INTERNAL_CHECK(sourceType) << "tensor.assemble requires second argument to be a TensorType, but got "
                               << args[1]->GetType()->TypeName();

    // Third argument must be TupleType (offset)
    auto offsetTupleType = As<TupleType>(args[2]->GetType());
    INTERNAL_CHECK(offsetTupleType) << "tensor.assemble requires offset to be TupleType, but got "
                                    << args[2]->GetType()->TypeName();

    // Validate all offset elements are ScalarType(INT64 or UINT64)
    for (size_t i = 0; i < offsetTupleType->types_.size(); ++i) {
        auto scalarType = As<ScalarType>(offsetTupleType->types_[i]);
        INTERNAL_CHECK(scalarType) << "tensor.assemble offset tuple element " << i << " must be ScalarType, but got "
                                   << offsetTupleType->types_[i]->TypeName();
        INTERNAL_CHECK(scalarType->dtype_ == DataType::INT64 || scalarType->dtype_ == DataType::UINT64)
            << "tensor.assemble offset tuple element " << i << " must have dtype INT64 or UINT64, but got "
            << scalarType->dtype_.ToString();
    }

    // Assemble returns a new TensorType with the same shape and dtype as target
    // We need to create a new type object to avoid sharing type instances
    return std::make_shared<TensorType>(targetType->shape_, targetType->dtype_);
}

// ============================================================================
// Registration Function for Tensor Memory Operations
// ============================================================================

REGISTER_OP("tensor.create")
    .SetOpCategory("TensorOp")
    .SetDescription("Create a new tensor with specified shape and dtype")
    .AddArgument("shape", "Shape dimensions (TupleType of ScalarType(UINT64))")
    .set_attr<DataType>("dtype")
    .SetDeduceType([](const std::vector<ExprPtr> &args, const std::vector<std::pair<std::string, std::any>> &kwargs) {
        return DeduceTensorCreateType(args, kwargs);
    });

REGISTER_OP("tensor.view")
    .SetOpCategory("TensorOp")
    .SetDescription("Create a view/slice of a tensor with new shape and offset")
    .AddArgument("input", "Input tensor (TensorType)")
    .AddArgument("shape", "New shape dimensions (TupleType of ScalarType(UINT64))")
    .AddArgument("offset", "Offset dimensions (TupleType of ScalarType(UINT64))")
    .SetDeduceType([](const std::vector<ExprPtr> &args, const std::vector<std::pair<std::string, std::any>> &kwargs) {
        return DeduceTensorViewType(args, kwargs);
    });

REGISTER_OP("tensor.assemble")
    .SetOpCategory("TensorOp")
    .SetDescription("Write/update tensor values at specified offset")
    .AddArgument("target", "Target tensor (TensorType)")
    .AddArgument("source", "Source tensor to write (TensorType)")
    .AddArgument("offset", "Offset dimensions (TupleType of ScalarType(UINT64))")
    .SetDeduceType([](const std::vector<ExprPtr> &args, const std::vector<std::pair<std::string, std::any>> &kwargs) {
        return DeduceTensorAssembleType(args, kwargs);
    });

} // namespace ir
} // namespace pypto
