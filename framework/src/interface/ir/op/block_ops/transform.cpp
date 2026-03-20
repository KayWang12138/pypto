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

TypePtr DeduceTileViewType(
    const std::vector<ExprPtr> &args, const std::vector<std::pair<std::string, std::any>> & /*kwargs*/, const Span &span) {
    // tile.view requires exactly 3 arguments: input tile, shape tuple, and offset tuple
    CHECK(args.size() == 3) << "tile.view requires exactly 3 arguments (input, shape, offset), but got " << args.size() << " at " << span.ToString();

    // First argument must be TileType
    auto tileType = As<TileType>(args[0]->GetType());
    CHECK(tileType) << "tile.view requires first argument to be a TileType, but got " << args[0]->GetType()->TypeName() << " at " << span.ToString();

    // Second argument must be TupleType (shape)
    auto shapeTupleType = As<TupleType>(args[1]->GetType());
    CHECK(shapeTupleType) << "tile.view requires shape to be TupleType, but got " << args[1]->GetType()->TypeName() << " at " << span.ToString();

    // Validate all shape elements are ScalarType(INT64 or UINT64)
    for (size_t i = 0; i < shapeTupleType->types_.size(); ++i) {
        auto scalarType = As<ScalarType>(shapeTupleType->types_[i]);
        CHECK(scalarType) << "tile.view shape tuple element " << i << " must be ScalarType, but got "
                          << shapeTupleType->types_[i]->TypeName() << " at " << span.ToString();
        CHECK(scalarType->dtype_ == DataType::INT64 || scalarType->dtype_ == DataType::UINT64)
            << "tile.view shape tuple element " << i << " must have dtype INT64 or UINT64, but got "
            << scalarType->dtype_.ToString() << " at " << span.ToString();
    }

    // Third argument must be TupleType (offset)
    auto offsetTupleType = As<TupleType>(args[2]->GetType());
    CHECK(offsetTupleType) << "tile.view requires offset to be TupleType, but got " << args[2]->GetType()->TypeName() << " at " << span.ToString();

    // Validate all offset elements are ScalarType(INT64 or UINT64)
    for (size_t i = 0; i < offsetTupleType->types_.size(); ++i) {
        auto scalarType = As<ScalarType>(offsetTupleType->types_[i]);
        CHECK(scalarType) << "tile.view offset tuple element " << i << " must be ScalarType, but got "
                          << offsetTupleType->types_[i]->TypeName() << " at " << span.ToString();
        CHECK(scalarType->dtype_ == DataType::INT64 || scalarType->dtype_ == DataType::UINT64)
            << "tile.view offset tuple element " << i << " must have dtype INT64 or UINT64, but got "
            << scalarType->dtype_.ToString() << " at " << span.ToString();
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
    return std::make_shared<TileType>(newShape, tileType->dtype_);
}

TypePtr DeduceTileReshapeType(
    const std::vector<ExprPtr> &args, const std::vector<std::pair<std::string, std::any>> & /*kwargs*/, const Span &span) {
    // tile.reshape requires exactly 2 arguments: input tile and shape tuple
    CHECK(args.size() == 2) << "tile.reshape requires exactly 2 arguments (input, shape), but got " << args.size() << " at " << span.ToString();

    // First argument must be TileType
    auto tileType = As<TileType>(args[0]->GetType());
    CHECK(tileType) << "tile.reshape requires first argument to be a TileType, but got "
                    << args[0]->GetType()->TypeName() << " at " << span.ToString();

    // Second argument must be TupleType (shape)
    auto shapeTupleType = As<TupleType>(args[1]->GetType());
    CHECK(shapeTupleType) << "tile.reshape requires shape to be TupleType, but got " << args[1]->GetType()->TypeName() << " at " << span.ToString();

    // Validate all shape elements are ScalarType(INT64 or UINT64)
    for (size_t i = 0; i < shapeTupleType->types_.size(); ++i) {
        auto scalarType = As<ScalarType>(shapeTupleType->types_[i]);
        CHECK(scalarType) << "tile.reshape shape tuple element " << i << " must be ScalarType, but got "
                          << shapeTupleType->types_[i]->TypeName() << " at " << span.ToString();
        CHECK(scalarType->dtype_ == DataType::INT64 || scalarType->dtype_ == DataType::UINT64)
            << "tile.reshape shape tuple element " << i << " must have dtype INT64 or UINT64, but got "
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
    int64_t oldProduct = ComputeShapeProduct(tileType->shape_);
    int64_t newProduct = ComputeShapeProduct(newShape);

    if (oldProduct > 0 && newProduct > 0) {
        CHECK(oldProduct == newProduct) << "tile.reshape: cannot reshape tile of size " << oldProduct
                                        << " into shape with size " << newProduct << " at " << span.ToString();
    }

    // Return new TileType with reshaped dimensions and same dtype
    return std::make_shared<TileType>(newShape, tileType->dtype_);
}

TypePtr DeduceTileTransposeType(
    const std::vector<ExprPtr> &args, const std::vector<std::pair<std::string, std::any>> & /*kwargs*/, const Span &span) {
    // tile.transpose requires exactly 3 arguments: input tile, axis1, axis2
    CHECK(args.size() == 3) << "tile.transpose requires exactly 3 arguments (input, axis1, axis2), but got "
                            << args.size() << " at " << span.ToString();

    // First argument must be TileType
    auto tileType = As<TileType>(args[0]->GetType());
    CHECK(tileType) << "tile.transpose requires first argument to be a TileType, but got "
                    << args[0]->GetType()->TypeName() << " at " << span.ToString();

    const auto &inputShape = tileType->shape_;
    size_t ndim = inputShape.size();

    CHECK(ndim >= 2) << "tile.transpose requires at least 2 dimensions, but got " << ndim << " at " << span.ToString();

    // Second argument is axis1 (ConstInt)
    auto axis1Const = As<ConstInt>(args[1]);
    CHECK(axis1Const) << "tile.transpose requires second argument (axis1) to be a ConstInt" << " at " << span.ToString();

    // Third argument is axis2 (ConstInt)
    auto axis2Const = As<ConstInt>(args[2]);
    CHECK(axis2Const) << "tile.transpose requires third argument (axis2) to be a ConstInt" << " at " << span.ToString();

    // Normalize axes (handle negative indexing)
    int axis1 = NormalizeAxis(static_cast<int>(axis1Const->value_), ndim, span);
    int axis2 = NormalizeAxis(static_cast<int>(axis2Const->value_), ndim, span);

    CHECK(axis1 != axis2) << "tile.transpose: axis1 and axis2 must be different, but got axis1=" << axis1
                          << ", axis2=" << axis2 << " at " << span.ToString();

    // Create new shape by swapping the specified dimensions
    std::vector<ExprPtr> newShape = inputShape;
    std::swap(newShape[axis1], newShape[axis2]);

    // Return new TileType with transposed shape and same dtype
    return std::make_shared<TileType>(newShape, tileType->dtype_);
}

// ============================================================================
// Registration Function for Tile Transform Operations
// ============================================================================

REGISTER_OP("block.view")
    .SetOpCategory("BlockOp")
    .SetDescription("Create a view/slice of a tile with new shape and offset")
    .SetPipe(PipeType::V)
    .AddArgument("input", "Input tile (TileType)")
    .AddArgument("shape", "New shape dimensions (TupleType of ScalarType(UINT64))")
    .AddArgument("offset", "Offset dimensions (TupleType of ScalarType(UINT64))")
    .SetDeduceType([](const std::vector<ExprPtr> &args, const std::vector<std::pair<std::string, std::any>> &kwargs,
        const Span &span) {
        return DeduceTileViewType(args, kwargs, span);
    });

REGISTER_OP("block.reshape")
    .SetOpCategory("BlockOp")
    .SetDescription("Reshape tile to new shape")
    .SetPipe(PipeType::V)
    .AddArgument("input", "Input tile (TileType)")
    .AddArgument("shape", "New shape dimensions (TupleType of ScalarType(UINT64))")
    .SetDeduceType([](const std::vector<ExprPtr> &args, const std::vector<std::pair<std::string, std::any>> &kwargs,
        const Span &span) {
        return DeduceTileReshapeType(args, kwargs, span);
    });

REGISTER_OP("block.transpose")
    .SetOpCategory("BlockOp")
    .SetDescription("Transpose tile by swapping two axes")
    .SetPipe(PipeType::V)
    .AddArgument("input", "Input tile (TileType)")
    .AddArgument("axis1", "First axis to swap (ConstInt)")
    .AddArgument("axis2", "Second axis to swap (ConstInt)")
    .SetDeduceType([](const std::vector<ExprPtr> &args, const std::vector<std::pair<std::string, std::any>> &kwargs,
        const Span &span) {
        return DeduceTileTransposeType(args, kwargs, span);
    });

} // namespace ir
} // namespace pypto
