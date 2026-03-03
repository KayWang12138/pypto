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

#include <cstdint>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "core/error.h"
#include "core/logging.h"
#include "ir/core.h"
#include "ir/kind_traits.h"
#include "ir/op_registry.h"
#include "ir/op_utils.h"
#include "ir/scalar_expr.h"
#include "ir/type.h"

namespace pypto {
namespace ir {

TypePtr DeduceBlockReductionType(const std::vector<ExprPtr> &args,
    const std::vector<std::pair<std::string, std::any>> &kwargs, const std::string &opName) {
    // block.sum and block.max require 1 argument (tile) and 2 attributes (axis, keepdim)
    CHECK(args.size() == 1) << "The operator " << opName << " requires 1 argument, but got " << args.size();

    // First argument must be TileType
    auto tileType = As<TileType>(args[0]->GetType());
    CHECK(tileType) << "The operator " << opName << " requires first argument to be a TileType, but got "
                    << args[0]->GetType()->TypeName();

    // Get the input shape
    const auto &inputShape = tileType->shape_;
    int64_t inputNdim = static_cast<int64_t>(inputShape.size());

    // Determine which axes to reduce
    std::set<int64_t> reduceAxes;

    // Extract axis from kwargs (required)
    int axisValue = GetKwarg<int>(kwargs, "axis");
    if (axisValue < 0) {
        // Negative axis: convert to positive
        axisValue = static_cast<int>(inputNdim) + axisValue;
    }
    CHECK(axisValue >= 0 && static_cast<int64_t>(axisValue) < inputNdim)
        << "The operator " << opName << " axis " << axisValue << " is out of range for shape with " << inputNdim
        << " dimensions";
    reduceAxes.insert(static_cast<int64_t>(axisValue));

    // Extract keepdim from kwargs (optional, default to false)
    bool keepdim = GetKwarg<bool>(kwargs, "keepdim", false);

    // If all axes are reduced and keepdim is false, return ScalarType
    if (static_cast<int64_t>(reduceAxes.size()) == inputNdim && !keepdim) {
        return std::make_shared<ScalarType>(tileType->dtype_);
    }

    // Build output shape
    std::vector<ExprPtr> outputShape;
    if (keepdim) {
        // When keepdim is true, keep all dimensions but set reduced axes to 1
        for (int64_t i = 0; i < inputNdim; ++i) {
            if (reduceAxes.find(i) != reduceAxes.end()) {
                // Reduced axis: set to 1
                outputShape.push_back(std::make_shared<ConstInt>(1, DataType::INT32, Span::Unknown()));
            } else {
                // Keep this dimension
                outputShape.push_back(inputShape[i]);
            }
        }
    } else {
        // When keepdim is false, remove reduced axes
        for (int64_t i = 0; i < inputNdim; ++i) {
            if (reduceAxes.find(i) == reduceAxes.end()) {
                // Keep this dimension
                outputShape.push_back(inputShape[i]);
            }
        }
    }

    // If output shape is empty, return ScalarType
    if (outputShape.empty()) {
        return std::make_shared<ScalarType>(tileType->dtype_);
    }

    // Return TileType with reduced shape
    return std::make_shared<TileType>(outputShape, tileType->dtype_);
}

TypePtr DeduceBlockRowReductionType(const std::vector<ExprPtr> &args,
    const std::vector<std::pair<std::string, std::any>> & /*kwargs*/, const std::string &opName) {
    // block.row_max and block.row_sum require 1 argument (tile)
    CHECK(args.size() == 1) << "The operator " << opName << " requires 1 argument, but got " << args.size();

    // First argument must be TileType
    auto tileType = As<TileType>(args[0]->GetType());
    CHECK(tileType) << "The operator " << opName << " requires first argument to be a TileType, but got "
                    << args[0]->GetType()->TypeName();

    // Get the input shape
    const auto &inputShape = tileType->shape_;
    int64_t inputNdim = static_cast<int64_t>(inputShape.size());

    // Row reduction requires at least 2D tile (operates on the last dimension)
    CHECK(inputNdim >= 2) << "The operator " << opName << " requires at least a 2D tile, but got " << inputNdim
                          << " dimensions";

    // Output shape is [...batch_dims, rows, 1] - reduce along the last axis (columns) with keepdim=True
    std::vector<ExprPtr> outputShape(inputShape.begin(), inputShape.end() - 1);             // Keep all but last dim
    outputShape.push_back(std::make_shared<ConstInt>(1, DataType::INT32, Span::Unknown())); // Reduced last dim

    return std::make_shared<TileType>(outputShape, tileType->dtype_);
}

// ============================================================================
// Registration Function for Block Reduction Operations
// ============================================================================

REGISTER_OP("block.sum")
    .SetOpCategory("BlockOp")
    .SetDescription("Sum reduction of a tile along specified axis")
    .AddArgument("tile", "Input tile (TileType)")
    .set_attr<int>("axis")
    .set_attr<bool>("keepdim")
    .SetDeduceType([](const std::vector<ExprPtr> &args, const std::vector<std::pair<std::string, std::any>> &kwargs) {
        return DeduceBlockReductionType(args, kwargs, "block.sum");
    });

REGISTER_OP("block.max")
    .SetOpCategory("BlockOp")
    .SetDescription("Max reduction of a tile along specified axis")
    .AddArgument("tile", "Input tile (TileType)")
    .set_attr<int>("axis")
    .set_attr<bool>("keepdim")
    .SetDeduceType([](const std::vector<ExprPtr> &args, const std::vector<std::pair<std::string, std::any>> &kwargs) {
        return DeduceBlockReductionType(args, kwargs, "block.max");
    });

REGISTER_OP("block.min")
    .SetOpCategory("BlockOp")
    .SetDescription("Min reduction of a tile along specified axis")
    .SetPipe(PipeType::V)
    .AddArgument("tile", "Input tile (TileType)")
    .set_attr<int>("axis")
    .set_attr<bool>("keepdim")
    .SetDeduceType([](const std::vector<ExprPtr> &args, const std::vector<std::pair<std::string, std::any>> &kwargs) {
        return DeduceBlockReductionType(args, kwargs, "block.min");
    });

REGISTER_OP("block.row_max")
    .SetOpCategory("BlockOp")
    .SetDescription("Row-wise max reduction of a 2D tile (output shape: [rows, 1])")
    .SetPipe(PipeType::V)
    .AddArgument("tile", "Input tile (TileType, 2D)")
    .SetDeduceType([](const std::vector<ExprPtr> &args, const std::vector<std::pair<std::string, std::any>> &kwargs) {
        return DeduceBlockRowReductionType(args, kwargs, "block.row_max");
    });

REGISTER_OP("block.row_sum")
    .SetOpCategory("BlockOp")
    .SetDescription("Row-wise sum reduction of a 2D tile (output shape: [rows, 1])")
    .SetPipe(PipeType::V)
    .AddArgument("tile", "Input tile (TileType, 2D)")
    .SetDeduceType([](const std::vector<ExprPtr> &args, const std::vector<std::pair<std::string, std::any>> &kwargs) {
        return DeduceBlockRowReductionType(args, kwargs, "block.row_sum");
    });

REGISTER_OP("block.row_min")
    .SetOpCategory("BlockOp")
    .SetDescription("Row-wise min reduction of a 2D tile (output shape: [rows, 1])")
    .SetPipe(PipeType::V)
    .AddArgument("tile", "Input tile (TileType, 2D)")
    .SetDeduceType([](const std::vector<ExprPtr> &args, const std::vector<std::pair<std::string, std::any>> &kwargs) {
        return DeduceBlockRowReductionType(args, kwargs, "block.row_min");
    });

} // namespace ir
} // namespace pypto
