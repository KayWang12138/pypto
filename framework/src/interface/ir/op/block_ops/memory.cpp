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

#include <any>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "core/common.h"
#include "core/dtype.h"
#include "core/error.h"
#include "core/logging.h"
#include "ir/core.h"
#include "ir/expr.h"
#include "ir/kind_traits.h"
#include "ir/op_registry.h"
#include "ir/scalar_expr.h"
#include "ir/type.h"
#include "ir/type_inference.h"

namespace pypto {
namespace ir {

// Helper to get kwargs value with default (uses vector to preserve order)
template <typename T>
T GetKwarg(const std::vector<std::pair<std::string, std::any>> &kwargs, const std::string &key,
    const std::optional<T> &defaultValue = std::nullopt) {
    for (const auto &[k, v] : kwargs) {
        if (k == key) {
            return AnyCast<T>(v, "kwarg key: " + key);
        }
    }
    if (defaultValue) {
        return *defaultValue;
    }
    throw ValueError("Missing kwarg: " + key);
}

TypePtr DeduceBlockGetBlockIdxType(const std::vector<ExprPtr> &args,
    const std::vector<std::pair<std::string, std::any>> &kwargs, const std::string &opName, const Span &span) {
    (void)kwargs;
    CHECK(args.size() == 0) << "The operator " << opName << " requires no arguments, but got " << args.size() << " at " << span.ToString();

    // get_block_idx returns INT32 scalar
    return std::make_shared<ScalarType>(DataType::INT32);
}

TypePtr DeduceBlockLoadType(const std::vector<ExprPtr> &args,
    const std::vector<std::pair<std::string, std::any>> &kwargs, const std::string &opName, const Span &span) {
    (void)kwargs;
    // load signature: (tensor, row_offset, col_offset, height, width)
    // We need at least the tensor argument
    CHECK(args.size() >= 1) << "The operator " << opName << " requires at least 1 argument, but got " << args.size() << " at " << span.ToString();

    // First argument must be TensorType
    auto tensorType = As<TensorType>(args[0]->GetType());
    CHECK(tensorType) << "The operator " << opName << " requires first argument to be a TensorType, but got "
                      << args[0]->GetType()->TypeName() << " at " << span.ToString();

    // If we have shape arguments (height, width), use them to determine tile shape
    // Otherwise, we need to infer from context or use dynamic dimensions
    std::vector<ExprPtr> tileShape;

    if (args.size() >= 5) {
        // We have height and width arguments (args[3] and args[4])
        // These should be scalar expressions that we can use as dimensions
        // For now, we'll use them directly as shape dimensions
        tileShape.push_back(args[3]);
        tileShape.push_back(args[4]);
    } else {
        // Use dynamic dimensions if shape is not provided
        // Create ConstInt expressions for dynamic dimensions
        auto dynamicDimHeight =
            std::make_shared<ConstInt>(static_cast<int>(kDynamicDim), DataType::INT32, Span::Unknown());
        auto dynamicDimWidth =
            std::make_shared<ConstInt>(static_cast<int>(kDynamicDim), DataType::INT32, Span::Unknown());
        tileShape.push_back(dynamicDimHeight);
        tileShape.push_back(dynamicDimWidth);
    }

    // Return TileType with same dtype as tensor
    return std::make_shared<TileType>(tileShape, tensorType->dtype_);
}

TypePtr DeduceBlockStoreType(const std::vector<ExprPtr> &args,
    const std::vector<std::pair<std::string, std::any>> &kwargs, const std::string &opName, const Span &span) {
    (void)kwargs;
    // store signature: (tile, row_offset, col_offset, height, width, output_tensor)
    // We need at least the tile and output_tensor arguments
    CHECK(args.size() >= 2) << "The operator " << opName << " requires at least 2 arguments, but got " << args.size() << " at " << span.ToString();

    // First argument must be TileType
    auto tileType = As<TileType>(args[0]->GetType());
    CHECK(tileType) << "The operator " << opName << " requires first argument to be a TileType, but got "
                    << args[0]->GetType()->TypeName() << " at " << span.ToString();

    // Last argument should be the output tensor
    auto outputTensorType = As<TensorType>(args.back()->GetType());
    CHECK(outputTensorType) << "The operator " << opName << " requires last argument to be a TensorType, but got "
                            << args.back()->GetType()->TypeName() << " at " << span.ToString();

    // store returns the output tensor (same type)
    return outputTensorType;
}

TypePtr DeduceBlockMoveType(const std::vector<ExprPtr> &args,
    const std::vector<std::pair<std::string, std::any>> &kwargs, const std::string &opName, const Span &span) {
    // Validate args: expect exactly 1 argument (tile)
    CHECK(args.size() == 1) << "The operator " << opName << " requires 1 argument, but got " << args.size() << " at " << span.ToString();

    // Validate first argument is TileType
    auto tileType = As<TileType>(args[0]->GetType());
    CHECK(tileType) << "The operator " << opName << " requires first argument to be a TileType, but got "
                    << args[0]->GetType()->TypeName() << " at " << span.ToString();

    // Extract transpose attribute (default: false)
    bool transpose = GetKwarg<bool>(kwargs, "transpose", false);

    // Determine output shape based on transpose flag
    const auto &inputShape = tileType->shape_;
    std::vector<ExprPtr> outputShape;

    if (transpose && inputShape.size() == 2) {
        // Transpose: swap dimensions [H, W] -> [W, H]
        outputShape = {inputShape[1], inputShape[0]};
    } else {
        // No transpose: keep original shape
        outputShape = inputShape;
    }

    // Return TileType with computed shape and same dtype (no explicit MemRef)
    return std::make_shared<TileType>(outputShape, tileType->dtype_);
}

TypePtr DeduceBlockAllocType(const std::vector<ExprPtr> &args,
    const std::vector<std::pair<std::string, std::any>> &kwargs, const std::string &opName, const Span &span) {
    (void)kwargs;
    // alloc signature: (memory_space, addr, size, id)
    // Takes MemRef fields as arguments and returns MemRefType
    CHECK(args.size() == 4) << "The operator " << opName << " requires exactly 4 arguments, but got " << args.size() << " at " << span.ToString();

    // Return MemRefType
    return GetMemRefType();
}

TypePtr DeduceBlockZerosType(const std::vector<ExprPtr> &args,
    const std::vector<std::pair<std::string, std::any>> &kwargs, const std::string &opName, const Span &span) {
    // zeros signature: (dim1, dim2, ..., dimN) with dtype kwarg
    // Creates a zero-initialized tile with arbitrary dimensions
    CHECK(args.size() >= 1) << "The operator " << opName << " requires at least 1 dimension argument, but got "
                            << args.size() << " at " << span.ToString();

    // Extract dtype from kwargs (required)
    int dtypeCode = GetKwarg<int>(kwargs, "dtype");
    DataType dtype = static_cast<DataType>(dtypeCode);

    // All arguments are shape dimensions
    std::vector<ExprPtr> shape;
    for (size_t i = 0; i < args.size(); ++i) {
        // Verify each dimension is a scalar type
        auto scalarType = As<ScalarType>(args[i]->GetType());
        CHECK(scalarType) << "The operator " << opName << " requires dimension " << i << " to be a scalar, but got "
                          << args[i]->GetType()->TypeName() << " at " << span.ToString();
        shape.push_back(args[i]);
    }

    // Return TileType with specified shape and dtype
    return std::make_shared<TileType>(shape, dtype);
}

// ============================================================================
// Registration Function for Block Memory Operations
// ============================================================================

REGISTER_OP("block.get_block_idx")
    .SetOpCategory("BlockOp")
    .SetDescription("Get the current block index")
    .NoArgument()
    .SetDeduceType([](const std::vector<ExprPtr> &args, const std::vector<std::pair<std::string, std::any>> &kwargs,
        const Span &span) {
        return DeduceBlockGetBlockIdxType(args, kwargs, "block.get_block_idx", span);
    });

REGISTER_OP("block.load")
    .SetOpCategory("BlockOp")
    .SetDescription("Copy data from tensor to unified buffer (tile)")
    .AddArgument("tensor", "Source tensor (TensorType)")
    .AddArgument("row_offset", "Row offset (scalar)")
    .AddArgument("col_offset", "Column offset (scalar)")
    .AddArgument("height", "Tile height (scalar)")
    .AddArgument("width", "Tile width (scalar)")
    .set_attr<int>("target_memory")
    .SetDeduceType([](const std::vector<ExprPtr> &args, const std::vector<std::pair<std::string, std::any>> &kwargs,
        const Span &span) {
        return DeduceBlockLoadType(args, kwargs, "block.load", span);
    });

REGISTER_OP("block.store")
    .SetOpCategory("BlockOp")
    .SetDescription("Copy data from unified buffer (tile) to tensor")
    .AddArgument("tile", "Source tile (TileType)")
    .AddArgument("row_offset", "Row offset (scalar)")
    .AddArgument("col_offset", "Column offset (scalar)")
    .AddArgument("height", "Output height (scalar)")
    .AddArgument("width", "Output width (scalar)")
    .AddArgument("output_tensor", "Output tensor (TensorType)")
    .SetDeduceType([](const std::vector<ExprPtr> &args, const std::vector<std::pair<std::string, std::any>> &kwargs,
        const Span &span) {
        return DeduceBlockStoreType(args, kwargs, "block.store", span);
    });

REGISTER_OP("block.l0c_store")
    .SetOpCategory("BlockOp")
    .SetDescription("Copy data from L0C tile to GM tensor")
    .AddArgument("tile", "Source tile (TileType)")
    .AddArgument("row_offset", "Row offset (scalar)")
    .AddArgument("col_offset", "Column offset (scalar)")
    .AddArgument("height", "Output height (scalar)")
    .AddArgument("width", "Output width (scalar)")
    .AddArgument("output_tensor", "Output tensor (TensorType)")
    .SetDeduceType([](const std::vector<ExprPtr> &args, const std::vector<std::pair<std::string, std::any>> &kwargs,
        const Span &span) {
        return DeduceBlockStoreType(args, kwargs, "block.l0c_store", span);
    });

REGISTER_OP("block.move")
    .SetOpCategory("BlockOp")
    .SetDescription("Move tile to memory levels (UB/L1/L0A/L0B) with optional transpose")
    .AddArgument("tile", "Input tile (TileType)")
    .set_attr<bool>("transpose")
    .set_attr<int>("target_memory")
    .SetDeduceType([](const std::vector<ExprPtr> &args, const std::vector<std::pair<std::string, std::any>> &kwargs,
        const Span &span) {
        return DeduceBlockMoveType(args, kwargs, "block.move", span);
    });

REGISTER_OP("block.alloc")
    .SetOpCategory("BlockOp")
    .SetDescription("Allocate memory for a MemRef object")
    .AddArgument("memory_space", "Memory space (int enum value)")
    .AddArgument("addr", "Starting address expression")
    .AddArgument("size", "Size in bytes (scalar)")
    .AddArgument("id", "MemRef ID (scalar)")
    .SetDeduceType([](const std::vector<ExprPtr> &args, const std::vector<std::pair<std::string, std::any>> &kwargs,
        const Span &span) {
        return DeduceBlockAllocType(args, kwargs, "block.alloc", span);
    });

REGISTER_OP("block.zeros")
    .SetOpCategory("BlockOp")
    .SetDescription("Create a zero-initialized tile with specified shape and dtype")
    .SetPipe(PipeType::V)
    .AddArgument("dims", "Shape dimensions (one or more scalars)")
    .set_attr<int>("dtype")
    .SetDeduceType([](const std::vector<ExprPtr> &args, const std::vector<std::pair<std::string, std::any>> &kwargs,
        const Span &span) {
        return DeduceBlockZerosType(args, kwargs, "block.zeros", span);
    });

} // namespace ir
} // namespace pypto
