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

#include <algorithm>
#include <any>
#include <memory>
#include <string>
#include <vector>

#include "core/logging.h"
#include "ir/core.h"
#include "ir/kind_traits.h"
#include "ir/op_registry.h"
#include "ir/type.h"
#include "ir/type_inference.h"

namespace pypto {
namespace ir {

TypePtr DeduceBlockUnaryType(const std::vector<ExprPtr> &args,
    const std::vector<std::pair<std::string, std::any>> &kwargs, const std::string &opName, const Span &span) {
    (void)kwargs;
    CHECK(args.size() == 1) << "The operator " << opName << " requires exactly 1 argument, but got " << args.size() << " at " << span.ToString();

    // Argument must be TileType
    auto tileType = As<TileType>(args[0]->GetType());
    CHECK(tileType) << "The operator " << opName << " requires argument to be a TileType, but got "
                    << args[0]->GetType()->TypeName() << " at " << span.ToString();

    // Unary operations preserve shape and data type
    return std::make_shared<TileType>(tileType->shape_, tileType->dtype_);
}

TypePtr DeduceBlockCastType(const std::vector<ExprPtr> &args,
    const std::vector<std::pair<std::string, std::any>> &kwargs, const std::string &opName, const Span &span) {
    CHECK(args.size() == 1) << "The operator " << opName << " requires exactly 1 argument, but got " << args.size() << " at " << span.ToString();

    // Argument must be TileType
    auto tileType = As<TileType>(args[0]->GetType());
    CHECK(tileType) << "The operator " << opName << " requires argument to be a TileType, but got "
                    << args[0]->GetType()->TypeName() << " at " << span.ToString();

    // Extract target_dtype from kwargs
    auto it = std::find_if(kwargs.begin(), kwargs.end(), [](const auto &pair) { return pair.first == "target_dtype"; });
    CHECK(it != kwargs.end()) << "The operator " << opName << " requires 'target_dtype' kwarg" << " at " << span.ToString();

    DataType targetDtype = static_cast<DataType>(std::any_cast<int>(it->second));

    // Cast operation preserves shape but changes data type
    return std::make_shared<TileType>(tileType->shape_, targetDtype);
}

// ============================================================================
// Op Registration
// ============================================================================

REGISTER_OP("block.neg")
    .SetOpCategory("BlockOp")
    .SetDescription("Negation of a tile (element-wise)")
    .AddArgument("tile", "Input tile (TileType)")
    .SetDeduceType([](const std::vector<ExprPtr> &args, const std::vector<std::pair<std::string, std::any>> &kwargs,
        const Span &span) {
        return DeduceBlockUnaryType(args, kwargs, "block.neg", span);
    });

REGISTER_OP("block.exp")
    .SetOpCategory("BlockOp")
    .SetDescription("Exponential function of a tile (element-wise)")
    .AddArgument("tile", "Input tile (TileType)")
    .SetDeduceType([](const std::vector<ExprPtr> &args, const std::vector<std::pair<std::string, std::any>> &kwargs,
        const Span &span) {
        return DeduceBlockUnaryType(args, kwargs, "block.exp", span);
    });

REGISTER_OP("block.recip")
    .SetOpCategory("BlockOp")
    .SetDescription("Reciprocal (1/x) of a tile (element-wise)")
    .AddArgument("tile", "Input tile (TileType)")
    .SetDeduceType([](const std::vector<ExprPtr> &args, const std::vector<std::pair<std::string, std::any>> &kwargs,
        const Span &span) {
        return DeduceBlockUnaryType(args, kwargs, "block.recip", span);
    });

REGISTER_OP("block.sqrt")
    .SetOpCategory("BlockOp")
    .SetDescription("Square root of a tile (element-wise)")
    .AddArgument("tile", "Input tile (TileType)")
    .SetDeduceType([](const std::vector<ExprPtr> &args, const std::vector<std::pair<std::string, std::any>> &kwargs,
        const Span &span) {
        return DeduceBlockUnaryType(args, kwargs, "block.sqrt", span);
    });

REGISTER_OP("block.rsqrt")
    .SetOpCategory("BlockOp")
    .SetDescription("Reciprocal square root (1/sqrt(x)) of a tile (element-wise)")
    .AddArgument("tile", "Input tile (TileType)")
    .SetDeduceType([](const std::vector<ExprPtr> &args, const std::vector<std::pair<std::string, std::any>> &kwargs,
        const Span &span) {
        return DeduceBlockUnaryType(args, kwargs, "block.rsqrt", span);
    });

REGISTER_OP("block.cast")
    .SetOpCategory("BlockOp")
    .SetDescription("Cast tile to target data type (element-wise)")
    .SetPipe(PipeType::V)
    .AddArgument("tile", "Input tile (TileType)")
    .set_attr<int>("target_dtype")
    .SetDeduceType([](const std::vector<ExprPtr> &args, const std::vector<std::pair<std::string, std::any>> &kwargs,
        const Span &span) {
        return DeduceBlockCastType(args, kwargs, "block.cast", span);
    });

REGISTER_OP("block.log")
    .SetOpCategory("BlockOp")
    .SetDescription("Natural logarithm of a tile (element-wise)")
    .SetPipe(PipeType::V)
    .AddArgument("tile", "Input tile (TileType)")
    .SetDeduceType([](const std::vector<ExprPtr> &args, const std::vector<std::pair<std::string, std::any>> &kwargs,
        const Span &span) {
        return DeduceBlockUnaryType(args, kwargs, "block.log", span);
    });

REGISTER_OP("block.abs")
    .SetOpCategory("BlockOp")
    .SetDescription("Absolute value of a tile (element-wise)")
    .SetPipe(PipeType::V)
    .AddArgument("tile", "Input tile (TileType)")
    .SetDeduceType([](const std::vector<ExprPtr> &args, const std::vector<std::pair<std::string, std::any>> &kwargs,
        const Span &span) {
        return DeduceBlockUnaryType(args, kwargs, "block.abs", span);
    });

REGISTER_OP("block.relu")
    .SetOpCategory("BlockOp")
    .SetDescription("ReLU activation function of a tile (element-wise)")
    .SetPipe(PipeType::V)
    .AddArgument("tile", "Input tile (TileType)")
    .SetDeduceType([](const std::vector<ExprPtr> &args, const std::vector<std::pair<std::string, std::any>> &kwargs,
        const Span &span) {
        return DeduceBlockUnaryType(args, kwargs, "block.relu", span);
    });

} // namespace ir
} // namespace pypto
