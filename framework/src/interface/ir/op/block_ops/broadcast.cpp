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

#include "core/logging.h"
#include "ir/kind_traits.h"
#include "ir/op_registry.h"
#include "ir/scalar_expr.h"
#include "ir/type.h"
#include "ir/type_inference.h"

namespace pypto {
namespace ir {

TypePtr DeduceBlockRowExpandType(const std::vector<ExprPtr> &args,
    const std::vector<std::pair<std::string, std::any>> & /*kwargs*/, const std::string &opName) {
    CHECK(args.size() == 2) << "The operator " << opName << " requires exactly 2 arguments, but got " << args.size();

    // First argument must be TileType (the main tile)
    auto tileType = As<TileType>(args[0]->GetType());
    CHECK(tileType) << "The operator " << opName << " requires first argument to be a TileType, but got "
                    << args[0]->GetType()->TypeName();

    // Second argument must be TileType (the row vector)
    auto rowType = As<TileType>(args[1]->GetType());
    CHECK(rowType) << "The operator " << opName << " requires second argument to be a TileType, but got "
                   << args[1]->GetType()->TypeName();

    // Get shapes
    const auto &tileShape = tileType->shape_;
    const auto &rowShape = rowType->shape_;

    // Both must have at least 2D (last 2 dimensions are used for broadcasting)
    CHECK(tileShape.size() >= 2) << "The operator " << opName
                                 << " requires first argument to have at least 2 dimensions, but got "
                                 << tileShape.size() << " dimensions";
    CHECK(rowShape.size() >= 2) << "The operator " << opName
                                << " requires second argument to have at least 2 dimensions, but got "
                                << rowShape.size() << " dimensions";

    // Last dimension of row vector must be 1
    auto rowColConst = As<ConstInt>(rowShape[rowShape.size() - 1]);
    CHECK(rowColConst && rowColConst->value_ == 1)
        << "The operator " << opName << " requires second argument's last dimension to be 1, but got "
        << (rowColConst ? std::to_string(rowColConst->value_) : "?");

    // Second-to-last dimension (rows) must match
    auto tileRowsConst = As<ConstInt>(tileShape[tileShape.size() - 2]);
    auto rowRowsConst = As<ConstInt>(rowShape[rowShape.size() - 2]);

    if (tileRowsConst && rowRowsConst) {
        CHECK(tileRowsConst->value_ == rowRowsConst->value_)
            << "The operator " << opName
            << " requires matching row dimensions, but got tile rows=" << tileRowsConst->value_
            << " and row_vec rows=" << rowRowsConst->value_;
    }

    // Promote data types
    auto resultDtype = PromoteDataTypes(tileType->dtype_, rowType->dtype_);
    CHECK(resultDtype) << "The operator " << opName << " requires compatible data types, but got "
                       << tileType->dtype_.ToString() << " and " << rowType->dtype_.ToString();

    // Output has the same shape as the main tile
    return std::make_shared<TileType>(tileShape, *resultDtype);
}

// ============================================================================
// Registration Function for Block Row Broadcast Operations
// ============================================================================

REGISTER_OP("block.row_expand_sub")
    .SetOpCategory("BlockOp")
    .SetDescription("Row-wise broadcast subtraction: tile - row_vec (broadcasted)")
    .AddArgument("tile", "Input tile (TileType, 2D [M, N])")
    .AddArgument("row_vec", "Row vector (TileType, 2D [M, 1])")
    .SetDeduceType([](const std::vector<ExprPtr> &args, const std::vector<std::pair<std::string, std::any>> &kwargs) {
        return DeduceBlockRowExpandType(args, kwargs, "block.row_expand_sub");
    });

REGISTER_OP("block.row_expand_div")
    .SetOpCategory("BlockOp")
    .SetDescription("Row-wise broadcast division: tile / row_vec (broadcasted)")
    .AddArgument("tile", "Input tile (TileType, 2D [M, N])")
    .AddArgument("row_vec", "Row vector (TileType, 2D [M, 1])")
    .SetDeduceType([](const std::vector<ExprPtr> &args, const std::vector<std::pair<std::string, std::any>> &kwargs) {
        return DeduceBlockRowExpandType(args, kwargs, "block.row_expand_div");
    });

REGISTER_OP("block.row_expand_mul")
    .SetOpCategory("BlockOp")
    .SetDescription("Row-wise broadcast multiplication: tile * row_vec (broadcasted)")
    .AddArgument("tile", "Input tile (TileType, 2D [M, N])")
    .AddArgument("row_vec", "Row vector (TileType, 2D [M, 1])")
    .SetDeduceType([](const std::vector<ExprPtr> &args, const std::vector<std::pair<std::string, std::any>> &kwargs) {
        return DeduceBlockRowExpandType(args, kwargs, "block.row_expand_mul");
    });

REGISTER_OP("block.row_expand_add")
    .SetOpCategory("BlockOp")
    .SetDescription("Row-wise broadcast addition: tile + row_vec (broadcasted)")
    .SetPipe(PipeType::V)
    .AddArgument("tile", "Input tile (TileType, 2D [M, N])")
    .AddArgument("row_vec", "Row vector (TileType, 2D [M, 1])")
    .SetDeduceType([](const std::vector<ExprPtr> &args, const std::vector<std::pair<std::string, std::any>> &kwargs) {
        return DeduceBlockRowExpandType(args, kwargs, "block.row_expand_add");
    });

} // namespace ir
} // namespace pypto
