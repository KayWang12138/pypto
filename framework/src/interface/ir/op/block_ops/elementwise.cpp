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

TypePtr DeduceBlockOpElementwiseBinaryType(const std::vector<ExprPtr> &args,
    const std::vector<std::pair<std::string, std::any>> &kwargs, const std::string &opName) {
    (void)kwargs;
    CHECK(args.size() == 2) << "The operator " << opName << " requires exactly 2 arguments, but got " << args.size();

    // Both arguments must be TileType
    auto tileType1 = As<TileType>(args[0]->GetType());
    auto tileType2 = As<TileType>(args[1]->GetType());

    CHECK(tileType1) << "The operator " << opName << " requires first argument to be a TileType, but got "
                     << args[0]->GetType()->TypeName();
    CHECK(tileType2) << "The operator " << opName << " requires second argument to be a TileType, but got "
                     << args[1]->GetType()->TypeName();

    // Use broadcasting
    auto resultDtype = PromoteDataTypes(tileType1->dtype_, tileType2->dtype_);
    CHECK(resultDtype) << "The operator " << opName << " requires compatible data types, but got "
                       << args[0]->GetType()->TypeName() << " and " << args[1]->GetType()->TypeName();

    auto broadcastResult = BroadcastShapes(tileType1->shape_, tileType2->shape_);
    CHECK(broadcastResult.success) << "The operator " << opName << " requires compatible shapes, but got "
                                   << FormatShape(tileType1->shape_) << " and " << FormatShape(tileType2->shape_);

    return std::make_shared<TileType>(broadcastResult.shape, *resultDtype);
}

TypePtr DeduceBlockOpScalarBinaryType(const std::vector<ExprPtr> &args,
    const std::vector<std::pair<std::string, std::any>> &kwargs, const std::string &opName) {
    (void)kwargs;
    CHECK(args.size() == 2) << "The operator " << opName << " requires exactly 2 arguments, but got " << args.size();

    // First argument must be TileType
    auto tileType = As<TileType>(args[0]->GetType());
    CHECK(tileType) << "The operator " << opName << " requires first argument to be a TileType, but got "
                    << args[0]->GetType()->TypeName();

    // Second argument MUST be ScalarType
    auto scalarType = As<ScalarType>(args[1]->GetType());
    CHECK(scalarType) << "The operator " << opName << " requires second argument to be a ScalarType, but got "
                      << args[1]->GetType()->TypeName();

    // Result has same shape as tile, with promoted dtype
    auto resultDtype = PromoteDataTypes(tileType->dtype_, scalarType->dtype_);
    CHECK(resultDtype) << "The operator " << opName << " requires compatible data types, but got "
                       << tileType->dtype_.ToString() << " and " << scalarType->dtype_.ToString();

    return std::make_shared<TileType>(tileType->shape_, *resultDtype);
}

// ============================================================================
// Op Registration
// ============================================================================

REGISTER_OP("block.mul")
    .SetOpCategory("BlockOp")
    .SetDescription("Element-wise multiplication of two tiles with broadcasting")
    .AddArgument("lhs", "Left-hand side tile (TileType)")
    .AddArgument("rhs", "Right-hand side tile (TileType)")
    .SetDeduceType([](const std::vector<ExprPtr> &args, const std::vector<std::pair<std::string, std::any>> &kwargs) {
        return DeduceBlockOpElementwiseBinaryType(args, kwargs, "block.mul");
    });

REGISTER_OP("block.add")
    .SetOpCategory("BlockOp")
    .SetDescription("Element-wise addition of two tiles with broadcasting")
    .AddArgument("lhs", "Left-hand side tile (TileType)")
    .AddArgument("rhs", "Right-hand side tile (TileType)")
    .SetDeduceType([](const std::vector<ExprPtr> &args, const std::vector<std::pair<std::string, std::any>> &kwargs) {
        return DeduceBlockOpElementwiseBinaryType(args, kwargs, "block.add");
    });

REGISTER_OP("block.div")
    .SetOpCategory("BlockOp")
    .SetDescription("Element-wise division of two tiles with broadcasting")
    .AddArgument("lhs", "Left-hand side tile (TileType)")
    .AddArgument("rhs", "Right-hand side tile (TileType)")
    .SetDeduceType([](const std::vector<ExprPtr> &args, const std::vector<std::pair<std::string, std::any>> &kwargs) {
        return DeduceBlockOpElementwiseBinaryType(args, kwargs, "block.div");
    });

REGISTER_OP("block.sub")
    .SetOpCategory("BlockOp")
    .SetDescription("Element-wise subtraction of two tiles with broadcasting")
    .AddArgument("lhs", "Left-hand side tile (TileType)")
    .AddArgument("rhs", "Right-hand side tile (TileType)")
    .SetDeduceType([](const std::vector<ExprPtr> &args, const std::vector<std::pair<std::string, std::any>> &kwargs) {
        return DeduceBlockOpElementwiseBinaryType(args, kwargs, "block.sub");
    });

REGISTER_OP("block.maximum")
    .SetOpCategory("BlockOp")
    .SetDescription("Element-wise maximum of two tiles with broadcasting")
    .AddArgument("lhs", "Left-hand side tile (TileType)")
    .AddArgument("rhs", "Right-hand side tile (TileType)")
    .SetDeduceType([](const std::vector<ExprPtr> &args, const std::vector<std::pair<std::string, std::any>> &kwargs) {
        return DeduceBlockOpElementwiseBinaryType(args, kwargs, "block.maximum");
    });

REGISTER_OP("block.minimum")
    .SetOpCategory("BlockOp")
    .SetDescription("Element-wise minimum of two tiles with broadcasting")
    .SetPipe(PipeType::V)
    .AddArgument("lhs", "Left-hand side tile (TileType)")
    .AddArgument("rhs", "Right-hand side tile (TileType)")
    .SetDeduceType([](const std::vector<ExprPtr> &args, const std::vector<std::pair<std::string, std::any>> &kwargs) {
        return DeduceBlockOpElementwiseBinaryType(args, kwargs, "block.minimum");
    });

REGISTER_OP("block.muls")
    .SetOpCategory("BlockOp")
    .SetDescription("Element-wise multiplication of tile and scalar")
    .AddArgument("lhs", "Tile (TileType)")
    .AddArgument("rhs", "Scalar (ScalarType)")
    .SetDeduceType([](const std::vector<ExprPtr> &args, const std::vector<std::pair<std::string, std::any>> &kwargs) {
        return DeduceBlockOpScalarBinaryType(args, kwargs, "block.muls");
    });

REGISTER_OP("block.adds")
    .SetOpCategory("BlockOp")
    .SetDescription("Element-wise addition of tile and scalar")
    .AddArgument("lhs", "Tile (TileType)")
    .AddArgument("rhs", "Scalar (ScalarType)")
    .SetDeduceType([](const std::vector<ExprPtr> &args, const std::vector<std::pair<std::string, std::any>> &kwargs) {
        return DeduceBlockOpScalarBinaryType(args, kwargs, "block.adds");
    });

REGISTER_OP("block.divs")
    .SetOpCategory("BlockOp")
    .SetDescription("Element-wise division of tile and scalar")
    .AddArgument("lhs", "Tile (TileType)")
    .AddArgument("rhs", "Scalar (ScalarType)")
    .SetDeduceType([](const std::vector<ExprPtr> &args, const std::vector<std::pair<std::string, std::any>> &kwargs) {
        return DeduceBlockOpScalarBinaryType(args, kwargs, "block.divs");
    });

REGISTER_OP("block.subs")
    .SetOpCategory("BlockOp")
    .SetDescription("Element-wise subtraction of tile and scalar")
    .AddArgument("lhs", "Tile (TileType)")
    .AddArgument("rhs", "Scalar (ScalarType)")
    .SetDeduceType([](const std::vector<ExprPtr> &args, const std::vector<std::pair<std::string, std::any>> &kwargs) {
        return DeduceBlockOpScalarBinaryType(args, kwargs, "block.subs");
    });

// Type deduction for block.cmp and block.cmps (comparison operations)
TypePtr DeduceBlockCmpType(const std::vector<ExprPtr> &args,
    const std::vector<std::pair<std::string, std::any>> &kwargs, const std::string &opName, bool isScalarRhs = false) {
    CHECK(args.size() == 2) << "The operator " << opName << " requires exactly 2 arguments, but got " << args.size();

    // Validate cmp_type attribute exists
    bool hasCmpType = false;
    for (const auto &kwarg : kwargs) {
        if (kwarg.first == "cmp_type") {
            hasCmpType = true;
            break;
        }
    }
    CHECK(hasCmpType) << "The operator " << opName << " requires 'cmp_type' attribute";

    // First argument must be TileType
    auto tileType1 = As<TileType>(args[0]->GetType());
    CHECK(tileType1) << "The operator " << opName << " requires first argument to be a TileType, but got "
                     << args[0]->GetType()->TypeName();

    if (isScalarRhs) {
        // Second argument MUST be ScalarType
        auto scalarType = As<ScalarType>(args[1]->GetType());
        CHECK(scalarType) << "The operator " << opName << " requires second argument to be a ScalarType, but got "
                          << args[1]->GetType()->TypeName();

        // Result has same shape as tile, with promoted dtype
        auto resultDtype = PromoteDataTypes(tileType1->dtype_, scalarType->dtype_);
        CHECK(resultDtype) << "The operator " << opName << " requires compatible data types, but got "
                           << tileType1->dtype_.ToString() << " and " << scalarType->dtype_.ToString();

        return std::make_shared<TileType>(tileType1->shape_, *resultDtype);
    } else {
        // Second argument must be TileType
        auto tileType2 = As<TileType>(args[1]->GetType());
        CHECK(tileType2) << "The operator " << opName << " requires second argument to be a TileType, but got "
                         << args[1]->GetType()->TypeName();

        // Use broadcasting
        auto resultDtype = PromoteDataTypes(tileType1->dtype_, tileType2->dtype_);
        CHECK(resultDtype) << "The operator " << opName << " requires compatible data types, but got "
                           << args[0]->GetType()->TypeName() << " and " << args[1]->GetType()->TypeName();

        auto broadcastResult = BroadcastShapes(tileType1->shape_, tileType2->shape_);
        CHECK(broadcastResult.success) << "The operator " << opName << " requires compatible shapes, but got "
                                       << FormatShape(tileType1->shape_) << " and " << FormatShape(tileType2->shape_);

        return std::make_shared<TileType>(broadcastResult.shape, *resultDtype);
    }
}

REGISTER_OP("block.cmp")
    .SetOpCategory("BlockOp")
    .SetDescription("Element-wise comparison of two tiles (returns boolean tile)")
    .SetPipe(PipeType::V)
    .AddArgument("lhs", "Left-hand side tile (TileType)")
    .AddArgument("rhs", "Right-hand side tile (TileType)")
    .set_attr<int>("cmp_type")
    .SetDeduceType([](const std::vector<ExprPtr> &args, const std::vector<std::pair<std::string, std::any>> &kwargs) {
        return DeduceBlockCmpType(args, kwargs, "block.cmp", false);
    });

REGISTER_OP("block.cmps")
    .SetOpCategory("BlockOp")
    .SetDescription("Element-wise comparison of tile and scalar (returns boolean tile)")
    .SetPipe(PipeType::V)
    .AddArgument("lhs", "Tile (TileType)")
    .AddArgument("rhs", "Scalar (ScalarType)")
    .set_attr<int>("cmp_type")
    .SetDeduceType([](const std::vector<ExprPtr> &args, const std::vector<std::pair<std::string, std::any>> &kwargs) {
        return DeduceBlockCmpType(args, kwargs, "block.cmps", true);
    });

// Type deduction for column expand operations
TypePtr DeduceBlockColExpandType(const std::vector<ExprPtr> &args,
    const std::vector<std::pair<std::string, std::any>> &kwargs, const std::string &opName) {
    (void)kwargs;
    CHECK(args.size() == 2) << "The operator " << opName << " requires exactly 2 arguments, but got " << args.size();

    // First argument is the target tile (shape to expand to)
    auto targetType = As<TileType>(args[0]->GetType());
    CHECK(targetType) << "The operator " << opName << " requires first argument to be a TileType, but got "
                      << args[0]->GetType()->TypeName();

    // Second argument is the column tile to expand (shape [1, cols])
    auto colType = As<TileType>(args[1]->GetType());
    CHECK(colType) << "The operator " << opName << " requires second argument to be a TileType, but got "
                   << args[1]->GetType()->TypeName();

    // Result has same shape as target, with promoted dtype
    auto resultDtype = PromoteDataTypes(targetType->dtype_, colType->dtype_);
    CHECK(resultDtype) << "The operator " << opName << " requires compatible data types";

    return std::make_shared<TileType>(targetType->shape_, *resultDtype);
}

// Type deduction for scalar expand operations
TypePtr DeduceBlockExpandScalarType(const std::vector<ExprPtr> &args,
    const std::vector<std::pair<std::string, std::any>> &kwargs, const std::string &opName) {
    (void)kwargs;
    CHECK(args.size() == 2) << "The operator " << opName << " requires exactly 2 arguments, but got " << args.size();

    // First argument is the target tile
    auto tileType = As<TileType>(args[0]->GetType());
    CHECK(tileType) << "The operator " << opName << " requires first argument to be a TileType, but got "
                    << args[0]->GetType()->TypeName();

    // Second argument is the scalar to expand
    auto scalarType = As<ScalarType>(args[1]->GetType());
    CHECK(scalarType) << "The operator " << opName << " requires second argument to be a ScalarType, but got "
                      << args[1]->GetType()->TypeName();

    // Result has same shape as tile, with promoted dtype
    auto resultDtype = PromoteDataTypes(tileType->dtype_, scalarType->dtype_);
    CHECK(resultDtype) << "The operator " << opName << " requires compatible data types";

    return std::make_shared<TileType>(tileType->shape_, *resultDtype);
}

REGISTER_OP("block.col_expand")
    .SetOpCategory("BlockOp")
    .SetDescription("Expand column tile [1, cols] to target shape [rows, cols]")
    .SetPipe(PipeType::V)
    .AddArgument("target", "Target tile defining output shape (TileType)")
    .AddArgument("col_tile", "Column tile to expand (TileType, shape [1, cols])")
    .SetDeduceType([](const std::vector<ExprPtr> &args, const std::vector<std::pair<std::string, std::any>> &kwargs) {
        return DeduceBlockColExpandType(args, kwargs, "block.col_expand");
    });

REGISTER_OP("block.col_expand_mul")
    .SetOpCategory("BlockOp")
    .SetDescription("Expand column tile and multiply with target tile")
    .SetPipe(PipeType::V)
    .AddArgument("target", "Target tile (TileType)")
    .AddArgument("col_tile", "Column tile to expand and multiply (TileType, shape [1, cols])")
    .SetDeduceType([](const std::vector<ExprPtr> &args, const std::vector<std::pair<std::string, std::any>> &kwargs) {
        return DeduceBlockColExpandType(args, kwargs, "block.col_expand_mul");
    });

REGISTER_OP("block.col_expand_div")
    .SetOpCategory("BlockOp")
    .SetDescription("Expand column tile and divide target tile by it")
    .SetPipe(PipeType::V)
    .AddArgument("target", "Target tile (TileType)")
    .AddArgument("col_tile", "Column tile to expand and divide by (TileType, shape [1, cols])")
    .SetDeduceType([](const std::vector<ExprPtr> &args, const std::vector<std::pair<std::string, std::any>> &kwargs) {
        return DeduceBlockColExpandType(args, kwargs, "block.col_expand_div");
    });

REGISTER_OP("block.col_expand_sub")
    .SetOpCategory("BlockOp")
    .SetDescription("Expand column tile and subtract from target tile")
    .SetPipe(PipeType::V)
    .AddArgument("target", "Target tile (TileType)")
    .AddArgument("col_tile", "Column tile to expand and subtract (TileType, shape [1, cols])")
    .SetDeduceType([](const std::vector<ExprPtr> &args, const std::vector<std::pair<std::string, std::any>> &kwargs) {
        return DeduceBlockColExpandType(args, kwargs, "block.col_expand_sub");
    });

REGISTER_OP("block.expands")
    .SetOpCategory("BlockOp")
    .SetDescription("Expand scalar to target tile shape")
    .SetPipe(PipeType::V)
    .AddArgument("target", "Target tile defining output shape (TileType)")
    .AddArgument("scalar", "Scalar to expand (ScalarType)")
    .SetDeduceType([](const std::vector<ExprPtr> &args, const std::vector<std::pair<std::string, std::any>> &kwargs) {
        return DeduceBlockExpandScalarType(args, kwargs, "block.expands");
    });

} // namespace ir
} // namespace pypto
