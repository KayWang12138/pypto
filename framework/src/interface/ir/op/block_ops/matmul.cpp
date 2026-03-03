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
#include <string>
#include <vector>

#include "core/logging.h"
#include "ir/expr.h"
#include "ir/kind_traits.h"
#include "ir/op_registry.h"
#include "ir/scalar_expr.h"
#include "ir/type.h"
#include "ir/type_inference.h"

namespace pypto {
namespace ir {

TypePtr DeduceBlockMatMulType(const std::vector<ExprPtr> &args,
    const std::vector<std::pair<std::string, std::any>> &kwargs, const std::string &opName) {
    (void)kwargs;
    CHECK(args.size() == 2) << "The operator " << opName << " requires exactly 2 arguments, but got " << args.size();

    // Both arguments must be TileType
    auto lhsType = As<TileType>(args[0]->GetType());
    auto rhsType = As<TileType>(args[1]->GetType());

    CHECK(lhsType) << "The operator " << opName << " requires first argument to be a TileType, but got "
                   << args[0]->GetType()->TypeName();
    CHECK(rhsType) << "The operator " << opName << " requires second argument to be a TileType, but got "
                   << args[1]->GetType()->TypeName();

    // Extract shapes
    const auto &lhsShape = lhsType->shape_;
    const auto &rhsShape = rhsType->shape_;

    // For block matmul, we require 2D tiles
    CHECK(lhsShape.size() == 2) << "The operator " << opName << " requires lhs to be 2D, but got " << lhsShape.size()
                                << " dimensions";
    CHECK(rhsShape.size() == 2) << "The operator " << opName << " requires rhs to be 2D, but got " << rhsShape.size()
                                << " dimensions";

    // Matrix multiplication: [M, K] @ [K, N] -> [M, N]
    // We need to verify that K dimensions match
    // Note: In PTO ISA, we see [M, K] @ [K, N] -> [M, N]

    ExprPtr mDim = lhsShape[0];
    ExprPtr kDimLhs = lhsShape[1];
    ExprPtr kDimRhs = rhsShape[0];
    ExprPtr nDim = rhsShape[1];

    // Try to verify K dimensions match if they are constant
    auto kLhsConst = As<ConstInt>(kDimLhs);
    auto kRhsConst = As<ConstInt>(kDimRhs);

    if (kLhsConst && kRhsConst) {
        CHECK(kLhsConst->value_ == kRhsConst->value_)
            << "The operator " << opName << " requires matching inner dimensions, but got lhs K=" << kLhsConst->value_
            << " and rhs K=" << kRhsConst->value_;
    }

    // Promote data types
    auto resultDtype = PromoteDataTypes(lhsType->dtype_, rhsType->dtype_);
    CHECK(resultDtype) << "The operator " << opName << " requires compatible data types, but got "
                       << lhsType->dtype_.ToString() << " and " << rhsType->dtype_.ToString();

    // Output shape is [M, N]
    std::vector<ExprPtr> outputShape = {mDim, nDim};

    return std::make_shared<TileType>(outputShape, *resultDtype);
}

TypePtr DeduceBlockMatMulAccType(const std::vector<ExprPtr> &args,
    const std::vector<std::pair<std::string, std::any>> &kwargs, const std::string &opName) {
    (void)kwargs;
    CHECK(args.size() == 3) << "The operator " << opName << " requires exactly 3 arguments, but got " << args.size();

    // All arguments must be TileType
    auto accType = As<TileType>(args[0]->GetType());
    auto lhsType = As<TileType>(args[1]->GetType());
    auto rhsType = As<TileType>(args[2]->GetType());

    CHECK(accType) << "The operator " << opName << " requires first argument (acc) to be a TileType, but got "
                   << args[0]->GetType()->TypeName();
    CHECK(lhsType) << "The operator " << opName << " requires second argument (lhs) to be a TileType, but got "
                   << args[1]->GetType()->TypeName();
    CHECK(rhsType) << "The operator " << opName << " requires third argument (rhs) to be a TileType, but got "
                   << args[2]->GetType()->TypeName();

    // Extract shapes
    const auto &accShape = accType->shape_;
    const auto &lhsShape = lhsType->shape_;
    const auto &rhsShape = rhsType->shape_;

    // For block matmul_acc, we require 2D tiles
    CHECK(accShape.size() == 2) << "The operator " << opName << " requires acc to be 2D, but got " << accShape.size()
                                << " dimensions";
    CHECK(lhsShape.size() == 2) << "The operator " << opName << " requires lhs to be 2D, but got " << lhsShape.size()
                                << " dimensions";
    CHECK(rhsShape.size() == 2) << "The operator " << opName << " requires rhs to be 2D, but got " << rhsShape.size()
                                << " dimensions";

    // Matrix multiplication with accumulation: acc[M, N] += lhs[M, K] @ rhs[K, N]
    ExprPtr mDimAcc = accShape[0];
    ExprPtr nDimAcc = accShape[1];

    // Verify dimensions match
    auto mAccConst = As<ConstInt>(mDimAcc);
    auto mLhsConst = As<ConstInt>(lhsShape[0]);
    auto nAccConst = As<ConstInt>(nDimAcc);
    auto nRhsConst = As<ConstInt>(rhsShape[1]);
    auto kLhsConst = As<ConstInt>(lhsShape[1]);
    auto kRhsConst = As<ConstInt>(rhsShape[0]);

    if (mAccConst && mLhsConst) {
        CHECK(mAccConst->value_ == mLhsConst->value_)
            << "The operator " << opName << " requires matching M dimensions, but got acc M=" << mAccConst->value_
            << " and lhs M=" << mLhsConst->value_;
    }

    if (nAccConst && nRhsConst) {
        CHECK(nAccConst->value_ == nRhsConst->value_)
            << "The operator " << opName << " requires matching N dimensions, but got acc N=" << nAccConst->value_
            << " and rhs N=" << nRhsConst->value_;
    }

    if (kLhsConst && kRhsConst) {
        CHECK(kLhsConst->value_ == kRhsConst->value_)
            << "The operator " << opName << " requires matching K dimensions, but got lhs K=" << kLhsConst->value_
            << " and rhs K=" << kRhsConst->value_;
    }

    // Promote data types
    auto lhsRhsDtype = PromoteDataTypes(lhsType->dtype_, rhsType->dtype_);
    CHECK(lhsRhsDtype) << "The operator " << opName << " requires compatible lhs and rhs data types, but got "
                       << lhsType->dtype_.ToString() << " and " << rhsType->dtype_.ToString();

    auto resultDtype = PromoteDataTypes(accType->dtype_, *lhsRhsDtype);
    CHECK(resultDtype) << "The operator " << opName << " requires compatible accumulator data type, but got "
                       << accType->dtype_.ToString() << " and " << lhsRhsDtype->ToString();

    // Output shape is [M, N] (same as accumulator)
    std::vector<ExprPtr> outputShape = {mDimAcc, nDimAcc};

    return std::make_shared<TileType>(outputShape, *resultDtype);
}

// ============================================================================
// Registration Function for Block Matrix Multiplication Operations
// ============================================================================

REGISTER_OP("block.matmul")
    .SetOpCategory("BlockOp")
    .SetDescription("Matrix multiplication of two tiles")
    .AddArgument("lhs", "Left-hand side tile (TileType, 2D)")
    .AddArgument("rhs", "Right-hand side tile (TileType, 2D)")
    .SetDeduceType([](const std::vector<ExprPtr> &args, const std::vector<std::pair<std::string, std::any>> &kwargs) {
        return DeduceBlockMatMulType(args, kwargs, "block.matmul");
    });

REGISTER_OP("block.matmul_acc")
    .SetOpCategory("BlockOp")
    .SetDescription("Matrix multiplication with accumulation: acc = acc + lhs @ rhs")
    .AddArgument("acc", "Accumulator tile (TileType, 2D)")
    .AddArgument("lhs", "Left-hand side tile (TileType, 2D)")
    .AddArgument("rhs", "Right-hand side tile (TileType, 2D)")
    .SetDeduceType([](const std::vector<ExprPtr> &args, const std::vector<std::pair<std::string, std::any>> &kwargs) {
        return DeduceBlockMatMulAccType(args, kwargs, "block.matmul_acc");
    });

} // namespace ir
} // namespace pypto
