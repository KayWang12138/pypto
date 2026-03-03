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

#include "core/dtype.h"
#include "core/error.h"
#include "core/logging.h"
#include "ir/kind_traits.h"
#include "ir/op_registry.h"
#include "ir/op_utils.h"
#include "ir/type.h"
#include "ir/type_inference.h"

namespace pypto {
namespace ir {

/**
 * \brief Deduce type for batch matrix multiplication
 *
 * Batch matmul operates on multi-dimensional TileTypes with batch dimensions.
 * For inputs with shape [...batch_dims, M, K] and [...batch_dims, K, N],
 * the output has shape [...broadcast_batch_dims, M, N].
 *
 * \param args Arguments: [lhs_tile, rhs_tile]
 * \param kwargs Keyword arguments (unused)
 * \param opName Operator name for error messages
 * \return TileType with output shape
 */
TypePtr DeduceBlockBatchMatMulType(const std::vector<ExprPtr> &args,
    const std::vector<std::pair<std::string, std::any>> & /*kwargs*/, const std::string &opName) {
    INTERNAL_CHECK(args.size() == 2) << "The operator " << opName << " requires exactly 2 arguments, but got "
                                     << args.size();

    // Both arguments must be TileType
    auto lhsType = As<TileType>(args[0]->GetType());
    auto rhsType = As<TileType>(args[1]->GetType());

    INTERNAL_CHECK(lhsType) << "The operator " << opName << " requires first argument to be a TileType, but got "
                            << args[0]->GetType()->TypeName();
    INTERNAL_CHECK(rhsType) << "The operator " << opName << " requires second argument to be a TileType, but got "
                            << args[1]->GetType()->TypeName();

    // Extract shapes
    const auto &lhsShape = lhsType->shape_;
    const auto &rhsShape = rhsType->shape_;

    // For batch matmul, we require at least 2D tiles
    INTERNAL_CHECK(lhsShape.size() >= 2) << "The operator " << opName
                                         << " requires lhs to have at least 2 dimensions, but got " << lhsShape.size()
                                         << " dimensions";
    INTERNAL_CHECK(rhsShape.size() >= 2) << "The operator " << opName
                                         << " requires rhs to have at least 2 dimensions, but got " << rhsShape.size()
                                         << " dimensions";

    size_t lhsNdim = lhsShape.size();
    size_t rhsNdim = rhsShape.size();

    // Extract matrix dimensions (last 2 dimensions)
    ExprPtr mDim = lhsShape[lhsNdim - 2];
    ExprPtr kDimLhs = lhsShape[lhsNdim - 1];
    ExprPtr kDimRhs = rhsShape[rhsNdim - 2];
    ExprPtr nDim = rhsShape[rhsNdim - 1];

    // Try to verify K dimensions match if they are constant
    VerifyKDimensionsMatch(kDimLhs, kDimRhs, opName);

    // Handle batch dimensions
    std::vector<ExprPtr> outputShape;

    if (lhsNdim == 2 && rhsNdim == 2) {
        // Simple 2D x 2D matrix multiplication: [M, K] @ [K, N] -> [M, N]
        outputShape = {mDim, nDim};
    } else {
        // Batch matrix multiplication
        // Extract batch dimensions (all except last 2)
        std::vector<ExprPtr> lhsBatch(lhsShape.begin(), lhsShape.end() - 2);
        std::vector<ExprPtr> rhsBatch(rhsShape.begin(), rhsShape.end() - 2);

        // Broadcast batch dimensions
        auto broadcastResult = BroadcastShapes(lhsBatch, rhsBatch);
        INTERNAL_CHECK(broadcastResult.success) << "Cannot broadcast batch dimensions for " << opName;

        outputShape = broadcastResult.shape;

        // Append matrix dimensions: [M, N]
        outputShape.push_back(mDim);
        outputShape.push_back(nDim);
    }

    // Promote data types
    auto resultDtype = PromoteDataTypes(lhsType->dtype_, rhsType->dtype_);
    INTERNAL_CHECK(resultDtype) << "The operator " << opName << " requires compatible data types, but got "
                                << lhsType->dtype_.ToString() << " and " << rhsType->dtype_.ToString();

    return std::make_shared<TileType>(outputShape, *resultDtype);
}

// ============================================================================
// Registration Function for Block Batch Matrix Multiplication Operations
// ============================================================================

REGISTER_OP("block.batch_matmul")
    .SetOpCategory("BlockOp")
    .SetDescription("Batch matrix multiplication of two tiles with broadcasting")
    .SetPipe(PipeType::M)
    .AddArgument("lhs", "Left-hand side tile (TileType, at least 2D)")
    .AddArgument("rhs", "Right-hand side tile (TileType, at least 2D)")
    .SetDeduceType([](const std::vector<ExprPtr> &args, const std::vector<std::pair<std::string, std::any>> &kwargs) {
        return DeduceBlockBatchMatMulType(args, kwargs, "block.batch_matmul");
    });

} // namespace ir
} // namespace pypto
