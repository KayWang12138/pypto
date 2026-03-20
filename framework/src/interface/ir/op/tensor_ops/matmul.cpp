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
#include <optional>
#include <string>
#include <vector>

#include "core/any_cast.h"
#include "core/dtype.h"
#include "core/error.h"
#include "core/logging.h"
#include "ir/core.h"
#include "ir/kind_traits.h"
#include "ir/op_registry.h"
#include "ir/op_utils.h"
#include "ir/type.h"
#include "ir/type_inference.h"

namespace pypto {
namespace ir {

TypePtr DeduceTensorMatMulType(
    const std::vector<ExprPtr> &args, const std::vector<std::pair<std::string, std::any>> &kwargs,
    const Span &span) {
    // tensor.matmul requires exactly 2 Expr arguments (lhs, rhs)
    INTERNAL_CHECK(args.size() == 2) << "tensor.matmul requires exactly 2 arguments (lhs, rhs), but got "
                                     << args.size() << " at " << span.ToString();

    // First two arguments must be TensorType
    auto lhsType = As<TensorType>(args[0]->GetType());
    auto rhsType = As<TensorType>(args[1]->GetType());

    INTERNAL_CHECK(lhsType) << "tensor.matmul requires first argument to be a TensorType, but got "
                            << args[0]->GetType()->TypeName() << " at " << span.ToString();
    INTERNAL_CHECK(rhsType) << "tensor.matmul requires second argument to be a TensorType, but got "
                            << args[1]->GetType()->TypeName() << " at " << span.ToString();

    // Extract shapes
    const auto &lhsShape = lhsType->shape_;
    const auto &rhsShape = rhsType->shape_;

    INTERNAL_CHECK(lhsShape.size() >= 1) << "tensor.matmul requires lhs to have at least 1 dimension"
                                         << " at " << span.ToString();
    INTERNAL_CHECK(rhsShape.size() >= 1) << "tensor.matmul requires rhs to have at least 1 dimension"
                                         << " at " << span.ToString();

    // Read kwargs (with defaults)
    DataType outDtype;
    try {
        outDtype = GetKwarg<DataType>(kwargs, "out_dtype");
    } catch (const ValueError &e) {
        auto promoted = PromoteDataTypes(lhsType->dtype_, rhsType->dtype_);
        INTERNAL_CHECK(promoted) << "Cannot promote data types for tensor.matmul"
                                 << " at " << span.ToString();
        outDtype = *promoted;
    } catch (const TypeError &e) {
        throw TypeError("Invalid kwarg type for out_dtype: " + std::string(e.what()));
    }

    bool aTrans = GetKwarg<bool>(kwargs, "a_trans", false);
    bool bTrans = GetKwarg<bool>(kwargs, "b_trans", false);

    // Compute output shape based on transpose flags
    // For 2D: lhs [M, K] x rhs [K, N] -> [M, N]
    // With transpose: lhs [K, M]^T x rhs [N, K]^T -> [M, N]

    std::vector<ExprPtr> outputShape;

    if (lhsShape.size() == 1 && rhsShape.size() == 1) {
        // Vector x vector (dot product): [K] x [K] -> scalar (0D tensor)
        outputShape = {};
    } else if (lhsShape.size() == 2 && rhsShape.size() == 1) {
        // Matrix x vector: [M, K] x [K] -> [M]
        outputShape = {lhsShape[0]};
    } else if (lhsShape.size() == 1 && rhsShape.size() == 2) {
        // Vector x matrix: [K] x [K, N] -> [N]
        outputShape = {rhsShape[1]};
    } else if (lhsShape.size() == 2 && rhsShape.size() == 2) {
        // 2D x 2D matrix multiplication
        ExprPtr mDim = aTrans ? lhsShape[1] : lhsShape[0];
        ExprPtr nDim = bTrans ? rhsShape[0] : rhsShape[1];
        outputShape = {mDim, nDim};
    } else {
        // For higher-dimensional tensors (both must have at least 2 dimensions),
        // use batched matmul semantics
        size_t lhsNdim = lhsShape.size();
        size_t rhsNdim = rhsShape.size();

        // Ensure both tensors have at least 2 dimensions for batched matmul
        INTERNAL_CHECK(lhsNdim >= 2 && rhsNdim >= 2)
            << "tensor.matmul requires both tensors to have at least 2 dimensions "
            << "for batched matmul, but got lhs shape size " << lhsNdim << " and rhs shape size " << rhsNdim
            << " at " << span.ToString();

        // Extract batch dimensions (all except last 2)
        std::vector<ExprPtr> lhsBatch(lhsShape.begin(), lhsShape.end() - 2);
        std::vector<ExprPtr> rhsBatch(rhsShape.begin(), rhsShape.end() - 2);

        // Broadcast batch dimensions
        auto broadcastResult = BroadcastShapes(lhsBatch, rhsBatch);
        INTERNAL_CHECK(broadcastResult.success) << "Cannot broadcast batch dimensions for tensor.matmul"
                                                 << " at " << span.ToString();

        outputShape = broadcastResult.shape;

        // Append matrix dimensions
        ExprPtr mDim = aTrans ? lhsShape[lhsNdim - 1] : lhsShape[lhsNdim - 2];
        ExprPtr nDim = bTrans ? rhsShape[rhsNdim - 2] : rhsShape[rhsNdim - 1];
        outputShape.push_back(mDim);
        outputShape.push_back(nDim);
    }

    return std::make_shared<TensorType>(outputShape, outDtype);
}

// ============================================================================
// Registration Function for Tensor Matrix Multiplication Operations
// ============================================================================

REGISTER_OP("tensor.matmul")
    .SetOpCategory("TensorOp")
    .SetDescription("Matrix multiplication of two tensors with optional transpose")
    .AddArgument("lhs", "Left-hand side tensor (TensorType)")
    .AddArgument("rhs", "Right-hand side tensor (TensorType)")
    .set_attr<DataType>("out_dtype")
    .set_attr<bool>("a_trans")
    .set_attr<bool>("b_trans")
    .set_attr<bool>("c_matrix_nz")
    .SetDeduceType([](const std::vector<ExprPtr> &args, const std::vector<std::pair<std::string, std::any>> &kwargs,
        const Span &span) {
        return DeduceTensorMatMulType(args, kwargs, span);
    });

} // namespace ir
} // namespace pypto
