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

namespace {

DataType DeduceMatMulOutputDtype(const TensorTypePtr &lhsType, const TensorTypePtr &rhsType,
    const std::vector<std::pair<std::string, std::any>> &kwargs, const Span &span) {
    try {
        return GetKwarg<DataType>(kwargs, "out_dtype");
    } catch (const ValueError &e) {
        auto promoted = PromoteDataTypes(lhsType->dtype_, rhsType->dtype_);
        INTERNAL_CHECK(promoted) << "Cannot promote data types for tensor.matmul"
                                 << " at " << span.ToString();
        return *promoted;
    } catch (const TypeError &e) {
        throw TypeError("Invalid kwarg type for out_dtype: " + std::string(e.what()));
    }
}

std::vector<ExprPtr> DeduceMatMulOutputShape(const std::vector<ExprPtr> &lhsShape,
    const std::vector<ExprPtr> &rhsShape, bool aTrans, bool bTrans, const Span &span) {
    if (lhsShape.size() == 1 && rhsShape.size() == 1) {
        return {};
    } else if (lhsShape.size() == 2 && rhsShape.size() == 1) {
        return {lhsShape[0]};
    } else if (lhsShape.size() == 1 && rhsShape.size() == 2) {
        return {rhsShape[1]};
    } else if (lhsShape.size() == 2 && rhsShape.size() == 2) {
        ExprPtr mDim = aTrans ? lhsShape[1] : lhsShape[0];
        ExprPtr nDim = bTrans ? rhsShape[0] : rhsShape[1];
        return {mDim, nDim};
    }

    size_t lhsNdim = lhsShape.size();
    size_t rhsNdim = rhsShape.size();
    INTERNAL_CHECK(lhsNdim >= 2 && rhsNdim >= 2)
        << "tensor.matmul requires both tensors to have at least 2 dimensions "
        << "for batched matmul, but got lhs shape size " << lhsNdim << " and rhs shape size " << rhsNdim
        << " at " << span.ToString();

    std::vector<ExprPtr> lhsBatch(lhsShape.begin(), lhsShape.end() - 2);
    std::vector<ExprPtr> rhsBatch(rhsShape.begin(), rhsShape.end() - 2);
    auto broadcastResult = BroadcastShapes(lhsBatch, rhsBatch);
    INTERNAL_CHECK(broadcastResult.success) << "Cannot broadcast batch dimensions for tensor.matmul"
                                             << " at " << span.ToString();

    std::vector<ExprPtr> outputShape = broadcastResult.shape;
    ExprPtr mDim = aTrans ? lhsShape[lhsNdim - 1] : lhsShape[lhsNdim - 2];
    ExprPtr nDim = bTrans ? rhsShape[rhsNdim - 2] : rhsShape[rhsNdim - 1];
    outputShape.push_back(mDim);
    outputShape.push_back(nDim);
    return outputShape;
}

} // namespace

TypePtr DeduceTensorMatMulType(
    const std::vector<ExprPtr> &args, const std::vector<std::pair<std::string, std::any>> &kwargs,
    const Span &span) {
    INTERNAL_CHECK(args.size() == 2) << "tensor.matmul requires exactly 2 arguments (lhs, rhs), but got "
                                     << args.size() << " at " << span.ToString();

    auto lhsType = As<TensorType>(args[0]->GetType());
    auto rhsType = As<TensorType>(args[1]->GetType());
    INTERNAL_CHECK(lhsType) << "tensor.matmul requires first argument to be a TensorType, but got "
                            << args[0]->GetType()->TypeName() << " at " << span.ToString();
    INTERNAL_CHECK(rhsType) << "tensor.matmul requires second argument to be a TensorType, but got "
                            << args[1]->GetType()->TypeName() << " at " << span.ToString();

    INTERNAL_CHECK(lhsType->shape_.size() >= 1) << "tensor.matmul requires lhs to have at least 1 dimension"
                                                 << " at " << span.ToString();
    INTERNAL_CHECK(rhsType->shape_.size() >= 1) << "tensor.matmul requires rhs to have at least 1 dimension"
                                                 << " at " << span.ToString();

    DataType outDtype = DeduceMatMulOutputDtype(lhsType, rhsType, kwargs, span);
    bool aTrans = GetKwarg<bool>(kwargs, "a_trans", false);
    bool bTrans = GetKwarg<bool>(kwargs, "b_trans", false);
    auto outputShape = DeduceMatMulOutputShape(lhsType->shape_, rhsType->shape_, aTrans, bTrans, span);

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
