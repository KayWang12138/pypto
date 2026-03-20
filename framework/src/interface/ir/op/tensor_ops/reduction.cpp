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

#include "core/any_cast.h"
#include "core/logging.h"
#include "ir/core.h"
#include "ir/kind_traits.h"
#include "ir/op_registry.h"
#include "ir/op_utils.h"
#include "ir/scalar_expr.h"
#include "ir/type.h"
#include "ir/type_inference.h"

namespace pypto {
namespace ir {

TypePtr DeduceTensorReductionType(const std::vector<ExprPtr> &args,
    const std::vector<std::pair<std::string, std::any>> &kwargs, const std::string &opName, const Span &span) {
    // Reduction operations require exactly 1 argument (input tensor)
    INTERNAL_CHECK(args.size() == 1) << "The operator " << opName << " requires exactly 1 argument, but got "
                                     << args.size() << " at " << span.ToString();

    // First argument must be TensorType
    auto tensorType = As<TensorType>(args[0]->GetType());
    INTERNAL_CHECK(tensorType) << "The operator " << opName << " requires first argument to be a TensorType, but got "
                               << args[0]->GetType()->TypeName() << " at " << span.ToString();

    const auto &inputShape = tensorType->shape_;
    int64_t inputNdim = static_cast<int64_t>(inputShape.size());

    // Extract axis from kwargs (default: -1, meaning last axis)
    int axis = GetKwarg<int>(kwargs, "axis", -1);

    // Normalize negative axis
    if (axis < 0) {
        axis = static_cast<int>(inputNdim) + axis;
    }
    INTERNAL_CHECK(axis >= 0 && static_cast<int64_t>(axis) < inputNdim)
        << "The operator " << opName << " axis " << axis << " is out of range for shape with " << inputNdim
        << " dimensions" << " at " << span.ToString();

    // Extract keep_dim flag from kwargs (default: true)
    bool keepDim = GetKwarg<bool>(kwargs, "keep_dim", true);

    // Build output shape
    std::vector<ExprPtr> outputShape;
    for (int64_t i = 0; i < inputNdim; ++i) {
        if (i == axis) {
            if (keepDim) {
                // Keep dimension as 1
                outputShape.push_back(std::make_shared<ConstInt>(1, DataType::INT32, Span::Unknown()));
            }
            // Otherwise, skip this dimension (reduce it out)
        } else {
            outputShape.push_back(inputShape[i]);
        }
    }

    // If output shape is empty (all dimensions reduced and keep_dim=false), return ScalarType
    if (outputShape.empty()) {
        return std::make_shared<ScalarType>(tensorType->dtype_);
    }

    return std::make_shared<TensorType>(outputShape, tensorType->dtype_);
}

// ============================================================================
// Registration Function for Tensor Reduction Operations
// ============================================================================

REGISTER_OP("tensor.row_max")
    .SetOpCategory("TensorOp")
    .SetDescription("Row-wise maximum reduction along specified axis")
    .AddArgument("input", "Input tensor (TensorType)")
    .set_attr<int>("axis")
    .set_attr<bool>("keep_dim")
    .SetDeduceType([](const std::vector<ExprPtr> &args, const std::vector<std::pair<std::string, std::any>> &kwargs,
        const Span &span) {
        return DeduceTensorReductionType(args, kwargs, "tensor.row_max", span);
    });

REGISTER_OP("tensor.row_sum")
    .SetOpCategory("TensorOp")
    .SetDescription("Row-wise sum reduction along specified axis")
    .AddArgument("input", "Input tensor (TensorType)")
    .set_attr<int>("axis")
    .set_attr<bool>("keep_dim")
    .SetDeduceType([](const std::vector<ExprPtr> &args, const std::vector<std::pair<std::string, std::any>> &kwargs,
        const Span &span) {
        return DeduceTensorReductionType(args, kwargs, "tensor.row_sum", span);
    });

} // namespace ir
} // namespace pypto
