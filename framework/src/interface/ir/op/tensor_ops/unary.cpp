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
#include "ir/kind_traits.h"
#include "ir/op_registry.h"
#include "ir/type.h"
namespace pypto {
namespace ir {

TypePtr DeduceTensorExpType(
    const std::vector<ExprPtr> &args, const std::vector<std::pair<std::string, std::any>> & /*kwargs*/) {
    INTERNAL_CHECK(args.size() == 1) << "tensor.exp requires exactly 1 argument, but got " << args.size();

    auto tensorType = As<TensorType>(args[0]->GetType());
    INTERNAL_CHECK(tensorType) << "tensor.exp requires first argument to be a TensorType, but got "
                               << args[0]->GetType()->TypeName();

    // exp should promote to float type if input is integer
    // Exponential always produces floating-point output (e.g., exp(1) = 2.718...)
    DataType outDtype = tensorType->dtype_;
    if (!outDtype.IsFloat()) {
        // Promote to default float type (FP32)
        outDtype = DataType::FP32;
    }

    return std::make_shared<TensorType>(tensorType->shape_, outDtype);
}

TypePtr DeduceTensorCastType(
    const std::vector<ExprPtr> &args, const std::vector<std::pair<std::string, std::any>> &kwargs) {
    INTERNAL_CHECK(args.size() == 1) << "tensor.cast requires exactly 1 argument (input), but got " << args.size();

    auto tensorType = As<TensorType>(args[0]->GetType());
    INTERNAL_CHECK(tensorType) << "tensor.cast requires first argument to be a TensorType, but got "
                               << args[0]->GetType()->TypeName();

    // Read target_type from kwargs
    bool foundTargetType = false;
    DataType targetDtype;
    for (const auto &[key, value] : kwargs) {
        if (key == "target_type") {
            // Handle both DataType and int for backward compatibility
            if (value.type() == typeid(DataType)) {
                targetDtype = AnyCast<DataType>(value, "kwarg key: target_type");
            } else if (value.type() == typeid(int)) {
                targetDtype = static_cast<DataType>(AnyCast<int>(value, "kwarg key: target_type"));
            } else {
                throw TypeError("target_type must be a DataType or int, but got " + std::string(value.type().name()));
            }
            foundTargetType = true;
            break;
        }
    }
    INTERNAL_CHECK(foundTargetType) << "tensor.cast requires 'target_type' kwarg";

    // mode kwarg is optional, not used in type deduction

    // Cast preserves shape but changes dtype
    return std::make_shared<TensorType>(tensorType->shape_, targetDtype);
}

// ============================================================================
// Registration Function for Tensor Unary Operations
// ============================================================================

REGISTER_OP("tensor.exp")
    .SetOpCategory("TensorOp")
    .SetDescription("Element-wise exponential operation")
    .AddArgument("input", "Input tensor (TensorType)")
    .SetDeduceType([](const std::vector<ExprPtr> &args, const std::vector<std::pair<std::string, std::any>> &kwargs) {
        return DeduceTensorExpType(args, kwargs);
    });

REGISTER_OP("tensor.cast")
    .SetOpCategory("TensorOp")
    .SetDescription("Type casting operation")
    .AddArgument("input", "Input tensor (TensorType)")
    .set_attr<DataType>("target_type")
    .set_attr<int>("mode")
    .SetDeduceType([](const std::vector<ExprPtr> &args, const std::vector<std::pair<std::string, std::any>> &kwargs) {
        return DeduceTensorCastType(args, kwargs);
    });

} // namespace ir
} // namespace pypto
