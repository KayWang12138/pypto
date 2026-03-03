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

TypePtr DeduceTensorOpElementwiseBinaryType(const std::vector<ExprPtr> &args,
    const std::vector<std::pair<std::string, std::any>> & /*kwargs*/, const std::string &opName) {
    INTERNAL_CHECK(args.size() == 2) << "The operator " << opName << " requires exactly 2 arguments, but got "
                                     << args.size();

    // Try TensorType first
    auto tensorType1 = As<TensorType>(args[0]->GetType());
    auto tensorType2 = As<TensorType>(args[1]->GetType());

    INTERNAL_CHECK(tensorType1) << "The operator " << opName << " requires first argument to be a TensorType, but got "
                                << args[0]->GetType()->TypeName();
    INTERNAL_CHECK(tensorType2) << "The operator " << opName << " requires second argument to be a TensorType, but got "
                                << args[1]->GetType()->TypeName();

    auto resultDtype = PromoteDataTypes(tensorType1->dtype_, tensorType2->dtype_);
    INTERNAL_CHECK(resultDtype) << "The operator " << opName << " requires compatible data types, but got "
                                << args[0]->GetType()->TypeName() << " and " << args[1]->GetType()->TypeName();

    auto broadcastResult = BroadcastShapes(tensorType1->shape_, tensorType2->shape_);
    INTERNAL_CHECK(broadcastResult.success)
        << "The operator " << opName << " requires compatible shapes, but got " << FormatShape(tensorType1->shape_)
        << " and " << FormatShape(tensorType2->shape_);

    return std::make_shared<TensorType>(broadcastResult.shape, *resultDtype);
}

TypePtr DeduceTensorOpElementwiseScalarType(const std::vector<ExprPtr> &args,
    const std::vector<std::pair<std::string, std::any>> & /*kwargs*/, const std::string &opName) {
    INTERNAL_CHECK(args.size() == 2) << "The operator " << opName << " requires exactly 2 arguments, but got "
                                     << args.size();

    auto tensorType1 = As<TensorType>(args[0]->GetType());
    auto scalarType2 = As<ScalarType>(args[1]->GetType());

    INTERNAL_CHECK(tensorType1) << "The operator " << opName << " requires first argument to be a TensorType, but got "
                                << args[0]->GetType()->TypeName();
    INTERNAL_CHECK(scalarType2) << "The operator " << opName << " requires second argument to be a ScalarType, but got "
                                << args[1]->GetType()->TypeName();

    // TensorType + ScalarType - result is TensorType with same shape as first argument
    auto resultDtype = PromoteDataTypes(tensorType1->dtype_, scalarType2->dtype_);
    INTERNAL_CHECK(resultDtype) << "The operator " << opName << " requires compatible data types, but got "
                                << args[0]->GetType()->TypeName() << " and " << args[1]->GetType()->TypeName();

    return std::make_shared<TensorType>(tensorType1->shape_, *resultDtype);
}

// ============================================================================
// Registration Function for Tensor Element-wise Operations
// ============================================================================

REGISTER_OP("tensor.add")
    .SetOpCategory("TensorOp")
    .SetDescription("Element-wise addition of two tensors with broadcasting")
    .AddArgument("lhs", "Left-hand side tensor (TensorType)")
    .AddArgument("rhs", "Right-hand side tensor (TensorType)")
    .SetDeduceType([](const std::vector<ExprPtr> &args, const std::vector<std::pair<std::string, std::any>> &kwargs) {
        return DeduceTensorOpElementwiseBinaryType(args, kwargs, "tensor.add");
    });

REGISTER_OP("tensor.add_scalar")
    .SetOpCategory("TensorOp")
    .SetDescription("Element-wise addition of tensor and scalar")
    .AddArgument("lhs", "Left-hand side tensor (TensorType)")
    .AddArgument("rhs", "Right-hand side scalar (ScalarType)")
    .SetDeduceType([](const std::vector<ExprPtr> &args, const std::vector<std::pair<std::string, std::any>> &kwargs) {
        return DeduceTensorOpElementwiseScalarType(args, kwargs, "tensor.add_scalar");
    });

REGISTER_OP("tensor.sub")
    .SetOpCategory("TensorOp")
    .SetDescription("Element-wise subtraction of two tensors with broadcasting")
    .AddArgument("lhs", "Left-hand side tensor (TensorType)")
    .AddArgument("rhs", "Right-hand side tensor (TensorType)")
    .SetDeduceType([](const std::vector<ExprPtr> &args, const std::vector<std::pair<std::string, std::any>> &kwargs) {
        return DeduceTensorOpElementwiseBinaryType(args, kwargs, "tensor.sub");
    });

REGISTER_OP("tensor.sub_scalar")
    .SetOpCategory("TensorOp")
    .SetDescription("Element-wise subtraction of tensor and scalar")
    .AddArgument("lhs", "Left-hand side tensor (TensorType)")
    .AddArgument("rhs", "Right-hand side scalar (ScalarType)")
    .SetDeduceType([](const std::vector<ExprPtr> &args, const std::vector<std::pair<std::string, std::any>> &kwargs) {
        return DeduceTensorOpElementwiseScalarType(args, kwargs, "tensor.sub_scalar");
    });

REGISTER_OP("tensor.mul")
    .SetOpCategory("TensorOp")
    .SetDescription("Element-wise multiplication of two tensors with broadcasting")
    .AddArgument("lhs", "Left-hand side tensor (TensorType)")
    .AddArgument("rhs", "Right-hand side tensor (TensorType)")
    .SetDeduceType([](const std::vector<ExprPtr> &args, const std::vector<std::pair<std::string, std::any>> &kwargs) {
        return DeduceTensorOpElementwiseBinaryType(args, kwargs, "tensor.mul");
    });

REGISTER_OP("tensor.mul_scalar")
    .SetOpCategory("TensorOp")
    .SetDescription("Element-wise multiplication of tensor and scalar")
    .AddArgument("lhs", "Left-hand side tensor (TensorType)")
    .AddArgument("rhs", "Right-hand side scalar (ScalarType)")
    .SetDeduceType([](const std::vector<ExprPtr> &args, const std::vector<std::pair<std::string, std::any>> &kwargs) {
        return DeduceTensorOpElementwiseScalarType(args, kwargs, "tensor.mul_scalar");
    });

REGISTER_OP("tensor.div")
    .SetOpCategory("TensorOp")
    .SetDescription("Element-wise division of two tensors with broadcasting")
    .AddArgument("lhs", "Left-hand side tensor (TensorType)")
    .AddArgument("rhs", "Right-hand side tensor (TensorType)")
    .SetDeduceType([](const std::vector<ExprPtr> &args, const std::vector<std::pair<std::string, std::any>> &kwargs) {
        return DeduceTensorOpElementwiseBinaryType(args, kwargs, "tensor.div");
    });

REGISTER_OP("tensor.div_scalar")
    .SetOpCategory("TensorOp")
    .SetDescription("Element-wise division of tensor and scalar")
    .AddArgument("lhs", "Left-hand side tensor (TensorType)")
    .AddArgument("rhs", "Right-hand side scalar (ScalarType)")
    .SetDeduceType([](const std::vector<ExprPtr> &args, const std::vector<std::pair<std::string, std::any>> &kwargs) {
        return DeduceTensorOpElementwiseScalarType(args, kwargs, "tensor.div_scalar");
    });

REGISTER_OP("tensor.maximum")
    .SetOpCategory("TensorOp")
    .SetDescription("Element-wise maximum of two tensors with broadcasting")
    .AddArgument("lhs", "Left-hand side tensor (TensorType)")
    .AddArgument("rhs", "Right-hand side tensor (TensorType)")
    .SetDeduceType([](const std::vector<ExprPtr> &args, const std::vector<std::pair<std::string, std::any>> &kwargs) {
        return DeduceTensorOpElementwiseBinaryType(args, kwargs, "tensor.maximum");
    });

} // namespace ir
} // namespace pypto
