/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once
#include <any>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "core/any_cast.h"
#include "core/error.h"
#include "core/logging.h"
#include "ir/kind_traits.h"
#include "ir/scalar_expr.h"

namespace pypto {
namespace ir {

/**
 * \brief Get a keyword argument value from kwargs vector with optional default
 *
 * \tparam T The expected value type
 * \param kwargs The keyword arguments vector
 * \param key The key to look up
 * \param defaultValue Optional default value if key is not found
 * \return The value associated with the key, or default_value if not found
 * \throws ValueError if key is not found and no default_value is provided
 */
template <typename T>
T GetKwarg(const std::vector<std::pair<std::string, std::any>> &kwargs, const std::string &key,
    const std::optional<T> &defaultValue = std::nullopt) {
    for (const auto &entry : kwargs) {
        if (entry.first == key) {
            return AnyCast<T>(entry.second, "kwarg key: " + key);
        }
    }
    if (defaultValue) {
        return *defaultValue;
    }
    throw ValueError("Missing kwarg: " + key);
}

/**
 * \brief Normalize axis index to handle negative indexing
 *
 * Converts negative axis indices to positive ones and validates the range.
 *
 * \param axis The axis index (can be negative)
 * \param ndim The number of dimensions
 * \return The normalized (non-negative) axis index
 */
inline int NormalizeAxis(int axis, size_t ndim) {
    if (axis < 0) {
        axis += static_cast<int>(ndim);
    }
    INTERNAL_CHECK(axis >= 0 && axis < static_cast<int>(ndim))
        << "Axis " << axis << " is out of range for " << ndim << "D shape";
    return axis;
}

/**
 * \brief Compute the product of shape dimensions (for static shapes)
 *
 * \param shape The shape dimensions
 * \return The product if all dimensions are ConstInt, -1 otherwise (dynamic shape)
 */
inline int64_t ComputeShapeProduct(const std::vector<ExprPtr> &shape) {
    int64_t product = 1;
    for (const auto &dim : shape) {
        auto constDim = As<ConstInt>(dim);
        if (!constDim) {
            return -1; // Dynamic shape, cannot compute product
        }
        product *= constDim->value_;
    }
    return product;
}

/**
 * \brief Verify that two K-dimensions match (for matmul operations)
 *
 * If both dimensions are ConstInt, checks that their values are equal.
 * If either is dynamic, the check is skipped (deferred to runtime).
 *
 * \param kLhs The K dimension from the left-hand side
 * \param kRhs The K dimension from the right-hand side
 * \param opName Operator name for error messages
 */
inline void VerifyKDimensionsMatch(const ExprPtr &kLhs, const ExprPtr &kRhs, const std::string &opName) {
    auto kLhsConst = As<ConstInt>(kLhs);
    auto kRhsConst = As<ConstInt>(kRhs);
    if (kLhsConst && kRhsConst) {
        INTERNAL_CHECK(kLhsConst->value_ == kRhsConst->value_)
            << "The operator " << opName << " requires matching inner dimensions, but got lhs K=" << kLhsConst->value_
            << " and rhs K=" << kRhsConst->value_;
    }
}

} // namespace ir
} // namespace pypto
