/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ir/expr.h"

#include <memory>
#include <utility>
#include <vector>

#include "core/error.h"
#include "core/logging.h"
#include "ir/kind_traits.h"
#include "ir/type.h"

namespace pypto {
namespace ir {

MakeTuple::MakeTuple(std::vector<ExprPtr> elements, Span span) : Expr(std::move(span)), elements_(std::move(elements))
{
    // Collect types from all element expressions
    std::vector<TypePtr> elementTypes;
    elementTypes.reserve(elements_.size());
    for (const auto& elem : elements_) {
        elementTypes.push_back(elem->GetType());
    }

    // Set result type to TupleType
    type_ = std::make_shared<TupleType>(std::move(elementTypes));
}

TupleGetItemExpr::TupleGetItemExpr(ExprPtr tuple, int index, Span span)
    : Expr(std::move(span)), tuple_(std::move(tuple)), index_(index)
{
    // Type checking: tuple must have TupleType
    auto tupleType = As<TupleType>(tuple_->GetType());
    INTERNAL_CHECK(tupleType) << "TupleGetItemExpr requires tuple to have TupleType, got "
                              << tuple_->GetType()->TypeName() << " at " << span_.ToString();

    // Bounds checking
    INTERNAL_CHECK(index >= 0 && index < static_cast<int>(tupleType->types_.size()))
        << "TupleGetItemExpr index " << index << " out of bounds for tuple with " << tupleType->types_.size()
        << " elements at " << span_.ToString();

    // Set result type to the accessed element's type
    type_ = tupleType->types_[index];
}

} // namespace ir
} // namespace pypto
