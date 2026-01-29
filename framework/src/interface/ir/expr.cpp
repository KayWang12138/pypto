/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

#include "ir/expr.h"

#include <sstream>
#include <stdexcept>
#include <utility>

// #include "core/error.h"
#include "ir/type.h"

// Temporary replacement for CHECK - depends on core/error.h which is temporarily removed
#ifndef CHECK
#define CHECK(expr) \
  if (!(expr)) throw std::logic_error(std::string("Check failed: " #expr " at ") + __FILE__ + ":" + std::to_string(__LINE__)); \
  std::ostringstream() /* Allow chaining with << */
#endif

namespace pypto {
namespace ir {

TupleGetItemExpr::TupleGetItemExpr(ExprPtr tuple, int index, Span span)
    : Expr(std::move(span)), tuple_(std::move(tuple)), index_(index) {
  // Type checking: tuple must have TupleType
  auto tuple_type = std::dynamic_pointer_cast<const TupleType>(tuple_->GetType());
  CHECK(tuple_type) << "TupleGetItemExpr requires tuple to have TupleType, got "
                    << tuple_->GetType()->TypeName();

  // Bounds checking
  CHECK(index >= 0 && index < static_cast<int>(tuple_type->types_.size()))
      << "TupleGetItemExpr index " << index << " out of bounds for tuple with " << tuple_type->types_.size()
      << " elements";

  // Set result type to the accessed element's type
  type_ = tuple_type->types_[index];
}

}  // namespace ir
}  // namespace pypto
