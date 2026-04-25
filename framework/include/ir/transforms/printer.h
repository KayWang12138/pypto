/*
 * Copyright (c) PyPTO Contributors.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */
#pragma once
#include <string>

#include "ir/core.h"
#include "ir/expr.h"
#include "ir/type.h"

namespace pypto {
namespace ir {

enum class Precedence : int {
    kOr = 1,
    kXor = 2,
    kAnd = 3,
    kNot = 4,
    kComparison = 5,
    kBitOr = 6,
    kBitXor = 7,
    kBitAnd = 8,
    kBitShift = 9,
    kAddSub = 10,
    kMulDivMod = 11,
    kUnary = 12,
    kPow = 13,
    kCall = 14,
    kAtom = 15
};

Precedence GetPrecedence(const ExprPtr& expr);
bool IsRightAssociative(const ExprPtr& expr);

std::string PythonPrint(const IRNodePtr& node, const std::string& prefix = "pl", bool concise = false);
std::string PythonPrint(const TypePtr& type, const std::string& prefix = "pl");
std::string PythonDslPrint(const IRNodePtr& node, const std::string& prefix = "pl");
std::string PythonDslPrint(const TypePtr& type, const std::string& prefix = "pl");

} // namespace ir
} // namespace pypto
