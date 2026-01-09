/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file hash_computer.h
 * \brief IR hash computer using visitor pattern for bottom-up hash computation
 */

#pragma once

#include "ir/ir_visitor.h"
#include "ir/function.h"
#include "ir/statement.h"
#include "ir/operation.h"
#include "ir/value.h"

#include <cstdint>
#include <sstream>

namespace pto {

// Hash computer for IR nodes using bottom-up approach:
// Operation hash -> Statement hash (contains Operation hashes) -> Function hash (contains Statement hashes)
class IRHashComputer : public IRVisitor<IRHashComputer, uint64_t> {
public:

    // Visitor methods for Operations
    uint64_t VisitOperation(Operation& op);

    // Visitor methods for Statements
    uint64_t VisitCompoundStatement(CompoundStatement& cs);
    uint64_t VisitOpStatement(OpStatement& os);
    uint64_t VisitForStatement(ForStatement& fs);
    uint64_t VisitIfStatement(IfStatement& is);
    uint64_t VisitYieldStatement(YieldStatement& ys);
    uint64_t VisitReturnStatement(ReturnStatement& rs);

    // Visitor method for Function
    uint64_t VisitFunction(Function& func);

private:
    // Helper: Combine multiple hash values
    uint64_t CombineHashes(const std::vector<uint64_t>& hashes);

    // Helper: Hash a Value (for operand hashing)
    uint64_t HashValue(const ValuePtr& value);

    // Helper: Hash function signature
    uint64_t HashFunctionSignature(const FunctionSignature& signature);

    // Helper: Hash a int
    std::hash<int> intHasher_;
    // Helper: Hash a string
    std::hash<std::string> stringHasher_;
};

} // namespace pto
