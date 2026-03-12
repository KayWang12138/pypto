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

#include "ir/op_registry.h"

#include <memory>
#include <string>
#include <typeindex>
#include <unordered_map>
#include <utility>
#include <vector>

#include "core/dtype.h"
#include "core/logging.h"

namespace pypto {
namespace ir {

void ValidateKwargs(const std::vector<std::pair<std::string, std::any>> &kwargs,
    const std::unordered_map<std::string, std::type_index> &allowedKwargs, const std::string &opName) {
    for (const auto &[key, value] : kwargs) {
        auto it = allowedKwargs.find(key);
        if (it == allowedKwargs.end()) {
            throw ValueError("Unknown kwarg '" + key + "' for operator '" + opName + "'");
        }

        // For DataType, accept both DataType and int (since Python may pass as int for backward compatibility)
        if (it->second == std::type_index(typeid(DataType))) {
            std::type_index value_type(value.type());
            if (value_type != std::type_index(typeid(DataType)) && value_type != std::type_index(typeid(int))) {
                throw TypeError("Kwarg '" + key + "' for operator '" + opName +
                                "' expects DataType or int, but got incompatible type");
            }
        } else if (std::type_index(value.type()) != it->second) {
            throw TypeError("Kwarg '" + key + "' for operator '" + opName + "' has incompatible type");
        }
    }
}

OpRegistry &OpRegistry::GetInstance() {
    static OpRegistry instance;
    return instance;
}

OpRegistryEntry &OpRegistry::Register(const std::string &opName) {
    // Check if operator is already registered
    INTERNAL_CHECK(registry_.find(opName) == registry_.end()) << "Operator '" + opName + "' is already registered";

    // Create and insert the entry into the registry
    auto result = registry_.emplace(opName, OpRegistryEntry());
    auto &entry = result.first->second;
    entry.SetName(opName);

    // Create the operator instance with the operator name
    entry.op_ = std::make_shared<Op>(opName);

    return entry;
}

// ============================================================================
// OpRegistry Implementation
// ============================================================================

CallPtr OpRegistry::Create(const std::string &opName, const std::vector<ExprPtr> &args, Span span) const {
    // Call new version with empty kwargs for backward compatibility
    return Create(opName, args, {}, std::move(span));
}

CallPtr OpRegistry::Create(const std::string &opName, const std::vector<ExprPtr> &args,
    const std::vector<std::pair<std::string, std::any>> &kwargs, Span span) const {
    // Look up operator in registry
    auto it = registry_.find(opName);
    INTERNAL_CHECK(it != registry_.end()) << "Operator '" + opName + "' not found in registry";

    const auto &entry = it->second;

    // Get operator instance (shared definition)
    OpPtr op = entry.GetOp();

    // Validate kwargs against allowed attributes (stored in Op)
    if (!kwargs.empty()) {
        const auto &allowedKwargs = op->GetAttrs();
        if (!allowedKwargs.empty()) {
            ValidateKwargs(kwargs, allowedKwargs, opName);
        }
    }

    const auto &deduceTypeFn = entry.GetDeduceType();

    // Deduce result type (pass args and kwargs separately)
    TypePtr ResultType = deduceTypeFn(args, kwargs);
    INTERNAL_CHECK(ResultType) << "Type deduction failed for '" + opName + "'";

    // Create Call with deduced type
    return std::make_shared<Call>(op, args, kwargs, ResultType, std::move(span));
}

const OpRegistryEntry &OpRegistry::GetEntry(const std::string &opName) const {
    auto it = registry_.find(opName);
    INTERNAL_CHECK(it != registry_.end()) << "Operator '" + opName + "' not found in registry";
    return it->second;
}

OpPtr OpRegistry::GetOp(const std::string &opName) const {
    auto it = registry_.find(opName);
    INTERNAL_CHECK(it != registry_.end()) << "Operator '" + opName + "' not found in registry";
    return it->second.GetOp();
}

} // namespace ir
} // namespace pypto
