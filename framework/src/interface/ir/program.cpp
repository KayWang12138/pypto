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

#include "ir/program.h"

#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "core/logging.h"
#include "ir/expr.h"
#include "ir/function.h"

namespace pypto {
namespace ir {

// Vector-based constructor: creates GlobalVars from function names
Program::Program(const std::vector<FunctionPtr> &functions, std::string name, Span span)
    : IRNode(std::move(span)), name_(std::move(name)) {
    // Create a map and populate it with GlobalVar -> Function mappings
    // The map automatically sorts by GlobalVar name via the GlobalVarPtrLess comparator
    std::set<std::string> functionNames;
    for (const auto &func : functions) {
        INTERNAL_CHECK(func) << "Program constructor encountered null function at " << span_.ToString();
        auto funcName = func->name_;
        INTERNAL_CHECK(!funcName.empty()) << "Program constructor encountered empty function name at "
                                          << span_.ToString();
        INTERNAL_CHECK(functionNames.find(funcName) == functionNames.end())
            << "Duplicate function name \"" << funcName << "\" at " << span_.ToString();
        functionNames.insert(funcName);
        auto globalVar = std::make_shared<const GlobalVar>(funcName);
        functions_.emplace(globalVar, func);
    }
}

FunctionPtr Program::GetFunction(const std::string &name) const {
    auto it = functions_.find(std::make_shared<const GlobalVar>(name));
    if (it != functions_.end()) {
        return it->second;
    }
    return nullptr;
}

GlobalVarPtr Program::GetGlobalVar(const std::string &name) const {
    auto it = functions_.find(std::make_shared<const GlobalVar>(name));
    if (it != functions_.end()) {
        return it->first;
    }
    return nullptr;
}

} // namespace ir
} // namespace pypto
