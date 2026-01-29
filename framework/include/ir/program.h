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

#include <memory>
#include <string>
#include <vector>
#include <ostream>

namespace pto {

// Forward declaration
class Function;
class BlockFunction;

/**
 * @brief Temporary stub for ProgramModule
 * This is a minimal placeholder to maintain compilation compatibility
 * Will be replaced with actual implementation from new IR
 */
class ProgramModule {
public:
    explicit ProgramModule(std::string name) : name_(std::move(name)) {}

    const std::string& GetName() const { return name_; }

    // Stub methods - will be replaced with actual implementation
    std::vector<std::shared_ptr<Function>> GetFunctions() const {
        return functions_;
    }

    void AddFunction(std::shared_ptr<Function> func) {
        functions_.push_back(func);
    }

    // Stub print operator
    friend std::ostream& operator<<(std::ostream& os, const ProgramModule& pm) {
        os << "ProgramModule(stub): " << pm.name_;
        return os;
    }

private:
    std::string name_;
    std::vector<std::shared_ptr<Function>> functions_;
};

using ProgramModulePtr = std::shared_ptr<ProgramModule>;

} // namespace pto
