/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * Temporary stub for IR program.h - waiting for new IR integration
 * This file provides minimal interface to keep compilation working
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
