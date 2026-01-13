/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file verify.h
 * \brief TileValue verification rules and Verify class
 */

#pragma once

#include "ir/value.h"

#include <algorithm>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace pto {

/**
 * @brief Verify class for TileValue validation
 *
 * This class provides a rule-based verification system for TileValue objects.
 * Each verification rule is implemented as a function pointer stored in a map.
 */

struct VerifyResult {
    bool passed;
    std::string errorMsg;
};

// Individual rule implementations
inline VerifyResult RuleVerifyShapeNotEmpty(const TileValue &tile) {
    const auto &shape = tile.GetShape();
    if (shape.empty()) {
        return {false, "Tile shape is empty"};
    }
    return {true, ""};
}

inline VerifyResult RuleVerifyShapeDimensions(const TileValue &tile) {
    const auto &shape = tile.GetShape();
    if (std::any_of(shape.begin(), shape.end(), [](int64_t x) { return x <= 0; })) {
        return {false, "Tile shape contains invalid dimensions (<= 0)"};
    }
    return {true, ""};
}

inline VerifyResult RuleVerifyValidShapesSize(const TileValue &tile) {
    const auto &shape = tile.GetShape();
    const auto &validShapes = tile.GetValidShape();
    if (validShapes.size() != shape.size()) {
        return {false, "ValidShapes size (" + std::to_string(validShapes.size()) + ") does not match shape size (" +
                       std::to_string(shape.size()) + ")"};
    }
    return {true, ""};
}

inline VerifyResult RuleVerifyValidShapesValues(const TileValue &tile) {
    const auto &validShapes = tile.GetValidShape();
    for (size_t i = 0; i < validShapes.size(); ++i) {
        if (!validShapes[i]) {
            return {false, "ValidShape[" + std::to_string(i) + "] is nullptr"};
        }
        if (validShapes[i]->HasImmediateValue()) {
            int64_t val = validShapes[i]->GetInt64Value();
            if (val <= 0) {
                return {false, "ValidShape[" + std::to_string(i) + "] has invalid value (<= 0): " + std::to_string(val)};
            }
        }
    }
    return {true, ""};
}

inline VerifyResult RuleVerifySSAId(const TileValue &tile) {
    if (tile.GetID() <= 0) {
        return {false, "Invalid SSA ID: " + std::to_string(tile.GetID())};
    }
    return {true, ""};
}

inline VerifyResult RuleVerifySSAName(const TileValue &tile) {
    std::string ssaName = tile.GetSSAName();
    if (ssaName.empty()) {
        return {false, "SSA name is empty"};
    }
    return {true, ""};
}

inline VerifyResult RuleVerifyStridesSize(const TileValue &tile) {
    const auto &shape = tile.GetShape();
    const auto &strides = tile.GetStrides();
    if (!strides.empty() && strides.size() != shape.size()) {
        return {false, "Strides size (" + std::to_string(strides.size()) + ") does not match shape size (" +
                       std::to_string(shape.size()) + ")"};
    }
    return {true, ""};
}

class TileValueVerify {
public:
    using RuleFunc = std::function<VerifyResult(const TileValue &)>;

    TileValueVerify() = default;

    VerifyResult VerifyRule(const std::string &ruleName, const TileValue &tile) const {
        auto it = rules_.find(ruleName);
        if (it == rules_.end()) {
            return {false, "Unknown rule: " + ruleName};
        }
        return it->second(tile);
    }

    VerifyResult VerifyAllRules(const TileValue &tile) const {
        std::map<std::string, VerifyResult> results;
        bool allPassed = true;
        std::string errorMsg;

        // Execute all rules and collect results
        for (const auto &[ruleName, ruleFunc] : rules_) {
            VerifyResult result = ruleFunc(tile);
            results[ruleName] = result;
            if (!result.passed) {
                allPassed = false;
                if (errorMsg.empty()) {
                    errorMsg = "Rule '" + ruleName + "' failed: " + result.errorMsg;
                } else {
                    errorMsg += "; Rule '" + ruleName + "' failed: " + result.errorMsg;
                }
            }
        }

        // Print verification table
        PrintVerificationTable(results);

        return {allPassed, errorMsg};
    }

    std::vector<std::string> GetRuleNames() const {
        std::vector<std::string> names;
        names.reserve(rules_.size());
        for (const auto &[name, _] : rules_) {
            names.push_back(name);
        }
        return names;
    }

    void RegisterRule(const std::string &ruleName, RuleFunc ruleFunc) { rules_[ruleName] = ruleFunc; }

private:
    void PrintVerificationTable(const std::map<std::string, VerifyResult> &results) const {
        // Calculate column widths
        size_t maxRuleNameLen = 0;
        size_t maxErrorMsgLen = 0;
        for (const auto &[ruleName, result] : results) {
            maxRuleNameLen = std::max(maxRuleNameLen, ruleName.length());
            maxErrorMsgLen = std::max(maxErrorMsgLen, result.errorMsg.length());
        }

        // Set minimum column widths
        const size_t ruleNameWidth = std::max(maxRuleNameLen, size_t(20));
        const size_t statusWidth = 10; // "Passed/Failed"
        const size_t errorMsgWidth = std::max(maxErrorMsgLen, size_t(30));

        // Print table header
        std::cout << std::string(ruleNameWidth + statusWidth + errorMsgWidth + 6, '=') << std::endl;
        std::cout << std::left << std::setw(ruleNameWidth) << "Rule Name" << " | " << std::setw(statusWidth) << "Status"
                  << " | " << std::setw(errorMsgWidth) << "Error Message" << std::endl;
        std::cout << std::string(ruleNameWidth + statusWidth + errorMsgWidth + 6, '-') << std::endl;

        // Print table rows
        for (const auto &[ruleName, result] : results) {
            std::cout << std::left << std::setw(ruleNameWidth) << ruleName << " | " << std::setw(statusWidth)
                      << (result.passed ? "Passed" : "Failed") << " | " << std::setw(errorMsgWidth)
                      << (result.errorMsg.empty() ? "-" : result.errorMsg) << std::endl;
        }

        // Print table footer
        std::cout << std::string(ruleNameWidth + statusWidth + errorMsgWidth + 6, '=') << std::endl;
    }

    // Dictionary of verification rules: key is rule name, value is rule function
    std::map<std::string, RuleFunc> rules_;
};

} // namespace pto
