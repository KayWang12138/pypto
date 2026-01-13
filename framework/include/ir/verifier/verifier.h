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
 * \file verifier/verifier.h
 * \brief TileValue verification rules and Verifier class
 */

#pragma once

#include "ir/value.h"

#include <functional>
#include <map>
#include <string>
#include <vector>

namespace pto {

/**
 * @brief Result of a verification operation
 */
struct VerifyResult {
    bool passed;
    std::string errorMsg;
};

/**
 * @brief Verify class for TileValue validation
 *
 * This class provides a rule-based verification system for TileValue objects.
 * Each verification rule is implemented as a function pointer stored in a map.
 */
class Verifier {
public:
    using RuleFunc = std::function<VerifyResult(const TileValue &)>;

    Verifier() = default;

    VerifyResult VerifyRule(const std::string &ruleName, const TileValue &tile) const {
        auto it = rules_.find(ruleName);
        if (it == rules_.end()) {
            return {false, "Unknown rule: " + ruleName};
        }
        return it->second(tile);
    }

    VerifyResult VerifyAllRules(const TileValue &tile) const;

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
    void PrintVerificationTable(const std::map<std::string, VerifyResult> &results) const;

    // Dictionary of verification rules: key is rule name, value is rule function
    std::map<std::string, RuleFunc> rules_;
};

} // namespace pto
