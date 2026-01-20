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

#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <vector>

namespace pto {

enum class VerifyStatus : uint32_t {
    PASS = 0,
    FAIL = 1
};

struct VerifyResult {
    VerifyStatus status;
    std::string errorMsg;
    bool Passed() const { return status == VerifyStatus::PASS; }
};

class Verifier {
public:
    using RuleFunc = std::function<VerifyResult(const TileValue &)>;

    Verifier() = default;

    VerifyResult VerifyRule(const std::string &ruleName, const TileValue &tile) const {
        // Boundary check: ensure rule exists
        auto it = rules_.find(ruleName);
        if (it == rules_.end()) {
            return {VerifyStatus::FAIL, "Unknown rule: " + ruleName};
        }
        return it->second(tile);
    }

    VerifyResult VerifyAllRules(const TileValue &tile) const;

    std::vector<std::string> GetRuleNames() const {
        std::vector<std::string> names;
        names.reserve(rules_.size());
        for (const auto &pair : rules_) {
            names.push_back(pair.first);
        }
        return names;
    }

    void RegisterRule(const std::string &ruleName, RuleFunc ruleFunc) {
        rules_[ruleName] = ruleFunc;
    }

    void RegisterRules(const std::map<std::string, RuleFunc> &rules) {
        for (const auto &pair : rules) {
            RegisterRule(pair.first, pair.second);
        }
    }

    bool HasRule(const std::string &ruleName) const {
        return rules_.find(ruleName) != rules_.end();
    }

    bool RemoveRule(const std::string &ruleName) {
        return rules_.erase(ruleName) > 0;
    }

    void ClearRules() {
        rules_.clear();
    }

    size_t GetRuleCount() const {
        return rules_.size();
    }

    bool IsEmpty() const {
        return rules_.empty();
    }

private:
    void PrintVerificationTable(const std::map<std::string, VerifyResult> &results) const;
    std::map<std::string, RuleFunc> rules_;
};

} // namespace pto
