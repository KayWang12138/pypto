/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ir/verifier/verifier.h"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>

namespace pto {

VerifyResult Verifier::VerifyAllRules(const TileValue &tile) const {
    std::map<std::string, VerifyResult> results;
    bool allPassed = true;
    std::string errorMsg;

    // Check if verifier has any rules
    if (rules_.empty()) {
        return {VerifyStatus::FAIL, "No verification rules registered"};
    }

    // Execute all rules and collect results
    for (const auto &pair : rules_) {
        const std::string &ruleName = pair.first;
        const RuleFunc &ruleFunc = pair.second;
        VerifyResult result = ruleFunc(tile);
        results[ruleName] = result;
        if (!result.Passed()) {
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

    return {allPassed ? VerifyStatus::PASS : VerifyStatus::FAIL, errorMsg};
}

void Verifier::PrintVerificationTable(const std::map<std::string, VerifyResult> &results) const {
    // Calculate column widths
    size_t maxRuleNameLen = 0;
    size_t maxErrorMsgLen = 0;
    for (const auto &pair : results) {
        const std::string &ruleName = pair.first;
        const VerifyResult &result = pair.second;
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
    for (const auto &pair : results) {
        const std::string &ruleName = pair.first;
        const VerifyResult &result = pair.second;
        std::cout << std::left << std::setw(ruleNameWidth) << ruleName << " | " << std::setw(statusWidth)
                  << (result.Passed() ? "Passed" : "Failed") << " | " << std::setw(errorMsgWidth)
                  << (result.errorMsg.empty() ? "-" : result.errorMsg) << std::endl;
    }

    // Print table footer
    std::cout << std::string(ruleNameWidth + statusWidth + errorMsgWidth + 6, '=') << std::endl;
}

} // namespace pto
