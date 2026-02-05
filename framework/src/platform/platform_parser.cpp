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
 * \file platform_parser.cpp
 * \brief
 */

#include <limits>
#include <fstream>
#include "tilefwk/file.h"
#include "tilefwk/platform.h"

#ifdef BUILD_WITH_CANN
#include "runtime/rt.h"
#endif

namespace npu {
namespace tile_fwk {
const uint32_t kMaxLength = 50;
const std::string platformConfigEnv = "PLATFORM_CONFIG_PATH";
const std::string version = "version";
const std::string instrinsicMap = "AICoreintrinsicDtypeMap";
const std::string aic = "AIC";
const std::string aiv = "AIV";
const std::string aicVersion = "AIC_version";
const std::string aivVersion = "AIV_version";
const std::string ccecAicVersion = "CCEC_AIC_version";
const std::string ccecAivVersion = "CCEC_AIV_version";
const std::string ccecCubeVersion = "CCEC_CUBE_version";
const std::string ccecVectorVersion = "CCEC_VECTOR_version";

bool PlatformParser::FilterCCECVersion(const std::string& key, std::string &coreType) const {
    const std::string prefix = "CCEC_";
    const std::string suffix = "_version";
    const size_t prefixLen = prefix.length();
    const size_t suffixLen = suffix.length();
    if (key.length() >= (prefixLen + suffixLen) &&
        key.substr(0, prefixLen) == prefix &&
        key.substr(key.length() - suffixLen) == suffix) {
        coreType = key.substr(prefixLen, key.length() - prefixLen - suffixLen);
        return true;
    } else {
        return false;
    }
}

bool PlatformParser::GetSizeVal(const std::string& column, const std::string& key, size_t& val) const {
    std::string valStr;
    const size_t max_size_t = std::numeric_limits<size_t>::max();
    if (!GetStringVal(column, key, valStr)) {
        return false;
    }
    val = 0UL;

    constexpr int    kRadix10    = 10;
    constexpr int    kMaxDigit10 = kRadix10 - 1;

    for (const char &c : valStr) {
        int digit = c - '0';
        if (digit < 0 || digit > kMaxDigit10) {
            return false;
        }
        if (val > (max_size_t - digit) / kRadix10) {
            return false;
        }
        val = val * kRadix10 + digit;
    }
    return true;
}

bool PlatformParser::GetCCECVersion(std::unordered_map<std::string, std::string>& ccecVersion) const {
    const std::vector<std::string> ccecVersions = {ccecAicVersion, ccecAivVersion, ccecCubeVersion, ccecVectorVersion};
    ccecVersion.clear();
    std::string coreType;
    std::string versionVal;
    for (const auto &curVersion : ccecVersions) {
        if (FilterCCECVersion(curVersion, coreType) && GetStringVal(version, curVersion, versionVal)) {
            ccecVersion[coreType] = versionVal;
        }
    }
    return true;
}

bool PlatformParser::GetCoreVersion(std::unordered_map<std::string, std::string>& curVersion) const {
    curVersion.clear();
    std::string versionVal;
    if (GetStringVal(version, aicVersion, versionVal)) {
        curVersion[aic] = versionVal;
    }
    if (GetStringVal(version, aivVersion, versionVal)) {
        curVersion[aiv] = versionVal;
    }
    return true;
}

bool INIParser::Initialize(const std::string &iniFilePath) {
    if (!ReadINIFile(iniFilePath)) {
        return false;
    }
    return true;
}

bool INIParser::ReadINIFile(const std::string& filepath) {
    data_.clear();
    std::ifstream file(filepath);
    if (!file.is_open()) {
        return false;
    }
    std::string line;
    std::string trimLine;
    std::string section;
    while (std::getline(file, line)) {
        trimLine = trim(line);
        if (trimLine.empty()) {
            continue;
        }
        if (trimLine.front() == '[' && trimLine.back() == ']') {
            constexpr size_t kLeftBracketLen  = std::char_traits<char>::length("[");
            constexpr size_t kRightBracketLen = std::char_traits<char>::length("]");
            if (trimLine.size() <= kLeftBracketLen + kRightBracketLen) {
                continue;
            }
            section = trimLine.substr(kLeftBracketLen, trimLine.size() - kLeftBracketLen - kRightBracketLen);
            continue;
        }
        size_t equalPos = trimLine.find('=');
        if (equalPos == std::string::npos) {
            continue;
        }
        std::string key = trimLine.substr(0, equalPos);
        std::string value = trimLine.substr(equalPos + 1);
        if (key.empty()) {
            continue;
        }
        data_[section][key] = value;
    }
    file.close();
    return true;
}

bool INIParser::GetStringVal(const std::string& column, const std::string& key, std::string& val) const {
    val.clear();
    if (data_.find(column) == data_.end()) {
        return false;
    }
    auto value = data_.at(column);
    if (value.find(key) == value.end()) {
        return true;
    }
    val = value[key];
    return true;
}

bool CmdParser::GetStringVal(const std::string& column, const std::string& key, std::string& val) const {
    char charVal[kMaxLength] = {0};
    if (rtGetSocSpec(column.c_str(), key.c_str(), charVal, kMaxLength) == 0) {
        val = std::string(charVal);
        return true;
    }
    return false;
}

}  // namespace tile_fwk
}  // namespace npu