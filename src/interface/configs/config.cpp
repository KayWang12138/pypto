/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file config.cpp
 * \brief
 */

#include "interface/inner/config.h"
#include "interface/utils/string_utils.h"

namespace npu::tile_fwk {
using ConfigType = std::variant<bool, int64_t, std::string, std::vector<int64_t>>;
static FunctionType g_funcType{FunctionType::DYNAMIC};
static std::unordered_map<std::string, ConfigType> g_optionsMap;

void Config::SetOption(const std::string &key, bool value) {
    auto iter = g_optionsMap.find(key);
    if (g_optionsMap.find(key) == g_optionsMap.end()) {
        throw std::runtime_error("Config key not found: " + key);
    }
    iter->second = value;
}

void Config::SetOption(const std::string &key, int64_t value) {
    auto iter = g_optionsMap.find(key);
    if (g_optionsMap.find(key) == g_optionsMap.end()) {
        throw std::runtime_error("Config key not found: " + key);
    }
    iter->second = value;
}

void Config::SetOption(const std::string &key, const std::string &value) {
    auto iter = g_optionsMap.find(key);
    if (g_optionsMap.find(key) == g_optionsMap.end()) {
        throw std::runtime_error("Config key not found: " + key);
    }
    iter->second = value;
}

void Config::SetOption(const std::string &key, std::vector<int64_t> &value) {
    auto iter = g_optionsMap.find(key);
    if (g_optionsMap.find(key) == g_optionsMap.end()) {
        throw std::runtime_error("Config key not found: " + key);
    }
    iter->second = value;
}

void Config::SetBuildStatic(bool isStatic) {
    g_funcType = isStatic ? FunctionType::STATIC : FunctionType::DYNAMIC;
}

std::string Config::ToString() {
    std::ostringstream oss;
    for (auto &it : g_optionsMap) {
        if (auto *valuePtr = std::get_if<bool>(&(it.second))) {
            oss << it.first << ": " << std::boolalpha << *valuePtr << std::endl;
        } else {
            std::visit([&](auto &value) {
                oss << it.first << ": " << value << std::endl;
            }, it.second);
        }
    }
    return oss.str();
}

void SetOptionOverlay(const std::string &key, bool value) {
    g_optionsMap[key] = value;
}

void SetOptionOverlay(const std::string &key, int64_t value) {
    g_optionsMap[key] = value;
}

void SetOptionOverlay(const std::string &key, const std::string &value) {
    g_optionsMap[key] = value;
}

void SetOptionOverlay(const std::string &key, std::vector<int64_t> &value) {
    g_optionsMap[key] = value;
}

bool GetOptionInner(const std::string &key, bool &value) {
    auto iter = g_optionsMap.find(key);
    if (iter == g_optionsMap.end()) {
        return false;
    }
    value = std::get<bool>(iter->second);
    return true;
}

bool GetOptionInner(const std::string &key, int64_t &value) {
    auto iter = g_optionsMap.find(key);
    if (iter == g_optionsMap.end()) {
        return false;
    }
    value = std::get<int64_t>(iter->second);
    return true;
}

bool GetOptionInner(const std::string &key, std::string &value) {
    auto iter = g_optionsMap.find(key);
    if (iter == g_optionsMap.end()) {
        return false;
    }
    value = std::get<std::string>(iter->second);
    return true;
}

bool GetOptionInner(const std::string &key, std::vector<int64_t> &value) {
    auto iter = g_optionsMap.find(key);
    if (iter == g_optionsMap.end()) {
        return false;
    }
    value = std::get<std::vector<int64_t>>(iter->second);
    return true;
}

FunctionType GetFunctionType() { return g_funcType; }

} // end npu::tile_fwk