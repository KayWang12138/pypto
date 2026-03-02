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

#define STRINGIFY(x) #x
#include "platform_parser.h"

namespace npu {
namespace tile_fwk {
const uint32_t kMaxLength = 50;
const std::string version = "version";
void *GetSymbol(const std::string &sym) {
    void *ptr = nullptr;
    const char* CannPath = STRINGIFY(ASCEND_CANN_PACKAGE_PATH);
    std::string LibPathDir = std::string(CannPath) + "/lib64/libruntime.so";
    std::string soPath = RealPath(LibPathDir);
    void* handle = dlopen(soPath.c_str(), RTLD_LAZY);
    if (handle != nullptr) {
        ptr = dlsym(handle, sym.c_str());
    }
    return ptr;
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
    int ret = 1;
    char charVal[kMaxLength] = {0};
#ifdef BUILD_WITH_CANN
    using GetSocSpecFunc = int (*)(const char *, const char *, char *, const uint32_t);
    std::string socSpecFuncName = "rtGetSocSpec";
    auto socSpecFunc = (GetSocSpecFunc)GetSymbol(socSpecFuncName);
    ret = socSpecFunc(column.c_str(), key.c_str(), charVal, kMaxLength);
#endif
    if (ret == 0) {
        val = std::string(charVal);
        return true;
    }
    (void)column;
    (void)key;
    (void)val;
    return false;
}
}  // namespace tile_fwk
}  // namespace npu