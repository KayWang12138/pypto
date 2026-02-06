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
 * \file internal_parser.cpp
 * \brief
 */

#include <iostream>
#include "internal_parser.h"

namespace npu {
namespace tile_fwk {
const std::string iniFile = "/configs/platforminfo.ini";
const std::string paths = "PATHS";
const std::string comma = ",";
const std::string direction = "->";

std::string GetCurrentSharedLibPath() {
    static std::string currentLibPath;
    if (!currentLibPath.empty()) {
        return currentLibPath;
    }

    Dl_info info;
    if (dladdr(reinterpret_cast<void*>(GetCurrentSharedLibPath), &info)) {
        currentLibPath = std::string(info.dli_fname);
        int32_t pos = currentLibPath.rfind('/');
        if (pos >= 0) {
            currentLibPath = currentLibPath.substr(0, pos);
        }
    }
    return currentLibPath;
}

std::vector<std::string> SplitByDelimiter(const std::string& str, const std::string& delimiter) {
    std::vector<std::string> res;
    size_t start = 0;
    size_t pos = str.find(delimiter);
    while (pos != std::string::npos) {
        res.emplace_back(str.substr(start, pos - start));
        start = pos + delimiter.size();
        pos = str.find(delimiter, start);
    }
    res.emplace_back(str.substr(start));
    return res;
}

// helper function
MemoryType StringToMemoryType(const std::string& memType) {
    const std::unordered_map<std::string, MemoryType> memTypeMap = {
        {"MEM_DEVICE_DDR", MemoryType::MEM_DEVICE_DDR},
        {"MEM_L1", MemoryType::MEM_L1},
        {"MEM_L0A", MemoryType::MEM_L0A},
        {"MEM_L0B", MemoryType::MEM_L0B},
        {"MEM_L0C", MemoryType::MEM_L0C},
        {"MEM_UB", MemoryType::MEM_UB},
        {"MEM_BT", MemoryType::MEM_BT}
    };
    auto it = memTypeMap.find(memType);
    if (it != memTypeMap.end()) {
        return it->second;
    }
    return MemoryType::MEM_UNKNOWN;
}

bool InternalParser::LoadInternalInfo() {
    std::string internalFile = RealPath(GetCurrentSharedLibPath() + iniFile);
    if (!IsPathExist(internalFile)) {
        return false;
    }
    std::ifstream file(internalFile);
    if (!file.is_open()) {
        return false;
    }
    std::string line;
    std::string trimLine;
    std::string section;
    std::string info;
    bool currentSoc = true;
    while (std::getline(file, line)) {
        trimLine = trim(line);
        if (trimLine.empty() || !currentSoc) {
            continue;
        }
        if (trimLine.find("]") != std::string::npos) {
            data_[section] = info;
            info.clear();
        } else if (trimLine.find("{") != std::string::npos) {
            if (trim(trimLine.substr(0, trimLine.find(':'))) != socVersion_) {
                currentSoc = false;
            }
        } else if (trimLine.find("[") != std::string::npos) {
            section = trim(trimLine.substr(0, trimLine.find(':')));
        } else if (trimLine.find("}") != std::string::npos) {
            currentSoc = true;
        } else {
            info += trimLine;
        }
    }
    file.close();
    return true;
}

bool InternalParser::GetDataPath(std::vector<std::pair<MemoryType, MemoryType>> &dataPath) {
    std::string currentPath = data_[paths];
    if (currentPath.empty()) {
        return false;
    }
    dataPath.clear();
    auto firstSplit = SplitByDelimiter(currentPath, comma);
    for (const auto& subStr : firstSplit) {
        auto secondSplit = SplitByDelimiter(subStr, "->");
        if (secondSplit.size() != 2) {
            continue;
        }
        dataPath.emplace_back(std::make_pair(StringToMemoryType(secondSplit[0]), StringToMemoryType(secondSplit[1])));
    }
    return true; 
}
 
}  // namespace tile_fwk
}  // namespace npu