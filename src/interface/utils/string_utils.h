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
 * \file string_utils.h
 * \brief
 */

#pragma once

#include <cstddef>
#include <cstring>
#include <string>
#include <vector>

namespace npu::tile_fwk {
class StringUtils {
public:
    static void Trim(std::string &str) {
        if (str.empty()) {
            return;
        }
        size_t startPos = str.find_first_not_of(" \t");
        size_t endPos = str.find_last_not_of(" \t");
        if (startPos == std::string::npos || startPos > endPos) {
            str.clear();
            return;
        }
        str = str.substr(startPos, endPos - startPos + 1);
    }

    static std::vector<std::string> Split(const std::string &str, const std::string &pattern) {
        std::vector<std::string> resVec;
        if (str.empty() || pattern.empty()) {
            return resVec;
        }
        std::string strAndPattern = str + pattern;
        size_t pos = strAndPattern.find(pattern);
        while (pos != std::string::npos) {
            resVec.push_back(strAndPattern.substr(0, pos));
            strAndPattern = strAndPattern.substr(pos + pattern.size());
            pos = strAndPattern.find(pattern);
        }
        return resVec;
    }

    static bool StartsWith(const std::string &str, const std::string &prefix) {
        if (str.size() < prefix.size())
            return false;
        for (size_t i = 0; i < prefix.size(); i++) {
            if (prefix[i] != str[i])
                return false;
        }
        return true;
    }

    static std::string BaseName(const char *fname) {
        if (auto start = strrchr(fname, '/')) {
            return start + 1;
        }
        return fname;
    }
};
}
