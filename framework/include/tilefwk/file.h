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
 * \file file.h
 * \brief
 */

#pragma once
#define BUFFERSIZE 4096
#include <dlfcn.h>
#include <sys/stat.h>

namespace npu {
namespace tile_fwk {
inline std::string RealPath(const std::string &path) {
    std::string res;
    if (path.empty()) {	
        return res;	
    }	
    if (path.size() >= BUFFERSIZE) {	
        return res;	
    }	
    char resovedPath[BUFFERSIZE] = {0x00};
    if (realpath(path.c_str(), resovedPath) != nullptr) {
        return std::string(resovedPath);
    } else {
        return res;
    }
}

inline bool FileExist(const std::string &filePath) {
    return !RealPath(filePath).empty();
}

inline bool IsPathExist(const std::string& path) {
    if (path.empty()) {
        return false;
    }
    struct stat buffer;
    return (stat(path.c_str(), &buffer) == 0);
}

inline std::string trim(const std::string& s) {
    size_t start = s.find_first_not_of(" \t\n\r");
    size_t end = s.find_last_not_of(" \t\n\r");
    return (start == std::string::npos) ? "" : s.substr(start, end - start + 1);
}

std::string GetCurrentSharedLibPath();
}
}