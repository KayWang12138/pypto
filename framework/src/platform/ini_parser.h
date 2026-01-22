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
 * \file ini_parser.h
 * \brief
 */
#ifndef INI_PARSER_H_
#define INI_PARSER_H_

#include <fstream>
#include <map>
#include <set>
#include <string>
#include <vector>
#include <unordered_map>
#include <numeric_limits>

namespace npu {
namespace tile_fwk {

class INIParser {
  public:
    INIParser() = default;
    ~INIParser() = default;
    bool Initialize(const std::string &iniFilePath); 
    bool GetStringVal(const std::string& column, const std::string& key, std::string& val);
    bool GetSizeVal(const std::string& column, const std::string& key, size_t& val);

    bool GetCCECVersion(std::unordered_map<std::string, std::string>& ccecVersion);
    bool GetCoreVersion(std::unordered_map<std::string, std::string>& curVersion);
    bool GetDataPath(std::vector<std::vector<std::string>>& dataPath);
  private:
    bool ReadINIFile(const std::string& filepath);
    bool FilterCCECVersion(const std::string& key, std::string &coreType);
    bool FilterDirections(const std::string& value, std::string &part);
    bool FilterDataPath(const std::string& part, std::string &from, std::string &to);

    std::unordered_map<std::string, std::unordered_map<std::string, std::string>> data_;
};
} // namespace tile_fwk
} // namepsace npu 
#endif