/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_distributed.h
 * \brief
 */

#include <nlohmann/json.hpp>
#include <fstream>
#include <vector>
#include <string>
#include "interface/configs/config_manager.h"
#include "distributed_op_test_suite.h"
#include <filesystem>

namespace npu::tile_fwk::Distributed {

struct OpMetaData {
    explicit OpMetaData(const nlohmann::json &testData) : testData_(testData) {}
    nlohmann::json testData_;
};

struct DisOpRegister {
    using opFunc = std::function<void(OpTestParam&, const std::string&, const nlohmann::json& testData)>;
    std::unordered_map<std::string, opFunc> disRegisterMap;
    
    static DisOpRegister& GetRegister()
    {
        static DisOpRegister disOpRegister;
        return disOpRegister;
    }

    template <typename TFunc>
    void RegisterOp(const std::string& opName, TFunc func)
    {
        disRegisterMap[opName] = [func](OpTestParam &testParam, const std::string &dtype, const nlohmann::json& testData)
        {
            if (dtype == "int32") {
                func.template operator()<int32_t>(testParam, testData);
            } else if (dtype == "float16") {
                func.template operator()<float16>(testParam, testData);
            } else if (dtype == "bfloat16") {
                func.template operator()<bfloat16>(testParam, testData);
            } else if (dtype == "float32") {
                func.template operator()<float>(testParam, testData);
            } else {
                FAIL() << "Unsupported dtype: " << dtype;
            }
        };
    }

    void Run(const std::string &opName, OpTestParam &testParam, const std::string &dtype, 
             const nlohmann::json& testData)
    {
        if (!disRegisterMap.count(opName)) {
            FAIL() << "Unsupported op: " << opName;
        }
        disRegisterMap[opName](testParam, dtype, testData);
    }
};

template <typename T>
std::vector<T> GetOpMetaDataFromFile(const std::filesystem::path& filePath)
{
    std::ifstream jsonFile(filePath);
    if (!jsonFile.is_open()) {
        std::cerr << "Failed to open JSON file: " << filePath << std::endl;
        return {};
    }
    std::vector<T> testCaseList;
    nlohmann::json jsonData = nlohmann::json::parse(jsonFile);
    for (auto &tc : jsonData.at("test_cases")) {
        testCaseList.emplace_back(tc);
    }
    if (testCaseList.empty()) {
        std::cerr << "No test cases found in json. "
        << "Please check the contents of: " << filePath << std::endl;
    }
    return testCaseList;
}

template <typename T>
std::vector<T> GetOpMetaDataFromDir(const std::filesystem::path& dirPath)
{
    static std::vector<T> allTestCases;
    if (!std::filesystem::exists(dirPath) || !std::filesystem::is_directory(dirPath)) {
        std::cerr << "Invaild directory: " << dirPath << std::endl;
        return {};
    }
    std::vector<std::filesystem::path> jsonFiles;
    for (const auto& file : std::filesystem::directory_iterator(dirPath)) {
        if (file.path().extension() != ".json") continue;
        jsonFiles.push_back(file.path());
    }
    std::sort(jsonFiles.begin(), jsonFiles.end(),
    [](const auto& a, const auto& b) {
        auto aLower = a.filename().string();
        auto bLower = b.filename().string();
        std::transform(aLower.begin(), aLower.end(), aLower.begin(), ::tolower);
        std::transform(bLower.begin(), bLower.end(), bLower.begin(), ::tolower);
        return aLower < bLower;
    });
    for (const auto& file : jsonFiles) {
        auto testCases = GetOpMetaDataFromFile<T>(file);
        allTestCases.insert(allTestCases.end(), testCases.begin(), testCases.end());
    }
    return allTestCases;
}

template <typename T>
std::vector<T> GetOpMetaData()
{
    const char* jsonPath = std::getenv("JSON_PATH");
    std::filesystem::path casePath;
    if (jsonPath != nullptr) {
        casePath = std::filesystem::path(jsonPath);
    } else {
        casePath = std::filesystem::path(TEST_CASE_DIR);
    }
    if (!std::filesystem::exists(casePath)) {
        std::cerr << "JSON path does not exist: " << casePath << std::endl;
    }
    if (std::filesystem::is_regular_file(casePath)) {
        return GetOpMetaDataFromFile<T>(casePath);
    }
    if (std::filesystem::is_directory(casePath)) {
        return GetOpMetaDataFromDir<T>(casePath);
    }
    std::cerr << "Invalid path type: " << casePath << std::endl;
    return {};
}

void GegisterOps();

} // namespace npu::tile_fwk::Distributed
