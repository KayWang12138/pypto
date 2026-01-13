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
    explicit OpMetaData(const nlohmann::json &testData)
        : testData_(testData) {}
    nlohmann::json testData_;
};

struct DisOpRegister {
    using opFunc = std::function<void(OpTestParam&, const std::string&)>;
    std::unordered_map <std::string, opFunc> disRegisterMap;
    
    static DisOpRegister& GetRegister()
    {
        static DisOpRegister disOpRegister;
        return disOpRegister;
    }

    template <typename TFunc>
    void RegisterOp(const std::string& opName, TFunc func)
    {
        disRegisterMap[opName] = [func](OpTestParam &testParam, const std::string &dtype)
        {
            if (dtype == "int32") func.template operator()<int32_t>(testParam);
            else if (dtype == "float16") func.template operator()<float16>(testParam);
            else if (dtype == "bfloat16") func.template operator()<bfloat16>(testParam);
            else if (dtype == "float32") func.template operator()<float>(testParam);
            else FAIL() << "Unsupported dtype: " << dtype;
        };
    }

    void Run(const std::string &opName, OpTestParam &testParam, const std::string &dtype)
    {
        if (!disRegisterMap.count(opName)) {
            FAIL() << "Unsupported op: " << opName;
        }
        disRegisterMap[opName](testParam, dtype);
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
}

template <typename T>
std::vector<T> GetOpMetaDataFromDir(const std::filesystem::path& disPath)
{
    static std::vector<T> allTestCases;
    if (!std::filesystem::exists(disPath) || !std::filesystem::is_directory(disPath)) {
        std::cerr << "Invaild directory: " << disPath << std::endl;
        return {};
    }
    for (const auto& file : std::filesystem::directory_iterator(disPath)) {
        if (file.path().extension() != ".json") continue;
        auto testCases = GetOpMetaDataFromFile<T>(file.path());
        allTestCases.insert(allTestCases.end(), testCases.begin(), testCases.end());
    }
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

// template <typename T>
// std::vector<T> GetOpMetaData(const std::string &op)
// {
//     auto caseFile = "../../../framework/tests/st/distributed/ops/test_case/" + op + "_st_test_cases.json";
//     std::ifstream jsonFile(caseFile);
//     if (!jsonFile.is_open()) {
//         std::cerr << "Failed to open JSON file for op " << op << ". "
//         << "Please check the path and ensure the file exists: " << caseFile << std::endl;
//         return {};
//     }
//     nlohmann::json jsonData = nlohmann::json::parse(jsonFile);
//     std::vector<T> testCaseList;
//     for (auto &tc : jsonData.at("test_cases")) {
//         testCaseList.emplace_back(tc);
//     }
//     if (testCaseList.empty()) {
//         std::cerr << "No test cases found in json for op: " << op << ". "
//         << "Please check the contents of: " << caseFile << std::endl;
//     }
//     return testCaseList;
// }

void GegisterOps();

} // namespace npu::tile_fwk::Distributed
