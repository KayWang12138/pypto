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

namespace npu::tile_fwk::Distributed {

struct OpMetaData {
    explicit OpMetaData(const nlohmann::json &testData)
        : testData_(testData) {}
    nlohmann::json testData_;
};

struct DisTemplateOpRegistry {
    using opFunc = std::function<void(OpTestParam&, const std::string&)>;
    std::unordered_map <std::string, opFunc> disRegisterMap;
    
    static DisTemplateOpRegistry& GetRegistry()
    {
        static DisTemplateOpRegistry disOpRegistry;
        return disOpRegistry;
    }

    void RegistryOp(const std::string& opName, opFunc func)
    {
        disRegisterMap[opName] = std::move(func);
    }

    void Run(const std::string &opName, OpTestParam &testParam, const std::string &dtype)
    {
        if (!disRegisterMap.count(opName)) {
            FAIL() << "Unsupported op: " << opName;
        }
        disRegisterMap[opName](testParam, dtype);
    }
};

#define REGISTER_DIS_TEMPLATE_OP(opname, opfunc) \
namespace { \
struct opname##Register { \
    opname##Register() { \
        DisTemplateOpRegistry::GetRegistry().RegistryOp( \
            #opname, \
            [](OpTestParam &testParam, const std::string &dtype) { \
                if (dtype == "int32") opfunc<int32_t>(testParam); \
                else if (dtype == "float16") opfunc<float16>(testParam); \
                else if (dtype == "bfloat16") opfunc<bfloat16>(testParam); \
                else if (dtype == "float32") opfunc<float>(testParam); \
                else FAIL() << "Unsupported dtype: " << dtype; \
            } \
        ); \
    } \
}; \
static opname##Register g_##opname##_Register; \
}

template <typename T>
std::vector<T> GetOpMetaData(const std::string &op)
{
    auto caseFile = "../../../framework/tests/st/distributed/ops/test_case/" + op + "_st_test_cases.json";
    std::ifstream jsonFile(caseFile);
    if (!jsonFile.is_open()) {
        std::cerr << "Failed to open JSON file for op " << op << ". "
        << "Please check the path and ensure the file exists: " << caseFile << std::endl;
        return {};
    }
    nlohmann::json jsonData = nlohmann::json::parse(jsonFile);
    std::vector<T> testCaseList;
    for (auto &tc : jsonData.at("test_cases")) {
        testCaseList.emplace_back(tc);
    }
    if (testCaseList.empty()) {
        std::cerr << "No test cases found in json for op: " << op << ". "
        << "Please check the contents of: " << caseFile << std::endl;
    }
    return testCaseList;
}

} // namespace npu::tile_fwk::Distributed
