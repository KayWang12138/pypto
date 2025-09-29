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
#include <variant>
#include <sstream>
#include "interface/inner/config.h"
#include "interface/utils/string_utils.h"

namespace npu::tile_fwk {

using ValueType = std::variant<int64_t, std::string, std::vector<int64_t>, std::map<int64_t, int64_t>>;

using MapType = std::map<int64_t, int64_t>;

static std::map<std::string, ValueType> g_passConfig = {
    {SG_PARALLEL_NUM, 20},
    {SG_CYCLE_UPPER_BOUND, 10000},
    {SG_CYCLE_LOWER_BOUND, 512},
    {L1_REUSE, 0},
    {L1_REUSE_MAP, std::map<int64_t, int64_t>{}},
    {CUBE_NBUFFER, 1},
    {CUBE_NBUFFER_MAP, std::map<int64_t, int64_t>{}},
    {COPYIN_THRESHOLD, 1024 * 1024},
    {MACHINE_CONFIG, 0},
    {OOO_PRESCHEDULE_METHOD, "PriorDFS"},
    {NBUFFER_MERGE_MODE, 1},
    {VEC_NBUFFER_MAP, std::map<int64_t, int64_t>{}},
    {SG_VEC_PARALLEL_NUM, 48},
    {SG_CUBE_PARALLEL_NUM, 24},
    {SG_SKIP_PARTITION, false},
    {COPYOUT_RESOLVE_COALESCING, 0},
};

struct ConfigStorage {
    ConfigStorage() { Reset(); }

    void Reset() {
        funcType = FunctionType::DYNAMIC;
        sematicLabel = "";
        edgeItems = 3; // 3 edge items
        precision = 4; // 4 float precision
        for (auto &[key, val] : g_passConfig) {
            options["pass." + key] = val;
        }
    }

    FunctionType funcType;
    std::string sematicLabel;
    std::unordered_map<std::string, ValueType> options;

    // print options
    int edgeItems;
    int precision;
};

namespace config {

static ConfigStorage g_config;

void SetBuildStatic(bool isStatic) {
    g_config.funcType = isStatic ? FunctionType::STATIC : FunctionType::DYNAMIC;
}

FunctionType GetFunctionType() {
    return g_config.funcType;
}

void SetSemanticLabel(const std::string &label) {
    g_config.sematicLabel = label;
}

std::string GetSemanticLabel() {
    return g_config.sematicLabel;
}

bool HasOption(const std::string &key) {
    return g_config.options.find(StringUtils::ToLower(key)) != g_config.options.end();
}

std::string Dump() {
    std::ostringstream oss;

    oss << "funcType: " << (g_config.funcType == FunctionType::DYNAMIC ? "dynamic" : "static") << std::endl;
    oss << "sematicLabel: " << g_config.sematicLabel << std::endl;
    oss << "printOption.edgeItems: " << g_config.edgeItems << std::endl;
    oss << "printOption.precision: " << g_config.precision << std::endl;

    for (auto &it : g_config.options) {
        if (std::holds_alternative<int64_t>(it.second)) {
            oss << it.first << ": " << std::get<int64_t>(it.second) << std::endl;
        } else if (std::holds_alternative<std::string>(it.second)) {
            oss << it.first << ": " << std::get<std::string>(it.second) << std::endl;
        } else if (std::holds_alternative<std::vector<int64_t>>(it.second)) {
            oss << it.first << ": " << std::get<std::vector<int64_t>>(it.second) << std::endl;
        } else if (std::holds_alternative<MapType>(it.second)) {
            oss << it.first << ": ";
            for (auto &[k, v] : std::get<MapType>(it.second)) {
                oss << "{" << k << ":" << v << "}";
            }
            oss << std::endl;
        } else {
            throw std::runtime_error("Config value type not supported: " + it.first);
        }
    }
    return oss.str();
}

#define DEFINE_GET_OPTION(Type)                                       \
    bool internal::GetOption(const std::string &key, Type &value) {   \
        auto iter = g_config.options.find(StringUtils::ToLower(key)); \
        if (iter == g_config.options.end()) {                         \
            return false;                                             \
        }                                                             \
        value = std::get<Type>(iter->second);                         \
        return true;                                                  \
    }

DEFINE_GET_OPTION(int64_t)
DEFINE_GET_OPTION(std::string)
DEFINE_GET_OPTION(std::vector<int64_t>)
DEFINE_GET_OPTION(MapType)

void internal::SetOption(const std::string &key, int64_t value) {
    g_config.options[StringUtils::ToLower(key)] = value;
}

void internal::SetOption(const std::string &key, const std::string &value) {
    g_config.options[StringUtils::ToLower(key)] = value;
}

void internal::SetOption(const std::string &key, const std::vector<int64_t> &value) {
    g_config.options[StringUtils::ToLower(key)] = value;
}

void internal::SetOption(const std::string &key, const std::map<int64_t, int64_t> &value) {
    g_config.options[StringUtils::ToLower(key)] = value;
}

void SetPrintOptions(int edgeItems, int precision) {
    g_config.edgeItems = edgeItems;
    g_config.precision = precision;
}

void GetPrintOptions(int &edgeItems, int &precision) {
    edgeItems = g_config.edgeItems;
    precision = g_config.precision;
}

void Reset() {
    g_config.Reset();
}

std::shared_ptr<ConfigStorage> Duplicate() {
    return std::make_shared<ConfigStorage>(g_config);
}

void Restore(std::shared_ptr<ConfigStorage> config) {
    g_config = *config;
}
} // namespace config
} // namespace npu::tile_fwk