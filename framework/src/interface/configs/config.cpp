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
 * \file config.cpp
 * \brief
 */

#include <variant>
#include <sstream>
#include <shared_mutex>

#include <nlohmann/json.hpp>

#include "interface/inner/config.h"
#include "interface/utils/common.h"
#include "interface/utils/string_utils.h"
#include "interface/utils/file_utils.h"

using json = nlohmann::json;

namespace npu::tile_fwk {

using ValueType = std::variant<bool, int64_t, std::string, std::vector<int64_t>, std::map<int64_t, int64_t>>;

using MapType = std::map<int64_t, int64_t>;

static std::map<std::string, ValueType> g_passConfig = {
    {SG_PARALLEL_NUM, 20L},
    {SG_CYCLE_UPPER_BOUND, 10000L},
    {SG_CYCLE_LOWER_BOUND, 512L},
    {L1_REUSE, 0L},
    {L1_REUSE_MAP, std::map<int64_t, int64_t>{}},
    {CUBE_NBUFFER, 1L},
    {CUBE_NBUFFER_MAP, std::map<int64_t, int64_t>{}},
    {COPYIN_THRESHOLD, 1024 * 1024L},
    {OOO_PRESCHEDULE_METHOD, std::string("PriorDFS")}, // bugs in gcc 9.4
    {NBUFFER_MERGE_MODE, 1L},
    {VEC_NBUFFER_MAP, std::map<int64_t, int64_t>{}},
    {SG_VEC_PARALLEL_NUM, 48L},
    {SG_CUBE_PARALLEL_NUM, 24L},
    {SG_SKIP_PARTITION, false},
    {COPYOUT_RESOLVE_COALESCING, 0L},
};

static std::map<std::string, ValueType> g_runtimeConfig = {
    {MACHINE_SCHED_MODE, 0L},
    {WORKSPACE_RECYCLE_PERIOD, 10L},
    {ESTIMATED_STITCH_TASK_MAX_LOOP_NUM, 50L},
    {FIRST_STITCH_TASK_LOOP_NUM, 30L},
    {SUBSEQ_STITCH_TASK_INCR_LOOP_NUM, 30L}, // Increasing loop number
    {CFGCACHE_DEVICE_TASK_NUM, 0L},
    {CFGCACHE_ROOT_TASK_NUM, 0L},
    {CFGCACHE_LEAF_TASK_NUM, 0L},
    {SINGLE_LOOP_CALLOP_MAX_NUM, 20000L},
};

static std::map<std::string, ValueType> g_hostConfig = {
    {ONLY_CODEGEN, false},
};

static std::map<std::string, ValueType> g_codegenConfig = {
    {SUPPORT_DYNAMIC_UNALIGNED, false},
    {CODEGEN_EXPRESSION_FUSION, false},
};

static std::map<std::string, ValueType> g_verifyConfig = {
    {KEY_VERIFY_TENSOR_GRAPH, false},
    {KEY_VERIFY_PASS, false},
    {KEY_VERIFY_EXECUTE_GRAPH, false},
    {KEY_VERIFY_CHECK_PRECISION, false},
    {KEY_VERIFY_DUMP_TENSOR, false},
    {KEY_VERIFY_DUMP_OPERATION, false},
    {KEY_VERIFY_PROFILE_ENABLE, false},
};

static std::map<std::string, ValueType> g_globalConfig = {
    {PROFILE_ENABLE, false},
};

struct ConfigStorage {
    ConfigStorage() { Init(); }

    void Init() {
        printOption.edgeItems = 3; // 3 edge items
        printOption.precision = 4; // 4 float precision
        printOption.threshold = 1000; // 1000 default threshold
        printOption.linewidth = 80; // 80 max line width

        for (auto &[key, val] : g_globalConfig) {
            options[key] = val;
        }

        Reset();
    }

    void Reset() {
        funcType = FunctionType::DYNAMIC;
        semanticLabel = nullptr;
        for (auto &[key, val] : g_passConfig) {
            options["pass." + key] = val;
        }
        for (auto &[key, val] : g_runtimeConfig) {
            options["runtime." + key] = val;
        }
        for (auto &[key, val] : g_hostConfig) {
            options["host." + key] = val;
        }
        for (auto &[key, val] : g_codegenConfig) {
            options["codegen." + key] = val;
        }
        for (auto &[key, val] : g_verifyConfig) {
            options["verify." + key] = val;
        }
    }

    FunctionType funcType;
    std::shared_ptr<SemanticLabel> semanticLabel;
    std::string rundataDir;
    std::unordered_map<std::string, ValueType> options;
    PrintOptions printOption;
};

namespace config {

static ConfigStorage g_config;
std::shared_mutex g_rwlock;

void SetBuildStatic(bool isStatic) {
    g_config.funcType = isStatic ? FunctionType::STATIC : FunctionType::DYNAMIC;
}

std::string GetRunDataDir() {
    return g_config.rundataDir;
}

FunctionType GetFunctionType() {
    return g_config.funcType;
}

void SetSemanticLabel(const std::string &label, const char *filename , int lineno) {
    g_config.semanticLabel = std::make_shared<SemanticLabel>(label, filename, lineno);
}

void SetSemanticLabel(std::shared_ptr<SemanticLabel> label) {
    g_config.semanticLabel = label;
}

std::shared_ptr<SemanticLabel> GetSemanticLabel() {
    return g_config.semanticLabel;
}

bool HasOption(const std::string &key) {
    return g_config.options.find(StringUtils::ToLower(key)) != g_config.options.end();
}

std::string Dump() {
    std::ostringstream oss;
    auto &printOption = g_config.printOption;

    std::shared_lock lock(g_rwlock);
    oss << "funcType: " << (g_config.funcType == FunctionType::DYNAMIC ? "dynamic" : "static") << std::endl;
    if (g_config.semanticLabel) {
        oss << "sematicLabel: " << g_config.semanticLabel->label << std::endl;
    }
    oss << "printOption.edgeItems: " << printOption.edgeItems << std::endl;
    oss << "printOption.precision: " << printOption.precision << std::endl;
    oss << "printOption.threshold: " << printOption.threshold << std::endl;
    oss << "printOption.linewidth: " << printOption.linewidth << std::endl;

    for (auto &it : g_config.options) {
        if (std::holds_alternative<bool>(it.second)) {
            oss << it.first << ": " << std::get<bool>(it.second) << std::endl;
        } else if (std::holds_alternative<int64_t>(it.second)) {
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

bool internal::IsType(const std::string &key, const std::type_info &type) {
    std::shared_lock lock(g_rwlock);

    auto iter = g_config.options.find(StringUtils::ToLower(key));
    if (iter == g_config.options.end()) {
        return false;
    }
    if (std::holds_alternative<bool>(iter->second)) {
        return type == typeid(bool);
    } else if (std::holds_alternative<int64_t>(iter->second)) {
        return type == typeid(int64_t);
    } else if (std::holds_alternative<std::string>(iter->second)) {
        return type == typeid(std::string);
    } else if (std::holds_alternative<std::vector<int64_t>>(iter->second)) {
        return type == typeid(std::vector<int64_t>);
    } else if (std::holds_alternative<MapType>(iter->second)) {
        return type == typeid(MapType);
    } else {
        return false;
    }
}

#define DEFINE_GET_OPTION(Type)                                       \
    bool internal::GetOption(const std::string &key, Type &value) {   \
        std::shared_lock lock(g_rwlock);                              \
        auto iter = g_config.options.find(StringUtils::ToLower(key)); \
        if (iter == g_config.options.end()) {                         \
            return false;                                             \
        }                                                             \
        value = std::get<Type>(iter->second);                         \
        return true;                                                  \
    }

DEFINE_GET_OPTION(bool)
DEFINE_GET_OPTION(int64_t)
DEFINE_GET_OPTION(std::string)
DEFINE_GET_OPTION(std::vector<int64_t>)
DEFINE_GET_OPTION(MapType)
#undef DEFINE_GET_OPTION

static json toJson(const std::string &prefix) {
    json j;
    std::shared_lock lock(g_rwlock);
    for (auto &it : g_config.options) {
        if (!StringUtils::StartsWith(it.first, prefix)) {
            continue;
        }
        auto key = it.first.substr(prefix.size());
        if (std::holds_alternative<int64_t>(it.second)) {
            j[key] = std::get<int64_t>(it.second);
        } else if (std::holds_alternative<std::string>(it.second)) {
            j[key] = std::get<std::string>(it.second);
        } else if (std::holds_alternative<std::vector<int64_t>>(it.second)) {
            j[key] = std::get<std::vector<int64_t>>(it.second);
        } else if (std::holds_alternative<MapType>(it.second)) {
            j[key] = std::get<MapType>(it.second);
        }
    }
    return j;
}

constexpr const char *ENV_VAR_PYPTO_HOME = "PYPTO_HOME";
constexpr const char *ENV_VAR_HOME = "HOME";
void CreateRunDataDir() {
    if (!g_config.rundataDir.empty()) {
        return;
    }
    auto time = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::stringstream timestamp;
    timestamp << std::put_time(std::localtime(&time), "%Y%m%d%H%M%S");

    std::string envStr = GetEnvVar(ENV_VAR_PYPTO_HOME);
    std::string dir = envStr.empty() ? (GetEnvVar(ENV_VAR_HOME) + "/.pypto") : envStr;

    dir = dir + "/run/rundata_" + timestamp.str();
    bool res = CreateMultiLevelDir(dir);
    ASSERT(res) << "Failed to create directory: " << dir;

    g_config.rundataDir = dir;
}

static void SetOptionPost(const std::string &key) {
    if (StringUtils::StartsWith(key, "rundata.")) {
        if (g_config.rundataDir.empty()) {
            CreateRunDataDir();
        }
        auto value = toJson("rundata.").dump(2);
        auto dir = GetRunDataDir() + "/rundata.json";
        SaveFileSafe(dir, reinterpret_cast<uint8_t*>(value.data()), value.size());
    }
}

void internal::SetOption(const std::string &key, int64_t value) {
    g_rwlock.lock();
    g_config.options[StringUtils::ToLower(key)] = value;
    g_rwlock.unlock();
    SetOptionPost(key);
}

void internal::SetOption(const std::string &key, bool value) {
    g_rwlock.lock();
    g_config.options[StringUtils::ToLower(key)] = value;
    g_rwlock.unlock();
    SetOptionPost(key);
}

void internal::SetOption(const std::string &key, const char *value) {
    g_rwlock.lock();
    g_config.options[StringUtils::ToLower(key)] = value;
    g_rwlock.unlock();
    SetOptionPost(key);
}

void internal::SetOption(const std::string &key, const std::string &value) {
    g_rwlock.lock();
    g_config.options[StringUtils::ToLower(key)] = value;
    g_rwlock.unlock();
    SetOptionPost(key);
}

void internal::SetOption(const std::string &key, const std::vector<int64_t> &value) {
    g_rwlock.lock();
    g_config.options[StringUtils::ToLower(key)] = value;
    g_rwlock.unlock();
    SetOptionPost(key);
}

void internal::SetOption(const std::string &key, const std::map<int64_t, int64_t> &value) {
    g_rwlock.lock();
    g_config.options[StringUtils::ToLower(key)] = value;
    g_rwlock.unlock();
    SetOptionPost(key);
}

void SetPrintOptions(int edgeItems, int precision, int threshold, int linewidth) {
    g_config.printOption.edgeItems = edgeItems;
    g_config.printOption.precision = precision;
    g_config.printOption.threshold = threshold;
    g_config.printOption.linewidth = linewidth;
}

PrintOptions &GetPrintOptions() {
    return g_config.printOption;
}

void Reset() {
    g_rwlock.lock();
    g_config.Reset();
    g_rwlock.unlock();
}

std::unordered_map<std::string, ValueType> GetOptions(){
    return g_config.options;
}

std::shared_ptr<ConfigStorage> Duplicate() {
    std::shared_lock lock(g_rwlock);
    return std::make_shared<ConfigStorage>(g_config);
}

void Restore(std::shared_ptr<ConfigStorage> config) {
    g_rwlock.lock();
    g_config = *config;
    g_rwlock.unlock();
}

} // namespace config
} // namespace npu::tile_fwk
