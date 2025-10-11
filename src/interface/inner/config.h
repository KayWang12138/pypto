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
 * \file config.h
 * \brief
 */

#pragma once
#include <iostream>
#include "tilefwk/tilefwk.h"

namespace npu::tile_fwk {

constexpr const char *SG_PARALLEL_NUM = "parallel_threshold";
constexpr const char *SG_CYCLE_UPPER_BOUND = "cycle_upper_bound";
constexpr const char *SG_CYCLE_LOWER_BOUND = "cycle_lower_bound";
constexpr const char *L1_REUSE = "l1_reuse";
constexpr const char *L1_REUSE_MAP = "l1_reuse_map";
constexpr const char *CUBE_NBUFFER = "cube_nbuffer";
constexpr const char *CUBE_NBUFFER_MAP = "cube_nbuffer_map";
constexpr const char *COPYIN_THRESHOLD = "copyin_threshold";
constexpr const char *MACHINE_CONFIG = "machine_config";
constexpr const char *OOO_PRESCHEDULE_METHOD = "ooo_preschedule_method";
constexpr const char *NBUFFER_MERGE_MODE = "nbuffer_merge_mode";
constexpr const char *VEC_NBUFFER_MAP = "vec_nbuffer_map";
constexpr const char *SG_CUBE_PARALLEL_NUM = "sg_cube_parallel_num";
constexpr const char *SG_VEC_PARALLEL_NUM = "sg_vec_parallel_num";
constexpr const char *SG_SKIP_PARTITION = "sg_skip_partition";
constexpr const char *NBUFFER_NUM = "nbuffer_num";
constexpr const char *L1_REUSE_NUM = "l1_reuse_num";
constexpr const char *CUBE_NBUFFER_NUM = "cube_nbuffer_num";
constexpr const char *DB_TYPE = "db_type";
constexpr const char *COPYOUT_RESOLVE_COALESCING = "copyout_resolve_coalescing";

struct ConfigStorage;

struct PrintOptions {
    int edgeItems;
    int precision;
    int threshold;
    int linewidth;
};

namespace config {
FunctionType GetFunctionType();

std::string GetSemanticLabel();

namespace internal {
bool GetOption(const std::string &key, int64_t &value);
bool GetOption(const std::string &key, std::string &value);
bool GetOption(const std::string &key, std::vector<int64_t> &value);
bool GetOption(const std::string &key, std::map<int64_t, int64_t> &value);
} // namespace internal

template <typename T>
T GetOption(const std::string &key) {
    bool exist = false;
    T val = {};
    if constexpr (std::is_integral_v<T>) {
        int64_t tmp = 0;
        exist = internal::GetOption(key, tmp);
        val = static_cast<T>(tmp);
    } else {
        exist = internal::GetOption(key, val);
    }
    if (!exist) {
        std::cout << Dump() << std::endl;
        throw std::runtime_error("config " + key + " not exist");
    }
    return val;
}

#define DEFINE_CONFIG_GROUP(group, prefix)                   \
    inline bool Has##group##Option(const std::string &key) { \
        return HasOption(prefix "." + key);                  \
    }                                                        \
    template <typename T>                                    \
    inline T Get##group##Option(const std::string &key) {    \
        return GetOption<T>(prefix "." + key);               \
    }

DEFINE_CONFIG_GROUP(CodeGen, "codegen")
DEFINE_CONFIG_GROUP(Pass, "pass")
DEFINE_CONFIG_GROUP(Runtime, "runtime")
DEFINE_CONFIG_GROUP(Host, "host")

std::shared_ptr<ConfigStorage> Duplicate();
void Restore(std::shared_ptr<ConfigStorage> config);

PrintOptions &GetPrintOptions();
} // namespace config
} // namespace npu::tile_fwk
