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
#include <variant>
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
constexpr const char *ONLY_CODEGEN = "only_codegen";
constexpr const char *SUPPORT_DYNAMIC_UNALIGNED = "support_dynamic_unaligned";
constexpr const char *CODEGEN_EXPRESSION_FUSION = "codegen_expression_fusion";

//runtime
constexpr const char *MACHINE_SCHED_MODE = "machine_sched_mode";
constexpr const char *WORKSPACE_RECYCLE_PERIOD = "workspace_recycle_period";
constexpr const char *ESTIMATED_STITCH_TASK_MAX_LOOP_NUM = "estimated_stitch_task_max_loop_num";
constexpr const char *FIRST_STITCH_TASK_LOOP_NUM = "first_stitch_task_loop_num";
constexpr const char *SUBSEQ_STITCH_TASK_INCR_LOOP_NUM = "subseq_stitch_task_incr_loop_num";
constexpr const char *PROFILE_ENABLE = "profile_enable";
constexpr const char *SINGLE_LOOP_CALLOP_MAX_NUM = "single_loop_callop_max_num";
constexpr const char *CFGCACHE_DEVICE_TASK_NUM = "cfgcache_device_task_num";
constexpr const char *CFGCACHE_ROOT_TASK_NUM = "cfgcache_root_task_num";
constexpr const char *CFGCACHE_LEAF_TASK_NUM = "cfgcache_leaf_task_num";

/* Rundata KEYS */
constexpr const char *KEY_RUNTYPE = "runtype";
constexpr const char *KEY_PTO_CONFIG_FILE = "pto_config_file";
constexpr const char *KEY_COMPUTE_GRAPH_PATH = "compute_graph_path";
constexpr const char *KEY_SWIM_GRAPH_PATH = "swim_graph_path";
constexpr const char *KEY_FLOW_VERIFY_PATH = "flow_verify_path";
constexpr const char *KEY_PROGRAM_PATH = "program_file";

struct ConfigStorage;

struct PrintOptions {
    int edgeItems;
    int precision;
    int threshold;
    int linewidth;
};

struct SemanticLabel {
    std::string label;
    std::string filename;
    int lineno;

    SemanticLabel(const std::string &tlabel, const char *tfilename, int tlineno)
        : label(tlabel), filename(tfilename), lineno(tlineno) {}
    SemanticLabel(const std::string &tlabel, const std::string &tfilename, int tlineno)
        : label(tlabel), filename(tfilename), lineno(tlineno) {}
};

namespace config {
FunctionType GetFunctionType();

std::shared_ptr<SemanticLabel> GetSemanticLabel();
void SetSemanticLabel(std::shared_ptr<SemanticLabel> label);

namespace internal {
bool IsType(const std::string &key, const std::type_info &type);
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

template <typename T>
bool IsType(const std::string &key) {
    if constexpr (std::is_integral_v<T>) {
        return internal::IsType(key, typeid(int64_t));
    } else {
        return internal::IsType(key, typeid(T));
    }
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

template <typename T>
void SetRunDataOption(const std::string &key, const T &value) {
    internal::SetOption("rundata." + key, value);
}
void CreateRunDataDir();

std::unordered_map<std::string, std::variant<int64_t, std::string, std::vector<int64_t>, std::map<int64_t, int64_t>>> GetOptions();
} // namespace config
} // namespace npu::tile_fwk
