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
 * \file config_storage.h
 * \brief
 */

#pragma once
#include <cstdint>
#include <memory>
#include <vector>
#include <cassert>
#include <array>
#include <iostream>
#include <sstream>
#include <map>
#include <string>
#include <set>
#include <unordered_set>
#include <variant>
#include <typeinfo>

#include "tilefwk/error.h"
#include "tilefwk/data_type.h"
#include "interface/inner/hash_buffer.h"
#include "interface/utils/common.h"

namespace npu::tile_fwk {

/* RuntimeConfig KEYS*/
const std::string SG_PARALLEL_NUM = "parallel_threshold";
const std::string SG_CYCLE_UPPER_BOUND = "cycle_upper_bound";
const std::string SG_CYCLE_LOWER_BOUND = "cycle_lower_bound";
const std::string L1_REUSE = "l1_reuse";
const std::string L1_REUSE_MAP = "l1_reuse_map";
const std::string CUBE_NBUFFER = "cube_nbuffer";
const std::string CUBE_NBUFFER_MAP = "cube_nbuffer_map";
const std::string COPYIN_THRESHOLD = "copyin_threshold";
const std::string MACHINE_CONFIG = "machine_config";
const std::string OOO_PRESCHEDULE_METHOD_DEFAULT = "ooo_preschedule_method_default";
const std::string OOO_PRESCHEDULE_METHOD = "ooo_preschedule_method";
const std::string NBUFFER_MERGE_MODE = "nbuffer_merge_mode";
const std::string VEC_NBUFFER_MAP = "vec_nbuffer_map";
const std::string SG_CUBE_PARALLEL_NUM = "sg_cube_parallel_num";
const std::string SG_VEC_PARALLEL_NUM = "sg_vec_parallel_num";
const std::string SG_SKIP_PARTITION = "sg_skip_partition";
const std::string NBUFFER_NUM = "nbuffer_num";
const std::string L1_REUSE_NUM = "l1_reuse_num";
const std::string CUBE_NBUFFER_NUM = "cube_nbuffer_num";
const std::string DB_TYPE = "db_type";
const std::string COPYOUT_RESOLVE_COALESCING = "copyout_resolve_coalescing";

class ConfigStorage {
public:
    using ConfigValue = std::variant<int, bool, std::string, std::map<int, int>, uint8_t, std::map<std::string,std::string>>;
    explicit ConfigStorage() {
        Reset();
    }

    void Reset() {
        const int parallel_num = 20;
        const int cycle_upper_bound = 10000;
        const int cycle_lower_bound = 512;
        const int copyin_threshold = 1024 * 1024;
        const int sg_vec_parallel_num = 48;
        const int sg_cube_parallel_num = 24;
        configs_[SG_PARALLEL_NUM] = parallel_num;  // default threshold
        configs_[SG_CYCLE_UPPER_BOUND] = cycle_upper_bound; // defalt cycle upper bound
        configs_[SG_CYCLE_LOWER_BOUND] = cycle_lower_bound; // default cycle threshold
        configs_[L1_REUSE] = 0;
        configs_[L1_REUSE_MAP] = std::map<int,int>({});
        configs_[CUBE_NBUFFER] = 1;
        configs_[CUBE_NBUFFER_MAP] = std::map<int,int>({});
        configs_[COPYIN_THRESHOLD] = copyin_threshold; // default copyin threshold
        configs_[MACHINE_CONFIG] = static_cast<uint8_t>(0);
        configs_[OOO_PRESCHEDULE_METHOD_DEFAULT] = std::move(std::string("PriorDFS"));
        configs_[OOO_PRESCHEDULE_METHOD] = std::map<std::string,std::string>({});
        configs_[NBUFFER_MERGE_MODE] = 1;
        configs_[SG_VEC_PARALLEL_NUM] = sg_vec_parallel_num;
        configs_[SG_CUBE_PARALLEL_NUM] = sg_cube_parallel_num;
        configs_[SG_SKIP_PARTITION] = false;
        configs_[VEC_NBUFFER_MAP] = std::map<int, int>({});
        configs_[COPYOUT_RESOLVE_COALESCING] = 0;
    }

    template <typename T>
    void Set(const std::string& key, const T& value) {
        configs_[key] = value;
    }

    template <typename T>
    T Get(const std::string& key) const {
        auto it = configs_.find(key);
        if (it == configs_.end()) {
            throw std::runtime_error("Config key not found: " + key);
        }

        try {
            return std::get<T>(it->second);
        } catch (const std::bad_variant_access&) {
            throw TILEFWK_ERROR() << "Type mismatch for config key: " << key << ", expected: " << typeid(T).name()
                                  << ", actual type index: " << it->second.index();
        }
    }

    template <typename T>
    T Get(const std::string& key, const T& defaultValue) const noexcept {
        auto it = configs_.find(key);
        if (it == configs_.end()) {
            return defaultValue;
        }

        const T* value = std::get_if<T>(&it->second);
        return value ? *value : defaultValue;
    }

    bool Has(const std::string& key) const {
        return configs_.find(key) != configs_.end();
    }

private:
    std::unordered_map<std::string, ConfigValue> configs_;
};

} // namespace npu::tile_fwk