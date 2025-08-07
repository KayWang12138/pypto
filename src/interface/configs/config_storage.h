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
#include "interface/utils/common.h"
#include "interface/cache/hash_buffer.h"

namespace npu::tile_fwk {

/* RuntimeConfig KEYS*/
const std::string PARALLEL_THRESHOLD = "parallel_threshold";
const std::string CYCLE_UPPER_BOUND = "cycle_upper_bound";
const std::string USE_NODE_HASH = "use_node_hash";
const std::string CYCLES_THRESHOLD = "cycles_threshold";
const std::string DB_TYPE = "db_type";
const std::string NBUFFER_NUM = "nbuffer_num";
const std::string L1_REUSE = "l1_reuse";
const std::string L1_REUSE_MAP = "l1_reuse_map";
const std::string CUBE_NBUFFER = "cube_nbuffer";
const std::string CUBE_NBUFFER_MAP = "cube_nbuffer_map";
const std::string LOAD_BALANCE = "load_balance";
const std::string COPYIN_THRESHOLD = "copyin_threshold";
const std::string MACHINE_CONFIG = "machine_config";

class ConfigStorage {
public:
    using ConfigValue = std::variant<int, bool, std::string, std::map<int, int>, uint8_t>;
    explicit ConfigStorage() {
        Reset();
    }

    void Reset() {
        configs_[PARALLEL_THRESHOLD] = 20;  // default threshold
        configs_[CYCLE_UPPER_BOUND] = 10000; // defalt cycle upper bound
        configs_[USE_NODE_HASH] = false;
        configs_[CYCLES_THRESHOLD] = 512; // default cycle threshold
        configs_[DB_TYPE] = 0;
        configs_[NBUFFER_NUM] = 1;
        configs_[L1_REUSE] = 0;
        configs_[L1_REUSE_MAP] = std::map<int,int>({});
        configs_[CUBE_NBUFFER] = 1;
        configs_[CUBE_NBUFFER_MAP] = std::map<int,int>({});
        configs_[LOAD_BALANCE] = false;
        configs_[COPYIN_THRESHOLD] = 1024 * 1024; // default copyin threshold
        configs_[MACHINE_CONFIG] = static_cast<uint8_t>(0);
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