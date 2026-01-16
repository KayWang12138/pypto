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
 * \file config.cpp
 * \brief Implementation of configuration management for IR compilation process
 */

#include "ir/config.h"

#include <stdexcept>

namespace pto {

// Static member definition for ConfigKey
std::map<ConfigKey, std::any> ConfigKey::defaultValues_;

ConfigRegistry& ConfigRegistry::GetInstance() {
    static ConfigRegistry instance;
    return instance;
}


bool ConfigRegistry::IsRegistered(const ConfigKey& key) const {
    return registeredKeys_.find(key) != registeredKeys_.end();
}

std::type_index ConfigRegistry::GetTypeIndex(const ConfigKey& key) const {
    auto it = registeredConfigs_.find(key);
    if (it == registeredConfigs_.end()) {
        throw std::runtime_error("Config key '" + key.Get() + "' not found. Make sure to register it first using REGISTER_CONFIG macro.");
    }
    return it->second;
}

std::vector<ConfigKey> ConfigRegistry::GetAllRegisteredKeys() const {
    std::vector<ConfigKey> keys;
    keys.reserve(registeredKeys_.size());
    for (const auto& key : registeredKeys_) {
        keys.push_back(key);
    }
    return keys;
}

bool Config::Has(const ConfigKey& key) const {
    if (!initialized_) {
        return false;
    }
    return configs_.find(key) != configs_.end();
}

} // namespace pto
