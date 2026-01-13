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

ConfigRegistry& ConfigRegistry::GetInstance() {
    static ConfigRegistry instance;
    return instance;
}

size_t ConfigRegistry::ComputeKeyHash(const std::string& keyName) {
    return ComputeConfigKeyHashImpl(keyName.c_str());
}

bool ConfigRegistry::IsRegistered(ConfigKey key) const {
    return registeredConfigs_.find(key) != registeredConfigs_.end();
}

std::string ConfigRegistry::GetKeyName(ConfigKey key) const {
    auto it = keyNames_.find(key);
    if (it == keyNames_.end()) {
        throw std::runtime_error("Config key not found");
    }
    return it->second;
}

ConfigKey ConfigRegistry::GetKeyByName(const std::string& name) const {
    auto it = nameToKey_.find(name);
    if (it == nameToKey_.end()) {
        throw std::runtime_error("Config key name '" + name + "' not found. Make sure to register it first using REGISTER_CONFIG macro.");
    }
    return it->second;
}

std::type_index ConfigRegistry::GetTypeIndex(ConfigKey key) const {
    auto it = registeredConfigs_.find(key);
    if (it == registeredConfigs_.end()) {
        throw std::runtime_error("Config key not found");
    }
    return it->second;
}

bool Config::Has(ConfigKey key) const {
    if (!initialized_) {
        return false;
    }
    return configs_.find(key) != configs_.end();
}

} // namespace pto
