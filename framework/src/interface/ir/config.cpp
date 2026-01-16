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
#include <fstream>
#include <nlohmann/json.hpp>

namespace pto {

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
    return configs_.find(key) != configs_.end();
}

std::string ConfigKey::FromRawName(const std::string& keyName) {
    std::string result;
    result.reserve(keyName.length());
    
    for (char c : keyName) {
        if (c >= 'A' && c <= 'Z') {
            result += static_cast<char>(c - 'A' + 'a');
        } else if (c >= 'a' && c <= 'z') {
            result += c;
        } else if (c >= '0' && c <= '9') {
            result += c;
        } else {
            result += '_';
        }
    }

    if (result.length() > MAX_CONFIG_NAME_LENGTH) {
        throw std::runtime_error("Config key name '" + keyName + 
            "' results in normalized name that is too long (" + std::to_string(result.length()) + 
            " characters). Maximum length is " + std::to_string(MAX_CONFIG_NAME_LENGTH) + " characters.");
    }
    
    return result;
}

// Implementation for Config::Initialize
void Config::Initialize(std::initializer_list<std::pair<ConfigKey, std::any>> configs) {
    auto& registry = ConfigRegistry::GetInstance();

    // Validate all provided configs and store them
    for (const auto& [key, value] : configs) {
        if (!registry.IsRegistered(key)) {
            throw std::runtime_error("Config key '" + key.Get() + "' is not registered");
        }

        std::type_index expectedType = registry.GetTypeIndex(key);
        std::type_index actualType = std::type_index(value.type());
        if (expectedType != actualType) {
            throw std::runtime_error("Config key '" + key.Get() + "' type mismatch. Expected: " + 
                expectedType.name() + ", Got: " + actualType.name());
        }

        configs_[key] = value;
    }
}

namespace {
// Helper function to convert JSON value to std::any based on type
std::any ConvertJsonValue(const nlohmann::json& j, const std::type_index& expectedType) {
    if (expectedType == std::type_index(typeid(int32_t))) {
        if (j.is_number_integer()) {
            return std::any(j.get<int32_t>());
        }
    }
    else if (expectedType == std::type_index(typeid(bool))) {
        if (j.is_boolean()) {
            return std::any(j.get<bool>());
        }
    }
    else if (expectedType == std::type_index(typeid(uint8_t))) {
        if (j.is_number_unsigned()) {
            return std::any(static_cast<uint8_t>(j.get<uint64_t>()));
        }
    }
    else if (expectedType == std::type_index(typeid(uint16_t))) {
        if (j.is_number_unsigned()) {
            return std::any(static_cast<uint16_t>(j.get<uint64_t>()));
        }
    }
    else if (expectedType == std::type_index(typeid(int64_t))) {
        if (j.is_number_integer()) {
            return std::any(j.get<int64_t>());
        }
    }
    else if (expectedType == std::type_index(typeid(std::map<int64_t, int64_t>))) {
        if (j.is_object()) {
            std::map<int64_t, int64_t> mapValue;
            for (auto& [key, val] : j.items()) {
                int64_t mapKey = std::stoll(key);
                if (val.is_number_integer()) {
                    mapValue[mapKey] = val.get<int64_t>();
                } else {
                    throw std::runtime_error("Map value must be an integer");
                }
            }
            return std::any(mapValue);
        }
    }
    else if (expectedType == std::type_index(typeid(std::string))) {
        if (j.is_string()) {
            return std::any(j.get<std::string>());
        }
    }

    throw std::runtime_error("Type conversion not supported or JSON type mismatch");
}
} // anonymous namespace

// Implementation for Config::LoadFromJsonFile
void Config::LoadFromJsonFile(const std::string& filePath) {
    std::ifstream ifs(filePath);
    if (!ifs.is_open()) {
        throw std::runtime_error("Failed to open JSON file: " + filePath);
    }

    nlohmann::json jsonObj;
    try {
        ifs >> jsonObj;
        ifs.close();
    } catch (const std::exception& e) {
        ifs.close();
        throw std::runtime_error("Failed to parse JSON file: " + filePath + ", error: " + e.what());
    }

    if (!jsonObj.is_object()) {
        throw std::runtime_error("JSON file must contain an object: " + filePath);
    }

    auto& registry = ConfigRegistry::GetInstance();

    // Process each key-value pair in JSON
    for (auto& [keyName, value] : jsonObj.items()) {
        try {
            ConfigKey key(keyName);
            std::type_index expectedType = registry.GetTypeIndex(key);
            std::any configValue = ConvertJsonValue(value, expectedType);
            configs_[key] = configValue;
        } catch (const std::runtime_error& e) {
            // If key is not registered or conversion fails, skip it
            continue;
        }
    }

    std::vector<ConfigKey> allRegisteredKeys = registry.GetAllRegisteredKeys();
    for (ConfigKey key : allRegisteredKeys) {
        if (configs_.find(key) == configs_.end()) {
            try {
                std::any defaultValue = registry.GetDefaultValueAny(key);
                configs_[key] = defaultValue;
            } catch (const std::runtime_error&) {
                // If no default value exists, skip this key
            }
        }
    }
}

} // namespace pto
