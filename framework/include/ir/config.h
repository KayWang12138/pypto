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
 * \file config.h
 * \brief Configuration management for IR compilation process
 */

#pragma once

#include <map>
#include <string>
#include <stdexcept>
#include <typeindex>
#include <typeinfo>
#include <any>

namespace pto {

// Configuration key enumeration
enum class ConfigKey : int {
    CONFIG_INVALID = 0,
    // Enum values are generated automatically by REGISTER_CONFIG macro
};

// Forward declaration
class ConfigRegistry;

// Configuration registry singleton for managing config item definitions
class ConfigRegistry {
public:
    static ConfigRegistry& GetInstance();

    // Register a config item with template type and return its generated enum key
    template<typename T>
    ConfigKey Register(const std::string& keyName, const T& defaultValue, ConfigKey key = ConfigKey::CONFIG_INVALID);

    // Check if a config key is registered
    bool IsRegistered(ConfigKey key) const;

    // Get the string name of a config key
    std::string GetKeyName(ConfigKey key) const;

    // Get the config key by string name
    ConfigKey GetKeyByName(const std::string& name) const;

    // Get the type index for a registered key
    std::type_index GetTypeIndex(ConfigKey key) const;

private:
    ConfigRegistry() = default;
    ~ConfigRegistry() = default;
    ConfigRegistry(const ConfigRegistry&) = delete;
    ConfigRegistry& operator=(const ConfigRegistry&) = delete;

    std::map<ConfigKey, std::type_index> registeredConfigs_;
    std::map<ConfigKey, std::string> keyNames_;
    std::map<std::string, ConfigKey> nameToKey_;
    int nextKeyValue_ = static_cast<int>(ConfigKey::CONFIG_INVALID) + 1;
    
    // Helper to compute hash for key name (same algorithm as macro)
    static int ComputeKeyHash(const std::string& keyName);
};

// Configuration container class using type erasure
class Config {
public:
    Config() : initialized_(false) {}
    ~Config() = default;

    // Initialize config with initializer list, can only be called once
    // Usage: config.Initialize({{CONFIG_KEY1, std::any(value1)}, {CONFIG_KEY2, std::any(value2)}})
    // Note: Values must be wrapped in std::any, type checking happens during initialization
    void Initialize(std::initializer_list<std::pair<ConfigKey, std::any>> configs);

    // Get config value by key with template type, throws if not found or type mismatch
    template<typename T>
    const T& Get(ConfigKey key) const;

    // Check if a config key exists
    bool Has(ConfigKey key) const;

    // Check if config is initialized
    bool IsInitialized() const { return initialized_; }

private:
    std::map<ConfigKey, std::any> configs_;
    bool initialized_;
};

// Helper function to convert key name to enum constant name
inline std::string ToEnumName(const std::string& keyName) {
    std::string result = "CONFIG_";
    for (char c : keyName) {
        if (c >= 'a' && c <= 'z') {
            result += static_cast<char>(c - 'a' + 'A');
        } else if (c >= 'A' && c <= 'Z') {
            result += c;
        } else if (c >= '0' && c <= '9') {
            result += c;
        } else {
            // Convert underscore and other characters to underscore
            result += '_';
        }
    }
    return result;
}


// Helper function to create a config map entry
// Usage: MakeConfigEntry(CONFIG_KEY, value)
template<typename T>
inline std::pair<ConfigKey, std::any> MakeConfigEntry(ConfigKey key, const T& value) {
    return std::make_pair(key, std::any(value));
}

// Template implementation for ConfigRegistry::Register
template<typename T>
ConfigKey ConfigRegistry::Register(const std::string& keyName, const T& /* defaultValue */, ConfigKey key) {
    // If key is not provided, compute it from name hash
    if (key == ConfigKey::CONFIG_INVALID) {
        key = static_cast<ConfigKey>(ComputeKeyHash(keyName));
    }

    // Check if already registered by name
    auto nameIt = nameToKey_.find(keyName);
    if (nameIt != nameToKey_.end()) {
        // Already registered, verify type matches and key matches
        ConfigKey existingKey = nameIt->second;
        if (existingKey != key) {
            throw std::runtime_error("Config key '" + keyName + 
                "' is already registered with a different enum value");
        }
        std::type_index expectedType = std::type_index(typeid(T));
        auto configIt = registeredConfigs_.find(existingKey);
        if (configIt != registeredConfigs_.end() && configIt->second != expectedType) {
            throw std::runtime_error("Config key '" + keyName + 
                "' is already registered with a different type");
        }
        return existingKey;
    }

    // Check if key is already used by a different name
    auto keyIt = keyNames_.find(key);
    if (keyIt != keyNames_.end()) {
        throw std::runtime_error("Config enum value is already used by key '" + 
            keyIt->second + "', cannot register '" + keyName + "'");
    }

    // Register the config
    registeredConfigs_.emplace(key, std::type_index(typeid(T)));
    keyNames_.emplace(key, keyName);
    nameToKey_.emplace(keyName, key);

    return key;
}

// Implementation for Config::Initialize with initializer list
inline void Config::Initialize(std::initializer_list<std::pair<ConfigKey, std::any>> configs) {
    if (initialized_) {
        throw std::runtime_error("Config is already initialized");
    }

    auto& registry = ConfigRegistry::GetInstance();

    // Validate all configs are registered and types match
    for (const auto& [key, value] : configs) {
        if (!registry.IsRegistered(key)) {
            std::string keyName = registry.GetKeyName(key);
            throw std::runtime_error("Config key '" + keyName + "' is not registered");
        }

        std::type_index expectedType = registry.GetTypeIndex(key);
        std::type_index actualType = std::type_index(value.type());
        if (expectedType != actualType) {
            std::string keyName = registry.GetKeyName(key);
            throw std::runtime_error("Config key '" + keyName + "' type mismatch. Expected: " + 
                expectedType.name() + ", Got: " + actualType.name());
        }
    }

    // Store all values
    configs_ = std::map<ConfigKey, std::any>(configs.begin(), configs.end());
    initialized_ = true;
}

// Template implementation for Config::Get
template<typename T>
const T& Config::Get(ConfigKey key) const {
    if (!initialized_) {
        throw std::runtime_error("Config is not initialized");
    }

    auto it = configs_.find(key);
    if (it == configs_.end()) {
        auto& registry = ConfigRegistry::GetInstance();
        std::string keyName = registry.GetKeyName(key);
        throw std::runtime_error("Config key '" + keyName + "' not found in config");
    }

    // Verify type matches
    std::type_index expectedType = std::type_index(typeid(T));
    std::type_index actualType = std::type_index(it->second.type());
    if (actualType != expectedType) {
        auto& registry = ConfigRegistry::GetInstance();
        std::string keyName = registry.GetKeyName(key);
        throw std::runtime_error("Config key '" + keyName + "' type mismatch. Expected: " + 
            expectedType.name() + ", Got: " + actualType.name());
    }

    return std::any_cast<const T&>(it->second);
}

// Helper function to compute hash for config key name (outside namespace for macro use)
// This ensures the same name always generates the same enum value
// Returns a value >= 1 (CONFIG_INVALID is 0)
// Using recursive constexpr function
constexpr int ComputeConfigKeyHashImpl(const char* str, int hash = 5381) {
    return (*str == '\0') ? ((hash % 1000000) + 1)
                          : ComputeConfigKeyHashImpl(str + 1, hash * 33 + *str);
}

} // namespace pto

// Macro to register a config item
// Usage: REGISTER_CONFIG(max_iterations, 100)
// This will generate CONFIG_max_iterations that can be used directly as enum value
#define REGISTER_CONFIG(keyName, defaultValue) \
    namespace { \
        struct ConfigRegistrar_##keyName { \
            ConfigRegistrar_##keyName() { \
                auto& registry = pto::ConfigRegistry::GetInstance(); \
                registry.Register(#keyName, defaultValue, static_cast<pto::ConfigKey>(pto::ComputeConfigKeyHashImpl(#keyName))); \
            } \
        }; \
        static ConfigRegistrar_##keyName g_configRegistrar_##keyName; \
    } \
    namespace pto { \
        inline const ConfigKey CONFIG_##keyName = static_cast<ConfigKey>(ComputeConfigKeyHashImpl(#keyName)); \
    }
