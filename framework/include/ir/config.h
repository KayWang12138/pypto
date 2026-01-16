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
 * \file config.h
 * \brief Configuration management for IR compilation process
 */

#pragma once

#include <map>
#include <string>
#include <vector>
#include <set>
#include <stdexcept>
#include <typeindex>
#include <typeinfo>
#include <any>

namespace pto {

// Maximum length for normalized config name
constexpr size_t MAX_CONFIG_NAME_LENGTH = 256;

class ConfigKey {
public:
    explicit ConfigKey(const std::string& key) {
        key_ = FromRawName(key);
    }
    explicit ConfigKey(const char* key) {
        key_ = FromRawName(key ? key : "CONFIG_INVALID_KEY");
    }
    const std::string& Get() const { return key_; }
    
    // Comparison operators for use in std::set and std::map
    bool operator<(const ConfigKey& other) const {
        return key_ < other.key_;
    }
    
    bool operator==(const ConfigKey& other) const {
        return key_ == other.key_;
    }
    
    bool operator!=(const ConfigKey& other) const {
        return key_ != other.key_;
    }

private:
    static std::string FromRawName(const std::string& keyName);
    
    std::string key_;
};

class ConfigRegistry {
public:
    static ConfigRegistry& GetInstance();

    template<typename T>
    ConfigKey Register(const std::string& keyName, const T& defaultValue);

    bool IsRegistered(const ConfigKey& key) const;

    std::string GetKeyName(const ConfigKey& key) const { return key.Get(); }

    ConfigKey GetKeyByName(const std::string& name) const { return ConfigKey(name); }

    std::type_index GetTypeIndex(const ConfigKey& key) const;

    template<typename T>
    const T& GetDefaultValue(const ConfigKey& key) const {
        if (!IsRegistered(key)) {
            throw std::runtime_error("Config key '" + key.Get() + "' is not registered");
        }
        auto it = defaultValues_.find(key);
        if (it == defaultValues_.end()) {
            throw std::runtime_error("Config key '" + key.Get() + "' has no default value");
        }
        
        std::type_index expectedType = std::type_index(typeid(T));
        std::type_index actualType = std::type_index(it->second.type());
        if (actualType != expectedType) {
            throw std::runtime_error("Config key '" + key.Get() + "' default value type mismatch. Expected: " + 
                expectedType.name() + ", Got: " + actualType.name());
        }
        
        return std::any_cast<const T&>(it->second);
    }

    // Get the default value as std::any for a registered key
    std::any GetDefaultValueAny(const ConfigKey& key) const {
        if (!IsRegistered(key)) {
            throw std::runtime_error("Config key '" + key.Get() + "' is not registered");
        }
        auto it = defaultValues_.find(key);
        if (it == defaultValues_.end()) {
            throw std::runtime_error("Config key '" + key.Get() + "' has no default value");
        }
        return it->second;
    }
    
    // Set default value for a key
    void SetDefaultValue(const ConfigKey& key, const std::any& value) {
        defaultValues_[key] = value;
    }
    
    // Check if a key has a default value
    bool HasDefaultValue(const ConfigKey& key) const {
        return defaultValues_.find(key) != defaultValues_.end();
    }

    // Get all registered config keys
    std::vector<ConfigKey> GetAllRegisteredKeys() const;

private:
    ConfigRegistry() {}
    ~ConfigRegistry() = default;
    ConfigRegistry(const ConfigRegistry&) = delete;
    ConfigRegistry& operator=(const ConfigRegistry&) = delete;

    std::set<ConfigKey> registeredKeys_;
    std::map<ConfigKey, std::type_index> registeredConfigs_;
    std::map<ConfigKey, std::any> defaultValues_;
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
    const T& Get(const ConfigKey& key) const;

    // Set/Update config value by key with template type, throws if not initialized or type mismatch
    template<typename T>
    void Set(const ConfigKey& key, const T& value);

    // Load config from JSON file
    // JSON format: {"config_key_name1": value1, "config_key_name2": value2, ...}
    // Throws if file cannot be read, JSON is invalid, or type mismatch
    void LoadFromJsonFile(const std::string& filePath);

    // Check if a config key exists
    bool Has(const ConfigKey& key) const;

    // Check if config is initialized
    bool IsInitialized() const { return initialized_; }

private:
    std::map<ConfigKey, std::any> configs_;
    bool initialized_;
};


// Helper function to create a config map entry
// Usage: MakeConfigEntry(CONFIG_KEY, value)
template<typename T>
inline std::pair<ConfigKey, std::any> MakeConfigEntry(ConfigKey key, const T& value) {
    return std::make_pair(key, std::any(value));
}

// Template implementation for ConfigRegistry::Register
template<typename T>
ConfigKey ConfigRegistry::Register(const std::string& keyName, const T& defaultValue) {
    ConfigKey key(keyName);
    
    // Check if already registered
    if (registeredKeys_.find(key) != registeredKeys_.end()) {
        // Already registered, verify type matches
        auto configIt = registeredConfigs_.find(key);
        if (configIt != registeredConfigs_.end()) {
            std::type_index expectedType = std::type_index(typeid(T));
            if (configIt->second != expectedType) {
                throw std::runtime_error("Config key '" + keyName + 
                    "' is already registered with a different type");
            }
        }
        // Update default value if already registered
        defaultValues_[key] = std::any(defaultValue);
        return key;
    }

    // Register the config by directly inserting the key
    registeredKeys_.insert(key);
    registeredConfigs_.emplace(key, std::type_index(typeid(T)));
    defaultValues_[key] = std::any(defaultValue);

    return key;
}


// Template implementation for Config::Get
template<typename T>
const T& Config::Get(const ConfigKey& key) const {
    if (!initialized_) {
        throw std::runtime_error("Config is not initialized");
    }

    auto it = configs_.find(key);
    if (it == configs_.end()) {
        throw std::runtime_error("Config key '" + key.Get() + "' not found in config");
    }

    // Verify type matches
    std::type_index expectedType = std::type_index(typeid(T));
    std::type_index actualType = std::type_index(it->second.type());
    if (actualType != expectedType) {
        throw std::runtime_error("Config key '" + key.Get() + "' type mismatch. Expected: " + 
            expectedType.name() + ", Got: " + actualType.name());
    }

    return std::any_cast<const T&>(it->second);
}

// Template implementation for Config::Set
template<typename T>
void Config::Set(const ConfigKey& key, const T& value) {
    if (!initialized_) {
        throw std::runtime_error("Config is not initialized");
    }

    auto& registry = ConfigRegistry::GetInstance();

    // Check if key is registered
    if (!registry.IsRegistered(key)) {
        throw std::runtime_error("Config key '" + key.Get() + "' is not registered");
    }

    // Verify type matches
    std::type_index expectedType = registry.GetTypeIndex(key);
    std::type_index actualType = std::type_index(typeid(T));
    if (actualType != expectedType) {
        throw std::runtime_error("Config key '" + key.Get() + "' type mismatch. Expected: " + 
            expectedType.name() + ", Got: " + actualType.name());
    }

    // Set or update the value
    configs_[key] = std::any(value);
}
} // namespace pto

#define REGISTER_CONFIG(keyName, defaultValue) \
    namespace { \
        struct ConfigRegistrar_##keyName { \
            ConfigRegistrar_##keyName() { \
                auto& registry = pto::ConfigRegistry::GetInstance(); \
                registry.Register(#keyName, defaultValue); \
            } \
        }; \
        ConfigRegistrar_##keyName g_configRegistrar_##keyName; \
    } \
    namespace pto { \
        inline const ConfigKey CONFIG_##keyName(#keyName); \
    }
