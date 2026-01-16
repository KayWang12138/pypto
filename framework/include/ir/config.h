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
#include <fstream>
#include <stdexcept>
#include <typeindex>
#include <typeinfo>
#include <any>
#include <nlohmann/json.hpp>

namespace pto {

// Maximum length for normalized config name
constexpr size_t MAX_CONFIG_NAME_LENGTH = 256;

// Configuration key class
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
    
    // Get default value for this key
    template<typename T>
    const T& GetDefaultValue() const {
        auto it = defaultValues_.find(*this);
        if (it == defaultValues_.end()) {
            throw std::runtime_error("Config key '" + key_ + "' has no default value");
        }
        
        // Verify type matches
        std::type_index expectedType = std::type_index(typeid(T));
        std::type_index actualType = std::type_index(it->second.type());
        if (actualType != expectedType) {
            throw std::runtime_error("Config key '" + key_ + "' default value type mismatch. Expected: " + 
                expectedType.name() + ", Got: " + actualType.name());
        }
        
        return std::any_cast<const T&>(it->second);
    }
    
    // Get default value as std::any for this key
    std::any GetDefaultValueAny() const {
        auto it = defaultValues_.find(*this);
        if (it == defaultValues_.end()) {
            throw std::runtime_error("Config key '" + key_ + "' has no default value");
        }
        return it->second;
    }
    
    // Set default value for this key
    template<typename T>
    void SetDefaultValue(const T& value) {
        defaultValues_[*this] = std::any(value);
    }
    
    // Set default value as std::any for this key
    void SetDefaultValueAny(const std::any& value) {
        defaultValues_[*this] = value;
    }
    
    // Check if this key has a default value
    bool HasDefaultValue() const {
        return defaultValues_.find(*this) != defaultValues_.end();
    }

private:
    // Converts all letters to lowercase, special characters to underscore, with max length 256
    static std::string FromRawName(const std::string& keyName) {
        std::string result;
        result.reserve(keyName.length());
        
        for (char c : keyName) {
            if (c >= 'A' && c <= 'Z') {
                // Convert uppercase to lowercase
                result += static_cast<char>(c - 'A' + 'a');
            } else if (c >= 'a' && c <= 'z') {
                // Keep lowercase letters
                result += c;
            } else if (c >= '0' && c <= '9') {
                // Keep digits
                result += c;
            } else {
                // Convert special characters to underscore
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
    
    std::string key_;
    
    // Static storage for default values: one default value per unique ConfigKey
    static std::map<ConfigKey, std::any> defaultValues_;
};

class ConfigRegistry {
public:
    static ConfigRegistry& GetInstance();

    // Register a config item with template type and return its key name
    template<typename T>
    ConfigKey Register(const std::string& keyName, const T& defaultValue);

    // Check if a config key is registered
    bool IsRegistered(const ConfigKey& key) const;

    // Get the string name of a config key
    std::string GetKeyName(const ConfigKey& key) const { return key.Get(); }

    // Get the config key by string name
    ConfigKey GetKeyByName(const std::string& name) const { return ConfigKey(name); }

    // Get the type index for a registered key
    std::type_index GetTypeIndex(const ConfigKey& key) const;

    // Get the default value for a registered key with template type
    template<typename T>
    const T& GetDefaultValue(const ConfigKey& key) const {
        if (!IsRegistered(key)) {
            throw std::runtime_error("Config key '" + key.Get() + "' is not registered");
        }
        return key.GetDefaultValue<T>();
    }

    // Get the default value as std::any for a registered key
    std::any GetDefaultValueAny(const ConfigKey& key) const {
        if (!IsRegistered(key)) {
            throw std::runtime_error("Config key '" + key.Get() + "' is not registered");
        }
        return key.GetDefaultValueAny();
    }

    // Get all registered config keys
    std::vector<ConfigKey> GetAllRegisteredKeys() const;

private:
    ConfigRegistry() = default;
    ~ConfigRegistry() = default;
    ConfigRegistry(const ConfigRegistry&) = delete;
    ConfigRegistry& operator=(const ConfigRegistry&) = delete;

    std::set<ConfigKey> registeredKeys_;
    std::map<ConfigKey, std::type_index> registeredConfigs_;
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
        key.SetDefaultValueAny(std::any(defaultValue));
        return key;
    }

    // Register the config by directly inserting the key
    registeredKeys_.insert(key);
    registeredConfigs_.emplace(key, std::type_index(typeid(T)));
    key.SetDefaultValueAny(std::any(defaultValue));

    return key;
}

// Implementation for Config::Initialize with initializer list
inline void Config::Initialize(std::initializer_list<std::pair<ConfigKey, std::any>> configs) {
    if (initialized_) {
        throw std::runtime_error("Config is already initialized");
    }

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

    initialized_ = true;
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

// Implementation for Config::LoadFromJsonFile
inline void Config::LoadFromJsonFile(const std::string& filePath) {
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

    // Helper function to convert JSON value to std::any based on type
    auto convertJsonValue = [](const nlohmann::json& j, const std::type_index& expectedType) -> std::any {
        // Check for int32_t
        if (expectedType == std::type_index(typeid(int32_t))) {
            if (j.is_number_integer()) {
                return std::any(j.get<int32_t>());
            }
        }
        // Check for bool
        else if (expectedType == std::type_index(typeid(bool))) {
            if (j.is_boolean()) {
                return std::any(j.get<bool>());
            }
        }
        // Check for uint8_t
        else if (expectedType == std::type_index(typeid(uint8_t))) {
            if (j.is_number_unsigned()) {
                return std::any(static_cast<uint8_t>(j.get<uint64_t>()));
            }
        }
        // Check for uint16_t
        else if (expectedType == std::type_index(typeid(uint16_t))) {
            if (j.is_number_unsigned()) {
                return std::any(static_cast<uint16_t>(j.get<uint64_t>()));
            }
        }
        // Check for int64_t
        else if (expectedType == std::type_index(typeid(int64_t))) {
            if (j.is_number_integer()) {
                return std::any(j.get<int64_t>());
            }
        }
        // Check for std::map<int64_t, int64_t>
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
        // Check for std::string
        else if (expectedType == std::type_index(typeid(std::string))) {
            if (j.is_string()) {
                return std::any(j.get<std::string>());
            }
        }

        throw std::runtime_error("Type conversion not supported or JSON type mismatch");
    };

    // Process each key-value pair in JSON
    for (auto& [keyName, value] : jsonObj.items()) {
        try {
            // Use key name directly as the key
            ConfigKey key(keyName);
            
            // Get expected type
            std::type_index expectedType = registry.GetTypeIndex(key);
            
            // Convert JSON value to std::any
            std::any configValue = convertJsonValue(value, expectedType);
            
            // Store the value (will update if key exists, add if new)
            configs_[key] = configValue;
        } catch (const std::runtime_error& e) {
            // If key is not registered or conversion fails, skip it
            // (unknown keys are silently ignored)
            continue;
        }
    }

    // Mark as initialized if not already
    // If already initialized, this just allows updates from JSON
    if (!initialized_) {
        // Add default values for all registered keys that were not provided in JSON
        std::vector<ConfigKey> allRegisteredKeys = registry.GetAllRegisteredKeys();
        for (ConfigKey key : allRegisteredKeys) {
            // If this key was not provided in JSON, use the default value
            if (configs_.find(key) == configs_.end()) {
                try {
                    std::any defaultValue = key.GetDefaultValueAny();
                    configs_[key] = defaultValue;
                } catch (const std::runtime_error&) {
                    // If no default value exists, skip this key
                    // (it will throw when accessed via Get())
                }
            }
        }
        initialized_ = true;
    }
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
                registry.Register(#keyName, defaultValue); \
            } \
        }; \
        ConfigRegistrar_##keyName g_configRegistrar_##keyName; \
    } \
    namespace pto { \
        inline const ConfigKey CONFIG_##keyName(#keyName); \
    }
