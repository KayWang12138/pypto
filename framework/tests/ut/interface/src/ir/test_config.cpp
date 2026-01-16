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
 * \file test_config.cpp
 * \brief Unit tests for Config class
 */

#include "gtest/gtest.h"
#include "ir/config.h"
#include "ir/program.h"
#include "ir/function.h"
#include <fstream>
#include <cstdio>

// Register test config items
REGISTER_CONFIG(test_int32, int32_t(100));
REGISTER_CONFIG(test_bool, true);
REGISTER_CONFIG(test_uint8, uint8_t(42));
REGISTER_CONFIG(test_uint16, uint16_t(1000));
REGISTER_CONFIG(test_map, (std::map<int64_t, int64_t>{
                              {1, 2},
                              {3, 4}
}));

namespace pto {
class ConfigTest : public testing::Test {
public:
    void SetUp() override {
        // Reset registry state if needed (in real scenario, registry is singleton)
    }

    void TearDown() override {}
};

// Test config registration
TEST_F(ConfigTest, TestConfigRegistration) {
    auto &registry = ConfigRegistry::GetInstance();

    // Test that registered configs are found using enum values directly
    EXPECT_TRUE(registry.IsRegistered(CONFIG_test_int32));
    EXPECT_TRUE(registry.IsRegistered(CONFIG_test_bool));
    EXPECT_TRUE(registry.IsRegistered(CONFIG_test_uint8));

    // Test GetKeyName
    EXPECT_EQ("test_int32", registry.GetKeyName(CONFIG_test_int32));
    EXPECT_EQ("test_bool", registry.GetKeyName(CONFIG_test_bool));

    // Test GetKeyByName
    EXPECT_EQ(CONFIG_test_int32, registry.GetKeyByName("test_int32"));
    EXPECT_EQ(CONFIG_test_bool, registry.GetKeyByName("test_bool"));

    // Test GetTypeIndex
    std::type_index int32Type = std::type_index(typeid(int32_t));
    EXPECT_EQ(int32Type, registry.GetTypeIndex(CONFIG_test_int32));

    std::type_index boolType = std::type_index(typeid(bool));
    EXPECT_EQ(boolType, registry.GetTypeIndex(CONFIG_test_bool));
}

// Test config initialization with initializer list
TEST_F(ConfigTest, TestConfigInitializeWithInitializerList) {
    Config config;

    config.Initialize({
        {CONFIG_test_int32, std::any(int32_t(300))},
        { CONFIG_test_bool,         std::any(true)},
        {CONFIG_test_uint8,  std::any(uint8_t(88))}
    });

    EXPECT_EQ(300, config.Get<int32_t>(CONFIG_test_int32));
    EXPECT_TRUE(config.Get<bool>(CONFIG_test_bool));
    EXPECT_EQ(88, config.Get<uint8_t>(CONFIG_test_uint8));
}

// Test config Get and Has
TEST_F(ConfigTest, TestConfigGetAndHas) {
    Config config;

    // Config constructor automatically initializes all registered keys with default values
    // So all registered keys should exist
    EXPECT_TRUE(config.Has(CONFIG_test_int32));
    EXPECT_TRUE(config.Has(CONFIG_test_bool));
    EXPECT_TRUE(config.Has(CONFIG_test_uint8));
    EXPECT_TRUE(config.Has(CONFIG_test_uint16));
    EXPECT_TRUE(config.Has(CONFIG_test_map));

    // Verify default values
    EXPECT_EQ(100, config.Get<int32_t>(CONFIG_test_int32));
    EXPECT_TRUE(config.Get<bool>(CONFIG_test_bool));
    EXPECT_EQ(42, config.Get<uint8_t>(CONFIG_test_uint8));
    EXPECT_EQ(1000, config.Get<uint16_t>(CONFIG_test_uint16));

    // Now override some values using Initialize
    config.Initialize({
        {CONFIG_test_int32, std::any(int32_t(200))},
        { CONFIG_test_bool,        std::any(false)}
    });

    // Verify overridden values
    EXPECT_EQ(200, config.Get<int32_t>(CONFIG_test_int32));
    EXPECT_FALSE(config.Get<bool>(CONFIG_test_bool));
    // Other keys should still have default values
    EXPECT_EQ(42, config.Get<uint8_t>(CONFIG_test_uint8));
}

// Test config with map type
TEST_F(ConfigTest, TestConfigWithMapType) {
    Config config;

    std::map<int64_t, int64_t> testMap = {
        {1, 10},
        {2, 20},
        {3, 30}
    };
    config.Initialize({
        {CONFIG_test_map, std::any(testMap)}
    });

    const auto &retrievedMap = config.Get<std::map<int64_t, int64_t>>(CONFIG_test_map);
    EXPECT_EQ(3, retrievedMap.size());
    EXPECT_EQ(10, retrievedMap.at(1));
    EXPECT_EQ(20, retrievedMap.at(2));
    EXPECT_EQ(30, retrievedMap.at(3));
}

// Test config type mismatch error
TEST_F(ConfigTest, TestConfigTypeMismatchError) {
    Config config;

    // Try to initialize with wrong type
    EXPECT_THROW(
        {
            config.Initialize({
                {CONFIG_test_int32, std::any(true)}  // Should be int32_t, not bool
            });
        },
        std::runtime_error);
}

// Test config unregistered key error
TEST_F(ConfigTest, TestConfigUnregisteredKeyError) {
    Config config;

    // Create an unregistered key
    ConfigKey unregisteredKey = static_cast<ConfigKey>("unregistered_key");

    EXPECT_THROW(
        {
            config.Initialize({
                {unregisteredKey, std::any(int32_t(100))}
            });
        },
        std::runtime_error);
}

// Test config constructor automatically initializes with default values
TEST_F(ConfigTest, TestConfigConstructorInitializesDefaults) {
    Config config;

    // All registered keys should be automatically initialized with default values
    EXPECT_FALSE(config.IsEmpty());
    EXPECT_TRUE(config.Has(CONFIG_test_int32));
    EXPECT_TRUE(config.Has(CONFIG_test_bool));
    EXPECT_TRUE(config.Has(CONFIG_test_uint8));
    EXPECT_TRUE(config.Has(CONFIG_test_uint16));
    EXPECT_TRUE(config.Has(CONFIG_test_map));

    // Verify all default values
    EXPECT_EQ(100, config.Get<int32_t>(CONFIG_test_int32));
    EXPECT_TRUE(config.Get<bool>(CONFIG_test_bool));
    EXPECT_EQ(42, config.Get<uint8_t>(CONFIG_test_uint8));
    EXPECT_EQ(1000, config.Get<uint16_t>(CONFIG_test_uint16));

    // Verify map default value
    const auto &defaultMap = config.Get<std::map<int64_t, int64_t>>(CONFIG_test_map);
    EXPECT_EQ(2, defaultMap.size());
    EXPECT_EQ(2, defaultMap.at(1));
    EXPECT_EQ(4, defaultMap.at(3));
}

// Test config get wrong type error
TEST_F(ConfigTest, TestConfigGetWrongTypeError) {
    Config config;

    config.Initialize({
        {CONFIG_test_int32, std::any(int32_t(100))}
    });

    // Try to get as wrong type
    EXPECT_THROW(
        {
            config.Get<bool>(CONFIG_test_int32); // Should be int32_t, not bool
        },
        std::runtime_error);
}

// Test ProgramModule config
TEST_F(ConfigTest, TestProgramModuleConfig) {
    auto module = std::make_shared<ProgramModule>("test_module");

    // Config constructor automatically initializes with default values, so HasConfig() should be true
    EXPECT_TRUE(module->HasConfig());

    // Verify default values are present
    EXPECT_EQ(100, module->GetConfig().Get<int32_t>(CONFIG_test_int32));
    EXPECT_TRUE(module->GetConfig().Get<bool>(CONFIG_test_bool));

    // Override with Initialize
    module->GetConfig().Initialize({
        {CONFIG_test_int32, std::any(int32_t(500))},
        { CONFIG_test_bool,        std::any(false)}
    });

    EXPECT_TRUE(module->HasConfig());
    EXPECT_EQ(500, module->GetConfig().Get<int32_t>(CONFIG_test_int32));
    EXPECT_FALSE(module->GetConfig().Get<bool>(CONFIG_test_bool));
}

// Test Function config
TEST_F(ConfigTest, TestFunctionConfig) {
    FunctionSignature sig;
    auto func = std::make_shared<Function>("test_func", FunctionKind::ControlFlow, sig);

    // Config constructor automatically initializes with default values, so HasConfig() should be true
    EXPECT_TRUE(func->HasConfig());

    // Verify default value is present
    EXPECT_EQ(100, func->GetConfig().Get<int32_t>(CONFIG_test_int32));

    // Override with Initialize
    func->GetConfig().Initialize({
        {CONFIG_test_int32, std::any(int32_t(600))}
    });

    EXPECT_TRUE(func->HasConfig());
    EXPECT_EQ(600, func->GetConfig().Get<int32_t>(CONFIG_test_int32));
}

// Test Function config cascade lookup
TEST_F(ConfigTest, TestFunctionConfigCascadeLookup) {
    // Create module and function
    auto module = std::make_shared<ProgramModule>("test_module");
    FunctionSignature sig;
    auto func = std::make_shared<Function>("test_func", FunctionKind::ControlFlow, sig);

    // Add function to module (this sets parentModule_)
    module->AddFunction(func);

    // Initialize module config
    module->GetConfig().Initialize({
        {CONFIG_test_int32, std::any(int32_t(100))},
        { CONFIG_test_bool,         std::any(true)}
    });

    // Initialize function config with one key that overrides module
    func->GetConfig().Initialize({
        {CONFIG_test_int32, std::any(int32_t(200))}  // Override module's value
    });

    // Function config should take precedence
    EXPECT_EQ(200, func->GetConfigValue<int32_t>(CONFIG_test_int32));

    // Function doesn't have this key, should get from module
    EXPECT_TRUE(func->GetConfigValue<bool>(CONFIG_test_bool));

    // Unregistered key should return default value 42
    EXPECT_EQ(42, func->GetConfigValue<uint8_t>(CONFIG_test_uint8));
}

// Test Function config cascade lookup without parent module
TEST_F(ConfigTest, TestFunctionConfigCascadeLookupWithoutParent) {
    FunctionSignature sig;
    auto func = std::make_shared<Function>("test_func", FunctionKind::ControlFlow, sig);

    // Function has no parent module
    func->GetConfig().Initialize({
        {CONFIG_test_int32, std::any(int32_t(300))}
    });

    // Should get from function config
    EXPECT_EQ(300, func->GetConfigValue<int32_t>(CONFIG_test_int32));

    // Unregistered key should return default value true
    EXPECT_TRUE(func->GetConfigValue<bool>(CONFIG_test_bool));
}

// Test MakeConfigEntry helper
TEST_F(ConfigTest, TestMakeConfigEntryHelper) {
    Config config;

    config.Initialize({MakeConfigEntry(CONFIG_test_int32, int32_t(400)), MakeConfigEntry(CONFIG_test_bool, true)});

    EXPECT_EQ(400, config.Get<int32_t>(CONFIG_test_int32));
    EXPECT_TRUE(config.Get<bool>(CONFIG_test_bool));
}

TEST_F(ConfigTest, TestFunctionGetConfigConst) {
    FunctionSignature sig;
    auto func = std::make_shared<Function>("test_func", FunctionKind::ControlFlow, sig);

    // Initialize config using non-const GetConfig()
    func->GetConfig().Initialize({
        {CONFIG_test_int32, std::any(int32_t(700))},
        { CONFIG_test_bool,        std::any(false)},
        {CONFIG_test_uint8,  std::any(uint8_t(99))}
    });

    // Test const GetConfig() - should return const reference
    const Function &constFunc = *func;
    const Config &constConfig = constFunc.GetConfig();

    // Verify const Config can be used for read operations
    EXPECT_TRUE(constConfig.Has(CONFIG_test_int32));
    EXPECT_TRUE(constConfig.Has(CONFIG_test_bool));
    EXPECT_TRUE(constConfig.Has(CONFIG_test_uint8));
    EXPECT_TRUE(constConfig.Has(CONFIG_test_uint16));

    // Test Get() method on const Config
    EXPECT_EQ(700, constConfig.Get<int32_t>(CONFIG_test_int32));
    EXPECT_FALSE(constConfig.Get<bool>(CONFIG_test_bool));
    EXPECT_EQ(99, constConfig.Get<uint8_t>(CONFIG_test_uint8));
    EXPECT_EQ(1000, constConfig.Get<uint16_t>(CONFIG_test_uint16));

    // Verify const Config reference points to the same object
    EXPECT_EQ(&constConfig, &func->GetConfig());
}

// Test Function::GetParentModule() method
TEST_F(ConfigTest, TestFunctionGetParentModule) {
    FunctionSignature sig;
    auto func = std::make_shared<Function>("test_func", FunctionKind::ControlFlow, sig);

    // Initially, function should have no parent module
    EXPECT_EQ(nullptr, func->GetParentModule());

    // Create a module and set it as parent using SetParentModule()
    auto module = std::make_shared<ProgramModule>("test_module");
    func->SetParentModule(module);

    // GetParentModule() should return the parent module
    auto parentModule = func->GetParentModule();
    EXPECT_NE(nullptr, parentModule);
    EXPECT_EQ(module, parentModule);
    EXPECT_EQ("test_module", parentModule->GetName());

    // Test with AddFunction() which also sets parent module
    auto module2 = std::make_shared<ProgramModule>("test_module2");
    auto func2 = std::make_shared<Function>("test_func2", FunctionKind::ControlFlow, sig);
    module2->AddFunction(func2);

    // GetParentModule() should return the parent module set by AddFunction()
    auto parentModule2 = func2->GetParentModule();
    EXPECT_NE(nullptr, parentModule2);
    EXPECT_EQ(module2, parentModule2);
    EXPECT_EQ("test_module2", parentModule2->GetName());

    // Test weak_ptr behavior: when parent module is released, GetParentModule() should return nullptr
    auto module3 = std::make_shared<ProgramModule>("test_module3");
    auto func3 = std::make_shared<Function>("test_func3", FunctionKind::ControlFlow, sig);
    func3->SetParentModule(module3);
    EXPECT_NE(nullptr, func3->GetParentModule());

    module3.reset();                              // Release the shared_ptr
    EXPECT_EQ(nullptr, func3->GetParentModule()); // Should return nullptr after parent is released
}

// Test loading config from JSON file
TEST_F(ConfigTest, TestLoadFromJsonFile) {
    Config config;
    std::string jsonPath = "../../../framework/tests/ut/interface/src/ir/json/test_config.json";
    config.LoadFromJsonFile(jsonPath);

    // Verify values loaded from JSON
    EXPECT_EQ(200, config.Get<int32_t>(CONFIG_test_int32));
    EXPECT_FALSE(config.Get<bool>(CONFIG_test_bool));
    EXPECT_EQ(88, config.Get<uint8_t>(CONFIG_test_uint8));
    EXPECT_EQ(2000, config.Get<uint16_t>(CONFIG_test_uint16));

    // Verify map value
    const auto &retrievedMap = config.Get<std::map<int64_t, int64_t>>(CONFIG_test_map);
    EXPECT_EQ(3, retrievedMap.size());
    EXPECT_EQ(10, retrievedMap.at(1));
    EXPECT_EQ(20, retrievedMap.at(2));
    EXPECT_EQ(30, retrievedMap.at(3));
}

// Test loading config from JSON file with partial configs
TEST_F(ConfigTest, TestLoadFromJsonFilePartial) {
    Config config;
    std::string jsonPath = "../../../framework/tests/ut/interface/src/ir/json/test_config_partial.json";
    config.LoadFromJsonFile(jsonPath);

    // Verify loaded values
    EXPECT_EQ(500, config.Get<int32_t>(CONFIG_test_int32));
    EXPECT_TRUE(config.Get<bool>(CONFIG_test_bool));

    // Verify default values are used for missing keys
    EXPECT_EQ(42, config.Get<uint8_t>(CONFIG_test_uint8));     // Should use default value
    EXPECT_EQ(1000, config.Get<uint16_t>(CONFIG_test_uint16)); // Should use default value
}

// Test loading config from non-existent JSON file
TEST_F(ConfigTest, TestLoadFromJsonFileNotFound) {
    Config config;
    std::string jsonPath = "../../../framework/tests/ut/interface/src/ir/json/test_nonexistent_file.json";
    EXPECT_THROW({ config.LoadFromJsonFile("non_existent_file.json"); }, std::runtime_error);
}

// Test SetDefault function
TEST_F(ConfigTest, TestSetDefault) {
    Config config;

    // Initially has default value
    EXPECT_EQ(100, config.Get<int32_t>(CONFIG_test_int32));
    EXPECT_TRUE(config.Get<bool>(CONFIG_test_bool));

    // Override with Set
    config.Set(CONFIG_test_int32, int32_t(500));
    config.Set(CONFIG_test_bool, false);

    // Verify overridden values
    EXPECT_EQ(500, config.Get<int32_t>(CONFIG_test_int32));
    EXPECT_FALSE(config.Get<bool>(CONFIG_test_bool));

    // Reset to default values using SetDefault
    config.SetDefault(CONFIG_test_int32);
    config.SetDefault(CONFIG_test_bool);

    // Verify default values are restored
    EXPECT_EQ(100, config.Get<int32_t>(CONFIG_test_int32));
    EXPECT_TRUE(config.Get<bool>(CONFIG_test_bool));
}

// Test SetDefault with unregistered key
TEST_F(ConfigTest, TestSetDefaultUnregisteredKey) {
    Config config;
    ConfigKey unregisteredKey("unregistered_key");

    EXPECT_THROW({ config.SetDefault(unregisteredKey); }, std::runtime_error);
}

// Test Set function (template version)
TEST_F(ConfigTest, TestSetFunction) {
    Config config;

    // Override default values using Set
    config.Set(CONFIG_test_int32, int32_t(777));
    config.Set(CONFIG_test_bool, false);
    config.Set(CONFIG_test_uint8, uint8_t(99));

    // Verify values are set correctly
    EXPECT_EQ(777, config.Get<int32_t>(CONFIG_test_int32));
    EXPECT_FALSE(config.Get<bool>(CONFIG_test_bool));
    EXPECT_EQ(99, config.Get<uint8_t>(CONFIG_test_uint8));

    // Other keys should still have default values
    EXPECT_EQ(1000, config.Get<uint16_t>(CONFIG_test_uint16));
}

// Test Set function with wrong type
TEST_F(ConfigTest, TestSetWrongType) {
    Config config;

    // Try to set with wrong type
    EXPECT_THROW({ config.Set(CONFIG_test_int32, true); }, std::runtime_error);        // bool instead of int32_t
    EXPECT_THROW({ config.Set(CONFIG_test_bool, int32_t(100)); }, std::runtime_error); // int32_t instead of bool
}

// Test Set function with unregistered key
TEST_F(ConfigTest, TestSetUnregisteredKey) {
    Config config;
    ConfigKey unregisteredKey("unregistered_key");

    EXPECT_THROW({ config.Set(unregisteredKey, int32_t(100)); }, std::runtime_error);
}

// Test Initialize with empty std::any (should use default value)
TEST_F(ConfigTest, TestInitializeWithEmptyAny) {
    Config config;

    // Initialize with empty std::any - should use default value
    config.Initialize({
        {CONFIG_test_int32,     std::any{}}, // Empty std::any
        { CONFIG_test_bool, std::any(true)}  // Normal value
    });

    // Empty std::any should result in default value
    EXPECT_EQ(100, config.Get<int32_t>(CONFIG_test_int32)); // Should use default value 100
    EXPECT_TRUE(config.Get<bool>(CONFIG_test_bool));        // Should use provided value true
}

// Test ConfigRegistry::GetDefaultValue (template version)
TEST_F(ConfigTest, TestConfigRegistryGetDefaultValue) {
    auto &registry = ConfigRegistry::GetInstance();

    // Test GetDefaultValue with correct type
    EXPECT_EQ(100, registry.GetDefaultValue<int32_t>(CONFIG_test_int32));
    EXPECT_TRUE(registry.GetDefaultValue<bool>(CONFIG_test_bool));
    EXPECT_EQ(42, registry.GetDefaultValue<uint8_t>(CONFIG_test_uint8));
    EXPECT_EQ(1000, registry.GetDefaultValue<uint16_t>(CONFIG_test_uint16));
}

// Test ConfigRegistry::GetDefaultValue with wrong type
TEST_F(ConfigTest, TestConfigRegistryGetDefaultValueWrongType) {
    auto &registry = ConfigRegistry::GetInstance();

    EXPECT_THROW({ registry.GetDefaultValue<bool>(CONFIG_test_int32); }, std::runtime_error);
    EXPECT_THROW({ registry.GetDefaultValue<int32_t>(CONFIG_test_bool); }, std::runtime_error);
}

// Test ConfigRegistry::HasDefaultValue
TEST_F(ConfigTest, TestConfigRegistryHasDefaultValue) {
    auto &registry = ConfigRegistry::GetInstance();

    // All registered keys should have default values
    EXPECT_TRUE(registry.HasDefaultValue(CONFIG_test_int32));
    EXPECT_TRUE(registry.HasDefaultValue(CONFIG_test_bool));
    EXPECT_TRUE(registry.HasDefaultValue(CONFIG_test_uint8));
    EXPECT_TRUE(registry.HasDefaultValue(CONFIG_test_uint16));
    EXPECT_TRUE(registry.HasDefaultValue(CONFIG_test_map));

    // Unregistered key should not have default value
    ConfigKey unregisteredKey("unregistered_key");
    EXPECT_FALSE(registry.HasDefaultValue(unregisteredKey));
}

// Test ConfigRegistry::SetDefaultValue
TEST_F(ConfigTest, TestConfigRegistrySetDefaultValue) {
    auto &registry = ConfigRegistry::GetInstance();

    // Get original default value
    int32_t originalValue = registry.GetDefaultValue<int32_t>(CONFIG_test_int32);
    EXPECT_EQ(100, originalValue);

    // Set new default value
    registry.SetDefaultValue(CONFIG_test_int32, std::any(int32_t(999)));

    // Verify new default value
    EXPECT_EQ(999, registry.GetDefaultValue<int32_t>(CONFIG_test_int32));

    // Create new Config instance - should use new default value
    Config config;
    EXPECT_EQ(999, config.Get<int32_t>(CONFIG_test_int32));

    // Restore original default value
    registry.SetDefaultValue(CONFIG_test_int32, std::any(int32_t(100)));
}

// Test ConfigRegistry::GetAllRegisteredKeys
TEST_F(ConfigTest, TestConfigRegistryGetAllRegisteredKeys) {
    auto &registry = ConfigRegistry::GetInstance();

    std::vector<ConfigKey> allKeys = registry.GetAllRegisteredKeys();

    // Should contain all registered keys
    EXPECT_GE(allKeys.size(), 5); // At least 5 registered keys

    // Verify specific keys are present
    bool foundInt32 = false;
    bool foundBool = false;
    bool foundUint8 = false;
    bool foundUint16 = false;
    bool foundMap = false;

    for (const auto &key : allKeys) {
        if (key == CONFIG_test_int32)
            foundInt32 = true;
        if (key == CONFIG_test_bool)
            foundBool = true;
        if (key == CONFIG_test_uint8)
            foundUint8 = true;
        if (key == CONFIG_test_uint16)
            foundUint16 = true;
        if (key == CONFIG_test_map)
            foundMap = true;
    }

    EXPECT_TRUE(foundInt32);
    EXPECT_TRUE(foundBool);
    EXPECT_TRUE(foundUint8);
    EXPECT_TRUE(foundUint16);
    EXPECT_TRUE(foundMap);
}

// Test ConfigKey normalization (FromRawName)
TEST_F(ConfigTest, TestConfigKeyNormalization) {
    auto &registry = ConfigRegistry::GetInstance();

    // Test case-insensitive matching - all should normalize to the same key
    ConfigKey key1("test_int32");
    ConfigKey key2("TEST_INT32");

    // All should normalize to the same key and match the registered key
    EXPECT_EQ(key1.Get(), key2.Get());
    EXPECT_TRUE(registry.IsRegistered(key1));
    EXPECT_TRUE(registry.IsRegistered(key2));

    // Test special character normalization
    ConfigKey key3("test-int32");
    ConfigKey key4("test.int32");
    ConfigKey key5("test int32");

    // All should normalize to the same key (special chars become underscores)
    EXPECT_EQ(key1.Get(), key3.Get());
    EXPECT_EQ(key1.Get(), key4.Get());
    EXPECT_EQ(key1.Get(), key5.Get());
    EXPECT_TRUE(registry.IsRegistered(key3));
    EXPECT_TRUE(registry.IsRegistered(key4));
    EXPECT_TRUE(registry.IsRegistered(key5));
}

// Test ConfigKey comparison operators
TEST_F(ConfigTest, TestConfigKeyComparison) {
    ConfigKey key1("test_int32");
    ConfigKey key2("test_int32");
    ConfigKey key3("test_bool");

    // Test equality
    EXPECT_TRUE(key1 == key2);
    EXPECT_FALSE(key1 == key3);

    // Test inequality
    EXPECT_FALSE(key1 != key2);
    EXPECT_TRUE(key1 != key3);

    // Test less-than (for std::map ordering)
    EXPECT_TRUE(key3 < key1); // "test_bool" < "test_int32" alphabetically
    EXPECT_FALSE(key1 < key3);
}

// Test multiple Initialize calls
TEST_F(ConfigTest, TestMultipleInitializeCalls) {
    Config config;

    // First Initialize
    config.Initialize({
        {CONFIG_test_int32, std::any(int32_t(100))},
        { CONFIG_test_bool,         std::any(true)}
    });

    EXPECT_EQ(100, config.Get<int32_t>(CONFIG_test_int32));
    EXPECT_TRUE(config.Get<bool>(CONFIG_test_bool));

    // Second Initialize - should override previous values
    config.Initialize({
        {CONFIG_test_int32, std::any(int32_t(200))},
        {CONFIG_test_uint8,  std::any(uint8_t(88))}
    });

    EXPECT_EQ(200, config.Get<int32_t>(CONFIG_test_int32));
    EXPECT_EQ(88, config.Get<uint8_t>(CONFIG_test_uint8));
    // test_bool should still have value from first Initialize
    EXPECT_TRUE(config.Get<bool>(CONFIG_test_bool));
}

// Test multiple SetDefault calls
TEST_F(ConfigTest, TestMultipleSetDefaultCalls) {
    Config config;

    // Set custom value
    config.Set(CONFIG_test_int32, int32_t(500));

    // First SetDefault
    config.SetDefault(CONFIG_test_int32);
    EXPECT_EQ(100, config.Get<int32_t>(CONFIG_test_int32));

    // Set custom value again
    config.Set(CONFIG_test_int32, int32_t(600));

    // Second SetDefault - should still reset to default
    config.SetDefault(CONFIG_test_int32);
    EXPECT_EQ(100, config.Get<int32_t>(CONFIG_test_int32));
}

// Test ConfigRegistry::GetTypeIndex with unregistered key
TEST_F(ConfigTest, TestConfigRegistryGetTypeIndexUnregistered) {
    auto &registry = ConfigRegistry::GetInstance();
    ConfigKey unregisteredKey("unregistered_key");

    EXPECT_THROW({ registry.GetTypeIndex(unregisteredKey); }, std::runtime_error);
}

// Test ConfigRegistry::GetDefaultValue with unregistered key
TEST_F(ConfigTest, TestConfigRegistryGetDefaultValueUnregistered) {
    auto &registry = ConfigRegistry::GetInstance();
    ConfigKey unregisteredKey("unregistered_key");

    EXPECT_THROW({ registry.GetDefaultValue<int32_t>(unregisteredKey); }, std::runtime_error);
}

// Test ConfigRegistry::GetDefaultValueAny with unregistered key
TEST_F(ConfigTest, TestConfigRegistryGetDefaultValueAnyUnregistered) {
    auto &registry = ConfigRegistry::GetInstance();
    ConfigKey unregisteredKey("unregistered_key");

    EXPECT_THROW({ registry.GetDefaultValueAny(unregisteredKey); }, std::runtime_error);
}

} // namespace pto