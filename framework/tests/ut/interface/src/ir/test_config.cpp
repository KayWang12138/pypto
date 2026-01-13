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


// Register test config items
REGISTER_CONFIG(test_int32, int32_t(100));
REGISTER_CONFIG(test_bool, true);
REGISTER_CONFIG(test_uint8, uint8_t(42));
REGISTER_CONFIG(test_uint16, uint16_t(1000));
REGISTER_CONFIG(test_map, (std::map<int64_t, int64_t>{{1, 2}, {3, 4}}));

namespace pto {
class ConfigTest : public testing::Test {
public:
    void SetUp() override {
        // Reset registry state if needed (in real scenario, registry is singleton)
    }

    void TearDown() override {
    }
};

// Test config registration
TEST_F(ConfigTest, TestConfigRegistration) {
    auto& registry = ConfigRegistry::GetInstance();
    
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
        {CONFIG_test_bool, std::any(true)},
        {CONFIG_test_uint8, std::any(uint8_t(88))}
    });
    
    EXPECT_TRUE(config.IsInitialized());
    EXPECT_EQ(300, config.Get<int32_t>(CONFIG_test_int32));
    EXPECT_TRUE(config.Get<bool>(CONFIG_test_bool));
    EXPECT_EQ(88, config.Get<uint8_t>(CONFIG_test_uint8));
}

// Test config Get and Has
TEST_F(ConfigTest, TestConfigGetAndHas) {
    Config config;
    
    config.Initialize({
        {CONFIG_test_int32, std::any(int32_t(100))},
        {CONFIG_test_bool, std::any(true)}
    });
    
    EXPECT_TRUE(config.Has(CONFIG_test_int32));
    EXPECT_TRUE(config.Has(CONFIG_test_bool));
    EXPECT_FALSE(config.Has(CONFIG_test_uint8));
    
    EXPECT_EQ(100, config.Get<int32_t>(CONFIG_test_int32));
    EXPECT_TRUE(config.Get<bool>(CONFIG_test_bool));
}

// Test config with map type
TEST_F(ConfigTest, TestConfigWithMapType) {
    Config config;
    
    std::map<int64_t, int64_t> testMap = {{1, 10}, {2, 20}, {3, 30}};
    config.Initialize({
        {CONFIG_test_map, std::any(testMap)}
    });
    
    const auto& retrievedMap = config.Get<std::map<int64_t, int64_t>>(CONFIG_test_map);
    EXPECT_EQ(3, retrievedMap.size());
    EXPECT_EQ(10, retrievedMap.at(1));
    EXPECT_EQ(20, retrievedMap.at(2));
    EXPECT_EQ(30, retrievedMap.at(3));
}

// Test config double initialization error
TEST_F(ConfigTest, TestConfigDoubleInitializationError) {
    Config config;
    
    config.Initialize({
        {CONFIG_test_int32, std::any(int32_t(100))}
    });
    
    // Second initialization should throw
    EXPECT_THROW({
        config.Initialize({
            {CONFIG_test_bool, std::any(true)}
        });
    }, std::runtime_error);
}

// Test config type mismatch error
TEST_F(ConfigTest, TestConfigTypeMismatchError) {
    Config config;
    
    // Try to initialize with wrong type
    EXPECT_THROW({
        config.Initialize({
            {CONFIG_test_int32, std::any(true)}  // Should be int32_t, not bool
        });
    }, std::runtime_error);
}

// Test config unregistered key error
TEST_F(ConfigTest, TestConfigUnregisteredKeyError) {
    Config config;
    
    // Create an unregistered key
    ConfigKey unregisteredKey = static_cast<ConfigKey>(99999);
    
    EXPECT_THROW({
        config.Initialize({
            {unregisteredKey, std::any(int32_t(100))}
        });
    }, std::runtime_error);
}

// Test config get uninitialized error
TEST_F(ConfigTest, TestConfigGetUninitializedError) {
    Config config;
    
    EXPECT_FALSE(config.IsInitialized());
    EXPECT_THROW({
        config.Get<int32_t>(CONFIG_test_int32);
    }, std::runtime_error);
}

// Test config get non-existent key error
TEST_F(ConfigTest, TestConfigGetNonExistentKeyError) {
    Config config;
    
    config.Initialize({
        {CONFIG_test_int32, std::any(int32_t(100))}
    });
    
    EXPECT_THROW({
        config.Get<bool>(CONFIG_test_bool);  // Not in config
    }, std::runtime_error);
}

// Test config get wrong type error
TEST_F(ConfigTest, TestConfigGetWrongTypeError) {
    Config config;
    
    config.Initialize({
        {CONFIG_test_int32, std::any(int32_t(100))}
    });
    
    // Try to get as wrong type
    EXPECT_THROW({
        config.Get<bool>(CONFIG_test_int32);  // Should be int32_t, not bool
    }, std::runtime_error);
}

// Test ProgramModule config
TEST_F(ConfigTest, TestProgramModuleConfig) {
    auto module = std::make_shared<ProgramModule>("test_module");
    
    EXPECT_FALSE(module->HasConfig());
    
    module->GetConfig().Initialize({
        {CONFIG_test_int32, std::any(int32_t(500))},
        {CONFIG_test_bool, std::any(false)}
    });
    
    EXPECT_TRUE(module->HasConfig());
    EXPECT_EQ(500, module->GetConfig().Get<int32_t>(CONFIG_test_int32));
    EXPECT_FALSE(module->GetConfig().Get<bool>(CONFIG_test_bool));
}

// Test Function config
TEST_F(ConfigTest, TestFunctionConfig) {
    FunctionSignature sig;
    auto func = std::make_shared<Function>("test_func", FunctionKind::ControlFlow, sig);
    
    EXPECT_FALSE(func->HasConfig());
    
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
        {CONFIG_test_bool, std::any(true)}
    });
    
    // Initialize function config with one key that overrides module
    func->GetConfig().Initialize({
        {CONFIG_test_int32, std::any(int32_t(200))}  // Override module's value
    });
    
    // Function config should take precedence
    EXPECT_EQ(200, func->GetConfigValue<int32_t>(CONFIG_test_int32));
    
    // Function doesn't have this key, should get from module
    EXPECT_TRUE(func->GetConfigValue<bool>(CONFIG_test_bool));
    
    // Key not in either should throw
    EXPECT_THROW({
        func->GetConfigValue<uint8_t>(CONFIG_test_uint8);
    }, std::runtime_error);
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
    
    // Key not in function and no parent module should throw
    EXPECT_THROW({
        func->GetConfigValue<bool>(CONFIG_test_bool);
    }, std::runtime_error);
}

// Test MakeConfigEntry helper
TEST_F(ConfigTest, TestMakeConfigEntryHelper) {
    Config config;
    
    config.Initialize({
        MakeConfigEntry(CONFIG_test_int32, int32_t(400)),
        MakeConfigEntry(CONFIG_test_bool, true)
    });
    
    EXPECT_EQ(400, config.Get<int32_t>(CONFIG_test_int32));
    EXPECT_TRUE(config.Get<bool>(CONFIG_test_bool));
}

} // namespace pto