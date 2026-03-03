/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_op_registry.cpp
 * \brief Unit tests for IR operator registry
 */

#include "gtest/gtest.h"

#include <memory>
#include <string>
#include <vector>

#include "core/dtype.h"
#include "core/error.h"
#include "ir/expr.h"
#include "ir/op_registry.h"
#include "ir/scalar_expr.h"
#include "ir/type.h"

namespace pypto {
namespace ir {

class OpRegistryTest : public testing::Test {};

// ============================================================================
// OpRegistry Singleton Tests
// ============================================================================

TEST_F(OpRegistryTest, TestGetInstance) {
    auto &registry1 = OpRegistry::GetInstance();
    auto &registry2 = OpRegistry::GetInstance();
    ASSERT_EQ(&registry1, &registry2);
}

// ============================================================================
// OpRegistry Lookup Tests (using already registered ops)
// ============================================================================

TEST_F(OpRegistryTest, TestGetOpTensorAdd) {
    auto &registry = OpRegistry::GetInstance();
    auto op = registry.GetOp("tensor.add");
    ASSERT_NE(op, nullptr);
    ASSERT_EQ(op->name_, "tensor.add");
}

TEST_F(OpRegistryTest, TestGetEntryTensorAdd) {
    auto &registry = OpRegistry::GetInstance();
    const auto &entry = registry.GetEntry("tensor.add");
    ASSERT_EQ(entry.GetOp()->name_, "tensor.add");
}

TEST_F(OpRegistryTest, TestGetOpNotFound) {
    auto &registry = OpRegistry::GetInstance();
    ASSERT_THROW(registry.GetOp("nonexistent.op"), InternalError);
}

// ============================================================================
// OpRegistry Create Tests
// ============================================================================

TEST_F(OpRegistryTest, TestCreateTensorAdd) {
    auto &registry = OpRegistry::GetInstance();

    std::vector<ExprPtr> shape = {std::make_shared<ConstInt>(10, DataType::INT64, Span::Unknown())};
    auto tensorType = std::make_shared<TensorType>(shape, DataType::FP32);

    auto arg1 = std::make_shared<Var>("a", tensorType, Span::Unknown());
    auto arg2 = std::make_shared<Var>("b", tensorType, Span::Unknown());

    std::vector<ExprPtr> args = {arg1, arg2};
    auto call = registry.Create("tensor.add", args, Span::Unknown());

    ASSERT_NE(call, nullptr);
    ASSERT_EQ(call->args_.size(), 2);
}

// ============================================================================
// ValidateKwargs Tests
// ============================================================================

TEST_F(OpRegistryTest, TestValidateKwargsEmpty) {
    std::vector<std::pair<std::string, std::any>> kwargs;
    std::unordered_map<std::string, std::type_index> allowed;
    // Empty kwargs should not throw
    ASSERT_NO_THROW(ValidateKwargs(kwargs, allowed, "test.op"));
}

TEST_F(OpRegistryTest, TestValidateKwargsUnknownKey) {
    std::vector<std::pair<std::string, std::any>> kwargs = {
        {"unknown_key", std::any(42)}
    };
    std::unordered_map<std::string, std::type_index> allowed;
    ASSERT_THROW(ValidateKwargs(kwargs, allowed, "test.op"), ValueError);
}

TEST_F(OpRegistryTest, TestValidateKwargsTypeMismatch) {
    std::vector<std::pair<std::string, std::any>> kwargs = {
        {"key", std::any(std::string("hello"))}
    };
    std::unordered_map<std::string, std::type_index> allowed = {
        {"key", std::type_index(typeid(int))}
    };
    ASSERT_THROW(ValidateKwargs(kwargs, allowed, "test.op"), TypeError);
}

TEST_F(OpRegistryTest, TestValidateKwargsCorrectType) {
    std::vector<std::pair<std::string, std::any>> kwargs = {
        {"key", std::any(42)}
    };
    std::unordered_map<std::string, std::type_index> allowed = {
        {"key", std::type_index(typeid(int))}
    };
    ASSERT_NO_THROW(ValidateKwargs(kwargs, allowed, "test.op"));
}

TEST_F(OpRegistryTest, TestValidateKwargsDataTypeAcceptsInt) {
    std::vector<std::pair<std::string, std::any>> kwargs = {
        {"dtype", std::any(42)}
    };
    std::unordered_map<std::string, std::type_index> allowed = {
        {"dtype", std::type_index(typeid(DataType))}
    };
    // DataType kwarg should accept int for backward compatibility
    ASSERT_NO_THROW(ValidateKwargs(kwargs, allowed, "test.op"));
}

TEST_F(OpRegistryTest, TestValidateKwargsDataTypeAcceptsDataType) {
    std::vector<std::pair<std::string, std::any>> kwargs = {
        {"dtype", std::any(DataType::FP32)}
    };
    std::unordered_map<std::string, std::type_index> allowed = {
        {"dtype", std::type_index(typeid(DataType))}
    };
    ASSERT_NO_THROW(ValidateKwargs(kwargs, allowed, "test.op"));
}

} // namespace ir
} // namespace pypto
