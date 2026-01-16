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
 * \file object.cpp
 * \brief
 */

#include "ir/object.h"
#include "ir/program.h"
#include "ir/value.h"
#include "ir/statement.h"

namespace pto {

Object::AttributeRegistry Object::registry_;

// Centralized attribute registration - all attribute registrations happen here
void RegisterAllAttributes()
{
    // ProgramModule attributes
    Object::RegisterAttrKey<ProgramModule, std::string>("arch");
    Object::RegisterAttrKey<ProgramModule, std::string>("tile_default");
    Object::RegisterAttrKey<ProgramModule, std::string>("enable_debug");
    Object::RegisterAttrKey<ProgramModule, std::string>("test_type");

    // Value attributes
    Object::RegisterAttrKey<ScalarValue, std::string>("io");
    Object::RegisterAttrKey<TileValue, std::string>("io");
    Object::RegisterAttrKey<TensorValue, std::string>("io");

    // ForStatement attributes
    Object::RegisterAttrKey<ForStatement, int64_t>("unroll");
}

// Auto-registration: register all attributes during static initialization
namespace {
struct AttributeRegistrar {
    AttributeRegistrar() {
        RegisterAllAttributes();
    }
};
static AttributeRegistrar g_attributeRegistrar;
} // namespace

} // namespace pto
