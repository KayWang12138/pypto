/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

#include <memory>
#include <string>
#include <vector>

#include "ir/op_registry.h"

namespace pypto {
namespace ir {

// ============================================================================
// Test Operator Registration
// ============================================================================

REGISTER_OP("test.op")
    .set_op_category("TestOp")
    .set_description("Test operation for operator registration system")
    .add_argument("arg1", "First test argument")
    .add_argument("arg2", "Second test argument")
    .set_attr<int>("int_attr")
    .set_attr<std::string>("string_attr")
    .set_attr<bool>("bool_attr")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& /*kwargs*/) {
      return args[0]->GetType();
    });

}  // namespace ir
}  // namespace pypto
