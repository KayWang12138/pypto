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
 * \file guard.cpp
 * \brief
 */

#include "ir/builder/ir_builder.h"
#include "ir/function.h"
#include "ir/builder/guard.h"

namespace pto {

ScopeGuard::ScopeGuard(IRBuilder& builder, std::shared_ptr<CompoundStatement> new_scope, std::shared_ptr<Function> new_func)
    : builder_(builder),
      prev_compound_(builder.compound_),
      prev_func_(builder_.func_) {
    builder_.compound_ = new_scope;
    builder_.opStmt_ = nullptr;
    if (new_func) {
      builder_.func_ = new_func;
    }
}

ScopeGuard::~ScopeGuard() {
    builder_.compound_ = prev_compound_;
    builder_.opStmt_ = nullptr;
    builder_.func_ = prev_func_;
}
}
