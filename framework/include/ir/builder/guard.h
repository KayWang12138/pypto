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
 * \file guard.h
 * \brief
 */

#pragma once

#include "ir/function.h"
namespace pto {

class IRBuilder;
class CompoundStatement;

class ScopeGuard {
public:
    ScopeGuard(IRBuilder& builder, std::shared_ptr<CompoundStatement> new_scope, std::shared_ptr<Function> new_func = nullptr);
    ~ScopeGuard();

private:
    IRBuilder& builder_;
    std::shared_ptr<CompoundStatement> prev_compound_;
    std::shared_ptr<Function> prev_func_;
};
}
