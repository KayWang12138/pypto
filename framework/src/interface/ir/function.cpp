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
 * \file function.cpp
 * \brief
 */

#include "ir/function.h"

namespace pto {

Function::Function(std::string name, FunctionKind kind, FunctionSignature signature)
    : Object(ObjectType::Function, std::move(name)),
      kind_(kind),
      signature_(std::move(signature)) {
    inputCompound_ = std::make_shared<CompoundStatement>();
    // Make the function body scope a child of the input scope.
    compound_ = std::make_shared<CompoundStatement>(inputCompound_);

    // Register function arguments into the dedicated input scope so that
    // Function::scope_ can see them via GetAncestorValues().
    for (const auto& arg : signature_.arguments) {
        if (arg) {
            // Use SSA name as the key in environment table
            inputCompound_->SetEnvVar(arg->GetName(), arg);
        }
    }
}

void Function::AddStatement(StatementPtr stmt) {
    compound_->AddStatement(std::move(stmt));
}

} // namespace pto


