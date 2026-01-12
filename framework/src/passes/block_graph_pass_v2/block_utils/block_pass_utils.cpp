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
 * \file pass_utils.cpp
 * \brief
 */

#include "block_pass_utils.h"
#include "ir/statement.h"

namespace npu::tile_fwk {

std::vector<pto::OperationPtr> BlockPassUtils::GetBlockFunctionOperations(pto::Function &function) {
    for (auto statement : function.GetCompound()->GetStatements()) {
        if (statement->GetKind() == pto::StatementKind::Op) {
            auto opStatement = std::dynamic_pointer_cast<pto::OpStatement>(statement);
            return opStatement->Operations();
        }
    }
    return {};
}
} // namespace npu::tile_fwk
