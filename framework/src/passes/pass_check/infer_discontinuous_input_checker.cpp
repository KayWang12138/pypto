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
 * \file infer_discontinuous_input_checker.cpp
 * \brief
 */

#include "infer_discontinuous_input_checker.h"
#include "passes/pass_log/pass_log.h"
#include <queue>
#include <set>
#define MODULE_NAME "InferDiscontinuousInput"

namespace npu {
namespace tile_fwk {
Status match() {}
Status InferDisContinuousInputChecker::DoPostCheck(Function &function) {
    APASS_LOG_INFO_F(Elements::Function, "PostCheck for DisContinuousInput.");
    if (CheckGraphLoop(function) != SUCCESS) {
        return FAILED;
    }
    std::queue<Operation *> q;
    std::unordered_set<Operation *> visited;
    for (auto &tensor : function.GetOutcast()) {
        std::unordered_map<RawTensor, LogicalTensor> tensorMap;
        std::unordered_map<LogicalTensor, std::pair<Offset, Offset>> offsetMap;
        for (auto producer : tensor->GetProducers()) {
            if (producer->GetOpcode() != Opcode::OP_ASSEMBLE) {
                continue;
            }
            std::shared_ptr<AssembleOpAttribute> attr =
                std::static_pointer_cast<AssembleOpAttribute>(op->GetOpAttribute());
            if (attr == nullptr) {
                APASS_LOG_ERROR_F(Elements::Operation, "assemble op %d do not have attribute. %s", op->GetOpMagic(),
                    GetFormatBacktrace(op).c_str());
                return FAILED;
            }
            auto inputTensor = e->GetIOperands().begin();
            tensorMap[inputTensor.tensor] = inputTensor;
            offsetMap[inputTensor] = std::make_pair<Offset, Offset>();
        }
        if (tensorMap.size() == 0 && offsetMap.size() == 0) {
            continue;
        }
        if (match() != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "assemble op %d do not have attribute. %s", op->GetOpMagic(),
                GetFormatBacktrace(op).c_str());
            return FAILED;
        }
    }
    return SUCCESS;
}
} // namespace tile_fwk
} // namespace npu