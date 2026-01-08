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
 * \file loopaxes_proc.cpp
 * \brief
 */

#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/utils/common.h"
#include "interface/function/function.h"
#include "interface/tensor/logical_tensor.h"
#include "passes/pass_log/pass_log.h"
#include "passes/pass_interface/pass.h"
#include "loopaxes_proc.h"

#define MODULE_NAME "LoopaxesProc"

namespace npu {
namespace tile_fwk {
Status LoopaxesProc::RunOnFunction(Function &function) {
    APASS_LOG_INFO_F(Elements::Operation, "===============================================================> Start LoopaxesProc.");
    UpdateFuncLoopAxes(function);
    APASS_LOG_INFO_F(Elements::Operation, "===============================================================> Finish LoopaxesProc.");
    return SUCCESS;
}

Status LoopaxesProc::UpdateOpLoopAxes(Operation &op) {
    std::vector<int64_t> loopAxes;
    auto output = op.GetOOperands().front();
    auto shape = output->GetShape();
    if (shape.size() <= 1) {
        // 被纳入group的要求维度大于2，否则将其设置为-1
        op.SetAttribute(OpAttributeKey::loopGroup, -1);
    } else {
        if (op.HasAttr(OpAttributeKey::loopAxes)) {
            loopAxes = op.GetVectorIntAttribute(OpAttributeKey::loopAxes);
        } else {
            for (size_t i = 0UL; i < shape.size() - 1UL; ++i) {
                loopAxes.push_back(shape[i]);
            }
        }
        // 当前节点的loopaxes和group的loopaxes一致，当前节点划入当前的loopaxes
        // 当前节点的loopaxes和group的loopaxes不一致，划入一个新的group起点，进行group
        if (loopAxes != previousLoopAxes) {
            groupIdx++;
            previousLoopAxes = loopAxes;
        }
        op.SetAttribute(OpAttributeKey::loopGroup, groupIdx);
        op.SetAttribute(OpAttributeKey::loopAxes, loopAxes);
    }
    return SUCCESS;
}

Status LoopaxesProc::UpdateFuncLoopAxes(Function &function) {
    for (auto &subProgram : function.rootFunc_->programs_) {
        for (auto &op : subProgram.second->Operations(false)) {
            UpdateOpLoopAxes(op);
        }
    }
    return SUCCESS;
}
} // namespace tile_fwk
} // namespace npu
