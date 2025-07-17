/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file vf_fusion.h
 * \brief
 */

#ifndef PASSES_VF_PATTERN_MATCH_H
#define PASSES_VF_PATTERN_MATCH_H
#include "interface/function/function.h"
#include "interface/tensor/logical_tensor.h"
#include "passes/pass_utils/reschedule_utils.h"
#include "passes/pass_interface/pass.h"
#include "passes/pass_utils/pass_utils.h"
#include "tilefwk/tilefwk.h"
#include "tilefwk.h"
#include "interface/program/program.h"

namespace npu::tile_fwk {

const std::unordered_set<Opcode> BinaryOps{Opcode::OP_ADD, Opcode::OP_SUB, Opcode::OP_MUL, Opcode::OP_DIV,
    Opcode::OP_PAIRMAX, Opcode::OP_PAIRSUM, Opcode::OP_EXPAND, Opcode::OP_FUSED_OP};
const std::unordered_set<Opcode> UnaryOps{
    Opcode::OP_EXP,
    Opcode::OP_SQRT,
    Opcode::OP_ABS,
    Opcode::OP_ROWMAX,
    Opcode::OP_ROWEXPSUM,
    Opcode::OP_ROWEXPMAX,
};
const std::unordered_set<Opcode> VectorScalarOps{
    Opcode::OP_ADDS,
    Opcode::OP_MULS,
};

const int VL_B16 = 32;

class VFFusionPass : public Pass {
public:
    VFFusionPass() : Pass("VFFusionPass") {}
    ~VFFusionPass() override = default;

    Status PostCheck(Function &function) override;

private:
    Status RunOnFunction(Function &function) override {
        ALOG_INFO("============== START VFFusionPass ==========");
        for (auto &subFunctionDict : function.rootFunc_->programs_) {
            auto opList = subFunctionDict.second->Operations().DuplicatedOpList();
            Fusion(subFunctionDict.second);
            RescheduleUtils::UpdateTensorConsProd(subFunctionDict.second);
        }
        ALOG_INFO("============== END VFFusionPass ==========");
        return SUCCESS;
    }
    void Fusion(Function *function) const;
    void AddCopyAndAlloc(const std::shared_ptr<npu::tile_fwk::Function> &function) const;
    void AssignMemory(const std::shared_ptr<npu::tile_fwk::Function> &vfFunc) const;
    void PatternMatch(Function *function, std::vector<std::vector<Operation *>> &fusedList,
        std::vector<std::set<Operation *>> &fusedSet) const;
    bool NextOpCanFuse(Operation *curOp) const;
};

} // namespace npu::tile_fwk

#endif // PASSES_VF_PATTERN_MATCH_H