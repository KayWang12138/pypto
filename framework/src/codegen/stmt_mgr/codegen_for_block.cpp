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
 * \file codegen_for_block.cpp
 * \brief
 */

#include "codegen_for_block.h"

#include "codegen/utils/codegen_utils.h"

namespace npu::tile_fwk {
constexpr const int MAX_LOOP_DEPTH = 3;

std::unordered_map<Opcode, std::string> SUPPORT_VF_FUSE_OPS{
    {    Opcode::OP_ADD,   "TAdd"},
    {    Opcode::OP_SUB,   "TSub"},
    {    Opcode::OP_DIV,   "TDiv"},
    {    Opcode::OP_MUL,   "TMul"},
    {   Opcode::OP_ADDS,  "TAddS"},
    {   Opcode::OP_MULS,  "TMulS"},
    {   Opcode::OP_DIVS,  "TDivS"},
    {  Opcode::OP_RSQRT, "TRsqrt"},
    {   Opcode::OP_SQRT,  "TSqrt"},
    {    Opcode::OP_EXP,   "TExp"},
    {Opcode::OP_MAXIMUM,   "TMax"},
    {Opcode::OP_MINIMUM,   "TMin"},
};

void ForBlockManager::UpdateAxesList(const std::vector<SymbolicScalar> &axesList) {
    axesList_ = axesList;
    FillIntVecWithDummyInHead<SymbolicScalar>(axesList_, MAX_LOOP_DEPTH - axesList.size(), 1);
    for (size_t i = 0; i < axesList_.size(); ++i) {
        std::string loopVar = "idx" + std::to_string(i);
        ForNode forNode(loopVar, 0, axesList_[i], 1);
        forNodes_.push_back(forNode);
    }
}

} // namespace npu::tile_fwk