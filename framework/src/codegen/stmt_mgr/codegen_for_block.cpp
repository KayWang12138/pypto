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

#include "interface/tensor/symbolic_scalar.h"
#include "codegen/utils/codegen_utils.h"
#include "codegen/symbol_mgr/codegen_symbol.h"

namespace npu::tile_fwk {
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

std::string ForNode::Print() const {
    std::ostringstream os;
    os << "for (" << PrintInit(os) << PrintCond(os) << PrintUpdate(os) << ") {";
}

void ForNode::PrintInit(std::ostringstream &os) const {
    os << "size_t " << loopVar << " = " << SymbolicExpressionTable::BuildExpression(start) << SEMICOLON_BLANK;
}

void ForNode::PrintCond(std::ostringstream &os) const {
    os << loopVar << " < " << SymbolicExpressionTable::BuildExpression(extent) << SEMICOLON_BLANK;
}

void ForNode::PrintUpdate(std::ostringstream &os) const {
    if (step.ConcreteValid() && step.Concrete() == 1) {
        os << "++" << loopVar << SEMICOLON_BLANK;
    } else {
        os << loopVar << " += " << SymbolicExpressionTable::BuildExpression(step) << SEMICOLON_BLANK
    }
}

void ForBlockManager::UpdateAxesList(const std::vector<SymbolicScalar> &axesList) {
    axesList_ = axesList;
    FillIntVecWithDummyInHead<SymbolicScalar>(axesList_, MAX_LOOP_DEPTH - axesList.size(), 1);
    ALOG_INFO_F("axesList_ after fill is : %s, ", IntVecToStr(axesList_).c_str());
    for (size_t i = 0; i < axesList_.size(); ++i) {
        std::string loopVar = "idx" + std::to_string(i);
        ForNode forNode(loopVar, 0, axesList_[i], 1);
        forNodes_.push_back(forNode);
    }
}

std::string ForBlockManager::Print() const {
    std::ostringstream os;
    PrintForHeader(os);
}

std::string ForBlockManager::PrintForHeader(std::ostringstream &os) const {
    for (size_t i = 0; i < MAX_LOOP_DEPTH; ++i) {
        PrintIndent(os, i);
        forNodes_[i].Print();
    }
}

std::string ForBlockManager::PrintForBody(std::ostringstream &os) const {
    PrintIndent(os, MAX_LOOP_DEPTH + 1);
    PrintOffsetDef(os);
}

std::string ForBlockManager::PrintOffsetDef(std::ostringstream &os) const {
    os << "auto tileOffsets = TileOffset";
    std::vector<std::string> loopVars;
    for (const auto &for : forNodes_){
        loopVars.emplace_back(for.loopVar);
    }
    os << WrapParamByParentheses(loopVars) << SEMICOLON;
}

std::string ForBlockManager::PrintSetAddrs(std::ostringstream &os) const {
    for (const auto &tensor : tensorNeedSetAddr_) {
        PrintIndent(os, MAX_LOOP_DEPTH + 1);
        PrintSetAddrSingle(os, tensor);
    }
}

std::string ForBlockManager::PrintSetAddrSingle(std::ostringstream &os, const std::string &tensor) const {
    os << tensor << ".SetAddr(";
    std::string fullDimTensor = sm->QueryTileTensorFullDimByTensorInLoop(tensor);
    os << fullDimTensor << ".GetLinearAddr(tileOffsets));"
}

} // namespace npu::tile_fwk