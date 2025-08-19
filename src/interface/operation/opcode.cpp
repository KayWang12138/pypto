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
 * \file opcode.cpp
 * \brief
 */

#include "opcode.h"
#include <string>
#include <array>
#include <sstream>
#include <unordered_set>
#include "interface/utils/common.h"
#include "tilefwk/data_type.h"
#include "tilefwk/error.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"

namespace npu::tile_fwk {
OpcodeManager::OpcodeManager() {
    std::unordered_set<Opcode> registered;
    auto registerInfo = [this, &registered](Opcode opcode, OpCoreType coreType, std::string str,
                            std::vector<MemoryType> inputsMemType, std::vector<MemoryType> outputsMemType,
                            const TileOpCfg tileOpCfg,
                            OpCalcType calcType,
                            const std::vector<std::string> &attrs = {}) {
        ASSERT(opcode < Opcode::OP_UNKNOWN);
        ASSERT(strToEnum_.count(str) == 0);
        ASSERT(registered.count(opcode) == 0);
        registered.emplace(opcode);
        strToEnum_.emplace(str, opcode);
        opcodeInfos_[static_cast<int>(opcode)] =
            OpcodeInfo{opcode, coreType, std::move(str), std::move(inputsMemType), std::move(outputsMemType),
            tileOpCfg, calcType, attrs};
    };

    std::vector<std::string> convAttrStrList{
        ConvOpAttributeKey::cin,
        ConvOpAttributeKey::cout,
        ConvOpAttributeKey::paddingLeft,
        ConvOpAttributeKey::paddingTop,
        ConvOpAttributeKey::paddingRight,
        ConvOpAttributeKey::paddingBottom,
        ConvOpAttributeKey::strideh,
        ConvOpAttributeKey::stridew,
        ConvOpAttributeKey::hposX,
        ConvOpAttributeKey::hsteP,
        ConvOpAttributeKey::wposX,
        ConvOpAttributeKey::wstep,
        ConvOpAttributeKey::hoffsetY,
        ConvOpAttributeKey::woffsetY,
        ConvOpAttributeKey::reluType,
        ConvOpAttributeKey::reluAlpha,
        ConvOpAttributeKey::clearFlag,
        ConvOpAttributeKey::hasAccFlag,
        ConvOpAttributeKey::hasEltFlag,
        ConvOpAttributeKey::hasBiasFlag,
        ConvOpAttributeKey::eltBrcbFlag,
        ConvOpAttributeKey::eltMode,
        ConvOpAttributeKey::fmapSrcNum,
        FixpOpAttributeKey::hasQuantPreVector,
        FixpOpAttributeKey::hasQuantPostVector,
        FixpOpAttributeKey::hasAntiqVector,
        FixpOpAttributeKey::quantPreScalar,
        FixpOpAttributeKey::quantPostScalar,
        FixpOpAttributeKey::antiqScalar,
    };
    // AIV
    // Unary
    // Cast Vector
    // clang-format off
    registerInfo(Opcode::OP_CAST, OpCoreType::AIV, "CAST", {MemoryType::MEM_UB}, {MemoryType::MEM_UB}, {"TileOp::Tcast", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::CAST, {OP_ATTR_PREFIX + "mode"});

    // Unary Vector
    registerInfo(Opcode::OP_EXP, OpCoreType::AIV, "EXP", {MemoryType::MEM_UB}, {MemoryType::MEM_UB}, {"TileOp::Texp", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::ELMWISE, {OpAttributeKey::inputCombineAxis, OpAttributeKey::outputCombineAxis});
    registerInfo(Opcode::OP_SQRT, OpCoreType::AIV, "SQRT", {MemoryType::MEM_UB}, {MemoryType::MEM_UB}, {"TileOp::Tsqrt", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::ELMWISE, {OpAttributeKey::inputCombineAxis, OpAttributeKey::outputCombineAxis});
    registerInfo(Opcode::OP_RECIPROCAL, OpCoreType::AIV, "RECIPROCAL", {MemoryType::MEM_UB}, {MemoryType::MEM_UB}, {"TileOp::Trec", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::ELMWISE, {OpAttributeKey::inputCombineAxis, OpAttributeKey::outputCombineAxis});
    registerInfo(Opcode::OP_ABS, OpCoreType::AIV, "ABS", {MemoryType::MEM_UB}, {MemoryType::MEM_UB}, {"TileOp::Tabs", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::ELMWISE, {OpAttributeKey::inputCombineAxis, OpAttributeKey::outputCombineAxis});

    registerInfo(Opcode::OP_TRANSPOSE_MOVEOUT, OpCoreType::ANY, "TRANSPOSE_MOVEOUT",
        {MemoryType::MEM_UB}, {MemoryType::MEM_DEVICE_DDR}, {"TileOp::TtransposeMoveOut", PIPE_MTE3, PIPE_MTE3, CoreType::AIV}, OpCalcType::MOVE_OUT, {OP_ATTR_PREFIX + "shape"});
    registerInfo(Opcode::OP_TRANSPOSE_VNCHWCONV, OpCoreType::ANY, "TRANSPOSE_VNCHWCONV",
        {MemoryType::MEM_UB}, {MemoryType::MEM_UB, MemoryType::MEM_UB}, {"TileOp::Ttranspose_vnchwconv", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::MOVE_LOCAL, {OP_ATTR_PREFIX + "shape"});

    registerInfo(Opcode::OP_EXPAND, OpCoreType::AIV, "EXPAND", {MemoryType::MEM_UB}, {MemoryType::MEM_UB}, {"TileOp::Texpand", PIPE_V, PIPE_V, CoreType::AIV},
        OpCalcType::ELMWISE, {OP_ATTR_PREFIX + "EXPANDDIM"});
    registerInfo(Opcode::OP_CONCAT, OpCoreType::AIV, "CONCAT", {MemoryType::MEM_UB}, {MemoryType::MEM_UB}, {"TileOp::Tconcat", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::ELMWISE, {"concat", OpAttributeKey::inputCombineAxis, OpAttributeKey::outputCombineAxis});
    registerInfo(Opcode::OP_COMPACT, OpCoreType::AIV, "COMPACT", {MemoryType::MEM_UB}, {MemoryType::MEM_UB}, {"TileOp::Tcompact", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::OTHER);
    registerInfo(Opcode::OP_ROWMAX, OpCoreType::AIV, "ROWMAX", {MemoryType::MEM_UB}, {MemoryType::MEM_UB}, {"TileOp::Trowmaxexpand", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::REDUCE);
    registerInfo(Opcode::OP_ROWSUM, OpCoreType::AIV, "ROWSUM", {MemoryType::MEM_UB}, {MemoryType::MEM_UB}, {"TileOp::Treducesum", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::REDUCE);
    registerInfo(Opcode::OP_ROWEXPMAX, OpCoreType::AIV, "ROWEXPMAX", {MemoryType::MEM_UB}, {MemoryType::MEM_UB}, {"TileOp::Trowmaxexpand", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::REDUCE);
    registerInfo(Opcode::OP_ROWEXPSUM, OpCoreType::AIV, "ROWEXPSUM", {MemoryType::MEM_UB}, {MemoryType::MEM_UB}, {"TileOp::Trowsumexpand", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::REDUCE);
    registerInfo(Opcode::OP_ADDS, OpCoreType::AIV, "ADDS", {MemoryType::MEM_UB}, {MemoryType::MEM_UB}, {"TileOp::Tadds", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::ELMWISE, {OpAttributeKey::scalar, OP_ATTR_PREFIX + "reverseOperand", OP_ATTR_PREFIX + "reverseOperand", OpAttributeKey::inputCombineAxis, OpAttributeKey::outputCombineAxis});
    registerInfo(Opcode::OP_SUBS, OpCoreType::AIV, "SUBS", {MemoryType::MEM_UB}, {MemoryType::MEM_UB}, {"TileOp::Tsubs", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::ELMWISE, {OpAttributeKey::scalar, OP_ATTR_PREFIX + "reverseOperand", OP_ATTR_PREFIX + "reverseOperand", OpAttributeKey::inputCombineAxis, OpAttributeKey::outputCombineAxis});
    registerInfo(Opcode::OP_MULS, OpCoreType::AIV, "MULS", {MemoryType::MEM_UB}, {MemoryType::MEM_UB}, {"TileOp::Tmuls", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::ELMWISE, {OpAttributeKey::scalar, OP_ATTR_PREFIX + "reverseOperand", OP_ATTR_PREFIX + "reverseOperand", OpAttributeKey::inputCombineAxis, OpAttributeKey::outputCombineAxis});
    registerInfo(Opcode::OP_DIVS, OpCoreType::AIV, "DIVS", {MemoryType::MEM_UB}, {MemoryType::MEM_UB}, {"TileOp::Tdivs", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::ELMWISE, {OpAttributeKey::scalar, OP_ATTR_PREFIX + "reverseOperand", OP_ATTR_PREFIX + "reverseOperand", OpAttributeKey::inputCombineAxis, OpAttributeKey::outputCombineAxis});
    registerInfo(Opcode::OP_S_ADDS, OpCoreType::AIV, "S_ADDS", {MemoryType::MEM_UB}, {MemoryType::MEM_UB}, {"TileOp::TSadds", PIPE_S, PIPE_S, CoreType::AIV}, OpCalcType::ELMWISE, {OpAttributeKey::scalar, OP_ATTR_PREFIX + "reverseOperand", OpAttributeKey::inputCombineAxis, OpAttributeKey::outputCombineAxis});
    registerInfo(Opcode::OP_S_SUBS, OpCoreType::AIV, "S_SUBS", {MemoryType::MEM_UB}, {MemoryType::MEM_UB}, {"TileOp::TSsubs", PIPE_S, PIPE_S, CoreType::AIV}, OpCalcType::ELMWISE, {OpAttributeKey::scalar, OP_ATTR_PREFIX + "reverseOperand", OpAttributeKey::inputCombineAxis, OpAttributeKey::outputCombineAxis});
    registerInfo(Opcode::OP_S_MULS, OpCoreType::AIV, "S_MULS", {MemoryType::MEM_UB}, {MemoryType::MEM_UB}, {"TileOp::TSmuls", PIPE_S, PIPE_S, CoreType::AIV}, OpCalcType::ELMWISE, {OpAttributeKey::scalar, OP_ATTR_PREFIX + "reverseOperand", OpAttributeKey::inputCombineAxis, OpAttributeKey::outputCombineAxis});
    registerInfo(Opcode::OP_S_DIVS, OpCoreType::AIV, "S_DIVS", {MemoryType::MEM_UB}, {MemoryType::MEM_UB}, {"TileOp::TSdivs", PIPE_S, PIPE_S, CoreType::AIV}, OpCalcType::ELMWISE, {OpAttributeKey::scalar, OP_ATTR_PREFIX + "reverseOperand", OpAttributeKey::inputCombineAxis, OpAttributeKey::outputCombineAxis});
    registerInfo(Opcode::OP_S_MAXS, OpCoreType::AIV, "S_MAXS", {MemoryType::MEM_UB}, {MemoryType::MEM_UB}, {"TileOp::TSmaxs", PIPE_S, PIPE_S, CoreType::AIV}, OpCalcType::ELMWISE, {OpAttributeKey::scalar, OP_ATTR_PREFIX + "reverseOperand", OpAttributeKey::inputCombineAxis, OpAttributeKey::outputCombineAxis});
    registerInfo(Opcode::OP_S_MINS, OpCoreType::AIV, "S_MINS", {MemoryType::MEM_UB}, {MemoryType::MEM_UB}, {"TileOp::TSmins", PIPE_S, PIPE_S, CoreType::AIV}, OpCalcType::ELMWISE, {OpAttributeKey::scalar, OP_ATTR_PREFIX + "reverseOperand", OpAttributeKey::inputCombineAxis, OpAttributeKey::outputCombineAxis});
    registerInfo(Opcode::OP_REGISTER_COPY, OpCoreType::AIV, "REGISTER_COPY", {MemoryType::MEM_UB}, {MemoryType::MEM_UB}, {"REGISTER_COPY", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::SYS);
    // Binary
    registerInfo(Opcode::OP_PAD, OpCoreType::ANY, "PAD", {MemoryType::MEM_UB}, {MemoryType::MEM_UB}, {}, OpCalcType::ELMWISE, {OpAttributeKey::inputCombineAxis, OpAttributeKey::outputCombineAxis});
    // Binary Vector
    registerInfo(Opcode::OP_ADD_BRC, OpCoreType::AIV, "ADD_BRC", {MemoryType::MEM_UB, MemoryType::MEM_UB}, {MemoryType::MEM_UB, MemoryType::MEM_UB}, {"TileOp::Taddbrc", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::BROADCAST, {OpAttributeKey::inputCombineAxis});
    registerInfo(Opcode::OP_SUB_BRC, OpCoreType::AIV, "SUB_BRC", {MemoryType::MEM_UB, MemoryType::MEM_UB}, {MemoryType::MEM_UB, MemoryType::MEM_UB}, {"TileOp::Tsubbrc", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::BROADCAST, {OpAttributeKey::inputCombineAxis});
    registerInfo(Opcode::OP_MUL_BRC, OpCoreType::AIV, "MUL_BRC", {MemoryType::MEM_UB, MemoryType::MEM_UB}, {MemoryType::MEM_UB, MemoryType::MEM_UB}, {"TileOp::Tmulbrc", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::BROADCAST, {OpAttributeKey::inputCombineAxis});
    registerInfo(Opcode::OP_DIV_BRC, OpCoreType::AIV, "DIV_BRC", {MemoryType::MEM_UB, MemoryType::MEM_UB}, {MemoryType::MEM_UB, MemoryType::MEM_UB}, {"TileOp::Tdivbrc", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::BROADCAST, {OpAttributeKey::inputCombineAxis});
    registerInfo(Opcode::OP_MAX_BRC, OpCoreType::AIV, "MAX_BRC", {MemoryType::MEM_UB, MemoryType::MEM_UB}, {MemoryType::MEM_UB, MemoryType::MEM_UB}, {"TileOp::Tmaxbrc", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::BROADCAST, {OpAttributeKey::inputCombineAxis});
    registerInfo(Opcode::OP_ADD, OpCoreType::AIV, "ADD", {MemoryType::MEM_UB, MemoryType::MEM_UB}, {MemoryType::MEM_UB}, {"TileOp::Tadd", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::BROADCAST, {OpAttributeKey::inputCombineAxis});
    registerInfo(Opcode::OP_SUB, OpCoreType::AIV, "SUB", {MemoryType::MEM_UB, MemoryType::MEM_UB}, {MemoryType::MEM_UB}, {"TileOp::Tsub", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::BROADCAST, {OpAttributeKey::inputCombineAxis});
    registerInfo(Opcode::OP_MUL, OpCoreType::AIV, "MUL", {MemoryType::MEM_UB, MemoryType::MEM_UB}, {MemoryType::MEM_UB}, {"TileOp::Tmul", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::BROADCAST, {OpAttributeKey::inputCombineAxis});
    registerInfo(Opcode::OP_DIV, OpCoreType::AIV, "DIV", {MemoryType::MEM_UB, MemoryType::MEM_UB}, {MemoryType::MEM_UB}, {"TileOp::Tdiv", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::BROADCAST, {OpAttributeKey::inputCombineAxis});
    registerInfo(Opcode::OP_S_ADD, OpCoreType::AIV, "S_ADD", {MemoryType::MEM_UB, MemoryType::MEM_UB}, {MemoryType::MEM_UB}, {"TileOp::TSadd", PIPE_S, PIPE_S, CoreType::AIV}, OpCalcType::BROADCAST, {OpAttributeKey::inputCombineAxis});
    registerInfo(Opcode::OP_S_SUB, OpCoreType::AIV, "S_SUB", {MemoryType::MEM_UB, MemoryType::MEM_UB}, {MemoryType::MEM_UB}, {"TileOp::TSsub", PIPE_S, PIPE_S, CoreType::AIV}, OpCalcType::BROADCAST, {OpAttributeKey::inputCombineAxis});
    registerInfo(Opcode::OP_S_MUL, OpCoreType::AIV, "S_MUL", {MemoryType::MEM_UB, MemoryType::MEM_UB}, {MemoryType::MEM_UB}, {"TileOp::TSmul", PIPE_S, PIPE_S, CoreType::AIV}, OpCalcType::BROADCAST, {OpAttributeKey::inputCombineAxis});
    registerInfo(Opcode::OP_S_DIV, OpCoreType::AIV, "S_DIV", {MemoryType::MEM_UB, MemoryType::MEM_UB}, {MemoryType::MEM_UB}, {"TileOp::TSdiv", PIPE_S, PIPE_S, CoreType::AIV}, OpCalcType::BROADCAST, {OpAttributeKey::inputCombineAxis});
    registerInfo(Opcode::OP_S_MAX, OpCoreType::AIV, "S_MAX", {MemoryType::MEM_UB, MemoryType::MEM_UB}, {MemoryType::MEM_UB}, {"TileOp::TSmax", PIPE_S, PIPE_S, CoreType::AIV}, OpCalcType::BROADCAST, {OpAttributeKey::inputCombineAxis});
    registerInfo(Opcode::OP_S_MIN, OpCoreType::AIV, "S_MIN", {MemoryType::MEM_UB, MemoryType::MEM_UB}, {MemoryType::MEM_UB}, {"TileOp::TSmin", PIPE_S, PIPE_S, CoreType::AIV}, OpCalcType::BROADCAST, {OpAttributeKey::inputCombineAxis});
    registerInfo(Opcode::OP_GATHER, OpCoreType::AIV, "GATHER", {MemoryType::MEM_UB, MemoryType::MEM_UB}, {MemoryType::MEM_UB}, {"TileOp::Tgather", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::OTHER, {OP_ATTR_PREFIX + "axis"});
    registerInfo(Opcode::OP_GATHER_ELEMENT, OpCoreType::AIV, "GATHER_ELEMENT",
        {MemoryType::MEM_UB, MemoryType::MEM_UB}, {MemoryType::MEM_UB}, {"TileOp::TgatherElement", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::OTHER, {OP_ATTR_PREFIX + "axis"});
    registerInfo(Opcode::OP_SCATTER_ELEMENT, OpCoreType::AIV, "SCATTER_ELEMENT",
        {MemoryType::MEM_UB, MemoryType::MEM_UB}, {MemoryType::MEM_UB}, {"TileOp::TscatterElement", PIPE_S, PIPE_S, CoreType::AIV}, OpCalcType::OTHER, {OP_ATTR_PREFIX + "axis", OpAttributeKey::scalar});
    registerInfo(Opcode::OP_INDEX_PUT, OpCoreType::AIV, "INDEX_PUT",
        {MemoryType::MEM_UB, MemoryType::MEM_UB}, {MemoryType::MEM_UB}, {"TileOp::Tindexput", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::OTHER, {"axis"});
    registerInfo(Opcode::OP_SCATTER_UPDATE, OpCoreType::ANY, "SCATTER_UPDATE",
        {MemoryType::MEM_UB, MemoryType::MEM_UB}, {MemoryType::MEM_UB}, {}, OpCalcType::OTHER);
    registerInfo(Opcode::OP_SCATTER_SCALAR, OpCoreType::ANY, "SCATTER_SCALAR",
        {MemoryType::MEM_UB, MemoryType::MEM_UB}, {MemoryType::MEM_UB}, {}, OpCalcType::OTHER);
    registerInfo(Opcode::OP_MAXIMUM, OpCoreType::AIV, "MAXIMUM", {MemoryType::MEM_UB, MemoryType::MEM_UB}, {MemoryType::MEM_UB}, {"TileOp::Tmax", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::BROADCAST, {OpAttributeKey::inputCombineAxis});
    registerInfo(Opcode::OP_PAIRMAX, OpCoreType::AIV, "PAIRMAX", {MemoryType::MEM_UB, MemoryType::MEM_UB}, {MemoryType::MEM_UB}, {"TileOp::Tmax", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::BROADCAST, {OpAttributeKey::inputCombineAxis});
    registerInfo(Opcode::OP_PAIRSUM, OpCoreType::AIV, "PAIRSUM", {MemoryType::MEM_UB, MemoryType::MEM_UB}, {MemoryType::MEM_UB}, {"TileOp::Tadd", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::BROADCAST, {OpAttributeKey::inputCombineAxis});
    registerInfo(Opcode::OP_ROWMAX_SINGLE, OpCoreType::AIV, "ROWMAX_SINGLE",
        {MemoryType::MEM_UB}, {MemoryType::MEM_UB, MemoryType::MEM_UB}, {"TileOp::Trowmaxsingle", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::REDUCE, {OP_ATTR_PREFIX + "AXIS"});
    registerInfo(Opcode::OP_ROWSUM_SINGLE, OpCoreType::AIV, "ROWSUM_SINGLE",
        {MemoryType::MEM_UB}, {MemoryType::MEM_UB, MemoryType::MEM_UB}, {"TileOp::Trowsumsingle", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::REDUCE, {OP_ATTR_PREFIX + "AXIS"});
    registerInfo(Opcode::OP_ROWMAX_COMBINE_AXIS_SINGLE, OpCoreType::AIV, "ROWMAX_COMBINE_AXIS_SINGLE",
        {MemoryType::MEM_UB}, {MemoryType::MEM_UB, MemoryType::MEM_UB}, {"TileOp::Trowmaxsinglecombine", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::REDUCE, {OP_ATTR_PREFIX + "AXIS", OpAttributeKey::outputCombineAxis});
    registerInfo(Opcode::OP_ROWSUM_COMBINE_AXIS_SINGLE, OpCoreType::AIV, "ROWSUM_COMBINE_AXIS_SINGLE",
        {MemoryType::MEM_UB}, {MemoryType::MEM_UB, MemoryType::MEM_UB}, {"TileOp::Trowsumsinglecombine", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::REDUCE, {OP_ATTR_PREFIX + "AXIS", OpAttributeKey::outputCombineAxis});
    registerInfo(Opcode::OP_ROWSUMLINE, OpCoreType::AIV, "ROWSUMLINE",
        {MemoryType::MEM_UB, MemoryType::MEM_UB}, {MemoryType::MEM_UB}, {"TileOp::Trowsumline", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::REDUCE);
    registerInfo(Opcode::OP_ROWMAXLINE, OpCoreType::AIV, "ROWMAXLINE",
        {MemoryType::MEM_UB, MemoryType::MEM_UB}, {MemoryType::MEM_UB}, {"TileOp::Trowmaxline", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::REDUCE);
    // AIV ALLOC
    registerInfo(Opcode::OP_VEC_DUP, OpCoreType::AIV, "VEC_DUP", {MemoryType::MEM_UB}, {MemoryType::MEM_UB}, {"TileOp::Tduplicate", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::OTHER, {OpAttributeKey::scalar, OP_ATTR_PREFIX + "shape", OP_ATTR_PREFIX + "validShape"});
    registerInfo(Opcode::OP_UB_ALLOC, OpCoreType::AIV, "UB_ALLOC", {}, {MemoryType::MEM_UB}, {"UB_ALLOC", PIPE_S, PIPE_S, CoreType::AIV}, OpCalcType::SYS);
    registerInfo(Opcode::OP_UB_COPY_IN, OpCoreType::AIV, "UB_COPY_IN", {MemoryType::MEM_DEVICE_DDR}, {MemoryType::MEM_UB}, {"TileOp::UBCopyIn", PIPE_MTE2, PIPE_MTE2, CoreType::AIV}, OpCalcType::MOVE_IN);
    registerInfo(Opcode::OP_UB_COPY_OUT, OpCoreType::AIV, "UB_COPY_OUT", {MemoryType::MEM_UB}, {MemoryType::MEM_DEVICE_DDR}, {"TileOp::UBCopyOut", PIPE_MTE3, PIPE_MTE3, CoreType::AIV}, OpCalcType::MOVE_OUT);
    registerInfo(Opcode::OP_REG_ALLOC, OpCoreType::AIV, "REG_ALLOC", {}, {MemoryType::MEM_VECTOR_REG}, {"REG_ALLOC", PIPE_S, PIPE_S, CoreType::AIV}, OpCalcType::SYS);

    // AIC
    registerInfo(Opcode::OP_A_MUL_B, OpCoreType::AIC, "A_MUL_B", {MemoryType::MEM_L0A, MemoryType::MEM_L0B, MemoryType::MEM_DEVICE_DDR}, {MemoryType::MEM_L0C}, {"TileOp::Tmad", PIPE_M, PIPE_M, CoreType::AIC}, OpCalcType::MATMUL);
    registerInfo(Opcode::OP_A_MULACC_B, OpCoreType::AIC, "A_MULACC_B", {MemoryType::MEM_L0A, MemoryType::MEM_L0B, MemoryType::MEM_L0C, MemoryType::MEM_DEVICE_DDR}, {MemoryType::MEM_L0C}, {"TileOp::Tmad", PIPE_M, PIPE_M, CoreType::AIC}, OpCalcType::MATMUL);
    registerInfo(Opcode::OP_A_MUL_BT, OpCoreType::AIC, "A_MUL_Bt", {MemoryType::MEM_L0A, MemoryType::MEM_L0B, MemoryType::MEM_DEVICE_DDR}, {MemoryType::MEM_L0C}, {"TileOp::Tmad", PIPE_M, PIPE_M, CoreType::AIC}, OpCalcType::MATMUL);
    registerInfo(Opcode::OP_A_MULACC_BT, OpCoreType::AIC, "A_MULACC_Bt", {MemoryType::MEM_L0A, MemoryType::MEM_L0B, MemoryType::MEM_L0C, MemoryType::MEM_DEVICE_DDR}, {MemoryType::MEM_L0C}, {"TileOp::Tmad", PIPE_M, PIPE_M, CoreType::AIC}, OpCalcType::MATMUL);

    registerInfo(Opcode::OP_CONV, OpCoreType::AIC, "CONV", {}, {}, {"CUBE_CONV", PIPE_M, PIPE_M, CoreType::AIC}, OpCalcType::CONV, convAttrStrList);
    registerInfo(Opcode::OP_CONV_ADD, OpCoreType::AIC, "CONV_ADD", {}, {}, {"CUBE_CONV_ADD", PIPE_M, PIPE_M, CoreType::AIC}, OpCalcType::CONV, convAttrStrList);
    registerInfo(Opcode::OP_CUBE_CONV_D2S, OpCoreType::AIC, "CONV_D2S", {}, {}, {"CUBE_CONV_D2S", PIPE_FIX, PIPE_FIX, CoreType::AIC}, OpCalcType::CONV);
    registerInfo(Opcode::OP_CUBE_CONCAT_C, OpCoreType::AIC, "CONCAT_C", {}, {}, {"CUBE_CONCAT_C", PIPE_MTE1, PIPE_FIX, CoreType::AIC}, OpCalcType::CONV);
    // AIC ALLOC
    registerInfo(Opcode::OP_L1_ALLOC, OpCoreType::AIC, "L1_ALLOC", {}, {MemoryType::MEM_L1}, {"L1_ALLOC", PIPE_S, PIPE_S, CoreType::AIC}, OpCalcType::SYS);
    registerInfo(Opcode::OP_L0A_ALLOC, OpCoreType::AIC, "L0A_ALLOC", {}, {MemoryType::MEM_L0A}, {"L0A_ALLOC", PIPE_S, PIPE_S, CoreType::AIC}, OpCalcType::SYS);
    registerInfo(Opcode::OP_L0B_ALLOC, OpCoreType::AIC, "L0B_ALLOC", {}, {MemoryType::MEM_L0B}, {"L0B_ALLOC", PIPE_S, PIPE_S, CoreType::AIC}, OpCalcType::SYS);
    registerInfo(Opcode::OP_L0C_ALLOC, OpCoreType::AIC, "L0C_ALLOC", {}, {MemoryType::MEM_L0C}, {"L0C_ALLOC", PIPE_S, PIPE_S, CoreType::AIC}, OpCalcType::SYS);
    registerInfo(Opcode::OP_FIX_ALLOC, OpCoreType::AIC, "FIX_ALLOC", {}, {}, {"FIX_ALLOC", PIPE_S, PIPE_S, CoreType::AIC}, OpCalcType::SYS);
    registerInfo(Opcode::OP_BT_ALLOC, OpCoreType::AIC, "BT_ALLOC", {}, {}, {"BT_ALLOC", PIPE_S, PIPE_S, CoreType::AIC}, OpCalcType::SYS);
    // AIC MOVE
    registerInfo(Opcode::OP_L1_COPY_IN, OpCoreType::AIC, "L1_COPY_IN", {MemoryType::MEM_DEVICE_DDR}, {MemoryType::MEM_L1}, {"TileOp::L1CopyIn", PIPE_MTE2, PIPE_MTE2, CoreType::AIC}, OpCalcType::MOVE_IN);
    registerInfo(Opcode::OP_L1_COPY_IN_FRACTAL_Z, OpCoreType::AIC, "L1_COPY_IN_FractalZ", {MemoryType::MEM_DEVICE_DDR}, {MemoryType::MEM_L1}, {"TileOp::L1CopyInFractalZ", PIPE_MTE2, PIPE_MTE2, CoreType::AIC}, OpCalcType::MOVE_IN, {ConvOpAttributeKey::fmapC0});
    registerInfo(Opcode::OP_L1_COPY_OUT, OpCoreType::AIC, "L1_COPY_OUT", {MemoryType::MEM_L1}, {MemoryType::MEM_DEVICE_DDR}, {"TileOp::L1CopyOut", PIPE_FIX, PIPE_FIX, CoreType::AIC}, OpCalcType::MOVE_OUT);
    registerInfo(Opcode::OP_L1_LOOP_ENHANCE, OpCoreType::AIC, "L1_LOOP_ENHANCE", {MemoryType::MEM_L1},
                 {MemoryType::MEM_L1}, {"TileOp::Depth2Space", PIPE_FIX, PIPE_FIX, CoreType::AIC},
                 OpCalcType::MOVE_LOCAL);
    registerInfo(Opcode::OP_L1_COPY_IN_DMA, OpCoreType::AIC, "L1_COPY_IN_DMA", {MemoryType::MEM_DEVICE_DDR}, {MemoryType::MEM_L1}, {"TileOp::L1CopyInDMA", PIPE_MTE2, PIPE_MTE2, CoreType::AIC}, OpCalcType::MOVE_IN);
    registerInfo(Opcode::OP_L1_COPY_OUT_DMA, OpCoreType::AIC, "L1_COPY_OUT_DMA", {MemoryType::MEM_L1}, {MemoryType::MEM_DEVICE_DDR}, {"TileOp::L1CopyOutDMA", PIPE_MTE3, PIPE_MTE3, CoreType::AIC}, OpCalcType::MOVE_OUT);
    registerInfo(Opcode::OP_L1_TO_L0A, OpCoreType::AIC, "L1_TO_L0A", {MemoryType::MEM_L1}, {MemoryType::MEM_L0A}, {"TileOp::L1ToL0A", PIPE_MTE1, PIPE_MTE1, CoreType::AIC}, OpCalcType::MOVE_LOCAL);
    registerInfo(Opcode::OP_L1_TO_L0B, OpCoreType::AIC, "L1_TO_L0B", {MemoryType::MEM_L1}, {MemoryType::MEM_L0B}, {"TileOp::L1ToL0B", PIPE_MTE1, PIPE_MTE1, CoreType::AIC}, OpCalcType::MOVE_LOCAL);
    registerInfo(Opcode::OP_L1_TO_L0_BT, OpCoreType::AIC, "L1_TO_L0Bt", {MemoryType::MEM_L1}, {MemoryType::MEM_L0B}, {"TileOp::L1ToL0Bt", PIPE_MTE1, PIPE_MTE1, CoreType::AIC}, OpCalcType::MOVE_LOCAL);
    registerInfo(Opcode::OP_L0C_COPY_OUT, OpCoreType::AIC, "L0C_COPY_OUT", {MemoryType::MEM_L0C}, {MemoryType::MEM_DEVICE_DDR}, {"TileOp::L0CCopyOut", PIPE_FIX, PIPE_FIX, CoreType::AIC}, OpCalcType::MOVE_OUT);
    registerInfo(Opcode::OP_FIX_COPY_IN, OpCoreType::AIC, "FIX_COPY_IN", {MemoryType::MEM_L1}, {MemoryType::MEM_FIX}, {"TileOp::L1CopyFix", PIPE_MTE1, PIPE_MTE1, CoreType::AIC}, OpCalcType::MOVE_LOCAL, {FixpOpAttributeKey::fbAddrSpace});
    registerInfo(Opcode::OP_FIX_COPY_IN_QUANT_PRE, OpCoreType::AIC, "FIX_COPY_IN_QUANT_PRE", {MemoryType::MEM_L1}, {MemoryType::MEM_FIX_QUANT_PRE}, {"TileOp::L1CopyFix", PIPE_MTE1, PIPE_MTE1, CoreType::AIC}, OpCalcType::MOVE_LOCAL, {FixpOpAttributeKey::fbAddrSpace});
    registerInfo(Opcode::OP_FIX_COPY_IN_RELU_PRE, OpCoreType::AIC, "FIX_COPY_IN_RELU_PRE", {MemoryType::MEM_L1}, {MemoryType::MEM_FIX_RELU_PRE}, {"TileOp::L1CopyFix", PIPE_MTE1, PIPE_MTE1, CoreType::AIC}, OpCalcType::MOVE_LOCAL, {FixpOpAttributeKey::fbAddrSpace});
    registerInfo(Opcode::OP_FIX_COPY_IN_RELU_POST, OpCoreType::AIC, "FIX_COPY_IN_RELU_POST", {MemoryType::MEM_L1}, {MemoryType::MEM_FIX_RELU_POST}, {"TileOp::L1CopyFix", PIPE_MTE1, PIPE_MTE1, CoreType::AIC}, OpCalcType::MOVE_LOCAL, {FixpOpAttributeKey::fbAddrSpace});
    registerInfo(Opcode::OP_FIX_COPY_IN_QUANT_POST, OpCoreType::AIC, "FIX_COPY_IN_QUANT_POST", {MemoryType::MEM_L1}, {MemoryType::MEM_FIX_QUANT_POST}, {"TileOp::L1CopyFix", PIPE_MTE1, PIPE_MTE1, CoreType::AIC}, OpCalcType::MOVE_LOCAL, {FixpOpAttributeKey::fbAddrSpace});
    registerInfo(Opcode::OP_FIX_COPY_IN_ELT_ANTIQ, OpCoreType::AIC, "FIX_COPY_IN_ELT_ANTIQ", {MemoryType::MEM_L1}, {MemoryType::MEM_FIX_ELT_ANTIQ}, {"TileOp::L1CopyFix", PIPE_MTE1, PIPE_MTE1, CoreType::AIC}, OpCalcType::MOVE_LOCAL, {FixpOpAttributeKey::fbAddrSpace});
    registerInfo(Opcode::OP_FIX_COPY_IN_MTE2_ANTIQ, OpCoreType::AIC, "FIX_COPY_IN_MTE2_ANTIQ", {MemoryType::MEM_L1}, {MemoryType::MEM_FIX_MTE2_ANTIQ}, {"TileOp::L1CopyFix", PIPE_MTE1, PIPE_MTE1, CoreType::AIC}, OpCalcType::MOVE_LOCAL, {FixpOpAttributeKey::fbAddrSpace});
    registerInfo(Opcode::OP_BT_COPY_IN, OpCoreType::AIC, "BT_COPY_IN", {MemoryType::MEM_L1}, {MemoryType::MEM_BT}, {"TileOp::L1ToBT", PIPE_MTE1, PIPE_MTE1, CoreType::AIC}, OpCalcType::MOVE_IN);

    registerInfo(Opcode::OP_L1_COPY_UB, OpCoreType::AIC, "L1_COPY_UB", {MemoryType::MEM_L1}, {MemoryType::MEM_UB}, {"TileOp::L1CopyUB", PIPE_FIX, PIPE_FIX, CoreType::AIC}, OpCalcType::MOVE_OUT);
    registerInfo(Opcode::OP_L0C_COPY_UB, OpCoreType::AIC, "L0C_COPY_UB", {MemoryType::MEM_L0C}, {MemoryType::MEM_UB}, {"TileOp::L0CCopyUB", PIPE_FIX, PIPE_FIX, CoreType::AIC}, OpCalcType::MOVE_OUT);
    registerInfo(Opcode::OP_UB_COPY_L1, OpCoreType::AIV, "UB_COPY_L1", {MemoryType::MEM_UB}, {MemoryType::MEM_L1}, {"TileOp::UBCopyL1", PIPE_MTE3, PIPE_MTE3, CoreType::AIV}, OpCalcType::MOVE_IN);
    registerInfo(Opcode::OP_UB_COPY_L1_ND, OpCoreType::AIV, "UB_COPY_L1_ND", {MemoryType::MEM_UB}, {MemoryType::MEM_L1}, {"TileOp::UBCopyL1ND", PIPE_MTE3, PIPE_MTE3, CoreType::AIV}, OpCalcType::MOVE_IN);
    registerInfo(Opcode::OP_COPY_L1_TO_L1, OpCoreType::AIC, "L1_TO_L1", {MemoryType::MEM_L1}, {MemoryType::MEM_L1}, {"TileOp::L1CopyL1", PIPE_MTE1, PIPE_MTE1, CoreType::AIC}, OpCalcType::MOVE_LOCAL);
    registerInfo(Opcode::OP_COPY_UB_TO_UB, OpCoreType::AIV, "UB_TO_UB", {MemoryType::MEM_UB}, {MemoryType::MEM_UB}, {"TileOp::Tvcopy", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::MOVE_LOCAL);

    // ANY
    registerInfo(Opcode::OP_DUPLICATE, OpCoreType::ANY, "DUPLICATE", {}, {MemoryType::MEM_DEVICE_DDR}, {},
                 OpCalcType::MOVE_LOCAL);
    // Move
    registerInfo(Opcode::OP_RESHAPE, OpCoreType::ANY, "RESHAPE", {}, {}, {"TileOp::Treshape", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::OTHER, {OP_ATTR_PREFIX + "validShape"});
    registerInfo(Opcode::OP_ASSEMBLE, OpCoreType::ANY, "ASSEMBLE", {}, {MemoryType::MEM_DEVICE_DDR}, {"ASSEMBLE", PIPE_S, PIPE_S, CoreType::AIV}, OpCalcType::MOVE_LOCAL);
    registerInfo(Opcode::OP_VIEW, OpCoreType::ANY, "VIEW", {}, {}, {"VIEW", PIPE_S, PIPE_S, CoreType::AIV}, OpCalcType::MOVE_LOCAL);
    registerInfo(Opcode::OP_INDEX_OUTCAST, OpCoreType::ANY, "INDEX_OUTCAST", {MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_DEVICE_DDR},
        {MemoryType::MEM_DEVICE_DDR}, {"TileOp::TIndexoutcast", PIPE_MTE3, PIPE_MTE3, CoreType::AIV}, OpCalcType::MOVE_OUT, {"axis", OpAttributeKey::inputCombineAxis, OpAttributeKey::cacheMode, OpAttributeKey::panzBlockSize});
    registerInfo(Opcode::OP_CONVERT, OpCoreType::ANY, "CONVERT", {}, {}, {}, OpCalcType::SYS);
    registerInfo(Opcode::OP_COPY_IN, OpCoreType::ANY, "COPY_IN", {}, {}, {}, OpCalcType::MOVE_IN, {OpAttributeKey::outputCombineAxis});
    registerInfo(Opcode::OP_COPY_OUT, OpCoreType::ANY, "COPY_OUT", {}, {}, {}, OpCalcType::MOVE_OUT, {OP_ATTR_PREFIX + "atomic_add", OpAttributeKey::inputCombineAxis});
    // Special
    registerInfo(Opcode::OP_CALL, OpCoreType::ANY, "CALL", {}, {}, {}, OpCalcType::SYS);
    registerInfo(Opcode::OP_PRINT, OpCoreType::ANY, "OP_DUMP", {}, {}, {}, OpCalcType::SYS, {});
    registerInfo(Opcode::OP_CALL_NOT_EXPAND, OpCoreType::ANY, "CALL_NOT_EXPAND", {}, {}, {}, OpCalcType::SYS);
    // Begin: add for TOPK and ArgSort
    registerInfo(Opcode::OP_BITSORT, OpCoreType::ANY, "BITSORT", {MemoryType::MEM_UB}, {MemoryType::MEM_UB}, {"TileOp::BitSort", PIPE_S, PIPE_V, CoreType::AIV}, OpCalcType::OTHER, {OP_ATTR_PREFIX + "axis", OP_ATTR_PREFIX + "order"});
    registerInfo(Opcode::OP_MRGSORT, OpCoreType::ANY, "MRGSORT", {MemoryType::MEM_UB}, {MemoryType::MEM_UB}, {"TileOp::MrgSort", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::OTHER, {OP_ATTR_PREFIX + "axis", OP_ATTR_PREFIX + "order", OP_ATTR_PREFIX + "kvalue"});
    registerInfo(Opcode::OP_ARGSORT, OpCoreType::ANY, "ARGSORT", {MemoryType::MEM_UB}, {MemoryType::MEM_UB}, {"TileOp::ArgSort", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::OTHER, {OP_ATTR_PREFIX + "axis", OP_ATTR_PREFIX + "order", OP_ATTR_PREFIX + "kvalue"});
    registerInfo(Opcode::OP_EXTRACT, OpCoreType::ANY, "EXTRACT", {MemoryType::MEM_UB}, {MemoryType::MEM_UB}, {"TileOp::Extract", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::OTHER,
    {OP_ATTR_PREFIX + "kvalue", OP_ATTR_PREFIX + "order", OP_ATTR_PREFIX + "makeMode"});

    registerInfo(Opcode::OP_FUSED_OP, OpCoreType::AIV, "FUSED_OP", {MemoryType::MEM_UB, MemoryType::MEM_UB}, {MemoryType::MEM_UB}, {"TileOp::fusedOP", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::BROADCAST);
    registerInfo(Opcode::OP_VLD, OpCoreType::ANY, "VLD", {}, {}, {}, OpCalcType::MOVE_IN);
    registerInfo(Opcode::OP_VST, OpCoreType::ANY, "VST", {}, {}, {}, OpCalcType::MOVE_IN);

    // sync need check
    registerInfo(Opcode::OP_SYNC_SRC, OpCoreType::ANY, "SYNC_SRC", {}, {}, {"SYNC_SRC", PIPE_S, PIPE_S, CoreType::AIC}, OpCalcType::SYNC);
    registerInfo(Opcode::OP_SYNC_DST, OpCoreType::ANY, "SYNC_DST", {}, {}, {"SYNC_DST", PIPE_S, PIPE_S, CoreType::AIC}, OpCalcType::SYNC);
    registerInfo(Opcode::OP_CV_SYNC_SRC, OpCoreType::ANY, "CV_SYNC_SRC", {}, {}, {"CV_SYNC_SRC", PIPE_S, PIPE_S, CoreType::AIC}, OpCalcType::SYNC);
    registerInfo(Opcode::OP_CV_SYNC_DST, OpCoreType::ANY, "CV_SYNC_DST", {}, {}, {"CV_SYNC_DST", PIPE_S, PIPE_S, CoreType::AIC}, OpCalcType::SYNC);
    registerInfo(Opcode::OP_PHASE1, OpCoreType::ANY, "PHASE1", {}, {}, {"PHASE1", PIPE_S, PIPE_S, CoreType::AIC}, OpCalcType::SYNC);
    registerInfo(Opcode::OP_PHASE2, OpCoreType::ANY, "PHASE2", {}, {}, {"PHASE2", PIPE_S, PIPE_S, CoreType::AIC}, OpCalcType::SYNC);
    registerInfo(Opcode::OP_BAR_V, OpCoreType::ANY, "BAR.V", {}, {}, {"BAR.V", PIPE_S, PIPE_S, CoreType::AIC}, OpCalcType::SYNC);
    registerInfo(Opcode::OP_BAR_M, OpCoreType::ANY, "BAR.M", {}, {}, {"BAR.M", PIPE_S, PIPE_S, CoreType::AIC}, OpCalcType::SYNC);
    registerInfo(Opcode::OP_BAR_ALL, OpCoreType::ANY, "BAR.ALL", {}, {}, {"BAR.ALL", PIPE_S, PIPE_S, CoreType::AIC}, OpCalcType::SYNC);
    registerInfo(Opcode::OP_REMOTE_GATHER, OpCoreType::ANY, "REMOTE_GATHER", {MemoryType::MEM_UB, MemoryType::MEM_UB}, {MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB}, {"TileOp::Distributed::RemoteGather", PIPE_S, PIPE_S, CoreType::AIV}, OpCalcType::DISTRIBUTED);
    registerInfo(Opcode::OP_LOCAL_COPY_OUT, OpCoreType::ANY, "LOCAL_COPY_OUT", {MemoryType::MEM_UB, MemoryType::MEM_UB}, {MemoryType::MEM_DEVICE_DDR}, {"TileOp::Distributed::LocalCopyOut", PIPE_S, PIPE_S, CoreType::AIV}, OpCalcType::DISTRIBUTED);
    registerInfo(Opcode::OP_WRITE_REMOTE, OpCoreType::ANY, "WRITE_REMOTE", {MemoryType::MEM_UB, MemoryType::MEM_UB}, {MemoryType::MEM_UB}, {"TileOp::Distributed::WriteRemote", PIPE_S, PIPE_S, CoreType::AIV}, OpCalcType::DISTRIBUTED);
    registerInfo(Opcode::OP_REMOTE_REDUCE, OpCoreType::ANY, "REMOTE_REDUCE", {MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB, MemoryType::MEM_UB}, {MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB}, {"TileOp::Distributed::RemoteReduce", PIPE_S, PIPE_S, CoreType::AIV}, OpCalcType::DISTRIBUTED);
    registerInfo(Opcode::OP_COMM_WAIT_FLAG, OpCoreType::AICPU, "COMM_WAIT_FLAG", {MemoryType::MEM_DEVICE_DDR}, {MemoryType::MEM_DEVICE_DDR}, TileOpCfg(), OpCalcType::DISTRIBUTED, {OP_ATTR_PREFIX + "distributed"});
    registerInfo(Opcode::OP_DIST_REDUCE, OpCoreType::ANY, "DIST_REDUCE", {MemoryType::MEM_UB, MemoryType::MEM_UB}, {MemoryType::MEM_DEVICE_DDR}, TileOpCfg(), OpCalcType::DISTRIBUTED);
    registerInfo(Opcode::OP_DIST_SCATTER, OpCoreType::ANY, "DIST_SCATTER", {MemoryType::MEM_UB, MemoryType::MEM_UB}, {},
        TileOpCfg(), OpCalcType::DISTRIBUTED, {OP_ATTR_PREFIX + "rankOffset"});
    registerInfo(Opcode::OP_DIST_GATHER, OpCoreType::ANY, "DIST_GATHER", {MemoryType::MEM_UB, MemoryType::MEM_UB}, {MemoryType::MEM_DEVICE_DDR}, TileOpCfg(), OpCalcType::DISTRIBUTED);
    registerInfo(Opcode::OP_DIST_BROADCAST, OpCoreType::ANY, "DIST_BROADCAST", {MemoryType::MEM_UB, MemoryType::MEM_UB}, {MemoryType::MEM_DEVICE_DDR}, TileOpCfg(), OpCalcType::DISTRIBUTED);
    registerInfo(Opcode::OP_DEPEND_ON, OpCoreType::AICPU, "DEPEND_ON", {MemoryType::MEM_UB}, {MemoryType::MEM_DEVICE_DDR}, TileOpCfg(), OpCalcType::DISTRIBUTED);
    registerInfo(Opcode::OP_MOE_FFN_TO_ATTN, OpCoreType::ANY, "MOE_FFN_TO_ATTN", {MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_UB}, {MemoryType::MEM_UB}, {"TileOp::Distributed::FFN2Attn", PIPE_S, PIPE_S, CoreType::AIV}, OpCalcType::DISTRIBUTED);
    registerInfo(Opcode::OP_MOE_ATTN_COMBINE, OpCoreType::ANY, "MOE_ATTN_COMBINE", {MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB}, {MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_UB}, {"TileOp::Distributed::AttnCombine", PIPE_S, PIPE_S, CoreType::AIV}, OpCalcType::DISTRIBUTED);
    registerInfo(Opcode::OP_SEND_TO_ROUTING_EXPERT, OpCoreType::ANY, "SEND_TO_ROUTING_EXPERT", {MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB}, {MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_UB}, {"TileOp::Distributed::SendToRoutingExpert", PIPE_S, PIPE_S, CoreType::AIV}, OpCalcType::DISTRIBUTED);
    registerInfo(Opcode::OP_SEND_TO_SHARED_EXPERT, OpCoreType::ANY, "SEND_TO_SHARED_EXPERT", {MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB}, {MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB}, {"TileOp::Distributed::SendToSharedExpert", PIPE_S, PIPE_S, CoreType::AIV}, OpCalcType::DISTRIBUTED);
    registerInfo(Opcode::OP_COPY_TO_LOCAL_EXPERT, OpCoreType::ANY, "COPY_TO_LOCAL_EXPERT", {MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB}, {MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB}, {"TileOp::Distributed::CopyToLocalExpert", PIPE_S, PIPE_S, CoreType::AIV}, OpCalcType::DISTRIBUTED);
    registerInfo(Opcode::OP_DISPATCH_SET_FLAG, OpCoreType::ANY, "DISPATCH_SET_FLAG", {MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB}, {MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_UB}, {"TileOp::Distributed::DispatchSetFlag", PIPE_S, PIPE_S, CoreType::AIV}, OpCalcType::DISTRIBUTED);
    registerInfo(Opcode::OP_FFN_SCHED, OpCoreType::ANY, "FFN_SCHED", {MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB}, {MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB}, {"TileOp::Distributed::FFNSched", PIPE_S, PIPE_S, CoreType::AIV}, OpCalcType::DISTRIBUTED);
    registerInfo(Opcode::OP_FFN_BATCHING, OpCoreType::ANY, "FFN_BATCHING", {MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB}, {MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB}, {"TileOp::Distributed::FFNBatching", PIPE_S, PIPE_S, CoreType::AIV}, OpCalcType::DISTRIBUTED);

    registerInfo(Opcode::OP_NOP, OpCoreType::ANY, "NOP", {}, {}, {}, OpCalcType::OTHER);
    registerInfo(Opcode::OP_REDUCE_ACC, OpCoreType::GMATOMIC, "REDUCE_ACC", {MEM_DEVICE_DDR, MEM_DEVICE_DDR}, {MEM_DEVICE_DDR}, {}, OpCalcType::OTHER, {OP_ATTR_PREFIX + "atomic_add"});
    registerInfo(Opcode::OP_MAX_POOL, OpCoreType::AIV, "MAX_POOL", {MemoryType::MEM_UB}, {MemoryType::MEM_UB}, {"TileOp::Tmaxpool", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::OTHER);
    // check register ok
    // clang-format on
    ASSERT(strToEnum_.size() == static_cast<size_t>(Opcode::OP_UNKNOWN));
    }
} // namespace npu::tile_fwk
