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
 * \file codegen_common.h
 * \brief
 */

#ifndef CODEGEN_COMMON_H
#define CODEGEN_COMMON_H

#include <iostream>

#include "interface/utils/common.h"
#include "codegen_symbol.h"

namespace npu::tile_fwk {
const std::string GM_TENSOR_PARAM_STR = "param";
const std::string PREFIX_STR_RAW_SHAPE = "RAWSHAPE";
const std::string PREFIX_STR_OFFSET = "OFFSET";
constexpr const int MAX_DIM = 5;

const std::string GET_PARAM_VALID_SHAPE_BY_IDX = "GET_PARAM_VALID_SHAPE_BY_IDX";
const std::string GET_PARAM_OFFSET_BY_IDX = "GET_PARAM_OFFSET_BY_IDX";
const std::string SUPPORT_DYNAMIC_UNALIGNED = "SUPPORT_DYNAMIC_UNALIGNED";

const std::string GM_PARAM_TYPE_FOR_STATIC = "__gm__ GMTensorInfo";
const std::string GM_PARAM_TYPE_FOR_DYN = "CoreFuncParam";
const std::string GM_STACK_BASE = "GMStackBase";

constexpr const int K_BYTES_OF16_BIT = 2;
constexpr const int K_BYTES_OF32_BIT = 4;

constexpr const unsigned ID0 = 0;
constexpr const unsigned ID1 = 1;
constexpr const unsigned ID2 = 2;
constexpr const unsigned ID3 = 3;
constexpr const unsigned ID4 = 4;

constexpr const int BUFFER_SIZE_256 = 256;
constexpr const int BUFFER_SIZE_512 = 512;
constexpr const int BUFFER_SIZE_1024 = 1024;

const std::unordered_map<OperandType, std::string> OPERAND_TYPE_TO_ADDR_TYPE{
    {BUF_DDR,   "__gm__"},
    { BUF_UB, "__ubuf__"},
    { BUF_L1, "__cbuf__"},
    {BUF_L0A,   "__ca__"},
    {BUF_L0B,   "__cb__"},
    {BUF_L0C,   "__cc__"},
    {BUF_FIX, "__fbuf__"},
    { BUF_BT,   "__cc__"},
};

const std::map<PipeType, std::string> PIPE_ID{
    {PIPE_MTE1, "PIPE_MTE1"},
    {PIPE_MTE2, "PIPE_MTE2"},
    {PIPE_MTE3, "PIPE_MTE3"},
    {   PIPE_V,    "PIPE_V"},
    {   PIPE_M,    "PIPE_M"},
    { PIPE_FIX,  "PIPE_FIX"},
    {   PIPE_S,    "PIPE_S"},
    { PIPE_ALL,  "PIPE_ALL"},
};

const std::map<OperandType, const char *> BUFFER_TYPE_TO_PREFIX = {
    { OperandType::BUF_UB,   "UB"},
    { OperandType::BUF_L1,   "L1"},
    {OperandType::BUF_L0A,  "L0A"},
    {OperandType::BUF_L0B,  "L0B"},
    {OperandType::BUF_L0C,  "LOC"},
    {OperandType::BUF_FIX, "FBUF"},
    { OperandType::BUF_BT,   "BT"},
    {OperandType::BUF_DDR,  "DDR"},
};

} // namespace npu::tile_fwk

#endif // CODEGEN_COMMON_H
