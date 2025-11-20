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
 * \file codegen_vf.h
 * \brief
 */

#ifndef CODEGEN_VF_H
#define CODEGEN_VF_H

#include <utility>
#include <unordered_set>

#include "codegen/codegen_common.h"
#include "tilefwk/data_type.h"
#include "interface/operation/operation.h"
#include "interface/tensor/tensormap.h"
#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"
#include "codegen/symbol_mgr/codegen_symbol.h"
#include "codegen_op_cloudnpu.h"

namespace npu::tile_fwk {
class VFCodeGen {
public:
    VFCodeGen() = default;

    void GenCode(Function *func, const std::string &file);

    std::string GenVFHeader(const std::string &KernelName, std::vector<std::shared_ptr<LogicalTensor>> &ubIn,
        std::vector<std::shared_ptr<LogicalTensor>> &ubOut);
    std::string GenVLD(const std::string &code);
    std::string GenVST(const std::string &code);
    std::string GenRegAlloc();
    std::string GenBinaryRegOp(const std::string &BinaryOp);
    std::string GenUnaryRegOp(const std::string &UnaryOp);
    std::string GenVFBody(const std::vector<Operation *> &OpList);
    std::string GenSingleOp(Operation *op);
    std::string GenVFEnd();
    std::string GenVarName(std::string loc, int id);
    bool IsGenSuccess() const { return isGenSuccess_; };
    std::string GetVFHeaderForInclude() const;

private:
    std::string path_;
    std::map<int, int> magicToBufferId_;
    int operand[MAX_OPERANDS];
    std::vector<int64_t> offset[MAX_OPERANDS] = {};
    std::vector<int64_t> shape[MAX_OPERANDS] = {};
    std::vector<int64_t> rawShape[MAX_OPERANDS] = {};
    std::vector<int64_t> originShape[MAX_OPERANDS] = {};
    bool isGenSuccess_{false};
    DataType operandDtype[MAX_OPERANDS] = {DataType::DT_BOTTOM, DataType::DT_BOTTOM, DataType::DT_BOTTOM,
        DataType::DT_BOTTOM, DataType::DT_BOTTOM, DataType::DT_BOTTOM, DataType::DT_BOTTOM, DataType::DT_BOTTOM};
    void UpdateVarOffset(std::vector<std::string *> vars, std::vector<unsigned int> operandIdxes) const;
    void InitOpParm(Operation *op);
    void AllocBufferId(Function *func, std::vector<Operation *> &opList);
};

} // namespace npu::tile_fwk

#endif // CODEGEN_VF_H
