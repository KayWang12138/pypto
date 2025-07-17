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
 * \file codegen_vf.h
 * \brief
 */

#ifndef CODEGEN_VF_H
#define CODEGEN_VF_H

#include <utility>
#include <unordered_set>

#include "codegen/codegen_common.h"
#include "common/data_type.h"
#include "interface/operation/operation.h"
#include "interface/tensor/tensormap.h"
#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"
#include "codegen/codegen_symbol.h"
#include "codegen_op_cloudnpu.h"

namespace npu::tile_fwk {
class VFCodegen {
public:
    VFCodegen(){};

    bool GenCode(Function *func, std::string file);

    std::string genVFHeader(const std::string &KernelName, std::vector<std::shared_ptr<LogicalTensor>> &ubIn,
        std::vector<std::shared_ptr<LogicalTensor>> &ubOut);
    std::string genVLD(const std::string &code);
    std::string genVST(const std::string &code);
    std::string genRegAlloc();
    std::string genBinaryRegOp(const std::string &BinaryOp);
    std::string genUnaryRegOp(const std::string &UnaryOp);
    std::string genVFBody(const std::vector<Operation *> &OpList);
    std::string genSingleOp(Operation *op);
    std::string genVFEnd();
    std::string genVarName(std::string loc, int id);

private:
    std::string path_;
    std::map<int, int> magicToBufferId_;
    int operand[MAX_OPERANDS];
    std::vector<int> offset[MAX_OPERANDS] = {};
    std::vector<int> shape[MAX_OPERANDS] = {};
    std::vector<int> rawShape[MAX_OPERANDS] = {};
    std::vector<int> originShape[MAX_OPERANDS] = {};
    DataType operandDtype[MAX_OPERANDS] = {DataType::DT_BOTTOM, DataType::DT_BOTTOM, DataType::DT_BOTTOM,
        DataType::DT_BOTTOM, DataType::DT_BOTTOM, DataType::DT_BOTTOM, DataType::DT_BOTTOM,
        DataType::DT_BOTTOM};
    void UpdateVarOffset(std::vector<std::string *> vars, std::vector<unsigned int> operandIdxes) const;
    void InitOpParm(Operation *op);
    void AllocBufferId(Function *func, std::vector<Operation *> &opList);
};

} // namespace npu::tile_fwk

#endif // CODEGEN_VF_H
