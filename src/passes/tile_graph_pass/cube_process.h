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
 * \file cube_process.h
 * \brief
 */

#ifndef CUBE_PROCESS_H
#define CUBE_PROCESS_H

#include <vector>

#include "interface/operation/opcode.h"
#include "common/data_type.h"

#include "passes/pass_interface/pass.h"
#include "passes/tile_graph_pass/dead_operation_eliminate.h"
#include "tilefwk/tilefwk.h"
#include "tilefwk.h"
#include "interface/program/program.h"
#include "interface/function/function.h"
#include "passes/pass_utils/pass_utils.h"

namespace npu::tile_fwk {

const std::string ACC_A_MUL_B = OP_ATTR_PREFIX + "atomic_add";
const std::string A_MUL_B_NZ_ATTR = OP_ATTR_PREFIX + "matmul_nz_attr";
const std::string A_MUL_B_ACT_M = OP_ATTR_PREFIX + "act_m";
const std::string A_MUL_B_ACT_K = OP_ATTR_PREFIX + "act_k";
const std::string A_MUL_B_ACT_N = OP_ATTR_PREFIX + "act_n";
const std::string L1_COPY_IN_IS_NZ = OP_ATTR_PREFIX + "is_nz";
const std::string L1_COPY_IN_INNER = OP_ATTR_PREFIX + "inner_value";
const std::string L1_COPY_IN_OUTER = OP_ATTR_PREFIX + "outer_value";

class CubeProcess : public Pass {
public:
    CubeProcess() : Pass("CubeProcess") {}
    ~CubeProcess() override = default;

    Status RunOnFunction(Function &function) override;
    void EliminateReduceAcc(Function &function);
    void UpdateCubeOp(Function &function);
    void UpdateL1CopyInNz(Operation &op) const;
    void AddL1CopyInAttr(const std::shared_ptr<LogicalTensor> input, int nzValue, int mValue, int kValue, int nValue) const;
    bool IsFloat(const std::shared_ptr<LogicalTensor> tensor) const;
    bool IsInt(const std::shared_ptr<LogicalTensor> tensor) const;
};
}
#endif // CUBE_PROCESS_H