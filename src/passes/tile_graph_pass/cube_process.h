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
#include "tilefwk/data_type.h"

#include "passes/pass_interface/pass.h"
#include "passes/tile_graph_pass/dead_operation_eliminate.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"
#include "interface/function/function.h"
#include "passes/pass_utils/pass_utils.h"

namespace npu::tile_fwk {

const std::string ACC_A_MUL_B = OP_ATTR_PREFIX + "atomic_add";
const std::string MATMUL_NZ_ATTR = OP_ATTR_PREFIX + "matmul_nz_attr";
const std::string A_MUL_B_ACT_M = OP_ATTR_PREFIX + "act_m";
const std::string A_MUL_B_ACT_K = OP_ATTR_PREFIX + "act_k";
const std::string A_MUL_B_ACT_N = OP_ATTR_PREFIX + "act_n";

/* 是否做搬运随路转Nz */
const std::string COPY_IS_NZ = OP_ATTR_PREFIX + "is_nz";

/* L1 Copy In 的内外轴大小 */
const std::string L1_COPY_IN_INNER = OP_ATTR_PREFIX + "inner_value";
const std::string L1_COPY_IN_OUTER = OP_ATTR_PREFIX + "outer_value";

/* L0C Copy Out 的内外轴大小 */
const std::string L0C_COPY_OUT_OUTER = OP_ATTR_PREFIX + "curH";
const std::string L0C_COPY_OUT_INNER = OP_ATTR_PREFIX + "curW";

class CubeProcess : public Pass {
public:
    CubeProcess() : Pass("CubeProcess") {}
    ~CubeProcess() override = default;

    Status PreCheck(Function &function) override;
    Status RunOnFunction(Function &function) override;
    Status EliminateReduceAcc(Function &function);
    Status UpdateCubeOp(Function &function);
    Status UpdateCopyAttr(Operation &op) const;
    Status AddL1CopyInAttr(
        const std::shared_ptr<LogicalTensor> input, int nzValue, int mValue, int kValue, int nValue) const;
    Status AddL0cCopyOutAttr(const std::shared_ptr<LogicalTensor> output, int nzValue, int mValue, int nValue) const;
    bool IsFloat(const std::shared_ptr<LogicalTensor> tensor) const;
    bool IsInt(const std::shared_ptr<LogicalTensor> tensor) const;
};
}
#endif // CUBE_PROCESS_H