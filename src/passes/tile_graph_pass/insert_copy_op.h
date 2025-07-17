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
 * \file insert_copy_op.h
 * \brief
 */

#ifndef PASS_INSERT_COPY_OP_H_
#define PASS_INSERT_COPY_OP_H_

#include <vector>
#include "interface/operation/opcode.h"
#include "common/data_type.h"
#include "passes/pass_interface/pass.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"

namespace npu::tile_fwk {
struct CopyOp {
    std::shared_ptr<LogicalTensor> input;
    std::shared_ptr<LogicalTensor> ddr;
    std::shared_ptr<LogicalTensor> output;
    Operation *usedOp;
};
class InsertCopyOpPass : public Pass {
public:
    InsertCopyOpPass() : Pass("InsertCopyOpPass") {}
    ~InsertCopyOpPass() override = default;

private:
    Status RunOnFunction(Function &function) override;
    void SplitTensor(Function &function);
    void CreateCopyOp(Function &function);

    std::vector<CopyOp> copysToCreate;
};
} // namespace npu::tile_fwk
#endif // PASS_INSERT_COPY_OP_H_