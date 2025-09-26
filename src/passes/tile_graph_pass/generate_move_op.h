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
 * \file generate_move_op.h
 * \brief
 */

#ifndef PASS_GENERATE_MOVE_OP_H_
#define PASS_GENERATE_MOVE_OP_H_

#include "passes/pass_interface/pass.h"
#include "passes/tile_graph_pass/dead_operation_eliminate.h"

namespace npu::tile_fwk {
/*
    GenerateMoveOp: 将view和sassemble翻译成copyin和copyout,并将连续的copyin和copyout合并为一个，删除冗余copyout
*/
class GenerateMoveOp : public Pass {
public:
    GenerateMoveOp() : Pass("GenerateMoveOp") {}
    ~GenerateMoveOp() override = default;
private:
    Status PreCheck(Function &function) override;
    Status PostCheck(Function &function) override;
    Status RunOnFunction(Function &function) override;
    void CreateMoveOp(Function &function) const;
    void MergeMoveOp(Function &function) const;
    void MergeCopyInCopyOut(Function &function, Operation &operation) const;
    void EraseRedundantCopyOut(Function &function) const;
    bool HasSpecificConsumer(const Operation &op) const;
    void ConvertViewToCopyInWhenInputGm(Operation &op, ViewOpAttribute *viewOpAttribute) const;
    void CreateMoveOpForView(Operation &op) const;
    void CreateMoveOpForAssemble(Operation &op) const;
    void CreateMoveOpForConvert(Operation &op) const;
};
}
#endif // PASS_GENERATE_MOVE_OP_H_