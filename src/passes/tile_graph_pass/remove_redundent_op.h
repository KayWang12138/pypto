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
 * \file remove_redundent_op.h
 * \brief
 */

#ifndef REMOVE_REDUNDENT_OP_H
#define REMOVE_REDUNDENT_OP_H
#include "interface/operation/opcode.h"
#include "tilefwk/data_type.h"

#include "passes/pass_interface/pass.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"
#include "interface/function/function.h"
#include "passes/pass_utils/pass_utils.h"

#include "interface/function/function.h"
#include "interface/operation/operation.h"
#include "interface/tensor/logical_tensor.h"

namespace npu::tile_fwk {

/*
    RemoveRedundentOp: 如果op的类型为VIEW且op的输入tensor和输出tensor相同，则认为该VIEW op为冗余op，将其删除，
    并改变图中的连接关系.
*/
class RemoveRedundentOp : public Pass {
public:
    RemoveRedundentOp() : Pass("RemoveRedundentOp") {}
    ~RemoveRedundentOp() override = default;
private:
    Status PreCheck(Function &function) override;
    Status PostCheck(Function &function) override;
    Status RunOnFunction(Function &function) override;
    Status NeedToDelete(const Operation &op, Function &function, bool &needToDelete) const;
    Status DeleteCopyIn(Operation &op, Function &function, bool &needToDelete) const;
    Status RemoveDummyExpand(Function &function) const;
    Status DeleteRedundantOps(Function &function) const;
};
} // namespace npu::tile_fwk
#endif  // REMOVE_REDUNDENT_OP_H