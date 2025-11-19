/**
 * Copyright (c)  Huawei Technologies Co., Ltd. 2025. 
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file duplicate_op.h
 * \brief
 */

#pragma once

#include <vector>
#include "passes/pass_interface/pass.h"
namespace npu::tile_fwk {
/*
    DuplicateOp: 对于一个View OP，如果存在多消费者的情况，则为每一个消费者创建一个新的View OP;对于一个GatherIn OP，如果存在多消费者的情况，则为每一个消费者创建一个新的GatherIn OP
*/
class DuplicateOp : public Pass {
public:
    DuplicateOp() : Pass("DuplicateOp") {}
    ~DuplicateOp() override = default;

private:
    Status RunOnFunction(Function &function) override;
    Status ProcessOp(Function &function, Operation &operation) const;
    Status Process(Function &function) const;
    Status ProcessGatherIn(Function &function, Operation &operation) const;
    Status ProcessView(Function &function, Operation &operation) const;
};

}