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
 * \file group_operations.h
 * \brief
 */

#ifndef PASS_GROUP_OPERATIONS_H_
#define PASS_GROUP_OPERATIONS_H_

#include <vector>
#include "passes/pass_interface/pass.h"
#include "tilefwk/data_type.h"

namespace npu::tile_fwk {
class GroupOperationsOp : public Pass {
public:
    GroupOperationsOp() : Pass("GroupOperationsOp") {}
    ~GroupOperationsOp() override = default;

private:
    Status RunOnFunction(Function &function) override;

    void RunOnOperation(Function &function, Operation &op) const;

    std::map<MemoryType, std::set<MemoryType>> memoryLinks;
};

}
#endif // PASS_GROUP_OPERATIONS_H_