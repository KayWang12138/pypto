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
 * \file assign_memory_type.h
 * \brief
 */

#ifndef TILE_FWK_ASSIGN_MEMORY_TYPE_H
#define TILE_FWK_ASSIGN_MEMORY_TYPE_H

#include <queue>
#include "passes/pass_interface/pass.h"
#include "interface/operation/opcode.h"
#include "passes/tile_graph_pass/convert_op_inserter.h"
#include "passes/pass_check/assign_memory_type_checker.h"
#include "tilefwk/data_type.h"

namespace npu::tile_fwk {
class AssignMemoryType : public Pass {
public:
    AssignMemoryType() : Pass("AssignMemoryType") {}
    void SpecialCallInterfaceToBeDeleted(Function &function) {
        RunOnFunction(function);
    }
private:
    Status PreCheck(Function &function) override;    
    Status RunOnFunction(Function &function) override;
    void AssignMoveOp(Operation &operation);
    void RunOnOperation(Operation &operation);
    void AssignMemUnknown(Function &function);
    void AssignL1CopyIn(Function &function);
    void AssignSpecialOpMemtype(Operation &op);
    std::string PrintTensorMem(std::shared_ptr<LogicalTensor>& tensor) const;
    ConvertInserter inserter;
};
} // namespace npu::tile_fwk

#endif // TILE_FWK_ASSIGN_MEMORY_TYPE_H
