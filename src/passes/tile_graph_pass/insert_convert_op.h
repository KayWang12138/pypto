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
 * \file insert_convert_op.h
 * \brief
 */

#ifndef PASS_INSERT_CONVERT_OP_H_
#define PASS_INSERT_CONVERT_OP_H_

#include "interface/function/function.h"
#include "interface/tensor/logical_tensor.h"
#include "interface/tensor/raw_tensor.h"
#include "interface/operation/attribute.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"
#include "passes/tile_graph_pass/dead_operation_eliminate.h"
#include "passes/pass_interface/pass.h"

namespace npu::tile_fwk {
struct ConvertOp {
    MemoryType from;
    MemoryType to;
    std::shared_ptr<LogicalTensor> input;
    std::shared_ptr<LogicalTensor> output;
};

class InsertConvertOp : public Pass, public DeadOperationEliminator {
public:
    InsertConvertOp() : Pass("InsertConvertOp") {}
    ~InsertConvertOp() override = default;

    Status PreCheck(Function &function) override;
    Status PostCheck(Function &function) override;
    Status RunOnFunction(Function &function) override;
    void RunOnOperation(Function &function, const Operation &operation);
    void CheckUnknown(Function &function) const;
    bool CrossCore(const std::shared_ptr<LogicalTensor>& tensor) const;
    void UpdateConsumerAndReconnect(std::shared_ptr<LogicalTensor> oldTensor, std::shared_ptr<LogicalTensor> newTensor, 
        Operation* op) const;

    std::vector<ConvertOp> converts;
    std::unordered_map<int, std::shared_ptr<RawTensor>> oldRawToNewRaw;
};
}
#endif // PASS_INSERT_CONVERT_OP_H_