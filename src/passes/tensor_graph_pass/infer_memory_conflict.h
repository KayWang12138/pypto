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
 * \file infer_memory_conflict.h
 * \brief
 */

#ifndef PASS_INFER_MEMORY_CONFLICT_H_
#define PASS_INFER_MEMORY_CONFLICT_H_

#include <vector>
#include <unordered_map>

#include "interface/operation/opcode.h"
#include "tilefwk/data_type.h"

#include "passes/pass_interface/pass.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"
#include "interface/function/function.h"
#include "passes/pass_utils/pass_utils.h"
#include "interface/configs/config_manager.h"
#include "interface/tensor/logical_tensor.h"

namespace npu {
namespace tile_fwk {
/*
key: Opcode类型
vaule: vector of pair, 每个pair记录了第几个输入和第几个输出存在inplace关系
*/
const std::unordered_map<Opcode, std::vector<std::pair<size_t, size_t>>> inplaceRelationshipMap = {
    {         Opcode::OP_VIEW, {std::pair<size_t, size_t>{0, 0}}},
    {     Opcode::OP_ASSEMBLE, {std::pair<size_t, size_t>{0, 0}}},
    {      Opcode::OP_RESHAPE, {std::pair<size_t, size_t>{0, 0}}},
    {Opcode::OP_INDEX_OUTCAST, {std::pair<size_t, size_t>{2, 0}}},
};

class InferMemoryConflict : public Pass {
public:
    InferMemoryConflict() : Pass("InferMemoryConflict") {}
    ~InferMemoryConflict() override = default;

private:
    // Status PreCheck(Function &function) override;
    // Status PostCheck(Function &function) override;
    Status RunOnFunction(Function &function) override;
    Status InferFromIncast(Function &function);
    Status InsertTensorCopy(Function &function);

    std::pair<Status, bool> IsInplace(Operation &op, std::shared_ptr<LogicalTensor> in, std::shared_ptr<LogicalTensor> out) const;

    std::unordered_map<std::shared_ptr<LogicalTensor>, std::shared_ptr<LogicalTensor>>
        parentRawForard; // key: 当前tensor的magic，value: parent tensor 的 raw magic
    std::set<std::shared_ptr<LogicalTensor>> conflictTensors;
};
} // namespace tile_fwk
} // namespace npu
#endif // PASS_INFER_MEMORY_CONFLICT_H_