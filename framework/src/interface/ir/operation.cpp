/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file operation.cpp
 * \brief
 */

#include "ir/operation.h"
#include "ir/tile_graph_base.h"

namespace pto {

Operation::Operation()
    : Object(ObjectType::Operation),
      opcode_(Opcode::OP_INVALID) {}

Operation::Operation(Opcode opcode)
    : Object(ObjectType::Operation),
      opcode_(opcode) {}

Operation::Operation(Opcode opcode, std::string name)
    : Object(ObjectType::Operation, std::move(name)),
      opcode_(opcode) {}

Operation::Operation(Opcode opcode,
                     ValuePtrs inputs,
                     ValuePtrs outputs,
                     std::string name)
    : Object(ObjectType::Operation, std::move(name)),
      ioperands_(std::move(inputs)),
      ooperands_(std::move(outputs)),
      opcode_(opcode) {
    for (size_t i = 0; i < inputs.size(); i++) {
        if (std::dynamic_pointer_cast<ScalarValue>(inputs[i])) {
            iScalarIndex_ = i;
            break;
        }
    }
    for (size_t i = 0; i < outputs.size(); i++) {
        if (std::dynamic_pointer_cast<ScalarValue>(outputs[i])) {
            oScalarIndex_ = i;
            break;
        }
    }
}

// TileBaseOp implementation
std::shared_ptr<TileValue> TileBaseOp::GetInOperand(size_t index) const {
    return std::static_pointer_cast<TileValue>(GetInputOperand(index));
}

std::shared_ptr<TileValue> TileBaseOp::GetOutOperand(size_t index) const {
    return std::static_pointer_cast<TileValue>(GetOutputOperand(index));
}

// ScalarBaseOp implementation
ScalarValuePtr ScalarBaseOp::GetInOperand(size_t index) const {
    return std::static_pointer_cast<ScalarValue>(GetInputOperand(index));
}

ScalarValuePtr ScalarBaseOp::GetOutOperand(size_t index) const {
    return std::static_pointer_cast<ScalarValue>(GetOutputOperand(index));
}

// MatmulTileBaseOp implementation
ScalarValuePtr MatmulTileBaseOp::GetOffset(size_t index) const {
    if (index >= offsets_.size()) {
        return nullptr;
    }
    return offsets_[index];
}

} // namespace pto
