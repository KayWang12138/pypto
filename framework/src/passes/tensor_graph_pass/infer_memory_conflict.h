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
#include <queue>
#include <unordered_map>

#include "passes/pass_interface/pass.h"
#include "interface/inner/tilefwk.h"
#include "interface/tensor/logical_tensor.h"

namespace npu {
namespace tile_fwk {
class InferMemoryConflict : public Pass {
public:
    InferMemoryConflict() : Pass("InferMemoryConflict") {}
    ~InferMemoryConflict() override = default;

private:
    Status RunOnFunction(Function &function) override;
    Status Init(Function& function);
    Status ForwardPropagation(Function &function);
    Status UpdateForwardTensor(Function &function, const LogicalTensorPtr &curTensor, Operation* consumer, std::queue<LogicalTensorPtr> &curTensors);
    Status BackwardPropagation(Function &function);
    Status UpdateBackwardTensor(const LogicalTensorPtr &curTensor, Operation* producer, std::queue<LogicalTensorPtr> &curTensors);
    Status InsertPrecededCopys(Function &function);
    Status InsertPostCopys(Function &function);
    Status InsertCopys(Function& function);
    Status InferTileShape(Operation &op, Operation *parentOp, const LogicalTensorPtr &tensor);
    Status SetDefaultShape(const LogicalTensorPtr &tensor, std::vector<int64_t> &defaultTile);

    bool CheckTransmit(Operation* curOp);
    bool CheckConflict(const LogicalTensorPtr &inTensor, const LogicalTensorPtr &outTensor);
    bool CheckRawShapeConflict(const LogicalTensorPtr &inTensor, const LogicalTensorPtr &outTensor);
    bool IsValidTileShape(const Operation &op) const;

    std::set<Operation*> preregcopys;
    std::set<Operation*> postregcopys;
    std::unordered_map<LogicalTensorPtr, LogicalTensorPtr> memoryInfo;
};
} // namespace tile_fwk
} // namespace npu
#endif // PASS_INFER_MEMORY_CONFLICT_H_