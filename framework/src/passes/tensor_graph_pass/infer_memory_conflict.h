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
    Status InferFromIncast(Function &function);
    Status InsertTensorCopy(Function &function);
    void Init(Function& function);
    bool IsValidTileShape(const Operation &op) const;
    std::vector<std::pair<LogicalTensorPtr, Operation *>> FilterCopyScenes(Function &function, LogicalTensorPtr targetTensor,
        const std::vector<std::pair<LogicalTensorPtr, Operation *>> &);
    std::unordered_map<LogicalTensorPtr, LogicalTensorPtr> parentRawTensor_; // key: 当前tensor的magic，value: parent tensor 的 raw magic
    std::map<LogicalTensorPtr, std::vector<std::pair<LogicalTensorPtr, Operation *>>> insertCopys_;
    std::map<Operation *, size_t> opInputDegree_;
};
} // namespace tile_fwk
} // namespace npu
#endif // PASS_INFER_MEMORY_CONFLICT_H_