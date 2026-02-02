/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file merge_view.h
 * \brief
 */

#ifndef PASS_MERGE_VIEW_H_
#define PASS_MERGE_VIEW_H_

#include "interface/function/function.h"
#include "interface/tensor/logical_tensor.h"
#include "interface/configs/config_manager.h"

namespace npu::tile_fwk {
class MergeView {
public:
    MergeView() = default;
    ~MergeView() = default;
    
    Status MergeViewOp(Function &function);

    Status Initialize();

    void InitOperationChain(Operation &operation, std::vector<Operation *> &chain);

    Status MergeViewChain(Function &function, Operation &operation, std::vector<Operation *> &chain);

    Status ProcessConsumerChain(Function &function,
                              const std::set<Operation*, LogicalTensor::CompareOp>& consumers,
                              std::vector<Operation *> &chain,
                              bool &chainEnd);

    Status ProcessChainEnd(Function &function,
                         std::vector<Operation *> &chain);

    Status CalculateMergedOffsets(const std::vector<Operation *> &chain, std::vector<int64_t> &newOffset,
        std::vector<SymbolicScalar> &newDynOffset, std::vector<SymbolicScalar> &newDynValidShape);

    void RecordMergedViewOperation(Operation* lastViewOp, const std::shared_ptr<LogicalTensor> &startTensor,
        const std::shared_ptr<LogicalTensor> &endTensor, const std::vector<int64_t> &newOffset,
        const std::vector<SymbolicScalar> &newDynOffset, const std::vector<SymbolicScalar> &newDynValidShape);
    
    Status AppendMergedViewOperations(Function &function);

    Status EraseRedundantAssemble(Function &function) const;

    Status CleanUp(Function &function);
    
    struct ViewOp {
        std::shared_ptr<LogicalTensor> input;
        std::shared_ptr<LogicalTensor> output;
        std::vector<int64_t> offset;
        std::vector<SymbolicScalar> dynOffset;
        std::vector<SymbolicScalar> dynValidShape;
        MemoryType toType = MemoryType::MEM_UNKNOWN;
        bool hasCopyInMode;     // 是否有copy_in_mode属性
        npu::tile_fwk::Any copyInModeValue;    // copy_in_mode属性值
    };
    std::unordered_set<int> visitedOp_;
    std::vector<ViewOp> viewOpToAppend_;
};
}
#endif // PASS_MERGE_VIEW_H_