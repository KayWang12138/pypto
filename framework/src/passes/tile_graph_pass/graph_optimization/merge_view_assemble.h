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
 * \file merge_view_assemble.h
 * \brief
 */

#ifndef PASS_MERGE_VIEW_ASSEMBLE_H_
#define PASS_MERGE_VIEW_ASSEMBLE_H_

#include "interface/function/function.h"
#include "interface/tensor/logical_tensor.h"
#include "interface/configs/config_manager.h"
#include "passes/pass_interface/pass.h"
#include "passes/pass_utils/merge_view.h"

namespace npu::tile_fwk {
class MergeViewAssemble : public Pass {
public:
    MergeViewAssemble() : Pass("MergeViewAssemble") {}
    ~MergeViewAssemble() override = default;

private:
    struct AssembleOp {
        std::shared_ptr<LogicalTensor> input;
        std::shared_ptr<LogicalTensor> output;
        std::vector<int64_t> offset;
        std::vector<SymbolicScalar> dynOffset;
    };
    Status RunOnFunction(Function &function) override;
    // View chain processing methods

    void InitOperationChain(Operation &operation, std::vector<Operation *> &chain);

    // Assemble chain processing methods
    Status MergeAssembleChain(Function &function, Operation &operation, std::vector<Operation *> &chain);
    void InitAssembleChain(Operation &operation, std::vector<Operation *> &chain);

    Status ProcessAssembleConsumers(Function &function,
                                  const std::set<Operation*, LogicalTensor::CompareOp>& consumers,
                                  std::vector<Operation *> &chain,
                                  bool &chainEnd, bool& hasAssembleConsumer);

    Status ProcessAssembleChainEnd(Function &function,
                                 std::vector<Operation *> &chain,
                                 Operation &operation);

    std::pair<std::vector<int64_t>, std::vector<SymbolicScalar>> CalculateAssembleOffsets(
        const std::vector<Operation *> &chain, size_t offsetSize);

    void RecordAssembleOperation(const std::shared_ptr<LogicalTensor> &input,
        const std::shared_ptr<LogicalTensor> &output, const std::vector<int64_t> &offset,
        const std::vector<SymbolicScalar> &dynOffset);

    // Processing methods
    Status ProcessOperations(Function &function);
    Status ProcessAssembleOperations(Function &function, Operation& op);

    // Operation appending methods
    Status AppendMergedAssembleOperations(Function &function);

    // Cleanup methods
    Status CleanUp(Function &function);
    Status EraseRedundantAssemble(Function &function) const;
    std::unordered_set<int> assembleWithoutAssembleConsumer_;
    std::vector<AssembleOp> assembleOpToAppend_;
    MergeView mergeView;
};
} // using namespace npu::tile_fwk
#endif // PASS_MERGE_VIEW_ASSEMBLE_H_
