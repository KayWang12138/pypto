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
 * \file merge_view_assemble.h
 * \brief
 */

#ifndef PASS_MERGE_VIEW_ASSEMBLE_H_
#define PASS_MERGE_VIEW_ASSEMBLE_H_

#include "interface/function/function.h"
#include "interface/tensor/logical_tensor.h"
#include "passes/tile_graph_pass/dead_operation_eliminate.h"
#include "interface/configs/config_manager.h"
#include "passes/pass_interface/pass.h"

namespace npu::tile_fwk {
class MergeViewAssemble : public Pass {
public:
    MergeViewAssemble() : Pass("MergeViewAssemble") {}
    ~MergeViewAssemble() override = default;

private:
    struct ViewOp {
        std::shared_ptr<LogicalTensor> input;
        std::shared_ptr<LogicalTensor> output;
        std::vector<int32_t> offset;
        std::vector<SymbolicScalar> dynOffset;
        std::vector<SymbolicScalar> dynValidShape;
    };
    struct AssembleOp {
        std::shared_ptr<LogicalTensor> input;
        std::shared_ptr<LogicalTensor> output;
        std::vector<int32_t> offset;
        std::vector<SymbolicScalar> dynOffset;
    };
    Status RunOnFunction(Function &function) override;
    // View chain processing methods
    Status MergeViewChain(Function &function, Operation &operation, std::vector<Operation *> &chain);
    
    void InitOperationChain(Operation &operation, 
                           std::vector<Operation *> &chain);
    
    Status ProcessConsumerChain(Function &function,
                              const std::set<Operation*, LogicalTensor::CompareOp>& consumers,
                              std::vector<Operation *> &chain,
                              bool &chainEnd);
    
    Status ProcessChainEnd(Function &function,
                         std::vector<Operation *> &chain);
    
    Status CalculateMergedOffsets(const std::vector<Operation *> &chain,
                                std::vector<int32_t> &newOffset,
                                std::vector<SymbolicScalar> &newDynOffset,
                                std::vector<SymbolicScalar> &newDynValidShape);
    
    void RecordMergedViewOperation(const std::shared_ptr<LogicalTensor> &startTensor,
                                 const std::shared_ptr<LogicalTensor> &endTensor,
                                 const std::vector<int32_t> &newOffset,
                                 const std::vector<SymbolicScalar> &newDynOffset,
                                const std::vector<SymbolicScalar> &newDynValidShape);

    // Assemble chain processing methods
    Status MergeAssembleChain(Function &function, Operation &operation, std::vector<Operation *> &chain);
    void InitAssembleChain(Operation &operation, 
                          std::vector<Operation *> &chain);
    
    Status ProcessAssembleConsumers(Function &function,
                                  const std::set<Operation*, LogicalTensor::CompareOp>& consumers,
                                  std::vector<Operation *> &chain,
                                  bool &chainEnd);
    
    Status ProcessAssembleChainEnd(Function &function,
                                 std::vector<Operation *> &chain,
                                 Operation &operation);
    
    std::pair<std::vector<int32_t>, std::vector<SymbolicScalar>> 
    CalculateAssembleOffsets(const std::vector<Operation *> &chain,
                            size_t offsetSize);
    
    void RecordAssembleOperation(const std::shared_ptr<LogicalTensor> &input,
                               const std::shared_ptr<LogicalTensor> &output,
                               const std::vector<int32_t> &offset,
                               const std::vector<SymbolicScalar> &dynOffset);

    // Common methods
    Status Initialize();

    // Processing methods
    Status ProcessViewOperations(Function &function);
    Status ProcessAssembleOperations(Function &function);

    // Operation appending methods
    Status AppendMergedViewOperations(Function &function);
    Status AppendMergedAssembleOperations(Function &function);

    // Cleanup methods
    Status CleanUp(Function &function);
    Status EraseRedundantAssemble(Function &function) const;
    std::set<int32_t> visitedOp_;
    std::vector<ViewOp> viewOpToAppend_;
    std::vector<AssembleOp> assembleOpToAppend_;
};
} // using namespace npu::tile_fwk
#endif // PASS_MERGE_VIEW_ASSEMBLE_H_
