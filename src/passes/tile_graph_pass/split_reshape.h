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
 * \file split_reshape.h
 * \brief
 */

#ifndef PASS_SPLIT_RESHAPE_H_
#define PASS_SPLIT_RESHAPE_H_

#include "interface/function/function.h"
#include "interface/tensor/logical_tensor.h"
#include "interface/configs/config_manager.h"
#include "passes/pass_utils/pass_common_defs.h"
#include "passes/tile_graph_pass/dead_operation_eliminate.h"
#include "passes/pass_interface/pass.h"

namespace npu::tile_fwk {
using InputMaigc = int;
using OutputMaigc = int;
using OverlaprawMagic = int;
inline constexpr uint32_t WARNING = 2;

class ReshapeOp {
    public:
        ReshapeOp(std::shared_ptr<LogicalTensor> aInput, std::shared_ptr<LogicalTensor> aOutput)
            : input(aInput), output(aOutput) {}

        std::shared_ptr<LogicalTensor> input;
        std::shared_ptr<LogicalTensor> output;
};

struct ReshapeTilePara {
    std::vector<int> shape;
    std::vector<int> newShape;
    std::vector<int> tileOffset;
    std::vector<int> tileShape;
};

struct copyOutTilePara {
    LogicalTensorPtr reshapeSource;
    LogicalTensorPtr inputView;
    LogicalTensorPtr newInputView;
    std::vector<int32_t> alignedShape;
    std::vector<SymbolicScalar> validShape;
};

struct PerfectlyMatchPara {
    LogicalTensorPtr input;
    LogicalTensorPtr output;
    LogicalTensorPtr overlap;
    LogicalTensorPtr reshapeSource;
    LogicalTensorPtr reshapeOutput;
};

struct BeCoveredPara {
    LogicalTensorPtr overlap;
    LogicalTensorPtr input;
    LogicalTensorPtr reshapeOutput;
    LogicalTensorPtr reshapeSource;
    std::vector<int32_t> newOffset;
};

struct PerfectlyMatchWithAllPara {
    LogicalTensorPtr input;
    LogicalTensorPtr output;
    LogicalTensorPtr overlap;
    LogicalTensorPtr reshapeOutput;
    LogicalTensorPtr newReshapeSource;
};

struct AssemblePara {
    LogicalTensorPtr input;
    LogicalTensorPtr output;
    LogicalTensorPtr reshapeSource;
    LogicalTensorPtr newInput;
    LogicalTensorPtr newReshapeOutput;
    LogicalTensorPtr inputView;
    LogicalTensorPtr overlap;
    std::vector<int32_t> newReshapeOutputTileOffset;
};

struct OpPara {
    LogicalTensorPtr oldInput;
    LogicalTensorPtr oldOutput;
    LogicalTensorPtr newInput;
    LogicalTensorPtr newOutput;
};

struct CalcOverlapPara {
    std::vector<int32_t> alignedShape;
    LogicalTensorPtr reshapeSource;
    std::vector<int32_t> newInputViewTileOffset;
    std::vector<int32_t> newInputViewTileShape;
    LogicalTensors overlaps;
    LogicalTensors newOverlaps;
    LogicalTensorPtr input;
    LogicalTensorPtr inputView;
    LogicalTensorPtr output;
};

class SplitReshape : public Pass, public DeadOperationEliminator {
public:
    SplitReshape() : Pass("SplitReshape") {}
    ~SplitReshape() override = default;
private:
    Status RunOnFunction(Function &function) override;
    Status Init();
    Status CollectCopyOut(Function &function);
    Status CheckCopyIn(Function &function);
    Status AddOperation(Function &function);
    Status EraseReshape(Function &function);
    Status SetMemoryType(Function &function);

    Status AddReshapeRemoveView(Operation &op, const OpPara &para);
    Status AddReshape(Operation &op, const OpPara &para);
    Status ObtainCopyOutTile(Function &function, const copyOutTilePara &copyOutTile, LogicalTensors &overlaps, LogicalTensors &newOverlaps);
    Status ConstructShapeOffset(const ReshapeTilePara &shapePara, size_t &i, size_t j, std::vector<int32_t> &newOffset, std::vector<int32_t> &newShape);

    Status CheckOp(Function &function, Operation &op);
    Status UpdateForPerfectlyMatchWithUB(Operation &op, const PerfectlyMatchPara &para);
    Status UpdateForPerfectlyMatchWithDDR(Operation &op, const PerfectlyMatchPara &para);
    Status UpdateForPerfectlyMatchOtherCase(Function &function, Operation &op, const PerfectlyMatchPara &para);
    Status UpdateForPerfectlyMatch(Function &function, Operation &op, const CalcOverlapPara &para);
    Status UpdateForBeCoveredUBDDR(Operation &op, const BeCoveredPara &para);
    Status UpdateForBeCoveredOtherCase(Function &function, Operation &op, const BeCoveredPara &para);
    Status UpdateForBeCovered(Function &function, Operation &op, const CalcOverlapPara &para);
    Status UpdateForAssembleAfterReshapeWithUB(Operation &op, const AssemblePara &para);
    Status UpdateForAssembleAfterReshapeWithDDR(Operation &op, const AssemblePara &para);
    Status UpdateForAssembleAfterReshapeOtherCase(Function &function, Operation &op, const AssemblePara &para);
    Status UpdateForAssembleAfterReshape(Function &function, Operation &op, const CalcOverlapPara &para);
    Status UpdateForPerfectlyMatchWithAllWithUB(Operation &op, const PerfectlyMatchWithAllPara &para);
    Status UpdateForPerfectlyMatchWithAllOtherCase(Operation &op, const PerfectlyMatchWithAllPara &para);
    Status UpdateForPerfectlyMatchWithAll(Function &function, Operation &op, const CalcOverlapPara &para);

    bool CheckSplit(const LogicalTensorPtr &reshapeSource);
    std::shared_ptr<ReshapeOp> ReshapeOperationExist(const std::shared_ptr<ReshapeOp> &isAddReshapeop);
    unsigned long ComputeReshapeHash(const LogicalTensorPtr &input, const LogicalTensorPtr &output) const;
    unsigned long ComputeReshapeHashOrderless(const LogicalTensorPtr &input, const LogicalTensorPtr &output) const;
    
    Status ShapeAlign(std::vector<int32_t> shape1, std::vector<int32_t> shape2, std::vector<int32_t> &alignedShape);
    Status RawToAlign(const ReshapeTilePara &shapePara, std::vector<int32_t> &newOffset, std::vector<int32_t> &newShape);
    Status AlignToRaw(const ReshapeTilePara &shapePara, std::vector<int32_t> &newOffset, std::vector<int32_t> &newShape);

    std::unordered_map<int, std::set<LogicalTensorPtr, TensorPtrComparator>> copyOutSources;
    std::unordered_map<InputMaigc, std::unordered_map<OutputMaigc, std::vector<int>>> mapOffset;
    std::unordered_map<int, LogicalTensorPtr> reshapeSources;
    std::vector<AssembleOp> assembles;
    std::unordered_map<unsigned long, std::shared_ptr<ReshapeOp>> reshapes;
    std::unordered_set<Operation *> redundentViewops;
    std::unordered_map<OverlaprawMagic, std::shared_ptr<RawTensor>> reshapeRawOutputs;
    std::unordered_map<OverlaprawMagic, std::shared_ptr<RawTensor>> reshapeRawInputs;
};

} // namespace npu::tile_fwk
#endif // PASS_SPLIT_RESHAPE_H_