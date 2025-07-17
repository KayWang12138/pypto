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
 * \file split_reshape_pvc2.h
 * \brief
 */

#ifndef PASS_SPLIT_RESHAPE_PVC2_H_
#define PASS_SPLIT_RESHAPE_PVC2_H_

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

class ReshapeOp{
    public:
        ReshapeOp(std::shared_ptr<LogicalTensor> aInput, std::shared_ptr<LogicalTensor> aOutput)
            : input(aInput), output(aOutput) {}

        std::shared_ptr<LogicalTensor> input;
        std::shared_ptr<LogicalTensor> output;
};

struct CalcOverlapPara {
    std::vector<int> &alignedShape;
    std::shared_ptr<LogicalTensor> &reshapeSource;
    std::pair<std::vector<int>, std::vector<int>> &newinputviewTileinfo;
    std::vector<std::shared_ptr<LogicalTensor>> &overlaps;
    std::vector<std::shared_ptr<LogicalTensor>> &newOverlaps;
    std::shared_ptr<LogicalTensor> &input;
    std::shared_ptr<LogicalTensor> &inputView;
    std::shared_ptr<LogicalTensor> &output;
};

class SplitReshapeOpPVC2 : public Pass, public DeadOperationEliminator {
public:
    SplitReshapeOpPVC2() : Pass("SplitReshapeOpPVC2") {}
    ~SplitReshapeOpPVC2() override = default;
    Status RunOnFunction(Function &function) override;

private:
    void CollectCopyOut(Function &function);
    void CheckCopyIn(Function &function);
    void EraseReshape(Function &function);
    void UpdateForPerfectlyMatchWithAll(Function &function, Operation &op, const CalcOverlapPara &para);
    void UpdateForAssembleAfterReshape(Function &function, Operation &op, const CalcOverlapPara &para);

    std::shared_ptr<ReshapeOp> ReshapeOperationExist(const std::shared_ptr<ReshapeOp> &isAddReshapeop);
    unsigned long ComputeReshapeHash(
        const std::shared_ptr<LogicalTensor> &input, const std::shared_ptr<LogicalTensor> &output) const;
    unsigned long ComputeReshapeHashOrderless(
        const std::shared_ptr<LogicalTensor> &input, const std::shared_ptr<LogicalTensor> &output) const;
    std::vector<int> ShapeAlign(std::vector<int> shape1, std::vector<int> shape2);
    std::pair<std::vector<int>, std::vector<int>> ReshapeTile(const std::vector<int> &shape,
        const std::vector<int> &alignedShape, const std::vector<int> &tileOffset, const std::vector<int> &tileShape);
    std::pair<std::vector<int>, std::vector<int>> ReshapeTile2(const std::vector<int> &rawshape,
        const std::vector<int> &newRawshape, const std::vector<int> &tileOffset, const std::vector<int> &tileShape);

    std::unordered_map<int, std::set<std::shared_ptr<LogicalTensor>, TensorPtrComparator>> copyOutSources;
    std::unordered_map<InputMaigc, std::unordered_map<OutputMaigc, std::vector<int>>> mappingOffset;
    std::unordered_map<int, std::shared_ptr<LogicalTensor>> reshapeSources;
    std::vector<AssembleOp> assembles;
    std::unordered_map<unsigned long, std::shared_ptr<ReshapeOp>> reshapes;
    std::unordered_set<Operation *> redundentViewops;
    std::unordered_map<OverlaprawMagic, std::shared_ptr<RawTensor>> reshapeRawOutputs;
    std::unordered_map<OverlaprawMagic, std::shared_ptr<RawTensor>> reshapeRawInputs;
};

} // namespace npu::tile_fwk
#endif // PASS_SPLIT_RESHAPE_PVC2_H_