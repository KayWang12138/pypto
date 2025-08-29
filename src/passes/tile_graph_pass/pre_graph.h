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
 * \file pre_graph.h
 * \brief
 */

#ifndef PRE_GRAPH_PASS_H
#define PRE_GRAPH_PASS_H
#include <vector>

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

namespace npu::tile_fwk {

struct SubgraphColorInfo {
    std::vector<bool> visited;
    std::vector<int> newColor;
};

class PreGraphProcess : public Pass {
public:
    PreGraphProcess() : Pass("PreGraphProcess") {}
    ~PreGraphProcess() override = default;

private:
    Status PreCheck(Function &function) override;
    Status PostCheck(Function &function) override;
    Status RunOnFunction(Function &function) override;
    Status PreColorSort(Function &function);
    void DeleteRedundantAssemble(Function &function) const;
    void ProcessSpecialMTEOperation(Operation &op) const;
    void ProcessMoveInOperation(Operation &op) const;
    void InsertTemporaryCopyIn(Function &function, Operation &op) const;
    void ResetMemoryMap(Function &function) const;
    void ProcessInplaceOp(Function &function) const;
    void UpdateCopyOpIsCube(Operation &op) const;
    void InitializeTensorMemorymap(Operation &op) const;
    void ProcessSameInOutOp(Function &function) const;
    void SetTensorBoundary(Function &function) const;
    void HandleForAssembleToOutcast(Function &function, std::unordered_set<Operation *> &concurrentAssembles, 
        std::set<Operation *, LogicalTensor::CompareOp> &producersBackup) const;
    void HandleForAssembleFromInOut(Function &function, std::unordered_set<Operation *> &concurrentAssembles, 
        std::set<Operation *, LogicalTensor::CompareOp> &producersBackup) const;
    void HandleForReshapeToOutcast(Function &function) const;
};
} // namespace npu::tile_fwk
#endif // PRE_GRAPH_PASS_H