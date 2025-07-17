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
 * \file partitioner.h
 * \brief
 */

#ifndef PASS_PARTITIONER_H
#define PASS_PARTITIONER_H

#include "interface/function/function.h"
#include "interface/tensor/logical_tensor.h"
#include "tilefwk/tilefwk.h"
#include "tilefwk.h"
#include "interface/program/program.h"
#include "passes/pass_interface/pass.h"

namespace npu::tile_fwk {

class GraphParitioner {
  public:
    explicit GraphParitioner() {}
    ~GraphParitioner() = default;
    void Run(Function *funcPtr);
  private:
    int color_{0};
    std::map<int, size_t> opMagic2Idx_;
    std::vector<Operation *> opList_;// all the operations
    std::vector<std::vector<int>> opInGraph_, opOutGraph_;// tile op's parent/child tile op
    std::vector<std::vector<int>> superNodes_;// each super node contains which tile ops
    std::vector<int> superNodesColor_;// color of each super node
    std::vector<int> op2superNodeIdx_;// tile op's super node index
    std::vector<std::vector<int>> superInGraph_, superOutGraph_; // super node's parent/child super node
    std::vector<OpCoreType> opCoreType_;
    std::vector<OpCoreType> nodeCoreType_;
    std::vector<std::string> superNodeHash_;
    std::vector<std::vector<int>> colorInGraph_;
    std::vector<std::vector<int>> colorOutGraph_;
    int FindParent(std::vector<int>& parent, int i);
    void MergeSrcToDstIsland(std::vector<int>& parent, int src, int dst);
    bool CoreTypeMergeable(const std::set<OpCoreType>& coreTypes);
    void UpdateInOutGraph(Function *funcPtr);
    void UpdateSuperNode(std::vector<int>& parent);
    void BuildSuperNodes();
    std::vector<int> GetSameLevelOpIdx(int opIdx, Opcode opLabel);
    void ProcessSuperNode(const int superNodeIdx, std::vector<int>& superDegree, std::queue<int>& superQueue, int color);
    void ColorSuperNodes();
    bool IsAllOutSame(const std::vector<int>& outSuperNodes);
    void MergeColor();
};

class PartitionVCPass : public Pass {
public:
    PartitionVCPass() : Pass("PartitionVCPass") {}
    ~PartitionVCPass() override = default;
    Status PreCheck(Function &function) override;
    Status PostCheck(Function &function) override;
    Status RunOnFunction(Function &function) override;
};

} // namespace npu::tile_fwk
#endif  // PASS_PARTITIONER_H