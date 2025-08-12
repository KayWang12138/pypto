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
 * \file l1_copy_reuse.h
 * \brief
 */

#ifndef PASS_L1_COPY_REUSE_H_
#define PASS_L1_COPY_REUSE_H_

#include "interface/function/function.h"
#include "interface/tensor/logical_tensor.h"
#include "passes/pass_utils/reschedule_utils.h"
#include "passes/pass_interface/pass.h"
#include "passes/tile_graph_pass/dead_operation_eliminate.h"
#include "passes/pass_utils/pass_utils.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"
#include "interface/utils/log.h"

namespace npu::tile_fwk {
class L1CopyInReuseRunner {
  public:
    explicit L1CopyInReuseRunner(const std::vector<std::vector<int>> &inGraph1) : inGraph(inGraph1) {}
    ~L1CopyInReuseRunner() {}
    void Run(Function &func, int color, std::vector<std::vector<int>> &colorNode);
  private:
    void GetOpHash(std::vector<uint64_t> &hashList, const std::string op, int idx);
    void GetColorHash(const OperationsViewer &opOriList, std::vector<uint64_t> &hashColor);
    int GetMaxInColor(const std::vector<int> &nodes, const OperationsViewer &opOriList, int curColor);
    void MergeDupL1CopyIn(Function &func, std::vector<std::vector<int>> &colorNode,
                          int color);
    std::vector<int> GetOpInputFeature(const OperationsViewer &opOriList,
                                      const int opIdx, const int ioperandIdx);
    void RemoveUselessViews(Function &func) const;
    void GetDuplicateOps(std::vector<Operation *> &opOriList,
                         const std::vector<int> &opIdx);
    void TackleOp(int i, Operation *op, std::vector<std::vector<int>> &replacedInputs, 
                        std::vector<std::vector<int>> &replacedOutputs);
    void Phase1(Function &func, int color, std::vector<std::vector<int>> &colorNode, 
                        std::vector<int> &colorCopyIn, std::vector<uint64_t> &hashColor);
    std::vector<int> SetNumLR();
    std::vector<int> SetNumDB();
    const std::vector<std::vector<int>> &inGraph;
    std::unordered_map<int, int> replacedCopyMap_;
    std::unordered_map<int, int> tensormagic2Op_;
    std::unordered_map<uint64_t, std::vector<int>> hashMap;
    std::unordered_map<uint64_t, int> hashOrder;
    int numLR;
    std::map<int, int> numLRMap;
    int numDB_;
    std::map<int, int> numDBMap;
    int copyInThreshold;
    bool isLoadBalance;
};

class L1CopyInReuseMerge : public Pass, public DeadOperationEliminator {
public:
    L1CopyInReuseMerge() : Pass("L1CopyInReuseMerge") {}
    ~L1CopyInReuseMerge() override = default;

private:
    Status L1CopyInReuse(Function &func) const;
    Status RunOnFunction(Function &function) override {
        ASLOGI("===> Start L1CopyInReusePass.");
        if (L1CopyInReuse(function) == FAILED) {
          return FAILED;
        }
        EliminateDeadOperationBackward(function);
        ALOG_INFO_F("===> Finish L1CopyInReuseMerge.");
        return SUCCESS;
    }
};
} // namespace npu::tile_fwk
#endif // PASS_L1_COPY_REUSE_H_