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
 * \file n_buffer_merge.h
 * \brief
 */

#ifndef PASS_N_BUFFER_MERGE_H_
#define PASS_N_BUFFER_MERGE_H_

#include "interface/function/function.h"
#include "interface/tensor/logical_tensor.h"
#include "passes/pass_interface/pass.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"
namespace npu::tile_fwk {
class NBufferMerge : public Pass {
public:
    NBufferMerge() : Pass("NBufferMerge") {}
    ~NBufferMerge() override = default;
private:
    Status RunOnFunction(Function &function) override;
    Status NBufferMergeProcess(Function &func);
    Status Init(Function &func);
    void InitParam(OperationsViewer &opOriList);
    void GetOpHash(std::vector<uint64_t> &hashList, const std::string op, int idx);
    void GetOpHashReverse(std::vector<uint64_t> &hashList, const std::string op, int idx);
    void GetColorHash(const OperationsViewer &opOriList, 
                      std::vector<uint64_t> &hashColor, 
                      std::map<uint64_t, std::vector<int>> &hashMap);
    Status CheckAndFixColorOrder(OperationsViewer &opOriList, 
                               int &color1, std::vector<int> &colorCycles1,
                               std::vector<std::vector<int>> &colorNode1);
    std::map<int, size_t> GetIsoColorMergeNum(const OperationsViewer &opOriList,
                                                   const std::map<uint64_t, std::vector<int>> &hashMap) const;
    std::vector<std::vector<int>> SortColorWithInput(std::vector<int> &colorValues) const;
    Status MergeProcess(const OperationsViewer &opOriList, 
                        std::map<uint64_t, std::vector<int>> &hashMap, 
                        std::map<int, size_t> &hashMergeNum, 
                        std::vector<uint64_t> &hashColor);
    Status ColorTopo(int &color1, 
                     std::vector<std::vector<int>> &inputColor, 
                     std::vector<std::vector<int>> &outputColor, 
                     OperationsViewer &opOriList);
    void MergePingPong(std::vector<std::vector<int>> &sortedColors, 
                       const OperationsViewer &opOriList, 
                       std::vector<uint64_t> &hashColor, 
                       int &numDBmerge);
    std::map<int, size_t> SetNumDB(std::map<uint64_t, std::vector<int>> &hashMap);
private:
    int color_{0};
    std::vector<std::vector<int>> inGraph_;
    std::vector<std::vector<int>> outGraph_;
    std::vector<std::vector<int>> inColor_;
    std::vector<std::vector<int>> outColor_;
    std::vector<std::vector<int>> colorNode_;
    std::vector<int> colorCycles_;
    int nBufferMergeMode;
    int sgVecParallelNum;
    int sgCubeParallelNum;
    std::map<int, int> vecNBufferMap;
    std::unordered_map<uint64_t, int> hashOrder;
    int noMerge = 0;
    int autoMerge = 1;
    int manualMerge = 2;
};
}  // namespace npu::tile_fwk
#endif  // PASS_N_BUFFER_MERGE_H_