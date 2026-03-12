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
 * \file reduce_copy.cpp
 * \brief Adapter layer for subgraph merge algorithm integration with pypto.
 */

#include "passes/tile_graph_pass/graph_partition/reduce_copy.h"
#include "passes/tile_graph_pass/graph_partition/subgraph_merger.h"
#include "interface/function/function.h"
#include "interface/utils/log.h"
#include "interface/tensor/logical_tensor.h"
#include "passes/pass_log/pass_log.h"
#include <set>
#include <unordered_map>

#define MODULE_NAME "ReduceCopy"

namespace npu::tile_fwk {

Status ReduceCopyMerge::RunOnFunction(Function &function) {
    APASS_LOG_INFO_F(Elements::Operation, "=== ReduceCopyMerge Pass Started ===");
    
    if (Platform::Instance().GetSoc().GetNPUArch() != NPUArch::DAV_3510) {
        APASS_LOG_INFO_F(Elements::Operation, "Platform not support CV mix graph, skip ReduceCopy Pass.");
        return SUCCESS;
    }
    
    auto ops = function.Operations();
    int numOp = static_cast<int>(ops.size());
    int numSubgraph = function.GetTotalSubGraphCount();
    
    if (numOp == 0) {
        APASS_LOG_INFO_F(Elements::Operation, "No operations in function, skip ReduceCopy Pass.");
        return SUCCESS;
    }
    
    APASS_LOG_INFO_F(Elements::Operation, "Input: numOp=%d, numSubgraph=%d", numOp, numSubgraph);
    
    std::unordered_map<int, int> magicToIdx;
    for (int i = 0; i < numOp; ++i) {
        magicToIdx[ops[i].GetOpMagic()] = i;
    }
    
    std::vector<int> opLatency(numOp);
    std::vector<int> opSubgraph(numOp);
    std::unordered_map<int, std::set<int>> opOutGraph;
    std::unordered_map<int, std::set<int>> opInGraph;
    
    for (int i = 0; i < numOp; ++i) {
        opLatency[i] = ops[i].GetLatency();
        opSubgraph[i] = ops[i].GetSubgraphID();
        
        if (opSubgraph[i] < 0) {
            APASS_LOG_ERROR_F(Elements::Operation, 
                "Op %d has invalid subgraphID %d", ops[i].GetOpMagic(), opSubgraph[i]);
            return FAILED;
        }
        
        for (auto* consumer : ops[i].ConsumerOps()) {
            auto it = magicToIdx.find(consumer->GetOpMagic());
            if (it != magicToIdx.end()) {
                opOutGraph[i].insert(it->second);
            }
        }
        
        for (auto* producer : ops[i].ProducerOps()) {
            auto it = magicToIdx.find(producer->GetOpMagic());
            if (it != magicToIdx.end()) {
                opInGraph[i].insert(it->second);
            }
        }
    }
    
    MergeSubgraphs(numOp, numSubgraph, opLatency, opSubgraph, opOutGraph, opInGraph);
    
    for (int i = 0; i < numOp; ++i) {
        ops[i].UpdateSubgraphID(opSubgraph[i]);
    }
    
    std::set<int> finalSubgraphs(opSubgraph.begin(), opSubgraph.end());
    function.SetTotalSubGraphCount(static_cast<int>(finalSubgraphs.size()));
    
    APASS_LOG_INFO_F(Elements::Operation, 
        "Subgraph count: %d -> %zu", numSubgraph, finalSubgraphs.size());
    APASS_LOG_INFO_F(Elements::Operation, "=== ReduceCopyMerge Pass Completed ===");
    
    return SUCCESS;
}

Status ReduceCopyMerge::PostCheck(Function &function) {
    APASS_LOG_INFO_F(Elements::Function, "PostCheck for ReduceCopy.");
    
    if (Platform::Instance().GetSoc().GetNPUArch() != NPUArch::DAV_3510) {
        APASS_LOG_INFO_F(Elements::Operation, 
            "Platform not support CV mix graph, skip PostCheck for ReduceCopy Pass.");
        return SUCCESS;
    }
    
    APASS_LOG_INFO_F(Elements::Operation, "===> Start PostCheck for ReduceCopy.");
    
    for (auto &op : function.Operations()) {
        if (op.GetSubgraphID() < 0) {
            APASS_LOG_ERROR_F(Elements::Operation, 
                "Op %d does not belong to any subgraph.", op.GetOpMagic());
            return FAILED;
        }
    }
    
    APASS_LOG_INFO_F(Elements::Operation, "===> Finish PostCheck for ReduceCopy.");
    return SUCCESS;
}

}