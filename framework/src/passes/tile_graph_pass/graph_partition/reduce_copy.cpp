/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You can not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file reduce_copy.cpp
 * \brief Adapter layer for mix subgraph merge algorithm integration with pypto.
 */

#include "passes/tile_graph_pass/graph_partition/reduce_copy.h"
#include "passes/tile_graph_pass/graph_partition/mixsubgraph_merger.h"
#include "interface/function/function.h"
#include "interface/utils/log.h"
#include "interface/tensor/logical_tensor.h"
#include "passes/pass_log/pass_log.h"
#include "interface/operation/opcode.h"
#include <set>
#include <unordered_map>
#include <map>

#define MODULE_NAME "ReduceCopy"

namespace npu::tile_fwk {

namespace {

constexpr int DEFAULT_MAX_LATENCY = 10000;
constexpr double DEFAULT_AIV_RATIO_MIN = 0.5;
constexpr double DEFAULT_AIV_RATIO_MAX = 2.0;

}

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
    }
    
    std::vector<bool> isCubeSubgraph(numSubgraph, false);
    for (int i = 0; i < numOp; ++i) {
        int sgId = opSubgraph[i];
        if (ops[i].GetCoreType() == CoreType::AIC) {
            isCubeSubgraph[sgId] = true;
        }
    }
    
    std::vector<std::pair<std::vector<int>, int>> mergeGroup;
    std::map<std::pair<int, int>, int> boundaryTensorSize;
    
    auto tensors = function.GetTensors();
    for (auto& tensor : tensors) {
        if (!tensor->isSubGraphBoundary) {
            continue;
        }
        
        const auto& producers = tensor->GetProducers();
        const auto& consumers = tensor->GetConsumers();
        
        if (producers.empty() || consumers.empty()) {
            continue;
        }
        
        int producerSg = -1;
        for (auto* producer : producers) {
            auto it = magicToIdx.find(producer->GetOpMagic());
            if (it != magicToIdx.end()) {
                producerSg = opSubgraph[it->second];
                break;
            }
        }
        
        if (producerSg < 0) {
            continue;
        }
        
        for (auto* consumer : consumers) {
            auto it = magicToIdx.find(consumer->GetOpMagic());
            if (it == magicToIdx.end()) {
                continue;
            }
            int consumerSg = opSubgraph[it->second];
            
            if (producerSg != consumerSg) {
                size_t tensorSize = tensor->MemorySize();
                int priority = static_cast<int>(tensorSize);
                
                int minSg = std::min(producerSg, consumerSg);
                int maxSg = std::max(producerSg, consumerSg);
                auto key = std::make_pair(minSg, maxSg);
                
                if (boundaryTensorSize.find(key) == boundaryTensorSize.end() ||
                    boundaryTensorSize[key] < priority) {
                    boundaryTensorSize[key] = priority;
                }
            }
        }
    }
    
    for (const auto& pair : boundaryTensorSize) {
        std::vector<int> sgPair = {pair.first.first, pair.first.second};
        mergeGroup.push_back({sgPair, pair.second});
        APASS_LOG_INFO_F(Elements::Operation, 
            "Merge group: subgraph %d <-> %d, priority=%d", 
            pair.first.first, pair.first.second, pair.second);
    }
    
    MixSubgraphMergerInput input;
    input.numOp = numOp;
    input.numSubgraph = numSubgraph;
    input.maxLatency = DEFAULT_MAX_LATENCY;
    input.aivRatio = {DEFAULT_AIV_RATIO_MIN, DEFAULT_AIV_RATIO_MAX};
    input.opSubgraph = opSubgraph;
    input.opLatency = opLatency;
    input.isCubeSubgraph = isCubeSubgraph;
    input.opOutGraph = opOutGraph;
    input.mergeGroup = mergeGroup;
    
    MixSubgraphMerger merger;
    MixSubgraphMergerOutput output = merger.Merge(input);
    
    for (int i = 0; i < numOp; ++i) {
        ops[i].UpdateSubgraphID(output.opSubgraphUpdated[i]);
    }
    
    function.SetTotalSubGraphCount(output.numSubgraphUpdated);
    
    APASS_LOG_INFO_F(Elements::Operation, 
        "Subgraph count: %d -> %d", numSubgraph, output.numSubgraphUpdated);
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