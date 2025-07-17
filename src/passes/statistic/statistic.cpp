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
 * \file statistic.cpp
 * \brief
 */

#include "statistic.h"

#include <sstream>
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <unordered_set>
#include <vector>
#include <unordered_map>
#include <queue>
#include <climits>

#include "interface/utils/log.h"
#include "passes/pass_manager.h"
#include "interface/configs/config_manager.h"
#include "interface/function/function.h"
#include "interface/tensor/logical_tensor.h"
#include "interface/tensor/raw_tensor.h"

namespace npu::tile_fwk {
void ReportTitle(std::string title)
{
    int fillWidth = TOTAL_WIDTH - title.length();
    int leftWidth = fillWidth / 2;
    int rightWidth = fillWidth - leftWidth;
    std::cout << std::string(leftWidth, '=') << title << std::string(rightWidth, '=') << std::endl;
}

void PrintName(std::string name, int len)
{
    if (len == 0) {
        std::cout << std::left << std::setw(NAME_WIDTH - 1) << std::setfill('.') << name << ':';
        return;
    }
    std::string output = std::string(INDENT_WIDTH * len, ' ') + "|--" + name;
    std::cout << std::left << std::setw(NAME_WIDTH - 1) << std::setfill('.') << output << ':';
}

void ReportVal(std::string name, int val, int level)
{
    PrintName(name, level);
    std::cout << std::right << std::setw(VAL_WIDTH) << std::setfill(' ') << std::dec << val << std::endl;
}

void ReportSeq(std::string name, int level)
{
    PrintName(name, level);
    std::cout << std::right << std::setw(VAL_WIDTH) << std::setfill(' ') << std::dec << std::endl;
}

void ReportValDouble(std::string name, double val, int level)
{
    PrintName(name, level);
    std::cout << std::right << std::setw(VAL_WIDTH) << std::setfill(' ') << std::setiosflags(std::ios::fixed) << std::setprecision(FLOAT_PREC) << val
         << std::endl;
}

void ProcessCurrentLayer(int layerSize,
                         std::queue<int>& q, 
                         const std::unordered_map<int, std::vector<int>>& outputMap,
                         std::unordered_map<int, int>& inDegree) {
    //std::cout << "============= this layer has ============" << std::endl;
    //std::cout << "layerSize = " << layerSize << std::endl;
    for (int i = 0; i < layerSize; i++) {
        int u = q.front();
        //std::cout << "u magic = " << u << std::endl;
        q.pop();
        if (outputMap.count(u) > 0) {
            for (int v : outputMap.at(u)) {
                if (--inDegree[v] == 0) {
                    q.push(v);
                }
            }
        }
    }
    //std::cout << "============= this layer END ============" << std::endl;
}

int dfsOp(int node, const std::unordered_map<int, std::vector<int>>& outputMap,
        std::unordered_map<int, int>& memo, std::unordered_map<int, int>& nextNodes) {
    // check whether the node has been visited
    if (memo.find(node) != memo.end()) {
        return memo[node];
    }
    
    int maxDepth = 0;
    int nextNode = -1;
    
    // check whether the node has consumer
    auto it = outputMap.find(node);
    if (it != outputMap.end()) {
        for (int consumer : it->second) {
            int depth = dfsOp(consumer, outputMap, memo, nextNodes);
            if (depth > maxDepth) {
                maxDepth = depth;
                nextNode = consumer;
            }
        }
    }
    
    nextNodes[node] = nextNode;
    memo[node] = maxDepth + 1;
    return memo[node];
}

LongestPathResult findLongestPath(const std::unordered_map<int, std::vector<int>>& outputMap, const std::unordered_set<int>& indegree0) {
    std::unordered_map<int, int> memo;
    std::unordered_map<int, int> nextNodes;
    LongestPathResult result;
    int maxLength = 1;
    int startNode = -1;

    // traverse incast node magic
    for (int node : indegree0) {
        // ensure every node can be processed (including outcast node)
        int currentLength = dfsOp(node, outputMap, memo, nextNodes);
        if (currentLength > maxLength) {
            maxLength = currentLength;
            startNode = node;
        }
    }

    result.maxLength = maxLength;
    if (startNode != -1) {
        int current = startNode;
        // record every node in longest path
        while (current != -1) {
            result.nodePath.push_back(current);
            auto it = nextNodes.find(current);
            current = (it != nextNodes.end()) ? it->second : -1;
        }
    }
    return result;
}

ConcurrencyStats CalculateOpConcurrency(
    const std::unordered_map<int, std::vector<int>>& inputMap,
    const std::unordered_map<int, std::vector<int>>& outputMap)
{
    ConcurrencyStats stats;
    std::unordered_set<int> allNodes;
    for (const auto& entry : inputMap) {
        allNodes.insert(entry.first);
    }
    for (const auto& entry : outputMap) {
        allNodes.insert(entry.first);
    }
    std::unordered_map<int, int> inDegree;
    for (int node : allNodes) {
        inDegree[node] = inputMap.count(node) > 0 ? inputMap.at(node).size() : 0;
    }
    std::queue<int> q;
    for (const auto& entry : inDegree) {
        if (entry.second == 0) {
            q.push(entry.first);
        }
    }
    int maxConc = 1;
    int totalConc = 0;
    int minConc = INT_MAX;
    int layers = 0;
    std::vector<std::vector<int>> maxLayers;
    while (!q.empty()) {
        int layerSize = q.size();
        std::vector<int> currentLayer;
        std::queue<int> tempQ = q;
        while (!tempQ.empty()) {
            currentLayer.push_back(tempQ.front());
            tempQ.pop();
        }
        int effectiveConc = currentLayer.size();
        if (effectiveConc > maxConc) {
            maxConc = effectiveConc;
            maxLayers.clear();
            maxLayers.push_back(currentLayer);
        } else if (effectiveConc == maxConc) {
            maxLayers.push_back(currentLayer);
        }
        minConc = std::min(minConc, effectiveConc);
        totalConc += effectiveConc;
        layers++;
        ProcessCurrentLayer(layerSize, q, outputMap, inDegree);
    }
    stats.maxConcurrency = maxConc;
    stats.minConcurrency = (minConc == INT_MAX) ? 0 : minConc;
    stats.avgConcurrency = (layers == 0) ? 0 : static_cast<double>(totalConc) / layers;
    stats.maxLayersNodes = maxLayers;
    return stats;
}

std::pair<std::unordered_map<int, std::vector<int>>, std::unordered_map<int, std::vector<int>>> GetOpConnectionMap(const std::vector<Operation *> opList) {
    std::vector<Operation *> opListNew;
    std::unordered_map<int, int> idx2Magic;
    std::unordered_map<int, int> magic2Idx;
    int count = 0;
    for (auto op: opList){
        opListNew.push_back(op);
        int magic = op->GetOpMagic();
        magic2Idx[magic] = count;
        idx2Magic[count] = magic;
        count += 1;
    }
    std::unordered_map<int, std::vector<int>> inMap;
    std::unordered_map<int, std::vector<int>> outMap;
    for (size_t i = 0; i < opListNew.size(); i++) {
        for (auto& input : opListNew[i]->GetIOperands()) {
            for (auto& parentOpPtr : input->GetProducers()) {
                if (magic2Idx.find(parentOpPtr->GetOpMagic()) != magic2Idx.end()) {
                    inMap[idx2Magic[i]].push_back(parentOpPtr->GetOpMagic());
                    outMap[parentOpPtr->GetOpMagic()].push_back(idx2Magic[i]);
                }
            }
        }
    }
    return {inMap, outMap};
}

void cleanMap(std::unordered_map<int, std::vector<int>>& myMap) {
    for (auto it = myMap.begin(); it != myMap.end();) {
        int key = it->first;
        std::vector<int>& values = it->second;
        std::unordered_set<int> uniqueValues(values.begin(), values.end());
 
        // reuse values
        values.clear();
        for (int value : uniqueValues) {
            if (value != key) {
                values.push_back(value);
            }
        }
 
        if (values.empty()) {
            it = myMap.erase(it);
        } else {
            ++it;
        }
    }
}

void ProgramStatistic::ReportExecuteGraphTopElements() const {
    ReportTitle("ExecuteGraph Top Elements");
    ReportVal("Max ExecuteGraph Tensor Fanin Count", maxRootTensorFanin, 0);
    ReportVal("Max ExecuteGraph Tensor Fanin Magic", maxRootTensorFaninMagic, 0);
    ReportVal("Max ExecuteGraph Tensor Fanout Count", maxRootTensorFanout, 0);
    ReportVal("Max ExecuteGraph Tensor Fanout Magic", maxRootTensorFanoutMagic, 0);

    ReportVal("Max ExecuteGraph Operation Input Count", maxRootOpInput, 0);
    ReportVal("Max ExecuteGraph Operation Input Magic", maxRootOpInputMagic, 0);
    ReportVal("Max ExecuteGraph Operation Output Count", maxRootOpOutput, 0);
    ReportVal("Max ExecuteGraph Operation Output Magic", maxRootOpOutputMagic, 0);

    ReportVal("ExecuteGraph's Tensor Fanin Top Elements Number", TOP_NUMBER, 0);
    for (const auto &[count, magic] : rootFaninMaps) {
        ReportSeq("=================================================", 0);
        ReportVal("Top ExecuteGraph Tensor Fanin Count", count, 1);
        ReportVal("Top ExecuteGraph Tensor Fanin Magic", magic, 1);
    }
    ReportVal("ExecuteGraph's Tensor Fanout Top Elements Number", TOP_NUMBER, 0);
    for (const auto &[count, magic] : rootFanoutMaps) {
        ReportSeq("=================================================", 0);
        ReportVal("Top ExecuteGraph Tensor Fanout Count", count, 1);
        ReportVal("Top ExecuteGraph Tensor Fanout Magic", magic, 1);
    }
    ReportVal("ExecuteGraph's Operation Input Top Elements Number", TOP_NUMBER, 0);
    for (const auto &[count, magic] : rootOpInputMaps) {
        ReportSeq("=================================================", 0);
        ReportVal("Top ExecuteGraph Operation Input Count", count, 1);
        ReportVal("Top ExecuteGraph Operation Input Magic", magic, 1);
    }
    ReportVal("ExecuteGraph's Operation Output Top Elements Number", TOP_NUMBER, 0);
    for (const auto &[count, magic] : rootOpOutputMaps) {
        ReportSeq("=================================================", 0);
        ReportVal("Top ExecuteGraph Operation Output Count", count, 1);
        ReportVal("Top ExecuteGraph Operation Output Magic", magic, 1);
    }
}

void ProgramStatistic::ReportKernelGraphTopElements() const {
    ReportTitle("KernelGraph Top Elements");
    ReportVal("Total Different Subgraph", totalUniqueSubgraph, 0);
    ReportVal("Total AIC Subgraph",  totalCalledAICSubgraph, 0);
    ReportVal("Total AIC Different Subgraph", totalUniqueAICSubgraph, 0);
    ReportVal("Total AIV Subgraph",  totalCalledAIVSubgraph, 0);
    ReportVal("Total AIV Different Subgraph", totalUniqueAIVSubgraph, 0);

    ReportVal("Max AIC Operation Count", maxAICOperationCount.Value(), 0);
    ReportVal("Max AIC Operation SubgraphID", maxAICOperationCount.SubgraphID(), 0);
    ReportVal("Min AIC Operation Count", minAICOperationCount.Value(), 0);
    ReportVal("Min AIC Operation SubgraphID", minAICOperationCount.SubgraphID(), 0);
    ReportVal("Max AIV Operation Count", maxAIVOperationCount.Value(), 0);
    ReportVal("Max AIV Operation SubgraphID", maxAIVOperationCount.SubgraphID(), 0);
    ReportVal("Min AIV Operation Count", minAIVOperationCount.Value(), 0);
    ReportVal("Min AIV Operation SubgraphID", minAIVOperationCount.SubgraphID(), 0);

    ReportVal("Max KernelGraph Tensor Fanin Count", maxTensorFanin.Value(), 0);
    ReportVal("Max KernelGraph Tensor Fanin SubgraphID", maxTensorFanin.SubgraphID(), 0);
    ReportVal("Max KernelGraph Tensor Fanin Magic", maxTensorFaninMagic, 0);
    ReportVal("Max KernelGraph Tensor Fanout Count", maxTensorFanout.Value(), 0);
    ReportVal("Max KernelGraph Tensor Fanout SubgraphID", maxTensorFanout.SubgraphID(), 0);
    ReportVal("Max KernelGraph Tensor Fanout Magic", maxTensorFanoutMagic, 0);

    ReportVal("KernelGraph's Tensor Fanin Top Elements Number", TOP_NUMBER, 0);
    for (const auto &[count, magic] : funcFaninMaps) {
        ReportSeq("=================================================", 0);
        ReportVal("Top KernelGraph Tensor Fanin Count", count, 1);
        int subGraphID = leafMagic2SubgraphID.at(magic);
        ReportVal("Top KernelGraph Tensor Fanin SubgraphID", subGraphID, 1);
        ReportVal("Top KernelGraph Tensor Fanin Magic", magic, 1);
    }
    ReportVal("KernelGraph's Tensor Fanout Top Elements Number", TOP_NUMBER, 0);
    for (const auto &[count, magic] : funcFanoutMaps) {
        ReportSeq("=================================================", 0);
        ReportVal("Top KernelGraph Tensor Fanout Count", count, 1);
        int subGraphID = leafMagic2SubgraphID.at(magic);
        ReportVal("Top KernelGraph Tensor Fanout SubgraphID", subGraphID, 1);
        ReportVal("Top KernelGraph Tensor Fanout Magic", magic, 1);
    }
}

void CheckAndUpdateForMax(int &dest, int uninit, int curr, std::function<void()> callback = nullptr) {
    if (dest == uninit || dest < curr) {
        dest = curr;
        if (callback != nullptr) {
            callback();
        }
    }
}

void ProgramStatistic::GetTopElement(int getSize, int getMagic,
                                     std::priority_queue<std::pair<int, int>, std::vector<std::pair<int, int>>, std::greater<>>& funcQueue,
                                     std::set<std::pair<int, int>> &existElements) {
    if (existElements.find({getSize, getMagic}) != existElements.end()) {
        return;
    }
    if (funcQueue.size() < TOP_NUMBER || getSize >= funcQueue.top().first) {
        funcQueue.push({getSize, getMagic});
        existElements.insert({getSize, getMagic});
        if (funcQueue.size() > TOP_NUMBER) {
            auto t = funcQueue.top();
            existElements.erase(t);
            funcQueue.pop();
        }
    }
}

void ProgramStatistic::SortTopElement(std::vector<std::pair<int, int>> &topElements,
                                      std::priority_queue<std::pair<int, int>, std::vector<std::pair<int, int>>, std::greater<>>& funcQueue) {
    while (!funcQueue.empty()) {
        topElements.push_back(funcQueue.top());
        funcQueue.pop();
    }
    sort(topElements.begin(), topElements.end(),
        [](const std::pair<int, int>& a, const std::pair<int, int>& b) {
        if (a.first != b.first) {
            return a.first > b.first;
        }
        return a.second < b.second;
    });
}
} // namespace npu::tile_fwk