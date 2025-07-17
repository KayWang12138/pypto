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
 * \file statistic.h
 * \brief
 */

#ifndef TILE_FWK_PROGRAM_JUDGEMENT_H
#define TILE_FWK_PROGRAM_JUDGEMENT_H

#include <map>
#include "tilefwk/tilefwk.h"
#include "tilefwk.h"
#include "interface/program/program.h"

namespace npu::tile_fwk {
constexpr int SYN_MIN = 6U;
constexpr int STATISTIC_UNINIT = -1;
constexpr int TOTAL_WIDTH = 80;
constexpr int NAME_WIDTH = 50;
constexpr int VAL_WIDTH = 10;
constexpr int INDENT_WIDTH = 2;
constexpr int FLOAT_PREC = 2;
constexpr int ReportLevelTwo = 2;
constexpr size_t TOP_NUMBER = 5;

class CoutRedirector {
public:
    explicit CoutRedirector(const std::string& filename) : oldBuf(std::cout.rdbuf()) {
        file.open(filename);
        if (file.is_open()) {
            std::cout.rdbuf(file.rdbuf());
        }
    }

    ~CoutRedirector() {
        if (file.is_open()) {
            file.close();
        }
        std::cout.rdbuf(oldBuf);
    }

private:
    std::ofstream file;
    std::streambuf* oldBuf;
};

void ReportTitle(std::string title);

void PrintName(std::string name, int len);

void ReportVal(std::string name, int val, int level);

void ReportSeq(std::string name, int level);

void ReportValDouble(std::string name, double val, int level);

struct ConcurrencyStats {
    int maxConcurrency;
    double avgConcurrency;
    int minConcurrency;
    std::vector<std::vector<int>> maxLayersNodes;
};

struct LongestPathResult {
    int maxLength = 0;
    std::vector<int> nodePath;
};

int dfsOp(int node, const std::unordered_map<int, std::vector<int>>& outputMap,
        std::unordered_map<int, int>& memo, std::unordered_map<int, int>& nextNodes);

LongestPathResult findLongestPath(const std::unordered_map<int, std::vector<int>>& outputMap, const std::unordered_set<int>& indegree0);

ConcurrencyStats CalculateOpConcurrency(
    const std::unordered_map<int, std::vector<int>>& inputMap,
    const std::unordered_map<int, std::vector<int>>& outputMap);

void HealthCheckIsomorphismSubgraph(std::multimap<int, int> psgToESgMap, std::vector<std::vector<OperationPtr>> nLIST);

void HealthCheckTensorGraph(Function &function);

void HealthCheckTileGraph(Function &function);

std::pair<std::unordered_map<int, std::vector<int>>, std::unordered_map<int, std::vector<int>>> GetOpConnectionMap(const std::vector<Operation *> opList);

void cleanMap(std::unordered_map<int, std::vector<int>>& myMap);

void ProcessCurrentLayer(int layerSize,
                         std::queue<int>& q, 
                         const std::unordered_map<int, std::vector<int>>& outputMap,
                         std::unordered_map<int, int>& inDegree);

struct SubgraphTopology {
    std::unordered_map<int, std::vector<int>> subgraphInputs; // subgraph -> input subgraph
    std::unordered_map<int, std::vector<int>> subgraphOutputs; // subgraph -> output subgraph
    std::map<int, std::unordered_set<int>> members; // all nodes in subgraph
};
class SubgraphBuilder {
private:
    // 原始图数据
    std::unordered_map<int, std::vector<int>> inputConn; // node -> input node
    std::unordered_map<int, std::vector<int>> outputConn; // node -> output node
    std::unordered_map<int, int> subgraphMap; // node -> subgraph id
 
public:
    SubgraphBuilder(const std::unordered_map<int, std::vector<int>>& inputs,
                   const std::unordered_map<int, std::vector<int>>& outputs,
                   const std::unordered_map<int, int>& subgraphs)
        : inputConn(inputs), outputConn(outputs), subgraphMap(subgraphs) {}
 
    // build subgraph topo
    SubgraphTopology buildSubgraphTopology() {
        SubgraphTopology result;
        
        for (const auto& [node, sgId] : subgraphMap) {
            result.members[sgId].insert(node);
        }
 
        std::map<std::pair<int, int>, bool> processedEdges;
        
        for (const auto& [srcNode, destNodes] : outputConn) {
            int srcSg = subgraphMap.at(srcNode);
            
            for (int destNode : destNodes) {
                int destSg = subgraphMap.at(destNode);
                // ignore same subgraph id
                if (srcSg == destSg) {
                    continue;
                }

                std::pair<int, int> edge(srcSg, destSg);
                if (!processedEdges[edge]) {
                    processedEdges[edge] = true;
                    result.subgraphOutputs[srcSg].push_back(destSg);
                    result.subgraphInputs[destSg].push_back(srcSg);
                }
            }
        }
 
        // sort list
        auto dedupSort = [](auto& container) {
            for (auto& list : container) {
                sort(list.second.begin(), list.second.end());
                auto last = unique(list.second.begin(), list.second.end());
                list.second.erase(last, list.second.end());
            }
        };
 
        dedupSort(result.subgraphInputs);
        dedupSort(result.subgraphOutputs);
 
        return result;
    }
};

struct PathInfo {
    int length;
    std::vector<int> path; // node in path
};

struct ParallelismMetrics {
    int maxParallelism;
    int minParallelism;
    double avgParallelism;
    int longestDepth; 
    std::vector<int> longestPath;
    std::vector<std::vector<int>> levelsDetail;
};
 
class ParallelismAnalyzer {
private:
    SubgraphTopology topology;
    std::map<int, std::vector<int>> edges; 
    std::unordered_map<int, PathInfo> pathMemo; 
 
    void buildEdges() {
        edges.clear();
        for (const auto& [src, dests] : topology.subgraphOutputs) {
            edges[src] = dests;
        }
    }

    PathInfo dfsGraph(int current) {
        if (pathMemo.find(current) != pathMemo.end()) {
            return pathMemo[current];
        }

        PathInfo maxPath = {0, {}};
        for (int neighbor : edges[current]) {
            PathInfo neighborPath = dfsGraph(neighbor);
            if (neighborPath.length > maxPath.length) {
                maxPath = neighborPath;
            }
        }

        // create current path: currant node + longest path
        PathInfo currentPath;
        currentPath.length = maxPath.length + 1;
        currentPath.path.reserve(currentPath.length);
        currentPath.path.push_back(current);
        currentPath.path.insert(currentPath.path.end(), 
                                maxPath.path.begin(),
                                maxPath.path.end());

        pathMemo[current] = currentPath;
        return currentPath;
    }
 
public:
    explicit ParallelismAnalyzer(const SubgraphTopology& topo) : topology(topo) {
        buildEdges();
    }
 
    ParallelismMetrics calculateParallelism() {
        ParallelismMetrics metrics;
        pathMemo.clear();

        std::map<int, int> inDegree;
        std::map<int, int> nodeLevel;
        std::vector<std::vector<int>> levels;
 
        // init inDegree and queue
        for (const auto& sg : topology.members) {
            inDegree[sg.first] = topology.subgraphInputs[sg.first].size();
        }
        std::queue<int> q;
        for (const auto& [sg, inputs] : topology.subgraphInputs) {
            if (inputs.empty()) {
                nodeLevel[sg] = 0;
                q.push(sg);
            }
        }
 
        while (!q.empty()) {
            int current = q.front();
            q.pop();
 
            if (static_cast<int>(levels.size()) <= nodeLevel[current]) {
                levels.resize(nodeLevel[current] + 1);
            }
            levels[nodeLevel[current]].push_back(current);
 
            for (int neighbor : edges[current]) {
                int new_level = nodeLevel[current] + 1;
                if (new_level > nodeLevel[neighbor]) {
                    nodeLevel[neighbor] = new_level;
                }
 
                if (--inDegree[neighbor] == 0) {
                    q.push(neighbor);
                }
            }
        }
 
        if (levels.empty()) {
            return metrics;
        }
        
        metrics.maxParallelism = std::numeric_limits<int>::min();
        metrics.minParallelism = std::numeric_limits<int>::max();
        int total = 0;
 
        for (const auto& level : levels) {
            int size = level.size();
            metrics.levelsDetail.push_back(level);
            metrics.maxParallelism = std::max(metrics.maxParallelism, size);
            metrics.minParallelism = std::min(metrics.minParallelism, size);
            total += size;
        }
 
        metrics.avgParallelism = static_cast<double>(total) / levels.size();

        metrics.longestDepth = 0;
        for (const auto& sg : topology.members) {
            PathInfo path = dfsGraph(sg.first);
            if (path.length > metrics.longestDepth) {
                metrics.longestDepth = path.length;
                metrics.longestPath = path.path;
            }
        }
        return metrics;
    }
};

struct SubgraphStatistic {
    int tensorCount{STATISTIC_UNINIT};
    int operationCount{STATISTIC_UNINIT};

    OpCoreType coreType{OpCoreType::ANY};

    int maxTensorFanin{STATISTIC_UNINIT};
    int maxTensorFaninMagic{STATISTIC_UNINIT};
    int maxTensorFanout{STATISTIC_UNINIT};
    int maxTensorFanoutMagic{STATISTIC_UNINIT};
};

struct ProgramSubgraphStatisticValue {
    int value{STATISTIC_UNINIT};
    int subgraphID{STATISTIC_UNINIT};

    const int &Value() const { return value; }
    int SubgraphID() const { return subgraphID; }
    void CheckAndUpdate(int currValue, int currSubgraphID)
    {
        if (this->value == STATISTIC_UNINIT || this->value < currValue) {
            this->value = currValue;
            this->subgraphID = currSubgraphID;
        }
    }
};

struct ProgramStatistic {
    std::unordered_map<int, SubgraphStatistic> statistic;
    std::priority_queue<std::pair<int, int>, std::vector<std::pair<int, int>>, std::greater<>> graphProducerQueue;
    std::priority_queue<std::pair<int, int>, std::vector<std::pair<int, int>>, std::greater<>> graphConsumerQueue;
    std::priority_queue<std::pair<int, int>, std::vector<std::pair<int, int>>, std::greater<>> graphOpInputQueue;
    std::priority_queue<std::pair<int, int>, std::vector<std::pair<int, int>>, std::greater<>> graphOpOutputQueue;
    std::priority_queue<std::pair<int, int>, std::vector<std::pair<int, int>>, std::greater<>> rootProducerQueue;
    std::priority_queue<std::pair<int, int>, std::vector<std::pair<int, int>>, std::greater<>> rootConsumerQueue;
    std::priority_queue<std::pair<int, int>, std::vector<std::pair<int, int>>, std::greater<>> rootOpInputQueue;
    std::priority_queue<std::pair<int, int>, std::vector<std::pair<int, int>>, std::greater<>> rootOpOutputQueue;
    std::priority_queue<std::pair<int, int>, std::vector<std::pair<int, int>>, std::greater<>> leafProducerQueue;
    std::priority_queue<std::pair<int, int>, std::vector<std::pair<int, int>>, std::greater<>> leafConsumerQueue;

    std::set<std::pair<int, int>> leafProducerSet;
    std::set<std::pair<int, int>> leafConsumerSet;
    std::map<int, int> leafMagic2SubgraphID;

    std::vector<std::pair<int, int>> rootFaninMaps; // first: fanin, second: faninMagic
    std::vector<std::pair<int, int>> rootFanoutMaps;
    std::vector<std::pair<int, int>> rootOpInputMaps;
    std::vector<std::pair<int, int>> rootOpOutputMaps;
    std::vector<std::pair<int, int>> graphFaninMaps; // first: fanin, second: faninMagic
    std::vector<std::pair<int, int>> graphFanoutMaps;
    std::vector<std::pair<int, int>> graphOpInputMaps;
    std::vector<std::pair<int, int>> graphOpOutputMaps;
    std::vector<std::pair<int, int>> funcFaninMaps; // first: fanin, second: faninMagic
    std::vector<std::pair<int, int>> funcFanoutMaps;

    int totalUniqueSubgraph{0};
    int totalUniqueAICSubgraph{0};
    int totalUniqueAIVSubgraph{0};
    int totalUniqueOperationCount{0};

    int totalCalledSubgraph{0};
    int totalCalledAICSubgraph{0};
    int totalCalledAIVSubgraph{0};
    int totalCalledOperationCount{0};

    int maxRootTensorFanin{STATISTIC_UNINIT};
    int maxRootTensorFaninMagic{STATISTIC_UNINIT};
    int maxRootTensorFanout{STATISTIC_UNINIT};
    int maxRootTensorFanoutMagic{STATISTIC_UNINIT};

    int maxRootOpInput{STATISTIC_UNINIT};
    int maxRootOpInputMagic{STATISTIC_UNINIT};
    int maxRootOpOutput{STATISTIC_UNINIT};
    int maxRootOpOutputMagic{STATISTIC_UNINIT};
    int maxGraphTensorFanin{STATISTIC_UNINIT};
    int maxGraphTensorFaninMagic{STATISTIC_UNINIT};
    int maxGraphTensorFanout{STATISTIC_UNINIT};
    int maxGraphTensorFanoutMagic{STATISTIC_UNINIT};
 
    int maxGraphOpInput{STATISTIC_UNINIT};
    int maxGraphOpInputMagic{STATISTIC_UNINIT};
    int maxGraphOpOutput{STATISTIC_UNINIT};
    int maxGraphOpOutputMagic{STATISTIC_UNINIT};

    ProgramSubgraphStatisticValue maxOperationCount;
    ProgramSubgraphStatisticValue minOperationCount;
    ProgramSubgraphStatisticValue maxAICOperationCount;
    ProgramSubgraphStatisticValue minAICOperationCount;
    ProgramSubgraphStatisticValue maxAIVOperationCount;
    ProgramSubgraphStatisticValue minAIVOperationCount;

    ProgramSubgraphStatisticValue maxTensorFanin;
    int maxTensorFaninMagic{STATISTIC_UNINIT};
    ProgramSubgraphStatisticValue maxTensorFanout;
    int maxTensorFanoutMagic{STATISTIC_UNINIT};

    void GetTopElement(int getSize, int getMagic, std::priority_queue<std::pair<int, int>, std::vector<std::pair<int, int>>, std::greater<>>& funcQueue, std::set<std::pair<int, int>> &existElements);

    void SortTopElement(std::vector<std::pair<int, int>> &topElements, std::priority_queue<std::pair<int, int>, std::vector<std::pair<int, int>>, std::greater<>>& funcQueue);

    void ReportExecuteGraphTopElements() const;

    void ReportKernelGraphTopElements() const;

    void HealthCheckExecuteGraph(Function &func);

    const SubgraphStatistic &HealthCheckKernelGraph(Function &func, int subgraphID);
};

ProgramStatistic SubgraphEvaluation(const Program &prog, std::multimap<int, int> psgToESgMap);

}

#endif