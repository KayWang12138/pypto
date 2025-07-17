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
 * \file tile_graph_statistic.cpp
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
struct SubgraphStats {
    int maxInput;
    int minInput;
    double avgInput;
    int maxOutput;
    int minOutput;
    double avgOutput;
};

SubgraphStats CalculateStats(
    const std::unordered_map<int, std::unordered_set<int>>& subGraph2StartTensor,
    const std::unordered_map<int, std::unordered_set<int>>& subGraph2EndTensor) 
{
    SubgraphStats stats;
    std::unordered_set<int> allSubgraphs;

    for (const auto& entry : subGraph2StartTensor) {
        allSubgraphs.insert(entry.first);
    }
    for (const auto& entry : subGraph2EndTensor) {
        allSubgraphs.insert(entry.first);
    }
        
    if (allSubgraphs.empty()) {
        return stats;
    }

    std::vector<int> inputCounts, outputCounts;
    
    for (int subgraphId : allSubgraphs) {
        // calculate input counts
        int input = 0;
        auto itStart = subGraph2StartTensor.find(subgraphId);
        if (itStart != subGraph2StartTensor.end()) {
            input = itStart->second.size();
        }
        inputCounts.push_back(input);

        // calculate output counts
        int output = 0;
        auto itEnd = subGraph2EndTensor.find(subgraphId);
        if (itEnd != subGraph2EndTensor.end()) {
            output = itEnd->second.size();
        }
        outputCounts.push_back(output);
    }

    if (!inputCounts.empty()) {
        stats.maxInput = *std::max_element(inputCounts.begin(), inputCounts.end());
        stats.minInput = *std::min_element(inputCounts.begin(), inputCounts.end());
        stats.avgInput = static_cast<double>(
            std::accumulate(inputCounts.begin(), inputCounts.end(), 0)) / inputCounts.size();
    }

    if (!outputCounts.empty()) {
        stats.maxOutput = *std::max_element(outputCounts.begin(), outputCounts.end());
        stats.minOutput = *std::min_element(outputCounts.begin(), outputCounts.end());
        stats.avgOutput = static_cast<double>(
            std::accumulate(outputCounts.begin(), outputCounts.end(), 0)) / outputCounts.size();
    }

    return stats;
}

void GetSubGraphInfo(Function &function) {
    std::vector<Operation *> opList = function.Operations().DuplicatedOpList();
    std::unordered_set<int> startTensor;
    std::unordered_set<int> endTensor;
    std::unordered_map<int, std::unordered_set<int>> subGraph2StartTensor;
    std::unordered_map<int, std::unordered_set<int>> subGraph2EndTensor;
    std::unordered_map<int, std::vector<CoreType>> subGraph2OpType;

    for (auto &op : opList) {
        subGraph2OpType[op->GetSubgraphID()].push_back(op->GetCoreType());
        for (auto &iTensor : op->GetIOperands()) {
            if (iTensor->GetProducers().size() == 0) {
                subGraph2StartTensor[op->GetSubgraphID()].insert(iTensor->GetMagic());
                startTensor.insert(iTensor->GetMagic());
                break;
            } else {
                for (auto& parentOpPtr : iTensor->GetProducers()) {
                    if (parentOpPtr->GetSubgraphID() != op->GetSubgraphID()) {
                        subGraph2StartTensor[op->GetSubgraphID()].insert(iTensor->GetMagic());
                        startTensor.insert(iTensor->GetMagic());
                        break;
                    }
                }
            }
        }
        for (auto &oTensor : op->GetOOperands()) {
            if (oTensor->GetConsumers().size() == 0) {
                subGraph2EndTensor[op->GetSubgraphID()].insert(oTensor->GetMagic());
                endTensor.insert(oTensor->GetMagic());
                break;
            } else {
                for (auto& childOpPtr : oTensor->GetConsumers()) {
                    if (childOpPtr->GetSubgraphID() != op->GetSubgraphID()) {
                        subGraph2EndTensor[op->GetSubgraphID()].insert(oTensor->GetMagic());
                        endTensor.insert(oTensor->GetMagic());
                        break;
                    }
                }
            }
        }
    }
    int subgraphTotal = subGraph2OpType.size();
    int aicGraphCount = 0;
    int aivGraphCount = 0;
    int mixGraphCount = 0;
    int aicpuGraphCount = 0;
    int otherGraphCount = 0;
    ReportVal("SubgraphTotal", subgraphTotal, 0);
    ReportSeq("OperationsTypeDistribution", 0);
    for (auto [subgraphId, opTypes] : subGraph2OpType) {
        int aicCount = 0;
        int aivCount = 0;
        int mixCount = 0;
        int aicpuCount = 0;
        int otherCount = 0;
        for (auto opType : opTypes) {
            if (opType == CoreType::AIC) {
                aicCount++;
            } else if (opType == CoreType::AICPU) {
                aicpuCount++;
            } else if (opType == CoreType::MIX) {
                mixCount++;
            } else if (opType == CoreType::AIV) {
                aivCount++;
            } else {
                otherCount++;
            }
        }

        if (mixCount > 0) {
            mixGraphCount++;
        } else if (aicCount > 0) {
            aicGraphCount++;
        } else if (aivCount > 0) {
            aivGraphCount++;
        } else if (aicpuCount > 0) {
            aicpuGraphCount++;
        } else {
            otherGraphCount++;
        }

        ReportSeq("=================================================", 0);
        ReportVal("SubgraphId", subgraphId, 1);
        ReportVal("AIC total", aicCount, ReportLevelTwo);
        ReportVal("AIV total", aivCount, ReportLevelTwo);
        ReportVal("AICPU total", aicpuCount, ReportLevelTwo);
        ReportVal("MIX total", mixCount, ReportLevelTwo);
        ReportVal("Other total", otherCount, ReportLevelTwo);
    }

    int graphTypeCount = 5;
    ReportVal("Subgraph Type Distribution", graphTypeCount, 0);
    ReportVal("AIC Subgraph Count", aicGraphCount, 1);
    ReportVal("AIV Subgraph Count", aivGraphCount, 1);
    ReportVal("MIX Subgraph Count", mixGraphCount, 1);
    ReportVal("AICPU Subgraph Count", aicpuGraphCount, 1);
    ReportVal("Other Subgraph Count", otherGraphCount, 1);

    SubgraphStats subgraphStats = CalculateStats(subGraph2StartTensor, subGraph2EndTensor);

    ReportVal("Operation MAX Fanin", subgraphStats.maxInput, 0);
    ReportVal("Operation MIN Fanin", subgraphStats.minInput, 0);
    ReportValDouble("Operation AVG Fanin", subgraphStats.avgInput, 0);
    ReportVal("Operation MAX Fanout", subgraphStats.maxOutput, 0);
    ReportVal("Operation MIN Fanout", subgraphStats.minOutput, 0);
    ReportValDouble("Operation AVG Fanout", subgraphStats.avgOutput, 0);
}

void HealthCheckTileGraph(Function &function) {
    std::vector<Operation *> opList = function.Operations().DuplicatedOpList();
    ReportVal("TileOpTotal", opList.size(), 0);

    std::unordered_set<std::shared_ptr<LogicalTensor>> tensorList;
    int index = 0;

    int maxProducerCount = INT_MIN;
    int minProducerCount = INT_MAX;
    int totalProducerCount = 0;

    int maxConsumerCount = INT_MIN;
    int minConsumerCount = INT_MAX;
    int totalConsumerCount = 0;

    uint64_t maxShape = std::numeric_limits<uint64_t>::min();
    uint64_t minShape = std::numeric_limits<uint64_t>::max();
    uint64_t totalShape = 0;
    double tensorCount = 0;

    int maxInputCount = INT_MIN;
    int minInputCount = INT_MAX;
    double totalInputCount = 0;

    int maxOutputCount = INT_MIN;
    int minOutputCount = INT_MAX;
    double totalOutputCount = 0;

    for (auto &op : opList) {
        auto iOperand = op->GetIOperands();
        auto iOperandSize = iOperand.size();
        maxInputCount = std::max(maxInputCount, static_cast<int>(iOperandSize));
        minInputCount = std::min(minInputCount, static_cast<int>(iOperandSize));
        totalInputCount += iOperandSize;
        
        for (auto &iTensor : iOperand) {
            if (tensorList.find(iTensor) == tensorList.end()) {
                tensorList.insert(iTensor);
                auto curShape = iTensor->GetShape();
                uint64_t res = 1;
                for (auto &dim : curShape) {
                    res *= dim;
                }
                maxShape = std::max(maxShape, res);
                minShape = std::min(minShape, res);
                totalShape += res;
                tensorCount++;

                maxProducerCount = std::max(maxProducerCount, static_cast<int>(iTensor->GetProducers().size()));
                minProducerCount = std::min(minProducerCount, static_cast<int>(iTensor->GetProducers().size()));
                totalProducerCount += iTensor->GetProducers().size();
                maxConsumerCount = std::max(maxConsumerCount, static_cast<int>(iTensor->GetConsumers().size()));
                minConsumerCount = std::min(minConsumerCount, static_cast<int>(iTensor->GetConsumers().size()));
                totalConsumerCount += iTensor->GetConsumers().size();
                index++;
            }
        }

        auto oOperand = op->GetOOperands();
        auto oOperandSize = oOperand.size();
        maxOutputCount = std::max(maxOutputCount, static_cast<int>(oOperandSize));
        minOutputCount = std::min(minOutputCount, static_cast<int>(oOperandSize));
        totalOutputCount += oOperandSize;

        for (auto &oTensor : oOperand) {
            if (tensorList.find(oTensor) == tensorList.end()) {
                tensorList.insert(oTensor);
                auto curShape = oTensor->GetShape();
                uint64_t res = 1;
                for (auto &dim : curShape) {
                    res *= dim;
                }
                maxShape = std::max(maxShape, res);
                minShape = std::min(minShape, res);
                totalShape += res;
                tensorCount++;

                maxProducerCount = std::max(maxProducerCount, static_cast<int>(oTensor->GetProducers().size()));
                minProducerCount = std::min(minProducerCount, static_cast<int>(oTensor->GetProducers().size()));
                totalProducerCount += oTensor->GetProducers().size();
                maxConsumerCount = std::max(maxConsumerCount, static_cast<int>(oTensor->GetConsumers().size()));
                minConsumerCount = std::min(minConsumerCount, static_cast<int>(oTensor->GetConsumers().size()));
                totalConsumerCount += oTensor->GetConsumers().size();
                index++;
            }
        }
    }

    const double avgInputCount = static_cast<double>(totalInputCount) / static_cast<double>(opList.size());
    const double avgOutputCount = static_cast<double>(totalOutputCount) / static_cast<double>(opList.size());

    const double avgProducerCount = static_cast<double>(totalProducerCount) / static_cast<double>(index);
    const double avgConsumerCount = static_cast<double>(totalConsumerCount) / static_cast<double>(index);

    const double avgTensorShape = static_cast<double>(totalShape) / static_cast<double>(tensorCount);

    ReportVal("TileTotal", tensorList.size(), 0);

    auto allMap = GetOpConnectionMap(opList);
    auto inputMap = allMap.first;
    auto outputMap = allMap.second;
    std::unordered_set<int> allNodes;
    for (const auto& entry : inputMap) {
        allNodes.insert(entry.first);
    }
    for (const auto& entry : outputMap) {
        allNodes.insert(entry.first);
    }

    std::unordered_map<int, int> inDegree;
    std::unordered_set<int> indegreeZeroNodes;
    for (int node : allNodes) {
        inDegree[node] = inputMap.count(node) > 0 ? inputMap.at(node).size() : 0;
    }
    for (const auto& entry : inDegree) {
        if (entry.second == 0) {
            indegreeZeroNodes.insert(entry.first);
        }
    }

    auto result = findLongestPath(outputMap, indegreeZeroNodes);
    ReportVal("MaxTileOpDepth", result.maxLength, 0);
    ALOG_INFO("All nodes in max TileOp depth path:");
    for (int n : result.nodePath) {
        ALOG_INFO("----------Op Magic: ", n);
    }

    ConcurrencyStats stats = CalculateOpConcurrency(inputMap, outputMap);
    ReportVal("MaxTileOpWidth", stats.maxConcurrency, 0);
    
    ALOG_INFO("Node's Op Magic In Max Concurrency Layers", 0);
    index = 0;

    for (const auto& layer : stats.maxLayersNodes) {
        ALOG_INFO("The Sequence Number Of This Maxlayer", index++);
        std::unordered_set<int> printedSubgraphs;
        for (int node : layer) {
            ALOG_INFO("----------Op Magic: ", node);
        }
    }
    ReportVal("TileTensor Max ProducerCount", maxProducerCount, 0);
    ReportVal("TileTensor Min ProducerCount", minProducerCount, 0);
    ReportValDouble("TileTensor Average ProducerCount", avgProducerCount, 0);
    ReportVal("TileTensor Max ConsumerCount", maxConsumerCount, 0);
    ReportVal("TileTensor Min ConsumerCount", minConsumerCount, 0);
    ReportValDouble("TileTensor Average ConsumerCount", avgConsumerCount, 0);
    ReportVal("TileTensor Max Shape", maxShape, 0);
    ReportVal("TileTensor Min Shape", minShape, 0);
    ReportValDouble("TileTensor Average Shape", avgTensorShape, 0);

    ReportVal("TileOp Max InputCount", maxInputCount, 0);
    ReportVal("TileOp Min InputCount", minInputCount, 0);
    ReportValDouble("TileOp Average InputCount", avgInputCount, 0);
    ReportVal("TileOp Max OutputCount", maxOutputCount, 0);
    ReportVal("TileOp Min OutputCount", minOutputCount, 0);
    ReportValDouble("TileOp Average OutputCount", avgOutputCount, 0);

    std::unordered_map<int, int> magic2subGraphID;
    std::unordered_map<int, std::vector<int>> subGraphID2Magic;
    for (auto op: opList){
        int subGraphId = op->GetSubgraphID();
        magic2subGraphID[op->GetOpMagic()] = subGraphId;
        subGraphID2Magic[subGraphId].push_back(op->GetOpMagic());
    }
    
    SubgraphBuilder builder(inputMap, outputMap, magic2subGraphID);
    auto topology = builder.buildSubgraphTopology();
 
    auto printSubgraphConnections = [](const auto& connMap, const std::string& title) {
        ALOG_INFO(" ");
        ALOG_INFO(title);
        for (const auto& [sg, links] : connMap) {
            ALOG_INFO("============");
            ALOG_INFO("Subgraph ", sg, " -> ");
            for (int x : links) {
                ALOG_INFO("Subgraph ", x);
            } 
        }
    };
 
    ALOG_INFO("Subgraph ID And All Nodes In This Subgraph:");
    for (const auto& [sg, nodes] : topology.members) {
        ALOG_INFO("============");
        ALOG_INFO_F("Subgraph %d:", sg);
        for (int n : nodes) {
            ALOG_INFO_F("Node's OpMagic = %d,", n);
        }
    }
    printSubgraphConnections(topology.subgraphInputs, "Subgraph Input Connections");
    printSubgraphConnections(topology.subgraphOutputs, "Subgraph Output Connections");
 
    ParallelismAnalyzer analyzer(topology);
    auto metrics = analyzer.calculateParallelism();
    ReportVal("MaxSubgraphDepth", metrics.longestDepth, 0);
    ReportVal("MaxSubgraphWidth", metrics.maxParallelism, 0);
 
    // level -> subgraph id
    for (size_t i = 0; i < metrics.levelsDetail.size(); ++i) {
        ALOG_INFO("============");
        ALOG_INFO("Level ", i, " has ", metrics.levelsDetail[i].size(), " subgraphs:");
        for (int sg : metrics.levelsDetail[i]) {
            ALOG_INFO_F("Subgraph %d ", sg);
        }
    }

    ALOG_INFO("============");
    ALOG_INFO("Longest Path (", metrics.longestDepth, " nodes): ");
    for (int sg : metrics.longestPath) {
        ALOG_INFO("Subgraph ", sg);
    }

    GetSubGraphInfo(function);
}
}