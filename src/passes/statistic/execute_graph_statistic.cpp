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
 * \file execute_graph_statistic.cpp
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
using namespace std;

namespace npu::tile_fwk {
void CheckAndUpdateForMaxRoot(int &dest, int uninit, int curr, std::function<void()> callback = nullptr) {
    if (dest == uninit || dest < curr) {
        dest = curr;
        if (callback != nullptr) {
            callback();
        }
    }
}
 
void ProgramStatistic::HealthCheckExecuteGraph(Function &func) {
    std::vector<Operation *> opList = func.Operations().DuplicatedOpList();
    ReportVal("CallOpTotal", func.Operations().size(), 0);

    int maxIncastCount = INT_MIN;
    int minIncastCount = INT_MAX;
    double totalIncastCount = 0;
    int maxOutcastCount = INT_MIN;
    int minOutcastCount = INT_MAX;
    double totalOutcastCount = 0;

    for (auto &op : opList) {
        auto iOperand = op->GetIOperands();
        auto iOperandSize = iOperand.size();
        maxIncastCount = std::max(maxIncastCount, static_cast<int>(iOperandSize));
        minIncastCount = std::min(minIncastCount, static_cast<int>(iOperandSize));
        totalIncastCount += iOperandSize;

        auto oOperand = op->GetOOperands();
        auto oOperandSize = oOperand.size();
        maxOutcastCount = std::max(maxOutcastCount, static_cast<int>(oOperandSize));
        minOutcastCount = std::min(minOutcastCount, static_cast<int>(oOperandSize));
        totalOutcastCount += oOperandSize;
    }
    const double avgIncastCount = static_cast<double>(totalIncastCount) / static_cast<double>(opList.size());
    const double avgOutcastCount = static_cast<double>(totalOutcastCount) / static_cast<double>(opList.size());

    auto allMap = GetOpConnectionMap(opList);
    auto inputMap = allMap.first;
    auto outputMap = allMap.second;
    cleanMap(inputMap);
    cleanMap(outputMap);
    ALOG_INFO("================================== INPUT MAP =====================================");
    for (auto &[key, value] : inputMap) {
        ALOG_INFO_F("Subgraph ID: %d", key);
        for (auto v : value) {
            ALOG_INFO_F("Its Input Subgraph ID: %d", v);
        }
    }
    ALOG_INFO("================================== OUTPUT MAP =====================================");
    for (auto &[key, value] : outputMap) {
        ALOG_INFO_F("Subgraph ID: %d", key);
        for (auto v : value) {
            ALOG_INFO_F("Its Input Subgraph ID: %d", v);
        }
    }

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
    ConcurrencyStats stats = CalculateOpConcurrency(inputMap, outputMap);

    ReportVal("MaxCallOpDepth", result.maxLength, 0);
    ALOG_INFO("All nodes in max depth path:");
    for (int n : result.nodePath) {
        ALOG_INFO("----------Op Magic: ", n);
    }

    ReportVal("MaxCallOpWidth", stats.maxConcurrency, 0);
    ALOG_INFO("Node's Op Magic In Max Concurrency Layers", 0);
    int index = 0;
    for (const auto& layer : stats.maxLayersNodes) {
        ALOG_INFO("The Sequence Number Of This Maxlayer", index++);
        std::unordered_set<int> printedSubgraphs;
        for (int node : layer) {
            ALOG_INFO("----------Op Magic: ", node);
        }
    }

    ReportVal("CallOp MAX IncastCount", maxIncastCount, 0);
    ReportVal("CallOp MIN IncastCount", minIncastCount, 0);
    ReportValDouble("CallOp AVG IncastCount", avgIncastCount, 0);
    ReportVal("CallOp MAX OutcastCount", maxOutcastCount, 0);
    ReportVal("CallOp MIN OutcastCount", minOutcastCount, 0);
    ReportValDouble("CallOp AVG OutcastCount", avgOutcastCount, 0);

    std::set<std::pair<int, int>> rootOpInputSet;
    std::set<std::pair<int, int>> rootOpOutputSet;
    std::set<std::pair<int, int>> rootProducerSet;
    std::set<std::pair<int, int>> rootConsumerSet;
    for (auto &op : func.Operations()) {
        GetTopElement(op.GetIOperands().size(), op.GetOpMagic(), rootOpInputQueue, rootOpInputSet);
        GetTopElement(op.GetOOperands().size(), op.GetOpMagic(), rootOpOutputQueue, rootOpOutputSet);
        CheckAndUpdateForMaxRoot(maxRootOpInput, STATISTIC_UNINIT, op.GetIOperands().size(), [&](){
            maxRootOpInputMagic = op.GetOpMagic();
        });
        CheckAndUpdateForMaxRoot(maxRootOpOutput, STATISTIC_UNINIT, op.GetOOperands().size(), [&](){
            maxRootOpOutputMagic = op.GetOpMagic();
        });
        for (auto &i : op.GetIOperands()) {
            GetTopElement(i->GetProducers().size(), i->GetMagic(), rootProducerQueue, rootProducerSet);
            GetTopElement(i->GetConsumers().size(), i->GetMagic(), rootConsumerQueue, rootConsumerSet);
            CheckAndUpdateForMaxRoot(maxRootTensorFanin, STATISTIC_UNINIT, i->GetProducers().size(), [&](){
                maxRootTensorFaninMagic = i->GetMagic();
            });
            CheckAndUpdateForMaxRoot(maxRootTensorFanout, STATISTIC_UNINIT, i->GetConsumers().size(), [&](){
                maxRootTensorFanoutMagic = i->GetMagic();
            });
        }
        for (auto &o : op.GetOOperands()) {
            GetTopElement(o->GetProducers().size(), o->GetMagic(), rootProducerQueue, rootProducerSet);
            GetTopElement(o->GetConsumers().size(), o->GetMagic(), rootConsumerQueue, rootConsumerSet);
            CheckAndUpdateForMaxRoot(maxRootTensorFanin, STATISTIC_UNINIT, o->GetProducers().size(), [&](){
                maxRootTensorFaninMagic = o->GetMagic();
            });
            CheckAndUpdateForMaxRoot(maxRootTensorFanout, STATISTIC_UNINIT, o->GetConsumers().size(), [&](){
                maxRootTensorFanoutMagic = o->GetMagic();
            });
        }
    }
    SortTopElement(rootFaninMaps, rootProducerQueue);
    SortTopElement(rootFanoutMaps, rootConsumerQueue);
    SortTopElement(rootOpInputMaps, rootOpInputQueue);
    SortTopElement(rootOpOutputMaps, rootOpOutputQueue);
}

void HealthCheckIsomorphismSubgraph(std::multimap<int, int> psgToESgMap, std::vector<std::vector<OperationPtr>> nLIST) {
    ProgramStatistic programStatistic;
    std::shared_ptr<Function> root;
    for (const auto &[name, func] : Program::GetInstance().GetFunctionMap()) {
        if (name.find("root") != std::string::npos) {
            root = func;
        }
    }
    ReportTitle("After PASS: SubgraphToFunction, Health Report: ExecuteGraph START");
    programStatistic.HealthCheckExecuteGraph(*root);
    programStatistic.ReportExecuteGraphTopElements();
    ReportTitle("After PASS: SubgraphToFunction, Health Report: ExecuteGraph END");

    ReportTitle("After PASS: SubgraphToFunction, Health Report: KernelGraph START");
    for (auto [psgId, esgId] : psgToESgMap) {
        auto iter = root->programs_.find(psgId);
        auto subFunction = iter->second;
        auto &subgraphStatistic = programStatistic.HealthCheckKernelGraph(*subFunction, esgId);

        programStatistic.totalCalledSubgraph++;
        switch (subgraphStatistic.coreType) {
        case OpCoreType::AIC:
            programStatistic.totalCalledAICSubgraph++;
            break;
        case OpCoreType::AIV:
            programStatistic.totalCalledAIVSubgraph++;
            break;
        default:
            ASSERT(false);
        }
        programStatistic.totalCalledOperationCount += subgraphStatistic.operationCount;
    }

    programStatistic.SortTopElement(programStatistic.funcFaninMaps, programStatistic.leafProducerQueue);
    programStatistic.SortTopElement(programStatistic.funcFanoutMaps, programStatistic.leafConsumerQueue);

    int totalIsomorphicSubgraphs = 0;
    double isomorphicRate = 0.0;
    std::map<int, std::vector<int>> isomorphicClass;
    std::map<int, int> isomorphicClassSize;
    std::map<int, int> isomorphicClassNodeCount;
    for (const auto &[psgId, esgId] : psgToESgMap) {
        int isomorphicClassId = psgId;
        isomorphicClass[psgId].push_back(esgId);
        isomorphicClassSize[isomorphicClassId]++;
        auto &subgraph = nLIST[esgId];
        isomorphicClassNodeCount[isomorphicClassId] += subgraph.size();
    }
    totalIsomorphicSubgraphs = isomorphicClassSize.size();
    if (nLIST.size() > 0) {
        isomorphicRate = static_cast<double>(totalIsomorphicSubgraphs) / nLIST.size();
    }

    ReportVal("Total Kernelgraph Count", nLIST.size(), 0);
    ReportVal("Total Isomorphic Kernelgraph Count", totalIsomorphicSubgraphs, 0);
    ReportValDouble("KernelgraphToSubgraphRatio", isomorphicRate, 0);
    ReportSeq("Isomorphic Class And Subgraph Count", 0);
    for (const auto &[classId, size] : isomorphicClassSize) {
        ReportSeq("=================================================", 0);
        ReportVal("Class", classId, 1);
        ReportVal("Subgraph Count", size, 1);
        for (const auto &esgId : isomorphicClass[classId]) {
            ReportVal("Subgraph Id", esgId, ReportLevelTwo);
        }
    }
    ReportSeq("Isomorphic Class And Op Count In This Subgraph:", 0);
    for (const auto &[classId, nodeCount] : isomorphicClassNodeCount) {
        ReportSeq("=================================================", 0);
        ReportVal("Class", classId, 1);
        ReportVal("Op Count", nodeCount, 1);
    }
    programStatistic.ReportKernelGraphTopElements();
    ReportTitle("After PASS: SubgraphToFunction, Health Report: KernelGraph END");
}
}