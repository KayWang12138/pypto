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
 * \file kernel_graph_statistic.cpp
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
void CheckAndUpdateForMaxLeaf(int &dest, int uninit, int curr, std::function<void()> callback = nullptr) {
    if (dest == uninit || dest < curr) {
        dest = curr;
        if (callback != nullptr) {
            callback();
        }
    }
}

const SubgraphStatistic &ProgramStatistic::HealthCheckKernelGraph(Function &func, int subgraphID)
{
    std::vector<Operation *> opList = func.Operations().DuplicatedOpList();
    if (statistic.count(func.GetFunctionHash().GetHash())) {
        return statistic[func.GetFunctionHash().GetHash()];
    }
    ReportSeq("================================================", 0);
    ReportVal("This KernelGraph's SubgraphID", subgraphID, 0);
    SubgraphStatistic &subgraphStatistic = statistic[func.GetFunctionHash().GetHash()];
    
    std::unordered_set<std::shared_ptr<LogicalTensor>> tensorList;
    int aicCount = 0;
    int aivCount = 0;
    int mixCount = 0;

    int maxProducerCount = INT_MIN;
    int minProducerCount = INT_MAX;
    int totalProducerCount = 0;

    int maxConsumerCount = INT_MIN;
    int minConsumerCount = INT_MAX;
    int totalConsumerCount = 0;

    int maxInputCount = INT_MIN;
    int minInputCount = INT_MAX;
    double totalInputCount = 0;

    int maxOutputCount = INT_MIN;
    int minOutputCount = INT_MAX;
    double totalOutputCount = 0;

    int index = 0;
    double tensorCount = 0;

    for (auto &op : opList) {
        ALOG_INFO_F("Op code = %s, op core type = %s", op->GetOpcodeStr().c_str(), op->GetCoreTypeStr().c_str());
        if (op->GetCoreType() == CoreType::AIC) {
            aicCount++;
        } else if (op->GetCoreType() == CoreType::AIV) {
            aivCount++;
        } else if (op->GetCoreType() == CoreType::MIX) {
            mixCount++;
        }

        auto iOperand = op->GetIOperands();
        auto iOperandSize = iOperand.size();
        maxInputCount = std::max(maxInputCount, static_cast<int>(iOperandSize));
        minInputCount = std::min(minInputCount, static_cast<int>(iOperandSize));
        totalInputCount += iOperandSize;

        for (auto &iTensor : op->GetIOperands()) {
            if (tensorList.find(iTensor) == tensorList.end()) {
                tensorList.insert(iTensor);
                int realProducerCount = 0;
                int realConsumerCount = 0;
                for (auto producer : iTensor->GetProducers()) {
                    if ((std::count(opList.begin(), opList.end(), producer))) {
                        realProducerCount++;
                    }
                }
                for (auto consumer : iTensor->GetConsumers()) {
                    if ((std::count(opList.begin(), opList.end(), consumer))) {
                        realConsumerCount++;
                    }
                }
                maxProducerCount = std::max(maxProducerCount, static_cast<int>(realProducerCount));
                minProducerCount = std::min(minProducerCount, static_cast<int>(realProducerCount));
                totalProducerCount += realProducerCount;
                maxConsumerCount = std::max(maxConsumerCount, static_cast<int>(realConsumerCount));
                minConsumerCount = std::min(minConsumerCount, static_cast<int>(realConsumerCount));
                totalConsumerCount += realConsumerCount;
                index++;
                tensorCount++;
            }
            if (iTensor->memorymap.size() == 1) {
                GetTopElement(iTensor->GetProducers().size(), iTensor->GetMagic(), leafProducerQueue, leafProducerSet);
                GetTopElement(iTensor->GetConsumers().size(), iTensor->GetMagic(), leafConsumerQueue, leafConsumerSet);
                leafMagic2SubgraphID[iTensor->GetMagic()] = subgraphID;
                CheckAndUpdateForMaxLeaf(subgraphStatistic.maxTensorFanin, STATISTIC_UNINIT, iTensor->GetProducers().size(), [&](){
                    subgraphStatistic.maxTensorFaninMagic = iTensor->GetMagic();
                });
                CheckAndUpdateForMaxLeaf(subgraphStatistic.maxTensorFanout, STATISTIC_UNINIT, iTensor->GetConsumers().size(), [&](){
                    subgraphStatistic.maxTensorFanoutMagic = iTensor->GetMagic();
                });
            }
        }

        auto oOperand = op->GetOOperands();
        auto oOperandSize = oOperand.size();
        maxOutputCount = std::max(maxOutputCount, static_cast<int>(oOperandSize));
        minOutputCount = std::min(minOutputCount, static_cast<int>(oOperandSize));
        totalOutputCount += oOperandSize;

        for (auto &oTensor : op->GetOOperands()) {
            if (tensorList.find(oTensor) == tensorList.end()) {
                tensorList.insert(oTensor);
                int realProducerCount = 0;
                int realConsumerCount = 0;
                for (auto producer : oTensor->GetProducers()) {
                    if ((std::count(opList.begin(), opList.end(), producer))) {
                        realProducerCount++;
                    }
                }
                for (auto consumer : oTensor->GetConsumers()) {
                    if ((std::count(opList.begin(), opList.end(), consumer))) {
                        realConsumerCount++;
                    }
                }
                maxProducerCount = std::max(maxProducerCount, static_cast<int>(realProducerCount));
                minProducerCount = std::min(minProducerCount, static_cast<int>(realProducerCount));
                totalProducerCount += realProducerCount;
                maxConsumerCount = std::max(maxConsumerCount, static_cast<int>(realConsumerCount));
                minConsumerCount = std::min(minConsumerCount, static_cast<int>(realConsumerCount));
                totalConsumerCount += realConsumerCount;
                index++;
                tensorCount++;
            }

            if (oTensor->memorymap.size() == 1) {
                GetTopElement(oTensor->GetProducers().size(), oTensor->GetMagic(), leafProducerQueue, leafProducerSet);
                GetTopElement(oTensor->GetConsumers().size(), oTensor->GetMagic(), leafConsumerQueue, leafConsumerSet);
                leafMagic2SubgraphID[oTensor->GetMagic()] = subgraphID;
                CheckAndUpdateForMaxLeaf(subgraphStatistic.maxTensorFanin, STATISTIC_UNINIT, oTensor->GetProducers().size(), [&](){
                    subgraphStatistic.maxTensorFaninMagic = oTensor->GetMagic();
                });
                CheckAndUpdateForMaxLeaf(subgraphStatistic.maxTensorFanout, STATISTIC_UNINIT, oTensor->GetConsumers().size(), [&](){
                    subgraphStatistic.maxTensorFanoutMagic = oTensor->GetMagic();
                });
            }
        }
    }

    subgraphStatistic.operationCount = func.Operations().size();
    subgraphStatistic.tensorCount = tensorList.size();
    ReportVal("TileOpTotal", subgraphStatistic.operationCount, 0);
    ReportVal("TileTotal", subgraphStatistic.tensorCount, 0);

    if (aicCount) {
        aicCount += mixCount;
    } else {
        aivCount += mixCount;
    }
    const double avgInputCount = static_cast<double>(totalInputCount) / static_cast<double>(opList.size());
    const double avgOutputCount = static_cast<double>(totalOutputCount) / static_cast<double>(opList.size());
    const double avgProducerCount = static_cast<double>(totalProducerCount) / static_cast<double>(index);
    const double avgConsumerCount = static_cast<double>(totalConsumerCount) / static_cast<double>(index);

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
    ALOG_INFO("All nodes in max depth path:");
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

    ReportVal("TileOp Max InputCount", maxInputCount, 0);
    ReportVal("TileOp Min InputCount", minInputCount, 0);
    ReportValDouble("TileOp Average InputCount", avgInputCount, 0);
    ReportVal("TileOp Max OutputCount", maxOutputCount, 0);
    ReportVal("TileOp Min OutputCount", minOutputCount, 0);
    ReportValDouble("TileOp Average OutputCount", avgOutputCount, 0);

    if (aicCount == 0 && aivCount == 0) {
        ALOG_INFO_F("Subgraph has %d operation, %d tensor", subgraphStatistic.operationCount, subgraphStatistic.tensorCount);
        ALOG_INFO_F("Subgraph %d is empty, ERROR", subgraphID);
    }

    if (aicCount != 0) {
        subgraphStatistic.coreType = OpCoreType::AIC;
    } else if (aivCount != 0) {
        subgraphStatistic.coreType = OpCoreType::AIV;
    } else {
        ASSERT(false) << func.Dump();
    }

    maxOperationCount.CheckAndUpdate(subgraphStatistic.operationCount, subgraphID);
    minOperationCount.CheckAndUpdate(subgraphStatistic.operationCount, subgraphID);

    CheckAndUpdateForMaxLeaf(maxTensorFanin.value, STATISTIC_UNINIT, subgraphStatistic.maxTensorFanin, [&](){
        maxTensorFanin.subgraphID = subgraphID;
        maxTensorFaninMagic = subgraphStatistic.maxTensorFaninMagic;
    });
    CheckAndUpdateForMaxLeaf(maxTensorFanout.value, STATISTIC_UNINIT, subgraphStatistic.maxTensorFanout, [&](){
        maxTensorFanout.subgraphID = subgraphID;
        maxTensorFanoutMagic = subgraphStatistic.maxTensorFanoutMagic;
    });

    totalUniqueSubgraph++;
    totalUniqueOperationCount += subgraphStatistic.operationCount;
    switch (subgraphStatistic.coreType) {
    case OpCoreType::AIC:
        maxAICOperationCount.CheckAndUpdate(subgraphStatistic.operationCount, subgraphID);
        minAICOperationCount.CheckAndUpdate(subgraphStatistic.operationCount, subgraphID);
        totalUniqueAICSubgraph++;
        break;
    case OpCoreType::AIV:
        maxAIVOperationCount.CheckAndUpdate(subgraphStatistic.operationCount, subgraphID);
        minAIVOperationCount.CheckAndUpdate(subgraphStatistic.operationCount, subgraphID);
        totalUniqueAIVSubgraph++;
        break;
    default:
        ASSERT(false);
    }
    return subgraphStatistic;
}
}