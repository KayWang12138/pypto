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
 * \file tensor_graph_statistic.cpp
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
    
void HealthCheckTensorGraph(Function &function) {
    std::vector<Operation *> opList = function.Operations().DuplicatedOpList();
    ReportVal("OperationTotal", opList.size(), 0);

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
                maxProducerCount = std::max(maxProducerCount, static_cast<int>(iTensor->GetProducers().size()));
                minProducerCount = std::min(minProducerCount, static_cast<int>(iTensor->GetProducers().size()));
                totalProducerCount += iTensor->GetProducers().size();
                maxConsumerCount = std::max(maxConsumerCount, static_cast<int>(iTensor->GetConsumers().size()));
                minConsumerCount = std::min(minConsumerCount, static_cast<int>(iTensor->GetConsumers().size()));
                totalConsumerCount += iTensor->GetConsumers().size();
                index++;
                auto curShape = iTensor->GetShape();
                uint64_t res = 1;
                for (auto &dim : curShape) {
                    res *= dim;
                }
                maxShape = std::max(maxShape, res);
                minShape = std::min(minShape, res);
                totalShape += res;
                tensorCount++;
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
                maxProducerCount = std::max(maxProducerCount, static_cast<int>(oTensor->GetProducers().size()));
                minProducerCount = std::min(minProducerCount, static_cast<int>(oTensor->GetProducers().size()));
                totalProducerCount += oTensor->GetProducers().size();
                maxConsumerCount = std::max(maxConsumerCount, static_cast<int>(oTensor->GetConsumers().size()));
                minConsumerCount = std::min(minConsumerCount, static_cast<int>(oTensor->GetConsumers().size()));
                totalConsumerCount += oTensor->GetConsumers().size();
                index++;
                auto curShape = oTensor->GetShape();
                uint64_t res = 1;
                for (auto &dim : curShape) {
                    res *= dim;
                }
                maxShape = std::max(maxShape, res);
                minShape = std::min(minShape, res);
                totalShape += res;
                tensorCount++;
            }
        }
    }

    const double avgInputCount = static_cast<double>(totalInputCount) / static_cast<double>(opList.size());
    const double avgOutputCount = static_cast<double>(totalOutputCount) / static_cast<double>(opList.size());
    const double avgProducerCount = static_cast<double>(totalProducerCount) / static_cast<double>(index);
    const double avgConsumerCount = static_cast<double>(totalConsumerCount) / static_cast<double>(index);
    const double avgTensorShape = static_cast<double>(totalShape) / static_cast<double>(tensorCount);

    ReportVal("TensorTotal", tensorList.size(), 0);

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
    ReportVal("Operation MaxDepth", result.maxLength, 0);
    ALOG_INFO("All nodes in max depth path:");
    for (int n : result.nodePath) {
        ALOG_INFO("----------Op Magic: ", n);
    }

    ConcurrencyStats stats = CalculateOpConcurrency(inputMap, outputMap);
    ReportVal("Operation MaxWidth", stats.maxConcurrency, 0);

    ALOG_INFO("Node's Op Magic In Max Concurrency Layers", 0);
    index = 0;
    for (const auto& layer : stats.maxLayersNodes) {
        ALOG_INFO("The Sequence Number Of This Maxlayer", index++);
        std::unordered_set<int> printedSubgraphs;
        for (int node : layer) {
            ALOG_INFO("----------Op Magic: ", node);
        }
    }

    ReportVal("Tensor Max ProducerCount", maxProducerCount, 0);
    ReportVal("Tensor Min ProducerCount", minProducerCount, 0);
    ReportValDouble("Tensor Average ProducerCount", avgProducerCount, 0);
    ReportVal("Tensor Max ConsumerCount", maxConsumerCount, 0);
    ReportVal("Tensor Min ConsumerCount", minConsumerCount, 0);
    ReportValDouble("Tensor Average ConsumerCount", avgConsumerCount, 0);
    ReportVal("Tensor Max Shape", maxShape, 0);
    ReportVal("Tensor Min Shape", minShape, 0);
    ReportValDouble("Tensor Average Shape", avgTensorShape, 0);

    ReportVal("Operation Max InputCount", maxInputCount, 0);
    ReportVal("Operation Min InputCount", minInputCount, 0);
    ReportValDouble("Operation Average InputCount", avgInputCount, 0);
    ReportVal("Operation Max OutputCount", maxOutputCount, 0);
    ReportVal("Operation Min OutputCount", minOutputCount, 0);
    ReportValDouble("Operation Average OutputCount", avgOutputCount, 0);
}
}