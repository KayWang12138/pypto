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
 * \file tensor_and_tile_graph_statistic.cpp
 * \brief
 */

#include "tensor_and_tile_graph_statistic.h"

#include <sstream>
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <unordered_set>
#include <vector>
#include <unordered_map>
#include <queue>
#include <climits>
#include <nlohmann/json.hpp>

#include "interface/utils/log.h"
#include "interface/configs/config_manager.h"
#include "interface/function/function.h"
#include "interface/tensor/logical_tensor.h"
#include "interface/tensor/raw_tensor.h"
#include "interface/utils/file_utils.h"

namespace npu {
namespace tile_fwk {
using json = nlohmann::json;

template <typename tType>
uint64_t CalcTensorSize(const std::vector<tType> &curShape) {
    uint64_t res = 1;
    for (auto &dim : curShape) {
        res *= dim;
    }
    return res;
}

void CalcOperatorInfo(Function &function, json &report) {
    MetricData memoryMetric;

    auto operationViewer = function.Operations();
    uint64_t totalCopySize = 0;
    for (size_t i = 0; i < operationViewer.size(); i++) {
        auto opCode = operationViewer[i].GetOpcode();
        int opMagic = operationViewer[i].GetOpMagic();
        uint64_t iShapeSize = 0;
        for (auto &inOperand : operationViewer[i].GetIOperands()) {
            iShapeSize += CalcTensorSize(inOperand->GetShape());
        }
        if (IsCopyOut(opCode)) {
            totalCopySize += iShapeSize;
        }
        uint64_t oShapeSize = 0;
        for (auto &outOperand : operationViewer[i].GetOOperands()) {
            oShapeSize += CalcTensorSize(outOperand->GetShape());
        }
        if (IsCopyIn(opCode)) {
            totalCopySize += oShapeSize;
        }
        memoryMetric.UpdateMetricData(iShapeSize + oShapeSize, opMagic);
    }

    // 写出memoryMetric和copyMetric
    report["totalOpCount"] = operationViewer.size();
    report["peakMemory"] = {
        {"peakMemoryUsage", memoryMetric.GetMaxSize()},
        {"peakMemoryUsageOps", *memoryMetric.GetMaxNodes()}
    };

    report["copyDataCount"] = totalCopySize;
    report["redundantCopyCount"] = 0;
}

void CalcTensorInfo(Function &function, json &report) {
    MetricData consumerMetric;
    MetricData producerMetric;
    for (auto ele : function.GetTensorMap().inverseMap_) {
        auto tensor = ele.second;
        int tensorMagic = tensor->GetMagic();
        uint64_t consumerSize = static_cast<uint64_t>(tensor->GetConsumers().size());
        consumerMetric.UpdateMetricData(consumerSize, tensorMagic);
        uint64_t producerSize = static_cast<uint64_t>(tensor->GetProducers().size());
        producerMetric.UpdateMetricData(producerSize, tensorMagic);
    }

    // 写出consumerMetric和producerMetric
    report["totalTensorCount"] = function.GetTensorMap().inverseMap_.size();
    report["maxConsumerCount"] = consumerMetric.GetMaxSize();
    report["maxConsumerTensors"] = *consumerMetric.GetMaxNodes();
    report["maxproducerCount"] = producerMetric.GetMaxSize();
    report["maxproducerTensors"] = *producerMetric.GetMaxNodes();
}

void GetOpConnectionMap(Function &function,
    std::vector<std::vector<int>> &inMap, std::vector<std::vector<int>> &outMap, std::vector<bool> &actualMagic)
{
    // 找到最大的magic编号
    MetricData magicNum;
    auto operationViewer = function.Operations();
    for (size_t i = 0; i < operationViewer.size(); i++) {
        magicNum.UpdateMetricData(static_cast<uint64_t>(operationViewer[i].GetOpMagic()), static_cast<int>(i));
    }
    size_t magicUpperRange = static_cast<size_t>(magicNum.GetMaxSize()) + static_cast<size_t>(1);

    // 生成inMap和outMap
    inMap.resize(magicUpperRange);
    outMap.resize(magicUpperRange);
    actualMagic.resize(magicUpperRange);
    std::fill(actualMagic.begin(), actualMagic.end(), false);
    for (size_t i = 0; i < operationViewer.size(); i++) {
        int childMagic = operationViewer[i].GetOpMagic();
        actualMagic[childMagic] = true;
        for (auto &input : operationViewer[i].GetIOperands()) {
            for (auto &parentOpPtr : input->GetProducers()) {
                int parentMagic = parentOpPtr->GetOpMagic();
                auto it = std::find(inMap[childMagic].begin(), inMap[childMagic].end(), parentMagic);
                if (it == inMap[childMagic].end()) {
                    inMap[childMagic].push_back(parentMagic);
                    outMap[parentMagic].push_back(childMagic);
                }
            }
        }
    }
}

void TraversePathUp(const int parent, const std::vector<std::vector<int>> &outMap, std::vector<int> &layerMap) {
    int parentLayer = layerMap[parent];
    for (auto child : outMap[parent]) {
        if (layerMap[child] <= parentLayer) {
            layerMap[child] = parentLayer + 1;
            TraversePathUp(child, outMap, layerMap);
        }
    }
}

void CalcGraphMetrics(const std::vector<std::vector<int>> &inMap, const std::vector<std::vector<int>> &outMap,
    const std::vector<bool> &actualVertex, json &report)
{
    MetricData inDegreeMetric;
    MetricData outDegreeMetric;
    std::vector<int> layerMap = std::vector<int>(inMap.size(), 0);

    // 计算每层节点层数，inDegree和outDegree
    for (size_t i = 0; i < inMap.size(); i++) {
        inDegreeMetric.UpdateMetricData(static_cast<uint64_t>(inMap[i].size()), static_cast<int>(i));
        outDegreeMetric.UpdateMetricData(static_cast<uint64_t>(outMap[i].size()), static_cast<int>(i));
        if (actualVertex[i] && (inMap[i].size() == 0)) {
            TraversePathUp(static_cast<int>(i), outMap, layerMap);
        }
    }

    // 找到最大层数，最大层数等于最长路径的长度
    int maxLayerNum = *std::max_element(layerMap.begin(), layerMap.end()) + 1;

    // 找到最宽层的节点数
    std::vector<int> layerCount = std::vector<int>(maxLayerNum, 0);
    for (size_t i = 0; i < inMap.size(); i++) {
        if (actualVertex[i]) {
            layerCount[layerMap[i]] ++;
        }
    }
    int maxLayerWidth = *std::max_element(layerCount.begin(), layerCount.end());

    // 写出inDegreeMetric, outDegreeMetric, maxLayerNum, maxLayerWidth
    report["maxFanin"] = inDegreeMetric.GetMaxSize();
    report["maxFaninOps"] = *inDegreeMetric.GetMaxNodes();
    report["maxFanout"] = outDegreeMetric.GetMaxSize();
    report["maxFanoutOps"] = *outDegreeMetric.GetMaxNodes();
    report["maxDepth"] = maxLayerNum;
    report["maxWidth"] = maxLayerWidth;
}

void WriteHealthReport(const json& report, const std::string &reportPath, const std::string& filename) {
    if (!CreateMultiLevelDir(reportPath)) {
        ALOG_ERROR_F("Failed to create directory for health report");
    }
    std::ofstream out(reportPath + "/" + filename);
    if (!out.is_open()) {
        ALOG_ERROR_F("Failed to open health report file for writing");
        return;
    }
    out << report.dump(DUMP_WIDTH);
    out.close();
}

void HealthCheckTensorGraph(Function &function, const std::string &reportPath, const std::string &fileName) {
    json tensorGraphReport;

    // 1. 计算operation节点信息
    tensorGraphReport["totalOpCount"] = function.Operations().size();

    // 2. 计算tensor节点信息
    CalcTensorInfo(function, tensorGraphReport);

    // 3. 构建operation节点图
    std::vector<std::vector<int>> inMap;
    std::vector<std::vector<int>> outMap;
    std::vector<bool> actualMagic;
    GetOpConnectionMap(function, inMap, outMap, actualMagic);
    if (inMap.size() == 0) {
        return;
    }

    // 4. 计算图信息
    CalcGraphMetrics(inMap, outMap, actualMagic, tensorGraphReport);

    // 5. 写出健康报告
    std::string graphName = fileName + "_TensorGraphHealthReport.json";
    WriteHealthReport(tensorGraphReport, reportPath, graphName);
}

void HealthCheckTileGraph(Function &function, const std::string &reportPath, const std::string &fileName) {
    json tileGraphReport;
    
    // 1. 计算operation节点信息
    CalcOperatorInfo(function, tileGraphReport);

    // 2. 计算tensor节点信息
    CalcTensorInfo(function, tileGraphReport);

    // 3. 构建operation节点图
    std::vector<std::vector<int>> inMap; // magic到magic的映射，in - parent, out - child
    std::vector<std::vector<int>> outMap;
    std::vector<bool> actualMagic;
    GetOpConnectionMap(function, inMap, outMap, actualMagic);
    if (inMap.size() == 0) {
        return;
    }

    // 4. 计算图信息
    CalcGraphMetrics(inMap, outMap, actualMagic, tileGraphReport);

    // 5. 写出健康报告
    std::string graphName = fileName + "_TileGraphHealthReport.json";
    WriteHealthReport(tileGraphReport, reportPath, graphName);
}

} // namespace tile_fwk
} // namespace npu