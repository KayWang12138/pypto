/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once
#ifndef OPENSHMEM_PTO_INTEGRATION_H
#define OPENSHMEM_PTO_INTEGRATION_H

#include "openshmem_npu.h"
#include "cost_model/simulation/arch/PipeSimulatorFast.h"
#include "cost_model/simulation/arch/A2A3/PostSimulatorA2A3.h"
#include "../../interface/operation/operation.h"
#include "../../interface/function/function.h"
#include <unordered_map>
#include <vector>
#include <string>

namespace npu {
namespace openshmem {

// OpenSHMEM PTO集成类
class OpenSHMEMPtoIntegration {
public:
    explicit OpenSHMEMPtoIntegration(OpenSHMEMContext* openshmemContext);
    ~OpenSHMEMPtoIntegration();

    // PTO模拟器集成
    int initializePtoSimulator();
    uint64_t simulateOpenSHMEMOperation(const std::string& operationName,
                                       size_t dataSize, int targetPe);

    // 通信性能建模
    uint64_t modelCommunicationLatency(int sourcePe, int targetPe, size_t messageSize);
    uint64_t modelSynchronizationLatency(int numPes);
    uint64_t modelMemoryOperationLatency(size_t dataSize, bool isNpuOptimized);

    // 网络拓扑建模
    uint64_t modelNetworkLatency(int sourcePe, int targetPe, const std::string& networkType = "ring");
    uint64_t modelCongestionLatency(size_t totalBandwidth, size_t concurrentOperations);

    // NPU特定建模
    uint64_t modelNpuTransferLatency(size_t dataSize, bool useHbm = true);
    uint64_t modelNpuComputationLatency(const std::string& operationType, size_t dataSize);

    // 端到端性能分析
    struct PerformanceReport {
        uint64_t totalLatency = 0;
        uint64_t communicationLatency = 0;
        uint64_t computationLatency = 0;
        uint64_t synchronizationLatency = 0;
        uint64_t memoryLatency = 0;
        double communicationEfficiency = 0.0;
        double bandwidthUtilization = 0.0;
        double npuUtilization = 0.0;
        std::vector<std::pair<std::string, uint64_t>> bottleneckOperations;
        std::unordered_map<std::string, double> resourceUtilization;
    };

    PerformanceReport generatePerformanceReport();
    void optimizeBasedOnPtoFeedback();

    // 配置管理
    void setPtoConfig(const std::string& archType = "A2A3", int accuracyLevel = 1);
    void updatePtoParameters(const std::unordered_map<std::string, double>& params);

    // 实时监控集成
    void enableRealTimeMonitoring(bool enable);
    void collectOperationMetrics(const std::string& operationName, uint64_t latency);

private:
    OpenSHMEMContext* openshmemContext_;
    std::string archType_;
    int accuracyLevel_;
    bool realTimeMonitoring_ = false;

    // PTO模拟器
    std::unique_ptr<CostModel::PipeSimulatorFast<CostModel::PostSimulatorA2A3>> fastSimulator_;

    // 性能指标收集
    struct PerformanceMetrics {
        uint64_t totalOperations = 0;
        uint64_t totalLatency = 0;
        uint64_t totalDataTransferred = 0;
        std::unordered_map<std::string, uint64_t> operationLatencies;
        std::unordered_map<std::string, size_t> operationCounts;
        std::unordered_map<std::string, size_t> operationDataSizes;
        std::vector<uint64_t> latencyHistory;
    } metrics_;

    // OpenSHMEM操作到PTO的映射
    static std::unordered_map<std::string, std::string> createOperationMapping();
    static std::unordered_map<std::string, CostModel::CorePipeType> createPipeMapping();

    // 内部辅助方法
    CostModel::TileOpPtr createTileOpFromOpenSHMEMOp(const std::string& operationName,
                                                    size_t dataSize, int targetPe);

    uint64_t runPtoSimulation(CostModel::TileOpPtr tileOp);
    void collectPerformanceData(const std::string& operationName, uint64_t latency, size_t dataSize);

    // 网络建模辅助
    uint64_t estimateNetworkHops(int sourcePe, int targetPe, const std::string& topology);
    uint64_t estimateBandwidthDelay(size_t dataSize, double bandwidthGBps);

    // NPU建模辅助
    uint64_t estimateNpuDmaLatency(size_t dataSize);
    uint64_t estimateNpuComputeLatency(const std::string& op, size_t dataSize);
};

// OpenSHMEM操作映射器
class OpenSHMEMOperationMapper {
public:
    static std::string mapOpenSHMEMToPtoOperation(const std::string& openshmemOperation);
    static CostModel::CorePipeType mapOpenSHMEMToPtoPipeType(const std::string& openshmemOperation);

    // OpenSHMEM操作分类
    static bool isCommunicationOperation(const std::string& operation);
    static bool isSynchronizationOperation(const std::string& operation);
    static bool isMemoryOperation(const std::string& operation);
    static bool isAtomicOperation(const std::string& operation);

    // 通信模式分析
    static std::string analyzeCommunicationPattern(const std::string& operation, int sourcePe, int targetPe);

private:
    static const std::unordered_map<std::string, std::string> operationMapping_;
    static const std::unordered_map<std::string, CostModel::CorePipeType> pipeMapping_;
};

// OpenSHMEM性能优化器 (基于PTO反馈)
class OpenSHMEMPerformanceOptimizer {
public:
    explicit OpenSHMEMPerformanceOptimizer(OpenSHMEMPtoIntegration* ptoIntegration);
    ~OpenSHMEMPerformanceOptimizer();

    // 基于PTO反馈的优化
    void optimizeCommunicationStrategy();
    void optimizeMemoryLayout();
    void optimizeSynchronizationPattern();
    void optimizeNpuResourceUtilization();

    // 自适应调整
    void adaptToWorkloadPattern(const std::string& workloadType);
    void adjustCommunicationBufferSize(size_t newSize);
    void adjustConcurrentOperationLimit(int newLimit);

    // 性能预测
    uint64_t predictLatencyWithConfig(const OpenSHMEMConfig& config);
    std::vector<std::string> suggestOptimizations();

    // 资源管理
    void balanceLoadAcrossPes();
    void optimizeNetworkTopology();
    void tuneNpuParameters();

private:
    OpenSHMEMPtoIntegration* ptoIntegration_;

    // 优化策略
    void applyCommunicationOptimization();
    void applyMemoryOptimization();
    void applySynchronizationOptimization();
    void applyNpuOptimization();

    // 分析方法
    std::vector<std::string> identifyPerformanceBottlenecks();
    double calculateOptimizationPotential(const std::string& optimizationType);
    std::unordered_map<std::string, double> analyzeResourceUtilization();
};

// OpenSHMEM网络拓扑管理器
class OpenSHMEMNetworkTopology {
public:
    explicit OpenSHMEMNetworkTopology(const OpenSHMEMConfig& config);
    ~OpenSHMEMNetworkTopology();

    // 拓扑类型
    enum class TopologyType {
        RING,
        MESH,
        TORUS,
        HYPERCUBE,
        TREE,
        BUTTERFLY
    };

    // 拓扑管理
    void setTopology(TopologyType type);
    TopologyType getCurrentTopology() const;

    // 路由计算
    int calculateHops(int sourcePe, int targetPe) const;
    std::vector<int> findShortestPath(int sourcePe, int targetPe) const;
    std::vector<int> findOptimalPath(int sourcePe, int targetPe, size_t dataSize) const;

    // 拓扑分析
    double calculateBisectionBandwidth() const;
    double calculateDiameter() const;
    std::unordered_map<int, int> analyzeConnectivity() const;

    // 动态重构
    void reconfigureTopology(const std::string& workloadPattern);
    void optimizeForCommunicationPattern(const std::string& pattern);

private:
    OpenSHMEMConfig config_;
    TopologyType currentTopology_ = TopologyType::RING;

    // 拓扑特定的计算方法
    int calculateRingHops(int sourcePe, int targetPe) const;
    int calculateMeshHops(int sourcePe, int targetPe) const;
    int calculateTorusHops(int sourcePe, int targetPe) const;
    int calculateHypercubeHops(int sourcePe, int targetPe) const;

    // 网络参数
    int networkDimension_ = 2;  // 网络维度 (for mesh/torus)
    int nodesPerDimension_ = 4; // 每维节点数
};

} // namespace openshmem
} // namespace npu

#endif // OPENSHMEM_PTO_INTEGRATION_H
