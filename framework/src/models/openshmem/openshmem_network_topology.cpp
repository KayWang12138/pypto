/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "openshmem_pto_integration.h"
#include <cmath>
#include <algorithm>

namespace npu {
namespace openshmem {

// OpenSHMEM网络拓扑管理器实现
OpenSHMEMNetworkTopology::OpenSHMEMNetworkTopology(const OpenSHMEMConfig& config)
    : config_(config) {
    // 根据PE数量自动确定网络维度
    int numPes = config.numPes;
    networkDimension_ = static_cast<int>(std::ceil(std::log2(numPes)));
    nodesPerDimension_ = static_cast<int>(std::pow(numPes, 1.0 / networkDimension_));
}

OpenSHMEMNetworkTopology::~OpenSHMEMNetworkTopology() {
}

void OpenSHMEMNetworkTopology::setTopology(TopologyType type) {
    currentTopology_ = type;
    std::cout << "[OpenSHMEM-Topology] Set topology to ";

    switch (type) {
        case TopologyType::RING:
            std::cout << "RING" << std::endl;
            break;
        case TopologyType::MESH:
            std::cout << "MESH" << std::endl;
            break;
        case TopologyType::TORUS:
            std::cout << "TORUS" << std::endl;
            break;
        case TopologyType::HYPERCUBE:
            std::cout << "HYPERCUBE" << std::endl;
            break;
        case TopologyType::TREE:
            std::cout << "TREE" << std::endl;
            break;
        case TopologyType::BUTTERFLY:
            std::cout << "BUTTERFLY" << std::endl;
            break;
    }
}

OpenSHMEMNetworkTopology::TopologyType OpenSHMEMNetworkTopology::getCurrentTopology() const {
    return currentTopology_;
}

int OpenSHMEMNetworkTopology::calculateHops(int sourcePe, int targetPe) const {
    switch (currentTopology_) {
        case TopologyType::RING:
            return calculateRingHops(sourcePe, targetPe);
        case TopologyType::MESH:
            return calculateMeshHops(sourcePe, targetPe);
        case TopologyType::TORUS:
            return calculateTorusHops(sourcePe, targetPe);
        case TopologyType::HYPERCUBE:
            return calculateHypercubeHops(sourcePe, targetPe);
        case TopologyType::TREE:
            // 简化的树拓扑
            return static_cast<int>(std::log2(std::max(sourcePe, targetPe) + 1));
        case TopologyType::BUTTERFLY:
            // 蝶形网络
            return static_cast<int>(std::log2(config_.numPes)) / 2;
        default:
            return std::abs(targetPe - sourcePe);
    }
}

std::vector<int> OpenSHMEMNetworkTopology::findShortestPath(int sourcePe, int targetPe) const {
    std::vector<int> path;

    if (sourcePe == targetPe) {
        path.push_back(sourcePe);
        return path;
    }

    switch (currentTopology_) {
        case TopologyType::RING: {
            // 环形拓扑的路径
            int clockwise = (targetPe - sourcePe + config_.numPes) % config_.numPes;
            int counterclockwise = (sourcePe - targetPe + config_.numPes) % config_.numPes;

            if (clockwise <= counterclockwise) {
                for (int i = 0; i <= clockwise; ++i) {
                    path.push_back((sourcePe + i) % config_.numPes);
                }
            } else {
                for (int i = 0; i <= counterclockwise; ++i) {
                    path.push_back((sourcePe - i + config_.numPes) % config_.numPes);
                }
            }
            break;
        }
        case TopologyType::MESH: {
            // 网格拓扑的路径 (简化实现)
            path.push_back(sourcePe);
            path.push_back(targetPe);
            break;
        }
        default:
            // 默认直接路径
            path.push_back(sourcePe);
            path.push_back(targetPe);
            break;
    }

    return path;
}

std::vector<int> OpenSHMEMNetworkTopology::findOptimalPath(int sourcePe, int targetPe, size_t dataSize) const {
    // 基于数据大小的路径优化
    // 大数据可能需要不同的路由策略

    if (dataSize > 1024 * 1024) { // > 1MB
        // 大数据可能需要避免拥塞路径
        return findShortestPath(sourcePe, targetPe);
    } else {
        // 小数据可以使用最短路径
        return findShortestPath(sourcePe, targetPe);
    }
}

double OpenSHMEMNetworkTopology::calculateBisectionBandwidth() const {
    // 计算二分带宽
    switch (currentTopology_) {
        case TopologyType::RING:
            return 2.0; // 环网的二分带宽很低
        case TopologyType::MESH: {
            int gridSize = static_cast<int>(std::sqrt(config_.numPes));
            return gridSize * 1.0; // 网格的二分带宽
        }
        case TopologyType::TORUS:
            return config_.numPes / 2.0; // 环面的二分带宽
        case TopologyType::HYPERCUBE:
            return config_.numPes / 2.0; // 超立方体的二分带宽
        case TopologyType::TREE:
            return 1.0; // 树的二分带宽很低
        case TopologyType::BUTTERFLY:
            return config_.numPes / 4.0; // 蝶形网络的二分带宽
        default:
            return 1.0;
    }
}

double OpenSHMEMNetworkTopology::calculateDiameter() const {
    // 计算网络直径 (最长最短路径)
    switch (currentTopology_) {
        case TopologyType::RING:
            return config_.numPes / 2.0;
        case TopologyType::MESH: {
            int gridSize = static_cast<int>(std::sqrt(config_.numPes));
            return 2 * (gridSize - 1);
        }
        case TopologyType::TORUS: {
            int gridSize = static_cast<int>(std::sqrt(config_.numPes));
            return gridSize;
        }
        case TopologyType::HYPERCUBE:
            return std::log2(config_.numPes);
        case TopologyType::TREE:
            return 2 * std::log2(config_.numPes);
        case TopologyType::BUTTERFLY:
            return std::log2(config_.numPes);
        default:
            return config_.numPes - 1;
    }
}

std::unordered_map<int, int> OpenSHMEMNetworkTopology::analyzeConnectivity() const {
    std::unordered_map<int, int> connectivity;

    // 计算每个节点的连接度
    for (int pe = 0; pe < config_.numPes; ++pe) {
        switch (currentTopology_) {
            case TopologyType::RING:
                connectivity[pe] = 2; // 每个节点连接2个邻居
                break;
            case TopologyType::MESH: {
                int gridSize = static_cast<int>(std::sqrt(config_.numPes));
                int row = pe / gridSize;
                int col = pe % gridSize;
                int degree = 0;
                if (row > 0) degree++;      // 上
                if (row < gridSize - 1) degree++; // 下
                if (col > 0) degree++;      // 左
                if (col < gridSize - 1) degree++; // 右
                connectivity[pe] = degree;
                break;
            }
            case TopologyType::TORUS: {
                connectivity[pe] = 4; // 环面每个节点连接4个邻居
                break;
            }
            case TopologyType::HYPERCUBE:
                connectivity[pe] = static_cast<int>(std::log2(config_.numPes));
                break;
            case TopologyType::TREE:
                // 简化的树连接度计算
                if (pe == 0) {
                    connectivity[pe] = 2; // 根节点
                } else {
                    connectivity[pe] = 3; // 其他节点 (父+2子)
                }
                break;
            case TopologyType::BUTTERFLY:
                connectivity[pe] = static_cast<int>(std::log2(config_.numPes)) + 1;
                break;
            default:
                connectivity[pe] = 1;
                break;
        }
    }

    return connectivity;
}

void OpenSHMEMNetworkTopology::reconfigureTopology(const std::string& workloadPattern) {
    std::cout << "[OpenSHMEM-Topology] Reconfiguring topology for workload: " << workloadPattern << std::endl;

    if (workloadPattern == "allreduce_heavy") {
        setTopology(TopologyType::TREE); // 树拓扑适合规约操作
    } else if (workloadPattern == "point_to_point") {
        setTopology(TopologyType::MESH); // 网格适合点对点通信
    } else if (workloadPattern == "broadcast_heavy") {
        setTopology(TopologyType::TREE); // 树拓扑也适合广播
    } else {
        setTopology(TopologyType::RING); // 默认环形拓扑
    }
}

void OpenSHMEMNetworkTopology::optimizeForCommunicationPattern(const std::string& pattern) {
    std::cout << "[OpenSHMEM-Topology] Optimizing for communication pattern: " << pattern << std::endl;

    if (pattern == "nearest_neighbor") {
        setTopology(TopologyType::MESH);
    } else if (pattern == "all_to_all") {
        setTopology(TopologyType::HYPERCUBE);
    } else if (pattern == "reduce_scatter") {
        setTopology(TopologyType::TREE);
    }
}

// 拓扑特定的跳数计算方法
int OpenSHMEMNetworkTopology::calculateRingHops(int sourcePe, int targetPe) const {
    int diff = std::abs(targetPe - sourcePe);
    return std::min(diff, config_.numPes - diff);
}

int OpenSHMEMNetworkTopology::calculateMeshHops(int sourcePe, int targetPe) const {
    int gridSize = static_cast<int>(std::sqrt(config_.numPes));

    int sourceRow = sourcePe / gridSize;
    int sourceCol = sourcePe % gridSize;
    int targetRow = targetPe / gridSize;
    int targetCol = targetPe % gridSize;

    return std::abs(targetRow - sourceRow) + std::abs(targetCol - sourceCol);
}

int OpenSHMEMNetworkTopology::calculateTorusHops(int sourcePe, int targetPe) const {
    int gridSize = static_cast<int>(std::sqrt(config_.numPes));

    int sourceRow = sourcePe / gridSize;
    int sourceCol = sourcePe % gridSize;
    int targetRow = targetPe / gridSize;
    int targetCol = targetPe % gridSize;

    // 环面拓扑可以绕过
    int rowDiff = std::min(std::abs(targetRow - sourceRow), gridSize - std::abs(targetRow - sourceRow));
    int colDiff = std::min(std::abs(targetCol - sourceCol), gridSize - std::abs(targetCol - sourceCol));

    return rowDiff + colDiff;
}

int OpenSHMEMNetworkTopology::calculateHypercubeHops(int sourcePe, int targetPe) const {
    // 超立方体中的汉明距离
    int xor_result = sourcePe ^ targetPe;
    int hops = 0;
    while (xor_result > 0) {
        if (xor_result & 1) hops++;
        xor_result >>= 1;
    }
    return hops;
}

} // namespace openshmem
} // namespace npu
