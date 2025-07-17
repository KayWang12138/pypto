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
 * \file config.cpp
 * \brief
 */

#include <algorithm>
#include "interface/utils/assert.h"
#include "interface/configs/platform_config.h"

namespace npu::tile_fwk {
AscendPlatformConfig::AscendPlatformConfig(ModelID modelId) : modelId_(modelId) {
    memoryGraph.AddPath(MemoryType::MEM_UB, MemoryType::MEM_DEVICE_DDR, true);
    memoryGraph.AddPath(MemoryType::MEM_L1, MemoryType::MEM_DEVICE_DDR, true);
    memoryGraph.AddPath(MemoryType::MEM_L1, MemoryType::MEM_L0A);
    memoryGraph.AddPath(MemoryType::MEM_L1, MemoryType::MEM_L0B);
    memoryGraph.AddPath(MemoryType::MEM_L1, MemoryType::MEM_L0A);
    memoryGraph.AddPath(MemoryType::MEM_L0C, MemoryType::MEM_DEVICE_DDR);
    SetMemoryLimitList(modelId);
}

void AscendPlatformConfig::MemoryNode::AddDest(const std::shared_ptr<MemoryNode> &to) {
    dests.insert({to->type, to});
}

void AscendPlatformConfig::MemoryGraph::AddPath(npu::tile_fwk::MemoryType from, npu::tile_fwk::MemoryType to, bool biDir) {
    if (from == to) {
        return;
    }

    std::shared_ptr<MemoryNode> fromNode = GetNode(from);
    std::shared_ptr<MemoryNode> toNode = GetNode(to);
    fromNode->AddDest(toNode);

    if (biDir) {
        toNode->AddDest(fromNode);
    }
}

std::shared_ptr<AscendPlatformConfig::MemoryNode> AscendPlatformConfig::MemoryGraph::GetNode(MemoryType type) {
    std::shared_ptr<MemoryNode> node;
    if (nodes.count(type) == 0) {
        node = std::make_shared<MemoryNode>();
        node->type = type;
        nodes.insert({type, node});
    } else {
        node = nodes[type];
    }

    return node;
}

void AscendPlatformConfig::MemoryGraph::DFS(MemoryType target, std::shared_ptr<MemoryNode> &node,
    std::vector<MemoryType> &candidate, std::vector<MemoryType> &paths) {
    for (auto &dest : node->dests) {
        if (std::find(candidate.begin(), candidate.end(), dest.first) != candidate.end()) {
            continue;
        }
        candidate.push_back(dest.first);
        if (dest.first == target) {
            if (paths.empty() || paths.size() > candidate.size()) {
                paths.clear();
                for (auto &t : candidate) {
                    paths.push_back(t);
                }
            }
        } else {
            DFS(target, dest.second, candidate, paths);
        }
        candidate.pop_back();
    }
}

void AscendPlatformConfig::MemoryGraph::FindNearestPath(
    MemoryType from, MemoryType to, std::vector<MemoryType> &paths) {
    ASSERT(nodes.count(from) != 0);
    ASSERT(nodes.count(to) != 0);

    std::vector<MemoryType> candidate = {from};

    DFS(to, nodes[from], candidate, paths);
}

std::string AscendPlatformConfig::MemoryGraph::ToString() const {
    std::stringstream ss;
    for (auto &node : nodes) {
        for (auto &dest : node.second->dests) {
            ss << MemoryTypeToString(node.first) << " --> " << MemoryTypeToString(dest.first) << "\n";
        }
    }

    return ss.str();
}

void AscendPlatformConfig::FindNearestPath(MemoryType from, MemoryType to, std::vector<MemoryType> &paths) {
    memoryGraph.FindNearestPath(from, to, paths);
}

} // namespace npu::tile_fwk
