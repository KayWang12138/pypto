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
 * \file platform_config.cpp
 * \brief
 */
#include "passes/pass_config/platform_config.h"
#include <algorithm>
#include <map>
#include "interface/utils/log.h"
#include "interface/utils/file_utils.h"
#include "passes/pass_config/json_node_paser.h"
namespace npu{
namespace tile_fwk {
const std::string platformConfigEnvName = "PLATFORM_CONFIG_PATH";

const std::string MEM_UB_STR = "MEM_UB";
const std::string MEM_L1_STR = "MEM_L1";
const std::string MEM_L0A_STR = "MEM_L0A";
const std::string MEM_L0B_STR = "MEM_L0B";
const std::string MEM_L0C_STR = "MEM_L0C";
const std::string MEM_FIX_STR = "MEM_FIX";
const std::string MEM_FIX_QUANT_PRE_STR = "MEM_FIX_QUANT_PRE";
const std::string MEM_FIX_RELU_PRE_STR = "MEM_FIX_RELU_PRE";
const std::string MEM_FIX_RELU_POST_STR = "MEM_FIX_RELU_POST";
const std::string MEM_FIX_QUANT_POST_STR = "MEM_FIX_QUANT_POST";
const std::string MEM_FIX_ELT_ANTIQ_STR = "MEM_FIX_ELT_ANTIQ";
const std::string MEM_FIX_MTE2_ANTIQ_STR = "MEM_FIX_MTE2_ANTIQ";
const std::string MEM_BT_STR = "MEM_BT";
const std::string MEM_L2_STR = "MEM_L2";
const std::string MEM_L3_STR = "MEM_L3";
const std::string MEM_DEVICE_DDR_STR = "MEM_DEVICE_DDR";
const std::string MEM_HOST1_STR = "MEM_HOST1";
const std::string MEM_FAR1_STR = "MEM_FAR1";
const std::string MEM_FAR2_STR = "MEM_FAR2";
const std::string MEM_WORKSPACE_STR = "MEM_WORKSPACE";
const std::string MEM_VECTOR_REG_STR = "MEM_VECTOR_REG";

const std::string PATHS_STR = "PATHS";
const std::string MEM_LIMITS_STR = "MEMORY_LIMITS";
const std::string CORE_NUM_STR = "CORE_NUM";
const std::string AICORE_STR = "AICORE";
const std::string CUBE_CORE_STR = "CUBE_CORE";
const std::string VECTOR_CORE_STR = "VECTOR_CORE";

static std::unordered_map<std::string, MemoryType> jsonNodeToMemoryType = {
    {MEM_UB_STR, MemoryType::MEM_UB},
    {MEM_L1_STR, MemoryType::MEM_L1},
    {MEM_L0A_STR, MemoryType::MEM_L0A},
    {MEM_L0B_STR, MemoryType::MEM_L0B},
    {MEM_L0C_STR, MemoryType::MEM_L0C},
    {MEM_FIX_STR, MemoryType::MEM_FIX},
    {MEM_FIX_QUANT_PRE_STR, MemoryType::MEM_FIX_QUANT_PRE},
    {MEM_FIX_RELU_PRE_STR, MemoryType::MEM_FIX_RELU_PRE},
    {MEM_FIX_RELU_POST_STR, MemoryType::MEM_FIX_RELU_POST},
    {MEM_FIX_QUANT_POST_STR, MemoryType::MEM_FIX_QUANT_POST},
    {MEM_FIX_ELT_ANTIQ_STR, MemoryType::MEM_FIX_ELT_ANTIQ},
    {MEM_FIX_MTE2_ANTIQ_STR, MemoryType::MEM_FIX_MTE2_ANTIQ},
    {MEM_BT_STR, MemoryType::MEM_BT},
    {MEM_L2_STR, MemoryType::MEM_L2},
    {MEM_L3_STR, MemoryType::MEM_L3},
    {MEM_DEVICE_DDR_STR, MemoryType::MEM_DEVICE_DDR},
    {MEM_HOST1_STR, MemoryType::MEM_HOST1},
    {MEM_FAR1_STR, MemoryType::MEM_FAR1},
    {MEM_FAR2_STR, MemoryType::MEM_FAR2},
    {MEM_WORKSPACE_STR, MemoryType::MEM_WORKSPACE},
    {MEM_VECTOR_REG_STR, MemoryType::MEM_VECTOR_REG}
};

static std::unordered_map<std::string, NpuCoreType> jsonNodeToCoreType = {
    {AICORE_STR, NpuCoreType::AICORE},
    {CUBE_CORE_STR, NpuCoreType::AICORE},
    {VECTOR_CORE_STR, NpuCoreType::VECTORCORE}
};

inline std::string PlatformIdToString(DPlatform platformId) {
    std::unordered_map<DPlatform, std::string> mappings = {
        {DPlatform::ASCEND_910B1, "ASCEND_910B1"},
        {DPlatform::ASCEND_910B2, "ASCEND_910B2"},
        {DPlatform::ASCEND_910B3, "ASCEND_910B3"},
        {DPlatform::ASCEND_910B4, "ASCEND_910B4"},
    };
    if (mappings.count(platformId)) {
        return mappings[platformId];
    }
    std::string res;
    return res;
}

inline Status AddCoreNum(const nlohmann::json *node, std::unordered_map<NpuCoreType, size_t>& corenum, std::string &platformIdStr) {
    if (node == nullptr) {
        ALOG_WARN_F("Platform %s built in config doesn't contain corenums, can add manually.");
        return SUCCESS;
    }
    for (auto &[coreTypeKey, coreNum] : (*node).get<std::map<std::string, size_t>>()) {
        auto coreType = jsonNodeToCoreType.find(coreTypeKey);
        if (coreType != jsonNodeToCoreType.end()) {
            corenum[coreType->second] = coreNum;
        } else {
            ALOG_WARN_F("Core type %s of Platform %s is not recognized.", coreTypeKey.c_str(), platformIdStr.c_str());
        }
    }
    return SUCCESS;
}

Status PlatformConfig::SetMemoryPath(std::vector<std::string>& pathDesc, std::string &platformIdStr) {
    if (pathDesc.size() != 2U) {
        ALOG_ERROR_F("Platform %s path is not legal.", platformIdStr.c_str());
        return FAILED;
    }
    auto from = jsonNodeToMemoryType.find(pathDesc[0]);
    auto to = jsonNodeToMemoryType.find(pathDesc[1]);
    if (from == jsonNodeToMemoryType.end()) {
        ALOG_ERROR_F("Memory Type %s of Platform %s is not recognized in memory path.", pathDesc[0].c_str(), platformIdStr.c_str());
        return FAILED;
    }
    if (to == jsonNodeToMemoryType.end()) {
        ALOG_ERROR_F("Memory Type %s of Platform %s is not recognized in memory path.", pathDesc[1].c_str(), platformIdStr.c_str());
        return FAILED;
    }
    memoryGraph_.AddPath(from->second, to->second);
    return SUCCESS;
}

Status PlatformConfig::InitPlatformConfig(DPlatform platformId) {
    platformId_ = platformId;
    auto platformIdStr = PlatformIdToString(platformId);
    std::string platformConfigPath = GetEnvVar(platformConfigEnvName);
    std::string builtinPlatformConfigPath = GetCurrentSharedLibPath() + "/../conf/tile_fwk_platform_info.json";
    JsonNodeParser jsonParser;
    if (platformConfigPath.size() > 0 && jsonParser.Initialize(platformConfigPath) != SUCCESS) {
        ALOG_ERROR_F("Platform %s config file %s is not available, please set %s properly.",
            platformIdStr.c_str(), platformConfigPath.c_str(), platformConfigEnvName.c_str());
        return FAILED;
    }
    if (platformConfigPath.size() == 0 && jsonParser.Initialize(builtinPlatformConfigPath) != SUCCESS) {
        ALOG_ERROR_F("Platform %s config is not available.", platformIdStr.c_str());
        return FAILED;
    }
    if (auto root = jsonParser.GetRootNode()) {
        auto memLimits = jsonParser.GetJsonInnerNode(*root, {platformIdStr, MEM_LIMITS_STR});
        if (memLimits == nullptr) {
            ALOG_ERROR_F("Platform %s config doesn't contain memory limits, please add limits manually.", platformIdStr.c_str());
            return FAILED;
        }
        for (auto &[memDesc, memLimit] : (*memLimits).get<std::map<std::string, size_t>>()) {
            auto mem = jsonNodeToMemoryType.find(memDesc);
            if (mem != jsonNodeToMemoryType.end()) {
                SetMemoryLimit(mem->second, memLimit);
            } else {
                ALOG_ERROR_F("Memory Type %s of Platform %s is not recognized in memory limit.", memDesc.c_str(), platformIdStr.c_str());
                return FAILED;
            }
        }
        if (AddCoreNum(jsonParser.GetJsonInnerNode(*root, {platformIdStr, CORE_NUM_STR}), coreNum_, platformIdStr) != SUCCESS) {
            ALOG_ERROR_F("Failed to set core number.");
            return FAILED;
        }
        auto paths = jsonParser.GetJsonInnerNode(*root, {platformIdStr, PATHS_STR});
        if (paths == nullptr) {
            ALOG_ERROR_F("Platform %s config doesn't contain memory path, please add path manually.", platformIdStr.c_str());
            return FAILED;
        }
        for(auto &pathDesc : (*paths).get<std::vector<std::vector<std::string>>>()) {
            if (SetMemoryPath(pathDesc, platformIdStr) != SUCCESS) {
                ALOG_ERROR_F("Failed to set memory path.");
                return FAILED;
            }
        }
    }
    isInit_ = true;
    return SUCCESS;
}

void PlatformConfig::SetMemoryLimit(MemoryType memoryType, size_t memSize) {
    memoryLimits_[memoryType] = memSize;
}

void PlatformConfig::ResetAndAddPath(const std::unordered_map<MemoryType, MemoryType>& memPaths) {
  memoryGraph_.Reset();
  for (auto &[from, to] : memPaths) {
      memoryGraph_.AddPath(from, to);
  }
}

size_t PlatformConfig::GetMemoryLimit(MemoryType memtype) const {
    auto memSizeIter = memoryLimits_.find(memtype);
    if (memSizeIter == memoryLimits_.end()) {
        return 0;
    }
    return memSizeIter->second;
}

size_t PlatformConfig::GetCoreNum(NpuCoreType coretype) const {
    auto coreIter = coreNum_.find(coretype);
    if (coreIter == coreNum_.end()) {
        return 1;
    }
    return coreIter->second;
}

void PlatformConfig::MemoryNode::AddDest(const std::shared_ptr<MemoryNode> &to) {
    dests.insert({to->type});
}

void PlatformConfig::MemoryGraph::AddPath(MemoryType from, MemoryType to) {
    if (from == to) {
        return;
    }
    std::shared_ptr<MemoryNode> fromNode = GetNode(from);
    std::shared_ptr<MemoryNode> toNode = GetNode(to);
    if ((fromNode == nullptr) || (toNode == nullptr)) {
        ALOG_WARN_F("Check from [%s] and to [%s] memtype is right or not.", MemoryTypeToString(from).c_str(),
            MemoryTypeToString(to).c_str());
        return;
    }
    fromNode->AddDest(toNode);
}

std::shared_ptr<PlatformConfig::MemoryNode> PlatformConfig::MemoryGraph::GetNode(MemoryType type) {
    std::shared_ptr<MemoryNode> node;
    if (nodes.count(type) != 0) {
        node = nodes[type];
        return node;
    }
    node = std::make_shared<MemoryNode>();
    if (node == nullptr) {
        ALOG_WARN_F("Create memory node failed.");
        return nullptr;
    }
    node->type = type;
    nodes.insert({type, node});
    return node;
}

void PlatformConfig::MemoryGraph::DFS(MemoryType target, const std::shared_ptr<MemoryNode> &node,
    std::vector<MemoryType> &candidate, std::vector<MemoryType> &paths) const {
    for (auto &dest : node->dests) {
        if (std::find(candidate.begin(), candidate.end(), dest) != candidate.end()) {
            continue;
        }
        candidate.push_back(dest);
        if (dest != target) {
            DFS(target, nodes.at(dest), candidate, paths);
            candidate.pop_back();
            continue;
        }
        if ((!paths.empty()) && (paths.size() <= candidate.size())) {
            candidate.pop_back();
            continue;
        }
        paths.clear();
        for (auto &t : candidate) {
            paths.push_back(t);
        }
        candidate.pop_back();
    }
}

Status PlatformConfig::MemoryGraph::FindNearestPath(
    MemoryType from, MemoryType to, std::vector<MemoryType> &paths) const {
    if (nodes.count(from) == 0) {
        ALOG_ERROR_F("Current platform doesn't support src mem: %s.", MemoryTypeToString(from).c_str());
        return FAILED;
    }
    if (nodes.count(to) == 0) {
        ALOG_ERROR_F("Current platform doesn't support dst mem: %s.", MemoryTypeToString(to).c_str());
        return FAILED;
    }
    std::vector<MemoryType> candidate = {from};
    paths.clear();
    const auto it = nodes.find(from);
    DFS(to, it->second, candidate, paths);
    return SUCCESS;
}

std::string PlatformConfig::MemoryGraph::ToString() const {
    std::stringstream ss;
    for (auto &node : nodes) {
        for (auto &dest : node.second->dests) {
            ss << MemoryTypeToString(node.first) << " --> " << MemoryTypeToString(dest) << "\n";
        }
    }
    return ss.str();
}

void PlatformConfig::MemoryGraph::Reset() {
    nodes.clear();
}

void PlatformConfig::FindNearestPath(MemoryType from, MemoryType to, std::vector<MemoryType> &paths) const {
    auto res = memoryGraph_.FindNearestPath(from, to, paths);
    if (res == SUCCESS) {
        return;
    }
    paths.clear();
    return;
}
} // namespace tile_fwk
} // namespace npu
