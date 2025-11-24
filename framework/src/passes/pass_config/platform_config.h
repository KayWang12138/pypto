/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file pass_dfx_config.h
 * \brief
 */

#ifndef PASSES_PASS_PLATFORM_CONFIG_H_
#define PASSES_PASS_PLATFORM_CONFIG_H_
#include <string>
#include <map>
#include "tilefwk/data_type.h"
#include "interface/utils/common.h"
#include "interface/configs/config_manager.h"

namespace npu {
namespace tile_fwk {
enum class NpuCoreType {
    CUBECORE = 0,
    VECTORCORE,
    AICORE
};

class PlatformConfig {
  public:
    PlatformConfig() {}
    void FindNearestPath(MemoryType from, MemoryType to, std::vector<MemoryType> &paths) const;
    Status InitPlatformConfig(DPlatform platformId);
    size_t GetMemoryLimit(MemoryType memtype) const;
    size_t GetCoreNum(NpuCoreType coreType) const;
    void SetMemoryLimit(MemoryType memoryType, size_t memSize);
    Status SetMemoryPath(std::vector<std::string>& pathDesc, std::string &platformIdStr);
    void ResetAndAddPath(const std::unordered_map<MemoryType, MemoryType>& memPaths);
    bool IsInited() const { return isInit_; }
  private:
    DPlatform platformId_;
    struct MemoryNode {
        MemoryType type;
        std::set<MemoryType> dests;
        void AddDest(const std::shared_ptr<MemoryNode> &to);
    };
    struct MemoryGraph {
        std::map<MemoryType, std::shared_ptr<MemoryNode>> nodes;
        void AddPath(MemoryType from, MemoryType to);
        std::shared_ptr<MemoryNode> GetNode(MemoryType type);
        void DFS(MemoryType target, const std::shared_ptr<MemoryNode> &node, std::vector<MemoryType> &candidate,
            std::vector<MemoryType> &paths) const;
        Status FindNearestPath(MemoryType from, MemoryType to, std::vector<MemoryType> &paths) const;
        std::string ToString() const;
        void Reset();
    };
    MemoryGraph memoryGraph_;
    std::unordered_map<MemoryType, size_t> memoryLimits_;
    std::unordered_map<NpuCoreType, size_t> coreNum_;
    bool isInit_{false};
};
} // namespace tile_fwk
} // namespace npu
#endif