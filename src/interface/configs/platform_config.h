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
 * \file platform_config.h
 * \brief
 */

#pragma once

#include <cstdint>
#include <memory>
#include <vector>
#include <map>
#include <string>
#include <set>
#include <unordered_set>
#include <sstream>

#include "common/data_type.h"

namespace npu::tile_fwk {
const size_t K_ONE_KB = 1024L;
constexpr int AICPU_CORE_NUM_910B = 6;
constexpr 
inline size_t KB(unsigned value) {
    return static_cast<size_t>(value * K_ONE_KB);
}

inline size_t MB(unsigned value) {
    return static_cast<size_t>(value * K_ONE_KB * K_ONE_KB);
}

inline size_t GB(unsigned value) {
    return static_cast<size_t>(value * K_ONE_KB * K_ONE_KB * K_ONE_KB);
}

enum class ModelID { ASCEND_910B };

enum class DPlatform {
    ASCEND_910B1,
    ASCEND_910B2,
    ASCEND_910B3,
    ASCEND_910B4,
};

struct DPlatformInfo {
    int c0ByteSize;
    int aICoreNum;
    int vectorCoreNum;
    int l0BSize;
    int l1Size;
    int ubSize;
    std::string coreTypeName;
};

const std::unordered_map<DPlatform, DPlatformInfo>  dPlatformInfos = 
    {{DPlatform::ASCEND_910B1, {32, 25, 50, 64 * 1024, 512 * 1024, 192 * 1024, "AIV"}},
     {DPlatform::ASCEND_910B2, {32, 24, 48, 64 * 1024, 512 * 1024, 192 * 1024, "AIC"}},
     {DPlatform::ASCEND_910B3, {32, 20, 40, 64 * 1024, 512 * 1024, 192 * 1024, "MIX"}},
     {DPlatform::ASCEND_910B4, {32, 20, 40, 64 * 1024, 512 * 1024, 192 * 1024, "GMATOMIC"}}};

class AscendPlatformConfig {
public:
    AscendPlatformConfig(ModelID modelId = ModelID::ASCEND_910B);
    void FindNearestPath(MemoryType from, MemoryType to, std::vector<MemoryType> &paths);

    void SetMemoryLimitList(ModelID modelId) {
        const std::map<ModelID, std::vector<size_t>> memorySizeLimitList = {
            /*ModelID  {MEM_UB,  MEM_L1,  MEM_L0A,MEM_L0B,MEM_L0C, MEM_BT, MEM_L2,  MEM_L3, MEM_DEVICE_DDR, MEM_HOST1, MEM_FAR1,
            MEM_FAR2}*/
            {ModelID::ASCEND_910B,
             {KB(192), KB(512), KB(64), KB(64), KB(128), KB(0), KB(0), KB(0), KB(0), KB(0), KB(0), KB(0), KB(0), MB(192), MB(64), GB(32), GB(1024), GB(1024), GB(1024)}}
        };
        if (memorySizeLimitList.count(modelId) != 0) {
            memorySizeLimit = memorySizeLimitList.at(modelId);
        }
    }

    [[nodiscard]] size_t GetMemoryLimit(MemoryType memoryType) const {
        ASSERT(static_cast<size_t>(memoryType) < memorySizeLimit.size());

        return memorySizeLimit[static_cast<size_t>(memoryType)];
    }

    void SetPlatform(DPlatform value) { platform_ = value; }

    DPlatform GetPlatform() { return platform_; }

    int GetC0ByteSize() const {
        auto iter = dPlatformInfos.find(platform_);
        ASSERT(iter != dPlatformInfos.end()) << "Can not support this platform";
        return iter->second.c0ByteSize;
    }

    int GetAICoreNum() const {
        auto iter = dPlatformInfos.find(platform_);
        ASSERT(iter != dPlatformInfos.end()) << "Can not support this platform";
        return iter->second.aICoreNum;
    }

    int GetVectorCoreNum() const {
        auto iter = dPlatformInfos.find(platform_);
        ASSERT(iter != dPlatformInfos.end()) << "Can not support this platform";
        return iter->second.vectorCoreNum;
    }

    int GetL0BSize() const {
        auto iter = dPlatformInfos.find(platform_);
        ASSERT(iter != dPlatformInfos.end()) << "Can not support this platform";
        return iter->second.l0BSize;
    }

    int GetL1Size() const {
        auto iter = dPlatformInfos.find(platform_);
        ASSERT(iter != dPlatformInfos.end()) << "Can not support this platform";
        return iter->second.l1Size;
    }

    int GetUBSize() const {
        auto iter = dPlatformInfos.find(platform_);
        ASSERT(iter != dPlatformInfos.end()) << "Can not support this platform";
        return iter->second.ubSize;
    }

    int GetAICpuCoreNum() const { return aiCpuCoreNum; }

    bool IsPlatformA2() { return platform_ >= DPlatform::ASCEND_910B1 && platform_ <= DPlatform::ASCEND_910B4; }

protected:
    std::vector<size_t> memorySizeLimit_;
    ModelID modelId_;
    DPlatform platform_{DPlatform::ASCEND_910B2};
    int aiCpuCoreNum{AICPU_CORE_NUM_910B};
    
    // Different device(UB/Cache/DDR/etc.) memory size in bytes
    std::vector<size_t> memorySizeLimit;

    struct MemoryNode {
        MemoryType type;
        std::map<MemoryType, std::shared_ptr<MemoryNode>> dests;

        void AddDest(const std::shared_ptr<MemoryNode> &to);
    };

    struct MemoryGraph {
        std::map<MemoryType, std::shared_ptr<MemoryNode>> nodes;

        void AddPath(MemoryType from, MemoryType to, bool biDir = false);
        std::shared_ptr<MemoryNode> GetNode(MemoryType type);
        void DFS(MemoryType target, std::shared_ptr<MemoryNode> &node, std::vector<MemoryType> &candidate,
            std::vector<MemoryType> &paths);
        void FindNearestPath(MemoryType from, MemoryType to, std::vector<MemoryType> &paths);

        std::string ToString() const;
    };

    MemoryGraph memoryGraph;
};

class Ascend910BConfig : public AscendPlatformConfig {
public:
    Ascend910BConfig() : AscendPlatformConfig(ModelID::ASCEND_910B) {
        memorySizeLimit_ = {
            KB(192), KB(512), KB(64), KB(64), KB(128), MB(192), MB(64), GB(32), GB(1024), GB(1024), GB(1024)};
    }
};
} // namespace npu::tile_fwk
