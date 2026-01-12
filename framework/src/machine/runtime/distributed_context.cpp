/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file distributed_context.cpp
 * \brief
 */

#include "distributed_context.h"
#include <variant>
#include <memory>
#include "securec.h"
#include "interface/tileop/distributed/hccl_context.h"
#include "runtime.h"
#ifdef BUILD_WITH_CANN
#ifdef BUILD_WITH_CANN_SUB
#include <hccl/hcom.h>
#else
#include "hcom.h"
#endif

extern "C" HcclResult HcclAllocComResourceByTiling(HcclComm comm, void* stream, void* mc2Tiling, void** commContext);
#endif
namespace {  
TileOp::HcclCombinOpParam g_hostAddr[4];

#pragma pack(push, 8)
struct Mc2ServerCfg {
    uint32_t version = 0;
    uint8_t debugMode = 0;
    uint8_t sendArgIndex = 0;
    uint8_t recvArgIndex = 0;
    uint8_t commOutArgIndex = 0;
    uint8_t reserved[8] = {};
};
#pragma pack(pop)

#pragma pack(push, 8)
struct Mc2HcommCfg {
    uint8_t skipLocalRankCopy = 0;
    uint8_t skipBufferWindowCopy = 0;
    uint8_t stepSize = 0;
    char reserved[13] = {};
    char groupName[128] = {};
    char algConfig[128] = {};
    uint32_t opType = 0;
    uint32_t reduceType = 0;
};
#pragma pack(pop)

struct Mc2CommConfig {
    uint32_t version;
    uint32_t hcommCnt;
    struct Mc2ServerCfg serverCfg;
    struct Mc2HcommCfg hcommCfg;
};

template<typename Mc2CommConfig>
class Mc2CommConfigMgrBase {
public:
    Mc2CommConfigMgrBase() {}
    virtual ~Mc2CommConfigMgrBase() {}
    virtual int32_t MakeMc2TilingStruct(const std::string& groupName) = 0;
    Mc2CommConfig Mc2CommConfig_;
private:
    std::string groupName_{};
};

class Mc2CommConfigMgr : public Mc2CommConfigMgrBase<Mc2CommConfig>  {
public:
    Mc2CommConfigMgr() {}
    ~Mc2CommConfigMgr() {}
    int32_t MakeMc2TilingStruct(const std::string& groupName) override
    {
        (void)memset_s(&Mc2CommConfig_, sizeof(Mc2CommConfig_), 0, sizeof(Mc2CommConfig_));
        constexpr uint32_t version = 2;
        constexpr uint32_t hcommCnt = 1;
        constexpr uint32_t opTypeAllToAll = 6; // numeric representation of AlltoAll
        const char *algConfig = "AllGather=level0:ring";
        constexpr uint32_t arraySize = 128;

        Mc2CommConfig_.version = version;
        Mc2CommConfig_.hcommCnt = hcommCnt;
        Mc2CommConfig_.hcommCfg.skipLocalRankCopy = 0;
        Mc2CommConfig_.hcommCfg.skipBufferWindowCopy = 0;
        Mc2CommConfig_.hcommCfg.stepSize = 0;
        Mc2CommConfig_.hcommCfg.opType = opTypeAllToAll;
        auto ret = strcpy_s(Mc2CommConfig_.hcommCfg.groupName, arraySize, groupName.c_str());
        if (ret != 0) {
            return -1;
        }
        ret = strcpy_s(Mc2CommConfig_.hcommCfg.algConfig, arraySize, algConfig);
        if (ret != 0) {
            return -1;
        }
        return 0;
    }
};

using ConfigMgrCommon = std::variant<
    std::shared_ptr<Mc2CommConfigMgr>
>;
}

namespace npu::tile_fwk::dynamic {
std::vector<uint64_t> DistributedContext::GetHcclContextToHost(const std::vector<std::string> &groupNames) {
    std::vector<uint64_t> devAddrs = GetHcclContext(groupNames);
#ifdef BUILD_WITH_CANN
    std::vector<uint64_t> hosts;
    for (size_t i = 0; i < groupNames.size(); i++) {
        (void)rtMemcpy(&g_hostAddr[i], sizeof(g_hostAddr[i]), (uint8_t *)devAddrs[0], sizeof(g_hostAddr[i]),
            RT_MEMCPY_DEVICE_TO_HOST);
        hosts.push_back((uint64_t)(&g_hostAddr[i]));
    }
    return hosts;
#else
    return {};
#endif
}

std::vector<uint64_t> DistributedContext::GetHcclContext([[maybe_unused]] const std::vector<std::string> &groupNames)
{
#ifdef BUILD_WITH_CANN
    ConfigMgrCommon commConfigMgr;
    commConfigMgr = std::make_shared<Mc2CommConfigMgr>();
    std::vector<uint64_t> hcclContext(groupNames.size(), 0);
    for (size_t groupIndex = 0; groupIndex < groupNames.size(); ++groupIndex) {
        auto groupName = groupNames[groupIndex];
        HcclComm commHandle = nullptr;
        HcclResult ret = HcomGetCommHandleByGroup(groupName.c_str(), &commHandle);
        ASSERT(ret == 0);
        bool makeTilingSuccess = std::visit(
            [groupName](const auto& cfgMgr) -> bool {
                return cfgMgr->MakeMc2TilingStruct(groupName) == 0;},
            commConfigMgr);
        ASSERT(makeTilingSuccess);
        ret = std::visit(
            [&](const auto& cfgMgr) -> HcclResult {
                return HcclAllocComResourceByTiling(commHandle, machine::GetRA()->GetStream(),
                    &(cfgMgr->Mc2CommConfig_),reinterpret_cast<void **>(&hcclContext[groupIndex]));
            },
            commConfigMgr);
        ASSERT((ret == 0) && (hcclContext[groupIndex] != 0UL));
        ALOG_INFO_F("groupIndex=%u, groupName=%s, hcclContext=%lu", groupIndex, groupName.c_str(),
            hcclContext[groupIndex]);
    }
    return hcclContext;
#else
    return {};
#endif
}
} // namespace npu::tile_fwk::dynamic