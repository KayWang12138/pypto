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
 * \file distributed_context.cpp
 * \brief
 */

#include "distributed_context.h"
#include <variant>
#include <memory>
#include "securec.h"
#include "interface/tileop/distributed/hccl_context.h"
#include "tilefwk/platform.h"
#include "hccl/hccl_types.h"
#include "machine/runtime/runtime.h"
#ifdef BUILD_WITH_CANN
#ifdef BUILD_WITH_CANN_SUB
#include<hccl/hcom.h>
#else
#include "hcom.h"
#endif

extern "C" HcclResult HcclAllocComResourceByTiling(HcclComm comm, void* stream, void* mc2Tiling, void** commContext);
#endif
namespace {  
TileOp::HcclCombinOpParam g_hostAddrV1[4];
TileOp::HcclOpResParam g_hostAddrV2[4];

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

struct Mc2CommConfigV1 {
    uint32_t version;
    uint32_t hcommCnt;
    struct Mc2ServerCfg serverCfg;
    struct Mc2HcommCfg hcommCfg;
};

constexpr uint32_t INIT_TILING_VERSION = 100U;
constexpr uint32_t MAX_CC_TILING_NUM = 8U;
struct Mc2InitTilingInner {
    uint32_t version;
    uint32_t mc2HcommCnt;
    uint32_t offset[MAX_CC_TILING_NUM];
    uint8_t debugMode;
    uint8_t preparePosition;
    uint16_t queueNum;
    uint16_t commBlockNum;
    uint8_t devType;
    char reserved[17];
};

constexpr uint32_t GROUP_NAME_SIZE = 128U;
constexpr uint32_t ALG_CONFIG_SIZE = 128U;
struct Mc2cCTilingInner {
    uint8_t skipLocalRankCopy;
    uint8_t skipBufferWindowCopy;
    uint8_t stepSize;
    uint8_t version;
    char reserved[9];
    uint8_t commEngine;
    uint8_t srcDataType;
    uint8_t dstDataType;
    char groupName[GROUP_NAME_SIZE];
    char algConfig[ALG_CONFIG_SIZE];
    uint32_t opType;
    uint32_t reduceType;
};

struct Mc2CommConfigV2 {
    Mc2InitTilingInner init;
    Mc2cCTilingInner inner;
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

class Mc2CommConfigMgrV1 : public Mc2CommConfigMgrBase<Mc2CommConfigV1>  {
public:
    Mc2CommConfigMgrV1() {}
    ~Mc2CommConfigMgrV1() {}
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

class Mc2CommConfigMgrV2 : public Mc2CommConfigMgrBase<Mc2CommConfigV2> {
public:
    Mc2CommConfigMgrV2() {}
    ~Mc2CommConfigMgrV2() {}
    int32_t MakeMc2TilingStruct(const std::string& groupName) override
    {
        (void)memset_s(&Mc2CommConfig_, sizeof(Mc2CommConfig_), 0, sizeof(Mc2CommConfig_));
        const char *algConfig = "BatchWrite=level0:fullmesh";
        Mc2CommConfig_.init.version = 100U;
        Mc2CommConfig_.init.mc2HcommCnt = 1;
        Mc2CommConfig_.init.queueNum = 0;
        Mc2CommConfig_.init.commBlockNum =48U;
        Mc2CommConfig_.init.devType = 4U;
        Mc2CommConfig_.inner.skipLocalRankCopy = 0;
        Mc2CommConfig_.inner.skipBufferWindowCopy =0;
        Mc2CommConfig_.inner.stepSize = 0;
        Mc2CommConfig_.inner.opType = 18U;
        Mc2CommConfig_.inner.version = 1;
        Mc2CommConfig_.init.offset[0] = static_cast<uint32_t>(reinterpret_cast<uint64_t>(&Mc2CommConfig_.inner) - reinterpret_cast<uint64_t>(&Mc2CommConfig_.init));
        auto ret = strcpy_s(Mc2CommConfig_.inner.groupName, GROUP_NAME_SIZE, groupName.c_str());
        if(ret != 0){
            return -1;
        }
        ret = strcpy_s(Mc2CommConfig_.inner.algConfig, ALG_CONFIG_SIZE, algConfig);
        if(ret != 0){
            return -1;
        }
        return 0;
    }
};

using ConfigMgrCommon = std::variant<
    std::shared_ptr<Mc2CommConfigMgrV1>,
    std::shared_ptr<Mc2CommConfigMgrV2>
>;
}

namespace npu::tile_fwk::dynamic {
std::vector<uint64_t> DistributedContext::GetHcclContextToHost(const std::vector<std::string> &groupNames) {
    std::vector<uint64_t> devAddrs = GetHcclContext(groupNames);
#ifdef BUILD_WITH_CANN
    std::vector<uint64_t> hosts;
    auto SoCVersion = Platform::Instance().GetSoc().GetShortSoCVersion();
    for (size_t i = 0; i < groupNames.size(); i++) {
        if (SoCVersion == ShortSoCVersion::SoC_910B) {
        (void)rtMemcpy(&g_hostAddrV1[i], sizeof(g_hostAddrV1[i]), (uint8_t *)devAddrs[0], sizeof(g_hostAddrV1[i]), RT_MEMCPY_DEVICE_TO_HOST);
        hosts.push_back((uint64_t)(&g_hostAddrV1[i]));
    }
    else {
        (void)rtMemcpy(&g_hostAddrV2[i], sizeof(g_hostAddrV2[i]), (uint8_t *)devAddrs[0], sizeof(g_hostAddrV2[i]), RT_MEMCPY_DEVICE_TO_HOST);
        hosts.push_back((uint64_t)(&g_hostAddrV2[i]));
    }
    }
    return hosts;
#else
    return {};
#endif
}

std::vector<uint64_t> DistributedContext::GetHcclContext([[maybe_unused]] const std::vector<std::string> &groupNames)
{
#ifdef BUILD_WITH_CANN
    auto SoCVersion = Platform::Instance().GetSoc().GetShortSoCVersion();
    ConfigMgrCommon commConfigMgr;
    if (SoCVersion == ShortSoCVersion::SoC_910B) {
        commConfigMgr = std::make_shared<Mc2CommConfigMgrV1>();
    }
    else {
        commConfigMgr = std::make_shared<Mc2CommConfigMgrV2>();
    }
    std::vector<uint64_t> hcclContext(groupNames.size(), 0);
    for (size_t groupIndex = 0; groupIndex < groupNames.size(); ++groupIndex) {
        auto groupName = groupNames[groupIndex];
        HcclComm commHandle = nullptr;
        HcclResult ret = HcomGetCommHandleByGroup(groupName.c_str(), &commHandle);
        ASSERT(ret == 0);
        bool makeTilingSuccess = std::visit(
            [groupName](const auto& cfgMgr) -> bool {
                return cfgMgr->MakeMc2TilingStruct(groupName) == 0;
            },
            commConfigMgr
        );
        ASSERT(makeTilingSuccess);
        ret = std::visit(
            [&](const auto& cfgMgr) -> HcclResult {
                return HcclAllocComResourceByTiling(
                    commHandle,
                    machine::GetRA()->GetStream(),
                    &(cfgMgr->Mc2CommConfig_),
                    reinterpret_cast<void **>(&hcclContext[groupIndex])
                );
            },
            commConfigMgr
        );
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