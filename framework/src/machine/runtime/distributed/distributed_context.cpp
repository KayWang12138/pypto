/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
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

#include <memory>
#include "securec.h"
#include "distributed_context.h"
#include "hccl_context.h"
#include "interface/tileop/distributed/comm_context.h"
#include "interface/utils/common.h"
#include "machine/runtime/runtime.h"
#include "machine/device/dynamic/device_utils.h"

#ifdef BUILD_WITH_CANN
#include "hcom.h"
#include "acl/acl.h"
extern "C" HcclResult HcclAllocComResourceByTiling(HcclComm comm, void* stream, void* mc2Tiling, void** commContext);
#endif
static constexpr uint32_t COMM_IS_NOT_SET_DEVICE = 0;
constexpr uint32_t COMM_MESH = 0b1u;
constexpr int winTypeNum = 3; // win区类型in,status,debug
TileOp::CommContext g_hostAddr[npu::tile_fwk::DIST_COMM_GROUP_NUM];
std::unordered_map<std::string, uint64_t> g_context; //key: groupname, value: deviceHcclContext
namespace {  
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

class TilingStructBase {
public:
    TilingStructBase() {}
    virtual ~TilingStructBase() {}
    virtual int32_t MakeMc2TilingStruct(const std::string& groupName) = 0;
    virtual void *GetMc2CommConfig() = 0;
private:
    std::string groupName_{};
};

class TilingStruct : public TilingStructBase {
public:
    TilingStruct() {}
    ~TilingStruct() {}
    int32_t MakeMc2TilingStruct(const std::string& groupName) override
    {
        (void)memset_s(&Mc2CommConfig_, sizeof(Mc2CommConfig_), 0, sizeof(Mc2CommConfig_));
        constexpr uint32_t version = 2;
        constexpr uint32_t hcommCnt = 1;
        constexpr uint32_t opTypeAllToAll = 6; // numeric representation of AlltoAll
        const char *algConfig = "AllGather=level0:ring";

        Mc2CommConfig_.version = version;
        Mc2CommConfig_.hcommCnt = hcommCnt;
        Mc2CommConfig_.hcommCfg.skipLocalRankCopy = 0;
        Mc2CommConfig_.hcommCfg.skipBufferWindowCopy = 0;
        Mc2CommConfig_.hcommCfg.stepSize = 0;
        Mc2CommConfig_.hcommCfg.opType = opTypeAllToAll;
        if (strcpy_s(Mc2CommConfig_.hcommCfg.groupName, sizeof(Mc2CommConfig_.hcommCfg.groupName), groupName.c_str()) != EOK) {
            return -1;
        }
        if (strcpy_s(Mc2CommConfig_.hcommCfg.algConfig, sizeof(Mc2CommConfig_.hcommCfg.algConfig), algConfig) != EOK) {
            return -1;
        }
        return 0;
    }
    void *GetMc2CommConfig() override
    {
        return &Mc2CommConfig_;
    }
private:
    Mc2CommConfig Mc2CommConfig_;
};
}

namespace npu::tile_fwk::dynamic {
std::vector<uint64_t> DistributedContext::GetCommContextToHost(const std::vector<std::string> &groupNames)
{
#ifdef BUILD_WITH_CANN
    std::vector<uint64_t> devAddrs = GetCommContext(groupNames);
    std::vector<uint64_t> host_context;
    ASSERT(groupNames.size() <= DIST_COMM_GROUP_NUM);
    for (size_t groupIndex = 0; groupIndex < groupNames.size(); groupIndex++) {
        (void)rtMemcpy(&g_hostAddr[groupIndex], sizeof(g_hostAddr[groupIndex]), (uint8_t *)devAddrs[0], sizeof(g_hostAddr[groupIndex]),
            RT_MEMCPY_DEVICE_TO_HOST);
        host_context.push_back((uint64_t)(&g_hostAddr[groupIndex]));
    }
    return host_context;
#endif
    (void)groupNames;
    return {};
}

template<ResType T>
uint64_t DistributedContext::AllocCommContext(uint64_t ctxAddr) {
    (void)ctxAddr;
    return 0;
}

template<>
uint64_t DistributedContext::AllocCommContext<ResType::HCCLMESH>(uint64_t ctxAddr)
{
#ifdef BUILD_WITH_CANN
    for (const auto& [key, value] : g_context) {
        if (value == ctxAddr) {
            return value;
        }
    }
    npu::tile_fwk::HcclCombinOpParam *hcclParamDevice = (npu::tile_fwk::HcclCombinOpParam *)ctxAddr;
    npu::tile_fwk::HcclCombinOpParam *hcclParamhost = nullptr;
    hcclParamhost = (npu::tile_fwk::HcclCombinOpParam *)machine::GetRuntimeHostAgent()->AllocHostAddr(sizeof(npu::tile_fwk::HcclCombinOpParam));
    ASSERT(hcclParamhost != nullptr);
    size_t offset_rankId = offsetof(npu::tile_fwk::HcclCombinOpParam, rankId);
    size_t offset_hcomId = offsetof(npu::tile_fwk::HcclCombinOpParam, hcomId);
    size_t offset_winExpSize = offsetof(npu::tile_fwk::HcclCombinOpParam, winExpSize);
    size_t offset_multiServerFlag = offsetof(npu::tile_fwk::HcclCombinOpParam, multiServerFlag);
    aclrtMemcpy(&(hcclParamhost->rankId), offset_hcomId - offset_rankId, &(hcclParamDevice->rankId),
                offset_hcomId - offset_rankId, ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(&(hcclParamhost->winExpSize), offset_multiServerFlag - offset_winExpSize,
                &(hcclParamDevice->winExpSize), offset_multiServerFlag - offset_winExpSize,
                ACL_MEMCPY_DEVICE_TO_HOST);

    size_t commCtxSize = sizeof(TileOp::CommContext) + sizeof(uint64_t) * hcclParamhost->rankNum * winTypeNum;
    TileOp::CommContext *ctxHost = nullptr;
    ctxHost = (TileOp::CommContext *)machine::GetRuntimeHostAgent()->AllocHostAddr(commCtxSize);
    ASSERT(ctxHost != nullptr);
    TileOp::CommContext *ctxDevice = nullptr;
    machine::GetRA()->AllocDevAddr((uint8_t **)&ctxDevice, commCtxSize);
    ASSERT(ctxDevice != nullptr);
    ctxHost->rankId = hcclParamhost->rankId;
    ctxHost->rankNum = hcclParamhost->rankNum;
    ctxHost->winStatusStart = hcclParamhost->rankNum;
    ctxHost->winDebugStart = hcclParamhost->rankNum * 2;
    ctxHost->winDataSize = hcclParamhost->winSize;
    ctxHost->winStatusSize = hcclParamhost->winSize;
    ctxHost->winDebugSize = hcclParamhost->winExpSize;
    ctxHost->totalWinNum = hcclParamhost->rankNum * winTypeNum;
    for (uint32_t i = 0; i < ctxHost->rankNum; i++) {
        ctxHost->winAddr[i + (0 * ctxHost->rankNum)] = hcclParamhost->windowsIn[i];
        ctxHost->winAddr[i + (1 * ctxHost->rankNum)] = hcclParamhost->windowsExp[i];
        ctxHost->winAddr[i + (2 * ctxHost->rankNum)] = hcclParamhost->windowsOut[i];
    }
    aclrtMemcpy(ctxDevice, sizeof(TileOp::CommContext) + sizeof(uint64_t) * hcclParamhost->rankNum * winTypeNum,
                ctxHost, sizeof(TileOp::CommContext) + sizeof(uint64_t) * hcclParamhost->rankNum * winTypeNum,
                ACL_MEMCPY_HOST_TO_DEVICE);
    return (uint64_t)ctxDevice;
#endif
    return 0;
}

std::vector<uint64_t> DistributedContext::GetCommContext(const std::vector<std::string> &groupNames)
{
#ifdef BUILD_WITH_CANN
    CommTopo topoRet;
    const char *group = groupNames[0].c_str();
    ASSERT(HcomGetL0TopoTypeEx(group, &topoRet, COMM_IS_NOT_SET_DEVICE) == HCCL_SUCCESS);
    uint32_t topoType = static_cast<uint32_t>(topoRet);
    std::shared_ptr<TilingStructBase> tilingStruct;
    if(topoType == COMM_MESH) {
        tilingStruct = std::make_shared<TilingStruct>();
    } else {
        return {};
    }
    std::vector<uint64_t> commContext(groupNames.size(), 0);
    ASSERT(groupNames.size() <= DIST_COMM_GROUP_NUM);
    for (size_t groupIndex = 0; groupIndex < groupNames.size(); ++groupIndex) {
        auto groupName = groupNames[groupIndex];
        if (g_context.find(groupName) != g_context.end()) {
            commContext[groupIndex] = g_context[groupName];
            continue;
        }
        HcclComm commHandle = nullptr;
        ASSERT(HcomGetCommHandleByGroup(groupName.c_str(), &commHandle) == 0);
        ASSERT(tilingStruct->MakeMc2TilingStruct(groupName) == 0);
        auto ret = HcclAllocComResourceByTiling(commHandle, machine::GetRA()->GetStream(), tilingStruct->GetMc2CommConfig(),
            reinterpret_cast<void **>(&commContext[groupIndex]));
        ASSERT((ret == 0) && (commContext[groupIndex] != 0UL));
        ALOG_INFO_F("groupIndex=%u, groupName=%s, commContext=%lu", groupIndex, groupName.c_str(),
            commContext[groupIndex]);
        if (topoType == COMM_MESH) {
            commContext[groupIndex] = AllocCommContext<ResType::HCCLMESH>(commContext[groupIndex]);
        } else {
            return {};
        }
        g_context[groupName] = commContext[groupIndex];
    }
    return commContext;
#endif
    (void)groupNames;
    return {};
}
} // namespace npu::tile_fwk::dynamic