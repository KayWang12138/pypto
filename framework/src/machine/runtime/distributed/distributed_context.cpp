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
#include "interface/utils/common.h"
#include "machine/runtime/runtime.h"
#include "machine/device/dynamic/device_utils.h"

#ifdef BUILD_WITH_CANN
#include "hcom.h"
#include "acl/acl.h"
extern "C" HcclResult HcclAllocComResourceByTiling(HcclComm comm, void* stream, void* mc2Tiling, void** commContext);
#endif
constexpr uint32_t COMM_IS_NOT_SET_DEVICE = 0;
constexpr uint32_t COMM_MESH = 0b1u;
TileOp::CommContext g_hostAddr[npu::tile_fwk::DIST_COMM_GROUP_NUM];
std::unordered_map<std::string, uint64_t> g_context; //key: groupname, value: deviceHcclContext

namespace npu::tile_fwk::dynamic {
std::vector<uint64_t> DistributedContext::GetCommContextToHost([[maybe_unused]] const std::vector<std::string> &groupNames)
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
    return {};
}

template<ResType T>
uint64_t DistributedContext::AllocCommContext([[maybe_unused]] uint64_t ctxAddr) 
{
    return 0;
}

template<>
uint64_t DistributedContext::AllocCommContext<ResType::MESH>([[maybe_unused]] uint64_t ctxAddr)
{
#ifdef BUILD_WITH_CANN
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

    size_t commCtxSize = sizeof(TileOp::CommContext) + sizeof(uint64_t) * hcclParamhost->rankNum * WIN_TYPE_NUM;
    TileOp::CommContext *ctxHost = 
            (TileOp::CommContext *)machine::GetRuntimeHostAgent()->AllocHostAddr(commCtxSize);
    ASSERT(ctxHost != nullptr);
    TileOp::CommContext *ctxDevice = nullptr;
    machine::GetRA()->AllocDevAddr((uint8_t **)&ctxDevice, commCtxSize);
    ASSERT(ctxDevice != nullptr);
    FillCommCtxAttr<npu::tile_fwk::HcclCombinOpParam>(ctxHost, hcclParamhost);
    for (uint32_t i = 0; i < ctxHost->rankNum; i++) {
        FillCommCtxWinArr<npu::tile_fwk::HcclCombinOpParam>(i, ctxHost, hcclParamhost);
    }
    aclrtMemcpy(ctxDevice, sizeof(TileOp::CommContext) + sizeof(uint64_t) * hcclParamhost->rankNum * WIN_TYPE_NUM,
                ctxHost, sizeof(TileOp::CommContext) + sizeof(uint64_t) * hcclParamhost->rankNum * WIN_TYPE_NUM,
                ACL_MEMCPY_HOST_TO_DEVICE);
    return (uint64_t)ctxDevice;
#endif
    return 0;
}

template<>
uint64_t DistributedContext::AllocCommContext<ResType::RING>([[maybe_unused]] uint64_t ctxAddr)
{
#ifdef BUILD_WITH_CANN
    npu::tile_fwk::HcclOpResParam *hcclParam = (npu::tile_fwk::HcclOpResParam *)ctxAddr;
    npu::tile_fwk::HcclOpResParamHead *hcclParamhost = 
            (npu::tile_fwk::HcclOpResParamHead *)machine::GetRuntimeHostAgent()->AllocHostAddr(sizeof(npu::tile_fwk::HcclOpResParamHead));
    ASSERT(hcclParamhost != nullptr);
    size_t offset_localUsrRankId = offsetof(npu::tile_fwk::HcclOpResParam, localUsrRankId);
    size_t offset_rWinStart = offsetof(npu::tile_fwk::HcclOpResParam, rWinStart);
    aclrtMemcpy(&(hcclParamhost->localUsrRankId), offset_rWinStart - offset_localUsrRankId,
                &(hcclParam->localUsrRankId), offset_rWinStart - offset_localUsrRankId,
                ACL_MEMCPY_DEVICE_TO_HOST);

    size_t remoteResSize = hcclParamhost->rankSize * sizeof(npu::tile_fwk::RemoteResPtr);
    npu::tile_fwk::RemoteResPtr *remoteResPtr = 
            (npu::tile_fwk::RemoteResPtr *)machine::GetRuntimeHostAgent()->AllocHostAddr(remoteResSize);
    ASSERT(remoteResPtr != nullptr);
    aclrtMemcpy(remoteResPtr, remoteResSize,
                &(hcclParam->remoteRes), remoteResSize,
                ACL_MEMCPY_DEVICE_TO_HOST);
    
    size_t commCtxSize = sizeof(TileOp::CommContext) + sizeof(uint64_t) * hcclParamhost->rankSize * WIN_TYPE_NUM;
    TileOp::CommContext *ctxHost = (TileOp::CommContext *)machine::GetRuntimeHostAgent()->AllocHostAddr(commCtxSize);
    ASSERT(ctxHost != nullptr);
    TileOp::CommContext *ctxDevice = nullptr;
    machine::GetRA()->AllocDevAddr((uint8_t **)&ctxDevice, commCtxSize);
    ASSERT(ctxDevice != nullptr);
    FillCommCtxAttr<npu::tile_fwk::HcclOpResParamHead>(ctxHost, hcclParamhost);
    for (uint64_t i = 0; i < hcclParamhost->rankSize; i++) {
        if (i == hcclParamhost->localUsrRankId) {
            FillCommCtxWinArr<npu::tile_fwk::HcclOpResParamHead>(i, ctxHost, hcclParamhost);
            continue;
        }
        uint64_t remoteResDevicePtr;
        aclrtMemcpy(&remoteResDevicePtr, sizeof(uint64_t), 
                    &(remoteResPtr[i].nextDevicePtr), sizeof(uint64_t), 
                    ACL_MEMCPY_DEVICE_TO_HOST); // 设备二级指针值拷贝到主机
        npu::tile_fwk::HcclRankRelationResV2 remoteParam;
        aclrtMemcpy(&remoteParam, sizeof(npu::tile_fwk::HcclRankRelationResV2), 
                (void *)remoteResDevicePtr, sizeof(npu::tile_fwk::HcclRankRelationResV2), ACL_MEMCPY_DEVICE_TO_HOST);
        FillCommCtxWinArr<npu::tile_fwk::HcclRankRelationResV2>(i, ctxHost, &remoteParam);
    }
    aclrtMemcpy(ctxDevice, sizeof(TileOp::CommContext) + sizeof(uint64_t) * hcclParamhost->rankSize * WIN_TYPE_NUM, 
                ctxHost, sizeof(TileOp::CommContext) + sizeof(uint64_t) * hcclParamhost->rankSize * WIN_TYPE_NUM,
                ACL_MEMCPY_HOST_TO_DEVICE);
    return (uint64_t)ctxDevice;
#endif
    return 0;
}

std::vector<uint64_t> DistributedContext::GetCommContext([[maybe_unused]] const std::vector<std::string> &groupNames)
{
#ifdef BUILD_WITH_CANN
    CommTopo topoRet;
    if(groupNames.size() != 0) {
        const char *group = groupNames[0].c_str();
        ASSERT(HcomGetL0TopoTypeEx(group, &topoRet, COMM_IS_NOT_SET_DEVICE) == HCCL_SUCCESS);
    }
    uint32_t topoType = static_cast<uint32_t>(topoRet);
    std::shared_ptr<TilingStructBase> tilingStruct;
    if(topoType == COMM_MESH) {
        tilingStruct = std::make_shared<TilingStruct>();
    } else {
        tilingStruct = std::make_shared<TilingStructV2>();
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
            commContext[groupIndex] = AllocCommContext<ResType::MESH>(commContext[groupIndex]);
        } else {
            commContext[groupIndex] = AllocCommContext<ResType::RING>(commContext[groupIndex]);
        }
        g_context[groupName] = commContext[groupIndex];
    }
    return commContext;
#endif
    return {};
}
} // namespace npu::tile_fwk::dynamic