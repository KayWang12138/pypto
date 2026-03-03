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
 * \file hccl_api.cpp
 * \brief
 */

#include "adapter/api/hccl_api.h"

#ifdef EXECUTE_WITH_CANN
#include "adapter/manager/adapter_manager.h"
#endif
#include "adapter/stubs/hccl_stubs.h"

namespace npu::tile_fwk {
HcclResult HcommGetCommName(HcclComm comm, char* commName) {
#ifdef EXECUTE_WITH_CANN
    void *func = AdapterManager::Instance().GetHcclAdapter().GetFunction(HcclFunc::GetCommName);
    if (func != nullptr) {
        HcclResult(*hcommFunc)(HcclComm, char*) = reinterpret_cast<HcclResult(*)(HcclComm, char*)>(func);
        return hcommFunc(comm, commName);
    }
#endif
    return StubGetCommName(comm, commName);
}

HcclResult HcommGetL0TopoTypeEx(const char *group, CommTopo *topoType, uint32_t flag) {
#ifdef EXECUTE_WITH_CANN
    void *func = AdapterManager::Instance().GetHcclAdapter().GetFunction(HcclFunc::GetL0TopoTypeEx);
    if (func != nullptr) {
        HcclResult(*hcommFunc)(const char*, CommTopo*, uint32_t) =
            reinterpret_cast<HcclResult(*)(const char*, CommTopo*, uint32_t)>(func);
        return hcommFunc(group, topoType, flag);
    }
#endif
    return StubGetL0TopoTypeEx(group, topoType, flag);
}

HcclResult HcommGetCommHandleByGroup(const char *group, HcclComm *commHandle) {
#ifdef EXECUTE_WITH_CANN
    void *func = AdapterManager::Instance().GetHcclAdapter().GetFunction(HcclFunc::GetCommHandleByGroup);
    if (func != nullptr) {
        HcclResult(*hcommFunc)(const char*, HcclComm*) = reinterpret_cast<HcclResult(*)(const char*, HcclComm*)>(func);
        return hcommFunc(group, commHandle);
    }
#endif
    return StubGetCommHandleByGroup(group, commHandle);
}

HcclResult HcommGetRootInfo(HcclRootInfo *rootInfo) {
#ifdef EXECUTE_WITH_CANN
    void *func = AdapterManager::Instance().GetHcclAdapter().GetFunction(HcclFunc::GetRootInfo);
    if (func != nullptr) {
        HcclResult(*hcommFunc)(HcclRootInfo*) = reinterpret_cast<HcclResult(*)(HcclRootInfo*)>(func);
        return hcommFunc(rootInfo);
    }
#endif
    return StubGetRootInfo(rootInfo);
}

HcclResult HcommCommInitRootInfo(uint32_t nRanks, const HcclRootInfo *rootInfo, uint32_t rank, HcclComm *comm) {
#ifdef EXECUTE_WITH_CANN
    void *func = AdapterManager::Instance().GetHcclAdapter().GetFunction(HcclFunc::CommInitRootInfo);
    if (func != nullptr) {
        HcclResult(*hcommFunc)(uint32_t, const HcclRootInfo*, uint32_t, HcclComm*) =
            reinterpret_cast<HcclResult(*)(uint32_t, const HcclRootInfo*, uint32_t, HcclComm*)>(func);
        return hcommFunc(nRanks, rootInfo, rank, comm);
    }
#endif
    return StubCommInitRootInfo(nRanks, rootInfo, rank, comm);
}

HcclResult HcommCommDestroy(HcclComm comm) {
#ifdef EXECUTE_WITH_CANN
    void *func = AdapterManager::Instance().GetHcclAdapter().GetFunction(HcclFunc::CommDestroy);
    if (func != nullptr) {
        HcclResult(*hcommFunc)(HcclComm) = reinterpret_cast<HcclResult(*)(HcclComm)>(func);
        return hcommFunc(comm);
    }
#endif
    return StubCommDestroy(comm);
}

HcclResult HcommAllocComResourceByTiling(HcclComm comm, void *stream, void *Mc2Tiling, void **commContext) {
#ifdef EXECUTE_WITH_CANN
    void *func = AdapterManager::Instance().GetHcclAdapter().GetFunction(HcclFunc::AllocComResourceByTiling);
    if (func != nullptr) {
        HcclResult(*hcommFunc)(HcclComm, void*, void*, void**) =
            reinterpret_cast<HcclResult(*)(HcclComm, void*, void*, void**)>(func);
        return hcommFunc(comm, stream, Mc2Tiling, commContext);
    }
#endif
    return StubAllocComResourceByTiling(comm, stream, Mc2Tiling, commContext);
}
}