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
 * \file hccl_stubs.cpp
 * \brief
 */

#include "adapter/stubs/hccl_stubs.h"
#include "tilefwk/pypto_fwk_log.h"

namespace npu::tile_fwk {
HcclResult StubGetCommName(HcclComm comm, char* commName) {
    ADAPTER_LOGD("Enter stub function of GetCommName.");
    (void)comm;
    (void)commName;
    return HCCL_SUCCESS;
}

HcclResult StubGetL0TopoTypeEx(const char *group, CommTopo *topoType, uint32_t flag) {
    ADAPTER_LOGD("Enter stub function of GetL0TopoTypeEx.");
    (void)group;
    (void)topoType;
    (void)flag;
    return HCCL_SUCCESS;
}

HcclResult StubGetCommHandleByGroup(const char *group, HcclComm *commHandle) {
    ADAPTER_LOGD("Enter stub function of GetCommHandleByGroup.");
    (void)group;
    (void)commHandle;
    return HCCL_SUCCESS;
}

HcclResult StubGetRootInfo(HcclRootInfo *rootInfo) {
    ADAPTER_LOGD("Enter stub function of GetRootInfo.");
    (void)rootInfo;
    return HCCL_SUCCESS;
}

HcclResult StubCommInitRootInfo(uint32_t nRanks, const HcclRootInfo *rootInfo, uint32_t rank, HcclComm *comm) {
    ADAPTER_LOGD("Enter stub function of CommInitRootInfo.");
    (void)nRanks;
    (void)rootInfo;
    (void)rank;
    (void)comm;
    return HCCL_SUCCESS;
}

HcclResult StubAllocComResourceByTiling(HcclComm comm, void *stream, void *Mc2Tiling, void **commContext) {
    ADAPTER_LOGD("Enter stub function of AllocComResourceByTiling.");
    (void)comm;
    (void)stream;
    (void)Mc2Tiling;
    (void)commContext;
    return HCCL_SUCCESS;
}
}