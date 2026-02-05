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
#include "distributed_context.h"
#include "machine/runtime/mc2_tiling.h"
#include "interface/tileop/distributed/hccl_context.h"
#include "interface/machine/device/tilefwk/core_func_data.h"
#include "runtime.h"
#ifdef BUILD_WITH_CANN
#include "hcom.h"
extern "C" HcclResult HcclAllocComResourceByTiling(HcclComm comm, void* stream, void* mc2Tiling, void** commContext);
#endif
namespace {  
TileOp::HcclCombinOpParam g_hostAddr[npu::tile_fwk::DIST_COMM_GROUP_NUM];
std::unordered_map<std::string, uint64_t> g_context; //key: groupname, value: deviceHcclContext

}

namespace npu::tile_fwk::dynamic {
std::vector<uint64_t> DistributedContext::GetHcclContextToHost(const std::vector<std::string> &groupNames) {
#ifdef BUILD_WITH_CANN
    std::vector<uint64_t> devAddrs = GetHcclContext(groupNames);
    std::vector<uint64_t> host_context;
    ASSERT(groupNames.size() <= npu::tile_fwk::DIST_COMM_GROUP_NUM);
    for (size_t groupIndex = 0; groupIndex < groupNames.size(); groupIndex++) {
        (void)rtMemcpy(&g_hostAddr[groupIndex], sizeof(g_hostAddr[groupIndex]),
            (uint8_t *)devAddrs[groupIndex], sizeof(g_hostAddr[groupIndex]), RT_MEMCPY_DEVICE_TO_HOST);
        host_context.push_back((uint64_t)(&g_hostAddr[groupIndex]));
    }
    return host_context;
#endif
    (void)groupNames;
    return {};
}

std::vector<uint64_t> DistributedContext::GetHcclContext(const std::vector<std::string> &groupNames)
{
#ifdef BUILD_WITH_CANN
    std::vector<uint64_t> hcclContext(groupNames.size(), 0);
    ASSERT(groupNames.size() <= npu::tile_fwk::DIST_COMM_GROUP_NUM);
    for (size_t groupIndex = 0; groupIndex < groupNames.size(); ++groupIndex) {
        auto groupName = groupNames[groupIndex];
        if (g_context.find(groupName) != g_context.end()) {
            hcclContext[groupIndex] = g_context[groupName];
            continue;
        }
        HcclComm commHandle = nullptr;
        HcclResult ret = HcomGetCommHandleByGroup(groupName.c_str(), &commHandle);
        ASSERT(ret == 0);
        void *commContext = nullptr;
        HcclResult retV1 = HCCL_E_INTERNAL;
        HcclResult retV2 = HCCL_E_INTERNAL;
        Mc2CommConfig commConfig = {};
        if (MakeMc2TilingStruct(commConfig, groupName) == 0) {
            retV1 = HcclAllocComResourceByTiling(commHandle, machine::GetRA()->GetStream(),
                &commConfig, &commContext);
        }
        if ((retV1 != 0) || (commContext == nullptr)) {
            commContext = nullptr;
            Mc2CommConfigV2 commConfigV2 = {};
            if (MakeMc2TilingStructV2(commConfigV2, groupName) == 0) {
                retV2 = HcclAllocComResourceByTiling(commHandle, machine::GetRA()->GetStream(),
                    &commConfigV2, &commContext);
            }
        }
        if ((retV1 != 0) && (retV2 != 0)) {
            ALOG_ERROR_F("HcclAllocComResourceByTiling failed: groupName=%s retV1=%d retV2=%d",
                groupName.c_str(), retV1, retV2);
        }
        ASSERT(commContext != nullptr);
        hcclContext[groupIndex] = reinterpret_cast<uint64_t>(commContext);
        ALOG_INFO_F("groupIndex=%u, groupName=%s, hcclContext=%lu", groupIndex, groupName.c_str(),
            hcclContext[groupIndex]);
        g_context[groupName] = hcclContext[groupIndex];
    }
    return hcclContext;
#endif
    (void)groupNames;
    return {};
}
} // namespace npu::tile_fwk::dynamic
