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
 * \file distributed_context.h
 * \brief
 */

#pragma once
#include <vector>
#include <string>
#include "hccl_context.h"
#include "interface/tileop/distributed/comm_context.h"

namespace npu::tile_fwk::dynamic {
constexpr int WIN_TYPE_NUM = 3; // win区类型in, status, debug
enum class ResType {
    MESH,
    RING,
    UNKNOWN
};

class DistributedContext {
public:
    DistributedContext(){};
    ~DistributedContext(){};
    static std::vector<uint64_t> GetCommContext(const std::vector<std::string> &groupNames);
    static std::vector<uint64_t> GetCommContextToHost(const std::vector<std::string> &groupNames);
    template<ResType T>
    static uint64_t AllocCommContext(uint64_t ctxAddr);
private:
    template<typename T>
    static void FillCommCtxAttr(TileOp::CommContext *ctxHost, T *hcclParamhost) {
        if constexpr (std::is_same_v<T, npu::tile_fwk::HcclCombinOpParam>) {
            ctxHost->rankId = hcclParamhost->rankId;
            ctxHost->rankNum = hcclParamhost->rankNum;
            ctxHost->statusIndex = hcclParamhost->rankNum;
            ctxHost->debugIndex = hcclParamhost->rankNum * 2;
            ctxHost->winDataSize = hcclParamhost->winSize;
            ctxHost->winStatusSize = hcclParamhost->winExpSize;
            ctxHost->winDebugSize = hcclParamhost->winSize;
            ctxHost->totalWinNum = hcclParamhost->rankNum * WIN_TYPE_NUM;
        } else {
            ctxHost->rankId = hcclParamhost->localUsrRankId;
            ctxHost->rankNum = hcclParamhost->rankSize;
            ctxHost->statusIndex = hcclParamhost->rankSize;
            ctxHost->debugIndex = hcclParamhost->rankSize * 2;
            ctxHost->winDataSize = hcclParamhost->winSize;
            ctxHost->winStatusSize = hcclParamhost->winSize;
            ctxHost->winDebugSize = hcclParamhost->winExpSize;
            ctxHost->totalWinNum = hcclParamhost->rankSize * WIN_TYPE_NUM;
        } 
    }
    template<typename T>
    static void FillCommCtxWinArr(int i, TileOp::CommContext *ctxHost, T *hcclParamhost) {
        if constexpr (std::is_same_v<T, npu::tile_fwk::HcclCombinOpParam>) {
            ctxHost->winAddr[i + (0 * ctxHost->rankNum)] = hcclParamhost->windowsIn[i];
            ctxHost->winAddr[i + (1 * ctxHost->rankNum)] = hcclParamhost->windowsExp[i];
            ctxHost->winAddr[i + (2 * ctxHost->rankNum)] = hcclParamhost->windowsOut[i];
        } else if constexpr (std::is_same_v<T, npu::tile_fwk::HcclOpResParamHead>) {
            ctxHost->winAddr[i + (0 * ctxHost->rankNum)] = hcclParamhost->localWindowsIn;
            ctxHost->winAddr[i + (1 * ctxHost->rankNum)] = hcclParamhost->localWindowsExp;
            ctxHost->winAddr[i + (2 * ctxHost->rankNum)] = hcclParamhost->localWindowsOut;
        } else (std::is_same_v<T, npu::tile_fwk::HcclRankRelationResV2>) {
            ctxHost->winAddr[i + (0 * ctxHost->rankNum)] = hcclParamhost->windowsIn;
            ctxHost->winAddr[i + (1 * ctxHost->rankNum)] = hcclParamhost->windowsExp;
            ctxHost->winAddr[i + (2 * ctxHost->rankNum)] = hcclParamhost->windowsOut;
        }
    }
};
} // namespace npu::tile_fwk::dynamic