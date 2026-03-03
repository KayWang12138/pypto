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
 * \file runtime_api.h
 * \brief
 */

#pragma once

#include "adapter/runtime/runtime_define.h"

namespace npu::tile_fwk {
rtError_t RuntimeMalloc(void **devPtr, uint64_t size, rtMemType_t type, const uint16_t moduleId);
rtError_t RuntimeMemset(void *devPtr, uint64_t destMax, uint32_t val, uint64_t cnt);
rtError_t RuntimeMemcpy(void *dst, uint64_t destMax, const void *src, uint64_t cnt, rtMemcpyKind_t kind);
rtError_t RuntimeMemcpyAsync(void *dst, uint64_t destMax, const void *src, uint64_t cnt, rtMemcpyKind_t kind,
                             rtStream_t stm);
rtError_t RuntimeFree(void *devPtr);

rtError_t RuntimeSetDevice(int32_t devId);
rtError_t RuntimeGetDevice(int32_t *devId);
rtError_t RuntimeGetSocSpec(const char* label, const char* key, char* val, const uint32_t maxLen);
rtError_t RuntimeGetSocVersion(char_t *ver, const uint32_t maxLen);
rtError_t RuntimeGetAiCpuCount(uint32_t *aiCpuCnt);
rtError_t RuntimeGetL2CacheOffset(uint32_t deviceId, uint64_t *offset);
rtError_t RuntimeGetLogicDevIdByUserDevId(const int32_t userDevId, int32_t * const logicDevId);

rtError_t RuntimeStreamCreate(rtStream_t *stm, int32_t priority);
rtError_t RuntimeStreamDestroy(rtStream_t stm);
rtError_t RuntimeStreamAddToModel(rtStream_t stm, rtModel_t captureMdl);
rtError_t RuntimeStreamSynchronize(rtStream_t stm);

rtError_t RuntimeDevBinaryUnRegister(void *handle);
rtError_t RuntimeRegisterAllKernel(const rtDevBinary_t *bin, void **hdl);

rtError_t RuntimeKernelLaunchWithHandleV2(void *hdl, const uint64_t tilingKey, uint32_t numBlocks,
                                          rtArgsEx_t *argsInfo, rtSmDesc_t *smDesc, rtStream_t stm,
                                          const rtTaskCfgInfo_t *cfgInfo);

rtError_t RuntimeAicpuKernelLaunchExWithArgs(const uint32_t kernelType, const char_t * const opName,
                                             const uint32_t numBlocks, const rtAicpuArgsEx_t *argsInfo,
                                             rtSmDesc_t * const smDesc, const rtStream_t stm, const uint32_t flags);
}
