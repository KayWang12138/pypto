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
 * \file runtime_stubs.h
 * \brief
 */

#pragma once

#include "adapter/api/runtime_define.h"

namespace npu::tile_fwk {
rtError_t StubMalloc(void **devPtr, uint64_t size, rtMemType_t type, const uint16_t moduleId);
rtError_t StubMemset(void *devPtr, uint64_t destMax, uint32_t val, uint64_t cnt);
rtError_t StubMemcpy(void *dst, uint64_t destMax, const void *src, uint64_t cnt, rtMemcpyKind_t kind);
rtError_t StubMemcpyAsync(void *dst, uint64_t destMax, const void *src, uint64_t cnt, rtMemcpyKind_t kind, rtStream_t stm);
rtError_t StubFree(void *devPtr);

rtError_t StubSetDevice(int32_t devId);
rtError_t StubGetDevice(int32_t *devId);
rtError_t StubGetSocSpec(const char* label, const char* key, char* val, const uint32_t maxLen);
rtError_t StubGetSocVersion(char_t *ver, const uint32_t maxLen);
rtError_t StubGetAiCpuCount(uint32_t *aiCpuCnt);
rtError_t StubGetL2CacheOffset(uint32_t deviceId, uint64_t *offset);
rtError_t StubGetLogicDevIdByUserDevId(const int32_t userDevId, int32_t * const logicDevId);

rtError_t StubStreamCreate(rtStream_t *stm, int32_t priority);
rtError_t StubStreamDestroy(rtStream_t stm);
rtError_t StubStreamAddToModel(rtStream_t stm, rtModel_t captureMdl);
rtError_t StubStreamSynchronize(rtStream_t stm);

rtError_t StubDevBinaryUnRegister(void *handle);
rtError_t StubRegisterAllKernel(const rtDevBinary_t *bin, void **hdl);

rtError_t StubKernelLaunchWithHandleV2(void *hdl, const uint64_t tilingKey, uint32_t numBlocks,
    rtArgsEx_t *argsInfo, rtSmDesc_t *smDesc, rtStream_t stm, const rtTaskCfgInfo_t *cfgInfo);

rtError_t StubAicpuKernelLaunchExWithArgs(const uint32_t kernelType, const char_t * const opName,
                                          const uint32_t numBlocks, const rtAicpuArgsEx_t *argsInfo,
                                          rtSmDesc_t * const smDesc, const rtStream_t stm, const uint32_t flags);
}