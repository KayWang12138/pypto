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
 * \file runtime_stubs.cpp
 * \brief
 */

#include "runtime/runtime_stubs.h"

namespace npu::tile_fwk {
rtError_t StubMalloc(void **devPtr, uint64_t size, rtMemType_t type, const uint16_t moduleId) {
    (void)devPtr;
    (void)size;
    (void)type;
    (void)moduleId;
    return RT_ERROR_NONE;
}

rtError_t StubMemset(void *devPtr, uint64_t destMax, uint32_t val, uint64_t cnt) {
    (void)devPtr;
    (void)destMax;
    (void)val;
    (void)cnt;
    return RT_ERROR_NONE;
}

rtError_t StubMemcpy(void *dst, uint64_t destMax, const void *src, uint64_t cnt, rtMemcpyKind_t kind) {
    (void)dst;
    (void)destMax;
    (void)src;
    (void)cnt;
    (void)kind;
    return RT_ERROR_NONE;
}

rtError_t StubMemcpyAsync(void *dst, uint64_t destMax, const void *src, uint64_t cnt, rtMemcpyKind_t kind,
                          rtStream_t stm) {
    (void)dst;
    (void)destMax;
    (void)src;
    (void)cnt;
    (void)kind;
    (void)stm;
    return RT_ERROR_NONE;
}

rtError_t StubFree(void *devPtr) {
    (void)devPtr;
    return RT_ERROR_NONE;
}

rtError_t StubSetDevice(int32_t devId) {
    (void)devId;
    return RT_ERROR_NONE;
}

rtError_t StubGetDevice(int32_t *devId) {
    (void)devId;
    return RT_ERROR_NONE;
}

rtError_t StubGetSocSpec(const char* label, const char* key, char* val, const uint32_t maxLen) {
    (void)label;
    (void)key;
    (void)val;
    (void)maxLen;
    return RT_ERROR_NONE;
}

rtError_t StubGetSocVersion(char_t *ver, const uint32_t maxLen) {
    (void)ver;
    (void)maxLen;
    return RT_ERROR_NONE;
}

rtError_t StubGetAiCpuCount(uint32_t *aiCpuCnt) {
    (void)aiCpuCnt;
    return RT_ERROR_NONE;
}

rtError_t StubGetL2CacheOffset(uint32_t deviceId, uint64_t *offset) {
    (void)deviceId;
    (void)offset;
    return RT_ERROR_NONE;
}

rtError_t StubGetLogicDevIdByUserDevId(const int32_t userDevId, int32_t * const logicDevId) {
    (void)userDevId;
    (void)logicDevId;
    return RT_ERROR_NONE;
}

rtError_t StubStreamCreate(rtStream_t *stm, int32_t priority) {
    (void)stm;
    (void)priority;
    return RT_ERROR_NONE;
}

rtError_t StubStreamDestroy(rtStream_t stm) {
    (void)stm;
    return RT_ERROR_NONE;
}

rtError_t StubStreamAddToModel(rtStream_t stm, rtModel_t captureMdl) {
    (void)stm;
    (void)captureMdl;
    return RT_ERROR_NONE;
}

rtError_t StubStreamSynchronize(rtStream_t stm) {
    (void)stm;
    return RT_ERROR_NONE;
}

rtError_t StubDevBinaryUnRegister(void *handle) {
    (void)handle;
    return RT_ERROR_NONE;
}

rtError_t StubRegisterAllKernel(const rtDevBinary_t *bin, void **hdl) {
    (void)bin;
    (void)hdl;
    return RT_ERROR_NONE;
}

rtError_t StubKernelLaunchWithHandleV2(void *hdl, const uint64_t tilingKey, uint32_t numBlocks,
                                       rtArgsEx_t *argsInfo, rtSmDesc_t *smDesc, rtStream_t stm,
                                       const rtTaskCfgInfo_t *cfgInfo) {
    (void)hdl;
    (void)tilingKey;
    (void)numBlocks;
    (void)argsInfo;
    (void)smDesc;
    (void)stm;
    (void)cfgInfo;
    return RT_ERROR_NONE;
}


rtError_t StubAicpuKernelLaunchExWithArgs(const uint32_t kernelType, const char_t * const opName,
                                          const uint32_t numBlocks, const rtAicpuArgsEx_t *argsInfo,
                                          rtSmDesc_t * const smDesc, const rtStream_t stm, const uint32_t flags) {
    (void)kernelType;
    (void)opName;
    (void)numBlocks;
    (void)argsInfo;
    (void)smDesc;
    (void)stm;
    (void)flags;
    return RT_ERROR_NONE;
}

}