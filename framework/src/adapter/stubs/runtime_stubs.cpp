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

#include "adapter/stubs/runtime_stubs.h"
#include "tilefwk/pypto_fwk_log.h"

namespace npu::tile_fwk {
rtError_t StubMalloc(void **devPtr, uint64_t size, rtMemType_t type, const uint16_t moduleId) {
    ADAPTER_LOGD("Enter stub function of Malloc.");
    (void)devPtr;
    (void)size;
    (void)type;
    (void)moduleId;
    return RT_ERROR_NONE;
}

rtError_t StubMemset(void *devPtr, uint64_t destMax, uint32_t val, uint64_t cnt) {
    ADAPTER_LOGD("Enter stub function of Memset.");
    (void)devPtr;
    (void)destMax;
    (void)val;
    (void)cnt;
    return RT_ERROR_NONE;
}

rtError_t StubMemcpy(void *dst, uint64_t destMax, const void *src, uint64_t cnt, rtMemcpyKind_t kind) {
    ADAPTER_LOGD("Enter stub function of Memcpy.");
    (void)dst;
    (void)destMax;
    (void)src;
    (void)cnt;
    (void)kind;
    return RT_ERROR_NONE;
}

rtError_t StubMemcpyAsync(void *dst, uint64_t destMax, const void *src, uint64_t cnt, rtMemcpyKind_t kind,
                          rtStream_t stm) {
    ADAPTER_LOGD("Enter stub function of MemcpyAsync.");
    (void)dst;
    (void)destMax;
    (void)src;
    (void)cnt;
    (void)kind;
    (void)stm;
    return RT_ERROR_NONE;
}

rtError_t StubFree(void *devPtr) {
    ADAPTER_LOGD("Enter stub function of Free.");
    (void)devPtr;
    return RT_ERROR_NONE;
}

rtError_t StubSetDevice(int32_t devId) {
    ADAPTER_LOGD("Enter stub function of SetDevice.");
    (void)devId;
    return RT_ERROR_NONE;
}

rtError_t StubGetDevice(int32_t *devId) {
    ADAPTER_LOGD("Enter stub function of GetDevice.");
    (void)devId;
    return RT_ERROR_NONE;
}

rtError_t StubGetSocSpec(const char* label, const char* key, char* val, const uint32_t maxLen) {
    ADAPTER_LOGD("Enter stub function of GetSocSpec.");
    (void)label;
    (void)key;
    (void)val;
    (void)maxLen;
    return RT_ERROR_NONE;
}

rtError_t StubGetSocVersion(char_t *ver, const uint32_t maxLen) {
    ADAPTER_LOGD("Enter stub function of GetSocVersion.");
    (void)ver;
    (void)maxLen;
    return RT_ERROR_NONE;
}

rtError_t StubGetAiCpuCount(uint32_t *aiCpuCnt) {
    ADAPTER_LOGD("Enter stub function of GetAiCpuCount.");
    (void)aiCpuCnt;
    return RT_ERROR_NONE;
}

rtError_t StubGetL2CacheOffset(uint32_t deviceId, uint64_t *offset) {
    ADAPTER_LOGD("Enter stub function of GetL2CacheOffset.");
    (void)deviceId;
    (void)offset;
    return RT_ERROR_NONE;
}

rtError_t StubGetLogicDevIdByUserDevId(const int32_t userDevId, int32_t * const logicDevId) {
    ADAPTER_LOGD("Enter stub function of GetLogicDevIdByUserDevId.");
    (void)userDevId;
    (void)logicDevId;
    return RT_ERROR_NONE;
}

rtError_t StubFuncGetByName(const rtBinHandle binHandle, const char_t *kernelName, rtFuncHandle *funcHandle) {
    (void)binHandle;
    (void)kernelName;
    (void)funcHandle;
    return RT_ERROR_NONE;
}

rtError_t StubBinaryLoadFromFile(const char_t * const binPath, const rtLoadBinaryConfig_t * const optionalCfg,
                                 rtBinHandle *handle) {
    (void)binPath;
    (void)optionalCfg;
    (void)handle;
    return RT_ERROR_NONE;
}

rtError_t StubStreamCreate(rtStream_t *stm, int32_t priority) {
    ADAPTER_LOGD("Enter stub function of StreamCreate.");
    (void)stm;
    (void)priority;
    return RT_ERROR_NONE;
}

rtError_t StubStreamDestroy(rtStream_t stm) {
    ADAPTER_LOGD("Enter stub function of StreamDestroy.");
    (void)stm;
    return RT_ERROR_NONE;
}

rtError_t StubStreamAddToModel(rtStream_t stm, rtModel_t captureMdl) {
    ADAPTER_LOGD("Enter stub function of StreamAddToModel.");
    (void)stm;
    (void)captureMdl;
    return RT_ERROR_NONE;
}

rtError_t StubStreamSynchronize(rtStream_t stm) {
    ADAPTER_LOGD("Enter stub function of StreamSynchronize.");
    (void)stm;
    return RT_ERROR_NONE;
}

rtError_t StubDevBinaryUnRegister(void *handle) {
    ADAPTER_LOGD("Enter stub function of DevBinaryUnRegister.");
    (void)handle;
    return RT_ERROR_NONE;
}

rtError_t StubRegisterAllKernel(const rtDevBinary_t *bin, void **hdl) {
    ADAPTER_LOGD("Enter stub function of RegisterAllKernel.");
    (void)bin;
    (void)hdl;
    return RT_ERROR_NONE;
}

rtError_t StubLaunchCpuKernel(const rtFuncHandle funcHandle, uint32_t numBlocks, rtStream_t stm,
    const rtKernelLaunchCfg_t *cfg, rtCpuKernelArgs_t *argsInfo) {
    ADAPTER_LOGD("Enter stub function of LaunchCpuKernel.");
    (void)funcHandle;
    (void)numBlocks;
    (void)stm;
    (void)cfg;
    (void)argsInfo;
    return RT_ERROR_NONE;
}

rtError_t StubKernelLaunchWithHandleV2(void *hdl, const uint64_t tilingKey, uint32_t numBlocks,
                                       rtArgsEx_t *argsInfo, rtSmDesc_t *smDesc, rtStream_t stm,
                                       const rtTaskCfgInfo_t *cfgInfo) {
    ADAPTER_LOGD("Enter stub function of KernelLaunchWithHandleV2.");
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
    ADAPTER_LOGD("Enter stub function of AicpuKernelLaunchExWithArgs.");
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