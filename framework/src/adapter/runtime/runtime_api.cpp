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
 * \file runtime_api.cpp
 * \brief
 */

#include "runtime/runtime_api.h"

#ifdef BUILD_WITH_CANN
#include "runtime/runtime_adapter.h"
#endif
#include "runtime/runtime_stubs.h"

namespace npu::tile_fwk {
rtError_t RuntimeMalloc(void **devPtr, uint64_t size, rtMemType_t type, const uint16_t moduleId) {
#ifdef BUILD_WITH_CANN
    void *func = RuntimeAdapter::Instance().GetFunction(RuntimeFunc::Malloc);
    if (func != nullptr) {
        rtError_t(*mallocFunc)(void**, uint64_t, rtMemType_t, const uint16_t) =
            reinterpret_cast<rtError_t(*)(void**, uint64_t, rtMemType_t, const uint16_t)>(func);
        return mallocFunc(devPtr, size, type, moduleId);
    }
#endif
    return StubMalloc(devPtr, size, type, moduleId);
}

rtError_t RuntimeMemset(void *devPtr, uint64_t destMax, uint32_t val, uint64_t cnt) {
#ifdef BUILD_WITH_CANN
    void *func = RuntimeAdapter::Instance().GetFunction(RuntimeFunc::Memset);
    if (func != nullptr) {
        rtError_t(*memsetFunc)(void*, uint64_t, uint32_t, uint64_t) =
            reinterpret_cast<rtError_t(*)(void*, uint64_t, uint32_t, uint64_t)>(func);
        return memsetFunc(devPtr, destMax, val, cnt);
    }
#endif
    return StubMemset(devPtr, destMax, val, cnt);
}

rtError_t RuntimeMemcpy(void *dst, uint64_t destMax, const void *src, uint64_t cnt, rtMemcpyKind_t kind) {
#ifdef BUILD_WITH_CANN
    void *func = RuntimeAdapter::Instance().GetFunction(RuntimeFunc::Memcpy);
    if (func != nullptr) {
        rtError_t(*memcpyFunc)(void *, uint64_t, const void *, uint64_t, rtMemcpyKind_t) =
            reinterpret_cast<rtError_t(*)(void *, uint64_t, const void *, uint64_t, rtMemcpyKind_t)>(func);
        return memcpyFunc(dst, destMax, src, cnt, kind);
    }
#endif
    return StubMemcpy(dst, destMax, src, cnt, kind);
}

rtError_t RuntimeMemcpyAsync(void *dst, uint64_t destMax, const void *src, uint64_t cnt, rtMemcpyKind_t kind,
                             rtStream_t stm) {
#ifdef BUILD_WITH_CANN
    void *func = RuntimeAdapter::Instance().GetFunction(RuntimeFunc::MemcpyAsync);
    if (func != nullptr) {
        rtError_t(*memcpyAsyncFunc)(void*, uint64_t, const void *, uint64_t, rtMemcpyKind_t, rtStream_t) =
            reinterpret_cast<rtError_t(*)(void*, uint64_t, const void *, uint64_t, rtMemcpyKind_t, rtStream_t)>(func);
        return memcpyAsyncFunc(dst, destMax, src, cnt, kind, stm);
    }
#endif
    return StubMemcpyAsync(dst, destMax, src, cnt, kind, stm);
}

rtError_t RuntimeFree(void *devPtr) {
#ifdef BUILD_WITH_CANN
    void *func = RuntimeAdapter::Instance().GetFunction(RuntimeFunc::Free);
    if (func != nullptr) {
        rtError_t(*freeFunc)(void*) = reinterpret_cast<rtError_t(*)(void*)>(func);
        return freeFunc(devPtr);
    }
#endif
    return StubFree(devPtr);
}

rtError_t RuntimeSetDevice(int32_t devId) {
#ifdef BUILD_WITH_CANN
    void *func = RuntimeAdapter::Instance().GetFunction(RuntimeFunc::SetDevice);
    if (func != nullptr) {
        rtError_t(*setDeviceFunc)(int32_t) = reinterpret_cast<rtError_t(*)(int32_t)>(func);
        return setDeviceFunc(devId);
    }
#endif
    return StubSetDevice(devId);
}

rtError_t RuntimeGetDevice(int32_t *devId) {
#ifdef BUILD_WITH_CANN
    void *func = RuntimeAdapter::Instance().GetFunction(RuntimeFunc::GetDevice);
    if (func != nullptr) {
        rtError_t(*getDeviceFunc)(int32_t*) = reinterpret_cast<rtError_t(*)(int32_t*)>(func);
        return getDeviceFunc(devId);
    }
#endif
    return StubGetDevice(devId);
}

rtError_t RuntimeGetSocSpec(const char* label, const char* key, char* val, const uint32_t maxLen) {
#ifdef BUILD_WITH_CANN
    void *func = RuntimeAdapter::Instance().GetFunction(RuntimeFunc::GetSocSpec);
    if (func != nullptr) {
        rtError_t(*getSocSpecFunc)(const char*, const char*, char*, const uint32_t) =
            reinterpret_cast<rtError_t(*)(const char*, const char*, char*, const uint32_t)>(func);
        return getSocSpecFunc(label, key, val, maxLen);
    }
#endif
    return StubGetSocSpec(label, key, val, maxLen);
}

rtError_t RuntimeGetSocVersion(char_t *ver, const uint32_t maxLen) {
#ifdef BUILD_WITH_CANN
    void *func = RuntimeAdapter::Instance().GetFunction(RuntimeFunc::GetSocVersion);
    if (func != nullptr) {
        rtError_t(*getSocVersionFunc)(char_t*, const uint32_t) =
            reinterpret_cast<rtError_t(*)(char_t*, const uint32_t)>(func);
        return getSocVersionFunc(ver, maxLen);
    }
#endif
    return StubGetSocVersion(ver, maxLen);
}

rtError_t RuntimeGetAiCpuCount(uint32_t *aiCpuCnt) {
#ifdef BUILD_WITH_CANN
    void *func = RuntimeAdapter::Instance().GetFunction(RuntimeFunc::GetAiCpuCount);
    if (func != nullptr) {
        rtError_t(*getAiCpuCountFunc)(uint32_t*) = reinterpret_cast<rtError_t(*)(uint32_t*)>(func);
        return getAiCpuCountFunc(aiCpuCnt);
    }
#endif
    return StubGetAiCpuCount(aiCpuCnt);
}

rtError_t RuntimeGetL2CacheOffset(uint32_t deviceId, uint64_t *offset) {
#ifdef BUILD_WITH_CANN
    void *func = RuntimeAdapter::Instance().GetFunction(RuntimeFunc::GetL2CacheOffset);
    if (func != nullptr) {
        rtError_t(*getL2CacheOffsetFunc)(uint32_t, uint64_t*) =
            reinterpret_cast<rtError_t(*)(uint32_t, uint64_t*)>(func);
        return getL2CacheOffsetFunc(deviceId, offset);
    }
#endif
    return StubGetL2CacheOffset(deviceId, offset);
}

rtError_t RuntimeGetLogicDevIdByUserDevId(const int32_t userDevId, int32_t * const logicDevId) {
#ifdef BUILD_WITH_CANN
    void *func = RuntimeAdapter::Instance().GetFunction(RuntimeFunc::GetLogicDevIdByUserDevId);
    if (func != nullptr) {
        rtError_t(*getLogicDevIdFunc)(const int32_t, int32_t* const) =
            reinterpret_cast<rtError_t(*)(const int32_t, int32_t* const)>(func);
        return getLogicDevIdFunc(userDevId, logicDevId);
    }
#endif
    return StubGetLogicDevIdByUserDevId(userDevId, logicDevId);
}

rtError_t RuntimeStreamCreate(rtStream_t *stm, int32_t priority) {
#ifdef BUILD_WITH_CANN
    void *func = RuntimeAdapter::Instance().GetFunction(RuntimeFunc::StreamCreate);
    if (func != nullptr) {
        rtError_t(*streamCreateFunc)(rtStream_t*, int32_t) = reinterpret_cast<rtError_t(*)(rtStream_t*, int32_t)>(func);
        return streamCreateFunc(stm, priority);
    }
#endif
    return StubStreamCreate(stm, priority);
}

rtError_t RuntimeStreamDestroy(rtStream_t stm) {
#ifdef BUILD_WITH_CANN
    void *func = RuntimeAdapter::Instance().GetFunction(RuntimeFunc::StreamDestroy);
    if (func != nullptr) {
        rtError_t(*streamDestroyFunc)(rtStream_t) = reinterpret_cast<rtError_t(*)(rtStream_t)>(func);
        return streamDestroyFunc(stm);
    }
#endif
    return StubStreamDestroy(stm);
}

rtError_t RuntimeStreamAddToModel(rtStream_t stm, rtModel_t captureMdl) {
#ifdef BUILD_WITH_CANN
    void *func = RuntimeAdapter::Instance().GetFunction(RuntimeFunc::StreamAddToModel);
    if (func != nullptr) {
        rtError_t(*streamAddToModelFunc)(rtStream_t, rtModel_t) =
            reinterpret_cast<rtError_t(*)(rtStream_t, rtModel_t)>(func);
        return streamAddToModelFunc(stm, captureMdl);
    }
#endif
    return StubStreamAddToModel(stm, captureMdl);
}

rtError_t RuntimeStreamSynchronize(rtStream_t stm) {
#ifdef BUILD_WITH_CANN
    void *func = RuntimeAdapter::Instance().GetFunction(RuntimeFunc::StreamSynchronize);
    if (func != nullptr) {
        rtError_t(*streamSyncFunc)(rtStream_t) = reinterpret_cast<rtError_t(*)(rtStream_t)>(func);
        return streamSyncFunc(stm);
    }
#endif
    return StubStreamSynchronize(stm);
}

rtError_t RuntimeDevBinaryUnRegister(void *handle) {
#ifdef BUILD_WITH_CANN
    void *func = RuntimeAdapter::Instance().GetFunction(RuntimeFunc::DevBinaryUnRegister);
    if (func != nullptr) {
        rtError_t(*devBinaryUnRegisterFunc)(void*) = reinterpret_cast<rtError_t(*)(void*)>(func);
        return devBinaryUnRegisterFunc(handle);
    }
#endif
    return StubDevBinaryUnRegister(handle);
}

rtError_t RuntimeRegisterAllKernel(const rtDevBinary_t *bin, void **hdl) {
#ifdef BUILD_WITH_CANN
    void *func = RuntimeAdapter::Instance().GetFunction(RuntimeFunc::RegisterAllKernel);
    if (func != nullptr) {
        rtError_t(*registerAllKernelFunc)(const rtDevBinary_t*, void**) =
            reinterpret_cast<rtError_t(*)(const rtDevBinary_t*, void**)>(func);
        return registerAllKernelFunc(bin, hdl);
    }
#endif
    return StubRegisterAllKernel(bin, hdl);
}

rtError_t RuntimeKernelLaunchWithHandleV2(void *hdl, const uint64_t tilingKey, uint32_t numBlocks,
                                          rtArgsEx_t *argsInfo, rtSmDesc_t *smDesc, rtStream_t stm,
                                          const rtTaskCfgInfo_t *cfgInfo) {
#ifdef BUILD_WITH_CANN
    void *func = RuntimeAdapter::Instance().GetFunction(RuntimeFunc::KernelLaunchWithHandleV2);
    if (func != nullptr) {
        rtError_t(*kernelLaunchFunc)(void*, const uint64_t, uint32_t, rtArgsEx_t*, rtSmDesc_t*, rtStream_t, const rtTaskCfgInfo_t*) =
            reinterpret_cast<rtError_t(*)(void*, const uint64_t, uint32_t, rtArgsEx_t*, rtSmDesc_t*, rtStream_t, const rtTaskCfgInfo_t*)>(func);
        return kernelLaunchFunc(hdl, tilingKey, numBlocks, argsInfo, smDesc, stm, cfgInfo);
    }
#endif
    return StubKernelLaunchWithHandleV2(hdl, tilingKey, numBlocks, argsInfo, smDesc, stm, cfgInfo);
}


rtError_t RuntimeAicpuKernelLaunchExWithArgs(const uint32_t kernelType, const char_t * const opName,
                                             const uint32_t numBlocks, const rtAicpuArgsEx_t *argsInfo,
                                             rtSmDesc_t * const smDesc, const rtStream_t stm, const uint32_t flags) {
#ifdef BUILD_WITH_CANN
    void *func = RuntimeAdapter::Instance().GetFunction(RuntimeFunc::AicpuKernelLaunchExWithArgs);
    if (func != nullptr) {
        rtError_t(*aicpuKernelLaunchFunc)(const uint32_t, const char_t* const, const uint32_t, const rtAicpuArgsEx_t*, rtSmDesc_t* const, const rtStream_t, const uint32_t) =
            reinterpret_cast<rtError_t(*)(const uint32_t, const char_t* const, const uint32_t, const rtAicpuArgsEx_t*, rtSmDesc_t* const, const rtStream_t, const uint32_t)>(func);
        return aicpuKernelLaunchFunc(kernelType, opName, numBlocks, argsInfo, smDesc, stm, flags);
    }
#endif
    return StubAicpuKernelLaunchExWithArgs(kernelType, opName, numBlocks, argsInfo, smDesc, stm, flags);
}

}
