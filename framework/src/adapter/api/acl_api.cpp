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
 * \file acl_api.cpp
 * \brief
 */

#include "adapter/api/acl_api.h"

#ifdef EXECUTE_WITH_CANN
#include "adapter/manager/adapter_manager.h"
#endif
#include "adapter/stubs/acl_stubs.h"

namespace npu::tile_fwk {
aclError AclInit(const char *configPath) {
#ifdef EXECUTE_WITH_CANN
    void *func = AdapterManager::Instance().GetAclAdapter().GetFunction(AclFunc::Init);
    if (func != nullptr) {
        aclError(*aclFunc)(const char*) = reinterpret_cast<aclError(*)(const char*)>(func);
        return aclFunc(configPath);
    }
#endif
    return StubAclInit(configPath);
}
aclError AclFinalize() {
#ifdef EXECUTE_WITH_CANN
    void *func = AdapterManager::Instance().GetAclAdapter().GetFunction(AclFunc::Finalize);
    if (func != nullptr) {
        aclError(*aclFunc)(void) = reinterpret_cast<aclError(*)(void)>(func);
        return aclFunc();
    }
#endif
    return StubAclFinalize();
}

aclError AclRtMemcpy(void *dst, size_t destMax, const void *src, size_t count, aclrtMemcpyKind kind) {
#ifdef EXECUTE_WITH_CANN
    void *func = AdapterManager::Instance().GetAclAdapter().GetFunction(AclFunc::RtMemcpy);
    if (func != nullptr) {
        aclError(*aclFunc)(void*, size_t, const void*, size_t, aclrtMemcpyKind) =
            reinterpret_cast<aclError(*)(void*, size_t, const void*, size_t, aclrtMemcpyKind)>(func);
        return aclFunc(dst, destMax, src, count, kind);
    }
#endif
    return StubRtMemcpy(dst, destMax, src, count, kind);
}

aclError AclRtSetDevice(int32_t deviceId) {
#ifdef EXECUTE_WITH_CANN
    void *func = AdapterManager::Instance().GetAclAdapter().GetFunction(AclFunc::RtSetDevice);
    if (func != nullptr) {
        aclError(*aclFunc)(int32_t) = reinterpret_cast<aclError(*)(int32_t)>(func);
        return aclFunc(deviceId);
    }
#endif
    return StubRtSetDevice(deviceId);
}

aclError AclRtCreateEvent(aclrtEvent *event) {
#ifdef EXECUTE_WITH_CANN
    void *func = AdapterManager::Instance().GetAclAdapter().GetFunction(AclFunc::RtCreateEvent);
    if (func != nullptr) {
        aclError(*aclFunc)(aclrtEvent*) = reinterpret_cast<aclError(*)(aclrtEvent*)>(func);
        return aclFunc(event);
    }
#endif
    return StubRtCreateEvent(event);
}

aclError AclRtRecordEvent(aclrtEvent event, aclrtStream stream) {
#ifdef EXECUTE_WITH_CANN
    void *func = AdapterManager::Instance().GetAclAdapter().GetFunction(AclFunc::RtRecordEvent);
    if (func != nullptr) {
        aclError(*aclFunc)(aclrtEvent, aclrtStream) = reinterpret_cast<aclError(*)(aclrtEvent, aclrtStream)>(func);
        return aclFunc(event, stream);
    }
#endif
    return StubRtRecordEvent(event, stream);
}

aclError AclRtCreateEventExWithFlag(aclrtEvent *event, uint32_t flag) {
#ifdef EXECUTE_WITH_CANN
    void *func = AdapterManager::Instance().GetAclAdapter().GetFunction(AclFunc::RtCreateEventExWithFlag);
    if (func != nullptr) {
        aclError(*aclFunc)(aclrtEvent*, uint32_t) = reinterpret_cast<aclError(*)(aclrtEvent*, uint32_t)>(func);
        return aclFunc(event, flag);
    }
#endif
    return StubRtCreateEventExWithFlag(event, flag);
}

aclError AclRtStreamWaitEvent(aclrtStream stream, aclrtEvent event) {
#ifdef EXECUTE_WITH_CANN
    void *func = AdapterManager::Instance().GetAclAdapter().GetFunction(AclFunc::RtStreamWaitEvent);
    if (func != nullptr) {
        aclError(*aclFunc)(aclrtStream, aclrtEvent) = reinterpret_cast<aclError(*)(aclrtStream, aclrtEvent)>(func);
        return aclFunc(stream, event);
    }
#endif
    return StubRtStreamWaitEvent(stream, event);
}

aclError AclRtGetStreamResLimit(aclrtStream stream, aclrtDevResLimitType type, uint32_t *value) {
#ifdef EXECUTE_WITH_CANN
    void *func = AdapterManager::Instance().GetAclAdapter().GetFunction(AclFunc::RtGetStreamResLimit);
    if (func != nullptr) {
        aclError(*aclFunc)(aclrtStream, aclrtDevResLimitType, uint32_t*) =
            reinterpret_cast<aclError(*)(aclrtStream, aclrtDevResLimitType, uint32_t*)>(func);
        return aclFunc(stream, type, value);
    }
#endif
    return StubRtGetStreamResLimit(stream, type, value);
}

aclError AclRtGetStreamAttribute(aclrtStream stream, aclrtStreamAttr stmAttrType, aclrtStreamAttrValue *value) {
#ifdef EXECUTE_WITH_CANN
    void *func = AdapterManager::Instance().GetAclAdapter().GetFunction(AclFunc::RtGetStreamAttribute);
    if (func != nullptr) {
        aclError(*aclFunc)(aclrtStream, aclrtStreamAttr, aclrtStreamAttrValue*) =
            reinterpret_cast<aclError(*)(aclrtStream, aclrtStreamAttr, aclrtStreamAttrValue*)>(func);
        return aclFunc(stream, stmAttrType, value);
    }
#endif
    return StubRtGetStreamAttribute(stream, stmAttrType, value);
}

aclError AclRtCacheLastTaskOpInfo(const void * const infoPtr, size_t infoSize) {
#ifdef EXECUTE_WITH_CANN
    void *func = AdapterManager::Instance().GetAclAdapter().GetFunction(AclFunc::RtCacheLastTaskOpInfo);
    if (func != nullptr) {
        aclError(*aclFunc)(const void* const, size_t) = reinterpret_cast<aclError(*)(const void* const, size_t)>(func);
        return aclFunc(infoPtr, infoSize);
    }
#endif
    return StubRtCacheLastTaskOpInfo(infoPtr, infoSize);
}

aclError AclRtSetExceptionInfoCallback(aclrtExceptionInfoCallback callback) {
#ifdef EXECUTE_WITH_CANN
    void *func = AdapterManager::Instance().GetAclAdapter().GetFunction(AclFunc::RtSetExceptionInfoCallback);
    if (func != nullptr) {
        aclError(*aclFunc)(aclrtExceptionInfoCallback) =
            reinterpret_cast<aclError(*)(aclrtExceptionInfoCallback)>(func);
        return aclFunc(callback);
    }
#endif
    return StubRtSetExceptionInfoCallback(callback);
}

aclError AclMdlRICaptureGetInfo(aclrtStream stream, aclmdlRICaptureStatus *status, aclmdlRI *modelRI) {
#ifdef EXECUTE_WITH_CANN
    void *func = AdapterManager::Instance().GetAclAdapter().GetFunction(AclFunc::MdlRICaptureGetInfo);
    if (func != nullptr) {
        aclError(*aclFunc)(aclrtStream, aclmdlRICaptureStatus*, aclmdlRI*) =
            reinterpret_cast<aclError(*)(aclrtStream, aclmdlRICaptureStatus*, aclmdlRI*)>(func);
        return aclFunc(stream, status, modelRI);
    }
#endif
    return StubMdlRICaptureGetInfo(stream, status, modelRI);
}

aclError AclMdlRICaptureThreadExchangeMode(aclmdlRICaptureMode *mode) {
#ifdef EXECUTE_WITH_CANN
    void *func = AdapterManager::Instance().GetAclAdapter().GetFunction(AclFunc::MdlRICaptureThreadExchangeMode);
    if (func != nullptr) {
        aclError(*aclFunc)(aclmdlRICaptureMode*) = reinterpret_cast<aclError(*)(aclmdlRICaptureMode*)>(func);
        return aclFunc(mode);
    }
#endif
    return StubMdlRICaptureThreadExchangeMode(mode);
}
}