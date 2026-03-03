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
 * \file acl_stubs.cpp
 * \brief
 */

#include "adapter/stubs/acl_stubs.h"
#include "tilefwk/pypto_fwk_log.h"

namespace npu::tile_fwk {
aclError StubAclInit(const char *configPath) {
    ADAPTER_LOGD("Enter stub function of aclInit.");
    (void)configPath;
    return ACL_SUCCESS;
}

aclError StubAclFinalize() {
    ADAPTER_LOGD("Enter stub function of aclFinalize.");
    return ACL_SUCCESS;
}

aclError StubRtMemcpy(void *dst, size_t destMax, const void *src, size_t count, aclrtMemcpyKind kind) {
    ADAPTER_LOGD("Enter stub function of aclrtMemcpy.");
    (void)dst;
    (void)destMax;
    (void)src;
    (void)count;
    (void)kind;
    return ACL_SUCCESS;
}

aclError StubRtSetDevice(int32_t deviceId) {
    ADAPTER_LOGD("Enter stub function of aclrtSetDevice.");
    (void)deviceId;
    return ACL_SUCCESS;
}

aclError StubRtResetDevice(int32_t deviceId) {
    ADAPTER_LOGD("Enter stub function of aclrtResetDevice.");
    (void)deviceId;
    return ACL_SUCCESS;
}

aclError StubRtCreateEvent(aclrtEvent *event) {
    ADAPTER_LOGD("Enter stub function of aclrtCreateEvent.");
    (void)event;
    return ACL_SUCCESS;
}

aclError StubRtRecordEvent(aclrtEvent event, aclrtStream stream) {
    ADAPTER_LOGD("Enter stub function of aclrtRecordEvent.");
    (void)event;
    (void)stream;
    return ACL_SUCCESS;
}

aclError StubRtCreateEventExWithFlag(aclrtEvent *event, uint32_t flag) {
    ADAPTER_LOGD("Enter stub function of aclrtCreateEventExWithFlag.");
    (void)event;
    (void)flag;
    return ACL_SUCCESS;
}

aclError StubRtStreamWaitEvent(aclrtStream stream, aclrtEvent event) {
    ADAPTER_LOGD("Enter stub function of aclrtStreamWaitEvent.");
    (void)stream;
    (void)event;
    return ACL_SUCCESS;
}

aclError StubRtGetStreamResLimit(aclrtStream stream, aclrtDevResLimitType type, uint32_t *value) {
    ADAPTER_LOGD("Enter stub function of aclrtGetStreamResLimit.");
    (void)stream;
    (void)type;
    *value = 20;
    return ACL_SUCCESS;
}

aclError StubRtGetStreamAttribute(aclrtStream stream, aclrtStreamAttr stmAttrType, aclrtStreamAttrValue *value) {
    ADAPTER_LOGD("Enter stub function of aclrtGetStreamAttribute.");
    (void)stream;
    (void)stmAttrType;
    if (value != nullptr) {
        value->cacheOpInfoSwitch = 1;
    }
    return ACL_SUCCESS;
}

aclError StubRtCacheLastTaskOpInfo(const void * const infoPtr, size_t infoSize) {
    ADAPTER_LOGD("Enter stub function of aclrtCacheLastTaskOpInfo.");
    (void)infoPtr;
    (void)infoSize;
    return ACL_SUCCESS;
}

aclError StubRtSetExceptionInfoCallback(aclrtExceptionInfoCallback callback) {
    ADAPTER_LOGD("Enter stub function of aclrtSetExceptionInfoCallback.");
    (void)callback;
    return ACL_SUCCESS;
}

aclError StubMdlRICaptureGetInfo(aclrtStream stream, aclmdlRICaptureStatus *status, aclmdlRI *modelRI) {
    ADAPTER_LOGD("Enter stub function of aclmdlRICaptureGetInfo.");
    (void)stream;
    (void)status;
    (void)modelRI;
    return ACL_SUCCESS;
}

aclError StubMdlRICaptureThreadExchangeMode(aclmdlRICaptureMode *mode) {
    ADAPTER_LOGD("Enter stub function of aclmdlRICaptureThreadExchangeMode.");
    (void)mode;
    return ACL_SUCCESS;
}
}