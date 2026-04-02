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
 * \file acl_stubs.h
 * \brief
 */

#pragma once

#include "adapter/api/acl_define.h"

namespace npu::tile_fwk {
aclError StubAclInit(const char *configPath);
aclError StubAclFinalize();
aclError StubRtMemcpy(void *dst, size_t destMax, const void *src, size_t count, aclrtMemcpyKind kind);
aclError StubRtSetDevice(int32_t deviceId);
aclError StubRtResetDevice(int32_t deviceId);
aclError StubRtCreateEvent(aclrtEvent *event);
aclError StubRtRecordEvent(aclrtEvent event, aclrtStream stream);
aclError StubRtCreateEventExWithFlag(aclrtEvent *event, uint32_t flag);
aclError StubRtStreamWaitEvent(aclrtStream stream, aclrtEvent event);
aclError StubRtGetStreamResLimit(aclrtStream stream, aclrtDevResLimitType type, uint32_t *value);
aclError StubRtGetStreamAttribute(aclrtStream stream, aclrtStreamAttr stmAttrType, aclrtStreamAttrValue *value);
aclError StubRtCacheLastTaskOpInfo(const void * const infoPtr, size_t infoSize);
aclError StubRtSetExceptionInfoCallback(aclrtExceptionInfoCallback callback);
aclError StubMdlRICaptureGetInfo(aclrtStream stream, aclmdlRICaptureStatus *status, aclmdlRI *modelRI);
aclError StubMdlRICaptureThreadExchangeMode(aclmdlRICaptureMode *mode);
}