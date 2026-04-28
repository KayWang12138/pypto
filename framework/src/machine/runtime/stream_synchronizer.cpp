/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file stream_synchronizer.cpp
 * \brief Implementation of stream synchronization primitives.
 */

#include "machine/runtime/stream_synchronizer.h"
#include "adapter/api/acl_api.h"
#include "adapter/api/runtime_api.h"
#include "tilefwk/pypto_fwk_log.h"
#include "tilefwk/error_code.h"

namespace npu::tile_fwk {

StreamSynchronizer::StreamSynchronizer() = default;

StreamSynchronizer::~StreamSynchronizer()
{
    if (event_ != nullptr) {
        AclRtDestroyEvent(event_);
        event_ = nullptr;
    }
}

int StreamSynchronizer::Init()
{
    int rc = AclRtCreateEventExWithFlag(&event_, ACL_EVENT_SYNC);
    if (rc < 0) {
        MACHINE_LOGE(RtErr::RT_EVENT_FAILED, "AclRtCreateEvent failed.");
        return -1;
    }
    initialized_ = true;
    return 0;
}

int StreamSynchronizer::Synchronize(RtStream aicpuStream, RtStream ctrlStream, RtStream aicoreStream)
{
    int rcAicore = RuntimeStreamSynchronize(aicoreStream);
    int rcAicpu = RuntimeStreamSynchronize(aicpuStream);
    int rcCtrl = 0;
    if (ctrlStream != nullptr) {
        rcCtrl = RuntimeStreamSynchronize(ctrlStream);
    }
    if (rcAicore != 0 || rcAicpu != 0 || rcCtrl != 0) {
        MACHINE_LOGW("sync stream failed aicpu:%d aicore:%d ctrl cpu:%d", rcAicpu, rcAicore, rcCtrl);
    }
    return rcAicore + rcAicpu + rcCtrl;
}

int StreamSynchronizer::PreSync(RtStream scheStream, RtStream ctrlStream, RtStream aicoreStream)
{
    if (!initialized_) {
        MACHINE_LOGE(RtErr::RT_EVENT_FAILED, "StreamSynchronizer not initialized");
        return -1;
    }
    int rc = AclRtRecordEvent(event_, aicoreStream);
    if (rc < 0) {
        MACHINE_LOGE(RtErr::RT_EVENT_FAILED, "AclRtRecordEvent failed %d\n", rc);
        return rc;
    }
    rc = AclRtStreamWaitEvent(scheStream, event_);
    if (rc < 0) {
        MACHINE_LOGE(RtErr::RT_EVENT_FAILED, "AclRtStreamWaitEvent failed %d\n", rc);
        return rc;
    }
    rc = AclRtStreamWaitEvent(ctrlStream, event_);
    if (rc < 0) {
        MACHINE_LOGE(RtErr::RT_EVENT_FAILED, "AclRtStreamWaitEvent failed %d\n", rc);
        return rc;
    }
    return 0;
}

int StreamSynchronizer::PostSync(RtStream aicpuStream, RtStream aicoreStream)
{
    SyncStreams(aicpuStream, aicoreStream, true);
    return 0;
}

void StreamSynchronizer::SyncStreams(RtStream fromStream, RtStream toStream, bool useSyncFlag)
{
    AclRtEvent event;
    int rc;

    if (useSyncFlag) {
        rc = AclRtCreateEventExWithFlag(&event, ACL_EVENT_SYNC);
    } else {
        rc = AclRtCreateEvent(&event);
    }

    if (rc < 0) {
        MACHINE_LOGI("CreateEvent failed rc=%d, useSyncFlag=%d", rc, useSyncFlag);
    }

    rc = AclRtRecordEvent(event, fromStream);
    if (rc < 0) {
        MACHINE_LOGI("RecordEvent failed rc=%d", rc);
    }

    rc = AclRtStreamWaitEvent(toStream, event);
    if (rc < 0) {
        MACHINE_LOGI("StreamWaitEvent failed rc=%d", rc);
    }
}

} // namespace npu::tile_fwk
