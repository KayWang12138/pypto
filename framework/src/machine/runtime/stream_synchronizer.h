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
 * \file stream_synchronizer.h
 * \brief Manages stream synchronization and cross-stream event dependencies.
 */

#pragma once

#include "adapter/api/acl_define.h"
#include "adapter/api/runtime_define.h"

namespace npu::tile_fwk {

class StreamSynchronizer {
public:
    StreamSynchronizer();
    ~StreamSynchronizer();

    int Init();

    // Synchronize all three streams and return combined error code
    int Synchronize(RtStream aicpuStream, RtStream ctrlStream, RtStream aicoreStream);

    // Pre-launch sync: establish aicore -> sche/ctrl dependency
    int PreSync(RtStream scheStream, RtStream ctrlStream, RtStream aicoreStream);

    // Post-launch sync: establish aicpu -> aicore dependency
    int PostSync(RtStream aicpuStream, RtStream aicoreStream);

private:
    void SyncStreams(RtStream fromStream, RtStream toStream, bool useSyncFlag);

    AclRtEvent event_{nullptr};
    bool initialized_{false};
};

} // namespace npu::tile_fwk
