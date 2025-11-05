/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file operation_common.h
 * \brief
 */

#ifndef INTERFACE_MAIN_OPERATION_COMMON_H
#define INTERFACE_MAIN_OPERATION_COMMON_H

#include "interface/utils/common.h"

namespace npu::tile_fwk {
class OpSyncQueue {
public:
    OpSyncQueue() {}
    OpSyncQueue(PipeType pipeId, PipeType trigPipeId, CoreType coreType, CoreType tirgCoreType, int evid)
        : pipeId_(pipeId), trigPipeId_(trigPipeId), coreType_(coreType), trigCoreType_(tirgCoreType), eventId_(evid) {}

    OpSyncQueue(int bufid, const std::vector<int> &offset, CoreType coreType, CoreType tirgCoreType)
        : coreType_(coreType), trigCoreType_(tirgCoreType), gMBufId(bufid), offset_(offset) {}

    PipeType pipeId_{PIPE_S};
    PipeType trigPipeId_{PIPE_S};
    CoreType coreType_{CoreType::AIV};
    CoreType trigCoreType_{CoreType::AIV};
    int eventId_{0};
    int gMBufId{0};
    std::vector<int> offset_;

    Json ToJson() const {
        Json j;
        j["pipe_id"] = pipeId_;
        j["trig_pipe"] = trigPipeId_;
        j["core_type"] = static_cast<int>(coreType_);
        j["tri_core_type"] = static_cast<int>(trigCoreType_);
        j["event_id"] = eventId_;
        j["gm_buf_id"] = gMBufId;
        j["offset"] = offset_;
        return j;
    }

    void FromJson(const Json &j) {
        pipeId_ = static_cast<PipeType>(j["pipe_id"].get<int>());
        trigPipeId_ = static_cast<PipeType>(j["trig_pipe"].get<int>());
        coreType_ = static_cast<CoreType>(j["core_type"].get<int>());
        trigCoreType_ = static_cast<CoreType>(j["tri_core_type"].get<int>());
        eventId_ = j["event_id"].get<int>();
        gMBufId = j["gm_buf_id"].get<int>();
        offset_ = j["offset"].get<std::vector<int>>();
    }

    std::string Dump() const {
        std::ostringstream oss;
        oss << GetPipeTypeDict().Find(pipeId_) << "," << GetPipeTypeDict().Find(trigPipeId_) << "," << eventId_;
        return oss.str();
    }
};
}

#endif // INTERFACE_MAIN_OPERATION_COMMON_H
