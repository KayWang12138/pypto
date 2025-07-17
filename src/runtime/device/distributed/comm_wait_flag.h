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
 * \file comm_wait_flag.h
 * \brief
 */

#ifndef COMM_WAIT_FLAG_H
#define COMM_WAIT_FLAG_H

#include "interface/utils/common.h"
#include "interface/cache/core_func_data.h"
#include <cstdint>
#include <vector>
#include <deque>
#include <map>

namespace npu::tile_fwk {
namespace Distributed {
class FlagPoller {
public:
    FlagPoller() {};
    ~FlagPoller() {};
    void Init(uint32_t rankId, uint32_t rankSize, uint8_t *winFlag);
    void EnqueueOp(uint64_t taskId, uint32_t rankShape, uint32_t rankOffset, uint32_t tileIndex);
    void PollCompleted(std::vector<uint64_t> &completed);

private:
    void SetUndoFlag(uint32_t startIndex, uint32_t endIndex);
    void ProcessFlag();
    void ProcessBlock(uint8_t *winFlag, uint8_t *undoFlag, size_t start);
    void ProcessRemaining(uint8_t *winFlag, uint8_t *undoFlag, size_t remaining, size_t start);
    struct OpInfo {
        uint64_t taskId;
        uint32_t flagCount;
        uint32_t offset;
    };
    uint32_t rankId_;
    uint32_t rankSize_;
    uint8_t *winFlag_{nullptr};
    std::vector<uint8_t> undoFlag_;
    std::vector<OpInfo> opInfo_;
    size_t opCount_{0};
    std::deque<size_t> readyQueue_;
};

class CommWaitFlag {
public:
    CommWaitFlag() {};
    ~CommWaitFlag() {};
    void Init(DeviceTask *deviceTask);
    void EnqueueOp(uint64_t taskId, uint64_t *paramList, uint32_t paramSize);
    void PollCompleted(std::vector<uint64_t> &completed);

private:
    bool Prepare(uint32_t groupIndex);
    uint64_t *hcclContextAddr_{nullptr};
    uint32_t commGroupNum_{0};
    FlagPoller flagPoller_[DIST_COMM_GROUP_NUM];
    bool inited_[DIST_COMM_GROUP_NUM]{false};
};

} // namespace Distributed
} // namespace npu::tile_fwk
#endif // COMM_WAIT_FLAG_H
