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
 * \file latency_estimator.h
 * \brief
 */

#ifndef PASS_ESTIMATE_LATENCY_H
#define PASS_ESTIMATE_LATENCY_H

#include "ooo_scheduler.h"
#include <vector>

namespace npu::tile_fwk {
class LatencyEstimator : public OoOScheduler {
public:
    LatencyEstimator() {}
    LatencyEstimator(std::vector<Operation*>& newTaskList, std::vector<Operation*>& newOperations,
        CoreLocationType coreLocation);
    ~LatencyEstimator() {}

    Status LatencyEstimatorMainLoop();

protected:
    Status PreMainLoop() override;
    Status PostMainLoop() override;
    void LaunchReadyIssue() override;
    Status FreeBuffer(Operation* op) override;
    Status ExecuteAllocIssue(uint64_t& commitCnt, MemoryType memType, IssueQueue& pipe) override;
    Status LaunchIssueStage(int& nextCycle) override;
    Status SpillOnBlock() override;

private:
    std::vector<Operation*> taskList;
    std::vector<Operation*> operations;
    std::set<int> spillblockMemIds;
    CoreLocationType coreLocation_;

    void InitMemWithoutAlloc();
    void InitLatencyEstimator();
    void InitLatencyAllocIssueQueues();
};
} // namespace npu::tile_fwk
#endif // PASS_ESTIMATE_LATENCY_H
