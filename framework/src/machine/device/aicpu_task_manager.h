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
 * \file aicpu_task_manager.h
 * \brief
 */

#pragma once
#include "machine/utils/machine_ws_intf.h"
#include "tilefwk/core_func_data.h"
#include "interface/operation/opcode.h"
#include "interface/utils/common.h"
#include <functional>
#include <atomic>
#include <vector>
#include <array>
#include <malloc.h>

namespace npu::tile_fwk {
constexpr uint32_t AICPU_QUEUE_SIZE = 1024;

class AicpuTaskManager {
public:
    enum TaskType {
        TASK_TYPE_NUM,
    };
    using InitCallBack = std::function<void(DeviceTask *)>;
    using EnqueueOpCallBack = std::function<void(uint64_t, uint64_t *, uint32_t)>;
    using PollCompletedCallBack = std::function<void(std::vector<uint64_t> &)>;

    inline void TaskCallBackRegister() {}

    AicpuTaskManager() {};
    ~AicpuTaskManager() {};

    template <typename T>
    inline void TaskCallBackResigter(TaskType taskType, T &obj) {
        auto index = static_cast<uint32_t>(taskType);
        initCallBack_[index] = std::bind(&T::Init, &obj, std::placeholders::_1);
        enqueueOpCallBack_[index] =
            std::bind(&T::EnqueueOp, &obj, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
        pollCompletedCallBack_[index] = std::bind(&T::PollCompleted, &obj, std::placeholders::_1);
    }

    inline TaskType GetTaskType() {
        return TaskType::TASK_TYPE_NUM;
    }

    // 每个AICPU都会调用
    inline void TaskEnqueue(uint64_t taskId) {
        readyQueue_->lock();
        readyQueue_->push(&taskId);
        readyQueue_->unlock();
    }

    // 仅AICPU_0会调用
    void Init(DeviceTask *deviceTask) {
        funcInfo_ = reinterpret_cast<CoreFunctionWsAddr *>(deviceTask->coreFuncData.coreFunctionWsAddr);
        readyQueue_ = reinterpret_cast<StaticReadyCoreFunctionQueue *>(deviceTask->readyAicpuFunctionQue);
        for (auto &init : initCallBack_) {
            init(deviceTask);
        }
    }

    // 仅AICPU_0会调用
    inline uint64_t  TaskProcess() {
        readyQueue_->lock();
        auto taskSet = readyQueue_->pop(readyQueue_->wasSize());
        readyQueue_->unlock();

        const uint64_t* taskSetAddress = taskSet.first;
        const size_t taskSetCount = taskSet.second;
        for (uint32_t i = 0; i < taskSetCount; ++i) {
            TaskDispatch(taskSetAddress[i]);
        }

        return taskSetCount;
    }

    inline std::vector<uint64_t> TaskPoll() {
        std::vector<uint64_t> completed;
        for (auto &pollCompleted : pollCompletedCallBack_) {
            pollCompleted(completed);
        }
        return completed;
    }

    inline bool Finished() {
        readyQueue_->lock();
        auto fin = readyQueue_->wasEmpty();
        readyQueue_->unlock();
        return fin;
    }

private:

    inline void TaskDispatch(uint64_t elem) {
        auto topo = reinterpret_cast<CoreFunctionTopo *>(funcInfo_[elem].topoAddr);
        auto taskType = GetTaskType();
        if (taskType < TaskType::TASK_TYPE_NUM) {
            auto enqueueOp = enqueueOpCallBack_[static_cast<uint64_t>(taskType)];
            enqueueOp(elem, reinterpret_cast<uint64_t *>(&topo->depIds[topo->depNum]), topo->extParamNum);
        }
    }

    StaticReadyCoreFunctionQueue *readyQueue_{nullptr};

    std::array<InitCallBack, TaskType::TASK_TYPE_NUM> initCallBack_;
    std::array<EnqueueOpCallBack, TaskType::TASK_TYPE_NUM> enqueueOpCallBack_;
    std::array<PollCompletedCallBack, TaskType::TASK_TYPE_NUM> pollCompletedCallBack_;

    CoreFunctionWsAddr *funcInfo_{nullptr};
};
} // namespace npu::tile_fwk
