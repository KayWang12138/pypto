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
 * \file aicpu_task_manager.h
 * \brief
 */

#pragma once

#include <functional>
#include <atomic>
#include <vector>
#include <array>
#include <malloc.h>
#include <queue>

#include "machine/utils/dynamic/dev_encode.h"
#include "machine/device/dynamic/device_context.h"
#include "machine/device/distributed/common.h"
#include "machine/device/distributed/shmem_wait_until.h"
#include "machine/utils/machine_ws_intf.h"
#include "tilefwk/core_func_data.h"
#include "interface/operation/opcode.h"
#include "interface/utils/common.h"
#include "tileop/hccl_context.h"

namespace npu::tile_fwk::dynamic {
constexpr uint32_t AICPU_QUEUE_SIZE = 1024;

class AicpuTaskManager {
public:
    enum TaskType {
        SHMEM_WAIT_UNTIL = 0,
        TASK_TYPE_NUM,
    };
    using InitCallBack = std::function<void(DeviceTask *)>;
    using EnqueueOpCallBack = std::function<void(uint64_t, npu::tile_fwk::Distributed::TensorInfo&)>;
    using PollCompletedCallBack = std::function<void(std::vector<uint64_t> &)>;

    inline void TaskCallBackRegister() {}

    AicpuTaskManager() {
        TaskCallBackResigter<npu::tile_fwk::Distributed::ShmemWaitUntil>(TaskType::SHMEM_WAIT_UNTIL, shmemWaitUntil_);
    };
    ~AicpuTaskManager() {};

    template <typename T>
    inline void TaskCallBackResigter(TaskType taskType, T &obj) {
        auto index = static_cast<uint32_t>(taskType);
        initCallBack_[index] = std::bind(&T::Init, &obj, std::placeholders::_1);
        enqueueOpCallBack_[index] =
            std::bind(&T::EnqueueOp, &obj, std::placeholders::_1, std::placeholders::_2);
        pollCompletedCallBack_[index] = std::bind(&T::PollCompleted, &obj, std::placeholders::_1);
    }

    // 每个AICPU都会调用
    inline void TaskEnqueue(uint64_t taskId) {
        ReadyQueueLock();
        readyQueue_->elem[readyQueue_->tail] = taskId;
        readyQueue_->tail += 1;
        ReadyQueueUnLock();
    }

    // 仅AICPU_0会调用
    void Init(DynDeviceTask *deviceTask) {
        curDevTask_ = deviceTask;
        funcDataList_ = reinterpret_cast<DynFuncData*>(curDevTask_->dynFuncData + 1);
        readyQueue_ = reinterpret_cast<ReadyCoreFunctionQueue *>(deviceTask->devTask.readyAicpuFunctionQue);
        hcclContextAddr_ = funcDataList_->hcclContext;
        for (auto &init : initCallBack_) {
            init(&(deviceTask->devTask));
        }
    }

    // 仅AICPU_0会调用
    inline uint64_t  TaskProcess() {
        ReadyQueueLock();
        uint64_t taskIdx = readyQueue_->head;
        uint64_t taskCount = readyQueue_->tail - readyQueue_->head;
        readyQueue_->head += taskCount;
        ReadyQueueUnLock();

        for (uint32_t i = 0; i < taskCount; ++i) {
            TaskDispatch(readyQueue_->elem[taskIdx + i]);
        }
        return taskCount;
    }

    inline std::vector<uint64_t> TaskPoll() {
        std::vector<uint64_t> completed;
        for (auto &pollCompleted : pollCompletedCallBack_) {
            pollCompleted(completed);
        }
        return completed;
    }

    inline bool Finished() {
        ReadyQueueLock();
        auto fin = readyQueue_->head == readyQueue_->tail;
        ReadyQueueUnLock();
        return fin;
    }

private:
    inline void ReadyQueueLock() {
        while (!__sync_bool_compare_and_swap(&readyQueue_->lock, 0, 1))
            ;
    }

    inline void ReadyQueueUnLock() {
        while (!__sync_bool_compare_and_swap(&readyQueue_->lock, 1, 0))
            ;
    }

    inline TaskType GetTaskType(uint64_t taskId) {
        auto funcId = FuncID(taskId);
        auto &funcDup = curDevTask_->stitchedList[funcId];
        auto opIndex = TaskID(taskId);
        auto extType = funcDup.GetSource()->GetOperationAicpuOpType(opIndex);
        auto taskType = TaskType::TASK_TYPE_NUM;
        switch (extType) {
            case static_cast<uint32_t>(Opcode::OP_SHMEM_WAIT_UNTIL): taskType = TaskType::SHMEM_WAIT_UNTIL; break;
            default: break;
        }
        return taskType;
    }

    inline uint64_t GetCoa(const uint32_t index, __gm__ uint64_t* opAttrs, __gm__ uint64_t* expressionTable)
    {
        constexpr uint64_t valueLength = 63;
        constexpr uint64_t valueMask = (1UL << valueLength) - 1;
        const uint64_t encodedValue = opAttrs[index];
        const bool isExpression = (encodedValue >> valueLength) & 1;
        const uint64_t decodedValue = encodedValue & valueMask;
        return isExpression ? expressionTable[decodedValue] : decodedValue;
    }

    inline std::vector<uint32_t> GetCoaVector(const uint32_t baseIndex, const uint32_t dim, __gm__ uint64_t* opAttrs,
        __gm__ uint64_t* expressionTable)
    {
        std::vector<uint32_t> vec(dim);
        for (uint32_t i = 0; i < dim; ++i) {
            vec[i] = GetCoa(baseIndex + i, opAttrs, expressionTable);
        }
        return vec;
    }

    inline uint64_t GetRawAddr(const uint64_t dstRankId, const uint64_t offsetPerRank) {
        constexpr uint32_t groupIndex = 0;
        struct TileOp::HcclCombinOpParam* hcclOpParam = (struct TileOp::HcclCombinOpParam*)hcclContextAddr_[groupIndex];
        return hcclOpParam->windowsIn[dstRankId] + offsetPerRank * hcclOpParam->rankNum;   
    }

    inline npu::tile_fwk::Distributed::TensorInfo GetTensorInfo(const uint64_t taskId,
        const std::vector<uint32_t>& previousOperandDims)
    {
        auto funcId = FuncID(taskId);
        auto opIndex = TaskID(taskId);
        auto &funcDup = curDevTask_->stitchedList[funcId];
        auto &funcData = funcDataList_[funcId];
        auto opAttrs = &funcData.opAttrs[funcData.opAtrrOffsets[opIndex]];
        auto expressionTable = funcData.exprTbl;

        uint32_t index = 0;
        ++index; // 跳过 function id
        for (uint32_t dim : previousOperandDims) {
            index += 1 + dim * 4; // 1 跳过 rawIndex，dim * 4 跳过 offset、shape、rawShape、dynValidShape
        }

        npu::tile_fwk::Distributed::TensorInfo info;
        ++index; // 跳过 rawIndex
        info.dim = funcDup.GetSource()->GetOperationIOperandInfo(opIndex, 1).GetDim(); // todo: shmemSignal 是第 1 个输入
        info.offset = GetCoaVector(index, info.dim, opAttrs, expressionTable);
        index += info.dim;
        info.shape = GetCoaVector(index, info.dim, opAttrs, expressionTable);
        index += info.dim;
        info.rawShape = GetCoaVector(index, info.dim, opAttrs, expressionTable);
        index += info.dim;
        info.dynValidShape = GetCoaVector(index, info.dim, opAttrs, expressionTable);
        const uint32_t dstRankId = info.offset[0];
        const uint32_t offsetPerRank = 128 * 256 * 4; // todo: 约定在 windowsIn 里先放 data 再放 signal，所以要加上 data 的偏移
        info.rawAddr = GetRawAddr(dstRankId, offsetPerRank);

        return info;
    }

    inline void TaskDispatch(uint64_t elem) {
        auto taskType = GetTaskType(elem);
        if (taskType < TaskType::TASK_TYPE_NUM) {
            auto enqueueOp = enqueueOpCallBack_[static_cast<uint64_t>(taskType)];
            auto tensor = GetTensorInfo(elem, std::vector<uint32_t>{2});
            enqueueOp(elem, tensor);
        }
        tasks_.push(elem);
    }

    ReadyCoreFunctionQueue *readyQueue_{nullptr};

    npu::tile_fwk::Distributed::ShmemWaitUntil shmemWaitUntil_;

    std::array<InitCallBack, TaskType::TASK_TYPE_NUM> initCallBack_;
    std::array<EnqueueOpCallBack, TaskType::TASK_TYPE_NUM> enqueueOpCallBack_;
    std::array<PollCompletedCallBack, TaskType::TASK_TYPE_NUM> pollCompletedCallBack_;

    DynDeviceTask *curDevTask_;
    npu::tile_fwk::DynFuncData *funcDataList_;
    uint64_t *hcclContextAddr_;
    std::queue<uint64_t> tasks_;
};
} // namespace npu::tile_fwk
