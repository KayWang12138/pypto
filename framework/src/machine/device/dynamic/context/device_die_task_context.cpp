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
 * \file device_mix_task_context.cpp
 * \brief
 */

 #include "machine/device/dynamic/context/device_task_context.h"

namespace npu::tile_fwk::dynamic {
int DeviceTaskContext::InitDieReadyQueues(DynDeviceTask *dyntask, DevAscendProgram *devProg,
    ReadyCoreFunctionQueue* dieAivQueue[DIE_NUM], ReadyCoreFunctionQueue* dieAicQueue[DIE_NUM]) {
    if (!IsMixArch(devProg)) {
        return DEVICE_MACHINE_OK;
    }
    ReadyCoreFunctionQueue* queue[DIE_READY_QUEUE_SIZE * DIE_NUM];
    uint32_t size = sizeof(ReadyCoreFunctionQueue) + dyntask->devTask.coreFunctionCnt * sizeof(taskid_t);
    for (size_t i = 0; i < DIE_READY_QUEUE_SIZE * DIE_NUM; ++i) {
        WsAllocation qalloc = ControlFlowAllocateSlab(devProg_, size, workspace_->SlabAlloc(size, WsAicpuSlabMemType::DIE_READY_QUE));
        ReadyCoreFunctionQueue *q = qalloc.As<ReadyCoreFunctionQueue>();
        q->lock = 0;
        q->head = 0;
        q->tail = 0;
        q->capacity = dyntask->devTask.coreFunctionCnt;
        q->elem = reinterpret_cast<taskid_t *>(q + 1);
        queue[i] = q;
        dyntask->readyQueue[i] = q;
    }
    for (int i = 0; i < DIE_NUM; i++) {
        dieAivQueue[i] = q[i];
        dieAicQueue[i] = q[DIE_NUM + i];
    }
    return DEVICE_MACHINE_OK;
}

void DeviceTaskContext::UpdateDeviceDieTaskQueueInfo(DynDeviceTask *dyntask, ReadyCoreFunctionQueue *dieAivQueue[DIE_NUM],
    ReadyCoreFunctionQueue *dieAicQueue[DIE_NUM]) {
    for (int i = 0; i < DIE_NUM; i++) {
        dyntask->devTask.dieReadyFunctionQue.readyDieAivCoreFunctionQue[i] = PtrToValue(aivQueue)(dieAivQueue[i]);
        dyntask->devTask.dieReadyFunctionQue.readyDieAicCoreFunctionQue[i] = PtrToValue(aivQueue)(dieAicQueue[i]);
    }
}

}