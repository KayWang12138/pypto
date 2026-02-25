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
 * \file aicore_manager.cpp
 * \brief
 */

#include "aicore_manager.h"
#include <chrono>

namespace npu::tile_fwk {

void SdmaPrefetch(DeviceTask *devTask)
{
    if (devTask == nullptr || devTask->l2Info.prefetchNum == 0) {
      return;
    }
    if (devTask->l2Info.prefetchNum > MAX_PREFETCH_NUM) {
      DEV_ERROR("Prefetch invalid num %ld.\n", devTask->l2Info.prefetchNum);
      return;
    }
    int fd = open(SDMA_FILE, O_RDWR);
    if (fd == -1) {
      return;
    }
    struct sdma_l2_cmo_desc desc;
    desc.cmo_opcode = 0x6;
    int ret = 0;
    for (int64_t i = 0; i < devTask->l2Info.prefetchNum; ++i) {
      desc.src_addr = devTask->l2Info.prefetchAddrs[i];
      desc.size = devTask->l2Info.prefetchSizes[i];
      ret |= ioctl(fd, IOCTL_SDMA_L2_CMO, &desc);
      DEV_DEBUG("Prefetch %lx %lu ret:%d\n", devTask->l2Info.prefetchAddrs[i],
          devTask->l2Info.prefetchSizes[i], ret);
    }
    DEV_INFO("Prefetch tensor num %ld ret %d.\n", devTask->l2Info.prefetchNum, ret);
    close(fd);
    return;
}

int AiCoreManager::RunTask(DeviceTaskCtrl *taskCtrl)
{
    int ret = 0;
    DEV_INFO("receive new task %lu\n", taskCtrl->taskId);
    InitTaskData();
    if (isLeaderScheduler_ == true)  aicpuTaskManager_.Init(curDevTask_);

    const auto t0 = std::chrono::high_resolution_clock::now();

    npu::tile_fwk::dynamic::TimeCheck tm;
    uint32_t globalTasksIssued = 0;
    while (globalTasksIssued < curDevTask_->coreFunctionCnt)
    {
        uint32_t tasksIssued = 0;
        while (TryBatchSendTask() > 0) tasksIssued++;
        ResolveDepForAllAiCore();
        while (TryBatchSendTask() > 0) tasksIssued++;
        globalTasksIssued = curTaskCtrl_->issuedTaskCount.fetch_add(tasksIssued, std::memory_order_relaxed);

        if (npu::tile_fwk::dynamic::CheckTimeOut("wait task send finish.", tm) != 0) return -1;
    }

    DEV_DEBUG("Aicpu %d proc finish send all task .\n", aicpuIdx_);
    ret = WaitAllAicoreFinish(aicStart_, aicEnd_);
    if (ret != npu::tile_fwk::dynamic::DEVICE_MACHINE_OK) {
        // DEV_ERROR("wait tail aic task timeout .\n");
    }

    if (isLeaderScheduler_ == true) {
        while (!aicpuTaskManager_.Finished()) {
            (void)aicpuTaskManager_.TaskProcess();
        }
    }

    const auto tf = std::chrono::high_resolution_clock::now();
    const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(tf - t0).count();
    DEV_ERROR("[AICPU %d] Running Time: %ldns", aicpuIdx_, ns);

    if (isLeaderScheduler_ == true) {
        delete availableTaskQueue_;
    }

    delete runningPairQueue_ ;
    delete availableCoreQueue_[(int)MachineType::AIV];
    delete availableCoreQueue_[(int)MachineType::AIC];

    return ret;
}

int AiCoreManager::Run(int threadIdx, DeviceArgs *deviceArgs, DeviceTaskCtrl *taskCtrl)
{
    aicpuIdx_ = threadIdx;
    curTaskCtrl_ = taskCtrl;
    curDevTask_ = static_cast<DeviceTask *>(taskCtrl->devTask);

    // Getting variable controlling who is the scheduler lead. The first to arrive here should take the lead so that this starts as fast as possible
    leadSchedulerId_ = (std::atomic<uint32_t>*) &curDevTask_->leadSchedulerId;
    
    // Putting myself as leader, if nobody has done it yet
    uint32_t expectedValue = AICPU_LEAD_SCHEDULER_NULL;
    isLeaderScheduler_ = leadSchedulerId_->compare_exchange_strong(expectedValue, (uint32_t)aicpuIdx_);

    aicNum_ = deviceArgs->nrAic;
    aivNum_ = deviceArgs->nrAiv;
    aicpuNum_ = START_STATIC_AICPU_NUM;
    aicValidNum_ = deviceArgs->nrValidAic;
    regAddrs_ = reinterpret_cast<int64_t *>(deviceArgs->coreRegAddr);
    sharedBuffer_ = deviceArgs->sharedBuffer;

    for (uint32_t idx = 0; idx < MAX_AICORE_NUM; idx++)
    {
        auto baseAddress = (uint64_t) regAddrs_[idx];
        readyRegQueues_[idx]  = (volatile uint64_t *)(baseAddress + (uint64_t)regSprDataMainBase_);
        finishRegQueues_[idx] = (volatile uint64_t *)(baseAddress + (uint64_t)regSprCond_);
    }

    auto f = [](int total, int idx, int part, int &start, int &end) {
        int perCpu = total / part;
        int remain = total % part;
        start = idx * perCpu + ((idx < remain) ? idx : remain);
        end = start + perCpu + ((idx < remain) ? 1 : 0);
    };

    f(aicValidNum_, aicpuIdx_, aicpuNum_, aicStart_, aicEnd_);
    f(AIV_NUM_PER_AI_CORE * aicValidNum_, aicpuIdx_, aicpuNum_, aivStart_, aivEnd_);
    aivStart_ += aicValidNum_;
    aivEnd_ += aicValidNum_;

    args_.fill(nullptr);
    blockIdToPhyCoreId_.fill(-1);
    
    ForEachManageAicore([this](int coreIdx)
    {
        auto args = reinterpret_cast<KernelArgs *>((static_cast<uint64_t>(sharedBuffer_)) + SHARED_BUFFER_SIZE * coreIdx);
        args->taskEntry.reserved[0] = dotStatus_;
        volatile int64_t *shakeBuffer = args->shakeBuffer;
        npu::tile_fwk::dynamic::TimeCheck tm;
        while ((*shakeBuffer & 0xFFFFFFFF) != AICORE_SAY_HELLO) {
            if (npu::tile_fwk::dynamic::CheckTimeOut("hand shake", tm) != 0) {
                DEV_ERROR("hand shake %d timeout.\n", coreIdx);
                AbnormalStop();
            }
        }
        args_[coreIdx] = args;
        // ENTRIES NEED TO BE PUT -1 and then filled by the leader
        blockIdToPhyCoreId_[coreIdx] = (*shakeBuffer >> NUM_THIRTY_TWO) & AICORE_COREID_MASK;
        curDevTask_->blockIdToPhyCoreId[coreIdx] = (*shakeBuffer >> NUM_THIRTY_TWO) & AICORE_COREID_MASK;
    });

    // Enabling fast path
    ForEachManageAicore([this](int coreIdx) { WriteReg32(coreIdx, REG_SPR_FAST_PATH_ENABLE, REG_SPR_FAST_PATH_OPEN); });

    /* write to MAINBASE reg need reg 0x18 open first */
    __sync_synchronize();

    int ret = npu::tile_fwk::dynamic::DEVICE_MACHINE_OK;
    if (taskCtrl != nullptr) {
        ret = RunTask(taskCtrl);
    } else {
        while (true) {
            taskCtrl = taskQueue_.Dequeue();
            if (taskCtrl == nullptr)
                break;
            ret = RunTask(taskCtrl);
            taskCtrl->PutTask(ret);
        }
    }

    NormalStop();
    return ret;
}

inline bool AiCoreManager::checkCoreFinished(const int coreIdx)
{
    uint64_t finTaskVal = GetFinishedTask(coreIdx);
    uint32_t regLFinTaskState = REG_LOW_TASK_STATE(finTaskVal);
    if (regLFinTaskState == TASK_FIN_STATE) return true;
    return false;
}

int AiCoreManager::WaitAllAicoreFinish(int coreIdxStart, int coreIdxEnd)
{
    npu::tile_fwk::dynamic::TimeCheck tm;
    for (int i = coreIdxStart; i < coreIdxEnd; i++) {
        while (checkCoreFinished(i) == false)
        {
            if (npu::tile_fwk::dynamic::CheckTimeOut("wait tail task", tm) != 0) {
                DEV_ERROR("wait tail task finish timeout coreindx=%d.\n", i);
                return -1;
            }
        }
    }
    return 0;
}

uint64_t AiCoreManager::TryBatchSendTask()
{
    auto taskIdx = (uint64_t)availableTaskQueue_->pop();
    if (taskIdx == aicoreNullTask) return 0;

    // Getting task's type
    const auto readyState = reinterpret_cast<CoreFunctionReadyState *>(curDevTask_->coreFunctionReadyStateAddr);
    const auto coreType = readyState[taskIdx].coreType;
    auto coreIdx = (uint64_t)availableCoreQueue_[coreType]->pop();
    if (coreIdx == aicoreNullCore)
    {
        availableTaskQueue_->push(taskIdx);
        return 0;
    }

    // DEV_ERROR("AICPU %d - Running Task: %lu", aicpuIdx_, *taskSetAddress);
    SetReadyQueue(coreIdx, taskIdx);
    runningPairQueue_->push(encodePair(taskIdx, coreIdx));
    return 1;
}

void AiCoreManager::ResolveDepForAllAiCore()
{
    const auto runningPair = runningPairQueue_->pop();
    if (runningPair == aicoreNullPair) return;

    const int coreIdx = (int)decodePairCore(runningPair);

    if (checkCoreFinished(coreIdx)) 
    {
        const uint64_t taskId = (int)decodePairTask(runningPair);
        ResolveDep(taskId);
        const auto readyState = reinterpret_cast<CoreFunctionReadyState *>(curDevTask_->coreFunctionReadyStateAddr);
        const auto coreType = readyState[taskId].coreType; 
        availableCoreQueue_[coreType]->push(coreIdx);
    }
    else
    {
        runningPairQueue_->push(runningPair);
    }
}

void AiCoreManager::ResolveVirtualPure(uint64_t dep, CoreFunctionReadyState* readyState) {
    DEV_DEBUG("new virtual pure task resolved. id: %lu\n", dep);
    auto virtualFuncInfo = &(reinterpret_cast<CoreFunctionWsAddr *>(curDevTask_->coreFuncData.coreFunctionWsAddr)[dep]);
    auto topo = reinterpret_cast<CoreFunctionTopo *>(virtualFuncInfo->topoAddr);
    for (uint64_t i = 0 ; i < topo->depNum; i++) {
        uint64_t depId = topo->depIds[i];
        if (readyState[depId].coreType == static_cast<uint64_t>(MachineType::AICPU)) {
            aicpuTaskManager_.TaskEnqueue(depId);
            continue;
        }
        availableTaskQueue_->push((uint32_t)depId);
    }
}

void AiCoreManager::ResolveVirtualMix(uint64_t dep, CoreFunctionReadyState* readyState) {
    DEV_DEBUG("new virtual mix task resolved. id: %lu\n", dep);
    auto virtualFuncInfo = &(reinterpret_cast<CoreFunctionWsAddr *>(curDevTask_->coreFuncData.coreFunctionWsAddr)[dep]);
    auto topo = reinterpret_cast<CoreFunctionTopo *>(virtualFuncInfo->topoAddr);
    for (uint64_t i = 0 ; i < topo->depNum; i++) {
        uint64_t depId = topo->depIds[i];
        if (readyState[depId].coreType == static_cast<uint64_t>(MachineType::AICPU))
        {
            aicpuTaskManager_.TaskEnqueue(depId);
            continue;
        }
        if (readyState[depId].readyCount == topo->readyCount) {
            availableTaskQueue_->push((uint32_t)depId);
        } else {
            if (__sync_add_and_fetch(&(readyState[depId].readyCount), (-1) * topo->readyCount) == 0) {
                availableTaskQueue_->push((uint32_t)depId);
            }
        }
    }
}

void AiCoreManager::ResolveByCoreType(int coretype, uint64_t depTaskId, CoreFunctionReadyState *readyState) {
    // Compiler optimizations reduce switch-case to O(1), rendering if-else unnecessary in such cases.
    switch (coretype) {
        case static_cast<int>(MachineType::AIV):
        case static_cast<int>(MachineType::AIC): {
            availableTaskQueue_->push((uint32_t)depTaskId);
            break;
        }
        case static_cast<int>(MachineType::MIX): {
            DEV_ERROR("in valid core type mix.");
            break;
        }
        case static_cast<int>(MachineType::AICPU): {
            aicpuTaskManager_.TaskEnqueue(depTaskId);
            break;
        }
        case static_cast<int>(MachineType::HUB): {
            ResolveDep(depTaskId);
            break;
        }
        case static_cast<int>(MachineType::VIRTUAL_PURE): {
            ResolveVirtualPure(depTaskId, readyState);
            break;
        }
        case static_cast<int>(MachineType::VIRTUAL_MIX): {
            ResolveVirtualMix(depTaskId, readyState);
            break;
        }
        default: {
            break;
        }
    }
    DEV_DEBUG("new task resolved. id: %lu, coretype: %d\n", depTaskId, coretype);
}

void AiCoreManager::ResolveDep(uint64_t finishId) {
    auto readyState = reinterpret_cast<CoreFunctionReadyState *>(curDevTask_->coreFunctionReadyStateAddr);
    auto funcInfo = &(reinterpret_cast<CoreFunctionWsAddr *>(curDevTask_->coreFuncData.coreFunctionWsAddr)[finishId]);
    auto topo = reinterpret_cast<CoreFunctionTopo *>(funcInfo->topoAddr);
    DEV_DEBUG("resolve %lx, Dep core function num: %lu\n", finishId, topo->depNum);
    for (uint64_t i = 0; i < topo->depNum; i++) {
        uint64_t dep = topo->depIds[i];
        int ret = __sync_add_and_fetch(&(readyState[dep].readyCount), 1);
        if (ret != 0) {
            continue;
        }
        ResolveByCoreType(readyState[dep].coreType, dep, readyState);
    }
}

void AiCoreManager::AbnormalStop() {
    WriteReg32ALl(regSprDataMainBase_, AICORE_TASK_STOP + 1);
    /* write to MAINBASE reg must be done before close 0x18 */
    if (isNeedWriteRegForFastPath_) WriteReg32ALl(REG_SPR_FAST_PATH_ENABLE, REG_SPR_FAST_PATH_CLOSE);
}

void AiCoreManager::NormalStop() {
    ForEachManageAicore([this](auto coreIdx) { SetReadyQueue(coreIdx, AICORE_TASK_STOP); });
    /* write to MAINBASE reg must be done before close 0x18 */
    __sync_synchronize();
    ForEachManageAicore([this](auto coreIdx)
    {
        if (isNeedWriteRegForFastPath_) WriteReg32(coreIdx, REG_SPR_FAST_PATH_ENABLE, REG_SPR_FAST_PATH_CLOSE);
        volatile KernelArgs *arg = reinterpret_cast<KernelArgs *>(sharedBuffer_ + coreIdx * SHARED_BUFFER_SIZE);
        arg->shakeBuffer[0] = 0;
        arg->shakeBuffer[SHAK_BUF_COREFUNC_DATA_INDEX] = 0;
    });
}

}