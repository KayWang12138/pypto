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
    InitTaskData(taskCtrl);
    if (aicpuIdx_ == LEAD_STATIC_SCHEDULER_AICPU_ID)  aicpuTaskManager_.Init(curDevTask_);

    const auto t0 = std::chrono::high_resolution_clock::now();

    // DEV_ERROR("AICPU %d - Task Vector Queue Size: %lu\n", aicpuIdx_, availableVectorTaskQueue_->wasSize());
    // DEV_ERROR("AICPU %d - Task Cube Queue Size: %lu\n", aicpuIdx_, availableVectorTaskQueue_->wasSize());
    // DEV_ERROR("AICPU %d - Core Queue Size: %lu\n", aicpuIdx_, availableCoreQueue_->wasSize());

    npu::tile_fwk::dynamic::TimeCheck tm;
    while (taskCtrl->finishedFunctionCnt.load(std::memory_order_relaxed) < curDevTask_->coreFunctionCnt)
    {
        TryBatchSendTask(CoreType::AIC, availableCubeTaskQueue_, aicStart_, aicEnd_);
        if (waitTaskCnt_[static_cast<int>(CoreType::AIC)] > 0) {
            ResolveDepForAllAiCore(CoreType::AIC);
            TryBatchSendTask(CoreType::AIC, availableCubeTaskQueue_, aicStart_, aicEnd_);
        }

        TryBatchSendTask(CoreType::AIV, availableVectorTaskQueue_, aivStart_, aivEnd_);
        if (waitTaskCnt_[static_cast<int>(CoreType::AIV)] > 0) {
            ResolveDepForAllAiCore(CoreType::AIV);
            TryBatchSendTask(CoreType::AIV, availableVectorTaskQueue_, aivStart_, aivEnd_);
        }

        const uint64_t sentAic = sendCnt_[static_cast<int>(CoreType::AIC)];
        const uint64_t sentAiv = sendCnt_[static_cast<int>(CoreType::AIV)];
        waitTaskCnt_[static_cast<int>(CoreType::AIC)] += sentAic;
        waitTaskCnt_[static_cast<int>(CoreType::AIV)] += sentAiv;
        sendCnt_[static_cast<int>(CoreType::AIC)] = 0;
        sendCnt_[static_cast<int>(CoreType::AIV)] = 0;

        taskCtrl->finishedFunctionCnt.fetch_add(sentAic + sentAiv + reSolveHubCnt_, std::memory_order_relaxed);
        reSolveHubCnt_ = 0;
        
        if (npu::tile_fwk::dynamic::CheckTimeOut("wait task send finish.", tm) != 0) {
            return -1;
        }
    }

    DEV_DEBUG("Aicpu %d proc finish send all task .\n", aicpuIdx_);
    ret = WaitAllAicoreFinish(aicStart_, aicEnd_);
    if (ret != npu::tile_fwk::dynamic::DEVICE_MACHINE_OK) {
        // DEV_ERROR("wait tail aic task timeout .\n");
    }

    if (aicpuIdx_ == LEAD_STATIC_SCHEDULER_AICPU_ID) {
        while (!aicpuTaskManager_.Finished()) {
            (void)aicpuTaskManager_.TaskProcess();
        }
    }

    const auto tf = std::chrono::high_resolution_clock::now();
    const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(tf - t0).count();
    DEV_ERROR("[AICPU %d] Running Time: %ldns", aicpuIdx_, ns);

    if (aicpuIdx_ == LEAD_STATIC_SCHEDULER_AICPU_ID) {
        delete availableVectorTaskQueue_;
        delete availableCubeTaskQueue_;
        delete availableCoreQueue_;
        delete runningPairQueue_ ;
    }

    return ret;
}

int AiCoreManager::Run(int threadIdx, DeviceArgs *deviceArgs, DeviceTaskCtrl *taskCtrl)
{
    Init(threadIdx, deviceArgs);

    int ret = HandkShake();
    if (ret != npu::tile_fwk::dynamic::DEVICE_MACHINE_OK) {
        DEV_DEBUG("hand shake timeout .\n");
        AbnormalStop();
        return ret;
    }
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


bool AiCoreManager::waitCoreFinish(int coreIdx) {
    uint64_t finTaskVal = GetFinishedTask(coreIdx);
    uint32_t regLFinTaskId = REG_LOW_TASK_ID(finTaskVal);
    uint32_t regLFinTaskState = REG_LOW_TASK_STATE(finTaskVal);
    if (regLFinTaskState == TASK_FIN_STATE &&
        (pendingIds_[coreIdx] == regLFinTaskId || runningIds_[coreIdx] == regLFinTaskId))
       {
            pendingIds_[coreIdx] = AICORE_TASK_INIT;
            runningIds_[coreIdx] = AICORE_TASK_INIT;
       }

    return pendingIds_[coreIdx] == AICORE_TASK_INIT && runningIds_[coreIdx] == AICORE_TASK_INIT;
}

int AiCoreManager::WaitAllAicoreFinish(int coreIdxStart, int coreIdxEnd)
{
    int stopSent = 0;
    bool coreStopped[MAX_MANAGER_AIV_NUM] = {false};
    npu::tile_fwk::dynamic::TimeCheck tm;
    while (stopSent < coreIdxEnd - coreIdxStart) {
        for (int i = coreIdxStart; i < coreIdxEnd; i++) {
            if (coreStopped[i - coreIdxStart]) {
                continue;
            }
            if (waitCoreFinish(i)) {
                stopSent++;
                coreStopped[i - coreIdxStart] = true;
                continue;
            }
            if (npu::tile_fwk::dynamic::CheckTimeOut("wait tail task", tm) != 0) {
                DEV_ERROR("wait tail task finish timeout coreindx=%d.\n", i);
                return -1;
            }
        }
    }
    return 0;
}

uint64_t AiCoreManager::TryBatchSendTask(CoreType type, taskQueue_t* readyQue, int coreIdxStart, int coreIdxEnd)
{
    if (readyQue->wasEmpty()) {
        DEV_DEBUG("AiCpud:%d, can not send task currently: no ready tasks in the queue\n", aicpuIdx_);
        return 0;
    }

    // check other aicpu if idle, Prioritize remaining tasks for scheduling by other AICPUs
    if (coreRunReadyCnt_[static_cast<int>(type)] == 0 && IsExistOtherAicpuIdle(type)) {
        return 0;
    }

    uint64_t readyCoreCount = corePendReadyCnt_[static_cast<int>(type)];
    if (readyCoreCount == 0 ) {
        DEV_DEBUG("AiCpud:%d, can not send task currently: no cores ready\n", aicpuIdx_);
        return 0;
    }

    auto taskQueueIdx = (uint64_t)readyQue->pop();
    if (taskQueueIdx == aicoreNullTask) return 0;

    // DEV_ERROR("AICPU %d - Popping Task: %lu", aicpuIdx_, taskQueueIdx);
    uint64_t* taskSetAddress = &taskQueueIdx;
    uint64_t taskSetCount = 1;
    if (taskSetCount == 0) {
        DEV_DEBUG("AiCpud:%d, taskCount is zero \n", aicpuIdx_);
        return 0;
    }

    // DEV_ERROR("AICPU %d - Running Task: %lu", aicpuIdx_, *taskSetAddress);

    DEV_DEBUG("AiCpud:%d, pop all new task count: %lu \n", aicpuIdx_, taskSetCount);
    BatchSendTask(type, taskSetAddress, taskSetCount, coreIdxStart, coreIdxEnd);
    DEV_DEBUG("core ready cnt: %u \n", corePendReadyCnt_[static_cast<int>(type)]);
    return taskSetCount;
}

uint32_t AiCoreManager::BatchSendTask(CoreType type, uint64_t *newTask, uint32_t taskCount,int coreIdxStart, int coreIdxEnd) {
    uint32_t sendCnt = 0;
    uint32_t taskIdx = 0;
    uint32_t idx = lastPendReadyCoreIdx_[static_cast<int>(type)];
    uint32_t coreNum = coreIdxEnd - coreIdxStart;
    uint32_t lastProcCore = idx;

    while (corePendReadyCnt_[static_cast<int>(type)] > 0 && sendCnt < taskCount)
    {
        if (pendingIds_[idx] == AICORE_TASK_INIT)
        {
            SendTaskToAiCore(type, idx, newTask[taskIdx++]);
            sendCnt++;
            corePendReadyCnt_[static_cast<int>(type)]--;
            lastProcCore = idx;
        }
        idx = coreIdxStart + (idx - coreIdxStart + 1) % coreNum;
    }

    if (lastProcCore != lastPendReadyCoreIdx_[static_cast<int>(type)])
    {
        lastPendReadyCoreIdx_[static_cast<int>(type)] = coreIdxStart + (lastProcCore - coreIdxStart + 1) % coreNum;
    }

    return sendCnt;
}

void AiCoreManager::SendTaskToAiCore(CoreType type, int coreIdx, uint64_t newTask) {
    SetReadyQueue(coreIdx, newTask + 1);
    pendingIds_[coreIdx] = newTask;
    sendCnt_[static_cast<int>(type)]++;

    const uint64_t pairCode = encodePair(newTask, coreIdx);
    runningPairQueue_->push(pairCode);
    // DEV_ERROR("AICPU: %d - Send task %lu, at core %d ,type:%d, code: 0x%0lX\n", aicpuIdx_, newTask, coreIdx, static_cast<int>(type), pairCode);
}

void AiCoreManager::ResolveDepForAllAiCore(CoreType type)
{
    const auto runningPair = runningPairQueue_->pop();
    if (runningPair == aicoreNullPair) return;

    const int coreIdx = (int)decodePairCore(runningPair);
    const uint64_t taskId = (int)decodePairTask(runningPair);

    // DEV_ERROR("AICPU: %d - checking1 on core %d - task %lu\n", aicpuIdx_, coreIdx, taskId);
    if (runningIds_[coreIdx] == AICORE_TASK_INIT && pendingIds_[coreIdx] == AICORE_TASK_INIT)
    {
        runningPairQueue_->push(runningPair);
        return;
    }
    
    if (isPairingFinished(runningPair)) 
    {
        DEV_DEBUG("core index: %d, PendingTask Finished. pending: %lx\n", coreIdx, pendingIds_[coreIdx]);
        ResolveDepWithDfx(type, coreIdx, taskId);
        pendingIds_[coreIdx] = AICORE_TASK_INIT;
        corePendReadyCnt_[static_cast<int>(type)]++;
        runReadyCoreIdx_[static_cast<int>(type)][coreRunReadyCnt_[static_cast<int>(type)]++] = coreIdx;
    }
    else
    {
        runningPairQueue_->push(runningPair);
    }
}

inline bool AiCoreManager::isPairingFinished(const aicorePair_t pairing)
{
    const int coreIdx = (int)decodePairCore(pairing);
    const uint64_t taskId = (int)decodePairTask(pairing);
    
    uint64_t finTaskRegVal = GetFinishedTask(coreIdx);
    uint32_t finTaskId = REG_LOW_TASK_ID(finTaskRegVal);
    uint32_t finTaskState = REG_LOW_TASK_STATE(finTaskRegVal);

    if (finTaskId == taskId && finTaskState == TASK_FIN_STATE) return true;
    return false;
}

void AiCoreManager::PushReadyTask(int coreType, int64_t taskId) {
    if (readyCount[coreType] < READY_ID_FIX_CACHE_NUM) 
        readyIds[coreType][readyCount[coreType]++] = taskId;
    if ((MachineType)coreType == MachineType::AIV) availableVectorTaskQueue_->push((uint32_t)taskId);
    if ((MachineType)coreType == MachineType::AIC) availableCubeTaskQueue_->push((uint32_t)taskId);
}

void AiCoreManager::ResolveVirtualPure(uint64_t dep, CoreFunctionReadyState* readyState) {
    DEV_DEBUG("new virtual pure task resolved. id: %lu\n", dep);
    auto virtualFuncInfo =
        &(reinterpret_cast<CoreFunctionWsAddr *>(curDevTask_->coreFuncData.coreFunctionWsAddr)[dep]);
    auto topo = reinterpret_cast<CoreFunctionTopo *>(virtualFuncInfo->topoAddr);
    for (uint64_t i = 0 ; i < topo->depNum; i++) {
        uint64_t depId = topo->depIds[i];
        if (readyState[depId].coreType == static_cast<uint64_t>(MachineType::AICPU)) {
            PushAicpuTaskQueue(depId);
            continue;
        }
        PushReadyTask(readyState[depId].coreType, depId);
    }
}

void AiCoreManager::ResolveVirtualMix(uint64_t dep, CoreFunctionReadyState* readyState) {
    DEV_DEBUG("new virtual mix task resolved. id: %lu\n", dep);
    auto virtualFuncInfo =
        &(reinterpret_cast<CoreFunctionWsAddr *>(curDevTask_->coreFuncData.coreFunctionWsAddr)[dep]);
    auto topo = reinterpret_cast<CoreFunctionTopo *>(virtualFuncInfo->topoAddr);
    for (uint64_t i = 0 ; i < topo->depNum; i++) {
        uint64_t depId = topo->depIds[i];
        if (readyState[depId].coreType == static_cast<uint64_t>(MachineType::AICPU)) {
            PushAicpuTaskQueue(depId);
            continue;
        }
        if (readyState[depId].readyCount == topo->readyCount) {
            PushReadyTask(readyState[depId].coreType, depId);
        } else {
            if (__sync_add_and_fetch(&(readyState[depId].readyCount), (-1) * topo->readyCount) == 0) {
                PushReadyTask(readyState[depId].coreType, depId);
            }
        }
    }
}

void AiCoreManager::ResolveByCoreType(int coretype, uint64_t depTaskId, CoreFunctionReadyState *readyState) {
    // Compiler optimizations reduce switch-case to O(1), rendering if-else unnecessary in such cases.
    switch (coretype) {
        case static_cast<int>(MachineType::AIV):
        case static_cast<int>(MachineType::AIC): {
            PushReadyTask(coretype, depTaskId);
            break;
        }
        case static_cast<int>(MachineType::MIX): {
            DEV_ERROR("in valid core type mix.");
            break;
        }
        case static_cast<int>(MachineType::AICPU): {
            PushAicpuTaskQueue(depTaskId);
            break;
        }
        case static_cast<int>(MachineType::HUB): {
            reSolveHubCnt_++;
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
    auto funcInfo =
        &(reinterpret_cast<CoreFunctionWsAddr *>(curDevTask_->coreFuncData.coreFunctionWsAddr)[finishId]);
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

void AiCoreManager::ResolveDepWithDfx(CoreType type, int coreIdx, uint64_t finishId) {
    ResolveDep(finishId);
    (void)coreIdx;
    if (waitTaskCnt_[static_cast<int>(type)] != 0) {
        waitTaskCnt_[static_cast<int>(type)]--;
    }
}

bool AiCoreManager::IsExistOtherAicpuIdle(CoreType type) {
    int idx = (aicpuIdx_ + 1) % START_STATIC_AICPU_NUM;
    while (idx != aicpuIdx_) {
        if (curTaskCtrl_->isAicpuIdle[static_cast<int>(type)][idx].load(std::memory_order_relaxed) == true){
            return true;
        }
        idx = (idx + 1) % START_STATIC_AICPU_NUM;
    }
    return false;
}

void AiCoreManager::Init(int threadIdx, DeviceArgs *deviceArgs) {
    aicNum_ = deviceArgs->nrAic;
    aivNum_ = deviceArgs->nrAiv;
    aicpuNum_ = START_STATIC_AICPU_NUM;
    aicpuIdx_ = threadIdx;
    aicValidNum_ = deviceArgs->nrValidAic;
    regAddrs_ = reinterpret_cast<int64_t *>(deviceArgs->coreRegAddr);
    sharedBuffer_ = deviceArgs->sharedBuffer;
    runningIds_.fill(AICORE_STATUS_INIT);
    pendingIds_.fill(AICORE_STATUS_INIT);
    taskDfxStatPos_.fill(REG_LOW_TASK_PING);

    blockIdToPhyCoreId_.fill(-1);
    readyRegQueues_.fill(nullptr);
    finishRegQueues_.fill(nullptr);
    UpdateAiCoreBlockIndexSection();
    MapRegistersForAllCores();

    args_.fill(nullptr);
    ResetCnt();
}

int AiCoreManager::HandkShake() {
    DEV_INFO("Aicpu %d handshake start.\n", aicpuIdx_);
    int rc = ForEachManageAicoreWithRet([this](int coreIdx) -> int {
        int ret = npu::tile_fwk::dynamic::DEVICE_MACHINE_OK;
        auto args =
            reinterpret_cast<KernelArgs *>((static_cast<uint64_t>(sharedBuffer_)) + SHARED_BUFFER_SIZE * coreIdx);
        args->taskEntry.reserved[0] = dotStatus_;
        volatile int64_t *shakeBuffer = args->shakeBuffer;
        npu::tile_fwk::dynamic::TimeCheck tm;
        while ((*shakeBuffer & 0xFFFFFFFF) != AICORE_SAY_HELLO) {
            if (npu::tile_fwk::dynamic::CheckTimeOut("hand shake", tm) != 0) {
                DEV_ERROR("hand shake %d timeout.\n", coreIdx);
                return -1;
            }
        }
        args_[coreIdx] = args;
        blockIdToPhyCoreId_[coreIdx] = (*shakeBuffer >> NUM_THIRTY_TWO) & AICORE_COREID_MASK;
        DEV_DEBUG("coreidx %d handshake  phycorid %d .\n", coreIdx, blockIdToPhyCoreId_[coreIdx]);
        return ret;
    });
    if (rc != npu::tile_fwk::dynamic::DEVICE_MACHINE_OK) {
        DEV_DEBUG("Aicpu %d handshake failed end.\n", aicpuIdx_);
        return rc;
    }

    if (isNeedWriteRegForFastPath_) {
        ForEachManageAicore(
            [this](int coreIdx) { WriteReg32(coreIdx, REG_SPR_FAST_PATH_ENABLE, REG_SPR_FAST_PATH_OPEN); });
    }
    /* write to MAINBASE reg need reg 0x18 open first */
    __sync_synchronize();
    DEV_INFO("Aicpu %d handshake sucess end.\n", aicpuIdx_);
    return 0;
}

/* assign aic and aiv core index section for this aicpu */
void AiCoreManager::UpdateAiCoreBlockIndexSection() {
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
    corePendReadyCnt_[static_cast<int>(CoreType::AIC)] = aicEnd_ - aicStart_;
    corePendReadyCnt_[static_cast<int>(CoreType::AIV)] = aivEnd_ - aivStart_;
    coreRunReadyCnt_[static_cast<int>(CoreType::AIC)] = 0;
    coreRunReadyCnt_[static_cast<int>(CoreType::AIV)] = 0;
    ForEachManageAicoreReverse(
        [this](int coreIdx) {
        int coreType = static_cast<int>(AicoreType(coreIdx));
        runReadyCoreIdx_[coreType][coreRunReadyCnt_[coreType]++] = coreIdx;
        });
    lastPendReadyCoreIdx_[static_cast<int>(CoreType::AIV)] = aivStart_;
    lastPendReadyCoreIdx_[static_cast<int>(CoreType::AIC)] = aicStart_;
}

void AiCoreManager::MapRegistersForAllCores() {
    for (uint32_t idx = 0; idx < aicNum_ * CORE_NUM_PER_AI_CORE; idx++) {
        void *addr = reinterpret_cast<void *>(regAddrs_[idx]);
        if (addr == nullptr) {
            continue;
        }
        volatile uint64_t *reqQueueReg = reinterpret_cast<volatile uint64_t *>(static_cast<uint8_t *>(addr) + regSprDataMainBase_);
        readyRegQueues_[idx] = reqQueueReg;
        volatile uint64_t *finishQueueReg = reinterpret_cast<volatile uint64_t *>(static_cast<uint8_t *>(addr) + regSprCond_);
        finishRegQueues_[idx] = finishQueueReg;
    }
}

void AiCoreManager::AbnormalStop() {
    WriteReg32ALl(regSprDataMainBase_, AICORE_TASK_STOP + 1);
    /* write to MAINBASE reg must be done before close 0x18 */
    if (isNeedWriteRegForFastPath_) WriteReg32ALl(REG_SPR_FAST_PATH_ENABLE, REG_SPR_FAST_PATH_CLOSE);
}

void AiCoreManager::NormalStop() {
    ForEachManageAicore([this](auto coreIdx) { SetReadyQueue(coreIdx, AICORE_TASK_STOP + 1); });
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