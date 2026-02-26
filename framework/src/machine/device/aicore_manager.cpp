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
    for (size_t coreIdx = 0; coreIdx < AIV_CORE_COUNT + AIC_CORE_COUNT; coreIdx++)
    {
        WriteReg32(coreIdx, regSprDataMainBase_, AICORE_TASK_STOP + 1);
        if (isNeedWriteRegForFastPath_) WriteReg32(coreIdx, REG_SPR_FAST_PATH_ENABLE, REG_SPR_FAST_PATH_CLOSE); // write to MAINBASE reg must be done before close 0x18
    }
}

void AiCoreManager::NormalStop() {
    for (size_t coreIdx = 0; coreIdx < AIV_CORE_COUNT + AIC_CORE_COUNT; coreIdx++) SetReadyQueue(coreIdx, AICORE_TASK_STOP);

    __sync_synchronize(); // write to MAINBASE reg must be done before close 0x18 */

    for (size_t coreIdx = 0; coreIdx < AIV_CORE_COUNT + AIC_CORE_COUNT; coreIdx++)
    {
        if (isNeedWriteRegForFastPath_) WriteReg32(coreIdx, REG_SPR_FAST_PATH_ENABLE, REG_SPR_FAST_PATH_CLOSE);
        volatile KernelArgs *arg = reinterpret_cast<KernelArgs *>(sharedBuffer_ + coreIdx * SHARED_BUFFER_SIZE);
        arg->shakeBuffer[0] = 0;
        arg->shakeBuffer[SHAK_BUF_COREFUNC_DATA_INDEX] = 0;
    };
}

} // namespace npu::tile_fwk