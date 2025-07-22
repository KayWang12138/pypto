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
 * \file device_machine.cpp
 * \brief
 */

#include "device_machine.h"
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <sched.h>
#include "machine/device/dynamic/device_utils.h"
#include "machine/kernel/aicore.h"
#include "machine/utils/device_log.h"

using namespace npu::tile_fwk;
using namespace npu::tile_fwk::dynamic;

namespace {
constexpr uint64_t CPUS_PER_CLUSTER = 4;

void DySdmaPrefetch(DevStartArgs *devArgs) {
    if (devArgs == nullptr || devArgs->devProg == nullptr) {
      return;
    }
    auto devProg = devArgs->devProg;
    size_t prefetchNum = devProg->prefetchInfoList.size();
    DEV_INFO("Prefetch num %zu.\n", prefetchNum);
    if (prefetchNum > devArgs->inputTensorSize) {
      DEV_ERROR("Prefetch invalid num %zu.\n", prefetchNum);
      return;
    }
    int fd = open(SDMA_FILE.c_str(), O_RDWR);
    if (fd == -1) {
      return;
    }
    struct sdma_l2_cmo_desc desc;
    desc.cmo_opcode = 0x6;
    int ret = 0;
    for (size_t i = 0; i < prefetchNum; ++i) {
      auto &preInfo = devProg->prefetchInfoList[i];
      DEV_INFO("Prefetch tensor idx[%lu] with size[%lu].\n", preInfo.tensorIdx, preInfo.tensorSize);
      if (preInfo.tensorIdx >= static_cast<uint64_t>(devArgs->GetInputTensorSize())) {
        DEV_WARN("TensorIdx[%lu] over inpust size[%d].\n", preInfo.tensorIdx, devArgs->GetInputTensorSize());
        continue;
      }
      auto &inTensor = devArgs->GetInputTensor(preInfo.tensorIdx);
      desc.src_addr = inTensor.address;
      desc.size = preInfo.tensorSize;
      ret |= ioctl(fd, IOCTL_SDMA_L2_CMO, &desc);
      DEV_DEBUG("Prefetch %lx %lu ret:%d\n", inTensor.address, preInfo.tensorSize, ret);
    }
    DEV_INFO("Prefetch tensor num %zu ret %d.\n", prefetchNum, ret);
    close(fd);
    return;
}
}

struct DynMachineManager {
    int allocThreadIdx(int nrAicpu) {
        int threadIdx = -1;
        if (schAicpuNum == 1) {
            return threadIdx_++;
        }
        int cpu = sched_getcpu();
        cpumask.fetch_or(1 << cpu, std::memory_order_release);
        while (__builtin_popcount(cpumask.load(std::memory_order_acquire)) != nrAicpu) {
            sched_yield();
        }

        auto maskval = cpumask.load(std::memory_order_relaxed);
        int cpuoff = 0;
        for (int i = 0; i < static_cast<int>(sizeof(uint64_t)); i++) {
            int mask = (maskval >> cpuoff) & 0xF;
            if (__builtin_popcount(static_cast<uint32_t>(mask)) >= schAicpuNum) {
                threadIdx = threadIdx_++;
                break;
            }
            cpuoff += CPUS_PER_CLUSTER;
            if (cpu < cpuoff) {
                break;
            }
        }
        return threadIdx;
    }

    int Run(AstKernelArgs *args) {
        char logfile[128];
        (void)logfile;
        int ret = npu::tile_fwk::dynamic::DEVICE_MACHINE_OK;
        auto devArgs = (DeviceArgs *)args->tilingdata;
        int threadIdx = allocThreadIdx(devArgs->nrAicpu);
        if ((threadIdx != -1) && threadIdx < schAicpuNum) {
#if defined(DEBUG_SWITCH) && DEBUG_SWITCH
            (void)sprintf_s(logfile, sizeof(logfile), "/tmp/tile_fwk_aicpu_sch%d.txt", threadIdx);
            GetLogger(logfile);
#endif
            DEV_INFO("devArgs->taskType %d\n", static_cast<int>(devArgs->taskType));
            DEV_INFO("threadIdx %d aicNum %u aivNum %u aicpuNum %u validAicNum%u \n", threadIdx, devArgs->nrAic,
                devArgs->nrAiv, devArgs->nrAicpu, devArgs->nrValidAic);
            DEV_INFO("devQueueAddr %lx, sharedBuffer %lx coreRegAddr %lx corePmuAdr %lx\n", devArgs->devQueueAddr,
                devArgs->sharedBuffer, devArgs->coreRegAddr, devArgs->corePmuAddr);
            ret = machine.Run(threadIdx, devArgs);
        } else {
            threadIdx = ctrlcpuIdx.fetch_add(1);
            DEV_INFO("devArgs->taskType %d\n",  static_cast<int>(devArgs->taskType));
            if (devArgs->taskType == DEVICE_TASK_TYPE_DYN && threadIdx == MAX_SCHEDULE_AICPU_NUM) {
                (void)sprintf_s(logfile, sizeof(logfile), "/tmp/tile_fwk_aicpu_ctrl.txt");
                GetLogger(logfile);
                ret = machine.ExecDyn(threadIdx, devArgs->taskId, args);
            } else if (threadIdx == MAX_SCHEDULE_AICPU_NUM + 1){
#if defined(DEBUG_SWITCH) && DEBUG_SWITCH
                (void)sprintf_s(logfile, sizeof(logfile), "/tmp/tile_fwk_aicpu_prefetch.txt");
                GetLogger(logfile);
#endif
                if (devArgs->taskType == DEVICE_TASK_TYPE_DYN) {
                  DevStartArgs *startArgs = (DevStartArgs *)args->workspace;
                  DySdmaPrefetch(startArgs);
                } else {
                  auto devTask = reinterpret_cast<DeviceTask *>(devArgs->taskData);
                  SdmaPrefetch(devTask);
                }
            }
        }
     
        DEV_INFO("threadIdx %d finished, ret %d\n", threadIdx, ret);
        GetLogger().Flush();
        if (++finished == static_cast<std::atomic<int>>(devArgs->nrAicpu)) {
            return npu::tile_fwk::dynamic::DEVICE_MACHINE_FINISHED;
        }
        return ret;
    }

    void init(DeviceArgs *args) {
        schAicpuNum = CalcSchAicpuNumByBlockDim(args->nrValidAic);
        machine.init(args);
    }

    std::atomic<int> threadIdx_{0};
    std::atomic<int> finished{0};
    std::atomic<uint64_t> cpumask{0};
    std::atomic<int> ctrlcpuIdx{MAX_SCHEDULE_AICPU_NUM};
    int schAicpuNum{MAX_SCHEDULE_AICPU_NUM};
    DeviceMachine machine;
};


static std::mutex g_mutex;

static int RunDynamic(AstKernelArgs *kargs) {
    auto devArgs = (DeviceArgs *)kargs->tilingdata;

    g_mutex.lock();
    DynMachineManager *machine = reinterpret_cast<DynMachineManager *>(devArgs->opaque);
    if (machine == nullptr) {
        machine = new DynMachineManager();
        machine->init(devArgs);
        devArgs->opaque = reinterpret_cast<uint64_t>(machine);
    }
    g_mutex.unlock();
    int rc = machine->Run(kargs);
    if (rc == npu::tile_fwk::dynamic::DEVICE_MACHINE_FINISHED) {
        DEV_INFO("all exited destroy the machine\n");
        delete machine;
        devArgs->opaque = 0;
        return DEVICE_MACHINE_OK;
    }
    return rc;
}

static bool CheckValidArgs(AstKernelArgs *kargs) {
    if (kargs == nullptr) {
        return false;
    }
    if (kargs->inputs == nullptr || kargs->outputs == nullptr || kargs->workspace == nullptr || kargs->tilingdata == nullptr) {
        return false;
    }
    return true;
}

extern "C" __attribute__((visibility("default"))) int DynTileFwkNSAKernelServerInit(void *targ) {
    PerfBegin(PERF_EVT_DEVICE_MACHINE_INIT_DYN);
    auto kargs = (AstKernelArgs *)targ;
    if (!CheckValidArgs(kargs)) {
        DEV_INFO("invalid parameter\n");
        return -EINVAL;
    }
    DeviceMachine::InitDyn(kargs);
    PerfEnd(PERF_EVT_DEVICE_MACHINE_INIT_DYN);
    return 0;
}

extern "C" __attribute__((visibility("default"))) int DynTileFwkNSAKernelServer(void *targ) {
    auto kargs = (AstKernelArgs *)targ;
    return RunDynamic(kargs);
}
