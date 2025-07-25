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
#include "machine/utils/device_log.h"
#include "dynamic/device_utils.h"
#include "machine/utils/barrier.h"

using namespace npu::tile_fwk;
using namespace npu::tile_fwk::dynamic;
namespace {
constexpr uint64_t CPUS_PER_CLUSTER = 4;
}

struct MachineManager {
    int allocThreadIdx(int nrAicpu) {
        int threadIdx = -1;
        int cpu = sched_getcpu();
        cpumask.fetch_or(1 << cpu, std::memory_order_release);
        while (__builtin_popcount(cpumask.load(std::memory_order_acquire)) != nrAicpu) {
            sched_yield();
        }
        auto maskval = cpumask.load(std::memory_order_relaxed);
        int cpuoff = 0;
        for (int i = 0; i < static_cast<int>(sizeof(uint64_t)); i++) {
            int mask = (maskval >> cpuoff) & 0xF;
            if (__builtin_popcount(static_cast<uint32_t>(mask)) >= static_cast<int>(MAX_SCHEDULE_AICPU_NUM)) {
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

    int Run(DeviceArgs *args) {
        char logfile[128];
        int ret = DEVICE_MACHINE_OK;

        int threadIdx = allocThreadIdx(args->nrAicpu);
        if (threadIdx != -1) {
#if DEBUG_SWITCH && !DEBUG_PLOG
            (void)sprintf_s(logfile, sizeof(logfile), "/tmp/aicpu%d.txt", threadIdx);
            GetLogger(logfile);
#endif
            DEV_INFO("threadIdx %d aicNum %u aivNum %u aicpuNum %u validAicNum%u \n", threadIdx, args->nrAic,
                args->nrAiv, args->nrAicpu, args->nrValidAic);
            DEV_INFO("devQueueAddr %lx, sharedBuffer %lx coreRegAddr %lx corePmuAdr %lx\n", args->devQueueAddr,
                args->sharedBuffer, args->coreRegAddr, args->corePmuAddr);
            ret = machine.Run(threadIdx, args);
            DEV_INFO("threadIdx %d finished, ret %d\n", threadIdx, ret);
#if !DEBUG_PLOG
            GetLogger().Flush();
#endif
        } else {
            int old = 0;
            if (args->taskType == DEVICE_TASK_TYPE_DYN && ctrlcpu.compare_exchange_weak(old, 1)) {
                (void)sprintf_s(logfile, sizeof(logfile), "/tmp/aicpu%u.txt", START_AICPU_NUM);
#if !DEBUG_PLOG
                GetLogger(logfile);
#endif
                ret = machine.ExecDyn(args->taskId, args->taskData);
#if !DEBUG_PLOG
                GetLogger().Flush();
#endif
            } else {
#if DEBUG_SWITCH && !DEBUG_PLOG
              (void)sprintf_s(logfile, sizeof(logfile), "/tmp/aicpu4.txt");
              GetLogger(logfile);
#endif
              auto devTask = reinterpret_cast<DeviceTask *>(args->taskData);
              SdmaPrefetch(devTask);
#if DEBUG_SWITCH && !DEBUG_PLOG
              GetLogger().Flush();
#endif
            }
        }
        if (++finished == static_cast<std::atomic<int>>(args->nrAicpu)) {
            return DEVICE_MACHINE_FINISHED;
        }
        return ret;
    }

    void GetTaskTotalWastTime(volatile uint64_t *totalWastTime) {
        uint64_t min_task_start_time = UINT64_MAX;
        uint64_t max_task_end_time = 0;
        for (uint32_t i = 0; i < MAX_SCHEDULE_AICPU_NUM; i++) {
            min_task_start_time = std::min(machine.GetMinTaskTime(i), min_task_start_time);
            max_task_end_time = std::max(machine.GetMaxTaskTime(i), max_task_end_time);
        }
        *totalWastTime = max_task_end_time - min_task_start_time;
        DEV_INFO("min_task_start_time %lu, max_task_end_time %lu\n", min_task_start_time, max_task_end_time);
    }

    void init(DeviceArgs *args) { machine.init(args); }

    std::atomic<int> threadIdx_{0};
    std::atomic<int> finished{0};
    std::atomic<uint64_t> cpumask{0};
    std::atomic<int> ctrlcpu{0};
    DeviceMachine machine;
};
static std::mutex g_mutex;

extern "C" __attribute__((visibility("default"))) int StaticTileFwkNSAKernelServer(void *targ) {
    auto args = (DeviceArgs *)targ;
    g_mutex.lock();
    MachineManager *machine = reinterpret_cast<MachineManager *>(args->opaque);
    if (machine == nullptr) {
        machine = new MachineManager();
        machine->init(args);
        args->opaque = reinterpret_cast<uint64_t>(machine);
    }
    g_mutex.unlock();

    int rc = machine->Run(args);
    if (rc == DEVICE_MACHINE_FINISHED) {
        DEV_INFO("all exited destroy the machine\n");
        machine->GetTaskTotalWastTime((uint64_t *)args->taskWastTime);
        wmb();
        DEV_INFO("Total wast time is %lu\n", *(uint64_t *)args->taskWastTime);
#if !DEBUG_PLOG
        GetLogger().Flush();
#endif
        delete machine;
        return 0;
    }
    return rc;
}
