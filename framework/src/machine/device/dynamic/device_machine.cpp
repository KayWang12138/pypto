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
 * \file device_machine.cpp
 * \brief
 */

#include "device_machine.h"
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <sched.h>
#include <signal.h>
#include <sys/ucontext.h>
#include "machine/device/dynamic/device_utils.h"
#include "machine/kernel/aicore.h"
#include "machine/utils/device_log.h"
#include "device_utils.h"

using namespace npu::tile_fwk;
using namespace npu::tile_fwk::dynamic;

namespace {
constexpr int CPUS_PER_CLUSTER = 4;
constexpr uint64_t SIGNAL_DELAY_SECONDS = 2;

extern void SigAct(int signum, siginfo_t* info, void* act);

void DySdmaPrefetch(DevStartArgs *devArgs) {
    if (devArgs == nullptr || devArgs->devProg == nullptr) {
      return;
    }
    auto devProg = devArgs->devProg;
    size_t prefetchNum = devProg->prefetchInfoList.size();
    DEV_INFO("Prefetch num %zu.", prefetchNum);
    if (prefetchNum > devArgs->inputTensorSize) {
      DEV_ERROR("Prefetch invalid num %zu.", prefetchNum);
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
      DEV_INFO("Prefetch tensor idx[%lu] with size[%lu].", preInfo.tensorIdx, preInfo.tensorSize);
      if (preInfo.tensorIdx >= static_cast<uint64_t>(devArgs->GetInputTensorSize())) {
        DEV_WARN("TensorIdx[%lu] over inpust size[%d].", preInfo.tensorIdx, devArgs->GetInputTensorSize());
        continue;
      }
      auto &inTensor = devArgs->GetInputTensor(preInfo.tensorIdx);
      desc.src_addr = inTensor.address;
      desc.size = preInfo.tensorSize;
      ret |= ioctl(fd, IOCTL_SDMA_L2_CMO, &desc);
      DEV_DEBUG("Prefetch %lx %lu ret:%d.", inTensor.address, preInfo.tensorSize, ret);
    }
    DEV_INFO("Prefetch tensor num %zu ret %d.", prefetchNum, ret);
    close(fd);
    return;
}

struct DynMachineManager {
    int allocThreadIdx(int nrAicpu) {
        if (schAicpuNum_ == 1) {
            return threadIdx_++;
        }
        int cpu = sched_getcpu();
        cpumask_.fetch_or(1 << cpu, std::memory_order_release);
        while (__builtin_popcount(cpumask_.load(std::memory_order_acquire)) != nrAicpu) {
            sched_yield();
        }

        auto maskval = cpumask_.load(std::memory_order_relaxed);
        int cpuoff = 0;
        int clus_id = -1;
        for (int i = 0; i < static_cast<int>(sizeof(uint64_t)); ++i) {
            int mask = (maskval >> cpuoff) & 0xF;
            if (__builtin_popcount(static_cast<uint32_t>(mask)) >= schAicpuNum_) {
                clus_id = i;
                break;
            }
            cpuoff += CPUS_PER_CLUSTER;
        }
        if (clus_id == -1) {
            return threadIdx_++;
        }
        if (cpu < cpuoff || cpu >= (cpuoff + CPUS_PER_CLUSTER)) {
            return -1;
        }
        return threadIdx_++;
    }

    void SignalReg() {
        DEV_INFO("Exception SignalReg.");
        struct sigaction myAct;
        (void)memset_s(&myAct, sizeof(myAct), 0, sizeof(myAct));
        sigemptyset(&myAct.sa_mask);
        myAct.sa_flags = SA_SIGINFO;
        myAct.sa_sigaction = SigAct;
        sigaction(SIGFPE, &myAct, &oriFPEAct_);
        sigaction(SIGBUS, &myAct, &oriBUSAct_);
        sigaction(SIGSEGV, &myAct, &oriSEGVAct_);
        sigaction(SIGPIPE, &myAct, &oriPIPEAct_);
        sigaction(SIGILL, &myAct, &oriILLAct_);
        sigaction(SIGABRT, &myAct, &oriBordAct_);
        return;
    }

    int Run(AstKernelArgs *args) {
        int ret = npu::tile_fwk::dynamic::DEVICE_MACHINE_OK;
        auto devArgs = PtrToPtr<int64_t, DeviceArgs>(args->cfgdata);
        if ((uint32_t)schAicpuNum_ > devArgs->nrAicpu - 1) {
            DEV_ERROR("Aicpu num[%u] less than sche num[%d].", devArgs->nrAicpu, schAicpuNum_);
            return npu::tile_fwk::dynamic::DEVICE_MACHINE_ERROR;
        }
        int threadIdx = allocThreadIdx(devArgs->nrAicpu);
        uint64_t allocThreadCycle = GetCycles();
        if ((threadIdx != -1) && threadIdx < schAicpuNum_) {
            CreateLogFile(LOG_TYPE_SCHEDULER, threadIdx);
            DEV_INFO("devArgs->taskType %d.", static_cast<int>(devArgs->taskType));
            DEV_INFO("threadIdx %d aicNum %u aivNum %u aicpuNum %u validAicNum %u.", threadIdx, devArgs->nrAic,
                devArgs->nrAiv, devArgs->nrAicpu, devArgs->nrValidAic);
            DEV_INFO("devQueueAddr %lx, sharedBuffer %lx coreRegAddr %lx corePmuAdr %lx.", devArgs->devQueueAddr,
                devArgs->sharedBuffer, devArgs->coreRegAddr, devArgs->corePmuAddr);
            DEV_TRACE_DEBUG(schema::ScheEvent(threadIdx, schema::ThreadStart()));
            ret = machine_.Run(threadIdx, devArgs);
        } else {
            threadIdx = ctrlcpuIdx_.fetch_add(1);
            DEV_INFO("devArgs->taskType %d.",  static_cast<int>(devArgs->taskType));
            if (devArgs->taskType == DEVICE_TASK_TYPE_DYN && threadIdx == CTRL_CPU_THREAD_IDX) {
                CreateLogFile(LOG_TYPE_CONTROLLER, 0);
                DEV_TRACE_DEBUG(schema::CtrlEvent(threadIdx, schema::ThreadStart()));
                ret = machine_.ExecDyn(threadIdx, devArgs->taskId, args);
            } else if (threadIdx == MAX_SCHEDULE_AICPU_NUM + 1) {
                CreateLogFile(LOG_TYPE_PREFETCH, 0); 
                if (devArgs->taskType == DEVICE_TASK_TYPE_DYN) {
                  auto startArgs = (DevStartArgs *)devArgs->startArgsAddr;
                  DySdmaPrefetch(startArgs);
                } else {
                  auto devTask = reinterpret_cast<DeviceTask *>(devArgs->taskData);
                  SdmaPrefetch(devTask);
                }
            }
        }
        PerfMtTrace(PERF_TRACE_BEGIN, threadIdx, args->taskWastTime);
        PerfMtTrace(PERF_TRACE_ALLOC_THREAD_ID, threadIdx, allocThreadCycle);
        DEV_INFO("ThreadIdx %d finished, ret %d.", threadIdx, ret);
        GetLogger().Flush();
        PerfMtTrace(PERF_TRACE_EXIT, threadIdx);
        if (++finished_ == static_cast<std::atomic<int>>(devArgs->nrAicpu)) {
            LastFinishThreadIdx_ = threadIdx;
            return npu::tile_fwk::dynamic::DEVICE_MACHINE_FINISHED;
        }
        return ret;
    }

    void Init(DeviceArgs *args) {
        SignalReg();
        schAicpuNum_ = CalcSchAicpuNumByBlockDim(args->nrValidAic);
        machine_.init(args, schAicpuNum_);
    }

    void SignalReset() {
        sigaction(SIGFPE, &oriFPEAct_, nullptr);
        sigaction(SIGBUS, &oriBUSAct_, nullptr);
        sigaction(SIGSEGV, &oriSEGVAct_, nullptr);
        sigaction(SIGPIPE, &oriPIPEAct_, nullptr);
        sigaction(SIGILL, &oriILLAct_, nullptr);
        sigaction(SIGABRT, &oriBordAct_, nullptr);
        return;
    }

    void DeInit() {
      threadIdx_ = 0;
      finished_ = 0;
      cpumask_ = 0;
      ctrlcpuIdx_ = MAX_SCHEDULE_AICPU_NUM;
      SignalReset();
    }

    int LastFinishThreadIdx_{0};
    std::atomic<int> threadIdx_{0};
    std::atomic<int> finished_{0};
    std::atomic<uint64_t> cpumask_{0};
    std::atomic<int> ctrlcpuIdx_{MAX_SCHEDULE_AICPU_NUM};
    int schAicpuNum_{MAX_SCHEDULE_AICPU_NUM};
    DeviceMachine machine_;
    struct sigaction oriFPEAct_;
    struct sigaction oriBUSAct_;
    struct sigaction oriSEGVAct_;
    struct sigaction oriPIPEAct_;
    struct sigaction oriILLAct_;
    struct sigaction oriBordAct_;
    std::atomic<bool> reset_{false};
};

DynMachineManager g_machine_mgr;

void SigAct(int signum, siginfo_t* info, void* act) {
    (void)info;
    (void)act;
    DEV_ERROR("Exception Signum[%d] Act.", signum);
    if (g_machine_mgr.reset_.load()) {
      DEV_ERROR("Exception Already reset.");
      sleep(SIGNAL_DELAY_SECONDS);
      return;
    }
    g_machine_mgr.reset_.store(true);
    g_machine_mgr.machine_.ResetRegAll();
    sigaction(SIGFPE, &g_machine_mgr.oriFPEAct_, nullptr);
    sigaction(SIGBUS, &g_machine_mgr.oriBUSAct_, nullptr);
    sigaction(SIGSEGV, &g_machine_mgr.oriSEGVAct_, nullptr);
    sigaction(SIGPIPE, &g_machine_mgr.oriPIPEAct_, nullptr);
    sigaction(SIGILL, &g_machine_mgr.oriILLAct_, nullptr);
    sigaction(SIGABRT, &g_machine_mgr.oriBordAct_, nullptr);
    (void)raise(signum);
    return;
}
}

static int RunDynamic(AstKernelArgs *kargs) {
    int rc = g_machine_mgr.Run(kargs);
    if (rc == npu::tile_fwk::dynamic::DEVICE_MACHINE_FINISHED) {
        DEV_INFO("All schedule exited, destroy the machine.\n");
        g_machine_mgr.DeInit();
#if ENABLE_PERF_TRACE
        DEV_ERROR("Begin dump machine perf trace:");
        PerfMtTrace(PERF_TRACE_EXIT, g_machine_mgr.LastFinishThreadIdx_);
        PerfEvtMgr::Instance().DumpPerfTrace("/tmp/tile_fwk_aicpu_perftrace.json");
        DEV_IF_DEVICE {
            g_machine_mgr.machine_.DumpAicorePerfTrace("tmp/tile_fwk_aicore_perftrace.json");
        }
        DEV_ERROR("Finish dump machine perf trace.");
#endif
        return DEVICE_MACHINE_OK;
    }
    return rc;
}

static bool CheckValidArgs(AstKernelArgs *kargs) {
    if (kargs == nullptr) {
        return false;
    }
    if (kargs->inputs == nullptr || kargs->outputs == nullptr || kargs->workspace == nullptr
        || kargs->cfgdata == nullptr) {
        DEV_ERROR("Args has null in inputs[%p] outputs[%p] work[%p] or cfg[%p].\n", kargs->inputs,
                 kargs->outputs, kargs->workspace, kargs->cfgdata);
        return false;
    }
    return true;
}

extern "C" __attribute__((visibility("default"))) int DynTileFwkBackendKernelServerInit(void *targ) {
    PerfBegin(PERF_EVT_DEVICE_MACHINE_INIT_DYN);
#if DEBUG_PLOG && defined(__DEVICE__)
    InitLogSwitch();
#endif
    auto kargs = (AstKernelArgs *)targ;
    if (!CheckValidArgs(kargs)) {
        DEV_ERROR("invalid parameter.");
        return -EINVAL;
    }
    auto devArgs = PtrToPtr<int64_t, DeviceArgs>(kargs->cfgdata);
    DeviceMachine::InitDyn(kargs);
    g_machine_mgr.Init(devArgs);
    PerfEnd(PERF_EVT_DEVICE_MACHINE_INIT_DYN);
    return 0;
}

extern "C" __attribute__((visibility("default"))) int DynTileFwkBackendKernelServer(void *targ) {
    auto kargs = (AstKernelArgs *)targ;
    kargs->taskWastTime = GetCycles();
    return RunDynamic(kargs);
}
