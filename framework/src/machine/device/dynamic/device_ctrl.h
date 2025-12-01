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
 * \file device_ctrl.h
 * \brief
 */

#pragma once

#include "device_common.h"
#include <cstdint>
#include <cstdlib>
#include "device_utils.h"
#include "device_perf.h"
#include "device_context.h"
#include "machine/utils/dynamic/dev_encode.h"
#include "machine/utils/machine_ws_intf.h"
#include "machine/utils/device_log.h"

#ifdef __USE_CUSTOM_CTRLFLOW__
extern "C" __attribute__((weak)) void* GetTilingKeyFunc(const uint64_t tilingKey);    
#endif

namespace npu::tile_fwk::dynamic {
class DeviceCtrlMachine {
 public:
    void InitTaskCtrl(int idx, int type, uint64_t taskId, DeviceTask *devTask, DeviceExecuteContext *ctx) {
        if (ctx == nullptr) {
            DEV_ERROR("Init Task control failed, which ctx is null.");
            return;
        }
        auto taskCtrl = &taskctrl_[idx];
        taskCtrl->taskType = type;
        taskCtrl->devTask = devTask;
        taskCtrl->taskId = taskId;
        taskCtrl->initAicFuncNum = ((ReadyCoreFunctionQueue *)devTask->readyAicCoreFunctionQue)->Size();
        taskCtrl->initAivFuncNum = ((ReadyCoreFunctionQueue *)devTask->readyAivCoreFunctionQue)->Size();
        taskCtrl->finishedAicFunctionCnt = 0;
        taskCtrl->finishedAivFunctionCnt = 0;
        taskCtrl->finishedAicpuFunctionCnt = 0;
        taskCtrl->finishedFunctionCnt.store(0, std::memory_order_relaxed);
        taskCtrl->runFlag.store(true, std::memory_order_relaxed);
        taskCtrl->runcnt.store(schAicpuNum_, std::memory_order_relaxed);
        taskCtrl->ctx = ctx;
        devTask->aicoreModel = reinterpret_cast<uint64_t>(ctx->aicoreModel);
        if (ctx->costModelData != nullptr) {
            devTask->costModelData = reinterpret_cast<uint64_t>(ctx->costModelData);
        }
        for (size_t i = 0; i < AICORE_TYPE_NUM; ++i) {
            for (size_t j = 0; j < MAX_SCHEDULE_AICPU_NUM; ++j) {
                taskCtrl->isAicpuIdle[i][j].store(true);
            }
        }
    }

    int AllocNewTaskCtrl() {
        while (true) {
            if (taskCtrlIndex_ == MAX_DEVICE_TASK_NUM)
                taskCtrlIndex_ = 0;
            if (!taskctrl_[taskCtrlIndex_].IsNotFree()) {
                return taskCtrlIndex_++;
            }
            taskCtrlIndex_++;
        }
    }

    int PushTask(int type, DynDeviceTask *dynTask, DeviceExecuteContext *ctx) {
        auto idx = AllocNewTaskCtrl();
        InitTaskCtrl(idx, type, dynTask->GetIndex(), &dynTask->devTask, ctx);
        for (uint32_t i = 0; i < schAicpuNum_; ++i) {
            taskQueue_[i].Enqueue(&taskctrl_[idx]);
        }
        return idx;
    }

    void StopAicoreManager() {
        for (uint32_t i = 0; i < schAicpuNum_; ++i) {
            taskQueue_[i].Enqueue(nullptr);
        }
    }

    int SyncTask(int idx) {
        while (taskctrl_[idx].IsNotFree())
            ;
        return taskctrl_[idx].retCode;
    }

    int SyncTask(DeviceTaskContext *taskContext = nullptr) {
        int ret = 0;
        for (int idx = 0; idx < MAX_DEVICE_TASK_NUM; idx++) {
            auto rc = SyncTask(idx);
            if (rc != 0) {
                ret = rc;
            }
            if (taskContext) {
                taskContext->ReleaseFinishedTasks(PERF_EVT_RELEASE_FINISH_TASK_INSYNC, PERF_EVT_DEALLOCATE_TASK_INSYNC);
            }
        }
        return ret;
    }

    int InitDyn(AstKernelArgs *kargs) {
        DEV_INFO("AscendCppDyInitTask begin");
        auto devProg = PtrToPtr<int64_t, DevAscendProgram>(kargs->cfgdata);
        auto devArgs = reinterpret_cast<DevStartArgs *>(devProg->devArgs.startArgsAddr);
        taskctrl_ = reinterpret_cast<DeviceTaskCtrl *>(devProg->devArgs.taskCtrl);
        taskQueue_ = reinterpret_cast<SPSCQueue<DeviceTaskCtrl *, DEFAULT_QUEUE_SIZE> *>(devProg->devArgs.taskQueue);
        schAicpuNum_ = devProg->devArgs.scheCpuNum;
        PerfBegin(PERF_EVT_INIT);
        bool firstInit = false;
        if (devProg->controlFlowBinaryAddr == nullptr) {
            devProg->RelocProgram(0, reinterpret_cast<uint64_t>(devProg), true);
            firstInit = true;
        }
#ifdef __USE_CUSTOM_CTRLFLOW__
        devProg->controlFlowBinaryAddr = GetTilingKeyFunc(devProg->configKey);
#else
        auto execProg = DeviceExecuteProgram(devProg, nullptr);
        devProg->controlFlowBinaryAddr = execProg.GetControlFlowEntry();
#endif
        devArgs->controlFlowEntry = devProg->controlFlowBinaryAddr;

        PerfEnd(PERF_EVT_INIT);

        auto inputPtr = PtrToPtr<DevStartArgs, DevTensorData>(devArgs + 1);
        auto inputSize = DevAscendTensorDataCreator::Decode(kargs->inputs, devProg, 0, inputPtr);

        auto outputPtr = inputPtr + inputSize;
        auto outputSize = DevAscendTensorDataCreator::Decode(kargs->outputs, devProg, inputSize, outputPtr);
        devArgs->devTensorList = inputPtr;
        devArgs->inputTensorSize = static_cast<uint64_t>(inputSize);
        devArgs->outputTensorSize = static_cast<uint64_t>(outputSize);
        devArgs->contextWorkspaceAddr = PtrToValue(kargs->workspace);
        devArgs->contextWorkspaceSize = devProg->workspaceSize;
        devArgs->devProg = devProg;

        devArgs->inputSymbolList = nullptr;
        devArgs->inputSymbolSize = 0;
        devArgs->hcclContextAddr = (uint64_t*)&devProg->hcclContext[0];

        if (devProg->controlFlowCache.isRecording) {
            devProg->controlFlowCache.contextWorkspaceAddr = devArgs->contextWorkspaceAddr;
        }
        DEV_INFO("ControlFlowCache: deviceTask:%d firstInit:%d\n", (int)devProg->controlFlowCache.deviceTaskCount, (int)firstInit);
        if (devProg->controlFlowCache.deviceTaskCount != 0) {
            // Actual run
            if (firstInit) {
                devProg->controlFlowCache.RelocProgram(0, reinterpret_cast<uint64_t>(devProg));
                devProg->controlFlowCache.RelocWorkspace(0, devArgs->contextWorkspaceAddr);
                devProg->controlFlowCache.RuntimeAddrRelocProgram(0, reinterpret_cast<uint64_t>(devProg));
            }
            devProg->controlFlowCache.IncastOutcastAddrRestore();
            devProg->controlFlowCache.IncastOutcastAddrReloc(0, devArgs->contextWorkspaceAddr, devArgs);
            devProg->ResetRerun();
        }
        DEV_INFO("AscendCppDyInitTask done.");
        return 0;
    }

    int ExecDyn(npu::tile_fwk::AstKernelArgs *args) {
        int ret = 0;
        DEV_INFO("start control flow.");
        auto devProg = PtrToPtr<int64_t, DevAscendProgram>(args->cfgdata);
        auto devStartArgs = (DevStartArgs *)devProg->devArgs.startArgsAddr;
        DeviceExecuteContext ctx(devStartArgs);
        ctx.costModelData = reinterpret_cast<CostModel::ModelData*>(args->costmodeldata);
        ctx.aicoreModel = args->aicoreModel;
        PerfBegin(PERF_EVT_EXEC_DYN);
        PerfBegin(PERF_EVT_CONTROL_FLOW_CALL);
        ctx.GELaunch(devStartArgs, [this](DynDeviceTask *dynTask, DeviceExecuteContext *exeCtx) {
            DEV_IF_DEBUG {
                DumpTask(dynTask->GetIndex(), (DeviceTask *)dynTask, true);
            }
            PushTask(DEVICE_TASK_TYPE_DYN, dynTask, exeCtx);
        });
        PerfEnd(PERF_EVT_CONTROL_FLOW_CALL);
        DEV_INFO("end control flow.");
        PerfBegin(PERF_EVT_STAGE_STOP_AICORE);
        StopAicoreManager();
        PerfEnd(PERF_EVT_STAGE_STOP_AICORE);
        DEV_INFO("aicore manager stopped");

        PerfBegin(PERF_EVT_STAGE_TASK_SYNC);
        ret = SyncTask(&ctx.taskContext);
        PerfMtTrace(PERF_TRACE_WAIT_ALL_DEV_TASK_FINISH, CTRL_CPU_THREAD_IDX);
        PerfEnd(PERF_EVT_STAGE_TASK_SYNC);
        PerfEnd(PERF_EVT_EXEC_DYN);
#if ENABLE_PERF_EVT
        ctx.ShowStats();
        PerfEvtMgr::Instance().Dump();
        PerfettoMgr::Instance().Dump("/tmp/perfetto.txt");
    #endif
        return ret;
    }

private:
    static void DumpTask(int64_t taskId, DeviceTask *devTask, bool isDyn) {
        DEV_DEBUG("devTask %ld %p.", taskId, devTask);
        if (devTask == nullptr) {
            return;
        }

        DEV_DEBUG("devtask { %lu, %lx, %lx, %lx, %lx, %lu, %lu}.", devTask->coreFunctionCnt,
            devTask->coreFunctionReadyStateAddr, devTask->readyAicCoreFunctionQue, devTask->readyAivCoreFunctionQue,
            devTask->coreFuncData.coreFunctionWsAddr, devTask->coreFuncData.stackWorkSpaceAddr,
            devTask->coreFuncData.stackWorkSpaceSize);

        DEV_DEBUG("===== ready aic func =====");
        ReadyCoreFunctionQueue* readyFunc = reinterpret_cast<ReadyCoreFunctionQueue*>(devTask->readyAicCoreFunctionQue);
        for (uint64_t i = readyFunc->head; i < readyFunc->tail; i++) {
            DEV_DEBUG( "taskId %u.", readyFunc->elem[i]);
        }

        DEV_DEBUG("===== ready aiv func =====");
        readyFunc = reinterpret_cast<ReadyCoreFunctionQueue *>(devTask->readyAivCoreFunctionQue);
        for (uint64_t i = readyFunc->head; i < readyFunc->tail; i++) {
            DEV_DEBUG( "taskId %u.", readyFunc->elem[i]);
        }
        DEV_DEBUG("===== ready aicpu func =====");
        readyFunc = reinterpret_cast<ReadyCoreFunctionQueue *>(devTask->readyAicpuFunctionQue);
        for (uint64_t i = readyFunc->head; i < readyFunc->tail; i++) {
            DEV_DEBUG( "taskId %u.", readyFunc->elem[i]);
        }

        if (isDyn) {
            DEV_DEBUG("===== dyn info =====");
            auto dyntask = PtrToPtr<DeviceTask, DynDeviceTask>(devTask);
            int funcIdx = 0;
            for (auto &func : dyntask->stitchedList) {
                DEV_DEBUG("func %d %s.", funcIdx, func.DumpDyn(funcIdx, dyntask->cceBinary).c_str());
                funcIdx++;
                (void)func;
            }
        } else {
            auto coreFunc = reinterpret_cast<CoreFunctionWsAddr *>(devTask->coreFuncData.coreFunctionWsAddr);
            DEV_DEBUG("===== core func =====");
            for (uint64_t i = 0; i < devTask->coreFunctionCnt; i++) {
                DEV_DEBUG("taskId %lu binAddr %lx invokeEntry %lx topo %lx.", i, coreFunc[i].functionBinAddr,
                    coreFunc[i].invokeEntryAddr, coreFunc[i].topoAddr);
                auto topo = reinterpret_cast<CoreFunctionTopo *>(coreFunc[i].topoAddr);
                DEV_DEBUG("coreType %lu pstId %lu readyCount %ld depNum %lu .", topo->coreType, topo->psgId,
                    topo->readyCount, topo->depNum);
                (void)topo;
            }
            DEV_DEBUG("===== ready state =====");
            auto readyState = reinterpret_cast<CoreFunctionReadyState *>(devTask->coreFunctionReadyStateAddr);
            for (uint64_t i = 0; i < devTask->coreFunctionCnt; i++) {
                DEV_DEBUG("taskId %lu readyCount %ld coreType %lu.", i, readyState[i].readyCount, readyState[i].coreType);
            }
            (void)(readyState);
        }
        (void)taskId;
        DEV_DEBUG("===== dev task end =====");
    }
 private:
    uint32_t taskCtrlIndex_{0};
    DeviceTaskCtrl *taskctrl_{nullptr};
    SPSCQueue<DeviceTaskCtrl *, DEFAULT_QUEUE_SIZE> *taskQueue_{nullptr};
    uint32_t schAicpuNum_{MAX_SCHEDULE_AICPU_NUM};
};
} // namespace npu::tile_fwk
