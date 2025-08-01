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
 * \file device_machine.h
 * \brief
 */

#pragma once

#include <atomic>
#include <cstdint>

#include "aicore_manager.h"
#include "device_utils.h"
#include "device_context.h"
#include "machine/utils/dynamic/dev_encode.h"
#include "machine/utils/dynamic/device_channel.h"
#include "machine/utils/machine_ws_intf.h"
#include "machine/utils/device_log.h"
#include "device_utils.h"
namespace npu::tile_fwk::dynamic {

class DeviceMachine {
public:
    DeviceMachine() {}

    void init(DeviceArgs *args) {
        DEV_INFO("device machine init .");
        if (args->devQueueAddr != 0) {
            serverMode_ = true;
            receiver.init(reinterpret_cast<uint8_t *>(args->devQueueAddr), DEVICE_QUEUE_SIZE);
        }
        schAicpuNum_ = CalcSchAicpuNumByBlockDim(args->nrValidAic);
        for (uint32_t i = 0; i < schAicpuNum_; ++i) {
            aicoreManager_.push_back(std::make_unique<AiCoreManager>(aicpuTaskManager_));
        }

        coreNum_ = args->nrAic + args->nrAiv;
        sharedBuffer_ = args->sharedBuffer;

        if (args->taskType == DEVICE_TASK_TYPE_STATIC) {
            auto devTask = reinterpret_cast<DeviceTask *>(args->taskData);
#if DEBUG_PLOG && defined(__DEVICE__)
            if (CheckDebug()) {
#else
            if (GetLogger().Level() == LOG_LEVEL_DEBUG) {
#endif
                DumpTask(args->taskId, devTask, false);
            }
            auto idx = AllocNewTaskCtrl();
            InitTaskCtrl(idx, DEVICE_TASK_TYPE_STATIC, args->taskId, devTask, nullptr);
            initTaskCtrl = &taskctrl_[idx];
        }
    }

    int AllocNewTaskCtrl() {
        while (true) {
            if (taskCtrlIndex_ == MAX_DEVICE_TASK_NUM)
                taskCtrlIndex_ = 0;
            if (taskctrl_[taskCtrlIndex_].IsFree()) {
                return taskCtrlIndex_++;
            }
            taskCtrlIndex_++;
        }
    }

    void InitTaskCtrl(int idx, int type, uint64_t taskId, DeviceTask *devTask, DeviceExecuteContext *ctx, FinishCallback callback = nullptr) {
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
        taskCtrl->refcnt.store(schAicpuNum_, std::memory_order_relaxed);
        taskCtrl->runcnt.store(schAicpuNum_, std::memory_order_relaxed);
        taskCtrl->finish = callback;
        taskCtrl->ctx = ctx;
        devTask->aicoreModel = reinterpret_cast<uint64_t>(ctx->aicoreModel);
        if (ctx->costModelData != nullptr) {
            devTask->costModelData = reinterpret_cast<uint64_t>(ctx->costModelData);
        }
        for (auto& eType : taskCtrl->isAicpuIdle) {
            for (auto& e : eType) {
                e.store(true);
            }
        }
    }

    int PushTask(int type, uint64_t taskId, DeviceTask *devTask, DeviceExecuteContext *ctx, FinishCallback callback = nullptr) {
        auto idx = AllocNewTaskCtrl();
        InitTaskCtrl(idx, type, taskId, devTask, ctx, callback);
        for (auto &m : aicoreManager_) {
            m->PushTask(&taskctrl_[idx]);
        }
        return idx;
    }

    void StopAicoreManager() {
        for (auto &m : aicoreManager_) {
            m->PushTask(nullptr);
        }
    }

    int SyncTask(int idx) {
        while (!taskctrl_[idx].IsFree())
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

    int Run(int threadIdx, DeviceArgs *args) {
        int ret = 0;
        if (args->nrAic == 0 || args->nrValidAic == 0 || args->nrAicpu < NEED_LAUNCH_AICPU_MINNUM) {
            DEV_ERROR("Device machinr run invalid args aicnum:%u, blockdim:%u, launchAicpu num:%u",
                args->nrAic, args->nrValidAic, args->nrAicpu);
            return DEVICE_MACHINE_ERROR;
        }

        DEV_INFO("thread %d start .", threadIdx);
        if (static_cast<uint32_t>(threadIdx) >= MAX_SCHEDULE_AICPU_NUM) {
            DEV_INFO("thread start ignore ");
            return DEVICE_MACHINE_OK;
        }

        ret = aicoreManager_[threadIdx]->Run(threadIdx, args, initTaskCtrl);
        DEV_INFO("thread  %d end , ret = %d", threadIdx, ret);
        return ret;
    }

    static int InitDyn(AstKernelArgs *args) {
        auto kargs = (AstKernelArgs *) args;

        DEV_INFO("AscendCppDyInitTask begin");
        DevStartArgs *devArgs = PtrToPtr<int64_t, DevStartArgs>(kargs->workspace);

        auto inputPtr = PtrToPtr<DevStartArgs, DevAscendTensorData>(devArgs + 1);
        auto inputSize = DevAscendTensorDataCreator::Decode(kargs->inputs, inputPtr);

        auto outputPtr = inputPtr + inputSize;
        auto outputSize = DevAscendTensorDataCreator::Decode(kargs->outputs, outputPtr);
        auto workspaceAddr = ALIGN_UP((uint64_t)(outputPtr + outputSize), 512);
        auto devArgsSize = workspaceAddr - PtrToValue(kargs->workspace);

        auto devProg = PtrToPtr<int64_t, DevAscendProgram>(kargs->cfgdata);
        devArgs->inputTensorList = inputPtr;
        devArgs->inputTensorSize = static_cast<uint64_t>(inputSize);
        devArgs->outputTensorList = outputPtr;
        devArgs->outputTensorSize = static_cast<uint64_t>(outputSize);
        devArgs->workspaceAddr = workspaceAddr;
        devArgs->devProg = devProg;
        devArgs->aicpuCoherentWorkspaceSize = devProg->aicpuCoherentWorkspaceSize - devArgsSize;
        devArgs->aicoreLocalWorkspaceSize = devProg->workspaceSize - devProg->aicpuCoherentWorkspaceSize;
        devArgs->inputSymbolList = nullptr;
        devArgs->inputSymbolSize = 0;

        PerfBegin(PERF_EVT_INIT);
        if (devProg->controlFlowBinaryAddr == nullptr) {
            devProg->Reloc((uint64_t)devProg, true);
            auto execProg = DeviceExecuteProgram(devProg, nullptr);
            devProg->controlFlowBinaryAddr = execProg.GetControlFlowEntry();
        }
        devArgs->controlFlowEntry = devProg->controlFlowBinaryAddr;

        PerfEnd(PERF_EVT_INIT);
        DEV_INFO("AscendCppDyInitTask done.");
        return 0;
    }

    int ExecDyn(int threadIdx, uint64_t taskId, npu::tile_fwk::AstKernelArgs *args) {
        int ret = 0;
        DEV_INFO("start control flow.");
        auto devArgs = PtrToPtr<int64_t, DevStartArgs>(args->workspace);

        DeviceExecuteContext ctx(devArgs);
        ctx.costModelData = reinterpret_cast<CostModel::ModelData*>(args->costmodeldata);
        ctx.aicoreModel = args->aicoreModel;
        PerfBegin(PERF_EVT_EXEC_DYN);
        ctx.GELaunch(devArgs, [this](uint64_t dynTaskId, DeviceTask *devTask, DeviceExecuteContext *ctx_) {
#if DEBUG_SWITCH
            DumpTask(dynTaskId, (DeviceTask *)devTask, true);
#endif
            PushTask(DEVICE_TASK_TYPE_DYN, dynTaskId, devTask, ctx_, DeviceExecuteContext::TaskFinish);
        });
        DEV_INFO("end control flow.");

        PerfBegin(PERF_EVT_STAGE_TASK_SYNC);
        ret = SyncTask(&ctx.taskContext);
        PerfEnd(PERF_EVT_STAGE_TASK_SYNC);

        PerfBegin(PERF_EVT_STAGE_STOP_AICORE);
        StopAicoreManager();
        PerfEnd(PERF_EVT_STAGE_STOP_AICORE);
        PerfEnd(PERF_EVT_EXEC_DYN);

        ctx.ShowStats();

        PerfEvtMgr::Instance().Dump();
        PerfettoMgr::Instance().Dump("/tmp/perfetto.txt");

        (void)threadIdx;
        (void)taskId;
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
    DeviceTaskReceiver receiver;
    uint64_t sharedBuffer_{0};
    uint64_t coreNum_{0};
    bool serverMode_{false};
    DeviceTaskCtrl taskctrl_[MAX_DEVICE_TASK_NUM];
    uint64_t taskCtrlIndex_{0};
    DeviceTaskCtrl *initTaskCtrl{nullptr};
    AicpuTaskManager aicpuTaskManager_;
    uint32_t schAicpuNum_{MAX_SCHEDULE_AICPU_NUM};
    std::vector<std::unique_ptr<AiCoreManager>> aicoreManager_;
};
} // namespace npu::tile_fwk
