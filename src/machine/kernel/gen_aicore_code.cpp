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
 * \file gen_aicore_code.cpp
 * \brief
 */

#include "machine/kernel/gen_aicore_code.h"
#include "interface/utils/file_utils.h"

namespace npu::tile_fwk {
namespace {
const std::string kAicoreSrcCode = R"!!!(
#include <stdint.h>
#include <cstdint>
#include "tilefwk/aicpu_common.h"
#include "tilefwk/aicore_runtime.h"
#include "tilefwk/core_func_data.h"

// aicore head file begin
#ifndef __gm__
#define __gm__
#endif

#ifndef __global__
#define __global__
#endif

#ifndef __aicore__
#define __aicore__ [aicore]
#endif

#ifndef INLINE
#define INLINE __attribute__((always_inline)) inline __aicore__
#endif

#define TO_ENTRY_IMPL(name, line, key, type) (name##line##key##type)
#define TO_ENTRY(name, key, type) TO_ENTRY_IMPL(name, _, key, type)

#ifdef __MIX__
#ifdef __AIV__
#define KERNEL_ENTRY(x, y) TO_ENTRY(x, y, _mix_aiv)
#else
#define KERNEL_ENTRY(x, y) TO_ENTRY(x, y, _mix_aic)
#endif
#else
#define KERNEL_ENTRY(x, y) x
#endif

constexpr uint32_t REG_HIGH_DTASKID_SHIFT = 32;
enum class TASK_POS : size_t { LOW_REG = 0, HIGH_REG = 1, ALL_REG = 2, REG_POS_BUTT = 3 };

struct TaskStat {
    int16_t seqNo;
    int16_t subGraphId;
    int32_t taskId;
    int64_t execStart;
    int64_t execEnd;
    int64_t waitStart; // 2.0 dfx 当前未使用
};

struct Metrics {
  int64_t handShakeStart;
  int64_t handShakeEnd;
  int64_t kernelRunStart;
  int64_t kernelRunEnd;
  int64_t blockIdx;
  int64_t taskCount;
  int64_t isMetricStop;
  int64_t reserver[1];
  TaskStat tasks[];
};

struct TaskEntry {
    int32_t subGraphId;
    int32_t taskId;
    int64_t funcAddr;
    int64_t tensorAddrs;
    int64_t gmStackSize;
    int64_t gmStackBase;
    int64_t reserved2[2];
    uint32_t tensorSize;
    uint32_t reserved[1];
};

struct KernelArgs {
    int64_t shakeBuffer[8];
    TaskEntry taskEntry;
    TaskStat taskStat[2];
};

static_assert(sizeof(KernelArgs) < SHARED_BUFFER_SIZE);
// aicore head file end

// device switch head file begin
namespace npu::tile_fwk {
#define PERF_PMU_TEST_SWITCH 0 // PMU test switch
#define PERF_AICPU_TEST_SWITCH 0 //性能AICPU数据测试

// Disable DFX during performance testing, disable logging and partial traceability data collection.
#define DEBUG_SWITCH 0
#define DEBUG_INFINITE_LIFETIME 0

#define PERF_SWITCH 0

#if DEBUG_SWITCH == 0
#define DEBUG_PLOG 1
#else
#define DEBUG_PLOG 0
#endif/*DEBUG_PLOG*/

// whether to use the pending and running async task mode(set macro 1) or just use running sync mode(set macro 0)
#define SCHEDULE_USE_PENDING_AND_RUNING_SWITCH 1

/* The DFX swimlane performance statistics use host pre-allocated memory mode, which avoids data collection during
   AICPU scheduling to minimize scheduling interference. However, each AICore only supports tracking up to
   MAX_DFX_TASK_NUM_PER_CORE tasks, with excess tasks being discarded.
*/
#define PROF_DFX_HOST_PREPARE_MEMORY_MODE 1

// For DynamicFunction
#define DEBUG_MEM_DUMP_DISABLE 0
#define DEBUG_MEM_DUMP_LIGHT 1
#define DEBUG_MEM_DUMP_FULL 2

#define DEBUG_MEM_DUMP_LEVEL DEBUG_MEM_DUMP_LIGHT

#ifdef CONFIG_BAREMETAL

#define CONFIG_PROF                             0
#define CONFIG_COMM_WAIT_FLAG                   0

#ifdef __aarch64__
#define BAREMETAL_RAW_START()                   __asm__ __volatile__("orr x3, x3, x3" : : :"memory")
#define BAREMETAL_RAW_GET_PMU()                 __asm__ __volatile__("orr x4, x4, x4" : : :"memory")
#define BAREMETAL_RAW_GET_AND_RESET_PMU()       __asm__ __volatile__("orr x0, x0, x0" : : :"memory")
#else
#define BAREMETAL_RAW_START()
#define BAREMETAL_RAW_GET_PMU()
#define BAREMETAL_RAW_GET_AND_RESET_PMU()
#endif

#ifdef CONFIG_BAREMETAL_NOLOG
#define DEV_PROF(fmt, args...)
#else
#define DEV_PROF(fmt, args...) GetLogger().Log(LOG_LEVEL_ERROR, __FILE__, __LINE__, fmt, ##args)
#endif

#define PROF_START(...) \
    do { \
        BAREMETAL_RAW_START(); \
        DEV_PROF("[baremetal]start: " __VA_ARGS__); \
    } while (0)
#define PROF_STAGE_BEGIN_DYN(perfkey, ...) \
    do { \
        DEV_PROF("[baremetal]get: " __VA_ARGS__); \
        BAREMETAL_RAW_GET_PMU(); \
        PerfBegin(perfkey); \
    } while (0)
#define PROF_STAGE_END_DYN(perfkey, ...) \
    do { \
        PerfEnd(perfkey); \
        BAREMETAL_RAW_GET_PMU(); \
        DEV_PROF("[baremetal]get: " __VA_ARGS__); \
    } while (0)
#define PROF_STAGE_BEGIN(perfkey, ...)              PROF_STAGE_BEGIN_DYN(perfkey, __VA_ARGS__)
#define PROF_STAGE_END(perfkey, ...)                PROF_STAGE_END_DYN(perfkey, __VA_ARGS__)
#define PROF_STAGE_BEGIN_MTSAFE(perfkey, tid, ...)  PROF_STAGE_BEGIN_DYN(perfkey, __VA_ARGS__)
#define PROF_STAGE_END_MTSAFE(perfkey, tid, ...)    PROF_STAGE_END_DYN(perfkey, __VA_ARGS__)

#else

#define CONFIG_PROF                                 1
#define CONFIG_COMM_WAIT_FLAG                       1

#define PROF_START(...)
#define PROF_STAGE_BEGIN(perfkey, ...)              PerfBegin(perfkey)
#define PROF_STAGE_END(perfkey, ...)                PerfEnd(perfkey)
#define PROF_STAGE_BEGIN_MTSAFE(perfkey, tid, ...)  PerfMtBegin(perfkey, tid)
#define PROF_STAGE_END_MTSAFE(perfkey, tid, ...)    PerfMtEnd(perfkey, tid)

#endif
}
// device switch head file end

#define TO_STRING_IMPL(str) #str
#define TO_STRING(str) TO_STRING_IMPL(str)

#ifdef __HAS_SUB_FUNC__
#if defined(__MIX__) && defined(__AIV__)
#include TO_STRING(__HEAD_FILE__)
#else
#include TO_STRING(__HEAD_FILE__)
#endif
#endif

using npu::tile_fwk::DynFuncHeader;
using npu::tile_fwk::DynFuncData;
using npu::tile_fwk::DynFuncBin;
using npu::tile_fwk::DevRawTensorDesc;
using npu::tile_fwk::CoreFunctionData;

constexpr uint32_t STATUS_TASKID_SHIFT = 32;

#if defined(__MIX__) && defined(__AIV__)
#define blockIdx __v_blockIdx
#define GmWorkspace __v_GmWorkspace
#endif

[[block_local]] int blockIdx;
[[block_local]] int64_t GmWorkspace;

enum DFX_STAGE_STATUS {
    STAGE_HANDSHAKE_START = 1,
    STAGE_HANDSHAKE_END = 2,
    STAGE_GET_COREFUNC_DATA_STOP = 3,
    STAGE_GET_NEXT_TASK_STOP = 4,
    STAGE_PRE_EXEC_COREFUNC_KERNEL = 5,
    STAGE_FINISH_EXEC_COREFUNC_KERNEL = 6,
    STAGE_FINISH_PIPE_SYNC = 7,
    STAGE_FINISH_CUR_TASK = 8
};

struct ExecuteContext {
    __gm__ KernelArgs *args;
    uint32_t seqNo;
    __gm__ DynFuncData *funcDataList;
    __gm__ CoreFunctionData *staticFuncData;
};

typedef void (*StaticKernelFunc)(__gm__ int64_t *param, int64_t gmStackAddr, __gm__ int64_t *hcclContext, __gm__ int64_t *oriAddr);

INLINE uint32_t GetNextTask(uint32_t lastTaskIdx) {
    uint32_t nextLowIdx;
    uint64_t coreStatus;
    uint64_t t0 = get_sys_cnt();
    uint64_t loop_count = 0;
    do {
        __asm__ volatile("MOV %0, DATA_MAIN_BASE\n" : "+l"(coreStatus));
        nextLowIdx = coreStatus & 0xFFFFFFFF;
        nextLowIdx -= 1;
        ++loop_count;
        if ((loop_count % 1000 == 0) && (get_sys_cnt() - t0 > 500000000)) {
            break;
        }
    } while (nextLowIdx == lastTaskIdx);

    return nextLowIdx;
}

INLINE void PipeSync() {
#if defined(__AIV__)
    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);
    wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);
#else
    set_flag(PIPE_FIX, PIPE_S, EVENT_ID7);
    wait_flag(PIPE_FIX, PIPE_S, EVENT_ID7);
#endif
}

INLINE void Barrier()
{
#if defined(__CCE_KT_TEST__) && __CCE_KT_TEST__ == 1
    __asm__ __volatile__("" ::: "memory");
#else
    __asm__ __volatile__("");
#endif
}

INLINE void HandshakeClient(volatile __gm__ int64_t *shakeBuf) {
    volatile __gm__ int64_t *hello = shakeBuf;

    set_cond(AICORE_TASK_INIT);
    *hello = (int64_t)get_coreid() << 32 | AICORE_SAY_HELLO;
    Barrier();
    dcci(hello, SINGLE_CACHE_LINE, CACHELINE_OUT);
    Barrier();
}

INLINE void SetStatus(__gm__ KernelArgs *args, int64_t val) {
#if DEBUG_SWITCH
    Barrier();
    args->shakeBuffer[2] = val;
    dcci(args->shakeBuffer, SINGLE_CACHE_LINE, CACHELINE_OUT);
#endif
}

INLINE void SendRegFinsh(uint32_t curTaskIdx) {
    set_cond(curTaskIdx | AICORE_FIN_MASK);
}

INLINE void SendRegDevTaskStop(uint32_t dTaskId) {
    set_cond(((uint64_t)dTaskId << REG_HIGH_DTASKID_SHIFT) | (AICORE_FUNC_STOP | AICORE_FIN_MASK));
}

INLINE void SendRegAck(uint32_t taskIdx) {
    set_cond(taskIdx);
}

INLINE void SetTaskStatistic(__gm__ KernelArgs *args, int32_t& dfxPose,
                             int32_t taskId, int32_t subGraphId, int64_t tStart)
{
    __gm__ volatile TaskStat *stat = &args->taskStat[dfxPose];
    stat->subGraphId = subGraphId;
    stat->taskId = taskId;
    stat->execStart = tStart;
    stat->execEnd = get_sys_cnt();
    dcci(stat, SINGLE_CACHE_LINE, CACHELINE_OUT);
}

INLINE void AddMetricStatistic(__gm__ KernelArgs *args, uint32_t seqNo, uint32_t taskId, int32_t subGraphId, int64_t t1) {
#if PROF_DFX_HOST_PREPARE_MEMORY_MODE
    auto m = (__gm__ Metrics*)(args->shakeBuffer[SHAK_BUF_DFX_DATA_INDEX]);
    if (m && m->taskCount < MAX_DFX_TASK_NUM_PER_CORE) {
        m->tasks[m->taskCount].subGraphId = subGraphId;
        m->tasks[m->taskCount].seqNo = seqNo;
        m->tasks[m->taskCount].taskId = taskId;
        m->tasks[m->taskCount].execStart = t1;
        m->tasks[m->taskCount].execEnd = get_sys_cnt();
        m->taskCount++;
    }
#endif
}

INLINE void FlushMetricStatistic(__gm__ volatile KernelArgs* args) {
    __gm__ volatile Metrics* m = (__gm__ volatile Metrics*)(args->shakeBuffer[SHAK_BUF_DFX_DATA_INDEX]);
    if (m == nullptr) {
        return;
    }
    for (uint32_t i = 0; i < m->taskCount; i++) {
        dcci(&m->tasks[i], SINGLE_CACHE_LINE, CACHELINE_OUT);
    }
    m->isMetricStop = 1;
    dcci(m, SINGLE_CACHE_LINE, CACHELINE_OUT);
}

INLINE uint64_t getCoreFuncionData(__gm__ KernelArgs *args, int64_t lastFunc) {
    uint32_t nextLowIdx;
    uint64_t coreStatus;
    uint64_t t0 = get_sys_cnt();
    uint64_t loop_count = 0;
    while (true) {
        // check if stop
        ++loop_count;
        if ((loop_count % 1000 == 0) && (get_sys_cnt() - t0 > 500000000)) {
            break;
        }
        volatile __gm__ int64_t *shakebuffer = args->shakeBuffer;
        dcci(shakebuffer, SINGLE_CACHE_LINE, CACHELINE_OUT);
        auto newFunc = args->shakeBuffer[SHAK_BUF_COREFUNC_DATA_INDEX];
        if (newFunc != lastFunc && newFunc != 0) {
            dcci((__gm__ void *)newFunc, SINGLE_CACHE_LINE, CACHELINE_OUT);
            return newFunc;
        }

        __asm__ volatile("MOV %0, DATA_MAIN_BASE\n" : "+l"(coreStatus));
        nextLowIdx = coreStatus & 0xFFFFFFFF;
        nextLowIdx -= 1;

        if (nextLowIdx == AICORE_TASK_STOP) {
            return 0;
        }
    }
    return 0;
}

INLINE void PmuTestBegin(__gm__ KernelArgs *args) {
#if PERF_PMU_TEST_SWITCH
    if (args->taskEntry.reserved[0] == PRO_LEVEL2) {
        set_ctrl((uint64_t) get_ctrl() | 0x1);
    }
#endif
}

INLINE void PmuTestEnd(__gm__ KernelArgs *args) {
#if PERF_PMU_TEST_SWITCH
        if (args->taskEntry.reserved[0] == PRO_LEVEL2) {
            set_ctrl((uint64_t) get_ctrl() - 1);
        }
#endif
}

#define FuncNum(id)      TaskID(id)

INLINE void ExecStaticCoreFunctionKernel(ExecuteContext *ctx, uint32_t taskId) {
#if PROF_DFX_HOST_PREPARE_MEMORY_MODE != 1
    static int32_t taskDfxPos = REG_LOW_TASK_PING;
#endif
    __gm__ CoreFunctionData* coreFuncData = ctx->staticFuncData;
    uint64_t t1 = get_sys_cnt();
    SetStatus(ctx->args,  ((uint64_t)taskId << STATUS_TASKID_SHIFT) | STAGE_PRE_EXEC_COREFUNC_KERNEL);
    __gm__ npu::tile_fwk::CoreFunctionWsAddr* functionInfo =
            &((__gm__ npu::tile_fwk::CoreFunctionWsAddr*)coreFuncData->coreFunctionWsAddr)[taskId];
    StaticKernelFunc kernel = (StaticKernelFunc)functionInfo->functionBinAddr;
    kernel((__gm__ int64_t *)functionInfo->invokeEntryAddr,
           coreFuncData->stackWorkSpaceAddr + blockIdx * coreFuncData->stackWorkSpaceSize,
           (__gm__ int64_t *)coreFuncData->hcclContextAddr,
           (__gm__ int64_t *)functionInfo->invokeEntryOriAddr);

    SetStatus(ctx->args, STAGE_FINISH_EXEC_COREFUNC_KERNEL);
    PipeSync();
    SetStatus(ctx->args, STAGE_FINISH_PIPE_SYNC);

#if PROF_DFX_HOST_PREPARE_MEMORY_MODE != 1
    SetTaskStatistic(ctx->args, taskDfxPos, taskId, (int32_t)functionInfo->psgId, t1);
#endif

    AddMetricStatistic(ctx->args, 0, taskId, (int32_t)functionInfo->psgId, t1);
}

#ifdef __HAS_SUB_FUNC__
INLINE void ExecDynCoreFunctionKernel(ExecuteContext *ctx, uint32_t taskId) {
    uint64_t t1 = get_sys_cnt();

    SetStatus(ctx->args, ((uint64_t)taskId << STATUS_TASKID_SHIFT) | STAGE_PRE_EXEC_COREFUNC_KERNEL); // high 32 bits used for taskId

    auto funcData = &ctx->funcDataList[FuncID(taskId)];
    auto opAttrs = &funcData->opAttrs[funcData->opAtrrOffsets[TaskID(taskId)]];
    CoreFuncParam param = {funcData, opAttrs, funcData->exprTbl, taskId};
    CallSubFuncTask(opAttrs[0], &param, funcData->stackWorkSpaceAddr + blockIdx * funcData->stackWorkSpaceSize,
                    (__gm__ int64_t *)funcData->hcclContext);
    SetStatus(ctx->args, STAGE_FINISH_EXEC_COREFUNC_KERNEL);
    PipeSync();
    SetStatus(ctx->args, STAGE_FINISH_PIPE_SYNC);
    AddMetricStatistic(ctx->args, ctx->seqNo, taskId, opAttrs[0], t1);
#if PROF_DFX_HOST_PREPARE_MEMORY_MODE != 1
    static int32_t taskDfxPos = REG_LOW_TASK_PING;
    SetTaskStatistic(ctx->args, taskDfxPos, taskId, opAttrs[0], t1);
#endif
}
#endif

INLINE void InitCtx(ExecuteContext *ctx, uint64_t coreFuncData, bool isDyn) {
    if (isDyn) {
        __gm__ DynFuncHeader *header = (__gm__ DynFuncHeader *)coreFuncData;
        ctx->seqNo = header->seqNo;
        ctx->funcDataList = (__gm__ npu::tile_fwk::DynFuncData *)(header + 1);
        dcci((__gm__ void *)0, ENTIRE_DATA_CACHE, CACHELINE_OUT);
        return;
    }

    ctx->staticFuncData = (__gm__ npu::tile_fwk::CoreFunctionData*)coreFuncData;
}

INLINE void ExecCoreFunctionKernel(ExecuteContext *ctx, uint32_t curTaskIdx, bool isDyn) {
#ifdef __HAS_SUB_FUNC__
    if (isDyn) {
        ExecDynCoreFunctionKernel(ctx, curTaskIdx);
        return;
    }
#endif
    ExecStaticCoreFunctionKernel(ctx, curTaskIdx);
}

extern "C" __global__ __aicore__ void KERNEL_ENTRY(__OPTYPE__, __TILINGKEY__)(int64_t ffts_addr, int64_t inputs,
        int64_t outputs, int64_t workspace, int64_t tilingdata, int64_t cfgdata) {
#if defined(__AIV__) and defined(__MIX__)
    blockIdx = get_block_idx() * get_subblockdim() + get_subblockid() + get_block_num();
#else
    blockIdx = get_block_idx();
#endif
    auto devArgs = (DeviceArgs*)cfgdata;
    __gm__ KernelArgs *args = (__gm__ KernelArgs *)(devArgs->sharedBuffer + blockIdx * SHARED_BUFFER_SIZE);
    bool isDyn = devArgs->taskType == DEVICE_TASK_TYPE_DYN ? true : false;

    SetStatus(args, STAGE_HANDSHAKE_START);
    HandshakeClient(args->shakeBuffer);
    SetStatus(args, STAGE_HANDSHAKE_END);
    set_mask_norm();
    uint32_t curTaskIdx;
    uint32_t lastTaskIdx;
    int64_t coreFuncData = 0;
    ExecuteContext ctx = {.args = args };
    //get core task data
    uint64_t t0 = get_sys_cnt();
    uint64_t loop_count = 0;
    while (true) {
        ++loop_count;
        if ((loop_count % 1000 == 0) && (get_sys_cnt() - t0 > 3000000000)) {
            break;
        }
        lastTaskIdx = AICORE_TASK_INIT;
        coreFuncData = getCoreFuncionData(args, coreFuncData);
        if (coreFuncData == 0) {
            FlushMetricStatistic(args);
            SetStatus(args, STAGE_GET_COREFUNC_DATA_STOP);
            return; // no data exit
        }
        InitCtx(&ctx, coreFuncData, isDyn);
        uint64_t t1 = get_sys_cnt();
        uint64_t inner_loop_count = 0;
        while (true) {
            ++inner_loop_count;
            if ((inner_loop_count % 1000 == 0) && (get_sys_cnt() - t1 > 3000000000)) {
                break;
            }
            curTaskIdx = GetNextTask(lastTaskIdx);
            if (curTaskIdx == AICORE_TASK_STOP || curTaskIdx == AICORE_FUNC_STOP) {
                SetStatus(args, STAGE_GET_NEXT_TASK_STOP);
                if (isDyn) {
                    SendRegDevTaskStop(ctx.seqNo);
                    break;
                } else {
                    FlushMetricStatistic(args);
                    return;
                }
            }

            SendRegAck(curTaskIdx);
            PmuTestBegin(args);
            ExecCoreFunctionKernel(&ctx, curTaskIdx, isDyn);
            PmuTestEnd(args);
            SendRegFinsh(curTaskIdx);
            lastTaskIdx = curTaskIdx;
            SetStatus(args, STAGE_FINISH_CUR_TASK);
        }
    }
}
)!!!";
}

std::string GenAndGetAicoreCodeSrcPath() {
    std::string codeSrcPath = "./aicore.cpp";
    DumpFile(kAicoreSrcCode, codeSrcPath);
    return RealPath(codeSrcPath);
}
}
