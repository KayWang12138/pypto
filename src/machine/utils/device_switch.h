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
 * \file device_switch.h
 * \brief
 */

#pragma once
#ifndef DEVICE_SWITCH_H
#define DEVICE_SWITCH_H

namespace npu::tile_fwk {
#define PERF_PMU_TEST_SWITCH 0 // PMU test switch
#define PERF_AICPU_TEST_SWITCH 0 //性能AICPU数据测试

// Disable DFX during performance testing, disable logging and partial traceability data collection.
#define DEBUG_SWITCH 0

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

// ready quene mode for aicore task : Last-in-first-out(LIFO stack mode) or first-in-first-out(FIFO quene mode)
constexpr bool READY_QUE_LIFO_SWITCH = true;

// tasks are dispatched immediately once dependencies are resolved, and the remaining tasks are enqueued.
constexpr bool SEND_TASK_IMMEDIATELY_SWITCH = true;

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
} // namespace npu::tile_fwk
#endif
