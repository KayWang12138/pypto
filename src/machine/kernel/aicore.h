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
 * \file aicore.h
 * \brief
 */

#pragma once
#include <cstdint>
#include "tilefwk/aicpu_common.h"
#include "tilefwk/core_func_data.h"
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
    TaskStat taskStat[2]; // 寄存器高低32位，两个task 和 pending & running task存储： 2 * 2 个
};

static_assert(sizeof(KernelArgs) < SHARED_BUFFER_SIZE);
