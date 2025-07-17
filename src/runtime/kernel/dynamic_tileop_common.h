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
 * \file dynamic_tileop_common.h
 * \brief
 */

#ifndef __DYNAMIC_ASCENDTENSOR_TILEOP_COMMON__
#define __DYNAMIC_ASCENDTENSOR_TILEOP_COMMON__

constexpr uint64_t REPEAT_MAX = 255;
constexpr uint64_t REPEAT_BYTE = 256;

#ifndef __aicore__
#define __aicore__ [aicore]
#endif

#ifndef TILEOP
#define TILEOP static __attribute__((always_inline)) __aicore__
#endif

#ifndef INLINE
#define INLINE __attribute__((always_inline)) inline __aicore__
#endif

#ifndef CORELOG
#define CORELOG(x...)
#endif

#define SUBKERNEL_PHASE1
#define SUBKERNEL_PHASE2

#if defined(__DAV_C220_VEC__)
#define WAIT_TASK_FIN                       \
    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID7); \
    wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID7)
#else
#define WAIT_TASK_FIN                      \
    set_flag(PIPE_FIX, PIPE_S, EVENT_ID7); \
    wait_flag(PIPE_FIX, PIPE_S, EVENT_ID7)
#endif

#if defined(__DAV_C220_VEC__)
#define WAIT_PRE_TASK                          \
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID7); \
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID7)
#else
#define WAIT_PRE_TASK                          \
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID7); \
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID7)
#endif

#endif
