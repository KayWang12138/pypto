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
 * \file device_utils.h
 * \brief Device utility functions and timeout detection macros
 */

#ifndef DEVICE_UTILS_H
#define DEVICE_UTILS_H
#include <cstddef>
#include <cstdint>
#include <stdint.h>
#include <time.h>
#include <atomic>
#include <vector>
#include <mutex>
#include <sstream>
#include <fstream>
#include <ostream>
#include "securec.h"
#include "machine/utils/device_log.h"
#include "interface/machine/device/tilefwk/aicpu_perf.h"

#ifndef CONFIG_MAX_DEVICE_TASK_NUM
#define CONFIG_MAX_DEVICE_TASK_NUM 1024
#endif

#define MAX_DEVICE_TASK_NUM CONFIG_MAX_DEVICE_TASK_NUM

#define BITS_PER_BYTES 8
#define BITS_PER_INT 32
#define BITS_PER_LONG 64

namespace npu::tile_fwk::dynamic {
inline constexpr bool IsDeviceMode()
{
#ifdef __DEVICE__
    return true;
#else
    return false;
#endif // __DEVICE__
}

constexpr int32_t DEVICE_MACHINE_INVALID_RUN_MODE = -40;
constexpr int32_t DEVICE_MACHINE_TIMEOUT_SCH_PARALLEL_DEVTASK = -7;
constexpr int32_t DEVICE_MACHINE_TIMEOUT_SYNC_AICPU_FINISH = -6;
constexpr int32_t DEVICE_MACHINE_TIMEOUT_SYNC_CORE_FINISH = -5;
constexpr int32_t DEVICE_MACHINE_TIMEOUT_AIV = -4;
constexpr int32_t DEVICE_MACHINE_TIMEOUT_AIC = -3;
constexpr int32_t DEVICE_MACHINE_TIMEOUT_CORETASK = -2;
constexpr int32_t DEVICE_MACHINE_ERROR = -1;
constexpr int32_t DEVICE_MACHINE_OK = 0;
constexpr int32_t DEVICE_MACHINE_FINISHED = 1;
constexpr int32_t TIME_OUT_THRESHOLD = 1000000;      // 超时阈值 1s
constexpr int32_t DFX_TIME_OUT_THRESHOLD = 50000000; // 超时阈值 50s
constexpr uint32_t CTRL_CPU_THREAD_IDX = 0;
constexpr int32_t START_AICPU_NUM = 3;
constexpr uint64_t NUM_FIFTY = 50;
constexpr uint64_t US_PER_SEC = 1000000;
constexpr uint64_t NSEC_PER_USEC = 1000;
constexpr uint64_t NSEC_PER_SEC = 1000000000;
constexpr uint64_t HAND_SHAKE_TIMEOUT = 48000000000; // aicpu stream wait hccl finish
constexpr int32_t MAX_MNG_AICORE_AVG_NUM = 8;
constexpr uint32_t CORE_IDX_AIV = 0;
constexpr uint32_t CORE_IDX_AIC = 1;
const uint32_t AIV_NUM_PER_AI_CORE = 2;
const int INVALID_CORE_IDX = 0xFF;

// ========== 统一超时常量定义（基于纳秒）==========
// 注意：A2/A3 频率约 50 MHz（50 cycles/ns），A5 频率约 1000 MHz（1000 cycles/ns）
// 统一使用纳秒定义超时时间，通过 GetFreq() 动态转换确保跨平台一致性

constexpr uint64_t TIMEOUT_NS_1MIN     = 60ULL * 1000ULL * 1000ULL * 1000ULL;   // 1 分钟
constexpr uint64_t TIMEOUT_NS_10MIN    = 600ULL * 1000ULL * 1000ULL * 1000ULL;  // 10 分钟
constexpr uint64_t TIMEOUT_NS_20MIN    = 1200ULL * 1000ULL * 1000ULL * 1000ULL; // 20 分钟

// Legacy constants（保持兼容性，但标记为 deprecated）
constexpr uint64_t TIMEOUT_ONE_MINUTE = 3000000000;  // DEPRECATED: 实际是 3 秒，命名误导

#ifdef __aarch64__
constexpr uint64_t TIMEOUT_CYCLES = 500 * 1000 * 1000;  // 约 250ms @2GHz（legacy）
#else
constexpr uint64_t TIMEOUT_CYCLES = NSEC_PER_SEC;
#endif

constexpr uint64_t PROF_DUMP_TIMEOUT_CYCLES = TIMEOUT_CYCLES;

// ========== 新增错误码 ==========
constexpr int32_t DEVICE_MACHINE_TIMEOUT_SLAB_ALLOC = -8;
constexpr int32_t DEVICE_MACHINE_TIMEOUT_THREAD_ALLOC = -9;
constexpr int32_t DEVICE_MACHINE_TIMEOUT_RINGBUFFER = -10;
constexpr int32_t DEVICE_MACHINE_TIMEOUT_CTRL_ALLOC = -11;

#define PERF_LEVEL 0
#define PERF_AICORE_THREAD_START 100

#define PERF_EVENTS                  \
    X(0, EXEC_DYN)                   \
    X(1, INIT)                       \
    X(2, CONTROL_FLOW_MAPEXE)        \
    X(3, CONTROL_FLOW_MAPEXE_MEMCPY) \
    X(1, CONTROL_FLOW_CALL)          \
    X(1, CONTROL_FLOW_INIT)          \
    X(1, CONTROL_FLOW)               \
    X(2, ROOT_FUNC)                  \
    X(3, STAGE_DUP_ROOT)             \
    X(3, ALLOCATE_WORKSPACE)         \
    X(3, STAGE_STITCH)               \
    X(4, FAST_STITCH)                \
    X(4, UPDATE_SLOT)                \
    X(3, SUBMIT_AICORE)              \
    X(4, DECIDE_SLOT_ADDRESS)        \
    X(4, DECIDE_INCAST_ADDRESS)      \
    X(4, RELEASE_FINISH_TASK)        \
    X(4, DEALLOCATE_TASK)            \
    X(4, STAGE_BUILD_TASK)           \
    X(5, ALLOCATE_TASK)              \
    X(5, BUILD_TASK_DATA)            \
    X(6, READY_QUEUE)                \
    X(7, READY_QUEUE_IN)             \
    X(6, RESOLVE_EARLY)              \
    X(6, CORE_FUNCDATA)              \
    X(5, SLAB_MEM_SUBMIT)            \
    X(4, DEALLOCATE_WORKSPACE)       \
    X(4, STAGE_PUSH_TASK)            \
    X(1, STAGE_TASK_SYNC)            \
    X(2, RELEASE_FINISH_TASK_INSYNC) \
    X(3, DEALLOCATE_TASK_INSYNC)     \
    X(1, STAGE_STOP_AICORE)          \
    X(1, DEVICE_MACHINE_INIT_DYN)    \
    X(1, DEVICE_MACHINE_SERVER_DYN)  \
    X5(1, STAGE_SCHEDULE)            \
    X5(2, RUN_TASK)                  \
    X5(3, POLLING_AICORES)           \
    X5(3, RESOLVE_DEPENDENCE)        \
    X5(3, SEND_AIC_TASK)             \
    X5(3, SEND_AIV_TASK)             \
    X5(3, WAIT_AICORE_FINISH)        \
    X5(3, DISPATCH_TASK)             \
    X5(2, SYNC_AICORE)               \
    X5(1, TASK)                      \
    X5(2, RECV_TASK)                 \
    X5(2, SEND_TASK)                 \
    X5(2, RESOLVE_DEP)               \
    X_L2(1, WSALLOC_CORE_A)          \
    X_L2(1, WSALLOC_CORE_D)          \
    X_L2(1, WSALLOC_CPU_A)           \
    X_L2(1, WSALLOC_CPU_D)           \
    X_L2(1, WSRTALLOC_CPU_A)         \
    X_L2(1, WSRTALLOC_CPU_D)         \
    X(0, MAX)

#define X5(a, b) \
    X(a, b)      \
    X(a, b##1)   \
    X(a, b##2)   \
    X(a, b##3)   \
    X(a, b##4)

enum PerfEventType {
#define X(ind, evt) PERF_EVT_##evt,
#define X_L2(idx, evt) X(idx, evt)
    PERF_EVENTS
#undef X_L2
#undef X
};

inline const char* PerfEventName[] = {
#define SPACE_0 ""
#define SPACE_1 "  " SPACE_0
#define SPACE_2 "  " SPACE_1
#define SPACE_3 "  " SPACE_2
#define SPACE_4 "  " SPACE_3
#define SPACE_5 "  " SPACE_4
#define SPACE_6 "  " SPACE_5
#define SPACE_7 "  " SPACE_6
#define X(ind, evt) SPACE_##ind #evt,
#define X_L2(idx, evt) X(idx, evt)
    PERF_EVENTS
#undef X_L2
#undef X
#undef SPACE_7
#undef SPACE_6
#undef SPACE_5
#undef SPACE_4
#undef SPACE_3
#undef SPACE_2
#undef SPACE_1
#undef SPACE_0
};

inline bool PerfEvtEnable[] = {
#define X(ind, evt) PERF_LEVEL >= 0,
#define X_L2(ind, evt) PERF_LEVEL >= 2,
    PERF_EVENTS
#undef X_L2
#undef X
};

// common of ptr
template <typename TI, typename TO>
inline TO* PtrToPtr(TI* const ptr)
{
    return reinterpret_cast<TO*>(ptr);
}

template <typename TI, typename TO>
inline const TO* PtrToPtr(const TI* const ptr)
{
    return reinterpret_cast<const TO*>(ptr);
}

inline uint64_t PtrToValue(const void* const ptr) { return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(ptr)); }

inline void* ValueToPtr(const uint64_t value) { return reinterpret_cast<void*>(static_cast<uintptr_t>(value)); }

inline std::vector<uint64_t> VPtrToValue(const std::vector<void*> v_ptr)
{
    std::vector<uint64_t> v_value;
    for (const auto& ptr : v_ptr) {
        v_value.emplace_back(PtrToValue(ptr));
    }
    return v_value;
}

inline uint64_t GetTimeMonotonic()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * NSEC_PER_SEC + ts.tv_nsec;
}

inline uint64_t GetCycles()
{
    uint64_t cycles;
#ifdef __aarch64__
    asm volatile("mrs %0, cntvct_el0" : "=r"(cycles));
#else
    cycles = GetTimeMonotonic();
#endif
    return cycles;
}

inline uint64_t GetFreq()
{
    uint64_t freq;
#ifdef __aarch64__
    asm volatile("mrs %0, cntfrq_el0" : "=r"(freq));
#else
    freq = NSEC_PER_SEC;
#endif
    return freq;
}

inline uint64_t CurrentTime()
{
    uint64_t mono = GetTimeMonotonic();
    return mono / NSEC_PER_USEC;
}

inline int CheckTimeOut(const uint64_t& tStart, uint64_t& tCnt, uint64_t& tCur, const std::string& opString)
{
    tCnt++;
    if (tCnt % 50 == 0) { // 50 is loop count to check whether timeout occurs
        tCur = CurrentTime();
        DEV_IF_VERBOSE_DEBUG
        {
            if (tCur - tStart > DFX_TIME_OUT_THRESHOLD) {
                DEV_ERROR(
                    SchedErr::TASK_WAIT_TIMEOUT,
                    "#sche.task.run.sync.timeout: %s dfx_timeout, aicpu force exit, ttl=%lu.", opString.c_str(), tCnt);
                return DEVICE_MACHINE_ERROR;
            }
        }
        else
        {
            if (tCur - tStart > TIME_OUT_THRESHOLD) {
                DEV_ERROR(
                    SchedErr::TASK_WAIT_TIMEOUT, "#sche.task.run.sync.timeout: %s timeout, aicpu force exit, ttl=%lu.",
                    opString.c_str(), tCnt);
                return DEVICE_MACHINE_ERROR;
            }
        }
    }
    return DEVICE_MACHINE_OK;
}

struct TimeCheck {
    uint64_t startTime = CurrentTime();
    uint64_t curTime = 0;
    uint64_t count = 0;
};

inline int CheckTimeOut(const std::string& operation, TimeCheck& timeCheck)
{
    return CheckTimeOut(timeCheck.startTime, timeCheck.count, timeCheck.curTime, operation);
}

// ========== 统一超时检测宏 ==========
// 核心设计：一个宏支持三种模式
// 1. MODE_EXIT: 超时后退出（返回错误码）
// 2. MODE_WARN: 超时后打印警告（不退出，周期性打印）
// 3. MODE_WARN_THEN_EXIT: 在 warn_threshold 时打印警告，在 timeout_threshold 时退出

enum TimeoutMode {
    MODE_EXIT,              // 超时后直接退出
    MODE_WARN,              // 周期性警告，不退出
    MODE_WARN_THEN_EXIT     // 先警告后退出
};

// 内部状态结构（用于跟踪警告打印状态）
struct TimeoutState {
    uint64_t startCycles;
    uint64_t freq;
    uint64_t lastWarnCycles;
    bool warnPrinted;
    
    TimeoutState() : startCycles(GetCycles()), freq(GetFreq()), 
                     lastWarnCycles(0), warnPrinted(false) {}
    
    // 纳秒转换为 cycles（考虑平台频率）
    inline uint64_t NsToCycles(uint64_t ns) const {
        return (ns * freq) / NSEC_PER_SEC;
    }
    
    // cycles 转换为纳秒
    inline uint64_t CyclesToNs(uint64_t cycles) const {
        return (cycles * NSEC_PER_SEC) / freq;
    }
    
    // 获取已消耗的纳秒
    inline uint64_t ElapsedNs() const {
        return CyclesToNs(GetCycles() - startCycles);
    }
    
    // 重置计时器
    inline void Reset() {
        startCycles = GetCycles();
        lastWarnCycles = 0;
        warnPrinted = false;
    }
};

// ========== 统一超时检测宏 ==========
// 用法示例：
//   TIMEOUT_CHECK(state, TIMEOUT_NS_1MIN, TIMEOUT_NS_1MIN/2, MODE_WARN_THEN_EXIT, err_code, warn_fmt, err_fmt, args...)
//   TIMEOUT_CHECK(state, TIMEOUT_NS_10MIN, 0, MODE_WARN, 0, warn_fmt, "", args...)
//   TIMEOUT_CHECK(state, TIMEOUT_NS_20MIN, 0, MODE_EXIT, err_code, "", err_fmt, args...)

#define TIMEOUT_CHECK(state, timeout_ns, warn_ns, mode, err_code, warn_fmt, err_fmt, ...) \
    do { \
        uint64_t elapsed_ns = state.ElapsedNs(); \
        uint64_t timeout_threshold = timeout_ns; \
        uint64_t warn_threshold = warn_ns; \
        \
        /* 模式 1: 直接退出 */ \
        if ((mode == MODE_EXIT) && (elapsed_ns > timeout_threshold)) { \
            DEV_ERROR(err_code, err_fmt, ##__VA_ARGS__); \
            return err_code; \
        } \
        \
        /* 模式 2: 周期性警告（不退出） */ \
        if ((mode == MODE_WARN) && (elapsed_ns > timeout_threshold)) { \
            /* 每个周期只打印一次 */ \
            if (!state.warnPrinted || elapsed_ns > state.lastWarnCycles + timeout_threshold) { \
                DEV_WARN(warn_fmt, ##__VA_ARGS__); \
                state.lastWarnCycles = elapsed_ns; \
                state.warnPrinted = true; \
            } \
        } \
        \
        /* 模式 3: 先警告后退出 */ \
        if (mode == MODE_WARN_THEN_EXIT) { \
            /* 达到警告阈值时打印 */ \
            if (elapsed_ns > warn_threshold && !state.warnPrinted) { \
                DEV_WARN(warn_fmt, ##__VA_ARGS__); \
                state.warnPrinted = true; \
            } \
            /* 达到超时阈值时退出 */ \
            if (elapsed_ns > timeout_threshold) { \
                DEV_ERROR(err_code, err_fmt, ##__VA_ARGS__); \
                return err_code; \
            } \
        } \
    } while (0)

// ========== 简化宏：超时退出 ==========
#define TIMEOUT_CHECK_EXIT(state, timeout_ns, err_code, fmt, ...) \
    TIMEOUT_CHECK(state, timeout_ns, 0, MODE_EXIT, err_code, "", fmt, ##__VA_ARGS__)

// ========== 简化宏：周期性警告（不退出，仅打印）==========
#define TIMEOUT_CHECK_WARN(state, timeout_ns, fmt, ...) \
    do { \
        uint64_t elapsed_ns = state.ElapsedNs(); \
        if (elapsed_ns > timeout_ns) { \
            if (!state.warnPrinted || elapsed_ns > state.lastWarnCycles + timeout_ns) { \
                DEV_WARN(fmt, ##__VA_ARGS__); \
                state.lastWarnCycles = elapsed_ns; \
                state.warnPrinted = true; \
            } \
        } \
    } while (0)

// ========== 简化宏：先警告后退出 ==========
#define TIMEOUT_CHECK_WARN_EXIT(state, timeout_ns, warn_ns, err_code, warn_fmt, err_fmt, ...) \
    TIMEOUT_CHECK(state, timeout_ns, warn_ns, MODE_WARN_THEN_EXIT, err_code, warn_fmt, err_fmt, ##__VA_ARGS__)

// ========== Legacy 宏（标记为 DEPRECATED，保持向后兼容）==========
// 注意：TIMEOUT_CHECK_AND_RESET 是伪超时，超时后重置继续循环，永不退出
// 建议使用新的 TIMEOUT_CHECK 系列宏替代
#define TIMEOUT_CHECK_START() uint64_t start = GetCycles()

#define TIMEOUT_CHECK_AND_RESET(timeout, ...)  \
    do {                                       \
        if (GetCycles() - start > (timeout)) { \
            DEV_ERROR(__VA_ARGS__);            \
            start = GetCycles();               \
        }                                      \
    } while (0)

} // namespace npu::tile_fwk::dynamic
#endif