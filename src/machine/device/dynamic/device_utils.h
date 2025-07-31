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
 * \file device_utils.h
 * \brief
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
#include <fstream>
#include "machine/utils/device_log.h"

#ifndef CONFIG_MAX_DEVICE_TASK_NUM
#define CONFIG_MAX_DEVICE_TASK_NUM 64
#endif

#define MAX_DEVICE_TASK_NUM CONFIG_MAX_DEVICE_TASK_NUM

#define likely(x)   __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)

#define BITS_PER_BYTES  8
#define BITS_PER_INT    32
#define BITS_PER_LONG   64

#ifndef UNUSED
#define UNUSED(n)       (void)(n)
#endif

namespace npu::tile_fwk::dynamic {
inline constexpr bool IsDeviceMode() {
#ifdef __DEVICE__
    return true;
#else
    return false;
#endif // __DEVICE__
}

constexpr int32_t DEVICE_MACHINE_ERROR = -1;
constexpr int32_t DEVICE_MACHINE_OK = 0;
constexpr int32_t DEVICE_MACHINE_FINISHED = 1;
constexpr int32_t TIME_OUT_THRESHOLD = 1000000; // 超时阈值 1s
constexpr int32_t DFX_TIME_OUT_THRESHOLD = 50000000; // 超时阈值 50s
constexpr uint32_t MAX_SCHEDULE_AICPU_NUM = 3;          // 真正负责调度aicore的aicpu个数
constexpr int32_t START_AICPU_NUM = 3; 
constexpr uint64_t NUM_FIFTY = 50;
constexpr uint64_t US_PER_SEC = 1000000;
constexpr uint64_t NSEC_PER_USEC = 1000;
constexpr uint64_t NSEC_PER_SEC = 10000000000;
constexpr int32_t MAX_MNG_AICORE_AVG_NUM = 8;
constexpr uint32_t NEED_LAUNCH_AICPU_MINNUM = 3;

#ifdef __aarch64__
constexpr uint64_t TIMEOUT_CYCLES = 500 * 1000 * 1000;
#else
constexpr uint64_t TIMEOUT_CYCLES = NSEC_PER_SEC;
#endif

constexpr uint64_t PROF_DUMP_TIMEOUT_CYCLES = 20 * TIMEOUT_CYCLES;

#define PERF_LEVEL 0
#define PERF_AICORE_THREAD_START 100

#define PERF_EVENTS                             \
    X(0, EXEC_DYN)                              \
    X(1, INIT)                                  \
    X(2, CONTROL_FLOW_MAPEXE)                   \
    X(3, CONTROL_FLOW_MAPEXE_MEMCPY)            \
    X(1, CONTROL_FLOW_INIT)                     \
    X(1, CONTROL_FLOW)                          \
    X(2, ROOT_FUNC)                             \
    X(3, STAGE_DUP_ROOT)                        \
    X(3, ALLOCATE_WORKSPACE)                    \
    X(3, STAGE_STITCH)                          \
    X(4, FAST_STITCH)                           \
    X(4, UPDATE_SLOT)                           \
    X(3, SUBMIT_AICORE)                         \
    X(4, STAGE_BUILD_TASK)                      \
    X(5, DECIDE_SLOT_ADDRESS)                   \
    X(5, DECIDE_INCAST_ADDRESS)                 \
    X(5, RELEASE_FINISH_TASK)                   \
    X(6, DEALLOCATE_TASK)                       \
    X(5, DEALLOCATE_WORKSPACE)                  \
    X(5, ALLOCATE_TASK)                         \
    X(5, READY_QUEUE)                           \
    X(5, RESOLVE_EARLY)                         \
    X(5, CORE_FUNCDATA)                         \
    X(4, STAGE_PUSH_TASK)                       \
    X(1, STAGE_TASK_SYNC)                       \
    X(2, RELEASE_FINISH_TASK_INSYNC)            \
    X(3, DEALLOCATE_TASK_INSYNC)                \
    X(1, STAGE_STOP_AICORE)                     \
    X(1, DEVICE_MACHINE_INIT_DYN)               \
    X(1, DEVICE_MACHINE_SERVER_DYN)             \
    X3(1, STAGE_SCHEDULE)                       \
    X3(2, RUN_TASK)                             \
    X3(3, POLLING_AICORES)                      \
    X3(3, RESOLVE_DEPENDENCE)                   \
    X3(3, SEND_AIC_TASK)                        \
    X3(3, SEND_AIV_TASK)                        \
    X3(3, WAIT_AICORE_FINISH)                   \
    X3(3, DISPATCH_TASK)                             \
    X3(2, SYNC_AICORE)                          \
    X3(1, TASK)                                 \
    X3(2, RECV_TASK)                            \
    X3(2, SEND_TASK)                            \
    X3(2, RESOLVE_DEP)                          \
    X_L2(1, WSALLOC_CORE_A)                     \
    X_L2(1, WSALLOC_CORE_D)                     \
    X_L2(1, WSALLOC_CPU_A)                      \
    X_L2(1, WSALLOC_CPU_D)                      \
    X_L2(1, WSRTALLOC_CPU_A)                    \
    X_L2(1, WSRTALLOC_CPU_D)                    \
    X(0, MAX)

#define X3(a, b) \
    X(a, b)      \
    X(a, b##1)   \
    X(a, b##2)

enum PerfEventType {
#define X(ind, evt) PERF_EVT_##evt,
#define X_L2(idx, evt) X(idx, evt)
    PERF_EVENTS
#undef X_L2
#undef X
};

inline const char *PerfEventName[] = {
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
#define X(ind, evt)  PERF_LEVEL >= 0,
#define X_L2(ind, evt) PERF_LEVEL >= 2,
    PERF_EVENTS
#undef X_L2
#undef X
};

// common of ptr
template<typename TI, typename TO>
inline TO *PtrToPtr(TI *const ptr) {
  return reinterpret_cast<TO *>(ptr);
}

template<typename TI, typename TO>
inline const TO *PtrToPtr(const TI *const ptr) {
  return reinterpret_cast<const TO *>(ptr);
}

inline uint64_t PtrToValue(const void *const ptr) {
  return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(ptr));
}

inline void *ValueToPtr(const uint64_t value) {
  return reinterpret_cast<void *>(static_cast<uintptr_t>(value));
}

inline std::vector<uint64_t> VPtrToValue(const std::vector<void *> v_ptr) {
  std::vector<uint64_t> v_value;
  for (const auto &ptr : v_ptr) {
    v_value.emplace_back(PtrToValue(ptr));
  }
  return v_value;
}

inline uint32_t CalcSchAicpuNumByBlockDim(uint32_t blockDim) {
    if (blockDim > (MAX_SCHEDULE_AICPU_NUM - 1) * MAX_MNG_AICORE_AVG_NUM) {
        return MAX_SCHEDULE_AICPU_NUM;
    }

    if (blockDim % MAX_MNG_AICORE_AVG_NUM == 0) {
        return blockDim / MAX_MNG_AICORE_AVG_NUM;
    }

    return blockDim / MAX_MNG_AICORE_AVG_NUM + 1;
}

inline uint64_t GetTimeMonotonic() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * NSEC_PER_SEC + ts.tv_nsec;
}

inline uint64_t GetCycles() {
    uint64_t cycles;
#ifdef __aarch64__
    asm volatile("mrs %0, cntvct_el0" : "=r"(cycles));
#else
    cycles = GetTimeMonotonic();
#endif
    return cycles;
}

inline uint64_t GetFreq() {
    uint64_t freq;
#ifdef __aarch64__
    asm volatile("mrs %0, cntfrq_el0" : "=r"(freq));
#else
    freq = NSEC_PER_SEC;
#endif
    return freq;
}

inline uint64_t CurrentTime() {
    uint64_t mono = GetTimeMonotonic();
    return mono / NSEC_PER_USEC;
}

inline int CheckTimeOut(const uint64_t &tStart, uint64_t &tCnt, uint64_t &tCur, const std::string &opString) {
    tCnt++;
    if (tCnt % 50 == 0) { // 50 is loop count to check whether timeout occurs
        tCur = CurrentTime();
        #if DEBUG_SWITCH
            if (tCur - tStart > DFX_TIME_OUT_THRESHOLD) {
                DEV_ERROR("%s dfx_timeout, aicpu force exit, ttl=%lu.", opString.c_str(), tCnt);
                return DEVICE_MACHINE_ERROR;
            }
        #else
            if (tCur - tStart > TIME_OUT_THRESHOLD) {
                DEV_ERROR("%s timeout, aicpu force exit, ttl=%lu.", opString.c_str(), tCnt);
                return DEVICE_MACHINE_ERROR;
            }
        #endif
    }
    return DEVICE_MACHINE_OK;
}

struct TimeCheck {
    uint64_t startTime = CurrentTime();
    uint64_t curTime = 0;
    uint64_t count = 0;
};

inline int CheckTimeOut(const std::string &operation, TimeCheck &timeCheck) {
    return CheckTimeOut(timeCheck.startTime, timeCheck.count, timeCheck.curTime, operation);
}
inline void RepeatPuts(char c, size_t count) {
    char buf[80];
    for (size_t i = 0; i < count; i++) {
        buf[i] = c;
    }
    buf[count] = '\0';
    DEV_ERROR("%s.", buf);
}

struct PerfettoMgr {
    static const int MAX_THEAD_NUM = 200;
    static const int TRUNK_SIZE = 32768;
    static const int MAX_EVT_DEPTH = 64;

    struct Record {
        int type;
        int tid;
        uint64_t start;
        uint64_t end;
        uint32_t index;
        uint32_t pIndex;
        std::string name = "-";
    };

    template <typename T, int N>
    struct Array {
        size_t used{0};
        T data[N];

        inline void Push(const T &t) { data[used++] = t; }
        inline void Pop() { used--; }
        inline bool Full() { return used == N; }
        inline bool Empty() { return used == 0; }
        inline T &Top() { return data[used - 1]; }
        inline T *Alloc() { return &data[used++]; }
    };

    using Trunk = Array<Record, TRUNK_SIZE>;
    using EvtStack = Array<Record *, MAX_EVT_DEPTH>;

    Record *allocRecord(int tid) {
        if (trunk_[tid] == nullptr || trunk_[tid]->Full()) {
            trunk_[tid] = new Trunk;
            mutex_.lock();
            trunks_.push_back(trunk_[tid]);
            mutex_.unlock();
        }
        return trunk_[tid]->Alloc();
    }

    void PerfBegin(int type, int tid) {
#if defined(CONFIG_PERFETTO) && CONFIG_PERFETTO
        auto r = allocRecord(tid);
        auto &stack = evtStack[tid];
        r->type = type;
        r->tid = tid;
        r->start = GetCycles();
        r->index = evtIndex++;
        r->pIndex = stack.Empty() ? -1 : stack.Top()->index;
        stack.Push(r);
#endif
        (void)type;
        (void)tid;
    }

    void PerfEnd(int type, int tid) {
#if defined(CONFIG_PERFETTO) && CONFIG_PERFETTO
        auto &stack = evtStack[tid];
        auto r = stack.Top();
        r->end = GetCycles();
        stack.Pop();
#endif
        (void)type;
        (void)tid;
    }

    void PerfEvent(int type, int tid, uint64_t start, uint64_t end, std::string name) {
#if defined(CONFIG_PERFETTO) && CONFIG_PERFETTO
        auto r = allocRecord(tid);
        auto &stack = evtStack[tid];
        r->type = type;
        r->tid = tid;
        r->name = name;
        r->start = start;
        r->end = end;
        r->index = evtIndex++;
        r->pIndex = stack.Empty() ? -1 : stack.Top()->index;
#endif
        (void)type;
        (void)tid;
        (void)start;
        (void)end;
        (void)name;
    }

    static PerfettoMgr &Instance() {
        static PerfettoMgr recorder;
        return recorder;
    }

    void Dump(const std::string &file) {
        std::ofstream os(file);
        for (auto &trunk : trunks_) {
            for (size_t i = 0; i < trunk->used; i++) {
                auto &r = trunk->data[i];
                os << PerfEventName[r.type] << " ";
                os << r.name << " ";
                os << r.index << " ";
                os << r.start << " ";
                os << r.end << " ";
                os << r.tid << ";";
                os << r.pIndex << " ";
                os << std::endl;
            }
        }
    }

private:
    PerfettoMgr() = default;

private:
    EvtStack evtStack[MAX_THEAD_NUM];
    Trunk *trunk_[MAX_THEAD_NUM] = {nullptr};

    std::mutex mutex_;
    std::vector<Trunk *> trunks_;
    std::atomic<int32_t> evtIndex;
};
struct PerfEvtMgr {
    struct Counter {
        int64_t start;
        int64_t total;
        int64_t count;
    };

    void PerfBegin(int type) {
        counters[type].start = GetCycles();
    }

    void PerfEnd(int type) {
        auto &c = counters[type];
        c.count++;
        c.total += GetCycles() - c.start;
    }

    static PerfEvtMgr &Instance() {
        static PerfEvtMgr recorder;
        return recorder;
    }

    void Dump() {
        uint64_t freq = GetFreq();
        static constexpr size_t SHEET_WIDTH = 40 + 3 + 10 + 3 + 10 + 3 + 10;

        RepeatPuts('=', SHEET_WIDTH);
        DEV_ERROR("%40s | %10s | %10s | %10s.", "EventType", "Count", "Total(us)", "Avg(us)");
        RepeatPuts('-', SHEET_WIDTH);

        for (int i = 0; i < PERF_EVT_MAX; i++) {
            auto evt = counters[i];
            if (evt.count != 0) {
                uint64_t total = evt.total * NSEC_PER_SEC / freq / NSEC_PER_USEC;
                float avg = static_cast<float>(total / evt.count);
                DEV_ERROR("%-40s | %10ld | %10lu | %10.1f.", PerfEventName[i], evt.count, total, avg);
            }
        }

        RepeatPuts('=', SHEET_WIDTH);
    }

private:
    PerfEvtMgr() { memset_s(counters, sizeof(counters), 0, sizeof(counters)); };

private:
    Counter counters[PERF_EVT_MAX];
};

inline void PerfBegin(int type) {
    if (PerfEvtEnable[type]) {
        PerfEvtMgr::Instance().PerfBegin(type);
        PerfettoMgr::Instance().PerfBegin(type, MAX_SCHEDULE_AICPU_NUM);
    }
}

inline void PerfEnd(int type) {
    if (PerfEvtEnable[type]) {
        PerfEvtMgr::Instance().PerfEnd(type);
        PerfettoMgr::Instance().PerfEnd(type, MAX_SCHEDULE_AICPU_NUM);
    }
}

inline void PerfMtBegin(int type, int tid) {
    if (PerfEvtEnable[type]) {
        PerfEvtMgr::Instance().PerfBegin(type + tid);
        PerfettoMgr::Instance().PerfBegin(type, tid);
    }
}

inline void PerfMtEnd(int type, int tid) {
    if (PerfEvtEnable[type]) {
        PerfEvtMgr::Instance().PerfEnd(type + tid);
        PerfettoMgr::Instance().PerfEnd(type, tid);
    }
}

inline void PerfMtEvent(int type, int tid, uint64_t start, uint64_t end, std::string name = "-") {
    if (PerfEvtEnable[type]) {
        PerfettoMgr::Instance().PerfEvent(type, tid, start, end, name);
    }
}

struct AutoScopedPerf {
    explicit AutoScopedPerf(int type) : type_(type) { PerfBegin(type); }
    ~AutoScopedPerf() { PerfEnd(type_); }
    int type_;
};
}

#endif
