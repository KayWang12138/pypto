#ifndef AICPU_PREF_H
#define AICPU_PREF_H
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

namespace npu::tile_fwk::dynamic {
constexpr uint32_t MAX_SCHEDULE_AICPU_NUM = 3;          // 真正负责调度aicore的aicpu个数
constexpr uint32_t MAX_USED_AICPU_NUM = MAX_SCHEDULE_AICPU_NUM + 2;
constexpr uint32_t CTRL_CPU_THREAD_IDX = MAX_SCHEDULE_AICPU_NUM;

#define PERF_TRACES                             \
    X(BEGIN)                                    \
    X(ALLOC_THREAD_ID)                          \
    X(INIT)                                     \
    X(CORE_HAND_SHAKE)                          \
    XDEVTASK(DEV_TASK_BUILD)                    \
    XDEVTASK(DEV_TASK_RCV)                      \
    XDEVTASK(DEV_TASK_SEND_FIRST_CALLOP_TASK)   \
    XDEVTASK(DEV_TASK_SCHED_EXEC)               \
    XDEVTASK(DEV_TASK_SYNC_CORE_STOP)           \
    XDEVTASK(DEV_TASK_RSP)                      \
    X(WAIT_ALL_DEV_TASK_FINISH)                 \
    X(WAIT_CORE_EXIT)                           \
    X(EXIT)                                     \
    X(MAX)                                      \

enum PerfTraceType {
#define X(trace) PERF_TRACE_##trace,
#define XDEVTASK(trace) PERF_TRACE_##trace,
    PERF_TRACES
#undef XDEVTASK
#undef X
};

inline bool PerfTraceIsDevTask[] = {
#define X(trace)  0,
#define XDEVTASK(trace) 1,
    PERF_TRACES
#undef XDEVTASK
#undef X
};

inline const char *PerfTraceName[] = {
#define X(trace) #trace,
#define XDEVTASK(trace) #trace,
    PERF_TRACES
#undef XDEVTASK
#undef X
};

#define DEVTASK_PERF_ARRY_INDEX(type) (type - PERF_TRACE_DEV_TASK_BUILD)
inline constexpr uint32_t DEVTASK_PERF_TYPE_NUM = (PERF_TRACE_DEV_TASK_RSP - PERF_TRACE_DEV_TASK_BUILD + 1);
inline constexpr uint32_t PERF_TRACE_COUNT_DEVTASK_MAX_NUM = 20;

#undef PERF_TRACES
}
#endif