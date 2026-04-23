
#include <atomic> 

#define SCHED_AICPU_NUM 3
#define CORE_TASK_TYPE 5

struct BCTasks {
    uint64_t buf[512];
    int cnt;
};

struct FinishTaskController {
    struct BCTasks bcTasks[CORE_TASK_TYPE];
};

extern FinishTaskController gController[];
extern std::atomic<int> gBCStartTCnt;
extern std::atomic<int> gBCEndTCnt;
extern volatile __attribute__((aligned(64))) bool gDeviceTaskFinish;

inline void WaitOnBCStartFlag() {
    while (true) {
        if (gBCStartTCnt % SCHED_AICPU_NUM == 0 || gDeviceTaskFinish) {
            break;
        }
    }
}

inline void WaitOnBCEndFlag() {
    while (true) {
        if (gBCEndTCnt % SCHED_AICPU_NUM == 0 || gDeviceTaskFinish) {
            break;
        }
    }
}


