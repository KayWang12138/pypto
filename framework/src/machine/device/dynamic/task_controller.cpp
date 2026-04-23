#include "machine/device/dynamic/task_controller.h"


FinishTaskController gController[SCHED_AICPU_NUM];
std::atomic<int> gBCStartTCnt = 0;
std::atomic<int> gBCEndTCnt = SCHED_AICPU_NUM;
volatile __attribute__((aligned(64))) bool gDeviceTaskFinish = false;