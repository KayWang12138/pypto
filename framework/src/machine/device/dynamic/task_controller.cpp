#include "machine/device/dynamic/task_controller.h"


FinishTaskController gController[SCHED_AICPU_NUM];
std::atomic<int> gBCStartTCnt = 0;
std::atomic<int> gBCEndTCnt = SCHED_AICPU_NUM;
volatile std::atomic<bool> gDeviceTaskFinish = false;