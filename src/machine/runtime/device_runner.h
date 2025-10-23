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
 * \file device_runner.h
 * \brief
 */

#ifndef SRC_MACHINE_DEVICE_RUNNER_H
#define SRC_MACHINE_DEVICE_RUNNER_H

#include <cstdint>
#include <fcntl.h>
#include <vector>
#include <mutex>
#include <unistd.h>
#include <sys/file.h>

#ifdef ENABLE_BUILD_WITH_CANN
#include <runtime/rt.h>
#include "machine/utils/machine_ws_intf.h"
constexpr int PMU_EVENT_TYPE_MAX = 8;
constexpr int CORE_DEFAULT_NUM = 70;
namespace npu::tile_fwk {

struct FileLock {
    FileLock() : fd(-1){};

    bool Init(const char *path) {
        fd = open(path, O_RDWR | O_CREAT, S_IRWXU | S_IRUSR | S_IXUSR | S_IROTH | S_IXOTH);
        return fd >= 0;
    }

    void lock() const { flock(fd, LOCK_EX); }

    void unlock() const { flock(fd, LOCK_UN); }

    ~FileLock() {
        if (fd != -1) {
            close(fd);
        }
    }

    int fd;
};
class DeviceRunner {
public:
    static DeviceRunner &Get();

    int Run(rtStream_t aicpuStream, rtStream_t aicoreStream, int64_t taskId, uint64_t taskData, int taskType = DEVICE_TASK_TYPE_STATIC);
    int RunAsync(rtStream_t aicpuStream, rtStream_t aicoreStream, int64_t taskId, uint64_t taskData, int taskType = DEVICE_TASK_TYPE_STATIC);
    uint64_t GetTasksTime() const;
    int DynamicLaunch(rtStream_t aicpuStream, rtStream_t aicoreStream, int64_t taskId, AstKernelArgs *kernelArgs, int blockdim, int launchAicpuNum);
    int DynamicLaunchSynchronize(rtStream_t aicpuStream, rtStream_t aicoreStream);
    int DynamicRun(rtStream_t aicpuStream, rtStream_t aicoreStream, int64_t taskId, AstKernelArgs *kernelArgs, int blockdim = 25, int launchAicpuNum = 5);
    void InitDynamicArgs(DeviceArgs &args, int nrCore = CORE_DEFAULT_NUM);
    static int RegiserKernelBin(void **hdl);
    static void SetBinData(const std::vector<uint8_t> &binBuf);

private:
    DeviceRunner() = default;
    void *DevAlloc(int size);
    int InitDeviceArgs(DeviceArgs &args);
    int Init();

    int LaunchAiCpu(const rtStream_t aicpuStream, const uint64_t taskId, const uint64_t taskData, int taskType) const;
    int LaunchAiCore(rtStream_t aicoreStream, int taskType);
    void Dump();
    void AllocDfxMetricMemory();
    void SetPmuEventType(int32_t &profPmuType);
    void GetPmuEventType();
    /**************DynamicFunction**************/
    int launchDynamicAiCore(rtStream_t aicoreStream, AstKernelArgs *kernelArgs);
    int launchDynamicAiCpu(rtStream_t aicpuStream, AstKernelArgs *kArgs);
    int RunPrepare(rtStream_t aicpuStream, rtStream_t aicoreStream);
    int RunPost(rtStream_t aicpuStream, rtStream_t aicoreStream);
    int launchDynamicAiCpuInit(rtStream_t aicpuStream, AstKernelArgs *kArgs);
    void InitAiCpuSoBin();
private:
    int devId_;
    int aicpuNum_{5};
    int blockDim_{0};
    std::vector<int64_t> pmuEvtType_;
    DeviceArgs args_;
    DeviceArgs *devArgs_;
    std::vector<void *> perfData_;
    std::once_flag once_;
    rtBinHandle binHdl_;
    FileLock lock_;
};

#else
namespace npu::tile_fwk {
class DeviceRunner {
public:
    static DeviceRunner &Get() {
        static DeviceRunner runner;
        return runner;
    }
    int Run(void *stream, int64_t taskId, uint64_t taskData) {
        (void)stream;
        (void)taskId;
        (void)taskData;
        return 0;
    }
    int RunAsync(void *stream, int64_t taskId, uint64_t taskData) {
        (void)stream;
        (void)taskId;
        (void)taskData;
        return 0;
    }
};
#endif
} // namespace npu::tile_fwk
#endif // SRC_MACHINE_DEVICE_RUNNER_H
