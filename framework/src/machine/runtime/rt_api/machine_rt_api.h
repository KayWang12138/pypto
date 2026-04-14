/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file machine_rt_api.h
 * \brief runtime rt api dispatcher and implementations
 */

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include "interface/function/function.h"

namespace npu::tile_fwk::dynamic {

class IMachineRtApi {
public:
    virtual ~IMachineRtApi() = default;

    virtual int Malloc(void** ptr, size_t size, bool useHugePage) = 0;
    virtual int Free(void* ptr, bool useHugePage) = 0;
    virtual int Memcpy(void* dst, size_t dstSize, const void* src, size_t size, int kind) = 0;
    virtual int Memset(void* dst, size_t dstSize, int value, size_t size) = 0;

    virtual int LaunchAicpu(void* rtArgs, bool tripleStream, bool debugEnable, Function* function) = 0;
    virtual int LaunchAicore(
        void* aicoreStream, void* kernel, void* rtArgs, void* rtTaskCfg, bool debugEnable, uint32_t blockDim,
        uint64_t tilingKey) = 0;
    virtual int StreamSync(void* aicpuStream, void* ctrlStream, void* aicoreStream) = 0;

    virtual int RealRtCallCount() const { return 0; }
};

class RealRtApi final : public IMachineRtApi {
public:
    int Malloc(void** ptr, size_t size, bool useHugePage) override;
    int Free(void* ptr, bool useHugePage) override;
    int Memcpy(void* dst, size_t dstSize, const void* src, size_t size, int kind) override;
    int Memset(void* dst, size_t dstSize, int value, size_t size) override;
    int LaunchAicpu(void* rtArgs, bool tripleStream, bool debugEnable, Function* function) override;
    int LaunchAicore(
        void* aicoreStream, void* kernel, void* rtArgs, void* rtTaskCfg, bool debugEnable, uint32_t blockDim,
        uint64_t tilingKey) override;
    int StreamSync(void* aicpuStream, void* ctrlStream, void* aicoreStream) override;
};

class HostRtApi final : public IMachineRtApi {
public:
    using AicpuLaunchFn = std::function<int(void*, bool, bool, Function*)>;
    using AicoreLaunchFn = std::function<int(void*, void*, void*, void*, bool, uint32_t, uint64_t)>;
    using StreamSyncFn = std::function<int(void*, void*, void*)>;

    HostRtApi(
        AicpuLaunchFn aicpuLaunch, AicoreLaunchFn aicoreLaunch, StreamSyncFn streamSync,
        bool enforceNoRealRt = true);

    int Malloc(void** ptr, size_t size, bool useHugePage) override;
    int Free(void* ptr, bool useHugePage) override;
    int Memcpy(void* dst, size_t dstSize, const void* src, size_t size, int kind) override;
    int Memset(void* dst, size_t dstSize, int value, size_t size) override;
    int LaunchAicpu(void* rtArgs, bool tripleStream, bool debugEnable, Function* function) override;
    int LaunchAicore(
        void* aicoreStream, void* kernel, void* rtArgs, void* rtTaskCfg, bool debugEnable, uint32_t blockDim,
        uint64_t tilingKey) override;
    int StreamSync(void* aicpuStream, void* ctrlStream, void* aicoreStream) override;

    int RealRtCallCount() const override { return realRtCallCount_.load(std::memory_order_relaxed); }

private:
    int RecordFallbackFailure() const;

private:
    AicpuLaunchFn aicpuLaunch_;
    AicoreLaunchFn aicoreLaunch_;
    StreamSyncFn streamSync_;
    bool enforceNoRealRt_{true};
    mutable std::atomic<int> realRtCallCount_{0};
};

class RtApiDispatcher {
public:
    class ScopedInstall {
    public:
        explicit ScopedInstall(IMachineRtApi* api);
        ~ScopedInstall();

        ScopedInstall(const ScopedInstall&) = delete;
        ScopedInstall& operator=(const ScopedInstall&) = delete;

    private:
        IMachineRtApi* previous_{nullptr};
    };

    static IMachineRtApi& Current();
    static IMachineRtApi* Install(IMachineRtApi* api);
    static RealRtApi& Real();

private:
    static std::atomic<IMachineRtApi*> currentApi_;
};

} // namespace npu::tile_fwk::dynamic
