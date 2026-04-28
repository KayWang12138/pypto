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
 * \file perf_event_sampler.h
 * \brief Linux perf_event_open based PMU sampler for AICPU performance analysis
 *        Uses PERF_TYPE_HARDWARE standard events for better compatibility
 */

#pragma once

#ifdef __DEVICE__
 
#include <unistd.h>
#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <cstdint>
#include <ctime>
#include <cerrno>
#include <sstream>
#include <iomanip>
#include <locale>

#include "machine/utils/device_switch.h"
#include "machine/utils/device_log.h"
#include "machine/device/dynamic/device_utils.h"

#define MAX_PERF_EVENT_NUM 8

namespace npu::tile_fwk {

static inline uint64_t MakeCacheEventConfig(uint64_t cacheId, uint64_t opId, uint64_t resultId)
{
    return cacheId | (opId << 8) | (resultId << 16);
}

struct GroupEvent {
    struct Event {
        int type_;
        uint64_t config_;
        int fd_{-1};
        const char* name;
        bool valid_{false};
    };

    int AddEvent(int type, uint64_t config, const char* name)
    {
        if (nrEvent == MAX_PERF_EVENT_NUM) {
            return -EINVAL;
        }

        struct perf_event_attr pe;
        memset(&pe, 0, sizeof(struct perf_event_attr));
        pe.type = type;
        pe.size = sizeof(struct perf_event_attr);
        pe.config = config;
        pe.disabled = 1;
        pe.exclude_kernel = 1;
        pe.exclude_hv = 1;
        if (groupFd == -1) {
            pe.read_format = PERF_FORMAT_GROUP;
        }

        int fd = syscall(__NR_perf_event_open, &pe, tid_, -1, groupFd, 0);
        if (fd < 0) {
            events[nrEvent++] = {type, config, -1, name, false};
            return -1;
        }
        if (groupFd == -1) {
            groupFd = fd;
        }
        events[nrEvent++] = {type, config, fd, name, true};
        validEventCount++;
        return 0;
    }

    bool Enable()
    {
        if (groupFd == -1) {
            return false;
        }
        if (ioctl(groupFd, PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP) < 0) {
            DEV_WARN("[AICPU_PMU] PERF_EVENT_IOC_RESET failed, errno=%d", errno);
            return false;
        }
        if (ioctl(groupFd, PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP) < 0) {
            DEV_WARN("[AICPU_PMU] PERF_EVENT_IOC_ENABLE failed, errno=%d", errno);
            return false;
        }
        return true;
    }

    void Disable()
    {
        if (groupFd != -1) {
            ioctl(groupFd, PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP);
        }
    }

    int Read(uint64_t* counts)
    {
        uint64_t buf[MAX_PERF_EVENT_NUM + 1];
        if (groupFd == -1 || validEventCount == 0) {
            return 0;
        }
        int len = read(groupFd, buf, sizeof(buf));
        if (len < 0) {
            DEV_WARN("[AICPU_PMU] read perf event group failed, errno=%d", errno);
            return 0;
        }
        size_t expectedLen = (validEventCount + 1) * sizeof(uint64_t);
        if ((size_t)len != expectedLen) {
            DEV_WARN("[AICPU_PMU] read perf event group length mismatch, actual=%d, expected=%lu",
                     len, expectedLen);
            return 0;
        }
        
        int validIdx = 0;
        for (int i = 0; i < nrEvent; i++) {
            if (events[i].valid_) {
                counts[i] = buf[validIdx + 1];
                validIdx++;
            } else {
                counts[i] = 0;
            }
        }
        return buf[0];
    }

    GroupEvent(int tid) : tid_(tid) {}

    ~GroupEvent()
    {
        for (int i = 0; i < nrEvent; i++) {
            if (events[i].fd_ >= 0) {
                close(events[i].fd_);
            }
        }
    }

    pid_t tid_;
    int nrEvent{0};
    int validEventCount{0};
    int groupFd{-1};
    Event events[MAX_PERF_EVENT_NUM];
};

// Event indices for standard PERF_TYPE_HARDWARE events
enum PerfEventIdx {
    IDX_CPU_CYCLES = 0,
    IDX_INSTRUCTIONS,
    IDX_BRANCH_INST,
    IDX_BRANCH_MISS,
    IDX_L1D_CACHE_REFS,
    IDX_L1D_CACHE_MISSES,
    IDX_LL_CACHE_REFS,
    IDX_LL_CACHE_MISSES,
    PERF_EVENT_COUNT
};

#if AICPU_PMU_EVENT_ENABLE
struct AicpuPerfEventSampler {
    AicpuPerfEventSampler() : events(gettid())
    {
        // 使用 PERF_TYPE_HARDWARE 标准事件（权限要求较低，兼容性更好）
        TryAddEvent(PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES, "cpu_cycles");
        TryAddEvent(PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS, "instructions");
        TryAddEvent(PERF_TYPE_HARDWARE, PERF_COUNT_HW_BRANCH_INSTRUCTIONS, "branch_inst");
        TryAddEvent(PERF_TYPE_HARDWARE, PERF_COUNT_HW_BRANCH_MISSES, "branch_miss");
        TryAddCacheEvent(PERF_COUNT_HW_CACHE_L1D, PERF_COUNT_HW_CACHE_OP_READ,
                         PERF_COUNT_HW_CACHE_RESULT_ACCESS, "l1d_cache_refs");
        TryAddCacheEvent(PERF_COUNT_HW_CACHE_L1D, PERF_COUNT_HW_CACHE_OP_READ,
                         PERF_COUNT_HW_CACHE_RESULT_MISS, "l1d_cache_misses");
        TryAddCacheEvent(PERF_COUNT_HW_CACHE_LL, PERF_COUNT_HW_CACHE_OP_READ,
                         PERF_COUNT_HW_CACHE_RESULT_ACCESS, "ll_cache_refs");
        TryAddCacheEvent(PERF_COUNT_HW_CACHE_LL, PERF_COUNT_HW_CACHE_OP_READ,
                         PERF_COUNT_HW_CACHE_RESULT_MISS, "ll_cache_misses");
        
        if (events.validEventCount > 0) {
            DEV_INFO("[AICPU_PMU] Registered %d/%d PMU events successfully", 
                     events.validEventCount, events.nrEvent);
        } else {
            DEV_WARN("[AICPU_PMU] PMU events unavailable (errno=%d). "
                     "Possible causes: container restrictions or missing capabilities. "
                     "Fallback to time-only mode.", errno);
            pmuAvailable = false;
        }
    }

    void Begin()
    {
        pmuEnabled = events.Enable();
        cycles = dynamic::GetCycles();
    }

    void End()
    {
        cycles = dynamic::GetCycles() - cycles;
        events.Disable();
    }

    void Dump()
    {
        double timeUs = dynamic::Cycles2Us(cycles);
        double timeMs = timeUs / 1000.0;
        
        if (!pmuAvailable || !pmuEnabled || events.validEventCount == 0) {
            DEV_ERROR(ERROR_CODE_UNDEFINED, "[AICPU_PMU] ExecDyn Summary (PMU unavailable)");
            DEV_ERROR(ERROR_CODE_UNDEFINED, "  Total Running Time: %.2f us (%.2f ms)", timeUs, timeMs);
            DEV_ERROR(ERROR_CODE_UNDEFINED, "  Note: PMU events disabled due to permission restrictions");
            return;
        }
        
        uint64_t counts[MAX_PERF_EVENT_NUM] = {0};
        int n = events.Read(counts);
        if (n == 0) {
            DEV_ERROR(ERROR_CODE_UNDEFINED, "[AICPU_PMU] ExecDyn Summary");
            DEV_ERROR(ERROR_CODE_UNDEFINED, "  Total Running Time: %.2f us (%.2f ms)", timeUs, timeMs);
            return;
        }
        
        DEV_ERROR(ERROR_CODE_UNDEFINED, "[AICPU_PMU] Performance Report");
        DEV_ERROR(ERROR_CODE_UNDEFINED, "============================================================");
        DEV_ERROR(ERROR_CODE_UNDEFINED, "Total Running Time: %.2f us (%.2f ms)", timeUs, timeMs);
        
        // Raw PMU event counters
        DEV_ERROR(ERROR_CODE_UNDEFINED, "------------------------------------------------------------");
        DEV_ERROR(ERROR_CODE_UNDEFINED, "PMU Event Counters");
        DEV_ERROR(ERROR_CODE_UNDEFINED, "------------------------------------------------------------");
        DEV_ERROR(ERROR_CODE_UNDEFINED, "  CPU Cycles:         %s", FormatNumber(counts[IDX_CPU_CYCLES]).c_str());
        DEV_ERROR(ERROR_CODE_UNDEFINED, "  Instructions:       %s", FormatNumber(counts[IDX_INSTRUCTIONS]).c_str());
        DEV_ERROR(ERROR_CODE_UNDEFINED, "  Branch Instructions:%s", FormatNumber(counts[IDX_BRANCH_INST]).c_str());
        DEV_ERROR(ERROR_CODE_UNDEFINED, "  Branch Misses:      %s", FormatNumber(counts[IDX_BRANCH_MISS]).c_str());
        DEV_ERROR(ERROR_CODE_UNDEFINED, "  L1D Cache Refs:     %s", FormatNumber(counts[IDX_L1D_CACHE_REFS]).c_str());
        DEV_ERROR(ERROR_CODE_UNDEFINED, "  L1D Cache Misses:   %s", FormatNumber(counts[IDX_L1D_CACHE_MISSES]).c_str());
        DEV_ERROR(ERROR_CODE_UNDEFINED, "  LL Cache Refs:      %s", FormatNumber(counts[IDX_LL_CACHE_REFS]).c_str());
        DEV_ERROR(ERROR_CODE_UNDEFINED, "  LL Cache Misses:    %s", FormatNumber(counts[IDX_LL_CACHE_MISSES]).c_str());

        // Derived metrics computed from the raw counters above.
        DEV_ERROR(ERROR_CODE_UNDEFINED, "------------------------------------------------------------");
        DEV_ERROR(ERROR_CODE_UNDEFINED, "Derived Metrics");
        DEV_ERROR(ERROR_CODE_UNDEFINED, "------------------------------------------------------------");
        double ipc = counts[IDX_CPU_CYCLES] > 0 ? 
            (double)counts[IDX_INSTRUCTIONS] / counts[IDX_CPU_CYCLES] : 0.0;
        double cpi = counts[IDX_INSTRUCTIONS] > 0 ?
            (double)counts[IDX_CPU_CYCLES] / counts[IDX_INSTRUCTIONS] : 0.0;
        DEV_ERROR(ERROR_CODE_UNDEFINED, "  IPC:                %.2f", ipc);
        DEV_ERROR(ERROR_CODE_UNDEFINED, "  CPI:                %.2f", cpi);

        double branchMissRate = counts[IDX_BRANCH_INST] > 0 ?
            (double)counts[IDX_BRANCH_MISS] / counts[IDX_BRANCH_INST] * 100.0 : 0.0;
        DEV_ERROR(ERROR_CODE_UNDEFINED, "  Branch Miss Rate:   %.2f%%", branchMissRate);
        
        DumpCacheDerivedMetric("L1D Cache", counts[IDX_L1D_CACHE_REFS], counts[IDX_L1D_CACHE_MISSES]);
        DumpCacheDerivedMetric("LL Cache", counts[IDX_LL_CACHE_REFS], counts[IDX_LL_CACHE_MISSES]);
        
        DEV_ERROR(ERROR_CODE_UNDEFINED, "============================================================");
    }

private:
    void TryAddEvent(int type, uint64_t config, const char* name)
    {
        int ret = events.AddEvent(type, config, name);
        if (ret < 0) {
            DEV_DEBUG("[AICPU_PMU] Failed to register: %s (type=%d, config=%lu, errno=%d)",
                      name, type, config, errno);
        }
    }

    void TryAddCacheEvent(uint64_t cacheId, uint64_t opId, uint64_t resultId, const char* name)
    {
        TryAddEvent(PERF_TYPE_HW_CACHE, MakeCacheEventConfig(cacheId, opId, resultId), name);
    }

    void DumpCacheDerivedMetric(const char* name, uint64_t refs, uint64_t misses)
    {
        if (refs == 0) {
            DEV_ERROR(ERROR_CODE_UNDEFINED, "  %s Hit Rate: N/A", name);
            DEV_ERROR(ERROR_CODE_UNDEFINED, "  %s Miss Rate:N/A", name);
            return;
        }
        double missRate = refs > 0 ? (double)misses / refs * 100.0 : 0.0;
        DEV_ERROR(ERROR_CODE_UNDEFINED, "  %s Hit Rate: %.2f%%", name, 100.0 - missRate);
        DEV_ERROR(ERROR_CODE_UNDEFINED, "  %s Miss Rate:%.2f%%", name, missRate);
    }
    
    std::string FormatNumber(uint64_t n)
    {
        std::stringstream ss;
        ss.imbue(std::locale(""));
        ss << n;
        return ss.str();
    }

    uint64_t cycles{0};
    GroupEvent events;
    bool pmuAvailable{true};
    bool pmuEnabled{false};
};

static inline AicpuPerfEventSampler& GetAicpuPerfEventSampler()
{
    static thread_local AicpuPerfEventSampler sampler;
    return sampler;
}

struct AicpuPerfScopedSampler {
    explicit AicpuPerfScopedSampler(const char* sectionName)
        : sectionName_(sectionName), sampler_(GetAicpuPerfEventSampler())
    {
        sampler_.Begin();
    }

    ~AicpuPerfScopedSampler()
    {
        sampler_.End();
        DEV_ERROR(ERROR_CODE_UNDEFINED, "[AICPU_PMU] %s", sectionName_);
        sampler_.Dump();
    }

private:
    const char* sectionName_{"unnamed"};
    AicpuPerfEventSampler& sampler_;
};

#define AICPU_PMU_SCOPE(section_name_literal) \
    ::npu::tile_fwk::AicpuPerfScopedSampler aicpuPerfScopedSampler_##__LINE__(section_name_literal)

#define AICPU_PMU_BEGIN(sampler_name) \
    auto& sampler_name = ::npu::tile_fwk::GetAicpuPerfEventSampler(); \
    (sampler_name).Begin()

#define AICPU_PMU_END(sampler_name, section_name_literal) \
    do { \
        (sampler_name).End(); \
        DEV_ERROR(ERROR_CODE_UNDEFINED, "[AICPU_PMU] %s", section_name_literal); \
        (sampler_name).Dump(); \
    } while (0)

// 外部对象式采样（跨函数场景）
#define AICPU_PMU_BEGIN_EXTERNAL(sampler_ptr) \
    do { (sampler_ptr)->Begin(); } while (0)

#define AICPU_PMU_END_EXTERNAL(sampler_ptr, section_name_literal) \
    do { \
        (sampler_ptr)->End(); \
        DEV_ERROR(ERROR_CODE_UNDEFINED, "[AICPU_PMU] %s", section_name_literal); \
        (sampler_ptr)->Dump(); \
    } while (0)
#else
struct AicpuPerfEventSampler {
    void Begin() {}
    void End() {}
    void Dump() {}
};

static inline AicpuPerfEventSampler& GetAicpuPerfEventSampler()
{
    static thread_local AicpuPerfEventSampler sampler;
    return sampler;
}

struct AicpuPerfScopedSampler {
    explicit AicpuPerfScopedSampler([[maybe_unused]] const char* sectionName)
        : sampler_(GetAicpuPerfEventSampler())
    {}

private:
    AicpuPerfEventSampler& sampler_;
};

#define AICPU_PMU_SCOPE(section_name_literal)
#define AICPU_PMU_BEGIN(sampler_name)
#define AICPU_PMU_END(sampler_name, section_name_literal)
#define AICPU_PMU_BEGIN_EXTERNAL(sampler_ptr)
#define AICPU_PMU_END_EXTERNAL(sampler_ptr, section_name_literal)
#endif

} // namespace npu::tile_fwk

#endif // __DEVICE__
