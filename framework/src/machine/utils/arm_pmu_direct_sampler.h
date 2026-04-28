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
 * \file arm_pmu_direct_sampler.h
 * \brief Direct ARMv8 PMUv3 register access sampler (MRS/MSR instructions).
 *
 *  本文件提供独立的 ARM PMU 直读采样能力，通过 MRS/MSR 指令直接访问 PMU 寄存器，
 *  完全绕过 Linux perf_event_open / ioctl / read 系统调用路径。
 *
 *  前置条件：
 *    - 目标架构为 aarch64
 *    - 内核已开启 PMUSERENR_EL0.EN，允许 EL0 访问 PMU 寄存器
 *      * Linux 5.17+：`echo 1 > /proc/sys/kernel/perf_user_access`
 *      * 否则：加载自定义内核模块，在每个 CPU 上设置 PMUSERENR_EL0 = 0xF
 *
 *  典型开销：Begin/End 各 ~10-20ns，对比 ioctl/read 路径的数百纳秒显著优化。
 *
 *  与 perf_event_sampler.h 完全解耦：
 *    - 独立的编译开关（ARM_PMU_DIRECT_ENABLE）
 *    - 独立的宏接口（ARM_PMU_DIRECT_SCOPE / BEGIN / END）
 *    - 独立的类 ArmPmuDirectSampler
 */

#pragma once


#include <cstdint>
#include <sstream>
#include <iomanip>
#include <string>
#include <locale>

#include "machine/utils/device_switch.h"
#include "machine/utils/device_log.h"
#include "machine/device/dynamic/device_utils.h"

// ============================================================
// 可用性判断：需同时满足 开关启用 + aarch64
// ============================================================
#if defined(ARM_PMU_DIRECT_ENABLE) && ARM_PMU_DIRECT_ENABLE && defined(__aarch64__)
#define ARM_PMU_DIRECT_AVAILABLE 1
#else
#define ARM_PMU_DIRECT_AVAILABLE 0
#endif

namespace npu::tile_fwk {

#if ARM_PMU_DIRECT_AVAILABLE

// ============================================================
// ARMv8 PMUv3 寄存器访问原语（MRS/MSR 指令）
// ============================================================
namespace arm_pmu_direct {

static inline uint64_t ReadPmccntr()
{
    uint64_t v;
    asm volatile("mrs %0, pmccntr_el0" : "=r"(v));
    return v;
}

static inline uint64_t ReadPmuserenr()
{
    uint64_t v;
    asm volatile("mrs %0, pmuserenr_el0" : "=r"(v));
    return v;
}

static inline uint64_t ReadPmcr()
{
    uint64_t v;
    asm volatile("mrs %0, pmcr_el0" : "=r"(v));
    return v;
}

static inline void WritePmcr(uint64_t v)
{
    asm volatile("msr pmcr_el0, %0" ::"r"(v));
}

static inline void WritePmcntenset(uint64_t v)
{
    asm volatile("msr pmcntenset_el0, %0" ::"r"(v));
}

static inline void WritePmcntenclr(uint64_t v)
{
    asm volatile("msr pmcntenclr_el0, %0" ::"r"(v));
}

// 通用事件计数器读取 PMEVCNTR<n>_EL0
#define ARM_PMU_DEFINE_READ_PMEVCNTR(n)                                      \
    static inline uint64_t ReadPmevcntr##n()                                 \
    {                                                                        \
        uint64_t v;                                                          \
        asm volatile("mrs %0, pmevcntr" #n "_el0" : "=r"(v));                \
        return v;                                                            \
    }
ARM_PMU_DEFINE_READ_PMEVCNTR(0)
ARM_PMU_DEFINE_READ_PMEVCNTR(1)
ARM_PMU_DEFINE_READ_PMEVCNTR(2)
ARM_PMU_DEFINE_READ_PMEVCNTR(3)
ARM_PMU_DEFINE_READ_PMEVCNTR(4)
ARM_PMU_DEFINE_READ_PMEVCNTR(5)
ARM_PMU_DEFINE_READ_PMEVCNTR(6)
ARM_PMU_DEFINE_READ_PMEVCNTR(7)
#undef ARM_PMU_DEFINE_READ_PMEVCNTR

static inline uint64_t ReadPmevcntr(int idx)
{
    switch (idx) {
        case 0: return ReadPmevcntr0();
        case 1: return ReadPmevcntr1();
        case 2: return ReadPmevcntr2();
        case 3: return ReadPmevcntr3();
        case 4: return ReadPmevcntr4();
        case 5: return ReadPmevcntr5();
        case 6: return ReadPmevcntr6();
        case 7: return ReadPmevcntr7();
        default: return 0;
    }
}

#define ARM_PMU_DEFINE_WRITE_PMEVTYPER(n)                                    \
    static inline void WritePmevtyper##n(uint64_t v)                         \
    {                                                                        \
        asm volatile("msr pmevtyper" #n "_el0, %0" ::"r"(v));                \
    }
ARM_PMU_DEFINE_WRITE_PMEVTYPER(0)
ARM_PMU_DEFINE_WRITE_PMEVTYPER(1)
ARM_PMU_DEFINE_WRITE_PMEVTYPER(2)
ARM_PMU_DEFINE_WRITE_PMEVTYPER(3)
ARM_PMU_DEFINE_WRITE_PMEVTYPER(4)
ARM_PMU_DEFINE_WRITE_PMEVTYPER(5)
ARM_PMU_DEFINE_WRITE_PMEVTYPER(6)
ARM_PMU_DEFINE_WRITE_PMEVTYPER(7)
#undef ARM_PMU_DEFINE_WRITE_PMEVTYPER

static inline void WritePmevtyper(int idx, uint64_t evt)
{
    switch (idx) {
        case 0: WritePmevtyper0(evt); break;
        case 1: WritePmevtyper1(evt); break;
        case 2: WritePmevtyper2(evt); break;
        case 3: WritePmevtyper3(evt); break;
        case 4: WritePmevtyper4(evt); break;
        case 5: WritePmevtyper5(evt); break;
        case 6: WritePmevtyper6(evt); break;
        case 7: WritePmevtyper7(evt); break;
        default: break;
    }
}

static inline void IsbBarrier()
{
    asm volatile("isb" ::: "memory");
}

// ---- ARMv8 PMUv3 Common Event Numbers ----
enum ArmPmuEvent : uint32_t {
    EVT_SW_INCR              = 0x00,
    EVT_L1I_CACHE_REFILL     = 0x01,
    EVT_L1D_CACHE_REFILL     = 0x03,
    EVT_L1D_CACHE            = 0x04,
    EVT_INST_RETIRED         = 0x08,
    EVT_BR_MIS_PRED          = 0x10,
    EVT_CPU_CYCLES           = 0x11,
    EVT_BR_PRED              = 0x12,
    EVT_MEM_ACCESS           = 0x13,
    EVT_L1I_CACHE            = 0x14,
    EVT_L2D_CACHE            = 0x16,
    EVT_L2D_CACHE_REFILL     = 0x17,
    EVT_BUS_ACCESS           = 0x19,
    EVT_STALL_FRONTEND       = 0x23,
    EVT_STALL_BACKEND        = 0x24,
};

// ---- PMCR_EL0 位定义 ----
constexpr uint64_t PMCR_E  = 1ULL << 0;   // Enable
constexpr uint64_t PMCR_P  = 1ULL << 1;   // Reset event counters
constexpr uint64_t PMCR_C  = 1ULL << 2;   // Reset cycle counter
constexpr uint64_t PMCR_LC = 1ULL << 6;   // Long cycle counter (64-bit)

constexpr uint64_t PMCCNTR_BIT = 1ULL << 31;  // 周期计数器使能位（PMCNTENSET.C）

// PMEVTYPER_EL0 bit31 (P): Exclude privileged modes (EL1/EL2/EL3), only count EL0
constexpr uint64_t PMEVTYPER_EXCLUDE_PRIVILEGED = 1ULL << 31;

} // namespace arm_pmu_direct

// ============================================================
// 事件索引
// ============================================================
enum ArmPmuDirectEventIdx {
    ARM_PMU_IDX_INSTRUCTIONS = 0,
    ARM_PMU_IDX_L1D_CACHE,
    ARM_PMU_IDX_L1D_MISS,
    ARM_PMU_IDX_L1I_CACHE,
    ARM_PMU_IDX_L1I_MISS,
    ARM_PMU_DIRECT_COLLECT_EVENT_COUNT,
    ARM_PMU_IDX_BRANCHES,
    ARM_PMU_IDX_BRANCH_MISS,
    ARM_PMU_IDX_STALL_BACKEND,
    ARM_PMU_DIRECT_EVENT_COUNT
};

// ============================================================
// ArmPmuDirectSampler: MRS/MSR 直读 PMU 寄存器采样器
// ============================================================
struct ArmPmuDirectSampler {
    static constexpr int MAX_COUNTERS = ARM_PMU_DIRECT_EVENT_COUNT;
    static constexpr int COLLECT_COUNTERS = ARM_PMU_DIRECT_COLLECT_EVENT_COUNT;

    struct EventDesc {
        uint32_t event;
        const char* name;
    };

    static const EventDesc& GetEvent(int idx)
    {
        static const EventDesc kEvents[COLLECT_COUNTERS] = {
            {arm_pmu_direct::EVT_INST_RETIRED,     "instructions"},
            {arm_pmu_direct::EVT_L1D_CACHE,        "l1d_cache"},
            {arm_pmu_direct::EVT_L1D_CACHE_REFILL, "l1d_miss"},
            {arm_pmu_direct::EVT_L1I_CACHE,        "l1i_cache"},
            {arm_pmu_direct::EVT_L1I_CACHE_REFILL, "l1i_miss"},
        };
        return kEvents[idx];
    }

    // 运行时探测：检查 EL0 是否允许直读 PMU（PMUSERENR_EL0.EN 位）
    static bool ProbeAvailable()
    {
        uint64_t v = arm_pmu_direct::ReadPmuserenr();
        return (v & 0x1) != 0;
    }

    bool Available() const { return available_; }

    ArmPmuDirectSampler()
    {
        if (!ProbeAvailable()) {
            DEV_WARN("[ARM_PMU_DIRECT] User-space PMU access disabled (PMUSERENR_EL0=0x%lx). "
                     "Enable via: echo 1 > /proc/sys/kernel/perf_user_access, "
                     "or load a kernel module that sets PMUSERENR_EL0.EN=1",
                     arm_pmu_direct::ReadPmuserenr());
            available_ = false;
            return;
        }

        // 读取硬件通用计数器数量（PMCR_EL0.N 字段：bits [15:11]）
        uint64_t pmcr = arm_pmu_direct::ReadPmcr();
        int hwCounters = static_cast<int>((pmcr >> 11) & 0x1F);
        activeCounters_ = (hwCounters < COLLECT_COUNTERS) ? hwCounters : COLLECT_COUNTERS;
        if (activeCounters_ == 0) {
            DEV_WARN("[ARM_PMU_DIRECT] No hardware event counters available");
            available_ = false;
            return;
        }

        // 清零全部计数器使能位
        arm_pmu_direct::WritePmcntenclr(0xFFFFFFFF);

        // 配置事件类型，同时设置 bit31 排除特权模式（仅统计 EL0 用户态）
        for (int i = 0; i < activeCounters_; ++i) {
            arm_pmu_direct::WritePmevtyper(i, GetEvent(i).event | arm_pmu_direct::PMEVTYPER_EXCLUDE_PRIVILEGED);
        }

        // 启用 PMU（E=1，LC=1 开启 64-bit 周期计数器）
        arm_pmu_direct::WritePmcr(pmcr | arm_pmu_direct::PMCR_E | arm_pmu_direct::PMCR_LC);

        // 使能选中的事件计数器 + 专用周期计数器
        uint64_t mask = (1ULL << activeCounters_) - 1;
        arm_pmu_direct::WritePmcntenset(mask | arm_pmu_direct::PMCCNTR_BIT);
        arm_pmu_direct::IsbBarrier();

        available_ = true;
        DEV_INFO("[ARM_PMU_DIRECT] Enabled, %d hardware counters active", activeCounters_);
    }

    void Begin()
    {
        if (!available_) {
            return;
        }
        arm_pmu_direct::IsbBarrier();
        startCycles_ = arm_pmu_direct::ReadPmccntr();
        for (int i = 0; i < activeCounters_; ++i) {
            startCounts_[i] = arm_pmu_direct::ReadPmevcntr(i);
        }
    }

    void End()
    {
        if (!available_) {
            return;
        }
        arm_pmu_direct::IsbBarrier();
        endCycles_ = arm_pmu_direct::ReadPmccntr();
        for (int i = 0; i < activeCounters_; ++i) {
            endCounts_[i] = arm_pmu_direct::ReadPmevcntr(i);
        }
    }

    void Dump()
    {
        uint64_t cycles = endCycles_ - startCycles_;
        double timeUs = dynamic::Cycles2Us(cycles);
        double timeMs = timeUs / 1000.0;

        if (!available_) {
            DEV_ERROR(ERROR_CODE_UNDEFINED, "[ARM_PMU_DIRECT] Sampler disabled (PMU user access unavailable)");
            DEV_ERROR(ERROR_CODE_UNDEFINED, "  Total Running Time: %.2f us (%.2f ms)", timeUs, timeMs);
            return;
        }

        uint64_t delta[MAX_COUNTERS] = {0};
        for (int i = 0; i < activeCounters_; ++i) {
            delta[i] = endCounts_[i] - startCounts_[i];
        }

        DEV_ERROR(ERROR_CODE_UNDEFINED, "[ARM_PMU_DIRECT] Performance Report (MRS/MSR mode)");
        DEV_ERROR(ERROR_CODE_UNDEFINED, "============================================================");
        DEV_ERROR(ERROR_CODE_UNDEFINED, "Total Running Time: %.2f us (%.2f ms)", timeUs, timeMs);

        // Raw PMU event counters
        DEV_ERROR(ERROR_CODE_UNDEFINED, "------------------------------------------------------------");
        DEV_ERROR(ERROR_CODE_UNDEFINED, "Raw PMU Event Counters");
        DEV_ERROR(ERROR_CODE_UNDEFINED, "------------------------------------------------------------");
        DEV_ERROR(ERROR_CODE_UNDEFINED, "  CPU Cycles (PMCCNTR): %s", FormatNumber(cycles).c_str());
        DEV_ERROR(ERROR_CODE_UNDEFINED, "  Instructions:         %s", FormatCounter(delta, ARM_PMU_IDX_INSTRUCTIONS).c_str());
        DEV_ERROR(ERROR_CODE_UNDEFINED, "  Branch Instructions:  %s", FormatCounter(delta, ARM_PMU_IDX_BRANCHES).c_str());
        DEV_ERROR(ERROR_CODE_UNDEFINED, "  Branch Misses:        %s", FormatCounter(delta, ARM_PMU_IDX_BRANCH_MISS).c_str());
        DEV_ERROR(ERROR_CODE_UNDEFINED, "  L1D Cache Refs:       %s", FormatCounter(delta, ARM_PMU_IDX_L1D_CACHE).c_str());
        DEV_ERROR(ERROR_CODE_UNDEFINED, "  L1D Cache Misses:     %s", FormatCounter(delta, ARM_PMU_IDX_L1D_MISS).c_str());
        DEV_ERROR(ERROR_CODE_UNDEFINED, "  L1I Cache Refs:       %s", FormatCounter(delta, ARM_PMU_IDX_L1I_CACHE).c_str());
        DEV_ERROR(ERROR_CODE_UNDEFINED, "  L1I Cache Misses:     %s", FormatCounter(delta, ARM_PMU_IDX_L1I_MISS).c_str());
        DEV_ERROR(ERROR_CODE_UNDEFINED, "  Stall Backend:        %s", FormatCounter(delta, ARM_PMU_IDX_STALL_BACKEND).c_str());

        // Derived metrics computed from the raw counters above.
        DEV_ERROR(ERROR_CODE_UNDEFINED, "------------------------------------------------------------");
        DEV_ERROR(ERROR_CODE_UNDEFINED, "Derived Metrics");
        DEV_ERROR(ERROR_CODE_UNDEFINED, "------------------------------------------------------------");
        DumpIpcMetric(cycles, delta);
        DumpRateMetric("Branch Miss Rate", delta, ARM_PMU_IDX_BRANCH_MISS, ARM_PMU_IDX_BRANCHES);
        DumpCacheDerivedMetric("L1D Cache", delta, ARM_PMU_IDX_L1D_CACHE, ARM_PMU_IDX_L1D_MISS);
        DumpCacheDerivedMetric("L1I Cache", delta, ARM_PMU_IDX_L1I_CACHE, ARM_PMU_IDX_L1I_MISS);
        DumpRateMetric("Stall Backend Rate", delta, ARM_PMU_IDX_STALL_BACKEND, cycles);
        DEV_ERROR(ERROR_CODE_UNDEFINED, "============================================================");
    }

private:
    bool IsEventActive(int idx) const
    {
        return idx >= 0 && idx < activeCounters_;
    }

    std::string FormatCounter(const uint64_t* delta, int idx)
    {
        if (!IsEventActive(idx)) {
            return "N/A";
        }
        return FormatNumber(delta[idx]);
    }

    std::string FormatNumber(uint64_t n)
    {
        std::stringstream ss;
        ss.imbue(std::locale(""));
        ss << n;
        return ss.str();
    }

    void DumpIpcMetric(uint64_t cycles, const uint64_t* delta)
    {
        if (!IsEventActive(ARM_PMU_IDX_INSTRUCTIONS) || cycles == 0) {
            DEV_ERROR(ERROR_CODE_UNDEFINED, "  IPC:                 N/A");
            DEV_ERROR(ERROR_CODE_UNDEFINED, "  CPI:                 N/A");
            return;
        }
        uint64_t inst = delta[ARM_PMU_IDX_INSTRUCTIONS];
        double ipc = static_cast<double>(inst) / cycles;
        double cpi = inst > 0 ? static_cast<double>(cycles) / inst : 0.0;
        DEV_ERROR(ERROR_CODE_UNDEFINED, "  IPC:                 %.3f", ipc);
        DEV_ERROR(ERROR_CODE_UNDEFINED, "  CPI:                 %.3f", cpi);
    }

    void DumpRateMetric(const char* name, const uint64_t* delta, int numeratorIdx, int denominatorIdx)
    {
        if (!IsEventActive(numeratorIdx) || !IsEventActive(denominatorIdx) || delta[denominatorIdx] == 0) {
            DEV_ERROR(ERROR_CODE_UNDEFINED, "  %s: N/A", name);
            return;
        }
        double rate = static_cast<double>(delta[numeratorIdx]) / delta[denominatorIdx] * 100.0;
        DEV_ERROR(ERROR_CODE_UNDEFINED, "  %s: %.2f%%", name, rate);
    }

    void DumpRateMetric(const char* name, const uint64_t* delta, int numeratorIdx, uint64_t denominator)
    {
        if (!IsEventActive(numeratorIdx) || denominator == 0) {
            DEV_ERROR(ERROR_CODE_UNDEFINED, "  %s: N/A", name);
            return;
        }
        double rate = static_cast<double>(delta[numeratorIdx]) / denominator * 100.0;
        DEV_ERROR(ERROR_CODE_UNDEFINED, "  %s: %.2f%%", name, rate);
    }

    void DumpCacheDerivedMetric(const char* name, const uint64_t* delta, int refsIdx, int missesIdx)
    {
        if (!IsEventActive(refsIdx) || !IsEventActive(missesIdx) || delta[refsIdx] == 0) {
            DEV_ERROR(ERROR_CODE_UNDEFINED, "  %s Hit Rate:  N/A", name);
            DEV_ERROR(ERROR_CODE_UNDEFINED, "  %s Miss Rate: N/A", name);
            return;
        }
        double missRate = static_cast<double>(delta[missesIdx]) / delta[refsIdx] * 100.0;
        DEV_ERROR(ERROR_CODE_UNDEFINED, "  %s Hit Rate:  %.2f%%", name, 100.0 - missRate);
        DEV_ERROR(ERROR_CODE_UNDEFINED, "  %s Miss Rate: %.2f%%", name, missRate);
    }

    bool available_{false};
    int activeCounters_{0};
    uint64_t startCycles_{0};
    uint64_t endCycles_{0};
    uint64_t startCounts_[MAX_COUNTERS]{};
    uint64_t endCounts_[MAX_COUNTERS]{};
};

// ============================================================
// RAII 作用域采样器
// ============================================================
struct ArmPmuDirectScopedSampler {
    explicit ArmPmuDirectScopedSampler(const char* sectionName) : sectionName_(sectionName)
    {
        sampler_.Begin();
    }

    ~ArmPmuDirectScopedSampler()
    {
        sampler_.End();
        DEV_ERROR(ERROR_CODE_UNDEFINED, "[ARM_PMU_DIRECT] %s", sectionName_);
        sampler_.Dump();
    }

private:
    const char* sectionName_{"unnamed"};
    ArmPmuDirectSampler sampler_;
};

// ============================================================
// 便捷宏接口（仅在 ARM_PMU_DIRECT_AVAILABLE 时有效）
// ============================================================
#define ARM_PMU_DIRECT_SCOPE(section_name_literal) \
    ::npu::tile_fwk::ArmPmuDirectScopedSampler armPmuDirectScoped_##__LINE__(section_name_literal)

#define ARM_PMU_DIRECT_BEGIN(sampler_name) \
    ::npu::tile_fwk::ArmPmuDirectSampler sampler_name; \
    (sampler_name).Begin()

#define ARM_PMU_DIRECT_END(sampler_name, section_name_literal) \
    do { \
        (sampler_name).End(); \
        DEV_ERROR(ERROR_CODE_UNDEFINED, "[ARM_PMU_DIRECT] %s", section_name_literal); \
        (sampler_name).Dump(); \
    } while (0)

#define ARM_PMU_DIRECT_BEGIN_EXTERNAL(sampler_ptr) \
    do { (sampler_ptr)->Begin(); } while (0)

#define ARM_PMU_DIRECT_END_EXTERNAL(sampler_ptr, section_name_literal) \
    do { \
        (sampler_ptr)->End(); \
        DEV_ERROR(ERROR_CODE_UNDEFINED, "[ARM_PMU_DIRECT] %s", section_name_literal); \
        (sampler_ptr)->Dump(); \
    } while (0)

#else  // !ARM_PMU_DIRECT_AVAILABLE

// ============================================================
// 关闭态空实现：保证业务代码无需条件编译也可编译通过
// ============================================================
struct ArmPmuDirectSampler {
    static bool ProbeAvailable() { return false; }
    bool Available() const { return false; }
    void Begin() {}
    void End() {}
    void Dump() {}
};

struct ArmPmuDirectScopedSampler {
    explicit ArmPmuDirectScopedSampler([[maybe_unused]] const char* sectionName) {}
};

#define ARM_PMU_DIRECT_SCOPE(section_name_literal)
#define ARM_PMU_DIRECT_BEGIN(sampler_name)
#define ARM_PMU_DIRECT_END(sampler_name, section_name_literal)
#define ARM_PMU_DIRECT_BEGIN_EXTERNAL(sampler_ptr)
#define ARM_PMU_DIRECT_END_EXTERNAL(sampler_ptr, section_name_literal)

#endif  // ARM_PMU_DIRECT_AVAILABLE

} // namespace npu::tile_fwk
