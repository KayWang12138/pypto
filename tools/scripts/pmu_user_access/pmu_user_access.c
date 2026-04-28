/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*
 * pmu_user_access.c
 *
 * Enable EL0 (user-space) access to ARMv8 PMUv3 registers by writing
 * PMUSERENR_EL0 = 0xF on every online CPU. The cycle counter is configured to
 * count EL0 only, matching perf_event exclude_kernel behavior. A CPU hotplug
 * callback keeps the bit set when a CPU is brought online after initial module
 * load.
 *
 * This module is a one-shot bootstrap for the direct-register sampler in
 * framework/src/machine/utils/arm_pmu_direct_sampler.h, targeted at kernels
 * that predate the 5.17 `kernel.perf_user_access` sysctl interface.
 *
 * PMUSERENR_EL0 bit layout (ARMv8):
 *   bit0 EN : EL0 access to event counters (PMEVCNTR<n>_EL0)
 *   bit1 SW : EL0 write to PMSWINC_EL0
 *   bit2 CR : EL0 read of cycle counter (PMCCNTR_EL0)
 *   bit3 ER : EL0 read of event counters via PMXEVCNTR_EL0
 * Writing 0xF opens all four capabilities.
 *
 * PMCCFILTR_EL0 relevant filter bits:
 *   bit31 P : Do not count cycles in privileged modes (EL1/EL2/EL3)
 *   bit30 U : Do not count cycles in EL0
 */

#include <linux/module.h>
#include <linux/init.h>
#include <linux/smp.h>
#include <linux/cpu.h>
#include <linux/cpuhotplug.h>
#include <linux/printk.h>

#define PMUSERENR_EN_ALL 0xFUL
#define PMU_CNTENSET_CYCLE_BIT (1UL << 31)
#define PMCCFILTR_EXCLUDE_PRIVILEGED (1UL << 31)

static enum cpuhp_state g_pmu_hp_state;

static void PmuWriteOnCpu(void *info)
{
    unsigned long val = (unsigned long)info;
    unsigned long pmcr = 0;

    asm volatile("msr pmuserenr_el0, %0" ::"r"(val));
    /*
     * Make sure the cycle counter is actually running; PMUSERENR_EL0 only
     * controls EL0 accessibility, not whether PMCCNTR_EL0 is enabled.
     */
    asm volatile("mrs %0, pmcr_el0" : "=r"(pmcr));
    pmcr |= 1UL; /* PMCR_EL0.E: enable all counters */
    asm volatile("msr pmcr_el0, %0" ::"r"(pmcr));
    asm volatile("msr pmccfiltr_el0, %0" ::"r"(PMCCFILTR_EXCLUDE_PRIVILEGED));
    asm volatile("msr pmcntenset_el0, %0" ::"r"(PMU_CNTENSET_CYCLE_BIT));
    asm volatile("isb" ::: "memory");
}

static int PmuCpuOnline(unsigned int cpu)
{
    /*
     * PMU EL0 access control is per-CPU; ensure we program the target CPU.
     * The CPUHP callback is not guaranteed to run on @cpu itself.
     */
    smp_call_function_single(cpu, PmuWriteOnCpu, (void *)PMUSERENR_EN_ALL, 1);
    return 0;
}

static int PmuCpuOffline(unsigned int cpu)
{
    smp_call_function_single(cpu, PmuWriteOnCpu, (void *)0UL, 1);
    return 0;
}

static int __init PmuUserAccessInit(void)
{
    int ret;

    on_each_cpu(PmuWriteOnCpu, (void *)PMUSERENR_EN_ALL, 1);

    ret = cpuhp_setup_state(CPUHP_AP_ONLINE_DYN,
                            "pmu_user_access:online",
                            PmuCpuOnline,
                            PmuCpuOffline);
    if (ret < 0) {
        on_each_cpu(PmuWriteOnCpu, (void *)0UL, 1);
        pr_err("pmu_user_access: cpuhp_setup_state failed: %d\n", ret);
        return ret;
    }
    g_pmu_hp_state = ret;

    pr_info("pmu_user_access: enabled (PMUSERENR_EL0=0x%lx) on all CPUs\n",
            PMUSERENR_EN_ALL);
    return 0;
}

static void __exit PmuUserAccessExit(void)
{
    cpuhp_remove_state(g_pmu_hp_state);
    on_each_cpu(PmuWriteOnCpu, (void *)0UL, 1);
    pr_info("pmu_user_access: disabled on all CPUs\n");
}

module_init(PmuUserAccessInit);
module_exit(PmuUserAccessExit);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("CANN pypto_pmu");
MODULE_DESCRIPTION("Enable EL0 access to ARMv8 PMUv3 registers and filter PMCCNTR_EL0 to EL0");
MODULE_VERSION("1.0");
