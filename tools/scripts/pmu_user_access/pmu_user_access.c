/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
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
 * PMUSERENR_EL0 = 0xF on every online CPU. A CPU hotplug callback keeps the
 * bit set when a CPU is brought online after initial module load.
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
 */

#include <linux/module.h>
#include <linux/init.h>
#include <linux/smp.h>
#include <linux/cpu.h>
#include <linux/cpuhotplug.h>
#include <linux/printk.h>

#define PMUSERENR_EN_ALL 0xFUL

static enum cpuhp_state g_pmu_hp_state;

static void PmuWriteOnCpu(void *info)
{
    unsigned long val = (unsigned long)info;

    asm volatile("msr pmuserenr_el0, %0" ::"r"(val));
    asm volatile("isb" ::: "memory");
}

static int PmuCpuOnline(unsigned int cpu)
{
    PmuWriteOnCpu((void *)PMUSERENR_EN_ALL);
    return 0;
}

static int PmuCpuOffline(unsigned int cpu)
{
    PmuWriteOnCpu((void *)0UL);
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
MODULE_DESCRIPTION("Enable EL0 access to ARMv8 PMUv3 registers (PMUSERENR_EL0=0xF)");
MODULE_VERSION("1.0");
