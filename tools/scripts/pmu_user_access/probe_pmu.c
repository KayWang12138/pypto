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
 * probe_pmu.c
 *
 * Minimal user-space probe to verify that EL0 PMU access is enabled after
 * loading pmu_user_access.ko. Reads PMUSERENR_EL0 and, if EN bit is set,
 * reads PMCCNTR_EL0 to confirm the cycle counter is actually accessible.
 *
 * Build:  gcc -O2 probe_pmu.c -o probe_pmu
 * Run  :  ./probe_pmu              # current CPU
 *         taskset -c N ./probe_pmu # pin to CPU N
 */

#include <stdio.h>
#include <stdint.h>
#include <sched.h>
#include <unistd.h>

int main(void)
{
    uint64_t pmuserenr = 0;
    uint64_t cycles = 0;
    int cpu = sched_getcpu();

    __asm__ volatile("mrs %0, pmuserenr_el0" : "=r"(pmuserenr));
    printf("[cpu%d] PMUSERENR_EL0 = 0x%lx\n", cpu, (unsigned long)pmuserenr);

    if ((pmuserenr & 0x1UL) == 0) {
        puts("[cpu?] EL0 PMU access is NOT enabled on this CPU.");
        puts("       Load the kernel module first: sudo insmod pmu_user_access.ko");
        return 1;
    }

    __asm__ volatile("mrs %0, pmccntr_el0" : "=r"(cycles));
    printf("[cpu%d] PMCCNTR_EL0  = %lu (cycles)\n", cpu, (unsigned long)cycles);
    return 0;
}
