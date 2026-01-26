/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aicore_prof_dav3510.h
 * \brief
 */

#ifndef AICORE_PROF_DAV3510_H
#define AICORE_PROF_DAV3510_H

#include <cstdint>
#include <vector>

namespace npu::tile_fwk::dynamic {

namespace DAV_3510 {
    const uint32_t PMU_CTRL_0 = 0x4200;
    const uint32_t PMU_CTRL_1 = 0X2400;
    const uint32_t PMU_CNT0 = 0x4210;
    const uint32_t PMU_CNT1 = 0x4218;
    const uint32_t PMU_CNT2 = 0x4220;
    const uint32_t PMU_CNT3 = 0x4228;
    const uint32_t PMU_CNT4 = 0x4230;
    const uint32_t PMU_CNT5 = 0x4238;
    const uint32_t PMU_CNT6 = 0x4240;
    const uint32_t PMU_CNT7 = 0x4248;
    const uint32_t PMU_CNT8 = 0x4250;
    const uint32_t PMU_CNT9 = 0x4254;
    const uint32_t PMU_CNT_TOTAL0 = 0x4260;
    const uint32_t PMU_CNT_TOTAL1 = 0x4264;
    const uint32_t PMU_CNT0_IDX = 0x2500;
    const uint32_t PMU_CNT1_IDX = 0x2504;
    const uint32_t PMU_CNT2_IDX = 0x2508;
    const uint32_t PMU_CNT3_IDX = 0x250C;
    const uint32_t PMU_CNT4_IDX = 0x2510;
    const uint32_t PMU_CNT5_IDX = 0x2514;
    const uint32_t PMU_CNT6_IDX = 0x2518;
    const uint32_t PMU_CNT7_IDX = 0x251C;
    const uint32_t PMU_CNT8_IDX = 0x2520;
    const uint32_t PMU_CNT9_IDX = 0x2524;
    const uint32_t PMU_START_CNT_CYC_0 = 0x42A0;
    const uint32_t PMU_START_CNT_CYC_1 = 0x42A4;
    const uint32_t PMU_STOP_CNT_CYC_0 = 0x42A8;
    const uint32_t PMU_STOP_CNT_CYC_1 = 0x42AC;
}

inline ArchPmuConfig InitDav3510PmuConfig() {
    return {
        {DAV_3510::PMU_CNT0_IDX, DAV_3510::PMU_CNT1_IDX, DAV_3510::PMU_CNT2_IDX, DAV_3510::PMU_CNT3_IDX,
         DAV_3510::PMU_CNT4_IDX, DAV_3510::PMU_CNT5_IDX, DAV_3510::PMU_CNT6_IDX, DAV_3510::PMU_CNT7_IDX,
         DAV_3510::PMU_CNT8_IDX, DAV_3510::PMU_CNT9_IDX},
        {DAV_3510::PMU_CNT0, DAV_3510::PMU_CNT1, DAV_3510::PMU_CNT2, DAV_3510::PMU_CNT3,
         DAV_3510::PMU_CNT4, DAV_3510::PMU_CNT5, DAV_3510::PMU_CNT6, DAV_3510::PMU_CNT7,
         DAV_3510::PMU_CNT8, DAV_3510::PMU_CNT9},
        DAV_3510::PMU_CNT_TOTAL0, DAV_3510::PMU_CNT_TOTAL1,
        DAV_3510::PMU_CTRL_0, DAV_3510::PMU_CTRL_1,
        DAV_3510::PMU_START_CNT_CYC_0, DAV_3510::PMU_START_CNT_CYC_1,
        DAV_3510::PMU_STOP_CNT_CYC_0, DAV_3510::PMU_STOP_CNT_CYC_1,
        USER_PMU_MODE_EN + (SAMPLE_PMU_MODE_EN << 1), GLB_PMU_EN
    };
}

} // namespace npu::tile_fwk::dynamic

#endif // AICORE_PROF_DAV3510_H
