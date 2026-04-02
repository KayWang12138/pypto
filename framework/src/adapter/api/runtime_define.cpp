/**
* Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file runtime_define.cpp
 * \brief
 */

#include "adapter/api/runtime_define.h"
#ifdef BUILD_WITH_CANN
#include <type_traits>
#include "runtime/base.h"
#include "runtime/mem.h"
#include "runtime/kernel.h"
#include "runtime/rts/rts_kernel.h"
#endif

namespace npu::tile_fwk {
#ifdef BUILD_WITH_CANN
static_assert(std::is_same<RtError, rtError_t>::value);
static_assert(std::is_same<RtMemType, rtMemType_t>::value);
static_assert(std::is_same<RtStream, rtStream_t>::value);
static_assert(std::is_same<RtModel, rtModel_t>::value);
static_assert(std::is_same<RtFuncHandle, rtFuncHandle>::value);
static_assert(std::is_same<RtBinHandle, rtBinHandle>::value);
static_assert(RT_SUCCESS == RT_ERROR_NONE);
static_assert(sizeof(RtDevBinary) == sizeof(rtDevBinary_t));
static_assert(sizeof(RtHostInputInfo) == sizeof(rtHostInputInfo_t));
static_assert(sizeof(RtArgsEx) == sizeof(rtArgsEx_t));
static_assert(sizeof(RtSmData) == sizeof(rtSmData_t));
static_assert(sizeof(RtSmDesc) == sizeof(rtSmDesc_t));
static_assert(sizeof(RtTaskCfgInfo) == sizeof(rtTaskCfgInfo_t));
static_assert(sizeof(RtAicpuArgsEx) == sizeof(rtAicpuArgsEx_t));
static_assert(sizeof(RtCpuKernelArgs) == sizeof(rtCpuKernelArgs_t));
static_assert(sizeof(RtTimeoutUs) == sizeof(rtTimeoutUs));
static_assert(sizeof(RtLaunchKernelAttrVal) == sizeof(rtLaunchKernelAttrVal_t));
static_assert(sizeof(RtKernelLaunchCfg) == sizeof(rtKernelLaunchCfg_t));
static_assert(sizeof(RtLoadBinaryOptionValue) == sizeof(rtLoadBinaryOptionValue_t));
static_assert(sizeof(RtLoadBinaryOption) == sizeof(rtLoadBinaryOption_t));
static_assert(sizeof(RtLoadBinaryConfig) == sizeof(rtLoadBinaryConfig_t));
static_assert(sizeof(RtArgsSizeInfo) == sizeof(rtArgsSizeInfo_t));
static_assert(sizeof(RtExceptionKernelInfo) == sizeof(rtExceptionKernelInfo_t));
static_assert(sizeof(RtExceptionArgsInfo) == sizeof(rtExceptionArgsInfo_t));
static_assert(sizeof(RtFftsPlusExDetailInfo) == sizeof(rtFftsPlusExDetailInfo_t));
static_assert(sizeof(RtAicoreExDetailInfo) == sizeof(rtAicoreExDetailInfo_t));
static_assert(sizeof(RtUbExDetailInfo) == sizeof(rtUbExDetailInfo_t));
static_assert(sizeof(RtCcuMissionDetailInfo) == sizeof(rtCcuMissionDetailInfo_t));
static_assert(sizeof(RtMultiCCUExDetailInfo) == sizeof(rtMultiCCUExDetailInfo_t));
static_assert(sizeof(RtFusionAICoreCCUExDetailInfo) == sizeof(rtFusionAICoreCCUExDetailInfo_t));
static_assert(sizeof(RtFusionExDetailInfo) == sizeof(rtFusionExDetailInfo_t));
static_assert(sizeof(RtExceptionExpandInfo) == sizeof(rtExceptionExpandInfo_t));
static_assert(sizeof(RtExceptionInfo) == sizeof(rtExceptionInfo_t));
#endif
}