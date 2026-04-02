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
 * \file acl_define.h
 * \brief
 */

#pragma once

#ifdef BUILD_WITH_CANN
#include "acl/acl_base_rt.h"
#include "acl/acl_rt.h"
#include "runtime/base.h"
#else
#include <cstdint>
#include <cstddef>

namespace npu::tile_fwk {
constexpr int ACL_ERROR_NONE = 0;
constexpr int ACL_SUCCESS = 0;
constexpr int ACL_ERROR_REPEAT_INITIALIZE = 100002;

#define  ACL_ERROR_RT_FEATURE_NOT_SUPPORT        207000 // feature not support
#define ACL_EVENT_SYNC                    0x00000001U

typedef int aclError;
typedef void *aclrtStream;
typedef void *aclrtEvent;

typedef enum aclrtMemcpyKind {
    ACL_MEMCPY_HOST_TO_HOST,
    ACL_MEMCPY_HOST_TO_DEVICE,
    ACL_MEMCPY_DEVICE_TO_HOST,
    ACL_MEMCPY_DEVICE_TO_DEVICE,
    ACL_MEMCPY_DEFAULT,
    ACL_MEMCPY_HOST_TO_BUF_TO_DEVICE,
    ACL_MEMCPY_INNER_DEVICE_TO_DEVICE,
    ACL_MEMCPY_INTER_DEVICE_TO_DEVICE,
} aclrtMemcpyKind;

typedef enum {
    ACL_STREAM_ATTR_FAILURE_MODE         = 1,
    ACL_STREAM_ATTR_FLOAT_OVERFLOW_CHECK = 2,
    ACL_STREAM_ATTR_USER_CUSTOM_TAG      = 3,
    ACL_STREAM_ATTR_CACHE_OP_INFO        = 4,
} aclrtStreamAttr;

typedef union {
    uint64_t failureMode;
    uint32_t overflowSwitch;
    uint32_t userCustomTag;
    uint32_t cacheOpInfoSwitch;
    uint32_t reserve[4];
} aclrtStreamAttrValue;

typedef enum {
    ACL_RT_DEV_RES_CUBE_CORE = 0,
    ACL_RT_DEV_RES_VECTOR_CORE,
} aclrtDevResLimitType;

typedef enum tagRtExceptionExpandType {
    RT_EXCEPTION_INVALID = 0,
    RT_EXCEPTION_FFTS_PLUS,
    RT_EXCEPTION_AICORE,
    RT_EXCEPTION_UB,
    RT_EXCEPTION_CCU,
    RT_EXCEPTION_FUSION
} rtExceptionExpandType_t;

typedef struct rtArgsSizeInfo {
    void *infoAddr; /* info : atomicIndex|input num input offset|size|size */
    uint32_t atomicIndex;
} rtArgsSizeInfo_t;

typedef void *rtBinHandle;

typedef struct rtExceptionKernelInfo {
    uint32_t binSize;
    rtBinHandle bin; // binHandle
    uint32_t kernelNameSize;
    const char *kernelName;
    const void *dfxAddr;
    uint16_t dfxSize;
    uint8_t reserved[2]; // 填补空间以保持四字节对齐
    int32_t elfDataFlag;
} rtExceptionKernelInfo_t;

typedef struct rtExceptionArgsInfo {
    uint32_t argsize;
    void *argAddr;
    rtArgsSizeInfo_t sizeInfo;
    rtExceptionKernelInfo_t exceptionKernelInfo; // 新增结构体，注意兼容性问题
} rtExceptionArgsInfo_t;

typedef struct rtFftsPlusExDetailInfo {
    uint16_t contextId;
    uint16_t threadId;
    rtExceptionArgsInfo_t exceptionArgs;
} rtFftsPlusExDetailInfo_t;

typedef struct rtAicoreExDetailInfo {
    rtExceptionArgsInfo_t exceptionArgs;
} rtAicoreExDetailInfo_t;

typedef enum rtUbExType {
    RT_UB_TYPE_DOORBELL,
    RT_UB_TYPE_DIRECT_WQE
} rtUbExType_t;

typedef struct rtUbInfo {
    uint8_t functionId;
    uint8_t dieId;
    uint16_t jettyId;
    uint16_t piValue;  // directWqe类型下该字段无效
} rtUbInfo_t;

#define UB_DB_SEND_MAX_NUM (4)
#define RT_CCU_SQE_ARGS_LEN     (13U)
#define MAX_CCU_EXCEPTION_INFO_SIZE (128U)
#define FUSION_SUB_TASK_MAX_CCU_NUM (8U)

typedef struct rtUbExDetailInfo {
    rtUbExType_t ubType;
    uint8_t ubNum;
    uint8_t resv[3];
    rtUbInfo_t info[UB_DB_SEND_MAX_NUM];
} rtUbExDetailInfo_t;

typedef struct rtCCUExDetailInfo {
    uint8_t dieId;
    uint8_t missionId;
    uint16_t instrId;
    uint64_t args[RT_CCU_SQE_ARGS_LEN];
    uint8_t status;
    uint8_t subStatus;
    uint8_t panicLog[MAX_CCU_EXCEPTION_INFO_SIZE];
} rtCcuMissionDetailInfo_t;

typedef struct rtMultiCCUExDetailInfo {
    uint16_t ccuMissionNum;
    rtCcuMissionDetailInfo_t missionInfo[FUSION_SUB_TASK_MAX_CCU_NUM];
} rtMultiCCUExDetailInfo_t;

typedef enum rtFusionType {
    RT_FUSION_AICORE_CCU,
    RT_FUSION_AICORE_AICPU
} rtFusionExType_t;

typedef struct rtFusionAICoreCCUExDetailInfo {
    rtExceptionArgsInfo_t exceptionArgs;
    rtMultiCCUExDetailInfo_t ccuDetailMsg;
} rtFusionAICoreCCUExDetailInfo_t;

typedef struct rtFusionExDetailInfo {
    rtFusionExType_t type;
    union {
        rtFusionAICoreCCUExDetailInfo_t aicoreCcuInfo;
    } u;
} rtFusionExDetailInfo_t;

typedef struct rtExceptionExpandInfo {
    rtExceptionExpandType_t type;
    union {
        rtFftsPlusExDetailInfo_t fftsPlusInfo;
        rtAicoreExDetailInfo_t aicoreInfo; // 关注下影响
        rtUbExDetailInfo_t ubInfo;
        rtMultiCCUExDetailInfo_t ccuInfo;       /* use for ccu task */
        rtFusionExDetailInfo_t fusionInfo;      /* use for fusion task */
    } u;
} rtExceptionExpandInfo_t;

typedef struct rtExceptionInfo {
    uint32_t taskid;
    uint32_t streamid;
    uint32_t tid;
    uint32_t deviceid;
    uint32_t retcode;
    rtExceptionExpandInfo_t expandInfo;
} rtExceptionInfo_t;

typedef struct rtExceptionInfo aclrtExceptionInfo;

typedef void (*aclrtExceptionInfoCallback)(aclrtExceptionInfo *exceptionInfo);

typedef void *aclmdlRI;

typedef enum {
    ACL_MODEL_RI_CAPTURE_STATUS_NONE = 0,
    ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE,
    ACL_MODEL_RI_CAPTURE_STATUS_INVALIDATED,
} aclmdlRICaptureStatus;

typedef enum {
    ACL_MODEL_RI_CAPTURE_MODE_GLOBAL = 0,
    ACL_MODEL_RI_CAPTURE_MODE_THREAD_LOCAL,
    ACL_MODEL_RI_CAPTURE_MODE_RELAXED,
} aclmdlRICaptureMode;
}
#endif
