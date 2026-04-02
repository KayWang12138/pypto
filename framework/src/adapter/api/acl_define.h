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

#include <cstdint>
#include <cstddef>

namespace npu::tile_fwk {
constexpr int ACL_RT_SUCCESS = 0;
constexpr int ACL_RT_ERROR_REPEAT_INITIALIZE = 100002;

#define ACL_RT_ERROR_FEATURE_NOT_SUPPORT        207000 // feature not support
#define ACL_RT_EVENT_SYNC                    0x00000001U

typedef int AclError;
typedef void *AclRtStream;
typedef void *AclRtEvent;

enum class AclRtMemcpyKind {
    HOST_TO_HOST,
    HOST_TO_DEVICE,
    DEVICE_TO_HOST,
    DEVICE_TO_DEVICE,
    DEFAULT,
    HOST_TO_BUF_TO_DEVICE,
    INNER_DEVICE_TO_DEVICE,
    INTER_DEVICE_TO_DEVICE,
};

enum class AclRtStreamAttr {
    FAILURE_MODE         = 1,
    FLOAT_OVERFLOW_CHECK = 2,
    USER_CUSTOM_TAG      = 3,
    CACHE_OP_INFO        = 4,
};

typedef union {
    uint64_t failureMode;
    uint32_t overflowSwitch;
    uint32_t userCustomTag;
    uint32_t cacheOpInfoSwitch;
    uint32_t reserve[4];
} AclRtStreamAttrValue;

enum class AclRtDevResLimitType {
    CUBE_CORE = 0,
    VECTOR_CORE,
};

enum class RtExceptionExpandType {
    INVALID = 0,
    FFTS_PLUS,
    AICORE,
    UB,
    CCU,
    FUSION
};

struct RtArgsSizeInfo {
    void *infoAddr; /* info : atomicIndex|input num input offset|size|size */
    uint32_t atomicIndex;
};

typedef void *RtBinHandle;

struct RtExceptionKernelInfo {
    uint32_t binSize;
    RtBinHandle bin; // binHandle
    uint32_t kernelNameSize;
    const char *kernelName;
    const void *dfxAddr;
    uint16_t dfxSize;
    uint8_t reserved[2]; // 填补空间以保持四字节对齐
    int32_t elfDataFlag;
};

struct RtExceptionArgsInfo {
    uint32_t argsize;
    void *argAddr;
    RtArgsSizeInfo sizeInfo;
    RtExceptionKernelInfo exceptionKernelInfo; // 新增结构体，注意兼容性问题
};

struct RtFftsPlusExDetailInfo {
    uint16_t contextId;
    uint16_t threadId;
    RtExceptionArgsInfo exceptionArgs;
};

struct RtAicoreExDetailInfo {
    RtExceptionArgsInfo exceptionArgs;
};

enum class RtUbExType {
    DOORBELL,
    DIRECT_WQE
};

struct RtUbInfo {
    uint8_t functionId;
    uint8_t dieId;
    uint16_t jettyId;
    uint16_t piValue;  // directWqe类型下该字段无效
};

#define UB_DB_SEND_MAX_NUM (4)
#define RT_CCU_SQE_ARGS_LEN     (13U)
#define MAX_CCU_EXCEPTION_INFO_SIZE (128U)
#define FUSION_SUB_TASK_MAX_CCU_NUM (8U)

struct RtUbExDetailInfo {
    RtUbExType ubType;
    uint8_t ubNum;
    uint8_t resv[3];
    RtUbInfo info[RT_UB_DB_SEND_MAX_NUM];
};

struct RtCcuMissionDetailInfo {
    uint8_t dieId;
    uint8_t missionId;
    uint16_t instrId;
    uint64_t args[RT_CCU_SQE_ARGS_LEN];
    uint8_t status;
    uint8_t subStatus;
    uint8_t panicLog[MAX_CCU_EXCEPTION_INFO_SIZE];
};

struct RtMultiCCUExDetailInfo {
    uint16_t ccuMissionNum;
    RtCcuMissionDetailInfo missionInfo[FUSION_SUB_TASK_MAX_CCU_NUM];
};

enum RtFusionExType {
    RT_FUSION_AICORE_CCU,
    RT_FUSION_AICORE_AICPU
};

struct RtFusionAICoreCCUExDetailInfo {
    RtExceptionArgsInfo exceptionArgs;
    RtMultiCCUExDetailInfo ccuDetailMsg;
};

struct RtFusionExDetailInfo {
    RtFusionExType type;
    union {
        RtFusionAICoreCCUExDetailInfo aicoreCcuInfo;
    } u;
};

struct RtExceptionExpandInfo {
    RtExceptionExpandType type;
    union {
        RtFftsPlusExDetailInfo fftsPlusInfo;
        RtAicoreExDetailInfo aicoreInfo; // 关注下影响
        RtUbExDetailInfo ubInfo;
        RtMultiCCUExDetailInfo ccuInfo;       /* use for ccu task */
        RtFusionExDetailInfo fusionInfo;      /* use for fusion task */
    } u;
};

struct RtExceptionInfo {
    uint32_t taskid;
    uint32_t streamid;
    uint32_t tid;
    uint32_t deviceid;
    uint32_t retcode;
    RtExceptionExpandInfo expandInfo;
};

typedef struct RtExceptionInfo AclRtExceptionInfo;

typedef void (*AclRtExceptionInfoCallback)(AclRtExceptionInfo *exceptionInfo);

typedef void *AclMdlRI;

enum class AclMdlRICaptureStatus {
    NONE = 0,
    ACTIVE,
    INVALIDATED,
};

enum class AclMdlRICaptureMode {
    GLOBAL = 0,
    THREAD_LOCAL,
    RELAXED,
};
}
