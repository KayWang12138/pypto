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
 * \file msprof_define.h
 * \brief
 */

#pragma once

#ifdef BUILD_WITH_CANN
#include "profiling/aprof_pub.h"
#else
#include <cstdint>

namespace npu::tile_fwk {
#define CCECPU 25
#define MSPROF_REPORT_NODE_BASIC_INFO_TYPE       0U  /* type info: node_basic_info */
#define MSPROF_REPORT_NODE_TENSOR_INFO_TYPE      1U  /* type info: tensor_info */
#define MSPROF_REPORT_NODE_CONTEXT_ID_INFO_TYPE  4U  /* type info: context_id_info */
#define MSPROF_REPORT_NODE_LAUNCH_TYPE           5U  /* type info: launch */
#define MSPROF_REPORT_NODE_LEVEL        10000U
#define MSPROF_GE_TENSOR_DATA_SHAPE_LEN 8
#define MSPROF_GE_TENSOR_DATA_NUM 5
#define PROF_TASK_TIME_L1_MASK         0x00000002ULL

typedef void* VOID_PTR;

enum MsprofGeTensorType {
    MSPROF_GE_TENSOR_TYPE_INPUT = 0,
    MSPROF_GE_TENSOR_TYPE_OUTPUT,
};

enum MsprofGeTaskType {
    MSPROF_GE_TASK_TYPE_AI_CORE = 0,
    MSPROF_GE_TASK_TYPE_AI_CPU,
    MSPROF_GE_TASK_TYPE_AIV,
    MSPROF_GE_TASK_TYPE_WRITE_BACK,
    MSPROF_GE_TASK_TYPE_MIX_AIC,
    MSPROF_GE_TASK_TYPE_MIX_AIV,
    MSPROF_GE_TASK_TYPE_FFTS_PLUS,
    MSPROF_GE_TASK_TYPE_DSA,
    MSPROF_GE_TASK_TYPE_DVPP,
    MSPROF_GE_TASK_TYPE_HCCL,
    MSPROF_GE_TASK_TYPE_FUSION,
    MSPROF_GE_TASK_TYPE_INVALID
};

struct MsrofTensorData {
    uint32_t tensorType;
    uint32_t format;
    uint32_t dataType;
    uint32_t shape[MSPROF_GE_TENSOR_DATA_SHAPE_LEN];
};
struct MsprofTensorInfo {
    uint64_t opName;
    uint32_t tensorNum;
    struct MsrofTensorData tensorData[MSPROF_GE_TENSOR_DATA_NUM];
};

#define MSPROF_CTX_ID_MAX_NUM 55
struct MsprofContextIdInfo {
    uint64_t opName;
    uint32_t ctxIdNum;
    uint32_t ctxIds[MSPROF_CTX_ID_MAX_NUM];
};

#define MSPROF_REPORT_DATA_MAGIC_NUM 0x5A5AU

struct MsprofApi { // for MsprofReportApi
#ifdef __cplusplus
    uint16_t magicNumber = MSPROF_REPORT_DATA_MAGIC_NUM;
#else
    uint16_t magicNumber;
#endif
    uint16_t level;
    uint32_t type;
    uint32_t threadId;
    uint32_t reserve;
    uint64_t beginTime;
    uint64_t endTime;
    uint64_t itemId;
};

#define MSPROF_ADDTIONAL_INFO_DATA_LENGTH (232)
struct MsprofAdditionalInfo {  // for MsprofReportAdditionalInfo buffer data
#ifdef __cplusplus
    uint16_t magicNumber = MSPROF_REPORT_DATA_MAGIC_NUM;
#else
    uint16_t magicNumber;
#endif
    uint16_t level;
    uint32_t type;
    uint32_t threadId;
    uint32_t dataLen;
    uint64_t timeStamp;
    uint8_t  data[MSPROF_ADDTIONAL_INFO_DATA_LENGTH];
};

struct MsprofRuntimeTrack {  // for MsprofReportCompactInfo buffer data
    uint16_t deviceId;
    uint16_t streamId;
    uint32_t taskId;
    uint64_t taskType;       // task message hash id
    uint64_t kernelName;     // kernelname hash id
};

struct MsprofCaptureStreamInfo {  // for MsprofReportCompactInfo buffer data
    uint16_t captureStatus;     // Whether the mark is destroyed: 0 indicates normal, 1 indicates destroyed.
    uint16_t modelStreamId;     // capture stream id. Destroy the stream ID of the record, set it to UINT16_MAX.
    uint16_t originalStreamId;  // ori stream id. Destroy the stream ID of the record, set it to UINT16_MAX.
    uint16_t modelId;           // capture model id, independent of GE
    uint16_t deviceId;
};

struct MsprofNodeBasicInfo {
    uint64_t opName;
    uint32_t taskType;
    uint64_t opType;
    uint32_t blockDim;
    uint32_t opFlag;
};

struct MsprofHCCLOPInfo {  // for MsprofReportCompactInfo buffer data
    uint8_t relay : 1;     // Communication
    uint8_t retry : 1;     // Retransmission flag
    uint8_t dataType;      // Consistent with Type HcclDataType preservation
    uint64_t algType;      // The algorithm used by the communication operator, the hash key, whose value is a string separated by "-".
    uint64_t count;        // Number of data sent
    uint64_t groupName;    // group hash id
};

struct MsprofDpuTrack {  // for MsprofReportCompactInfo buffer data
    uint16_t deviceId;   // high 4 bits, devType: dpu: 1, low 12 bits device id
    uint16_t streamId;
    uint32_t taskId;
    uint32_t taskType;    // task type enum
    uint32_t res;
    uint64_t startTime;   // start time
};

struct MsprofStreamExpandSpecInfo {
    uint8_t expandStatus;
    uint8_t reserve1;
    uint16_t reserve2;
};

#define MSPROF_COMPACT_INFO_DATA_LENGTH 40
struct MsprofCompactInfo {  // for MsprofReportCompactInfo buffer data
#ifdef __cplusplus
    uint16_t magicNumber = MSPROF_REPORT_DATA_MAGIC_NUM;
#else
    uint16_t magicNumber;
#endif
    uint16_t level;
    uint32_t type;
    uint32_t threadId;
    uint32_t dataLen;
    uint64_t timeStamp;
    union {
        uint8_t info[MSPROF_COMPACT_INFO_DATA_LENGTH];
        struct MsprofRuntimeTrack runtimeTrack;
        struct MsprofCaptureStreamInfo captureStreamInfo;
        struct MsprofNodeBasicInfo nodeBasicInfo;
        struct MsprofHCCLOPInfo hcclopInfo;
        struct MsprofDpuTrack dpuTack;
        struct MsprofStreamExpandSpecInfo streamExpandInfo;
    } data;
};

typedef int32_t (*ProfCommandHandle)(uint32_t type, void *data, uint32_t len);

#define PATH_LEN_MAX 1023
#define PARAM_LEN_MAX 4095
#define MSPROF_MAX_DEV_NUM 64

struct MsprofCommandHandleParams {
    uint32_t pathLen;
    uint32_t storageLimit;  // MB
    uint32_t profDataLen;
    char path[PATH_LEN_MAX + 1];
    char profData[PARAM_LEN_MAX + 1];
};

/**
 * @brief profiling command info
 */
struct MsprofCommandHandle {
    uint64_t profSwitch;
    uint64_t profSwitchHi;
    uint32_t devNums;
    uint32_t devIdList[MSPROF_MAX_DEV_NUM];
    uint32_t modelId;
    uint32_t type;
    uint32_t cacheFlag;
    struct MsprofCommandHandleParams params;
};

enum MsprofCommandHandleType {
    PROF_COMMANDHANDLE_TYPE_INIT = 0,
    PROF_COMMANDHANDLE_TYPE_START,
    PROF_COMMANDHANDLE_TYPE_STOP,
    PROF_COMMANDHANDLE_TYPE_FINALIZE,
    PROF_COMMANDHANDLE_TYPE_MODEL_SUBSCRIBE,
    PROF_COMMANDHANDLE_TYPE_MODEL_UNSUBSCRIBE,
    PROF_COMMANDHANDLE_TYPE_MAX
};
}
#endif