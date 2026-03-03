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
 * \file runtime_define.h
 * \brief
 */

#pragma once

#ifdef BUILD_WITH_CANN
#include "runtime/base.h"
#include "runtime/mem.h"
#include "runtime/kernel.h"
#include "runtime/rts/rts_kernel.h"
#else
#include <cstdint>
#include <cstddef>

namespace npu::tile_fwk {
const int32_t RT_ERROR_NONE = 0; // success
#define RT_MEMORY_HBM (0x2U)       // HBM memory on device
#define RT_MEMORY_POLICY_HUGE_PAGE_FIRST (0x400U)    // Malloc mem prior huge page, then default page, 0x1U << 10U
#define RT_MEMORY_POLICY_HUGE1G_PAGE_ONLY (0x10000U)   // Malloc mem only use 1G huge page, 0x1U << 16U
#define RT_KERNEL_USE_SPECIAL_TIMEOUT             (0x100U)
#define RT_STREAM_PRIORITY_DEFAULT (0U)
#define RT_DEV_BINARY_MAGIC_ELF 0x43554245U

#ifndef char_t
typedef char char_t;
#endif
typedef int32_t rtError_t;
typedef uint32_t rtMemType_t;
typedef void *rtStream_t;
typedef void *rtModel_t;
typedef void *rtFuncHandle;
typedef void *rtBinHandle;

typedef enum tagRtMemcpyKind {
    RT_MEMCPY_HOST_TO_HOST = 0,  // host to host
    RT_MEMCPY_HOST_TO_DEVICE,    // host to device
    RT_MEMCPY_DEVICE_TO_HOST,    // device to host
    RT_MEMCPY_DEVICE_TO_DEVICE,  // device to device, 1P && P2P
    RT_MEMCPY_MANAGED,           // managed memory
    RT_MEMCPY_ADDR_DEVICE_TO_DEVICE,
    RT_MEMCPY_HOST_TO_DEVICE_EX, // host  to device ex (only used for 8 bytes)
    RT_MEMCPY_DEVICE_TO_HOST_EX, // device to host ex
    RT_MEMCPY_DEFAULT,           // auto infer copy dir
    RT_MEMCPY_RESERVED,
} rtMemcpyKind_t;

typedef struct tagRtDevBinary {
    uint32_t magic;    // magic number
    uint32_t version;  // version of binary
    const void *data;  // binary data
    uint64_t length;   // binary length
} rtDevBinary_t;

typedef struct rtHostInputInfo {
    uint32_t addrOffset;
    uint32_t dataOffset;
} rtHostInputInfo_t;

typedef struct tagRtArgsEx {
    void *args;                     // args host mem addr
    rtHostInputInfo_t *hostInputInfoPtr;     // nullptr means no host mem input
    uint32_t argsSize;              // input + output + tiling addr size + tiling data size + host mem
    uint32_t tilingAddrOffset;      // tiling addr offset
    uint32_t tilingDataOffset;      // tiling data offset
    uint16_t hostInputInfoNum;      // hostInputInfo num
    uint8_t hasTiling;              // if has tiling: 0 means no tiling
    uint8_t isNoNeedH2DCopy;        // is no need host to device copy: 0 means need H2D copy,
    // others means doesn't need H2D copy.
    uint8_t reserved[4];
} rtArgsEx_t;

typedef struct tagRtSmData {
    uint64_t L2_mirror_addr;          // preload or swap source addr
    uint32_t L2_data_section_size;    // every data size
    uint8_t L2_preload;               // 1 - preload from mirrorAddr, 0 - no preload
    uint8_t modified;                 // 1 - data will be modified by kernel, 0 - no modified
    uint8_t priority;                 // data priority
    int8_t prev_L2_page_offset_base;  // remap source section offset
    uint8_t L2_page_offset_base;      // remap destination section offset
    uint8_t L2_load_to_ddr;           // 1 - need load out, 0 - no need
    uint8_t reserved[2];              // reserved
} rtSmData_t;

typedef struct tagRtSmCtrl {
    rtSmData_t data[8];  // data description
    uint64_t size;       // max page Num
    uint8_t remap[64];   /* just using for static remap mode, default:0xFF
                          array index: virtual l2 page id, array value: physic l2 page id */
    uint8_t l2_in_main;  // 0-DDR, 1-L2, default:0xFF
    uint8_t reserved[3];
} rtSmDesc_t;

typedef struct tagRtTaskCfgInfo {
    uint8_t qos;
    uint8_t partId;
    uint8_t schemMode; // rtschemModeType_t 0:normal;1:batch;2:sync
    bool d2dCrossFlag; // d2dCrossFlag true:D2D_CROSS flase:D2D_INNER
    uint32_t blockDimOffset;
    uint8_t dumpflag; // dumpflag 0:fault 2:RT_KERNEL_DUMPFLAG 4:RT_FUSION_KERNEL_DUMPFLAG
    uint8_t neverTimeout; // 1: never timeout, 0: will timeout
    uint8_t rev[2];
    uint32_t localMemorySize;  // for simt ub_size
} rtTaskCfgInfo_t;

typedef struct tagRtAicpuArgsEx {
    void *args; // args host mem addr
    rtHostInputInfo_t *hostInputInfoPtr; // nullptr means no host mem input
    rtHostInputInfo_t *kernelOffsetInfoPtr; // KernelOffsetInfo, it is different for CCE Kernel and fwk kernel
    uint32_t argsSize;
    uint16_t hostInputInfoNum; // hostInputInfo num
    uint16_t kernelOffsetInfoNum; // KernelOffsetInfo num
    uint32_t soNameAddrOffset; // just for CCE Kernel, default value is 0xffff for FWK kernel
    uint32_t kernelNameAddrOffset; // just for CCE Kernel, default value is 0xffff for FWK kernel
    bool isNoNeedH2DCopy; // is no need host to device copy: 0 means need H2D copy,
    // other means doesn't need H2D copy.
    uint16_t timeout;  // timeout for aicpu exit
    uint8_t reserved;
} rtAicpuArgsEx_t;

typedef struct {
    rtAicpuArgsEx_t baseArgs;
    size_t cpuParamHeadOffset;
    uint32_t rsv[4];
} rtCpuKernelArgs_t;

typedef enum {
    RT_LAUNCH_KERNEL_ATTR_SCHEM_MODE = 1,
    RT_LAUNCH_KERNEL_ATTR_LOCAL_MEM_SIZE = 2, // DEPRECATED: Use RT_LAUNCH_KERNEL_ATTR_DYN_UBUF_SIZE
    RT_LAUNCH_KERNEL_ATTR_DYN_UBUF_SIZE = 2,
    // vector core使能使用
    RT_LAUNCH_KERNEL_ATTR_ENGINE_TYPE,
    // vector core使能使用
    RT_LAUNCH_KERNEL_ATTR_BLOCKDIM_OFFSET,
    RT_LAUNCH_KERNEL_ATTR_BLOCK_TASK_PREFETCH,
    RT_LAUNCH_KERNEL_ATTR_DATA_DUMP,
    RT_LAUNCH_KERNEL_ATTR_TIMEOUT,
    RT_LAUNCH_KERNEL_ATTR_TIMEOUT_US,
    RT_LAUNCH_KERNEL_ATTR_MAX
} rtLaunchKernelAttrId;

typedef enum {
    RT_ENGINE_TYPE_AIC = 0,
    RT_ENGINE_TYPE_AIV
} rtEngineType;

typedef struct {
    uint32_t timeoutLow;  // low  32bit
    uint32_t timeoutHigh; // high 32bit
} rtTimeoutUs;

typedef union {
    uint8_t schemMode;
    uint32_t localMemorySize; // DEPRECATED: Use dynUBufSize
    uint32_t dynUBufSize;
    rtEngineType engineType;
    uint32_t blockDimOffset;
    uint8_t isBlockTaskPrefetch;  // 任务下发时判断是否sqe后续需要刷新标记（tiling key依赖下沉场景）0:disable 1:enable
    uint8_t isDataDump; // 0:disable 1:enable
    uint16_t timeout;       // uint:s
    rtTimeoutUs timeoutUs;  // uint:us
    uint32_t rsv[4];
} rtLaunchKernelAttrVal_t;

typedef struct {
    rtLaunchKernelAttrId id;
    rtLaunchKernelAttrVal_t value;
} rtLaunchKernelAttr_t;

typedef struct {
    rtLaunchKernelAttr_t *attrs;
    size_t numAttrs;
} rtKernelLaunchCfg_t;

typedef enum {
    RT_LOAD_BINARY_OPT_LAZY_LOAD = 1,
    RT_LOAD_BINARY_OPT_MAGIC = 2,
    RT_LOAD_BINARY_OPT_CPU_KERNEL_MODE = 3,
    RT_LOAD_BINARY_OPT_MAX
} rtLoadBinaryOption;

typedef union {
    uint32_t isLazyLoad;
    uint32_t magic;
    int32_t cpuKernelMode; // 0 ：仅需要加载json，1 ：加载cpu so & json，2: LoadFromData
    uint32_t rsv[4];
} rtLoadBinaryOptionValue_t;

typedef struct {
    rtLoadBinaryOption optionId;
    rtLoadBinaryOptionValue_t value;
} rtLoadBinaryOption_t;

typedef struct {
    rtLoadBinaryOption_t *options;
    size_t numOpt;
} rtLoadBinaryConfig_t;

typedef enum rtKernelType {
    KERNEL_TYPE_CCE = 0,
    KERNEL_TYPE_FWK = 1,
    KERNEL_TYPE_AICPU = 2,
    KERNEL_TYPE_AICPU_CUSTOM = 4,
    KERNEL_TYPE_AICPU_KFC = 5,
    KERNEL_TYPE_CUSTOM_KFC = 6,
    KERNEL_TYPE_HWTS = 10,
    KERNEL_TYPE_RESERVED = 99,
} rtKernelType_t;

typedef enum {
    RT_PROF_CTRL_INVALID = 0,
    RT_PROF_CTRL_SWITCH,
    RT_PROF_CTRL_REPORTER,
    RT_PROF_CTRL_BUTT
} rtProfCtrlType_t;

typedef enum schemModeType {
    RT_SCHEM_MODE_NORMAL = 0,
    RT_SCHEM_MODE_BATCH,
    RT_SCHEM_MODE_END
} rtschemModeType_t;
}
#endif