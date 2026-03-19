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
 * \file machine_error.h
 * \brief
 */

#pragma once

#include <cstdint>

namespace npu::tile_fwk {
enum class MachineError : uint32_t {
    SCHEDULE       = 0x70000U, // 调度链路
    CONTROL_FLOW   = 0x71000U, // 控制流执行
    WORKSPACE      = 0x72000U, // Workspace / Slab
    DUMP_DFX       = 0x73000U, // Dump / DFX / Profiling
    PROGRAM_ENCODE = 0x74000U, // 编解码与一致性
    TENSOR_META    = 0x75000U, // 张量元信息
    SERVER_KERNEL  = 0x76000U, // AICPU server / kernel
    THREAD_MACHINE = 0x77000U, // 线程/机器级
    DATA_STRUCTURE = 0x78000U, // 内部数据结构
    UNKNOWN        = 0x79000U, // 未知/预留
};

enum class SchedErr : uint32_t {
    PREFETCH_CHECK_FAILED     = 0x70001U,
    AIC_TASK_WAIT_TIMEOUT     = 0x70002U,
    AIV_TASK_WAIT_TIMEOUT     = 0x70003U,
    TAIL_TASK_WAIT_TIMEOUT    = 0x70004U,
    ALL_AICORE_SYNC_TIMEOUT   = 0x70005U,
    HANDSHAKE_TIMEOUT         = 0x70006U,
    READY_QUEUE_OVERFLOW      = 0x70007U,
    SIGNAL_QUEUE_OVERFLOW     = 0x70008U,
    QUEUE_DEQUEUE_WHEN_EMPTY  = 0x70009U,
    CORE_TASK_PROCESS_FAILED  = 0x70010U,
    AICPU_TASK_SYNC_TIMEOUT   = 0x70011U,
    EXCEPTION_RESET_TRIGGERED = 0x70012U,
    EXCEPTION_SIGNAL_RECEIVED = 0x70013U,
    THREAD_INIT_ARGS_INVALID  = 0x70014U,
    UNKNOWN                   = 0x70099U
};

enum class CtrlErr : uint32_t {
    CTRL_FLOW_EXEC_FAILED     = 0x71001U,
    ROOT_ALLOC_CTX_NULL       = 0x71002U,
    ROOT_STITCH_CTX_NULL      = 0x71003U,
    SYNC_FLAG_WAIT_TIMEOUT    = 0x71004U,
    DEVICE_TASK_BUILD_FAILED  = 0x71005U,
    READY_QUEUE_INIT_FAILED   = 0x71006U,
    DEP_DUMP_FAILED           = 0x71007U,
    READY_QUEUE_DUMP_FAILED   = 0x71008U,
    TASK_STATS_ABNORMAL       = 0x71009U,
    UNKNOWN                   = 0x71099U
};

enum class WsErr : uint32_t {
    SLAB_ADD_CACHE_FAILED       = 0x72001U,
    SLAB_STAGE_LIST_INCONSISTENT = 0x72002U,
    SLAB_TYPE_INVALID           = 0x72003U,
    WORKSPACE_INIT_RESOURCE_ERROR = 0x72004U,
    WORKSPACE_INIT_PARAM_INVALID  = 0x72005U,
    WS_TENSOR_ADDRESS_OUT_OF_RANGE = 0x72006U,
    SLAB_CAPACITY_CALC_INVALID  = 0x72007U,
    UNKNOWN                     = 0x72099U
};

enum class DumpDfxErr : uint32_t {
    DUMP_MEMCPY_FAILED            = 0x73001U,
    DUMP_TENSOR_INFO_FAILED       = 0x73002U,
    DUMP_TENSOR_DATA_FAILED       = 0x73003U,
    METRIC_ALLOC_OR_WAIT_TIMEOUT  = 0x73004U,
    PERF_TRACE_FORMAT_ERROR       = 0x73005U,
    PERF_TRACE_DUMP_ERROR         = 0x73006U,
    DFX_AICPU_TIMEOUT            = 0x73007U,
    UNKNOWN                       = 0x73099U
};

enum class ProgEncodeErr : uint32_t {
    DYNFUNC_DATA_ALIGNMENT_ERROR = 0x74001U,
    FUNC_OP_SIZE_MISMATCH        = 0x74002U,
    STITCH_PRED_SUCC_MISMATCH     = 0x74003U,
    STITCH_LIST_TOO_LARGE         = 0x74004U,
    STITCH_HANDLE_INDEX_OUT_OF_RANGE = 0x74005U,
    CELL_MATCH_PARAM_INVALID     = 0x74006U,
    PROGRAM_RANGE_VERIFY_FAILED   = 0x74007U,
    CACHE_RELOC_KIND_INVALID      = 0x74008U,
    UNKNOWN                       = 0x74099U
};

enum class TensorMetaErr : uint32_t {
    TENSOR_DIM_COUNT_EXCEEDED   = 0x75001U,
    TENSOR_ENCODE_PTR_MISMATCH  = 0x75002U,
    RAW_TENSOR_INDEX_OUT_OF_RANGE = 0x75003U,
    SHAPE_VALUE_MISMATCH        = 0x75004U,
    TENSOR_DUMP_INFO_INCONSISTENT = 0x75005U,
    UNKNOWN                      = 0x75099U
};

enum class ServerKernelErr : uint32_t {
    DYN_SERVER_ARGS_NULL      = 0x76001U,
    DYN_SERVER_SAVE_SO_FAILED  = 0x76002U,
    KERNEL_EXEC_FUNC_FAILED    = 0x76003U,
    KERNEL_SO_OR_FUNC_LOAD_FAILED = 0x76004U,
    DYN_SERVER_RUN_FAILED      = 0x76005U,
    DYN_SERVER_INIT_FAILED     = 0x76006U,
    UNKNOWN                    = 0x76099U
};

enum class ThreadErr : uint32_t {
    DEVICE_ARGS_INVALID     = 0x77001U,
    SIGNAL_HANDLER_ABNORMAL = 0x77002U,
    RESET_REG_ALL_TRIGGERED = 0x77003U,
    UNKNOWN                 = 0x77099U
};

enum class DataStructErr : uint32_t {
    DEV_RELOC_VECTOR_INDEX_OOB = 0x78001U,
    SMALL_ARRAY_RESIZE_OOB     = 0x78002U,
    UNKNOWN                    = 0x78099U
};

enum class MachineFunctionErr : uint32_t {
    RESERVED = 0x87000U
};
enum class MachinePassErr : uint32_t {
    RESERVED = 0x87500U
};

enum class MachineCodegenErr : uint32_t {
    RESERVED = 0x88000U
};

enum class MachineSimulationErr : uint32_t {
    RESERVED = 0x88500U
};

enum class MachineDistributedErr : uint32_t {
    RESERVED = 0x89000U
};

enum class MachineOperationErr : uint32_t {
    RESERVED = 0x89500U 
};

}  // namespace npu::tile_fwk

