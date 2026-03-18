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
#include <type_traits>
#include "interface/utils/common.h"

namespace npu::tile_fwk {

enum class MachineError : uint32_t {
    HOST = 0x70000U,           // Host侧错误
    SCHEDULE = 0x71000U,       // 调度链路
    CONTROL_FLOW = 0x72000U,   // 控制流执行
    WORKSPACE = 0x73000U,      // Workspace / Slab
    DUMP_DFX = 0x74000U,       // Dump / DFX / Profiling
    PROGRAM_ENCODE = 0x75000U, // 编解码与一致性
    TENSOR_META = 0x76000U,    // 张量元信息
    SERVER_KERNEL = 0x77000U,  // AICPU server / kernel
    THREAD_MACHINE = 0x78000U, // 线程/机器级
    DATA_STRUCTURE = 0x79000U, // 内部数据结构
    UNKNOWN = 0x7A000U,        // 未知/预留
};

// Host侧错误码 (0x70000)
enum class HostErr : uint32_t {
    // Host基础错误 (0x70001 - 0x7000F)
    RUNTIME_INIT_FAILED = ToUnderlying(MachineError::HOST) + 0x01U,
    DEVICE_COMM_FAILED = ToUnderlying(MachineError::HOST) + 0x02U,
    MEMORY_ALLOC_FAILED = ToUnderlying(MachineError::HOST) + 0x03U,
    // 新增错误码 (0x70010 - 0x7002F)
    BACKTRACE_INFO = ToUnderlying(MachineError::HOST) + 0x10U,
    BACKTRACE_FRAME = ToUnderlying(MachineError::HOST) + 0x11U,
    DEVICE_ID_GET_FAILED = ToUnderlying(MachineError::HOST) + 0x12U,
    DEVICE_ADDR_ALLOC_FAILED = ToUnderlying(MachineError::HOST) + 0x13U,
    INVALID_FREE_OPERATION = ToUnderlying(MachineError::HOST) + 0x14U,
    NULL_DEVICE_ADDRESS = ToUnderlying(MachineError::HOST) + 0x15U,
    NULL_POINTER_FREE = ToUnderlying(MachineError::HOST) + 0x16U,
    UNKNOWN_POINTER_FREE = ToUnderlying(MachineError::HOST) + 0x17U,
    MEMORY_CORRUPTION = ToUnderlying(MachineError::HOST) + 0x18U,
    SENTINEL_CHECK_FAILED = ToUnderlying(MachineError::HOST) + 0x19U,
    BASE_ADDR_NOT_FOUND = ToUnderlying(MachineError::HOST) + 0x1AU,
    DEVICE_ALLOC_FAILED = ToUnderlying(MachineError::HOST) + 0x1BU,
    DEPRECATED_FEATURE = ToUnderlying(MachineError::HOST) + 0x1CU,
    CAPTURE_INFO_FAILED = ToUnderlying(MachineError::HOST) + 0x1DU,
    UNSUPPORTED_CAPTURE_STATUS = ToUnderlying(MachineError::HOST) + 0x1EU,
    NULL_MODEL = ToUnderlying(MachineError::HOST) + 0x1FU,
    STREAM_ADD_FAILED = ToUnderlying(MachineError::HOST) + 0x20U,
    DUMP_FUNCTION_NOT_FOUND = ToUnderlying(MachineError::HOST) + 0x21U,
    DUMP_INIT_FAILED = ToUnderlying(MachineError::HOST) + 0x22U,
    DUMP_UNINIT_FAILED = ToUnderlying(MachineError::HOST) + 0x23U,
    JSON_CONSTRUCT_FAILED = ToUnderlying(MachineError::HOST) + 0x24U,
    EMPTY_JSON_PATH = ToUnderlying(MachineError::HOST) + 0x25U,
    JSON_LOAD_FAILED = ToUnderlying(MachineError::HOST) + 0x26U,
    FUNC_HANDLE_GET_FAILED = ToUnderlying(MachineError::HOST) + 0x27U,
    BUILTIN_BIN_HANDLE_FAILED = ToUnderlying(MachineError::HOST) + 0x28U,
    FUNC_NAME_INVALID = ToUnderlying(MachineError::HOST) + 0x29U,
    HAL_FUNCTION_NOT_FOUND = ToUnderlying(MachineError::HOST) + 0x2AU,
    MAP_REG_ADDR_FAILED = ToUnderlying(MachineError::HOST) + 0x2BU,
    RT_MALLOC_FAILED = ToUnderlying(MachineError::HOST) + 0x2CU,
    RT_MEMCPY_FAILED = ToUnderlying(MachineError::HOST) + 0x2DU,
    CACHE_DIR_CREATE_FAILED = ToUnderlying(MachineError::HOST) + 0x2EU,
    ENV_HOME_NOT_SET = ToUnderlying(MachineError::HOST) + 0x2FU,
    PRECOMPILE_FAILED = ToUnderlying(MachineError::HOST) + 0x30U,
    COMPILE_FAILED = ToUnderlying(MachineError::HOST) + 0x31U,
    GEN_OP_FAILED = ToUnderlying(MachineError::HOST) + 0x32U,
    OP_PRECOMPILE_FAILED = ToUnderlying(MachineError::HOST) + 0x33U,
    COMPILE_CMD_FAILED = ToUnderlying(MachineError::HOST) + 0x34U,
    CCEC_FAILED = ToUnderlying(MachineError::HOST) + 0x35U,
    FILE_OPEN_FAILED = ToUnderlying(MachineError::HOST) + 0x36U,
    LINK_CMD_FAILED = ToUnderlying(MachineError::HOST) + 0x37U,
    LINK_FAILED = ToUnderlying(MachineError::HOST) + 0x38U,
    NO_CCE_PATH = ToUnderlying(MachineError::HOST) + 0x39U,
    SRC_FILE_GEN_FAILED = ToUnderlying(MachineError::HOST) + 0x3AU,
    MACHINE_TASK_NULL = ToUnderlying(MachineError::HOST) + 0x3BU,
    CACHE_NOT_FOUND = ToUnderlying(MachineError::HOST) + 0x3CU,
    LOOP_OP_SIZE_EXCEEDED = ToUnderlying(MachineError::HOST) + 0x3DU,
    DIR_CREATE_FAILED = ToUnderlying(MachineError::HOST) + 0x3EU,
    DUPLICATE_FUNC_HASH = ToUnderlying(MachineError::HOST) + 0x3FU,
    DYNAMIC_COMPILE_FAILED = ToUnderlying(MachineError::HOST) + 0x40U,
    UNKNOWN = ToUnderlying(MachineError::HOST) + 0x99U
};

enum class SchedErr : uint32_t {
    PREFETCH_CHECK_FAILED = ToUnderlying(MachineError::SCHEDULE) + 0x01U,
    AIC_TASK_WAIT_TIMEOUT = ToUnderlying(MachineError::SCHEDULE) + 0x02U,
    AIV_TASK_WAIT_TIMEOUT = ToUnderlying(MachineError::SCHEDULE) + 0x03U,
    TAIL_TASK_WAIT_TIMEOUT = ToUnderlying(MachineError::SCHEDULE) + 0x04U,
    ALL_AICORE_SYNC_TIMEOUT = ToUnderlying(MachineError::SCHEDULE) + 0x05U,
    HANDSHAKE_TIMEOUT = ToUnderlying(MachineError::SCHEDULE) + 0x06U,
    READY_QUEUE_OVERFLOW = ToUnderlying(MachineError::SCHEDULE) + 0x07U,
    SIGNAL_QUEUE_OVERFLOW = ToUnderlying(MachineError::SCHEDULE) + 0x08U,
    QUEUE_DEQUEUE_WHEN_EMPTY = ToUnderlying(MachineError::SCHEDULE) + 0x09U,
    CORE_TASK_PROCESS_FAILED = ToUnderlying(MachineError::SCHEDULE) + 0x0AU,
    AICPU_TASK_SYNC_TIMEOUT = ToUnderlying(MachineError::SCHEDULE) + 0x0BU,
    EXCEPTION_RESET_TRIGGERED = ToUnderlying(MachineError::SCHEDULE) + 0x0CU,
    EXCEPTION_SIGNAL_RECEIVED = ToUnderlying(MachineError::SCHEDULE) + 0x0DU,
    THREAD_INIT_ARGS_INVALID = ToUnderlying(MachineError::SCHEDULE) + 0x0EU,
    THREAD_ALLOC_FAILED = ToUnderlying(MachineError::SCHEDULE) + 0x0FU,
    INVALID_RUN_MODE = ToUnderlying(MachineError::SCHEDULE) + 0x10U,
    // 新增错误码 (0x71020 - 0x7102F)
    DFX_TIMEOUT = ToUnderlying(MachineError::SCHEDULE) + 0x20U,
    TIMEOUT = ToUnderlying(MachineError::SCHEDULE) + 0x21U,
    INVALID_ITERATOR = ToUnderlying(MachineError::SCHEDULE) + 0x22U,
    INVALID_ADDRESS = ToUnderlying(MachineError::SCHEDULE) + 0x23U,
    INVALID_OUTCAST = ToUnderlying(MachineError::SCHEDULE) + 0x24U,
    INVALID_CACHE_KIND = ToUnderlying(MachineError::SCHEDULE) + 0x25U,
    INVALID_ARGS = ToUnderlying(MachineError::SCHEDULE) + 0x26U,
    UNKNOWN = ToUnderlying(MachineError::SCHEDULE) + 0x99U
};

enum class CtrlErr : uint32_t {
    CTRL_FLOW_EXEC_FAILED = ToUnderlying(MachineError::CONTROL_FLOW) + 0x01U,
    ROOT_ALLOC_CTX_NULL = ToUnderlying(MachineError::CONTROL_FLOW) + 0x02U,
    ROOT_STITCH_CTX_NULL = ToUnderlying(MachineError::CONTROL_FLOW) + 0x03U,
    SYNC_FLAG_WAIT_TIMEOUT = ToUnderlying(MachineError::CONTROL_FLOW) + 0x04U,
    DEVICE_TASK_BUILD_FAILED = ToUnderlying(MachineError::CONTROL_FLOW) + 0x05U,
    READY_QUEUE_INIT_FAILED = ToUnderlying(MachineError::CONTROL_FLOW) + 0x06U,
    DEP_DUMP_FAILED = ToUnderlying(MachineError::CONTROL_FLOW) + 0x07U,
    READY_QUEUE_DUMP_FAILED = ToUnderlying(MachineError::CONTROL_FLOW) + 0x08U,
    TASK_STATS_ABNORMAL = ToUnderlying(MachineError::CONTROL_FLOW) + 0x09U,
    ROOT_INCAST_NO_FROMSLOTLIST = ToUnderlying(MachineError::CONTROL_FLOW) + 0x0AU,
    ROOT_INCAST_EMPTY_ADDRESS = ToUnderlying(MachineError::CONTROL_FLOW) + 0x0BU,
    ASSEMBLE_SLOT_MISSING_ALLOC = ToUnderlying(MachineError::CONTROL_FLOW) + 0x0CU,
    INCAST_DESC_NOT_RTOUTCAST = ToUnderlying(MachineError::CONTROL_FLOW) + 0x0DU,
    INCAST_ITER_INVALID = ToUnderlying(MachineError::CONTROL_FLOW) + 0x0EU,
    OUTCAST_DESC_NOT_RTOUTCAST = ToUnderlying(MachineError::CONTROL_FLOW) + 0x0FU,
    OUTCAST_ITER_INVALID = ToUnderlying(MachineError::CONTROL_FLOW) + 0x10U,
    RUNTIME_WS_ZERO = ToUnderlying(MachineError::CONTROL_FLOW) + 0x11U,
    // 控制流缓存错误 (0x72020 - 0x7202F)
    CACHE_MALLOC_FAILED = ToUnderlying(MachineError::CONTROL_FLOW) + 0x20U,
    CACHE_MEMCPY_FAILED = ToUnderlying(MachineError::CONTROL_FLOW) + 0x21U,
    CACHE_ALLOC_FAILED = ToUnderlying(MachineError::CONTROL_FLOW) + 0x22U,
    CACHE_REGISTRATION_FAILED = ToUnderlying(MachineError::CONTROL_FLOW) + 0x23U,
    // 新增错误码 (0x72030 - 0x7203F)
    CACHE_SHOULD_NOT_RECORD = ToUnderlying(MachineError::CONTROL_FLOW) + 0x30U,
    CACHE_ACTIVATION_MISMATCH = ToUnderlying(MachineError::CONTROL_FLOW) + 0x31U,
    NULL_ARGS = ToUnderlying(MachineError::CONTROL_FLOW) + 0x32U,
    BACKUP_FAILED = ToUnderlying(MachineError::CONTROL_FLOW) + 0x33U,
    INIT_TASK_CTRL_FAILED = ToUnderlying(MachineError::CONTROL_FLOW) + 0x34U,
    INVALID_SYMBOL_HANDLER = ToUnderlying(MachineError::CONTROL_FLOW) + 0x35U,
    HANDLER_NOT_FOUND = ToUnderlying(MachineError::CONTROL_FLOW) + 0x36U,
    INVALID_CTX = ToUnderlying(MachineError::CONTROL_FLOW) + 0x37U,
    MEM_TYPE_INVALID = ToUnderlying(MachineError::CONTROL_FLOW) + 0x38U,
    GROUP_INDEX_INVALID = ToUnderlying(MachineError::CONTROL_FLOW) + 0x39U,
    WINSIZE_EXCEEDED = ToUnderlying(MachineError::CONTROL_FLOW) + 0x3AU,
    UNKNOWN = ToUnderlying(MachineError::CONTROL_FLOW) + 0x99U
};

enum class WsErr : uint32_t {
    SLAB_ADD_CACHE_FAILED = ToUnderlying(MachineError::WORKSPACE) + 0x01U,
    SLAB_STAGE_LIST_INCONSISTENT = ToUnderlying(MachineError::WORKSPACE) + 0x02U,
    SLAB_TYPE_INVALID = ToUnderlying(MachineError::WORKSPACE) + 0x03U,
    WORKSPACE_INIT_RESOURCE_ERROR = ToUnderlying(MachineError::WORKSPACE) + 0x04U,
    WORKSPACE_INIT_PARAM_INVALID = ToUnderlying(MachineError::WORKSPACE) + 0x05U,
    WS_TENSOR_ADDRESS_OUT_OF_RANGE = ToUnderlying(MachineError::WORKSPACE) + 0x06U,
    SLAB_CAPACITY_CALC_INVALID = ToUnderlying(MachineError::WORKSPACE) + 0x07U,
    WS_TENSOR_NOT_INSIDE_SEGMENT = ToUnderlying(MachineError::WORKSPACE) + 0x08U,
    WS_MEMORY_CROSS_BOUNDARY = ToUnderlying(MachineError::WORKSPACE) + 0x09U,
    WS_TENSOR_OUTSIDE_WORKSPACE = ToUnderlying(MachineError::WORKSPACE) + 0x0AU,
    SLAB_ALLOC_NULL = ToUnderlying(MachineError::WORKSPACE) + 0x0BU,
    METADATA_ALLOCATOR_NO_MEMORY = ToUnderlying(MachineError::WORKSPACE) + 0x0CU,
    // 内存池错误 (0x73010 - 0x7301F)
    MEMORY_POOL_ALLOC_FAILED = ToUnderlying(MachineError::WORKSPACE) + 0x10U,
    MEMORY_POOL_FREE_INVALID = ToUnderlying(MachineError::WORKSPACE) + 0x11U,
    MEMORY_POOL_CHECK_FAILED = ToUnderlying(MachineError::WORKSPACE) + 0x12U,
    MEMORY_POOL_BLOCK_ALLOC_FAILED = ToUnderlying(MachineError::WORKSPACE) + 0x13U,
    // Workspace配置错误 (0x73020 - 0x7302F)
    DYNAMIC_WORKSPACE_DEPRECATED = ToUnderlying(MachineError::WORKSPACE) + 0x20U,
    TRIPLE_STREAM_CONFIG_INVALID = ToUnderlying(MachineError::WORKSPACE) + 0x21U,
    // 新增错误码 (0x73030 - 0x7304F)
    ROOT_INNER_ALLOC_FAILED = ToUnderlying(MachineError::WORKSPACE) + 0x30U,
    OUTCAST_ALLOC_FAILED = ToUnderlying(MachineError::WORKSPACE) + 0x31U,
    INCAST_NO_FROMSLOTLIST = ToUnderlying(MachineError::WORKSPACE) + 0x32U,
    INCAST_EMPTY_ADDRESS = ToUnderlying(MachineError::WORKSPACE) + 0x33U,
    SLOTLIST_RTOUTCAST_INVALID = ToUnderlying(MachineError::WORKSPACE) + 0x34U,
    ASSEMBLE_SLOT_ALLOC_FAILED = ToUnderlying(MachineError::WORKSPACE) + 0x35U,
    DUMP_TENSOR_ALLOC_FAILED = ToUnderlying(MachineError::WORKSPACE) + 0x36U,
    RTOUTCAST_ITER_INVALID = ToUnderlying(MachineError::WORKSPACE) + 0x37U,
    RTOUTCAST_REFCNT_INVALID = ToUnderlying(MachineError::WORKSPACE) + 0x38U,
    METADATA_FREE_MEMORY_ZERO = ToUnderlying(MachineError::WORKSPACE) + 0x39U,
    SLAB_CAPACITY_NULL = ToUnderlying(MachineError::WORKSPACE) + 0x3AU,
    SLAB_TYPE_NUM_EXCEEDED = ToUnderlying(MachineError::WORKSPACE) + 0x3BU,
    SLAB_ALLOC_NULL_DETAIL = ToUnderlying(MachineError::WORKSPACE) + 0x3CU,
    WS_MEM_CATEGORY_INVALID = ToUnderlying(MachineError::WORKSPACE) + 0x3DU,
    TENSOR_ADDRESS_INVALID = ToUnderlying(MachineError::WORKSPACE) + 0x3EU,
    LEAST_SLAB_REQ_MEM_EXCEEDED = ToUnderlying(MachineError::WORKSPACE) + 0x3FU,
    MEMBASE_NULL = ToUnderlying(MachineError::WORKSPACE) + 0x40U,
    SLAB_SIZE_TOO_SMALL = ToUnderlying(MachineError::WORKSPACE) + 0x41U,
    UNKNOWN = ToUnderlying(MachineError::WORKSPACE) + 0x99U
};

enum class DumpDfxErr : uint32_t {
    DUMP_MEMCPY_FAILED = ToUnderlying(MachineError::DUMP_DFX) + 0x01U,
    DUMP_TENSOR_INFO_FAILED = ToUnderlying(MachineError::DUMP_DFX) + 0x02U,
    DUMP_TENSOR_DATA_FAILED = ToUnderlying(MachineError::DUMP_DFX) + 0x03U,
    METRIC_ALLOC_OR_WAIT_TIMEOUT = ToUnderlying(MachineError::DUMP_DFX) + 0x04U,
    PERF_TRACE_FORMAT_ERROR = ToUnderlying(MachineError::DUMP_DFX) + 0x05U,
    PERF_TRACE_DUMP_ERROR = ToUnderlying(MachineError::DUMP_DFX) + 0x06U,
    DFX_AICPU_TIMEOUT = ToUnderlying(MachineError::DUMP_DFX) + 0x07U,
    // Dump/DFX服务器错误 (0x74010 - 0x7401F)
    SERVER_INIT_FAILED = ToUnderlying(MachineError::DUMP_DFX) + 0x10U,
    SERVER_UNINIT_FAILED = ToUnderlying(MachineError::DUMP_DFX) + 0x11U,
    SERVER_FUNC_NOT_FOUND = ToUnderlying(MachineError::DUMP_DFX) + 0x12U,
    INFO_ALLOC_FAILED = ToUnderlying(MachineError::DUMP_DFX) + 0x13U,
    DUMP_FUNCTIONS_NOT_FOUND = ToUnderlying(MachineError::DUMP_DFX) + 0x14U,
    AICORE_PROFILING = ToUnderlying(MachineError::DUMP_DFX) + 0x15U,
    PERF_TRACE_START = ToUnderlying(MachineError::DUMP_DFX) + 0x16U,
    PERF_TRACE_END = ToUnderlying(MachineError::DUMP_DFX) + 0x17U,
    PERF_MESSAGE = ToUnderlying(MachineError::DUMP_DFX) + 0x18U,
    DUMP_HEADER = ToUnderlying(MachineError::DUMP_DFX) + 0x19U,
    DUMP_START = ToUnderlying(MachineError::DUMP_DFX) + 0x1AU,
    DUMP_DATA = ToUnderlying(MachineError::DUMP_DFX) + 0x1BU,
    DUMP_END = ToUnderlying(MachineError::DUMP_DFX) + 0x1CU,
    DFX_TIMEOUT = ToUnderlying(MachineError::DUMP_DFX) + 0x1DU,
    DFX_FORCE_EXIT = ToUnderlying(MachineError::DUMP_DFX) + 0x1EU,
    UNKNOWN = ToUnderlying(MachineError::DUMP_DFX) + 0x99U
};

enum class ProgEncodeErr : uint32_t {
    DYNFUNC_DATA_ALIGNMENT_ERROR = ToUnderlying(MachineError::PROGRAM_ENCODE) + 0x01U,
    FUNC_OP_SIZE_MISMATCH = ToUnderlying(MachineError::PROGRAM_ENCODE) + 0x02U,
    STITCH_PRED_SUCC_MISMATCH = ToUnderlying(MachineError::PROGRAM_ENCODE) + 0x03U,
    STITCH_LIST_TOO_LARGE = ToUnderlying(MachineError::PROGRAM_ENCODE) + 0x04U,
    STITCH_HANDLE_INDEX_OUT_OF_RANGE = ToUnderlying(MachineError::PROGRAM_ENCODE) + 0x05U,
    CELL_MATCH_PARAM_INVALID = ToUnderlying(MachineError::PROGRAM_ENCODE) + 0x06U,
    PROGRAM_RANGE_VERIFY_FAILED = ToUnderlying(MachineError::PROGRAM_ENCODE) + 0x07U,
    CACHE_RELOC_KIND_INVALID = ToUnderlying(MachineError::PROGRAM_ENCODE) + 0x08U,
    // 编解码错误 (0x75010 - 0x7501F)
    ADDR_OFFSET_INVALID = ToUnderlying(MachineError::PROGRAM_ENCODE) + 0x10U,
    CALLOP_SIZE_EXCEEDED = ToUnderlying(MachineError::PROGRAM_ENCODE) + 0x11U,
    CELLMATCH_DIM_ZERO = ToUnderlying(MachineError::PROGRAM_ENCODE) + 0x12U,
    OUTCAST_MEMORY_EXCEEDED = ToUnderlying(MachineError::PROGRAM_ENCODE) + 0x13U,
    LEAF_ATTR_NULL = ToUnderlying(MachineError::PROGRAM_ENCODE) + 0x14U,
    OPERATION_SIZE_MISMATCH = ToUnderlying(MachineError::PROGRAM_ENCODE) + 0x15U,
    MIX_TASK_OP_WRAP_BACKUP_FAILED = ToUnderlying(MachineError::PROGRAM_ENCODE) + 0x16U,
    DATA_ALIGNMENT_ERROR = ToUnderlying(MachineError::PROGRAM_ENCODE) + 0x17U,
    RANGE_VERIFY_FAILED = ToUnderlying(MachineError::PROGRAM_ENCODE) + 0x18U,
    RANGES_OVERLAP = ToUnderlying(MachineError::PROGRAM_ENCODE) + 0x19U,
    INVALID_RANGE = ToUnderlying(MachineError::PROGRAM_ENCODE) + 0x1AU,
    LAST_RANGE_END_MISMATCH = ToUnderlying(MachineError::PROGRAM_ENCODE) + 0x1BU,
    UNKNOWN = ToUnderlying(MachineError::PROGRAM_ENCODE) + 0x99U
};

enum class TensorMetaErr : uint32_t {
    TENSOR_DIM_COUNT_EXCEEDED = ToUnderlying(MachineError::TENSOR_META) + 0x01U,
    TENSOR_ENCODE_PTR_MISMATCH = ToUnderlying(MachineError::TENSOR_META) + 0x02U,
    RAW_TENSOR_INDEX_OUT_OF_RANGE = ToUnderlying(MachineError::TENSOR_META) + 0x03U,
    SHAPE_VALUE_MISMATCH = ToUnderlying(MachineError::TENSOR_META) + 0x04U,
    TENSOR_DUMP_INFO_INCONSISTENT = ToUnderlying(MachineError::TENSOR_META) + 0x05U,
    // 新增错误码 (0x76010 - 0x7602F)
    DIMENSION_COUNT_EXCEEDED = ToUnderlying(MachineError::TENSOR_META) + 0x10U,
    POINTER_MISMATCH = ToUnderlying(MachineError::TENSOR_META) + 0x11U,
    FUNC_COUNT_INFO = ToUnderlying(MachineError::TENSOR_META) + 0x12U,
    TASK_COUNT_INFO = ToUnderlying(MachineError::TENSOR_META) + 0x13U,
    DATA_SIZE_INFO = ToUnderlying(MachineError::TENSOR_META) + 0x14U,
    CORE_FUNC_COUNT = ToUnderlying(MachineError::TENSOR_META) + 0x15U,
    QUEUE_INFO = ToUnderlying(MachineError::TENSOR_META) + 0x16U,
    QUEUE_DETAIL = ToUnderlying(MachineError::TENSOR_META) + 0x17U,
    READY_COUNT = ToUnderlying(MachineError::TENSOR_META) + 0x18U,
    WORKSPACE_ADDR = ToUnderlying(MachineError::TENSOR_META) + 0x19U,
    INPUT_TENSOR_ADDR = ToUnderlying(MachineError::TENSOR_META) + 0x1AU,
    OUTPUT_TENSOR_ADDR = ToUnderlying(MachineError::TENSOR_META) + 0x1BU,
    PRED_INFO = ToUnderlying(MachineError::TENSOR_META) + 0x1CU,
    SUCC_INFO = ToUnderlying(MachineError::TENSOR_META) + 0x1DU,
    EXPR_INFO = ToUnderlying(MachineError::TENSOR_META) + 0x1EU,
    INCAST_INFO = ToUnderlying(MachineError::TENSOR_META) + 0x1FU,
    OUTCAST_INFO = ToUnderlying(MachineError::TENSOR_META) + 0x20U,
    WORKSPACE_INFO = ToUnderlying(MachineError::TENSOR_META) + 0x21U,
    OUTCAST_WORKSPACE_INFO = ToUnderlying(MachineError::TENSOR_META) + 0x22U,
    OP_ATTR_LIST_INFO = ToUnderlying(MachineError::TENSOR_META) + 0x23U,
    OP_ATTR_OFFSET_LIST_INFO = ToUnderlying(MachineError::TENSOR_META) + 0x24U,
    EXPR_TBL_INFO = ToUnderlying(MachineError::TENSOR_META) + 0x25U,
    RAW_TENSOR_DESC_INFO = ToUnderlying(MachineError::TENSOR_META) + 0x26U,
    DUMP_TENSOR_INFO = ToUnderlying(MachineError::TENSOR_META) + 0x27U,
    CCE_BINARY_ALIGNMENT_ERROR = ToUnderlying(MachineError::TENSOR_META) + 0x28U,
    OP_ATTRS_ALIGNMENT_ERROR = ToUnderlying(MachineError::TENSOR_META) + 0x29U,
    RAW_TENSOR_ADDR_ALIGNMENT_ERROR = ToUnderlying(MachineError::TENSOR_META) + 0x2AU,
    UNKNOWN = ToUnderlying(MachineError::TENSOR_META) + 0x99U
};

enum class ServerKernelErr : uint32_t {
    DYN_SERVER_ARGS_NULL = ToUnderlying(MachineError::SERVER_KERNEL) + 0x01U,
    DYN_SERVER_SAVE_SO_FAILED = ToUnderlying(MachineError::SERVER_KERNEL) + 0x02U,
    KERNEL_EXEC_FUNC_FAILED = ToUnderlying(MachineError::SERVER_KERNEL) + 0x03U,
    KERNEL_SO_OR_FUNC_LOAD_FAILED = ToUnderlying(MachineError::SERVER_KERNEL) + 0x04U,
    DYN_SERVER_RUN_FAILED = ToUnderlying(MachineError::SERVER_KERNEL) + 0x05U,
    DYN_SERVER_INIT_FAILED = ToUnderlying(MachineError::SERVER_KERNEL) + 0x06U,
    // AICPU算子加载错误 (0x77010 - 0x7701F)
    OP_JSON_PARSE_FAILED = ToUnderlying(MachineError::SERVER_KERNEL) + 0x10U,
    OP_JSON_PATH_EMPTY = ToUnderlying(MachineError::SERVER_KERNEL) + 0x11U,
    OP_LOAD_FAILED = ToUnderlying(MachineError::SERVER_KERNEL) + 0x12U,
    OP_FUNC_HANDLE_GET_FAILED = ToUnderlying(MachineError::SERVER_KERNEL) + 0x13U,
    OP_BUILTIN_BIN_HANDLE_FAILED = ToUnderlying(MachineError::SERVER_KERNEL) + 0x14U,
    OP_FUNC_NAME_INVALID = ToUnderlying(MachineError::SERVER_KERNEL) + 0x15U,
    UNKNOWN = ToUnderlying(MachineError::SERVER_KERNEL) + 0x99U
};

enum class ThreadErr : uint32_t {
    DEVICE_ARGS_INVALID = ToUnderlying(MachineError::THREAD_MACHINE) + 0x01U,
    SIGNAL_HANDLER_ABNORMAL = ToUnderlying(MachineError::THREAD_MACHINE) + 0x02U,
    RESET_REG_ALL_TRIGGERED = ToUnderlying(MachineError::THREAD_MACHINE) + 0x03U,
    // Runtime运行时错误 (0x78010 - 0x7801F)
    DEVICE_INIT_FAILED = ToUnderlying(MachineError::THREAD_MACHINE) + 0x10U,
    DEVICE_ID_GET_FAILED = ToUnderlying(MachineError::THREAD_MACHINE) + 0x11U,
    DEVADDR_ALLOC_FAILED = ToUnderlying(MachineError::THREAD_MACHINE) + 0x12U,
    STREAM_CAPTURE_FAILED = ToUnderlying(MachineError::THREAD_MACHINE) + 0x13U,
    STREAM_CAPTURE_STATUS_INVALID = ToUnderlying(MachineError::THREAD_MACHINE) + 0x14U,
    KERNEL_REGISTER_FAILED = ToUnderlying(MachineError::THREAD_MACHINE) + 0x15U,
    KERNEL_UNREGISTER_FAILED = ToUnderlying(MachineError::THREAD_MACHINE) + 0x16U,
    MODEL_OR_STREAM_FAILED = ToUnderlying(MachineError::THREAD_MACHINE) + 0x17U,
    LAUNCH_FAILED = ToUnderlying(MachineError::THREAD_MACHINE) + 0x18U,
    SYNC_FAILED = ToUnderlying(MachineError::THREAD_MACHINE) + 0x19U,
    // 设备运行器错误 (0x78020 - 0x7803F)
    MEMSET_FAILED = ToUnderlying(MachineError::THREAD_MACHINE) + 0x20U,
    DFX_ALLOC_FAILED = ToUnderlying(MachineError::THREAD_MACHINE) + 0x21U,
    SO_READ_FAILED = ToUnderlying(MachineError::THREAD_MACHINE) + 0x22U,
    AICPU_SERVER_INIT_FAILED = ToUnderlying(MachineError::THREAD_MACHINE) + 0x23U,
    EVENT_CREATE_FAILED = ToUnderlying(MachineError::THREAD_MACHINE) + 0x24U,
    EVENT_RECORD_FAILED = ToUnderlying(MachineError::THREAD_MACHINE) + 0x25U,
    STREAM_WAIT_EVENT_FAILED = ToUnderlying(MachineError::THREAD_MACHINE) + 0x26U,
    AICPU_LAUNCH_FAILED = ToUnderlying(MachineError::THREAD_MACHINE) + 0x27U,
    AICORE_LAUNCH_FAILED = ToUnderlying(MachineError::THREAD_MACHINE) + 0x28U,
    PREPARE_FAILED = ToUnderlying(MachineError::THREAD_MACHINE) + 0x29U,
    BUILTIN_LAUNCH_FAILED = ToUnderlying(MachineError::THREAD_MACHINE) + 0x2AU,
    RUNNER_KERNEL_REGISTER_FAILED = ToUnderlying(MachineError::THREAD_MACHINE) + 0x2BU,
    BUILTIN_HANDLE_GET_FAILED = ToUnderlying(MachineError::THREAD_MACHINE) + 0x2CU,
    PREPARE_ARGS_FAILED = ToUnderlying(MachineError::THREAD_MACHINE) + 0x2DU,
    UNKNOWN = ToUnderlying(MachineError::THREAD_MACHINE) + 0x99U
};

enum class DataStructErr : uint32_t {
    DEV_RELOC_VECTOR_INDEX_OOB = ToUnderlying(MachineError::DATA_STRUCTURE) + 0x01U,
    SMALL_ARRAY_RESIZE_OOB = ToUnderlying(MachineError::DATA_STRUCTURE) + 0x02U,
    VECTOR_ALREADY_INITIALIZED = ToUnderlying(MachineError::DATA_STRUCTURE) + 0x03U,
    VECTOR_UNEXPECTED_EXPAND = ToUnderlying(MachineError::DATA_STRUCTURE) + 0x04U,
    VECTOR_INVALID_POP_SIZE = ToUnderlying(MachineError::DATA_STRUCTURE) + 0x05U,
    REF_COUNT_INVALID = ToUnderlying(MachineError::DATA_STRUCTURE) + 0x06U,
    ITEMPOOL_ALREADY_INITIALIZED = ToUnderlying(MachineError::DATA_STRUCTURE) + 0x07U,
    ITEMPOOL_NO_AVAILABLE_ITEMS = ToUnderlying(MachineError::DATA_STRUCTURE) + 0x08U,
    ITEMPOOL_DOUBLE_FREE = ToUnderlying(MachineError::DATA_STRUCTURE) + 0x09U,
    RTOUTCAST_ITER_EXCEEDS_MAX = ToUnderlying(MachineError::DATA_STRUCTURE) + 0x0AU,
    ADDRESS_DESC_TYPE_MISMATCH = ToUnderlying(MachineError::DATA_STRUCTURE) + 0x0BU,
    // 新增错误码 (0x79010 - 0x7902F)
    INDEX_OUT_OF_BOUNDS = ToUnderlying(MachineError::DATA_STRUCTURE) + 0x10U,
    SIZE_EXCEEDS_LIMIT = ToUnderlying(MachineError::DATA_STRUCTURE) + 0x11U,
    COLUMN_COUNT_MISMATCH = ToUnderlying(MachineError::DATA_STRUCTURE) + 0x12U,
    COLUMN_INDEX_OUT_OF_BOUNDS = ToUnderlying(MachineError::DATA_STRUCTURE) + 0x13U,
    POOL_EMPTY = ToUnderlying(MachineError::DATA_STRUCTURE) + 0x14U,
    INVALID_FREE_INDEX = ToUnderlying(MachineError::DATA_STRUCTURE) + 0x15U,
    ALREADY_INITIALIZED = ToUnderlying(MachineError::DATA_STRUCTURE) + 0x16U,
    EMPTY_CONTAINER_ACCESS = ToUnderlying(MachineError::DATA_STRUCTURE) + 0x17U,
    CAPACITY_INSUFFICIENT = ToUnderlying(MachineError::DATA_STRUCTURE) + 0x18U,
    NOT_INITIALIZED = ToUnderlying(MachineError::DATA_STRUCTURE) + 0x19U,
    RESIZE_TOO_LARGE = ToUnderlying(MachineError::DATA_STRUCTURE) + 0x1AU,
    ALLOCATOR_NOT_INITIALIZED = ToUnderlying(MachineError::DATA_STRUCTURE) + 0x1BU,
    POINTER_OUT_OF_RANGE = ToUnderlying(MachineError::DATA_STRUCTURE) + 0x1CU,
    INVALID_CACHE_INDEX = ToUnderlying(MachineError::DATA_STRUCTURE) + 0x1DU,
    ALLOCATION_FAILED = ToUnderlying(MachineError::DATA_STRUCTURE) + 0x1EU,
    GROUP_INDEX_OUT_OF_BOUNDS = ToUnderlying(MachineError::DATA_STRUCTURE) + 0x1FU,
    ARRAY_SIZE_EXCEEDED = ToUnderlying(MachineError::DATA_STRUCTURE) + 0x20U,
    QUEUE_FULL = ToUnderlying(MachineError::DATA_STRUCTURE) + 0x21U,
    QUEUE_EMPTY = ToUnderlying(MachineError::DATA_STRUCTURE) + 0x22U,
    TASK_NOT_FOUND = ToUnderlying(MachineError::DATA_STRUCTURE) + 0x23U,
    COUNT_MISMATCH = ToUnderlying(MachineError::DATA_STRUCTURE) + 0x24U,
    SLOT_INDEX_INVALID = ToUnderlying(MachineError::DATA_STRUCTURE) + 0x25U,
    QUEUE_OVERFLOW = ToUnderlying(MachineError::DATA_STRUCTURE) + 0x26U,
    DIMENSION_ZERO = ToUnderlying(MachineError::DATA_STRUCTURE) + 0x27U,
    DIMENSION_EXCEEDS_LIMIT = ToUnderlying(MachineError::DATA_STRUCTURE) + 0x28U,
    SIZE_MISMATCH = ToUnderlying(MachineError::DATA_STRUCTURE) + 0x29U,
    INVALID_STITCH_INDEX = ToUnderlying(MachineError::DATA_STRUCTURE) + 0x2AU,
    DIMENSION_MISMATCH = ToUnderlying(MachineError::DATA_STRUCTURE) + 0x2BU,
    SHAPE_MISMATCH = ToUnderlying(MachineError::DATA_STRUCTURE) + 0x2CU,
    ADDRESS_MISMATCH = ToUnderlying(MachineError::DATA_STRUCTURE) + 0x2DU,
    POINTER_MISMATCH = ToUnderlying(MachineError::DATA_STRUCTURE) + 0x2EU,
    NULL_POINTER = ToUnderlying(MachineError::DATA_STRUCTURE) + 0x2FU,
    UNKNOWN = ToUnderlying(MachineError::DATA_STRUCTURE) + 0x99U
};

enum class MachineFunctionErr : uint32_t { RESERVED = 0x87000U };

enum class MachinePassErr : uint32_t { RESERVED = 0x87500U };

enum class MachineCodegenErr : uint32_t { RESERVED = 0x88000U };

enum class MachineSimulationErr : uint32_t { RESERVED = 0x88500U };

enum class MachineDistributedErr : uint32_t { RESERVED = 0x89000U };

enum class MachineOperationErr : uint32_t { RESERVED = 0x89500U };

} // namespace npu::tile_fwk
