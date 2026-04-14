/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for the details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file error_code.h
 * \brief PyPTO 各组件错误码枚举统一定义（原分散于各 `*_error.h`）。
 *
 * 分段与 `docs/trouble_shooting/README.md` 总体范围表一致：
 *   F1XXXX 外部限制（无独立枚举段）；F2–F3 FUNCTION；F4–F5 PASS；F6 CODEGEN；
 *   F7–F8 MACHINE；F9 SIMULATION；FA DISTRIBUTED；FB VERIFY；FC OPERATION
 *   （FC0–FC2 VECTOR，FC3–FC5 MATMUL，FC6–FC8 CONV）；Calculator 使用 `npu::tile_fwk::calc_error`。
 */

#pragma once

#include <cstdint>
#include <type_traits>

namespace npu::tile_fwk {

// -----------------------------------------------------------------------------
// 枚举底层值（避免依赖 interface/utils/common.h 中的 ToUnderlying，防止与 common 的包含环）
// -----------------------------------------------------------------------------
template <typename E, typename = std::enable_if_t<std::is_enum_v<E>>>
constexpr std::underlying_type_t<E> err_u(E e) noexcept
{
    return static_cast<std::underlying_type_t<E>>(e);
}

// =============================================================================
// F2XXXX - F3XXXX：FUNCTION
// =============================================================================
enum class FError : uint32_t {
    EINTERNAL = 0x21001U,           // 内部Error
    INVALID_OPERATION = 0x21002U,   // 不允许的操作
    INVALID_TYPE = 0x21003U,        // 错误的类型
    INVALID_VAL = 0x21004U,         // 无效的值
    INVALID_PTR = 0x21005U,         // 无效的指针
    OUT_OF_RANGE = 0x21006U,        // 参数超出范围
    IS_EXIST = 0x21007U,            // 参数/操作已存在
    NOT_EXIST = 0x21008U,           // 参数/操作不存在
    BAD_FD = 0x29001U,              // 错误的文件描述符状态
    INVALID_FILE = 0x29002U,        // 无效的文件(内容)
    UNKNOWN = 0x3FFFFU
};

// =============================================================================
// F4XXXX - F5XXXX：PASS
// =============================================================================
enum class TensorErr : uint32_t {
    TENSOR_NULL_POINTER = 0x40000U,
    TENSOR_INVALID_MEMORY_TYPE = 0x40001U,
    TENSOR_SUBGRAPH_BOUNDARY = 0x40002U,
    TENSOR_SHAPE_MISMATCH = 0x40003U,
    TENSOR_UNSUPPORTED_DATATYPE = 0x40004U,
    TENSOR_MEMORY_ALLOCATION = 0x40005U,
    TENSOR_DYNAMIC_ATTR = 0x40006U
};

enum class OperationErr : uint32_t {
    OP_INVALID_OPERAND_COUNT = 0x41000U,
    OP_NULL_POINTER = 0x41001U,
    OP_INVALID_OPCODE = 0x41002U,
    OP_PRODUCER_CONSUMER = 0x41003U,
    OP_SPECIAL_CONSTRAINT = 0x41004U,
    OP_NESTING_DEPTH = 0x41005U,
    OP_SEQUENCE_ERROR = 0x41006U
};

enum class FunctionErr : uint32_t {
    FUNCTION_GRAPH_STRUCTURE = 0x42000U,
    FUNCTION_BOUNDARY_COMPLETENESS = 0x42001U,
    FUNCTION_GRAPH_CONNECTION = 0x42002U,
    FUNCTION_EXPAND_FEATURE = 0x42003U,
    FUNCTION_MEMORY_REACHABILITY = 0x42004U,
    FUNCTION_UNIQUENESS = 0x42005U,
    FUNCTION_SPECIAL_STRUCTURE = 0x42006U
};

enum class GraphErr : uint32_t {
    GRAPH_LOOP_DETECTION = 0x43000U,
    GRAPH_TOPOLOGY_STRUCTURE = 0x43001U,
    GRAPH_SUBGRAPH_EMPTY = 0x43002U,
    GRAPH_SUBGRAPH_ID_INVALID = 0x43003U,
    GRAPH_EDGE_CONSISTENCY = 0x43004U,
    GRAPH_COLOR_CONSISTENCY = 0x43005U,
    GRAPH_READY_STATE = 0x43006U,
    GRAPH_AIV_AIC_MIX = 0x43007U
};

enum class ConfigErr : uint32_t {
    CONFIG_MEMORY_TYPE_REACHABLE = 0x44000U,
    CONFIG_SUBGRAPH_BOUNDARY = 0x44001U,
    CONFIG_TENSOR_MEMORY_TYPE = 0x44002U
};

enum class ManagerErr : uint32_t {};

// =============================================================================
// F6XXXX：CODEGEN
// =============================================================================
enum class CodeGenErrorCategory {
    FRAMEWORK = 60000U,
    OPERATION_ADAPTER = 61000U,
    GEN_OP_CODE = 62000U,
    COMPILE_CODE = 63000U,
};

enum class FwkErr : uint32_t {
    PLATFORM_NOT_SUPPORTED = err_u(CodeGenErrorCategory::FRAMEWORK) + 1U,
    INVALID_FUNCTION = err_u(CodeGenErrorCategory::FRAMEWORK) + 2U,
};

enum class OperErr : uint32_t {
    ATTRIBUTE_INVALID = err_u(CodeGenErrorCategory::OPERATION_ADAPTER) + 1U,
    TENSOR_DIM_EXCEEDED = err_u(CodeGenErrorCategory::OPERATION_ADAPTER) + 2U,
    OPERAND_COUNT_EXCEEDED = err_u(CodeGenErrorCategory::OPERATION_ADAPTER) + 3U,
    OPERAND_COUNT_NOT_MATCHED = err_u(CodeGenErrorCategory::OPERATION_ADAPTER) + 4U,
    OPERATION_INIT_FAILED = err_u(CodeGenErrorCategory::OPERATION_ADAPTER) + 5U,
    OPERAND_TYPE_UNSUPPORTED = err_u(CodeGenErrorCategory::OPERATION_ADAPTER) + 6U,
};

enum class GenCodeErr : uint32_t {
    GEN_OP_CODE_FAILED = err_u(CodeGenErrorCategory::GEN_OP_CODE) + 1U,
    OP_CODE_UNSUPPORTED = err_u(CodeGenErrorCategory::GEN_OP_CODE) + 2U,
    PRINT_FAILED = err_u(CodeGenErrorCategory::GEN_OP_CODE) + 3U,
    PRINT_MODE_ERROR = err_u(CodeGenErrorCategory::GEN_OP_CODE) + 4U,
    DATA_TYPE_MISMATCHED = err_u(CodeGenErrorCategory::GEN_OP_CODE) + 5U,
    DATA_TYPE_UNSUPPORTED = err_u(CodeGenErrorCategory::GEN_OP_CODE) + 6U,
    TENSOR_SHAPE_INVALID = err_u(CodeGenErrorCategory::GEN_OP_CODE) + 7U,
    TENSOR_SHAPE_MISMATCHED = err_u(CodeGenErrorCategory::GEN_OP_CODE) + 8U,
    TENSOR_DIM_UNSUPPORTED = err_u(CodeGenErrorCategory::GEN_OP_CODE) + 9U,
    TENSOR_OFFSET_INVALID = err_u(CodeGenErrorCategory::GEN_OP_CODE) + 10U,
    TENSOR_MAGIC_CONFLICT = err_u(CodeGenErrorCategory::GEN_OP_CODE) + 11U,
    PARAM_IDX_INVALID = err_u(CodeGenErrorCategory::GEN_OP_CODE) + 12U,
    TENSOR_NOT_FOUND = err_u(CodeGenErrorCategory::GEN_OP_CODE) + 13U,
    SYMBOL_NOT_FOUND = err_u(CodeGenErrorCategory::GEN_OP_CODE) + 14U,
    PIPE_ID_NOT_FOUND = err_u(CodeGenErrorCategory::GEN_OP_CODE) + 15U,
    SYMBOL_ID_INVALID = err_u(CodeGenErrorCategory::GEN_OP_CODE) + 16U,
};

enum class CmpCodeErr : uint32_t {
    COMPILE_CODE_FAILED = err_u(CodeGenErrorCategory::COMPILE_CODE) + 1U,
    INCLUDE_FILE_NOT_FOUND = err_u(CodeGenErrorCategory::COMPILE_CODE) + 2U,
    PTO_ISA_NOT_FOUND = err_u(CodeGenErrorCategory::COMPILE_CODE) + 3U,
    CMD_CHECK_FAILED = err_u(CodeGenErrorCategory::COMPILE_CODE) + 4U,
    FILE_IO_FAILED = err_u(CodeGenErrorCategory::COMPILE_CODE) + 5U,
};

// =============================================================================
// F7XXXX - F8XXXX：MACHINE
// =============================================================================
enum class MachineError : uint32_t {
    HOST_BACKEND = 0x70000U,
    HOST_LAUNCHER = 0x71000U,
    SCHEDULE = 0x72000U,
    CONTROL_FLOW = 0x73000U,
    WORKSPACE = 0x74000U,
    DUMP_DFX = 0x75000U,
    PROGRAM_ENCODE = 0x76000U,
    TENSOR_META = 0x77000U,
    SERVER_KERNEL = 0x78000U,
    THREAD_MACHINE = 0x79000U,
    DEV_DATA = 0x7A000U,
    DEV_COMMON = 0x7A000U,
    RUNTIME_ERROR = 0x7B000U,
    UNKNOWN = 0x7F000U,
};

enum class DevDataErr : uint32_t {
    DEV_RELOC_VECTOR_INDEX_OOB = err_u(MachineError::DEV_DATA) + 0x01U,
    SMALL_ARRAY_RESIZE_OOB = err_u(MachineError::DEV_DATA) + 0x02U,
    VECTOR_UNINITIALIZED = err_u(MachineError::DEV_DATA) + 0x03U,
    VECTOR_INDEX_OUT_OF_RANGE = err_u(MachineError::DEV_DATA) + 0x04U,
    VECTOR_EMPTY_ACCESS = err_u(MachineError::DEV_DATA) + 0x05U,
    ITEM_POOL_UNINITIALIZED = err_u(MachineError::DEV_DATA) + 0x06U,
    ITEM_POOL_FREE_LIST_INVALID = err_u(MachineError::DEV_DATA) + 0x07U,
    ITEM_POOL_INDEX_OUT_OF_RANGE = err_u(MachineError::DEV_DATA) + 0x08U,
    SHEET_COLUMN_MISMATCH = err_u(MachineError::DEV_DATA) + 0x09U,
    SHEET_COLUMN_INDEX_OUT_OF_RANGE = err_u(MachineError::DEV_DATA) + 0x0AU,
};

enum class DevCommonErr : uint32_t {
    MEMCPY_FAILED = err_u(MachineError::DEV_COMMON) + 0x01U,
    ALLOC_FAILED = err_u(MachineError::DEV_COMMON) + 0x02U,
    MALLOC_FAILED = err_u(MachineError::DEV_COMMON) + 0x03U,
    NULLPTR = err_u(MachineError::DEV_COMMON) + 0x04U,
    PARAM_INVALID = err_u(MachineError::DEV_COMMON) + 0x05U,
    PARAM_CHECK_FAILED = err_u(MachineError::DEV_COMMON) + 0x06U,
    FILE_ERROR = err_u(MachineError::DEV_COMMON) + 0x07U,
    CMD_ERROR = err_u(MachineError::DEV_COMMON) + 0x08U,
    GET_ENV_FAILED = err_u(MachineError::DEV_COMMON) + 0x09U,
    GET_HANDLE_FAILED = err_u(MachineError::DEV_COMMON) + 0x0AU,
    FREE_FAILED = err_u(MachineError::DEV_COMMON) + 0x0BU,
};

enum class HostBackEndErr : uint32_t {
    COMPILE_AICORE_FAILED = err_u(MachineError::HOST_BACKEND) + 0x01U,
    COMPILE_CCEC_FAILED = err_u(MachineError::HOST_BACKEND) + 0x02U,
    LINK_FAILED = err_u(MachineError::HOST_BACKEND) + 0x03U,
    GEN_AICORE_FILE_FAILED = err_u(MachineError::HOST_BACKEND) + 0x04U,
    GEN_DYNAMIC_OP_FAILED = err_u(MachineError::HOST_BACKEND) + 0x05U,
    PRECOMPILE_FAILED = err_u(MachineError::HOST_BACKEND) + 0x06U,
    FUNCTION_CACHE_HASH_MISS = err_u(MachineError::HOST_BACKEND) + 0x07U,
    DUPLICATE_LEAF_FUNC_HASH = err_u(MachineError::HOST_BACKEND) + 0x08U,
};

enum class HostLauncherErr : uint32_t {
    LAUNCH_AICPU_FAILED = err_u(MachineError::HOST_LAUNCHER) + 0x01U,
    LAUNCH_PREPARE_FAILED = err_u(MachineError::HOST_LAUNCHER) + 0x02U,
    LAUNCH_CUSTOM_AICPU_FAILED = err_u(MachineError::HOST_LAUNCHER) + 0x03U,
    LAUNCH_AICORE_FAILED = err_u(MachineError::HOST_LAUNCHER) + 0x04U,
    LAUNCH_BUILTIN_OP_NULL_FAILED = err_u(MachineError::HOST_LAUNCHER) + 0x05U,
    REGISTER_KERNEL_FAILED = err_u(MachineError::HOST_LAUNCHER) + 0x06U,
    PREPARE_ARGS_FAILED = err_u(MachineError::HOST_LAUNCHER) + 0x07U,
    MAP_REG_ADDR_FAILED = err_u(MachineError::HOST_LAUNCHER) + 0x08U,
    MEM_POOL_CHECK_ALL_SENTINELS_FAILED = err_u(MachineError::HOST_LAUNCHER) + 0x09U,
    TRIPLE_STREAM_ERROR = err_u(MachineError::HOST_LAUNCHER) + 0x0AU,
    SYNC_FAILED = err_u(MachineError::HOST_LAUNCHER) + 0x0BU,
};

enum class SchedErr : uint32_t {
    TASK_WAIT_TIMEOUT = err_u(MachineError::SCHEDULE) + 0x01U,
    HANDSHAKE_TIMEOUT = err_u(MachineError::SCHEDULE) + 0x02U,
    READY_QUEUE_OVERFLOW = err_u(MachineError::SCHEDULE) + 0x03U,
    CORE_TASK_EXEC_FAILED = err_u(MachineError::SCHEDULE) + 0x04U,
    CORE_TASK_PROCESS_FAILED = err_u(MachineError::SCHEDULE) + 0x05U,
    RINGBUFFER_WAIT_TIMEOUT = err_u(MachineError::SCHEDULE) + 0x06U,
    ABNOMAL_LAST_WORD = err_u(MachineError::SCHEDULE) + 0x07U,
    SCH_DEVTASK_CTX_FULL = err_u(MachineError::SCHEDULE) + 0x08U,
    FSM_STATUS_ERROR = err_u(MachineError::SCHEDULE) + 0x09U,
    SCH_PARALLEL_DEVTASK_TIMEOUT = err_u(MachineError::SCHEDULE) + 0x0aU,
};

enum class CtrlErr : uint32_t {
    CTRL_FLOW_EXEC_FAILED = err_u(MachineError::CONTROL_FLOW) + 0x01U,
    ROOT_ALLOC_CTX_NULL = err_u(MachineError::CONTROL_FLOW) + 0x02U,
    ROOT_STITCH_CTX_NULL = err_u(MachineError::CONTROL_FLOW) + 0x03U,
    DEVICE_TASK_BUILD_FAILED = err_u(MachineError::CONTROL_FLOW) + 0x04U,
    TASK_STATS_ABNORMAL = err_u(MachineError::CONTROL_FLOW) + 0x05U,
    CTRL_INIT_FAILED = err_u(MachineError::CONTROL_FLOW) + 0x06U,
    CTRL_SIM_FAILED = err_u(MachineError::CONTROL_FLOW) + 0x07U,
    CTRL_ALLOC_TIMEOUT = err_u(MachineError::CONTROL_FLOW) + 0x08U,
};

enum class WsErr : uint32_t {
    SLAB_ADD_CACHE_FAILED = err_u(MachineError::WORKSPACE) + 0x01U,
    SLAB_STAGE_LIST_INCONSISTENT = err_u(MachineError::WORKSPACE) + 0x02U,
    SLAB_TYPE_INVALID = err_u(MachineError::WORKSPACE) + 0x03U,
    WORKSPACE_INIT_RESOURCE_ERROR = err_u(MachineError::WORKSPACE) + 0x04U,
    WORKSPACE_INIT_PARAM_INVALID = err_u(MachineError::WORKSPACE) + 0x05U,
    WS_TENSOR_ADDRESS_OUT_OF_RANGE = err_u(MachineError::WORKSPACE) + 0x06U,
    WORKSPACE_ITER_INVALID = err_u(MachineError::WORKSPACE) + 0x07U,
    WORKSPACE_REFCOUNT_INVALID = err_u(MachineError::WORKSPACE) + 0x08U,
    WORKSPACE_ALLOCATOR_REGIST_FAILED = err_u(MachineError::WORKSPACE) + 0x09U,
    WORKSPACE_CATEGORY_INVALID = err_u(MachineError::WORKSPACE) + 0x0AU,
    WORKSPACE_CAPACITY_INSUFFICIENT = err_u(MachineError::WORKSPACE) + 0x0BU,
    WORKSPACE_BASE_ADDR_OUT_OF_RANGE = err_u(MachineError::WORKSPACE) + 0x0CU,
};

enum class ProgEncodeErr : uint32_t {
    DYNFUNC_DATA_ALIGNMENT_ERROR = err_u(MachineError::PROGRAM_ENCODE) + 0x01U,
    FUNC_OP_SIZE_MISMATCH = err_u(MachineError::PROGRAM_ENCODE) + 0x02U,
    STITCH_PRED_SUCC_MISMATCH = err_u(MachineError::PROGRAM_ENCODE) + 0x03U,
    STITCH_LIST_TOO_LARGE = err_u(MachineError::PROGRAM_ENCODE) + 0x04U,
    STITCH_HANDLE_INDEX_OUT_OF_RANGE = err_u(MachineError::PROGRAM_ENCODE) + 0x05U,
    CELL_MATCH_PARAM_INVALID = err_u(MachineError::PROGRAM_ENCODE) + 0x06U,
    PROGRAM_RANGE_VERIFY_FAILED = err_u(MachineError::PROGRAM_ENCODE) + 0x07U,
    CACHE_RELOC_KIND_INVALID = err_u(MachineError::PROGRAM_ENCODE) + 0x08U,
    ADDR_OFFSET_RAW_MAGIC_MISMATCH = err_u(MachineError::PROGRAM_ENCODE) + 0x09U,
    CALL_OP_COUNT_EXCEEDS_UINT16_MAX = err_u(MachineError::PROGRAM_ENCODE) + 0x0AU,
    CELL_MATCH_DIM_ZERO = err_u(MachineError::PROGRAM_ENCODE) + 0x0BU,
    ASSEMBLE_STITCH_MEMORY_EXCESS = err_u(MachineError::PROGRAM_ENCODE) + 0x0CU,
    LEAF_CALLEE_ATTR_NULL = err_u(MachineError::PROGRAM_ENCODE) + 0x0DU,
};

enum class TensorMetaErr : uint32_t {
    TENSOR_DIM_COUNT_EXCEEDED = err_u(MachineError::TENSOR_META) + 0x01U,
    TENSOR_ENCODE_PTR_MISMATCH = err_u(MachineError::TENSOR_META) + 0x02U,
    RAW_TENSOR_INDEX_OUT_OF_RANGE = err_u(MachineError::TENSOR_META) + 0x03U,
    SHAPE_VALUE_MISMATCH = err_u(MachineError::TENSOR_META) + 0x04U,
    INCAST_ADDRESS_NULL = err_u(MachineError::TENSOR_META) + 0x05U,
    OUTCAST_ADDRESS_NULL = err_u(MachineError::TENSOR_META) + 0x06U,
    RUNTIME_WORKSPACE_NULL = err_u(MachineError::TENSOR_META) + 0x07U,
};

enum class ServerKernelErr : uint32_t {
    KERNEL_EXEC_FAILED = err_u(MachineError::SERVER_KERNEL) + 0x01U,
};

enum class ThreadErr : uint32_t {
    SIGNAL_HANDLER_ABNORMAL = err_u(MachineError::THREAD_MACHINE) + 0x01U,
    THREAD_CPU_ALLOC_FAILED = err_u(MachineError::THREAD_MACHINE) + 0x03U,
};

enum class RtErr : uint32_t {
    RT_INIT_FAILED = err_u(MachineError::RUNTIME_ERROR) + 0x01U,
    RT_MEMCPY_FAILED = err_u(MachineError::RUNTIME_ERROR) + 0x02U,
    RT_MEMSET_FAILED = err_u(MachineError::RUNTIME_ERROR) + 0x03U,
    RT_MALLOC_FAILED = err_u(MachineError::RUNTIME_ERROR) + 0x04U,
    RT_LAUNCH_FAILED = err_u(MachineError::RUNTIME_ERROR) + 0x05U,
    RT_EVENT_FAILED = err_u(MachineError::RUNTIME_ERROR) + 0x06U,
    RT_CAPTURE_FAILED = err_u(MachineError::RUNTIME_ERROR) + 0x07U,
    RT_REGISTER_FAILED = err_u(MachineError::RUNTIME_ERROR) + 0x08U,
    RT_LOAD_FAILED = err_u(MachineError::RUNTIME_ERROR) + 0x09U,
    RT_GET_FUNC_FAILED = err_u(MachineError::RUNTIME_ERROR) + 0x0AU,
    RT_DEVICE_FAILED = err_u(MachineError::RUNTIME_ERROR) + 0x0BU,
};

} // namespace npu::tile_fwk

// =============================================================================
// F9XXXX：SIMULATION（命名空间 CostModel）
// =============================================================================
namespace CostModel {

enum class SimulationErrorCategory {
    INTERNEL_ERROR = 90000U,
    EXTERNAL_ERROR = 91000U,
    FORWARD_SIM = 92000U,
    POST_SIM = 93000U,
    PRECISION_SIM = 94000U,
    UNKNOWN = 99000U,
};

enum class InternelErrorScene : uint32_t { UNKNOWN = 90099U };

enum class ExternalErrorScene : uint32_t {
    INVALID_CONFIG = 91001U,
    CONFIG_OUT_OF_RANGE = 91002U,
    INVALID_CONFIG_NAME = 91003U,
    PERMISSION_CHECK_ERROR = 91004U,
    FILE_FORMAT_ERROR = 91005U,
    FILE_CONTENT_ERROR = 91006U,
    INVALID_PATH = 91007U,
    FILE_OPEN_FAILED = 91008U,
    PYTHON_CMD_ERROR = 91009U,
    UNKNOWN = 91099U
};

enum class ForwardSimErrorScene : uint32_t {
    INVALID_PIPE_TYPE = 92001U,
    INVALID_DATA_TYPE = 92002U,
    DEAD_LOCK = 92003U,
    FUNC_NOT_SUPPORT = 92004U,
    UNKNOWN = 92099U
};

enum class PostSimErrorScene : uint32_t { UNKNOWN = 93099U };

enum class PrecisionSimErrorScene : uint32_t { NO_SO_EXISTS = 94001U, CANN_LOAD_FAILED = 94002U, UNKNOWN = 94099U };
} // namespace CostModel

namespace npu::tile_fwk {

// =============================================================================
// FAXXXX：DISTRIBUTED
// =============================================================================
enum class DistributedErrorCode : uint32_t {
    INVALID_GROUP_NAME = 0xA0000,
    INVALID_WORLD_SIZE = 0xA0001,
    INVALID_TENSOR_DIM = 0xA0002,
    INVALID_TENSOR_SHAPE = 0xA0003,
    INVALID_TENSOR_DTYPE = 0xA0004,
    INVALID_TENSOR_FORMAT = 0xA0005,
    INVALID_OPERAND_NUM = 0xA0006,
    INVALID_SHMEM_TENSOR = 0xA0007,
    INVALID_SHMEM_VIEW_PARAM = 0xA0008,
    INVALID_OP_TYPE = 0xA0009,
    INVALID_TILE_DIM = 0xA1000,
    INVALID_TILE_SHAPE = 0xA1001,
    INVALID_ALIGNMENT = 0xA1002,
    WIN_SIZE_EXCEED_LIMIT = 0xA2000,
    TILE_NUM_EXCEED_LIMIT = 0xA2001,
    DIVISION_BY_ZERO = 0xA2002,
    AICPU_TASK_TIMEOUT = 0xA3000,
    AICPU_TASK_NUM_EXCEED_LIMIT = 0xA3001,
    AICPU_TASK_QUEUE_EMPTY = 0xA3002,
    AICPU_TASKID_NOT_IN_MAP = 0xA3003,
    INVALID_GROUP_INDEX = 0xA3004,
    NULLPTR = 0xA3005,
};

// =============================================================================
// FBXXXX：VERIFY
// =============================================================================
enum class VerifyErrorCategory : uint32_t {
    VERIFY_ENABLE = 0xB0000U,
    CONTROL_FLOW = 0xB1000U,
    EXECUTE_OPERATION = 0xB2000U,
    OP_DUMP = 0xB3000U,
    VERIFY_RESULT = 0xB4000U,
};

enum class VerifyEnableScene : uint32_t {
    VERIFY_NOT_ENABLE = 0xB0001U,
    VERIFY_LOAD_CALC_OPS_FAILED = 0xB0002U,
};

enum class ControlFlowScene : uint32_t {
    INVALID_FUNC_IO_SPEC = 0xB1001U,
    INVALID_INPLACE_CHAIN = 0xB1002U,
    INVALID_CALLEE_MAPPING = 0xB1003U,
    FUNC_IO_DATAVIEW_NULL = 0xB1004U,
    FUNC_INCAST_COUNT_MISMATCH = 0xB1005U,
    FUNC_OUTCAST_COUNT_MISMATCH = 0xB1006U,
    FUNC_TENSOR_DATAVIEW_MISMATCH = 0xB1007U,
    FUNC_TENSOR_DATAVIEW_LIST_SIZE_MISMATCH = 0xB1008U,
    FUNC_INPLACE_ALLOC_CONFLICT = 0xB1009U,
    FUNC_TENSOR_DATAVIEW_DUP = 0xB100AU,
    FUNC_SPILL_RAW_TENSOR_DUP = 0xB100BU,
    FUNC_INPLACE_GROUP_NO_FUNC_IO = 0xB100CU,
    FUNC_SLOT_IO_COUNT_MISMATCH = 0xB100DU,
    FUNC_SLOT_MISSING = 0xB100EU,
    FUNC_UNKNOWN_IO_TYPE = 0xB100FU,
};

enum class ExecuteOperationScene : uint32_t {
    INVALID_TENSOR_SHAPE = 0xB2001U,
    INVALID_TENSOR_DTYPE = 0xB2002U,
    INVALID_TENSOR_SIZE = 0xB2003U,
    CTX_NULL = 0xB2004U,
    CTX_OP_NULL = 0xB2005U,
    CTX_INPUT_COUNT_MISMATCH = 0xB2006U,
    CTX_OUTPUT_COUNT_MISMATCH = 0xB2007U,
    CTX_INPUT_VIEW_NULL = 0xB2008U,
    CTX_OUTPUT_VIEW_NULL = 0xB2009U,
    UNSUPPORTED_OPCODE = 0xB200AU,
    EMPTY_VALIDSHAPE = 0xB200BU,
    VIEWTYPE_BYTES_MISMATCH = 0xB200CU,
    AMULACC_ACC_DTYPE_UNSUPPORTED = 0xB200DU,
    L0C_TO_L1_SHAPE_NOT_2D = 0xB200EU,
    RUNTIME_EXCEPTION = 0xB200FU,
};

enum class OpDumpScene : uint32_t {
    DUMP_OPEN_FILE_FAILED = 0xB3001U,
    DUMP_WRITE_FILE_FAILED = 0xB3002U,
};

enum class VerifyResultScene : uint32_t {
    VERIFY_RESULT_MISMATCH = 0xB4001U,
    VERIFY_RESULT_SHAPE_DIFF = 0xB4002U,
    VERIFY_RESULT_DTYPE_DIFF = 0xB4003U,
    VERIFY_RESULT_INDEX_OUTOFBOUNDS = 0xB4004U
};

enum class ElementScene : uint32_t {
    INVALID_ELEMENT_DTYPE = 0xB5001U,
};

// =============================================================================
// FC0–FC2：VECTOR
// =============================================================================
enum class VectorErrorCode : uint32_t {
    ERR_PARAM_INVALID = 0xC0000U,
    ERR_PARAM_DTYPE_UNSUPPORTED = 0xC0001U,
    ERR_CONFIG_TILE = 0xC1000U,
    ERR_CONFIG_ALIGNMENT = 0xC1001U,
    ERR_RUNTIME_NULLPTR = 0xC2000U,
    ERR_RUNTIME_LOGIC = 0xC2001U,
};

// =============================================================================
// FC3–FC5：MATMUL
// =============================================================================
enum class MatmulErrorCode : uint32_t {
    ERR_PARAM_INVALID = 0xC3000U,
    ERR_PARAM_MISMATCH = 0xC3001U,
    ERR_PARAM_UNSUPPORTED = 0xC3002U,
    ERR_CONFIG_TILE = 0xC4000U,
    ERR_CONFIG_ALIGNMENT = 0xC4001U,
    ERR_CONFIG_UNSUPPORTED = 0xC4002U,
    ERR_RUNTIME_NULLPTR = 0xC5000U,
    ERR_RUNTIME_STATE = 0xC5001U,
    ERR_RUNTIME_LOGIC = 0xC5002U,
};

// =============================================================================
// FC6–FC8：CONV
// =============================================================================
enum class ConvOperationError : uint32_t {
    INPUT_INVALID = 0xC6101U,
    OVER_BUFFER_LIMIT = 0xC6102U,
    UNKNOWN = 0xC6199U
};

enum class ConvExpandFuncError : uint32_t {
    EXPANDFUNC_TENSOR_OP_NULLPTR = 0xC6201U,
    EXPANDFUNC_TENSOR_ATTR_GET_FAILED = 0xC6202U,
    EXPANDFUNC_TILE_OP_NULLPTR = 0xC6203U,
    EXPANDFUNC_PARAMS_INVALID = 0xC6204U,
    EXPANDFUNC_INNER_STATUS_FAILED = 0xC6205U,
    UNKNOWN = 0xC6299U
};

enum class ConvCodenGenError : uint32_t {
    CODEGEN_GET_ATTR_FAILED = 0xC6301U,
    CODEGEN_CHECK_ATTR_INVALID = 0xC6302U,
    CODEGEN_CHECK_DIM_INVALID = 0xC6303U,
    UNKNOWN = 0xC6399U
};

enum class ConvTileOpError : uint32_t {
    TILEOP_TENSOR_FORMAT_FAILED = 0xC6401U,
    TILEOP_SHAPE_SIZE_FAILED = 0xC6402U,
    TILEOP_STC_SHAPE_INVALID = 0xC6403U,
    TILEOP_INDEX_INVALID = 0xC6404U,
    UNKNOWN = 0xC6499U
};

} // namespace npu::tile_fwk

// =============================================================================
// Calculator（Interpreter）：npu::tile_fwk::calc_error
// =============================================================================
namespace npu::tile_fwk::calc_error {

enum class CalculatorErrorScene : uint32_t {
    RANGE_NUMEL_MISMATCH = 0xBF000U,
    COMPARE_UNSUPPORTED_TYPE = 0xBF001U,
    BITMODE_LAST_DIM_INVALID = 0xBF002U,
    FORMAT_ND2NZ_RANK_LT_2 = 0xBF003U,
    FORMAT_NZ2ND_RANK_LT_2 = 0xBF004U,
    QUANTPRECOMPUTE_NULL_DATAPTR = 0xBF005U,
    QUANTPRECOMPUTE_DTYPE_MISMATCH = 0xBF006U,
    GATHERMASK_PATTERNMODE_INVALID = 0xBF007U,
    MRGSORT_AXIS_OUT_OF_RANGE = 0xBF008U,
    SCATTER_BLOCKSIZE_ZERO = 0xBF009U,
    SCATTER_INDICES_DIM_INVALID = 0xBF00AU,
    SCATTER_SRC_RET_DIM_UNSUPPORTED = 0xBF00BU,
    SCATTER_SRC_RET_DIM_MISMATCH = 0xBF00CU,
};

} // namespace npu::tile_fwk::calc_error

#include "tilefwk/error.h"

namespace npu::tile_fwk {

#ifndef FUNCTION_ASSERT
#define FUNCTION_ASSERT_SELECT(_1, _2, NAME, ...) NAME
#define FUNCTION_ASSERT_WITH_UNKNOWN(cond) ASSERT(FError::UNKNOWN, cond)
#define FUNCTION_ASSERT(...) FUNCTION_ASSERT_SELECT(__VA_ARGS__, ASSERT, FUNCTION_ASSERT_WITH_UNKNOWN)(__VA_ARGS__)
#endif

} // namespace npu::tile_fwk
