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
 * \file function_error.h
 * \brief PyPTO 组件错误分类、场景枚举与错误码常量。
 *        - F2xxxx：PyPTO 内部已梳理报错（大流程 Category -> 子流程 Scene）。
 */

#pragma once

#include <cstdint>

namespace npu::tile_fwk {

// =============================================================================
// 一、大流程：ErrorCategory（对应 F2Cxxx 中的 C）
// =============================================================================

enum class ErrorCategory {
    PROGRAM = 0x20000U,    // 0: 程序相关
    CONFIG = 0x21000U,     // 1: 配置相关
    FUNCTION = 0x22000U,   // 2: 函数相关
    OPERATION = 0x23000U,  // 3: 操作相关
    TENSOR = 0x24000U,     // 4: 张量相关
    TENSORSLOT = 0x25000U, // 5: 张量槽相关
    SYMBOLIC = 0x26000U,   // 6: 符号相关
    UNKNOWN = 0x29000U,    // 8: 未知/预留
};

// =============================================================================
// 二、子流程：各 Category 下的 ErrorScene 枚举（枚举值即错误码 F2xxxx）
// =============================================================================

enum class ProgErr : uint32_t {
    FUNCTION_NOT_FOUND = 0x20001U,
    FUNCTION_ALREADY_EXISTS = 0x20002U,
    FUNCTION_NAME_DUPLICATE = 0x20003U,
    NO_ACTIVE_FUNCTION = 0x20004U,
    FUNCTION_STACK_NULL = 0x20005U,
    FUNCTION_NO_PARENT = 0x20006U,
    CURRENT_MAGIC_NOT_FOUND = 0x20007U,
    PROGRAM_ENTRY_NOT_FOUND = 0x20008U,
    CURRENT_FUNCTION_NOT_FOUND = 0x20009U,
    DUMP_FILE_OPEN_FAILED = 0x20000AU,
    LOOP_STACK_EMPTY = 0x2000BU,
    MUST_HAVE_UNROLL_ONE = 0x20000CU,
    LOOP_UNROLL_TIMES_INVALID = 0x2000DU,
    LOOP_UNROLL_TIMES_EXISTS = 0x2000EU,
    LOOP_UNROLL_TIMES_EMPTY = 0x2000FU,
    LOOP_INDEX_NAME_DUPLICATE = 0x20010U,
    RECORD_FUNCTION_FAILED = 0x20011U,
    RECORD_LOOP_FAILED = 0x20012U,
    RECORD_CONDITION_FAILED = 0x20013U,

    UNKNOWN = 0x20099U
};

enum class ConfErr : uint32_t {
    INVALID_TYPE = 0x21001U,
    FIELD_MISSING = 0x21002U,
    KEY_NOT_LOADED = 0x21003U,
    VALUE_OVERFLOW = 0x21004U,
    SET_FAILED = 0x21005U,
    READ_FAILED = 0x21006U,
    UNKNOWN = 0x21099U
};

enum class FuncErr : uint32_t {
    GRAPH_TYPE_MISMATCH = 0x22001U,
    FUNCTION_TYPE_MISMATCH = 0x22002U,
    NULLPTR = 0x22002U,
    INDEX_OUT_OF_BOUNDS = 0x22003U,
    NOT_FOUND = 0x22004U,
    CYCLE_DETECTED = 0x22005U,
    SIZE_MISMATCH = 0x22006U,
    DUPLICATE = 0x22007U,
    UNKNOWN = 0x22099U
};

enum class OpErr : uint32_t {
    ATTRIBUTE_NOT_FOUND = 0x23001U,
    ATTRIBUTE_VALUE_INVALID = 0x23002U,
    SYMBOLIC_SCALAR_INVALID = 0x23003U,
    OPERAND_NULL = 0x23004U,
    OPERAND_INDEX_OUT_OF_BOUNDS = 0x23005U,
    TENSOR_NOT_IN_DICT = 0x23006U,
    OP_TYPE_MISMATCH = 0x23007U,
    OP_ATTR_NULL = 0x23008U,
    UNKNOWN = 0x23099U
};

enum class TensorErr : uint32_t {
    NO_DIMENSIONS = 0x24001U,
    AXIS_OUT_OF_RANGE = 0x24002U,
    DIMS_INCONSISTENT = 0x24003U,
    VIEW_DIMENSION_MISMATCH = 0x24004U,
    VIEW_OFFSET_MISMATCH = 0x24005U,
    SHAPE_OUT_OF_BOUNDS = 0x24006U,
    INVALID_SHAPE = 0x24007U,
    NZ_ALIGNMENT_ERROR = 0x24008U,
    DATATYPE_MISMATCH = 0x24009U,
    STORAGE_NULL = 0x2400AU,
    SELF_ASSIGNMENT = 0x2400BU,
    UNINITIALIZED_ASSIGNMENT = 0x2400CU,
    MAGIC_NOT_FOUND = 0x2400DU,
    NOT_UNDER_DYNAMIC_FUNCTION = 0x2400EU,
    NOT_IN_INPUT_LIST = 0x2400FU,
    NO_ACTIVE_FUNCTION = 0x24010U,
    MAGIC_NAME_EXISTS = 0x24011U,
    UNKNOWN = 0x24099U
};

enum class SlotErr : uint32_t {
    LOGICAL_TENSOR_NOT_FOUND_IN_INPUT = 0x25001U,
    LOGICAL_TENSOR_NOT_FOUND_IN_OUTPUT = 0x25002U,
    NOT_FOUND_IN_INDEX_DICT = 0x25003U,
    ALREADY_EXISTS_IN_INPUT = 0x25004U,
    ALREADY_EXISTS_IN_OUTPUT = 0x25005U,
    NOT_FOUND_IN_INPUT_DICT = 0x25006U,
    CHECKPOINT_STACK_ERROR = 0x25007U,
    INPUT_NOT_IN_INDEX_DICT = 0x25008U,
    OUTPUT_NOT_IN_INDEX_DICT = 0x25009U,
    CURRENT_NO_PARENT = 0x2500AU,
    CURRENT_NULL = 0x2500BU,
    UNKNOWN = 0x25099U
};

enum class SymErr : uint32_t {
    // 符号编译相关 (26001-26009)
    BINARY_OPEN_FAILED = 0x26001U,
    GCC_FAILED = 0x26002U,
    ASSEMBLE_FAILED = 0x26003U,
    OBJCOPY_FAILED = 0x26004U,
    BINARY_READ_FAILED = 0x26005U,

    // 符号操作数相关 (26021-26029)
    OPERAND_COUNT_MISMATCH = 0x26021U,
    OPERAND_SIZE_INVALID = 0x26022U,
    LOOP_BEGIN_END_NESTED = 0x26023U,

    // 符号表达式相关 (26031-26039)
    EXPRESSION_NOT_FOUND_IN_TABLE = 0x26031U,
    EXPRESSION_NOT_FOUND_IN_PRIMARY_DICT = 0x26032U,
    EXPRESSION_NOT_FOUND_IN_PRIMARY_SET = 0x26033U,
    SYMBOL_TABLE_SIZE_MISMATCH = 0x26034U,
    ELEMENT_KEY_MISMATCH = 0x26035U,
    TITLE_MISMATCH = 0x26036U,
    SYMBOL_NOT_FOUND_IN_VALUE_DICT = 0x26037U,

    UNKNOWN = 0x26099U
};

} // namespace npu::tile_fwk
