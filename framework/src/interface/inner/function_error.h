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
    PROGRAM = 20000U,    // 0: 程序相关
    CONFIG = 21000U,     // 1: 配置相关
    FUNCTION = 22000U,   // 2: 函数相关
    OPERATION = 23000U,  // 3: 操作相关
    TENSOR = 24000U,     // 4: 张量相关
    TENSORSLOT = 25000U, // 5: 张量槽相关
    SYMBOLIC = 26000U,   // 6: 符号相关
    UNKNOWN = 29000U,    // 8: 未知/预留
};

// =============================================================================
// 二、子流程：各 Category 下的 ErrorScene 枚举（枚举值即错误码 F2xxxx）
// =============================================================================

enum class ProgErr : uint32_t {
    FUNCTION_NOT_FOUND = 20001U,
    FUNCTION_ALREADY_EXISTS = 20002U,
    FUNCTION_NAME_DUPLICATE = 20003U,
    NO_ACTIVE_FUNCTION = 20004U,
    FUNCTION_STACK_NULL = 20005U,
    FUNCTION_NO_PARENT = 20006U,
    CURRENT_MAGIC_NOT_FOUND = 20007U,
    PROGRAM_ENTRY_NOT_FOUND = 20008U,
    CURRENT_FUNCTION_NOT_FOUND = 20009U,
    DUMP_FILE_OPEN_FAILED = 200010U,
    LOOP_STACK_EMPTY = 20011U,
    MUST_HAVE_UNROLL_ONE = 200012U,
    LOOP_UNROLL_TIMES_INVALID = 20013U,
    LOOP_UNROLL_TIMES_EXISTS = 20014U,
    LOOP_UNROLL_TIMES_EMPTY = 20015U,
    LOOP_INDEX_NAME_DUPLICATE = 20016U,
    RECORD_FUNCTION_FAILED = 20017U,
    RECORD_LOOP_FAILED = 20018U,
    RECORD_CONDITION_FAILED = 20019U,

    UNKNOWN = 20099U
};

enum class ConfErr : uint32_t {
    INVALID_TYPE = 21001U,
    FIELD_MISSING = 21002U,
    KEY_NOT_LOADED = 21003U,
    VALUE_OVERFLOW = 21004U,
    SET_FAILED = 21005U,
    READ_FAILED = 21006U,
    UNKNOWN = 21099U
};

enum class FuncErr : uint32_t {
    GRAPH_TYPE_MISMATCH = 22001U,
    FUNCTION_TYPE_MISMATCH = 22002U,
    NULLPTR = 22002U,
    INDEX_OUT_OF_BOUNDS = 22003U,
    NOT_FOUND = 22004U,
    CYCLE_DETECTED = 22005U,
    SIZE_MISMATCH = 22006U,
    DUPLICATE = 22007U,
    UNKNOWN = 22099U
};

enum class OpErr : uint32_t {
    ATTRIBUTE_NOT_FOUND = 23001U,
    ATTRIBUTE_VALUE_INVALID = 23002U,
    SYMBOLIC_SCALAR_INVALID = 23003U,
    OPERAND_NULL = 23004U,
    OPERAND_INDEX_OUT_OF_BOUNDS = 23005U,
    TENSOR_NOT_IN_DICT = 23006U,
    OP_TYPE_MISMATCH = 23007U,
    OP_ATTR_NULL = 23008U,
    UNKNOWN = 23099U
};

enum class TensorErr : uint32_t {
    NO_DIMENSIONS = 24001U,
    AXIS_OUT_OF_RANGE = 24002U,
    DIMS_INCONSISTENT = 24003U,
    VIEW_DIMENSION_MISMATCH = 24004U,
    VIEW_OFFSET_MISMATCH = 24005U,
    SHAPE_OUT_OF_BOUNDS = 24006U,
    INVALID_SHAPE = 24007U,
    NZ_ALIGNMENT_ERROR = 24008U,
    DATATYPE_MISMATCH = 24009U,
    STORAGE_NULL = 24010U,
    SELF_ASSIGNMENT = 24011U,
    UNINITIALIZED_ASSIGNMENT = 24012U,
    MAGIC_NOT_FOUND = 24013U,
    NOT_UNDER_DYNAMIC_FUNCTION = 24014U,
    NOT_IN_INPUT_LIST = 24015U,
    NO_ACTIVE_FUNCTION = 24016U,
    MAGIC_NAME_EXISTS = 24017U,
    UNKNOWN = 24099U
};

enum class SlotErr : uint32_t {
    LOGICAL_TENSOR_NOT_FOUND_IN_INPUT = 25001U,
    LOGICAL_TENSOR_NOT_FOUND_IN_OUTPUT = 25002U,
    NOT_FOUND_IN_INDEX_DICT = 25003U,
    ALREADY_EXISTS_IN_INPUT = 25004U,
    ALREADY_EXISTS_IN_OUTPUT = 25005U,
    NOT_FOUND_IN_INPUT_DICT = 25006U,
    CHECKPOINT_STACK_ERROR = 25007U,
    INPUT_NOT_IN_INDEX_DICT = 25008U,
    OUTPUT_NOT_IN_INDEX_DICT = 25009U,
    CURRENT_NO_PARENT = 25010U,
    CURRENT_NULL = 25011U,
    UNKNOWN = 25099U
};

enum class SymErr : uint32_t {
    // 符号编译相关 (26001-26009)
    BINARY_OPEN_FAILED = 26001U,
    GCC_FAILED = 26002U,
    ASSEMBLE_FAILED = 26003U,
    OBJCOPY_FAILED = 26004U,
    BINARY_READ_FAILED = 26005U,

    // 符号操作数相关 (26021-26029)
    OPERAND_COUNT_MISMATCH = 26021U,
    OPERAND_SIZE_INVALID = 26022U,
    LOOP_BEGIN_END_NESTED = 26023U,

    // 符号表达式相关 (26031-26039)
    EXPRESSION_NOT_FOUND_IN_TABLE = 26031U,
    EXPRESSION_NOT_FOUND_IN_PRIMARY_DICT = 26032U,
    EXPRESSION_NOT_FOUND_IN_PRIMARY_SET = 26033U,
    SYMBOL_TABLE_SIZE_MISMATCH = 26034U,
    ELEMENT_KEY_MISMATCH = 26035U,
    TITLE_MISMATCH = 26036U,
    SYMBOL_NOT_FOUND_IN_VALUE_DICT = 26037U,

    UNKNOWN = 26099U
};

} // namespace npu::tile_fwk
