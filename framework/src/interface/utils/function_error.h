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
    SCALAR = 0x26000U,     // 6: 标量相关
    FILE = 0x27000U,       // 7: 文件相关
    UNKNOWN = 0x29000U,    // 8: 未知/预留
};

// =============================================================================
// 二、子流程：各 Category 下的 ErrorScene 枚举（枚举值即错误码 F2xxxx）
// =============================================================================

// program error code
enum class ProgErr : uint32_t {
    // 函数查找与管理相关错误 (20001-20008)
    FUNCTION_NOT_FOUND = 0x20001U,
    FUNCTION_ALREADY_EXISTS = 0x20002U,
    FUNCTION_NAME_DUPLICATE = 0x20003U,
    NO_ACTIVE_FUNCTION = 0x20004U,
    FUNCTION_STACK_NULL = 0x20005U,
    CURRENT_MAGIC_NOT_FOUND = 0x20006U,
    PROGRAM_ENTRY_NOT_FOUND = 0x20007U,
    CURRENT_FUNCTION_NOT_FOUND = 0x20008U,

    // 循环展开相关错误 (20011-20015)
    MUST_HAVE_UNROLL_ONE = 0x20011U,
    LOOP_UNROLL_TIMES_INVALID = 0x20012U,
    LOOP_UNROLL_TIMES_EXISTS = 0x20013U,
    LOOP_UNROLL_TIMES_EMPTY = 0x20014U,
    LOOP_INDEX_NAME_DUPLICATE = 0x20015U,

    UNKNOWN = 0x20099U
};

// config error code
enum class ConfErr : uint32_t {
    // 配置类型与字段相关错误 (21001-21006)
    INVALID_TYPE = 0x21001U,
    FIELD_MISSING = 0x21002U,
    KEY_NOT_LOADED = 0x21003U,
    VALUE_OVERFLOW = 0x21004U,
    SET_FAILED = 0x21005U,
    READ_FAILED = 0x21006U,

    UNKNOWN = 0x21099U
};

// function error code
enum class FuncErr : uint32_t {
    // 函数类型与图相关错误 (22001-22009)
    FUNCTION_TYPE_MISMATCH = 0x22001U,
    GRAPH_TYPE_MISMATCH = 0x22002U,
    GRAPH_CYCLE_DETECTED = 0x22003U,

    // 函数槽与转换相关错误 (22011-22019)
    SLOT_SCOPE_NULL = 0x22011U,
    ORIGIN_INCAST_NOT_FOUND = 0x22012U,
    ORIGIN_OUTCAST_NOT_FOUND = 0x22013U,

    // 槽索引与消费者生产者相关错误 (22021-22029)
    SLOT_SET_INDEX_SIZE_INVALID = 0x22021U,
    INCAST_HAS_NO_CONSUMER = 0x22022U,
    OUTCAST_HAS_NO_PRODUCER = 0x22023U,

    // 函数参数相关错误 (22031-22039)
    PARAM_INDEX_OUT_OF_BOUNDS = 0x22031U,
    PARAM_ADDRESS_NOT_STORED = 0x22032U,
    PARAM_INFO_MISMATCH = 0x22033U,
    CALLEE_FUNCTION_NULL = 0x22041U,
    CALLEE_NOT_IN_FUNCTION_MAP = 0x22042U,
    TARGET_FUNCTION_NULL = 0x22043U,
    FUNCTION_NO_PARENT = 0x22044U,

    // 张量数据与导入相关错误 (22051-22055)
    GET_TENSOR_DATA_INDEX_INVALID = 0x22051U,
    FUNCTION_NOT_IN_USAGE_DICT = 0x22052U,
    IMPORT_INDEX_NOT_FOUND = 0x22053U,
    INCAST_OUTCAST_INDEX_INVALID = 0x22054U,
    OUTCAST_INDEX_NOT_FOUND = 0x22055U,

    // JSON反序列化相关错误 (22071)
    JSON_FUNCTION_KIND_INVALID = 0x22071U,

    // 操作数与生产者相关错误 (22101-22109)
    OPERAND_NOT_BELONGS_TO_CURRENT_FUNCTION = 0x22101U,
    PRODUCER_NOT_FOUND = 0x22102U,
    OPERATION_ALREADY_IN_MAP = 0x22103U,

    // 转换冲突相关错误 (22111-22119)
    INCAST_HAS_CONFLICT_TENSORS = 0x22111U,

    // 操作列表与分组相关错误 (22121-22129)
    OP_LIST_MISMATCH = 0x22121U,
    OP_NOT_FOUND_IN_POSITION = 0x22122U,
    OP_ALREADY_IN_GROUP = 0x22123U,
    OP_GROUP_MISMATCH = 0x22124U,
    OP_ALREADY_IN_MAP = 0x22125U,

    UNKNOWN = 0x22199U
};

// operation error code
enum class OpErr : uint32_t {
    // 操作基本属性与验证错误 (23001-2300B)
    OP_ATTRIBUTE_NULL = 0x23001U,
    ATTRIBUTE_VALUE_INVALID = 0x23002U,
    SYMBOLIC_SCALAR_INVALID = 0x23003U,
    OPERAND_NULL = 0x23004U,
    OPERAND_INDEX_OUT_OF_BOUNDS = 0x23005U,
    TENSOR_NOT_IN_DICT = 0x23006U,
    OP_TYPE_MISMATCH = 0x23007U,
    OP_ATTR_NULLPTR = 0x23008U,
    OP_NOT_REGISTER = 0x23009U,
    OP_NOT_MARKED_DELETED = 0x2300AU,
    HASH_DUPLICATE = 0x2300BU,

    // 操作魔法值相关错误 (23011-23019)
    OP_DUPLICATE = 0x23011U,
    OP_MAGIC_OUT_OF_BOUNDS = 0x23012U,
    OP_MAGIC_RANGE_INVALID = 0x23013U,

    // 循环依赖错误 (23021-23029)
    OP_CYCLE_DETECTED = 0x23021U,

    // 生产者相关错误 (23051-23059)
    CONSUMER_NOT_FOUND = 0x23051U,

    // 连接相关错误 (23081)
    MATCHES_SHOULD_NOT_BE_EMPTY = 0x23081U,

    // 操作属性相关错误 (23091-23093)
    OOP_ATTR_OFFSET_NOT_EMPTY = 0x23091U,
    ARG_LIST_NOT_EMPTY = 0x23092U,
    OUTCAST_INDEX_TO_EXPR_NOT_EMPTY = 0x23093U,

    UNKNOWN = 0x23099U
};

// tensor error code
enum class TensorErr : uint32_t {
    // 张量维度与形状相关错误 (24001-24008)
    NO_DIMENSIONS = 0x24001U,
    AXIS_OUT_OF_RANGE = 0x24002U,
    DIMS_INCONSISTENT = 0x24003U,
    VIEW_DIMENSION_MISMATCH = 0x24004U,
    OFFSET_MISMATCH = 0x24005U,
    SHAPE_OUT_OF_BOUNDS = 0x24006U,
    INVALID_SHAPE = 0x24007U,
    SHAPE_MISMATCH = 0x24008U,

    // 张量数据类型与存储相关错误 (2400A-2400F)
    DATATYPE_MISMATCH = 0x2400AU,
    STORAGE_NULL = 0x2400BU,
    SELF_ASSIGNMENT = 0x2400CU,
    MAGIC_NOT_FOUND = 0x2400EU,
    NOT_UNDER_DYNAMIC_FUNCTION = 0x2400FU,

    // 张量魔法值与索引相关错误 (24010, 24012-24015)
    NOT_IN_INPUT_LIST = 0x24010U,
    INVALID_DESC_INDEX = 0x24012U,
    OPERAND_NOT_CONSUMER = 0x24013U,
    OPERAND_NOT_PRODUCER = 0x24014U,
    TENSOR_HAS_PRODUCERS = 0x24015U,

    UNKNOWN = 0x24099U
};

// tensor slot error code
enum class TensorSlotErr : uint32_t {
    // 张量槽查找相关错误 (25001-25003)
    LOGICAL_TENSOR_NOT_FOUND_IN_INPUT = 0x25001U,
    LOGICAL_TENSOR_NOT_FOUND_IN_OUTPUT = 0x25002U,
    NOT_FOUND_IN_INDEX_DICT = 0x25003U,

    // 张量槽作用域相关错误 (2500C)
    SLOT_SCOPE_NULL = 0x2500CU,

    UNKNOWN = 0x25099U
};

// symbolic scalar error code
enum class SymScalarErr : uint32_t {
    // 符号操作数相关 (26022)
    OPERAND_SIZE_INVALID = 0x26022U,

    // 符号表达式相关 (26035-26037)
    ELEMENT_KEY_MISMATCH = 0x26035U,
    TITLE_MISMATCH = 0x26036U,
    SYMBOL_NOT_FOUND_IN_VALUE_DICT = 0x26037U,

    // Symbolic scalar Type相关 (26041)
    TYPE_MISMATCH = 0x26041U,

    UNKNOWN = 0x26099U
};

// file error code
enum class FileErr : uint32_t {
    FILE_OPEN_FAILED = 0x27001U,

    UNKNOWN = 0x27099U
};

} // namespace npu::tile_fwk
