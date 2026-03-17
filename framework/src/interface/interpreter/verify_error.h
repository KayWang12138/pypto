/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file verify_error.h
 * \brief
 */

#pragma once

#include <cstdint>

namespace npu::tile_fwk {

// 一级错误大类，用于和 ASSERT(err_code, cond) 搭配使用
// 具体场景（Scene）可以在后续根据需要扩展
enum class VerifyErrorCategory : uint32_t {
    VERIFY_ENABLE     = 0xB0000U, // 0: 校验环境
    CONTROL_FLOW      = 0xB1000U, // 1: 执行控制流
    EXECUTE_OPERATION = 0xB2000U, // 2：op执行
    OP_DUMP           = 0xB3000U, // 3：op Dump
    VERIFY_RESULT     = 0xB4000U, // 4: 精度比对
};

// 预留二级场景枚举：
//  - VerifyErrorCategory 仅表示大类范围，不直接作为具体错误码使用
//  - 具体错误码请使用对应 Scene 枚举值
enum class VerifyEnableScene : uint32_t {
    VERIFY_NOT_ENABLE = 0xB0001U, // 校验功能未开启
};

// 控制流相关场景：函数图/控制流结构不合法
enum class ControlFlowScene : uint32_t {
    INVALID_FUNC_IO_SPEC      = 0xB1001U, // 函数 incast/outcast 规格不一致
    INVALID_INPLACE_CHAIN     = 0xB1002U, // inplace 链路不满足预期约束
    INVALID_CALLEE_MAPPING    = 0xB1003U, // callee hash 映射不一致
};

// 执行算子相关场景：shape/dtype/参数非法等
enum class ExecuteOperationScene : uint32_t {
    INVALID_TENSOR_SHAPE      = 0xB2001U, // 张量 shape / validShape 不匹配
    INVALID_TENSOR_DTYPE      = 0xB2002U, // 张量 dtype 不符合预期
    INVALID_TENSOR_SIZE       = 0xB2003U, // 数据长度/字节数不匹配
    INVALID_OP_CONTEXT        = 0xB2004U, // ExecuteOperationContext 内部状态异常
    UNSUPPORTED_OPCODE        = 0xB2005U, // 不支持的 Opcode
};

// Dump/IO 相关场景：文件读写错误等
enum class OpDumpScene : uint32_t {
    DUMP_OPEN_FILE_FAILED     = 0xB3001U, // 打开 dump 文件失败
    DUMP_WRITE_FILE_FAILED    = 0xB3002U, // 写入 dump 文件失败
};

// 精度/结果验证相关场景
enum class VerifyResultScene : uint32_t {
    VERIFY_RESULT_MISMATCH    = 0xB4001U, // 精度比对失败
    VERIFY_RESULT_SHAPE_DIFF  = 0xB4002U, // 比对双方 shape 不一致
    VERIFY_RESULT_DTYPE_DIFF  = 0xB4003U, // 比对双方 dtype 不一致
};

} // namespace npu::tile_fwk