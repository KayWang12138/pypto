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
 * \file utils_defop_switch.h
 * \brief
 */

#pragma once

// ===== Token concatenation =====
#define PTO_PP_CAT_(a, b) a##b
#define PTO_PP_CAT(a, b) PTO_PP_CAT_(a, b)

// ===== Count variadic args up to 32 =====
#define PTO_PP_NARG_( \
     _1,  _2,  _3,  _4,  _5,  _6,  _7,  _8,  _9, _10, \
    _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, \
    _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, \
    _31, _32, N, ...) N

#define PTO_PP_RSEQ_N() \
    32,31,30,29,28,27,26,25,24,23,22,21,20, \
    19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0

#define PTO_PP_NARG_I(...) PTO_PP_NARG_(__VA_ARGS__)
#define PTO_PP_NARG(...) PTO_PP_NARG_I(__VA_ARGS__, PTO_PP_RSEQ_N())

// ===== Expand OPCODE(...) list in *.def files =====
#define PTO_EXPAND_OPCODE_LIST(opcode_token) PTO_PP_CAT(PTO_EXPAND_OPCODE_LIST_, opcode_token)
#define PTO_EXPAND_OPCODE_LIST_OPCODE(...) __VA_ARGS__

// ===== FOR_EACH over opcode list (HANDLER(OPCLASS, OPC)) =====
#define PTO_FE_1(H, C, a1) H(C, a1)
#define PTO_FE_2(H, C, a1, ...) H(C, a1) PTO_FE_1(H, C, __VA_ARGS__)
#define PTO_FE_3(H, C, a1, ...) H(C, a1) PTO_FE_2(H, C, __VA_ARGS__)
#define PTO_FE_4(H, C, a1, ...) H(C, a1) PTO_FE_3(H, C, __VA_ARGS__)
#define PTO_FE_5(H, C, a1, ...) H(C, a1) PTO_FE_4(H, C, __VA_ARGS__)
#define PTO_FE_6(H, C, a1, ...) H(C, a1) PTO_FE_5(H, C, __VA_ARGS__)
#define PTO_FE_7(H, C, a1, ...) H(C, a1) PTO_FE_6(H, C, __VA_ARGS__)
#define PTO_FE_8(H, C, a1, ...) H(C, a1) PTO_FE_7(H, C, __VA_ARGS__)
#define PTO_FE_9(H, C, a1, ...) H(C, a1) PTO_FE_8(H, C, __VA_ARGS__)
#define PTO_FE_10(H, C, a1, ...) H(C, a1) PTO_FE_9(H, C, __VA_ARGS__)
#define PTO_FE_11(H, C, a1, ...) H(C, a1) PTO_FE_10(H, C, __VA_ARGS__)
#define PTO_FE_12(H, C, a1, ...) H(C, a1) PTO_FE_11(H, C, __VA_ARGS__)
#define PTO_FE_13(H, C, a1, ...) H(C, a1) PTO_FE_12(H, C, __VA_ARGS__)
#define PTO_FE_14(H, C, a1, ...) H(C, a1) PTO_FE_13(H, C, __VA_ARGS__)
#define PTO_FE_15(H, C, a1, ...) H(C, a1) PTO_FE_14(H, C, __VA_ARGS__)
#define PTO_FE_16(H, C, a1, ...) H(C, a1) PTO_FE_15(H, C, __VA_ARGS__)
#define PTO_FE_17(H, C, a1, ...) H(C, a1) PTO_FE_16(H, C, __VA_ARGS__)
#define PTO_FE_18(H, C, a1, ...) H(C, a1) PTO_FE_17(H, C, __VA_ARGS__)
#define PTO_FE_19(H, C, a1, ...) H(C, a1) PTO_FE_18(H, C, __VA_ARGS__)
#define PTO_FE_20(H, C, a1, ...) H(C, a1) PTO_FE_19(H, C, __VA_ARGS__)
#define PTO_FE_21(H, C, a1, ...) H(C, a1) PTO_FE_20(H, C, __VA_ARGS__)
#define PTO_FE_22(H, C, a1, ...) H(C, a1) PTO_FE_21(H, C, __VA_ARGS__)
#define PTO_FE_23(H, C, a1, ...) H(C, a1) PTO_FE_22(H, C, __VA_ARGS__)
#define PTO_FE_24(H, C, a1, ...) H(C, a1) PTO_FE_23(H, C, __VA_ARGS__)
#define PTO_FE_25(H, C, a1, ...) H(C, a1) PTO_FE_24(H, C, __VA_ARGS__)
#define PTO_FE_26(H, C, a1, ...) H(C, a1) PTO_FE_25(H, C, __VA_ARGS__)
#define PTO_FE_27(H, C, a1, ...) H(C, a1) PTO_FE_26(H, C, __VA_ARGS__)
#define PTO_FE_28(H, C, a1, ...) H(C, a1) PTO_FE_27(H, C, __VA_ARGS__)
#define PTO_FE_29(H, C, a1, ...) H(C, a1) PTO_FE_28(H, C, __VA_ARGS__)
#define PTO_FE_30(H, C, a1, ...) H(C, a1) PTO_FE_29(H, C, __VA_ARGS__)
#define PTO_FE_31(H, C, a1, ...) H(C, a1) PTO_FE_30(H, C, __VA_ARGS__)
#define PTO_FE_32(H, C, a1, ...) H(C, a1) PTO_FE_31(H, C, __VA_ARGS__)

#define PTO_FOR_EACH_OPCODE(HANDLER, OPCLASS, ...) \
    PTO_PP_CAT(PTO_FE_, PTO_PP_NARG(__VA_ARGS__))(HANDLER, OPCLASS, __VA_ARGS__)

// ===== Expand one DEFOP's opcode list into HANDLER(OPCLASS, OPC) cases =====
// DEFOP(OPCLASS, INHERIT(...), OPCODE(...), ATTR(...)...)
// -> PTO_FOR_EACH_OPCODE(HANDLER, OPCLASS, OP1, OP2, ...)
#define PTO_DEFOP_SWITCH(OPCLASS, opcode_token, HANDLER) \
    PTO_FOR_EACH_OPCODE(HANDLER, OPCLASS, PTO_EXPAND_OPCODE_LIST(opcode_token))

// (No cleanup macros here: users may include this header in many TU safely.)
