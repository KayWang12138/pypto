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
 * \file utils.h
 * \brief
 */

#pragma once

#include <vector>
#include <string>
#include <map>
#include <memory>
#include <unordered_set>
#include "type.h"

#define MAP_SIZE_( a0,  a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8,  a9, a10, a11, a12, a13, a14, a15, \
                  a16, a17, a18, a19, a20, a21, a22, a23, a24, a25, a26, a27, a28, a29, a30, a31, v, ...) v
#define MAP_SIZE(...) MAP_SIZE_(__VA_ARGS__, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, \
                                             16, 15, 14, 13, 12, 11, 10,  9,  8,  7,  6,  5,  4,  3,  2,  1)

#define MAP_1(fn, a0)  fn(a0)
#define MAP_2(fn, a0, ...)  fn(a0) MAP_1(fn, __VA_ARGS__)
#define MAP_3(fn, a0, ...)  fn(a0) MAP_2(fn, __VA_ARGS__)
#define MAP_4(fn, a0, ...)  fn(a0) MAP_3(fn, __VA_ARGS__)
#define MAP_5(fn, a0, ...)  fn(a0) MAP_4(fn, __VA_ARGS__)
#define MAP_6(fn, a0, ...)  fn(a0) MAP_5(fn, __VA_ARGS__)
#define MAP_7(fn, a0, ...)  fn(a0) MAP_6(fn, __VA_ARGS__)
#define MAP_8(fn, a0, ...)  fn(a0) MAP_7(fn, __VA_ARGS__)

namespace pto {

} // namespace pto
