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
 * \file tile_graph.h
 * \brief
 */

#pragma once

#include "utils_defop.h"
#include "tile_graph_base.h"

namespace pto {

enum CastMode {
    CAST_NONE = 0,
    CAST_RINT = 1,  // round to nearest, tie to even
    CAST_ROUND = 2, // round to nearest, tie away from zero
    CAST_FLOOR = 3, // round to minus infinity
    CAST_CEIL = 4,  // round to positive infinity
    CAST_TRUNC = 5, // round to zero
    CAST_ODD = 6   // round to odd (Von Neumann rounding)
};

#define DEFOP DEFOP_CLASS
#include "tile_graph.def"
#undef DEFOP

} // namespace pto