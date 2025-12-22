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
 * \file dfs_sort_utils.h
 * \brief
 */

#pragma once
#ifndef DFS_SORT_UTILS_H
#define DFS_SORT_UTILS_H
#include <vector>
#include <unordered_set>
#include "interface/function/function.h"
#include "interface/tensor/logical_tensor.h"

namespace npu {
namespace tile_fwk {
class DFSSortUtils {
public:
    static void DFSSortColor(int color,
                             std::vector<std::vector<int>> &inColor,
                             std::vector<std::vector<int>> &outColor,
                             std::unordered_map<int, int> &dfsColorOrder);
};
}
}

#endif