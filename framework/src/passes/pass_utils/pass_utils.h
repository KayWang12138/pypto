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
 * \file pass_utils.h
 * \brief
 */

#pragma once

#include "interface/operation/operation.h"
#include "interface/tensor/logical_tensor.h"

namespace npu::tile_fwk {
class PassUtils {
public:
    static void RelinkOperationInput(Operation *op, const size_t inputIndex, const Operation *targetOp,
                                     const size_t outputIndex);

    static bool IsContinuous(const std::vector<std::shared_ptr<LogicalTensor>> &tensors);

    static bool IsCVMixPlatform();
};

class CommonUtils {
public:
    template <typename Container>
    static std::string ContainerToStr(const Container &container, const std::string &delimiter = ", ") {
        if (container.empty()) {
            return "{}";
        }
        std::ostringstream oss;
        oss << "{";
        auto it = container.begin();
        oss << *it;
        std::for_each(std::next(it), container.end(),
            [&oss, &delimiter](const auto& elem) {
                oss << delimiter << elem;
            });
        oss << "}";
        return oss.str();
    }

    // 判断 Tensor 的 shape 是否存在-1
    static bool ContainsNegativeOne(const Shape &shape) {
        return std::any_of(shape.begin(), shape.end(), [](int64_t val) { return val == -1; });
    }

    // Number of Elements, 用来计算给定（tensor的）shape的总元素数量
    static int64_t Numel(const Shape &shape) {
        if (shape.empty())
            return 0;
        int64_t numel = 1;
        for (int64_t num : shape) {
            numel *= num;
        }
        return numel;
    }

    static std::unordered_map<MemoryType, int64_t> GetLocalMemorySize();
};
}
