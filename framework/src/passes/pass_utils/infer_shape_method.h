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
 * \file infer_shape_method.h
 * \brief
 */

#pragma once
#ifndef INFER_SHAPE_METHOD_H
#define INFER_SHAPE_METHOD_H
#include <map>
#include <vector>
#include "interface/operation/op_infer_shape_impl.h"
#include "interface/function/function.h"

namespace npu {
namespace tile_fwk {
class InferShapeMethod {
public:
    InferShapeMethod() = default;
    ~InferShapeMethod() = default;
    Status InferShape(Function& function);

private:
    Status BuildGraph(Function& function, std::vector<Operation*>& opList,
                     std::vector<std::vector<size_t>>& opInGraph, std::vector<std::vector<size_t>>& opOutGraph);
};
} // namespace tile_fwk
} // namespace npu
#endif