/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file topo_program.h
 * \brief
 */

#pragma once
#ifndef TOPO_PROGRAM_H
#define TOPO_PROGRAM_H
#include <vector>
#include <queue>
#include "interface/operation/op_infer_shape_impl.h"
#include "interface/function/function.h"
#include "interface/tensor/logical_tensor.h"

namespace npu {
namespace tile_fwk {
class TopoProgramUtils{
public:
    static void TopoProgram(const std::vector<Operation*>& opList,
                            const std::vector<std::vector<size_t>>& opInGraph,
                            const std::vector<std::vector<size_t>>& opOutGraph,
                            bool isParamIndex);
};
}
}
#endif