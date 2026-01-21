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
 * \file conv_operation_impl.cpp
 * \brief
 */

#include "interface/configs/config_manager.h"
#include "interface/inner/pre_def.h"
#include "interface/operation/operation.h"
#include "interface/operation/operation_common.h"
#include "interface/program/program.h"
#include "interface/tensor/logical_tensor.h"
#include "interface/utils/common.h"
#include "interface/utils/log.h"
#include "interface/utils/operator_tracer.h"
#include "operation_impl.h"
#include "tilefwk/data_type.h"
#include "tilefwk/tile_shape.h"

namespace npu {
namespace tile_fwk {
namespace Conv {

void CheckConvOperands(DataType outType, const Tensor &operand1, const Tensor &operand2, const Tensor &operand3) {
    // todo
}

Tensor ConstructTensorGraph(DataType dataType, const Tensor &inputTensor, const Tensor &weightTensor, const Tensor &biasTensor,
    const Tensor &gmMatrix, const std::vector<int> &strides, const std::vector<int> &paddings, const std::vector<int> &dilations,
    const int groups) {
    
    // infer hout, wout
    int batchOut = 1;
    int Ci1 = 1;
    int hOut = 1;
    int wOut = 1;
    int C0 = 16;
    Tensor resTensor(dataType, {batchOut, Ci1, hOut, wOut, C0}, "TensorC");
    return resTensor;
}

Tensor Conv(DataType outType, const Tensor &inputTensor, const Tensor &weightTensor, const Tensor &biasTensor,
            const std::vector<int> &strides, const std::vector<int> &paddings, const std::vector<int> &dilations,
            const int groups) {
    CheckConvOperands(outType, inputTensor, weightTensor, biasTensor);
    // auto &conveTile = TileShape::Current().GetConvTile();
    return ConstructTensorGraph(outType, inputTensor, weightTensor, biasTensor, Tensor(), strides, paddings, dilations, groups);
}

} //namespace Conv
}
}