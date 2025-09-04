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
 * \file soft_max.cpp
 * \brief
 */

#include "interface/operation/operation_impl.h"
#include "interface/operation/operation.h"
#include "interface/function/function.h"
#include "tilefwk/tensor.h"
#include "interface/tensor/logical_tensor.h"
#include "interface/tensor/raw_tensor.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"
#include "interface/tensor/tensormap.h"
#include "interface/configs/config_manager.h"
#include "interface/configs/config_storage.h"
#include "interface/utils/common.h"
#include "interface/utils/id_gen.h"
#include "interface/utils/log.h"
using namespace npu::tile_fwk;

namespace npu::tile_fwk {
Tensor Softmax(const Tensor &operand) {
    auto tRowmax = RowMaxExpand(operand);
    auto tSub = Sub(operand, tRowmax);
    auto tExp = Exp(tSub);
    auto tEsum = RowSumExpand(tExp);
    auto tSoftmax = Div(tExp, tEsum);

    return tSoftmax;
}

Tensor SoftmaxNew(const Tensor &operand) {
    auto inputDtype = operand->Datatype();
    Tensor castOperand = operand;
    if (inputDtype != DataType::DT_FP32) {
        castOperand = Cast(operand, DataType::DT_FP32);
    }
    // M=rowMax(xi)
    auto rowmax = RowMaxSingle(castOperand);
    // S=rowSum(exp(xi-M))
    auto sub = Sub(castOperand, rowmax);
    auto exp = Exp(sub);
    auto esum = RowSumSingle(exp);
    // softmax(zi)=exp(xi-M)/S
    auto softmax = Div(exp, esum);
    if (inputDtype != softmax->Datatype()) {
        softmax = Cast(softmax, inputDtype);
    }
    return softmax;
}

void SoftmaxDynamicCompute(Tensor &input, Tensor &output) {
    // input_shape: [b, n1, n2, d] fp16/bf16
    // int b = input->shape[0]; batch轴动态
    SymbolicScalar b = GetInputShapeDim(input, 0);
    int n1 = input->shape[1];
    int n2 = input->shape[2];
    int dim = input->shape[3];
    int tileB = 1;
    SymbolicScalar bLoop = b / tileB;
    LOOP("SOFTMAX_LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bLoop, 1), {}, true) {
        SymbolicScalar bOffset = bIdx * tileB;
        std::vector<SymbolicScalar> outOffset = {bOffset, 0, 0, 0};
        TileShape::Current().SetVecTile({1, 1, 32, 256});
        auto inputView = View(input, {tileB, n1, n2, dim}, {bOffset, 0, 0, 0});
        auto outputView = SoftmaxNew(inputView);
        Assemble(outputView, outOffset, output);
    }
}

void SoftmaxDynamic(Tensor &input, Tensor &output) {
    FunctionConfig funConfig;
    FUNCTION("SOFTMAX_DYNAMIC", funConfig, {input}, {output}) {
        SoftmaxDynamicCompute(input, output);
    }
}
} // namespace npu::tile_fwk