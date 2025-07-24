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
    OperatorChecker checker;

    auto tRowmax = RowMaxExpand(operand);
    auto tSub = Sub(operand, tRowmax);
    auto tExp = Exp(tSub);
    auto tEsum = RowSumExpand(tExp);
    auto tSoftmax = Div(tExp, tEsum);

    return tSoftmax;
}

Tensor SoftmaxNew(const Tensor &operand) {
    OperatorChecker checker;
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
} // namespace npu::tile_fwk