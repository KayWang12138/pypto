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
 * \file deepseek_moeinfer.cpp
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
#include "interface/tensor/tensormap.h"
#include "interface/configs/config_manager.h"
#include "interface/configs/config_storage.h"
#include "interface/utils/common.h"
#include "interface/utils/id_gen.h"
#include "interface/utils/log.h"
#include "page_attention.h"

using namespace npu::tile_fwk;

namespace npu::tile_fwk {

constexpr float F_1 = 1.0;
constexpr float F_NEGA_1 = -1.0;

    void DynamicFFN(const Tensor &hiddenStates, const Tensor &ffnWeight1, const Tensor &ffnWeight2, const Tensor &ffnWeight3,
                    Tensor &out, int BASIC_BATCH) {
    const int H = hiddenStates.GetShape()[1];
    FUNCTION("main", FunctionType::DYNAMIC, {hiddenStates, ffnWeight1, ffnWeight2, ffnWeight3}, {out}) {
        LOOP("L0", FunctionType::DYNAMIC_LOOP, loopIdx, LoopRange(GetInputShapeDim(hiddenStates, 0) / BASIC_BATCH)) {
            SymbolicScalar batchIdx = BASIC_BATCH * loopIdx;
            auto hiddenStatesTemp = DView(hiddenStates, {BASIC_BATCH, H}, {batchIdx, 0});
            auto castRes = Cast(hiddenStatesTemp, DataType::DT_FP16);
            auto gate = Matrix::Matmul(DataType::DT_FP32, castRes, ffnWeight1);
            auto swish = MulS(gate, Element(DataType::DT_FP32, F_NEGA_1));
            swish = Exp(swish);
            swish = AddS(swish, Element(DataType::DT_FP32, F_1));
            swish = Div(gate, swish);

            auto up = Matrix::Matmul(DataType::DT_FP32, castRes, ffnWeight2);
            swish = Mul(swish, up);
            auto swish_fp16 = Cast(swish, DataType::DT_FP16);

            // down_proj
            auto mlpRes = Matrix::Matmul<false, true>(DataType::DT_FP32, swish_fp16, ffnWeight3);
            DAssemble(mlpRes, {batchIdx, 0}, out);
        }
    }
    }

    void DynamicFFNQuant(const Tensor &hiddenStatesQuant, const Tensor &hiddenStatesScale,const Tensor &ffnWeight1, const Tensor &ffnWeight2, const Tensor &ffnWeight3,
                    const Tensor &ffnScale1, const Tensor &ffnScale2, const Tensor &ffnScale3,
                    Tensor &out, int BASIC_BATCH) {
    const int H = hiddenStatesQuant.GetShape()[1];
    FUNCTION("main", FunctionType::DYNAMIC, {hiddenStatesQuant, hiddenStatesScale, ffnWeight1, ffnWeight2, ffnWeight3, ffnScale1, ffnScale2, ffnScale3}, {out}) {
        LOOP("L0", FunctionType::DYNAMIC_LOOP, loopIdx, LoopRange(GetInputShapeDim(hiddenStatesQuant, 0) / BASIC_BATCH)) {
            SymbolicScalar batchIdx = BASIC_BATCH * loopIdx;

            auto castRes = DView(hiddenStatesQuant, {BASIC_BATCH, H}, {batchIdx, 0});
            auto castResScale = DView(hiddenStatesScale, {BASIC_BATCH, 1}, {batchIdx, 0});
            auto gateInt32 = Matrix::Matmul(DataType::DT_INT32, castRes, ffnWeight1);

            // dequant: int32 -> fp32 -> *scale -> fp16/bf16
            auto gateTmpFp32 = Cast(gateInt32, DataType::DT_FP32);
            auto gateTmpDequantPerToken = Mul(gateTmpFp32, castResScale);
            auto gate = Mul(gateTmpDequantPerToken, ffnScale1);

            // swish: x / (1 + e^(-x))
            auto swish = MulS(gate, Element(DataType::DT_FP32, F_NEGA_1));
            swish = Exp(swish);
            swish = AddS(swish, Element(DataType::DT_FP32, F_1));
            swish = Div(gate, swish);

            auto upInt32 = Matrix::Matmul(DataType::DT_INT32, castRes, ffnWeight2);
            // upProj
            auto upTmpFp32 = Cast(upInt32, DataType::DT_FP32);
            auto upTmpDequantPerToken = Mul(upTmpFp32, castResScale);
            auto up = Mul(upTmpDequantPerToken, ffnScale2);

            swish = Mul(swish, up);

            // downProj
            auto swishQuantRes = Quant(swish); // int8
            Tensor swishRes = std::get<0>(swishQuantRes);
            Tensor swishScale = std::get<1>(swishQuantRes);

            Tensor resInt32 = Matrix::Matmul<false, true>(DataType::DT_INT32, swishRes, ffnWeight3);
            auto resTmpFp32 = Cast(resInt32, DataType::DT_FP32);
            auto resTmpDequantPerToken = Mul(resTmpFp32, swishScale);
            auto res = Mul(resTmpDequantPerToken, ffnScale3);
            DAssemble(res, {batchIdx, 0}, out);
        }
    }
    }

} // namespace npu::tile_fwk