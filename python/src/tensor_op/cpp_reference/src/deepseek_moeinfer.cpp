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
#include "interface/function/function.h"
#include "interface/configs/config_manager.h"


namespace npu::tile_fwk {

constexpr float F_1 = 1.0;
constexpr float F_NEGA_1 = -1.0;
constexpr int NUM_32 = 32;
constexpr int NUM_64 = 64;
constexpr int NUM_128 = 128;
constexpr int NUM_2048 = 2048;
constexpr int NUM_7168 = 7168;

void DynamicFFN(const Tensor &hiddenStates, const Tensor &ffnWeight1, const Tensor &ffnWeight2,
                const Tensor &ffnWeight3, Tensor &out, int basicBatch)
{
    int h = hiddenStates.GetShape(1);
    if (basicBatch == 0) {
        throw std::invalid_argument("basic_batch is zero");
    }
    FUNCTION("main", {hiddenStates, ffnWeight1, ffnWeight2, ffnWeight3}, {out}) {
        LOOP("L0", FunctionType::DYNAMIC_LOOP, loopIdx,
                    LoopRange(CeilDiv(GetInputShape(hiddenStates, 0), basicBatch))) {
            SymbolicScalar batchIdx = basicBatch * loopIdx;
            auto hiddenStatesTemp = View(hiddenStates, {basicBatch, h}, {batchIdx, 0});
            auto castRes = Cast(hiddenStatesTemp, DataType::DT_FP16);
            auto gate = Matrix::Matmul(DataType::DT_FP32, castRes, ffnWeight1);
            auto swish = MulS(gate, Element(DataType::DT_FP32, F_NEGA_1));
            swish = Exp(swish);
            swish = AddS(swish, Element(DataType::DT_FP32, F_1));
            swish = Div(gate, swish);

            auto up = Matrix::Matmul(DataType::DT_FP32, castRes, ffnWeight2);
            swish = Mul(swish, up);
            auto swishFp16 = Cast(swish, DataType::DT_FP16);

            // down_proj
            auto mlpRes = Matrix::Matmul<false, true>(DataType::DT_FP32, swishFp16, ffnWeight3);
            Assemble(mlpRes, {batchIdx, 0}, out);
        }
    }
}

void DynamicFFNQuant(const Tensor &hiddenStatesQuant, const Tensor &hiddenStatesScale, const Tensor &ffnWeight1,
                     const Tensor &ffnWeight2, const Tensor &ffnWeight3, const Tensor &ffnScale1,
                     const Tensor &ffnScale2, const Tensor &ffnScale3, Tensor &out, int basicBatch)
{
    int h = hiddenStatesQuant.GetShape(1);
    if (basicBatch == 0) {
        throw std::invalid_argument("basic_batch is zero");
    }
    FUNCTION("main", {hiddenStatesQuant, hiddenStatesScale, ffnWeight1, ffnWeight2, ffnWeight3, ffnScale1,
             ffnScale2, ffnScale3}, {out}) {
        LOOP("L0", FunctionType::DYNAMIC_LOOP, loopIdx,
                    LoopRange(CeilDiv(GetInputShape(hiddenStatesQuant, 0), basicBatch))) {
            SymbolicScalar batchIdx = basicBatch * loopIdx;

            auto castRes = View(hiddenStatesQuant, {basicBatch, h}, {batchIdx, 0});
            auto castResScale = View(hiddenStatesScale, {basicBatch, 1}, {batchIdx, 0});
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
            Assemble(res, {batchIdx, 0}, out);
        }
    }
}

void TestDynamicFFN()
{
    config::SetHostOption(ONLY_CODEGEN, true);

    TileShape::Current().SetVecTile(NUM_32, NUM_128);
    TileShape::Current().SetCubeTile({NUM_32, NUM_32}, {NUM_128, NUM_128}, {NUM_128, NUM_128});

    constexpr int batchSize = NUM_64;
    constexpr int sequence = 1;
    constexpr int h = NUM_7168;
    constexpr int expertDim = NUM_2048;
    constexpr int bs = batchSize * sequence;
    constexpr int basicBatch = NUM_32;

    std::vector<int64_t> hiddenStatesShape{bs, h};
    std::vector<int64_t> weightShape{h, expertDim};
    std::vector<int64_t> outShape{bs, h};

    Tensor hiddenStates(DT_FP32, hiddenStatesShape, "hiddenStates");
    Tensor ffnWeight1(DT_FP16, weightShape, "weightShape1");
    Tensor ffnWeight2(DT_FP16, weightShape, "weightShape2");
    Tensor ffnWeight3(DT_FP16, weightShape, "weightShape3");
    Tensor ffnOut(DT_FP32, outShape, "ffnout");

    DynamicFFN(hiddenStates, ffnWeight1, ffnWeight2, ffnWeight3, ffnOut, basicBatch);
}

void TestDynamicFFNQuant()
{
    config::SetHostOption(ONLY_CODEGEN, true);
    TileShape::Current().SetVecTile(NUM_32, NUM_128);
    TileShape::Current().SetCubeTile({NUM_32, NUM_32}, {NUM_128, NUM_128}, {NUM_128, NUM_128});

    constexpr int batchSize = NUM_32;
    constexpr int sequence = 1;
    constexpr int h = NUM_7168;
    constexpr int expertDim = NUM_2048;
    constexpr int bs = batchSize * sequence;
    constexpr int basicBatch = NUM_32;

    std::vector<int64_t> hiddenStatesShape{bs, h};
    std::vector<int64_t> weightShape{h, expertDim};
    std::vector<int64_t> outShape{bs, h};

    Tensor hiddenStates(DT_INT8, hiddenStatesShape, "hiddenStates");
    Tensor hiddenStatesScale(DT_FP32, {bs, 1}, "hiddenStatesScale");
    Tensor ffnWeight1(DT_INT8, weightShape, "ffnWeight1", TileOpFormat::TILEOP_NZ);
    Tensor ffnWeight2(DT_INT8, weightShape, "ffnWeight2", TileOpFormat::TILEOP_NZ);
    Tensor ffnWeight3(DT_INT8, weightShape, "ffnWeight3", TileOpFormat::TILEOP_NZ);
    Tensor ffnScale1(DT_FP32, {1, expertDim}, "ffnScale1");
    Tensor ffnScale2(DT_FP32, {1, expertDim}, "ffnScale2");
    Tensor ffnScale3(DT_FP32, {1, h}, "ffnScale3");
    Tensor ffnOut(DT_FP32, outShape, "ffnout");

    DynamicFFNQuant(hiddenStates, hiddenStatesScale, ffnWeight1, ffnWeight2, ffnWeight3, ffnScale1, ffnScale2,
        ffnScale3, ffnOut, basicBatch);
}
} // namespace npu::tile_fwk