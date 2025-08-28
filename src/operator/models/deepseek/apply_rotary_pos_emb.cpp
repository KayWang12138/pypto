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
 * \file apply_rotary_pos_emb.cpp
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
using namespace npu::tile_fwk;

namespace npu::tile_fwk {
Tensor RoPEInputCast(const Tensor &input) {
    auto inputDtype = input->Datatype();
    if (inputDtype == DataType::DT_FP32) { // fp32，不需要进行cast
        return input;
    }

    // RoPE: bf16->fp32
    return Cast(input, DataType::DT_FP32);
}

Tensor RotateHalf(const Tensor &input) {
    auto shape = input->shape;
    auto shapeSize = shape.size();
    assert(shapeSize >= 1 && "rope rotate_half input dim less than 1");
    assert(shape[shapeSize - 1] % NUM2 == 0 && "rope rotate_half last dim shape is even.");

    shape[shapeSize - 1] /= NUM2;
    std::vector<int64_t> offset1(shapeSize, 0);
    std::vector<int64_t> offset2(shapeSize, 0);
    offset2[shapeSize - 1] = shape[shapeSize - 1];

    // x1 = [..., : x.shape[-1] // 2]
    // x2 = [..., x.shape[-1] // 2 :]
    Tensor x1 = View(input, shape, offset1);
    Tensor x2 = View(input, shape, offset2);

    // cat((-x2, x1), -1)
    return Concat(
        {MulS(x2, Element(x2->Datatype(), -1.0)), AddS(x1, Element(x1->Datatype(), 0.0))}, -1); // x1 add 0, 规避pass view+assemble未翻译registor_copy的问题
}

void ApplyRotaryPosEmbV2(const Tensor &q, const Tensor &k, const Tensor &cos, const Tensor &sin, Tensor &qEmbed,
    Tensor &kEmbed, const int unsqueezeDim, const RoPETileShapeConfigNew &ropeTileShapeConfig) {
    auto outputDtype = qEmbed->Datatype();

    // q/k仅支持四维，cos/sin仅支持san维
    assert(q->shape.size() == SHAPE_DIM4 && k->shape.size() == SHAPE_DIM4 && cos->shape.size() == SHAPE_DIM3 &&
           sin->shape.size() == SHAPE_DIM3);

    assert(!ropeTileShapeConfig.threeDimsTileShape.empty() && "rope ThreeDims Tile need to set!");
    assert(!ropeTileShapeConfig.fourDimsTileShapeQ.empty() && "rope FourDimsQ Tile need to set!");
    assert(!ropeTileShapeConfig.fourDimsTileShapeK.empty() && "rope FourDimsK Tile need to set!");
    assert(!ropeTileShapeConfig.fiveDimsTileShape.empty() && "rope FiveDims Tile need to set!");

    Program::GetInstance().GetTileShape().SetVecTileShapes(ropeTileShapeConfig.fourDimsTileShapeQ); // 设置四维Tile
    auto castQ = RoPEInputCast(q);                                                        // [b,s,n,qk_d]
    Program::GetInstance().GetTileShape().SetVecTileShapes(ropeTileShapeConfig.fourDimsTileShapeK);
    auto castK = RoPEInputCast(k);

    Program::GetInstance().GetTileShape().SetVecTileShapes(
        ropeTileShapeConfig.threeDimsTileShape); // cos/sin设置san维Tile
    auto castCos = RoPEInputCast(cos);           // [b, s, qk_d]
    auto castSin = RoPEInputCast(sin);

    auto cosUnsqueeze = Unsqueeze(castCos, unsqueezeDim); // [b,1,s,qk_d]
    auto sinUnsqueeze = Unsqueeze(castSin, unsqueezeDim);

    // q=View(q, b,h,s,d//2,2).transpose(4,3).reshape(b,h,s,d)
    // q/k: [b,s,n,qk_d]
    int b = castQ->shape[0];
    int s = castQ->shape[1]; // use h in source code
    int h = castQ->shape[2];
    int d = castQ->shape[NUM_VALUE_3];

    auto qView = Reshape(castQ, {b, s, h, d / 2, 2}); // [b,n,s,qk_d//2,2]
    Program::GetInstance().GetTileShape().SetVecTileShapes(ropeTileShapeConfig.fiveDimsTileShape);
    auto qTrans = Transpose(qView, {NUM_VALUE_3, NUM_VALUE_4}); // [b,n,s,2,qk_d//2]
    auto qReshape = Reshape(qTrans, {b, s, h, d});              // [b,n,s,qk_d]

    // k=View(k, b,h,s,d//2,2).transpose(4,3).reshape(b,h,s,d)
    b = castK->shape[0];
    s = castK->shape[1];
    h = castK->shape[2];
    d = castK->shape[3];

    Program::GetInstance().GetTileShape().SetVecTileShapes(ropeTileShapeConfig.fourDimsTileShapeK);
    auto kView = Reshape(castK, {b, s, h, d / 2, 2});
    Program::GetInstance().GetTileShape().SetVecTileShapes(ropeTileShapeConfig.fiveDimsTileShape);
    auto kTrans = Transpose(kView, {NUM_VALUE_3, NUM_VALUE_4});
    auto kReshape = Reshape(kTrans, {b, s, h, d});

    // q_embed=(q*cos)+(rotare_half(q)*sin)
    // k_embed=(k*cos)+(rotare_half(k)*sin)
    Program::GetInstance().GetTileShape().SetVecTileShapes(ropeTileShapeConfig.fourDimsTileShapeQ);
    qEmbed = Add(Mul(qReshape, cosUnsqueeze), Mul(RotateHalf(qReshape), sinUnsqueeze));
    Program::GetInstance().GetTileShape().SetVecTileShapes(ropeTileShapeConfig.fourDimsTileShapeK);
    kEmbed = Add(Mul(kReshape, cosUnsqueeze), Mul(RotateHalf(kReshape), sinUnsqueeze));

    if (outputDtype != qEmbed->Datatype()) {
        Program::GetInstance().GetTileShape().SetVecTileShapes(ropeTileShapeConfig.fourDimsTileShapeQ);
        qEmbed = Cast(qEmbed, outputDtype);
        Program::GetInstance().GetTileShape().SetVecTileShapes(ropeTileShapeConfig.fourDimsTileShapeK);
        kEmbed = Cast(kEmbed, outputDtype);
    }
}

void ApplyRotaryPosEmb(const Tensor &q, const Tensor &k, const Tensor &cos, const Tensor &sin,
    const Tensor &positionIds, Tensor &qEmbed, Tensor &kEmbed, const int unsqueezeDim,
    const RoPETileShapeConfig &ropeTileShapeConfig) {
    auto outputDtype = qEmbed->Datatype();

    // q/k仅支持四维，cos/sin仅支持两维
    assert(q->shape.size() == SHAPE_DIM4 && k->shape.size() == SHAPE_DIM4 && cos->shape.size() == SHAPE_DIM2 &&
           sin->shape.size() == SHAPE_DIM2);

    assert(!ropeTileShapeConfig.twoDimsTileShape.empty() && "rope TwoDims Tile need to set!");
    assert(!ropeTileShapeConfig.threeDimsTileShape.empty() && "rope ThreeDims Tile need to set!");
    assert(!ropeTileShapeConfig.fourDimsTileShape.empty() && "rope FourDims Tile need to set!");
    assert(!ropeTileShapeConfig.fiveDimsTileShape.empty() && "rope FiveDims Tile need to set!");

    Program::GetInstance().GetTileShape().SetVecTileShapes(ropeTileShapeConfig.fourDimsTileShape); // 设置四维Tile
    auto castQ = RoPEInputCast(q);                                                                       // [b,n,s,qk_d]
    auto castK = RoPEInputCast(k);

    Program::GetInstance().GetTileShape().SetVecTileShapes(
        ropeTileShapeConfig.twoDimsTileShape); // cos/sin设置两维Tile
    auto castCos = RoPEInputCast(cos);         // [s, qk_d]
    auto castSin = RoPEInputCast(sin);

    // cos = cos[position_ids].unsqueeze(unsqueezeDimNum)
    // sin = sin[position_ids].unsqueeze(unsqueezeDimNum)
    Program::GetInstance().GetTileShape().SetVecTileShapes(
        ropeTileShapeConfig.threeDimsTileShape);               // TensorIndex, 设置三维Tile
    auto cosTensorIndexes = TensorIndex(castCos, positionIds); // [s,qk_d],[b,s]->[b,s,qk_d]
    auto sinTensorIndexes = TensorIndex(castSin, positionIds);

    auto cosUnsqueeze = Unsqueeze(cosTensorIndexes, unsqueezeDim); // [b,1,s,qk_d]
    auto sinUnsqueeze = Unsqueeze(sinTensorIndexes, unsqueezeDim);

    // q=View(q, b,h,s,d//2,2).transpose(4,3).reshape(b,h,s,d)
    // q/k: [b,n,s,qk_d]
    int b = castQ->shape[0];
    int h = castQ->shape[1]; // use h in source code
    int s = castQ->shape[2];
    int d = castQ->shape[NUM_VALUE_3];

    Program::GetInstance().GetTileShape().SetVecTileShapes(ropeTileShapeConfig.fourDimsTileShape);
    auto qView = Reshape(castQ, {b, h, s, d / 2, 2}); // [b,n,s,qk_d//2,2]
    Program::GetInstance().GetTileShape().SetVecTileShapes(ropeTileShapeConfig.fiveDimsTileShape);
    auto qTrans = Transpose(qView, {NUM_VALUE_3, NUM_VALUE_4}); // [b,n,s,2,qk_d//2]
    auto qReshape = Reshape(qTrans, {b, h, s, d});              // [b,n,s,qk_d]

    // k=View(k, b,h,s,d//2,2).transpose(4,3).reshape(b,h,s,d)
    b = castK->shape[0];
    h = castK->shape[1];
    s = castK->shape[2];
    d = castK->shape[3];

    Program::GetInstance().GetTileShape().SetVecTileShapes(ropeTileShapeConfig.fourDimsTileShape);
    auto kView = Reshape(castK, {b, h, s, d / 2, 2});
    Program::GetInstance().GetTileShape().SetVecTileShapes(ropeTileShapeConfig.fiveDimsTileShape);
    auto kTrans = Transpose(kView, {NUM_VALUE_3, NUM_VALUE_4});
    auto kReshape = Reshape(kTrans, {b, h, s, d});

    // q_embed=(q*cos)+(rotare_half(q)*sin)
    // k_embed=(k*cos)+(rotare_half(k)*sin)
    Program::GetInstance().GetTileShape().SetVecTileShapes(ropeTileShapeConfig.fourDimsTileShape);
    qEmbed = Add(Mul(qReshape, cosUnsqueeze), Mul(RotateHalf(qReshape), sinUnsqueeze));
    kEmbed = Add(Mul(kReshape, cosUnsqueeze), Mul(RotateHalf(kReshape), sinUnsqueeze));

    if (outputDtype != qEmbed->Datatype()) {
        qEmbed = Cast(qEmbed, outputDtype);
        kEmbed = Cast(kEmbed, outputDtype);
    }
}
} // namespace npu::tile_fwk
