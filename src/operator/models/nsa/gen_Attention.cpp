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
 * \file gen_Attention.cpp
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
#include "gen_Attention.h"

using namespace npu::tile_fwk;

namespace npu::tile_fwk {
void GenAttention(Tensor &cmpAtten, Tensor &selAtten, Tensor &winAtten, Tensor &gatingScore, Tensor &attentionOut) {
    int nDimSize = cmpAtten->shape[2];
    int dDimSize = cmpAtten->shape[3];
    int tileB = 8;
    int tileS = 1;
    FUNCTION("main", FunctionType::DYNAMIC, {cmpAtten, selAtten, winAtten, gatingScore}, {attentionOut}) {
        SymbolicScalar bDimSize = GetInputShapeDim(cmpAtten, 0);
        SymbolicScalar sDimSize = GetInputShapeDim(cmpAtten, 1);
        SymbolicScalar bLoop = bDimSize / tileB;
        SymbolicScalar sLoop = sDimSize / tileS;
        DataType dType = cmpAtten->Datatype();
        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(bLoop)) {
            SymbolicScalar bOffset = bIdx * tileB;
            SymbolicScalar actualBSize = std::min(tileB, (bDimSize - bIdx * tileB));
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(sLoop)) {
                SymbolicScalar sOffset = sIdx * tileS;
                std::vector<SymbolicScalar> outOffset = {bOffset, sOffset, 0, 0};
                SymbolicScalar actualsSize = std::min(tileS, (sDimSize - sIdx * tileS));
                Program::GetInstance().GetTileShape().SetVecTileShapes(1, 1, NUM_16, dDimSize);
                auto cmpAttenTile = DViewPad(cmpAtten, {tileB, tileS, nDimSize, dDimSize},
                    {actualBSize, actualsSize, nDimSize, dDimSize}, {bOffset, sOffset, 0, 0});
                auto selAttenTile = DViewPad(selAtten, {tileB, tileS, nDimSize, dDimSize},
                    {actualBSize, actualsSize, nDimSize, dDimSize}, {bOffset, sOffset, 0, 0});
                auto winAttenTile = DViewPad(winAtten, {tileB, tileS, nDimSize, dDimSize},
                    {actualBSize, actualsSize, nDimSize, dDimSize}, {bOffset, sOffset, 0, 0});
                auto cmpAttenFP32Tile = Cast(cmpAttenTile, DT_FP32);
                auto selAttenFP32Tile = Cast(selAttenTile, DT_FP32);
                auto winAttenFP32Tile = Cast(winAttenTile, DT_FP32);
                Program::GetInstance().GetTileShape().SetVecTileShapes(1, 1, nDimSize, NUM_3);
                auto gatingScoreTile = DViewPad(gatingScore, {tileB, tileS, nDimSize, NUM_3},
                    {actualBSize, actualsSize, nDimSize, NUM_3}, {bOffset, sOffset, 0, 0});
                auto gatingScoreFP32 = Cast(gatingScoreTile, DT_FP32);
                auto cmpWeight = View(gatingScoreFP32, {tileB, tileS, nDimSize, 1}, {0, 0, 0, 0});
                auto selWeight = View(gatingScoreFP32, {tileB, tileS, nDimSize, 1}, {0, 0, 0, 1});
                auto winWeight = View(gatingScoreFP32, {tileB, tileS, nDimSize, 1}, {0, 0, 0, 2});
                Program::GetInstance().GetTileShape().SetVecTileShapes(1, 1, NUM_16, dDimSize);
                auto mulCmp = Mul(cmpAttenFP32Tile, cmpWeight);
                auto mulSel = Mul(selAttenFP32Tile, selWeight);
                auto mulWin = Mul(winAttenFP32Tile, winWeight);
                auto addCmpSel = Add(mulCmp, mulSel);
                auto outFP32 = Add(addCmpSel, mulWin);
                Program::GetInstance().GetTileShape().SetVecTileShapes(1, 1, NUM_16, dDimSize);
                auto attentionOutTile = Cast(outFP32, dType, CAST_RINT);
                Assemble(attentionOutTile, outOffset, attentionOut);
            }
        }
    }
}

} // namespace tile_fwk
