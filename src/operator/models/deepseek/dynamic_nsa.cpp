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
 * \file dynamic_mla.cpp
 * \brief
 */

#include "operator/models/deepseek/deepseek_mla.h"
#include "operator/models/deepseek/dynamic_nsa.h"

#include "interface/operation/operation.h"
#include "interface/function/function.h"
#include "tilefwk/tensor.h"
#include "interface/tensor/logical_tensor.h"
#include "interface/utils/common.h"
namespace npu::tile_fwk {
void GenGatedScore(const Tensor &x, const Tensor &gateW1, const Tensor &gateW2, const Tensor &gateSimW1,
    Tensor &gatingScore, Tensor &mm1, Tensor &tempOut, GateMode gateMode) {
    int b = x->shape[0];
    int s = x->shape[1]; // s=1
    int h = x->shape[2];
    int n = gateW2->shape[1] / 3;
    int tileB = b;
    int tileS = s;
    int tileBS = tileB * tileS;
    SymbolicScalar bLoop = b / tileB;
    SymbolicScalar sLoop = s / tileS;
    DataType dType = x->Datatype();
    if (gateMode != standard) {
        return;
    }
    FUNCTION("main", FunctionType::DYNAMIC, {x, gateW1, gateW2, gateSimW1}, {gatingScore, mm1, tempOut}) {
        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bLoop, 1)) {
            LOOP("LOOP_L0_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, sLoop, 1)) {
                Program::GetInstance().GetTileShape().SetVecTileShapes({tileB, tileS, h});
                Program::GetInstance().GetTileShape().SetCubeTileShapes(
                    {tileBS, tileBS}, {NUM_128, NUM_128}, {NUM_128, NUM_128});
                SymbolicScalar bOfs = bIdx * tileB;
                SymbolicScalar sOfs = sIdx * tileS;
                SymbolicScalar bsOfs = bOfs * sOfs;

                auto xReshape = Reshape(x, {b * s, h});
                auto xView = DView(xReshape, {tileBS, h}, {bsOfs, 0});
                auto mm1Res = Matrix::Matmul(dType, xReshape, gateW1);

                Program::GetInstance().GetTileShape().SetVecTileShapes({1, h});
                auto sigmoidRes = Sigmoid(mm1Res);

                auto mm2Res = Matrix::Matmul(dType, sigmoidRes, gateW2);
                Program::GetInstance().GetTileShape().SetVecTileShapes({tileBS, n});

                auto res = Reshape(mm2Res, {tileB, tileS, 3, n});
                Program::GetInstance().GetTileShape().SetVecTileShapes({2, tileS, 3, n});

                res = Transpose(Cast(res, DataType::DT_FP32), {2, 3});

                DAssemble(Cast(res, DataType::DT_FP16), {bOfs, sIdx, 0, 0}, gatingScore);
            }
        }
    }
}

std::vector<Tensor> GenTopkIndices(const Tensor &tmpOut, int s_slc, int actualTopk, int actualValidLen, bool isDyn) {
    std::vector<Tensor> res;
    Program::GetInstance().GetTileShape().SetVecTileShapes({1, s_slc});
    auto view0 = DViewPad(tmpOut, {1, 128}, {1, actualValidLen}, {0, 1});
    if (!isDyn) {
        view0 = View(tmpOut, {1, actualValidLen}, {0, 1});
    }
    Program::GetInstance().GetTileShape().SetVecTileShapes({1, s_slc});
    auto topk_idx = std::get<1>(TopK(view0, 16, -1, true)); // 13
    topk_idx = Cast(topk_idx, DataType::DT_FP32);
    topk_idx = AddS(topk_idx, Element(DT_FP32, 1.0f));
    res.emplace_back(topk_idx);

    topk_idx = DViewPad(topk_idx, {1, 16}, {1, actualTopk}, {0, 0});
    if (!isDyn) {
        topk_idx = View(topk_idx, {1, actualTopk}, {0, 0});
    }
    auto out32 = std::get<0>(TopK(topk_idx, actualTopk, -1, false));
    res.emplace_back(out32);
    return res;
}

std::vector<Tensor> singleTopk(const Tensor &tmpOut, int actualValidLen) {
    std::vector<Tensor> res;
    Program::GetInstance().GetTileShape().SetVecTileShapes({1, 128});
    auto view0 = DViewPad(tmpOut, {1, 128}, {1, actualValidLen}, {0, 1});
    Program::GetInstance().GetTileShape().SetVecTileShapes({1, 128});
    auto topk_idx = std::get<1>(TopK(view0, 16, -1, true));
    topk_idx = Cast(topk_idx, DataType::DT_FP32);
    res.emplace_back(topk_idx);
    return res;
}

void GenSlc(const Tensor &x, Tensor &trans0res, Tensor &reduce0res, Tensor &trans1res, Tensor &reduce1res,
    Tensor &topkInd, Tensor &topkVal, Tensor &out, int actualLen, int l_prime, int d, int front, int near, int topk) {
    int n2 = x->shape[0]; // 1
    assert(n2 == 1);
    int g = x->shape[1];         // 128
    int s_cmp = x->shape[2];     // 511
    int s_slc = (s_cmp + 3) / 4; // 128
    int loop = s_slc;
    int out_loop = l_prime / d;                      // 4
    int actualTopk = topk - (front + near);          // 13
    int actualVaildLen = actualLen - (front + near); // 125

    int tileS2 = s_cmp;
    SymbolicScalar sLoop = s_cmp / tileS2;
    Tensor tmpOut(DataType::DT_FP32, {1, g}, "tmpout");
    Tensor tmpOut1(DataType::DT_FP32, {1, 16}, "tmpout1");
    Tensor tmpTrans2(DataType::DT_FP32, {1, s_cmp, 128}, "trans1");
    FUNCTION(
        "main", FunctionType::DYNAMIC, {x}, {trans0res, reduce0res, trans1res, reduce1res, topkInd, topkVal, out}) {
        LOOP("LOOP_L0_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, sLoop, 1), {}, true) {
            SymbolicScalar sOfs = sIdx * tileS2;
            Program::GetInstance().GetTileShape().SetVecTileShapes({1, 4, s_cmp});
            auto viewer = DView(x, {n2, g, s_cmp}, {0, 0, sOfs});
            auto input32 = Cast(viewer, DataType::DT_FP32); // 1,128,511
            auto tmpTrans = Transpose(input32, {1, 2});     // 1,511,128
            DAssemble(tmpTrans, {0, 0, 0}, tmpTrans2);
            Program::GetInstance().GetTileShape().SetVecTileShapes({1, 16, g});
            trans0res = Cast(tmpTrans2, DataType::DT_FP16);
            Tensor abc(DataType::DT_FP16, {n2, loop, g}, "reduce0");
            for (int i = 0; i < loop; i++) {
                auto maxLen0 = std::min(out_loop, s_cmp - i * out_loop);
                auto view0 = View(tmpTrans, {1, maxLen0, g}, {0, i * out_loop, 0}); // 1,4,128
                auto maxLen1 = std::min(out_loop, s_cmp - i * out_loop - 1);
                Program::GetInstance().GetTileShape().SetVecTileShapes({1, 8, g});
                auto reduce0 = RowSumSingle(view0, 1); // 1,1,128
                if (maxLen1 > 0) {
                    auto view1 = View(tmpTrans, {1, maxLen1, g}, {0, i * out_loop + 1, 0}); // 1,4,128
                    auto reduce1 = RowSumSingle(view1, 1);                                  // 1,1,128
                    auto sum = Add(reduce0, reduce1);                                       // 1,1,128
                    auto sumTmp = Cast(sum, DataType::DT_FP16);
                    DAssemble(sumTmp, {0, i, 0}, abc);
                } else {
                    auto reduceTmp = Cast(reduce0, DataType::DT_FP16);
                    DAssemble(reduceTmp, {0, i, 0}, abc);
                }
            }
            reduce0res = abc;
            auto trans1 = Transpose(Cast(abc, DataType::DT_FP32), {1, 2}); // 1,128,128
            trans1res = Cast(trans1, DataType::DT_FP16);
            Program::GetInstance().GetTileShape().SetVecTileShapes({1, g, 8});
            auto reduce2 = RowSumSingle(trans1, 1); // 1,1,128
            tmpOut = Reshape(reduce2, {1, 128});
            reduce1res = Cast(reduce2, DataType::DT_FP16);
        }
        LOOP("LOOP_topk1", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, 1, 1), {}, true) {
            (void)sIdx;
            config::SetCodeGenConfig(KEY_SUPPORT_DYNAMIC_UNALIGNED, true);
            std::vector<Tensor> res = GenTopkIndices(tmpOut, s_slc, actualTopk, actualVaildLen, true);
            out = res[1];
            topkInd = res[0];
        }
    }
}

void GenTopkIndicesFun(const Tensor &x, Tensor &trans0res, Tensor &reduce0res, Tensor &trans1res, Tensor &reduce1res,
    Tensor &topkInd, Tensor &topkVal, Tensor &out, int actualLen, int front, int near) {
    int s_slc = x->shape[1];                         // 128
    int actualVaildLen = actualLen - (front + near); // 125
    Tensor tmpOut(DataType::DT_FP32, {1, s_slc}, "tmpout");
    Tensor tmpOut1(DataType::DT_FP32, {1, 16}, "tmpout1");

    FUNCTION(
        "main", FunctionType::DYNAMIC, {x}, {trans0res, reduce0res, trans1res, reduce1res, topkInd, topkVal, out}) {
        LOOP("LOOP_topk0", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, 1, 1), {}, true) {
            (void)sIdx;
            Program::GetInstance().GetTileShape().SetVecTileShapes({1, s_slc});
            tmpOut = Cast(x, DT_FP32);
        }
        LOOP("LOOP_topk1", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, 1, 1), {}, true) {
            (void)sIdx;
#define single_topk
#ifdef single_topk
            std::vector<Tensor> res = singleTopk(tmpOut, actualVaildLen);
            topkInd = res[0];
#else
            std::vector<Tensor> res = GenTopkIndices(tmpOut, s_slc, actualTopk, actualVaildLen, isDyn);
            out = res[1];
            topkInd = res[0];
#endif
    }
}
}
} // namespace npu::tile_fwk
