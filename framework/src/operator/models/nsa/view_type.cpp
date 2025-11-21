/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file view_type.cpp
 * \brief
 */

#include "interface/function/function.h"
#include "tilefwk/tilefwk_op.h"
#include "tilefwk/tensor.h"
#include "interface/tensor/logical_tensor.h"
#include "interface/utils/common.h"
#include "interface/configs/config_manager.h"
#include "operator/models/deepseek/deepseek_mla.h"
#include "gen_gated_score.h"

using namespace npu::tile_fwk;

namespace npu::tile_fwk {

void ViewTypeFunc(const Tensor &x, Tensor &result, DataType dstDtype) {
    FUNCTION("VIEWTYPE", {x}, {result}) {
        int n = x.GetShape()[2];
        int tileN = n;
        SymbolicScalar nLoop = n / tileN;
        
        LOOP("LOOP_L0_nIdx_view_type", FunctionType::DYNAMIC_LOOP, nIdx, LoopRange(0, nLoop, 1)) {
            SymbolicScalar nOffset = nIdx * tileN;
            TileShape::Current().SetVecTile({2, 4, n});
            auto resultView = View(x, dstDtype);
            auto resultRes = resultView;

            if(dstDtype == DT_FP32) {
                resultRes = Add(resultView, Element(dstDtype, float(0)));
            }
            Assemble(resultRes, {0, 0, nOffset}, result);
        }
    }
}

void ViewTypeCastFunc(const Tensor &x, Tensor &result, DataType dstDtype, DataType castDtype) {
    FUNCTION("VIEWTYPE", {x}, {result}) {
        int n = x.GetShape()[2];
        int tileN = n;
        SymbolicScalar nLoop = n / tileN;
        
        LOOP("LOOP_L0_nIdx_view_type", FunctionType::DYNAMIC_LOOP, nIdx, LoopRange(0, nLoop, 1)) {
            SymbolicScalar nOffset = nIdx * tileN;
            TileShape::Current().SetVecTile({2, 4, n});
            auto resultView = View(x, dstDtype);
            auto resultCast = Cast(resultView, castDtype);
            auto resultRes = resultCast;

            if(castDtype == DT_FP32) {
                resultRes = Add(resultCast, Element(castDtype, float(0)));
            }
            Assemble(resultRes, {0, 0, nOffset}, result);
        }
    }
}

} // namespace tile_fwk