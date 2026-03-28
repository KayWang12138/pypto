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
 * \file tile_tensor_litenpu.h
 * \brief
 */

#ifndef TILE_TENSOR_LITENPU_H
#define TILE_TENSOR_LITENPU_H

#include <string>
#include <vector>
#include "codegen/symbol_mgr/codegen_symbol.h"

namespace npu::tile_fwk {

// e.g.
// UBTileTensorFP32Dim2 ubTile_0((__ubuf__ float*)UB_S0_E16384, DimLayout2(Shape<int, int>(sym_18_dim_0, sym_18_dim_1),
// Stride<int, int>(64, 1)));
struct TileTensorLiteNPU : TileTensor {
    TileTensorLiteNPU(bool pIsConstant, int pMagic, int pDim, DataType pDtype, BufferType pBufType, std::string pBufVar,
        std::string pUsingType, std::string pTensorName, std::vector<std::string> pShape,
        std::vector<std::string> pStride, std::vector<int64_t> pRawShape, std::vector<int64_t> pLocalBufOffset,
        ShapeInLoop pShapeInLoop)
        : TileTensor(pIsConstant, pMagic, pDim, pDtype, pBufType, pBufVar, pUsingType, pTensorName, pShape, pStride,
              pRawShape, pLocalBufOffset, pShapeInLoop) {}
    TileTensorLiteNPU() = default;
    virtual ~TileTensorLiteNPU() override = default;

    /*  e.g.
        ((__ubuf__ float*)UB_S0_E16384,
        Layout2Dim(Shape2Dim<int, int>(sym_18_dim_0, sym_18_dim_1), Stride2Dim<int, int>(64, 1)));
    */
    std::string GenInitParamLiteNPU() const {
        std::ostringstream oss;
        std::vector<std::string> params;
        // ddr: e.g. (__gm__ float*)GET_PARAM_ADDR(...)
        // local: e.g. (uint64_t)UB_S0_E16384
        oss << "(uint64_t)";
        // cast local buffer pointer to uint64_t to adapt TileTensor mode
        int64_t linearOffset{0};
        if (!localBufOffset.empty() && shapeInLoop.loopDepth == 0) {
            // only calc linear offset in the outermost loop, tensor in loop use base addr from tensor out of loop
            linearOffset = CalcLinearOffset(rawShape, localBufOffset);
        }
        if (linearOffset != 0) {
            // append linear offset, e.g. UBTileTensorFP32Dim2_1 ubTensor_1((uint64_t)((float *)UB_S0_E4096 + 32))
            oss << "((" << DataType2CCEStr(dtype) << " *)" << bufVar << " + " << linearOffset << ")";
        } else {
            oss << bufVar;
        }

        return "(" + oss.str() + ")";
    }

    std::string ToString() const override {
        std::cout<<"called TileTensorLiteNPU ToString...\n";
        std::ostringstream oss;
        oss << usingType << " " << tensorName << GenInitParamLiteNPU() << STMT_END;
        return oss.str();
    }
};

struct TileTensorUsingLiteNPU : TileTensorUsing {
    TileTensorUsingLiteNPU(bool pIsConstant, DataType pDtype, BufferType pBufType, int pDim,
        std::vector<int64_t> pOriginShape, std::vector<int64_t> pRawShape)
        : TileTensorUsing(pIsConstant, pDtype, pBufType, pDim, pOriginShape, pRawShape) {}
    TileTensorUsingLiteNPU() = default;
    virtual ~TileTensorUsingLiteNPU() = default;

    // static shape: e.g. "TileTensor<float, LocalLayout4Dim<16, 16>, Hardware::UB>"
    std::string ToString() const override {
        std::ostringstream ss;
        ss << TILE_TENSOR << "<";
        ss << DataType2CCEStr(dtype) << ", ";
        ss << GetLayoutType(bufType, dim, isConstant);
        ss << GetLayoutParams();
        ss << ", " << SCOPE_NAMESPACE << "::" << BUFFER_TYPE_TO_PREFIX.at(bufType) << ">;\n";
        return ss.str();
    }
};
} // namespace npu::tile_fwk

#endif // TILE_TENSOR_LITENPU_H
