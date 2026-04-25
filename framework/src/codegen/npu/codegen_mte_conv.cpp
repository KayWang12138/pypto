/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file codegen_mte_conv.cpp
 * \brief
 */
#include <iterator>
#include <string>

#include "codegen_op_npu.h"
#include "codegen/symbol_mgr/codegen_symbol.h"
#include "codegen/utils/codegen_utils.h"
#include "tilefwk/error_code.h"
#include "securec.h"

namespace npu::tile_fwk {

std::vector<std::string> CodeGenOpNPU::GetDynamicOffsetExpr(
    const std::vector<SymbolicScalar>& dynOffset, bool isConv3D, std::vector<int64_t>& staticOffsets) const
{
    std::vector<std::string> gmOffsetExpr;
    size_t dimSize = isConv3D ? SHAPE_DIM5 : SHAPE_DIM4;

    if (!(functionType == FunctionType::STATIC) && dynOffset[ID0].IsValid()) {
        gmOffsetExpr = GenSymbolicArgument(dynOffset);
        staticOffsets.resize(dimSize, 0);
    } else {
        staticOffsets.resize(dimSize);
        for (size_t i = 0; i < dimSize; i++) {
            staticOffsets[i] = dynOffset[i].Concrete();
        }
    }
    return gmOffsetExpr;
}

std::vector<std::string> CodeGenOpNPU::BuildCopyInParamList(
    const std::string& dstTensor, const std::string& srcTensor, const std::vector<std::string>& gmOffsetExpr,
    const std::vector<int64_t>& staticOffsets, const std::vector<int64_t>& srcShape, bool isConv3D) const
{
    std::vector<std::string> tileOpParamList;
    tileOpParamList.emplace_back(dstTensor);
    tileOpParamList.emplace_back(srcTensor);

    if (functionType == FunctionType::STATIC) {
        if (isConv3D) {
            for (size_t i = 0; i < staticOffsets.size(); i++) {
                tileOpParamList.emplace_back(std::to_string(staticOffsets[i]));
            }
        } else {
            tileOpParamList.emplace_back(std::to_string(staticOffsets[ID0]));
            tileOpParamList.emplace_back(std::to_string(staticOffsets[ID1]));
            tileOpParamList.emplace_back("0");
            tileOpParamList.emplace_back(std::to_string(staticOffsets[ID2]));
            tileOpParamList.emplace_back(std::to_string(staticOffsets[ID3]));
        }
    } else if (isConv3D) {
        for (size_t i = 0; i < gmOffsetExpr.size(); i++) {
            tileOpParamList.emplace_back(gmOffsetExpr[i]);
        }
    } else {
        tileOpParamList.emplace_back(gmOffsetExpr[0]);
        tileOpParamList.emplace_back(gmOffsetExpr[1]);
        tileOpParamList.emplace_back("0");
        tileOpParamList.emplace_back(gmOffsetExpr[2]);
        tileOpParamList.emplace_back(gmOffsetExpr[3]);
    }

    if (isConv3D) {
        for (size_t i = 0; i < srcShape.size(); i++) {
            tileOpParamList.emplace_back(std::to_string(srcShape[i]));
        }
    } else {
        tileOpParamList.emplace_back(std::to_string(srcShape[0]));
        tileOpParamList.emplace_back(std::to_string(srcShape[1]));
        tileOpParamList.emplace_back("0");
        tileOpParamList.emplace_back(std::to_string(srcShape[2]));
        tileOpParamList.emplace_back(std::to_string(srcShape[3]));
    }

    return tileOpParamList;
}

std::vector<std::string> CodeGenOpNPU::BuildCopyOutParamList(
    const std::string& dstTensor, const std::string& srcTensor, const std::vector<std::string>& gmOffsetExpr,
    const std::vector<int64_t>& staticOffsets, int64_t realM, int64_t realN, bool isConv3D) const
{
    std::vector<std::string> tileOpParamList;
    tileOpParamList.emplace_back(dstTensor);
    tileOpParamList.emplace_back(srcTensor);

    if (functionType == FunctionType::STATIC) {
        if (isConv3D) {
            for (size_t i = 0; i < staticOffsets.size(); i++) {
                tileOpParamList.emplace_back(std::to_string(staticOffsets[i]));
            }
        } else {
            tileOpParamList.emplace_back(std::to_string(staticOffsets[ID0]));
            tileOpParamList.emplace_back(std::to_string(staticOffsets[ID1]));
            tileOpParamList.emplace_back("0");
            tileOpParamList.emplace_back(std::to_string(staticOffsets[ID2]));
            tileOpParamList.emplace_back(std::to_string(staticOffsets[ID3]));
        }
    } else if (isConv3D) {
        for (size_t i = 0; i < gmOffsetExpr.size(); i++) {
            tileOpParamList.emplace_back(gmOffsetExpr[i]);
        }
    } else {
        tileOpParamList.emplace_back(gmOffsetExpr[0]);
        tileOpParamList.emplace_back(gmOffsetExpr[1]);
        tileOpParamList.emplace_back("0");
        tileOpParamList.emplace_back(gmOffsetExpr[2]);
        tileOpParamList.emplace_back(gmOffsetExpr[3]);
    }

    tileOpParamList.emplace_back(std::to_string(realM));
    tileOpParamList.emplace_back(std::to_string(realN));

    return tileOpParamList;
}

std::string CodeGenOpNPU::GetConvCopyInMode() const
{
    int64_t copyInMode = -1;
    auto ret = GetOpAttr(Conv::LoadStoreConvOpAttributeKey::copyInMode, copyInMode);
    ASSERT(ConvCodenGenError::CODEGEN_GET_ATTR_FAILED, ret) << "GenMemL1CopyInConv get CopyInMode failed.";
    bool isValidMode =
        copyInMode >= ToUnderlying(Matrix::CopyInMode::ND2NZ) && copyInMode <= ToUnderlying(Matrix::CopyInMode::DN2NZ);
    ASSERT(ConvCodenGenError::CODEGEN_CHECK_ATTR_INVALID, isValidMode)
        << "GenMemL1CopyInConv CopyInMode is invalid: " << copyInMode;
    return CopyInModeToString(static_cast<Matrix::CopyInMode>(copyInMode));
}

std::string CodeGenOpNPU::GenMemL1CopyInConv() const
{
    std::string srcTensor = QueryTileTensorNameByIdx(ToUnderlying(MISOIdx::SRC0_IDX));
    std::string dstTensor = QueryTileTensorNameByIdx(ToUnderlying(MISOIdx::DST_IDX));
    std::string copyInModeStr = GetConvCopyInMode();

    bool isFmap = true, isConv3D = false;
    GetOpAttr(Conv::LoadStoreConvOpAttributeKey::isFmap, isFmap);
    GetOpAttr(Conv::LoadStoreConvOpAttributeKey::isConv3D, isConv3D);

    auto dynOffset = offsetFromAttr[ToUnderlying(MISOIdx::SRC0_IDX)];
    auto srcShapeVec = shapeFromAttr[ToUnderlying(MISOIdx::SRC0_IDX)];

    size_t expectedDim = isConv3D ? SHAPE_DIM5 : SHAPE_DIM4;
    ASSERT(ConvCodenGenError::CODEGEN_CHECK_DIM_INVALID, dynOffset.size() == expectedDim)
        << "GenMemL1CopyInConv offset should be " << expectedDim << "-dim!";
    ASSERT(ConvCodenGenError::CODEGEN_CHECK_DIM_INVALID, srcShapeVec.size() == expectedDim)
        << "GenMemL1CopyInConv shape should be " << expectedDim << "-dim!";

    std::vector<int64_t> staticOffsets;
    std::vector<std::string> gmOffsetExpr = GetDynamicOffsetExpr(dynOffset, isConv3D, staticOffsets);

    std::vector<int64_t> srcShape;
    for (size_t i = 0; i < srcShapeVec.size(); i++) {
        srcShape.emplace_back(srcShapeVec[i]);
    }

    std::vector<std::string> tileOpParamList =
        BuildCopyInParamList(dstTensor, srcTensor, gmOffsetExpr, staticOffsets, srcShape, isConv3D);

    std::ostringstream oss;
    oss << tileOpName << WrapParamByAngleBrackets({copyInModeStr, std::to_string(isConv3D), std::to_string(isFmap)});
    oss << WrapParamByParentheses(tileOpParamList) << STMT_END;
    return oss.str();
}

std::string CodeGenOpNPU::GetConvCopyOutMode() const
{
    int64_t copyOutMode = -1;
    auto ret = GetOpAttr(Conv::LoadStoreConvOpAttributeKey::copyOutMode, copyOutMode);
    ASSERT(ConvCodenGenError::CODEGEN_GET_ATTR_FAILED, ret) << "GenMemL1CopyOutConv get CopyOutMode failed.";
    bool isValidMode = copyOutMode == ToUnderlying(Matrix::CopyOutMode::NZ2ND) ||
                       copyOutMode == ToUnderlying(Matrix::CopyOutMode::NZ2NZ) ||
                       copyOutMode == ToUnderlying(Matrix::CopyOutMode::NZ2DN);
    ASSERT(ConvCodenGenError::CODEGEN_CHECK_ATTR_INVALID, isValidMode)
        << "GenMemL1CopyOutConv CopyOutMode is invalid: " << copyOutMode;
    return CopyOutModeToString(static_cast<Matrix::CopyOutMode>(copyOutMode));
}

std::string CodeGenOpNPU::GenMemL1CopyOutConv() const
{
    std::string dstTensor = QueryTileTensorNameByIdx(ToUnderlying(MISOIdx::DST_IDX));
    std::string srcTensor = QueryTileTensorNameByIdx(ToUnderlying(MISOIdx::SRC0_IDX));
    std::string copyOutModeStr = GetConvCopyOutMode();

    bool isConv3D = false;
    GetOpAttr(Conv::LoadStoreConvOpAttributeKey::isConv3D, isConv3D);

    auto realShape = shapeFromAttr[ToUnderlying(MISOIdx::DST_IDX)];
    ASSERT(ConvCodenGenError::CODEGEN_CHECK_DIM_INVALID, realShape.size() == SHAPE_DIM2)
        << "GenMemL1CopyOutConv valid shape should be 2-dim!";
    int64_t realM = realShape[ID0];
    int64_t realN = realShape[ID1];

    auto dynOffset = offsetFromAttr[ToUnderlying(MISOIdx::DST_IDX)];
    size_t expectedDim = isConv3D ? SHAPE_DIM5 : SHAPE_DIM4;
    ASSERT(ConvCodenGenError::CODEGEN_CHECK_DIM_INVALID, dynOffset.size() == expectedDim)
        << "GenMemL1CopyOutConv offset should be " << expectedDim << "-dim!";

    std::vector<int64_t> staticOffsets;
    std::vector<std::string> gmOffsetExpr = GetDynamicOffsetExpr(dynOffset, isConv3D, staticOffsets);

    std::vector<std::string> tileOpParamList =
        BuildCopyOutParamList(dstTensor, srcTensor, gmOffsetExpr, staticOffsets, realM, realN, isConv3D);

    std::ostringstream oss;
    oss << tileOpName << WrapParamByAngleBrackets({copyOutModeStr, std::to_string(isConv3D)});
    oss << WrapParamByParentheses(tileOpParamList) << STMT_END;
    return oss.str();
}

std::string CodeGenOpNPU::GenMemL1ToL0Load3D() const
{
    std::vector<std::variant<std::string, uint8_t, uint16_t, int, int64_t>> paramList;

    std::string dstVar = QueryTileTensorNameByIdx(ToUnderlying(MISOIdx::DST_IDX));
    std::string srcVar = QueryTileTensorNameByIdx(ToUnderlying(MISOIdx::SRC0_IDX));
    paramList.emplace_back(dstVar);
    paramList.emplace_back(srcVar);

    int64_t mPos = 0, kPos = 0;
    GetOpAttr(Conv::L12L0ConvOpAttributeKey::postM, mPos);
    GetOpAttr(Conv::L12L0ConvOpAttributeKey::postK, kPos);
    paramList.emplace_back(mPos);
    paramList.emplace_back(kPos);

    std::vector<int64_t> fmapL1Shape = rawShape[ID1];
    CODEGEN_LOGI("GenMemL1ToL0Load3D %s, fmapL1Shape is %s", tileOpName.c_str(), IntVecToStr(fmapL1Shape).c_str());

    int64_t padLeft = 0, padRight = 0, padTop = 0, padBottom = 0, padValue = 0;
    GetOpAttr(Conv::L12L0ConvOpAttributeKey::paddingLeft, padLeft);
    GetOpAttr(Conv::L12L0ConvOpAttributeKey::paddingRight, padRight);
    GetOpAttr(Conv::L12L0ConvOpAttributeKey::paddingTop, padTop);
    GetOpAttr(Conv::L12L0ConvOpAttributeKey::paddingBottom, padBottom);
    GetOpAttr(Conv::L12L0ConvOpAttributeKey::padValue, padValue);
    paramList.emplace_back(padLeft);
    paramList.emplace_back(padRight);
    paramList.emplace_back(padTop);
    paramList.emplace_back(padBottom);
    paramList.emplace_back(padValue);

    int64_t filterH = 0, filterW = 0;
    GetOpAttr(Conv::L12L0ConvOpAttributeKey::filterH, filterH);
    GetOpAttr(Conv::L12L0ConvOpAttributeKey::filterW, filterW);
    paramList.emplace_back(filterH);
    paramList.emplace_back(filterW);

    int64_t dilationH = 0, dilationW = 0;
    GetOpAttr(Conv::L12L0ConvOpAttributeKey::dilationH, dilationH);
    GetOpAttr(Conv::L12L0ConvOpAttributeKey::dilationW, dilationW);
    paramList.emplace_back(dilationH);
    paramList.emplace_back(dilationW);

    int64_t strideH = 0, strideW = 0;
    GetOpAttr(Conv::L12L0ConvOpAttributeKey::strideH, strideH);
    GetOpAttr(Conv::L12L0ConvOpAttributeKey::strideW, strideW);
    paramList.emplace_back(strideH);
    paramList.emplace_back(strideW);

    std::vector<int64_t> fmapL0Shape = rawShape[ID0];
    CODEGEN_LOGI("GenMemL1ToL0Load3D %s, fmapL0Shape is %s", tileOpName.c_str(), IntVecToStr(fmapL0Shape).c_str());
    ASSERT(ConvCodenGenError::CODEGEN_CHECK_DIM_INVALID, fmapL0Shape.size() == SHAPE_DIM2)
        << "GenMemL1ToL0Load3D L0 fmap only support 2-dim!";

    bool isConv3D = false;
    GetOpAttr(Conv::LoadStoreConvOpAttributeKey::isConv3D, isConv3D);

    std::ostringstream oss;
    oss << tileOpName.c_str() << WrapParamByAngleBrackets({std::to_string(isConv3D)});
    oss << WrapParamByParentheses(paramList) << STMT_END;
    return oss.str();
}

std::string CodeGenOpNPU::GenMemL1ToL0Load2D() const
{
    std::vector<std::variant<std::string, uint16_t, int, int64_t>> paramList;

    std::string dstVar = QueryTileTensorNameByIdx(ToUnderlying(MISOIdx::DST_IDX));
    std::string srcVar = QueryTileTensorNameByIdx(ToUnderlying(MISOIdx::SRC0_IDX));
    paramList.emplace_back(dstVar);
    paramList.emplace_back(srcVar);

    int64_t kPos = 0, nPos = 0;
    GetOpAttr(Conv::L12L0ConvOpAttributeKey::postK, kPos);
    GetOpAttr(Conv::L12L0ConvOpAttributeKey::postN, nPos);
    paramList.emplace_back(kPos);
    paramList.emplace_back(nPos);

    std::vector<int64_t> weightL1Shape = rawShape[ID1];
    CODEGEN_LOGI("GenMemL1ToL0Load2D %s, weightL1Shape is %s", tileOpName.c_str(), IntVecToStr(weightL1Shape).c_str());

    std::vector<int64_t> weightL0Shape = rawShape[ID0];
    CODEGEN_LOGI("GenMemL1ToL0Load2D %s, weightL0Shape is %s", tileOpName.c_str(), IntVecToStr(weightL0Shape).c_str());
    ASSERT(ConvCodenGenError::CODEGEN_CHECK_DIM_INVALID, weightL0Shape.size() == SHAPE_DIM2)
        << "GenMemL1ToL0Load2D L0 weight only support 2-dim!";

    std::ostringstream oss;
    oss << tileOpName.c_str() << WrapParamByParentheses(paramList) << STMT_END;
    return oss.str();
}

} // namespace npu::tile_fwk
