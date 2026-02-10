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
 * \file conv_operation_impl.cpp
 * \brief
 */

#include "interface/configs/config_manager.h"
#include "interface/inner/pre_def.h"
#include "interface/operation/operation.h"
#include "interface/operation/operation_common.h"
#include "interface/program/program.h"
#include "interface/tensor/logical_tensor.h"
#include "interface/utils/common.h"
#include "interface/utils/log.h"
#include "interface/utils/operator_tracer.h"
#include "operation_impl.h"
#include "tilefwk/data_type.h"
#include "tilefwk/tile_shape.h"
#include "tilefwk/platform.h"

namespace npu {
namespace tile_fwk {
namespace Conv {

#define OP_CHECK(cond, exec_expr) \
    do { \
        if (cond) { \
            exec_expr; \
        } \
    } while (0)

const std::string Im2ColOpAttributeKey::postK = "POST_K";
const std::string Im2ColOpAttributeKey::postM = "POST_M";
const std::string Im2ColOpAttributeKey::filterH = "FILTER_H";
const std::string Im2ColOpAttributeKey::filterW = "FILTER_W";
const std::string Im2ColOpAttributeKey::strideH = "STRIDE_H";
const std::string Im2ColOpAttributeKey::strideW = "STRIDE_W";
const std::string Im2ColOpAttributeKey::dilationH = "DILATION_H";
const std::string Im2ColOpAttributeKey::dilationW = "DILATION_W";
const std::string Im2ColOpAttributeKey::paddingLeft = "PAD_LEFT";
const std::string Im2ColOpAttributeKey::paddingRight = "PAD_RIGHT";
const std::string Im2ColOpAttributeKey::paddingTop = "PAD_TOP";
const std::string Im2ColOpAttributeKey::paddingBottom = "PAD_BOTTOM";
const std::string Im2ColOpAttributeKey::padValue = "PAD_VALUE";
std::vector<int64_t> rotateVector(const std::vector<int64_t>& input, size_t shift) {
    std::vector<int64_t> result = input;
    std::rotate(result.begin(), result.begin() + shift, result.end());
    return result;
}
void CheckValueRange(int64_t value, const std::string& name, int64_t min, int64_t max)
{
    OP_CHECK(true, {
            ASSERT(value >= min && value <= max)
            << "Invalid " << name << ":" << value
            << ", expected range [" << min << "," << max << "]." << std::endl;
    });
}
int64_t ConvComputeHo(const Tensor &inputTensor, const Tensor &weightTensor, const ConvAttrParam &attrParam)
{
    uint32_t indexH = NCHW_H_IDX;
    if (attrParam.isConv3D) {
        indexH = NCDHW_H_IDX;
    }
    std::vector<int64_t> strides = attrParam.strides;
    int64_t strideH = strides[PAD_STRIDE_H];
    if (strideH == 0) {
        return 1;
    }
    std::vector<int64_t> paddings = attrParam.paddings;
    std::vector<int64_t> dilations = attrParam.dilations;
    int64_t padTop = paddings[PAD_TOP_INDEX];
    int64_t padBottom = paddings[PAD_BOTTOM_INDEX];
    int64_t dilationH = dilations[PAD_STRIDE_H];
    int64_t hin = inputTensor.GetShape()[indexH];
    int64_t kh = weightTensor.GetShape()[indexH];
    int64_t cmpHo = (hin + padTop + padBottom - dilationH * (kh - 1) - 1) / strideH + 1;
    return cmpHo;
}
int64_t ConvComputeWo(const Tensor &inputTensor, const Tensor &weightTensor, const ConvAttrParam &attrParam)
{
    if (attrParam.isConv1D) {
        return 1;
    }
    uint32_t indexW= NCHW_W_IDX;
    if (attrParam.isConv3D) {
        indexW = NCDHW_W_IDX;
    }
    std::vector<int64_t> strides = attrParam.strides;
    int64_t strideW = strides[PAD_STRIDE_W];
    if (strideW == 0) {
        return 1;
    }
    std::vector<int64_t> paddings = attrParam.paddings;
    std::vector<int64_t> dilations = attrParam.dilations;
    int64_t dilationW = dilations[PAD_STRIDE_W];
    int64_t padLeft = paddings[PAD_LEFT_INDEX];
    int64_t padRight = paddings[PAD_RIGHT_INDEX];
    int64_t win = inputTensor.GetShape()[indexW];
    int64_t kw = weightTensor.GetShape()[indexW];
    int64_t cmpWo = (win + padLeft + padRight - dilationW * (kw - 1) - 1) / strideW + 1;
    return cmpWo;
}
int64_t ConvComputeDo(const Tensor &inputTensor, const Tensor &weightTensor, const ConvAttrParam &attrParam)
{
    std::vector<int64_t> strides = attrParam.strides;
    int64_t strideD = strides[PAD_STRIDE_D];
    if (strideD == 0) {
        return 1;
    }
    std::vector<int64_t> paddings = attrParam.paddings;
    std::vector<int64_t> dilations = attrParam.dilations;
    int64_t padHead = paddings[PAD_HEAD_INDEX];
    int64_t padTail = paddings[PAD_TAIL_INDEX];
    int64_t dilationD = dilations[PAD_STRIDE_D];
    int64_t din = inputTensor.GetShape()[NCDHW_D_IDX];
    int64_t kd = weightTensor.GetShape()[NCDHW_D_IDX];//NCHW或者NCDHW
    int64_t cmpDo = (din + padHead + padTail - dilationD * (kd - 1) - 1) / strideD + 1;
    return cmpDo;
}
void CheckOutputShape(const Tensor &inputTensor, const Tensor &weightTensor, const ConvAttrParam &attrParam)
{
    int64_t hOut = ConvComputeHo(inputTensor, weightTensor, attrParam);
    CheckValueRange(hOut, "hOut" , NUM1, MAX_SIZE);
    int64_t wOut = ConvComputeWo(inputTensor, weightTensor, attrParam);
    CheckValueRange(wOut, "wOut" , NUM1, MAX_SIZE);
    if (attrParam.isConv3D) {
        int64_t dOut = ConvComputeDo(inputTensor, weightTensor, attrParam);
        CheckValueRange(dOut, "dOut" , NUM1, MAX_SIZE);
    }
}
void CheckAlignment(int64_t value, int64_t alignment, const std::string& valueName, bool isByte = false)
{
        OP_CHECK(true, {
            ASSERT(value % alignment == 0)
                << "Invalid " << valueName << ":" << value
                << ", requires " << alignment << (isByte ? "-byte alignment." : "-element alignment.") << std::endl;
        });
}
void CheckHowoTile(const Tensor &inputTensor, const Tensor &weightTensor, const ConvAttrParam &attrParam)
{
    auto &convTile = TileShape::Current().GetConvTile();
    int64_t tileHout = convTile.tileL1Info.tileHout;
    int64_t tileWout = convTile.tileL1Info.tileWout;
    int64_t hOut = ConvComputeHo(inputTensor, weightTensor, attrParam);
    int64_t wOut = ConvComputeWo(inputTensor, weightTensor, attrParam);
    if (wOut % 16 != 0) {
        OP_CHECK(true, {
            ASSERT(tileHout == 1)
                << "When wout is not a multiple of 16, tileHout should be 1." << std::endl;
        });
    }
    CheckValueRange(tileHout, "tileHout" , NUM1, hOut);
    CheckValueRange(tileWout, "tileWout" , NUM1, wOut);
    if (!attrParam.isConv1D) {
        CheckAlignment(tileWout, NUM16, "tileWout");
    }
}
void ValidateL0Constraint(int64_t tile1, int64_t tile2, int64_t tile3, size_t dtypeSize, size_t cacheSize, const std::string& cacheName,
    const std::string& dim1Name, const std::string& dim2Name, const std::string& dim3Name)
{
    OP_CHECK(true, {
        ASSERT(tile1 * tile2 * tile3 * dtypeSize <= cacheSize)
            << "Shape does not satisfy " << cacheName 
            << " load constraints, " << dim1Name << ":" << tile1
            << ", " << dim2Name << ":" << tile2 << ", " << dim3Name << ":" << tile3
            << ", which must satisfy " << dim1Name << " × " << dim2Name << " × "
            << dim3Name << " × dtypesize ≤ " << cacheName << "Size(" << cacheSize << ")." 
            << std::endl;
    });
}
void CheckL0TileTiling(DataType outType)
{
    auto &convTile = TileShape::Current().GetConvTile();
    int64_t tileH = convTile.tileL0Info.tileH;
    int64_t tileW = convTile.tileL0Info.tileW;
    int64_t tileN = convTile.tileL0Info.tileN;
    int64_t tileK = convTile.tileL0Info.tileK;
    int64_t k0 = ALIGN_SIZE_32 / BytesOf(outType);
    CheckAlignment(tileK , k0, "tileK", true);
    CheckAlignment(tileN, NUM16, "tileN");
    CheckAlignment(tileW, NUM16, "tileW");

    Platform& platform = Platform::Instance();
    platform.ObtainPlatformInfo();
    size_t l0aSize = platform.GetAICCore().GetMemorySize(MemoryType::MEM_L0A);
    size_t l0bSize = platform.GetAICCore().GetMemorySize(MemoryType::MEM_L0B);
    size_t l0cSize = platform.GetAICCore().GetMemorySize(MemoryType::MEM_L0C);
    ValidateL0Constraint(tileH, tileW, tileK, BytesOf(outType), l0aSize, "L0A", "tileH", "tileW", "tileK");
    ValidateL0Constraint(tileK, tileN, 1, BytesOf(outType), l0bSize, "L0B", "tileK", "tileN", "");
    ValidateL0Constraint(tileH, tileW, tileN, BytesOf(DataType::DT_FP32), l0cSize, "L0C", "tileH", "tileW", "tileN");
}
void CheckDivisible(int64_t value, int64_t divisor, const std::string& valueName, const std::string& divisorName)
{
    OP_CHECK(true, {
            ASSERT(value % divisor == 0)
            << "The value of " << divisorName << " (" << divisor
            << ") does not divide "<< valueName
            << "(" << value << "). Adjusting " << divisorName 
            << " to the nearest value such that "<< valueName 
            << " % " << divisorName << " == 0." << std::endl;
    });
}
int64_t ConvAlignB(int64_t a, int64_t b)
{
    if (b == 0) {
        return 0;
    }
    return ((a + b - 1) / b) * b;
}
void CheckTileTiling(DataType outType, const Tensor &inputTensor, const Tensor &weightTensor, const ConvAttrParam &attrParam)
{
    auto convTile = TileShape::Current().GetConvTile();
    int64_t tileHin = convTile.tileL1Info.tileHin;
    int64_t tileWin = convTile.tileL1Info.tileWin;
    int64_t tileCinFmap = convTile.tileL1Info.tileCinFmap;
    int64_t tileCinWeight = convTile.tileL1Info.tileCinWeight;
    int64_t tileCout = convTile.tileL1Info.tileCout;
    int64_t tileBatch = convTile.tileL1Info.tileN;
    uint32_t indexH = NCHW_H_IDX;
    uint32_t indexW = NCHW_W_IDX;
    if (attrParam.isConv3D) {
        indexH = NCDHW_H_IDX;
        indexW = NCDHW_W_IDX;
    }
    int64_t cin = inputTensor.GetShape()[NCHW_C_IDX];
    int64_t hin = inputTensor.GetShape()[indexH];
    int64_t win = 1;
    if (!attrParam.isConv1D) {
        win = inputTensor.GetShape()[indexW];
    }
    int64_t cOut = weightTensor.GetShape()[NCHW_N_IDX];
    CheckValueRange(tileHin, "tileHin", NUM1, hin);
    CheckValueRange(tileBatch, "tileN", NUM1, NUM1);
    CheckValueRange(tileWin, "tileWin", NUM1, win);
    CheckValueRange(tileCout, "tileCout", NUM1, cOut);

    CheckHowoTile(inputTensor, weightTensor, attrParam);

    CheckDivisible(ConvAlignB(cin, NUM16), tileCinFmap, "ceil(Cin / C0) × C0", "tileCinFmap");
    CheckDivisible(ConvAlignB(cin, NUM16), tileCinWeight, "ceil(Cin / C0) × C0", "tileCinWeight");
    CheckAlignment(tileCinFmap, NUM16, "tileCinFmap");
    CheckAlignment(tileCinWeight, NUM16, "tileCinWeight");
    if (convTile.setL0Tile){
        CheckL0TileTiling(outType);
    }
}
uint64_t Conv2DInferHiL1(uint64_t inputHoL1, uint64_t khDilated, uint64_t hi, uint64_t strideH)
{
    uint64_t tmpHiL1 = (inputHoL1 - 1) * strideH + khDilated;
    if (tmpHiL1 > hi) {
        tmpHiL1 = hi;
    }
    return tmpHiL1;
}
void CheckL1SizeTiling(DataType outType, const Tensor &inputTensor, const Tensor &weightTensor, const Tensor &biasTensor, ConvAttrParam &attrParam)
{
    auto convTile = TileShape::Current().GetConvTile();
    uint64_t l1Size = Platform::Instance().GetAICCore().GetMemorySize(MemoryType::MEM_L0A);
    uint32_t indexH = NCHW_H_IDX;
    uint32_t indexW = NCHW_W_IDX;
    if (attrParam.isConv3D) {
        indexH = NCDHW_H_IDX;
        indexW = NCDHW_W_IDX;
    }
    int64_t kh = weightTensor.GetShape()[indexH];
    int64_t hin = inputTensor.GetShape()[indexH];
    int64_t kw = 1;
    int64_t win = 1;
    if (!attrParam.isConv1D) {
        kw = weightTensor.GetShape()[indexW];
        win = inputTensor.GetShape()[indexW];
    }
    int64_t k0 = ALIGN_SIZE_32 / BytesOf(outType);
    std::vector<int64_t> strides = attrParam.strides;
    std::vector<int64_t> dilations = attrParam.dilations;
    int64_t strideH = strides[PAD_STRIDE_H];
    int64_t strideW = strides[PAD_STRIDE_W];
    int64_t dilationH = dilations[PAD_STRIDE_H];
    int64_t dilationW = dilations[PAD_STRIDE_W];

    uint64_t biasL1Size = 0;
    uint64_t nBL1min = NUM16;
    if (!biasTensor.IsEmpty()) {
        biasL1Size = ConvAlignB(nBL1min * BytesOf(outType), ALIGN_SIZE_32);
    }
    uint64_t kBL1min = k0 * kh * kw;
    uint64_t weightL1Size = ConvAlignB(kBL1min * nBL1min * BytesOf(outType), ALIGN_SIZE_32);
    uint64_t inputL1Size = 0;
    uint64_t m0 = NUM16;
    uint64_t wo = ConvComputeWo(inputTensor, weightTensor, attrParam);
    uint64_t hoAL1min = wo < m0 ? (m0 + wo - 1) / wo : 1;
    uint64_t khDilated = (kh - 1) * dilationH + 1;
    uint64_t hiAL1min = Conv2DInferHiL1(hoAL1min, khDilated, hin, strideH);
    uint64_t kAL1min = k0;
    uint64_t woAL1min = m0;
    uint64_t kwDilated = (kw - 1) * dilationW + 1;
    uint64_t wiAL1min = Conv2DInferHiL1(woAL1min, kwDilated, win, strideW);
    inputL1Size = ConvAlignB(hiAL1min * wiAL1min * kAL1min * BytesOf(outType), ALIGN_SIZE_32);
    uint64_t minL1LoadSize = biasL1Size + inputL1Size + weightL1Size;
    OP_CHECK(true, {
        ASSERT(minL1LoadSize <= l1Size)
            << "MinL1LoadSize > L1size, current L1size: " << l1Size
            << ", maxL1Size: " << minL1LoadSize
            << "." << std::endl;
    });
}

void CheckGroupsShape(const int64_t cinFmap, const int64_t cinWeight,const int64_t cOut, const int64_t groups)
{
    CheckValueRange(groups, "groups", NUM1, SHAPE_INNER_AXIS_MAX_SIZE);

    CheckDivisible(cinFmap, groups, "Cin", "groups");
    CheckDivisible(cOut, groups, "Cout", "groups");

    OP_CHECK(true, {
            ASSERT(cinFmap == cinWeight * groups)
            << "Fmap Cin (" << cinFmap
            << ") != weight Cin (" << cinWeight
            << ") * groups (" << groups
            << ")." << std::endl;
    });
}

void CheckDimParam(const std::vector<int64_t>& vec, const std::string& name, int expectedDim)
{
    OP_CHECK(true, {
            ASSERT(vec.size() == static_cast<size_t>(expectedDim))
                << "Input attr " << name << " dim: " << vec.size()
                << " != " << expectedDim << "." << std::endl;
    });
}

void CheckDimensionRange(const std::vector<int64_t>& vec, const std::string& name, int minVal, int maxVal)
{
    for (size_t i = 0; i < vec.size(); ++i) {
        OP_CHECK(true, {
            ASSERT(vec[i] >= minVal && vec[i] <= maxVal)
                << "The value of the " << i
                << "-th dimension of " << name
                << " must be in the range [" << minVal
                << "," << maxVal << "].Current value:" << vec[i]
                << "." << std::endl;
        });
    }
}

void CheckLoad3dShape(DataType outType, const Tensor &weightTensor, const ConvAttrParam &attrParam)
{
    std::vector<int64_t> paddings = attrParam.paddings;
    std::vector<int64_t> dilations = attrParam.dilations;
    std::vector<int64_t> strides = attrParam.strides;
    if (attrParam.isConv3D) {
        paddings = rotateVector(paddings, 4);
        dilations = rotateVector(dilations, 2);
        strides = rotateVector(strides, 2);
    }
    CheckDimensionRange(paddings, "paddings", 0, MAX_PAD_KERNEL);
    CheckDimensionRange(dilations, "dilations", NUM1, MAX_DILATION_STRIDE);
    CheckDimensionRange(strides, "strides", NUM1, MAX_DILATION_STRIDE);

    uint32_t indexH = NCHW_H_IDX;
    uint32_t indexW = NCHW_W_IDX;
    if (attrParam.isConv3D) {
        indexH = NCDHW_H_IDX;
        indexW = NCDHW_W_IDX;
    }
    int64_t kh = weightTensor.GetShape()[indexH];
    int64_t kw = 1;
    if (!attrParam.isConv1D) {
        kw = weightTensor.GetShape()[indexW];
    }
    OP_CHECK(true, {
        ASSERT(kh <= MAX_PAD_KERNEL && kw  <= MAX_PAD_KERNEL)
        << "Weight shapes do not satisfy Load3D's limits: kh=" << kh
        << ", kw=" << kw
        << ", which must <=" << MAX_PAD_KERNEL
        << "." << std::endl;
    });

    int64_t k0 = ALIGN_SIZE_32 / BytesOf(outType);
    OP_CHECK(true, {
        ASSERT(kh * kw * k0 <= SHAPE_INNER_AXIS_MAX_SIZE)
        << "Weight shapes do not satisfy Load3D's limits: kh*kw*k0=" << kh * kw * k0
        << ", which must <=" << SHAPE_INNER_AXIS_MAX_SIZE
        << "." << std::endl;
    });
}

void CheckAttrShape(DataType outType, const Tensor &inputTensor, const Tensor &weightTensor, const ConvAttrParam &attrParam)
{
    std::vector<int64_t> paddings = attrParam.paddings;
    uint32_t index = NUM2;
    if (attrParam.isConv3D) {
        paddings = rotateVector(paddings, 4);
        index = CONV3D_INPUT_DIM - 2;
    } else if (attrParam.isConv1D) {
        index = 1;
    }
    CheckDimParam(attrParam.paddings, "paddings", index * 2);
    CheckDimParam(attrParam.dilations, "dilations", index);
    CheckDimParam(attrParam.strides, "strides", index);
    int64_t groups = attrParam.groups;
    int64_t cinFmap = inputTensor.GetShape()[NCHW_C_IDX];
    int64_t cinWeight = weightTensor.GetShape()[NCHW_C_IDX];
    int64_t cOut = weightTensor.GetShape()[NCHW_N_IDX];
    for (size_t i = 0; i < paddings.size() / 2; ++i) {
        int weightVal = weightTensor.GetShape()[i + 2];
        int paddingLeft = paddings[i * 2];
        int paddingRight = paddings[i * 2 + 1];
        OP_CHECK(true, {
            ASSERT(paddingLeft < weightVal && paddingRight < weightVal)
                << "The value of the " << i + 2
                << "-th dimension of weight must be >= padding.Current weight value:" << weightVal
                << ",padding value:" << paddingLeft
                << " and " << paddingRight
                << "." << std::endl;
        });
    }
    CheckGroupsShape(cinFmap, cinWeight, cOut, groups);
    CheckLoad3dShape(outType, weightTensor, attrParam);
}

void CheckOriginShape(const Tensor &inputTensor, const Tensor &weightTensor, const Tensor &biasTensor)
{
    CheckDimensionRange(inputTensor.GetShape(), "fmap", NUM1, MAX_SIZE);
    CheckDimensionRange(weightTensor.GetShape(), "weight", NUM1, MAX_SIZE);

    if (biasTensor.IsEmpty()) {
        return;
    }
    int64_t cOut = weightTensor.GetShape()[NCHW_N_IDX];
    OP_CHECK(true, {
        ASSERT(biasTensor.GetShape()[0] == cOut)
        << "Input illegal bias shape:" << biasTensor.GetShape()[0]
        << ", which must equal to Cout:" << cOut
        << "." << std::endl;
    });
}
void CheckConvOperands(DataType outType, const Tensor &inputTensor, const Tensor &weightTensor, const Tensor &biasTensor, ConvAttrParam &attrParam)
{
    OP_CHECK(true, {
        ASSERT(outType == DataType::DT_FP32 || outType == DataType::DT_FP16 || outType == DataType::DT_BF16)
        << "Unsupported output data type. Only DT_FP32, DT_FP16, DT_BF16 are supported.";
    });
    if (inputTensor.Dim() == CONV1D_INPUT_DIM && weightTensor.Dim() == CONV1D_INPUT_DIM) {
        attrParam.isConv1D = true;
    } else if (inputTensor.Dim() == CONV3D_INPUT_DIM && weightTensor.Dim() == CONV3D_INPUT_DIM) {
        attrParam.isConv3D = true;
    }
    CheckOriginShape(inputTensor, weightTensor, biasTensor);
    CheckOutputShape(inputTensor, weightTensor, attrParam);
    CheckAttrShape(outType, inputTensor, weightTensor, attrParam);
    CheckTileTiling(outType, inputTensor, weightTensor, attrParam);
    CheckL1SizeTiling(outType, inputTensor, weightTensor, biasTensor, attrParam);
}

void SetTensorOpAttr(Operation &op, const LogicalTensorPtr &inputTensor, const LogicalTensorPtr &weightTensor,
                     const LogicalTensorPtr &resTensor, const ConvAttrParam &convAttrParam)
{
    op.SetAttribute(CONV_BIAS_ATTR, convAttrParam.hasBias);
    op.SetAttribute(CONV_GROUPS_ATTR, convAttrParam.groups);
    op.SetAttribute(CONV_PADDINGS_ATTR, convAttrParam.paddings);
    op.SetAttribute(CONV_STRIDES_ATTR, convAttrParam.strides);
    op.SetAttribute(CONV_DILATIONS_ATTR, convAttrParam.dilations);
    op.SetAttribute(CONV_3D_FLAG, convAttrParam.isConv3D);
    op.SetAttribute(CONV_ORI_FMAP_SHAPE_ATTR, inputTensor->GetShape());
    op.SetAttribute(CONV_ORI_WEIGHT_SHAPE_ATTR, weightTensor->GetShape());
    op.SetAttribute(CONV_ORI_RES_SHAPE_ATTR, resTensor->GetShape());
}

Tensor ConstructTensorGraph(const Tensor &inputTensor, const Tensor &weightTensor,
                            const Tensor &biasTensor, const Tensor &resTensor, ConvAttrParam &convAttrParam)
{
    // add Conv node
    Function *functionPtr = Program::GetInstance().GetCurrentFunction();
    OP_CHECK(true, { ASSERT(functionPtr != nullptr) << "functionPtr is nullptr." << std::endl; });
    std::vector<LogicalTensorPtr> operandVecIn = {inputTensor.GetStorage(), weightTensor.GetStorage()};
    std::vector<LogicalTensorPtr> operandVecOut = {resTensor.GetStorage()};
    if (inputTensor.Dim() == CONV1D_INPUT_DIM && weightTensor.Dim() == CONV1D_INPUT_DIM) {
        // conv1d case, unsqueeze input to NC1W
        std::vector<int64_t> fmap4DimShape{inputTensor.GetShape()[NCHW_N_IDX], inputTensor.GetShape()[NCHW_C_IDX],
                                           1, inputTensor.GetShape()[NCHW_H_IDX]};
        Tensor fmap4DimTensor(inputTensor.GetStorage()->Datatype(), fmap4DimShape, "", inputTensor.Format());
        std::vector<int64_t> weight4DimShape{weightTensor.GetShape()[NCHW_N_IDX], weightTensor.GetShape()[NCHW_C_IDX],
                                             1, weightTensor.GetShape()[NCHW_H_IDX]};
        Tensor weigth4DimTensor(weightTensor.GetStorage()->Datatype(), weight4DimShape, "", weightTensor.Format());
        operandVecIn = {fmap4DimTensor.GetStorage(), weigth4DimTensor.GetStorage()};
    }
    if (!biasTensor.IsEmpty()) {
        convAttrParam.hasBias = true;
        operandVecIn.push_back(biasTensor.GetStorage());
    }

    if (inputTensor.Dim() == CONV1D_INPUT_DIM && weightTensor.Dim() == CONV1D_INPUT_DIM) {
        // conv1d case, squeeze output to NCL
        std::vector<int64_t> res4DimShape{inputTensor.GetShape()[NCHW_N_IDX], weightTensor.GetShape()[NCHW_N_IDX],
                                          1, resTensor.GetShape()[-1]};
        Tensor res4DimTensor(resTensor.GetStorage()->Datatype(), res4DimShape, "", resTensor.Format());
        operandVecOut = {res4DimTensor.GetStorage()};
    }
    auto &op = functionPtr->AddOperation(Opcode::OP_CONV, operandVecIn, operandVecOut);
    SetTensorOpAttr(op, operandVecIn[INPUT_FMAP_IDX], operandVecIn[INPUT_WEIGHT_IDX], operandVecOut[0], convAttrParam);
    return resTensor;
}

void SetConvAttrParam(const Operation &op, ConvAttrParam &convAttrParam)
{
    convAttrParam.isConv3D = (op.HasAttr(CONV_3D_FLAG)) ? op.GetBoolAttribute(CONV_3D_FLAG) : false;
    convAttrParam.paddings = (op.HasAttr(CONV_PADDINGS_ATTR)) ? op.GetVectorIntAttribute(CONV_PADDINGS_ATTR) :
        convAttrParam.isConv3D ? CONV3D_ATTR_DEFAULT_LIST : CONV2D_PAD_ATTR_DEFAULT_LIST;
    convAttrParam.strides = (op.HasAttr(CONV_STRIDES_ATTR)) ? op.GetVectorIntAttribute(CONV_STRIDES_ATTR) :
        convAttrParam.isConv3D ? CONV3D_ATTR_DEFAULT_LIST : CONV2D_ATTR_DEFAULT_LIST;
    convAttrParam.dilations = (op.HasAttr(CONV_DILATIONS_ATTR)) ? op.GetVectorIntAttribute(CONV_DILATIONS_ATTR) :
        convAttrParam.isConv3D ? CONV3D_ATTR_DEFAULT_LIST : CONV2D_ATTR_DEFAULT_LIST;
    convAttrParam.groups = (op.HasAttr(CONV_GROUPS_ATTR)) ? op.GetIntAttribute(CONV_GROUPS_ATTR) : 1;
    convAttrParam.hasBias = (op.HasAttr(CONV_BIAS_ATTR)) ? op.GetBoolAttribute(CONV_BIAS_ATTR) : false;
    convAttrParam.isInOutTensorNZ = false;
    OP_CHECK(true, {
        ASSERT(op.HasAttr(CONV_ORI_FMAP_SHAPE_ATTR))
        << "Conv ori fmapshape should be set when InOut Tensor NZ mode." << std::endl;
    });
    OP_CHECK(true, {
        ASSERT(op.HasAttr(CONV_ORI_WEIGHT_SHAPE_ATTR))
        << "Conv ori weightshape should be set when InOut Tensor NZ mode." << std::endl;
    });
    convAttrParam.oriFmapShape = op.GetVectorIntAttribute(CONV_ORI_FMAP_SHAPE_ATTR);
    convAttrParam.oriWeightShape = op.GetVectorIntAttribute(CONV_ORI_WEIGHT_SHAPE_ATTR);
    convAttrParam.oriResShape = op.GetVectorIntAttribute(CONV_ORI_RES_SHAPE_ATTR);
}

void SetTensorGraphNodes(const std::vector<LogicalTensorPtr> &operandVec, const LogicalTensorPtr &cTensorPtr,
                         const ConvAttrParam &convAttrParam, ConvGraphNodes &tensorGraphNodes)
{
    // set tensor GraphNodes
    size_t operandVecSize = SHAPE_DIM2 + static_cast<size_t>(convAttrParam.hasBias);
    OP_CHECK(true, {
            ASSERT(operandVec.size() == operandVecSize)
        << "Operand vector size mismatch: "
        << "Expected size: " << operandVecSize << ", actual size: " << operandVec.size()
        << ", Conv Common Input: " << SHAPE_DIM2 << ", hasBias: " << convAttrParam.hasBias
        << std::endl;
    });

    tensorGraphNodes.fmapTensorPtr = operandVec[INPUT_FMAP_IDX];
    tensorGraphNodes.weightTensorPtr = operandVec[INPUT_WEIGHT_IDX];
    if (convAttrParam.hasBias) {
        tensorGraphNodes.biasTensorPtr = operandVec[INPUT_BIAS_IDX];
    }
    OP_CHECK(true,
    {     ASSERT(tensorGraphNodes.fmapTensorPtr != nullptr && tensorGraphNodes.weightTensorPtr != nullptr)
        << "Expected aTensorPtr and bTensorPtr to be non-nullptr." << std::endl; });

    OP_CHECK(true, {ASSERT(cTensorPtr != nullptr) << "cTensorPtr is nullptr." << std::endl;});
    tensorGraphNodes.resTensorPtr = cTensorPtr;
}

void SetConvShapeInfo(const TileShape &tileShape, const ConvGraphNodes &tensorGraphNodes,
                      const ConvAttrParam &convAttrParam, ConvTileInfo &convTileInfo)
{
    // set org shape
    convTileInfo.orgBatch = convAttrParam.isConv3D ?
        convAttrParam.oriFmapShape[NCDHW_N_IDX] : convAttrParam.oriFmapShape[NCHW_N_IDX];
    convTileInfo.orgHin = convAttrParam.isConv3D ?
        convAttrParam.oriFmapShape[NCDHW_H_IDX] : convAttrParam.oriFmapShape[NCHW_H_IDX];
    convTileInfo.orgWin = convAttrParam.isConv3D ?
        convAttrParam.oriFmapShape[NCDHW_W_IDX] : convAttrParam.oriFmapShape[NCHW_W_IDX];
    convTileInfo.orgCin = convAttrParam.isConv3D ?
        convAttrParam.oriFmapShape[NCDHW_C_IDX] : convAttrParam.oriFmapShape[NCHW_C_IDX];
    convTileInfo.orgHout = convAttrParam.isConv3D ?
        convAttrParam.oriResShape[NCDHW_H_IDX] : convAttrParam.oriResShape[NCHW_H_IDX];
    convTileInfo.orgWout = convAttrParam.isConv3D ?
        convAttrParam.oriResShape[NCDHW_W_IDX] : convAttrParam.oriResShape[NCHW_W_IDX];
    convTileInfo.orgDin = convAttrParam.isConv3D ? convAttrParam.oriFmapShape[NCDHW_D_IDX] : 1;
    convTileInfo.orgDout = convAttrParam.isConv3D ? convAttrParam.oriResShape[NCDHW_D_IDX] : 1;
    convTileInfo.cin0 = ALIGN_SIZE_32 / BytesOf(tensorGraphNodes.fmapTensorPtr->Datatype());
    convTileInfo.orgCout = convAttrParam.isConv3D ?
        convAttrParam.oriWeightShape[NCDHW_N_IDX] : convAttrParam.oriWeightShape[NCHW_N_IDX];
    convTileInfo.orgKh = convAttrParam.isConv3D ?
        convAttrParam.oriWeightShape[NCDHW_H_IDX] : convAttrParam.oriWeightShape[NCHW_H_IDX];
    convTileInfo.orgKw = convAttrParam.isConv3D ?
        convAttrParam.oriWeightShape[NCDHW_W_IDX] : convAttrParam.oriWeightShape[NCHW_W_IDX];
    convTileInfo.orgKd = convAttrParam.isConv3D ? convAttrParam.oriWeightShape[NCDHW_D_IDX] : 1;
    int64_t cinPerGroup = convTileInfo.orgCin / convAttrParam.groups;
    convTileInfo.orgHoutWout = convTileInfo.orgHout * convTileInfo.orgWout;
    convTileInfo.kPerGroup = ConvAlignB(cinPerGroup, convTileInfo.cin0) *
        convTileInfo.orgKh * convTileInfo.orgKw;
    convTileInfo.coutPerGroup = convTileInfo.orgCout / convAttrParam.groups;
    // set tileshape info
    auto &convTile = tileShape.GetConvTile();
    convTileInfo.kAL1 = convTile.tileL1Info.tileCinFmap * convTileInfo.orgKh * convTileInfo.orgKw;
    convTileInfo.kBL1 = convTile.tileL1Info.tileCinWeight * convTileInfo.orgKh * convTileInfo.orgKw;
    convTileInfo.nBL1 = convTile.tileL1Info.tileCout;
    convTileInfo.hAL1In = convTile.tileL1Info.tileHin;
    convTileInfo.wAL1In = convTile.tileL1Info.tileWin;
    convTileInfo.hAL1Out = convTile.tileL1Info.tileHout;
    convTileInfo.wAL1Out = convTile.tileL1Info.tileWout;
    convTileInfo.kL0 = convTile.tileL0Info.tileK;
    convTileInfo.hL0 = convTile.tileL0Info.tileH;
    convTileInfo.wL0 = convTile.tileL0Info.tileW;
    convTileInfo.nL0 = convTile.tileL0Info.tileN;
}

LogicalTensorPtr ConstructBiasTile(Function &function, const ConvGraphNodes &tensorGraphNodes, ConvIterInfo &iterInfo)
{
    std::vector<int64_t> dstBiasL1Shape = std::vector<int64_t>{ConvAlignB(iterInfo.nL0Size, MKN_N_VALUE)};
    std::vector<int64_t> dstBiasL1Offset = std::vector<int64_t>{iterInfo.nL0Offset};
    LogicalTensorPtr dstBiasl1TensorPtr =
        std::make_shared<LogicalTensor>(function, tensorGraphNodes.biasTensorPtr->Datatype(),
                                        dstBiasL1Shape, SymbolicScalar::FromConcrete(dstBiasL1Shape),
                                        tensorGraphNodes.biasTensorPtr->Format(), "biasL1Tensor", NodeType::LOCAL);
    dstBiasl1TensorPtr->UpdateDynValidShape(SymbolicScalar::FromConcrete(dstBiasL1Shape));
    auto &viewOpBiasL1 = function.AddOperation(Opcode::OP_VIEW, {tensorGraphNodes.biasTensorPtr},
                                               {dstBiasl1TensorPtr});
    auto viewAttributeBiasL1 = std::make_shared<ViewOpAttribute>(dstBiasL1Offset, MemoryType::MEM_L1,
            SymbolicScalar::FromConcrete(dstBiasL1Offset), dstBiasl1TensorPtr->GetDynValidShape());
    viewOpBiasL1.SetOpAttribute(viewAttributeBiasL1);

    std::vector<int64_t> dstBiasBtShape = std::vector<int64_t>{ConvAlignB(iterInfo.nL0Size, MKN_N_VALUE)};
    std::vector<int64_t> dstBiasBtOffset = std::vector<int64_t>{0};
    LogicalTensorPtr dstBiasBtTensorPtr =
        std::make_shared<LogicalTensor>(function, tensorGraphNodes.biasTensorPtr->Datatype(), dstBiasBtShape,
                                        SymbolicScalar::FromConcrete(dstBiasBtShape),
                                        tensorGraphNodes.biasTensorPtr->Format(), "biasBtTensor", NodeType::LOCAL);
    dstBiasBtTensorPtr->UpdateDynValidShape(SymbolicScalar::FromConcrete(dstBiasBtShape));
    auto &viewOpBiasBt = function.AddOperation(Opcode::OP_VIEW, {dstBiasl1TensorPtr}, {dstBiasBtTensorPtr});
    auto viewAttributeBiasBt = std::make_shared<ViewOpAttribute>(dstBiasBtOffset, MemoryType::MEM_BT,
            SymbolicScalar::FromConcrete(dstBiasBtOffset), dstBiasBtTensorPtr->GetDynValidShape());
    viewOpBiasBt.SetOpAttribute(viewAttributeBiasBt);

    return dstBiasBtTensorPtr;
}

void SetImg2ColAttr(Operation &load3dOpAl0, const ConvAttrParam &convAttrParam, ConvIterInfo &iterInfo,
                    const ConvTileInfo &convTileInfo)
{
    int64_t strideH = convAttrParam.strides[0];
    int64_t strideW = convAttrParam.strides[1];
    int64_t dilationH = convAttrParam.dilations[0];
    int64_t dilationW = convAttrParam.dilations[1];
    int64_t dilatedKernelH = (convTileInfo.orgKh - 1) * dilationH + 1;
    int64_t dilatedKernelW = (convTileInfo.orgKw - 1) * dilationW + 1;
    load3dOpAl0.SetAttribute(Im2ColOpAttributeKey::strideH, strideH);
    load3dOpAl0.SetAttribute(Im2ColOpAttributeKey::strideW, strideW);
    load3dOpAl0.SetAttribute(Im2ColOpAttributeKey::dilationH, dilationH);
    load3dOpAl0.SetAttribute(Im2ColOpAttributeKey::dilationW, dilationW);
    load3dOpAl0.SetAttribute(Im2ColOpAttributeKey::filterH, convTileInfo.orgKh);
    load3dOpAl0.SetAttribute(Im2ColOpAttributeKey::filterW, convTileInfo.orgKw);
    // cal H padding
    if (iterInfo.hL1InOffset >= 0) {
        load3dOpAl0.SetAttribute(Im2ColOpAttributeKey::paddingTop, 0);
    } else {
        load3dOpAl0.SetAttribute(Im2ColOpAttributeKey::paddingTop, 0 - iterInfo.hL1InOffset);
    }
    int64_t hinAL1Used = (iterInfo.houtL1Size - 1) * strideH + dilatedKernelH;
    int64_t hinBottomPadOffset = iterInfo.hL1InOffset + hinAL1Used;
    if (hinBottomPadOffset > convAttrParam.oriFmapShape[NCHW_H_IDX]) {
        load3dOpAl0.SetAttribute(Im2ColOpAttributeKey::paddingBottom,
                                 hinBottomPadOffset - convAttrParam.oriFmapShape[NCHW_H_IDX]);
    } else {
        load3dOpAl0.SetAttribute(Im2ColOpAttributeKey::paddingBottom, 0);
    }
    // cal W padding
    if (iterInfo.wL1InOffset >= 0) {
        load3dOpAl0.SetAttribute(Im2ColOpAttributeKey::paddingLeft, 0);
    } else {
        load3dOpAl0.SetAttribute(Im2ColOpAttributeKey::paddingLeft, 0 - iterInfo.wL1InOffset);
    }
    int64_t winAL1Used = (iterInfo.woutL1Size - 1) * strideW + dilatedKernelW;
    int64_t winRightPadOffset = iterInfo.wL1InOffset + winAL1Used;
    if (winRightPadOffset > convAttrParam.oriFmapShape[NCHW_W_IDX]) {
        load3dOpAl0.SetAttribute(Im2ColOpAttributeKey::paddingRight,
                                 winRightPadOffset - convAttrParam.oriFmapShape[NCHW_W_IDX]);
    } else {
        load3dOpAl0.SetAttribute(Im2ColOpAttributeKey::paddingRight, 0);
    }
    // cal postm postk
    int64_t mStartPt = iterInfo.hL0Offset * iterInfo.winL1Size + iterInfo.wL0Offset;
    int64_t kStartPt = iterInfo.kL0Offset % convTileInfo.kAL1;
    load3dOpAl0.SetAttribute(Im2ColOpAttributeKey::postM, mStartPt);
    load3dOpAl0.SetAttribute(Im2ColOpAttributeKey::postK, kStartPt);
    // set pad value
    load3dOpAl0.SetAttribute(Im2ColOpAttributeKey::padValue, 0);
}

LogicalTensorPtr ConstructFmapTile(Function &function, const ConvGraphNodes &tensorGraphNodes,
                                   const ConvTileInfo &convTileInfo, ConvIterInfo &iterInfo,
                                   LogicalTensorPtr &dstAL1TensorPtr, const ConvAttrParam &convAttrParam)
{
    if (iterInfo.kL0Offset % convTileInfo.kAL1 == 0) {
        iterInfo.aL1UpadateFlag = true;
    }
    // L1层级 Fmap 展开
    if (iterInfo.aL1UpadateFlag) {
        iterInfo.kAL1Size =
            std::min((convTileInfo.kPerGroup * iterInfo.dkL1Size - iterInfo.kL0Offset), convTileInfo.kAL1);
        std::vector<int64_t> dstAL1Shape = std::vector<int64_t>{1, iterInfo.kAL1Size / convTileInfo.cin0,
            iterInfo.hinL1Size, iterInfo.winL1Size, convTileInfo.cin0};
        int64_t srcCinOffset =
            (iterInfo.kL0Offset % convTileInfo.kPerGroup) / (convTileInfo.orgKh * convTileInfo.orgKw);
        if (convAttrParam.isConv3D) {
            iterInfo.dkAL1Size = 1;
            if (iterInfo.kAL1Size > convTileInfo.kPerGroup) {
                srcCinOffset = 0;
                iterInfo.dkAL1Size = iterInfo.kAL1Size / convTileInfo.kPerGroup;
            }
            dstAL1Shape = std::vector<int64_t>{1, iterInfo.dkAL1Size,
                                               iterInfo.kAL1Size / (iterInfo.dkAL1Size * convTileInfo.cin0),
                                               iterInfo.hinL1Size, iterInfo.winL1Size, convTileInfo.cin0};
        }
        dstAL1TensorPtr =
            std::make_shared<LogicalTensor>(function, tensorGraphNodes.fmapTensorPtr->Datatype(), dstAL1Shape,
                                            SymbolicScalar::FromConcrete(dstAL1Shape),
                                            tensorGraphNodes.fmapTensorPtr->Format(), "aL1Tensor", NodeType::LOCAL);
        dstAL1TensorPtr->UpdateDynValidShape(SymbolicScalar::FromConcrete(dstAL1Shape));
        auto &copyInOpAl1 = function.AddOperation(Opcode::OP_L1_COPY_IN_CONV, {tensorGraphNodes.fmapTensorPtr},
                                                  {dstAL1TensorPtr});
        copyInOpAl1.SetAttribute("is_fmap", true);
        copyInOpAl1.SetAttribute("src_d_offset", iterInfo.dinL1Offset + (iterInfo.kL0Offset / convTileInfo.kPerGroup) *
                                 convAttrParam.dilations[2]);
        copyInOpAl1.SetAttribute("src_d_stride", convAttrParam.dilations[2]);
        copyInOpAl1.SetAttribute("src_n_offset", iterInfo.batchOffset);
        copyInOpAl1.SetAttribute("src_h_offset", iterInfo.hL1InOffset);
        copyInOpAl1.SetAttribute("src_w_offset", iterInfo.wL1InOffset);
        copyInOpAl1.SetAttribute("src_c_offset", iterInfo.groupOffset * (convTileInfo.orgCin / convAttrParam.groups) +
                                 srcCinOffset);
        iterInfo.aL1UpadateFlag = false;
    }

    // 二层展开
    // load3dv2()
    std::vector<int64_t> dstAL0Shape =
        std::vector<int64_t>{ConvAlignB(iterInfo.mL0Size, MKN_M_VALUE), iterInfo.kL0Size};
    LogicalTensorPtr dstAL0TensorPtr =
        std::make_shared<LogicalTensor>(function, tensorGraphNodes.fmapTensorPtr->Datatype(), dstAL0Shape,
                                        SymbolicScalar::FromConcrete(dstAL0Shape),
                                        tensorGraphNodes.fmapTensorPtr->Format(), "aL0Tensor", NodeType::LOCAL);
    dstAL1TensorPtr->UpdateDynValidShape(SymbolicScalar::FromConcrete(dstAL0Shape));
    auto &load3dOpAl0 = function.AddOperation(Opcode::OP_LOAD3D_CONV, {dstAL1TensorPtr}, {dstAL0TensorPtr});
    SetImg2ColAttr(load3dOpAl0, convAttrParam, iterInfo, convTileInfo);
    return dstAL0TensorPtr;
}

LogicalTensorPtr ConstructWeightTile(Function &function, const ConvGraphNodes &tensorGraphNodes,
                                     const ConvTileInfo &convTileInfo, ConvIterInfo &iterInfo,
                                     LogicalTensorPtr &dstBL1TensorPtr, const ConvAttrParam &convAttrParam)
{
    if (iterInfo.kL0Offset % convTileInfo.kBL1 == 0) {
        iterInfo.bL1UpadateFlag = true;
    }
    // L1层级 Weight 展开
    if (iterInfo.bL1UpadateFlag) {
        iterInfo.kBL1Size =
            std::min(convTileInfo.kPerGroup * iterInfo.dkL1Size - iterInfo.kL0Offset, convTileInfo.kBL1);
        std::vector<int64_t> dstBL1Shape =
            std::vector<int64_t>{iterInfo.kBL1Size / convTileInfo.cin0, iterInfo.nL1Size / MKN_N_VALUE,
                                 MKN_N_VALUE, convTileInfo.cin0};
        int64_t srcCinOffset =
            (iterInfo.kL0Offset % convTileInfo.kPerGroup) / (convTileInfo.orgKh * convTileInfo.orgKw);
        if (convAttrParam.isConv3D) {
            iterInfo.dkBL1Size = 1;
            if (iterInfo.kBL1Size > convTileInfo.kPerGroup) {
                srcCinOffset = 0;
                iterInfo.dkBL1Size = iterInfo.kBL1Size / convTileInfo.kPerGroup;
            }
            dstBL1Shape = 
                std::vector<int64_t>{iterInfo.kBL1Size / convTileInfo.cin0, iterInfo.nL1Size / MKN_N_VALUE,
                                     MKN_N_VALUE, convTileInfo.cin0};
        }
        dstBL1TensorPtr =
            std::make_shared<LogicalTensor>(function, tensorGraphNodes.weightTensorPtr->Datatype(), dstBL1Shape,
                                            SymbolicScalar::FromConcrete(dstBL1Shape),
                                            tensorGraphNodes.weightTensorPtr->Format(), "bL1Tensor", NodeType::LOCAL);
        dstBL1TensorPtr->UpdateDynValidShape(SymbolicScalar::FromConcrete(dstBL1Shape));
        auto &copyInOpBl1 = function.AddOperation(Opcode::OP_L1_COPY_IN_CONV, {tensorGraphNodes.weightTensorPtr},
                                                  {dstBL1TensorPtr});
        copyInOpBl1.SetAttribute("is_fmap", false);
        copyInOpBl1.SetAttribute("src_c_offset", srcCinOffset);
        copyInOpBl1.SetAttribute("src_d_offset", (convTileInfo.orgKd - iterInfo.dkL1Size) +
                                 (iterInfo.kL0Offset / convTileInfo.kPerGroup));
        copyInOpBl1.SetAttribute("src_n_offset",
                                 iterInfo.groupOffset * convTileInfo.coutPerGroup + iterInfo.coutOffset);
        iterInfo.bL1UpadateFlag = false;
    }
    // load2d()
    std::vector<int64_t> dstBL0Shape =
        std::vector<int64_t>{iterInfo.kL0Size, ConvAlignB(iterInfo.nL0Size, MKN_N_VALUE)};
    LogicalTensorPtr dstBL0TensorPtr =
        std::make_shared<LogicalTensor>(function, tensorGraphNodes.weightTensorPtr->Datatype(), dstBL0Shape,
                                        SymbolicScalar::FromConcrete(dstBL0Shape),
                                        tensorGraphNodes.weightTensorPtr->Format(), "bL0Tensor", NodeType::LOCAL);
    dstBL0TensorPtr->UpdateDynValidShape(SymbolicScalar::FromConcrete(dstBL0Shape));
    auto &load2dOpBl0 = function.AddOperation(Opcode::OP_LOAD2D_CONV, {dstBL1TensorPtr}, {dstBL0TensorPtr});
    load2dOpBl0.SetAttribute("postK", iterInfo.kL0Offset % convTileInfo.kBL1);
    load2dOpBl0.SetAttribute("postN", iterInfo.nL0Offset);

    return dstBL0TensorPtr;
}

void SetAMulBAttr(const ConvGraphNodes &tensorGraphNodes, const ConvTileInfo &convTileInfo, Operation &op)
{
    OP_CHECK(true,
        {
            ASSERT(tensorGraphNodes.fmapTensorPtr != nullptr && tensorGraphNodes.weightTensorPtr != nullptr &&
            tensorGraphNodes.resTensorPtr != nullptr)
            << "Expected fmapTensorPtr, weightTensorPtr, and resTensorPtr to be non-nullptr." << std::endl;
        });

    int64_t nzAttr = (static_cast<int64_t>(tensorGraphNodes.fmapTensorPtr->Format())) |
                     (static_cast<int64_t>(tensorGraphNodes.weightTensorPtr->Format()) << 1) |
                     (static_cast<int64_t>(tensorGraphNodes.resTensorPtr->Format()) << 2);
    op.SetAttribute(MATMUL_NZ_ATTR, nzAttr);
    op.SetAttribute(A_MUL_B_ACT_M, convTileInfo.hL0 * convTileInfo.wL0);
    op.SetAttribute(A_MUL_B_ACT_K, convTileInfo.kL0);
    op.SetAttribute(A_MUL_B_ACT_N, convTileInfo.nL0);

    if (op.GetOpcode() == Opcode::OP_A_MUL_B) {
        op.SetAttribute(A_MUL_B_BIAS_ATTR, tensorGraphNodes.biasTensorPtr != nullptr);
    }
}

LogicalTensorPtr DoMmad(Function &function, const ConvAttrParam &convAttrParam, const ConvGraphNodes &tensorGraphNodes,
                        ConvGraphNodes &tileGraphNodes, const ConvTileInfo &convTileInfo, const ConvIterInfo &iterInfo)
{
    OP_CHECK(true, {
        ASSERT(tileGraphNodes.fmapTensorPtr != nullptr && tileGraphNodes.weightTensorPtr != nullptr &&
               tileGraphNodes.resTensorPtr != nullptr)
            << "Inputs and res must be non-nullptr." << std::endl;
    });
    // MMAD node add
    std::vector<LogicalTensorPtr> mmadInputs;
    std::vector<LogicalTensorPtr> mmadOutputs;
    const std::string MmadOpStr = iterInfo.isFirstK ? "TILE_A_MUL_B" : "TILE_A_MULACC_B";
    if (iterInfo.isFirstK) {
        mmadInputs = {tileGraphNodes.fmapTensorPtr, tileGraphNodes.weightTensorPtr};
        if (convAttrParam.hasBias) {
            OP_CHECK(true, { ASSERT(tileGraphNodes.biasTensorPtr != nullptr)
                << "bias must be non-nullptr when hasBias Flag." << std::endl;});
            mmadInputs.push_back(tileGraphNodes.biasTensorPtr);
        }
    } else {
        mmadInputs = {tileGraphNodes.fmapTensorPtr, tileGraphNodes.weightTensorPtr, tileGraphNodes.cL0PartialSumPtr};
    }

    if (iterInfo.isLastK) {
        mmadOutputs = {tileGraphNodes.resTensorPtr};
    } else {
        std::vector<int64_t> cL0PartialSumShape =
            {ConvAlignB(iterInfo.mL0Size, MKN_M_VALUE), ConvAlignB(iterInfo.nL0Size, MKN_N_VALUE)};
        tileGraphNodes.cL0PartialSumPtr = 
            std::make_shared<LogicalTensor>(function, DataType::DT_FP32, cL0PartialSumShape,
                                            SymbolicScalar::FromConcrete(cL0PartialSumShape),
                                            TileOpFormat::TILEOP_NZ, "cL0PartialSumTensor", NodeType::LOCAL);
        tileGraphNodes.cL0PartialSumPtr->UpdateDynValidShape({iterInfo.mL0Size, iterInfo.nL0Size});
        mmadOutputs = {tileGraphNodes.cL0PartialSumPtr};
    }
    auto &aMulBOp = function.AddOperation(MmadOpStr, mmadInputs, mmadOutputs);

    SetAMulBAttr(tensorGraphNodes, convTileInfo, aMulBOp);

    return mmadOutputs[0];
}

void UpdateL1IterInfo(const ConvTileInfo &convTileInfo, ConvIterInfo &iterInfo, const ConvAttrParam &convAttrParam)
{
    // update iterInfo L1
    iterInfo.hL1InOffset = iterInfo.hL1OutOffset * convAttrParam.strides[0] - convAttrParam.paddings[0];
    if (iterInfo.hL1InOffset < 0) {
        iterInfo.hinL1Size = convTileInfo.hAL1In + iterInfo.hL1InOffset;
        if (iterInfo.hL1InOffset + convTileInfo.hAL1In <= 0) {
            iterInfo.hinL1Size = 0;
        }
    } else if (convTileInfo.orgHin - iterInfo.hL1InOffset <= 0){
        iterInfo.hinL1Size = 0;
    } else {
        iterInfo.hinL1Size = std::min(convTileInfo.orgHin - iterInfo.hL1InOffset, convTileInfo.hAL1In);
    }
    iterInfo.houtL1Size = std::min(convTileInfo.orgHout - iterInfo.hL1OutOffset, convTileInfo.hAL1Out);
    // cal winL1Size
    iterInfo.wL1InOffset = iterInfo.wL1OutOffset * convAttrParam.strides[1] - convAttrParam.paddings[2];
    if (iterInfo.wL1InOffset < 0) {
        iterInfo.winL1Size = convTileInfo.wAL1In + iterInfo.wL1InOffset;
        if (iterInfo.wL1InOffset + convTileInfo.wAL1In <= 0) {
            iterInfo.winL1Size = 0;
        }
    } else if (convTileInfo.orgWin - iterInfo.wL1InOffset <= 0){
        iterInfo.winL1Size = 0;
    } else {
        iterInfo.winL1Size = std::min(convTileInfo.orgWin - iterInfo.wL1InOffset, convTileInfo.wAL1In);
    }
    iterInfo.woutL1Size = std::min(convTileInfo.orgWout - iterInfo.wL1OutOffset, convTileInfo.wAL1Out);
    // cal nL1Size
    iterInfo.nL1Size = std::min(convTileInfo.coutPerGroup - iterInfo.nL1Offset, convTileInfo.nBL1);
    // cal dk in L1, not support dk in L1 = 0 now, kerneld <= padd
    iterInfo.dkL1Size = 1;
    if (convAttrParam.isConv3D) {
        iterInfo.dkL1Size = convTileInfo.orgKd;
        iterInfo.dinL1Offset = iterInfo.doL1Offset * convAttrParam.strides[2] - convAttrParam.paddings[4];
        int64_t srcDkOffset = iterInfo.dinL1Offset;
        if (iterInfo.dinL1Offset < 0) {
            int64_t tmpKd = ConvAlignB(-iterInfo.dinL1Offset, convAttrParam.dilations[2]) / convAttrParam.dilations[2];
            iterInfo.dkL1Size -= tmpKd;
            srcDkOffset = iterInfo.dinL1Offset + tmpKd * convAttrParam.dilations[2];
        }
        int64_t kdL1EndOffset = iterInfo.dinL1Offset + (convTileInfo.orgKd - 1) * convAttrParam.dilations[2] + 1;
        if (kdL1EndOffset > convTileInfo.orgDin) {
            int64_t tmpKd = ConvAlignB(kdL1EndOffset - convTileInfo.orgDin, convAttrParam.dilations[2]) / convAttrParam.dilations[2];
            iterInfo.dkL1Size -= tmpKd;
        }
        iterInfo.dinL1Offset = srcDkOffset;
    }
}

void UpdateL0IterInfo(const ConvTileInfo &convTileInfo, ConvIterInfo &iterInfo)
{
    // update iterInfo
    iterInfo.kL0Size = convTileInfo.kL0;
    iterInfo.isFirstK = iterInfo.kL0Offset == 0 ? true : false;
    iterInfo.isLastK = iterInfo.kL0Offset + convTileInfo.kL0 >= convTileInfo.kPerGroup ? true : false;
}

void IterL0ExpandFunc(Function &function, ConvIterInfo &iterInfo, ConvTileInfo &convTileInfo,
                      const ConvAttrParam &convAttrParam, const ConvGraphNodes &tensorGraphNodes,
                      ConvGraphNodes &tileGraphNodes)
{
    LogicalTensorPtr fmapL1TensorPtr = nullptr;
    LogicalTensorPtr weightL1TensorPtr = nullptr;
    LogicalTensorPtr resCl0TensorPtr = nullptr;
    for (iterInfo.nL0Offset = 0; iterInfo.nL0Offset < iterInfo.nL1Size; iterInfo.nL0Offset += convTileInfo.nL0) {
        iterInfo.nL0Size = std::min(iterInfo.nL1Size - iterInfo.nL0Offset, convTileInfo.nL0);
        // bias 载入
        if (convAttrParam.hasBias) {
            // get bias in bt tile for mmad
            tileGraphNodes.biasTensorPtr = ConstructBiasTile(function, tensorGraphNodes, iterInfo);
        }
        for (iterInfo.hL0Offset = 0; iterInfo.hL0Offset < iterInfo.houtL1Size;
             iterInfo.hL0Offset += convTileInfo.hL0) {
            for (iterInfo.wL0Offset = 0; iterInfo.wL0Offset < iterInfo.woutL1Size;
                 iterInfo.wL0Offset += convTileInfo.wL0) {
                if (convTileInfo.wL0 == convTileInfo.wAL1Out) {
                    iterInfo.mL0Size =
                        std::min(iterInfo.houtL1Size * iterInfo.woutL1Size - iterInfo.hL0Offset * iterInfo.woutL1Size,
                                 convTileInfo.hL0 * convTileInfo.wL0);
                } else {
                    iterInfo.mL0Size = std::min(iterInfo.woutL1Size - iterInfo.wL0Offset, convTileInfo.wL0);
                }
                // set res tile
                std::vector<int64_t> dstCL0Shape = std::vector<int64_t>{ConvAlignB(iterInfo.mL0Size, MKN_M_VALUE),
                                                                        ConvAlignB(iterInfo.nL0Size, MKN_N_VALUE)};
                tileGraphNodes.resTensorPtr =
                    std::make_shared<LogicalTensor>(function, tensorGraphNodes.fmapTensorPtr->Datatype(), dstCL0Shape,
                                                    SymbolicScalar::FromConcrete(dstCL0Shape),
                                                    tensorGraphNodes.fmapTensorPtr->Format(), "cL0Tensor",
                                                    NodeType::LOCAL);
                for (iterInfo.kL0Offset = 0; iterInfo.kL0Offset < convTileInfo.kPerGroup * iterInfo.dkL1Size;
                     iterInfo.kL0Offset += convTileInfo.kL0) {
                    UpdateL0IterInfo(convTileInfo, iterInfo);
                    // fmap and weight link
                    tileGraphNodes.fmapTensorPtr = ConstructFmapTile(function, tensorGraphNodes, convTileInfo,
                                                                     iterInfo, fmapL1TensorPtr, convAttrParam);
                    tileGraphNodes.weightTensorPtr = ConstructWeightTile(function, tensorGraphNodes, convTileInfo,
                                                                         iterInfo, weightL1TensorPtr, convAttrParam);
                    // add mmad node
                    resCl0TensorPtr = DoMmad(function, convAttrParam, tensorGraphNodes, tileGraphNodes,
                                             convTileInfo, iterInfo);
                }
                auto &fixpipeOpRes = function.AddOperation(Opcode::OP_L0C_COPY_OUT_CONV, {resCl0TensorPtr},
                                                        {tensorGraphNodes.resTensorPtr});
                // set fixpipe copy out validshape
                fixpipeOpRes.SetAttribute("realM", iterInfo.mL0Size);
                fixpipeOpRes.SetAttribute("realN", iterInfo.nL0Size);
            }
        }
    }
}

void IterOneBatchFunc(Function &function, ConvIterInfo &iterInfo, ConvTileInfo &convTileInfo,
                      const ConvAttrParam &convAttrParam, const ConvGraphNodes &tensorGraphNodes,
                      ConvGraphNodes &tileGraphNodes)
{
    for (iterInfo.doL1Offset = 0; iterInfo.doL1Offset < convTileInfo.orgDout; iterInfo.doL1Offset +=1) {
        for (iterInfo.nL1Offset = 0; iterInfo.nL1Offset < convTileInfo.coutPerGroup;
            iterInfo.nL1Offset += convTileInfo.nBL1) {
            iterInfo.bL1UpadateFlag = true;
            for (iterInfo.hL1OutOffset = 0; iterInfo.hL1OutOffset < convTileInfo.orgHout;
                iterInfo.hL1OutOffset += convTileInfo.hAL1Out) {
                for (iterInfo.wL1OutOffset = 0; iterInfo.wL1OutOffset < convTileInfo.orgWout;
                    iterInfo.wL1OutOffset += convTileInfo.wAL1Out) {
                    iterInfo.aL1UpadateFlag = true;
                    UpdateL1IterInfo(convTileInfo, iterInfo, convAttrParam);
                    // iterate L0 buffer expand
                    IterL0ExpandFunc(function, iterInfo, convTileInfo, convAttrParam,
                                     tensorGraphNodes, tileGraphNodes);
                }
            }
        }
    }
}

void ConstructTileGraph(Function &function, const TileShape &tileShape,
                        const std::vector<LogicalTensorPtr> &operandVec, const LogicalTensorPtr &cTensorPtr,
                        const Operation &op)
{
    // op attr set
    ConvAttrParam convAttrParam;
    SetConvAttrParam(op, convAttrParam);
    // set tensor graph node info
    ConvGraphNodes tensorGraphNodes;
    SetTensorGraphNodes(operandVec, cTensorPtr, convAttrParam, tensorGraphNodes);
    // save tile info
    ConvTileInfo convTileInfo;
    SetConvShapeInfo(tileShape, tensorGraphNodes, convAttrParam, convTileInfo);
    // save iter info
    ConvIterInfo iterInfo;
    // set tile graph node info
    ConvGraphNodes tileGraphNodes;

    for (iterInfo.groupOffset = 0; iterInfo.groupOffset < convAttrParam.groups; iterInfo.groupOffset += 1) {
        for (iterInfo.batchOffset = 0; iterInfo.batchOffset < convTileInfo.orgBatch; iterInfo.batchOffset += 1) {
            IterOneBatchFunc(function, iterInfo, convTileInfo, convAttrParam, tensorGraphNodes, tileGraphNodes);
        }
    }
}
Tensor Conv(DataType outType, const Tensor &inputTensor, const Tensor &weightTensor, const std::vector<int64_t> &strides, 
            const std::vector<int64_t> &paddings, const std::vector<int64_t> &dilations, const ConvExtendParam &extendParam, 
            const int64_t groups)
{
    std::vector<int64_t> finalPaddings = paddings;
    std::vector<int64_t> finalDilations = dilations;
    std::vector<int64_t> finalStrides = strides;
    if (dilations.size() == CONV3D_INPUT_DIM - 2 && strides.size() == CONV3D_INPUT_DIM - 2 && paddings.size() == NUM6) {
        finalDilations = rotateVector(dilations, 1);
        finalStrides = rotateVector(strides, 1);
        finalPaddings = rotateVector(paddings, 2);
    }
    const Tensor& biasTensor = extendParam.biasTensor;
    // ConvAttrParam convAttrParam(paddings, strides, dilations, groups);
    ConvAttrParam convAttrParam(finalPaddings, finalStrides, finalDilations, groups);
    CheckConvOperands(outType, inputTensor, weightTensor, biasTensor, convAttrParam);
    int64_t batchOut = inputTensor.GetShape()[NCHW_N_IDX];
    int64_t cOut = weightTensor.GetShape()[NCHW_N_IDX];
    int64_t hOut = ConvComputeHo(inputTensor, weightTensor, convAttrParam);
    int64_t wOut = ConvComputeWo(inputTensor, weightTensor, convAttrParam);
    std::vector<int64_t> resTensorShape{batchOut, cOut, hOut, wOut};
    if (convAttrParam.isConv1D) {
        resTensorShape = {batchOut, cOut, wOut};
    }
    Tensor resTensor(outType, resTensorShape, "TensorC");
    resTensor.GetStorage()->UpdateDynValidShape(SymbolicScalar::FromConcrete(resTensorShape));
    return ConstructTensorGraph(inputTensor, weightTensor, biasTensor, resTensor, convAttrParam);
}

} //namespace Conv
}
}