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

namespace npu {
namespace tile_fwk {
namespace Conv {

#define OP_CHECK(cond, exec_expr) \
    do { \
        if (cond) { \
            exec_expr; \
        } \
    } while (0)

void CheckValueRange(int64_t value, const std::string& name, int64_t min, int64_t max) {
    OP_CHECK(true, {
            ASSERT(value >= min && value <= max)
            << "Invalid " << name << " value: " << value
            << ", expected range [" << min << ", " << max << "]." << std::endl;
    });
}

int64_t ConvComputeHo(int64_t hin, int64_t kh, int64_t padTop, int64_t padBottom, int64_t dilationH, int64_t strideH)
{
    if (strideH == 0) {
        return 1;
    }
    int64_t cmpHo = (hin + padTop + padBottom - dilationH * (kh - 1) - 1) / strideH + 1;
    return cmpHo;
}

int64_t ConvComputeWo(int64_t win, int64_t kw, int64_t padLeft, int64_t padRight, int64_t dilationW, int64_t strideW)
{
    if (strideW == 0) {
        return 1;
    }
    int64_t cmpWo = (win + padLeft + padRight - dilationW * (kw - 1) - 1) / strideW + 1;
    return cmpWo;
}

void CheckOutputShape(const Tensor &inputTensor, const Tensor &weightTensor, const ConvAttrParam &attrParam){
    std::vector<int64_t> paddings = attrParam.paddings;
    std::vector<int64_t> dilations = attrParam.dilations;
    std::vector<int64_t> strides = attrParam.strides;

    int64_t hin = inputTensor.GetShape()[2];
    int64_t win = inputTensor.GetShape()[3];
    int64_t kh = weightTensor.GetShape()[2];
    int64_t kw = weightTensor.GetShape()[3];
    int64_t padTop = paddings[0];
    int64_t padBottom = paddings[1];
    int64_t padLeft = paddings[2];
    int64_t padRight = paddings[3];
    int64_t dilationH = dilations[0];
    int64_t dilationW = dilations[1];
    int64_t strideH = strides[0];
    int64_t strideW = strides[1];

    int64_t hout = ConvComputeHo(hin, kh, padTop, padBottom, dilationH, strideH);
    CheckValueRange(hout, "hout" , NUM1, MAX_SIZE);
    int64_t wout = ConvComputeWo(win, kw, padLeft, padRight, dilationW, strideW);
    CheckValueRange(wout, "wout" , NUM1, MAX_SIZE);
}

void CheckHowoTile(const Tensor &inputTensor, const Tensor &weightTensor, const ConvAttrParam &attrParam) {
    int64_t hin = inputTensor.GetShape()[2];
    int64_t win = inputTensor.GetShape()[3];
    int64_t kh = weightTensor.GetShape()[2];
    int64_t kw = weightTensor.GetShape()[3];
    std::vector<int64_t> paddings = attrParam.paddings;
    std::vector<int64_t> dilations = attrParam.dilations;
    std::vector<int64_t> strides = attrParam.strides;
    int64_t padTop = paddings[0];
    int64_t padBottom = paddings[1];
    int64_t padLeft = paddings[2];
    int64_t padRight = paddings[3];
    int64_t dilationH = dilations[0];
    int64_t dilationW = dilations[1];
    int64_t strideH = strides[0];
    int64_t strideW = strides[1];
    auto &convTile = TileShape::Current().GetConvTile();
    int64_t tileHout = convTile.tileL1Info.tileHout;
    int64_t tileWout = convTile.tileL1Info.tileWout;
    int64_t hout = ConvComputeHo(hin, kh, padTop, padBottom, dilationH, strideH);
    int64_t wout = ConvComputeWo(win, kw, padLeft, padRight, dilationW, strideW);
    CheckValueRange(hout, "tileHout" , NUM1, hout);
    CheckValueRange(wout, "tileWout" , NUM1, wout);
}

void CheckL0TileTiling(const Tensor &weightTensor, const ConvAttrParam &attrParam) {
    auto &convTile = TileShape::Current().GetConvTile();
    int64_t tileM = convTile.tileL0Info.tileM;
    int64_t tileN = convTile.tileL0Info.tileN;
    int64_t tileK = convTile.tileL0Info.tileK;

    int64_t tileHout = convTile.tileL1Info.tileHout;
    int64_t tileWout = convTile.tileL1Info.tileWout;
    OP_CHECK(true, {
        ASSERT(tileM > 0 && tileM <= tileHout * tileWout)
            << "Invalid tileHin value: " << tileM
            << ",expected range [1, " << tileHout * tileWout
            << ".Current tileHout= " << tileHout
            << ", tileWout=" << tileWout
            << " → maximum allowed = tileHout * tileWout" << std::endl;
    });

    int64_t tileCinWeight = convTile.tileL1Info.tileCinWeight;
    int64_t kh = weightTensor.GetShape()[2];
    int64_t kw = weightTensor.GetShape()[3];
    int64_t maxK = kh * kh * tileCinWeight;
    OP_CHECK(true, {
        ASSERT(tileK > 0 && tileK <= maxK)
            << "Invalid tileHin value: " << tileK
            << ", expected range [1, " << maxK
            << ".Current tileCinWeight= " << tileCinWeight
            << ", kh=" << kh
            << ", kw=" << kw
            << " → maximum allowed = kh * kh * tileCinWeight" << std::endl;
    });

    int tileCout = convTile.tileL1Info.tileCout;
    OP_CHECK(true, {
        ASSERT(tileN > 0 && tileN <= tileCout)
            << "Invalid tileHin value: " << tileN
            << ", expected range [1, " << tileCout
            << "]." << std::endl;
    });

}

void CheckTileTiling(const Tensor &inputTensor, const Tensor &weightTensor, const ConvAttrParam &attrParam) {
    auto convTile = TileShape::Current().GetConvTile();
    int64_t tileHin = convTile.tileL1Info.tileHin;
    int64_t tileWin = convTile.tileL1Info.tileWin;
    int64_t tileCinFmap = convTile.tileL1Info.tileCinFmap;
    int64_t tileCinWeight = convTile.tileL1Info.tileCinWeight;
    int64_t tileCout = convTile.tileL1Info.tileCout;
    int64_t tileBatch = convTile.tileL1Info.tileN;

    int64_t batch = inputTensor.GetShape()[0];
    int64_t cin = inputTensor.GetShape()[1];
    int64_t hin = inputTensor.GetShape()[2];
    int64_t win = inputTensor.GetShape()[3];
    int64_t cout = weightTensor.GetShape()[0];
    CheckValueRange(tileHin, "tileHin", NUM1, hin);
    CheckValueRange(tileBatch, "tileN", NUM1, batch);
    CheckValueRange(tileCinFmap, "tileCinFmap", NUM1, cin);
    CheckValueRange(tileCinWeight, "tileCinWeight", NUM1, cin);
    CheckValueRange(tileWin, "tileWin", NUM1, win);
    CheckValueRange(tileCout, "tileCout", NUM1, cout);

    CheckHowoTile(inputTensor, weightTensor, attrParam);

    if (convTile.setL0Tile){
        CheckL0TileTiling(weightTensor, attrParam);
    }
}
uint64_t ConvAlignB(uint64_t a, uint64_t b)
{
    if (b == 0) {
        return 0;
    }
    return ((a + b - 1) / b) * b;
}
/*
void CheckL1SizeTiling(DataType outType, const Tensor &weightTensor){
    auto &convTile = TileShape::Current().GetConvTile();
    int64_t l1Size = pipeConfig.l1SizeThreshold;
    int64_t tileHout = convTile.tileL1Info.tileHout;
    int64_t tileWout = convTile.tileL1Info.tileWout;
    int64_t mL1 = tileHout * tileWout;
    int64_t nL1 = convTile.tileL1Info.tileCout;
    int64_t tileCinWeight = convTile.tileL1Info.tileCinWeight;
    int64_t kh = weightTensor.GetShape()[2];
    int64_t kw = weightTensor.GetShape()[3];
    int64_t kL1 = kh * kh * tileCinWeight;
    int64_t k0 = ALIGN_SIZE_32 / BytesOf(outType);
    int64_t MinL1LoadSize = ConvAlignB(mL1, NUM16)* ConvAlignB(kL1, k0) * BytesOf(outType) +
                ConvAlignB(nL1, NUM16) * ConvAlignB(kL1, k0) * BytesOf(outType);
    OP_CHECK(true, {
        ASSERT(MinL1LoadSize <= l1Size)
            << "MinL1LoadSize > L1size, current L1size: " << l1Size
            << ", maxL1Size: " << MinL1LoadSize
            << "." << std::endl;
    });
}
*/


void CheckGroupsShape(const int64_t cinFmap, const int64_t cinWeight,const int64_t Cout, const int64_t groups){

    OP_CHECK(true, {
            ASSERT(groups > 0 && groups <= SHAPE_INNER_AXIS_MAX_SIZE)
            << "Invalid groups value: groups =" << groups
            << "expected range [1, " << SHAPE_INNER_AXIS_MAX_SIZE
            << "]." << std::endl
    });

    OP_CHECK(true, {
            ASSERT(cinFmap % groups == 0)
            << "Cin ( " << cinFmap
            << ") is not divisible by groups ( " << groups
            << ");adjusting Cin to the nearest value such that Cin % groups == 0." << std::endl
    });

    OP_CHECK(true, {
            ASSERT(Cout % groups == 0)
            << "Cout ( " << Cout
            << ") is not divisible by groups ( " << groups
            << ");adjusting Cout to the nearest value such that Cout % groups == 0." << std::endl
    });

    OP_CHECK(true, {
            ASSERT(cinFmap == cinWeight * groups)
            << "Fmap Cin ( " << cinFmap
            << ") != weight Cin ( " << cinWeight
            << ") * groups ( " << groups
            << ")." << std::endl
    });
}

void CheckDimParam(const std::vector<int64_t>& vec, const std::string& name, int expected_dim) {
    OP_CHECK(true, {
            ASSERT(vec.size() == expected_dim)
                << "Input attr " << name << " dim: " << vec.size()
                << " != " << expected_dim << ".";
    });
}

void CheckDimensionRange(const std::vector<int64_t>& vec, const std::string& name, int min_val, int max_val) {
    for (size_t i = 0; i < vec.size(); ++i) {
        OP_CHECK(true, {
            ASSERT(vec[i] >= min_val && vec[i] <= max_val)
                << "The value of the " << i
                << "-th dimension of " << name
                << " must be in the range [ " << min_val 
                << "," << max_val << "]." << std::endl;
        });
    }
}

void CheckLoad3dShape(DataType outType, const Tensor &weightTensor, const ConvAttrParam &attrParam){
    CheckDimensionRange(attrParam.paddings, "paddings", 0, MAX_PAD_KERNEL);
    CheckDimensionRange(attrParam.dilations, "dilations", NUM1, MAX_DILATION_STRIDE);
    CheckDimensionRange(attrParam.strides, "strides", NUM1, MAX_DILATION_STRIDE);

    int64_t kh = weightTensor.GetShape()[2];
    int64_t kw = weightTensor.GetShape()[3];
    OP_CHECK(true, {
        ASSERT(kh <= MAX_PAD_KERNEL && kw  <= MAX_PAD_KERNEL)
        << "Weight shapes do not satisfy Load3D's limits: kh=" << kh
        << ", kw=" << kw
        << ", which must <=" << MAX_PAD_KERNEL
        << "." << std::endl
    });

    int64_t k0 = ALIGN_SIZE_32 / BytesOf(outType);
    OP_CHECK(true, {
        ASSERT(kh * kw * k0 <= SHAPE_INNER_AXIS_MAX_SIZE)
        << "Weight shapes do not satisfy Load3D's limits: kh*kw*k0=" << kh * kw * k0
        << ", which must <=" << MAX_PAD_KERNEL
        << "." << std::endl
    });
}

void CheckAttrShape(DataType outType, const Tensor &inputTensor, const Tensor &weightTensor, const ConvAttrParam &attrParam){

    CheckDimParam(attrParam.paddings, "paddings", NUM2);
    CheckDimParam(attrParam.dilations, "dilations", NUM2);
    CheckDimParam(attrParam.strides, "strides", NUM4);

    int64_t groups = attrParam.groups;
    int64_t cinFmap = inputTensor.GetShape()[2];
    int64_t cinWeight = weightTensor.GetShape()[2];
    int64_t cout = weightTensor.GetShape()[0];

    CheckGroupsShape(cinFmap, cinWeight, cout, groups);
    CheckLoad3dShape(outType, weightTensor, attrParam);

    std::vector<int64_t> paddings = attrParam.paddings;
    int64_t kh = weightTensor.GetShape()[2];
    int64_t kw = weightTensor.GetShape()[3];
    for (size_t i = 0; i < paddings.size(); ++i) {
        OP_CHECK(true, {
            ASSERT(paddings[i] <= kh && paddings[i] <= kw)
                << "The value of the " << i
                << "-th dimension of paddings must be <= kh ( " << kh 
                << ") and <= kw (" << kw << ")." << std::endl;
        });
    }
}

void CheckOriginShape(const Tensor &inputTensor, const Tensor &weightTensor, const Tensor &biasTensor)
{
    CheckDimensionRange(inputTensor.GetShape(), "fmap", NUM1, MAX_SIZE);
    CheckDimensionRange(weightTensor.GetShape(), "weight", NUM1, MAX_SIZE);

    int64_t Cout = biasTensor.GetShape()[0];
    OP_CHECK(true, {
        ASSERT(biasTensor.GetShape()[i] == Cout)
        << "Input illegal bias shape:" << biasTensor.GetShape()[0]
        << ", which must euqal to Cout:" << Cout
        << "." << std::endl
    });
}
void CheckConvOperands(DataType outType, const Tensor &inputTensor, const Tensor &weightTensor, const Tensor &biasTensor, const ConvAttrParam &attrParam) {
    OP_CHECK(true, {
        ASSERT(outType == DataType::DT_FP32 || outType == DataType::DT_FP16 || outType == DataType::DT_BF16)
        << "Unsupported output data type. Only DT_FP32, DT_FP16, DT_BF16 are supported.";
    });
    CheckOriginShape(inputTensor, weightTensor, biasTensor);
    CheckOutputShape(inputTensor, weightTensor, biasTensor);
    CheckAttrShape(outType, attrParam);
    CheckTileTiling(inputTensor, weightTensor, attrParam);
    // CheckL1SizeTiling(outType, weightTensor);
}

void SetTensorOpAttr(Operation &op, const ConvAttrParam &convAttrParam)
{
    op.SetAttribute(CONV_BIAS_ATTR, convAttrParam.hasBias);
    op.SetAttribute(CONV_GROUPS_ATTR, convAttrParam.groups);
    op.SetAttribute(CONV_PADDINGS_ATTR, convAttrParam.paddings);
    op.SetAttribute(CONV_STRIDES_ATTR, convAttrParam.strides);
    op.SetAttribute(CONV_DILATIONS_ATTR, convAttrParam.dilations);
    op.SetAttribute(IS_MATRIX_NZ, true);
    std::vector<int64_t> fmapOriShape = {1, 32, 8, 8};
    std::vector<int64_t> wegihtOriShape = {32, 32, 1, 1};
    op.SetAttribute(CONV_ORI_FMAP_SHAPE_ATTR, fmapOriShape);
    op.SetAttribute(CONV_ORI_WEIGHT_SHAPE_ATTR, wegihtOriShape);
}

Tensor ConstructTensorGraph(DataType dataType, const Tensor &inputTensor, const Tensor &weightTensor, const Tensor &biasTensor,
    const Tensor &resTensor, ConvAttrParam &convAttrParam)
{
    // add Conv node
    Function *functionPtr = Program::GetInstance().GetCurrentFunction();
    OP_CHECK(true, { ASSERT(functionPtr != nullptr) << "functionPtr is nullptr." << std::endl; });
    std::vector<LogicalTensorPtr> operandVec = {inputTensor.GetStorage(), weightTensor.GetStorage()};
    if (biasTensor.GetStorage() != nullptr) {
        convAttrParam.hasBias = true;
        operandVec.push_back(biasTensor.GetStorage());
    }
    auto &op = functionPtr->AddOperation(Opcode::OP_CONV, operandVec, {resTensor.GetStorage()});
    SetTensorOpAttr(op, convAttrParam);

    return resTensor;
}

void SetConvAttrParam(const Operation &op, ConvAttrParam &convAttrParam)
{
    convAttrParam.isConv3D = (op.HasAttr(CONV_3D_FLAG)) ? op.GetBoolAttribute(CONV_3D_FLAG) : false;
    convAttrParam.paddings = (op.HasAttr(CONV_PADDINGS_ATTR)) ? op.GetVectorIntAttribute(CONV_PADDINGS_ATTR) :
        convAttrParam.isConv3D ? CONV3D_ATTR_DEFAULT_LIST : CONV2D_ATTR_DEFAULT_LIST;
    convAttrParam.strides = (op.HasAttr(CONV_STRIDES_ATTR)) ? op.GetVectorIntAttribute(CONV_STRIDES_ATTR) :
        convAttrParam.isConv3D ? CONV3D_ATTR_DEFAULT_LIST : CONV2D_ATTR_DEFAULT_LIST;
    convAttrParam.dilations = (op.HasAttr(CONV_DILATIONS_ATTR)) ? op.GetVectorIntAttribute(CONV_DILATIONS_ATTR) :
        convAttrParam.isConv3D ? CONV3D_ATTR_DEFAULT_LIST : CONV2D_ATTR_DEFAULT_LIST;
    convAttrParam.groups = (op.HasAttr(CONV_GROUPS_ATTR)) ? op.GetIntAttribute(CONV_GROUPS_ATTR) : 1;
    convAttrParam.hasBias = (op.HasAttr(CONV_BIAS_ATTR)) ? op.GetBoolAttribute(CONV_BIAS_ATTR) : false;
    convAttrParam.isInOutTensorNZ = (op.HasAttr(IS_MATRIX_NZ)) ? op.GetBoolAttribute(IS_MATRIX_NZ) : false;
    if (convAttrParam.isInOutTensorNZ) {
        OP_CHECK(true, {
            ASSERT(op.HasAttr(CONV_ORI_FMAP_SHAPE_ATTR))
            << "Conv ori fmapshape should be set when InOut Tensor NZ mode." << std::endl;
        });
        OP_CHECK(true, {
            ASSERT(op.HasAttr(CONV_ORI_WEIGHT_SHAPE_ATTR))
            << "Conv ori weightshape should be set when InOut Tensor NZ mode." << std::endl;
        });
        convAttrParam.oriFmapShape = op.GetVectorIntAttribute(CONV_ORI_FMAP_SHAPE_ATTR);
        convAttrParam.oriweightShape = op.GetVectorIntAttribute(CONV_ORI_WEIGHT_SHAPE_ATTR);
    }
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

    tensorGraphNodes.fmapTensorPtr = operandVec[0];
    tensorGraphNodes.weightTensorPtr = operandVec[1];
    if (convAttrParam.hasBias) {
        tensorGraphNodes.biasTensorPtr = operandVec[2];
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
    // 设计线讨论如何获取私有格式tensor传入的ori_shape
    // set org shape
    convTileInfo.orgBatch = tensorGraphNodes.fmapTensorPtr->shape[0];
    // convTileInfo.orgHout = tensorGraphNodes.resTensorPtr->shape[2];
    // convTileInfo.orgWout = tensorGraphNodes.resTensorPtr->shape[3];
    convTileInfo.orgHin = 8;
    convTileInfo.orgWin = 8;
    convTileInfo.orgHout = 8;
    convTileInfo.orgWout = 8;
    if (convAttrParam.isInOutTensorNZ) {
        convTileInfo.orgCout = convAttrParam.oriweightShape[0];
        convTileInfo.orgkh = convAttrParam.oriweightShape[2];
        convTileInfo.orgkw = convAttrParam.oriweightShape[3];
        convTileInfo.orgCin = tensorGraphNodes.fmapTensorPtr->shape[1] * tensorGraphNodes.fmapTensorPtr->shape[4];
    } else {
        convTileInfo.orgCout = tensorGraphNodes.weightTensorPtr->shape[0];
        convTileInfo.orgkh = tensorGraphNodes.weightTensorPtr->shape[2];
        convTileInfo.orgkw = tensorGraphNodes.weightTensorPtr->shape[3];
        convTileInfo.orgCin = tensorGraphNodes.fmapTensorPtr->shape[1];
    }
    convTileInfo.orgK = convTileInfo.orgCin * convTileInfo.orgkh * convTileInfo.orgkw;
    convTileInfo.orgHoutWout = convTileInfo.orgHout * convTileInfo.orgWout;
    // set tileshape info
    // auto &convTile = tileShape.GetConvTile();
    convTileInfo.kAL1 = 16;
    convTileInfo.kBL1 = 32;
    convTileInfo.nBL1 = 16;
    convTileInfo.hAL1In = 8;
    convTileInfo.wAL1In = 8;
    convTileInfo.hAL1Out = 8;
    convTileInfo.wAL1Out = 8;
    convTileInfo.kL0 = 16;
    convTileInfo.hL0 = 8;
    convTileInfo.wL0 = 8;
    convTileInfo.nL0 = 16;
    convTileInfo.cin0 = ALIGN_SIZE_32 / BytesOf(tensorGraphNodes.fmapTensorPtr->Datatype());
}

LogicalTensorPtr ConstructBiasTile(Function &function, const ConvGraphNodes &tensorGraphNodes,
                                   const ConvTileInfo &convTileInfo, ConvIterInfo &iterInfo)
{
    std::vector<int64_t> dstBiasL1Shape = std::vector<int64_t>{iterInfo.nL0Size};
    std::vector<int64_t> dstBiasL1Offset = std::vector<int64_t>{iterInfo.nL0Offset};
    LogicalTensorPtr dstBiasl1TensorPtr =
        std::make_shared<LogicalTensor>(function, tensorGraphNodes.biasTensorPtr->Datatype(),
                                        dstBiasL1Shape, SymbolicScalar::FromConcrete(dstBiasL1Shape),
                                        tensorGraphNodes.biasTensorPtr->Format(), "biasL1Tensor", NodeType::LOCAL);
    dstBiasl1TensorPtr->UpdateDynValidShape(
        GetViewValidShape(tensorGraphNodes.biasTensorPtr->GetDynValidShape(), dstBiasL1Offset, {}, dstBiasL1Shape));
    auto &viewOpBiasL1 = function.AddOperation(Opcode::OP_VIEW, {tensorGraphNodes.biasTensorPtr}, {dstBiasl1TensorPtr});
    auto viewAttributeBiasL1 = std::make_shared<ViewOpAttribute>(dstBiasL1Offset, MemoryType::MEM_L1,
            SymbolicScalar::FromConcrete(dstBiasL1Offset), dstBiasl1TensorPtr->GetDynValidShape());
    viewOpBiasL1.SetOpAttribute(viewAttributeBiasL1);

    std::vector<int64_t> dstBiasBtShape = std::vector<int64_t>{iterInfo.nL0Size};
    std::vector<int64_t> dstBiasBtOffset = std::vector<int64_t>{iterInfo.nL0Offset};
    LogicalTensorPtr dstBiasBtTensorPtr =
        std::make_shared<LogicalTensor>(function, DataType::DT_FP32, dstBiasBtShape,
                                        SymbolicScalar::FromConcrete(dstBiasBtShape),
                                        tensorGraphNodes.biasTensorPtr->Format(), "biasBtTensor", NodeType::LOCAL);
    dstBiasBtTensorPtr->UpdateDynValidShape(
        GetViewValidShape(dstBiasl1TensorPtr->GetDynValidShape(), dstBiasBtOffset, {}, dstBiasBtShape));
    auto &viewOpBiasBt = function.AddOperation(Opcode::OP_VIEW, {dstBiasl1TensorPtr}, {dstBiasBtTensorPtr});
    auto viewAttributeBiasBt = std::make_shared<ViewOpAttribute>(dstBiasBtOffset, MemoryType::MEM_BT,
            SymbolicScalar::FromConcrete(dstBiasBtOffset), dstBiasBtTensorPtr->GetDynValidShape());
    viewOpBiasBt.SetOpAttribute(viewAttributeBiasBt);

    return dstBiasBtTensorPtr;
}

LogicalTensorPtr ConstructFmapTile(Function &function, const ConvGraphNodes &tensorGraphNodes,
                                   const ConvTileInfo &convTileInfo, ConvIterInfo &iterInfo,
                                   LogicalTensorPtr &dstAL1TensorPtr)
{
    if (iterInfo.kL0Offset % convTileInfo.kAL1 == 0) {
        iterInfo.aL1UpadateFlag = true;
    }
    // L1层级 Fmap 展开
    if (iterInfo.aL1UpadateFlag) {
        iterInfo.kAL1Size = std::min(convTileInfo.orgK - iterInfo.nL0Offset, convTileInfo.kAL1);
        std::vector<int64_t> dstAL1Shape = std::vector<int64_t>{1, iterInfo.kAL1Size / convTileInfo.cin0, iterInfo.hinL1Size, iterInfo.winL1Size, convTileInfo.cin0};
        std::vector<int64_t> dstAL1Offset = std::vector<int64_t>{iterInfo.batchOffset, 0 , 0, 0, 0};
        dstAL1TensorPtr =
            std::make_shared<LogicalTensor>(function, tensorGraphNodes.fmapTensorPtr->Datatype(), dstAL1Shape,
                                            SymbolicScalar::FromConcrete(dstAL1Shape),
                                            tensorGraphNodes.fmapTensorPtr->Format(), "aL1Tensor", NodeType::LOCAL);
        dstAL1TensorPtr->UpdateDynValidShape(
            GetViewValidShape(tensorGraphNodes.fmapTensorPtr->GetDynValidShape(), dstAL1Offset, {}, dstAL1Shape));
        auto &viewOpAl1 = function.AddOperation(Opcode::OP_VIEW, {tensorGraphNodes.fmapTensorPtr}, {dstAL1TensorPtr});
        viewOpAl1.SetAttribute("IS_CONV", true);
        auto viewAttribute = std::make_shared<ViewOpAttribute>(dstAL1Offset, MemoryType::MEM_L1,
             SymbolicScalar::FromConcrete(dstAL1Offset), dstAL1TensorPtr->GetDynValidShape());
        viewOpAl1.SetOpAttribute(viewAttribute);
        iterInfo.aL1UpadateFlag = false;
    }

    // 二层展开
    // load3dv2()
    std::vector<int64_t> dstAL0Shape = std::vector<int64_t>{64, 16};
    LogicalTensorPtr dstAL0TensorPtr =
        std::make_shared<LogicalTensor>(function, tensorGraphNodes.fmapTensorPtr->Datatype(), dstAL0Shape,
                                        SymbolicScalar::FromConcrete(dstAL0Shape),
                                        tensorGraphNodes.fmapTensorPtr->Format(), "aL0Tensor", NodeType::LOCAL);
    // dstAL1TensorPtr->UpdateDynValidShape(
    //     GetViewValidShape(operandVec[0]->GetDynValidShape(), dstAL1Offset, {}, dstAL1Shape));
    auto &viewOpAl0 = function.AddOperation(Opcode::OP_LOAD3D_CONV, {dstAL1TensorPtr}, {dstAL0TensorPtr});
    return dstAL0TensorPtr;
}

LogicalTensorPtr ConstructWeightTile(Function &function, const ConvGraphNodes &tensorGraphNodes,
                                     const ConvTileInfo &convTileInfo, ConvIterInfo &iterInfo,
                                     LogicalTensorPtr &dstBL1TensorPtr)
{
    if (iterInfo.kL0Offset % convTileInfo.kBL1 == 0) {
        iterInfo.bL1UpadateFlag = true;
    }
    // L1层级 Weight 展开
    if (iterInfo.bL1UpadateFlag) {
        iterInfo.nL1Size = std::min(convTileInfo.orgCout - iterInfo.nL0Offset, convTileInfo.nBL1);
        iterInfo.kBL1Size = std::min(convTileInfo.orgK - iterInfo.kL0Offset, convTileInfo.kBL1);
        std::vector<int64_t> dstBL1Shape = std::vector<int64_t>{iterInfo.kBL1Size / convTileInfo.cin0, iterInfo.nL1Size / 16, 16, convTileInfo.cin0};
        std::vector<int64_t> dstBL1Offset = std::vector<int64_t>{iterInfo.kL0Offset / 16, iterInfo.nL0Offset / 16, 0, 0};
        dstBL1TensorPtr =
            std::make_shared<LogicalTensor>(function, tensorGraphNodes.weightTensorPtr->Datatype(), dstBL1Shape,
                                            SymbolicScalar::FromConcrete(dstBL1Shape),
                                            tensorGraphNodes.weightTensorPtr->Format(), "bL1Tensor", NodeType::LOCAL);
        dstBL1TensorPtr->UpdateDynValidShape(
            GetViewValidShape(tensorGraphNodes.weightTensorPtr->GetDynValidShape(), dstBL1Offset, {}, dstBL1Shape));
        auto &viewOpBl1 = function.AddOperation(Opcode::OP_VIEW, {tensorGraphNodes.weightTensorPtr}, {dstBL1TensorPtr});
        viewOpBl1.SetAttribute("IS_CONV", true);
        auto viewAttribute = std::make_shared<ViewOpAttribute>(dstBL1Offset, MemoryType::MEM_L1,
             SymbolicScalar::FromConcrete(dstBL1Offset), dstBL1TensorPtr->GetDynValidShape());
        viewOpBl1.SetOpAttribute(viewAttribute);
        iterInfo.bL1UpadateFlag = false;
    }
    // load2d()
    std::vector<int64_t> dstBL0Shape = std::vector<int64_t>{16, 16};
    LogicalTensorPtr dstBL0TensorPtr =
        std::make_shared<LogicalTensor>(function, tensorGraphNodes.weightTensorPtr->Datatype(), dstBL0Shape,
                                        SymbolicScalar::FromConcrete(dstBL0Shape),
                                        tensorGraphNodes.weightTensorPtr->Format(), "bL0Tensor", NodeType::LOCAL);
    // dstAL1TensorPtr->UpdateDynValidShape(
    //     GetViewValidShape(operandVec[0]->GetDynValidShape(), dstAL1Offset, {}, dstAL1Shape));
    auto &viewOpBl0 = function.AddOperation(Opcode::OP_LOAD2D_CONV, {dstBL1TensorPtr}, {dstBL0TensorPtr});

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
        std::vector<int64_t> cL0PartialSumShape = {iterInfo.mL0Size, iterInfo.nL0Size};
        tileGraphNodes.cL0PartialSumPtr = 
            std::make_shared<LogicalTensor>(function, DataType::DT_FP32, cL0PartialSumShape,
                                            SymbolicScalar::FromConcrete(cL0PartialSumShape),
                                            TileOpFormat::TILEOP_NZ, "cL0PartialSumTensor", NodeType::LOCAL);
        // if (CheckValidShape(tileGraphNodes.aTensorPtr) && CheckValidShape(tileGraphNodes.bTensorPtr)) {
        //     tileGraphNodes.cL0PartialSumPtr->UpdateDynValidShape(
        //         {tileGraphNodes.aTensorPtr->GetDynValidShape()[0], tileGraphNodes.bTensorPtr->GetDynValidShape()[1]});
        // }
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
    iterInfo.nL1Size = std::min((convTileInfo.orgCout / convAttrParam.groups) - iterInfo.nL1Offset, convTileInfo.nBL1);
}

void UpdateL0IterInfo(const ConvTileInfo &convTileInfo, ConvIterInfo &iterInfo)
{
    // todo update iterInfo
    iterInfo.isFirstK = iterInfo.kL0Offset == 0 ? true : false;
    iterInfo.isLastK = iterInfo.kL0Offset + convTileInfo.kL0 >= convTileInfo.orgK ? true : false;
}

void ConstructTileGraph(Function &function, const TileShape &tileShape, const std::vector<LogicalTensorPtr> &operandVec,
                        const LogicalTensorPtr &cTensorPtr, const Operation &op)
{
    // op attr set
    ConvAttrParam convAttrParam;
    SetConvAttrParam(op, convAttrParam);
    // tensor graph中的数据节点
    ConvGraphNodes tensorGraphNodes;
    SetTensorGraphNodes(operandVec, cTensorPtr, convAttrParam, tensorGraphNodes);
    // Tile 信息存储
    ConvTileInfo convTileInfo;
    SetConvShapeInfo(tileShape, tensorGraphNodes, convAttrParam, convTileInfo);
    // iter信息存储
    ConvIterInfo iterInfo;
    // tile graph中的数据节点
    ConvGraphNodes tileGraphNodes;

    for (int64_t groupIdx = 0; groupIdx < convAttrParam.groups; groupIdx += 1) {
        for (iterInfo.batchOffset = 0; iterInfo.batchOffset < convTileInfo.orgBatch; iterInfo.batchOffset += 1) {
            for (iterInfo.nL1Offset = 0; iterInfo.nL1Offset < (convTileInfo.orgCout / convAttrParam.groups); iterInfo.nL1Offset += convTileInfo.nBL1) {
                iterInfo.bL1UpadateFlag = true;
                for (iterInfo.hL1OutOffset = 0; iterInfo.hL1OutOffset < convTileInfo.orgHout; iterInfo.hL1OutOffset += convTileInfo.hAL1Out) {
                    for (iterInfo.wL1OutOffset = 0; iterInfo.wL1OutOffset < convTileInfo.orgWout; iterInfo.wL1OutOffset += convTileInfo.wAL1Out) {
                        iterInfo.aL1UpadateFlag = true;
                        // int64_t dilatedKernelH = (convTileInfo.orgkh - 1) * convAttrParam.dilations[0] + 1;
                        UpdateL1IterInfo(convTileInfo, iterInfo, convAttrParam);
                        // set res tile
                        // tileGraphNodes.resTensorPtr =
                        //     cTensorPtr->View(function, {1, iterInfo.nL0Size / 16, 8, 8, 16}, {iterInfo.batchOffset, iterInfo.nL0Offset / 16, 0, 0, 0});
                        // tileGraphNodes.resTensorPtr =
                        //     cTensorPtr->View(function, {iterInfo.mL0Size, iterInfo.nL0Size}, {iterInfo.mOffset, iterInfo.nL0Offset});
                        LogicalTensorPtr resCl0TensorPtr = nullptr;
                        for (iterInfo.nL0Offset = 0; iterInfo.nL0Offset < iterInfo.nL1Size; iterInfo.nL0Offset += convTileInfo.nL0) {
                            iterInfo.nL0Size = std::min(iterInfo.nL1Size - iterInfo.nL0Offset, convTileInfo.nL0);
                            // bias 载入
                            if (convAttrParam.hasBias) {
                                // get bias in bt tile for mmad
                                tileGraphNodes.biasTensorPtr = ConstructBiasTile(function, tensorGraphNodes, convTileInfo, iterInfo);
                            }
                            for (iterInfo.hL0Offset = 0; iterInfo.hL0Offset < iterInfo.houtL1Size; iterInfo.hL0Offset += convTileInfo.hL0) {
                                for (iterInfo.wL0Offset = 0; iterInfo.wL0Offset < iterInfo.woutL1Size; iterInfo.wL0Offset += convTileInfo.wL0) {
                                    if (convTileInfo.wL0 == convTileInfo.wAL1Out) {
                                        iterInfo.mL0Size = std::min(iterInfo.houtL1Size * iterInfo.woutL1Size - iterInfo.hL0Offset * iterInfo.woutL1Size, convTileInfo.hL0 * convTileInfo.wL0); // 需要对齐
                                    } else {
                                        iterInfo.mL0Size = std::min(iterInfo.woutL1Size - iterInfo.woutL1Size, convTileInfo.wL0);
                                    }
                                    std::vector<int64_t> dstCL0Shape = std::vector<int64_t>{iterInfo.mL0Size, iterInfo.nL0Size};
                                    tileGraphNodes.resTensorPtr =
                                        std::make_shared<LogicalTensor>(function, tensorGraphNodes.fmapTensorPtr->Datatype(), dstCL0Shape,
                                                                        SymbolicScalar::FromConcrete(dstCL0Shape),
                                                                        tensorGraphNodes.fmapTensorPtr->Format(), "cL0Tensor", NodeType::LOCAL);
                                    LogicalTensorPtr fmapL1TensorPtr = nullptr;
                                    LogicalTensorPtr weightL1TensorPtr = nullptr;
                                    for (iterInfo.kL0Offset = 0; iterInfo.kL0Offset < (convTileInfo.orgK / convAttrParam.groups); iterInfo.kL0Offset += convTileInfo.kL0) {
                                        UpdateL0IterInfo(convTileInfo, iterInfo);
                                        // fmap and weight link
                                        tileGraphNodes.fmapTensorPtr =
                                            ConstructFmapTile(function, tensorGraphNodes, convTileInfo, iterInfo, fmapL1TensorPtr);
                                        tileGraphNodes.weightTensorPtr =
                                            ConstructWeightTile(function, tensorGraphNodes, convTileInfo, iterInfo, weightL1TensorPtr);
                                        // add mmad node
                                        resCl0TensorPtr = DoMmad(function, convAttrParam, tensorGraphNodes, tileGraphNodes, convTileInfo, iterInfo);
                                    }   
                                }
                            }
                        }
                        auto &viewOpRes = function.AddOperation(Opcode::OP_L0C_COPY_OUT_CONV, {resCl0TensorPtr}, {cTensorPtr});
                    }
                }
            }
        }
    }
}

Tensor Conv(DataType outType, const Tensor &inputTensor, const Tensor &weightTensor, const Tensor &biasTensor,
            const std::vector<int64_t> &strides, const std::vector<int64_t> &paddings, const std::vector<int64_t> &dilations,
            const int64_t groups)
{
    ConvAttrParam convAttrParam(paddings, strides, dilations, groups);
    CheckConvOperands(outType, inputTensor, weightTensor, biasTensor, convAttrParam);
    // auto &convTile = TileShape::Current().GetConvTile();
    // Check ConvTile Valid
    // infer hout, wout
    int64_t batchOut = inputTensor.GetShape()[0];
    int64_t C0 = ALIGN_SIZE_32 / BytesOf(outType);
    int64_t co1 = (weightTensor.GetShape()[1] * weightTensor.GetShape()[2]) / C0;
    int64_t hOut = 8;
    int64_t wOut = 8;
    std::vector<int64_t> resTensorShape{batchOut, co1, hOut, wOut, C0};
    // std::vector<int64_t> resTensorShape{64, 32};
    Tensor resTensor(outType, resTensorShape, "TensorC");
    resTensor.GetStorage()->UpdateDynValidShape({batchOut, co1, hOut, wOut, C0});
    // resTensor.GetStorage()->UpdateDynValidShape({64, 32});
    return ConstructTensorGraph(outType, inputTensor, weightTensor, biasTensor, resTensor, convAttrParam);
}

} //namespace Conv
}
}