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

void CheckConvOperands(DataType outType, const Tensor &operand1, const Tensor &operand2, const Tensor &operand3, const convAttrParam &attrParam) {
    // todo
     // 1、dtype校验
    OP_CHECK(true, {
        ASSERT(outType == DataType::DT_FP32 || outType == DataType::DT_FP16 || outType == DataType::DT_BF16)
        << "Unsupported output data type. Only DT_FP32, DT_FP16, DT_BF16 are supported.";
    });

    CheckOperandShape(operand1, operand2, operand3);
    
    CheckOutputShape(operand1, operand2, operand3, attrParam);

    CheckAttrShape(attrParam);

    CheckTileTiling(operand1, operand2, attrParam);
    CheckL1SizeTiling(outType, operand2, attrParam);
    
}

void CheckTileTiling(const Tensor &operand1, const Tensor &operand2, const MatmulAttrParam &attrParam) {
    auto convTile = TileShape::Current().GetConvTile();
    int64_t tileHin    = convTile.tileL1Info.tileHin;
    uint64_t tileWin    = convTile.tileL1Info.tileWin;
    uint64_t tileCinFmap    = convTile.tileL1Info.tileCinFmap;
    uint64_t tileCinWeight    = convTile.tileL1Info.tileCinWeight;
    uint64_t tileCout   = convTile.tileL1Info.tileCout;
    uint64_t tileBatch = convTile.tileL1Info.tileN;

    uint64_t batch =  operand1.GetShape()[0];
    uint64_t cin =  operand1.GetShape()[1];
    uint64_t hin =  operand1.GetShape()[2];
    uint64_t win =  operand1.GetShape()[3];
    OP_CHECK(true, {
        ASSERT(tileBatch > 0 && tileBatch <= batch)
            << "Invalid tileHin value: " << tileBatch 
            << ", expected range [1, " << batch
            << "]." << std::endl;
    });
    OP_CHECK(true, {
        ASSERT(tileCinFmap > 0 && tileCinFmap <= cin)
            << "Invalid tileHin value: " << tileCinFmap 
            << ", expected range [1, " << cin
            << "]." << std::endl;
    });
    OP_CHECK(true, {
        ASSERT(tileCinWeight > 0 && tileCinWeight <= cin)
            << "Invalid tileHin value: " << tileCinWeight 
            << ", expected range [1, " << cin
            << "]." << std::endl;
    });
    OP_CHECK(true, {
        ASSERT(tileHin > 0 && tileHin <= hin)
            << "Invalid tileHin value: " << tileHin 
            << ", expected range [1, " << hin
            << "]." << std::endl;
    });

    OP_CHECK(true, {
        ASSERT(tileWin > 0 && tileWin <= win)
            << "Invalid tileHin value: " << tileWin 
            << ", expected range [1, " << win
            << "]." << std::endl;
    });

    CheckHoWoTiling(operand1, operand2, attrParam);

    bool isSetL0Tile = convTile.setL0Tile;
    if (isSetL0Tile){
        CheckTileTiling(operand2, attrParam);
    }
   
}

CheckL1SizeTiling(DataType outType, ){
   uint64_t l1Size = pipeConfig.l1SizeThreshold;
    %lu %lu

    uint64_t tileHout   = convTile.tileL1Info.tileHout;
    uint64_t tileWout   = convTile.tileL1Info.tileWout;

    uint64_t mL1   = tileHout * tileWout;
    uint64_t nL1   = convTile.tileL1Info.tileCout;

    int64_t tileCinWeight = convTile.tileL1Info.tileCinWeight;
    int64_t kH =  operand2.GetShape()[2];
    int64_t kW =  operand2.GetShape()[3];   
    int64_t kL1 = kH * kH * tileCinWeight;

    int64_t k0 = ALIGN_SIZE_32 / BytesOf(outType);

    uint64_t MinL1LoadSize = ConvAlignB(mL1, NUM16)* ConvAlignB(kL1, k0) * BytesOf(outType) + 
                ConvAlignB(nL1, NUM16) * ConvAlignB(kL1, k0) * BytesOf(outType);

    OP_CHECK(true, {
        ASSERT(MinL1LoadSize <= l1Size)
            << "MinL1LoadSize > L1size, current L1size: " << l1Size 
            << ", maxL1Size: " << MinL1LoadSize
            << "." << std::endl;
    });


   



}
uint64_t ConvAlignB(uint64_t a, uint64_t b)
{
    if (b == 0) {
        return 0;
    }
    return ((a + b - 1) / b) * b;
}

void CheckHoWoTiling(const Tensor &operand1, const Tensor &operand2, const MatmulAttrParam &attrParam) {
    int64_t hin =  operand1.GetShape()[2];
    int64_t win =  operand1.GetShape()[3];
    int64_t kH =  operand2.GetShape()[2];
    int64_t kW =  operand2.GetShape()[3];
    int64_t padTop =  paddings[0];
    int64_t padBottom =  paddings[1];
    int64_t padLeft =  paddings[2];
    int64_t padRight =  paddings[3];
    int64_t dilationH =  dilations[2];
    int64_t dilationW =  dilations[3];
    int64_t strideH =  strides[2];
    int64_t strideH =  strides[3]; 

    int64_t tileHout   = convTile.tileL1Info.tileHout;
    int64_t tileWout   = convTile.tileL1Info.tileWout;

    int64_t Ho = ConvComputeHo(hin, kH, padTop, padBottom, dilationH, strideH);
    int64_t Wo = ConvComputeWo(win, kW, padLeft, padRight, dilationW, strideW);
    OP_CHECK(true, {
        ASSERT(tileHout > 0 && tileHout <= Ho)
            << "Invalid tileHout value:: " << tileHout 
            << ",, expected range [1, " << Ho
            << "]." << std::endl;
    });

    OP_CHECK(true, {
        ASSERT(tileWout > 0 && tileWout <= Wo)
            << "Invalid tileWout value:: " << tileWout 
            << ",, expected range [1, " << Wo
            << "]." << std::endl;
    });

}

void CheckTileTiling(const Tensor &operand2, const MatmulAttrParam &attrParam) {
    int64_t tileM = convTile.tileL0Info.tileM;
    int64_t tileN = convTile.tileL0Info.tileN;
    int64_t tileK = convTile.tileL0Info.tileK;

    int64_t tileHout   = convTile.tileL1Info.tileHout;
    int64_t tileWout   = convTile.tileL1Info.tileWout;
    OP_CHECK(true, {
        ASSERT(tileM > 0 && tileM <= tileHout * tileWout)
            << "Invalid tileHin value:: " << tileM 
            << ",, expected range [1, " << tileHout * tileWout
            << ".Current tileHout= " << tileHout
            << ", tileWout=" << tileWout
            << " → maximum allowed = tileHout * tileWout" << std::endl;
    });

    int64_t tileCinWeight = convTile.tileL1Info.tileCinWeight;
    int64_t kH =  operand2.GetShape()[2];
    int64_t kW =  operand2.GetShape()[3];   
    int64_t maxK = kH * kH * tileCinWeight;
    OP_CHECK(true, {
        ASSERT(tileK > 0 && tileK <= maxK)
            << "Invalid tileHin value:: " << tileK 
            << ",, expected range [1, " << maxK
            << ".Current tileCinWeight= " << tileCinWeight
            << ", kH=" << kH
            << ", kW=" << kW
            << " → maximum allowed = kH * kH * tileCinWeight" << std::endl;
    });

    int tileCout   = convTile.tileL1Info.tileCout;
    OP_CHECK(true, {
        ASSERT(tileN > 0 && tileN <= tileCout)
            << "Invalid tileHin value:: " << tileN 
            << ",, expected range [1, " << tileCout
            << "]." << std::endl;
    });

}

void CheckPadShape(const std::vector<int64_t> &paddings){
    OP_CHECK(true, {
            ASSERT(paddings.size() <= NUM4)
        << "Input attr stride dim: " << paddings.size()
        << "!=" << NUM4
        << "]." << std::endl
    });

    for (size_t i = 0; i < paddings.size(); ++i) {
        OP_CHECK(true, {
            ASSERT(paddings[i] <= MAX_PAD)
            << "The value of the " << i 
            << "-th dimension of stride must be in the range [1, " << MAX_PAD
            << "]." << std::endl
        });
    }

}

void CheckDilationShape(const std::vector<int64_t> &dilations){
    OP_CHECK(true, {
        ASSERT(dilations.size() <= NUM4)
        << "Input attr dilations dim: " << dilations.size()
        << "!=" << NUM4
        << "]." << std::endl
    });

    for (size_t i = 0; i < dilations.size(); ++i) {
        if(i > 1) {
            OP_CHECK(true, {
                ASSERT(dilations[i] <= MAX_DILATION_STRIDE)
                << "The value of the " << i 
                << "-th dimension of stride must be in the range [1, " << MAX_DILATION_STRIDE
                << "]." << std::endl
            });
        }
        else {
            OP_CHECK(true, {
                ASSERT(dilations[i] != NUM1)
                << "The value of the " << i 
                << "-th dimension of stride must be " << NUM1
                << "]." << std::endl
            });
        }
        
    }
}

void heckStrideShape(const std::vector<int64_t> &strides){
    OP_CHECK(true, {
        ASSERT(strides.size() <= NUM4)
        << "Input attr strides dim: " << strides.size()
        << "!=" << NUM4
        << "]." << std::endl
    });

    for (size_t i = 0; i < strides.size(); ++i) {
        if(i > 1) {
            OP_CHECK(true, {
                ASSERT(strides[i] <= MAX_DILATION_STRIDE)
                << "The value of the " << i 
                << "-th dimension of stride must be in the range [1, " << MAX_DILATION_STRIDE
                << "]." << std::endl
            });
        }
        else {
            OP_CHECK(true, {
                ASSERT(strides[i] != NUM1)
                << "The value of the " << i 
                << "-th dimension of stride must be " << NUM1
                << "]." << std::endl
            });
        }
    }
}

void CheckGroupsShape(const int64_t cinFmap, const int64_t cinWeight,const int64_t Cout, const int64_t groups){

    OP_CHECK(true, {
            ASSERT(groups <= SHAPE_INNER_AXIS_MAX_SIZE)
            << "Invalid groups value: groups =" << i 
            << "expected range [1, " << SHAPE_INNER_AXIS_MAX_SIZE
            << "]." << std::endl
    });


    OP_CHECK(true, {
            ASSERT(cinFmap % groups == 0)
            << "Cin = " << cinFmap
            << "is not divisible by groups: " << groups
            << "adjusting Cin to the nearest value such that Cin % groups == 0." << std::endl
    });

    OP_CHECK(true, {
            ASSERT(Cout % groups == 0)
            << "Cout = " << Cout
            << "is not divisible by groups: " << groups
            << "adjusting Cout to the nearest value such that Cout % groups == 0." << std::endl
    });

    OP_CHECK(true, {
            ASSERT(cinFmap != cinWeight * groups)
            << "Fmap Cin : " << cinFmap
            << "!= weight Cin: " << cinWeight
            << "groups: " << groups
            << "." << std::endl
    });
}

void CheckAttrShape(const Tensor &operand1, const Tensor &operand2, const convAttrParam &attrParam){
    std::vector<int64_t> paddings = attrParam.paddings;
    CheckPadShape(paddings);

    std::vector<int64_t> dilations = attrParam.dilations;

    CheckDilationShape(dilations);

    std::vector<int64_t> strides = attrParam.strides;

    heckStrideShape(strides);
    
    int64_t groups = attrParam.groups;

    int64_t cinFmap =  operand1.GetShape()[2];
    int64_t cinWeight =  operand2.GetShape()[2];
    int64_t cout =  operand2.GetShape()[0];
    CheckGroupsShape(cinFmap, cinWeight, cout, groups)


    int64_t hin =  operand1.GetShape()[2];
    int64_t win =  operand1.GetShape()[3];
    int64_t kH =  operand2.GetShape()[2];
    int64_t kW =  operand2.GetShape()[3];
    int64_t padTop =  paddings[0];
    int64_t padBottom =  paddings[1];
    int64_t padLeft =  paddings[2];
    int64_t padRight =  paddings[3];
    int64_t dilationH =  dilations[2];
    int64_t dilationW =  dilations[3];
    int64_t strideH =  strides[2];
    int64_t strideH =  strides[3];
    

    int64_t Ho = ConvComputeHo(hin, kH, padTop, padBottom, dilationH, strideH);
    OP_CHECK(true, {
            ASSERT(Ho <= MAX_SIZE)
        << "Invalid hout value: " << Hout
        << ", expected range[1, " << MAX_SIZE
        << "]." << std::endl
    });

    int64_t Wo = ConvComputeWo(win, kW, padLeft, padRight, dilationW, strideW);
    OP_CHECK(true, {
            ASSERT(Wo <= MAX_SIZE)
        << "Invalid Wout value: " << Wout
        << ", expected range[1,  " << MAX_SIZE
        << "]." << std::endl
    });
    
}

void CheckOperandShape(const Tensor &operand1, const Tensor &operand2)
{
    for (size_t i = 0; i < operand1.size(); ++i) {
        OP_CHECK(true, {
            ASSERT(operand1.GetShape()[i] <= MAX_SIZE)
            << "The value of the " << i 
            << "-th dimension of fmap must be in the range [1, " << MAX_SIZE
            << "]." << std::endl
        });
    }

    for (size_t i = 0; i < operand2.size(); ++i) {
        OP_CHECK(true, {
            ASSERT(operand2.GetShape()[i] <= MAX_SIZE)
            << "The value of the " << i 
            << "-th dimension of weight must be in the range [1, " << MAX_SIZE
            << "]." << std::endl
        });
    }

    int64_t Cout = operand2.GetShape()[0];
    OP_CHECK(true, {
        ASSERT(operand3.GetShape()[i] == Cout)
        << "Input illegal bias shape:" << operand3.GetShape()[i] 
        << ", which must euqal to Cout:" << Cout
        << "." << std::endl
    });

    int64_t kH = operand2.GetShape()[2];
    int64_t kw = operand2.GetShape()[3];
    OP_CHECK(true, {
        ASSERT(kH * kW <= MAX_PAD_KERNEL)
        << "Weight shape not satisfy Load3D's limits: kh=" << kH
        << ", kw=" << kW
        << ", which must <=" << MAX_PAD_KERNEL
        << "." << std::endl
    });
    
}


int64_t ConvComputeHo(int64_t hin, int64_t kH, int64_t padTop, int64_t padBottom, int64_t dilationH, int64_t strideH)
{
    if (strideH == 0) {
        return 1;
    }
    int64_t cmpHo = (hin + padTop + padBottom - dilationH * (kH - 1) - 1) / strideH + 1;
    return cmpHo;
}

int64_t ConvComputeWo(int64_t win, int64_t kW, int64_t padLeft, int64_t padRight, int64_t dilationW, int64_t strideW)
{
    if (strideW == 0) {
        return 1;
    }
    int64_t cmpWo = (win + padLeft + padRight - dilationH * (hk - 1) - 1) / strideW + 1;
    return cmpWo;
}


void SetTensorOpAttr(Operation &op, const ConvAttrParam &convAttrParam)
{
    op.SetAttribute(CONV_BIAS_ATTR, convAttrParam.hasBias);
    op.SetAttribute(CONV_GROUPS_ATTR, convAttrParam.groups);
    op.SetAttribute(CONV_PADDINGS_ATTR, convAttrParam.paddings);
    op.SetAttribute(CONV_STRIDES_ATTR, convAttrParam.strides);
    op.SetAttribute(CONV_DILATIONS_ATTR, convAttrParam.dilations);
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
    // convAttrParam.paddings = (op.HasAttr(CONV_PADDINGS_ATTR)) ? op.GetVectorElementAttribute(CONV_PADDINGS_ATTR) :
    //     convAttrParam.isConv3D ? CONV2D_ATTR_DEFAULT_LIST : CONV2D_ATTR_DEFAULT_LIST;
    // convAttrParam.strides = (op.HasAttr(CONV_STRIDES_ATTR)) ? op.GetVectorElementAttribute(CONV_STRIDES_ATTR) :
    //     convAttrParam.isConv3D ? CONV2D_ATTR_DEFAULT_LIST : CONV2D_ATTR_DEFAULT_LIST;
    // convAttrParam.dilations = (op.HasAttr(CONV_DILATIONS_ATTR)) ? op.GetVectorElementAttribute(CONV_DILATIONS_ATTR) :
    //     convAttrParam.isConv3D ? CONV2D_ATTR_DEFAULT_LIST : CONV2D_ATTR_DEFAULT_LIST;
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
        // convAttrParam.oriFmapShape = op.GetVectorElementAttribute(CONV_ORI_FMAP_SHAPE_ATTR);
        // convAttrParam.oriweightShape = op.GetVectorElementAttribute(CONV_ORI_WEIGHT_SHAPE_ATTR);
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

    OP_CHECK(true, { ASSERT(cTensorPtr != nullptr) << "cTensorPtr is nullptr." << std::endl;});
    tensorGraphNodes.resTensorPtr = cTensorPtr;
}

void SetConvShapeInfo(const TileShape &tileShape, const ConvGraphNodes &tensorGraphNodes,
                      const ConvAttrParam &convAttrParam, ConvTileInfo &convTileInfo)
{
    // 设计线讨论如何获取私有格式tensor传入的ori_shape
    // set org shape
    convTileInfo.orgBatch = tensorGraphNodes.fmapTensorPtr->shape[0];
    convTileInfo.orgHout = tensorGraphNodes.resTensorPtr->shape[2];
    convTileInfo.orgWout = tensorGraphNodes.resTensorPtr->shape[3];
    if (convAttrParam.isInOutTensorNZ) {
        convTileInfo.orgCout = convAttrParam.oriweightShape[0];
        convTileInfo.orgKh = convAttrParam.oriweightShape[2];
        convTileInfo.orgKw = convAttrParam.oriweightShape[3];
        convTileInfo.orgCin = tensorGraphNodes.fmapTensorPtr->shape[1] * tensorGraphNodes.fmapTensorPtr->shape[4];
    } else {
        convTileInfo.orgCout = tensorGraphNodes.weightTensorPtr->shape[0];
        convTileInfo.orgKh = tensorGraphNodes.weightTensorPtr->shape[2];
        convTileInfo.orgKw = tensorGraphNodes.weightTensorPtr->shape[3];
        convTileInfo.orgCin = tensorGraphNodes.fmapTensorPtr->shape[1];
    }
    convTileInfo.orgK = convTileInfo.orgCin * convTileInfo.orgKh * convTileInfo.orgKw;
    convTileInfo.orgHoutWout = convTileInfo.orgHout * convTileInfo.orgWout;
    // set tileshape info
    // auto &convTile = tileShape.GetConvTile();
    convTileInfo.kAL1 = 16;
    convTileInfo.kBL1 = 32;
    convTileInfo.nBL1 = 16;
    convTileInfo.hAL1In = 8;
    convTileInfo.wAL1In = 8;
    convTileInfo.hAL1In = 8;
    convTileInfo.wAL1In = 8;
    convTileInfo.kL0 = 16;
    convTileInfo.mL0 = 64;
    convTileInfo.nL0 = 16;
}

LogicalTensorPtr ConstructBiasTile(Function &function, const ConvGraphNodes &tensorGraphNodes,
                                   const ConvIterInfo &iterInfo)
{
    std::vector<int64_t> dstBiasL1Shape = std::vector<int64_t>{iterInfo.nL0Size};
    LogicalTensorPtr dstBiasl1TensorPtr =
        std::make_shared<LogicalTensor>(function, tensorGraphNodes.biasTensorPtr->Datatype(),
                                        dstBiasL1Shape, SymbolicScalar::FromConcrete(dstBiasL1Shape),
                                        tensorGraphNodes.biasTensorPtr->Format(), "biasL1Tensor", NodeType::LOCAL);
    // dstBiasl1TensorPtr->UpdateDynValidShape(
    //     GetViewValidShape(operandVec[1]->GetDynValidShape(), dstTensorInfo.offset, {}, dstTensorInfo.shape));
    auto &viewOpBiasL1 = function.AddOperation(Opcode::OP_VIEW, {tensorGraphNodes.biasTensorPtr}, {dstBiasl1TensorPtr});

    std::vector<int64_t> dstBiasBtShape = std::vector<int64_t>{iterInfo.nL0Size};
    LogicalTensorPtr dstBiasBtTensorPtr =
        std::make_shared<LogicalTensor>(function, tensorGraphNodes.biasTensorPtr->Datatype(), dstBiasBtShape,
                                        SymbolicScalar::FromConcrete(dstBiasBtShape),
                                        tensorGraphNodes.biasTensorPtr->Format(), "biasBtTensor", NodeType::LOCAL);
    // dstBl1TensorPtr->UpdateDynValidShape(
    //     GetViewValidShape(operandVec[1]->GetDynValidShape(), dstTensorInfo.offset, {}, dstTensorInfo.shape));
    auto &viewOpBiasBt = function.AddOperation(Opcode::OP_VIEW, {dstBiasl1TensorPtr}, {dstBiasBtTensorPtr});

    return dstBiasBtTensorPtr;
}

LogicalTensorPtr ConstructFmapTile(Function &function, const ConvGraphNodes &tensorGraphNodes,
                                   const ConvTileInfo &convTileInfo, const ConvIterInfo &iterInfo,
                                   LogicalTensorPtr &dstAL1TensorPtr)
{
    bool aL1UpadateFlag = false;
    if (iterInfo.kOffset % convTileInfo.kAL1 == 0) {
        aL1UpadateFlag = true;
    }
    // L1层级 Fmap 展开
    if (aL1UpadateFlag) {
        std::vector<int64_t> dstAL1Shape = std::vector<int64_t>{1, 1, 8, 8, 16};
        dstAL1TensorPtr =
            std::make_shared<LogicalTensor>(function, tensorGraphNodes.fmapTensorPtr->Datatype(), dstAL1Shape,
                                            SymbolicScalar::FromConcrete(dstAL1Shape),
                                            tensorGraphNodes.fmapTensorPtr->Format(), "aL1Tensor", NodeType::LOCAL);
        // dstAL1TensorPtr->UpdateDynValidShape(
        //     GetViewValidShape(operandVec[0]->GetDynValidShape(), dstAL1Offset, {}, dstAL1Shape));
        auto &viewOpAl1 = function.AddOperation(Opcode::OP_VIEW, {tensorGraphNodes.fmapTensorPtr}, {dstAL1TensorPtr});
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
    auto &viewOpAl0 = function.AddOperation(Opcode::OP_VIEW, {dstAL1TensorPtr}, {dstAL0TensorPtr});
    return dstAL0TensorPtr;
}

LogicalTensorPtr ConstructWeightTile(Function &function, const ConvGraphNodes &tensorGraphNodes,
                                     const ConvTileInfo &convTileInfo, const ConvIterInfo &iterInfo,
                                     LogicalTensorPtr &dstBL1TensorPtr)
{
    bool bL1UpadateFlag = false;
    if (iterInfo.kOffset % convTileInfo.kBL1 == 0 || iterInfo.nOffset % convTileInfo.nBL1) {
        bL1UpadateFlag = true;
    }
    // L1层级 Weight 展开
    if (bL1UpadateFlag) {
        std::vector<int64_t> dstBL1Shape = std::vector<int64_t>{2, 1, 16, 16};
        dstBL1TensorPtr =
            std::make_shared<LogicalTensor>(function, tensorGraphNodes.weightTensorPtr->Datatype(), dstBL1Shape,
                                            SymbolicScalar::FromConcrete(dstBL1Shape),
                                            tensorGraphNodes.weightTensorPtr->Format(), "bL1Tensor", NodeType::LOCAL);
        // dstBL1TensorPtr->UpdateDynValidShape(
        //     GetViewValidShape(operandVec[1]->GetDynValidShape(), dstTensorInfo.offset, {}, dstTensorInfo.shape));
        auto &viewOpBl1 = function.AddOperation(Opcode::OP_VIEW, {tensorGraphNodes.weightTensorPtr}, {dstBL1TensorPtr});
    }
    // load2d()
    std::vector<int64_t> dstBL0Shape = std::vector<int64_t>{16, 16};
    LogicalTensorPtr dstBL0TensorPtr =
        std::make_shared<LogicalTensor>(function, tensorGraphNodes.weightTensorPtr->Datatype(), dstBL0Shape,
                                        SymbolicScalar::FromConcrete(dstBL0Shape),
                                        tensorGraphNodes.weightTensorPtr->Format(), "bL0Tensor", NodeType::LOCAL);
    // dstAL1TensorPtr->UpdateDynValidShape(
    //     GetViewValidShape(operandVec[0]->GetDynValidShape(), dstAL1Offset, {}, dstAL1Shape));
    auto &viewOpBl0 = function.AddOperation(Opcode::OP_VIEW, {dstBL1TensorPtr}, {dstBL0TensorPtr});

    return dstBL0TensorPtr;
}

void DoMmad(Function &function, const ConvAttrParam &convAttrParam, const ConvGraphNodes &tensorGraphNodes,
            ConvGraphNodes &tileGraphNodes, const ConvIterInfo &iterInfo)
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
            OP_CHECK(true, {ASSERT(tileGraphNodes.biasTensorPtr != nullptr)
                << "bias must be non-nullptr when hasBias Flag." << std::endl;});
            mmadInputs.push_back(tileGraphNodes.biasTensorPtr);
        }
    } else {
        mmadInputs = {tileGraphNodes.fmapTensorPtr, tileGraphNodes.weightTensorPtr, tileGraphNodes.cL0PartialSumPtr};
    }

    if (iterInfo.isLastK) {
        mmadOutputs = {tileGraphNodes.resTensorPtr};
    } else {
        tileGraphNodes.cL0PartialSumPtr = std::make_shared<LogicalTensor>(
            function, tileGraphNodes.resTensorPtr->Datatype(), tileGraphNodes.resTensorPtr->GetShape());
        // if (CheckValidShape(tileGraphNodes.aTensorPtr) && CheckValidShape(tileGraphNodes.bTensorPtr)) {
        //     tileGraphNodes.cL0PartialSumPtr->UpdateDynValidShape(
        //         {tileGraphNodes.aTensorPtr->GetDynValidShape()[0], tileGraphNodes.bTensorPtr->GetDynValidShape()[1]});
        // }
        tileGraphNodes.cL0PartialSumPtr->UpdateDynValidShape({iterInfo.mL0Size, iterInfo.nL0Size});
        mmadOutputs = {tileGraphNodes.cL0PartialSumPtr};
    }
    auto &aMulBOp = function.AddOperation(MmadOpStr, mmadInputs, mmadOutputs);
}

void UpdateIterInfo(const ConvTileInfo &convTileInfo, ConvIterInfo &iterInfo)
{
    // todo update iterInfo
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
            for (iterInfo.nOffset = 0; iterInfo.nOffset < (convTileInfo.orgCout / convAttrParam.groups);
                 iterInfo.nOffset += convTileInfo.nL0) {
                // bias 载入
                if (convAttrParam.hasBias) {
                    // get bias in bt tile for mmad
                    tileGraphNodes.biasTensorPtr = ConstructBiasTile(function, tensorGraphNodes, iterInfo);
                }
                for (iterInfo.mOffset = 0; iterInfo.mOffset < convTileInfo.orgHoutWout; iterInfo.mOffset += convTileInfo.mL0) {
                    // set res tile
                    tileGraphNodes.resTensorPtr =
                        cTensorPtr->View(function, {iterInfo.mL0Size, iterInfo.nL0Size}, {iterInfo.mOffset, iterInfo.nOffset});
                    LogicalTensorPtr fmapL1TensorPtr = nullptr;
                    LogicalTensorPtr weightL1TensorPtr = nullptr;
                    for (iterInfo.kOffset = 0; iterInfo.kOffset < (convTileInfo.orgCin / convAttrParam.groups);
                         iterInfo.kOffset += convTileInfo.kL0) {
                        UpdateIterInfo(convTileInfo, iterInfo);
                        // fmap and weight link
                        tileGraphNodes.fmapTensorPtr =
                            ConstructFmapTile(function, tensorGraphNodes, convTileInfo, iterInfo, fmapL1TensorPtr);
                        tileGraphNodes.weightTensorPtr =
                            ConstructWeightTile(function, tensorGraphNodes, convTileInfo, iterInfo, weightL1TensorPtr);
                        // add mmad node
                        DoMmad(function, convAttrParam, tensorGraphNodes, tileGraphNodes, iterInfo);
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
    // auto &conveTile = TileShape::Current().GetConvTile();
    // Check ConvTile Valid
    // infer hout, wout
    int64_t batchOut = inputTensor.GetShape()[0];
    int64_t C0 = ALIGN_SIZE_32 / BytesOf(outType);
    int64_t co1 = (weightTensor.GetShape()[1] * weightTensor.GetShape()[2]) / C0;
    int64_t hOut = 8;
    int64_t wOut = 8;
    std::vector<int64_t> resTensorShape{batchOut, co1, hOut, wOut, C0};
    Tensor resTensor(outType, resTensorShape, "TensorC");
    resTensor.GetStorage()->UpdateDynValidShape({{batchOut, co1, hOut, wOut, C0}});
    return ConstructTensorGraph(outType, inputTensor, weightTensor, biasTensor, resTensor, convAttrParam);
}

} //namespace Conv
}
}