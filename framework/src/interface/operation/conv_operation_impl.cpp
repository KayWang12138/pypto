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

void CheckConvOperands(DataType outType, const Tensor &operand1, const Tensor &operand2, const Tensor &operand3) {
    // todo
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
        // convAttrParam.oriFmapShape = {1, 32, 8, 8};
        // convAttrParam.oriweightShape = {32, 32, 1, 1};
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
    // convTileInfo.orgHout = tensorGraphNodes.resTensorPtr->shape[2];
    // convTileInfo.orgWout = tensorGraphNodes.resTensorPtr->shape[3];
    convTileInfo.orgHout = 8;
    convTileInfo.orgWout = 8;
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
                                   const ConvTileInfo &convTileInfo, ConvIterInfo &iterInfo)
{
    std::vector<int64_t> dstBiasL1Shape = std::vector<int64_t>{iterInfo.nL0Size};
    std::vector<int64_t> dstBiasL1Offset = std::vector<int64_t>{iterInfo.nOffset};
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
    std::vector<int64_t> dstBiasBtOffset = std::vector<int64_t>{iterInfo.nOffset};
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
    bool aL1UpadateFlag = false;
    if (iterInfo.kOffset % convTileInfo.kAL1 == 0) {
        aL1UpadateFlag = true;
    }
    // L1层级 Fmap 展开
    if (aL1UpadateFlag) {
        iterInfo.kAL1Size = std::min(convTileInfo.orgK - iterInfo.kOffset, convTileInfo.kAL1);
        std::vector<int64_t> dstAL1Shape = std::vector<int64_t>{1, iterInfo.kAL1Size / 16, 8, 8, 16};
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
    bool bL1UpadateFlag = false;
    if (iterInfo.kOffset % convTileInfo.kBL1 == 0 || iterInfo.nOffset % convTileInfo.nBL1) {
        bL1UpadateFlag = true;
    }
    // L1层级 Weight 展开
    if (bL1UpadateFlag) {
        iterInfo.nL1Size = std::min(convTileInfo.orgCout - iterInfo.nOffset, convTileInfo.nBL1);
        iterInfo.kBL1Size = std::min(convTileInfo.orgK - iterInfo.kOffset, convTileInfo.kBL1);
        std::vector<int64_t> dstBL1Shape = std::vector<int64_t>{iterInfo.kBL1Size / 16, iterInfo.nL1Size / 16, 16, 16};
        std::vector<int64_t> dstBL1Offset = std::vector<int64_t>{iterInfo.kOffset / 16, iterInfo.nOffset / 16, 0, 0};
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
    op.SetAttribute(A_MUL_B_ACT_M, convTileInfo.mL0);
    op.SetAttribute(A_MUL_B_ACT_K, convTileInfo.kL0);
    op.SetAttribute(A_MUL_B_ACT_N, convTileInfo.nL0);

    if (op.GetOpcode() == Opcode::OP_A_MUL_B) {
        op.SetAttribute(A_MUL_B_BIAS_ATTR, tensorGraphNodes.biasTensorPtr != nullptr);
        // op.SetAttribute(A_MUL_B_RELU_ATTR, static_cast<int64_t>(attrParam.reluType));
        // op.SetAttribute(A_MUL_B_SCALE_ATTR, Element(DataType::DT_UINT64, attrParam.scaleValue));
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

void UpdateIterInfo(const ConvTileInfo &convTileInfo, ConvIterInfo &iterInfo)
{
    // todo update iterInfo
    iterInfo.isFirstK = iterInfo.kOffset == 0 ? true : false;
    iterInfo.isLastK = iterInfo.kOffset + convTileInfo.kL0 >= convTileInfo.orgK ? true : false;
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
                iterInfo.nL0Size = std::min((convTileInfo.orgCout / convAttrParam.groups) - iterInfo.nOffset, convTileInfo.nL0);
                // bias 载入
                if (convAttrParam.hasBias) {
                    // get bias in bt tile for mmad
                    tileGraphNodes.biasTensorPtr = ConstructBiasTile(function, tensorGraphNodes, convTileInfo, iterInfo);
                }
                for (iterInfo.mOffset = 0; iterInfo.mOffset < convTileInfo.orgHoutWout; iterInfo.mOffset += convTileInfo.mL0) {
                    iterInfo.mL0Size = std::min(convTileInfo.orgHoutWout - iterInfo.mOffset, convTileInfo.mL0);
                    // set res tile
                    // tileGraphNodes.resTensorPtr =
                    //     cTensorPtr->View(function, {1, iterInfo.nL0Size / 16, 8, 8, 16}, {iterInfo.batchOffset, iterInfo.nOffset / 16, 0, 0, 0});
                    // tileGraphNodes.resTensorPtr =
                    //     cTensorPtr->View(function, {iterInfo.mL0Size, iterInfo.nL0Size}, {iterInfo.mOffset, iterInfo.nOffset});
                    std::vector<int64_t> dstCL0Shape = std::vector<int64_t>{iterInfo.mL0Size, iterInfo.nL0Size};
                    tileGraphNodes.resTensorPtr =
                        std::make_shared<LogicalTensor>(function, tensorGraphNodes.fmapTensorPtr->Datatype(), dstCL0Shape,
                                                        SymbolicScalar::FromConcrete(dstCL0Shape),
                                                        tensorGraphNodes.fmapTensorPtr->Format(), "cL0Tensor", NodeType::LOCAL);
                    LogicalTensorPtr fmapL1TensorPtr = nullptr;
                    LogicalTensorPtr weightL1TensorPtr = nullptr;
                    LogicalTensorPtr resCl0TensorPtr = nullptr;
                    for (iterInfo.kOffset = 0; iterInfo.kOffset < (convTileInfo.orgK / convAttrParam.groups);
                         iterInfo.kOffset += convTileInfo.kL0) {
                        UpdateIterInfo(convTileInfo, iterInfo);
                        // fmap and weight link
                        tileGraphNodes.fmapTensorPtr =
                            ConstructFmapTile(function, tensorGraphNodes, convTileInfo, iterInfo, fmapL1TensorPtr);
                        tileGraphNodes.weightTensorPtr =
                            ConstructWeightTile(function, tensorGraphNodes, convTileInfo, iterInfo, weightL1TensorPtr);
                        // add mmad node
                        resCl0TensorPtr = DoMmad(function, convAttrParam, tensorGraphNodes, tileGraphNodes, iterInfo);
                    }
                    auto &viewOpRes = function.AddOperation(Opcode::OP_L0C_COPY_OUT_CONV, {resCl0TensorPtr}, {cTensorPtr});
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
    CheckConvOperands(outType, inputTensor, weightTensor, biasTensor);
    // auto &conveTile = TileShape::Current().GetConvTile();
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