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
    param.isConv3D = (op.HasAttr(CONV_3D_FLAG)) ? op.GetBoolAttribute(CONV_3D_FLAG) : false;
    param.paddings = (op.HasAttr(CONV_PADDINGS_ATTR)) ? op.GetVectorElementAttribute(CONV_PADDINGS_ATTR) :
                                                        isConv3D ? {0, 0, 0, 0} : {0, 0, 0, 0, 0, 0};
    param.strides = (op.HasAttr(CONV_STRIDES_ATTR)) ? op.GetVectorElementAttribute(CONV_STRIDES_ATTR) :
                                                      isConv3D ? {0, 0, 0, 0} : {0, 0, 0, 0, 0, 0};;
    param.dilations = (op.HasAttr(CONV_DILATIONS_ATTR)) ? op.GetVectorElementAttribute(CONV_DILATIONS_ATTR) :
                                                          isConv3D ? {0, 0, 0, 0} : {0, 0, 0, 0, 0, 0};;
    param.groups = (op.HasAttr(CONV_GROUPS_ATTR)) ? op.GetIntAttribute(CONV_GROUPS_ATTR) : 1;
    param.hasBias = (op.HasAttr(CONV_BIAS_ATTR)) ? op.GetBoolAttribute(CONV_BIAS_ATTR) : false;
}

void ConstructTileGraph(Function &function, const TileShape &tileShape, const std::vector<LogicalTensorPtr> &operandVec,
                        const LogicalTensorPtr &cTensorPtr, const Operation &op)
{
    // op attr set
    ConvAttrParam attrParam;
    SetConvAttrParam(op, attrParam);
    // tensor graph中的数据节点
    ConvGraphNodes tensorGraphNodes;
    SetTensorGraphNodes(operandVec, cTensorPtr, attrParam, tensorGraphNodes);
    // Tile 信息存储
    ConvTileInfo tileInfo;
    SetConvTileInfo(tileShape, attrParam, tensorGraphNodes, tileInfo);
    // iter信息存储
    ConvIterInfo iterInfo;
    // tile graph中的数据节点
    ConvGraphNodes tileGraphNodes;

    int64_t groups = 1;
    int64_t orgBatch = 1;
    int64_t orgCout = 32;
    int64_t orgHWout = 16;
    int64_t orgK = 32 * 1 * 1;
    int64_t kAL1 = 16;
    int64_t kBL1 = 32;
    int64_t hAL1In = 8;
    int64_t wAL1In = 8;
    int64_t hAL1Out = 8;
    int64_t wAL1Out = 8;
    int64_t nBL1 = 16;

    for (int64_t groupIdx = 0; groupIdx < groups; groupIdx += 1) {
        for (int64_t batchIdx = 0; batchIdx < orgBatch; batchIdx += 1) {
            for (int64_t nl0Idx = 0; nl0Idx < orgCout; nl0Idx += 16) {
                // bias 载入
                if (operandVec.size() > 2) {
                    std::vector<int64_t> dstBiasL1Shape = std::vector<int64_t>{16};
                    dstBiasl1TensorPtr =
                        std::make_shared<LogicalTensor>(function, operandVec[2]->Datatype(), dstBiasL1Shape, SymbolicScalar::FromConcrete(dstBiasL1Shape),
                                                        operandVec[2]->Format(), "biasL1Tensor", NodeType::LOCAL);
                    // dstBl1TensorPtr->UpdateDynValidShape(
                    //     GetViewValidShape(operandVec[1]->GetDynValidShape(), dstTensorInfo.offset, {}, dstTensorInfo.shape));
                    auto &viewOpBiasL1 = function.AddOperation(Opcode::OP_VIEW, {operandVec[2]}, {dstBiasl1TensorPtr});

                    std::vector<int64_t> dstBiasBtShape = std::vector<int64_t>{16};
                    dstBiasBtTensorPtr =
                        std::make_shared<LogicalTensor>(function, operandVec[2]->Datatype(), dstBiasBtShape, SymbolicScalar::FromConcrete(dstBiasBtShape),
                                                        operandVec[2]->Format(), "biasBtTensor", NodeType::LOCAL);
                    // dstBl1TensorPtr->UpdateDynValidShape(
                    //     GetViewValidShape(operandVec[1]->GetDynValidShape(), dstTensorInfo.offset, {}, dstTensorInfo.shape));
                    auto &viewOpBiasBt = function.AddOperation(Opcode::OP_VIEW, {dstBiasl1TensorPtr}, {dstBiasBtTensorPtr});
                }

                for (int64_t ml0Idx = 0; ml0Idx < orgHWout; ml0Idx += 64) {
                    for (int64_t kl0Idx = 0; kl0Idx < orgK; kl0Idx += 16) {
                        bool aL1UpadateFlag = false;
                        bool bL1UpadateFlag = false;
                        // updateFlag
                        if (kl0Idx % kAL1 == 0) {
                            aL1UpadateFlag = true;
                        }
                        if (kl0Idx % kBL1 == 0 || nl0Idx % nBL1) {
                            bL1UpadateFlag = true;
                        }
                        // L1层级 Fmap 展开
                        if (aL1UpadateFlag) {
                            std::vector<int64_t> dstAL1Shape = std::vector<int64_t>{1, 1, 8, 8, 16};
                            dstAL1TensorPtr =
                                std::make_shared<LogicalTensor>(function, operandVec[0]->Datatype(), dstAL1Shape, SymbolicScalar::FromConcrete(dstAL1Shape),
                                                                operandVec[0]->Format(), "aL1Tensor", NodeType::LOCAL);
                            // dstAL1TensorPtr->UpdateDynValidShape(
                            //     GetViewValidShape(operandVec[0]->GetDynValidShape(), dstAL1Offset, {}, dstAL1Shape));
                            auto &viewOpAl1 = function.AddOperation(Opcode::OP_VIEW, {operandVec[0]}, {dstAL1TensorPtr});
                        }
                        // L1层级 Weight 展开
                        if (bL1UpadateFlag) {
                            std::vector<int64_t> dstBL1Shape = std::vector<int64_t>{2, 1, 16, 16};
                            dstBL1TensorPtr =
                                std::make_shared<LogicalTensor>(function, operandVec[1]->Datatype(), dstBL1Shape, SymbolicScalar::FromConcrete(dstBL1Shape),
                                                                operandVec[1]->Format(), "bL1Tensor", NodeType::LOCAL);
                            // dstBL1TensorPtr->UpdateDynValidShape(
                            //     GetViewValidShape(operandVec[1]->GetDynValidShape(), dstTensorInfo.offset, {}, dstTensorInfo.shape));
                            auto &viewOpBl1 = function.AddOperation(Opcode::OP_VIEW, {operandVec[1]}, {dstBL1TensorPtr});
                        }
                        // 二层展开
                        // load3dv2()
                        std::vector<int64_t> dstAL0Shape = std::vector<int64_t>{64, 16};
                        dstAL0TensorPtr =
                            std::make_shared<LogicalTensor>(function, operandVec[0]->Datatype(), dstAL0Shape, SymbolicScalar::FromConcrete(dstAL0Shape),
                                                            operandVec[0]->Format(), "aL0Tensor", NodeType::LOCAL);
                        // dstAL1TensorPtr->UpdateDynValidShape(
                        //     GetViewValidShape(operandVec[0]->GetDynValidShape(), dstAL1Offset, {}, dstAL1Shape));
                        auto &viewOpAl0 = function.AddOperation(Opcode::OP_VIEW, {dstAL1TensorPtr}, {dstAL0TensorPtr});
                        // load2d()
                        std::vector<int64_t> dstBL0Shape = std::vector<int64_t>{16, 16};
                        dstBL0TensorPtr =
                            std::make_shared<LogicalTensor>(function, operandVec[1]->Datatype(), dstBL0Shape, SymbolicScalar::FromConcrete(dstBL0Shape),
                                                            operandVec[1]->Format(), "bL0Tensor", NodeType::LOCAL);
                        // dstAL1TensorPtr->UpdateDynValidShape(
                        //     GetViewValidShape(operandVec[0]->GetDynValidShape(), dstAL1Offset, {}, dstAL1Shape));
                        auto &viewOpBl0 = function.AddOperation(Opcode::OP_VIEW, {dstBL1TensorPtr}, {dstBL0TensorPtr});
                        // MMAD node add
                        std::vector<LogicalTensorPtr> mmadInputs;
                        std::vector<LogicalTensorPtr> mmadOutputs;
                        const std::string MmadOpStr = kl0Idx == 0 ? "TILE_A_MUL_B" : "TILE_A_MULACC_B";
                        if (kl0Idx == 0) {
                            mmadInputs = {dstAL0TensorPtr, dstBL0TensorPtr};
                            if (operandVec.size() > 2) {
                                mmadInputs.push_back(dstBiasBtTensorPtr);
                            }
                        } else {
                            mmadInputs = {dstAL0TensorPtr, dstBL0TensorPtr, cL0PartialSumPtr};
                        }

                        if (kl0Idx + 16 >= orgK) {
                            mmadOutputs = {cTensorPtr};
                        } else {
                            cL0PartialSumPtr = std::make_shared<LogicalTensor>(
                                function, cTensorPtr->Datatype(), cTensorPtr->GetShape());
                            // if (CheckValidShape(tileGraphNodes.aTensorPtr) && CheckValidShape(tileGraphNodes.bTensorPtr)) {
                            //     tileGraphNodes.cL0PartialSumPtr->UpdateDynValidShape(
                            //         {tileGraphNodes.aTensorPtr->GetDynValidShape()[0], tileGraphNodes.bTensorPtr->GetDynValidShape()[1]});
                            // }
                            cL0PartialSumPtr->UpdateDynValidShape({64, 16});
                            mmadOutputs = {cL0PartialSumPtr};
                        }
                        auto &aMulBOp = function.AddOperation(MmadOpStr, mmadInputs, mmadOutputs);
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
    Tensor resTensor(dataType, resTensorShape, "TensorC");
    resTensor.GetStorage()->UpdateDynValidShape({resTensorShape});
    return ConstructTensorGraph(outType, inputTensor, weightTensor, biasTensor, resTensor, convAttrParam);
}

} //namespace Conv
}
}