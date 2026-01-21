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

Tensor ConstructTensorGraph(DataType dataType, const Tensor &inputTensor, const Tensor &weightTensor, const Tensor &biasTensor,
    const Tensor &gmMatrix, const std::vector<int> &strides, const std::vector<int> &paddings, const std::vector<int> &dilations,
    const int groups) {
    
    // infer hout, wout
    int batchOut = 1;
    int Ci1 = 2;
    int hOut = 8;
    int wOut = 8;
    int C0 = 16;
    Tensor resTensor(dataType, {batchOut, Ci1, hOut, wOut, C0}, "TensorC");
    resTensor.GetStorage()->UpdateDynValidShape({batchOut, Ci1, hOut, wOut, C0});

    // add Conv node
    Function *functionPtr = Program::GetInstance().GetCurrentFunction();

    OP_CHECK(true, { ASSERT(functionPtr != nullptr) << "functionPtr is nullptr." << std::endl; });
    std::vector<LogicalTensorPtr> operandVec = {inputTensor.GetStorage(), weightTensor.GetStorage()};
    if (biasTensor.GetStorage() != nullptr) {
        operandVec.push_back(biasTensor.GetStorage());
    }
    auto &op = functionPtr->AddOperation(Opcode::OP_CONV, operandVec, {resTensor.GetStorage()});

    return resTensor;
}

void ConstructTileGraph(Function &function, const TileShape &tileShape, const std::vector<LogicalTensorPtr> &operandVec,
                        const LogicalTensorPtr &cTensorPtr, const Operation &op)
{
    /*
    * 从Tile shape 和 attr 获取规格，属性 todo
    * convTile
    */

    int groups = 1;
    int orgBatch = 1;
    int orgCout = 32;
    int orgHWout = 16;
    int orgK = 32 * 1 * 1;
    int kAL1 = 16;
    int kBL1 = 32;
    int hAL1In = 8;
    int wAL1In = 8;
    int hAL1Out = 8;
    int wAL1Out = 8;
    int nBL1 = 16;


    LogicalTensorPtr dstAL1TensorPtr = nullptr;
    LogicalTensorPtr dstBL1TensorPtr = nullptr;
    LogicalTensorPtr dstAL0TensorPtr = nullptr;
    LogicalTensorPtr dstBL0TensorPtr = nullptr;
    LogicalTensorPtr cL0PartialSumPtr = nullptr;
    LogicalTensorPtr dstBiasl1TensorPtr = nullptr;
    LogicalTensorPtr dstBiasBtTensorPtr = nullptr;

    for (int groupIdx = 0; groupIdx < groups; groupIdx += 1) {
        for (int batchIdx = 0; batchIdx < orgBatch; batchIdx += 1) {
            for (int nl0Idx = 0; nl0Idx < orgCout; nl0Idx += 16) {
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

                for (int ml0Idx = 0; ml0Idx < orgHWout; ml0Idx += 64) {
                    for (int kl0Idx = 0; kl0Idx < orgK; kl0Idx += 16) {
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
            const std::vector<int> &strides, const std::vector<int> &paddings, const std::vector<int> &dilations,
            const int groups) {
    CheckConvOperands(outType, inputTensor, weightTensor, biasTensor);
    // auto &conveTile = TileShape::Current().GetConvTile();
    return ConstructTensorGraph(outType, inputTensor, weightTensor, biasTensor, Tensor(), strides, paddings, dilations, groups);
}

} //namespace Conv
}
}