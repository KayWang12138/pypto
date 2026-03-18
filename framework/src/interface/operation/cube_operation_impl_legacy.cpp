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
 * \file cube_operation_impl_legacy.cpp
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
#include "interface/utils/matmul_error.h"
#include "interface/utils/operator_tracer.h"
#include "operation_impl.h"
#include "tilefwk/data_type.h"
#include "tilefwk/platform.h"
#include "tilefwk/tile_shape.h"

namespace npu {
namespace tile_fwk {
namespace Matrix {

namespace Deprecate {

const int32_t GMACC = 3;
const int32_t BIASINDEX = 2;
const int32_t SCALEINDEX = 3;

using AggregationMap = std::map<std::vector<int64_t>,
    std::vector<std::tuple<LogicalTensorPtr, LogicalTensorPtr, LogicalTensorPtr, LogicalTensorPtr>>>;

struct TensorAttributes {
    int64_t tileSize;
    int64_t offset;
    std::string name;
    MemoryType memType{MemoryType::MEM_UNKNOWN};
};

struct MatmulInputs {
    LogicalTensorPtr aTensorPtr = nullptr;
    LogicalTensorPtr bTensorPtr = nullptr;
    LogicalTensorPtr cTensorPtr = nullptr;
    LogicalTensorPtr biasTensorPtr = nullptr;
    LogicalTensorPtr scaleTensorPtr = nullptr;
};

void SetMatmulAttr(Operation &op, const std::tuple<LogicalTensorPtr, LogicalTensorPtr, LogicalTensorPtr> &tensorPtrs,
    const std::vector<int64_t> &matrixSize) {
    int64_t nzAttr = (static_cast<int64_t>(std::get<0>(tensorPtrs)->Format())) |
                     (static_cast<int64_t>(std::get<1>(tensorPtrs)->Format()) << 1) |
                     // 2含义：cTensorPtr的索引，同时也是cTensor NZ信息的编码偏移位数
                     (static_cast<int64_t>(std::get<2>(tensorPtrs)->Format()) << 2);
    op.SetAttribute(MATMUL_NZ_ATTR, nzAttr);
    if (matrixSize.size() < MATRIX_MAXSIZE) {
        op.SetAttribute(A_MUL_B_ACT_M, 0);
        op.SetAttribute(A_MUL_B_ACT_K, 0);
        op.SetAttribute(A_MUL_B_ACT_N, 0);
        return;
    }
    op.SetAttribute(A_MUL_B_ACT_M, matrixSize[M_INDEX]);
    op.SetAttribute(A_MUL_B_ACT_K, matrixSize[K_INDEX]);
    op.SetAttribute(A_MUL_B_ACT_N, matrixSize[N_INDEX]);
}

void SetBiasAndScaleAttr(
    const MatmulInputs &matmulInputs, const MatmulAttrParam &matmulAttrParam, bool isFirstTile, Operation &op) {
    if (matmulInputs.biasTensorPtr != nullptr && isFirstTile) {
        op.SetAttribute(A_MUL_B_BIAS_ATTR, true);
    }
    op.SetAttribute(A_MUL_B_TRANS_MODE_ATTR, static_cast<int64_t>(matmulAttrParam.transMode));
    if (isFirstTile) {
        op.SetAttribute(A_MUL_B_RELU_ATTR, static_cast<int64_t>(matmulAttrParam.reluType));
    }
    if (matmulAttrParam.scaleValue != 0 && isFirstTile) {
        op.SetAttribute(A_MUL_B_SCALE_ATTR, Element(DataType::DT_UINT64, matmulAttrParam.scaleValue));
    }
    if (matmulAttrParam.gmAccumulationFlag) {
        op.SetAttribute(A_MUL_B_GM_ACC, matmulAttrParam.gmAccumulationFlag);
    }
};

struct L1DataLoadParam {
    const LogicalTensorPtr &cTilePtr;
    const int64_t mL1Idx;
    const int64_t nL1Idx;
    const int64_t stepK;
    const int64_t mL1Size;
    const int64_t nL1Size;
    const int64_t orgK;
};

struct CollectSubAMulBPara {
    const TileShape &tileShape;
    const std::array<int64_t, 3> &posK;
    const LogicalTensorPtr &aTensorPtr;
    const LogicalTensorPtr &bTensorPtr;
    const LogicalTensorPtr &cTensorPtr;
    const LogicalTensorPtr &biasTensorPtr = nullptr;
    const LogicalTensorPtr &scaleTensorPtr = nullptr;
};

struct DoAMulBParam {
    const TileShape &tileShape;
    const LogicalTensorPtr &cTensorPtr;
};

template <bool isTrans>
std::vector<SymbolicScalar> GetValidShapeFromTranspose(LogicalTensorPtr &l0Tensor) {
    auto l0ValidShape = l0Tensor->GetDynValidShape();
    if (l0ValidShape.empty()) {
        return l0ValidShape;
    }
    if constexpr (isTrans) {
        std::swap(l0ValidShape.at(0), l0ValidShape.at(1));
    }
    return l0ValidShape;
}

void AddOpView(
    Function &function, const LogicalTensorPtr &operand, LogicalTensorPtr &viewTensor, const TensorAttributes &attrs) {
    DataType dtype = ((attrs.name == "bias_BT" && operand->Datatype() == DataType::DT_FP16) ? DataType::DT_FP32 :
                                                                                              operand->Datatype());
    viewTensor = std::make_shared<LogicalTensor>(function, dtype, std::vector<int64_t>{1, attrs.tileSize},
        SymbolicScalar::FromConcrete({1, attrs.tileSize}), operand->Format(), attrs.name, operand->nodetype);
    viewTensor->UpdateDynValidShape(
        GetViewValidShape(operand->GetDynValidShape(), {0, attrs.offset}, {}, {1, attrs.tileSize}));
    auto &viewOperand = function.AddOperation(Opcode::OP_VIEW, {operand}, {viewTensor});
    std::vector<int64_t> newoffset{0, attrs.offset};

    auto viewAttribute = std::make_shared<ViewOpAttribute>(
        newoffset, SymbolicScalar::FromConcrete(newoffset), viewTensor->GetDynValidShape());
    viewAttribute->SetToType(attrs.memType);
    viewOperand.SetOpAttribute(viewAttribute);
    if (attrs.name.find("l1") != std::string::npos) {
        viewOperand.SetAttribute(A_MUL_B_COPY_IN_MODE, static_cast<int64_t>(0));
    }
}

template <bool isTransA = false, bool isTransB = false>
void CollectSubAMulB(Function &function, const CollectSubAMulBPara &args, AggregationMap &aggregations,
    const std::vector<int64_t> &l1Offset) {
    const CubeTile &cubeTile = args.tileShape.GetCubeTile();
    const LogicalTensorPtr &aTensorPtr = args.aTensorPtr;
    const LogicalTensorPtr &bTensorPtr = args.bTensorPtr;
    const LogicalTensorPtr &cTensorPtr = args.cTensorPtr;
    const LogicalTensorPtr &biasTensorPtr = args.biasTensorPtr;
    const LogicalTensorPtr &scaleTensorPtr = args.scaleTensorPtr;

    const std::array<int64_t, 3> &posK = args.posK;
    const int32_t kL1SizeIndex = 2;
    const int64_t mL1 = isTransA ? aTensorPtr->shape[1] : aTensorPtr->shape[0];
    const int64_t nL1 = isTransB ? bTensorPtr->shape[0] : bTensorPtr->shape[1];
    const auto opCodeA = isTransA ? Opcode::OP_L1_TO_L0_AT : Opcode::OP_L1_TO_L0A;
    const auto opCodeB = isTransB ? Opcode::OP_L1_TO_L0_BT : Opcode::OP_L1_TO_L0B;

    for (int64_t mL0Idx = 0; mL0Idx < mL1; mL0Idx += cubeTile.m[0]) {
        for (int64_t nL0Idx = 0; nL0Idx < nL1; nL0Idx += cubeTile.n[0]) {
            int64_t mL0size = std::min(mL1 - mL0Idx, cubeTile.m[0]);
            int64_t nL0size = std::min(nL1 - nL0Idx, cubeTile.n[0]);
            LogicalTensorPtr biasToBTLogicalTensor = nullptr;
            LogicalTensorPtr scaleToFBLogicalTensor = nullptr;
            if (biasTensorPtr != nullptr) {
                TensorAttributes biasAttrs = {nL0size, nL0Idx, "bias_BT", MemoryType::MEM_BT};
                AddOpView(function, biasTensorPtr, biasToBTLogicalTensor, biasAttrs);
            }
            if (scaleTensorPtr != nullptr) {
                TensorAttributes scaleAttrs = {nL0size, nL0Idx, "scale_FB", MemoryType::MEM_FIX_QUANT_PRE};
                AddOpView(function, scaleTensorPtr, scaleToFBLogicalTensor, scaleAttrs);
            }
            auto cL0Tensor = cTensorPtr->View(function, {mL0size, nL0size}, {mL0Idx, nL0Idx});
            for (int64_t kL0Idx = 0; kL0Idx < posK[kL1SizeIndex]; kL0Idx += cubeTile.k[0]) {
                int64_t kL0size = std::min(posK[kL1SizeIndex] - kL0Idx, cubeTile.k[0]);
                const std::vector<int64_t> sizeVecA =
                    isTransA ? std::vector<int64_t>{kL0size, mL0size} : std::vector<int64_t>{mL0size, kL0size};
                const std::vector<int64_t> sizeVecB =
                    isTransB ? std::vector<int64_t>{nL0size, kL0size} : std::vector<int64_t>{kL0size, nL0size};
                auto aL0Tensor = isTransA ? aTensorPtr->View(function, sizeVecA, {posK[0] + kL0Idx, mL0Idx}) :
                                            aTensorPtr->View(function, sizeVecA, {mL0Idx, posK[0] + kL0Idx});
                auto bL0Tensor = isTransB ? bTensorPtr->View(function, sizeVecB, {nL0Idx, posK[1] + kL0Idx}) :
                                            bTensorPtr->View(function, sizeVecB, {posK[1] + kL0Idx, nL0Idx});
                auto aL0ValidShape = GetValidShapeFromTranspose<isTransA>(aL0Tensor);
                auto aL0LogicalTensor = std::make_shared<LogicalTensor>(function, aTensorPtr->Datatype(),
                    std::vector<int64_t>{mL0size, kL0size}, aL0ValidShape, aTensorPtr->Format(), "a_l0",
                    aTensorPtr->nodetype);
                auto bL0ValidShape = GetValidShapeFromTranspose<isTransB>(bL0Tensor);
                auto bL0LogicalTensor = std::make_shared<LogicalTensor>(function, bTensorPtr->Datatype(),
                    std::vector<int64_t>{kL0size, nL0size}, bL0ValidShape, bTensorPtr->Format(), "b_l0",
                    bTensorPtr->nodetype);
                function.AddOperation(opCodeA, {aL0Tensor}, {aL0LogicalTensor});
                function.AddOperation(opCodeB, {bL0Tensor}, {bL0LogicalTensor});
                auto l0Offset = l1Offset;
                l0Offset[0] += mL0Idx;
                l0Offset[1] += nL0Idx;

                std::tuple<LogicalTensorPtr, LogicalTensorPtr, LogicalTensorPtr, LogicalTensorPtr> logicalTensor;
                std::get<0>(logicalTensor) = aL0LogicalTensor;
                std::get<1>(logicalTensor) = bL0LogicalTensor;
                if (biasTensorPtr != nullptr) {
                    std::get<BIASINDEX>(logicalTensor) = biasToBTLogicalTensor;
                }
                if (scaleTensorPtr != nullptr) {
                    std::get<SCALEINDEX>(logicalTensor) = scaleToFBLogicalTensor;
                }
                aggregations[l0Offset].emplace_back(logicalTensor);
            }
        }
    }
}

template <bool isTransA = false, bool isTransB = false>
void L1NormalLoad(Function &function, const MatmulInputs &matmulInputs, const L1DataLoadParam &L1DataLoadParam,
    const TileShape &tileShape, AggregationMap &aggregations) {
    const int64_t orgK = L1DataLoadParam.orgK;
    const LogicalTensorPtr &cTilePtr = L1DataLoadParam.cTilePtr;
    const int64_t mL1Idx = L1DataLoadParam.mL1Idx;
    const int64_t nL1Idx = L1DataLoadParam.nL1Idx;
    const int64_t mL1Size = L1DataLoadParam.mL1Size;
    const int64_t nL1Size = L1DataLoadParam.nL1Size;
    const auto operand1 = matmulInputs.aTensorPtr;
    const auto operand2 = matmulInputs.bTensorPtr;
    auto &cubeTile = tileShape.GetCubeTile();
    for (int64_t kL1Idx = 0; kL1Idx < orgK; kL1Idx += cubeTile.k[1]) {
        int64_t kL1Size = std::min(orgK - kL1Idx, cubeTile.k[1]);
        auto aL1Tensor = isTransA ? operand1->View(function, {kL1Size, mL1Size}, {kL1Idx, mL1Idx}) :
                                    operand1->View(function, {mL1Size, kL1Size}, {mL1Idx, kL1Idx});
        auto bL1Tensor = isTransB ? operand2->View(function, {nL1Size, kL1Size}, {nL1Idx, kL1Idx}) :
                                    operand2->View(function, {kL1Size, nL1Size}, {kL1Idx, nL1Idx});

        LogicalTensorPtr biasL1Tensor = nullptr;
        LogicalTensorPtr scaleL1Tensor = nullptr;
        if (matmulInputs.biasTensorPtr != nullptr) {
            TensorAttributes biasAttrs = {nL1Size, nL1Idx, "bias_l1", MemoryType::MEM_L1};
            AddOpView(function, matmulInputs.biasTensorPtr, biasL1Tensor, biasAttrs);
        }
        if (matmulInputs.scaleTensorPtr != nullptr) {
            TensorAttributes scaleAttrs = {nL1Size, nL1Idx, "scale_l1", MemoryType::MEM_L1};
            AddOpView(function, matmulInputs.scaleTensorPtr, scaleL1Tensor, scaleAttrs);
        }
        CollectSubAMulB<isTransA, isTransB>(function,
            {
                tileShape, {0, 0, kL1Size},
                 aL1Tensor, bL1Tensor, cTilePtr, biasL1Tensor, scaleL1Tensor
        },
            aggregations, {mL1Idx, nL1Idx});
    }
}

void ExtendTileOprandInputs(
    const std::vector<LogicalTensorPtr> &operandVec, MatmulInputs &matmulInputs, const MatmulAttrParam &params) {
    matmulInputs.aTensorPtr = operandVec[0];
    matmulInputs.bTensorPtr = operandVec[1];
    if (params.hasBias) {
        matmulInputs.biasTensorPtr = operandVec[SHAPE_DIM2];
        if (params.hasScale) {
            matmulInputs.scaleTensorPtr = operandVec[SHAPE_DIM3];
        }
    } else if (params.hasScale) {
        matmulInputs.scaleTensorPtr = operandVec[SHAPE_DIM2];
    } else if (operandVec.size() == GMACC) {
        matmulInputs.cTensorPtr = operandVec[SHAPE_DIM2];
    }
}

template <bool hasThirdInput = false>
void DoAMulB(Function &function, const AggregationMap &aggregations, const MatmulInputs &matmulInputs,
    const DoAMulBParam &DoAMulBPara, const MatmulAttrParam &matmulAttrParam) {
    const auto &cubeTile = DoAMulBPara.tileShape.GetCubeTile();
    const LogicalTensorPtr &cTensorPtr = DoAMulBPara.cTensorPtr;
    LogicalTensorPtr inputOperand = hasThirdInput ? matmulInputs.cTensorPtr :
                                                    std::make_shared<LogicalTensor>(function, cTensorPtr->Datatype(),
                                                        cTensorPtr->shape, cTensorPtr->GetDynValidShape());
    auto dataType = cTensorPtr->Datatype();
    std::vector<int64_t> shape = {cubeTile.m[0], cubeTile.n[0]};
    const std::vector<int64_t> matrixSize = {matmulAttrParam.mValue, matmulAttrParam.kValue, matmulAttrParam.nValue};
    for (const auto &[offset, aggregation] : aggregations) {
        ASSERT(MatmulErrorCode::ERR_RUNTIME_LOGIC, !aggregation.empty()) << "AggregationMap is empty.";
        auto cL0PartialTensor = std::make_shared<LogicalTensor>(function, dataType, shape);
        shape[0] = std::min(cubeTile.m[0], cTensorPtr->shape[0] - offset[0]);
        shape[1] = std::min(cubeTile.n[0], cTensorPtr->shape[1] - offset[1]);
        auto cTilePtr = cTensorPtr->View(function, shape, offset);
        auto partialGM = inputOperand->View(function, shape, offset);
        for (size_t i = 0; i < aggregation.size(); i++) {
            bool isFirstTile = i == 0;
            std::vector<LogicalTensorPtr> inputWithThird = {std::get<0>(aggregation[i]), std::get<1>(aggregation[i])};
            if (matmulInputs.biasTensorPtr != nullptr && isFirstTile) {
                inputWithThird.push_back(std::get<BIASINDEX>(aggregation[i]));
            }
            if (matmulInputs.scaleTensorPtr != nullptr && isFirstTile) {
                inputWithThird.push_back(std::get<SCALEINDEX>(aggregation[i]));
            }
            std::vector<LogicalTensorPtr> inputWithOutThird = inputWithThird;
            if (isFirstTile) {
                inputWithThird.push_back(partialGM);
            } else {
                inputWithOutThird.push_back(cL0PartialTensor);
                inputWithThird.push_back(cL0PartialTensor);
                inputWithThird.push_back(partialGM);
            }
            const std::vector<LogicalTensorPtr> inputVec =
                (isFirstTile && hasThirdInput) ? inputWithThird : inputWithOutThird;
            const std::string matmulOpStr = isFirstTile ? "TILE_A_MUL_B" : "TILE_A_MULACC_B";
            if (i == aggregation.size() - 1) {
                auto &op = function.AddOperation(matmulOpStr, inputVec, {cTilePtr});
                SetBiasAndScaleAttr(matmulInputs, matmulAttrParam, isFirstTile, op);
                if (!isFirstTile) {
                    op.oOperand.front()->SetIsDummy();
                }
                SetMatmulAttr(op, std::make_tuple(std::get<0>(aggregation[i]), std::get<1>(aggregation[i]), cTensorPtr),
                    matrixSize);
            } else {
                auto cL0PartialSum = std::make_shared<LogicalTensor>(function, dataType, shape);
                if (!inputVec[0]->GetDynValidShape().empty() && !inputVec[1]->GetDynValidShape().empty()) {
                    cL0PartialSum->UpdateDynValidShape(
                        {inputVec[0]->GetDynValidShape()[0], inputVec[1]->GetDynValidShape()[1]});
                }
                auto &op = function.AddOperation(matmulOpStr, inputVec, {cL0PartialSum});
                SetBiasAndScaleAttr(matmulInputs, matmulAttrParam, isFirstTile, op);
                if (!isFirstTile) {
                    op.oOperand.front()->SetIsDummy();
                }
                SetMatmulAttr(op, std::make_tuple(std::get<0>(aggregation[i]), std::get<1>(aggregation[i]), cTensorPtr),
                    matrixSize);
                cL0PartialTensor = cL0PartialSum;
            }
        }
    }
}

template <bool isTransA, bool isTransB>
void TiledInnerAMulB(Function &function, const TileShape &tileShape, const std::vector<LogicalTensorPtr> &operandVec,
    const LogicalTensorPtr &cTensorPtr, const MatmulAttrParam &params) {
    MatmulInputs matmulInputs;
    ExtendTileOprandInputs(operandVec, matmulInputs, params);
    const auto operand1 = matmulInputs.aTensorPtr;
    const auto operand2 = matmulInputs.bTensorPtr;

    // 2为shape的维度，当前只支持2维
    ASSERT(MatmulErrorCode::ERR_PARAM_UNSUPPORTED, operand1->shape.size() == 2) << "Only supported two dimension.";
    const int64_t orgM = isTransA ? operand1->shape[1] : operand1->shape[0];
    const int64_t orgKa = isTransA ? operand1->shape[0] : operand1->shape[1];
    const int64_t orgKb = isTransB ? operand2->shape[1] : operand2->shape[0];
    const int64_t orgN = isTransB ? operand2->shape[0] : operand2->shape[1];
    ASSERT(MatmulErrorCode::ERR_PARAM_MISMATCH, orgKa == orgKb)
        << "K-axis mismatch: orgKa: " << orgKa << ", orgKb: " << orgKb;

    const int32_t kBL1Idx = 2;
    auto &cubeTile = tileShape.GetCubeTile();
    const int64_t stepK = std::gcd(cubeTile.k[1], cubeTile.k[kBL1Idx]);

    AggregationMap aggregations;
    // 增加计算尾块的逻辑
    for (int64_t mL1Idx = 0; mL1Idx < orgM; mL1Idx += cubeTile.m[1]) {
        for (int64_t nL0Idx = 0; nL0Idx < orgN; nL0Idx += cubeTile.n[0]) {
            auto mL1Size = std::min(orgM - mL1Idx, cubeTile.m[1]);
            auto nL1Size = std::min(orgN - nL0Idx, cubeTile.n[0]);

            auto cTilePtr = cTensorPtr->View(function, {mL1Size, nL1Size}, {mL1Idx, nL0Idx});
            L1NormalLoad<isTransA, isTransB>(function, matmulInputs,
                {cTilePtr, mL1Idx, nL0Idx, stepK, mL1Size, nL1Size, orgKa}, tileShape, aggregations);
        }
    }

    if (matmulInputs.cTensorPtr != nullptr) {
        DoAMulB<true>(function, aggregations, matmulInputs, {tileShape, cTensorPtr}, params);
    } else {
        DoAMulB<false>(function, aggregations, matmulInputs, {tileShape, cTensorPtr}, params);
    }
}

void TiledInnerAMulB(Function &function, const TileShape &tileShape, const std::vector<LogicalTensorPtr> &operandVec,
    const LogicalTensorPtr &cTensorPtr, const MatmulAttrParam &params) {
    if (params.transA && params.transB) {
        TiledInnerAMulB<true, true>(function, tileShape, operandVec, cTensorPtr, params);
    } else if (params.transA && !params.transB) {
        TiledInnerAMulB<true, false>(function, tileShape, operandVec, cTensorPtr, params);
    } else if (!params.transA && params.transB) {
        TiledInnerAMulB<false, true>(function, tileShape, operandVec, cTensorPtr, params);
    } else {
        TiledInnerAMulB<false, false>(function, tileShape, operandVec, cTensorPtr, params);
    }
}

bool IsSFANonDB(
    const TileShape &tileShape, const std::vector<LogicalTensorPtr> &operandVec, const MatmulAttrParam &params) {
    constexpr int64_t tileSize = 128;
    constexpr int64_t dimM = 128;
    constexpr int64_t dimK = 576;
    constexpr int64_t dimN = 2048;

    auto &cubeTile = tileShape.GetCubeTile();
    const int64_t aMatrixM = params.transA ? operandVec[0]->shape[1] : operandVec[0]->shape[0];
    const int64_t aMatrixK = params.transA ? operandVec[0]->shape[0] : operandVec[0]->shape[1];
    const int64_t bMatrixK = params.transB ? operandVec[1]->shape[1] : operandVec[0]->shape[0];
    const int64_t bMatrixN = params.transB ? operandVec[1]->shape[0] : operandVec[0]->shape[1];

    const bool kTileValid = (cubeTile.k[0] == tileSize) && (cubeTile.k[1] == tileSize);
    const bool mTileValid = (cubeTile.m[0] == tileSize) && (cubeTile.m[1] == tileSize);
    const bool nTileValid = (cubeTile.n[0] == tileSize) && (cubeTile.n[1] == tileSize);
    const bool tileValid = kTileValid && mTileValid && nTileValid;

    const bool aMatrixValid = (aMatrixM == dimM && aMatrixK == dimK);
    const bool bMatrixValid = (bMatrixK == dimK && bMatrixN == dimN);
    const bool IsInputValid = aMatrixValid && bMatrixValid;

    const bool dataTypeValid = (operandVec[0]->Datatype() == DataType::DT_BF16);

    return (tileValid && IsInputValid && dataTypeValid);
}

} // namespace Deprecate
} // namespace Matrix
} // namespace tile_fwk
} // namespace npu
