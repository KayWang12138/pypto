/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file cube_operation_impl.cpp
 * \brief
 */

#include "operation_impl.h"
#include "common/pre_def.h"
#include "common/tile_shape.h"
#include "interface/configs/config_manager.h"
#include "interface/operation/operation.h"
#include "interface/program/program.h"
#include "interface/tensor/logical_tensor.h"
#include "interface/utils/common.h"
#include "interface/utils/log.h"
#include "interface/utils/operator_tracer.h"


#define CALL(n, ...) Tensor##n(__VA_ARGS__)

namespace npu {
namespace tile_fwk {
namespace Matrix {
template <typename T>
auto CeilAlign(T num_1, T num_2) -> T {
    if (num_2 == 0) {
        return 0;
    }
    return (num_1 + num_2 - 1) / num_2 * num_2;
}

using AggregationMap = std::map<std::vector<int64_t>, std::vector<std::pair<LogicalTensorPtr, LogicalTensorPtr>>>;

void SetMatmulAttr(Operation &op) {
    auto matrixSize = Program::GetInstance().GetMatrixSize();
    if (matrixSize.Size() < MATRIX_MAXSIZE) {
        op.SetAttribute(A_MUL_B_ACT_M, 0);
        op.SetAttribute(A_MUL_B_ACT_K, 0);
        op.SetAttribute(A_MUL_B_ACT_N, 0);
        return;
    }
    op.SetAttribute(A_MUL_B_ACT_M, matrixSize.V(M_INDEX));
    op.SetAttribute(A_MUL_B_ACT_K, matrixSize.V(K_INDEX));
    op.SetAttribute(A_MUL_B_ACT_N, matrixSize.V(N_INDEX));
}

void SetMatmulAttr(Operation &op, const std::tuple<LogicalTensorPtr, LogicalTensorPtr, LogicalTensorPtr> &tensorPtrs,
    const std::vector<int64_t> &matrixSize) {
    int nzAttr = (static_cast<int>(std::get<0>(tensorPtrs)->tensorfmt)) |
                        (static_cast<int>(std::get<1>(tensorPtrs)->tensorfmt) << 1) |
                        // 2含义：cTensorPtr的索引，同时也是cTensor NZ信息的编码偏移位数
                        (static_cast<int>(std::get<2>(tensorPtrs)->tensorfmt) << 2);
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

template <bool isTrans>
std::vector<SymbolicScalar> GetValidShapeFromTranspose(LogicalTensorPtr &l0Tensor)
{
    auto l0ValidShape = l0Tensor->GetDynValidShape();
    if (l0ValidShape.empty()) {
        return l0ValidShape;
    }
    if constexpr (isTrans) {
        std::swap(l0ValidShape.at(0), l0ValidShape.at(1));
    }
    return l0ValidShape;
}

template <bool isTransA = false, bool isTransB = false>
void CollectSubAMulB(Function &function, const CollectSubAMulBPara &args, AggregationMap &aggregations,
    const std::vector<int64_t> &l1Offset) {
    const TileShape &tileShape = args.tileShape;
    const LogicalTensorPtr &aTensorPtr = args.aTensorPtr;
    const LogicalTensorPtr &bTensorPtr = args.bTensorPtr;
    const LogicalTensorPtr &cTensorPtr = args.cTensorPtr;
    const std::array<int, 3> &posK = args.posK;
    const int kL1SizeIndex = 2;
    const int mL1 = isTransA ? aTensorPtr->shape[1] : aTensorPtr->shape[0];
    const int nL1 = isTransB ? bTensorPtr->shape[0] : bTensorPtr->shape[1];
    const auto opCodeA = isTransA ? Opcode::OP_L1_TO_L0_AT : Opcode::OP_L1_TO_L0A;
    const auto opCodeB = isTransB ? Opcode::OP_L1_TO_L0_BT : Opcode::OP_L1_TO_L0B;
    for (int mL0Idx = 0; mL0Idx < mL1; mL0Idx += tileShape.M(0)) {
        for (int nL0Idx = 0; nL0Idx < nL1; nL0Idx += tileShape.N(0)) {
            int mL0size = std::min(mL1 - mL0Idx, tileShape.M(0));
            int nL0size = std::min(nL1 - nL0Idx, tileShape.N(0));
            auto cL0Tensor = cTensorPtr->View(function, {mL0size, nL0size}, {mL0Idx, nL0Idx});
            for (int kL0Idx = 0; kL0Idx < posK[kL1SizeIndex]; kL0Idx += tileShape.K(0)) {
                int kL0size = std::min(posK[kL1SizeIndex] - kL0Idx, tileShape.K(0));
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
                    std::vector<int64_t>{mL0size, kL0size}, aL0ValidShape, "a_l0", aTensorPtr->nodetype,
                    aTensorPtr->tensorfmt);
                auto bL0ValidShape = GetValidShapeFromTranspose<isTransB>(bL0Tensor);
                auto bL0LogicalTensor = std::make_shared<LogicalTensor>(function, bTensorPtr->Datatype(),
                    std::vector<int64_t>{kL0size, nL0size}, bL0ValidShape, "b_l0", bTensorPtr->nodetype,
                    bTensorPtr->tensorfmt);
                function.AddOperation(opCodeA, {aL0Tensor}, {aL0LogicalTensor});
                function.AddOperation(opCodeB, {bL0Tensor}, {bL0LogicalTensor});
                auto l0Offset = l1Offset;
                l0Offset[0] += mL0Idx;
                l0Offset[1] += nL0Idx;
                aggregations[l0Offset].emplace_back(aL0LogicalTensor, bL0LogicalTensor);
            }
        }
    }
}

template <bool hasThirdInput = false, bool isTransA = false, bool isTransB = false>
void DoAMulB(Function &function, const AggregationMap &aggregations, const LogicalTensorPtr &inputOperand,
    const DoAMulBParam &DoAMulBPara, const std::vector<int64_t> &matrixSize) {
    const TileShape &tileShape = DoAMulBPara.tileShape;
    const LogicalTensorPtr &cTensorPtr = DoAMulBPara.cTensorPtr;
    auto dataType = cTensorPtr->Datatype();
    std::vector<int64_t> shape = {tileShape.M(0), tileShape.N(0)};
    for (const auto &[offset, aggregation] : aggregations) {
        ASSERT(!aggregation.empty());
        auto cL0PartialTensor = std::make_shared<LogicalTensor>(function, dataType, shape);
        shape[0] = std::min(tileShape.M(0), static_cast<int>(cTensorPtr->shape[0] - offset[0]));
        shape[1] = std::min(tileShape.N(0), static_cast<int>(cTensorPtr->shape[1] - offset[1]));

        auto cTilePtr = cTensorPtr->View(function, shape, offset);
        for (size_t i = 0; i < aggregation.size(); i++) {
            bool isFirstTile = i == 0;
            const auto inputWithThird =
                isFirstTile ? std::vector<LogicalTensorPtr>{aggregation[i].first, aggregation[i].second, inputOperand} :
                              std::vector<LogicalTensorPtr>{
                                  aggregation[i].first, aggregation[i].second, cL0PartialTensor, inputOperand};
            const auto inputWithOutThird =
                isFirstTile ?
                    std::vector<LogicalTensorPtr>{aggregation[i].first, aggregation[i].second} :
                    std::vector<LogicalTensorPtr>{aggregation[i].first, aggregation[i].second, cL0PartialTensor};
            const std::vector<LogicalTensorPtr> inputVec =
                (isFirstTile && hasThirdInput) ? inputWithThird : inputWithOutThird;
            const std::string matmulOpStr = isFirstTile ? "TILE_A_MUL_B" : "TILE_A_MULACC_B";
            if (i == aggregation.size() - 1) {
                auto &op = function.AddOperation(matmulOpStr, inputVec, {cTilePtr});
                if (!isFirstTile) {
                    op.oOperand.front()->SetIsDummy();
                }
                SetMatmulAttr(op, std::make_tuple(aggregation[i].first, aggregation[i].second, cTensorPtr), matrixSize);
            } else {
                auto cL0PartialSum = std::make_shared<LogicalTensor>(function, dataType, shape);
                if (!inputVec[0]->GetDynValidShape().empty() && !inputVec[1]->GetDynValidShape().empty()) {
                    cL0PartialSum->UpdateDynValidShape(
                        {inputVec[0]->GetDynValidShape()[0], inputVec[1]->GetDynValidShape()[1]});
                }
                auto &op = function.AddOperation(matmulOpStr, inputVec, {cL0PartialSum});
                if (!isFirstTile) {
                    op.oOperand.front()->SetIsDummy();
                }
                SetMatmulAttr(op, std::make_tuple(aggregation[i].first, aggregation[i].second, cTensorPtr), matrixSize);
                cL0PartialTensor = cL0PartialSum;
            }
        }
    }
}

template <bool isTransA = false, bool isTransB = false>
void L1MultiDataLoadAL1Tiles(Function &function, const std::vector<LogicalTensorPtr> &operandVec,
    const std::vector<std::pair<int, int>> &kAL1Tiles, std::vector<std::shared_ptr<LogicalTensor>> &aL1Tiles,
    const L1DataLoadParam &L1DataLoadParam) {
    const int &mL1Size = L1DataLoadParam.mL1Size;
    const int &mL1Idx = L1DataLoadParam.mL1Idx;
    const auto operand1 = operandVec[0];
    for (const auto &tile : kAL1Tiles) {
        int startK = tile.first;
        int endK = tile.second;
        auto kL1PartialSiza = endK - startK;
        auto aL1TileTensor = operand1->View(function, {mL1Size, kL1PartialSiza}, {mL1Idx, startK});
        auto inputATile = std::make_shared<LogicalTensor>(function, operand1->Datatype(),
            std::vector<int64_t>{mL1Size, kL1PartialSiza}, aL1TileTensor->GetDynValidShape(), "a_l1",
            aL1TileTensor->nodetype, aL1TileTensor->tensorfmt);
        auto &copyInA = function.AddOperation(Opcode::OP_COPY_IN, {aL1TileTensor}, {inputATile});
        copyInA.SetOpAttribute(std::make_shared<CopyOpAttribute>(OpImmediate::Specified({0, 0}), MemoryType::MEM_L1,
            OpImmediate::Specified(inputATile->GetShape()),
            OpImmediate::Specified(inputATile->tensor->GetDynRawShape()),
            OpImmediate::Specified(aL1TileTensor->GetDynValidShape())));
        aL1Tiles.push_back(inputATile);
    }
}

template <bool isTransA = false, bool isTransB = false>
void L1MultiDataLoadBL1Tiles(Function &function, const std::vector<LogicalTensorPtr> &operandVec,
    const std::vector<std::pair<int, int>> &kBL1Tiles, std::vector<std::shared_ptr<LogicalTensor>> &bL1Tiles,
    const L1DataLoadParam &L1DataLoadParam) {
    const int &nL1Size = L1DataLoadParam.nL1Size;
    const int &nL1Idx = L1DataLoadParam.nL1Idx;
    const auto operand2 = operandVec[1];
    for (const auto &tile : kBL1Tiles) {
        int startK = tile.first;
        int endK = tile.second;
        auto kL1PartialSiza = endK - startK;
        auto bL1TileTensor = isTransB ? operand2->View(function, {nL1Size, kL1PartialSiza}, {nL1Idx, startK}) :
                                        operand2->View(function, {kL1PartialSiza, nL1Size}, {startK, nL1Idx});
        auto inputBTile = isTransB ? std::make_shared<LogicalTensor>(function, operand2->Datatype(),
                                         std::vector<int64_t>{nL1Size, kL1PartialSiza}, bL1TileTensor->GetDynValidShape(),
                                         "b_l1", bL1TileTensor->nodetype, bL1TileTensor->tensorfmt) :
                                     std::make_shared<LogicalTensor>(function, operand2->Datatype(),
                                         std::vector<int64_t>{kL1PartialSiza, nL1Size}, bL1TileTensor->GetDynValidShape(),
                                         "b_l1", bL1TileTensor->nodetype, bL1TileTensor->tensorfmt);
        auto &copyInB = function.AddOperation(Opcode::OP_COPY_IN, {bL1TileTensor}, {inputBTile});
        copyInB.SetOpAttribute(std::make_shared<CopyOpAttribute>(OpImmediate::Specified({0, 0}), MemoryType::MEM_L1,
            OpImmediate::Specified(inputBTile->GetShape()),
            OpImmediate::Specified(inputBTile->tensor->GetDynRawShape()),
            OpImmediate::Specified(bL1TileTensor->GetDynValidShape())));
        bL1Tiles.push_back(inputBTile);
    }
}

template <bool isTransA = false, bool isTransB = false>
void L1MultiDataLoad(Function &function, const std::vector<LogicalTensorPtr> &operandVec,
    const L1DataLoadParam &L1DataLoadParam, const TileShape &tileShape, AggregationMap &aggregations) {
    const LogicalTensorPtr &cTilePtr = L1DataLoadParam.cTilePtr;
    const int &mL1Idx = L1DataLoadParam.mL1Idx;
    const int &nL1Idx = L1DataLoadParam.nL1Idx;
    const int &stepK = L1DataLoadParam.stepK;
    const int &mL1Size = L1DataLoadParam.mL1Size;
    const int &nL1Size = L1DataLoadParam.nL1Size;
    const int &orgK = L1DataLoadParam.orgK;
    const int lenK = 2;
    std::vector<std::pair<int, int>> kAL1Tiles, kBL1Tiles;
    for (int kL1Idx = 0; kL1Idx < orgK; kL1Idx += tileShape.K(1)) {
        int endK = std::min(kL1Idx + tileShape.K(1), orgK);
        kAL1Tiles.emplace_back(kL1Idx, endK);
    }
    for (int kL1Idx = 0; kL1Idx < orgK; kL1Idx += tileShape.K(lenK)) {
        int endK = std::min(kL1Idx + tileShape.K(lenK), orgK);
        kBL1Tiles.emplace_back(kL1Idx, endK);
    }
    std::vector<std::shared_ptr<LogicalTensor>> aL1Tiles, bL1Tiles;
    L1MultiDataLoadAL1Tiles<isTransA, isTransB>(
        function, operandVec, kAL1Tiles, aL1Tiles, {cTilePtr, mL1Idx, nL1Idx, stepK, mL1Size, nL1Size, orgK});
    L1MultiDataLoadBL1Tiles<isTransA, isTransB>(
        function, operandVec, kBL1Tiles, bL1Tiles, {cTilePtr, mL1Idx, nL1Idx, stepK, mL1Size, nL1Size, orgK});
    for (int kL1Idx = 0; kL1Idx < orgK; kL1Idx += stepK) {
        int i = kL1Idx / tileShape.K(1);
        int j = kL1Idx / tileShape.K(lenK);
        int kAL1Start = kL1Idx - kAL1Tiles[i].first;
        int kBL1Start = kL1Idx - kBL1Tiles[j].first;
        int kL1Size = std::min(stepK, orgK - kL1Idx);
        CollectSubAMulB<isTransA, isTransB>(
            function, {tileShape, {kAL1Start, kBL1Start, kL1Size}, aL1Tiles[i], bL1Tiles[j], cTilePtr},
            aggregations, {mL1Idx, nL1Idx});
    }
}

template <bool isTransA = false, bool isTransB = false>
void L1NormalLoad(Function &function, const std::vector<LogicalTensorPtr> &operandVec,
    const L1DataLoadParam &L1DataLoadParam, const TileShape &tileShape, AggregationMap &aggregations) {
    const int &orgK = L1DataLoadParam.orgK;
    const LogicalTensorPtr &cTilePtr = L1DataLoadParam.cTilePtr;
    const int &mL1Idx = L1DataLoadParam.mL1Idx;
    const int &nL1Idx = L1DataLoadParam.nL1Idx;
    const int &mL1Size = L1DataLoadParam.mL1Size;
    const int &nL1Size = L1DataLoadParam.nL1Size;
    const auto operand1 = operandVec[0];
    const auto operand2 = operandVec[1];
    for (int kL1Idx = 0; kL1Idx < orgK; kL1Idx += tileShape.K(1)) {
        auto kL1Size = std::min(orgK - kL1Idx, tileShape.K(1));
        auto aL1Tensor = isTransA ? operand1->View(function, {kL1Size, mL1Size}, {kL1Idx, mL1Idx}) :
                                    operand1->View(function, {mL1Size, kL1Size}, {mL1Idx, kL1Idx});
        auto bL1Tensor = isTransB ? operand2->View(function, {nL1Size, kL1Size}, {nL1Idx, kL1Idx}) :
                                     operand2->View(function, {kL1Size, nL1Size}, {kL1Idx, nL1Idx});
        CollectSubAMulB<isTransA, isTransB>(
            function, {tileShape, {0, 0, kL1Size}, aL1Tensor, bL1Tensor, cTilePtr},
            aggregations, {mL1Idx, nL1Idx});
    }
}

template <bool isTransA, bool isTransB>
void TiledInnerAMulB(Function &function, const TileShape &tileShape, const std::vector<LogicalTensorPtr> &operandVec,
    const LogicalTensorPtr &cTensorPtr, const std::vector<int64_t> &matmulSize) {
    const auto operand1 = operandVec[0];
    const auto operand2 = operandVec[1];

    // 2为shape的维度，当前只支持2维
    if (operand1->shape.size() != 2) {
        ASSERT(false && "only supported two dimension");
    }

    const int orgM = isTransA ? operand1->shape[1] : operand1->shape[0];
    const int orgK = isTransA ? operand1->shape[0] : operand1->shape[1];
    const int orgKb = isTransB ? operand2->shape[1] : operand2->shape[0];
    const int orgN = isTransB ? operand2->shape[0] : operand2->shape[1];
    ASSERT(orgK == orgKb);

    bool checkOrgK = isTransB ? orgK == operand2->shape[1] : orgK == operand2->shape[0];
    ASSERT(checkOrgK) << "k axis of a shape is not equal to b shape ";

    const int kBL1Idx = 2;
    ASSERT(tileShape.K(0) > 0 && tileShape.K(1) > 0 && tileShape.K(kBL1Idx) > 0 && tileShape.M(0) > 0 &&
           tileShape.M(1) > 0 && tileShape.N(0) > 0 && tileShape.N(1) > 0);
    ASSERT(tileShape.K(1) % tileShape.K(0) == 0 && "kTile[0] does not divide kTile[1]");
    ASSERT(tileShape.K(kBL1Idx) % tileShape.K(0) == 0 && "kTile[0] does not divide kTile[2]");
    const int stepK = std::gcd(tileShape.K(1), tileShape.K(kBL1Idx));

    AggregationMap aggregations;
    // 增加计算尾块的逻辑
    for (int mL1Idx = 0; mL1Idx < orgM; mL1Idx += tileShape.M(1)) {
        for (int nL1Idx = 0; nL1Idx < orgN; nL1Idx += tileShape.N(1)) {
            auto mL1Size = std::min(orgM - mL1Idx, tileShape.M(1));
            auto nL1Size = std::min(orgN - nL1Idx, tileShape.N(1));
            auto cTilePtr = cTensorPtr->View(function, {mL1Size, nL1Size}, {mL1Idx, nL1Idx});
            if (!tileShape.SetL1Tile()) {
                L1NormalLoad<isTransA, isTransB>(function, operandVec,
                    {cTilePtr, mL1Idx, nL1Idx, stepK, mL1Size, nL1Size, orgK}, tileShape, aggregations);
            } else {
                L1MultiDataLoad<isTransA, isTransB>(function, operandVec,
                    {cTilePtr, mL1Idx, nL1Idx, stepK, mL1Size, nL1Size, orgK}, tileShape, aggregations);
            }
        }
    }
    // 3的含义：gm累加场景的输入tensor数量为3
    if (operandVec.size() == 3) {
        DoAMulB<true, isTransA, isTransB>(
            function, aggregations, operandVec[SHAPE_DIM2], {tileShape, cTensorPtr}, matmulSize);
    } else {
        DoAMulB<false, isTransA, isTransB>(function, aggregations,
            std::make_shared<LogicalTensor>(
                function, cTensorPtr->Datatype(), cTensorPtr->shape, cTensorPtr->GetDynValidShape()),
            {tileShape, cTensorPtr}, matmulSize);
    }
}

template void TiledInnerAMulB<false, false>(Function &, const TileShape &, const std::vector<LogicalTensorPtr> &,
    const LogicalTensorPtr &, const std::vector<int64_t> &);
template void TiledInnerAMulB<false, true>(Function &, const TileShape &, const std::vector<LogicalTensorPtr> &,
    const LogicalTensorPtr &, const std::vector<int64_t> &);
template void TiledInnerAMulB<true, false>(Function &, const TileShape &, const std::vector<LogicalTensorPtr> &,
    const LogicalTensorPtr &, const std::vector<int64_t> &);
template void TiledInnerAMulB<true, true>(Function &, const TileShape &, const std::vector<LogicalTensorPtr> &,
    const LogicalTensorPtr &, const std::vector<int64_t> &);

void TensorInnerAMulB(
    Function &function, const std::vector<LogicalTensorPtr> &operandVec, const LogicalTensorPtr &result) {
    const auto operand1 = operandVec[0];
    const auto operand2 = operandVec[1];
    ASSERT(operand1->shape.size() == operand2->shape.size());
    ASSERT(operand1->shape[1] == operand2->shape[0]);
    if (!operand1->GetDynValidShape().empty() && !operand2->GetDynValidShape().empty()) {
        result->UpdateDynValidShape({operand1->GetDynValidShape()[0], operand2->GetDynValidShape()[1]});
    }
    auto &op = function.AddOperation(Opcode::OP_A_MUL_B, operandVec, {result});
    SetMatmulAttr(op);
}

void CheckOperandsValid(const Tensor &operand1, const Tensor &operand2) {
    ASSERT(operand1->shape.size() == operand2->shape.size());
    ASSERT(operand1->shape.size() == operand1->offset.size());
    ASSERT(operand2->shape.size() == operand2->offset.size());
}
template <bool isTransA, bool isTransB, bool isCMatrixNZ>
void CheckMatMulOperandsValid(DataType outType, const Tensor &operand1, const Tensor &operand2) {
    // shape valid check
    ASSERT(operand1->shape.size() != 0 && operand2->shape.size() != 0);
    for (size_t i = 0; i < operand1->shape.size(); ++i) {
        ASSERT(operand1->shape[i] > 0);
        ASSERT(operand2->shape[i] > 0);
    }
    // 内轴值要小于等于65535，只有ND2NZ指令需要
    auto opFormatA = operand1->GetTileOpFormat();
    auto opFormatB = operand2->GetTileOpFormat();
    // tile valid check
    auto tileShape = Program::GetInstance().GetTileShape().GetCubeTileShapes();
    int kL0 = tileShape.GetTileShape<TileShapeType::K>(0);
    int kL1 = tileShape.GetTileShape<TileShapeType::K>(1);
    int mL0 = tileShape.GetTileShape<TileShapeType::M>(0);
    int mL1 = tileShape.GetTileShape<TileShapeType::M>(1);
    int nL0 = tileShape.GetTileShape<TileShapeType::N>(0);
    int nL1 = tileShape.GetTileShape<TileShapeType::N>(1);
    if (opFormatA == TileOpFormat::TILEOP_ND) {
        ASSERT(operand1->shape.back() <= SHAPE_INNER_AXIS_MAX_SIZE);
        if (isTransA) { // nd A转置 ml0需要32B对齐
            ASSERT(mL0 * BytesOf(operand1.GetDataType()) % ALIGN_SIZE_32 == 0);
        }
    } else {
        if (isTransA) {
            ASSERT(mL0 * BytesOf(operand1.GetDataType()) % ALIGN_SIZE_32 == 0);
            ASSERT(kL0  % ALIGN_SIZE_16 == 0);
        }else {
            ASSERT(mL0 % ALIGN_SIZE_16 == 0);
            ASSERT(kL0 * BytesOf(operand1.GetDataType()) % ALIGN_SIZE_32 == 0);
        }
        // shape 满足内轴32B对齐 外轴16对齐
        ASSERT(operand1->shape[0]  % ALIGN_SIZE_16 == 0);
        ASSERT(operand1->shape[1] * BytesOf(operand1.GetDataType()) % ALIGN_SIZE_32 == 0);
    }
    if (opFormatB == TileOpFormat::TILEOP_ND) {
        ASSERT(operand2->shape.back() <= SHAPE_INNER_AXIS_MAX_SIZE);
    } else {
        if (isTransB) {
            ASSERT(kL0 * BytesOf(operand2.GetDataType()) % ALIGN_SIZE_32 == 0);
            ASSERT(nL0 % ALIGN_SIZE_16 == 0);
        }else {
            ASSERT(kL0 % ALIGN_SIZE_16 == 0);
            ASSERT(nL0 * BytesOf(operand2.GetDataType()) % ALIGN_SIZE_32 == 0);
        }
        ASSERT(operand2->shape[0] % ALIGN_SIZE_16 == 0);
        ASSERT(operand2->shape[1] * BytesOf(operand2.GetDataType()) % ALIGN_SIZE_32 == 0);
    }
    if constexpr (isCMatrixNZ) {
        int64_t nView = isTransB ? operand2->shape[0] : operand2->shape[1];
        if (outType == DataType::DT_INT32) {
            ASSERT(nView % ALIGN_SIZE_16 == 0);
            ASSERT(nL0 % ALIGN_SIZE_16 == 0);
        } else {
            ASSERT(nView * BytesOf(outType) % ALIGN_SIZE_32 == 0);
            ASSERT(nL0 * BytesOf(outType) % ALIGN_SIZE_32 == 0);
        }
    }
    ASSERT(kL0 > 0 && kL1 > 0 && mL0 > 0 && mL1 > 0 && nL0 > 0 && nL1 > 0);
    ASSERT(kL0 <= kL1 && kL1 % kL0 == 0);
    ASSERT(nL0 <= nL1 && nL1 % nL0 == 0);
    ASSERT(mL0 <= mL1 && mL1 % mL0 == 0);
    ASSERT(kL0 * BytesOf(operand1.GetDataType()) % ALIGN_SIZE_32 == 0);
    ASSERT(nL0 * BytesOf(operand1.GetDataType()) % ALIGN_SIZE_32 == 0);
}

template <bool isCMatrixNZ>
void MatmulImpl(DataType dataType, const std::vector<LogicalTensorPtr> &iOperand, LogicalTensorPtr &result) {
    const auto operand1 = iOperand[0];
    const auto operand2 = iOperand[1];
    CheckOperandsValid(operand1, operand2);
    CheckMatMulOperandsValid<false, false, isCMatrixNZ>(dataType, operand1, operand2);
    ASSERT(dataType == DT_FP32 || dataType == DT_FP16 || dataType == DT_BF16 || dataType == DT_INT32);
    if constexpr (isCMatrixNZ) {
        result->tensorfmt = TileOpFormat::TILEOP_NZ;
    }
    CALL(InnerAMulB, *Program::GetInstance().GetCurrentFunction(), iOperand, result);
}

void TensorInnerAMulBt(Function &function, const LogicalTensorPtr &operand1, const LogicalTensorPtr &operand2,
    const LogicalTensorPtr &result) {
    ASSERT(operand1->shape.size() == operand2->shape.size());
    if (!operand1->GetDynValidShape().empty() && !operand2->GetDynValidShape().empty()) {
        result->UpdateDynValidShape({operand1->GetDynValidShape()[0], operand2->GetDynValidShape()[0]});
    }
    auto &op = function.AddOperation(Opcode::OP_A_MUL_BT, {operand1, operand2}, {result});
    SetMatmulAttr(op);
}

void TensorInnerAMulBt(Function &function, const LogicalTensorPtr &operand1, const LogicalTensorPtr &operand2,
    const LogicalTensorPtr &operand3, const LogicalTensorPtr &result) {
    ASSERT(operand1->shape.size() == operand2->shape.size());
    auto &op = function.AddOperation(Opcode::OP_A_MUL_BT, {operand1, operand2, operand3}, {result});
    SetMatmulAttr(op);
}

void TensorInnerAtMulB(Function &function, const LogicalTensorPtr &operand1, const LogicalTensorPtr &operand2,
    const LogicalTensorPtr &result) {
    ASSERT(operand1->shape.size() == operand2->shape.size());
    if (!operand1->GetDynValidShape().empty() && !operand2->GetDynValidShape().empty()) {
        result->UpdateDynValidShape({operand1->GetDynValidShape()[1], operand2->GetDynValidShape()[1]});
    }
    auto &op = function.AddOperation(Opcode::OP_AT_MUL_B, {operand1, operand2}, {result});
    SetMatmulAttr(op);
}

void TensorInnerAtMulBt(Function &function, const LogicalTensorPtr &operand1, const LogicalTensorPtr &operand2,
    const LogicalTensorPtr &result) {
    ASSERT(operand1->shape.size() == operand2->shape.size());
    if (!operand1->GetDynValidShape().empty() && !operand2->GetDynValidShape().empty()) {
        result->UpdateDynValidShape({operand1->GetDynValidShape()[1], operand2->GetDynValidShape()[0]});
    }
    auto &op = function.AddOperation(Opcode::OP_AT_MUL_BT, {operand1, operand2}, {result});
    SetMatmulAttr(op);
}

template <bool isCMatrixNZ>
void AMulBtImpl(
    DataType dataType, const LogicalTensorPtr &operand1, const LogicalTensorPtr &operand2, LogicalTensorPtr &result) {
    CheckOperandsValid(operand1, operand2);
    CheckMatMulOperandsValid<false, true, isCMatrixNZ>(dataType, operand1, operand2);
    ASSERT(dataType == DataType::DT_FP32 || dataType == DataType::DT_FP16 || dataType == DataType::DT_BF16 ||
           dataType == DataType::DT_INT32);
    if constexpr (isCMatrixNZ) {
        result->tensorfmt = TileOpFormat::TILEOP_NZ;
    }
    CALL(InnerAMulBt, *Program::GetInstance().GetCurrentFunction(), operand1, operand2, result);
}

template <bool isCMatrixNZ>
void AMulBtImpl(DataType dataType, const LogicalTensorPtr &operand1, const LogicalTensorPtr &operand2,
    const LogicalTensorPtr &operand3, LogicalTensorPtr &result) {
    CheckOperandsValid(operand1, operand2);
    CheckMatMulOperandsValid<false, true, isCMatrixNZ>(dataType, operand1, operand2);
    ASSERT(dataType == DataType::DT_FP32 || dataType == DataType::DT_FP16 || dataType == DataType::DT_BF16 ||
           dataType == DataType::DT_INT32);
    if constexpr (isCMatrixNZ) {
        result->tensorfmt = TileOpFormat::TILEOP_NZ;
    }
    CALL(InnerAMulBt, *Program::GetInstance().GetCurrentFunction(), operand1, operand2, operand3, result);
}

template <bool isCMatrixNZ>
void AtMulBImpl(
    DataType dataType, const LogicalTensorPtr &operand1, const LogicalTensorPtr &operand2, LogicalTensorPtr &result) {
    CheckOperandsValid(operand1, operand2);
    CheckMatMulOperandsValid<true, false, isCMatrixNZ>(dataType, operand1, operand2);
    ASSERT(dataType == DataType::DT_FP32 || dataType == DataType::DT_FP16 || dataType == DataType::DT_BF16 ||
           dataType == DataType::DT_INT32);
    if constexpr (isCMatrixNZ) {
        result->tensorfmt = TileOpFormat::TILEOP_NZ;
    }
    CALL(InnerAtMulB, *Program::GetInstance().GetCurrentFunction(), operand1, operand2, result);
}

template <bool isCMatrixNZ>
void AtMulBtImpl(
    DataType dataType, const LogicalTensorPtr &operand1, const LogicalTensorPtr &operand2, LogicalTensorPtr &result) {
    CheckOperandsValid(operand1, operand2);
    CheckMatMulOperandsValid<true, true, isCMatrixNZ>(dataType, operand1, operand2);
    ASSERT(dataType == DataType::DT_FP32 || dataType == DataType::DT_FP16 || dataType == DataType::DT_BF16 ||
           dataType == DataType::DT_INT32);
    if constexpr (isCMatrixNZ) {
        result->tensorfmt = TileOpFormat::TILEOP_NZ;
    }
    CALL(InnerAtMulBt, *Program::GetInstance().GetCurrentFunction(), operand1, operand2, result);
}

// A mul transpose B
template <bool isCMatrixNZ>
Tensor A_MUL_Bt(DataType dataType, const Tensor &operand1, const Tensor &operand2, const void *lr) {
    DECLARE_TRACERX(lr);
    Tensor result(dataType, {operand1->shape[0], operand2->shape[0]});
    if constexpr (isCMatrixNZ) {
        ASSERT(BytesOf(dataType) > 0);
        int64_t c0Size = dataType == DataType::DT_INT32 ? ALIGN_SIZE_16 : ALIGN_SIZE_32 / BytesOf(dataType);
        result = Tensor(dataType, {operand1->shape[0], CeilAlign(operand2->shape[0], c0Size)});
    }
    AMulBtImpl<isCMatrixNZ>(dataType, operand1.GetStorage(), operand2.GetStorage(), result.GetStorage());
    return result;
}

template <bool isCMatrixNZ>
Tensor A_MUL_Bt(
    DataType dataType, const Tensor &operand1, const Tensor &operand2, const Tensor &operand3, const void *lr) {
    DECLARE_TRACERX(lr);
    Tensor result(dataType, {operand1->shape[0], operand2->shape[0]});
    if constexpr (isCMatrixNZ) {
        ASSERT(BytesOf(dataType) > 0);
        int64_t c0Size = dataType == DataType::DT_INT32 ? ALIGN_SIZE_16 : ALIGN_SIZE_32 / BytesOf(dataType);
        result = Tensor(dataType, {operand1->shape[0], CeilAlign(operand2->shape[0], c0Size)});
    }
    AMulBtImpl<isCMatrixNZ>(
        dataType, operand1.GetStorage(), operand2.GetStorage(), operand3.GetStorage(), result.GetStorage());
    return result;
}

template <bool isCMatrixNZ>
Tensor A_MUL_B(DataType dataType, const Tensor &operand1, const Tensor &operand2, const void *lr) {
    DECLARE_TRACERX(lr);
    Tensor result(dataType, {operand1->shape[0], operand2->shape[1]});
    if constexpr (isCMatrixNZ) {
        ASSERT(BytesOf(dataType) > 0);
        int64_t c0Size = dataType == DataType::DT_INT32 ? ALIGN_SIZE_16 : ALIGN_SIZE_32 / BytesOf(dataType);
        result = Tensor(dataType, {operand1->shape[0], CeilAlign(operand2->shape[1], c0Size)});
    }
    MatmulImpl<isCMatrixNZ>(dataType, {operand1.GetStorage(), operand2.GetStorage()}, result.GetStorage());
    return result;
}

template <bool isCMatrixNZ>
Tensor A_MUL_B(
    DataType dataType, const Tensor &operand1, const Tensor &operand2, const Tensor &operand3, const void *lr) {
    DECLARE_TRACERX(lr);
    Tensor result(dataType, {operand3->shape[0], operand3->shape[1]});
    if constexpr (isCMatrixNZ) {
        ASSERT(BytesOf(dataType) > 0);
        int64_t c0Size = dataType == DataType::DT_INT32 ? ALIGN_SIZE_16 : ALIGN_SIZE_32 / BytesOf(dataType);
        result = Tensor(dataType, {operand3->shape[0], CeilAlign(operand3->shape[1], c0Size)});
    }
    MatmulImpl<isCMatrixNZ>(
        dataType, {operand1.GetStorage(), operand2.GetStorage(), operand3.GetStorage()}, result.GetStorage());
    return result;
}

template <bool isCMatrixNZ>
Tensor At_MUL_B(DataType dataType, const Tensor &operand1, const Tensor &operand2, const void *lr) {
    DECLARE_TRACERX(lr);
    Tensor result(dataType, {operand1->shape[1], operand2->shape[1]});
    if constexpr (isCMatrixNZ) {
        ASSERT(BytesOf(dataType) > 0);
        int64_t c0Size = dataType == DataType::DT_INT32 ? ALIGN_SIZE_16 : ALIGN_SIZE_32 / BytesOf(dataType);
        result = Tensor(dataType, {operand1->shape[1], CeilAlign(operand2->shape[1], c0Size)});
    }
    AtMulBImpl<isCMatrixNZ>(dataType, operand1.GetStorage(), operand2.GetStorage(), result.GetStorage());
    return result;
}

template <bool isCMatrixNZ>
Tensor At_MUL_Bt(DataType dataType, const Tensor &operand1, const Tensor &operand2, const void *lr) {
    DECLARE_TRACERX(lr);
    Tensor result(dataType, {operand1->shape[1], operand2->shape[0]});
    if constexpr (isCMatrixNZ) {
        ASSERT(BytesOf(dataType) > 0);
        int64_t c0Size = dataType == DataType::DT_INT32 ? ALIGN_SIZE_16 : ALIGN_SIZE_32 / BytesOf(dataType);
        result = Tensor(dataType, {operand1->shape[1], CeilAlign(operand2->shape[0], c0Size)});
    }
    AtMulBtImpl<isCMatrixNZ>(dataType, operand1.GetStorage(), operand2.GetStorage(), result.GetStorage());
    return result;
}

template <bool isATrans, bool isBTrans, bool isCMatrixNZ>
Tensor Matmul(DataType outType, const Tensor &aMatrix, const Tensor &bMatrix) {
    if constexpr (!isATrans && !isBTrans) {
        return A_MUL_B<isCMatrixNZ>(outType, aMatrix, bMatrix, __builtin_return_address(0));
    } else if constexpr (!isATrans && isBTrans) {
        return A_MUL_Bt<isCMatrixNZ>(outType, aMatrix, bMatrix, __builtin_return_address(0));
    } else if constexpr (isATrans && !isBTrans) {
        return At_MUL_B<isCMatrixNZ>(outType, aMatrix, bMatrix, __builtin_return_address(0));
    } else {
        return At_MUL_Bt<isCMatrixNZ>(outType, aMatrix, bMatrix, __builtin_return_address(0));
    }
}

// inteface: k spilt
template <bool isATrans, bool isBTrans, bool isCMatrixNZ>
Tensor Matmul(DataType outType, const Tensor &aMatrix, const Tensor &bMatrix, const Tensor &cMatrix) {
    static_assert(!isATrans); // A trans not supported now
    if constexpr (!isATrans && !isBTrans) {
        return A_MUL_B<isCMatrixNZ>(outType, aMatrix, bMatrix, cMatrix, __builtin_return_address(0));
    } else if constexpr (!isATrans && isBTrans) {
        return A_MUL_Bt<isCMatrixNZ>(outType, aMatrix, bMatrix, cMatrix, __builtin_return_address(0));
    }
    return Tensor();
}

template Tensor Matmul<false, false, false>(DataType, const Tensor &, const Tensor &);
template Tensor Matmul<false, true, false>(DataType, const Tensor &, const Tensor &);
template Tensor Matmul<false, false, true>(DataType, const Tensor &, const Tensor &);
template Tensor Matmul<false, true, true>(DataType, const Tensor &, const Tensor &);
template Tensor Matmul<true, false, false>(DataType, const Tensor &, const Tensor &);
template Tensor Matmul<true, true, false>(DataType, const Tensor &, const Tensor &);
template Tensor Matmul<true, false, true>(DataType, const Tensor &, const Tensor &);
template Tensor Matmul<true, true, true>(DataType, const Tensor &, const Tensor &);
template Tensor Matmul<false, false, false>(DataType, const Tensor &, const Tensor &, const Tensor &);
template Tensor Matmul<false, true, false>(DataType, const Tensor &, const Tensor &, const Tensor &);
template Tensor Matmul<false, false, true>(DataType, const Tensor &, const Tensor &, const Tensor &);
template Tensor Matmul<false, true, true>(DataType, const Tensor &, const Tensor &, const Tensor &);

template <bool isTransA, bool isTransB, bool isCMatrixNZ>
Tensor ABatchMulB3D(DataType dataType, const Tensor &operand1, const Tensor &operand2) {
    ASSERT(operand1->shape.size() == operand2->shape.size() && operand1->shape.size() == SHAPE_DIM3);
    const int batchSizeA = operand1->shape[0];
    const int batchSizeB = operand2->shape[0];
    ASSERT(batchSizeA == batchSizeB || batchSizeB == 1 || batchSizeA == 1);

    const int orgM = isTransA ? operand1->shape[SHAPE_DIM2] : operand1->shape[1];
    const int orgKa = isTransA ? operand1->shape[1] : operand1->shape[SHAPE_DIM2];
    const int orgKb = isTransB ? operand2->shape[2] : operand2->shape[1];
    const int orgN = isTransB ? operand2->shape[1] :operand2->shape[SHAPE_DIM2];
    ASSERT(orgKa == orgKb);
    int firstDimA = isTransA ? orgKa : orgM;
    int secondDimA = isTransA ? orgM : orgKa;
    int firstDimB = isTransB ? orgN : orgKb;
    int secondDimB = isTransB ? orgKb : orgN;
    int batchSize = std::max(batchSizeA, batchSizeB);
    auto operand2D1 = Reshape(operand1, {batchSizeA * firstDimA, secondDimA});
    auto operand2D2 = Reshape(operand2, {batchSizeB * firstDimB, secondDimB});
    Tensor result(dataType, {batchSize * orgM, orgN});
    if constexpr (isCMatrixNZ) {
        result = Tensor(dataType, {batchSize * orgM, orgN}, "BatchMatmulOutputNz",
            TileOpFormat::TILEOP_NZ);
    }
    auto &curFunc = *Program::GetInstance().GetCurrentFunction();
    for (int i = 0; i < batchSize; i++) {
        int offsetA = batchSizeA == 1 ? 0 : i * firstDimA;
        int offsetB = batchSizeB == 1 ? 0 : i * firstDimB;
        int offsetC = i * orgM;
        auto tensorA = operand2D1->View(curFunc, {firstDimA, secondDimA}, {offsetA, 0});
        auto tensorB = operand2D2->View(curFunc, {firstDimB, secondDimB}, {offsetB, 0});
        auto tensorC = result->View(curFunc, {orgM, orgN}, {offsetC, 0});
        if (!isTransA  && !isTransB) {
            MatmulImpl<isCMatrixNZ>(dataType, {tensorA, tensorB}, tensorC);
        } else if (!isTransA && isTransB) {
            AMulBtImpl<isCMatrixNZ>(dataType, tensorA, tensorB, tensorC);
        } else if (isTransA && !isTransB) {
            AtMulBImpl<isCMatrixNZ>(dataType, tensorA, tensorB, tensorC);
        } else {
           AtMulBtImpl<isCMatrixNZ>(dataType, tensorA, tensorB, tensorC);
        }
    }
    return Reshape(result, {batchSize, orgM, orgN});
};

template <bool isTransA, bool isTransB, bool isCMatrixNZ>
Tensor ABatchMulB4D(DataType dataType, const Tensor &operand1, const Tensor &operand2) {
    ASSERT(operand1->shape.size() == SHAPE_DIM4 && operand2->shape.size() == SHAPE_DIM4);

    const int batchSizeA1 = operand1->shape[0];
    const int batchSizeA2 = operand1->shape[1];
    const int batchSizeB1 = operand2->shape[0];
    const int batchSizeB2 = operand2->shape[1];
    ASSERT(batchSizeA1 == batchSizeB1 || batchSizeB1 == 1 || batchSizeA1 == 1);
    ASSERT(batchSizeA2 == batchSizeB2 || batchSizeB2 == 1 || batchSizeA2 == 1);

    const int orgM = isTransA ? operand1->shape[SHAPE_DIM3] :operand1->shape[SHAPE_DIM2];
    const int orgKa = isTransA ? operand1->shape[SHAPE_DIM2] : operand1->shape[SHAPE_DIM3];
    const int orgKb = isTransB ? operand2->shape[SHAPE_DIM3] : operand2->shape[SHAPE_DIM2];
    const int orgN = isTransB ? operand2->shape[SHAPE_DIM2] : operand2->shape[SHAPE_DIM3];
    ASSERT(orgKa == orgKb);
    int firstDimA = isTransA ? orgKa : orgM;
    int secondDimA = isTransA ? orgM : orgKa;
    int firstDimB = isTransB ? orgN : orgKb;
    int secondDimB = isTransB ? orgKb : orgN;
    auto operand2D1 = Reshape(operand1, {batchSizeA1 * batchSizeA2 * firstDimA, secondDimA});
    auto operand2D2 = Reshape(operand2, {batchSizeB1 * batchSizeB2 * firstDimB, secondDimB});
    int batchSize1 = std::max(batchSizeA1, batchSizeB1);
    int batchSize2 = std::max(batchSizeA2, batchSizeB2);
    Tensor result(dataType, {batchSize1 * batchSize2 * orgM, orgN});
    if constexpr (isCMatrixNZ) {
        result = Tensor(dataType, {batchSize1 * batchSize2 * orgM, orgN}, "BatchMatmulOutputNz",
            TileOpFormat::TILEOP_NZ);
    }

    int strideA = batchSizeA2 == 1 ? 0 : firstDimA;
    int strideB = batchSizeB2 == 1 ? 0 : firstDimB;
    int offsetC = 0;
    auto &curFunc = *Program::GetInstance().GetCurrentFunction();
    for (int i = 0; i < batchSize1; i++) {
        int offsetA = batchSizeA1 == 1 ? 0 : i * batchSizeA2 * firstDimA;
        int offsetB = batchSizeB1 == 1 ? 0 : i * batchSizeB2 * firstDimB;
        for (int j = 0; j < batchSize2; j++) {
            auto tensorA = operand2D1->View(curFunc, {firstDimA, secondDimA}, {offsetA, 0});
            auto tensorB = operand2D2->View(curFunc, {firstDimB, secondDimB}, {offsetB, 0});
            auto tensorC = result->View(curFunc, {orgM, orgN}, {offsetC, 0});
            if constexpr (!isTransA && !isTransB) {
                MatmulImpl<isCMatrixNZ>(dataType, {tensorA, tensorB}, tensorC);
            } else if constexpr (!isTransA && isTransB) {
                AMulBtImpl<isCMatrixNZ>(dataType, tensorA, tensorB, tensorC);
            } else if constexpr (isTransA && !isTransB) {
                AtMulBImpl<isCMatrixNZ>(dataType, tensorA, tensorB, tensorC);
            } else {
                AtMulBtImpl<isCMatrixNZ>(dataType, tensorA, tensorB, tensorC);
            }
            offsetC += orgM;
            offsetA += strideA;
            offsetB += strideB;
        }
    }
    return Reshape(result, {batchSize1, batchSize2, orgM, orgN});
};

template <bool isTransA, bool isTransB, bool isCMatrixNZ>
Tensor BatchMatmul(DataType dataType, const Tensor &aMatrix, const Tensor &bMatrix) {
    DECLARE_TRACER();
    ASSERT(aMatrix.GetShape().size() == bMatrix.GetShape().size());
    Tensor res;
    if constexpr (!isTransA && isTransB) {
        if (aMatrix.GetShape().size() == SHAPE_DIM4) {
            res = ABatchMulB4D<isTransA, isTransB, isCMatrixNZ>(dataType, aMatrix, bMatrix);
        } else {
            res = ABatchMulB3D<isTransA, isTransB, isCMatrixNZ>(dataType, aMatrix, bMatrix);
        }
    } else if constexpr (!isTransA && !isTransB) {
        if (aMatrix.GetShape().size() == SHAPE_DIM4) {
            res = ABatchMulB4D<isTransA, isTransB, isCMatrixNZ>(dataType, aMatrix, bMatrix);
        } else {
            res = ABatchMulB3D<isTransA, isTransB, isCMatrixNZ>(dataType, aMatrix, bMatrix);
        }
    } else if constexpr (isTransA && !isTransB){
        if (aMatrix.GetShape().size() == SHAPE_DIM4) {
            res = ABatchMulB4D<isTransA, isTransB, isCMatrixNZ>(dataType, aMatrix, bMatrix);
        } else {
            res = ABatchMulB3D<isTransA, isTransB, isCMatrixNZ>(dataType, aMatrix, bMatrix);
        }
    } else {
        if (aMatrix.GetShape().size() == SHAPE_DIM4) {
            res = ABatchMulB4D<isTransA, isTransB, isCMatrixNZ>(dataType, aMatrix, bMatrix);
        } else {
            res = ABatchMulB3D<isTransA, isTransB, isCMatrixNZ>(dataType, aMatrix, bMatrix);
        }
    }
    return res;
}

template Tensor BatchMatmul<false, false, false>(DataType, const Tensor &, const Tensor &);
template Tensor BatchMatmul<false, true, false>(DataType, const Tensor &, const Tensor &);
template Tensor BatchMatmul<false, false, true>(DataType, const Tensor &, const Tensor &);
template Tensor BatchMatmul<false, true, true>(DataType, const Tensor &, const Tensor &);
template Tensor BatchMatmul<true, false, false>(DataType, const Tensor &, const Tensor &);
template Tensor BatchMatmul<true, true, false>(DataType, const Tensor &, const Tensor &);
template Tensor BatchMatmul<true, false, true>(DataType, const Tensor &, const Tensor &);
template Tensor BatchMatmul<true, true, true>(DataType, const Tensor &, const Tensor &);
} // namespace Matrix
} // namespace tile_fwk
} // namespace npu