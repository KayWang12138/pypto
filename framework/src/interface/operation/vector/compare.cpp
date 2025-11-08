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
 * \file basic.cpp
 * \brief
 */

#include "binary.h"
#include "interface/utils/operator_tracer.h"
#include "passes/pass_utils/graph_utils.h"

namespace npu::tile_fwk {

void TiledCompareOperationImpl(Function &function, const TileShape &tileShape, size_t cur, Input &input1, Input &input2,
    const LogicalTensorPtr &result, TileInfo &resultTileInfo, OpType operation, OutType mode) {
    if (cur == result->shape.size()) {
        auto inputTile1 = input1.tensor.GetStorage()->View(function, input1.tileInfo.shape, input1.tileInfo.offset);
        auto inputTile2 = input2.tensor.GetStorage()->View(function, input2.tileInfo.shape, input2.tileInfo.offset);
        auto resultTile = result->View(function, resultTileInfo.shape, resultTileInfo.offset);

        const int64_t COUNT_MODE_SIZE = 4096;
        std::vector<int64_t> vcmpBitResultShape({COUNT_MODE_SIZE / (int64_t)BytesOf(input1.tensor.GetDataType()) / 8});
        auto vcmpBitResultTensor = std::make_shared<LogicalTensor>(function, DT_UINT8, vcmpBitResultShape);
        std::vector<int64_t> zeroCondShape({COUNT_MODE_SIZE / (int64_t)BytesOf(input1.tensor.GetDataType())});
        auto zeroCondTensor = std::make_shared<LogicalTensor>(function, input1.tensor.GetDataType(), zeroCondShape);
        std::vector<int64_t> oneCondition({COUNT_MODE_SIZE / (int64_t)BytesOf(input1.tensor.GetDataType())});
        auto oneCondTensor = std::make_shared<LogicalTensor>(function, input1.tensor.GetDataType(), oneCondition);
        std::vector<int64_t> vselResult({COUNT_MODE_SIZE / (int64_t)BytesOf(input1.tensor.GetDataType())});
        auto vselResultTensor = std::make_shared<LogicalTensor>(function, input1.tensor.GetDataType(), vselResult);
        std::vector<int64_t> startAddrUBShape({1});
        auto startAddrUBTensor = std::make_shared<LogicalTensor>(function, DT_UINT64, startAddrUBShape);
        auto &op = function.AddOperation(Opcode::OP_CMP, {inputTile1, inputTile2},
            {resultTile, vcmpBitResultTensor, zeroCondTensor, oneCondTensor, vselResultTensor, startAddrUBTensor});

        op.SetAttribute(OP_ATTR_PREFIX + "cmp_operation", static_cast<int64_t>(operation));
        op.SetAttribute(OP_ATTR_PREFIX + "cmp_mode", static_cast<int64_t>(mode));
        return;
    }

    auto &vecTile = tileShape.GetVecTile();
    for (int i = 0; i < result->shape[cur]; i += vecTile[cur]) {
        resultTileInfo.offset[cur] = i;
        resultTileInfo.shape[cur] = std::min(result->shape[cur] - resultTileInfo.offset[cur], vecTile[cur]);
        input1.tileInfo.offset[cur] = i % input1.tensor.GetShape()[cur];
        input1.tileInfo.shape[cur] =
            std::min(input1.tensor.GetShape()[cur] - input1.tileInfo.offset[cur], vecTile[cur]);
        input2.tileInfo.offset[cur] = i % input2.tensor.GetShape()[cur];
        input2.tileInfo.shape[cur] =
            std::min(input2.tensor.GetShape()[cur] - input2.tileInfo.offset[cur], vecTile[cur]);
        TiledCompareOperationImpl(
            function, tileShape, cur + 1, input1, input2, result, resultTileInfo, operation, mode);
    }
}

void TiledCompareOperation(Function &function, const TileShape &tileShape, LogicalTensorPtr operand1,
    LogicalTensorPtr operand2, const LogicalTensorPtr &result, OpType operation, OutType mode) {
    auto broadcastOperand = [&](LogicalTensorPtr &operand, LogicalTensorPtr &other) {
        auto dstShape = result->shape;
        if (mode == OutType::BIT) {
            dstShape[dstShape.size() - 1] *= 8; // compare output 8 bit to 1 byte
        }
        if (operand->shape == dstShape) {
            return;
        }
        auto expanded = std::make_shared<LogicalTensor>(function, operand->Datatype(), dstShape);
        Expand(function, tileShape, operand, {other}, expanded);
        operand = expanded;
    };
    broadcastOperand(operand1, operand2);
    broadcastOperand(operand2, operand1);

    TileInfo tileInfo1(result->shape.size(), result->offset.size());
    TileInfo tileInfo2(result->shape.size(), result->offset.size());
    TileInfo resultTileInfo(result->shape.size(), result->offset.size());
    auto input1 = Input{operand1, tileInfo1};
    auto input2 = Input{operand2, tileInfo2};

    TiledCompareOperationImpl(function, tileShape, 0, input1, input2, result, resultTileInfo, operation, mode);
}

LogicalTensorPtr TensorCompareOperation(
    Function &function, const Tensor &self, const Tensor &other, OpType operation, OutType mode) {
    auto operandT1 = self.GetStorage();
    auto operandT2 = other.GetStorage();
    if (operandT1->shape.size() != operandT2->shape.size()) {
        std::vector<int> broadCastShape = GetBroadCastShape(operandT1, operandT2);
        operandT1 = BinaryOperationBroadCast(operandT1, broadCastShape);
        operandT2 = BinaryOperationBroadCast(operandT2, broadCastShape);
    }
    std::vector<SymbolicScalar> resultValidShape;
    std::vector<int64_t> resultShape = BinaryOperationResultShape(operandT1, operandT2);
    if (!operandT1->GetDynValidShape().empty() && !operandT2->GetDynValidShape().empty()) {
        for (size_t i = 0; i < resultShape.size(); ++i) {
            if (resultShape[i] == operandT1->shape[i]) {
                resultValidShape.push_back(operandT1->GetDynValidShape()[i]);
            } else {
                resultValidShape.push_back(operandT2->GetDynValidShape()[i]);
            }
        }
    }
    auto resultType = DT_BOOL;
    if (mode == OutType::BIT) {
        resultType = DT_UINT8;
        if (!resultShape.empty() && resultShape.back() % NUM_VALUE_8 != 0) {
            ALOG_ERROR_F("Last dimension must be divisible by 8 in BIT mode");
        }
        if (!resultShape.empty()) {
            resultShape.back() /= NUM_VALUE_8;
            if (!resultValidShape.empty()) {
                resultValidShape.back() = resultValidShape.back() / NUM_VALUE_8;
            }
        }
    }
    auto result = std::make_shared<LogicalTensor>(function, resultType, resultShape, resultValidShape);
    auto &op = function.AddOperation(Opcode::OP_CMP, {operandT1, operandT2}, {result});
    op.SetAttribute(OP_ATTR_PREFIX + "cmp_operation", static_cast<int64_t>(operation));
    op.SetAttribute(OP_ATTR_PREFIX + "cmp_mode", static_cast<int64_t>(mode));
    return result;
}

Tensor Compare(const Tensor &self, const Tensor &other, OpType op, OutType mode) {
    DECLARE_TRACER();
    RETURN_CALL(CompareOperation, *Program::GetInstance().GetCurrentFunction(), self, other, op, mode);
}

void CompareOperationTileFunc(Function &function, const TileShape &tileShape,
    const std::vector<LogicalTensorPtr> &iOperand, const std::vector<LogicalTensorPtr> &oOperand, const Operation &op) {
    BinaryOperationOperandCheck(iOperand, oOperand);
    auto operation = static_cast<OpType>(op.GetIntAttribute(OP_ATTR_PREFIX + "cmp_operation"));
    auto mode = static_cast<OutType>(op.GetIntAttribute(OP_ATTR_PREFIX + "cmp_mode"));
    TiledCompareOperation(function, tileShape, iOperand[0], iOperand[1], oOperand[0], operation, mode);
}

REGISTER_OPERATION_TILED_FUNC(OP_CMP, Opcode::OP_CMP, CompareOperationTileFunc);

} // namespace npu::tile_fwk