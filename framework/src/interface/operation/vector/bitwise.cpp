/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file bitwise.cpp
 * \brief
 */

#include "bitwise.h"
#include "tensor_transformation.h"
#include "interface/utils/operator_tracer.h"
#include "interface/configs/config_manager.h"

namespace npu::tile_fwk {

void CheckOperandsValid(const LogicalTensorPtr &operand1, const LogicalTensorPtr &operand2) {
    ASSERT(operand1->shape.size() == operand2->shape.size()) << "The shape size of the two input tensors must be equal";
}

void CheckBinOpOperandsValid(const LogicalTensorPtr &operand1, const LogicalTensorPtr &operand2) {
    CheckOperandsValid(operand1, operand2);
    for (size_t i = 0; i < operand1->shape.size(); ++i) {
        if (operand1->shape[i] != operand2->shape[i] && (operand1->shape[i] != 1 && operand2->shape[i] != 1)) {
            ASSERT(false && "shape not support bitwise operation");
        }
    }
}

void CheckBitwiseInputTensors(const LogicalTensorPtr &tensor1, const LogicalTensorPtr &tensor2, std::string &op) {
    CheckTensorShape(tensor1, op);
    CheckTensorShape(tensor2, op);
    CheckBinOpOperandsValid(tensor1, tensor2);
    if (tensor1->Datatype() != tensor2->Datatype()) {
        ASSERT(false && "The dtype of input tensors are not same.");
    }
    if (tensor1->Format() != tensor2->Format()) {
        ASSERT(false && "The format of input tensors are not same.");
    }
}

void BitwiseOperationOperandCheck(
    const std::vector<LogicalTensorPtr> &iOperand, const std::vector<LogicalTensorPtr> &oOperand) {
    constexpr size_t inOpSize = 2;
    constexpr size_t outOpSize = 1;
    ASSERT(iOperand.size() == inOpSize && "iOperand size should be 2");
    ASSERT(oOperand.size() == outOpSize && "oOperand size should be 1");
}

// [m,n] + [m, 1]
bool CallBrcBinOp(LogicalTensorPtr operand1, LogicalTensorPtr operand2) {
    ASSERT(operand1->shape.size() == operand2->shape.size() && "Dims not match");
    size_t shapeSize = operand1->shape.size();
    for (size_t i = 0; i < shapeSize - 1; ++i) {
        if (operand1->shape[i] != operand2->shape[i]) {
            return false;
        }
    }

    return ((operand1->shape[shapeSize - 1] != 1) && (operand2->shape[shapeSize - 1] == 1)) ||
           ((operand1->shape[shapeSize - 1] == 1) && (operand2->shape[shapeSize - 1] != 1));
}

struct LogicalInput {
    const LogicalTensorPtr tensor;
    TileInfo tileInfo;
};

template <BitwiseOpType T>
void TiledBitwiseOperation(Function &function, const TileShape &tileShape, size_t cur, LogicalInput &input1,
    LogicalInput &input2, const LogicalTensorPtr &result, TileInfo &resultTileInfo, bool withBrc) {
    constexpr size_t shapeSize = 2;
    if (cur == input1.tensor->GetShape().size()) {
        auto inputTile1 = input1.tensor->View(function, input1.tileInfo.shape, input1.tileInfo.offset);
        auto inputTile2 = input2.tensor->View(function, input2.tileInfo.shape, input2.tileInfo.offset);
        auto resultTile = result->View(function, resultTileInfo.shape, resultTileInfo.offset);
        if (withBrc) {
            std::vector<int64_t> tmpShape(input1.tileInfo.shape);
            auto alignSize = BLOCK_SIZE / BytesOf(input2.tensor->Datatype());
            tmpShape[input1.tileInfo.shape.size() - 1] = alignSize;
            if (input1.tileInfo.shape.size() == shapeSize) {
                tmpShape[input1.tileInfo.shape.size() - shapeSize] =
                    (tmpShape[input1.tileInfo.shape.size() - shapeSize] + alignSize - 1) / alignSize * alignSize;
            }
            auto tempTensor = std::make_shared<LogicalTensor>(function, input2.tensor->Datatype(), tmpShape);
            function.AddOperation(
                GetBitwiseOpNameCode<T, false, true>(), {inputTile1, inputTile2}, {resultTile, tempTensor});
        } else {
            function.AddOperation(GetBitwiseOpNameCode<T, false, false>(), {inputTile1, inputTile2}, {resultTile});
        }
        return;
    }
    auto &vecTile = tileShape.GetVecTile();
    for (int i = 0; i < result->shape[cur]; i += vecTile[cur]) {
        resultTileInfo.offset[cur] = i;
        resultTileInfo.shape[cur] = std::min(result->shape[cur] - resultTileInfo.offset[cur], vecTile[cur]);
        input1.tileInfo.offset[cur] = i % input1.tensor->GetShape()[cur];
        input1.tileInfo.shape[cur] =
            std::min(input1.tensor->GetShape()[cur] - input1.tileInfo.offset[cur], vecTile[cur]);
        input2.tileInfo.offset[cur] = i % input2.tensor->GetShape()[cur];
        input2.tileInfo.shape[cur] =
            std::min(input2.tensor->GetShape()[cur] - input2.tileInfo.offset[cur], vecTile[cur]);
        TiledBitwiseOperation<T>(function, tileShape, cur + 1, input1, input2, result, resultTileInfo, withBrc);
    }
}

template <BitwiseOpType T>
void TiledBitwiseOperation(Function &function, const TileShape &tileShape, LogicalTensorPtr operand1,
    LogicalTensorPtr operand2, const LogicalTensorPtr &result) {
    CheckBinOpOperandsValid(operand1, operand2);
    bool withBrc = CallBrcBinOp(operand1, operand2) &&
                   (ConfigManager::Instance().GetOperationConfig(KEY_FORCE_COMBINE_AXIS, false) ||
                       ConfigManager::Instance().GetOperationConfig(KEY_COMBINE_AXIS, false));
    // nolast brc will be inline
    if (!withBrc) {
        if (operand1->shape != result->shape) {
            auto targetShape = result->shape;
            auto tmp = std::make_shared<LogicalTensor>(function, operand1->Datatype(), targetShape);
            Expand(function, tileShape, operand1, {operand2}, tmp);
            operand1 = tmp;
        }

        if (operand2->shape != result->shape) {
            auto targetShape = result->shape;
            auto tmp = std::make_shared<LogicalTensor>(function, operand2->Datatype(), targetShape);
            Expand(function, tileShape, operand2, {operand1}, tmp);
            operand2 = tmp;
        }
    }

    TileInfo tileInfo1(result->shape.size(), result->offset.size());
    TileInfo tileInfo2(result->shape.size(), result->offset.size());
    TileInfo resultTileInfo(result->shape.size(), result->offset.size());
    auto input1 = LogicalInput{operand1, tileInfo1};
    auto input2 = LogicalInput{operand2, tileInfo2};
    // 如果使能了Combine Axis逻辑，需要将withbrc置为false，避免后续走OP_XX_BRC逻辑
    if (ConfigManager::Instance().GetOperationConfig(KEY_COMBINE_AXIS, false)) {
        withBrc = false;
    }
    TiledBitwiseOperation<T>(function, tileShape, 0, input1, input2, result, resultTileInfo, withBrc);
}

Tensor BitwiseAnd(const Tensor &self, const Tensor &other) {
 	DECLARE_TRACER();
 	ASSERT(self.GetDataType() == other.GetDataType()) << "The datatype of the two input must be equal";
 	std::vector<DataType> MAXS_SUPPORT_DATATYPES = {
 	         DataType::DT_UINT32, DataType::DT_UINT16, DataType::DT_INT32, DataType::DT_INT16};
 	ASSERT(std::find(MAXS_SUPPORT_DATATYPES.begin(), MAXS_SUPPORT_DATATYPES.end(), self.GetDataType()) !=
 	    MAXS_SUPPORT_DATATYPES.end())
 	    << "The datatype is not supported";
 	RETURN_CALL(BitwiseOperation<BitwiseOpType::BITWISEAND>, *Program::GetInstance().GetCurrentFunction(), self, other);
}
 	 

template <BitwiseOpType T>
void TiledBitwiseOperationScalar(Function &function, const TileShape &tileShape, size_t cur, LogicalInput &input1,
    Element &value, const LogicalTensorPtr &result, TileInfo &resultTileInfo, bool reverseOperand) {
    if (cur == input1.tensor->GetShape().size()) {
        auto inputTile1 = input1.tensor->View(function, input1.tileInfo.shape, input1.tileInfo.offset);
        auto resultTile = result->View(function, resultTileInfo.shape, resultTileInfo.offset);
        // 确认接口
        auto &op = function.AddOperation(GetBitwiseOpNameCode<T, true>(), {inputTile1}, {resultTile});
        op.SetAttribute(OpAttributeKey::scalar, value);
        op.SetAttribute(OP_ATTR_PREFIX + "reverseOperand", reverseOperand);
        return;
    }
    auto &vecTile = tileShape.GetVecTile();
    for (int i = 0; i < result->shape[cur]; i += vecTile[cur]) {
        resultTileInfo.offset[cur] = i;
        resultTileInfo.shape[cur] = std::min(result->shape[cur] - resultTileInfo.offset[cur], vecTile[cur]);
        input1.tileInfo.offset[cur] = i % input1.tensor->GetShape()[cur];
        input1.tileInfo.shape[cur] =
            std::min(input1.tensor->GetShape()[cur] - input1.tileInfo.offset[cur], vecTile[cur]);

        TiledBitwiseOperationScalar<T>(
            function, tileShape, cur + 1, input1, value, result, resultTileInfo, reverseOperand);
    }
}

template <BitwiseOpType T>
void TiledBitwiseOperationScalar(Function &function, const TileShape &tileShape, LogicalTensorPtr operand1,
    Element value, const LogicalTensorPtr &result, bool reverseOperand = false) {
    TileInfo tileInfo1(result->shape.size(), result->offset.size());
    TileInfo resultTileInfo(result->shape.size(), result->offset.size());
    auto input1 = LogicalInput{operand1, tileInfo1};
    TiledBitwiseOperationScalar<T>(function, tileShape, 0, input1, value, result, resultTileInfo, reverseOperand);
}

// Tensor Add(const Tensor &self, const Element &other) {
//     DECLARE_TRACER();
//     RETURN_CALL(BitwiseOperationScalar<BitwiseOpType::ADD>, *Program::GetInstance().GetCurrentFunction(),
//         self.GetStorage(), other);
// }

// OP_BITWISEAND
template <BitwiseOpType T>
void BitwiseOperationTileFunc(Function &function, const TileShape &tileShape,
    const std::vector<LogicalTensorPtr> &iOperand, const std::vector<LogicalTensorPtr> &oOperand,
    [[maybe_unused]] const Operation &op) {
    BitwiseOperationOperandCheck(iOperand, oOperand);
    TiledBitwiseOperation<T>(function, tileShape, iOperand[0], iOperand[1], oOperand[0]);
}

// OP_ADDS OP_SUBS OP_MULS OP_DIVS OP_MAXS OP_MINS
template <BitwiseOpType T>
void BitwiseOperationScalarTileFunc(Function &function, const TileShape &tileShape,
    const std::vector<LogicalTensorPtr> &iOperand, const std::vector<LogicalTensorPtr> &oOperand,
    [[maybe_unused]] const Operation &op) {
    TiledBitwiseOperationScalar<T>(
        function, tileShape, iOperand[0], op.GetElementAttribute(OpAttributeKey::scalar), oOperand[0]);
}


REGISTER_OPERATION_TILED_FUNC(OP_BITWISEAND, Opcode::OP_BITWISEAND, BitwiseOperationTileFunc<BitwiseOpType::BITWISEAND>);

} // namespace npu::tile_fwk