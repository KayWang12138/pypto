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
 * \file operation_impl.cpp
 * \brief
 */

#include "operation_impl.h"
#include <memory>
#include "tilefwk/data_type.h"
#include "interface/operation/operation.h"
#include "distributed/distributed_expand.h"
#include "interface/function/function.h"
#include "tilefwk/symbolic_scalar.h"
#include "tilefwk/tensor.h"
#include "interface/tensor/logical_tensor.h"
#include "interface/program/program.h"
#include "interface/configs/config_manager.h"
#include "interface/utils/common.h"
#include "interface/utils/log.h"
#include "interface/utils/operator_tracer.h"

using namespace npu::tile_fwk;

namespace npu::tile_fwk {

} // namespace npu::tile_fwk

namespace {

struct TileInfo {
    std::vector<int> shape;
    std::vector<int> offset;
    std::vector<SymbolicScalar> validShape;

    TileInfo(size_t shapeSize, size_t offsetSize) : shape(shapeSize), offset(offsetSize), validShape(shapeSize) {}

    TileInfo(std::vector<int> aShape, std::vector<int> aOffset, std::vector<SymbolicScalar> aValidShape = {})
        : shape(std::move(aShape)), offset(std::move(aOffset)), validShape(aValidShape) {}
};

struct Input {
    const Tensor tensor;
    TileInfo tileInfo;
};

enum class TransposeOpType {
    TRANSPOSE_DATAMOVE,
    TRANSPOSE_VNCHWCONV,
};

template <TransposeOpType T>
Opcode GetTransposeOpName() {
#define CASE(X) \
case TransposeOpType::X: return Opcode::OP_##X
    switch (T) {
        CASE(TRANSPOSE_DATAMOVE);
        CASE(TRANSPOSE_VNCHWCONV);
        default: assert(false && "unknown unary op type");
    }
#undef CASE
}

template <CastOpType T>
Opcode GetCastOpName() {
#define CASE(X) \
    case CastOpType::X: return Opcode::OP_##X
    switch (T) {
        CASE(CAST);
        default: assert(false && "unknown cast op type");
    }
#undef CASE
}

template <UnaryOpType T>
std::string GetUnaryOpName() {
    switch (T) {
        case UnaryOpType::EXP:
            return "EXP";
        case UnaryOpType::SQRT:
            return "SQRT";
        case UnaryOpType::RECIPROCAL:
            return "RECIPROCAL";
        case UnaryOpType::DUPLICATE:
            return "DUPLICATE";
        case UnaryOpType::ABS:
            return "ABS";
        default:
            assert(false && "unknown unary op type");
            return "";
    }
}

template <UnaryOpType T>
Opcode GetUnaryOpNameCode() {
#define CASE(X) \
case UnaryOpType::X: return Opcode::OP_## X
    switch (T) {
        CASE(EXP);
        CASE(SQRT);
        CASE(RECIPROCAL);
        CASE(ABS);
        default: assert(false && "unknown unary op type");
    }
#undef CASE
}

template <BinaryOpType T>
std::string GetBinaryOpName() {
    switch (T) {
        case BinaryOpType::ADD:
            return "ADD";
        case BinaryOpType::SUB:
            return "SUB";
        case BinaryOpType::MUL:
            return "MUL";
        case BinaryOpType::DIV:
            return "DIV";
        case BinaryOpType::MAXIMUM:
            return "MAXIMUM";
        default:
            assert(false && "unknown binary op type");
            return "";
    }
}

template <BinaryOpType T, bool WithElement = false, bool WithBrc = false>
Opcode GetBinaryOpNameCode() {
    if constexpr (WithElement) {
#define CASE(X) \
    case BinaryOpType::X: return Opcode::OP_##X##S
        switch (T) {
            CASE(ADD);
            CASE(SUB);
            CASE(MUL);
            CASE(DIV);
            CASE(S_ADD);
            CASE(S_SUB);
            CASE(S_MUL);
            CASE(S_DIV);
            CASE(S_MAX);
            default: assert(false && "unknown binary op type");
        }
#undef CASE
    }

    if constexpr (WithBrc) {
#define CASE(X) \
    case BinaryOpType::X: return Opcode::OP_##X##_BRC
        switch (T) {
            CASE(ADD);
            CASE(SUB);
            CASE(MUL);
            CASE(DIV);
            default: assert(false && "unknown binary op type");
        }
#undef CASE
    }

#define CASE(X) \
    case BinaryOpType::X: return Opcode::OP_##X
    switch (T) {
        CASE(ADD);
        CASE(SUB);
        CASE(MUL);
        CASE(DIV);
        CASE(S_ADD);
        CASE(S_SUB);
        CASE(S_MUL);
        CASE(S_DIV);
        CASE(S_MAX);
        CASE(MAXIMUM);
        default: assert(false && "unknown binary op type");
    }
#undef CASE
}

std::vector<int> GetBroadCastShape(LogicalTensorPtr &operand1, LogicalTensorPtr &operand2) {
    std::vector<int> opShape1(operand1->shape);
    std::vector<int> opShape2(operand2->shape);
    auto maxShapeSize = std::max(opShape1.size(), opShape2.size());
    if (opShape1.size() != maxShapeSize) {
        opShape1.insert(opShape1.begin(), maxShapeSize - opShape1.size(), 1);
    }
    if (opShape2.size() != maxShapeSize) {
        opShape2.insert(opShape2.begin(), maxShapeSize - opShape2.size(), 1);
    }
    std::vector<int> broadCastShape(maxShapeSize, 0);
    for (size_t i = 0; i < maxShapeSize; i++) {
        broadCastShape[i] = std::max(opShape1[i], opShape2[i]);
    }
    return broadCastShape;
}

LogicalTensorPtr BinaryOperationBroadCast(const LogicalTensorPtr &operand,
    const std::vector<int> &broadCastShape) {
    if(operand->shape.size() < broadCastShape.size()) {
        auto broadCastDims = broadCastShape.size() - operand->shape.size();
        std::vector<int> unsqueezeShape(operand->shape);
        unsqueezeShape.insert(unsqueezeShape.begin(), broadCastDims, 1);
        auto tmpOperand = Reshape(operand,unsqueezeShape).GetStorage();
        return tmpOperand;
    }
    return operand;
}

void CheckOperandsValid(const Tensor &operand1, const Tensor &operand2) {
    assert(operand1->shape.size() == operand2->shape.size());
    assert(operand1->shape.size() == operand1->offset.size());
    assert(operand2->shape.size() == operand2->offset.size());
}

void CheckBinOpOperandsValid(
    const LogicalTensorPtr &operand1, const LogicalTensorPtr &operand2) {
    CheckOperandsValid(operand1, operand2);
    for (size_t i = 0; i < operand1->shape.size(); ++i) {
        if (operand1->shape[i] != operand2->shape[i] && (operand1->shape[i] != 1 && operand2->shape[i] != 1)) {
            assert(false && "shape not support binary operation");
        }
    }
}

template <UnaryOpType T>
Tensor UnaryOperation(Tensor operand) {
    Tensor result(operand->tensor->datatype, operand->shape);
    assert(operand->shape.size() == operand->offset.size());
    Program::GetInstance().AddOperation(GetUnaryOpName<T>(), {operand.GetStorage()}, {result.GetStorage()});
    return result;
}

template <UnaryOpType T>
LogicalTensorPtr TensorUnaryOperation(Function &function, LogicalTensorPtr operand) {
    auto result = std::make_shared<LogicalTensor>(function, operand->tensor->datatype, operand->shape, operand->GetDynValidShape());
    function.AddOperation(GetUnaryOpNameCode<T>(), {operand}, {result});
    return result;
}

void CheckExpandTensorVaild(const LogicalTensorPtr &operand, const LogicalTensorPtr &result) {
    if (operand->shape.size() != result->shape.size()) {
        assert(false && "Dims not match");
    }

    for (size_t i = 0; i < result->shape.size(); ++i) {
        if (operand->shape[i] != result->shape[i] && operand->shape[i] != 1) {
            assert(0 && "shape not match");
        }
    }
}

void ExpandTile(Function &function, const struct ExpandInfo &expandInfo) {
    auto resultTile = expandInfo.result->View(function, expandInfo.viewShape, expandInfo.offset);

    std::vector<int> srcShape(expandInfo.srcTensor->shape.size(), 1);
    for (size_t i = 0; i < expandInfo.result->shape.size(); i++) {
        srcShape[i] = std::min(expandInfo.viewShape[i], expandInfo.srcTensor->shape[i]);
    }

    std::vector<int> srcOffset = expandInfo.offset;
    for (size_t j = 0; j < srcOffset.size(); j++) {
        if (expandInfo.srcTensor->shape[j] < expandInfo.result->shape[j]) {
            srcOffset[j] = expandInfo.offset[j] % expandInfo.srcTensor->shape[j];
        }
    }
    auto srcTile = expandInfo.srcTensor->View(function, srcShape, srcOffset);
    auto &newOp = function.AddOperation("TILE_EXPAND", {srcTile}, {resultTile});
    newOp.SetAttribute(OP_ATTR_PREFIX + "EXPANDDIM", expandInfo.expandDim);
    newOp.SetAttribute(OP_ATTR_PREFIX + "validShape", resultTile->GetDynValidShape());
}

void ExpandTile(Function &function, const TileShape &tileShape, int dimIdx, const struct ExpandInfo &expandInfo) {
    if (static_cast<size_t>(dimIdx) == expandInfo.result->shape.size()) {
        ExpandTile(function, expandInfo);
        return;
    }
    for (int i = 0; i < expandInfo.result->shape[dimIdx]; i += tileShape.V(dimIdx)) {
        expandInfo.offset[dimIdx] = i;
        expandInfo.viewShape[dimIdx] = std::min(expandInfo.result->shape[dimIdx] - i, tileShape.V(dimIdx));
        ExpandTile(function, tileShape, dimIdx + 1, expandInfo);
    }
}

void Expand(Function &function, const TileShape &tileShape, const LogicalTensorPtr &operand,
    const LogicalTensorPtr &result) {
    CheckExpandTensorVaild(operand, result);
    ASSERT(function.GetGraphType() == GraphType::TILE_GRAPH);

    std::vector<int> offset(result->shape.size(), 0);
    std::vector<int> viewShape(result->shape.size(), 1);
    std::vector<SymbolicScalar> outValidShape;
    int expandDim = -1;
    for (size_t i = 0; i < result->shape.size(); ++i) {
        if (operand->shape[i] != result->shape[i]) {
            expandDim = i;
            outValidShape.push_back(result->shape[i]);
        } else {
            outValidShape.push_back(operand->shape[i]);
        }
    }

    result->UpdateDynValidShape(outValidShape);
    struct ExpandInfo expandInfo(operand, result, viewShape, offset, expandDim);
    ExpandTile(function, tileShape, 0, expandInfo);
}

void TiledExpand(Function &function, const TileShape &tileShape, const LogicalTensorPtr &operand,
    const LogicalTensorPtr &result) {
    Expand(function, tileShape, operand, result);
}

void TensorExpand(Function &function, const LogicalTensorPtr &operand, const LogicalTensorPtr &result) {
    function.AddOperation(Opcode::OP_EXPAND, {operand}, {result});
}

// [m,n] + [m, 1]
bool CallBrcBinOp(LogicalTensorPtr operand1, LogicalTensorPtr operand2) {
    assert(operand1->shape.size() == operand2->shape.size() && "Dims not match");
    size_t shapeSize = operand1->shape.size();
    for(size_t i = 0; i < shapeSize - 1; ++i) {
        if (operand1->shape[i] != operand2->shape[i]) {
            return false;
        }
    }

    return (operand1->shape[shapeSize - 1] != 1) && (operand2->shape[shapeSize - 1] == 1);
}

template <BinaryOpType T>
void TiledBinaryOperation(Function &function, const TileShape &tileShape, size_t cur, Input &input1,
    Input &input2, const LogicalTensorPtr &result, TileInfo &resultTileInfo, bool withBrc) {
    if (cur == input1.tensor->shape.size()) {
        auto inputTile1 = input1.tensor->View(function, input1.tileInfo.shape, input1.tileInfo.offset);
        auto inputTile2 = input2.tensor->View(function, input2.tileInfo.shape, input2.tileInfo.offset);
        auto resultTile = result->View(function, resultTileInfo.shape, resultTileInfo.offset);
        if(withBrc) {
            std::vector<int> tmpShape(input1.tileInfo.shape);
            tmpShape[input1.tileInfo.shape.size() - 1] = BLOCK_SIZE / BytesOf(input2.tensor.GetDataType());
            auto tempTensor = std::make_shared<LogicalTensor>(function, input2.tensor->Datatype(), tmpShape);
            function.AddOperation(GetBinaryOpNameCode<T, false, true>(), {inputTile1, inputTile2}, {resultTile, tempTensor});
        } else {
            function.AddOperation(GetBinaryOpNameCode<T, false, false>(), {inputTile1, inputTile2}, {resultTile});
        }
        return;
    }
    for (int i = 0; i < result->shape[cur]; i += tileShape.V(cur)) {
        resultTileInfo.offset[cur] = i;
        resultTileInfo.shape[cur] = std::min(result->shape[cur] - resultTileInfo.offset[cur], tileShape.V(cur));
        input1.tileInfo.offset[cur] = i % input1.tensor->shape[cur];
        input1.tileInfo.shape[cur] =
            std::min(input1.tensor->shape[cur] - input1.tileInfo.offset[cur], tileShape.V(cur));
        input2.tileInfo.offset[cur] = i % input2.tensor->shape[cur];
        input2.tileInfo.shape[cur] =
            std::min(input2.tensor->shape[cur] - input2.tileInfo.offset[cur], tileShape.V(cur));
        TiledBinaryOperation<T>(function, tileShape, cur + 1, input1, input2, result, resultTileInfo, withBrc);
    }
}

template <BinaryOpType T>
void TiledBinaryOperation(Function &function, const TileShape &tileShape, LogicalTensorPtr operand1,
    LogicalTensorPtr operand2, const LogicalTensorPtr &result) {
    CheckBinOpOperandsValid(operand1, operand2);
    bool withBrc = CallBrcBinOp(operand1, operand2) && ConfigManager::Instance().GetOperationConfig("FORCE_COMBINE_AXIS", false);
    if (!withBrc) {
        if (operand1->shape != result->shape) {
            auto targetShape = result->shape;
            auto tmp = std::make_shared<LogicalTensor>(function, operand1->Datatype(), targetShape);
            Expand(function, tileShape, operand1, tmp);
            operand1 = tmp;
        }

        if (operand2->shape != result->shape) {
            auto targetShape = result->shape;
            auto tmp = std::make_shared<LogicalTensor>(function, operand2->Datatype(), targetShape);
            Expand(function, tileShape, operand2, tmp);
            operand2 = tmp;
        }
    }

    TileInfo tileInfo1(result->shape.size(), result->offset.size());
    TileInfo tileInfo2(result->shape.size(), result->offset.size());
    TileInfo resultTileInfo(result->shape.size(), result->offset.size());
    auto input1 = Input{operand1, tileInfo1};
    auto input2 = Input{operand2, tileInfo2};
    TiledBinaryOperation<T>(function, tileShape, 0, input1, input2, result, resultTileInfo, withBrc);
}

const std::string ACC_A_MUL_B = OP_ATTR_PREFIX + "atomic_add";
const std::string A_MUL_B_NZ_ATTR = OP_ATTR_PREFIX + "matmul_nz_attr";
const std::string A_MUL_B_ACT_M = OP_ATTR_PREFIX + "act_m";
const std::string A_MUL_B_ACT_K = OP_ATTR_PREFIX + "act_k";
const std::string A_MUL_B_ACT_N = OP_ATTR_PREFIX + "act_n";

std::vector<int> BinaryOperationResultShape(
    LogicalTensorPtr operand1, LogicalTensorPtr operand2) {
    std::vector<int> resultShape(operand1->shape.size());
    for (size_t i = 0; i < resultShape.size(); i++) {
        resultShape[i] = std::max(operand1->shape[i], operand2->shape[i]);
    }
    return resultShape;
}

template <UnaryOpType T>
void TiledUnaryOperation(Function &function, const TileShape &tileShape, size_t cur, Input &input,
    const LogicalTensorPtr &result) {
    if (cur == input.tensor->shape.size()) {
        auto tile = input.tensor->View(function, input.tileInfo.shape, input.tileInfo.offset);
        auto resultTile = result->View(function, input.tileInfo.shape, input.tileInfo.offset);
        function.AddOperation(GetUnaryOpNameCode<T>(), {tile}, {resultTile});
        return;
    }
    for (int i = 0; i < input.tensor->shape[cur]; i += tileShape.V(cur)) {
        input.tileInfo.shape[cur] = std::min(input.tensor->shape[cur] - i, tileShape.V(cur));
        input.tileInfo.offset[cur] = i;
        TiledUnaryOperation<T>(function, tileShape, cur + 1, input, result);
    }
}

template <UnaryOpType T>
void TiledUnaryOperation(Function &function, const TileShape &tileShape,
    const LogicalTensorPtr &operand, const LogicalTensorPtr &result) {
    assert(operand->shape.size() == operand->offset.size());

    TileInfo tileInfo(result->shape.size(), result->offset.size());
    auto input = Input{operand, tileInfo};
    TiledUnaryOperation<T>(function, tileShape, 0, input, result);
}

void TiledAssemble(Function &function, const TileShape &tileShape, size_t cur, Input &input,
    const std::shared_ptr<LogicalTensor> &result, AssembleOpAttribute *attr) {
    if (cur == input.tensor->shape.size()) {
        auto tile = input.tensor->View(function, input.tileInfo.shape, input.tileInfo.offset);
        auto &assemble = function.AddOperation(Opcode::OP_ASSEMBLE, {tile}, {result});
        assemble.SetAttr("NeedCopy", true);
        auto &toDynOffset = attr->GetToDynOffset();
        std::vector<SymbolicScalar> newDynOffset;
        newDynOffset.resize(toDynOffset.size());
        for (size_t i = 0; i < toDynOffset.size(); ++i) {
            newDynOffset[i] = toDynOffset[i] + SymbolicScalar(input.tileInfo.offset[i]);
        }
        assemble.iOperand[0]->SetMemoryTypeOriginal(MemoryType::MEM_UB);
        assemble.SetOpAttribute(std::make_shared<AssembleOpAttribute>(MemoryType::MEM_UB, input.tileInfo.offset, newDynOffset));
        return;
    }
    for (int i = 0; i < input.tensor->shape[cur]; i += tileShape.V(cur)) {
        input.tileInfo.shape[cur] = std::min(input.tensor->shape[cur] - i, tileShape.V(cur));
        input.tileInfo.offset[cur] = i;
        TiledAssemble(function, tileShape, cur + 1, input, result, attr);
    }
}

void TiledAssemble(Function &function, const TileShape &tileShape,
    const std::shared_ptr<LogicalTensor> &operand, const std::shared_ptr<LogicalTensor> &result,
    AssembleOpAttribute *attr) {
    assert(operand->shape.size() == operand->offset.size());

    TileInfo tileInfo(result->shape.size(), result->offset.size());
    auto input = Input{operand, tileInfo};
    TiledAssemble(function, tileShape, 0, input, result, attr);
}

template <BinaryOpType T>
LogicalTensorPtr TensorBinaryOperation(Function &function, const Tensor &operand1,
    const Tensor &operand2) {
    auto oprandT1 = operand1.GetStorage();
    auto oprandT2 = operand2.GetStorage();
    if(oprandT1->shape.size() != oprandT2->shape.size()) {
        std::vector<int> broadCastShape = GetBroadCastShape(oprandT1, oprandT2);
        oprandT1 = BinaryOperationBroadCast(oprandT1, broadCastShape);
        oprandT2 = BinaryOperationBroadCast(oprandT2, broadCastShape);
    }
    CheckBinOpOperandsValid(oprandT1, oprandT2);

    std::vector<SymbolicScalar> resultValidShape;
    std::vector<int> resultShape = BinaryOperationResultShape(oprandT1, oprandT2);
        if ((!oprandT1->GetDynValidShape().empty()) && (!oprandT2->GetDynValidShape().empty())) {
            for (size_t i = 0; i < resultShape.size(); ++i) {
                if (resultShape[i] == oprandT1->shape[i]) {
                    resultValidShape.push_back(operand1->GetDynValidShape()[i]);
                } else {
                    resultValidShape.push_back(operand2->GetDynValidShape()[i]);
                }
        }
    }
    auto result = std::make_shared<LogicalTensor>(function, oprandT1->Datatype(), resultShape, resultValidShape);
    function.AddOperation(GetBinaryOpNameCode<T>(), {oprandT1, oprandT2}, {result});
    return result;
}

template <BinaryOpType T>
LogicalTensorPtr TensorBinaryOperationScalar(Function &function, LogicalTensorPtr operand1, const Element &value) {
    auto result = std::make_shared<LogicalTensor>(function, operand1->Datatype(), operand1->shape, operand1->GetDynValidShape());
    auto &op = function.AddOperation(GetBinaryOpNameCode<T, true>(), {operand1}, {result});
    op.SetAttribute(OpAttributeKey::scalar, value);
    return result;
}

template <BinaryOpType T>
void TiledBinaryOperationScalar(Function &function, const TileShape &tileShape, size_t cur, Input &input1,
    Element &value, const LogicalTensorPtr &result, TileInfo &resultTileInfo, bool reverseOperand) {
    if (cur == input1.tensor->shape.size()) {
        auto inputTile1 = input1.tensor->View(function, input1.tileInfo.shape, input1.tileInfo.offset);
        auto resultTile = result->View(function, resultTileInfo.shape, resultTileInfo.offset);
        // 确认接口
        auto &op = function.AddOperation(GetBinaryOpNameCode<T, true>(), {inputTile1}, {resultTile});
        op.SetAttribute(OpAttributeKey::scalar, value);
        op.SetAttribute(OP_ATTR_PREFIX + "reverseOperand", reverseOperand);
        return;
    }
    for (int i = 0; i < result->shape[cur]; i += tileShape.V(cur)) {
        resultTileInfo.offset[cur] = i;
        resultTileInfo.shape[cur] = std::min(result->shape[cur] - resultTileInfo.offset[cur], tileShape.V(cur));
        input1.tileInfo.offset[cur] = i % input1.tensor->shape[cur];
        input1.tileInfo.shape[cur] =
            std::min(input1.tensor->shape[cur] - input1.tileInfo.offset[cur], tileShape.V(cur));

        TiledBinaryOperationScalar<T>(function, tileShape, cur + 1, input1, value, result, resultTileInfo, reverseOperand);
    }
}

template <BinaryOpType T>
void TiledBinaryOperationScalar(Function &function, const TileShape &tileShape,
    LogicalTensorPtr operand1, Element value, const LogicalTensorPtr &result, bool reverseOperand=false) {
    TileInfo tileInfo1(result->shape.size(), result->offset.size());
    TileInfo resultTileInfo(result->shape.size(), result->offset.size());
    auto input1 = Input{operand1, tileInfo1};
    TiledBinaryOperationScalar<T>(function, tileShape, 0, input1, value, result, resultTileInfo, reverseOperand);
}

template <BinaryOpType T>
LogicalTensorPtr TensorBinaryOperationAllScalar(Function &function,
    const Tensor &operand1, const Element &value, bool reverseOperand) {
    auto result = std::make_shared<LogicalTensor>(function, operand1->Datatype(), operand1->shape);
    auto &op = function.AddOperation(GetBinaryOpNameCode<T, true>(), {operand1.GetStorage()}, {result});
    op.SetAttribute(OpAttributeKey::scalar, value);
    op.SetAttribute(OP_ATTR_PREFIX + "reverseOperand", reverseOperand);
    return result;
}

template <BinaryOpType T>
void TiledBinaryOperationAllScalar(Function &function, const TileShape &tileShape, size_t cur, Input &input1,
    Element &value, const LogicalTensorPtr &result, TileInfo &resultTileInfo, bool reverseOperand) {
    if (cur == input1.tensor->shape.size()) {
        auto inputTile1 = input1.tensor->View(function, input1.tileInfo.shape, input1.tileInfo.offset);
        auto resultTile = result->View(function, resultTileInfo.shape, resultTileInfo.offset);
        // 确认接口
        auto &op = function.AddOperation(GetBinaryOpNameCode<T, true>(), {inputTile1}, {resultTile});
        op.SetAttribute(OpAttributeKey::scalar, value);
        op.SetAttribute(OP_ATTR_PREFIX + "reverseOperand", reverseOperand);
        return;
    }
    for (int i = 0; i < result->shape[cur]; i += tileShape.V(cur)) {
        resultTileInfo.offset[cur] = i;
        resultTileInfo.shape[cur] = std::min(result->shape[cur] - resultTileInfo.offset[cur], tileShape.V(cur));
        input1.tileInfo.offset[cur] = i % input1.tensor->shape[cur];
        input1.tileInfo.shape[cur] =
            std::min(input1.tensor->shape[cur] - input1.tileInfo.offset[cur], tileShape.V(cur));

        TiledBinaryOperationScalar<T>(function, tileShape, cur + 1, input1, value, result, resultTileInfo, reverseOperand);
    }
}

template <BinaryOpType T>
void TiledBinaryOperationAllScalar(Function &function, const TileShape &tileShape,
    LogicalTensorPtr operand1, Element value, const LogicalTensorPtr &result, bool reverseOperand) {
    TileInfo tileInfo1(result->shape.size(), result->offset.size());
    TileInfo resultTileInfo(result->shape.size(), result->offset.size());
    auto input1 = Input{operand1, tileInfo1};
    TiledBinaryOperationAllScalar<T>(function, tileShape, 0, input1, value, result, resultTileInfo, reverseOperand);
}

template <BinaryOpType T>
LogicalTensorPtr TensorBinaryOperationAllScalar(Function &function,
    const Tensor &operand1, const Tensor &operand2) {
    auto result = std::make_shared<LogicalTensor>(function, operand1->Datatype(), operand1->shape);
    function.AddOperation(GetBinaryOpNameCode<T, false>(), {operand1.GetStorage(), operand2.GetStorage()}, {result});
    return result;
}

template <BinaryOpType T>
void TiledBinaryOperationAllScalar(Function &function, const TileShape &tileShape, size_t cur, Input &input1,
    Input &input2, const LogicalTensorPtr &result, TileInfo &resultTileInfo) {
    if (cur == input1.tensor->shape.size()) {
        auto inputTile1 = input1.tensor->View(function, input1.tileInfo.shape, input1.tileInfo.offset);
        auto inputTile2 = input2.tensor->View(function, input2.tileInfo.shape, input2.tileInfo.offset);
        auto resultTile = result->View(function, resultTileInfo.shape, resultTileInfo.offset);
        function.AddOperation(GetBinaryOpNameCode<T, false>(), {inputTile1, inputTile2}, {resultTile});
        return;
    }
    for (int i = 0; i < result->shape[cur]; i += tileShape.V(cur)) {
        resultTileInfo.offset[cur] = i;
        resultTileInfo.shape[cur] = std::min(result->shape[cur] - resultTileInfo.offset[cur], tileShape.V(cur));
        input1.tileInfo.offset[cur] = i % input1.tensor->shape[cur];
        input1.tileInfo.shape[cur] =
            std::min(input1.tensor->shape[cur] - input1.tileInfo.offset[cur], tileShape.V(cur));
        input2.tileInfo.offset[cur] = i % input2.tensor->shape[cur];
        input2.tileInfo.shape[cur] =
            std::min(input2.tensor->shape[cur] - input2.tileInfo.offset[cur], tileShape.V(cur));
        TiledBinaryOperationAllScalar<T>(function, tileShape, cur + 1, input1, input2, result, resultTileInfo);
    }
}

template <BinaryOpType T>
void TiledBinaryOperationAllScalar(Function &function, const TileShape &tileShape, LogicalTensorPtr operand1,
    LogicalTensorPtr operand2, const LogicalTensorPtr &result) {
    CheckBinOpOperandsValid(operand1, operand2);

    if (operand1->shape != result->shape) {
        auto targetShape = result->shape;
        auto tmp = std::make_shared<LogicalTensor>(function, operand1->Datatype(), targetShape);
        Expand(function, tileShape, operand1, tmp);
        operand1 = tmp;
    }

    if (operand2->shape != result->shape) {
        auto targetShape = result->shape;
        auto tmp = std::make_shared<LogicalTensor>(function, operand2->Datatype(), targetShape);
        Expand(function, tileShape, operand2, tmp);
        operand2 = tmp;
    }

    TileInfo tileInfo1(result->shape.size(), result->offset.size());
    TileInfo tileInfo2(result->shape.size(), result->offset.size());
    TileInfo resultTileInfo(result->shape.size(), result->offset.size());
    auto input1 = Input{operand1, tileInfo1};
    auto input2 = Input{operand2, tileInfo2};
    TiledBinaryOperationAllScalar<T>(function, tileShape, 0, input1, input2, result, resultTileInfo);
}

void TileReduceNew(Function &function, const TileShape &tileShape, const std::string &op, ReduceType reduceType,
    const LogicalTensorPtr &in, const LogicalTensorPtr &result, int axis = -1) {
    axis = axis < 0 ? in->shape.size() + axis : axis;
    std::vector<int> tileAshape = in->shape;
    std::vector<int> tileBshape = in->shape;
    std::vector<int> regShape = in->shape;
    std::vector<int> regAshape = in->shape;
    std::vector<int> regBshape = in->shape;
    std::vector<int> regOffset(regShape.size(), 0);
    std::vector<int> tileAoffset(regShape.size(), 0);
    std::vector<int> tileBoffset(regShape.size(), 0);

    std::vector<int> remainderShape = in->shape;
    std::vector<int> remainderOffset(remainderShape.size(), 0);

    auto opNew = op;
    if (opNew == "MAX_COMBINE_AXIS") {
        opNew = "MAX";
    }
    if (opNew == "SUM_COMBINE_AXIS") {
        opNew = "SUM";
    }

    auto source = std::make_shared<LogicalTensor>(function, in->tensor, in->offset, in->shape, in->GetDynValidShape(), in->nodetype);

    int width = (source->shape[axis] + tileShape.V(axis) - 1) / tileShape.V(axis) * tileShape.V(axis); // 向上对齐
    int padSize = width - source->shape[axis];
    int remainder = 0;

    int p2width = tileShape.V(axis);
    while (width >= p2width) {
        p2width = p2width << 1;
    }
    p2width = p2width >> 1;

    remainder = width - p2width;
    remainderShape[axis] = remainder;
    remainderOffset[axis] = p2width;

    width = p2width;

    while (width >= NUM2 * tileShape.V(axis)) // hierarchically pair wise reduce to a
    // single TILE_SHAPE1
    {
        width = width >> 1;

        tileAshape[axis] = width;
        tileBshape[axis] = std::min(width, source->shape[axis] - width); // 带tail的部分
        tileBoffset[axis] = width;

        auto tileA = source->View(function, tileAshape, tileAoffset);
        auto tileB = source->View(function, tileBshape, tileBoffset);

        auto resultA = std::make_shared<LogicalTensor>(
            function, in->Datatype(), reduceType == npu::tile_fwk::ReduceType::EXPAND ? result->shape : source->shape, reduceType == npu::tile_fwk::ReduceType::EXPAND ? result->GetDynValidShape() : source->GetDynValidShape());
        for (int j = 0; j < width; j += tileShape.V(axis)) {
            regAshape[axis] = tileShape.V(axis);
            regBshape[axis] = std::min(tileShape.V(axis), tileB->shape[axis] - j); // 带tail的部分
            regOffset[axis] = j;

            auto regA = tileA->View(function, regAshape, regOffset);
            auto regB = tileB->View(function, regBshape, regOffset);
            auto regResult = resultA->View(function, regAshape, regOffset);
            function.AddOperation("TILE_PAIR" + opNew, {regA, regB}, {regResult});
        }

        if (remainder < width) {
            source = resultA;
            continue;
        }

        if ((remainderShape[axis] + remainderOffset[axis] > in->shape[axis])) {
            remainderShape[axis] = remainderShape[axis] - padSize;
        }

        auto tileRemainder = in->View(function, remainderShape, remainderOffset);
        auto resultAnext = std::make_shared<LogicalTensor>(function, in->Datatype(), resultA->shape, resultA->GetDynValidShape());
        for (int j = 0; j < width; j += tileShape.V(axis)) {
            regAshape[axis] = tileShape.V(axis);
            regBshape[axis] = std::min(tileShape.V(axis), tileRemainder->shape[axis] - j); // 带tail的部分
            regOffset[axis] = j;

            auto regA = resultA->View(function, regAshape, regOffset);
            auto regB = tileRemainder->View(function, regBshape, regOffset);
            auto regResult = resultAnext->View(function, regAshape, regOffset);
            function.AddOperation("TILE_PAIR" + opNew, {regA, regB}, {regResult});
        }
        remainder -= width;
        remainderOffset[axis] += width;
        remainderShape[axis] -= width;

        source = resultAnext;
    }

    // reduce to a single TILE_SHAPE1
    regShape[axis] = std::min(in->shape[axis], tileShape.V(axis));
    regOffset[axis] = 0;

    auto temp =
        std::make_shared<LogicalTensor>(function, in->Datatype(), reduceType == npu::tile_fwk::ReduceType::EXPAND ? result->shape : source->shape, reduceType == npu::tile_fwk::ReduceType::EXPAND ? result->GetDynValidShape() : source->GetDynValidShape());
    auto sourceReg = source->View(function, regShape, regOffset);
    switch (reduceType) {
        case npu::tile_fwk::ReduceType::NORMAL: {
            auto resultReg = result->View(function, regShape, regOffset);
            // now the max is in resultReg
            function.AddOperation("TILE_ROW" + op, {sourceReg}, {resultReg});
            break;
        }
        case npu::tile_fwk::ReduceType::EXPAND: {
            auto resultReg = temp->View(function, regShape, regOffset);
            // now the max is in resultReg
            function.AddOperation("TILE_ROWEXP" + op, {sourceReg}, {resultReg});

            auto resultReg1 = temp->View(function, regShape, regOffset);

            for (int j = 0; j < result->shape[1]; j += tileShape.V(1)) // duplicate to fill result tensor
            {
                regShape[0] = in->shape[0];
                regShape[1] = tileShape.V(1);

                regOffset[0] = 0;
                regOffset[1] = j;

                resultReg = result->View(function, regShape, regOffset);
                function.AddOperation("TILE_REGISTER_COPY", {resultReg1}, {resultReg});
            }
            break;
        }
        case npu::tile_fwk::ReduceType::SINGLE: {
            std::vector<int> tmpShape = {1, static_cast<int>(BLOCK_SIZE / BytesOf(in->Datatype()))};
            if (axis > 0) {
                tmpShape[0] = sourceReg->shape[axis - 1];
            }
            if (op == "SUM") {
                if (static_cast<size_t>(sourceReg->shape[axis]) <= REPEAT_BYTE / BytesOf(in->Datatype())) {
                    tmpShape[0] = 1;
                } else if (static_cast<size_t>(sourceReg->shape[axis]) <=
                           NUM2 * REPEAT_BYTE / BytesOf(in->Datatype())) {
                    tmpShape[1] = REPEAT_BYTE / BytesOf(in->Datatype());
                } else {
                    tmpShape[1] = (((sourceReg->shape[axis] * BytesOf(in->Datatype())) / REPEAT_BYTE) / NUM2) *
                                  REPEAT_BYTE / BytesOf(in->Datatype());
                }
            } else {
                tmpShape[1] = REPEAT_BLOCK_NUM;
            }
            if ((sourceReg->shape[0] % BLOCK_NUM == 0) &&
                ((tileShape.V(axis) == LEN1024 && sourceReg->shape[0] * NUM_VALUE_16 <= MAX_REPEAT) ||
                    (tileShape.V(axis) == LEN512 && sourceReg->shape[0] * BLOCK_NUM <= MAX_REPEAT))) {
                tmpShape[1] = tileShape.V(axis) / BLOCK_NUM;
            }
            unsigned tmpBufSize = tmpShape[0] * tmpShape[1] * BytesOf(in->Datatype());
            // Set the threshold of tmp buffer size as 32KB
            assert(tmpBufSize <= MAX_TMP_BUF_SHAPE);
            auto tempTensor = std::make_shared<LogicalTensor>(function, in->Datatype(), tmpShape);
            auto &newOp = function.AddOperation("TILE_ROW" + op + "_SINGLE", {sourceReg}, {result, tempTensor});
            newOp.SetAttribute(OP_ATTR_PREFIX + "AXIS", axis);
            break;
        }
        default:
            break;
    }
}


void TiledReduceExpand(Function &function, const TileShape &tileShape, const std::string &op,
    const LogicalTensorPtr &operand, const LogicalTensorPtr &result) {
    ASSERT(op == "MAX" || op == "SUM");
    assert(operand->shape.size() == operand->offset.size());

    // 目前只支持2维操作
    if (operand->shape.size() != 2) {
        assert(false && "unsupported dimension");
    }

    TileInfo tileInfo({tileShape.V(0), operand->shape[1]}, std::vector<int>(operand->offset.size()));

    for (int i = 0; i < operand->shape[0]; i += tileShape.V(0)) {
        tileInfo.offset[0] = i;
        auto inputTile = operand->View(function, tileInfo.shape, tileInfo.offset);
        auto resultTile = result->View(function, tileInfo.shape, tileInfo.offset);
        TileReduceNew(function, tileShape, op, npu::tile_fwk::ReduceType::EXPAND, inputTile, resultTile);
    }
}

[[maybe_unused]] void TensorReduceExpand(Function &function, const std::string &op,
    const LogicalTensorPtr &operand, const LogicalTensorPtr &result) {
    ASSERT(op == "MAX" || op == "SUM");
    assert(operand->shape.size() == operand->offset.size());
    function.AddOperation(op == "MAX" ? Opcode::OP_ROWEXPMAX : Opcode::OP_ROWEXPSUM, {operand}, {result});
}

void ReduceSingle(size_t cur, const std::string &op, Input &input, const LogicalTensorPtr result,
    TileInfo &resultTileInfo, int axis, Function &function, const TileShape &tileShape) {
    if (cur == static_cast<size_t>(axis) && static_cast<size_t>(axis) == input.tileInfo.shape.size() - 1) {
        auto inputTile = input.tensor->View(function, input.tileInfo.shape, input.tileInfo.offset);
        auto resultTile = result->View(function, resultTileInfo.shape, resultTileInfo.offset);
        if (static_cast<size_t>(axis) <  input.tileInfo.shape.size() - 1) {
            auto &newOp  = function.AddOperation(Opcode::OP_ROWSUMLINE, {inputTile}, {resultTile});
            newOp .SetAttribute(OP_ATTR_PREFIX + "AXIS", axis);
        } else {
            TileReduceNew(function, tileShape, op, npu::tile_fwk::ReduceType::SINGLE, inputTile, resultTile, axis);
        }
        return;
    } else if (cur == input.tileInfo.shape.size() && static_cast<size_t>(axis) < input.tileInfo.shape.size() - 1){
        auto inputTile = input.tensor->View(function, input.tileInfo.shape, input.tileInfo.offset);
        auto resultTile = result->View(function, resultTileInfo.shape, resultTileInfo.offset);
        auto &newOp = function.AddOperation(Opcode::OP_ROWSUMLINE, {inputTile}, {resultTile});
        newOp.SetAttribute(OP_ATTR_PREFIX + "AXIS", axis);
        return;
    }
    for (int i = 0; i < result->shape[cur]; i += tileShape.V(cur)) {
        resultTileInfo.offset[cur] = i;
        resultTileInfo.shape[cur] = std::min(result->shape[cur] - resultTileInfo.offset[cur], tileShape.V(cur));
        input.tileInfo.offset[cur] = i % input.tensor->shape[cur];
        input.tileInfo.shape[cur] = std::min(input.tensor->shape[cur] - input.tileInfo.offset[cur], tileShape.V(cur));
        ReduceSingle(cur + 1, op, input, result, resultTileInfo, axis, function, tileShape);
    }
}

void TiledReduceSingle(Function &function, const TileShape &tileShape, const std::string &op,
    const LogicalTensorPtr &operand, const LogicalTensorPtr &result, int axis = -1) {
    ASSERT(op == "MAX" || op == "SUM" || op == "MAX_COMBINE_AXIS" || op == "SUM_COMBINE_AXIS");
    assert(operand->shape.size() == operand->offset.size());

    if (axis < 0) {
        axis = operand->shape.size() + axis;
    }

    // for loops before reduce axis
    TileInfo tileInfo(operand->shape, operand->offset);
    TileInfo resultTileInfo(result->shape, result->offset);
    auto input = Input{operand, tileInfo};
    ReduceSingle(0, op, input, result, resultTileInfo, axis, function, tileShape);
}

void TensorReduceExpand(Function &function, const std::string &op,
    const Tensor &operand, const Tensor &result) {
    ASSERT(op == "MAX" || op == "SUM");
    assert(operand->shape.size() == operand->offset.size());
    function.AddOperation(op == "MAX" ? Opcode::OP_ROWEXPMAX : Opcode::OP_ROWEXPSUM, {operand.GetStorage()}, {result.GetStorage()});
    return;
}

[[maybe_unused]] Tensor ReduceExpand(const std::string &op, const Tensor &operand) {
    Tensor result(operand->tensor->datatype, operand->shape);
    assert(operand->shape.size() == operand->offset.size());
    Program::GetInstance().AddOperation("ROW_" + op + "_EXPAND", {operand.GetStorage()}, {result.GetStorage()});
    return result;
}

[[maybe_unused]] void TensorReduceSingle(Function &function, const std::string &op,
    const Tensor &operand, Tensor &result, int axis) {
    ASSERT(op == "MAX" || op == "SUM" || op == "MAX_COMBINE_AXIS" || op == "SUM_COMBINE_AXIS");
    assert(operand->shape.size() == operand->offset.size());
    auto opCode = Opcode::OP_ROWMAX_SINGLE;
    if (op == "MAX") {
        opCode = Opcode::OP_ROWMAX_SINGLE;
    } else if (op == "SUM") {
        opCode = Opcode::OP_ROWSUM_SINGLE;
    } else if (op == "MAX_COMBINE_AXIS") {
        opCode = Opcode::OP_ROWMAX_COMBINE_AXIS_SINGLE;
    } else {  // SUM_COMBINE_AXIS
        opCode = Opcode::OP_ROWSUM_COMBINE_AXIS_SINGLE;
    }

    if (!operand->GetDynValidShape().empty()) {
        std::vector<SymbolicScalar> outValidShape;
        for (auto shape : operand->GetDynValidShape()) {
            outValidShape.push_back(shape);
        }
        outValidShape[axis] = SymbolicScalar(1);
        result->UpdateDynValidShape(outValidShape);
    }

    auto &newOp = function.AddOperation(opCode, {operand.GetStorage()}, {result.GetStorage()});
    newOp.SetAttribute(OP_ATTR_PREFIX + "AXIS", static_cast<int>(axis));
    return;
}

[[maybe_unused]] Tensor ReduceSingle(const std::string &op, const Tensor &operand) {
    Tensor result(operand->tensor->datatype, {operand->shape[0], 1});
    assert(operand->shape.size() == operand->offset.size());
    Program::GetInstance().AddOperation("REDUCE_" + op + "_SINGLE", {operand.GetStorage()}, {result.GetStorage()});
    return result;
}

} // namespace

namespace npu::tile_fwk {
#define CALL(n, ...) Tensor##n(__VA_ARGS__)
#define RETURN_CALL(n, ...) return Tensor##n(__VA_ARGS__)
constexpr int NCHW_DIM_NUM = 4;
constexpr int NC1HWC0_DIM_NUM = 5;
constexpr int STRIDE_DIM_NUM = 2;
constexpr int PADS_DIM_NUM = 4;
constexpr int WEIGHT_DIM_NUM = 4;
constexpr int BIAS_DIM_NUM = 1;
constexpr int SMALL_CHANNEL_4 = 4;
constexpr int SMALL_CHANNEL_8 = 8;
constexpr int SMALL_CHANNEL_16 = 16;

static void MaybeAppendGetTensorData(Operation *op, const std::vector<SymbolicScalar> &offset) {
    (void)op;
    auto currDynFunc = Program::GetInstance().GetCurrentDynamicFunction();
    if (currDynFunc == nullptr) {
        return;
    }

    auto currDynAttr = currDynFunc->GetDyndevAttribute();
    auto getTensorDataDict = GetTensorDataDict(offset);
    for (auto &[getTensorDataIndex, _] : getTensorDataDict) {
        (void)_;
        ASSERT(currDynAttr->getTensorDataDict.count(getTensorDataIndex)) << "Invalid index!";
        auto import = *currDynAttr->getTensorDataDict[getTensorDataIndex].outcastTensor;
        // The goal of this view is to add the tensor as incast.
        std::vector<int> importShape(import.GetShape().size(), 1);
        std::vector<int> importOffset(import.GetShape().size(), 0);
        auto importLoad = View(import, importShape, importOffset);
        auto importLoadOp = *importLoad->GetProducers().begin();
        importLoadOp->SetAttr<int>(OP_EMUOP_PREFIX + "GetTensorData_tensor_to_scalar", getTensorDataIndex);
    }
}

Tensor View(const Tensor &operand, const std::vector<int> &shapes, const std::vector<int> &offsets) {
    DECLARE_TRACER();
    Tensor result(operand->Datatype(), shapes, "View_" + operand->GetRawTensor()->GetSymbol(), operand->nodetype, operand->tensorfmt);
    auto &op = Program::GetInstance().GetCurrentFunction()->AddOperation(
        Opcode::OP_VIEW, {operand.GetStorage()}, {result.GetStorage()});
    auto validShape = GetViewValidShape(operand->GetDynValidShape(), offsets, {}, shapes);
    result->UpdateDynValidShape(validShape);
    auto newOffsets = SymbolicScalar::FromConcrete(offsets);
    op.SetOpAttribute(std::make_shared<ViewOpAttribute>(offsets, newOffsets, validShape));
    return result;
}

Tensor DView(const Tensor &operand, const std::vector<int> &shapes, const std::vector<SymbolicScalar> &newOffsets) {
    DECLARE_TRACER();
    Tensor result(operand->Datatype(), shapes, "DView_" + operand->GetRawTensor()->GetSymbol(), operand->nodetype, operand->tensorfmt);
    result->UpdateDynValidShape(SymbolicScalar::FromConcrete(shapes));
    auto &op = Program::GetInstance().GetCurrentFunction()->AddOperation(
        Opcode::OP_VIEW, {operand.GetStorage()}, {result.GetStorage()});
    auto validShape = GetViewValidShape(operand->GetDynValidShape(), {}, newOffsets, shapes);
    result->UpdateDynValidShape(validShape);
    std::vector<int> newOffsetsConcrete = SymbolicScalar::Concrete(newOffsets, 0);
    op.SetOpAttribute(std::make_shared<ViewOpAttribute>(newOffsetsConcrete, newOffsets, validShape));
    MaybeAppendGetTensorData(&op, newOffsets);
    return result;
}

Tensor DViewPad(const Tensor &operand, const std::vector<int> &shapes,
    const std::vector<SymbolicScalar> &newValidShapes, const std::vector<SymbolicScalar> &newOffsets) {
    DECLARE_TRACER();
    Tensor result(operand->Datatype(), shapes, "DViewPad_" + operand->GetRawTensor()->GetSymbol(), operand->nodetype, operand->tensorfmt);
    auto &op = Program::GetInstance().GetCurrentFunction()->AddOperation(
        Opcode::OP_VIEW, {operand.GetStorage()}, {result.GetStorage()});
    std::vector<int> newOffsetsConcrete = SymbolicScalar::Concrete(newOffsets, 0);
    op.SetOpAttribute(std::make_shared<ViewOpAttribute>(newOffsetsConcrete, newOffsets, newValidShapes));
    result->UpdateDynValidShape(newValidShapes);
    MaybeAppendGetTensorData(&op, newOffsets);
    return result;
}

void TiledGatherOperation(Function &function, const TileShape &tileShape, size_t cur, Input &paramsInput,
    Input &indicesInput, int axis, const LogicalTensorPtr &result, TileInfo &resultTileInfo) {
    if (cur == result->shape.size()) {
        // add Operation
        auto paramsTile = paramsInput.tensor->View(function, paramsInput.tileInfo.shape, paramsInput.tileInfo.offset);
        auto indicesTile =
            indicesInput.tensor->View(function, indicesInput.tileInfo.shape, indicesInput.tileInfo.offset);
        auto resultTile = result->View(function, resultTileInfo.shape, resultTileInfo.offset);
        auto &op = function.AddOperation(Opcode::OP_GATHER, {paramsTile, indicesTile}, {resultTile});
        op.SetAttribute(OP_ATTR_PREFIX + "axis", axis);

        return;
    }

    // 按照resultShape进行切分
    int tmpTile = tileShape.V(cur);
    for (int i = 0; i < result->shape[cur]; i += tmpTile) {
        if (cur < static_cast<size_t>(axis)) {
            // 在result中gather轴的外层轴
            paramsInput.tileInfo.offset[cur] = i % paramsInput.tensor->shape[cur];
            paramsInput.tileInfo.shape[cur] =
                std::min(paramsInput.tensor->shape[cur] - paramsInput.tileInfo.offset[cur], tmpTile);
        } else if (cur >= static_cast<size_t>(axis) && (cur < static_cast<size_t>(axis) + indicesInput.tensor->shape.size())) {
            // 当前属于indices的gather轴
            // params[axis]不切
            paramsInput.tileInfo.offset[axis] = 0;
            paramsInput.tileInfo.shape[axis] = paramsInput.tensor->shape[axis];
            // 处理indices的tileInfo
            indicesInput.tileInfo.offset[cur] = i % indicesInput.tensor->shape[cur];
            indicesInput.tileInfo.shape[cur] =
                std::min(indicesInput.tensor->shape[cur] - indicesInput.tileInfo.offset[cur], tmpTile);
        } else {
            // 在result中gather轴的内层轴
            int paramHighAxis = cur - indicesInput.tensor->shape.size() + 1;
            paramsInput.tileInfo.offset[paramHighAxis] = i % paramsInput.tensor->shape[paramHighAxis];
            paramsInput.tileInfo.shape[paramHighAxis] = std::min(
                paramsInput.tensor->shape[paramHighAxis] - paramsInput.tileInfo.offset[paramHighAxis], tmpTile);
        }

        resultTileInfo.offset[cur] = i;
        resultTileInfo.shape[cur] = std::min(result->shape[cur] - resultTileInfo.offset[cur], tmpTile);
        TiledGatherOperation(function, tileShape, cur + 1, paramsInput, indicesInput, axis, result, resultTileInfo);
    }
}

std::vector<int> GatherOperationResultShape(
    LogicalTensorPtr params, LogicalTensorPtr indices, int axis) {
    assert(params->shape.size() == params->offset.size());
    assert(indices->shape.size() == indices->offset.size());
    int paramsRank = params->shape.size();
    if (axis < 0) {
        axis = axis + paramsRank;
    }
    assert(axis == paramsRank - 2); // 当前支持-2轴

    // result shape: params.shape[:aixs] + indices.shape + params.shape[axis+1:]
    std::vector<int> resultShape = params->shape;
    resultShape.erase(resultShape.begin() + axis);
    resultShape.insert(resultShape.begin() + axis, indices->shape.begin(), indices->shape.end());

    return resultShape;
}

void TiledGatherOperation(Function &function, const TileShape &tileShape,
    const LogicalTensorPtr &params, const LogicalTensorPtr &indices, int axis,
    const LogicalTensorPtr &result) {
    // Check Operands Valid
    std::vector<int> expectedShape = GatherOperationResultShape(params, indices, axis);
    assert(result->shape.size() == expectedShape.size());
    assert(result->shape.size() == result->offset.size());
    assert(params->shape.size() == params->offset.size());
    assert(indices->shape.size() == indices->offset.size());

    TileInfo paramsTileInfo(params->shape.size(), params->offset.size());
    TileInfo indicesTileInfo(indices->shape.size(), indices->offset.size());
    TileInfo resultTileInfo(result->shape.size(), result->offset.size());
    auto paramsInput = Input{params, paramsTileInfo};
    auto indicesInput = Input{indices, indicesTileInfo};
    TiledGatherOperation(function, tileShape, 0, paramsInput, indicesInput, axis, result, resultTileInfo);
}

LogicalTensorPtr TiledGatherOperation(Function &function, const TileShape &tileShape,
    const LogicalTensorPtr &params, const LogicalTensorPtr &indices, int axis) {
    std::vector<int> resultShape = GatherOperationResultShape(params, indices, axis);
    auto result = std::make_shared<LogicalTensor>(function, params->Datatype(), resultShape);

    TileInfo paramsTileInfo(params->shape.size(), params->offset.size());
    TileInfo indicesTileInfo(indices->shape.size(), indices->offset.size());
    TileInfo resultTileInfo(result->shape.size(), result->offset.size());
    auto paramsInput = Input{params, paramsTileInfo};
    auto indicesInput = Input{indices, indicesTileInfo};
    TiledGatherOperation(function, tileShape, 0, paramsInput, indicesInput, axis, result, resultTileInfo);

    return result;
}

LogicalTensorPtr TensorGatherOperation(Function &function,
    const LogicalTensorPtr &params, const LogicalTensorPtr &indices, int axis) {
    std::vector<int> resultShape = GatherOperationResultShape(params, indices, axis);
    auto result = std::make_shared<LogicalTensor>(function, params->Datatype(), resultShape);

    auto &op = function.AddOperation(Opcode::OP_GATHER, {params, indices}, {result});
    op.SetAttribute(OP_ATTR_PREFIX + "axis", axis);

    return result;
}

void TiledGatherElementOperation(Function &function, const TileShape &tileShape, size_t cur, Input &paramsInput,
    Input &indicesInput, int axis, const LogicalTensorPtr &result, TileInfo &resultTileInfo) {
    if (cur == result->shape.size()) {
        // add Operation
        auto paramsTile = paramsInput.tensor->View(function, paramsInput.tileInfo.shape, paramsInput.tileInfo.offset);
        auto indicesTile =
            indicesInput.tensor->View(function, indicesInput.tileInfo.shape, indicesInput.tileInfo.offset);
        auto resultTile = result->View(function, resultTileInfo.shape, resultTileInfo.offset);
        auto &op = function.AddOperation(Opcode::OP_GATHER_ELEMENT, {paramsTile, indicesTile}, {resultTile});
        op.SetAttribute(OP_ATTR_PREFIX + "axis", axis);
        return;
    }

    // 按照resultShape进行切分
    int tmpTile = tileShape.V(cur);
    for (int i = 0; i < result->shape[cur]; i += tmpTile) {
        if (cur == static_cast<size_t>(axis)) {
            // params[axis]不切
            paramsInput.tileInfo.offset[cur] = 0;
            paramsInput.tileInfo.shape[cur] = paramsInput.tensor->shape[cur];
            // 处理indices的tileInfo
            indicesInput.tileInfo.offset[cur] = i % indicesInput.tensor->shape[cur];
            indicesInput.tileInfo.shape[cur] =
                std::min(indicesInput.tensor->shape[cur] - indicesInput.tileInfo.offset[cur], tmpTile);
        } else {
            paramsInput.tileInfo.offset[cur] = i % paramsInput.tensor->shape[cur];
            paramsInput.tileInfo.shape[cur] =
                std::min(indicesInput.tensor->shape[cur] - indicesInput.tileInfo.offset[cur], tmpTile);
            // 处理indices的tileInfo
            indicesInput.tileInfo.offset[cur] = i % indicesInput.tensor->shape[cur];
            indicesInput.tileInfo.shape[cur] =
                std::min(indicesInput.tensor->shape[cur] - indicesInput.tileInfo.offset[cur], tmpTile);
        }

        resultTileInfo.offset[cur] = i;
        resultTileInfo.shape[cur] = std::min(result->shape[cur] - resultTileInfo.offset[cur], tmpTile);
        TiledGatherElementOperation(
            function, tileShape, cur + 1, paramsInput, indicesInput, axis, result, resultTileInfo);
    }
}

void TiledGatherElementOperation(Function &function, const TileShape &tileShape,
    const LogicalTensorPtr &params, const LogicalTensorPtr &indices, int axis,
    const LogicalTensorPtr &result) {
    // Check Operands Valid
    assert(result->shape.size() == result->offset.size());
    assert(params->shape.size() == params->offset.size());
    assert(indices->shape.size() == indices->offset.size());

    TileInfo paramsTileInfo(params->shape.size(), params->offset.size());
    TileInfo indicesTileInfo(indices->shape.size(), indices->offset.size());
    TileInfo resultTileInfo(result->shape.size(), result->offset.size());
    auto paramsInput = Input{params, paramsTileInfo};
    auto indicesInput = Input{indices, indicesTileInfo};
    TiledGatherElementOperation(function, tileShape, 0, paramsInput, indicesInput, axis, result, resultTileInfo);
}

LogicalTensorPtr TensorGatherElementOperation(Function &function,
    const LogicalTensorPtr &params, const LogicalTensorPtr &indices, int axis) {
    auto result = std::make_shared<LogicalTensor>(function, params->Datatype(), indices->shape);

    auto &op = function.AddOperation(Opcode::OP_GATHER_ELEMENT, {params, indices}, {result});
    op.SetAttribute(OP_ATTR_PREFIX + "axis", axis);

    return result;
}

struct ScatterTileInfoPara {
    TileInfo srcTileInfo;
    TileInfo idxTileInfo;
    TileInfo dstTileInfo;
};

struct ScatterElementPara {
    const LogicalTensorPtr &dstTensor;
    const LogicalTensorPtr &srcInput;
    const LogicalTensorPtr &idxInput;
    const Element& scalar;
    const int axis;
};

void InnerTiledScatterElement(size_t cur, Function &function, const TileShape &tileShape,
    const ScatterElementPara& scatterPara, ScatterTileInfoPara& scatterTileInfo) {
    const LogicalTensorPtr &dstTensor = scatterPara.dstTensor;
    const LogicalTensorPtr &srcInput = scatterPara.srcInput;
    const LogicalTensorPtr &idxInput = scatterPara.idxInput;
    const Element& scalar = scatterPara.scalar;
    const int axis = scatterPara.axis;

    if (cur == dstTensor->shape.size()) {
        // add Operation
        auto srcTile = srcInput->View(function, scatterTileInfo.srcTileInfo.shape, scatterTileInfo.srcTileInfo.offset);
        auto idxTile = idxInput->View(function, scatterTileInfo.idxTileInfo.shape, scatterTileInfo.idxTileInfo.offset);
        auto dstTile = dstTensor->View(function, scatterTileInfo.dstTileInfo.shape, scatterTileInfo.dstTileInfo.offset);
        auto &op = function.AddOperation(Opcode::OP_SCATTER_ELEMENT, {srcTile, idxTile}, {dstTile});
        op.SetAttribute(OP_ATTR_PREFIX + "axis", axis);
        op.SetAttribute(OpAttributeKey::scalar, scalar);
        return;
    }

    // 按照dstShape进行切分
    int tmpTile = tileShape.V(cur);
    if (static_cast<int>(cur) == axis) {
        tmpTile = dstTensor->shape[cur];
    }
    for (int i = 0; i < idxInput->shape[cur]; i += tmpTile) {
        if (static_cast<int>(cur) == axis) {
            scatterTileInfo.idxTileInfo.offset[cur] = 0;
            scatterTileInfo.idxTileInfo.shape[cur] = idxInput->shape[cur];
            scatterTileInfo.dstTileInfo.offset[cur] = 0;
            scatterTileInfo.dstTileInfo.shape[cur] = dstTensor->shape[cur];
            scatterTileInfo.srcTileInfo.offset[cur] = 0;
            scatterTileInfo.srcTileInfo.shape[cur] = srcInput->shape[cur];
        } else {
            scatterTileInfo.idxTileInfo.offset[cur] = i % idxInput->shape[cur];
            scatterTileInfo.idxTileInfo.shape[cur] =
                std::min(idxInput->shape[cur] - scatterTileInfo.idxTileInfo.offset[cur], tmpTile);
            scatterTileInfo.dstTileInfo.offset[cur] = i;
            scatterTileInfo.dstTileInfo.shape[cur] =
                std::min(idxInput->shape[cur] - scatterTileInfo.idxTileInfo.offset[cur], tmpTile);
            scatterTileInfo.srcTileInfo.offset[cur] = i;
            scatterTileInfo.srcTileInfo.shape[cur] =
                std::min(idxInput->shape[cur] - scatterTileInfo.idxTileInfo.offset[cur], tmpTile);
        }
        InnerTiledScatterElement(cur + 1, function, tileShape, scatterPara, scatterTileInfo);
    }
}

void TiledScatterElement(Function &function, const TileShape &tileShape, const ScatterElementPara& scatterPara) {
    // Check Operands Valid
    assert(scatterPara.srcInput->shape.size() == scatterPara.srcInput->offset.size());
    assert(scatterPara.idxInput->shape.size() == scatterPara.idxInput->offset.size());
    assert(scatterPara.dstTensor->shape.size() == scatterPara.dstTensor->offset.size());

    ScatterTileInfoPara scatterTileInfo{
        TileInfo(scatterPara.srcInput->shape.size(), scatterPara.srcInput->offset.size()),
        TileInfo(scatterPara.idxInput->shape.size(), scatterPara.idxInput->offset.size()),
        TileInfo(scatterPara.dstTensor->shape.size(), scatterPara.dstTensor->offset.size()),
    };
    InnerTiledScatterElement(0, function, tileShape, scatterPara, scatterTileInfo);
}

void TensorScatterElement(Function &function, const ScatterElementPara& scatterPara) {
    auto &op = function.AddOperation(Opcode::OP_SCATTER_ELEMENT, {scatterPara.srcInput, scatterPara.idxInput}, {scatterPara.dstTensor});
    op.SetAttribute(OP_ATTR_PREFIX + "axis", scatterPara.axis);
    op.SetAttribute(OpAttributeKey::scalar, scatterPara.scalar);
}

void UnalignPadTmpBufTile(std::vector<int> &shape) {
    // tmpbuf按16 8对齐
    auto size = shape.size();
    if (size >= NUM_VALUE_2) {
        shape[size - NUM_VALUE_2] = (shape[size - NUM_VALUE_2] + NUM_VALUE_16 - 1) / NUM_VALUE_16 * NUM_VALUE_16;
        shape[size - 1] = (shape[size - 1] + NUM_VALUE_8 - 1) / NUM_VALUE_8 * NUM_VALUE_8;
    }
}

template <TransposeOpType T>
void TiledInnerTranspose(Function &function, const TileShape &tileShape, const int cur, Input &input,
    const LogicalTensorPtr &result, const std::vector<int> &shape) {
    int shapeSize = input.tensor->shape.size();
    if (cur == shapeSize) {
        auto tile = input.tensor->View(function, input.tileInfo.shape, input.tileInfo.offset);
        std::vector<int> resultTileShape(input.tileInfo.shape);
        std::swap(resultTileShape[shape[0]], resultTileShape[shape[1]]);
        std::vector<int> resultTileOfs(input.tileInfo.offset);
        std::swap(resultTileOfs[shape[0]], resultTileOfs[shape[1]]);
        auto resultTile = result->View(function, resultTileShape, resultTileOfs);
        std::vector<int> tmpShape(input.tileInfo.shape);
        if (tmpShape.size() == SHAPE_DIM5) {
            // 临时tensor的transpose轴对应的shape对齐: 受指令限制，last轴按32Byte对齐，nlast轴按16对齐
            int blockNum = BLOCK_SIZE / static_cast<int>(BytesOf(tile->Datatype()));
            tmpShape[SHAPE_DIM5 - 1] = AlignUp(tmpShape[SHAPE_DIM5 - 1], blockNum);
            tmpShape[SHAPE_DIM5 - 2] = AlignUp(tmpShape[SHAPE_DIM5 - 2], VNCHWCONV_REPEAT);
        }
        if (T == TransposeOpType::TRANSPOSE_VNCHWCONV) {
            UnalignPadTmpBufTile(tmpShape);
        }
        auto tempTensor = std::make_shared<LogicalTensor>(function, tile->Datatype(), tmpShape);
        if (T == TransposeOpType::TRANSPOSE_DATAMOVE) {
            auto &op = function.AddOperation(GetTransposeOpName<T>(), {tile}, {resultTile});
            op.SetAttribute(OP_ATTR_PREFIX + "shape", shape);
        } else {
            auto &op = function.AddOperation(GetTransposeOpName<T>(), {tile}, {resultTile, tempTensor});
            op.SetAttribute(OP_ATTR_PREFIX + "shape", shape);
        }
        return;
    }
    for (int i = 0; i < input.tensor->shape[cur]; i += tileShape.V(cur)) {
        int dimTileSize = tileShape.V(cur);
        if (cur == shapeSize - 1 && tileShape.V(cur) != input.tensor->shape[cur]) {
            dimTileSize = input.tensor->shape[cur];
            ALOG_INFO_F("transpose dont tile last dim ,tensor shape %d, tile shape %d.", input.tensor->shape[cur],
                tileShape.V(cur));
        }
        input.tileInfo.shape[cur] = std::min(input.tensor->shape[cur] - i, dimTileSize);
        input.tileInfo.offset[cur] = i;
        TiledInnerTranspose<T>(function, tileShape, cur + 1, input, result, shape);
    }
}

template <TransposeOpType T>
void TiledInnerTranspose(Function &function, const TileShape &tileShape,
    const LogicalTensorPtr &operand, const LogicalTensorPtr &result,
    const std::vector<int> &shape) {
    TileInfo tileInfo(result->shape.size(), result->offset.size());
    auto input = Input{operand, tileInfo};
    TiledInnerTranspose<T>(function, tileShape, 0, input, result, shape);
}

void TensorInnerTranspose(Function &function, const LogicalTensorPtr &operand,
    const LogicalTensorPtr &result, std::vector<int> transposeShape) {
    constexpr size_t dimSizeTwo = 2;
    if (operand->shape.size() != dimSizeTwo && (transposeShape[0] != static_cast<int>(operand->shape.size() - dimSizeTwo) ||
        transposeShape[1] != static_cast<int>(operand->shape.size() - 1))) {
        auto &operation = function.AddOperation(Opcode::OP_TRANSPOSE_DATAMOVE, {operand}, {result});
        operation.SetAttribute(OP_ATTR_PREFIX + "shape", transposeShape);
    } else {
        auto &operation = function.AddOperation(Opcode::OP_TRANSPOSE_VNCHWCONV, {operand}, {result});
        operation.SetAttribute(OP_ATTR_PREFIX + "shape", transposeShape);
    }
}

void TiledMaxpool(Function &function, const TileShape &tileShape, const std::shared_ptr<LogicalTensor> &input,
    const std::shared_ptr<LogicalTensor> &output, const Operation &op) {
    const int dimN = output->shape[NUM_VALUE_0];
    const int dimC1 = output->shape[NUM_VALUE_1];
    const int dimOutH = output->shape[NUM_VALUE_2];
    const int dimOutW = output->shape[NUM_VALUE_3];
    const int dimInH = input->shape[NUM_VALUE_2];
    const int dimInW = input->shape[NUM_VALUE_3];
    const int c0 = output->shape[NUM_VALUE_4];
    const int paddingLeft = op.GetIntAttribute(ConvOpAttributeKey::paddingLeft);
    const int paddingTop = op.GetIntAttribute(ConvOpAttributeKey::paddingTop);
    const int paddingRight = op.GetIntAttribute(ConvOpAttributeKey::paddingRight);
    const int paddingBottom = op.GetIntAttribute(ConvOpAttributeKey::paddingBottom);
    const int strideH = op.GetIntAttribute(ConvOpAttributeKey::strideh);
    const int strideW = op.GetIntAttribute(ConvOpAttributeKey::stridew);
    const int poolH = op.GetIntAttribute(PoolOpAttributeKey::poolh);
    const int poolW = op.GetIntAttribute(PoolOpAttributeKey::poolw);
    int tileOutH = tileShape.V(NUM_VALUE_0);
    int tileOutW = tileShape.V(NUM_VALUE_1);
    bool isOnlyNeedCopy = strideH == 1 && strideW == 1 && poolH == 1 && poolW == 1;

    for (int n = 0; n < dimN; n++) {
        const int tileN = 1;
        for (int c1 = 0; c1 < dimC1; c1++) {
            const int tileC1 = 1;
            for (int h = 0; h < dimOutH; h += tileOutH) {
                const int tileHOut = Min(dimOutH - h, tileOutH);
                int startHIn = -paddingTop + h * strideH;
                int curStartHIn = startHIn > 0 ? startHIn : 0;
                int endHIn = -paddingTop + (h + tileHOut - 1) * strideH + poolH - 1;
                int curEndHIn = endHIn < dimInH ? endHIn : dimInH - 1;
                int tileHIn = curEndHIn - curStartHIn + 1;
                const int curPaddingTop = startHIn > 0 ? 0 : paddingTop;
                const int curPaddingBottom = endHIn < dimInH ? 0 : paddingBottom;
                for (int w = 0; w < dimOutW; w += tileOutW) {
                    const int tileWOut = Min(dimOutW - w, tileOutW);
                    int startWIn = -paddingLeft + w * strideW;
                    int curStartWIn = startWIn > 0 ? startWIn : 0;
                    int endWIn = -paddingLeft + (w + tileWOut - 1) * strideW + poolW - 1;
                    int curEndWIn = endWIn < dimInW ? endWIn : dimInW - 1;
                    int tileWIn = curEndWIn - curStartWIn + 1;
                    const int curPaddingLeft = startWIn > 0 ? 0 : paddingLeft;
                    const int curPaddingRight = endWIn < dimInW ? 0 : paddingRight;

                    auto inTile = input->View(
                        function, {tileN, tileC1, tileHIn, tileWIn, c0}, {n, c1, curStartHIn, curStartWIn, 0});
                    auto outTile = output->View(function, {tileN, tileC1, tileHOut, tileWOut, c0}, {n, c1, h, w, 0});
                    if (isOnlyNeedCopy) {
                        function.AddOperation(Opcode::OP_COPY_UB_TO_UB, {inTile}, {outTile});
                        continue;
                    }

                    auto &maxpoolOp = function.AddOperation(Opcode::OP_MAX_POOL, {inTile}, {outTile});

                    maxpoolOp.SetAttribute(ConvOpAttributeKey::paddingLeft, curPaddingLeft);
                    maxpoolOp.SetAttribute(ConvOpAttributeKey::paddingTop, curPaddingTop);
                    maxpoolOp.SetAttribute(ConvOpAttributeKey::paddingRight, curPaddingRight);
                    maxpoolOp.SetAttribute(ConvOpAttributeKey::paddingBottom, curPaddingBottom);
                    maxpoolOp.SetAttribute(ConvOpAttributeKey::strideh, strideH);
                    maxpoolOp.SetAttribute(ConvOpAttributeKey::stridew, strideW);
                    maxpoolOp.SetAttribute(PoolOpAttributeKey::poolh, poolH);
                    maxpoolOp.SetAttribute(PoolOpAttributeKey::poolw, poolH);
                }
            }
        }
    }
}

void TensorMaxpool(Function &function, const std::shared_ptr<LogicalTensor> &operand,
    const std::shared_ptr<LogicalTensor> &result, const std::vector<int> &pools, const std::vector<int> &strides,
    const std::vector<int> &paddings) {
    const int paddingLeft = paddings[NUM_VALUE_0];
    const int paddingTop = paddings[NUM_VALUE_1];
    const int paddingRight = paddings[NUM_VALUE_2];
    const int paddingBottom = paddings[NUM_VALUE_3];
    const int strideH = strides[NUM_VALUE_0];
    const int strideW = strides[NUM_VALUE_1];
    const int poolH = pools[NUM_VALUE_0];
    const int poolW = pools[NUM_VALUE_1];

    auto& maxpoolTensorOp = function.AddOperation(Opcode::OP_MAX_POOL, {operand}, {result});
    maxpoolTensorOp.SetAttribute(ConvOpAttributeKey::paddingLeft, paddingLeft);
    maxpoolTensorOp.SetAttribute(ConvOpAttributeKey::paddingTop, paddingTop);
    maxpoolTensorOp.SetAttribute(ConvOpAttributeKey::paddingRight, paddingRight);
    maxpoolTensorOp.SetAttribute(ConvOpAttributeKey::paddingBottom, paddingBottom);
    maxpoolTensorOp.SetAttribute(ConvOpAttributeKey::strideh, strideH);
    maxpoolTensorOp.SetAttribute(ConvOpAttributeKey::stridew, strideW);
    maxpoolTensorOp.SetAttribute(PoolOpAttributeKey::poolh, poolH);
    maxpoolTensorOp.SetAttribute(PoolOpAttributeKey::poolw, poolW);
}

Tensor Maxpool(const Tensor &operand, const std::vector<int> &pools, const std::vector<int> &strides,
    const std::vector<int> &paddings) {
    DECLARE_TRACER();
    // 目前只支持5D操作
    ASSERT((operand->shape.size() == NC1HWC0_DIM_NUM) && pools.size() == NUM_VALUE_2 &&
           strides.size() == STRIDE_DIM_NUM && paddings.size() == PADS_DIM_NUM);

    const int inDimH = operand->shape[NUM_VALUE_2];
    const int inDimW = operand->shape[NUM_VALUE_3];
    const int paddingLeft = paddings[NUM_VALUE_0];
    const int paddingTop = paddings[NUM_VALUE_1];
    const int paddingRight = paddings[NUM_VALUE_2];
    const int paddingBottom = paddings[NUM_VALUE_3];
    const int strideH = strides[NUM_VALUE_0];
    const int strideW = strides[NUM_VALUE_1];
    const int kh = pools[NUM_VALUE_0];
    const int kw = pools[NUM_VALUE_1];
    const int outHeight = CeilDiv(inDimH + paddingTop + paddingBottom - kh + 1, strideH);
    const int outWidth = CeilDiv(inDimW + paddingLeft + paddingRight - kw + 1, strideW);
    const std::vector<int> outShape = {
        operand->shape[NUM_VALUE_0], operand->shape[NUM_VALUE_1], outHeight, outWidth, operand->shape[NUM_VALUE_4]};
    Tensor result(operand->tensor->datatype, outShape);
    CALL(Maxpool, *Program::GetInstance().GetCurrentFunction(), operand.GetStorage(), result.GetStorage(), pools, strides, paddings);

    return result;
}

template <CastOpType T>
void TiledCastOperation(Function &function, const TileShape &tileShape, const int cur, Input &input,
    const LogicalTensorPtr &result, const CastMode &mode) {
    if (cur == static_cast<int>(input.tensor->shape.size())) {
        auto tile = input.tensor->View(function, input.tileInfo.shape, input.tileInfo.offset);
        auto resultTile = result->View(function, input.tileInfo.shape, input.tileInfo.offset);
        auto &op = function.AddOperation(GetCastOpName<T>(), {tile}, {resultTile});
        op.SetAttribute(OP_ATTR_PREFIX + "mode", mode);
        return;
    }
    for (int i = 0; i < input.tensor->shape[cur]; i += tileShape.V(cur)) {
        input.tileInfo.shape[cur] = std::min(input.tensor->shape[cur] - i, tileShape.V(cur));
        input.tileInfo.offset[cur] = i;
        TiledCastOperation<T>(function, tileShape, cur + 1, input, result, mode);
    }
}

template <CastOpType T>
void TiledCastOperation(Function &function, const TileShape &tileShape,
    const LogicalTensorPtr &operand, const LogicalTensorPtr &result, const CastMode &mode) {
    assert(operand->shape.size() == operand->offset.size());

    TileInfo tileInfo(result->shape.size(), result->offset.size());
    auto input = Input{operand, tileInfo};
    TiledCastOperation<T>(function, tileShape, 0, input, result, mode);
}

template <CastOpType T>
LogicalTensorPtr TensorCastOperation(Function &function, LogicalTensorPtr operand,
    const DataType &newType, const CastMode &mode) {
    auto result = std::make_shared<LogicalTensor>(function, newType, operand->shape);
    auto &op = function.AddOperation(GetCastOpName<T>(), {operand}, {result});
    op.SetAttribute(OP_ATTR_PREFIX + "mode", mode);
    return result;
}

Tensor Cast(const Tensor &operand, DataType newDataType, CastMode mode) {
    DECLARE_TRACER();
    assert(operand->shape.size() == operand->offset.size());
    // Cast to same dType with no mode will do nothing
    if (operand->tensor->datatype == newDataType && (mode == CAST_NONE || mode == CAST_RINT)) {
      return operand;
    }
    RETURN_CALL(CastOperation<CastOpType::CAST>, *Program::GetInstance().GetCurrentFunction(),
        operand.GetStorage(), newDataType, mode);
}

Tensor Exp(const Tensor &operand) {
    DECLARE_TRACER();

    RETURN_CALL(
        UnaryOperation<UnaryOpType::EXP>, *Program::GetInstance().GetCurrentFunction(), operand.GetStorage());
}

Tensor Maximum(const Tensor &operand1, const Tensor &operand2) {
    DECLARE_TRACER();

    RETURN_CALL(BinaryOperation<BinaryOpType::MAXIMUM>, *Program::GetInstance().GetCurrentFunction(), operand1, operand2);
}

Tensor Add(const Tensor &operand1, const Tensor &operand2) {
    DECLARE_TRACER();

    RETURN_CALL(BinaryOperation<BinaryOpType::ADD>, *Program::GetInstance().GetCurrentFunction(), operand1, operand2);
}

Tensor Sub(const Tensor &operand1, const Tensor &operand2) {
    DECLARE_TRACER();

    RETURN_CALL(BinaryOperation<BinaryOpType::SUB>, *Program::GetInstance().GetCurrentFunction(), operand1, operand2);
}

Tensor Mul(const Tensor &operand1, const Tensor &operand2) {
    DECLARE_TRACER();

    RETURN_CALL(BinaryOperation<BinaryOpType::MUL>, *Program::GetInstance().GetCurrentFunction(), operand1, operand2);
}

Tensor Div(const Tensor &operand1, const Tensor &operand2) {
    DECLARE_TRACER();

    RETURN_CALL(BinaryOperation<BinaryOpType::DIV>, *Program::GetInstance().GetCurrentFunction(), operand1, operand2);
}

Tensor ScalarAdd(const Tensor &operand1, const Tensor &operand2) {
    DECLARE_TRACER();

    RETURN_CALL(BinaryOperationAllScalar<BinaryOpType::S_ADD>, *Program::GetInstance().GetCurrentFunction(),
        operand1.GetStorage(), operand2.GetStorage());
}
Tensor ScalarSub(const Tensor &operand1, const Tensor &operand2) {
    DECLARE_TRACER();

    RETURN_CALL(BinaryOperationAllScalar<BinaryOpType::S_SUB>, *Program::GetInstance().GetCurrentFunction(),
        operand1.GetStorage(), operand2.GetStorage());
}

Tensor ScalarMul(const Tensor &operand1, const Tensor &operand2) {
    DECLARE_TRACER();

    RETURN_CALL(BinaryOperationAllScalar<BinaryOpType::S_MUL>, *Program::GetInstance().GetCurrentFunction(),
        operand1.GetStorage(), operand2.GetStorage());
}

Tensor ScalarDiv(const Tensor &operand1, const Tensor &operand2) {
    DECLARE_TRACER();

    RETURN_CALL(BinaryOperationAllScalar<BinaryOpType::S_DIV>, *Program::GetInstance().GetCurrentFunction(),
        operand1.GetStorage(), operand2.GetStorage());
}

Tensor ScalarMax(const Tensor &operand1, const Tensor &operand2) {
    DECLARE_TRACER();

    RETURN_CALL(BinaryOperationAllScalar<BinaryOpType::S_MAX>, *Program::GetInstance().GetCurrentFunction(),
        operand1.GetStorage(), operand2.GetStorage());
}

Tensor Sqrt(const Tensor &operand) {
    DECLARE_TRACER();

    RETURN_CALL(
        UnaryOperation<UnaryOpType::SQRT>, *Program::GetInstance().GetCurrentFunction(), operand.GetStorage());
}

Tensor Reciprocal(const Tensor &operand) {
    DECLARE_TRACER();

    RETURN_CALL(UnaryOperation<UnaryOpType::RECIPROCAL>, *Program::GetInstance().GetCurrentFunction(),
        operand.GetStorage());
}

Tensor Abs(const Tensor &operand) {
    DECLARE_TRACER();

    RETURN_CALL(UnaryOperation<UnaryOpType::ABS>, *Program::GetInstance().GetCurrentFunction(),
        operand.GetStorage());
}

Tensor Duplicate(const Tensor &operand) {
    DECLARE_TRACER();

    RETURN_CALL(UnaryOperation<UnaryOpType::DUPLICATE>, *Program::GetInstance().GetCurrentFunction(),
        operand.GetStorage());
}

Tensor AddS(const Tensor &operand1, const Element &operand2) {
    DECLARE_TRACER();
    RETURN_CALL(BinaryOperationScalar<BinaryOpType::ADD>, *Program::GetInstance().GetCurrentFunction(),
        operand1.GetStorage(), operand2);
}

Tensor SubS(const Tensor &operand1, const Element &operand2) {
    DECLARE_TRACER();
    RETURN_CALL(BinaryOperationScalar<BinaryOpType::SUB>, *Program::GetInstance().GetCurrentFunction(),
        operand1.GetStorage(), operand2);
}

Tensor MulS(const Tensor &operand1, const Element &operand2) {
    DECLARE_TRACER();
    RETURN_CALL(BinaryOperationScalar<BinaryOpType::MUL>, *Program::GetInstance().GetCurrentFunction(),
        operand1.GetStorage(), operand2);
}

Tensor DivS(const Tensor &operand1, const Element &operand2) {
    DECLARE_TRACER();
    RETURN_CALL(BinaryOperationScalar<BinaryOpType::DIV>, *Program::GetInstance().GetCurrentFunction(),
        operand1.GetStorage(), operand2);
}

Tensor ScalarAddS(const Tensor &operand, const Element &value, bool reverseOperand) {
    DECLARE_TRACER();

    RETURN_CALL(BinaryOperationAllScalar<BinaryOpType::S_ADD>, *Program::GetInstance().GetCurrentFunction(),
        operand.GetStorage(), value, reverseOperand);
}

Tensor ScalarSubS(const Tensor &operand, const Element &value, bool reverseOperand) {
    DECLARE_TRACER();

    RETURN_CALL(BinaryOperationAllScalar<BinaryOpType::S_SUB>, *Program::GetInstance().GetCurrentFunction(),
        operand.GetStorage(), value, reverseOperand);
}

Tensor ScalarMulS(const Tensor &operand, const Element &value, bool reverseOperand) {
    DECLARE_TRACER();

    RETURN_CALL(BinaryOperationAllScalar<BinaryOpType::S_MUL>, *Program::GetInstance().GetCurrentFunction(),
        operand.GetStorage(), value, reverseOperand);
}

Tensor ScalarDivS(const Tensor &operand, const Element &value, bool reverseOperand) {
    DECLARE_TRACER();

    RETURN_CALL(BinaryOperationAllScalar<BinaryOpType::S_DIV>, *Program::GetInstance().GetCurrentFunction(),
        operand.GetStorage(), value, reverseOperand);
}

Tensor ScalarMaxS(const Tensor &operand, const Element &value, bool reverseOperand) {
    DECLARE_TRACER();

    RETURN_CALL(BinaryOperationAllScalar<BinaryOpType::S_MAX>, *Program::GetInstance().GetCurrentFunction(),
        operand.GetStorage(), value, reverseOperand);
}

Tensor Expand(const Tensor &operand, DataType dataType, const std::vector<int> &shape) {
    DECLARE_TRACER();

    ASSERT(operand->shape.size() == shape.size());
    auto result = Tensor(dataType, shape);
    Program::GetInstance().AddOperation(Opcode::OP_EXPAND, {operand.GetStorage()}, {result.GetStorage()});
    return result;
}

Tensor Expand(const Tensor &operand, const std::vector<int> &dstShape) {
    DECLARE_TRACER();

    ASSERT(operand->shape.size() == dstShape.size());
    Tensor result(operand.GetStorage()->Datatype(), dstShape);
    CALL(Expand, *Program::GetInstance().GetCurrentFunction(), operand.GetStorage(), result.GetStorage());
    return result;
}

void TiledReduceExpandNew(Function &function, const TileShape &tileShape, const std::string &op,
    const LogicalTensorPtr &operand, const LogicalTensorPtr &result) {
    ASSERT(op == "MAX" || op == "SUM");
    assert(operand->shape.size() == operand->offset.size());

    // 目前只支持2维操作
    if (operand->shape.size() != 2) {
        assert(false && "unsupported dimension");
    }

    TileInfo tileInfo({tileShape.V(0), operand->shape[1]}, std::vector<int>(operand->offset.size()));

    for (int i = 0; i < operand->shape[0]; i += tileShape.V(0)) {
        tileInfo.offset[0] = i;
        auto inputTile = operand->View(function, tileInfo.shape, tileInfo.offset);
        auto resultTile = result->View(function, tileInfo.shape, tileInfo.offset);
        TileReduceNew(function, tileShape, op, npu::tile_fwk::ReduceType::EXPAND, inputTile, resultTile);
    }
}

Tensor RowSumExpand(const Tensor &operand) {
    DECLARE_TRACER();
    Tensor result(operand->tensor->datatype, operand->shape);
    CALL(ReduceExpand, *Program::GetInstance().GetCurrentFunction(), "SUM", operand, result);
    return result;
}

Tensor RowMaxExpand(const Tensor &operand) {
    DECLARE_TRACER();
    Tensor result(operand->tensor->datatype, operand->shape);
    CALL(ReduceExpand, *Program::GetInstance().GetCurrentFunction(), "MAX", operand, result);
    return result;
}

Tensor RowMaxSingle(const Tensor &operand, int axis) {
    DECLARE_TRACER();
    auto resultShape = operand->shape;
    axis = axis < 0 ? operand->shape.size() + axis : axis;

    resultShape[axis] = 1;

    ASSERT(Program::GetInstance().tileShape.V(axis) % NUM_VALUE_8 == 0)
    << "RowMaxSingle op: the tileShape of reduce axis need to align 8!";

    Tensor result(operand->tensor->datatype, resultShape);
    int shapeSize = static_cast<int>(resultShape.size());
    auto tileShape = Program::GetInstance().GetTileShape().GetVecTileShapes();
    if (ConfigManager::Instance().GetOperationConfig("FORCE_COMBINE_AXIS", false) &&
        axis == shapeSize - 1 && shapeSize >= NUM2 &&
        (resultShape[shapeSize - NUM2] % NUM_VALUE_8 == 0 && tileShape[tileShape.size() - NUM2] % NUM_VALUE_8 == 0)) {
        CALL(ReduceSingle, *Program::GetInstance().GetCurrentFunction(), "MAX_COMBINE_AXIS", operand, result, axis);
    } else {
        CALL(ReduceSingle, *Program::GetInstance().GetCurrentFunction(), "MAX", operand, result, axis);
    }
    return result;
}

Tensor RowMinSingle(const Tensor &operand, int axis) {
    DECLARE_TRACER();
    auto resultShape = operand->shape;
    axis = axis < 0 ? operand->shape.size() + axis : axis;

    resultShape[axis] = 1;

    ASSERT(Program::GetInstance().tileShape.V(axis) % BLOCK_NUM == 0)
    << "RowMinSingle op: the tileShape of reduce axis need to align 8!";

    Tensor result(operand->tensor->datatype, resultShape);
    CALL(ReduceSingle, *Program::GetInstance().GetCurrentFunction(), "MIN", operand, result, axis);
    return result;
}

Tensor RowSumSingle(const Tensor &operand, int axis) {
    DECLARE_TRACER();
    auto resultShape = operand->shape;
    axis = axis < 0 ? operand->shape.size() + axis : axis;

    resultShape[axis] = 1;

    ASSERT(Program::GetInstance().tileShape.V(axis) % NUM_VALUE_8 == 0)
    << "RowSumSingle op: the tileShape of reduce axis need to align 8!";

    Tensor result(operand->tensor->datatype, resultShape);
    int shapeSize = static_cast<int>(resultShape.size());
    auto tileShape = Program::GetInstance().GetTileShape().GetVecTileShapes();
    if (ConfigManager::Instance().GetOperationConfig("FORCE_COMBINE_AXIS", false) &&
        axis == shapeSize - 1 && shapeSize >= NUM2 &&
        (resultShape[shapeSize - NUM2] % NUM_VALUE_8 == 0 && tileShape[tileShape.size() - NUM2] % NUM_VALUE_8 == 0)) {
        CALL(ReduceSingle, *Program::GetInstance().GetCurrentFunction(), "SUM_COMBINE_AXIS", operand, result, axis);
    } else {
        CALL(ReduceSingle, *Program::GetInstance().GetCurrentFunction(), "SUM", operand, result, axis);
    }
    return result;
}

Tensor Compact(const Tensor &operand) {
    DECLARE_TRACER();

    assert(operand->shape.size() == operand->offset.size());
    Tensor result(operand->tensor->datatype, {operand->shape[0], 1});
    Tensor workspace(operand->tensor->datatype, {operand->shape[0], NUM_VALUE_8});
    Program::GetInstance().AddOperation(Opcode::OP_COMPACT, {operand.GetStorage()}, {result.GetStorage()});
    return result;
}

Tensor Gather(const Tensor &params, const Tensor &indices, int axis) {
    DECLARE_TRACER();

    RETURN_CALL(GatherOperation, *Program::GetInstance().GetCurrentFunction(), params.GetStorage(), indices.GetStorage(), axis);
}

void TiledVecDup(Function &function, const TileShape &tileShape, size_t cur, const Element &value, const SymbolicScalar &dynValue,
    std::vector<int> &shape, const std::vector<SymbolicScalar> &validShape, const LogicalTensorPtr &results,
    TileInfo &resultTileInfo) {
    if (cur == results->shape.size()) {
        auto resultTile = results->View(function, resultTileInfo.shape, resultTileInfo.offset);
        auto lastIndex = resultTile->shape.size() - 1;
        auto bytes = BytesOf(resultTile->Datatype());
        auto paddingIter = BLOCK_PADDING_DIM.find(bytes);
        int paddingValue = 1;
        if (paddingIter != BLOCK_PADDING_DIM.end()) {
            paddingValue = paddingIter->second;
        }
        auto tempTileShape = resultTile->shape;
        tempTileShape[lastIndex] = (tempTileShape[lastIndex] + paddingValue -1) / paddingValue * paddingValue;
        Program::GetInstance().tileShape.SetVecTileShapes(tempTileShape);
        auto &op = function.AddOperation("TILE_VEC_DUP", {}, {resultTile});

        op.SetAttribute(OpAttributeKey::scalar, value);
        if (dynValue.IsValid()) {
            op.SetAttribute(OpAttributeKey::dynScalar, dynValue);
        }
        op.SetAttribute(OP_ATTR_PREFIX + "shape", resultTileInfo.shape);
        op.SetAttribute(OP_ATTR_PREFIX + "validShape", resultTile->GetDynValidShape());
        return;
    }

    for (int i = 0; i < results->shape[cur]; i += tileShape.V(cur)) {
        resultTileInfo.offset[cur] = i;
        resultTileInfo.shape[cur] = std::min(results->shape[cur] - i, tileShape.V(cur));
        TiledVecDup(function, tileShape, cur + 1, value, dynValue, shape, validShape, results, resultTileInfo);
    }
}

void TiledVecDup(Function &function, const TileShape &tileShape, const Element &value, const SymbolicScalar &dynValue,
    std::vector<int> &shape, const std::vector<SymbolicScalar> &validShape, const LogicalTensorPtr &results) {
    TileInfo resultTileInfo(results->shape.size(), results->offset.size());
    TiledVecDup(function, tileShape, 0, value, dynValue, shape, validShape, results, resultTileInfo);
}

Tensor TensorVectorDuplicateOperation(Function &function, const Element& src, const SymbolicScalar &dynValue,
    DataType dtype, const std::vector<int> &dstShape, const std::vector<SymbolicScalar> &validShape) {
    auto result = std::make_shared<LogicalTensor>(function, dtype, dstShape, validShape);
    auto &op = function.AddOperation(Opcode::OP_VEC_DUP, {}, {result}); //输入没有tensor
    op.SetAttribute(OpAttributeKey::scalar, src);
    if (dynValue.IsValid()) {
        op.SetAttribute(OpAttributeKey::dynScalar, dynValue);
    }
    op.SetAttribute(OP_ATTR_PREFIX + "shape", dstShape);
    op.SetAttribute(OP_ATTR_PREFIX + "validShape", validShape);
    return result;
}

Tensor VectorDuplicate(const Element &src, DataType dtype, std::vector<int> dstShape,
    std::vector<SymbolicScalar> validShape) {
    DECLARE_TRACER();
    if (validShape.empty()) {
        for (auto x : dstShape)
            validShape.emplace_back(x);
    }
    RETURN_CALL(VectorDuplicateOperation, *Program::GetInstance().GetCurrentFunction(), src, SymbolicScalar(), dtype, dstShape, validShape);
}

Tensor VectorDuplicate(const SymbolicScalar &dynSrc, DataType dtype, std::vector<int> dstShape,
    std::vector<SymbolicScalar> validShape) {
    DECLARE_TRACER();
    if (validShape.empty()) {
        for (auto x : dstShape)
            validShape.emplace_back(x);
    }
    RETURN_CALL(VectorDuplicateOperation, *Program::GetInstance().GetCurrentFunction(), Element(dtype, (int64_t)0), dynSrc, dtype, dstShape, validShape);
}

Tensor GatherElement(const Tensor &params, const Tensor &indices, int axis) {
    DECLARE_TRACER();

    RETURN_CALL(GatherElementOperation, *Program::GetInstance().GetCurrentFunction(), params.GetStorage(),
        indices.GetStorage(), axis);
}

Tensor Transpose(const Tensor &operand, std::vector<int> transposeShape) {
    DECLARE_TRACER();
    constexpr int32_t TRANS_EXPERT_SHAPE_2 = 2;
    constexpr int32_t TRANS_EXPERT_SHAPE_4 = 4;
    assert(
        operand->shape.size() == TRANS_EXPERT_SHAPE_2 ||
        (transposeShape.size() == TRANS_EXPERT_SHAPE_2 && transposeShape[0] < static_cast<int>(operand->shape.size()) &&
            transposeShape[1] < static_cast<int>(operand->shape.size())));
    std::sort(transposeShape.begin(), transposeShape.end());
    assert(transposeShape[0] + 1 == transposeShape[1]);
    if ((operand->shape[transposeShape[0]] == 1 && operand->shape[transposeShape[1]] == 1) ||
        transposeShape[0] == transposeShape[1]) {
        return operand;
    }
    std::vector<int> resultShape(operand->shape);
    auto leftIdx = transposeShape[0];
    auto rightIdx = transposeShape[1];
    auto tmp = resultShape[leftIdx];
    resultShape[leftIdx] = resultShape[rightIdx];
    resultShape[rightIdx] = tmp;
    if (operand->shape.size() == TRANS_EXPERT_SHAPE_4 && transposeShape[0] + transposeShape[1] == 1) {
        auto lastTwoDim = operand->shape[NUM_VALUE_2] * operand->shape[NUM_VALUE_3];
        std::vector<int> tmpInputShape = {operand->shape[0], operand->shape[1], lastTwoDim};
        auto tmpInputTensor = Reshape(operand, tmpInputShape);
        auto oldVecTileShapes = Program::GetInstance().tileShape.GetVecTileShapes();
        if (!oldVecTileShapes.empty()) {
            std::vector<int> thirdDimVecTileShapes(NUM_VALUE_3);
            thirdDimVecTileShapes[0] = oldVecTileShapes[0];
            thirdDimVecTileShapes[1] = oldVecTileShapes[1];
            thirdDimVecTileShapes[NUM_VALUE_2] = lastTwoDim;
            Program::GetInstance().tileShape.SetVecTileShapes(thirdDimVecTileShapes);
        }
        auto tmpOutputTensor = Transpose(tmpInputTensor, transposeShape);
        if (!oldVecTileShapes.empty()) {
            Program::GetInstance().tileShape.SetVecTileShapes(oldVecTileShapes);
        }
        auto outputTensor = Reshape(tmpOutputTensor, resultShape);
        return outputTensor;
    }
    Tensor result(operand->Datatype(), resultShape);

    CALL(InnerTranspose, *Program::GetInstance().GetCurrentFunction(), operand.GetStorage(), result.GetStorage(),
        transposeShape);
    return result;
}

Tensor TensorIndex(const Tensor &params, const Tensor &indices) {
    DECLARE_TRACER();

    // TensorIndex默认按0轴进行gather
    RETURN_CALL(GatherOperation, *Program::GetInstance().GetCurrentFunction(), params.GetStorage(), indices.GetStorage(), 0);
}

Tensor Unsqueeze(const Tensor &old, int unsqueezeDimNum) {
    DECLARE_TRACER();

    ASSERT(unsqueezeDimNum < static_cast<int>(old->shape.size()) + 1 && unsqueezeDimNum >= -static_cast<int>(old->shape.size()) - 1);
    size_t unsqueezeDim = unsqueezeDimNum;
    if (unsqueezeDimNum < 0) {
        unsqueezeDim = unsqueezeDimNum + old->shape.size() + 1;
    }
    std::vector<int> newShape(old.GetStorage()->shape);
    newShape.insert(newShape.begin() + unsqueezeDim, 1);
    Tensor result(old->tensor->datatype, newShape);
    result = Reshape(old, newShape);

    return result;
}

void TiledScatterUpdate(size_t cur, Function &function, const TileShape &tileShape, Input &srcInput,
   Input &indexInput, Input &dstInput, int axis, const LogicalTensorPtr &dst,
    TileInfo &dstTileInfo, std::string cacheMode, int blockSize) {
    if (cur == dst->shape.size()) {
        // add Operation
        auto srcTile = srcInput.tensor->View(function, srcInput.tileInfo.shape, srcInput.tileInfo.offset);
        auto dstTile = dstInput.tensor->View(function, dstTileInfo.shape, dstTileInfo.offset);
        auto resultTile = dst->View(function, dstTileInfo.shape, dstTileInfo.offset);
        auto indexTile = indexInput.tensor->View(function, indexInput.tileInfo.shape, indexInput.tileInfo.offset);
        auto &op = function.AddOperation("TILE_INDEX_OUTCAST", {srcTile, indexTile, dstTile}, {resultTile});
        op.SetAttribute("axis", axis);
        op.SetAttribute(OpAttributeKey::panzBlockSize, blockSize);
        op.SetAttribute(OpAttributeKey::cacheMode, cacheMode);
        return;
    }

    // 按照dstShape进行切分
    int tmpTile = tileShape.V(cur);
    if (static_cast<int>(cur) == axis) {
        tmpTile = dst->shape[cur];
    }
    for (int i = 0; i < dst->shape[cur]; i += tmpTile) {
        if (static_cast<int>(cur) == axis) {
            srcInput.tileInfo.offset[cur] = 0;
            srcInput.tileInfo.shape[cur] = srcInput.tensor->shape[cur];
            if (cur <= 1) {
                indexInput.tileInfo.offset[cur] = 0;
                indexInput.tileInfo.shape[cur] = indexInput.tensor->shape[cur];
            }
            dstTileInfo.offset[cur] = 0;
            dstTileInfo.shape[cur] = dst->shape[cur];
        } else {
            srcInput.tileInfo.offset[cur] = i % srcInput.tensor->shape[cur];
            srcInput.tileInfo.shape[cur] =
                std::min(srcInput.tensor->shape[cur] - srcInput.tileInfo.offset[cur], tmpTile);
            if (cur == 0) { // only cut index first axis
                indexInput.tileInfo.offset[cur] = i % indexInput.tensor->shape[cur];
                indexInput.tileInfo.shape[cur] =
                    std::min(indexInput.tensor->shape[cur] - indexInput.tileInfo.offset[cur], tmpTile);
            } else {
                indexInput.tileInfo.offset[1] = 0;
                indexInput.tileInfo.shape[1] = indexInput.tensor->shape[1];
            }
            dstTileInfo.offset[cur] = i;
            dstTileInfo.shape[cur] = std::min(dst->shape[cur] - dstTileInfo.offset[cur], tmpTile);
        }
        TiledScatterUpdate(cur + 1, function, tileShape, srcInput, indexInput, dstInput, axis, dst, dstTileInfo, cacheMode, blockSize);
    }
}

void TiledIndexScatterUpdate(size_t cur, Function &function, const TileShape &tileShape, Input &srcInput,
   Input &indexInput, Input &dstInput, int axis, const std::shared_ptr<LogicalTensor> &dst,
    TileInfo &dstTileInfo, std::string cacheMode, int blockSize) {
    if (cur == dst->shape.size()) {
        // add Operation
        auto srcTile = srcInput.tensor->View(function, srcInput.tileInfo.shape, srcInput.tileInfo.offset);
        auto dstTile = dstInput.tensor->View(function, dstTileInfo.shape, dstTileInfo.offset);
        auto indexTile = indexInput.tensor->View(function, indexInput.tileInfo.shape, indexInput.tileInfo.offset);
        auto &op = function.AddOperation("TILE_INDEX_OUTCAST", {srcTile, indexTile, dstTile}, {dst});
        op.SetAttribute("axis", axis);
        op.SetAttribute(OpAttributeKey::panzBlockSize, blockSize);
        op.SetAttribute(OpAttributeKey::cacheMode, cacheMode);
        return;
    }

    // 按照srcShape进行切分
    int tmpTile = tileShape.V(cur);
    if (static_cast<int>(cur) == axis) {
        tmpTile = srcInput.tensor->shape[cur];
    }

    for (int i = 0; i < srcInput.tensor->shape[cur]; i += tmpTile) {
        if (static_cast<int>(cur) == axis) { // asis == 1
            srcInput.tileInfo.offset[cur] = 0;
            srcInput.tileInfo.shape[cur] = srcInput.tensor->shape[cur];

            int indexTileLen = tileShape.V(0);
            indexInput.tileInfo.offset[cur] = 0;
            indexInput.tileInfo.shape[cur] =
                std::min(indexInput.tensor->shape[cur] - indexInput.tileInfo.offset[0], indexTileLen);

            // indextileinfo need trans : [16,0] -> [0,16]
            indexInput.tileInfo.offset[cur] = indexInput.tileInfo.offset[0];
            indexInput.tileInfo.offset[0] = 0;

            dstTileInfo.offset[cur] = 0;
            dstTileInfo.shape[cur] = dst->shape[cur];
        } else {
            srcInput.tileInfo.offset[cur] = i % srcInput.tensor->shape[cur];
            srcInput.tileInfo.shape[cur] =
                std::min(srcInput.tensor->shape[cur] - srcInput.tileInfo.offset[cur], tmpTile);

            indexInput.tileInfo.offset[0] = i % indexInput.tensor->shape[1];
            indexInput.tileInfo.shape[0] = indexInput.tensor->shape[0]; // index axis 0

            dstTileInfo.offset[cur] = i;
            dstTileInfo.shape[cur] = tmpTile;
        }
        TiledIndexScatterUpdate(cur + 1, function, tileShape, srcInput, indexInput, dstInput, axis, dst, dstTileInfo, cacheMode, blockSize);
    }
}

void TiledScatterUpdate(Function &function, const TileShape &tileShape,
    const LogicalTensorPtr &result, const LogicalTensorPtr &src,
    const LogicalTensorPtr &index, const LogicalTensorPtr &dst, int axis, std::string cacheMode, int blockSize) {
    // Check Operands Valid
    assert(result->shape.size() == result->offset.size());
    assert(src->shape.size() == src->offset.size());
    assert(index->shape.size() == index->offset.size());

    TileInfo srcTileInfo(src->shape.size(), src->offset.size());
    TileInfo indexTileInfo(index->shape.size(), index->offset.size());
    TileInfo dstTileInfo(dst->shape.size(), dst->offset.size());
    TileInfo resultTileInfo(result->shape.size(), result->offset.size());

    auto srcInput = Input{src, srcTileInfo};
    auto indexInput = Input{index, indexTileInfo};
    auto dstInput = Input{dst, dstTileInfo};
    if (axis == 1 && src->shape.size() == NUM_VALUE_2 && tileShape.V(1) == src->shape[1]) { // 2维切index场景
        TiledIndexScatterUpdate(0, function, tileShape, srcInput, indexInput, dstInput, axis, result, resultTileInfo, cacheMode, blockSize);
    } else {
        TiledScatterUpdate(0, function, tileShape, srcInput, indexInput, dstInput, axis, result, resultTileInfo, cacheMode, blockSize);
    }
}

void TensorScatterUpdate(Function &function,
    const LogicalTensorPtr &result, const LogicalTensorPtr &dst,
    const LogicalTensorPtr &index, const LogicalTensorPtr &src, int axis, std::string cacheMode, int blockSize) {
    std::vector<int> newOffset(src->shape.size(), 0);

    // src: ub
    // index: ub
    // dst: gm
    // result: gm
    auto &op = function.AddOperation(Opcode::OP_INDEX_OUTCAST, {src, index, dst}, {result});
    op.SetAttribute("axis", axis);
    op.SetAttribute(OpAttributeKey::panzBlockSize, blockSize);
    op.SetAttribute(OpAttributeKey::cacheMode, cacheMode);
}

Tensor ScatterUpdate(const Tensor &dst, const Tensor &index, const Tensor &src, int axis, std::string cacheMode, int blockSize) {
    DECLARE_TRACER();

    ASSERT(dst->shape.size() == src->shape.size());
    axis = axis < 0 ? dst->shape.size() + axis : axis;
    ASSERT(static_cast<size_t>(axis)  < dst->shape.size());
    ASSERT(index->shape.size() == 2); // only support 2 dim

    Tensor result(dst->tensor->datatype, dst->shape);
    result.GetStorage()->tensor->SetTensorInfo(dst.GetStorage()->tensor->GetTensorInfo());
    result.GetStorage()->tensorfmt = dst.GetStorage()->tensorfmt;

    if (cacheMode == "PA_BSND" || cacheMode == "PA_NZ") {
        axis = 1;
        ASSERT(src->shape.size() == NUM_VALUE_2); // only support 2 dim

        Tensor newIndex = Reshape(index, {1, index->shape[0] * index->shape[1]});
        CALL(ScatterUpdate, *Program::GetInstance().GetCurrentFunction(), result.GetStorage(), dst.GetStorage(),
            newIndex.GetStorage(), src.GetStorage(), axis, cacheMode, blockSize);
    } else {
        CALL(ScatterUpdate, *Program::GetInstance().GetCurrentFunction(), result.GetStorage(), dst.GetStorage(),
         index.GetStorage(), src.GetStorage(), axis, cacheMode, blockSize);
    }
    return result;
}

Tensor ScatterElement(const Tensor &src, const Tensor &idx, const Element &scalar, int axis) {
    DECLARE_TRACER();
    // 目前只支持2维操作
    constexpr int kScatterDim = 2;
    ASSERT(src->shape.size() == kScatterDim);
    ASSERT(idx->shape.size() == kScatterDim);
    axis = axis < 0 ? src->shape.size() + axis : axis;
    Tensor result(src->tensor->datatype, src->shape);
    result.GetStorage()->tensor->SetTensorInfo(src.GetStorage()->tensor->GetTensorInfo());
    CALL(ScatterElement, *Program::GetInstance().GetCurrentFunction(), {result.GetStorage(), src.GetStorage(),
         idx.GetStorage(), scalar, axis});
    return result;
}

Tensor IndexPut(const Tensor &src, std::vector<Tensor> indices, const Tensor &values) {
    DECLARE_TRACER();

    Tensor result(src->tensor->datatype, src->shape);
    result.GetStorage()->tensor->SetTensorInfo(src.GetStorage()->tensor->GetTensorInfo());
    for (auto index : indices) {
        CALL(ScatterUpdate, *Program::GetInstance().GetCurrentFunction(), result.GetStorage(), src.GetStorage(),
        index.GetStorage(), values.GetStorage(), 0, "PA_PNSD", 1);
    }

    return result;
}

void TensorInnerAssemble(Function &function, const LogicalTensorPtr &operand,
    const LogicalTensorPtr &result, const VecTileShapes &offset) {
    auto &op = function.AddOperation(Opcode::OP_ASSEMBLE, {operand}, {result});
    op.SetOpAttribute(std::make_shared<AssembleOpAttribute>(offset));
}

void InnerAssemble(Function &function, const LogicalTensorPtr &operand,
    const LogicalTensorPtr &result, const VecTileShapes &offset) {
    CALL(InnerAssemble, function, operand, result, offset);
}

Tensor Assemble(const std::vector<std::pair<Tensor, std::vector<int>>> &tensors) {
    DECLARE_TRACER();

    ASSERT(!tensors.empty());
    std::vector<int> shape = tensors.front().first->shape;
    for (const auto &[tensor, offset] : tensors) {
        // 目前只支持2维操作
        if (tensor->shape.size() != 2) {
            ASSERT(false) << "unsupported dimension";
        }
        ASSERT(tensor->shape.size() == tensor->offset.size());
        ASSERT(tensor->shape.size() == offset.size());
    }

    auto shapeSize = tensors[0].first->shape.size(); // 2

    std::vector<int> rawShape(shapeSize, 0);

    std::set<std::vector<int>> position;
    for (const auto &[tensor, offset] : tensors) {
        (void)tensor;
        ASSERT(position.find(offset) == position.end());
        position.emplace(offset);
    }
    ASSERT(position.find(std::vector<int>(shapeSize, 0)) != position.end());

    for (const auto &[tensor, offset] : tensors) {
        for (int j = 0; static_cast<size_t>(j) < shapeSize; j++) {
            rawShape[j] = std::max(rawShape[j], tensor->shape[j] + offset[j]);
            ASSERT(offset[j] % shape[j] == 0);
            if (offset[j] > 0) {
                auto tmpOffset = offset;
                tmpOffset[j] -= shape[j];
                ASSERT(position.find(tmpOffset) != position.end());
            }
        }
    }

    for (int i = 0; static_cast<size_t>(i) < shapeSize; i++) {
        ASSERT(rawShape[i] > 0);
    }

    Tensor result(tensors[0].first->Datatype(), rawShape, "", NodeType::LOCAL, tensors[0].first->tensorfmt);
    auto &curFunc = *Program::GetInstance().GetCurrentFunction();
    for (const auto &[tensor, offset] : tensors) {
        InnerAssemble(curFunc, tensor.GetStorage(), result.GetStorage(), offset);
    }
    Program::GetInstance().GetTensorSlotManager()->TensorWrite(result, true);
    return result;
}

void InferDassembleValidShape(const LogicalTensorPtr &operand, const LogicalTensorPtr &result,
    const std::vector<int> &offset, std::vector<SymbolicScalar> &resultValidShape) {
    const std::vector<SymbolicScalar> operandDynValidShape = operand->GetDynValidShape();
    const std::vector<SymbolicScalar> resultDynValidShape = result->GetDynValidShape();

    for (size_t i = 0; i < offset.size(); i++) {
        if (resultDynValidShape.empty()) {
            resultValidShape.push_back(operandDynValidShape[i] + offset[i]);
        } else {
            resultValidShape.push_back(std::max(resultDynValidShape[i], operandDynValidShape[i] + offset[i]));
        }
    }
    return;
}

void TensorDInnerAssemble(Function &function, const LogicalTensorPtr &operand,
    const LogicalTensorPtr &result, const std::vector<SymbolicScalar> &dynOffset) {
    std::vector<int> offset = SymbolicScalar::Concrete(dynOffset, 0);

    std::vector<SymbolicScalar> resultValidShape;
    resultValidShape.reserve(offset.size());
    if (!operand->GetDynValidShape().empty()) {
        InferDassembleValidShape(operand, result, offset, resultValidShape);
        result->UpdateDynValidShape(resultValidShape);
    }

    auto &op = function.AddOperation(Opcode::OP_ASSEMBLE, {operand}, {result});
    op.SetAssembleOpAttribute(offset, dynOffset);
    op.SetAttribute("dassemble", true);
    MaybeAppendGetTensorData(&op, dynOffset);
}

void DInnerAssemble(Function &function, const LogicalTensorPtr &operand,
    const LogicalTensorPtr &result, const std::vector<SymbolicScalar> &dynOffset) {
    CALL(DInnerAssemble, function, operand, result, dynOffset);
}

void DAssemble(const Tensor &tensor, const std::vector<SymbolicScalar> &dynOffset, Tensor &dest) {
    DECLARE_TRACER();

    ASSERT(dest.GetStorage(false)->tensorfmt == tensor.GetStorage(false)->tensorfmt)<<"DAssemble: src and dest requires same format";
    ASSERT(dest.GetShape().size() == tensor.GetShape().size())<<"DAssemble: src and dest requires same shape";
    ASSERT(dest.GetShape().size() == dynOffset.size())<<"DAssemble: dynOffset and dest requires same shape";
    DInnerAssemble(*Program::GetInstance().GetCurrentFunction(), tensor.GetStorage(), dest.GetStorage(), dynOffset);

    Program::GetInstance().GetTensorSlotManager()->TensorWrite(dest, true);
}

void TensorInnerAssign(Function &function, const LogicalTensorPtr &operand, const LogicalTensorPtr &result) {
    function.AddOperation(Opcode::OP_REGISTER_COPY, {operand}, {result});
}

Tensor Assign(const Tensor &operand) {
    Tensor result(operand->Datatype(), operand->shape);
    CALL(InnerAssign, *Program::GetInstance().GetCurrentFunction(), operand.GetStorage(), result.GetStorage());
    return result;
}

static int64_t CalculateCapacity(const std::vector<int> &shape) {
    int64_t capacity = 1;
    for (size_t i = 0; i < shape.size(); i++) {
        capacity = capacity * shape[i];
    }
    return capacity;
}

void TiledInnerReshape(Function &function, const LogicalTensorPtr &operand, const LogicalTensorPtr &result) {
    auto &op = function.AddOperation("TILE_RESHAPE", {operand}, {result});
    op.SetAttribute(OP_ATTR_PREFIX + "validShape", result->GetDynValidShape());
    op.oOperand.front()->SetIsDummy();
}

void TensorInnerReshape(Function &function, const LogicalTensorPtr &operand, const LogicalTensorPtr &result, const std::vector<SymbolicScalar> &validShape) {
    auto &operation = function.AddOperation(Opcode::OP_RESHAPE, {operand}, {result});
    result->UpdateDynValidShape(validShape);
    operation.SetAttribute("reshape", result->shape);
}

static std::vector<int> CheckAndInferShape(const std::vector<int> &oriShape, const std::vector<int> &dstshape) {
    int negIdx = -1;
    std::vector<int> newShape = dstshape;
    auto capacity = CalculateCapacity(oriShape);

    for (size_t i = 0; i < newShape.size(); i++) {
        int x = newShape[i];
        ASSERT(x >= -1) << "Invalid shape " << x;
        if (x == -1) {
            ASSERT(negIdx == -1) << "Only one dim can be inferred";
            negIdx = i;
        }
        ASSERT(capacity % x == 0) << "Invalid dstshape";
        capacity /= x;
    }

    if (negIdx != -1) {
        newShape[negIdx] = capacity;
        capacity = 1;
    }
    ASSERT(capacity == 1) << "Shape size not match";
    return newShape;
}

static bool ReshapeNeedCopy(const Tensor &operand) {
    if (operand->shape != operand->tensor->rawshape) {
        return true;
    }
    if (operand->GetProducers().empty()) {
        return false;
    }

    auto op = *operand->GetProducers().begin();
    while (op->GetOpcode() == Opcode::OP_VIEW) {
        if (op->GetInputOperand(0)->GetShape() != op->GetOutputOperand(0)->GetShape()) {
            return true;
        }
        if (op->GetInputOperand(0) != nullptr && !op->GetInputOperand(0)->GetProducers().empty()) {
            op = *op->GetInputOperand(0)->GetProducers().begin();
        } else {
            break;
        }
    }
    return false;
}

Tensor Reshape(const Tensor &operand, const std::vector<int> &dstshape, const std::vector<SymbolicScalar> &validShape) {
    if (operand->shape == dstshape) {
        return operand;
    }
    auto newShape = CheckAndInferShape(operand->shape, dstshape);
    if (ReshapeNeedCopy(operand)) {
        Tensor copyOperand(operand->Datatype(), operand->shape, "", operand->nodetype, operand->tensorfmt);
        CALL(InnerAssign, *Program::GetInstance().GetCurrentFunction(), operand.GetStorage(),
            copyOperand.GetStorage());
        Tensor result(copyOperand->Datatype(), newShape, "", operand->nodetype, operand->tensorfmt);
        CALL(InnerReshape, *Program::GetInstance().GetCurrentFunction(), copyOperand.GetStorage(),
            result.GetStorage(), validShape);
        return result;
    } else {
        Tensor result(operand->Datatype(), newShape, "", operand->nodetype, operand->tensorfmt);
        CALL(InnerReshape, *Program::GetInstance().GetCurrentFunction(), operand.GetStorage(), result.GetStorage(), validShape);
        return result;
    }
}

#define CALL(n, ...) Tensor##n(__VA_ARGS__)
#define RETURN_CALL(n, ...) return Tensor##n(__VA_ARGS__)

void TiledInnerConcatLoop(const int dimIdx, Function &function, const TileShape &tileShape,
    const LogicalTensorPtr &operand, const LogicalTensorPtr &result,
    std::vector<int> actTileShape, std::vector<int> actOffset, std::vector<int> tensorOffset)
{
    if (static_cast<size_t>(dimIdx)  == result->GetShape().size()) {
        auto inputTile = operand->View(function, actTileShape, actOffset);
        auto tileOffset = tensorOffset;
        for (int i = 0; static_cast<size_t>(i) < actOffset.size(); ++i) {
            tileOffset[i] += actOffset[i];
        }
        auto &op = function.AddOperation(Opcode::OP_ASSEMBLE, {inputTile}, {result});
        op.SetOpAttribute(std::make_shared<AssembleOpAttribute>(tileOffset));
        return;
    }
    for (auto i = 0; i < operand->GetShape()[dimIdx]; i += tileShape.V(dimIdx)) {
        actTileShape[dimIdx] = std::min(operand->GetShape()[dimIdx] - i, tileShape.V(dimIdx));
        actOffset[dimIdx] = i;
        TiledInnerConcatLoop(dimIdx + 1, function, tileShape, operand, result, actTileShape, actOffset, tensorOffset);
    }
}

void TiledInnerConcat(Function &function, const TileShape &tileShape,
    const LogicalTensorPtr &operand, const LogicalTensorPtr &result,
    std::vector<int> tensorOffset)
{
    std::vector<int> actOffset(operand->GetShape().size(), 0);
    std::vector<int> actTileShape(operand->GetShape().size(), 1);
    TiledInnerConcatLoop(0, function, tileShape, operand, result, actTileShape, actOffset, tensorOffset);
}

void TensorInnerConcat(Function &function,
    const LogicalTensorPtr &operand, const LogicalTensorPtr &result,
    std::vector<int> tensorOffset)
{
    auto &op = function.AddOperation(Opcode::OP_CONCAT, {operand}, {result});
    op.SetAttribute("concat", tensorOffset);
}

void InnerConcat(Function &function, const LogicalTensorPtr &operand,
    const LogicalTensorPtr &result, std::vector<int> tensorOffset)
{
    CALL(InnerConcat, function, operand, result, tensorOffset);
}


void TiledInnerRegisterCopy(const int dimIdx, Function &function, const TileShape &tileShape,
    const LogicalTensorPtr &operand, const LogicalTensorPtr &result,
    std::vector<int> actTileShape, std::vector<int> actOffset)
{
    if (static_cast<size_t>(dimIdx)  == result->GetShape().size()) {
        auto inputTile = operand->View(function, actTileShape, actOffset);
        auto resultTile = result->View(function, actTileShape, actOffset);
        function.AddOperation("TILE_REGISTER_COPY", { inputTile }, { resultTile });
        return;
    }
    for (auto i = 0; i < result->GetShape()[dimIdx]; i += tileShape.V(dimIdx)) {
        actTileShape[dimIdx] = std::min(result->GetShape()[dimIdx] - i, tileShape.V(dimIdx));
        actOffset[dimIdx] = i;
        TiledInnerRegisterCopy(dimIdx + 1, function, tileShape, operand, result, actTileShape, actOffset);
    }
}

void TiledInnerRegisterCopy(Function &function, const TileShape &tileShape,
    const LogicalTensorPtr &operand, const LogicalTensorPtr &result)
{
    std::vector<int> actOffset(result->GetShape().size(), 0);
    std::vector<int> actTileShape(result->GetShape().size(), 1);
    TiledInnerRegisterCopy(0, function, tileShape, operand, result, actTileShape, actOffset);
}

void TensorInnerConcatNew(Function &function,
    const LogicalTensorPtr &operand, const LogicalTensorPtr &result)
{
    function.AddOperation(Opcode::OP_REGISTER_COPY, {operand}, {result});
}

void InnerConcatNew(Function &function, const LogicalTensorPtr &operand,
    const LogicalTensorPtr &result)
{
    CALL(InnerConcatNew, function, operand, result);
}


Tensor Concat(const std::vector<Tensor> &tensorList, int axis) {
    DECLARE_TRACER();

    if (tensorList.size() > MAX_CAT_NUM_ONCE) {
        std::vector<Tensor> front(tensorList.begin(), tensorList.begin() + MAX_CAT_NUM_ONCE);
        std::vector<Tensor> back(tensorList.begin() + MAX_CAT_NUM_ONCE, tensorList.end());
        Tensor concatFront = Concat(front, axis);
        back.insert(back.begin(), concatFront);
        return Concat(back, axis);
    }

    auto shape = tensorList[0]->GetShape();
    auto shapeSize = shape.size();
    if (axis < 0) {
        axis = shapeSize + axis;
    }
    ASSERT(static_cast<size_t>(axis) < shapeSize);
    for (auto tensor : tensorList) {
        ASSERT(tensor->GetShape().size() == shapeSize);
    }

    for (auto tensor : tensorList) {
        for (int i = 0; static_cast<size_t>(i) < shapeSize; ++i) {
            if (i == axis) {
                continue;
            }
            ASSERT(shape[i] == tensor->GetShape()[i]);
        }
    }

    auto resultShape = shape;
    int axisSize = 0;
    for (auto tensor : tensorList) {
        axisSize += tensor->GetShape()[axis];
    }
    resultShape[axis] = axisSize;

    Tensor result(tensorList[0]->Datatype(), resultShape);
    Tensor tmp(tensorList[0]->Datatype(), resultShape);
    auto &function = *Program::GetInstance().GetCurrentFunction();
    std::vector<int> offset(shapeSize, 0);
    for (auto tensor : tensorList) {
        auto tmpView = tmp->View(function, tensor->GetShape(), offset);
        InnerConcatNew(*Program::GetInstance().GetCurrentFunction(), tensor.GetStorage(), tmpView);
        offset[axis] += tensor->GetShape()[axis];
    }
    auto &op = function.AddOperation(Opcode::OP_ASSEMBLE, {tmp.GetStorage()}, {result.GetStorage()});
    op.SetOpAttribute(std::make_shared<AssembleOpAttribute>(std::vector<int>(shapeSize, 0)));

    return result;
}

void TiledPadOperation(Function &function, const std::vector<int> &tileShape, const std::vector<int> &tileOffset,
    const LogicalTensorPtr &result, const LogicalTensorPtr &operand)
{
    auto resultTile = result->View(function, tileShape, tileOffset);
    std::vector<int> originTileShape(tileShape);
    for (auto i = 0; static_cast<size_t>(i) < tileOffset.size(); i++) {
        if (tileOffset[i] >= operand->GetShape()[i]) {
            function.AddOperation("TILE_PAD", {}, { resultTile });
            return;
        } else if ((tileOffset[i] + tileShape[i]) > operand->GetShape()[i]) {
            originTileShape[i] = std::min(operand->GetShape()[i] - tileOffset[i], tileShape[i]);
        }
    }
    auto operandTile = View(operand, originTileShape, tileOffset);
    function.AddOperation("TILE_PAD", { operandTile.GetStorage() }, { resultTile });
}

void TiledPadLoop(Function &function, int dimIdx, std::vector<int> &tileShape, std::vector<int> &tileOffset,
    const LogicalTensorPtr &result, const LogicalTensorPtr &operand,
    std::vector<int> &cfgShape)
{
    if (static_cast<size_t>(dimIdx) == result->GetShape().size()) {
        TiledPadOperation(function, tileShape, tileOffset, result, operand);
        return;
    }
    for (auto i = 0; i < result->GetShape()[dimIdx]; i += cfgShape[dimIdx]) {
        tileShape[dimIdx] = std::min(result->GetShape()[dimIdx] - i, cfgShape[dimIdx]);
        tileOffset[dimIdx] = i;
        TiledPadLoop(function, dimIdx + 1, tileShape, tileOffset, result, operand, cfgShape);
    }
}

void TileInnerPad(Function &function, const TileShape &tileShape, const LogicalTensorPtr &operand,
    const LogicalTensorPtr &result)
{
    std::vector<int> offset(result->shape.size(), 0);
    std::vector<int> padTileShape(result->GetShape().size(), 1);
    std::vector<int> cfgShape(result->GetShape().size(), 1);
    cfgShape[cfgShape.size() - 1] = tileShape.V(1);
    cfgShape[cfgShape.size() - 2] = tileShape.V(0);
    TiledPadLoop(function, 0, padTileShape, offset, result, operand, cfgShape);
}

LogicalTensorPtr TensorPadOperation(Function &function, const TileShape &tileShape,
    const LogicalTensorPtr operand, const std::vector<int> &newShape)
{
    auto tmpResult = std::make_shared<LogicalTensor>(function, operand->Datatype(), newShape);
    auto result = std::make_shared<LogicalTensor>(function, operand->Datatype(), newShape);
    TileInnerPad(function, tileShape, operand, tmpResult);
    auto &assembleOp = function.AddOperation(Opcode::OP_ASSEMBLE, {tmpResult}, {result});
    assembleOp.SetAssembleOpAttribute(std::vector<int>(newShape.size(), 0));
    return result;
}

Tensor Pad(const Tensor &old, const std::vector<int> &newShape)
{
    DECLARE_TRACER();
    auto oldShape = old->GetShape();
    auto oldShapeSize = oldShape.size();
    assert(oldShapeSize == newShape.size());
    assert(oldShape.size() >= SHAPE_DIM2);
    for (int i = 0; static_cast<size_t>(i) < oldShapeSize; ++i) {
        // 目前只支持最后一维做pad
        if (static_cast<size_t>(i) == (oldShapeSize - 1)) {
            assert(newShape[i] >= oldShape[i]);
            continue;
        }
        ASSERT(oldShape[i] == newShape[i]);
    }
    RETURN_CALL(PadOperation, *Program::GetInstance().GetCurrentFunction(), Program::GetInstance().tileShape, old.GetStorage(), newShape);
}

void TiledInnerCompact(Function &function, const TileShape &tileShape,
    const LogicalTensorPtr &operand, const LogicalTensorPtr &result)
{
    assert(operand->shape.size() == operand->offset.size());
    auto workspace =
        std::make_shared<LogicalTensor>(function, operand->tensor->datatype,
                                       std::vector<int>{ operand->shape[0], NUM_VALUE_8 });

    // 目前只支持2维操作
    if (operand->shape.size() != 2) {
        assert(false && "unsupported dimension");
    }

    int tileShape1 = std::min(operand->shape[1], tileShape.V(1));
    for (int i = 0; i < operand->shape[0]; i += tileShape.V(0)) {
        int tileShape0 = std::min(operand->shape[0] - i, tileShape.V(0));
        auto inputTile = operand->View(function, { tileShape0, tileShape1 }, { i, 0 });
        auto resultTile = result->View(function, { tileShape0, 1 }, { i, 0 });
        auto workspaceTile = workspace->View(function, { tileShape0, NUM_VALUE_8 }, { i, 0 });
        function.AddOperation("TILE_COMPACT", { inputTile, workspaceTile }, { resultTile });
    }
}

void TensorInnerCompact(Function &function, const TileShape &tileShape,
    const LogicalTensorPtr &operand, const LogicalTensorPtr &result)
{
    TiledInnerCompact(function, tileShape, operand, result);
}

Tensor NewCompact(const Tensor &operand)
{
    DECLARE_TRACER();

    Tensor result(operand->tensor->datatype, { operand->shape[0], 1 });
    CALL(InnerCompact, *Program::GetInstance().GetCurrentFunction(), Program::GetInstance().tileShape, operand.GetStorage(), result.GetStorage());
    return result;
}

const std::string TOPK_AXIS = OP_ATTR_PREFIX + "axis";
const std::string TOPK_ORDER = OP_ATTR_PREFIX + "order";
const std::string TOPK_KVALUE = OP_ATTR_PREFIX + "kvalue";
const std::string EXTRACT_MASKMODE = OP_ATTR_PREFIX + "makeMode";

// 针对axis全排序,当前只支持axis为-1,输出为结果每32数据排序
void TensorBitsortOperation(Function &function,
    LogicalTensorPtr operand, LogicalTensorPtr resOp, int axis, bool isLargest) {
    auto &op = function.AddOperation(Opcode::OP_BITSORT, {operand}, {resOp});
    op.SetAttribute(TOPK_AXIS, axis);
    op.SetAttribute(TOPK_ORDER, static_cast<int>(isLargest));
}

// 全排序结果根据axis进行归并,当前只支持axis为-1
void TensorMrgSortOperation(Function &function, LogicalTensorPtr operand,
    LogicalTensorPtr resOp, int axis, int k, bool isLargest) {
    auto &op = function.AddOperation(Opcode::OP_MRGSORT, {operand}, {resOp});
    op.SetAttribute(TOPK_AXIS, axis);
    op.SetAttribute(TOPK_KVALUE, k);
    op.SetAttribute(TOPK_ORDER, static_cast<int>(isLargest));
}

void TensorExtractOperation(Function &function, LogicalTensorPtr operand, LogicalTensorPtr resOp,
    int maskMode, int k, bool isLargest) {
    auto &op = function.AddOperation(Opcode::OP_EXTRACT, {operand}, {resOp});
    op.SetAttribute(EXTRACT_MASKMODE, maskMode);
    op.SetAttribute(TOPK_KVALUE, k);
    op.SetAttribute(TOPK_ORDER, static_cast<int>(isLargest));
}

std::tuple<Tensor, Tensor> TopK(const Tensor &operand, const int &k, int axis = -1, bool isLargest) {
    DECLARE_TRACER();
    const auto len = static_cast<int>(operand->shape.size());
    assert(axis == 1 || axis == -1);
    axis = axis >= 0 ? axis : (axis + len);
    // 首先进行全排序,全排序的输出是输入shape的2倍,另外需要在输出中增加临时空间,size变为原有的4倍
    // 需要注意,这里由于芯片限制需要对k做32元素对齐
    // [index value 2] + [index_tmp_buffer 1] / [index_value_tmp_buffer 2] * 2 = 4 * origin_size
    constexpr int32_t blockSize = 32;
    constexpr int32_t kFactorSize = 4;
    constexpr int32_t kBlockFpNum = 8;
    auto bitsortOutShape = operand->shape;
    bitsortOutShape[axis] = (bitsortOutShape[axis] + blockSize -1) / blockSize * blockSize;
    bitsortOutShape[axis] *= kFactorSize;

    auto bitsortResTensor = Tensor(operand->tensor->datatype, bitsortOutShape);
    CALL(BitsortOperation, *Program::GetInstance().GetCurrentFunction(),
        operand.GetStorage(), bitsortResTensor.GetStorage(), axis, isLargest);

    // 归并排序,输入为全排序的结果,输出为原始输入shape的2倍
    auto mrgSortOutShape = operand->shape;
    // 输出往32Bytes对齐考虑
    mrgSortOutShape[axis] = (k + kBlockFpNum - 1) / kBlockFpNum * kBlockFpNum * 2;
    auto mrgsortResultTensor = Tensor(operand->tensor->datatype, mrgSortOutShape);
    CALL(MrgSortOperation, *Program::GetInstance().GetCurrentFunction(), bitsortResTensor.GetStorage(), mrgsortResultTensor.GetStorage(),
        axis, k, isLargest);

    // index拆分
    auto topkOutShape = operand->shape;
    topkOutShape[axis] = k;
    auto resultTensor = Tensor(operand->tensor->datatype, topkOutShape);
    CALL(ExtractOperation, *Program::GetInstance().GetCurrentFunction(), mrgsortResultTensor.GetStorage(), resultTensor.GetStorage(), 0, k, isLargest);

    // value 拆分
    auto resIndicesTensor = Tensor(DataType::DT_INT32, topkOutShape);
    CALL(ExtractOperation, *Program::GetInstance().GetCurrentFunction(), mrgsortResultTensor.GetStorage(), resIndicesTensor.GetStorage(), 1, k, isLargest);
    return std::tie(resultTensor, resIndicesTensor);
}

void TiledBitSort(Function &function, const TileShape & tileShape, size_t cur, Input &input,
    const LogicalTensorPtr &result, TileInfo &resultTileInfo, int axis, int isLargest) {
    if (cur == input.tensor->shape.size()) {
        auto inputTile = input.tensor->View(function, input.tileInfo.shape, input.tileInfo.offset);
        auto resultTile = result->View(function, resultTileInfo.shape, resultTileInfo.offset);
        auto &op = function.AddOperation(Opcode::OP_BITSORT, {inputTile}, {resultTile});
        op.SetAttribute(TOPK_AXIS, axis);
        op.SetAttribute(TOPK_ORDER, static_cast<int>(isLargest));
        return;
    }
    // Jump cur axis
    if (cur == static_cast<size_t>(axis)) {
        TiledBitSort(function, tileShape, cur + 1, input, result, resultTileInfo, axis, isLargest);
        return;
    }
    for (int i = 0; i < input.tensor->shape[cur]; i += tileShape.V(cur)) {
        // update input && result && resultDices shape and offset info
        input.tileInfo.offset[cur] = i % input.tensor->shape[cur];
        input.tileInfo.shape[cur] = std::min(input.tensor->shape[cur] - input.tileInfo.offset[cur], tileShape.V(cur));

        resultTileInfo.offset[cur] = i;
        resultTileInfo.shape[cur]= std::min(result->shape[cur] - resultTileInfo.offset[cur], tileShape.V(cur));
        TiledBitSort(function, tileShape, cur + 1, input, result, resultTileInfo, axis, isLargest);
    }
}

void TiledBitSort(Function &function, const TileShape &tileShape,
     const LogicalTensorPtr operand, const LogicalTensorPtr resOperand,int axis, int isLargest) {
    // Build Init tile info
    TileInfo tileInfo(operand->shape.size(), operand->offset.size());
    TileInfo resultTileInfo(resOperand->shape.size(), resOperand->offset.size());
    tileInfo.shape = operand->shape;
    resultTileInfo.shape = resOperand->shape;
    auto input = Input{operand, tileInfo};
    TiledBitSort(function, tileShape, 0, input, resOperand, resultTileInfo, axis, isLargest);
}

void TiledMrgSort(Function &function, const TileShape & tileShape, size_t cur, Input &input,
    const LogicalTensorPtr &result, TileInfo &resultTileInfo, int axis, int k, int isLargest) {
    if (cur == input.tensor->shape.size()) {
        auto inputTile = input.tensor->View(function, input.tileInfo.shape, input.tileInfo.offset);
        auto resultTile = result->View(function, resultTileInfo.shape, resultTileInfo.offset);
        auto &op = function.AddOperation(Opcode::OP_MRGSORT, {inputTile}, {resultTile});
        op.SetAttribute(TOPK_AXIS, axis);
        op.SetAttribute(TOPK_KVALUE, k);
        op.SetAttribute(TOPK_ORDER, static_cast<int>(isLargest));
        return;
    }
    // Jump cur axis
    if (static_cast<int>(cur) == axis) {
        TiledMrgSort(function, tileShape, cur + 1, input, result, resultTileInfo, axis, k, isLargest);
        return;
    }
    for (int i = 0; i < input.tensor->shape[cur]; i += tileShape.V(cur)) {
        // update input && result && resultDices shape and offset info
        input.tileInfo.offset[cur] = i % input.tensor->shape[cur];
        input.tileInfo.shape[cur] = std::min(input.tensor->shape[cur] - input.tileInfo.offset[cur], tileShape.V(cur));

        resultTileInfo.offset[cur] = i;
        resultTileInfo.shape[cur]= std::min(result->shape[cur] - resultTileInfo.offset[cur], tileShape.V(cur));
        TiledMrgSort(function, tileShape, cur + 1, input, result, resultTileInfo, axis, k, isLargest);
    }
}

void TiledMrgSort(Function &function, const TileShape &tileShape,
     const LogicalTensorPtr operand, const LogicalTensorPtr resOperand,
     int axis, int k, int isLargest) {
    // Build Init tile info
    TileInfo tileInfo(operand->shape.size(), operand->offset.size());
    TileInfo resultTileInfo(resOperand->shape.size(), resOperand->offset.size());
    tileInfo.shape = operand->shape;
    resultTileInfo.shape = resOperand->shape;
    auto input = Input{operand, tileInfo};
    TiledMrgSort(function, tileShape, 0, input, resOperand, resultTileInfo, axis, k, isLargest);
}


void TiledExtract(Function &function, const TileShape & tileShape, size_t cur, Input &input,
    const LogicalTensorPtr &result, TileInfo &resultTileInfo, int maskMode, int kValue,
    bool isLargest) {
    if (cur == input.tensor->shape.size()) {
        auto inputTile = input.tensor->View(function, input.tileInfo.shape, input.tileInfo.offset);
        auto resultTile = result->View(function, resultTileInfo.shape, resultTileInfo.offset);
        auto &op = function.AddOperation(Opcode::OP_EXTRACT, {inputTile}, {resultTile});
        op.SetAttribute(EXTRACT_MASKMODE, maskMode);
        op.SetAttribute(TOPK_KVALUE, kValue);
        op.SetAttribute(TOPK_ORDER, static_cast<int>(isLargest));
        return;
    }

    // Jump last axis
    if (cur == input.tensor->shape.size() - 1) {
        TiledExtract(function, tileShape, cur + 1, input, result,  resultTileInfo, maskMode, kValue, isLargest);
        return;
    }

    for (int i = 0; i < input.tensor->shape[cur]; i += tileShape.V(cur)) {
        // update input && result && resultDices shape and offset info
        input.tileInfo.offset[cur] = i % input.tensor->shape[cur];
        input.tileInfo.shape[cur] = std::min(input.tensor->shape[cur] - input.tileInfo.offset[cur], tileShape.V(cur));

        resultTileInfo.offset[cur] = i;
        resultTileInfo.shape[cur]= std::min(result->shape[cur] - resultTileInfo.offset[cur], tileShape.V(cur));
        TiledExtract(function, tileShape, cur + 1, input, result,  resultTileInfo, maskMode, kValue, isLargest);
    }
}

void TiledExtract(Function &function, const TileShape &tileShape,
     const LogicalTensorPtr operand, const LogicalTensorPtr resOperand,
     int maskMode, int kValue, int isLargest) {
    // Build Init tile info
    TileInfo tileInfo(operand->shape.size(), operand->offset.size());
    TileInfo resultTileInfo(resOperand->shape.size(), resOperand->offset.size());
    tileInfo.shape = operand->shape;
    resultTileInfo.shape = resOperand->shape;
    auto input = Input{operand, tileInfo};
    TiledExtract(function, tileShape, 0, input, resOperand, resultTileInfo, maskMode, kValue, isLargest);
}

Tensor ArgSort(const Tensor &operand, int axis = -1, bool isLargest) {
    DECLARE_TRACER();
    const auto len = static_cast<int>(operand->shape.size());
    assert(axis == 1 || axis == -1);
    axis = axis >= 0 ? axis : (axis + len);
    // 首先进行全排序,全排序的输出是输入shape的2倍,另外需要在输出中增加临时空间,size变为原有的4倍
    // 需要注意,这里由于芯片限制需要对k做32元素对齐
    // [index value 2] + [index_tmp_buffer 1] / [index_value_tmp_buffer 2] * 2 = 4 * origin_size
    auto bitsortOutShape = operand->shape;
    bitsortOutShape[axis] = (bitsortOutShape[axis] + NUM_VALUE_31) / NUM_VALUE_32 * NUM_VALUE_32;
    bitsortOutShape[axis] *= NUM_VALUE_4; // size变成原来的4倍
    auto bitsortResTensor = Tensor(operand->tensor->datatype, bitsortOutShape);
    CALL(BitsortOperation, *Program::GetInstance().GetCurrentFunction(),
        operand.GetStorage(), bitsortResTensor.GetStorage(), axis, isLargest);

    auto k = operand->shape[axis];
    // 归并排序,输入为全排序的结果,输出为原始输入shape的2倍
    auto mrgSortOutShape = operand->shape;
    constexpr int32_t MRG_SORT_TIMES_2 = 2;
    mrgSortOutShape[axis] = k * MRG_SORT_TIMES_2;
    auto mrgsortResultTensor = Tensor(operand->tensor->datatype, mrgSortOutShape);
    CALL(MrgSortOperation, *Program::GetInstance().GetCurrentFunction(), bitsortResTensor.GetStorage(), mrgsortResultTensor.GetStorage(),
        axis, k, isLargest);

    // // index拆分
    auto topkOutShape = operand->shape;
    topkOutShape[axis] = k;

    // value 拆分
    auto resIndicesTensor = Tensor(operand->tensor->datatype, topkOutShape);
    CALL(ExtractOperation, *Program::GetInstance().GetCurrentFunction(), mrgsortResultTensor.GetStorage(), resIndicesTensor.GetStorage(), 1, k, isLargest);
    return resIndicesTensor;
}

void TiledArgSort(Function &function, const TileShape & tileShape, size_t cur, Input &input,
    const LogicalTensorPtr &resultDices, TileInfo &resultDicesTileInfo, int axis, int isLargest) {
    if (cur == input.tensor->shape.size()) {
        auto inputTile = input.tensor->View(function, input.tileInfo.shape, input.tileInfo.offset);
        auto resultDicesTile = resultDices->View(function, resultDicesTileInfo.shape, resultDicesTileInfo.offset);
        function.AddOperation(Opcode::OP_ARGSORT, {inputTile}, {resultDicesTile});
        return;
    }
    if (cur == static_cast<size_t>(axis)) {
        input.tileInfo.offset[cur] = 0;
        input.tileInfo.shape[cur] = input.tensor->shape[cur];
        TiledArgSort(function, tileShape, cur + 1, input, resultDices, resultDicesTileInfo, axis, isLargest);
        return;
    }
    for (int i = 0; i < input.tensor->shape[cur]; i += tileShape.V(cur)) {
        // update input && result && resultDices shape and offset info
        input.tileInfo.offset[cur] = i % input.tensor->shape[cur];
        input.tileInfo.shape[cur] = std::min(input.tensor->shape[cur] - input.tileInfo.offset[cur], tileShape.V(cur));

        resultDicesTileInfo.offset[cur] = i;
        resultDicesTileInfo.shape[cur]= std::min(resultDices->shape[cur] - resultDicesTileInfo.offset[cur], tileShape.V(cur));
        TiledArgSort(function, tileShape, cur + 1, input, resultDices, resultDicesTileInfo, axis, isLargest);
    }
}

void TiledArgSort(Function &function, const TileShape &tileShape,
     const LogicalTensorPtr operand,
     const LogicalTensorPtr resDicesOperand, int axis, int isLargest) {
    // Build Init tile info
    TileInfo tileInfo(operand->shape.size(), operand->offset.size());
    TileInfo resultDicesTileInfo(resDicesOperand->shape.size(), resDicesOperand->offset.size());
    auto input = Input{operand, tileInfo};
    TiledArgSort(function, tileShape, 0, input, resDicesOperand,
        resultDicesTileInfo, axis, isLargest);
}
/* End: Start for TOPK and ArgSort */
/* Begin: Start for Reduce*/

Tensor Reduce(const std::vector<Tensor> &aggregation, const ReduceMode reduceMode) {
    DECLARE_TRACER();
    // Support Reduce::Add only
    if (reduceMode != ReduceMode::ATOMIC_ADD) {
        return Tensor();
    }
    std::vector<LogicalTensorPtr> iOperand;
    std::vector<LogicalTensorPtr> oOperand;
    iOperand.reserve(aggregation.size());
    std::transform(aggregation.begin(), aggregation.end(),
        std::back_inserter(iOperand),
        [](const Tensor& elem)
        {
            return elem.GetStorage();
        });
    auto o0 = iOperand[0];
    Tensor result(o0->Datatype(), o0->shape);
    auto& op = Program::GetInstance().AddOperation(Opcode::OP_REDUCE_ACC, iOperand, { result.GetStorage() });
    op.SetAttribute(ACC_A_MUL_B, 1);
    return result;
}

void TiledReduceAcc(Function &function, const TileShape &tileShape, size_t cur,
    std::vector<Input> inputVec, const LogicalTensorPtr &result, TileInfo &resultTileInfo) {
    if (cur == inputVec[0].tensor->shape.size()) {
        std::vector<LogicalTensorPtr> inputTileVec;
        for (size_t index = 0; index < inputVec.size(); ++index) {
            auto inputTile = inputVec[index].tensor->View(function, inputVec[index].tileInfo.shape, inputVec[index].tileInfo.offset);
            inputTileVec.emplace_back(inputTile);
        }

        auto resultTile = result->View(function, resultTileInfo.shape, resultTileInfo.offset);
        auto &op = function.AddOperation(Opcode::OP_REDUCE_ACC, inputTileVec, {resultTile});
        op.SetAttribute(ACC_A_MUL_B, 1);
        return;
    }
    for (auto i = 0; i < result->shape[cur]; i += tileShape.V(cur)) {
        resultTileInfo.offset[cur] = i;
        resultTileInfo.shape[cur] = std::min(result->shape[cur] - resultTileInfo.offset[cur], tileShape.V(cur));
        for (size_t index = 0; index < inputVec.size(); ++index) {
            inputVec[index].tileInfo.offset[cur]  = i % inputVec[index].tensor->shape[cur];
            inputVec[index].tileInfo.shape[cur] = std::min(inputVec[index].tensor->shape[cur] -
                inputVec[index].tileInfo.offset[cur], tileShape.V(cur));
        }
        TiledReduceAcc(function, tileShape, cur + 1, inputVec, result, resultTileInfo);
    }
}

void TiledReduceAcc(Function &function, const TileShape &tileShape,
    std::vector<LogicalTensorPtr> operandVec,
    const LogicalTensorPtr &result) {
    TileInfo tileInfo1(result->shape.size(), result->offset.size());
    TileInfo tileInfo2(result->shape.size(), result->offset.size());
    TileInfo resultTileInfo(result->shape.size(), result->offset.size());
    auto input1 = Input{operandVec[0], tileInfo1};
    auto input2 = Input{operandVec[1], tileInfo2};
    std::vector<Input> inputVec;
    for (size_t index = 0; index < operandVec.size(); ++index) {
        TileInfo tileInfo(result->shape.size(), result->offset.size());
        Input input = Input{operandVec[index], tileInfo};
        inputVec.push_back(input);
    }
    TiledReduceAcc(function, tileShape, 0, inputVec, result, resultTileInfo);
}
/* Begin: Start for Reduce*/
} // namespace npu::tile_fwk

namespace npu::tile_fwk::Matrix {
using AggregationMap =
    std::map<std::vector<int>, std::vector<std::pair<LogicalTensorPtr, LogicalTensorPtr>>>;

struct CollectSubAMulBPara {
    const TileShape &tileShape;
    const std::array<int, 3> &posK;
    const LogicalTensorPtr &a;
    const LogicalTensorPtr &b;
    const LogicalTensorPtr &c;
};

template <bool isTransA = false, bool isTransB = false>
void CollectSubAMulB(Function &function, const CollectSubAMulBPara &args, AggregationMap &aggregations,
    const std::vector<int> &outerOffset) {
    const TileShape &tileShape = args.tileShape;
    const LogicalTensorPtr &a = args.a;
    const LogicalTensorPtr &b = args.b;
    const LogicalTensorPtr &c = args.c;
    const std::array<int, 3> &posK = args.posK;
    const int l1M = a->shape[0];
    const int l1N = isTransB ? b->shape[0] : b->shape[1];
    const auto opCode = isTransB ? Opcode::OP_L1_TO_L0_BT : Opcode::OP_L1_TO_L0B;
    for (int m = 0; m < l1M; m += tileShape.M(0)) {
        for (int n = 0; n < l1N; n += tileShape.N(0)) {
            int actL0M = std::min(l1M - m, tileShape.M(0));
            int actL0N = std::min(l1N - n, tileShape.N(0));
            auto cc = c->View(function, {actL0M, actL0N}, {m, n});
            for (int k = 0; k < posK[2]; k += tileShape.K(0)) {
                int actL0K = std::min(posK[2] - k, tileShape.K(0));
                const std::vector<int> sizeVec =
                    isTransB ? std::vector<int>{actL0N, actL0K} : std::vector<int>{actL0K, actL0N};
                auto aa = a->View(function, {actL0M, actL0K}, {m, posK[0] + k});
                auto bb = isTransB ? b->View(function, sizeVec, {n, posK[1] + k}) : b->View(function, sizeVec, {posK[1] + k, n});
                if (config::UseTIG()) {
                    auto A_l0 = std::make_shared<LogicalTensor>(
                        function, a->Datatype(), std::vector<int>{actL0M, actL0K}, aa->GetDynValidShape(), "a_l0", a->nodetype , a->tensorfmt);
                    auto B_l0 =
                        std::make_shared<LogicalTensor>(function, b->Datatype(), std::vector<int>{actL0K, actL0N}, bb->GetDynValidShape(), "b_l0",
                        b->nodetype, b->tensorfmt);
                    function.AddOperation(Opcode::OP_L1_TO_L0A, {aa}, {A_l0});
                    function.AddOperation(opCode, {bb}, {B_l0});
                    auto innerOffset = outerOffset;
                    innerOffset[0] += m;
                    innerOffset[1] += n;
                    aggregations[innerOffset].emplace_back(A_l0, B_l0);
                } else {
                    aggregations[outerOffset].emplace_back(aa, bb);
                }
            }
        }
    }
}

void AddMatmulAttr(Operation& op)
{
    auto matrixSize = Program::GetInstance().GetMatrixSize();
    const int matrixMaxSize = 3;
    if (matrixSize.Size() < matrixMaxSize) {
        op.SetAttribute(A_MUL_B_ACT_M, 0);
        op.SetAttribute(A_MUL_B_ACT_K, 0);
        op.SetAttribute(A_MUL_B_ACT_N, 0);
        return;
    }
    const int mIndex = 0;
    const int kIndex = 1;
    const int nIndex = 2;
    op.SetAttribute(A_MUL_B_ACT_M, matrixSize.V(mIndex));
    op.SetAttribute(A_MUL_B_ACT_K, matrixSize.V(kIndex));
    op.SetAttribute(A_MUL_B_ACT_N, matrixSize.V(nIndex));
}

void AddMatmulAttr(Operation& op, int32_t nzAttr, const std::vector<int32_t>& matrixSize)
{
    op.SetAttribute(A_MUL_B_NZ_ATTR, nzAttr);
    const int matrixMaxSize = 3;
    if (matrixSize.size() < matrixMaxSize) {
        op.SetAttribute(A_MUL_B_ACT_M, 0);
        op.SetAttribute(A_MUL_B_ACT_K, 0);
        op.SetAttribute(A_MUL_B_ACT_N, 0);
        return;
    }
    if (nzAttr == 0) return;
    const int mIndex = 0;
    const int kIndex = 1;
    const int nIndex = 2;
    op.SetAttribute(A_MUL_B_ACT_M, matrixSize[mIndex]);
    op.SetAttribute(A_MUL_B_ACT_K, matrixSize[kIndex]);
    op.SetAttribute(A_MUL_B_ACT_N, matrixSize[nIndex]);
}


template <bool hasThirdInput = false, bool isTransA = false, bool isTransB = false>
void DoAMulB(Function &function, const TileShape &tileShape, const AggregationMap &aggregations,
    const LogicalTensorPtr &inputOperand,
    const LogicalTensorPtr &result, const std::vector<int>& matrixSize) {
    auto dataType = result->tensor->datatype;
    std::vector<int> shape = {tileShape.M(0), tileShape.N(0)};
    for (const auto &[offset, aggregation] : aggregations) {
        ASSERT(!aggregation.empty());
        auto preTmp = std::make_shared<LogicalTensor>(function, dataType, shape);
        shape[0] = std::min(tileShape.M(0), result->shape[0] - offset[0]);
        shape[1] = std::min(tileShape.N(0), result->shape[1] - offset[1]);

        auto resultTile = result->View(function, shape, offset);
        for (size_t i = 0; i < aggregation.size(); i++) {
            if (i == 0) {
                const auto inputWithThird = std::vector<LogicalTensorPtr>{aggregation[i].first, aggregation[i].second, inputOperand};
                const auto inputWithOutThird = std::vector<LogicalTensorPtr>{aggregation[i].first, aggregation[i].second};
                const std::vector<LogicalTensorPtr> inputVec =
                    (hasThirdInput == true) ? inputWithThird : inputWithOutThird;
                const std::string matmulOpStr = "TILE_A_MUL_B";
                if (i == aggregation.size() - 1) {
                    auto &op = function.AddOperation(matmulOpStr, inputVec, {resultTile});
                    int32_t nzAttr = (static_cast<int8_t>(aggregation[i].first->tensorfmt) << 0) +
                        (static_cast<int8_t>(aggregation[i].second->tensorfmt) << 1);
                    AddMatmulAttr(op, nzAttr, matrixSize);
                } else {
                    auto tmp = std::make_shared<LogicalTensor>(function, dataType, shape);
                    if (inputVec[0]->GetDynValidShape().size() != 0 && inputVec[1]->GetDynValidShape().size() != 0) {
                        tmp->UpdateDynValidShape({inputVec[0]->GetDynValidShape()[0], inputVec[1]->GetDynValidShape()[1]});
                    }
                    auto &op = function.AddOperation(matmulOpStr, inputVec, {tmp});
                    int32_t nzAttr = (static_cast<int8_t>(aggregation[i].first->tensorfmt) << 0) +
                        (static_cast<int8_t>(aggregation[i].second->tensorfmt) << 1);
                    AddMatmulAttr(op, nzAttr, matrixSize);
                    preTmp = tmp;
                }
            } else {
                const auto inputWithThird = std::vector<LogicalTensorPtr>{aggregation[i].first, aggregation[i].second, preTmp, inputOperand};
                const auto inputWithOutThird = std::vector<LogicalTensorPtr>{aggregation[i].first, aggregation[i].second, preTmp};
                const std::string matmulOpStr = "TILE_A_MULACC_B";

                const std::vector<LogicalTensorPtr> inputVec = inputWithOutThird;
                if (i == aggregation.size() - 1) {
                    auto &op = function.AddOperation(matmulOpStr, inputVec, {resultTile});
                    op.oOperand.front()->SetIsDummy();
                    int32_t nzAttr = (static_cast<int8_t>(aggregation[i].first->tensorfmt) << 0) +
                        (static_cast<int8_t>(aggregation[i].second->tensorfmt) << 1);
                    AddMatmulAttr(op, nzAttr, matrixSize);
                } else {
                    auto tmp = std::make_shared<LogicalTensor>(function, dataType, shape);
                    if (inputVec[0]->GetDynValidShape().size() != 0 && inputVec[1]->GetDynValidShape().size() != 0) {
                        tmp->UpdateDynValidShape({inputVec[0]->GetDynValidShape()[0], inputVec[1]->GetDynValidShape()[1]});
                    }
                    auto &op = function.AddOperation(matmulOpStr, inputVec, {tmp});
                    op.oOperand.front()->SetIsDummy();
                    int32_t nzAttr = (static_cast<int8_t>(aggregation[i].first->tensorfmt) << 0) +
                        (static_cast<int8_t>(aggregation[i].second->tensorfmt) << 1);
                    AddMatmulAttr(op, nzAttr, matrixSize);
                    preTmp = tmp;
                }
            }
        }
    }
}

template <bool isTransA = false, bool isTransB = false>
void TiledInnerAMulB(Function &function, const TileShape &tileShape,
    const std::vector<LogicalTensorPtr> operandVec,
    const LogicalTensorPtr &result, const std::vector<int32_t> matmulSize) {
    const auto opCode = Opcode::OP_A_MUL_B;
    const auto operand1 = operandVec[0];
    const auto operand2 = operandVec[1];
    if (function.GetGraphType() != GraphType::TILE_GRAPH) {
        assert(operand1->shape.size() == operand2->shape.size());
        auto &op = function.AddOperation(opCode, operandVec, {result});
        int32_t nzAttr = (static_cast<int8_t>(operand1->tensorfmt) << 0) + (static_cast<int8_t>(operand2->tensorfmt) << 1);
        AddMatmulAttr(op, nzAttr, matmulSize);
        return;
    }

    // 目前只支持2维操作
    if (operand1->shape.size() != 2) {
        assert(false && "unsupported dimension");
    }

    const int formatDimM = operand1->shape[0];
    const int formatDimK = operand1->shape[1];
    const int formatDimKb = isTransB ? operand2->shape[1] : operand2->shape[0];
    const int formatDimN = isTransB ? operand2->shape[0] : operand2->shape[1];
    assert(formatDimK == formatDimKb);

    bool checkDimK = isTransB ? formatDimK == operand2->shape[1] : formatDimK == operand2->shape[0];
    ASSERT(checkDimK) << "MATMUL: a.shape[1] != b.shape[0]";

    //增加左右矩阵L1搬运的K轴tilesize不同的逻辑
    const int lenK = 2;
    assert(tileShape.K(0) > 0 && tileShape.K(1) > 0 && tileShape.K(lenK) > 0 && tileShape.M(0) > 0 && tileShape.M(1) > 0 && tileShape.N(0) > 0 && tileShape.N(1) > 0);
    assert(tileShape.K(1) % tileShape.K(0) == 0 && "kTile[0] does not divide kTile[1]");
    assert(tileShape.K(lenK) % tileShape.K(0) == 0 && "kTile[0] does not divide kTile[2]");
    const int stepK = std::gcd(tileShape.K(1), tileShape.K(lenK));

    AggregationMap aggregations;
    // 增加计算尾块的逻辑
    for (int m = 0; m < formatDimM; m += tileShape.M(1)) {
        for (int n = 0; n < formatDimN; n += tileShape.N(1)) {
            auto tileM = std::min(formatDimM - m, tileShape.M(1));
            auto tileN = std::min(formatDimN - n, tileShape.N(1));
            auto resultTile = result->View(function, {tileM, tileN}, {m, n});
            if (!tileShape.SetL1Tile()) {
                for (int k = 0; k < formatDimK; k += tileShape.K(1)) {
                    auto tempTileK = std::min(formatDimK - k, tileShape.K(1));
                    auto inputTile1 = operand1->View(function, {tileM, tempTileK}, {m, k});
                    auto inputTile2 = isTransB ? operand2->View(function, {tileN, tempTileK}, {n, k}) :
                                        operand2->View(function, {tempTileK, tileN}, {k, n});
                    CollectSubAMulB<isTransA, isTransB>(
                        function, {tileShape, {0, 0, tempTileK}, inputTile1, inputTile2, resultTile}, aggregations, {m, n});
                }
            } else {
                std::vector<std::pair<int, int>> leftTiles, rightTiles;
                for (int k = 0; k < formatDimK; k += tileShape.K(1)) {
                    int endK = std::min(k + tileShape.K(1), formatDimK);
                    leftTiles.emplace_back(k, endK);
                }
                for (int k = 0; k < formatDimK; k += tileShape.K(lenK)) {
                    int endK = std::min(k + tileShape.K(lenK), formatDimK);
                    rightTiles.emplace_back(k, endK);
                }
                std::vector<std::shared_ptr<LogicalTensor>> leftTilesL1, rightTilesL1;
                for (const auto& tile : leftTiles) {
                    int startK = tile.first;
                    int endK = tile.second;
                    auto tempTileK1 = endK - startK;
                    auto Tile1 = operand1->View(function, {tileM, tempTileK1}, {m, startK});
                    auto inputTile1 = std::make_shared<LogicalTensor>(function, operand1->Datatype(), std::vector<int>{tileM, tempTileK1}, Tile1->GetDynValidShape(),
                                                    "a_l1", Tile1->nodetype , Tile1->tensorfmt);
                    auto &CopyInA = function.AddOperation(Opcode::OP_COPY_IN, {Tile1}, {inputTile1});
                    CopyInA.SetOpAttribute(std::make_shared<CopyOpAttribute>(OpImmediate::Specified({0, 0}),
                        MemoryType::MEM_L1, OpImmediate::Specified(inputTile1->GetShape()),
                        OpImmediate::Specified(inputTile1->tensor->GetDynRawShape())));
                    leftTilesL1.push_back(inputTile1);
                }
                for (const auto& tile : rightTiles) {
                    int startK = tile.first;
                    int endK = tile.second;
                    auto tempTileK2 = endK - startK;
                    auto Tile2 = isTransB ? operand2->View(function, {tileN, tempTileK2}, {n, startK}) :
                                                operand2->View(function, {tempTileK2, tileN}, {startK, n});
                    auto inputTile2 = isTransB ? std::make_shared<LogicalTensor>(function, operand2->Datatype(), std::vector<int>{tileN, tempTileK2}, Tile2->GetDynValidShape(),
                                                "b_l1", Tile2->nodetype , Tile2->tensorfmt) :
                                                std::make_shared<LogicalTensor>(function, operand2->Datatype(), std::vector<int>{tempTileK2, tileN}, Tile2->GetDynValidShape(),
                                                "b_l1", Tile2->nodetype , Tile2->tensorfmt);
                    auto &CopyInB = function.AddOperation(Opcode::OP_COPY_IN, {Tile2}, {inputTile2});
                    CopyInB.SetOpAttribute(std::make_shared<CopyOpAttribute>(OpImmediate::Specified({0, 0}),
                        MemoryType::MEM_L1, OpImmediate::Specified(inputTile2->GetShape()),
                        OpImmediate::Specified(inputTile2->tensor->GetDynRawShape())));
                    rightTilesL1.push_back(inputTile2);
                }
                for (int k = 0; k < formatDimK; k += stepK) {
                    int i = k / tileShape.K(1);
                    int j = k / tileShape.K(lenK);
                    int startK1 = k - leftTiles[i].first;
                    int startK2 = k - rightTiles[j].first;
                    int lengthK = std::min(stepK, formatDimK - k);
                    CollectSubAMulB<isTransA, isTransB>(
                        function, {tileShape, {startK1, startK2, lengthK}, leftTilesL1[i], rightTilesL1[j], resultTile}, aggregations, {m, n});
                }
            }
        }
    }
    const int accOperandSize = 3;
    if (operandVec.size() == accOperandSize) {
        DoAMulB<true, isTransA, isTransB>(function, tileShape, aggregations, operandVec[2], result, matmulSize);
    } else {
        DoAMulB<false, isTransA, isTransB>(function, tileShape, aggregations,
            std::make_shared<LogicalTensor>(function, result->Datatype(), result->shape, result->GetDynValidShape()), result, matmulSize);
    }
}

void TensorInnerAMulB(Function &function, const std::vector<LogicalTensorPtr> &operandVec,
    const LogicalTensorPtr &result) {
    const auto operand1 = operandVec[0];
    const auto operand2 = operandVec[1];
    assert(operand1->shape.size() == operand2->shape.size());
    assert(operand1->shape[1] == operand2->shape[0]);
    if (operand1->GetDynValidShape().size() != 0 && operand2->GetDynValidShape().size() != 0) {
        result->UpdateDynValidShape({operand1->GetDynValidShape()[0], operand2->GetDynValidShape()[1]});
    }
    auto& op = function.AddOperation(Opcode::OP_A_MUL_B, operandVec, {result});
    AddMatmulAttr(op);
}

void CheckMatMulOperandsValid(DataType dataType, const Tensor &operand1, const Tensor &operand2) {
    // shape valid check
    assert(operand1->shape.size() != 0 && operand2->shape.size() != 0);
    for (size_t i = 0; i < operand1->shape.size(); ++i) {
        assert(operand1->shape[i] > 0);
    }
    for (size_t i = 0; i < operand2->shape.size(); ++i) {
        assert(operand2->shape[i] > 0);
    }
    // 内轴值要小于等于65535，只有ND2NZ指令需要
    auto opFormatA = operand1->GetTileOpFormat();
    auto opFormatB = operand2->GetTileOpFormat();
    if (opFormatA == TileOpFormat::TILEOP_ND) {
        assert(operand1->shape.back() <= SHAPE_INNER_AXIS_MAX_SIZE);
    }
    if (opFormatB == TileOpFormat::TILEOP_ND) {
        assert(operand2->shape.back() <= SHAPE_INNER_AXIS_MAX_SIZE);
    }
    // tile valid check
    auto tileShape = Program::GetInstance().GetTileShape().GetCubeTileShapes();
    int kL0 = tileShape.GetTileShape<TileShapeType::K>(0);
    int kL1 = tileShape.GetTileShape<TileShapeType::K>(1);
    int mL0 = tileShape.GetTileShape<TileShapeType::M>(0);
    int mL1 = tileShape.GetTileShape<TileShapeType::M>(1);
    int nL0 = tileShape.GetTileShape<TileShapeType::N>(0);
    int nL1 = tileShape.GetTileShape<TileShapeType::N>(1);
    assert(kL0 > 0 && kL1 > 0 && mL0 > 0 && mL1 > 0 && nL0 > 0 && nL1 > 0);
    assert(kL0 <= kL1 && kL1 % kL0 == 0);
    assert(nL0 <= nL1 && nL1 % nL0 == 0);
    assert(mL0 <= mL1 && mL1 % mL0 == 0);
    assert(kL0 * BytesOf(dataType) % ALIGN_SIZE_32 == 0);
    assert(nL0 * BytesOf(dataType) % ALIGN_SIZE_32 == 0);
}

void MatmulImpl(DataType dataType, const std::vector<LogicalTensorPtr>& iOperand, LogicalTensorPtr &result) {
    const auto operand1 = iOperand[0];
    const auto operand2 = iOperand[1];

    CheckOperandsValid(operand1, operand2);
    CheckMatMulOperandsValid(dataType, operand1, operand2);
    assert(dataType == DT_FP32 || dataType == DT_FP16 || dataType == DT_BF16 || dataType == DT_INT32);
    CALL(InnerAMulB, *Program::GetInstance().GetCurrentFunction(), iOperand, result);
}

namespace Internel {
Tensor A_MUL_B(DataType dataType, const Tensor &operand1, const Tensor &operand2, const void *lr) {
    DECLARE_TRACERX(lr);
    Tensor result(dataType, {operand1->shape[0], operand2->shape[1]});
    MatmulImpl(dataType, {operand1.GetStorage(), operand2.GetStorage()}, result.GetStorage());
    return result;
}

/* Add for Matmul acc*/
Tensor A_MUL_B(
    DataType dataType, const Tensor &operand1, const Tensor &operand2, const Tensor &operand3, const void *lr) {
    DECLARE_TRACERX(lr);
    Tensor result(dataType, {operand3->shape[0], operand3->shape[1]});
    MatmulImpl(dataType, {operand1.GetStorage(), operand2.GetStorage(), operand3.GetStorage()}, result.GetStorage());
    return result;
}
}

void TensorInnerAMulBt(Function &function, const LogicalTensorPtr &operand1,
    const LogicalTensorPtr &operand2, const LogicalTensorPtr &result) {
    assert(operand1->shape.size() == operand2->shape.size());
    if (operand1->GetDynValidShape().size() != 0 && operand2->GetDynValidShape().size() != 0) {
        result->UpdateDynValidShape({operand1->GetDynValidShape()[0], operand2->GetDynValidShape()[0]});
    }
    auto &op = function.AddOperation(Opcode::OP_A_MUL_BT, {operand1, operand2}, {result});
    AddMatmulAttr(op);
}

void TensorInnerAMulBt(Function &function, const LogicalTensorPtr &operand1,
    const LogicalTensorPtr &operand2, const LogicalTensorPtr &operand3, const LogicalTensorPtr &result) {
    assert(operand1->shape.size() == operand2->shape.size());
    auto &op = function.AddOperation(Opcode::OP_A_MUL_BT, {operand1, operand2, operand3}, {result});
    AddMatmulAttr(op);
}

void AMulBtImpl(DataType dataType, const LogicalTensorPtr &operand1, const LogicalTensorPtr &operand2, LogicalTensorPtr &result) {
    CheckOperandsValid(operand1, operand2);
    CheckMatMulOperandsValid(dataType, operand1, operand2);
    assert(dataType == DataType::DT_FP32 || dataType == DataType::DT_FP16 || dataType == DataType::DT_BF16 || dataType == DataType::DT_INT32);
    CALL(InnerAMulBt, *Program::GetInstance().GetCurrentFunction(), operand1, operand2, result);
}

void AMulBtImpl(DataType dataType, const LogicalTensorPtr &operand1, const LogicalTensorPtr &operand2, const LogicalTensorPtr &operand3, LogicalTensorPtr &result) {
    CheckOperandsValid(operand1, operand2);
    CheckMatMulOperandsValid(dataType, operand1, operand2);
    assert(dataType == DataType::DT_FP32 || dataType == DataType::DT_FP16 || dataType == DataType::DT_BF16 || dataType == DataType::DT_INT32);
    CALL(InnerAMulBt, *Program::GetInstance().GetCurrentFunction(), operand1, operand2, operand3, result);
}

// normal matmul intf c = a * b
namespace Internel {
// A mul transpose B
Tensor A_MUL_Bt(DataType dataType, const Tensor &operand1, const Tensor &operand2, const void *lr) {
    DECLARE_TRACERX(lr);
    Tensor result(dataType, {operand1->shape[0], operand2->shape[0]});
    AMulBtImpl(dataType, operand1.GetStorage(), operand2.GetStorage(), result.GetStorage());
    return result;
}

Tensor A_MUL_Bt(
    DataType dataType, const Tensor &operand1, const Tensor &operand2, const Tensor &operand3, const void *lr) {
    DECLARE_TRACERX(lr);
    Tensor result(dataType, {operand1->shape[0], operand2->shape[0]});
    AMulBtImpl(dataType, operand1.GetStorage(), operand2.GetStorage(), operand3.GetStorage(), result.GetStorage());
    return result;
}
} // namespace Internel

Tensor ABatchMulB3D(DataType dataType, const Tensor &operand1, const Tensor &operand2) {
    assert(operand1->shape.size() == operand2->shape.size() && operand1->shape.size() == NUM_VALUE_3);
    assert(operand1->shape[2] == operand2->shape[1]);
    const int batchSizeA = operand1->shape[0];
    const int batchSizeB = operand2->shape[0];
    assert(batchSizeA == batchSizeB || batchSizeB == 1 || batchSizeA == 1);

    const int formatDimM = operand1->shape[1];
    const int formatDimK = operand1->shape[2];
    const int formatDimN = operand2->shape[2];
    int batchSize = batchSizeA > batchSizeB ? batchSizeA : batchSizeB;
    auto operand2D1 = Reshape(operand1, {batchSizeA * formatDimM, formatDimK});
    auto operand2D2 = Reshape(operand2, {batchSizeB * formatDimK, formatDimN});
    Tensor result(dataType, {batchSize * formatDimM, formatDimN});
    auto &curFunc = *Program::GetInstance().GetCurrentFunction();
    for (int i = 0; i < batchSize; i++) {
        int offsetA = batchSizeA == 1 ? 0 : i * formatDimM;
        int offsetB = batchSizeB == 1 ? 0 : i * formatDimK;
        int offsetC = i * formatDimM;
        auto tensorA = operand2D1->View(curFunc, {formatDimM, formatDimK}, {offsetA, 0});
        auto tensorB = operand2D2->View(curFunc, {formatDimK, formatDimN}, {offsetB, 0});
        auto tensorC = result->View(curFunc, {formatDimM, formatDimN}, {offsetC, 0});
        MatmulImpl(dataType, {tensorA, tensorB}, tensorC);
    }
    return Reshape(result, {batchSize, formatDimM, formatDimN});
};

Tensor ABatchMulB4D(DataType dataType, const Tensor &operand1, const Tensor &operand2) {
    assert(operand1->shape.size() == SHAPE_DIM4 && operand2->shape.size() == SHAPE_DIM4);
    assert(operand1->shape[NUM_VALUE_3] == operand2->shape[2]);

    const int batchSizeA1 = operand1->shape[0];
    const int batchSizeA2 = operand1->shape[1];
    const int batchSizeB1 = operand2->shape[0];
    const int batchSizeB2 = operand2->shape[1];
    assert(batchSizeA1 == batchSizeB1 || batchSizeB1 == 1 || batchSizeA1 == 1);
    assert(batchSizeA2 == batchSizeB2 || batchSizeB2 == 1 || batchSizeA2 == 1);

    const int formatDimM = operand1->shape[2];
    const int formatDimK = operand1->shape[NUM_VALUE_3];
    const int formatDimN = operand2->shape[NUM_VALUE_3];
    auto operand2D1 = Reshape(operand1, {batchSizeA1 * batchSizeA2 * formatDimM, formatDimK});
    auto operand2D2 = Reshape(operand2, {batchSizeB1 * batchSizeB2 * formatDimK, formatDimN});
    int batchSize1 = batchSizeA1 > batchSizeB1 ? batchSizeA1 : batchSizeB1;
    int batchSize2 = batchSizeA2 > batchSizeB2 ? batchSizeA2 : batchSizeB2;
    Tensor result(dataType, {batchSize1 * batchSize2 * formatDimM, formatDimN});

    int strideA = batchSizeA2 == 1 ? 0 : formatDimM;
    int strideB = batchSizeB2 == 1 ? 0 : formatDimK;
    int offsetC = 0;
    auto &curFunc = *Program::GetInstance().GetCurrentFunction();
    for (int i = 0; i < batchSize1; i++) {
        int offsetA = batchSizeA1 == 1 ? 0 : i * batchSizeA2 * formatDimM;
        int offsetB = batchSizeB1 == 1 ? 0 : i * batchSizeB2 * formatDimK;
        for (int j = 0; j < batchSize2; j++) {
            auto tensorA = operand2D1->View(curFunc, {formatDimM, formatDimK}, {offsetA, 0});
            auto tensorB = operand2D2->View(curFunc, {formatDimK, formatDimN}, {offsetB, 0});
            auto tensorC = result->View(curFunc, {formatDimM, formatDimN}, {offsetC, 0});
            MatmulImpl(dataType, {tensorA, tensorB}, tensorC);
            offsetC += formatDimM;
            offsetA += strideA;
            offsetB += strideB;
        }
    }
    return Reshape(result, {batchSize1, batchSize2, formatDimM, formatDimN});
};

Tensor ABatchMulBT3D(DataType dataType, const Tensor &operand1, const Tensor &operand2) {
    assert(operand1->shape.size() == SHAPE_DIM3 && operand2->shape.size() == SHAPE_DIM3);
    assert(operand1->shape[2] == operand2->shape[2]);

    const int batchSizeA = operand1->shape[0];
    const int batchSizeB = operand2->shape[0];
    assert(batchSizeA == batchSizeB || batchSizeB == 1 || batchSizeA == 1);

    auto &curFunc = *Program::GetInstance().GetCurrentFunction();

    const int formatDimM = operand1->shape[1];
    const int formatDimK = operand1->shape[2];
    const int formatDimN = operand2->shape[1];
    auto operand2D1 = Reshape(operand1, {batchSizeA * formatDimM, formatDimK});
    auto operand2D2 = Reshape(operand2, {batchSizeB * formatDimN, formatDimK});
    int batchSize = batchSizeA > batchSizeB ? batchSizeA : batchSizeB;
    Tensor result(dataType, {batchSize * formatDimM, formatDimN});
    for (int i = 0; i < batchSize; i++) {
        int offsetA = batchSizeA == 1 ? 0 : i * formatDimM;
        int offsetB = batchSizeB == 1 ? 0 : i * formatDimN;
        int offsetC = i * formatDimM;
        auto tensorA = operand2D1->View(curFunc, {formatDimM, formatDimK}, {offsetA, 0});
        auto tensorB = operand2D2->View(curFunc, {formatDimN, formatDimK}, {offsetB, 0});
        auto tensorC = result->View(curFunc, {formatDimM, formatDimN}, {offsetC, 0});
        AMulBtImpl(dataType, tensorA, tensorB, tensorC);
    }
    return Reshape(result, {batchSize, formatDimM, formatDimN});
};

Tensor ABatchMulBT4D(DataType dataType, const Tensor &operand1, const Tensor &operand2) {
    assert(operand1->shape.size() == NUM_VALUE_4 && operand2->shape.size() == NUM_VALUE_4); // 两个输入都只支持4维的
    assert(operand1->shape[NUM_VALUE_3] == operand2->shape[NUM_VALUE_3]);

    const int batchSizeA1 = operand1->shape[0];
    const int batchSizeA2 = operand1->shape[1];
    const int batchSizeB1 = operand2->shape[0];
    const int batchSizeB2 = operand2->shape[1];
    assert(batchSizeA1 == batchSizeB1 || batchSizeB1 == 1 || batchSizeA1 == 1);
    assert(batchSizeA2 == batchSizeB2 || batchSizeB2 == 1 || batchSizeA2 == 1);

    const int formatDimM = operand1->shape[2];
    const int formatDimK = operand1->shape[NUM_VALUE_3];
    const int formatDimN = operand2->shape[2];
    auto operand2D1 = Reshape(operand1, {batchSizeA1 * batchSizeA2 * formatDimM, formatDimK});
    auto operand2D2 = Reshape(operand2, {batchSizeB1 * batchSizeB2 * formatDimN, formatDimK});
    int batchSize1 = batchSizeA1 > batchSizeB1 ? batchSizeA1 : batchSizeB1;
    int batchSize2 = batchSizeA2 > batchSizeB2 ? batchSizeA2 : batchSizeB2;
    std::vector<std::pair<Tensor, std::vector<int>>> aggregation;
    Tensor result(dataType, {batchSize1 * batchSize2 * formatDimM, formatDimN});

    auto &curFunc = *Program::GetInstance().GetCurrentFunction();

    int strideA = batchSizeA2 == 1 ? 0 : formatDimM;
    int strideB = batchSizeB2 == 1 ? 0 : formatDimN;
    int offsetC = 0;
    for (int i = 0; i < batchSize1; i++) {
        int offsetA = batchSizeA1 == 1 ? 0 : i * batchSizeA2 * formatDimM;
        int offsetB = batchSizeB1 == 1 ? 0 : i * batchSizeB2 * formatDimN;
        for (int j = 0; j < batchSize2; j++) {
            auto tensorA = operand2D1->View(curFunc, {formatDimM, formatDimK}, {offsetA, 0});
            auto tensorB = operand2D2->View(curFunc, {formatDimN, formatDimK}, {offsetB, 0});
            auto tensorC = result->View(curFunc, {formatDimM, formatDimN}, {offsetC, 0});
            AMulBtImpl(dataType, tensorA, tensorB, tensorC);
            offsetC += formatDimM;
            offsetA += strideA;
            offsetB += strideB;
        }
    }
    return Reshape(result, {batchSize1, batchSize2, formatDimM, formatDimN});
};

template<bool isATrans, bool isBTrans>
Tensor BatchMatmul(DataType dataType, const Tensor& aMatrix, const Tensor& bMatrix) {
    DECLARE_TRACER();
    assert(aMatrix.GetShape().size() == bMatrix.GetShape().size());
    Tensor res;
    if constexpr (!isATrans && isBTrans) {
        if (aMatrix.GetShape().size() == SHAPE_DIM4) {
            res = ABatchMulBT4D(dataType, aMatrix, bMatrix);
        } else {
            res = ABatchMulBT3D(dataType, aMatrix, bMatrix);
        }
    } else if constexpr (!isATrans && !isBTrans){
        if (aMatrix.GetShape().size() == SHAPE_DIM4) {
            res = ABatchMulB4D(dataType, aMatrix, bMatrix);
        } else {
            res = ABatchMulB3D(dataType, aMatrix, bMatrix);
        }
    } else {
        assert("only support B trans currently!");
    }
    return res;
}

template Tensor BatchMatmul<false, false>(DataType, const Tensor&, const Tensor&);
template Tensor BatchMatmul<false, true>(DataType, const Tensor&, const Tensor&);
} // namespace npu::tile_fwk::Matrix

static void BinaryOperationOperandCheck(const std::vector<LogicalTensorPtr> &iOperand,
    const std::vector<LogicalTensorPtr> &oOperand) {
    constexpr size_t inOpSize = 2;
    constexpr size_t outOpSize = 1;
    ASSERT(iOperand.size() == inOpSize && "iOperand size should be 2");
    ASSERT(oOperand.size() == outOpSize && "oOperand size should be 1");
}

static void UnaryOperationOperandCheck(const std::vector<LogicalTensorPtr> &iOperand,
    const std::vector<LogicalTensorPtr> &oOperand) {
    ASSERT(iOperand.size() == 1);
    ASSERT(oOperand.size() == 1);
}

static void CastOperationOperandCheck(const std::vector<LogicalTensorPtr> &iOperand,
    const std::vector<LogicalTensorPtr> &oOperand) {
    ASSERT(iOperand.size() == 1);
    ASSERT(oOperand.size() == 1);
}

void npu::tile_fwk::ExpandOperationInto(Function &function, const TileShape &tileShape, Opcode opCode,
    const std::vector<LogicalTensorPtr> &iOperand,
    const std::vector<LogicalTensorPtr> &oOperand, const Operation &op) {
    switch (opCode) {
        case Opcode::OP_ADDS: {
            TiledBinaryOperationScalar<BinaryOpType::ADD>(function, tileShape, iOperand[0],
                op.GetElementAttribute(OpAttributeKey::scalar), oOperand[0]);
            break;
        }
        case Opcode::OP_MULS: {
            TiledBinaryOperationScalar<BinaryOpType::MUL>(function, tileShape, iOperand[0],
                op.GetElementAttribute(OpAttributeKey::scalar), oOperand[0]);
            break;
        }
        case Opcode::OP_SUBS: {
            TiledBinaryOperationScalar<BinaryOpType::SUB>(function, tileShape, iOperand[0],
            op.GetElementAttribute(OpAttributeKey::scalar), oOperand[0]);
            break;
        }
        case Opcode::OP_DIVS: {
            TiledBinaryOperationScalar<BinaryOpType::DIV>(function, tileShape, iOperand[0],
                op.GetElementAttribute(OpAttributeKey::scalar), oOperand[0]);
            break;
        }
        case Opcode::OP_S_ADDS: {
            TiledBinaryOperationAllScalar<BinaryOpType::S_ADD>(function, tileShape, iOperand[0],
                op.GetElementAttribute(OpAttributeKey::scalar), oOperand[0],
                op.GetBoolAttribute(OP_ATTR_PREFIX + "reverseOperand"));
            break;
        }
        case Opcode::OP_S_MULS: {
            TiledBinaryOperationAllScalar<BinaryOpType::S_MUL>(function, tileShape, iOperand[0],
            op.GetElementAttribute(OpAttributeKey::scalar), oOperand[0],
            op.GetBoolAttribute(OP_ATTR_PREFIX + "reverseOperand"));
            break;
        }
        case Opcode::OP_S_SUBS: {
            TiledBinaryOperationAllScalar<BinaryOpType::S_SUB>(function, tileShape, iOperand[0],
                op.GetElementAttribute(OpAttributeKey::scalar), oOperand[0],
                op.GetBoolAttribute(OP_ATTR_PREFIX + "reverseOperand"));
            break;
        }
        case Opcode::OP_S_DIVS: {
            TiledBinaryOperationAllScalar<BinaryOpType::S_DIV>(function, tileShape, iOperand[0],
                op.GetElementAttribute(OpAttributeKey::scalar), oOperand[0],
                op.GetBoolAttribute(OP_ATTR_PREFIX + "reverseOperand"));
            break;
        }
        case Opcode::OP_S_MAXS: {
            TiledBinaryOperationAllScalar<BinaryOpType::S_MAX>(function, tileShape, iOperand[0],
                op.GetElementAttribute(OpAttributeKey::scalar), oOperand[0],
                op.GetBoolAttribute(OP_ATTR_PREFIX + "reverseOperand"));
            break;
        }
        case Opcode::OP_ADD: {
            BinaryOperationOperandCheck(iOperand, oOperand);
            TiledBinaryOperation<BinaryOpType::ADD>(function, tileShape, iOperand[0], iOperand[1], oOperand[0]);
            break;
        }
        case Opcode::OP_MUL: {
            BinaryOperationOperandCheck(iOperand, oOperand);
            TiledBinaryOperation<BinaryOpType::MUL>(function, tileShape, iOperand[0], iOperand[1], oOperand[0]);
            break;
        }
        case Opcode::OP_SUB: {
            BinaryOperationOperandCheck(iOperand, oOperand);
            TiledBinaryOperation<BinaryOpType::SUB>(function, tileShape, iOperand[0], iOperand[1], oOperand[0]);
            break;
        }
        case Opcode::OP_DIV: {
            BinaryOperationOperandCheck(iOperand, oOperand);
            TiledBinaryOperation<BinaryOpType::DIV>(function, tileShape, iOperand[0], iOperand[1], oOperand[0]);
            break;
        }
        case Opcode::OP_S_ADD: {
            BinaryOperationOperandCheck(iOperand, oOperand);
            TiledBinaryOperationAllScalar<BinaryOpType::S_ADD>(function, tileShape, iOperand[0], iOperand[1], oOperand[0]);
            break;
        }
        case Opcode::OP_S_MUL: {
            BinaryOperationOperandCheck(iOperand, oOperand);
            TiledBinaryOperationAllScalar<BinaryOpType::S_MUL>(function, tileShape, iOperand[0], iOperand[1], oOperand[0]);
            break;
        }
        case Opcode::OP_S_SUB: {
            BinaryOperationOperandCheck(iOperand, oOperand);
            TiledBinaryOperationAllScalar<BinaryOpType::S_SUB>(function, tileShape, iOperand[0], iOperand[1], oOperand[0]);
            break;
        }
        case Opcode::OP_S_DIV: {
            BinaryOperationOperandCheck(iOperand, oOperand);
            TiledBinaryOperationAllScalar<BinaryOpType::S_DIV>(function, tileShape, iOperand[0], iOperand[1], oOperand[0]);
            break;
        }
        case Opcode::OP_S_MAX: {
            BinaryOperationOperandCheck(iOperand, oOperand);
            TiledBinaryOperationAllScalar<BinaryOpType::S_MAX>(function, tileShape, iOperand[0], iOperand[1], oOperand[0]);
            break;
        }
        case Opcode::OP_MAXIMUM: {
            BinaryOperationOperandCheck(iOperand, oOperand);
            TiledBinaryOperation<BinaryOpType::MAXIMUM>(function, tileShape, iOperand[0], iOperand[1], oOperand[0]);
            break;
        }
        case Opcode::OP_EXP: {
            UnaryOperationOperandCheck(iOperand, oOperand);
            TiledUnaryOperation<UnaryOpType::EXP>(function, tileShape, iOperand[0], oOperand[0]);
            break;
        }
        case Opcode::OP_SQRT: {
            UnaryOperationOperandCheck(iOperand, oOperand);
            TiledUnaryOperation<UnaryOpType::SQRT>(function, tileShape, iOperand[0], oOperand[0]);
            break;
        }
        case Opcode::OP_RECIPROCAL: {
            UnaryOperationOperandCheck(iOperand, oOperand);
            TiledUnaryOperation<UnaryOpType::RECIPROCAL>(function, tileShape, iOperand[0], oOperand[0]);
            break;
        }
        case Opcode::OP_ABS: {
            UnaryOperationOperandCheck(iOperand, oOperand);
            TiledUnaryOperation<UnaryOpType::ABS>(function, tileShape, iOperand[0], oOperand[0]);
            break;
        }
        case Opcode::OP_GATHER: {
            int axis = op.GetIntAttribute(OP_ATTR_PREFIX + "axis");
            TiledGatherOperation(function, tileShape, iOperand[0], iOperand[1], axis, oOperand[0]);
            break;
        }
        case Opcode::OP_GATHER_ELEMENT: {
            int axis = op.GetIntAttribute(OP_ATTR_PREFIX + "axis");
            TiledGatherElementOperation(function, tileShape, iOperand[0], iOperand[1], axis, oOperand[0]);
            break;
        }
        case Opcode::OP_SCATTER_ELEMENT: {
            int axis = op.GetIntAttribute(OP_ATTR_PREFIX + "axis");
            Element scalar = op.GetElementAttribute(OpAttributeKey::scalar);
            TiledScatterElement(function, tileShape, {oOperand[0], iOperand[0], iOperand[1], scalar, axis});
            break;
        }
        case Opcode::OP_INDEX_PUT: {
            TiledScatterUpdate(function, tileShape, oOperand[0], iOperand[0], iOperand[1], iOperand[2], 0, "PA_BNSD", 1);
            break;
        }
        case Opcode::OP_INDEX_OUTCAST: {
            int axis = op.GetIntAttribute("axis");
            int blockSize = op.GetIntAttribute(OpAttributeKey::panzBlockSize);
            std::string cacheMode = op.GetStringAttribute(OpAttributeKey::cacheMode);
            TiledScatterUpdate(function, tileShape, oOperand[0], iOperand[0], iOperand[1], iOperand[2], axis, cacheMode, blockSize);
            break;
        }
        case Opcode::OP_CONCAT: {
            auto tensorOffset = op.GetVectorIntAttribute("concat");
            TiledInnerConcat(function, tileShape, iOperand[0], oOperand[0], tensorOffset);
            break;
        }
        case Opcode::OP_CAST: {
            CastOperationOperandCheck(iOperand, oOperand);
            auto mode = op.GetCastModeAttribute(OP_ATTR_PREFIX + "mode");
            TiledCastOperation<CastOpType::CAST>(function, tileShape, iOperand[0], oOperand[0], mode);
            break;
        }
        case Opcode::OP_EXPAND: {
            UnaryOperationOperandCheck(iOperand, oOperand);
            TiledExpand(function, tileShape, iOperand[0], oOperand[0]);
            break;
        }
        case Opcode::OP_ROWEXPMAX: {
            UnaryOperationOperandCheck(iOperand, oOperand);
            TiledReduceExpand(function, tileShape, "MAX", iOperand[0], oOperand[0]);
            break;
        }
        case Opcode::OP_ROWEXPSUM: {
            UnaryOperationOperandCheck(iOperand, oOperand);
            TiledReduceExpand(function, tileShape, "SUM", iOperand[0], oOperand[0]);
            break;
        }
        case Opcode::OP_ROWMAX_SINGLE: {
            UnaryOperationOperandCheck(iOperand, oOperand);
            TiledReduceSingle(function, tileShape, "MAX", iOperand[0], oOperand[0]);
            break;
        }
        case Opcode::OP_ROWSUM_SINGLE: {
            UnaryOperationOperandCheck(iOperand, oOperand);
            auto axis = op.GetIntAttribute(OP_ATTR_PREFIX + "AXIS");
            TiledReduceSingle(function, tileShape, "SUM", iOperand[0], oOperand[0], axis);
            break;
        }
        case Opcode::OP_ROWMAX_COMBINE_AXIS_SINGLE: {
            UnaryOperationOperandCheck(iOperand, oOperand);
            auto axis = op.GetIntAttribute(OP_ATTR_PREFIX + "AXIS");
            TiledReduceSingle(function, tileShape, "MAX_COMBINE_AXIS", iOperand[0], oOperand[0], axis);
            break;
        }
        case Opcode::OP_ROWSUM_COMBINE_AXIS_SINGLE: {
            UnaryOperationOperandCheck(iOperand, oOperand);
            auto axis = op.GetIntAttribute(OP_ATTR_PREFIX + "AXIS");
            TiledReduceSingle(function, tileShape, "SUM_COMBINE_AXIS", iOperand[0], oOperand[0], axis);
            break;
        }
        case Opcode::OP_TRANSPOSE_DATAMOVE: {
            auto shape = op.GetVectorIntAttribute(OP_ATTR_PREFIX + "shape");
            TiledInnerTranspose<TransposeOpType::TRANSPOSE_DATAMOVE>(function, tileShape, iOperand[0], oOperand[0], shape);
            break;
        }
        case Opcode::OP_TRANSPOSE_VNCHWCONV: {
            auto shape = op.GetVectorIntAttribute(OP_ATTR_PREFIX + "shape");
            TiledInnerTranspose<TransposeOpType::TRANSPOSE_VNCHWCONV>(function, tileShape, iOperand[0], oOperand[0], shape);
            break;
        }
        case Opcode::OP_RESHAPE: {
            TiledInnerReshape(function, iOperand[0], oOperand[0]);
            break;
        }
        case Opcode::OP_REGISTER_COPY: {
            UnaryOperationOperandCheck(iOperand, oOperand);
            TiledInnerRegisterCopy(function, tileShape, iOperand[0], oOperand[0]);
            break;
        }
        case Opcode::OP_A_MUL_B: {
            auto mValue = (op.HasAttr(OP_ATTR_PREFIX + "act_m")) ? op.GetIntAttribute(OP_ATTR_PREFIX + "act_m") : 0;
            auto kValue = (op.HasAttr(OP_ATTR_PREFIX + "act_k")) ? op.GetIntAttribute(OP_ATTR_PREFIX + "act_k") : 0;
            auto nValue = (op.HasAttr(OP_ATTR_PREFIX + "act_n")) ? op.GetIntAttribute(OP_ATTR_PREFIX + "act_n") : 0;
            Matrix::TiledInnerAMulB(function, tileShape, iOperand, oOperand[0], {mValue, kValue, nValue});
            break;
        }
        case Opcode::OP_A_MUL_BT: {
            auto mValue = (op.HasAttr(OP_ATTR_PREFIX + "act_m")) ? op.GetIntAttribute(OP_ATTR_PREFIX + "act_m") : 0;
            auto kValue = (op.HasAttr(OP_ATTR_PREFIX + "act_k")) ? op.GetIntAttribute(OP_ATTR_PREFIX + "act_k") : 0;
            auto nValue = (op.HasAttr(OP_ATTR_PREFIX + "act_n")) ? op.GetIntAttribute(OP_ATTR_PREFIX + "act_n") : 0;
            Matrix::TiledInnerAMulB<false, true>(function, tileShape, iOperand, oOperand[0], {mValue, kValue, nValue});
            break;
        }
        case Opcode::OP_BITSORT: {
            int axis = op.GetIntAttribute(TOPK_AXIS);
            int isLargest = op.GetIntAttribute(TOPK_ORDER);
            TiledBitSort(function, tileShape, iOperand[0], oOperand[0], axis, isLargest);
            break;
        }
        case Opcode::OP_MRGSORT: {
            int axis = op.GetIntAttribute(TOPK_AXIS);
            int kValue = op.GetIntAttribute(TOPK_KVALUE);
            int isLargest = op.GetIntAttribute(TOPK_ORDER);
            TiledMrgSort(function, tileShape, iOperand[0], oOperand[0], axis, kValue, isLargest);
            break;
        }
        case Opcode::OP_ARGSORT: {
            int axis = op.GetIntAttribute("axis");
            int isLargest = op.GetIntAttribute("order");
            TiledArgSort(function, tileShape, iOperand[0], oOperand[0], axis, isLargest);
            break;
        }
        case Opcode::OP_EXTRACT: {
            int maskMode = op.GetIntAttribute(EXTRACT_MASKMODE);
            int kValue = op.GetIntAttribute(TOPK_KVALUE);
            int isLargest = op.GetIntAttribute(TOPK_ORDER);
            TiledExtract(function, tileShape, iOperand[0], oOperand[0], maskMode, kValue, isLargest);
            break;
        }
        case Opcode::OP_VEC_DUP: {
            Element scalar = op.GetElementAttribute(OpAttributeKey::scalar);
            SymbolicScalar dynScalar;
            if (op.HasAttr(OpAttributeKey::dynScalar)) {
                dynScalar = op.GetSymbolicScalarAttribute(OpAttributeKey::dynScalar);
            }
            std::vector<int> shape = op.GetVectorIntAttribute(OP_ATTR_PREFIX + "shape");
            std::vector<SymbolicScalar> validShape;
            op.GetAttr(OP_ATTR_PREFIX + "validShape", validShape);
            TiledVecDup(function, tileShape, scalar, dynScalar, shape, validShape, oOperand[0]);
            break;
        }
        case Opcode::OP_DIST_SCATTER: {
            npu::tile_fwk::Distributed::TiledDistScatter(function, tileShape, iOperand, op);
            break;
        }
        case Opcode::OP_DIST_REDUCE: {
            npu::tile_fwk::Distributed::TiledDistReduce(function, tileShape, iOperand, oOperand, op);
            break;
        }
        case Opcode::OP_DIST_GATHER: {
            npu::tile_fwk::Distributed::TiledDistGather(function, tileShape, iOperand, oOperand, op);
            break;
        }
        case Opcode::OP_DIST_BROADCAST: {
            npu::tile_fwk::Distributed::TiledDistBroadCast(function, tileShape, iOperand, oOperand, op);
            break;
        }
        case Opcode::OP_MOE_FFN_TO_ATTN: {
            npu::tile_fwk::Distributed::TiledMoeFFN2Attn(function, tileShape, iOperand, op);
            break;
        }
        case Opcode::OP_MOE_ATTN_COMBINE: {
            npu::tile_fwk::Distributed::TiledMoeAttnCombine(function, tileShape, iOperand, oOperand, op);
            break;
        }
        case Opcode::OP_REDUCE_ACC: {
            TiledReduceAcc(function, tileShape, iOperand, oOperand[0]);
            break;
        }
        case Opcode::OP_ASSEMBLE: {
            auto assembleOpAttribute = dynamic_cast<AssembleOpAttribute *>(op.GetOpAttribute().get());
            TiledAssemble(function, tileShape, iOperand[0], oOperand[0], assembleOpAttribute);
            break;
        }
        case Opcode::OP_MAX_POOL: {
            TiledMaxpool(function, tileShape, iOperand[0], oOperand[0], op);
            break;
        }
        case Opcode::OP_SEND_TO_ROUTING_EXPERT: {
            npu::tile_fwk::Distributed::TiledSendToRoutingExpert(function, tileShape, iOperand, oOperand, op);
            break;
        }
        case Opcode::OP_SEND_TO_SHARED_EXPERT: {
            npu::tile_fwk::Distributed::TiledSendToSharedExpert(function, tileShape, iOperand, oOperand, op);
            break;
        }
        case Opcode::OP_COPY_TO_LOCAL_EXPERT: {
            npu::tile_fwk::Distributed::TiledCopyToLocalExpert(function, tileShape, iOperand, oOperand, op);
            break;
        }
        case Opcode::OP_DISPATCH_SET_FLAG: {
            npu::tile_fwk::Distributed::TiledDispatchSetFlag(function, tileShape, iOperand, oOperand, op);
            break;
        }
        case Opcode::OP_FFN_SCHED: {
            npu::tile_fwk::Distributed::TiledDispatchFFNSched(function, tileShape, iOperand, oOperand, op);
            break;
        }
        case Opcode::OP_FFN_BATCHING: {
            npu::tile_fwk::Distributed::TiledDispatchFFNBatching(function, tileShape, iOperand, oOperand, op);
            break;
        }
        default: {
            ASLOGE("OpCode is %d", static_cast<int>(opCode));
            ASSERT(false) << "unsupported now";
        }
    }
}
