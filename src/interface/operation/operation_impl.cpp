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
#include "passes/pass_utils/graph_utils.h"

using namespace npu::tile_fwk;

namespace {

struct TileInfo {
    std::vector<int64_t> shape;
    std::vector<int64_t> offset;
    std::vector<SymbolicScalar> validShape;

    TileInfo(size_t shapeSize, size_t offsetSize) : shape(shapeSize), offset(offsetSize), validShape(shapeSize) {}

    TileInfo(std::vector<int64_t> aShape, std::vector<int64_t> aOffset, std::vector<SymbolicScalar> aValidShape = {})
        : shape(std::move(aShape)), offset(std::move(aOffset)), validShape(aValidShape) {}
};

struct Input {
    const Tensor tensor;
    TileInfo tileInfo;
};

enum class LogicalNotOpType {
    OP_LOGICALNOT,
};

enum class TransposeOpType {
    TRANSPOSE_MOVEIN,
    TRANSPOSE_MOVEOUT,
    TRANSPOSE_VNCHWCONV,
};

template <TransposeOpType T>
Opcode GetTransposeOpName() {
#define CASE(X) \
case TransposeOpType::X: return Opcode::OP_##X
    switch (T) {
        CASE(TRANSPOSE_MOVEIN);
        CASE(TRANSPOSE_MOVEOUT);
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
        case UnaryOpType::NEG:
            return "NEG";
        case UnaryOpType::RSQRT:
            return "RSQRT";
        case UnaryOpType::SQRT:
            return "SQRT";
        case UnaryOpType::RECIPROCAL:
            return "RECIPROCAL";
        case UnaryOpType::DUPLICATE:
            return "DUPLICATE";
        case UnaryOpType::ABS:
            return "ABS";
        case UnaryOpType::LN:
            return "LN";
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
        CASE(NEG);
        CASE(RSQRT);
        CASE(SQRT);
        CASE(RECIPROCAL);
        CASE(ABS);
        CASE(LN);
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
        case BinaryOpType::MAX:
            return "MAX";
        case BinaryOpType::MIN:
            return "MIN";
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
            CASE(MAX);
            CASE(MIN);
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
    std::vector<int64_t> opShape1(operand1->shape);
    std::vector<int64_t> opShape2(operand2->shape);
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

LogicalTensorPtr BinaryOperationBroadCast(const LogicalTensorPtr &operand, const std::vector<int> &broadCastShape) {
    if (operand->shape.size() < broadCastShape.size()) {
        auto broadCastDims = broadCastShape.size() - operand->shape.size();
        std::vector<int64_t> unsqueezeShape(operand->shape);
        unsqueezeShape.insert(unsqueezeShape.begin(), broadCastDims, 1);
        auto tmpOperand = Reshape(operand, unsqueezeShape).GetStorage();
        return tmpOperand;
    }
    return operand;
}

inline const std::vector<size_t> &GetShapeLenLimit(const std::string &op) {
    // if the limit of op is not [1, 4], should add here
    static std::unordered_map<std::string, const std::vector<size_t>> op_shape_len_limit = {
        {    "ADD", {1, 4}},
        {   "CAST", {1, 4}},
        {"DEFAULT", {1, 4}}
    };
    if (op_shape_len_limit.find(op) == op_shape_len_limit.end()) {
        return op_shape_len_limit.at("DEFAULT");
    }
    return op_shape_len_limit.at(op);
}

inline void CheckTensorShape(const LogicalTensorPtr &tensor, const std::string &op) {
    auto shape = tensor->shape;
    // valid input dims must in [1, 4]
    auto shape_len_limit = GetShapeLenLimit(op);
    if (shape.size() < shape_len_limit[0] || shape.size() > shape_len_limit[1]) {
        assert(false && "The dims of tensor out of range.");
    }
    size_t shapeSize = 1;
    for (const auto &value : shape) {
        if (value > INT32_MAX) {
            assert(false && "The dim value of tensor must less than or equal to INT32_MAX(2,147,483,647)");
        }
        shapeSize *= static_cast<size_t>(value);
        if (shapeSize > INT32_MAX) {
            assert(false && "The shape size of tensor must less than or equal to INT32_MAX(2,147,483,647)");
        }
    }
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

inline void CheckBinaryInputTensors(const LogicalTensorPtr &tensor1, const LogicalTensorPtr &tensor2, std::string &op) {
    CheckTensorShape(tensor1, op);
    CheckTensorShape(tensor2, op);
    CheckBinOpOperandsValid(tensor1, tensor2);
    if (tensor1->Datatype() != tensor2->Datatype()) {
        assert(false && "The dtype of input tensors are not same.");
    }
}

template <UnaryOpType T>
Tensor UnaryOperation(Tensor operand) {
    auto opName = GetUnaryOpName<T>();
    CheckTensorShape(operand.GetStorage(), opName);
    Tensor result(operand->tensor->datatype, operand->shape);
    assert(operand->shape.size() == operand->offset.size());
    Program::GetInstance().AddOperation(GetUnaryOpName<T>(), {operand.GetStorage()}, {result.GetStorage()});
    return result;
}

template <UnaryOpType T>
LogicalTensorPtr TensorUnaryOperation(Function &function, LogicalTensorPtr operand) {
    auto opName = GetUnaryOpName<T>();
    CheckTensorShape(operand, opName);
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

    std::vector<int64_t> srcShape(expandInfo.srcTensor->shape.size(), 1);
    for (size_t i = 0; i < expandInfo.result->shape.size(); i++) {
        srcShape[i] = std::min(expandInfo.viewShape[i], expandInfo.srcTensor->shape[i]);
    }

    std::vector<int64_t> srcOffset = expandInfo.offset;
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

void ExpandTile(Function &function, const TileShape &tileShape, int dimIdx, const struct ExpandInfo &expandInfo,
    std::vector<SymbolicScalar> validShape) {
    if (static_cast<size_t>(dimIdx) == expandInfo.result->shape.size()) {
        ExpandTile(function, expandInfo);
        return;
    }
    auto &vecTile = tileShape.GetVecTile();
    for (int i = 0; i < expandInfo.result->shape[dimIdx]; i += vecTile[dimIdx]) {
        expandInfo.offset[dimIdx] = i;
        expandInfo.viewShape[dimIdx] =
            std::min(expandInfo.result->shape[dimIdx] - i, static_cast<int64_t>(vecTile[dimIdx]));
        ExpandTile(function, tileShape, dimIdx + 1, expandInfo, validShape);
    }
}

void Expand(Function &function, const TileShape &tileShape, const LogicalTensorPtr &operand,
    const LogicalTensorPtr &result) {
    CheckExpandTensorVaild(operand, result);
    ASSERT(function.GetGraphType() == GraphType::TILE_GRAPH);

    std::vector<int64_t> offset(result->shape.size(), 0);
    std::vector<int64_t> viewShape(result->shape.size(), 1);
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
    ExpandTile(function, tileShape, 0, expandInfo, outValidShape);
}

void Expand(Function &function, const TileShape &tileShape, const LogicalTensorPtr &operand,
    const LogicalTensorPtr &other, const LogicalTensorPtr &result) {
    CheckExpandTensorVaild(operand, result);
    ASSERT(function.GetGraphType() == GraphType::TILE_GRAPH);

    std::vector<int64_t> offset(result->shape.size(), 0);
    std::vector<int64_t> viewShape(result->shape.size(), 1);
    std::vector<SymbolicScalar> outValidShape;
    int expandDim = -1;
    for (size_t i = 0; i < result->shape.size(); ++i) {
        if (operand->shape[i] != result->shape[i]) {
            expandDim = i;
            outValidShape.push_back(other->GetDynValidShape()[i]);
        } else {
            outValidShape.push_back(operand->GetDynValidShape()[i]);
        }
    }

    result->UpdateDynValidShape(outValidShape);
    struct ExpandInfo expandInfo(operand, result, viewShape, offset, expandDim);
    ExpandTile(function, tileShape, 0, expandInfo, outValidShape);
}

void TiledExpand(Function &function, const TileShape &tileShape, const LogicalTensorPtr &operand,
    const LogicalTensorPtr &result, const std::vector<SymbolicScalar> &validShape) {
    CheckExpandTensorVaild(operand, result);
    ASSERT(function.GetGraphType() == GraphType::TILE_GRAPH);
    
    std::vector<int64_t> offset(result->shape.size(), 0);
    std::vector<int64_t> viewShape(result->shape.size(), 1);
    int expandDim = -1;
    for (size_t i = 0; i < result->shape.size(); ++i) {
        if (operand->shape[i] != result->shape[i]) {
            expandDim = i; 
        }
    }
    result->UpdateDynValidShape(validShape);
    struct ExpandInfo expandInfo(operand, result, viewShape, offset, expandDim);
    ExpandTile(function, tileShape, 0, expandInfo, validShape);
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

bool IsLastBrc(LogicalTensorPtr operand1, LogicalTensorPtr operand2) {
    assert(operand1->shape.size() == operand2->shape.size() && "Dims not match");
    size_t shapeSize = operand1->shape.size();
    return ((operand1->shape[shapeSize - 1] != 1) && (operand2->shape[shapeSize - 1] == 1)) ||
           ((operand2->shape[shapeSize - 1] != 1) && (operand1->shape[shapeSize - 1] == 1));
}

template <BinaryOpType T>
void TiledBinaryOperation(Function &function, const TileShape &tileShape, size_t cur, Input &input1,
    Input &input2, const LogicalTensorPtr &result, TileInfo &resultTileInfo, bool withBrc) {
    if (cur == input1.tensor->shape.size()) {
        auto inputTile1 = input1.tensor->View(function, input1.tileInfo.shape, input1.tileInfo.offset);
        auto inputTile2 = input2.tensor->View(function, input2.tileInfo.shape, input2.tileInfo.offset);
        auto resultTile = result->View(function, resultTileInfo.shape, resultTileInfo.offset);
        if(withBrc) {
            std::vector<int64_t> tmpShape(input1.tileInfo.shape);
            tmpShape[input1.tileInfo.shape.size() - 1] = BLOCK_SIZE / BytesOf(input2.tensor.GetDataType());
            auto tempTensor = std::make_shared<LogicalTensor>(function, input2.tensor->Datatype(), tmpShape);
            function.AddOperation(GetBinaryOpNameCode<T, false, true>(), {inputTile1, inputTile2}, {resultTile, tempTensor});
        } else {
            function.AddOperation(GetBinaryOpNameCode<T, false, false>(), {inputTile1, inputTile2}, {resultTile});
        }
        return;
    }
    auto &vecTile = tileShape.GetVecTile();
    for (int i = 0; i < result->shape[cur]; i += vecTile[cur]) {
        resultTileInfo.offset[cur] = i;
        resultTileInfo.shape[cur] = std::min(result->shape[cur] - resultTileInfo.offset[cur], vecTile[cur]);
        input1.tileInfo.offset[cur] = i % input1.tensor->shape[cur];
        input1.tileInfo.shape[cur] = std::min(input1.tensor->shape[cur] - input1.tileInfo.offset[cur], vecTile[cur]);
        input2.tileInfo.offset[cur] = i % input2.tensor->shape[cur];
        input2.tileInfo.shape[cur] = std::min(input2.tensor->shape[cur] - input2.tileInfo.offset[cur], vecTile[cur]);
        TiledBinaryOperation<T>(function, tileShape, cur + 1, input1, input2, result, resultTileInfo, withBrc);
    }
}

template <BinaryOpType T>
void TiledBinaryOperation(Function &function, const TileShape &tileShape, LogicalTensorPtr operand1,
    LogicalTensorPtr operand2, const LogicalTensorPtr &result) {
    CheckBinOpOperandsValid(operand1, operand2);
    bool withBrc = CallBrcBinOp(operand1, operand2) && ConfigManager::Instance().GetOperationConfig("FORCE_COMBINE_AXIS", false);
    // nolast brc will be inline
    if ((!withBrc) && IsLastBrc(operand1, operand2)) {
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

std::vector<int64_t> BinaryOperationResultShape(
    LogicalTensorPtr operand1, LogicalTensorPtr operand2) {
    std::vector<int64_t> resultShape(operand1->shape.size());
    for (size_t i = 0; i < resultShape.size(); i++) {
        resultShape[i] = std::max(operand1->shape[i], operand2->shape[i]);
    }
    return resultShape;
}

void TiledCompareOperationImpl(Function &function, const TileShape &tileShape, size_t cur, Input &input1, Input &input2, 
    const LogicalTensorPtr & result, TileInfo &resultTileInfo, CmpOperationType operation, CmpModeType mode) 
{
    if (cur == result->shape.size()) {
        auto inputTile1 = input1.tensor->View(function, input1.tileInfo.shape, input1.tileInfo.offset);
        auto inputTile2 = input2.tensor->View(function, input2.tileInfo.shape, input2.tileInfo.offset);
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
        auto& op = function.AddOperation(Opcode::OP_CMP, {inputTile1, inputTile2}, 
            {resultTile, vcmpBitResultTensor, zeroCondTensor, oneCondTensor, vselResultTensor, startAddrUBTensor});

        op.SetAttribute(OP_ATTR_PREFIX + "cmp_operation", static_cast<int64_t>(operation));
        op.SetAttribute(OP_ATTR_PREFIX + "cmp_mode", static_cast<int64_t>(mode));
        return;
    }

    auto &vecTile = tileShape.GetVecTile();
    for (int i = 0; i < result->shape[cur]; i += vecTile[cur]) {
        resultTileInfo.offset[cur] = i;
        resultTileInfo.shape[cur] = std::min(result->shape[cur] - resultTileInfo.offset[cur], vecTile[cur]);
        input1.tileInfo.offset[cur] = i % input1.tensor->shape[cur];
        input1.tileInfo.shape[cur] = std::min(input1.tensor->shape[cur] - input1.tileInfo.offset[cur], vecTile[cur]);
        input2.tileInfo.offset[cur] = i % input2.tensor->shape[cur];
        input2.tileInfo.shape[cur] = std::min(input2.tensor->shape[cur] - input2.tileInfo.offset[cur], vecTile[cur]);
        TiledCompareOperationImpl(function, tileShape, cur + 1, input1, input2, result, resultTileInfo, 
            operation, mode);
    }
}

void TiledCompareOperation(Function &function, const TileShape &tileShape, LogicalTensorPtr operand1,
    LogicalTensorPtr operand2, const LogicalTensorPtr &result, CmpOperationType operation, CmpModeType mode)
{
    auto broadcastOperand = [&](LogicalTensorPtr &operand, LogicalTensorPtr &other) {
        auto dstShape = result->shape;
        if (mode == CmpModeType::BIT) {
            dstShape[dstShape.size() - 1] *= 8; // compare output 8 bit to 1 byte
        }
        if (operand->shape == dstShape) {
            return;
        }
        auto expanded = std::make_shared<LogicalTensor>(function, operand->Datatype(), dstShape);
        Expand(function, tileShape, operand, other, expanded);
        operand = expanded;
    };
    broadcastOperand(operand1, operand2);
    broadcastOperand(operand2, operand1);

    TileInfo tileInfo1(result->shape.size(), result->offset.size());
    TileInfo tileInfo2(result->shape.size(), result->offset.size());
    TileInfo resultTileInfo(result->shape.size(), result->offset.size());
    auto input1 = Input{operand1, tileInfo1};
    auto input2 = Input{operand2, tileInfo2};

    TiledCompareOperationImpl(function, tileShape, 0, input1, input2, result, resultTileInfo,
        operation, mode);
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
    auto &vecTile = tileShape.GetVecTile();
    for (int i = 0; i < input.tensor->shape[cur]; i += vecTile[cur]) {
        input.tileInfo.shape[cur] = std::min(input.tensor->shape[cur] - i, vecTile[cur]);
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
    auto &vecTile = tileShape.GetVecTile();
    for (int i = 0; i < input.tensor->shape[cur]; i += vecTile[cur]) {
        input.tileInfo.shape[cur] = std::min(input.tensor->shape[cur] - i, vecTile[cur]);
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

LogicalTensorPtr TensorCompareOperation(Function& function, const Tensor& operand1, const Tensor& operand2, 
    CmpOperationType operation, CmpModeType mode) 
{
    auto operandT1 = operand1.GetStorage();
    auto operandT2 = operand2.GetStorage();
    if (operandT1->shape.size() != operandT2->shape.size()) {
        std::vector<int> broadCastShape = GetBroadCastShape(operandT1, operandT2);
        operandT1 = BinaryOperationBroadCast(operandT1, broadCastShape);
        operandT2 = BinaryOperationBroadCast(operandT2, broadCastShape);
    }
    std::vector<SymbolicScalar> resultValidShape;
    std::vector<int64_t> resultShape = BinaryOperationResultShape(operandT1, operandT2);
    if(!operandT1->GetDynValidShape().empty() && !operandT2->GetDynValidShape().empty()) {
        for (size_t i = 0; i < resultShape.size(); ++i) {
            if (resultShape[i] == operandT1->shape[i]) {
                resultValidShape.push_back(operandT1->GetDynValidShape()[i]);
            } else {
                resultValidShape.push_back(operandT2->GetDynValidShape()[i]);
            }
        }
    }
    auto resultType = DT_BOOL;
    if (mode == CmpModeType::BIT) {
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
    auto& op = function.AddOperation(Opcode::OP_CMP, {operandT1, operandT2}, {result});
    op.SetAttribute(OP_ATTR_PREFIX + "cmp_operation", static_cast<int64_t>(operation));
    op.SetAttribute(OP_ATTR_PREFIX + "cmp_mode", static_cast<int64_t>(mode));
    return result;
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
    auto opName = GetBinaryOpName<T>();
    CheckBinaryInputTensors(oprandT1, oprandT2, opName);

    std::vector<SymbolicScalar> resultValidShape;
    std::vector<int64_t> resultShape = BinaryOperationResultShape(oprandT1, oprandT2);
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
    auto opName = GetBinaryOpName<T>();
    CheckTensorShape(operand1, opName);
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
    auto &vecTile = tileShape.GetVecTile();
    for (int i = 0; i < result->shape[cur]; i += vecTile[cur]) {
        resultTileInfo.offset[cur] = i;
        resultTileInfo.shape[cur] = std::min(result->shape[cur] - resultTileInfo.offset[cur], vecTile[cur]);
        input1.tileInfo.offset[cur] = i % input1.tensor->shape[cur];
        input1.tileInfo.shape[cur] = std::min(input1.tensor->shape[cur] - input1.tileInfo.offset[cur], vecTile[cur]);

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
    auto &vecTile = tileShape.GetVecTile();
    for (int i = 0; i < result->shape[cur]; i += vecTile[cur]) {
        resultTileInfo.offset[cur] = i;
        resultTileInfo.shape[cur] = std::min(result->shape[cur] - resultTileInfo.offset[cur], vecTile[cur]);
        input1.tileInfo.offset[cur] = i % input1.tensor->shape[cur];
        input1.tileInfo.shape[cur] = std::min(input1.tensor->shape[cur] - input1.tileInfo.offset[cur], vecTile[cur]);

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
    auto opName = GetBinaryOpName<T>();
    CheckBinaryInputTensors(operand1.GetStorage(), operand2.GetStorage(), opName);
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
    auto &vecTile = tileShape.GetVecTile();
    for (int i = 0; i < result->shape[cur]; i += vecTile[cur]) {
        resultTileInfo.offset[cur] = i;
        resultTileInfo.shape[cur] = std::min(result->shape[cur] - resultTileInfo.offset[cur], vecTile[cur]);
        input1.tileInfo.offset[cur] = i % input1.tensor->shape[cur];
        input1.tileInfo.shape[cur] = std::min(input1.tensor->shape[cur] - input1.tileInfo.offset[cur], vecTile[cur]);
        input2.tileInfo.offset[cur] = i % input2.tensor->shape[cur];
        input2.tileInfo.shape[cur] = std::min(input2.tensor->shape[cur] - input2.tileInfo.offset[cur], vecTile[cur]);
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
    std::vector<int64_t> tileAshape = in->shape;
    std::vector<int64_t> tileBshape = in->shape;
    std::vector<int64_t> regShape = in->shape;
    std::vector<int64_t> regAshape = in->shape;
    std::vector<int64_t> regBshape = in->shape;
    std::vector<int64_t> regOffset(regShape.size(), 0);
    std::vector<int64_t> tileAoffset(regShape.size(), 0);
    std::vector<int64_t> tileBoffset(regShape.size(), 0);

    std::vector<int64_t> remainderShape = in->shape;
    std::vector<int64_t> remainderOffset(remainderShape.size(), 0);

    auto opNew = op;
    if (opNew == "MAX_COMBINE_AXIS") {
        opNew = "MAX";
    }
    if (opNew == "SUM_COMBINE_AXIS") {
        opNew = "SUM";
    }

    auto source = std::make_shared<LogicalTensor>(function, in->tensor, in->offset, in->shape, in->GetDynValidShape(), in->nodetype);

    auto &vecTile = tileShape.GetVecTile();
    int64_t width = (source->shape[axis] + vecTile[axis] - 1) / vecTile[axis] * vecTile[axis]; // 向上对齐
    int padSize = width - source->shape[axis];
    int remainder = 0;

    int p2width = vecTile[axis];
    while (width >= p2width) {
        p2width = p2width << 1;
    }
    p2width = p2width >> 1;

    remainder = width - p2width;
    remainderShape[axis] = remainder;
    remainderOffset[axis] = p2width;

    width = p2width;

    while (width >= NUM2 * vecTile[axis]) // hierarchically pair wise reduce to a
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
        for (int j = 0; j < width; j += vecTile[axis]) {
            regAshape[axis] = vecTile[axis];
            regBshape[axis] = std::min(vecTile[axis], tileB->shape[axis] - j); // 带tail的部分
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
        for (int j = 0; j < width; j += vecTile[axis]) {
            regAshape[axis] = vecTile[axis];
            regBshape[axis] = std::min(vecTile[axis], tileRemainder->shape[axis] - j); // 带tail的部分
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
    regShape[axis] = std::min(in->shape[axis], vecTile[axis]);
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

            for (int j = 0; j < result->shape[1]; j += vecTile[1]) // duplicate to fill result tensor
            {
                regShape[0] = in->shape[0];
                regShape[1] = vecTile[1];

                regOffset[0] = 0;
                regOffset[1] = j;

                resultReg = result->View(function, regShape, regOffset);
                function.AddOperation("TILE_REGISTER_COPY", {resultReg1}, {resultReg});
            }
            break;
        }
        case npu::tile_fwk::ReduceType::SINGLE: {
            std::vector<int64_t> tmpShape = {1, static_cast<int>(BLOCK_SIZE / BytesOf(in->Datatype()))};
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
                tmpShape[1] = REPEAT_BYTE / BytesOf(in->Datatype());
            }
            if ((sourceReg->shape[0] % BLOCK_NUM == 0) &&
                ((vecTile[axis] == LEN1024 && sourceReg->shape[0] * NUM_VALUE_16 <= MAX_REPEAT) ||
                    (vecTile[axis] == LEN512 && sourceReg->shape[0] * BLOCK_NUM <= MAX_REPEAT))) {
                tmpShape[1] = vecTile[axis] / BLOCK_NUM;
            }
            unsigned tmpBufSize = tmpShape[0] * tmpShape[1] * BytesOf(in->Datatype());
            if (op == "SUM" && static_cast<size_t>(axis) == (in->shape.size() - 1)) {
                assert(tmpBufSize <= MAX_TMP_BUF_SHAPE);
            } else if (op != "SUM" && static_cast<size_t>(axis) == (in->shape.size() - 1)) {
                assert(tmpBufSize <= MAX_TMP_BUF_SHAPE * NUM2);
            }
            if (static_cast<size_t>(axis) == (in->shape.size() - 1)) {
                auto tempTensor = std::make_shared<LogicalTensor>(function, in->Datatype(), tmpShape);
                tempTensor->dynValidShape_ = SymbolicScalar::FromConcrete(tmpShape);
                auto &newOp = function.AddOperation("TILE_ROW" + op + "_SINGLE", {sourceReg}, {result, tempTensor});
                newOp.SetAttribute(OP_ATTR_PREFIX + "AXIS", axis);
            } else {
                auto &newOp = function.AddOperation("TILE_ROW" + op + "LINE", {sourceReg}, {result});
                newOp.SetAttribute(OP_ATTR_PREFIX + "AXIS", axis);
            }
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

    auto &vecTile = tileShape.GetVecTile();
    TileInfo tileInfo({vecTile[0], operand->shape[1]}, std::vector<int64_t>(operand->offset.size()));

    for (int i = 0; i < operand->shape[0]; i += vecTile[0]) {
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
    TileInfo &resultTileInfo, int axis, Function &function, const TileShape &tileShape, std::vector<int> order) {
    if (order[cur] == axis && cur < order.size() - 1) {
        std::swap(order[cur], order[cur + 1]);
    }
    if (order[cur] == axis) {
        auto inputTile = input.tensor->View(function, input.tileInfo.shape, input.tileInfo.offset);
        auto resultTile = result->View(function, resultTileInfo.shape, resultTileInfo.offset);
        TileReduceNew(function, tileShape, op, npu::tile_fwk::ReduceType::SINGLE, inputTile, resultTile, axis);
        return;
    }
    auto &vecTile = tileShape.GetVecTile();
    for (int i = 0; i < result->shape[order[cur]]; i += vecTile[order[cur]]) {
        resultTileInfo.offset[order[cur]] = i;
        resultTileInfo.shape[order[cur]] = std::min(result->shape[order[cur]] - resultTileInfo.offset[order[cur]],
            vecTile[order[cur]]);
        input.tileInfo.offset[order[cur]] = i % input.tensor->shape[order[cur]];
        input.tileInfo.shape[order[cur]] = std::min(input.tensor->shape[order[cur]] - input.tileInfo.offset[order[cur]],
            vecTile[order[cur]]);
        ReduceSingle(cur + 1, op, input, result, resultTileInfo, axis, function, tileShape, order);
    }
}

void TiledReduceSingle(Function &function, const TileShape &tileShape, const std::string &op,
    const LogicalTensorPtr &operand, const LogicalTensorPtr &result, int axis = -1) {
    ASSERT(op == "MAX" || op == "MIN" || op == "SUM" || op == "MAX_COMBINE_AXIS" || op == "SUM_COMBINE_AXIS");
    assert(operand->shape.size() == operand->offset.size());

    if (axis < 0) {
        axis = operand->shape.size() + axis;
    }

    // for loops before reduce axis
    TileInfo tileInfo(operand->shape, operand->offset);
    TileInfo resultTileInfo(result->shape, result->offset);
    auto input = Input{operand, tileInfo};
    std::vector<int> defaultAxisOrder;
    for (size_t i = 0; i < operand->shape.size(); i++) {
        defaultAxisOrder.push_back(i);
    }
    ReduceSingle(0, op, input, result, resultTileInfo, axis, function, tileShape, defaultAxisOrder);
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
    ASSERT(op == "MAX" || op == "MIN" || op == "SUM" || op == "MAX_COMBINE_AXIS" || op == "SUM_COMBINE_AXIS");
    assert(operand->shape.size() == operand->offset.size());
    auto opCode = Opcode::OP_ROWMAX_SINGLE;
    if (op == "MAX") {
        opCode = Opcode::OP_ROWMAX_SINGLE;
    } else if (op == "MIN") {
        opCode = Opcode::OP_ROWMIN_SINGLE;
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
    auto &vecTile = tileShape.GetVecTile();
    int64_t tmpTile = vecTile[cur];
    for (int i = 0; i < result->shape[cur]; i += tmpTile) {
        if (cur < static_cast<size_t>(axis)) {
            // 在result中gather轴的外层轴
            paramsInput.tileInfo.offset[cur] = i % paramsInput.tensor->shape[cur];
            paramsInput.tileInfo.shape[cur] =
                std::min(paramsInput.tensor->shape[cur] - paramsInput.tileInfo.offset[cur], tmpTile);
        } else if (cur >= static_cast<size_t>(axis) &&
                   (cur < static_cast<size_t>(axis) + indicesInput.tensor->shape.size())) {
            // 当前属于indices的gather轴
            // params[axis]不切
            paramsInput.tileInfo.offset[axis] = 0;
            paramsInput.tileInfo.shape[axis] = paramsInput.tensor->shape[axis];
            // 处理indices的tileInfo
            indicesInput.tileInfo.offset[cur - axis] = i % indicesInput.tensor->shape[cur - axis];
            indicesInput.tileInfo.shape[cur - axis] =
                std::min(indicesInput.tensor->shape[cur - axis] - indicesInput.tileInfo.offset[cur - axis], tmpTile);
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

std::vector<int64_t> GatherOperationResultShape(
    LogicalTensorPtr params, LogicalTensorPtr indices, int axis) {
    assert(params->shape.size() == params->offset.size());
    assert(indices->shape.size() == indices->offset.size());
    int paramsRank = params->shape.size();
    if (axis < 0) {
        axis = axis + paramsRank;
    }
    // result shape: params.shape[:aixs] + indices.shape + params.shape[axis+1:]
    std::vector<int64_t> resultShape = params->shape;
    resultShape.erase(resultShape.begin() + axis);
    resultShape.insert(resultShape.begin() + axis, indices->shape.begin(), indices->shape.end());

    return resultShape;
}

void TiledGatherOperation(Function &function, const TileShape &tileShape, const LogicalTensorPtr &params,
    const LogicalTensorPtr &indices, int axis, const LogicalTensorPtr &result) {
    // Check Operands Valid
    std::vector<int64_t> expectedShape = GatherOperationResultShape(params, indices, axis);
    assert(result->shape.size() == expectedShape.size());
    assert(result->shape.size() == result->offset.size());
    assert(params->shape.size() == params->offset.size());
    assert(indices->shape.size() == indices->offset.size());

    assert(result->shape.size() <= NUM_VALUE_5);
    assert(indices->shape.size() <= NUM_VALUE_2);
    if (axis < 0) {
        axis += params->shape.size();
    }
    assert(axis >= 0 && axis < static_cast<int>(params->shape.size()));
    TileInfo paramsTileInfo(params->shape.size(), params->offset.size());
    TileInfo indicesTileInfo(indices->shape.size(), indices->offset.size());
    TileInfo resultTileInfo(result->shape.size(), result->offset.size());
    auto paramsInput = Input{params, paramsTileInfo};
    auto indicesInput = Input{indices, indicesTileInfo};
    TiledGatherOperation(function, tileShape, 0, paramsInput, indicesInput, axis, result, resultTileInfo);
}

LogicalTensorPtr TiledGatherOperation(Function &function, const TileShape &tileShape, const LogicalTensorPtr &params,
    const LogicalTensorPtr &indices, int axis) {
    std::vector<int64_t> resultShape = GatherOperationResultShape(params, indices, axis);
    auto result = std::make_shared<LogicalTensor>(function, params->Datatype(), resultShape);

    assert(params->shape.size() == params->offset.size());
    assert(indices->shape.size() == indices->offset.size());

    assert(result->shape.size() <= NUM_VALUE_5);
    assert(indices->shape.size() <= NUM_VALUE_2);
    if (axis < 0) {
        axis += params->shape.size();
    }
    
    assert(axis >= 0 && axis < static_cast<int>(params->shape.size()));
    TileInfo paramsTileInfo(params->shape.size(), params->offset.size());
    TileInfo indicesTileInfo(indices->shape.size(), indices->offset.size());
    TileInfo resultTileInfo(result->shape.size(), result->offset.size());
    auto paramsInput = Input{params, paramsTileInfo};
    auto indicesInput = Input{indices, indicesTileInfo};
    TiledGatherOperation(function, tileShape, 0, paramsInput, indicesInput, axis, result, resultTileInfo);

    return result;
}

LogicalTensorPtr TensorGatherOperation(
    Function &function, const LogicalTensorPtr &params, const LogicalTensorPtr &indices, int axis) {
    const auto &paramsDynShape = params->GetDynValidShape();
    const auto &indicesDynShape = indices->GetDynValidShape();
    const int paramsRank = paramsDynShape.size();
    if (axis < 0) {
        axis += paramsRank;
        ASSERT(axis >= 0 && axis < paramsRank) << "The configuration of the axis is incorrect";
    }
    std::vector<int64_t> resultShape = GatherOperationResultShape(params, indices, axis);
    auto result = std::make_shared<LogicalTensor>(function, params->Datatype(), resultShape);
    std::vector<SymbolicScalar> outValidShape = paramsDynShape;
    outValidShape.erase(outValidShape.begin() + axis);
    outValidShape.insert(outValidShape.begin() + axis, indicesDynShape.begin(), indicesDynShape.end());
    auto &op = GraphUtils::AddDynOperation(function, Opcode::OP_GATHER, {params, indices}, {result}, {outValidShape});
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
    auto &vecTile = tileShape.GetVecTile();
    int64_t tmpTile = vecTile[cur];
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
                std::min(paramsInput.tensor->shape[cur] - paramsInput.tileInfo.offset[cur], tmpTile);
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

struct ScatterElementSPara {
    const LogicalTensorPtr &dstTensor;
    const LogicalTensorPtr &srcInput;
    const LogicalTensorPtr &idxInput;
    const Element& scalar;
    const int axis;
    const std::string &reduceMode;
};

void InnerTiledScatterElementS(size_t cur, Function &function, const TileShape &tileShape,
    const ScatterElementSPara& scatterPara, ScatterTileInfoPara& scatterTileInfo) {
    const LogicalTensorPtr &dstTensor = scatterPara.dstTensor;
    const LogicalTensorPtr &srcInput = scatterPara.srcInput;
    const LogicalTensorPtr &idxInput = scatterPara.idxInput;
    const Element& scalar = scatterPara.scalar;
    const int axis = scatterPara.axis;
    const std::string &reduceMode = scatterPara.reduceMode;

    if (cur == dstTensor->shape.size()) {
        // add Operation
        auto srcTile = srcInput->View(function, scatterTileInfo.srcTileInfo.shape, scatterTileInfo.srcTileInfo.offset);
        auto idxTile = idxInput->View(function, scatterTileInfo.idxTileInfo.shape, scatterTileInfo.idxTileInfo.offset);
        auto dstTile = dstTensor->View(function, scatterTileInfo.dstTileInfo.shape, scatterTileInfo.dstTileInfo.offset);
        auto &op = function.AddOperation(Opcode::OP_SCATTER_ELEMENT, {srcTile, idxTile}, {dstTile});
        op.SetAttribute(OP_ATTR_PREFIX + "axis", axis);
        op.SetAttribute(OpAttributeKey::scalar, scalar);
        op.SetAttribute(OpAttributeKey::reduceMode, reduceMode);
        return;
    }

    // 按照dstShape进行切分
    auto &vecTile = tileShape.GetVecTile();
    if (vecTile[axis] < dstTensor->shape[axis]) {
        ALOG_ERROR_F("the axis:%d is not allowed to be cut. tileshape:%lld dstshape:%lld", axis, vecTile[axis],
            dstTensor->shape[axis]);
        ASSERT(vecTile[axis] >= dstTensor->shape[axis]);
    }
    int64_t tmpTile = vecTile[cur];
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
        InnerTiledScatterElementS(cur + 1, function, tileShape, scatterPara, scatterTileInfo);
    }
}

void TiledScatterElementS(Function &function, const TileShape &tileShape, const ScatterElementSPara& scatterPara) {
    // Check Operands Valid
    assert(scatterPara.srcInput->shape.size() == scatterPara.srcInput->offset.size());
    assert(scatterPara.idxInput->shape.size() == scatterPara.idxInput->offset.size());
    assert(scatterPara.dstTensor->shape.size() == scatterPara.dstTensor->offset.size());

    ScatterTileInfoPara scatterTileInfo{
        TileInfo(scatterPara.srcInput->shape.size(), scatterPara.srcInput->offset.size()),
        TileInfo(scatterPara.idxInput->shape.size(), scatterPara.idxInput->offset.size()),
        TileInfo(scatterPara.dstTensor->shape.size(), scatterPara.dstTensor->offset.size()),
    };
    InnerTiledScatterElementS(0, function, tileShape, scatterPara, scatterTileInfo);
}

void TensorScatterElementS(Function &function, const ScatterElementSPara& scatterPara) {
    auto &op = GraphUtils::AddDynOperation(function, Opcode::OP_SCATTER_ELEMENT, {scatterPara.srcInput, 
        scatterPara.idxInput}, {scatterPara.dstTensor});
    op.SetAttribute(OP_ATTR_PREFIX + "axis", scatterPara.axis);
    op.SetAttribute(OpAttributeKey::scalar, scatterPara.scalar);
    op.SetAttribute(OpAttributeKey::reduceMode, scatterPara.reduceMode);
}

void UnalignPadTmpBufTile(std::vector<int64_t> &shape, int blockElem) {
    // tmpbuf按16 8对齐
    auto size = shape.size();
    if (size >= NUM_VALUE_2) {
        shape[size - NUM_VALUE_2] = AlignUp(shape[size - NUM_VALUE_2], (int64_t)VNCHWCONV_REPEAT);
        shape[size - 1] = AlignUp(shape[size - 1], blockElem);
    }
}

template <TransposeOpType T>
void TiledInnerTranspose(Function &function, const TileShape &tileShape, const int cur, Input &input,
    const LogicalTensorPtr &result, const std::vector<int> &shape) {
    int shapeSize = input.tensor->shape.size();
    if (cur == shapeSize) {
        auto tile = input.tensor->View(function, input.tileInfo.shape, input.tileInfo.offset);
        std::vector<int64_t> resultTileShape(input.tileInfo.shape);
        std::swap(resultTileShape[shape[0]], resultTileShape[shape[1]]);
        std::vector<int64_t> resultTileOfs(input.tileInfo.offset);
        std::swap(resultTileOfs[shape[0]], resultTileOfs[shape[1]]);
        auto resultTile = result->View(function, resultTileShape, resultTileOfs);
        if (T == TransposeOpType::TRANSPOSE_MOVEOUT || T == TransposeOpType::TRANSPOSE_MOVEIN) {
            auto &op = function.AddOperation(GetTransposeOpName<T>(), {tile}, {resultTile});
            op.SetAttribute(OP_ATTR_PREFIX + "shape", shape);
        } else {
            std::vector<int64_t> tmpShape(input.tileInfo.shape);
            int64_t blockElem = BLOCK_SIZE / static_cast<int>(BytesOf(tile->Datatype()));
            UnalignPadTmpBufTile(tmpShape, blockElem);
            auto tempTensor = std::make_shared<LogicalTensor>(function, tile->Datatype(), tmpShape);
            tempTensor->dynValidShape_ = SymbolicScalar::FromConcrete(tmpShape);
            auto &op = function.AddOperation(GetTransposeOpName<T>(), {tile}, {resultTile, tempTensor});
            op.SetAttribute(OP_ATTR_PREFIX + "shape", shape);
        }
        return;
    }
    auto &vecTile = tileShape.GetVecTile();
    for (int i = 0; i < input.tensor->shape[cur]; i += vecTile[cur]) {
        input.tileInfo.shape[cur] = std::min(input.tensor->shape[cur] - i, vecTile[cur]);
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
    if (transposeShape[0] != (int)operand->shape.size() - 1 && transposeShape[1] != (int)operand->shape.size() - 1) {
        auto &operation = function.AddOperation(Opcode::OP_TRANSPOSE_MOVEOUT, {operand}, {result});
        operation.SetAttribute(OP_ATTR_PREFIX + "shape", transposeShape);
        return;
    }

    if (transposeShape[0] == (int)operand->shape.size() - 2 &&        // last 2 dims transpose
        transposeShape[1] == (int)operand->shape.size() - 1) {
        auto &operation = function.AddOperation(Opcode::OP_TRANSPOSE_VNCHWCONV, {operand}, {result});
        operation.SetAttribute(OP_ATTR_PREFIX + "shape", transposeShape);
        return;
    }

    ASSERT(operand->shape.size() == 3 || operand->shape.size() == 4)  // input should be 3 or 4 dims
        << "Transpose shape should be [A1,T1,A2,T2] or [T1,A2,T2].";

    // [A1,T1,A2,T2] to [A1,A2,T1,T2] or [T1,A2,T2] to [A2,T1,T2]
    auto oldVecTileShapes = TileShape::Current().GetVecTile();
    auto newVecTileShape = oldVecTileShapes;
    std::vector<int64_t> tmpShape(operand->shape);
    int dim1 = (tmpShape.size() == 3) ? 0 : 1;   // if input is 3 dims, dim1 = 0, otherwise dim1 = 1
    int dim2 = (tmpShape.size() == 3) ? 1 : 2;   // if input is 3 dims, dim2 = 1, otherwise dim2 = 2
    std::swap(tmpShape[dim1], tmpShape[dim2]);
    std::swap(newVecTileShape[dim1], newVecTileShape[dim2]);
    auto moveInResult = std::make_shared<LogicalTensor>(function, operand->Datatype(), tmpShape);
    auto &inOp = function.AddOperation(Opcode::OP_TRANSPOSE_MOVEIN, {operand}, {moveInResult});
    inOp.SetAttribute(OP_ATTR_PREFIX + "shape", std::vector<int>{dim1, dim2});
    TileShape::Current().SetVecTile(newVecTileShape);

    // [A1,A2,T1,T2] to [A1,A2,T2,T1] or [A2,T1,T2] to [A2,T2,T1]
    tmpShape = moveInResult->shape;
    dim1 = (tmpShape.size() == 3) ? 1 : 2;   // if input is 3 dims, dim1 = 1, otherwise dim1 = 2
    dim2 = (tmpShape.size() == 3) ? 2 : 3;   // if input is 3 dims, dim2 = 2, otherwise dim2 = 3
    std::swap(tmpShape[dim1], tmpShape[dim2]);
    std::swap(newVecTileShape[dim1], newVecTileShape[dim2]);
    auto vnchwconvResult = std::make_shared<LogicalTensor>(function, operand->Datatype(), tmpShape);
    auto &convOp = function.AddOperation(Opcode::OP_TRANSPOSE_VNCHWCONV, {moveInResult}, {vnchwconvResult});
    convOp.SetAttribute(OP_ATTR_PREFIX + "shape", std::vector<int>{dim1, dim2});
    TileShape::Current().SetVecTile(newVecTileShape);

    // [A1,A2,T2,T1] to [A1,T2,A2,T1] or [A2,T2,T1] to [T2,A2,T1]
    tmpShape = vnchwconvResult->shape;
    dim1 = (tmpShape.size() == 3) ? 0 : 1;   // if input is 3 dims, dim1 = 0, otherwise dim1 = 1
    dim2 = (tmpShape.size() == 3) ? 1 : 2;   // if input is 3 dims, dim2 = 1, otherwise dim2 = 2
    std::swap(tmpShape[dim1], tmpShape[dim2]);
    auto &outOp = function.AddOperation(Opcode::OP_TRANSPOSE_MOVEOUT, {vnchwconvResult}, {result});
    outOp.SetAttribute(OP_ATTR_PREFIX + "shape", std::vector<int>{dim1, dim2});
    TileShape::Current().SetVecTile(oldVecTileShapes);
}

bool MergeTransposeAxis(const Tensor &operand, std::vector<int64_t>& inputShape, std::vector<int64_t>& vecTileShape,
                        std::vector<SymbolicScalar>& validShape, std::vector<int>& transposeShape) {
    auto oldTransposeShape = transposeShape;
    int64_t pre = 1;
    int64_t mid = 1;
    int64_t after = 1;
    int64_t preTileShape = 1;
    int64_t midTileShape = 1;
    int64_t afterTileShape = 1;
    SymbolicScalar preValidShape = 1;
    SymbolicScalar midValidShape = 1;
    SymbolicScalar afterValidShape = 1;
    int preNum = 0;
    int midNum = 0;
    int afterNum = 0;
    auto oldVecTileShapes = TileShape::Current().GetVecTile();
    auto oldValidShapes = validShape;
    for (int i = 0; i < (int)operand->shape.size(); i++) {
        if (i < oldTransposeShape[0]) {
            pre *= operand->shape[i];
            preTileShape *= oldVecTileShapes[i];
            preValidShape = preValidShape * oldValidShapes[i];
            preNum++;
        } else if (i < oldTransposeShape[1] && i > oldTransposeShape[0]) {
            mid *= operand->shape[i];
            midTileShape *= oldVecTileShapes[i];
            midValidShape = midValidShape * oldValidShapes[i];
            midNum++;
        } else if (i > oldTransposeShape[1]) {
            after *= operand->shape[i];
            afterTileShape *= oldVecTileShapes[i];
            afterValidShape = afterValidShape * oldValidShapes[i];
            afterNum++;
        }
    }

    if (preNum <= 1 && midNum <= 1 && afterNum <= 1) {
        return false;
    }
    if (operand->shape.size() <= 5 && oldTransposeShape[0] == (int)operand->shape.size() - 2 &&  // tileop支持5维，最后2维转置
        oldTransposeShape[1] == (int)operand->shape.size() - 1) {
        return false;
    }

    // [A1,T1,A2,T2,A3]
    validShape.clear();
    if (preNum > 0) {
        inputShape.push_back(pre);
        vecTileShape.push_back(preTileShape);
        validShape.push_back(preValidShape);
        transposeShape[0] -= (preNum - 1);
        transposeShape[1] -= (preNum - 1);
    }
    inputShape.push_back(operand->shape[oldTransposeShape[0]]);
    vecTileShape.push_back(oldVecTileShapes[oldTransposeShape[0]]);
    validShape.push_back(oldValidShapes[oldTransposeShape[0]]);
    if (midNum > 0) {
        inputShape.push_back(mid);
        vecTileShape.push_back(midTileShape);
        validShape.push_back(midValidShape);
        transposeShape[1] -= (midNum - 1);
    }
    inputShape.push_back(operand->shape[oldTransposeShape[1]]);
    vecTileShape.push_back(oldVecTileShapes[oldTransposeShape[1]]);
    validShape.push_back(oldValidShapes[oldTransposeShape[1]]);
    if (afterNum > 0) {
        inputShape.push_back(after);
        vecTileShape.push_back(afterTileShape);
        validShape.push_back(afterValidShape);
    }
    return true;
}

Tensor Transpose(const Tensor &operand, std::vector<int> transposeShape) {
    DECLARE_TRACER();
    ASSERT(transposeShape.size() == 2) << "Transpose dim num should be 2."; // transposeShape should be 2 dims
    ASSERT(transposeShape[0] < (int)operand->shape.size()) << "Transpose dim should less than " << operand->shape.size();
    ASSERT(transposeShape[1] < (int)operand->shape.size()) << "Transpose dim should less than " << operand->shape.size();

    std::sort(transposeShape.begin(), transposeShape.end());
    if ((operand->shape[transposeShape[0]] == 1 && operand->shape[transposeShape[1]] == 1) ||
        transposeShape[0] == transposeShape[1]) {
        return operand;
    }
    auto oldVecTileShapes = TileShape::Current().GetVecTile();
    ASSERT(oldVecTileShapes.size() == operand->shape.size()) << "TileShape dim num should same to input.";
    auto oldValidShapes = operand.GetStorage()->GetDynValidShape();
    if (oldValidShapes.empty()) {
        oldValidShapes = SymbolicScalar::FromConcrete(operand->shape);
    }
    ASSERT(oldValidShapes.size() == operand->shape.size()) << "ValidShape dim num should same to input.";

    std::vector<int64_t> newInputShape;
    std::vector<int64_t> newVecTileShape;
    std::vector<int> newTransposeShape = transposeShape;
    std::vector<SymbolicScalar> newValidShape = oldValidShapes;
    std::vector<int64_t> resultShape(operand->shape);
    std::swap(resultShape[transposeShape[0]], resultShape[transposeShape[1]]);
    if (!MergeTransposeAxis(operand, newInputShape, newVecTileShape, newValidShape, newTransposeShape)) {
        Tensor result(operand->Datatype(), resultShape);
        CALL(InnerTranspose, *Program::GetInstance().GetCurrentFunction(), operand.GetStorage(), result.GetStorage(),
             transposeShape);
        return result;
    }
    std::swap(oldValidShapes[transposeShape[0]], oldValidShapes[transposeShape[1]]);

    auto tmpInputTensor = Reshape(operand, newInputShape, newValidShape);
    TileShape::Current().SetVecTile(newVecTileShape);
    auto tmpOutputTensor = Transpose(tmpInputTensor, newTransposeShape);
    TileShape::Current().SetVecTile(oldVecTileShapes);
    return Reshape(tmpOutputTensor, resultShape, oldValidShapes);
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

    auto &vecTile = tileShape.GetVecTile();
    int tileOutH = vecTile[NUM_VALUE_0];
    int tileOutW = vecTile[NUM_VALUE_1];
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
    const std::vector<int64_t> outShape = {
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
    auto &vecTile = tileShape.GetVecTile();
    for (int i = 0; i < input.tensor->shape[cur]; i += vecTile[cur]) {
        input.tileInfo.shape[cur] = std::min(input.tensor->shape[cur] - i, vecTile[cur]);
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
    auto result = std::make_shared<LogicalTensor>(function, newType, operand->shape, operand->dynValidShape_);
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

Tensor Ln(const Tensor &operand) {
    DECLARE_TRACER();

    RETURN_CALL(UnaryOperation<UnaryOpType::LN>, *Program::GetInstance().GetCurrentFunction(),
        operand.GetStorage());
}

Tensor Log(const Tensor &operand, LogBaseType base) {
    DECLARE_TRACER();
    ASSERT(base == LogBaseType::LOG_e || base == LogBaseType::LOG_2 || base == LogBaseType::LOG_10);
    ASSERT(operand->tensor->datatype == DataType::DT_FP16 || operand->tensor->datatype == DataType::DT_FP32);

    auto operandCast = Tensor(DataType::DT_FP32, operand->shape);
    if (operand->tensor->datatype == DataType::DT_FP16) {
        operandCast = CALL(CastOperation<CastOpType::CAST>, *Program::GetInstance().GetCurrentFunction(),
            operand.GetStorage(), DataType::DT_FP32, CastMode::CAST_NONE);
    } else {
        operandCast = operand;
    }

    auto resTensor = Tensor(DataType::DT_FP32, operand->shape);
    resTensor = CALL(UnaryOperation<UnaryOpType::LN>, *Program::GetInstance().GetCurrentFunction(), operandCast.GetStorage());

    auto resTensorBeforeCast = Tensor(DataType::DT_FP32, operand->shape);
    if (base == LogBaseType::LOG_2) {
        resTensorBeforeCast = CALL(BinaryOperationScalar<BinaryOpType::DIV>, *Program::GetInstance().GetCurrentFunction(),
            resTensor.GetStorage(), Element(DataType::DT_FP32, std::log(static_cast<float>(NUM_VALUE_2))));
    } else if (base == LogBaseType::LOG_10) {
        resTensorBeforeCast = CALL(BinaryOperationScalar<BinaryOpType::DIV>, *Program::GetInstance().GetCurrentFunction(),
            resTensor.GetStorage(), Element(DataType::DT_FP32, std::log(static_cast<float>(NUM_VALUE_10))));
    } else {
        resTensorBeforeCast = resTensor;
    }

    if (operand->tensor->datatype == DataType::DT_FP16) {
        RETURN_CALL(CastOperation<CastOpType::CAST>, *Program::GetInstance().GetCurrentFunction(),
            resTensorBeforeCast.GetStorage(), DataType::DT_FP16, CastMode::CAST_NONE);
    }
    return resTensorBeforeCast;
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

Tensor Neg(const Tensor &operand) {
    DECLARE_TRACER();

    if (IsFloat(operand->Datatype())) {
        RETURN_CALL(BinaryOperationScalar<BinaryOpType::MUL>, *Program::GetInstance().GetCurrentFunction(),
            operand.GetStorage(), Element(operand->Datatype(), -1.0));
    } else {
        RETURN_CALL(BinaryOperationScalar<BinaryOpType::MUL>, *Program::GetInstance().GetCurrentFunction(),
            operand.GetStorage(), Element(operand->Datatype(), -1));
    }
}

Tensor Rsqrt(const Tensor &operand) {
    DECLARE_TRACER();

    RETURN_CALL(
        UnaryOperation<UnaryOpType::RSQRT>, *Program::GetInstance().GetCurrentFunction(), operand.GetStorage());
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

Tensor MaxS(const Tensor &operand1, const Element &operand2) {
    DECLARE_TRACER();
    RETURN_CALL(BinaryOperationScalar<BinaryOpType::MAX>, *Program::GetInstance().GetCurrentFunction(),
        operand1.GetStorage(), operand2);
}

Tensor MinS(const Tensor &operand1, const Element &operand2) {
    DECLARE_TRACER();
    RETURN_CALL(BinaryOperationScalar<BinaryOpType::MIN>, *Program::GetInstance().GetCurrentFunction(),
        operand1.GetStorage(), operand2);
}

Tensor Compare(const Tensor& operand1, const Tensor& operand2, CmpOperationType operation, CmpModeType mode) {
    DECLARE_TRACER();
    RETURN_CALL(CompareOperation, *Program::GetInstance().GetCurrentFunction(), operand1, operand2, operation, mode);
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

Tensor Expand(const Tensor &operand, DataType dataType, const std::vector<int64_t> &shape) {
    DECLARE_TRACER();

    ASSERT(operand->shape.size() == shape.size());
    auto result = Tensor(dataType, shape);
    CALL(Expand, *Program::GetInstance().GetCurrentFunction(), operand.GetStorage(), result.GetStorage());
    return result;
}

Tensor TensorExpandOperation(Function &function, const LogicalTensorPtr &operand, const std::vector<int64_t> &dstShape,
    const std::vector<SymbolicScalar> &validShape) {
    auto result = std::make_shared<LogicalTensor>(function, operand->Datatype(), dstShape, validShape);
    auto &op = function.AddOperation(Opcode::OP_EXPAND, {operand}, {result});

    op.SetAttribute(OP_ATTR_PREFIX + "shape", dstShape);
    op.SetAttribute(OP_ATTR_PREFIX + "validShape", validShape);
    function.UpdateTensorDataUsage(op);
    return result;
}

Tensor Expand(const Tensor &operand, const std::vector<int64_t> &dstShape, std::vector<SymbolicScalar> validShape) {
    DECLARE_TRACER();

    ASSERT(operand->shape.size() == dstShape.size());
    RETURN_CALL(ExpandOperation, *Program::GetInstance().GetCurrentFunction(), operand.GetStorage(), dstShape, validShape);
}

void TiledReduceExpandNew(Function &function, const TileShape &tileShape, const std::string &op,
    const LogicalTensorPtr &operand, const LogicalTensorPtr &result) {
    ASSERT(op == "MAX" || op == "SUM");
    assert(operand->shape.size() == operand->offset.size());

    // 目前只支持2维操作
    if (operand->shape.size() != 2) {
        assert(false && "unsupported dimension");
    }

    auto &vecTile = tileShape.GetVecTile();
    TileInfo tileInfo({vecTile[0], operand->shape[1]}, std::vector<int64_t>(operand->offset.size()));

    for (int i = 0; i < operand->shape[0]; i += vecTile[0]) {
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

    const int lastDim = operand->shape.size() - 1;
    const int alignNum = BLOCK_SIZE / BytesOf(operand->tensor->datatype);
    auto &vecTile = TileShape::Current().GetVecTile();
    if (axis == lastDim) {
        ASSERT(vecTile[lastDim] % alignNum == 0) << "RowMaxSingle op: the tileShape of last axis need to 32Byte align!";
    }

    Tensor result(operand->tensor->datatype, resultShape);
    int shapeSize = static_cast<int>(resultShape.size());
    if (ConfigManager::Instance().GetOperationConfig("FORCE_COMBINE_AXIS", false) && axis == shapeSize - 1 &&
        shapeSize >= NUM2 &&
        (resultShape[shapeSize - NUM2] % NUM_VALUE_8 == 0 && vecTile[vecTile.size() - NUM2] % NUM_VALUE_8 == 0)) {
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

    const int lastDim = operand->shape.size() - 1;
    const int alignNum = BLOCK_SIZE / BytesOf(operand->tensor->datatype);
    auto &vecTile = TileShape::Current().GetVecTile();
    if (axis == lastDim) {
        ASSERT(vecTile[lastDim] % alignNum == 0) << "RowMinSingle op: the tileShape of last axis need to 32Byte align!";
    }

    Tensor result(operand->tensor->datatype, resultShape);
    int shapeSize = static_cast<int>(resultShape.size());
    if (ConfigManager::Instance().GetOperationConfig("FORCE_COMBINE_AXIS", false) && axis == shapeSize - 1 &&
        shapeSize >= NUM2 &&
        (resultShape[shapeSize - NUM2] % NUM_VALUE_8 == 0 && vecTile[vecTile.size() - NUM2] % NUM_VALUE_8 == 0)) {
        CALL(ReduceSingle, *Program::GetInstance().GetCurrentFunction(), "MIN_COMBINE_AXIS", operand, result, axis);
    } else {
        CALL(ReduceSingle, *Program::GetInstance().GetCurrentFunction(), "MIN", operand, result, axis);
    }
    return result;
}

Tensor RowSumSingle(const Tensor &operand, int axis) {
    DECLARE_TRACER();
    auto resultShape = operand->shape;
    axis = axis < 0 ? operand->shape.size() + axis : axis;

    resultShape[axis] = 1;

    const int lastDim = operand->shape.size() - 1;
    const int alignNum = BLOCK_SIZE / BytesOf(operand->tensor->datatype);
    auto &vecTile = TileShape::Current().GetVecTile();
    if (axis == lastDim) {
        ASSERT(vecTile[lastDim] % alignNum == 0) << "RowSumSingle op: the tileShape of last axis need to 32Byte align!";
    }

    Tensor result(operand->tensor->datatype, resultShape);
    int shapeSize = static_cast<int>(resultShape.size());
    if (ConfigManager::Instance().GetOperationConfig("FORCE_COMBINE_AXIS", false) && axis == shapeSize - 1 &&
        shapeSize >= NUM2 &&
        (resultShape[shapeSize - NUM2] % NUM_VALUE_8 == 0 && vecTile[vecTile.size() - NUM2] % NUM_VALUE_8 == 0)) {
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
    std::vector<int64_t> &shape, const std::vector<SymbolicScalar> &validShape, const LogicalTensorPtr &results,
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
        TileShape::Current().SetVecTile(tempTileShape);
        auto &op = function.AddOperation("TILE_VEC_DUP", {}, {resultTile});

        op.SetAttribute(OpAttributeKey::scalar, value);
        if (dynValue.IsValid()) {
            op.SetAttribute(OpAttributeKey::dynScalar, dynValue);
        }
        op.SetAttribute(OP_ATTR_PREFIX + "shape", resultTileInfo.shape);
        op.SetAttribute(OP_ATTR_PREFIX + "validShape", resultTile->GetDynValidShape());
        return;
    }

    auto &vecTile = tileShape.GetVecTile();
    for (int i = 0; i < results->shape[cur]; i += vecTile[cur]) {
        resultTileInfo.offset[cur] = i;
        resultTileInfo.shape[cur] = std::min(results->shape[cur] - i, vecTile[cur]);
        TiledVecDup(function, tileShape, cur + 1, value, dynValue, shape, validShape, results, resultTileInfo);
    }
}

void TiledLogicalNotOperation(Function& function, const TileShape& tileShape, size_t cur,
        Input& input, const LogicalTensorPtr& result) {
    if (cur == input.tensor->shape.size()) {
        auto tile = input.tensor->View(function, input.tileInfo.shape, input.tileInfo.offset);
        auto resultTile = result->View(function, input.tileInfo.shape, input.tileInfo.offset);

        std::vector<int64_t> castConditionShape({2048});
        auto castConditionTensor = std::make_shared<LogicalTensor>(function, DT_FP16, castConditionShape);

        DataType selectDtype;
        if (input.tensor.GetDataType() == DT_FP32) {
            selectDtype = DT_FP32;
        } else {
            selectDtype = DT_FP16;
        }
        auto compareConditionTensor = std::make_shared<LogicalTensor>(function, selectDtype, castConditionShape);
        auto oneConditionTensor = std::make_shared<LogicalTensor>(function, selectDtype, castConditionShape);

        std::vector<int64_t> vcmpBitResultShape({2048 / 8});
        auto vcmpBitResultTensor = std::make_shared<LogicalTensor>(function, DT_INT8, vcmpBitResultShape);
        std::vector<int64_t> startAddrUBShape({1});
        auto startAddrUBTensor = std::make_shared<LogicalTensor>(function, DT_UINT64, startAddrUBShape);
        function.AddOperation(Opcode::OP_LOGICALNOT, {tile}, {resultTile, castConditionTensor, compareConditionTensor,
                                                            vcmpBitResultTensor, startAddrUBTensor, oneConditionTensor});
        return;
    }

    auto& vecTile = tileShape.GetVecTile();
    for (int i = 0; i < input.tensor->shape[cur]; i += vecTile[cur]) {
        input.tileInfo.shape[cur] = std::min(input.tensor->shape[cur] - i, vecTile[cur]);
        input.tileInfo.offset[cur] = i;
        TiledLogicalNotOperation(function, tileShape, cur + 1, input, result);
    }
}

void TiledLogicalNotOperation(Function& function, const TileShape& tileShape,
        const LogicalTensorPtr& operand, const LogicalTensorPtr& result) {
    assert(operand->shape.size() == operand->offset.size());

    TileInfo tileInfo(result->shape.size(), result->offset.size());
    auto input = Input{operand, tileInfo};
    TiledLogicalNotOperation(function, tileShape, 0, input, result);
}

LogicalTensorPtr TensorLogicalNotOperation(Function& function, LogicalTensorPtr operand) {
    auto result = std::make_shared<LogicalTensor>(function, DT_BOOL, operand->shape, operand->GetDynValidShape());
    function.AddOperation(Opcode::OP_LOGICALNOT, {operand}, {result});
    return result;
}

Tensor LogicalNot(const Tensor &operand) {
    DECLARE_TRACER();

    RETURN_CALL(LogicalNotOperation, *Program::GetInstance().GetCurrentFunction(), operand.GetStorage());
}

void TiledVecDup(Function &function, const TileShape &tileShape, const Element &value, const SymbolicScalar &dynValue,
    std::vector<int64_t> &shape, const std::vector<SymbolicScalar> &validShape, const LogicalTensorPtr &results) {
    TileInfo resultTileInfo(results->shape.size(), results->offset.size());
    TiledVecDup(function, tileShape, 0, value, dynValue, shape, validShape, results, resultTileInfo);
}

Tensor TensorVectorDuplicateOperation(Function &function, const Element& src, const SymbolicScalar &dynValue,
    DataType dtype, const std::vector<int64_t> &dstShape, const std::vector<SymbolicScalar> &validShape) {
    auto result = std::make_shared<LogicalTensor>(function, dtype, dstShape, validShape);
    auto &op = function.AddOperation(Opcode::OP_VEC_DUP, {}, {result}); //输入没有tensor
    op.SetAttribute(OpAttributeKey::scalar, src);
    if (dynValue.IsValid()) {
        op.SetAttribute(OpAttributeKey::dynScalar, dynValue);
    }
    op.SetAttribute(OP_ATTR_PREFIX + "shape", dstShape);
    op.SetAttribute(OP_ATTR_PREFIX + "validShape", validShape);
    function.UpdateTensorDataUsage(op);
    return result;
}

Tensor VectorDuplicate(const Element &src, DataType dtype, const std::vector<int64_t> &dstShape,
    std::vector<SymbolicScalar> validShape) {
    DECLARE_TRACER();
    if (validShape.empty()) {
        for (auto x : dstShape)
            validShape.emplace_back(x);
    }
    RETURN_CALL(VectorDuplicateOperation, *Program::GetInstance().GetCurrentFunction(), src, SymbolicScalar(), dtype, dstShape, validShape);
}

Tensor VectorDuplicate(const SymbolicScalar &dynSrc, DataType dtype, const std::vector<int64_t> &dstShape,
    std::vector<SymbolicScalar> validShape) {
    DECLARE_TRACER();
    if (validShape.empty()) {
        for (auto x : dstShape)
            validShape.emplace_back(x);
    }
    RETURN_CALL(VectorDuplicateOperation, *Program::GetInstance().GetCurrentFunction(), Element(dtype, (int64_t)0), dynSrc, dtype, dstShape, validShape);
}

void internal::Print(SymbolicScalar cond, const std::string &format, const std::vector<Tensor> &tensors,
    const std::vector<SymbolicScalar> &scalars){
    auto function = Program::GetInstance().GetCurrentFunction();
    std::vector<LogicalTensorPtr> inputs;
    for (auto &t : tensors) {
        inputs.push_back(t.GetStorage());
    }
    auto &op = function->AddOperation(Opcode::OP_PRINT, inputs, {});
    op.SetAttr(OP_ATTR_PREFIX + "format", format);
    op.SetAttr(OP_ATTR_PREFIX + "scalars", scalars);
    op.SetAttribute(OP_ATTR_PREFIX + "cond", cond);
    function->UpdateTensorDataUsage(op);
}

void ToFile(const Tensor &operand, const std::string &fname, const std::vector<SymbolicScalar> &scalars, SymbolicScalar cond) {
    auto function = Program::GetInstance().GetCurrentFunction();
    auto &op = function->AddOperation(Opcode::OP_PRINT, {operand.GetStorage()}, {});
    ASSERT(!fname.empty()) << "Invalid file name";
    op.SetAttribute(OP_ATTR_PREFIX + "fname", fname);
    op.SetAttribute(OP_ATTR_PREFIX + "scalars", scalars);
    op.SetAttribute(OP_ATTR_PREFIX + "cond", cond);
    function->UpdateTensorDataUsage(op);
}

Tensor GatherElement(const Tensor &params, const Tensor &indices, int axis) {
    DECLARE_TRACER();
    ASSERT(axis < static_cast<int>(params->shape.size()) && axis >= - static_cast<int>(params->shape.size()));
    axis = axis < 0 ? params->shape.size() + axis : axis; //支持负轴
    RETURN_CALL(GatherElementOperation, *Program::GetInstance().GetCurrentFunction(), params.GetStorage(),
        indices.GetStorage(), axis);
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
    std::vector<int64_t> newShape(old.GetStorage()->shape);
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
    auto &vecTile = tileShape.GetVecTile();
    int64_t tmpTile = vecTile[cur];
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
    auto &vecTile = tileShape.GetVecTile();
    int64_t tmpTile = vecTile[cur];
    if (static_cast<int>(cur) == axis) {
        tmpTile = srcInput.tensor->shape[cur];
    }

    for (int i = 0; i < srcInput.tensor->shape[cur]; i += tmpTile) {
        if (static_cast<int>(cur) == axis) { // asis == 1
            srcInput.tileInfo.offset[cur] = 0;
            srcInput.tileInfo.shape[cur] = srcInput.tensor->shape[cur];

            int64_t indexTileLen = vecTile[0];
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

void TiledScatterUpdateFor2Dims(Function &function, const TileShape &tileShape,
    const LogicalTensorPtr &result, const LogicalTensorPtr &src,
    const LogicalTensorPtr &index, const LogicalTensorPtr &dst, int axis, std::string cacheMode, int blockSize) {
    auto &vecTile = tileShape.GetVecTile();
    int64_t tileBS = vecTile[NUM_VALUE_0];
    int64_t tileD = vecTile[NUM_VALUE_1];
    int64_t s = index->shape[1];
    if (s == 0 || tileBS == 0) {
        ALOG_ERROR_F("error: s == 0 || tileBS == 0");
        ASSERT(s == 0 || tileBS == 0);
    }
    if ((tileBS < s && s % tileBS != 0) || (tileBS > s && tileBS % s != 0)) {
        ALOG_ERROR_F("tileshape 0 is invalid, tileshape(%d, %d)", tileBS, tileD);
    }
    ASSERT((tileBS <= s && s % tileBS == 0) || (tileBS > s && tileBS % s == 0));
    ASSERT(tileD == src->shape[NUM_VALUE_1]);
    int64_t tileB = CeilDiv(tileBS, s);
    int64_t tileS = tileBS < s ? tileBS : s;
    int64_t bsOffset = 0;
    for (int64_t bIdx = 0; bIdx < index->shape[0]; bIdx += tileB) {
        for (int64_t sIdx = 0; sIdx < index->shape[1]; sIdx += tileS) {
            auto indexTile = index->View(function, {std::min(index->shape[0] - bIdx, tileB),
                std::min(index->shape[1] - sIdx, tileS)}, {bIdx, sIdx});
            for (int64_t j = 0; j < src->shape[1]; j += tileD) {
                auto srcTile = src->View(function, {std::min(src->shape[0] - bsOffset, tileBS),
                    std::min(src->shape[1] - j, tileD)}, {bsOffset, j});
                auto &op = function.AddOperation("TILE_INDEX_OUTCAST", {srcTile, indexTile, dst}, {result});
                op.SetAttribute("axis", axis);
                op.SetAttribute(OpAttributeKey::panzBlockSize, blockSize);
                op.SetAttribute(OpAttributeKey::cacheMode, cacheMode);
            }
            bsOffset += tileBS;
        }
    }
}

void TiledScatterUpdateFor4Dims(Function &function, const TileShape &tileShape,
    const LogicalTensorPtr &result, const LogicalTensorPtr &src,
    const LogicalTensorPtr &index, const LogicalTensorPtr &dst, int axis, std::string cacheMode, int blockSize) {
    auto &vecTile = tileShape.GetVecTile();
    int64_t tileB = vecTile[NUM_VALUE_0];
    int64_t tileS = vecTile[NUM_VALUE_1];
    int64_t tileN = vecTile[NUM_VALUE_2];
    int64_t tileD = vecTile[NUM_VALUE_3];
    for (int64_t i = 0; i < src->shape[0]; i += tileB) {
        for (int64_t j = 0; j < src->shape[1]; j += tileS) {
            auto indexTile = index->View(function, {std::min(index->shape[0] - i, tileB), std::min(index->shape[1] - j, tileS)}, {i, j});
            for (int64_t n = 0; n < src->shape[2]; n += tileN) {
                for (int64_t d = 0; d < src->shape[3]; d += tileD) {
                    auto srcTile = src->View(function, {std::min(src->shape[0] - i, tileB),
                        std::min(src->shape[1] - j, tileS),
                        std::min(src->shape[2] - n, tileN),
                        std::min(src->shape[3] - d, tileD)},
                        {i, j, n, d});
                    auto &op = function.AddOperation("TILE_INDEX_OUTCAST", {srcTile, indexTile, dst}, {result});
                    op.SetAttribute("axis", axis);
                    op.SetAttribute(OpAttributeKey::panzBlockSize, blockSize);
                    op.SetAttribute(OpAttributeKey::cacheMode, cacheMode);
                }
            }
        }
    }
}

void TiledScatterUpdate(Function &function, const TileShape &tileShape,
    const LogicalTensorPtr &result, const LogicalTensorPtr &src,
    const LogicalTensorPtr &index, const LogicalTensorPtr &dst, int axis, std::string cacheMode, int blockSize) {
    if (cacheMode == "PA_BSND") {
        if (src->shape.size() == NUM_VALUE_2) {
            TiledScatterUpdateFor2Dims(function, tileShape, result, src, index, dst, axis, cacheMode, blockSize);
        } else if (src->shape.size() == NUM_VALUE_4) {
            TiledScatterUpdateFor4Dims(function, tileShape, result, src, index, dst, axis, cacheMode, blockSize);
        } else {
            ALOG_ERROR_F("shape must be 2 or 4");
        }
        ASSERT(src->shape.size() == NUM_VALUE_2 || src->shape.size() == NUM_VALUE_4);
        return;
    }
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
    auto &vecTile = tileShape.GetVecTile();
    if (axis == 1 && src->shape.size() == NUM_VALUE_2 && vecTile[1] == src->shape[1]) { // 2维切index场景
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

static void CheckScatterUpdateInput(const Tensor &input)
{
    if ((input.GetShape().size() == NUM_VALUE_2 && (input.GetShape(NUM_VALUE_0) == NUM_VALUE_0 || input.GetShape(NUM_VALUE_1) == NUM_VALUE_0)) ||
        (input.GetShape().size() == NUM_VALUE_4 && (input.GetShape(NUM_VALUE_0) == NUM_VALUE_0 || input.GetShape(NUM_VALUE_1) == NUM_VALUE_0 ||
         input.GetShape(NUM_VALUE_2) == NUM_VALUE_0 || input.GetShape(NUM_VALUE_3) == NUM_VALUE_0))) {
        ALOG_ERROR_F("input shape is zero");
    }
    ASSERT((input.GetShape().size() == NUM_VALUE_2 && (input.GetShape(NUM_VALUE_0) != NUM_VALUE_0 && input.GetShape(NUM_VALUE_1) != NUM_VALUE_0)) ||
        (input.GetShape().size() == NUM_VALUE_4 && (input.GetShape(NUM_VALUE_0) != NUM_VALUE_0 && input.GetShape(NUM_VALUE_1) != NUM_VALUE_0 &&
        input.GetShape(NUM_VALUE_2) != NUM_VALUE_0 && input.GetShape(NUM_VALUE_3) != NUM_VALUE_0)));
    ASSERT(input.GetShape().size() == NUM_VALUE_2 || input.GetShape().size() == NUM_VALUE_4);
}

static void CheckScatterUpdateIndex(const Tensor &index)
{
    if (index.GetDataType() != DT_INT64 && index.GetDataType() != DT_INT32 && index.GetDataType() != DT_INT16) {
        ALOG_ERROR_F("index.GetDataType() != DT_INT64 && index.GetDataType() != DT_INT32 && index.GetDataType() != DT_INT16");
    }
    ASSERT(index.GetDataType() == DT_INT64 || index.GetDataType() == DT_INT32 || index.GetDataType() == DT_INT16);
    if (index.GetShape().size() != NUM_VALUE_2 || index.GetShape(NUM_VALUE_0) == NUM_VALUE_0 || index.GetShape(NUM_VALUE_1) == NUM_VALUE_0) {
        ALOG_ERROR_F("index.GetShape().size() is %d, shoud be 2", index.GetShape().size());
    }
    ASSERT(index.GetShape().size() == NUM_VALUE_2 && index.GetShape(NUM_VALUE_0) != NUM_VALUE_0 && index.GetShape(NUM_VALUE_1) != NUM_VALUE_0);
}

static void CheckScatterUpdateInvalid(const Tensor &dst, const Tensor &index, const Tensor &src)
{
    if (src.GetShape().size() != dst.GetShape().size()) {
        ALOG_ERROR_F("src.GetShape().size() == dst.GetShape().size()");
    }
    ASSERT(src.GetShape().size() == dst.GetShape().size());
    CheckScatterUpdateIndex(index);
    CheckScatterUpdateInput(src);
    CheckScatterUpdateInput(dst);
}

Tensor ScatterUpdate(const Tensor &dst, const Tensor &index, const Tensor &src, int axis, std::string cacheMode, int chunkSize) {
    DECLARE_TRACER();

    CheckScatterUpdateInvalid(dst, index, src);
    axis = axis < 0 ? dst->shape.size() + axis : axis;
    Tensor result(dst->tensor->datatype, dst->shape);
    result.GetStorage()->tensor->SetTensorInfo(dst.GetStorage()->tensor->GetTensorInfo());
    result.GetStorage()->tensorfmt = dst.GetStorage()->tensorfmt;

    if (cacheMode == "PA_NZ") {
        axis = 1;
        ASSERT(src->shape.size() == NUM_VALUE_2); // only support 2 dim

        Tensor newIndex = Reshape(index, {1, index->shape[0] * index->shape[1]});
        CALL(ScatterUpdate, *Program::GetInstance().GetCurrentFunction(), result.GetStorage(), dst.GetStorage(),
            newIndex.GetStorage(), src.GetStorage(), axis, cacheMode, chunkSize);
    } else {
        CALL(ScatterUpdate, *Program::GetInstance().GetCurrentFunction(), result.GetStorage(), dst.GetStorage(),
            index.GetStorage(), src.GetStorage(), axis, cacheMode, chunkSize);
    }
    return result;
}

static void CheckScatterElementSParamsInvalid(const Tensor &self, const Tensor &indices, int axis, 
    const std::string &reduceMode)
{
    ASSERT(self->shape.size() == indices->shape.size());
    ASSERT(axis < static_cast<int>(self->shape.size()));
    ASSERT(reduceMode.empty() || (reduceMode == "add") || (reduceMode == "multiply"));
    for (size_t i = 0; i < self->shape.size(); i++) {
        ASSERT(indices->shape[i] <= self->shape[i]);
    }
}

Tensor Scatter(const Tensor &self, const Tensor &indices, const Element &src, int axis, std::string reduce) {
    DECLARE_TRACER();

    axis = axis < 0 ? self->shape.size() + axis : axis;
    CheckScatterElementSParamsInvalid(self, indices, axis, reduce);
    Tensor result(self->tensor->datatype, self->shape);
    GraphUtils::AddDynOperation(*Program::GetInstance().GetCurrentFunction(), Opcode::OP_REGISTER_COPY, 
        {self.GetStorage()}, {result.GetStorage()});

    return Scatter_(result, indices, src, axis, reduce);
}

Tensor Scatter_(const Tensor &self, const Tensor &indices, const Element &src, int axis, std::string reduce) {
    DECLARE_TRACER();

    axis = axis < 0 ? self->shape.size() + axis : axis;
    CheckScatterElementSParamsInvalid(self, indices, axis, reduce);
    Tensor result(self->tensor->datatype, self->shape);
    result.GetStorage()->tensor->SetTensorInfo(self.GetStorage()->tensor->GetTensorInfo());
    CALL(ScatterElementS, *Program::GetInstance().GetCurrentFunction(), {result.GetStorage(), self.GetStorage(),
         indices.GetStorage(), src, axis, reduce});
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

void TensorInnerAssign(Function &function, const LogicalTensorPtr &operand, const LogicalTensorPtr &result) {
    function.AddOperation(Opcode::OP_REGISTER_COPY, {operand}, {result});
}

Tensor Assign(const Tensor &operand) {
    Tensor result(operand->Datatype(), operand->shape);
    CALL(InnerAssign, *Program::GetInstance().GetCurrentFunction(), operand.GetStorage(), result.GetStorage());
    return result;
}

#define CALL(n, ...) Tensor##n(__VA_ARGS__)
#define RETURN_CALL(n, ...) return Tensor##n(__VA_ARGS__)

void TiledInnerConcatLoop(const int dimIdx, Function &function, const TileShape &tileShape,
    const LogicalTensorPtr &operand, const LogicalTensorPtr &result,
    std::vector<int64_t> actTileShape, std::vector<int64_t> actOffset, std::vector<int64_t> tensorOffset)
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
    auto &vecTile = tileShape.GetVecTile();
    for (auto i = 0; i < operand->GetShape()[dimIdx]; i += vecTile[dimIdx]) {
        actTileShape[dimIdx] = std::min(operand->GetShape()[dimIdx] - i, vecTile[dimIdx]);
        actOffset[dimIdx] = i;
        TiledInnerConcatLoop(dimIdx + 1, function, tileShape, operand, result, actTileShape, actOffset, tensorOffset);
    }
}

void TiledInnerConcat(Function &function, const TileShape &tileShape,
    const LogicalTensorPtr &operand, const LogicalTensorPtr &result,
    std::vector<int64_t> tensorOffset)
{
    std::vector<int64_t> actOffset(operand->GetShape().size(), 0);
    std::vector<int64_t> actTileShape(operand->GetShape().size(), 1);
    TiledInnerConcatLoop(0, function, tileShape, operand, result, actTileShape, actOffset, tensorOffset);
}

void TensorInnerConcat(Function &function,
    const LogicalTensorPtr &operand, const LogicalTensorPtr &result,
    std::vector<int64_t> tensorOffset)
{
    auto &op = function.AddOperation(Opcode::OP_CONCAT, {operand}, {result});
    op.SetAttribute("concat", tensorOffset);
}

void InnerConcat(Function &function, const LogicalTensorPtr &operand,
    const LogicalTensorPtr &result, std::vector<int64_t> tensorOffset)
{
    CALL(InnerConcat, function, operand, result, tensorOffset);
}


void TiledInnerRegisterCopy(const int dimIdx, Function &function, const TileShape &tileShape,
    const LogicalTensorPtr &operand, const LogicalTensorPtr &result,
    std::vector<int64_t> actTileShape, std::vector<int64_t> actOffset)
{
    if (static_cast<size_t>(dimIdx)  == result->GetShape().size()) {
        auto inputTile = operand->View(function, actTileShape, actOffset);
        auto resultTile = result->View(function, actTileShape, actOffset);
        function.AddOperation("TILE_REGISTER_COPY", { inputTile }, { resultTile });
        return;
    }
    auto &vecTile = tileShape.GetVecTile();
    for (auto i = 0; i < result->GetShape()[dimIdx]; i += vecTile[dimIdx]) {
        actTileShape[dimIdx] = std::min(result->GetShape()[dimIdx] - i, vecTile[dimIdx]);
        actOffset[dimIdx] = i;
        TiledInnerRegisterCopy(dimIdx + 1, function, tileShape, operand, result, actTileShape, actOffset);
    }
}

void TiledInnerRegisterCopy(Function &function, const TileShape &tileShape,
    const LogicalTensorPtr &operand, const LogicalTensorPtr &result)
{
    std::vector<int64_t> actOffset(result->GetShape().size(), 0);
    std::vector<int64_t> actTileShape(result->GetShape().size(), 1);
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
    std::vector<int64_t> offset(shapeSize, 0);
    for (auto tensor : tensorList) {
        auto tmpView = tmp->View(function, tensor->GetShape(), offset);
        InnerConcatNew(*Program::GetInstance().GetCurrentFunction(), tensor.GetStorage(), tmpView);
        offset[axis] += tensor->GetShape()[axis];
    }
    auto &op = function.AddOperation(Opcode::OP_ASSEMBLE, {tmp.GetStorage()}, {result.GetStorage()});
    op.SetOpAttribute(std::make_shared<AssembleOpAttribute>(std::vector<int64_t>(shapeSize, 0)));

    return result;
}

void TiledPadOperation(Function &function, const std::vector<int64_t> &tileShape, const std::vector<int64_t> &tileOffset,
    const LogicalTensorPtr &result, const LogicalTensorPtr &operand)
{
    auto resultTile = result->View(function, tileShape, tileOffset);
    std::vector<int64_t> originTileShape(tileShape);
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

void TiledPadLoop(Function &function, int dimIdx, std::vector<int64_t> &tileShape, std::vector<int64_t> &tileOffset,
    const LogicalTensorPtr &result, const LogicalTensorPtr &operand,
    std::vector<int64_t> &cfgShape)
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
    std::vector<int64_t> offset(result->shape.size(), 0);
    std::vector<int64_t> padTileShape(result->GetShape().size(), 1);
    std::vector<int64_t> cfgShape(result->GetShape().size(), 1);
    auto &vecTile = tileShape.GetVecTile();
    cfgShape[cfgShape.size() - 1] = vecTile[1];
    cfgShape[cfgShape.size() - 2] = vecTile[0];
    TiledPadLoop(function, 0, padTileShape, offset, result, operand, cfgShape);
}

LogicalTensorPtr TensorPadOperation(Function &function, const TileShape &tileShape,
    const LogicalTensorPtr operand, const std::vector<int64_t> &newShape)
{
    auto tmpResult = std::make_shared<LogicalTensor>(function, operand->Datatype(), newShape);
    auto result = std::make_shared<LogicalTensor>(function, operand->Datatype(), newShape);
    TileInnerPad(function, tileShape, operand, tmpResult);
    auto &assembleOp = function.AddOperation(Opcode::OP_ASSEMBLE, {tmpResult}, {result});
    assembleOp.SetAssembleOpAttribute(std::vector<int64_t>(newShape.size(), 0));
    return result;
}

Tensor Pad(const Tensor &old, const std::vector<int64_t> &newShape)
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
    RETURN_CALL(
        PadOperation, *Program::GetInstance().GetCurrentFunction(), TileShape::Current(), old.GetStorage(), newShape);
}

void TiledInnerCompact(Function &function, const TileShape &tileShape,
    const LogicalTensorPtr &operand, const LogicalTensorPtr &result)
{
    assert(operand->shape.size() == operand->offset.size());
    auto workspace =
        std::make_shared<LogicalTensor>(function, operand->tensor->datatype,
                                       std::vector<int64_t>{ operand->shape[0], NUM_VALUE_8 });

    // 目前只支持2维操作
    if (operand->shape.size() != 2) {
        assert(false && "unsupported dimension");
    }
    auto &vecTile = tileShape.GetVecTile();
    int tileShape1 = std::min(operand->shape[1], vecTile[1]);
    for (int i = 0; i < operand->shape[0]; i += vecTile[0]) {
        int tileShape0 = std::min(operand->shape[0] - i, vecTile[0]);
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
    CALL(InnerCompact, *Program::GetInstance().GetCurrentFunction(), TileShape::Current(), operand.GetStorage(),
        result.GetStorage());
    return result;
}

template <typename T, DataType dataType>
Element GetCurStartElement(Element start, Element step, int id){
    T startValue;
    T stepValue;
    if (dataType == DT_INT32 || dataType == DT_INT64) {
        startValue = start.GetSignedData();
        stepValue = step.GetSignedData();
    } else if (dataType == DT_FP32) {
        startValue = (float)start.GetFloatData();
        stepValue = (float)step.GetFloatData();
    }
    T curStartValue = startValue + id * stepValue;
    Element curStart(dataType, curStartValue);
    return curStart;
}

void TiledRange(Function &function, const TileShape &tileShape, const Element start, const Element step,
    const LogicalTensorPtr &result) {
    TileInfo resultTileInfo(result->shape.size(), result->offset.size());
    auto &vecTile = tileShape.GetVecTile();
    for (int i = 0; i < result->shape[0]; i += vecTile[0]) {
        resultTileInfo.offset[0] = i;
        resultTileInfo.shape[0] = std::min(result->shape[0] - resultTileInfo.offset[0], vecTile[0]);
        int64_t curSizeValue = resultTileInfo.shape[0];
        Element curSize(DT_INT64, curSizeValue);
        Element curStart;
        if (start.GetDataType() == DT_INT32) {
            curStart = GetCurStartElement<int32_t, DT_INT32>(start, step, i);
        } else if (start.GetDataType() == DT_INT64) {
            curStart = GetCurStartElement<int64_t, DT_INT64>(start, step, i);
        } else if (start.GetDataType() == DT_FP32) {
            curStart = GetCurStartElement<float, DT_FP32>(start, step, i);
        } else {
            throw std::invalid_argument("Unknown DataType");
        }
        auto resultTile = result->View(function, resultTileInfo.shape, resultTileInfo.offset);
        auto &op = function.AddOperation(Opcode::OP_RANGE, {}, {resultTile});
        op.SetAttribute(OP_ATTR_PREFIX + "START", curStart);
        op.SetAttribute(OP_ATTR_PREFIX + "SIZE", curSize);
        op.SetAttribute(OP_ATTR_PREFIX + "STEP", step);
    }
    return;
}

LogicalTensorPtr TensorRange(Function &function, LogicalTensorPtr &result, Element &start, Element &step) {
    auto &op = function.AddOperation(Opcode::OP_RANGE, {}, {result});
    op.SetAttribute(OP_ATTR_PREFIX + "START", start);
    op.SetAttribute(OP_ATTR_PREFIX + "STEP", step);
    return result;
}

inline long LongCeilDiv(long a, long b){
    if(b == 0){
        return 0;
    }
    return (a + (b - 1)) / b;
}

const float EPSILON = (float)1e-8;
template <typename T, DataType dataType>
int64_t GetRangeResSize(Element &start, Element &end, Element &step){
    int64_t resultSize;
    if(dataType == DT_INT32 || dataType == DT_INT64){
        T startValue = start.GetSignedData();
        T endValue = end.GetSignedData();
        T stepValue = step.GetSignedData();
        assert(abs(stepValue) > 0);
        resultSize = static_cast<int64_t>(LongCeilDiv(static_cast<int64_t>(endValue) - static_cast<int64_t>(startValue), static_cast<int64_t>(stepValue)));
    }else if(dataType == DT_FP32){
        T startValue = (float)start.GetFloatData();
        T endValue = (float)end.GetFloatData();
        T stepValue = (float)step.GetFloatData();
        assert(abs(stepValue) > EPSILON);
        resultSize = static_cast<int64_t>(std::ceil((endValue - startValue) / stepValue));
    }
    return resultSize;
}

Tensor RealRange(Element &start, Element &end, Element &step) {
    DECLARE_TRACER();
    std::vector<int64_t> resTensorShape;
    int64_t resultSize;
    if (start.GetDataType() == DT_INT32) {
        resultSize = GetRangeResSize<int32_t, DT_INT32>(start, end, step);
    } else if(start.GetDataType() == DT_INT64) {
        resultSize = GetRangeResSize<int64_t, DT_INT64>(start, end, step);
    } else if(start.GetDataType() == DT_FP32) {
        resultSize = GetRangeResSize<float, DT_FP32>(start, end, step);
    } else {
        throw std::invalid_argument("Unknown DataType");
    }    
    assert(resultSize >= 0);
    resTensorShape.push_back(resultSize);
    auto resTensor = Tensor(start.GetDataType(), resTensorShape);
    RETURN_CALL(Range, *Program::GetInstance().GetCurrentFunction(), resTensor.GetStorage(), start, step);
}

DataType GetResultDataType(const Element &start, const Element &end, const Element &step) {
    DataType startType = start.GetDataType();
    DataType endType = end.GetDataType();
    DataType stepType = step.GetDataType();
    assert(startType == DT_FP32 || startType == DT_INT64 || startType == DT_INT32);
    assert(endType == DT_FP32 || endType == DT_INT64 || endType == DT_INT32);
    assert(stepType == DT_FP32 || stepType == DT_INT64 || stepType == DT_INT32);
    if (startType == DT_FP32 || endType == DT_FP32 || stepType == DT_FP32) {
        return DT_FP32;
    }
    int64_t startValue = start.GetSignedData();
    int64_t endValue = end.GetSignedData();
    int64_t stepValue = step.GetSignedData();
    bool startFlag = startValue <= INT_MAX_VALUE && startValue >= INT_MIN_VALUE;
    bool endFlag = endValue <= INT_MAX_VALUE && endValue >= INT_MIN_VALUE;
    bool stepFlag = stepValue <= INT_MAX_VALUE && stepValue >= INT_MIN_VALUE;
    if (startFlag && endFlag && stepFlag) {
        return DT_INT32;
    }
    return DT_INT64;
}

Element GetElementWithDataType(const Element &element, DataType dataType) {
    if (element.GetDataType() == DT_FP32) {
        return Element(dataType, element.GetFloatData());
    }
    return Element(dataType, element.GetSignedData());
}

Tensor Range(const Element &start, const Element &end, const Element &step) {
    DataType dataType = GetResultDataType(start, end, step);
    assert(dataType == DT_FP32 || dataType == DT_INT32);
    Element realStart = GetElementWithDataType(start, dataType);
    Element realEnd = GetElementWithDataType(end, dataType);
    Element realStep = GetElementWithDataType(step, dataType);
    return RealRange(realStart, realEnd, realStep);
}

const std::string TOPK_AXIS = OP_ATTR_PREFIX + "axis";
const std::string TOPK_ORDER = OP_ATTR_PREFIX + "order";
const std::string TOPK_KVALUE = OP_ATTR_PREFIX + "kvalue";
const std::string EXTRACT_MASKMODE = OP_ATTR_PREFIX + "makeMode";
const std::string TOPK_OFFSET = OP_ATTR_PREFIX + "offset";
const std::string TOPK_VALIDBIT = OP_ATTR_PREFIX + "validBit";

// 针对axis全排序,当前只支持axis为-1,输出为结果每32数据排序
void TensorBitsortOperation(Function &function,
    LogicalTensorPtr operand, LogicalTensorPtr resOp, int axis, bool isLargest) {
    auto &op = function.AddOperation(Opcode::OP_BITSORT, {operand}, {resOp});
    op.SetAttribute(TOPK_AXIS, axis);
    op.SetAttribute(TOPK_ORDER, static_cast<int>(isLargest));
    op.SetAttribute(TOPK_OFFSET, static_cast<int>(0));
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

void TensorTopK(Function &function, const LogicalTensorPtr &operand, LogicalTensorPtr &valueResult,
    LogicalTensorPtr &indexResult, int k, int axis, bool isLargest) {
    if (!operand->GetDynValidShape().empty()) {
        std::vector<SymbolicScalar> outValidShape;
        for (auto shape : operand->GetDynValidShape()) {
            outValidShape.push_back(shape);
        }
        outValidShape[axis] = SymbolicScalar(k);
        valueResult->UpdateDynValidShape(outValidShape);
        indexResult->UpdateDynValidShape(outValidShape);
    }

    auto &op = function.AddOperation(Opcode::OP_TOPK, {operand}, {valueResult, indexResult});
    op.SetAttribute(TOPK_AXIS, axis);
    op.SetAttribute(TOPK_KVALUE, k);
    op.SetAttribute(TOPK_ORDER, static_cast<int>(isLargest));
    return;
}

std::tuple<Tensor, Tensor> TopK(const Tensor &operand, const int &k, int axis = -1, bool isLargest) {
    DECLARE_TRACER();
    const auto len = static_cast<int>(operand->shape.size());
    ASSERT(axis == (len - 1) || axis == -1) << "TopK only support last axis";
    axis = axis >= 0 ? axis : (axis + len);

    auto topkOutShape = operand->shape;
    topkOutShape[axis] = k;
    auto valueResult = Tensor(operand->tensor->datatype, topkOutShape);
    auto indexResult = Tensor(DataType::DT_INT32, topkOutShape);
    CALL(TopK, *Program::GetInstance().GetCurrentFunction(), operand.GetStorage(), valueResult.GetStorage(),
     indexResult.GetStorage(), k, axis, isLargest);
    return std::tie(valueResult, indexResult);
}

void TiledBitSort(Function &function, const TileShape & tileShape, size_t cur, Input &input,
    const LogicalTensorPtr &result, TileInfo &resultTileInfo, int axis, int isLargest) {
    if (cur == input.tensor->shape.size()) {
        auto inputTile = input.tensor->View(function, input.tileInfo.shape, input.tileInfo.offset);
        auto resultTile = result->View(function, resultTileInfo.shape, resultTileInfo.offset);
        auto &op = function.AddOperation(Opcode::OP_BITSORT, {inputTile}, {resultTile});
        op.SetAttribute(TOPK_AXIS, axis);
        op.SetAttribute(TOPK_ORDER, static_cast<int>(isLargest));
        op.SetAttribute(TOPK_OFFSET, static_cast<int>(0));
        return;
    }
    // Jump cur axis
    if (cur == static_cast<size_t>(axis)) {
        TiledBitSort(function, tileShape, cur + 1, input, result, resultTileInfo, axis, isLargest);
        return;
    }
    auto &vecTile = tileShape.GetVecTile();
    for (int i = 0; i < input.tensor->shape[cur]; i += vecTile[cur]) {
        // update input && result && resultDices shape and offset info
        input.tileInfo.offset[cur] = i % input.tensor->shape[cur];
        input.tileInfo.shape[cur] = std::min(input.tensor->shape[cur] - input.tileInfo.offset[cur], vecTile[cur]);

        resultTileInfo.offset[cur] = i;
        resultTileInfo.shape[cur] = std::min(result->shape[cur] - resultTileInfo.offset[cur], vecTile[cur]);
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
    auto &vecTile = tileShape.GetVecTile();
    for (int i = 0; i < input.tensor->shape[cur]; i += vecTile[cur]) {
        // update input && result && resultDices shape and offset info
        input.tileInfo.offset[cur] = i % input.tensor->shape[cur];
        input.tileInfo.shape[cur] = std::min(input.tensor->shape[cur] - input.tileInfo.offset[cur], vecTile[cur]);

        resultTileInfo.offset[cur] = i;
        resultTileInfo.shape[cur] = std::min(result->shape[cur] - resultTileInfo.offset[cur], vecTile[cur]);
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

void TiledTopK(Function &function, const TileShape & tileShape, size_t cur, Input &input,
    const LogicalTensorPtr &valueResult, const LogicalTensorPtr &indexResult, TileInfo &resultTileInfo,
    int axis, int k, int isLargest) {
    auto &vecTile = tileShape.GetVecTile();
    ASSERT(k <= vecTile[axis]);
    if (static_cast<int>(cur) == axis) {
        auto source = input.tensor->View(function, input.tileInfo.shape, input.tileInfo.offset);
        constexpr int32_t blockSize = 32;
        constexpr int32_t kFactorSize = 4;
        constexpr int32_t kBlockFpNum = 8;
        std::vector<int64_t> vecTileAlign = vecTile.tile;
        vecTileAlign[axis] = (vecTile[axis] + blockSize - 1) / blockSize * blockSize;
        auto axisTileNum = (source->shape[axis] + vecTileAlign[axis] - 1) / vecTileAlign[axis];

        auto bitsortShape = source->shape;
        auto axisBlockSizeAlign = vecTileAlign[axis];
        bitsortShape[axis] = axisTileNum * axisBlockSizeAlign * kFactorSize;
        auto bitsortResult = std::make_shared<LogicalTensor>(function, source->Datatype(),
            bitsortShape, source->GetDynValidShape());
        auto mrgsortShape = source->shape;
        mrgsortShape[axis] = (k + kBlockFpNum - 1) / kBlockFpNum * kBlockFpNum * NUM_VALUE_2 * axisTileNum;
        auto mrgsortResult0 = std::make_shared<LogicalTensor>(function, source->Datatype(),
            mrgsortShape, source->GetDynValidShape());

        std::vector<int64_t> tileBitsortShape = bitsortShape;
        std::vector<int64_t> tileBitsortOffset(tileBitsortShape.size(), 0);
        std::vector<int64_t> tileMrgsortShape = mrgsortShape;
        std::vector<int64_t> tileMrgsortOffset(tileMrgsortShape.size(), 0);
        std::vector<int64_t> tileSourceShape = source->shape;
        std::vector<int64_t> tileSourceOffset(tileSourceShape.size(), 0);

        LogicalTensorPtr mrgsortTile;
        std::vector<LogicalTensorPtr> sortList;
        std::vector<int64_t> mrgsortTileShape;
        for (int i = 0; i < input.tensor->shape[axis]; i += vecTileAlign[axis]) {
            tileSourceShape[axis] = std::min(vecTileAlign[axis], source->shape[axis] - i);
            tileSourceOffset[axis] = i;
            auto inputTile = source->View(function, tileSourceShape, tileSourceOffset);
            auto tileBitsortRemain = (source->shape[axis] - i + blockSize -1) / blockSize * blockSize;
            tileBitsortShape[axis] = std::min(axisBlockSizeAlign * kFactorSize, tileBitsortRemain * kFactorSize);
            tileBitsortOffset[axis] = static_cast<int64_t>(i / vecTileAlign[axis] * axisBlockSizeAlign * kFactorSize);
            auto bitsortTile = bitsortResult->View(function, tileBitsortShape, tileBitsortOffset);
            auto &bitsortOp = function.AddOperation(Opcode::OP_BITSORT, {inputTile}, {bitsortTile});
            bitsortOp.SetAttribute(TOPK_AXIS, axis);
            bitsortOp.SetAttribute(TOPK_ORDER, static_cast<int>(isLargest));
            bitsortOp.SetAttribute(TOPK_OFFSET, static_cast<int>(i));

            tileMrgsortShape[axis] = (k + kBlockFpNum - 1) / kBlockFpNum * kBlockFpNum * NUM_VALUE_2;
            tileMrgsortOffset[axis] = static_cast<int64_t>(i / vecTileAlign[axis] * tileMrgsortShape[axis]);
            mrgsortTile = mrgsortResult0->View(function, tileMrgsortShape, tileMrgsortOffset);
            auto &mrgsortOp = function.AddOperation(Opcode::OP_MRGSORT, {bitsortTile}, {mrgsortTile});
            mrgsortOp.SetAttribute(TOPK_AXIS, axis);
            mrgsortOp.SetAttribute(TOPK_KVALUE, k);
            mrgsortOp.SetAttribute(TOPK_ORDER, static_cast<int>(isLargest));
            sortList.push_back(mrgsortTile);
        }
        std::vector<int64_t> mrgsortResultOffset(mrgsortShape.size(), 0);
        std::vector<int64_t> tempShape = sortList[0]->shape;
        tempShape[axis] = NUM_VALUE_4 * tempShape[axis];
        for (size_t i = 0; i < tempShape.size() - 1; ++i) {
            tempShape[i] = 1;
        }
        tileMrgsortShape[axis] = tileMrgsortShape[axis] * axisTileNum;
        auto mrgsortBuffer = std::make_shared<LogicalTensor>(function, valueResult->Datatype(),
            tileMrgsortShape, source->GetDynValidShape());
        std::vector<LogicalTensorPtr> tiledMrgsortList;
        for (int i = 0; i < axisTileNum; i+=NUM_VALUE_4) {
            if ((axisTileNum - i) == NUM_VALUE_3) {
                auto tempTensor = std::make_shared<LogicalTensor>(function, valueResult->Datatype(),
                    tempShape, source->GetDynValidShape());
                mrgsortResultOffset[axis] = i / NUM_VALUE_4 * sortList[0]->shape[axis];
                auto mrgsortRepeatResult = mrgsortBuffer->View(function, sortList[0]->shape, mrgsortResultOffset);
                auto &mrgSortMultiQue = function.AddOperation(Opcode::OP_TILEDMRGSORT, {sortList[i],
                    sortList[i + 1], sortList[i + NUM_VALUE_2], sortList[i + NUM_VALUE_2]},
                    {mrgsortRepeatResult, tempTensor});
                mrgSortMultiQue.SetAttribute(TOPK_VALIDBIT, NUM_VALUE_3);
                mrgSortMultiQue.SetAttribute(TOPK_KVALUE, k);
                tiledMrgsortList.push_back(mrgsortRepeatResult);
            } else if ((axisTileNum - i) == NUM_VALUE_2) {
                auto tempTensor = std::make_shared<LogicalTensor>(function, valueResult->Datatype(),
                    tempShape, source->GetDynValidShape());
                mrgsortResultOffset[axis] = i / NUM_VALUE_4 * sortList[0]->shape[axis];
                auto mrgsortRepeatResult = mrgsortBuffer->View(function, sortList[0]->shape, mrgsortResultOffset);
                auto &mrgSortMultiQue = function.AddOperation(Opcode::OP_TILEDMRGSORT, {sortList[i],
                    sortList[i + 1], sortList[i + 1], sortList[i + 1]},
                    {mrgsortRepeatResult, tempTensor});
                mrgSortMultiQue.SetAttribute(TOPK_VALIDBIT, NUM_VALUE_2);
                mrgSortMultiQue.SetAttribute(TOPK_KVALUE, k);
                tiledMrgsortList.push_back(mrgsortRepeatResult);
            } else if ((axisTileNum - i) == 1) {
                tiledMrgsortList.push_back(sortList[i]);
            } else {
                auto tempTensor = std::make_shared<LogicalTensor>(function, valueResult->Datatype(),
                    tempShape, source->GetDynValidShape());
                mrgsortResultOffset[axis] = i / NUM_VALUE_4 * sortList[0]->shape[axis];
                auto mrgsortRepeatResult = mrgsortBuffer->View(function, sortList[0]->shape, mrgsortResultOffset);
                auto &mrgSortMultiQue = function.AddOperation(Opcode::OP_TILEDMRGSORT, {sortList[i],
                    sortList[i + 1], sortList[i + NUM_VALUE_2], sortList[i + NUM_VALUE_3]},
                    {mrgsortRepeatResult, tempTensor});
                mrgSortMultiQue.SetAttribute(TOPK_VALIDBIT, NUM_VALUE_4);
                mrgSortMultiQue.SetAttribute(TOPK_KVALUE, k);
                tiledMrgsortList.push_back(mrgsortRepeatResult);
            }
        }
        int roundNum = 0;
        int width = 1;
        while (width < axisTileNum) {
            width = width << NUM_VALUE_2;
            roundNum++;
        }
        int tileResultIdx = 0;
        for (int i = 1; i < roundNum; ++i) {
            int tileResultNum = tiledMrgsortList.size();
            for (int j = tileResultIdx; j < tileResultNum; j+=NUM_VALUE_4) {
                if ((tileResultNum - j) == NUM_VALUE_3) {
                    auto tempTensor = std::make_shared<LogicalTensor>(function, valueResult->Datatype(),
                        tempShape, source->GetDynValidShape());
                    mrgsortResultOffset[axis] = tileResultNum * sortList[0]->shape[axis];
                    auto mrgsortRepeatResult = mrgsortBuffer->View(function, sortList[0]->shape, mrgsortResultOffset);
                    auto &mrgSortMultiQue = function.AddOperation(Opcode::OP_TILEDMRGSORT, {tiledMrgsortList[j],
                        tiledMrgsortList[j + 1], tiledMrgsortList[j + NUM_VALUE_2], tiledMrgsortList[j + NUM_VALUE_2]},
                        {mrgsortRepeatResult, tempTensor});
                    mrgSortMultiQue.SetAttribute(TOPK_VALIDBIT, NUM_VALUE_3);
                    mrgSortMultiQue.SetAttribute(TOPK_KVALUE, k);
                    tiledMrgsortList.push_back(mrgsortRepeatResult);
                } else if ((tileResultNum - j) == NUM_VALUE_2) {
                    auto tempTensor = std::make_shared<LogicalTensor>(function, valueResult->Datatype(),
                        tempShape, source->GetDynValidShape());
                    mrgsortResultOffset[axis] = tileResultNum * sortList[0]->shape[axis];
                    auto mrgsortRepeatResult = mrgsortBuffer->View(function, sortList[0]->shape, mrgsortResultOffset);
                    auto &mrgSortMultiQue = function.AddOperation(Opcode::OP_TILEDMRGSORT, {tiledMrgsortList[j],
                        tiledMrgsortList[j + 1], tiledMrgsortList[j + 1], tiledMrgsortList[j + 1]},
                        {mrgsortRepeatResult, tempTensor});
                    mrgSortMultiQue.SetAttribute(TOPK_VALIDBIT, NUM_VALUE_2);
                    mrgSortMultiQue.SetAttribute(TOPK_KVALUE, k);
                    tiledMrgsortList.push_back(mrgsortRepeatResult);
                } else if ((tileResultNum - j) == 1) {
                    tiledMrgsortList.push_back(tiledMrgsortList[j]);
                } else {
                    auto tempTensor = std::make_shared<LogicalTensor>(function, valueResult->Datatype(),
                        tempShape, source->GetDynValidShape());
                    mrgsortResultOffset[axis] = tileResultNum * sortList[0]->shape[axis];
                    auto mrgsortRepeatResult = mrgsortBuffer->View(function, sortList[0]->shape, mrgsortResultOffset);
                    auto &mrgSortMultiQue = function.AddOperation(Opcode::OP_TILEDMRGSORT, {tiledMrgsortList[j],
                        tiledMrgsortList[j + 1], tiledMrgsortList[j + NUM_VALUE_2], tiledMrgsortList[j + NUM_VALUE_3]},
                        {mrgsortRepeatResult, tempTensor});
                    mrgSortMultiQue.SetAttribute(TOPK_VALIDBIT, NUM_VALUE_4);
                    mrgSortMultiQue.SetAttribute(TOPK_KVALUE, k);
                    tiledMrgsortList.push_back(mrgsortRepeatResult);
                }
            }
            tileResultIdx = tileResultNum;
        }
        auto valueTile = valueResult->View(function, resultTileInfo.shape, resultTileInfo.offset);
        auto &valueOp = function.AddOperation(Opcode::OP_EXTRACT, {tiledMrgsortList.back()}, {valueTile});
        valueOp.SetAttribute(EXTRACT_MASKMODE, 0);
        valueOp.SetAttribute(TOPK_KVALUE, k);
        valueOp.SetAttribute(TOPK_ORDER, static_cast<int>(isLargest));

        auto indexTile = indexResult->View(function, resultTileInfo.shape, resultTileInfo.offset);
        auto &indexOp = function.AddOperation(Opcode::OP_EXTRACT, {tiledMrgsortList.back()}, {indexTile});
        indexOp.SetAttribute(EXTRACT_MASKMODE, 1);
        indexOp.SetAttribute(TOPK_KVALUE, k);
        indexOp.SetAttribute(TOPK_ORDER, static_cast<int>(isLargest));
        return;
    }

    for (int i = 0; i < input.tensor->shape[cur]; i += vecTile[cur]) {
        // update input && result && resultDices shape and offset info
        input.tileInfo.offset[cur] = i % input.tensor->shape[cur];
        input.tileInfo.shape[cur] = std::min(input.tensor->shape[cur] - input.tileInfo.offset[cur], vecTile[cur]);

        resultTileInfo.offset[cur] = i;
        resultTileInfo.shape[cur] = std::min(valueResult->shape[cur] - resultTileInfo.offset[cur], vecTile[cur]);
        TiledTopK(function, tileShape, cur + 1, input, valueResult, indexResult, resultTileInfo, axis, k, isLargest);
    }
}

void TiledTopK(Function &function, const TileShape &tileShape,
     const LogicalTensorPtr operand, const LogicalTensorPtr valueResult, const LogicalTensorPtr indexResult,
     int axis, int k, int isLargest) {
    // Build Init tile info
    TileInfo tileInfo(operand->shape, operand->offset);
    TileInfo resultTileInfo(valueResult->shape, valueResult->offset);
    auto input = Input{operand, tileInfo};
    TiledTopK(function, tileShape, 0, input, valueResult, indexResult, resultTileInfo, axis, k, isLargest);
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

    auto &vecTile = tileShape.GetVecTile();
    for (int i = 0; i < input.tensor->shape[cur]; i += vecTile[cur]) {
        // update input && result && resultDices shape and offset info
        input.tileInfo.offset[cur] = i % input.tensor->shape[cur];
        input.tileInfo.shape[cur] = std::min(input.tensor->shape[cur] - input.tileInfo.offset[cur], vecTile[cur]);

        resultTileInfo.offset[cur] = i;
        resultTileInfo.shape[cur] = std::min(result->shape[cur] - resultTileInfo.offset[cur], vecTile[cur]);
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
    auto &vecTile = tileShape.GetVecTile();
    for (int i = 0; i < input.tensor->shape[cur]; i += vecTile[cur]) {
        // update input && result && resultDices shape and offset info
        input.tileInfo.offset[cur] = i % input.tensor->shape[cur];
        input.tileInfo.shape[cur] = std::min(input.tensor->shape[cur] - input.tileInfo.offset[cur], vecTile[cur]);

        resultDicesTileInfo.offset[cur] = i;
        resultDicesTileInfo.shape[cur] =
            std::min(resultDices->shape[cur] - resultDicesTileInfo.offset[cur], vecTile[cur]);
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
    Tensor result(o0->Datatype(), o0->shape, "", o0->GetTileOpFormat());
    auto& op = Program::GetInstance().AddOperation(Opcode::OP_REDUCE_ACC, iOperand, { result.GetStorage() });
    op.SetAttribute(Matrix::ACC_A_MUL_B, 1);
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
        op.SetAttribute(Matrix::ACC_A_MUL_B, 1);
        return;
    }
    auto &vecTile = tileShape.GetVecTile();
    for (auto i = 0; i < result->shape[cur]; i += vecTile[cur]) {
        resultTileInfo.offset[cur] = i;
        resultTileInfo.shape[cur] = std::min(result->shape[cur] - resultTileInfo.offset[cur], vecTile[cur]);
        for (size_t index = 0; index < inputVec.size(); ++index) {
            inputVec[index].tileInfo.offset[cur]  = i % inputVec[index].tensor->shape[cur];
            inputVec[index].tileInfo.shape[cur] =
                std::min(inputVec[index].tensor->shape[cur] - inputVec[index].tileInfo.offset[cur], vecTile[cur]);
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

// parallel sort
const std::string SORT_ORDER = OP_ATTR_PREFIX + "order";
const std::string SORT_START_INDEX = OP_ATTR_PREFIX + "start_index";
const std::string SORT_FULL = OP_ATTR_PREFIX + "full_sort";

void TiledSort(Function &function, const LogicalTensorPtr &x, const LogicalTensorPtr &y, const LogicalTensorPtr &yIdx, const LogicalTensorPtr &temp, int idxStart, int descending) {
    auto &op = function.AddOperation(Opcode::OP_SORT, {x}, {y, yIdx, temp});
    op.SetAttribute(SORT_START_INDEX, static_cast<int>(idxStart));
    op.SetAttribute(SORT_ORDER, static_cast<int>(descending));
    std::map<int, int> inplaceInfo = {{0, 0}};
    op.SetAttr(OpAttributeKey::inplaceInfo, inplaceInfo);
}

std::tuple<Tensor, Tensor, Tensor> L1Sort(const Tensor &x, int idxStart, bool descending) {
    constexpr int32_t kFactorSize = NUM_VALUE_4;
    auto tempShape = x->shape;
    tempShape[1] *= kFactorSize;
    auto y = Tensor(x->tensor->datatype, x->shape);
    auto yIdx = Tensor(DataType::DT_INT32, x->shape);
    auto temp = Tensor(x->tensor->datatype, tempShape);
    TiledSort(*Program::GetInstance().GetCurrentFunction(), x.GetStorage(), y.GetStorage(), yIdx.GetStorage(), temp.GetStorage(), idxStart, descending);
    return std::tie(y, yIdx, temp);
}

void TiledCompareAndSwap(Function &function, const LogicalTensorPtr &x0, const LogicalTensorPtr &idx0, const LogicalTensorPtr &x1, const LogicalTensorPtr &idx1, 
    const LogicalTensorPtr &y0, const LogicalTensorPtr &yIdx0, const LogicalTensorPtr &y1, const LogicalTensorPtr &yIdx1, int descending) {
    auto &op = function.AddOperation(Opcode::OP_COMPARE_SWAP, {x0, idx0, x1, idx1}, {y0, yIdx0, y1, yIdx1});
    op.SetAttribute(SORT_ORDER, static_cast<int>(descending));
    std::map<int, int> inplaceInfo = {{0, 0}, {1, 1}};
    op.SetAttr(OpAttributeKey::inplaceInfo, inplaceInfo);
}

std::tuple<Tensor, Tensor, Tensor, Tensor> L1CompareAndSwap(const Tensor &x0, const Tensor &idx0, const Tensor &x1, const Tensor &idx1, bool descending) {
    Tensor y0(x0->Datatype(), x0->shape);
    Tensor yIdx0(idx0->Datatype(), idx0->shape);
    Tensor y1(x1->Datatype(), x1->shape);
    Tensor yIdx1(idx1->Datatype(), idx1->shape);
    TiledCompareAndSwap(*Program::GetInstance().GetCurrentFunction(), x0.GetStorage(), idx0.GetStorage(), x1.GetStorage(), idx1.GetStorage(), 
        y0.GetStorage(), yIdx0.GetStorage(), y1.GetStorage(), yIdx1.GetStorage(), descending);
    return std::tie(y0, yIdx0, y1, yIdx1);
}

void TiledMerge(Function &function, const LogicalTensorPtr &x, const LogicalTensorPtr &idx, const LogicalTensorPtr &y, const LogicalTensorPtr &yIdx, const LogicalTensorPtr &temp, int fullSort, int descending) {
    auto &op = function.AddOperation(Opcode::OP_MERGE, {x, idx}, {y, yIdx, temp});
    op.SetAttribute(SORT_ORDER, static_cast<int>(descending));
    op.SetAttribute(SORT_FULL, static_cast<int>(fullSort));
    std::map<int, int> inplaceInfo = {{0, 0}, {1, 1}};
    op.SetAttr(OpAttributeKey::inplaceInfo, inplaceInfo);
}

std::tuple<Tensor, Tensor, Tensor> L1Merge(const Tensor &x, const Tensor &idx, bool descending, bool fullSort) {
    constexpr int32_t kFactorSize = NUM_VALUE_4;
    auto tempShape = x->shape;
    tempShape[1] *= kFactorSize;
    auto y = Tensor(x->tensor->datatype, x->shape);
    auto yIdx = Tensor(idx->tensor->datatype, idx->shape);
    auto temp = Tensor(x->tensor->datatype, tempShape);
    TiledMerge(*Program::GetInstance().GetCurrentFunction(), x.GetStorage(), idx.GetStorage(), y.GetStorage(), yIdx.GetStorage(), temp.GetStorage(), fullSort, descending);
    return std::tie(y, yIdx, temp);
}

using SortTileMap = std::map<int, std::tuple<Tensor, Tensor>>;

bool IsMaxTile(SortTileMap &map, int index) {
    return map.find(index) == map.end();
}

void CompareAndSwapStep(SortTileMap &tileMap, int offset, int mergeSize, bool descending) {
    int nTile = mergeSize;
    for (int step = nTile; step >= NUM2; step /= NUM2) {
        for (int start = 0; start < nTile * NUM2; start += step * NUM2) {
            // within each swap size = step * tileSize
            for (int i = 0; i < step; i++) {
                int idx0 = offset + start + i;
                int idx1 = idx0 + step;

                // no need to comp & swap
                if (IsMaxTile(tileMap, idx0) && descending) {
                    continue;
                }
                if (IsMaxTile(tileMap, idx1) && !descending) {
                    continue;
                }
                if (IsMaxTile(tileMap, idx0) && !descending) {
                    tileMap[idx0] = tileMap[idx1];
                    tileMap.erase(idx1);
                    continue;
                } else if (IsMaxTile(tileMap, idx1) && descending) {
                    tileMap[idx1] = tileMap[idx0];
                    tileMap.erase(idx0);
                    continue;
                }

                // use L1CompareAndSwap
                auto [x0, xIdx0] = tileMap[idx0];
                auto [x1, xIdx1] = tileMap[idx1];
                auto [y0, yIdx0, y1, yIdx1] = L1CompareAndSwap(x0, xIdx0, x1, xIdx1, descending);
                tileMap[idx0] = std::tie(y0, yIdx0);
                tileMap[idx1] = std::tie(y1, yIdx1);
            }
        }
    }
}

void MergeStep(SortTileMap &tileMap, int offset, int mergeSize, int tileSize, bool descending) {
    // Compare & Swap
    CompareAndSwapStep(tileMap, offset, mergeSize, descending);

    // maxStep is the minimum orders of 2 that >= n
    int n = tileMap.size() / NUM2;
    int maxStep = 1;
    while (maxStep < n) {
        maxStep <<= 1;
    }
    int halfSize = tileSize / NUM2;

    // Merge within each tile
    for (int i = 0; i < mergeSize; i++) {
        int idx0 = offset + NUM2 * i;
        int idx1 = idx0 + 1;
        if (IsMaxTile(tileMap, idx0) || IsMaxTile(tileMap, idx1)) {
            continue;
        }
        auto [x0, xIdx0] = tileMap[idx0];
        auto [x1, xIdx1] = tileMap[idx1];
        Tensor src(x0.GetDataType(), {1, tileSize});
        Tensor srcIdx(DT_INT32, {1, tileSize});
        Assemble(x0, {0, 0}, src);
        Assemble(x1, {0, halfSize}, src);
        Assemble(xIdx0, {0, 0}, srcIdx);
        Assemble(xIdx1, {0, halfSize}, srcIdx);
        auto mergeResult = L1Merge(src, srcIdx, descending, false);
        auto res = std::get<0>(mergeResult);
        auto resIdx = std::get<1>(mergeResult);
        
        if (mergeSize < maxStep) {
            tileMap[idx0] = {View(res, {1, halfSize}, {0, 0}), View(resIdx, {1, halfSize}, {0, 0})};
            tileMap[idx1] = {View(res, {1, halfSize}, {0, halfSize}), View(resIdx, {1, halfSize}, {0, halfSize})};
        } else {
            // For assemble, no need to split into half
            tileMap[idx0] = {res, resIdx};
        }
    }
}

bool IsPowerOfTwo(int n) {
    return (n & (n - 1)) == 0;
}

int NextPowerofTwo(int n) {
    int power = 1;
    while (power < n) {
        power <<= 1;
    }
    return power;
}

std::tuple<Tensor, Tensor> Sort(const Tensor &x, bool descending) {
    DECLARE_TRACER();
    ASSERT(x->shape.size() == NUM2);
    ASSERT(x->shape[0] == 1);
    auto &vecTile = TileShape::Current().GetVecTile();
    ASSERT(vecTile.size() == NUM2);
    ASSERT(vecTile[0] == 1);
    auto tileSize = vecTile[1];
    ASSERT(IsPowerOfTwo(tileSize));
    int length = x->shape[1];
    int padLength = NextPowerofTwo(length);

    int nTile = padLength / tileSize;
    int halfSize = tileSize / NUM2;
    SortTileMap tileMap;

    if (nTile <= 1) {
        auto res = L1Sort(x, 0, descending);
        auto y = std::get<0>(res);
        auto yIdx = std::get<1>(res);
        return std::tie(y, yIdx);
    }

    // Tile Sort
    for (int i = 0; i < nTile; i++) {
        bool flag = (i % NUM2 == (descending ? 0 : 1));
        int idxStart = i;
        auto src = View(x, {1, tileSize}, {0, tileSize * i});
        auto sortResult = L1Sort(src, idxStart, flag);
        auto res = std::get<0>(sortResult);
        auto resIdx = std::get<1>(sortResult);
        tileMap[i * NUM2] = {View(res, {1, halfSize}, {0, 0}), View(resIdx, {1, halfSize}, {0, 0})};
        tileMap[i * NUM2 + 1] = {View(res, {1, halfSize}, {0, halfSize}), View(resIdx, {1, halfSize}, {0, halfSize})};
    }

    // Merge
    for (int step = NUM2; step <= nTile; step *= NUM2) {
        for (int i = 0; i < nTile / step; ++i) {
            int offset = i * step * NUM2;
            bool flag = (i % NUM2 == 0) ? descending : !descending;
            MergeStep(tileMap, offset, step, tileSize, flag);
        }
    }

    // Assemble result
    Tensor y(x.GetDataType(), {1, length});
    Tensor yIdx(DT_INT32, {1, length});
    for (int i = 0; i < nTile; i++) {
        if (IsMaxTile(tileMap, NUM2 * i)) {
            continue;
        }
        auto [res, resIdx] = tileMap[NUM2 * i];
        Assemble(res, {0, i * tileSize}, y);
        Assemble(resIdx, {0, i * tileSize}, yIdx);
    }
    return std::tie(y, yIdx);
}

std::tuple<Tensor, Tensor> SortWithIndex(const Tensor &x, const Tensor &idx, bool descending) {
    DECLARE_TRACER();
    ASSERT(x->shape.size() == NUM2);
    ASSERT(x->shape[0] == 1);
    auto &vecTile = TileShape::Current().GetVecTile();
    ASSERT(vecTile.size() == NUM2);
    ASSERT(vecTile[0] == 1);
    auto tileSize = vecTile[1];
    ASSERT(IsPowerOfTwo(tileSize));
    int length = x->shape[1];
    int padLength = NextPowerofTwo(length);
    int nTile = padLength / tileSize;
    int halfSize = tileSize / NUM2;
    SortTileMap tileMap;

    if (nTile <= 1) {
        auto res = L1Merge(x, idx, descending, true);   // L1Sort with index
        auto y = std::get<0>(res);
        auto yIdx = std::get<1>(res);
        return std::tie(y, yIdx);
    }

    // Tile Sort
    for (int i = 0; i < nTile; i++) {
        bool flag = (i % NUM2 == (descending ? 0 : 1));
        auto src = View(x, {1, tileSize}, {0, tileSize * i});
        auto srcIdx = View(idx, {1, tileSize}, {0, tileSize * i});
        auto sortResult = L1Merge(src, srcIdx, flag, true);   // L1Sort with index
        auto res = std::get<0>(sortResult);
        auto resIdx = std::get<1>(sortResult);
        tileMap[i * NUM2] = {View(res, {1, halfSize}, {0, 0}), View(resIdx, {1, halfSize}, {0, 0})};
        tileMap[i * NUM2 + 1] = {View(res, {1, halfSize}, {0, halfSize}), View(resIdx, {1, halfSize}, {0, halfSize})};
    }

    // Merge
    for (int step = NUM2; step <= nTile; step *= NUM2) {
        for (int i = 0; i < nTile / step; ++i) {
            int offset = i * step * NUM2;
            bool flag = (i % NUM2 == 0) ? descending : !descending;
            MergeStep(tileMap, offset, step, tileSize, flag);
        }
    }

    // Assemble result
    Tensor y(x.GetDataType(), {1, length});
    Tensor yIdx(idx.GetDataType(), {1, length});
    for (int i = 0; i < nTile; i++) {
        if (IsMaxTile(tileMap, NUM2 * i)) {
            continue;
        }
        auto [res, resIdx] = tileMap[NUM2 * i];
        Assemble(res, {0, i * tileSize}, y);
        Assemble(resIdx, {0, i * tileSize}, yIdx);
    }
    return std::tie(y, yIdx);
}

// view op
Tensor View(const Tensor &operand, const std::vector<int64_t> &shapes, const std::vector<int64_t> &offsets) {
    DECLARE_TRACER();
    Tensor result(operand->Datatype(), shapes, "View_" + operand->GetRawTensor()->GetSymbol(), operand->tensorfmt);
    auto &op = Program::GetInstance().GetCurrentFunction()->AddOperation(
        Opcode::OP_VIEW, {operand.GetStorage()}, {result.GetStorage()});
    auto validShape = GetViewValidShape(operand->GetDynValidShape(), offsets, {}, shapes);
    result->UpdateDynValidShape(validShape);
    auto newOffsets = SymbolicScalar::FromConcrete(offsets);
    op.SetOpAttribute(std::make_shared<ViewOpAttribute>(offsets, newOffsets, validShape));
    return result;
}

Tensor View(const Tensor &operand, const std::vector<int64_t> &shapes, const std::vector<SymbolicScalar> &newOffsets,
    const void *lr) {
    DECLARE_TRACERX(lr);
    Tensor result(operand->Datatype(), shapes, "View_" + operand->GetRawTensor()->GetSymbol(), operand->tensorfmt);
    result->UpdateDynValidShape(SymbolicScalar::FromConcrete(shapes));
    auto function = Program::GetInstance().GetCurrentFunction();
    auto &op = function->AddOperation(Opcode::OP_VIEW, {operand.GetStorage()}, {result.GetStorage()});
    auto validShape = GetViewValidShape(operand->GetDynValidShape(), {}, newOffsets, shapes);
    result->UpdateDynValidShape(validShape);
    std::vector<int64_t> newOffsetsConcrete = SymbolicScalar::Concrete(newOffsets, 0);
    op.SetOpAttribute(std::make_shared<ViewOpAttribute>(newOffsetsConcrete, newOffsets, validShape));
    function->UpdateTensorDataUsage(op);
    return result;
}

Tensor View(const Tensor &operand, const std::vector<int64_t> &shapes, const std::vector<SymbolicScalar> &newOffset) {
    return View(operand, shapes, newOffset, __builtin_return_address(0));
}

//重载View，initializer_list避免歧义
Tensor View(const Tensor &operand, const std::vector<int64_t> &shapes, const std::initializer_list<SymbolicScalar> &newOffsets) {
    return View(operand, shapes, std::vector<SymbolicScalar>(newOffsets), __builtin_return_address(0));
}

Tensor View(const Tensor &operand, const std::vector<int64_t> &shapes,
    const std::vector<SymbolicScalar> &newValidShapes, const std::vector<SymbolicScalar> &newOffsets) {
    DECLARE_TRACER();
    Tensor result(operand->Datatype(), shapes, "View_" + operand->GetRawTensor()->GetSymbol(), operand->tensorfmt);
    auto function = Program::GetInstance().GetCurrentFunction();
    auto &op = function->AddOperation(Opcode::OP_VIEW, {operand.GetStorage()}, {result.GetStorage()});
    std::vector<int64_t> newOffsetsConcrete = SymbolicScalar::Concrete(newOffsets, 0);
    op.SetOpAttribute(std::make_shared<ViewOpAttribute>(newOffsetsConcrete, newOffsets, newValidShapes));
    result->UpdateDynValidShape(newValidShapes);
    function->UpdateTensorDataUsage(op);
    return result;
}

void TensorInnerAssemble(Function &function, const LogicalTensorPtr &operand, const LogicalTensorPtr &result,
    const std::vector<int64_t> &offset) {
    auto &op = function.AddOperation(Opcode::OP_ASSEMBLE, {operand}, {result});
    op.SetOpAttribute(std::make_shared<AssembleOpAttribute>(offset));
}

void InnerAssemble(Function &function, const LogicalTensorPtr &operand, const LogicalTensorPtr &result,
    const std::vector<int64_t> &offset) {
    CALL(InnerAssemble, function, operand, result, offset);
}

Tensor Assemble(const std::vector<std::pair<Tensor, std::vector<int64_t>>> &tensors) {
    DECLARE_TRACER();

    ASSERT(!tensors.empty());
    std::vector<int64_t> shape = tensors.front().first->shape;
    for (const auto &[tensor, offset] : tensors) {
        // 目前只支持2维操作
        if (tensor->shape.size() != 2) {
            ASSERT(false) << "unsupported dimension";
        }
        ASSERT(tensor->shape.size() == tensor->offset.size());
        ASSERT(tensor->shape.size() == offset.size());
    }

    auto shapeSize = tensors[0].first->shape.size(); // 2

    std::vector<int64_t> rawShape(shapeSize, 0);

    std::set<std::vector<int64_t>> position;
    for (const auto &[tensor, offset] : tensors) {
        (void)tensor;
        ASSERT(position.find(offset) == position.end());
        position.emplace(offset);
    }
    ASSERT(position.find(std::vector<int64_t>(shapeSize, 0)) != position.end());

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

    Tensor result(tensors[0].first->Datatype(), rawShape, "Assemble", tensors[0].first->tensorfmt);
    auto &curFunc = *Program::GetInstance().GetCurrentFunction();
    for (const auto &[tensor, offset] : tensors) {
        InnerAssemble(curFunc, tensor.GetStorage(), result.GetStorage(), offset);
    }
    Program::GetInstance().GetTensorSlotManager()->TensorWrite(result, true);
    return result;
}

void TensorDInnerAssemble(Function &function, const LogicalTensorPtr &operand,
    const LogicalTensorPtr &result, const std::vector<SymbolicScalar> &dynOffset) {
    std::vector<int64_t> offset = SymbolicScalar::Concrete(dynOffset, 0);
    auto &op = function.AddOperation(Opcode::OP_ASSEMBLE, {operand}, {result});
    op.SetAssembleOpAttribute(offset, dynOffset);
    op.SetAttribute("dassemble", true);
    function.UpdateTensorDataUsage(op);
}

void DInnerAssemble(Function &function, const LogicalTensorPtr &operand,
    const LogicalTensorPtr &result, const std::vector<SymbolicScalar> &dynOffset) {
    CALL(DInnerAssemble, function, operand, result, dynOffset);
}

void Assemble(const Tensor &tensor, const std::vector<SymbolicScalar> &dynOffset, Tensor &dest) {
    DECLARE_TRACER();

    ASSERT(dest.GetStorage(false)->tensorfmt == tensor.GetStorage(false)->tensorfmt)<<"Assemble: src and dest requires same format";
    ASSERT(dest.GetShape().size() == tensor.GetShape().size())<<"Assemble: src and dest requires same shape";
    ASSERT(dest.GetShape().size() == dynOffset.size())<<"Assemble: dynOffset and dest requires same shape";
    DInnerAssemble(*Program::GetInstance().GetCurrentFunction(), tensor.GetStorage(), dest.GetStorage(), dynOffset);

    Program::GetInstance().GetTensorSlotManager()->TensorWrite(dest, true);
}

static int64_t CalculateCapacity(const std::vector<int64_t> &shape) {
    int64_t capacity = 1;
    for (size_t i = 0; i < shape.size(); i++) {
        capacity = capacity * shape[i];
    }
    return capacity;
}

void TiledInnerReshape(Function &function, const LogicalTensorPtr &operand, const LogicalTensorPtr &result, const bool isInplace = false) {
    auto &op = function.AddOperation("TILE_RESHAPE", {operand}, {result});
    op.SetAttribute(OP_ATTR_PREFIX + "isInplace", isInplace);
    op.SetAttribute(OP_ATTR_PREFIX + "validShape", result->GetDynValidShape());
    op.oOperand.front()->SetIsDummy();
}

void TensorInnerReshape(Function &function, const LogicalTensorPtr &operand, const LogicalTensorPtr &result, const std::vector<SymbolicScalar> &validShape) {
    auto &operation = function.AddOperation(Opcode::OP_RESHAPE, {operand}, {result});
    if(validShape.empty()) {
        result->UpdateDynValidShape(SymbolicScalar::FromConcrete(result->GetShape()));
    } else {
        result->UpdateDynValidShape(validShape);
    }
    operation.SetAttribute("reshape", result->shape);
}

static std::vector<int64_t> CheckAndInferShape(const std::vector<int64_t> &oriShape, const std::vector<int64_t> &dstshape) {
    int negIdx = -1;
    std::vector<int64_t> newShape = dstshape;
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

Tensor Reshape(const Tensor &operand, const std::vector<int64_t> &dstshape, const std::vector<SymbolicScalar> &validShape) {
    if (operand->shape == dstshape) {
        return operand;
    }
    std::vector<SymbolicScalar> validShapeDefault = validShape;
    if (validShape.empty()) {
        validShapeDefault = SymbolicScalar::FromConcrete(dstshape);
    }
    auto newShape = CheckAndInferShape(operand->shape, dstshape);
    if (ReshapeNeedCopy(operand)) {
        Tensor copyOperand(operand->Datatype(), operand->shape, "", operand->tensorfmt);
        CALL(InnerAssign, *Program::GetInstance().GetCurrentFunction(), operand.GetStorage(),
            copyOperand.GetStorage());
        Tensor result(copyOperand->Datatype(), newShape, "", operand->tensorfmt);
        CALL(InnerReshape, *Program::GetInstance().GetCurrentFunction(), copyOperand.GetStorage(),
            result.GetStorage(), validShapeDefault);
        return result;
    } else {
        Tensor result(operand->Datatype(), newShape, "", operand->tensorfmt);
        CALL(InnerReshape, *Program::GetInstance().GetCurrentFunction(), operand.GetStorage(), result.GetStorage(), validShapeDefault);
        return result;
    }
}

void ReshapeInplace(const Tensor &operand, Tensor &dst) {
    auto &operation = Program::GetInstance().GetCurrentFunction()->AddOperation(Opcode::OP_RESHAPE, {operand.GetStorage()}, {dst.GetStorage()});
    operation.SetAttribute(OP_ATTR_PREFIX + "isInplace", true);
    Program::GetInstance().GetTensorSlotManager()->TensorWrite(dst, true);
    Program::GetInstance().GetCurrentFunction()->SetSameMemId(operand, dst);
}

} // namespace npu::tile_fwk

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
        case Opcode::OP_MAXS: {
            TiledBinaryOperationScalar<BinaryOpType::MAX>(function, tileShape, iOperand[0],
            op.GetElementAttribute(OpAttributeKey::scalar), oOperand[0]);
            break;
        }
        case Opcode::OP_MINS: {
            TiledBinaryOperationScalar<BinaryOpType::MIN>(function, tileShape, iOperand[0],
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
        case Opcode::OP_LOGICALNOT: {
            TiledLogicalNotOperation(function, tileShape, iOperand[0], oOperand[0]);
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
        case Opcode::OP_NEG: {
            UnaryOperationOperandCheck(iOperand, oOperand);
            TiledUnaryOperation<UnaryOpType::NEG>(function, tileShape, iOperand[0], oOperand[0]);
            break;
        }
        case Opcode::OP_RSQRT: {
            UnaryOperationOperandCheck(iOperand, oOperand);
            TiledUnaryOperation<UnaryOpType::RSQRT>(function, tileShape, iOperand[0], oOperand[0]);
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
        case Opcode::OP_LN: {
            UnaryOperationOperandCheck(iOperand, oOperand);
            TiledUnaryOperation<UnaryOpType::LN>(function, tileShape, iOperand[0], oOperand[0]);
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
            std::string reduceMode = op.GetStringAttribute(OpAttributeKey::reduceMode);
            TiledScatterElementS(function, tileShape, {oOperand[0], iOperand[0], iOperand[1], scalar, axis, reduceMode});
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
        case Opcode::OP_CMP: {
            BinaryOperationOperandCheck(iOperand, oOperand);
            auto operation = static_cast<CmpOperationType>(op.GetIntAttribute(OP_ATTR_PREFIX + "cmp_operation"));
            auto mode = static_cast<CmpModeType>(op.GetIntAttribute(OP_ATTR_PREFIX + "cmp_mode"));
            TiledCompareOperation(function, tileShape, iOperand[0], iOperand[1], oOperand[0], operation, mode);
            break;
        }
        case Opcode::OP_EXPAND: {
            UnaryOperationOperandCheck(iOperand, oOperand);
            std::vector<SymbolicScalar> validShape;
            op.GetAttr(OP_ATTR_PREFIX + "validShape", validShape);
            TiledExpand(function, tileShape, iOperand[0], oOperand[0], validShape);
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
            auto axis = op.GetIntAttribute(OP_ATTR_PREFIX + "AXIS");
            TiledReduceSingle(function, tileShape, "MAX", iOperand[0], oOperand[0], axis);
            break;
        }
        case Opcode::OP_ROWMIN_SINGLE: {
            UnaryOperationOperandCheck(iOperand, oOperand);
            auto axis = op.GetIntAttribute(OP_ATTR_PREFIX + "AXIS");
            TiledReduceSingle(function, tileShape, "MIN", iOperand[0], oOperand[0], axis);
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
        case Opcode::OP_TRANSPOSE_MOVEOUT: {
            auto shape = op.GetVectorIntAttribute<int>(OP_ATTR_PREFIX + "shape");
            TiledInnerTranspose<TransposeOpType::TRANSPOSE_MOVEOUT>(function, tileShape, iOperand[0], oOperand[0], shape);
            break;
        }
        case Opcode::OP_TRANSPOSE_MOVEIN: {
            auto shape = op.GetVectorIntAttribute<int>(OP_ATTR_PREFIX + "shape");
            TiledInnerTranspose<TransposeOpType::TRANSPOSE_MOVEIN>(function, tileShape, iOperand[0], oOperand[0], shape);
            break;
        }
        case Opcode::OP_TRANSPOSE_VNCHWCONV: {
            auto shape = op.GetVectorIntAttribute<int>(OP_ATTR_PREFIX + "shape");
            TiledInnerTranspose<TransposeOpType::TRANSPOSE_VNCHWCONV>(function, tileShape, iOperand[0], oOperand[0], shape);
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
            Matrix::TiledInnerAMulB<false, true>(
                function, tileShape, iOperand, oOperand[0], {mValue, kValue, nValue});
            break;
        }
        case Opcode::OP_AT_MUL_B: {
            auto mValue = (op.HasAttr(OP_ATTR_PREFIX + "act_m")) ? op.GetIntAttribute(OP_ATTR_PREFIX + "act_m") : 0;
            auto kValue = (op.HasAttr(OP_ATTR_PREFIX + "act_k")) ? op.GetIntAttribute(OP_ATTR_PREFIX + "act_k") : 0;
            auto nValue = (op.HasAttr(OP_ATTR_PREFIX + "act_n")) ? op.GetIntAttribute(OP_ATTR_PREFIX + "act_n") : 0;
            Matrix::TiledInnerAMulB<true, false>(
                function, tileShape, iOperand, oOperand[0], {mValue, kValue, nValue});
            break;
        }
        case Opcode::OP_AT_MUL_BT: {
            auto mValue = (op.HasAttr(OP_ATTR_PREFIX + "act_m")) ? op.GetIntAttribute(OP_ATTR_PREFIX + "act_m") : 0;
            auto kValue = (op.HasAttr(OP_ATTR_PREFIX + "act_k")) ? op.GetIntAttribute(OP_ATTR_PREFIX + "act_k") : 0;
            auto nValue = (op.HasAttr(OP_ATTR_PREFIX + "act_n")) ? op.GetIntAttribute(OP_ATTR_PREFIX + "act_n") : 0;
            Matrix::TiledInnerAMulB<true, true>(
                function, tileShape, iOperand, oOperand[0], {mValue, kValue, nValue});
            break;
        }
        case Opcode::OP_RANGE: {
            Element start = op.GetElementAttribute(OP_ATTR_PREFIX + "START");
            Element step = op.GetElementAttribute(OP_ATTR_PREFIX + "STEP");
            TiledRange(function, tileShape, start, step, oOperand[0]);
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
        case Opcode::OP_TOPK: {
            int axis = op.GetIntAttribute(TOPK_AXIS);
            int kValue = op.GetIntAttribute(TOPK_KVALUE);
            int isLargest = op.GetIntAttribute(TOPK_ORDER);
            TiledTopK(function, tileShape, iOperand[0], oOperand[0], oOperand[1], axis, kValue, isLargest);
            break;
        }
        case Opcode::OP_SORT: {
            int idxStart = op.GetIntAttribute(SORT_START_INDEX);
            int descending = op.GetIntAttribute(SORT_ORDER);
            TiledSort(function, iOperand[0], oOperand[0], oOperand[1], oOperand[2], idxStart, descending);
            break;
        }
        case Opcode::OP_COMPARE_SWAP: {
            int descending = op.GetIntAttribute(SORT_ORDER);
            TiledCompareAndSwap(function, iOperand[0], iOperand[1], iOperand[2], iOperand[3], oOperand[0], oOperand[1], oOperand[2], oOperand[3], descending);
            break;
        }
        case Opcode::OP_MERGE: {
            int descending = op.GetIntAttribute(SORT_ORDER);
            int fullSort = op.GetIntAttribute(SORT_FULL);
            TiledMerge(function, iOperand[0], iOperand[1], oOperand[0], oOperand[1], oOperand[2], fullSort, descending);
            break;
        }
        case Opcode::OP_VEC_DUP: {
            Element scalar = op.GetElementAttribute(OpAttributeKey::scalar);
            SymbolicScalar dynScalar;
            if (op.HasAttr(OpAttributeKey::dynScalar)) {
                dynScalar = op.GetSymbolicScalarAttribute(OpAttributeKey::dynScalar);
            }
            std::vector<int64_t> shape = op.GetVectorIntAttribute(OP_ATTR_PREFIX + "shape");
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
        case Opcode::OP_RESHAPE: {
            bool isInplace = false;
            op.GetAttr(OP_ATTR_PREFIX + "isInplace", isInplace);
            TiledInnerReshape(function, iOperand[0], oOperand[0], isInplace);
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
        case Opcode::OP_SHMEM_PUT: {
            npu::tile_fwk::Distributed::TiledShmemPut(function, tileShape, iOperand, oOperand, op);
            break;
        }
        case Opcode::OP_SHMEM_GET: {
            npu::tile_fwk::Distributed::TiledShmemGet(function, tileShape, iOperand, oOperand, op);
            break;
        }
        case Opcode::OP_SHMEM_SIGNAL: {
            npu::tile_fwk::Distributed::TiledShmemSignal(function, tileShape, iOperand, oOperand, op);
            break;
        }
        case Opcode::OP_SHMEM_WAIT_UNTIL: {
            npu::tile_fwk::Distributed::TiledShmemWaitUntil(function, tileShape, iOperand, oOperand, op);
            break;
        }
        case Opcode::OP_SHMEM_REDUCE: {
            npu::tile_fwk::Distributed::TiledShmemReduce(function, tileShape, iOperand, oOperand, op);
            break;
        }
        case Opcode::OP_BIND_TENSOR: {
            npu::tile_fwk::Distributed::TiledShmemBindTensor(function, tileShape, iOperand, oOperand, op);
            break;
        }
        case Opcode::OP_SHMEM_CLEAR_SIGNAL: {
            npu::tile_fwk::Distributed::TiledShmemClearSignal(function, tileShape, iOperand, oOperand, op);
            break;
        }
        case Opcode::OP_SHMEM_BARRIER_ALL: {
            npu::tile_fwk::Distributed::TiledShmemBarrier(function, tileShape, iOperand, oOperand, op);
            break;
        }
        default: {
            ASLOGE("Unsupported opcode %d, opmagic is %d", static_cast<int>(opCode), op.GetOpMagic());
            ASSERT(false) << "Unsupported opcode " << static_cast<int>(opCode) << ", opmagic is " << op.GetOpMagic();
        }
    }
}