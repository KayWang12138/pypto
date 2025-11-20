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
 * \file math.cpp
 * \brief
 */

#include "unary.h"
#include "binary.h"
#include "tensor_transformation.h"
#include "interface/utils/operator_tracer.h"
#include "passes/pass_utils/graph_utils.h"

namespace npu::tile_fwk {

struct IndexAddPara {
    const LogicalTensorPtr &selfInput;
    const LogicalTensorPtr &srcInput;
    const LogicalTensorPtr &idxInput;
    const LogicalTensorPtr &dstTensor;
    const int axis;
    const Element &alpha;
};

struct IndexAddTileInfoPara {
    TileInfo selfTileInfo;
    TileInfo srcTileInfo;
    TileInfo idxTileInfo;
    TileInfo dstTileInfo;
};

void InnerTiledIndexAdd(size_t cur, Function &function, const TileShape &tileShape, const IndexAddPara indexaddPara,
    IndexAddTileInfoPara &indexaddTileInfo) {
    const LogicalTensorPtr &selfInput = indexaddPara.selfInput;
    const LogicalTensorPtr &srcInput = indexaddPara.srcInput;
    const LogicalTensorPtr &idxInput = indexaddPara.idxInput;
    const LogicalTensorPtr &dstTensor = indexaddPara.dstTensor;
    const int axis = indexaddPara.axis;
    const Element &alpha = indexaddPara.alpha;

    if (cur == dstTensor->shape.size()) {
        auto dstTile =
            dstTensor->View(function, indexaddTileInfo.dstTileInfo.shape, indexaddTileInfo.dstTileInfo.offset);
        auto selfTile =
            selfInput->View(function, indexaddTileInfo.selfTileInfo.shape, indexaddTileInfo.selfTileInfo.offset);
        auto srcTile =
            srcInput->View(function, indexaddTileInfo.srcTileInfo.shape, indexaddTileInfo.srcTileInfo.offset);
        indexaddTileInfo.idxTileInfo.offset = {
            indexaddTileInfo.srcTileInfo.offset[axis]}; // idxShape只需要按照srcShape所在的axis轴切分
        indexaddTileInfo.idxTileInfo.shape = {indexaddTileInfo.srcTileInfo.shape[axis]};
        auto idxTile =
            idxInput->View(function, indexaddTileInfo.idxTileInfo.shape, indexaddTileInfo.idxTileInfo.offset);
        auto &op = function.AddOperation(Opcode::OP_INDEX_ADD, {selfTile, srcTile, idxTile}, {dstTile});
        op.SetAttribute(OP_ATTR_PREFIX + "axis", axis);
        op.SetAttribute(OpAttributeKey::scalar, alpha);
        return;
    }

    auto &vecTile = tileShape.GetVecTile();
    int64_t tmpTile = vecTile[cur];
    // axis所在轴按照dstShape[axis]进行切分
    if (static_cast<int>(cur) == axis) {
        tmpTile = dstTensor->GetShape()[cur];
    }
    // srcInput在axis的维度=idxInput的维度，且可能比selfInput.shape[axis]大
    for (int i = 0; i < srcInput->GetShape()[cur]; i += tmpTile) {
        if (static_cast<int>(cur) == axis) {
            // self和dst不切
            indexaddTileInfo.dstTileInfo.offset[cur] = 0;
            indexaddTileInfo.dstTileInfo.shape[cur] = dstTensor->shape[cur];
            indexaddTileInfo.selfTileInfo.offset[cur] = 0;
            indexaddTileInfo.selfTileInfo.shape[cur] = selfInput->shape[cur];
        } else {
            indexaddTileInfo.dstTileInfo.offset[cur] = i;
            indexaddTileInfo.dstTileInfo.shape[cur] = std::min(dstTensor->shape[cur] - i, tmpTile);
            indexaddTileInfo.selfTileInfo.offset[cur] = i;
            indexaddTileInfo.selfTileInfo.shape[cur] = std::min(selfInput->shape[cur] - i, tmpTile);
        }
        indexaddTileInfo.srcTileInfo.offset[cur] = i;
        indexaddTileInfo.srcTileInfo.shape[cur] = std::min(srcInput->GetShape()[cur] - i, tmpTile);
        InnerTiledIndexAdd(cur + 1, function, tileShape, indexaddPara, indexaddTileInfo);
    }
}

void TiledIndexAdd(Function &function, const TileShape &tileShape, const IndexAddPara indexaddPara) {
    // Check Operands Valid
    ASSERT(indexaddPara.selfInput->GetShape().size() == indexaddPara.selfInput->GetOffset().size());
    ASSERT(indexaddPara.srcInput->GetShape().size() == indexaddPara.srcInput->GetOffset().size());
    ASSERT(indexaddPara.idxInput->GetShape().size() == indexaddPara.idxInput->GetOffset().size());

    IndexAddTileInfoPara indexaddTileInfo{
        TileInfo(indexaddPara.selfInput->GetShape().size(), indexaddPara.selfInput->GetOffset().size()),
        TileInfo(indexaddPara.srcInput->GetShape().size(), indexaddPara.srcInput->GetOffset().size()),
        TileInfo(indexaddPara.idxInput->GetShape().size(), indexaddPara.idxInput->GetOffset().size()),
        TileInfo(indexaddPara.dstTensor->GetShape().size(), indexaddPara.dstTensor->GetOffset().size())};
    InnerTiledIndexAdd(0, function, tileShape, indexaddPara, indexaddTileInfo);
}

void TensorIndexAdd(Function &function, const IndexAddPara indexaddPara) {
    auto &op = GraphUtils::AddDynOperation(function, Opcode::OP_INDEX_ADD,
        {indexaddPara.selfInput, indexaddPara.srcInput, indexaddPara.idxInput}, {indexaddPara.dstTensor});
    op.SetAttribute(OP_ATTR_PREFIX + "axis", indexaddPara.axis);
    op.SetAttribute(OpAttributeKey::scalar, indexaddPara.alpha);
}

Tensor IndexAdd(const Tensor &self, const Tensor &src, const Tensor &indices, int axis, const Element &alpha) {
    DECLARE_TRACER();
    Tensor result = self;
    return IndexAdd_(result, src, indices, axis, alpha);
}

Tensor IndexAdd_(const Tensor &self, const Tensor &src, const Tensor &indices, int axis, const Element &alpha) {
    DECLARE_TRACER();
    axis = axis < 0 ? self.GetStorage()->GetShape().size() + axis : axis;
    Tensor result(self.GetDataType(), self.GetShape());
    result.GetStorage()->tensor->SetTensorInfo(self.GetStorage()->tensor->GetTensorInfo());
    CALL(IndexAdd, *Program::GetInstance().GetCurrentFunction(),
        {self.GetStorage(), src.GetStorage(), indices.GetStorage(), result.GetStorage(), axis, alpha});
    return result;
}

void TiledLogicalNotOperation(
    Function &function, const TileShape &tileShape, size_t cur, Input &input, const LogicalTensorPtr &result) {
    if (cur == input.tensor.GetShape().size()) {
        auto tile = input.tensor.GetStorage()->View(function, input.tileInfo.shape, input.tileInfo.offset);
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
        function.AddOperation(Opcode::OP_LOGICALNOT, {tile},
            {resultTile, castConditionTensor, compareConditionTensor, vcmpBitResultTensor, startAddrUBTensor,
                oneConditionTensor});
        return;
    }

    auto &vecTile = tileShape.GetVecTile();
    for (int i = 0; i < input.tensor.GetShape()[cur]; i += vecTile[cur]) {
        input.tileInfo.shape[cur] = std::min(input.tensor.GetShape()[cur] - i, vecTile[cur]);
        input.tileInfo.offset[cur] = i;
        TiledLogicalNotOperation(function, tileShape, cur + 1, input, result);
    }
}

void TiledLogicalNotOperation(
    Function &function, const TileShape &tileShape, const LogicalTensorPtr &self, const LogicalTensorPtr &result) {
    ASSERT(self->shape.size() == self->offset.size());

    TileInfo tileInfo(result->shape.size(), result->offset.size());
    auto input = Input{self, tileInfo};
    TiledLogicalNotOperation(function, tileShape, 0, input, result);
}

LogicalTensorPtr TensorLogicalNotOperation(Function &function, LogicalTensorPtr self) {
    auto result = std::make_shared<LogicalTensor>(function, DT_BOOL, self->shape, self->GetDynValidShape());
    function.AddOperation(Opcode::OP_LOGICALNOT, {self}, {result});
    return result;
}

Tensor LogicalNot(const Tensor &self) {
    DECLARE_TRACER();
    bool dtypeIsValid = self.GetDataType() == DT_FP32 || self.GetDataType() == DT_FP16 ||
                        self.GetDataType() == DT_UINT8 || self.GetDataType() == DT_INT8 ||
                        self.GetDataType() == DT_BOOL;
    if (!dtypeIsValid) {
        std::string errorMessage = "Unsurpported Dtype " + DataType2String(self.GetDataType());
        ASSERT(false) << errorMessage;
    }
    RETURN_CALL(LogicalNotOperation, *Program::GetInstance().GetCurrentFunction(), self.GetStorage());
}

Tensor Neg(const Tensor &self) {
    DECLARE_TRACER();

    if (IsFloat(self.GetStorage()->Datatype())) {
        RETURN_CALL(BinaryOperationScalar<BinaryOpType::MUL>, *Program::GetInstance().GetCurrentFunction(),
            self.GetStorage(), Element(self.GetStorage()->Datatype(), -1.0));
    } else {
        RETURN_CALL(BinaryOperationScalar<BinaryOpType::MUL>, *Program::GetInstance().GetCurrentFunction(),
            self.GetStorage(), Element(self.GetStorage()->Datatype(), -1));
    }
}

Tensor Log(const Tensor &self, LogBaseType base) {
    DECLARE_TRACER();
    ASSERT(base == LogBaseType::LOG_E || base == LogBaseType::LOG_2 || base == LogBaseType::LOG_10);
    ASSERT(self.GetStorage()->tensor->datatype == DataType::DT_FP16 ||
           self.GetStorage()->tensor->datatype == DataType::DT_FP32);

    auto operandCast = Tensor(DataType::DT_FP32, self.GetShape());
    if (self.GetStorage()->tensor->datatype == DataType::DT_FP16) {
        operandCast = CALL(CastOperation<CastOpType::CAST>, *Program::GetInstance().GetCurrentFunction(),
            self.GetStorage(), DataType::DT_FP32, CastMode::CAST_NONE);
    } else {
        operandCast = self;
    }

    auto resTensor = Tensor(DataType::DT_FP32, self.GetShape());
    resTensor =
        CALL(UnaryOperation<UnaryOpType::LN>, *Program::GetInstance().GetCurrentFunction(), operandCast.GetStorage());

    auto resTensorBeforeCast = Tensor(DataType::DT_FP32, self.GetShape());
    if (base == LogBaseType::LOG_2) {
        resTensorBeforeCast =
            CALL(BinaryOperationScalar<BinaryOpType::DIV>, *Program::GetInstance().GetCurrentFunction(),
                resTensor.GetStorage(), Element(DataType::DT_FP32, std::log(static_cast<float>(NUM_VALUE_2))));
    } else if (base == LogBaseType::LOG_10) {
        resTensorBeforeCast =
            CALL(BinaryOperationScalar<BinaryOpType::DIV>, *Program::GetInstance().GetCurrentFunction(),
                resTensor.GetStorage(), Element(DataType::DT_FP32, std::log(static_cast<float>(NUM_VALUE_10))));
    } else {
        resTensorBeforeCast = resTensor;
    }

    if (self.GetStorage()->tensor->datatype == DataType::DT_FP16) {
        RETURN_CALL(CastOperation<CastOpType::CAST>, *Program::GetInstance().GetCurrentFunction(),
            resTensorBeforeCast.GetStorage(), DataType::DT_FP16, CastMode::CAST_NONE);
    }
    return resTensorBeforeCast;
}

LogicalTensorPtr GenAllOneTensor(const Shape &shape, std::vector<SymbolicScalar> validShape, const DataType &dataType) {
    auto result = CALL(FullOperation, *Program::GetInstance().GetCurrentFunction(),
        Element(DataType::DT_FP32, 1.0), SymbolicScalar(), DataType::DT_FP32, shape, validShape);
    if (dataType == DataType::DT_FP16) {
        RETURN_CALL(CastOperation<CastOpType::CAST>, *Program::GetInstance().GetCurrentFunction(), result.GetStorage(),
            DataType::DT_FP16, CastMode::CAST_NONE);
    }
    return result.GetStorage();
}

LogicalTensorPtr NormalPow(const Tensor &self, const double &exponent) {
    // a^b = e^(b * lna)
    // lna
    auto lnSelf =
        CALL(UnaryOperation<UnaryOpType::LN>, *Program::GetInstance().GetCurrentFunction(), self.GetStorage());
    // b * lna
    auto expMulLnSelf = CALL(BinaryOperationScalar<BinaryOpType::MUL>, *Program::GetInstance().GetCurrentFunction(),
        lnSelf, Element(DataType::DT_FP32, exponent));
    // e ^ (b * lna)
    RETURN_CALL(UnaryOperation<UnaryOpType::EXP>, *Program::GetInstance().GetCurrentFunction(), expMulLnSelf);
}

LogicalTensorPtr IntegerPow(const Tensor &self, int32_t intExponent) {
    // 快速幂
    auto result = GenAllOneTensor(self.GetShape(), self.GetStorage()->GetDynValidShape(), self.GetDataType());
    auto current = CALL(BinaryOperation<BinaryOpType::MUL>, *Program::GetInstance().GetCurrentFunction(), self, result);

    while (intExponent != NUM_VALUE_0) {
        if (intExponent % NUM_VALUE_2 != NUM_VALUE_0) {
            result =
                CALL(BinaryOperation<BinaryOpType::MUL>, *Program::GetInstance().GetCurrentFunction(), result, current);
        }
        current =
            CALL(BinaryOperation<BinaryOpType::MUL>, *Program::GetInstance().GetCurrentFunction(), current, current);
        intExponent /= NUM_VALUE_2;
    }
    return result;
}

LogicalTensorPtr GeneralPow(const Tensor &self, double exponent) {
    // 如果指数小于0，先计算a^(-b)，最后再取倒数
    bool expLessThanZero = exponent < NUM_VALUE_0;
    exponent = std::abs(exponent);

    LogicalTensorPtr result;
    // 将指数分为整数部分和小数部分
    int32_t intExponent = static_cast<int32_t>(std::floor(exponent));
    if (intExponent != NUM_VALUE_0) {
        result = IntegerPow(self, intExponent);
        if (exponent - intExponent > NUM_VALUE_EPS) {
            result = CALL(BinaryOperation<BinaryOpType::MUL>, *Program::GetInstance().GetCurrentFunction(), result,
                NormalPow(self, exponent - intExponent));
        }
    } else {
        result = NormalPow(self, exponent);
    }

    // 指数小于零，结果取倒数
    if (expLessThanZero) {
        auto oneTensor = GenAllOneTensor(self.GetShape(), self.GetStorage()->GetDynValidShape(), self.GetDataType());
        // 求倒数
        RETURN_CALL(
            BinaryOperation<BinaryOpType::DIV>, *Program::GetInstance().GetCurrentFunction(), oneTensor, result);
    }
    return result;
}

Tensor Pow(const Tensor &self, const Element &other) {
    DECLARE_TRACER();

    double exponent = other.Cast<double>();
    // 指数为0，输出全1
    if (std::abs(exponent) < NUM_VALUE_EPS) {
        return GenAllOneTensor(self.GetShape(), self.GetStorage()->GetDynValidShape(), self.GetDataType());
    }
    // 特殊指数
    if (std::abs(exponent - -NUM_VALUE_0_5) < NUM_VALUE_EPS) {
        RETURN_CALL(
            UnaryOperation<UnaryOpType::RSQRT>, *Program::GetInstance().GetCurrentFunction(), self.GetStorage());
    }
    if (std::abs(exponent - NUM_VALUE_0_5) < NUM_VALUE_EPS) {
        RETURN_CALL(UnaryOperation<UnaryOpType::SQRT>, *Program::GetInstance().GetCurrentFunction(), self.GetStorage());
    }
    if (std::abs(exponent - NUM_VALUE_2) < NUM_VALUE_EPS) {
        RETURN_CALL(BinaryOperation<BinaryOpType::MUL>, *Program::GetInstance().GetCurrentFunction(), self, self);
    }
    if (std::abs(exponent - NUM_VALUE_3) < NUM_VALUE_EPS) {
        auto doubleSelf =
            CALL(BinaryOperation<BinaryOpType::MUL>, *Program::GetInstance().GetCurrentFunction(), self, self);
        RETURN_CALL(BinaryOperation<BinaryOpType::MUL>, *Program::GetInstance().GetCurrentFunction(), doubleSelf, self);
    }
    // 其余情况处理
    return GeneralPow(self, exponent);
}

void TiledOneHot(
    Function &function, const TileShape &tileShape, size_t cur, Input &input, Input &output, int numClasses) {
    if (cur == output.tensor.GetShape().size()) {
        auto inputTile = input.tensor.GetStorage()->View(function, input.tileInfo.shape, input.tileInfo.offset);
        auto outputTile = output.tensor.GetStorage()->View(function, output.tileInfo.shape, output.tileInfo.offset);
        auto &newOp = function.AddOperation(Opcode::OP_ONEHOT, {inputTile}, {outputTile});
        newOp.SetAttribute(OP_ATTR_PREFIX + "numClasses", numClasses);
        return;
    }
    auto &vecTile = tileShape.GetVecTile();
    for (int i = 0; i < output.tensor.GetShape()[cur]; i += vecTile[cur]) {
        if (cur < input.tensor.GetShape().size()) {
            input.tileInfo.shape[cur] = std::min(input.tensor.GetShape()[cur] - i, vecTile[cur]);
            input.tileInfo.offset[cur] = i;
        }
        output.tileInfo.shape[cur] = std::min(output.tensor.GetShape()[cur] - i, vecTile[cur]);
        output.tileInfo.offset[cur] = i;
        TiledOneHot(function, tileShape, cur + 1, input, output, numClasses);
    }
}

void TiledOneHot(Function &function, const TileShape &tileShape, const LogicalTensorPtr &self,
    const LogicalTensorPtr &result, int numClasses) {
    ASSERT(self->shape.size() == self->offset.size());
    ASSERT(numClasses == tileShape.GetVecTile()[result->shape.size() - 1]);

    TileInfo inputTileInfo(self->shape.size(), self->offset.size());
    TileInfo outputTileInfo(result->shape.size(), result->offset.size());
    auto input = Input{self, inputTileInfo};
    auto output = Input{result, outputTileInfo};
    TiledOneHot(function, tileShape, 0, input, output, numClasses);
}

Tensor TensorOneHot(Function &function, const LogicalTensorPtr &self, int numClasses) {
    Shape shape(self->shape);
    std::vector<SymbolicScalar> validShape(self->dynValidShape_);
    shape.push_back(static_cast<int64_t>(numClasses));
    validShape.push_back(SymbolicScalar(numClasses));
    auto result = std::make_shared<LogicalTensor>(function, DataType::DT_INT64, shape, validShape);
    auto &op = function.AddOperation(Opcode::OP_ONEHOT, {self}, {result});
    op.SetAttribute(OP_ATTR_PREFIX + "numClasses", numClasses);
    function.UpdateTensorDataUsage(op);
    return result;
}

Tensor OneHot(const Tensor &self, int numClasses) {
    DECLARE_TRACER();

    RETURN_CALL(OneHot, *Program::GetInstance().GetCurrentFunction(), self.GetStorage(), numClasses);
}

void TiledLogicalAndOperation(Function& function, const TileShape& tileShape, size_t cur,
        Input& input0, Input& input1, const LogicalTensorPtr& result, TileInfo &resultTileInfo) {
    if (cur == input0.tensor.GetShape().size()) {
        auto tile0 = input0.tensor.GetStorage()->View(function, input0.tileInfo.shape, input0.tileInfo.offset);
        auto tile1 = input1.tensor.GetStorage()->View(function, input1.tileInfo.shape, input1.tileInfo.offset);
        auto resultTile = result->View(function, resultTileInfo.shape, resultTileInfo.offset);

        std::vector<int64_t> castConditionShape({64});
        auto castConditionTensor0 = std::make_shared<LogicalTensor>(function, DT_FP32, castConditionShape);
        auto castConditionTensor1 = std::make_shared<LogicalTensor>(function, DT_FP32, castConditionShape);
        auto tempConditionTensor = std::make_shared<LogicalTensor>(function, DT_FP16, castConditionShape);
        auto oneConditionTensor = std::make_shared<LogicalTensor>(function, DT_FP32, castConditionShape);
        auto zeroConditionTensor = std::make_shared<LogicalTensor>(function, DT_FP32, castConditionShape);

        std::vector<int64_t> vcmpBitResultShape({64 / 8});
        auto vcmpBitResultTensor = std::make_shared<LogicalTensor>(function, DT_UINT8, vcmpBitResultShape);
        std::vector<int64_t> startAddrUBShape({1});
        auto startAddrUBTensor = std::make_shared<LogicalTensor>(function, DT_UINT64, startAddrUBShape);

        function.AddOperation(Opcode::OP_LOGICALAND, {tile0, tile1}, 
                            {resultTile, castConditionTensor0, castConditionTensor1, tempConditionTensor, 
                            oneConditionTensor, zeroConditionTensor, vcmpBitResultTensor, startAddrUBTensor});    
        return;
    }

    auto& vecTile = tileShape.GetVecTile();
    for (int i = 0; i < result->shape[cur]; i += vecTile[cur]) {
        resultTileInfo.offset[cur] = i;
        input0.tileInfo.offset[cur] = i % input0.tensor.GetShape()[cur];
        input1.tileInfo.offset[cur] = i % input1.tensor.GetShape()[cur];
        resultTileInfo.shape[cur] = std::min(result->shape[cur] - resultTileInfo.offset[cur], vecTile[cur]);
        input0.tileInfo.shape[cur] = std::min(input0.tensor.GetShape()[cur] - input0.tileInfo.offset[cur], vecTile[cur]);
        input1.tileInfo.shape[cur] = std::min(input1.tensor.GetShape()[cur] - input1.tileInfo.offset[cur], vecTile[cur]);
        TiledLogicalAndOperation(function, tileShape, cur + 1, input0, input1, result, resultTileInfo);
    }
}

void BroadcastOperand(LogicalTensorPtr &operand, LogicalTensorPtr &other, LogicalTensorPtr result,
                                      Function& function, const TileShape& tileShape) {
    auto dstShape = result->shape;
    if (operand->shape == dstShape) {
        return;
    }
    auto expanded = std::make_shared<LogicalTensor>(function, operand->Datatype(), dstShape);
    Expand(function, tileShape, operand, {other}, expanded);
    operand = expanded;
}

void TiledLogicalAndOperation(Function& function, const TileShape& tileShape, LogicalTensorPtr operand0, LogicalTensorPtr operand1, const LogicalTensorPtr& result) {
    BroadcastOperand(operand0, operand1, result, function, tileShape);
    BroadcastOperand(operand1, operand0, result, function, tileShape);

    TileInfo tileInfo0(result->shape.size(), result->offset.size());
    TileInfo tileInfo1(result->shape.size(), result->offset.size());
    TileInfo resultTileInfo(result->shape.size(), result->offset.size());
    auto input0 = Input{operand0, tileInfo0};
    auto input1 = Input{operand1, tileInfo1};
    TiledLogicalAndOperation(function, tileShape, 0, input0, input1, result, resultTileInfo);
}

LogicalTensorPtr TensorLogicalAndOperation(Function& function, const Tensor& self, const Tensor& other) {
    auto operandT0 = self.GetStorage();
    auto operandT1 = other.GetStorage();
    if (operandT0->shape.size() != operandT1->shape.size()) {
        std::vector<int> broadCastShape = GetBroadCastShape(operandT0, operandT1);
        operandT0 = BinaryOperationBroadCast(operandT0, broadCastShape);
        operandT1 = BinaryOperationBroadCast(operandT1, broadCastShape);
    }

    std::vector<SymbolicScalar> resultValidShape;
    std::vector<int64_t> resultShape = BinaryOperationResultShape(operandT0, operandT1);
    if ((!operandT0->GetDynValidShape().empty()) && (!operandT1->GetDynValidShape().empty())) {
        for (size_t i = 0; i < resultShape.size(); ++i) {
            if (resultShape[i] == operandT0->shape[i]) {
                resultValidShape.push_back(operandT0->GetDynValidShape()[i]);
            } else {
                resultValidShape.push_back(operandT1->GetDynValidShape()[i]);
            }
        }
    }

    auto result = std::make_shared<LogicalTensor>(function, DT_BOOL, resultShape, resultValidShape);
    function.AddOperation(Opcode::OP_LOGICALAND, {operandT0, operandT1}, {result});
    return result;
}

Tensor LogicalAnd(const Tensor &self, const Tensor &other) {
    DECLARE_TRACER();
    RETURN_CALL(LogicalAndOperation, *Program::GetInstance().GetCurrentFunction(), self.GetStorage(), other.GetStorage());
}

void IndexAddOperationTileFunc(Function &function, const TileShape &tileShape,
    const std::vector<LogicalTensorPtr> &iOperand, const std::vector<LogicalTensorPtr> &oOperand, const Operation &op) {
    int axis = op.GetIntAttribute(OP_ATTR_PREFIX + "axis");
    Element alpha = op.GetElementAttribute(OpAttributeKey::scalar);
    TiledIndexAdd(function, tileShape, {iOperand[0], iOperand[1], iOperand[2], oOperand[0], axis, alpha});
}

void LogicNotOperationTileFunc(Function &function, const TileShape &tileShape,
    const std::vector<LogicalTensorPtr> &iOperand, const std::vector<LogicalTensorPtr> &oOperand,
    [[maybe_unused]] const Operation &op) {
    TiledLogicalNotOperation(function, tileShape, iOperand[0], oOperand[0]);
}

void OneHotOperationTileFunc(Function &function, const TileShape &tileShape,
    const std::vector<LogicalTensorPtr> &iOperand, const std::vector<LogicalTensorPtr> &oOperand,
    [[maybe_unused]] const Operation &op) {
    UnaryOperationOperandCheck(iOperand, oOperand);
    int numClasses = op.GetIntAttribute(OP_ATTR_PREFIX + "numClasses");
    TiledOneHot(function, tileShape, iOperand[0], oOperand[0], numClasses);
}

// beginregin: Clip

Tensor Clip(const Tensor &self, const Element &min, const Element &max) {
    ASSERT(self.GetShape().size() >= SHAPE_DIM2 && self.GetShape().size() <= SHAPE_DIM4);
    std::vector<DataType> CLIP_SUPPORT_DATATYPES = {
        DataType::DT_FP32, DataType::DT_FP16, DataType::DT_INT32, DataType::DT_INT16};
    ASSERT(std::find(CLIP_SUPPORT_DATATYPES.begin(), CLIP_SUPPORT_DATATYPES.end(), self.GetDataType()) != 
        CLIP_SUPPORT_DATATYPES.end());

    Element min_ = min, max_ = max;

    Tensor result = self;
    ASSERT(min_.GetDataType() == self.GetDataType());
    result = Maximum(result, min_);
    ASSERT(max_.GetDataType() == self.GetDataType());
    result = Minimum(result, max_);
    result.GetStorage()->UpdateDynValidShape(self.GetStorage()->GetDynValidShape());
    return result;
}

Tensor Clip(const Tensor &self, const Tensor &min, const Tensor &max) {
    ASSERT(self.GetShape().size() >= SHAPE_DIM2 && self.GetShape().size() <= SHAPE_DIM4);
    std::vector<DataType> CLIP_SUPPORT_DATATYPES = {
        DataType::DT_FP32, DataType::DT_FP16, DataType::DT_INT32, DataType::DT_INT16};
    ASSERT(std::find(CLIP_SUPPORT_DATATYPES.begin(), CLIP_SUPPORT_DATATYPES.end(), self.GetDataType()) != 
        CLIP_SUPPORT_DATATYPES.end());
    
    Tensor result = self;
    if (min.GetStorage() != nullptr) {
        ASSERT(min.GetDataType() == self.GetDataType());
        std::vector minBroadcastAxes = GetBroadcastAxes(min.GetShape(), self.GetShape());
        ASSERT(minBroadcastAxes.size() <= 1);
        result = Maximum(result, min);
    }
    if (max.GetStorage() != nullptr) {
        std::vector maxBroadcastAxes = GetBroadcastAxes(max.GetShape(), self.GetShape());
        ASSERT(maxBroadcastAxes.size() <= 1);
        result = Minimum(result, max);
    }
    result.GetStorage()->UpdateDynValidShape(self.GetStorage()->GetDynValidShape());
    return result;
}
// endregion: Clip

void LogicAndOperationTileFunc(Function &function, const TileShape &tileShape,
    const std::vector<LogicalTensorPtr> &iOperand, const std::vector<LogicalTensorPtr> &oOperand,
    [[maybe_unused]] const Operation &op) {
    TiledLogicalAndOperation(function, tileShape, iOperand[0], iOperand[1], oOperand[0]);
}

REGISTER_OPERATION_TILED_FUNC(OP_INDEX_ADD, Opcode::OP_INDEX_ADD, IndexAddOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_LOGICALNOT, Opcode::OP_LOGICALNOT, LogicNotOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_ONEHOT, Opcode::OP_ONEHOT, OneHotOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_LOGICALAND, Opcode::OP_LOGICALAND, LogicAndOperationTileFunc);
} // namespace npu::tile_fwk