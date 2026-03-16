/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quantize.cpp
 * \brief Quantize operation implementation
 */

#include "quantize.h"
#include "interface/inner/tilefwk.h"
#include "interface/utils/operator_tracer.h"
#include "passes/pass_utils/graph_utils.h"

namespace npu::tile_fwk {

void CheckQuantize(const LogicalTensorPtr &input, const LogicalTensorPtr &scale,
                   DataType otype, int axis, const LogicalTensorPtr &zeroPoints) {
    // 1. Input must be FP32
    ASSERT(input->datatype == DataType::DT_FP32)
        << "Quantize input must be DT_FP32, but got " << DataType2String(input->datatype);

    // 2. Dimension check: 2-4 dimensions
    auto shapeSize = input->shape.size();
    ASSERT(shapeSize >= 2 && shapeSize <= 4)
        << "Quantize input must be 2-4 dimensional, but got " << shapeSize << " dimensions";

    // 3. Shape size check
    int64_t totalSize = 1;
    for (auto dim : input->shape) {
        ASSERT(dim > 0) << "Input dimension must be positive";
        totalSize *= dim;
    }
    ASSERT(totalSize <= INT32_MAX)
        << "Input size " << totalSize << " exceeds INT32_MAX " << INT32_MAX;

    // 4. Scale type check
    ASSERT(scale->datatype == DataType::DT_FP32)
        << "Scale must be DT_FP32, but got " << DataType2String(scale->datatype);

    // 5. Axis check: -1, -2, or relative dimensions
    int normalizedAxis = axis;
    if (axis >= 0) {
        normalizedAxis = axis - static_cast<int>(shapeSize);
    }
    ASSERT(normalizedAxis == -1 || normalizedAxis == -2)
        << "Quantize axis must be -1 or -2 (last two dimensions), but got " << axis
        << " (normalized to " << normalizedAxis << ")";

    // 6. Output type check
    ASSERT(otype == DataType::DT_INT8 || otype == DataType::DT_UINT8)
        << "Quantize output type must be DT_INT8 or DT_UINT8, but got " << DataType2String(otype);

    // 7. Asymmetric quantization requires zero_points
    if (otype == DataType::DT_UINT8) {
        ASSERT(zeroPoints != nullptr)
            << "Asymmetric quantization (DT_UINT8) requires zero_points";
        ASSERT(zeroPoints->datatype == DataType::DT_FP32)
            << "zero_points must be DT_FP32, but got " << DataType2String(zeroPoints->datatype);
    }
}

template <QuantizeType quantType>
void QuantizeOperation(Function &function, const TileShape &tileShape, size_t cur,
                       const LogicalTensorPtr &input, const LogicalTensorPtr &scale,
                       const LogicalTensorPtr &result, const LogicalTensorPtr &zeroPoints) {
    if (cur == input->shape.size()) {
        auto inputTile = input->View(function, input->GetViewShape(), input->offset);
        auto scaleTile = scale->View(function, scale->GetViewShape(), scale->offset);
        auto resultTile = result->View(function, result->GetViewShape(), result->offset);

        std::vector<LogicalTensorPtr> inputs = {inputTile, scaleTile};
        if (quantType == QuantizeType::INT8_ASYM && zeroPoints != nullptr) {
            auto zeroTile = zeroPoints->View(function, zeroPoints->GetViewShape(), zeroPoints->offset);
            inputs.push_back(zeroTile);
        }

        function.AddOperation(GetQuantizeOpCode<quantType>(), inputs, {resultTile});
        return;
    }

    auto &vecTile = tileShape.GetVecTile();
    for (int i = 0; i < input->shape[cur]; i += vecTile[cur]) {
        input->tileShape[cur] = std::min(input->shape[cur] - i, static_cast<int64_t>(vecTile[cur]));
        input->offset[cur] = i;
        scale->tileShape[cur] = std::min(scale->shape[cur] - i, static_cast<int64_t>(vecTile[cur]));
        scale->offset[cur] = (scale->shape[cur] == 1) ? 0 : i;
        result->tileShape[cur] = input->tileShape[cur];
        result->offset[cur] = i;

        if (zeroPoints != nullptr) {
            zeroPoints->tileShape[cur] = std::min(zeroPoints->shape[cur] - i, static_cast<int64_t>(vecTile[cur]));
            zeroPoints->offset[cur] = (zeroPoints->shape[cur] == 1) ? 0 : i;
        }

        QuantizeOperation<quantType>(function, tileShape, cur + 1, input, scale, result, zeroPoints);
    }
}

LogicalTensorPtr QuantizeOperation(Function &function, const LogicalTensorPtr &input,
                                   const LogicalTensorPtr &scale, DataType otype,
                                   int axis, const LogicalTensorPtr &zeroPoints) {
    DECLARE_TRACER();

    // Parameter validation
    CheckQuantize(input, scale, otype, axis, zeroPoints);

    // Create result tensor
    auto result = std::make_shared<LogicalTensor>(function, otype, input->shape, input->GetDynValidShape());

    // Determine quantize type
    QuantizeType quantType = (otype == DataType::DT_INT8) ? QuantizeType::INT8_SYM : QuantizeType::INT8_ASYM;

    // Call tiled operation based on graph type
    if (function.GetGraphType() == GraphType::TILE_GRAPH) {
        auto tileShape = function.GetTileShape();
        if (quantType == QuantizeType::INT8_SYM) {
            QuantizeOperation<QuantizeType::INT8_SYM>(function, tileShape, 0, input, scale, result, nullptr);
        } else {
            QuantizeOperation<QuantizeType::INT8_ASYM>(function, tileShape, 0, input, scale, result, zeroPoints);
        }
    } else {
        // Logical graph: add operation directly
        std::vector<LogicalTensorPtr> inputs = {input, scale};
        if (quantType == QuantizeType::INT8_ASYM && zeroPoints != nullptr) {
            inputs.push_back(zeroPoints);
        }
        function.AddOperation(GetQuantizeOpCode<quantType>(), inputs, {result});
    }

    return result;
}

Tensor Quantize(const Tensor &input, const Tensor &scale, DataType otype, int axis, const Tensor &zeroPoints) {
    DECLARE_TRACER();
    ASSERT(input.GetShape().size() == input.GetStorage()->offset.size())
        << "The shape size of input and offset should be equal";

    // Convert to LogicalTensor pointers
    LogicalTensorPtr zeroPointsPtr = nullptr;
    if (zeroPoints.GetStorage() != nullptr) {
        zeroPointsPtr = zeroPoints.GetStorage();
    }

    RETURN_CALL(QuantizeOperation, *Program::GetInstance().GetCurrentFunction(),
        input.GetStorage(), scale.GetStorage(), otype, axis, zeroPointsPtr);
}

} // namespace npu::tile_fwk
