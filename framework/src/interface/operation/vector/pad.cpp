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
 * \file pad.cpp
 * \brief
 */

#include "tensor_transformation.h"
#include "interface/utils/operator_tracer.h"
#include "passes/pass_utils/graph_utils.h"

namespace npu::tile_fwk {


LogicalTensorPtr TensorPadOperation(Function &function, const Tensor &self,
    const std::vector<int64_t> &padding, const std::string &mode, float value)
{
    auto operand = self.GetStorage();
    size_t ndim = operand->shape.size();

    ASSERT(mode == "constant") << "Pad: only 'constant' mode is supported.";

    size_t numPadDims = padding.size() / 2;

    int64_t padRight = 0;
    int64_t padBottom = 0;
    if (numPadDims >= 1) {
        padRight = padding[1];
    }
    if (numPadDims >= 2) {
        padBottom = padding[3];
    }

    // --- 计算输出 shape ---
    std::vector<int64_t> outputShape = operand->shape;
    if (numPadDims >= 1) {
        outputShape[ndim - 1] += padRight;
    }
    if (numPadDims >= 2) {
        outputShape[ndim - 2] += padBottom;
    }


    std::vector<SymbolicScalar> resultValidShape;
    const auto &inputValidShape = operand->GetDynValidShape();

    if (!inputValidShape.empty()) {
        resultValidShape = inputValidShape;

        if (numPadDims >= 1 && padRight > 0) {
            resultValidShape[ndim - 1] = resultValidShape[ndim - 1] + padRight;
        }
        
        if (numPadDims >= 2 && padBottom > 0) {
            resultValidShape[ndim - 2] = resultValidShape[ndim - 2] + padBottom;
        }
    }

    auto result = std::make_shared<LogicalTensor>(function, operand->Datatype(), outputShape, resultValidShape);

    auto &op = function.AddOperation(Opcode::OP_PAD, {operand}, {result});
    op.SetAttribute(OP_ATTR_PREFIX + "pad_right", static_cast<int64_t>(padRight));
    op.SetAttribute(OP_ATTR_PREFIX + "pad_bottom", static_cast<int64_t>(padBottom));
    op.SetAttribute(OpAttributeKey::scalar, Element(self.GetDataType(), value));

    return result;
}

// ============================================================================
// Tiled Implementation
// ============================================================================
//
// 核心思路:
//   在输出空间做 tiling，对每个 output tile 计算其中包含多少 input 有效数据。
//   将 srcValidRow / srcValidCol 作为 attribute 传给后端：
//     - srcValid > 0 : 后端调 TFillPad，拷贝有效数据 + 右/下填充
//     - srcValid = 0 : 后端直接 vector_dup 全部填充 padValue
//
//   因为只有右/下 padding，input 数据从 output 的 (0,0) 开始，所以：
//     input 对应位置: [0, inputDimSize) 在 output 坐标系中
//     对于 output tile [i, i+step):
//       srcValid = clamp(inputDimSize - i, 0, step)
//       srcOffset = i  (当 srcValid > 0 时)

struct PadInput {
    LogicalTensorPtr tensor;
    TileInfo tileInfo;
};

void TiledPadImpl(Function &function, const TileShape &tileShape, size_t cur,
    PadInput &input, const LogicalTensorPtr &result, TileInfo &resultTileInfo,
    int64_t padRight, int64_t padBottom, const Element &padValue)
{
    size_t ndim = result->shape.size();

    if (cur == ndim) {
        // ---- Base case ----
        // 计算最后两维的 srcValid (有效数据量)
        // srcValid=0 表示该 tile 完全在 padding 区域，后端应全部填充 padValue

        int64_t srcValidRow = input.tileInfo.shape[ndim - 2]; // 倒数第二维的有效行数，可为 0
        int64_t srcValidCol = input.tileInfo.shape[ndim - 1]; // 最后一维的有效列数，可为 0

        // 创建 input tile view：view shape 各维至少为 1，避免 0-size view 非法
        std::vector<int64_t> viewShape = input.tileInfo.shape;
        viewShape[ndim - 2] = std::max(viewShape[ndim - 2], static_cast<int64_t>(1));
        viewShape[ndim - 1] = std::max(viewShape[ndim - 1], static_cast<int64_t>(1));

        auto inputTile = input.tensor->View(function, viewShape, input.tileInfo.offset);
        auto resultTile = result->View(function, resultTileInfo.shape, resultTileInfo.offset);

        auto &op = function.AddOperation(Opcode::OP_PAD, {inputTile}, {resultTile});
        op.SetAttribute(OP_ATTR_PREFIX + "src_valid_row", static_cast<int64_t>(srcValidRow));
        op.SetAttribute(OP_ATTR_PREFIX + "src_valid_col", static_cast<int64_t>(srcValidCol));
        op.SetAttribute(OP_ATTR_PREFIX + "pad_right", padRight);
        op.SetAttribute(OP_ATTR_PREFIX + "pad_bottom", padBottom);
        op.SetAttribute(OpAttributeKey::scalar, padValue);
        return;
    }

    auto &vecTile = tileShape.GetVecTile(); // 得到切分大小
    int64_t step = vecTile[cur]; // 当前切分值
    int64_t inputDimSize = input.tensor->shape[cur];
    int64_t outputDimSize = result->shape[cur];

    for (int64_t i = 0; i < outputDimSize; i += step) {
        int64_t outTileSize = std::min(outputDimSize - i, step);

        resultTileInfo.offset[cur] = i;
        resultTileInfo.shape[cur] = outTileSize;

        // 计算 input 在此维度上的有效范围
        // input 数据占 output 坐标 [0, inputDimSize)
        int64_t srcValid = std::max(static_cast<int64_t>(0),
                                    std::min(inputDimSize - i, outTileSize));
        if (srcValid > 0) {
            input.tileInfo.offset[cur] = i;
            input.tileInfo.shape[cur] = srcValid;
        } else {
            // 完全在 padding 区域，input 不被使用
            // shape 置 0，base case 据此将 srcValid 报给后端为 0，触发全填充
            input.tileInfo.offset[cur] = 0;
            input.tileInfo.shape[cur] = 0;
        }

        TiledPadImpl(function, tileShape, cur + 1, input, result,
                     resultTileInfo, padRight, padBottom, padValue);
    }
}

void TiledPadOperation(Function &function, const TileShape &tileShape,
    const LogicalTensorPtr &input, const LogicalTensorPtr &result,
    int64_t padRight, int64_t padBottom, const Element &padValue)
{
    size_t ndim = result->shape.size();
    TileInfo resultTileInfo(ndim, ndim);
    TileInfo inputTileInfo(ndim, ndim);

    PadInput padInput{input, inputTileInfo};

    TiledPadImpl(function, tileShape, 0, padInput, result,
                 resultTileInfo, padRight, padBottom, padValue);
}


Tensor Pad(const Tensor &self, const std::vector<int64_t> &padding, std::string mode, float value)
{
    DECLARE_TRACER();
    RETURN_CALL(PadOperation, *Program::GetInstance().GetCurrentFunction(), self, padding, mode, value);
}


void PadOperationTileFunc(Function &function, const TileShape &tileShape,
    const std::vector<LogicalTensorPtr> &iOperand, const std::vector<LogicalTensorPtr> &oOperand,
    const Operation &op)
{
    int64_t padRight = op.GetIntAttribute(OP_ATTR_PREFIX + "pad_right");
    int64_t padBottom = op.GetIntAttribute(OP_ATTR_PREFIX + "pad_bottom");
    Element padValue = op.GetElementAttribute(OpAttributeKey::scalar);

    TiledPadOperation(function, tileShape, iOperand[0], oOperand[0],
                      padRight, padBottom, padValue);
}

REGISTER_OPERATION_TILED_FUNC(OP_PAD, Opcode::OP_PAD, PadOperationTileFunc);

} // namespace npu::tile_fwk
