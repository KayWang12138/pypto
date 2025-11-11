/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file tensor_transformation.h
 * \brief
 */

#pragma once
#include <string>
#include "interface/utils/common.h"
#include "interface/operation/opcode.h"
#include "interface/operation/operation_common.h"
#include "interface/function/function.h"
#include "interface/program/program.h"
#include "interface/configs/config_manager.h"

namespace npu::tile_fwk {

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
        default: ASSERT(false && "unknown transpose op type");
    }
#undef CASE
}

inline void UnalignPadTmpBufTile(std::vector<int64_t> &shape, int blockElem) {
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
    int shapeSize = input.tensor.GetShape().size();
    if (cur == shapeSize) {
        auto tile = input.tensor.GetStorage()->View(function, input.tileInfo.shape, input.tileInfo.offset);
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
    for (int i = 0; i < input.tensor.GetShape()[cur]; i += vecTile[cur]) {
        input.tileInfo.shape[cur] = std::min(input.tensor.GetShape()[cur] - i, vecTile[cur]);
        input.tileInfo.offset[cur] = i;
        TiledInnerTranspose<T>(function, tileShape, cur + 1, input, result, shape);
    }
}

template <TransposeOpType T>
void TiledInnerTranspose(Function &function, const TileShape &tileShape, const LogicalTensorPtr &operand,
    const LogicalTensorPtr &result, const std::vector<int> &shape) {
    TileInfo tileInfo(result->shape.size(), result->offset.size());
    auto input = Input{operand, tileInfo};
    TiledInnerTranspose<T>(function, tileShape, 0, input, result, shape);
}

struct ExpandInfo {
    const std::shared_ptr<LogicalTensor> &srcTensor;
    const std::shared_ptr<LogicalTensor> &result;
    std::vector<int64_t> &viewShape;
    std::vector<int64_t> &offset;
    const int expandDim;
    ExpandInfo(const std::shared_ptr<LogicalTensor> &srcTensor0, const std::shared_ptr<LogicalTensor> &result0,
        std::vector<int64_t> &viewShape0, std::vector<int64_t> &offset0, const int expandDim0)
        : srcTensor(srcTensor0), result(result0), viewShape(viewShape0), offset(offset0), expandDim(expandDim0) {}
};

void Expand(Function &function, const TileShape &tileShape, const LogicalTensorPtr &operand,
    const std::vector<LogicalTensorPtr> &other, const LogicalTensorPtr &result);
Tensor TensorVectorDuplicateOperation(Function &function, const Element &src, const SymbolicScalar &dynValue,
    DataType dtype, const std::vector<int64_t> &dstShape, const std::vector<SymbolicScalar> &validShape);

enum class CastOpType {
    CAST,
    // FLOOR,
    // ROUND,
};

template <CastOpType T>
Opcode GetCastOpName() {
#define CASE(X) \
    case CastOpType::X: return Opcode::OP_##X
    switch (T) {
        CASE(CAST);
        default: ASSERT(false && "unknown cast op type");
    }
#undef CASE
}

template <CastOpType T>
void TiledCastOperation(Function &function, const TileShape &tileShape, const int cur, Input &input,
    const LogicalTensorPtr &result, const CastMode &mode) {
    if (cur == static_cast<int>(input.tensor.GetShape().size())) {
        auto tile = input.tensor.GetStorage()->View(function, input.tileInfo.shape, input.tileInfo.offset);
        auto resultTile = result->View(function, input.tileInfo.shape, input.tileInfo.offset);
        auto &op = function.AddOperation(GetCastOpName<T>(), {tile}, {resultTile});
        op.SetAttribute(OP_ATTR_PREFIX + "mode", mode);
        return;
    }
    auto &vecTile = tileShape.GetVecTile();
    for (int i = 0; i < input.tensor.GetShape()[cur]; i += vecTile[cur]) {
        input.tileInfo.shape[cur] = std::min(input.tensor.GetShape()[cur] - i, vecTile[cur]);
        input.tileInfo.offset[cur] = i;
        TiledCastOperation<T>(function, tileShape, cur + 1, input, result, mode);
    }
}

template <CastOpType T>
void TiledCastOperation(Function &function, const TileShape &tileShape, const LogicalTensorPtr &operand,
    const LogicalTensorPtr &result, const CastMode &mode) {
    ASSERT(operand->shape.size() == operand->offset.size());

    TileInfo tileInfo(result->shape.size(), result->offset.size());
    auto input = Input{operand, tileInfo};
    TiledCastOperation<T>(function, tileShape, 0, input, result, mode);
}

template <CastOpType T>
LogicalTensorPtr TensorCastOperation(
    Function &function, LogicalTensorPtr self, const DataType &dstDataType, const CastMode &mode = CAST_NONE) {
    auto result = std::make_shared<LogicalTensor>(function, dstDataType, self->shape, self->dynValidShape_);
    auto &op = function.AddOperation(GetCastOpName<T>(), {self}, {result});
    op.SetAttribute(OP_ATTR_PREFIX + "mode", mode);
    return result;
}

} // namespace npu::tile_fwk
