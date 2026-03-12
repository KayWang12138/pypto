/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FIT FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file permute_element.cpp
 * \brief Element-wise permute operation implementation
 */

#include "interface/utils/operator_tracer.h"
#include "interface/operation/operation_common.h"
#include "interface/function/function.h"
#include "interface/program/program.h"
#include "interface/configs/config_manager.h"
#include "permute_element.h"
#include "permute.h"
#include <algorithm>
#include <sstream>

namespace npu::tile_fwk {

namespace {

std::vector<int64_t> PermuteElementTileVector(const std::vector<int64_t> &values, const std::vector<int> &perm) {
    std::vector<int64_t> result;
    result.reserve(perm.size());
    for (int axis : perm) {
        result.push_back(values[axis]);
    }
    return result;
}

}

LogicalTensorPtr TensorElementPermuteOperation(Function &function, LogicalTensorPtr self, const std::vector<int> &perm) {
    std::vector<int64_t> resultShape = PermuteResultShape(self->shape, perm);
    std::vector<SymbolicScalar> resultValidShape;
    if (!self->GetDynValidShape().empty()) {
        for (int p : perm) {
            resultValidShape.push_back(self->GetDynValidShape()[p]);
        }
    } else {
        resultValidShape = SymbolicScalar::FromConcrete(resultShape);
    }

    auto result = std::make_shared<LogicalTensor>(function, self->tensor->datatype, resultShape, resultValidShape);

    auto &op = function.AddOperation(Opcode::OP_PERMUTE_ELEMENT, {self}, {result});

    int axis0 = perm.size() > 0 ? perm[0] : -1;
    int axis1 = perm.size() > 1 ? perm[1] : -1;
    int axis2 = perm.size() > 2 ? perm[2] : -1;
    int axis3 = perm.size() > 3 ? perm[3] : -1;
    int axis4 = perm.size() > 4 ? perm[4] : -1;
    int dimCount = static_cast<int>(perm.size());

    op.SetAttribute(OP_ATTR_PREFIX + "axis0", axis0);
    op.SetAttribute(OP_ATTR_PREFIX + "axis1", axis1);
    op.SetAttribute(OP_ATTR_PREFIX + "axis2", axis2);
    op.SetAttribute(OP_ATTR_PREFIX + "axis3", axis3);
    op.SetAttribute(OP_ATTR_PREFIX + "axis4", axis4);
    op.SetAttribute(OP_ATTR_PREFIX + "dimCount", dimCount);

    return result;
}

void TiledPermuteElementOperation(Function &function, const TileShape &tileShape, size_t cur, Input &input,
    const LogicalTensorPtr &result, const std::vector<int> &perm) {
    int shapeSize = static_cast<int>(input.tensor.GetShape().size());

    if (cur == static_cast<size_t>(shapeSize)) {
        auto srcTile = input.tensor.GetStorage()->View(function, input.tileInfo.shape, input.tileInfo.offset);
        auto resultTileShape = PermuteElementTileVector(input.tileInfo.shape, perm);
        auto resultTileOffset = PermuteElementTileVector(input.tileInfo.offset, perm);
        auto resultTile = result->View(function, resultTileShape, resultTileOffset);

        auto &op = function.AddOperation(Opcode::OP_PERMUTE_ELEMENT, {srcTile}, {resultTile});

        int axis0 = perm.size() > 0 ? perm[0] : -1;
        int axis1 = perm.size() > 1 ? perm[1] : -1;
        int axis2 = perm.size() > 2 ? perm[2] : -1;
        int axis3 = perm.size() > 3 ? perm[3] : -1;
        int axis4 = perm.size() > 4 ? perm[4] : -1;
        int dimCount = static_cast<int>(perm.size());

        op.SetAttribute(OP_ATTR_PREFIX + "axis0", axis0);
        op.SetAttribute(OP_ATTR_PREFIX + "axis1", axis1);
        op.SetAttribute(OP_ATTR_PREFIX + "axis2", axis2);
        op.SetAttribute(OP_ATTR_PREFIX + "axis3", axis3);
        op.SetAttribute(OP_ATTR_PREFIX + "axis4", axis4);
        op.SetAttribute(OP_ATTR_PREFIX + "dimCount", dimCount);
        op.SetAttribute(OP_ATTR_PREFIX + "validShape", resultTile->GetDynValidShape());
        return;
    }

    auto &vecTile = tileShape.GetVecTile();
    for (int i = 0; i < input.tensor.GetShape()[cur]; i += vecTile[cur]) {
        input.tileInfo.shape[cur] = std::min(input.tensor.GetShape()[cur] - i, vecTile[cur]);
        input.tileInfo.offset[cur] = i;
        TiledPermuteElementOperation(function, tileShape, cur + 1, input, result, perm);
    }
}

void PermuteElementOperationTileFunc(Function &function, const TileShape &tileShape,
    const std::vector<LogicalTensorPtr> &iOperand, const std::vector<LogicalTensorPtr> &oOperand, const Operation &op) {
    PermuteOperationOperandCheck(iOperand, oOperand);

    int dimCount = static_cast<int>(op.GetIntAttribute(OP_ATTR_PREFIX + "dimCount"));
    int axis0 = static_cast<int>(op.GetIntAttribute(OP_ATTR_PREFIX + "axis0"));
    int axis1 = static_cast<int>(op.GetIntAttribute(OP_ATTR_PREFIX + "axis1"));
    int axis2 = static_cast<int>(op.GetIntAttribute(OP_ATTR_PREFIX + "axis2"));
    int axis3 = static_cast<int>(op.GetIntAttribute(OP_ATTR_PREFIX + "axis3"));
    int axis4 = static_cast<int>(op.GetIntAttribute(OP_ATTR_PREFIX + "axis4"));

    std::vector<int> perm;
    if (dimCount > 0)
        perm.push_back(axis0);
    if (dimCount > 1)
        perm.push_back(axis1);
    if (dimCount > 2)
        perm.push_back(axis2);
    if (dimCount > 3)
        perm.push_back(axis3);
    if (dimCount > 4)
        perm.push_back(axis4);

    TileInfo tileInfo(iOperand[0]->shape.size(), iOperand[0]->offset.size());
    Input input{iOperand[0], tileInfo};
    TiledPermuteElementOperation(function, tileShape, 0, input, oOperand[0], perm);
}

REGISTER_OPERATION_TILED_FUNC(OP_PERMUTE_ELEMENT, Opcode::OP_PERMUTE_ELEMENT, PermuteElementOperationTileFunc);

} // namespace npu::tile_fwk
