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
 * \file test_scatter_tensor_operation.cpp
 * \brief Scatter Tensor GM inplace operation test
 *
 * Strategy:
 *   - axis dim: LOOP + View on src/index (value-driven, safe to cut)
 *   - non-axis dims: no LOOP, no View (position-driven, must match self)
 *   - self/outputs[0]: never View-cut (same as IndexAdd_)
 *
 * viewShape constraints:
 *   1. axis dim:     viewShape[axis] <= max(src_shape[axis], idx_shape[axis])
 *                    (only axis dim is LOOP-cut, src/idx View sliced together)
 *   2. non-axis dim: viewShape[d] >= max(src_shape[d], idx_shape[d])
 *                    (no LOOP on non-axis dims, validShape = full shape)
 *   3. src_shape[d] <= self_shape[d] for non-axis dim d
 *   4. idx_shape[d] <= self_shape[d] for non-axis dim d
 *   5. self/outputs[0] is never View-cut (GM direct access)
 *
 * Example (2D, axis=1, self=[4,10], src=[3,9], idx=[3,7]):
 *   - non-axis dim0: viewShape[0] >= max(3,3) = 3, and 3 <= 4 (self)
 *   - axis dim1:     viewShape[1] <= 9, e.g. viewShape = [3,8]
 */

#include "test_operation.h"

using namespace tile_fwk::test_operation;
namespace ScatterOperation {
extern const std::map<std::string, ScatterMode>& GetScatterModeMap();
}
namespace ScatterTensorOperation {
struct ScatterTensorOpFuncArgs : public OpFuncArgs {
    ScatterTensorOpFuncArgs(
        const std::vector<int64_t>& viewShape, const std::vector<int64_t> tileShape, int axis, ScatterMode reduce)
        : viewShape_(viewShape), tileShape_(tileShape), axis_(axis), reduce_(reduce)
    {
        this->inplaceInfo[0] = 0;
    }

    std::vector<int64_t> viewShape_;
    std::vector<int64_t> tileShape_;
    int axis_;
    ScatterMode reduce_;
};

struct ScatterTensorOpMetaData {
    explicit ScatterTensorOpMetaData(const OpFunc& opFunc, const nlohmann::json& test_data)
        : opFunc_(opFunc), test_data_(test_data)
    {}

    OpFunc opFunc_;
    nlohmann::json test_data_;
};

static void ScatterTensorOperationExeFunc2Dims(
    const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs, const OpFuncArgs* opArgs)
{
    FUNCTION("main", {inputs[0], inputs[1], inputs[2]}, {outputs[0]})
    {
        SymbolicScalar src_firstDim = inputs[2].GetShape()[0];
        SymbolicScalar src_secondDim = inputs[2].GetShape()[1];
        SymbolicScalar idx_firstDim = inputs[1].GetShape()[0];
        SymbolicScalar idx_secondDim = inputs[1].GetShape()[1];
        auto args = static_cast<const ScatterTensorOpFuncArgs*>(opArgs);
        int axis = args->axis_;
        axis = axis >= 0 ? axis : axis + 2;
        std::vector<int64_t> viewShape = args->viewShape_;
        const int64_t firstViewShape = viewShape[0];
        const int64_t secondViewShape = viewShape[1];

        SymbolicScalar axisSrcDim = (axis == 0) ? src_firstDim : src_secondDim;
        SymbolicScalar axisIdxDim = (axis == 0) ? idx_firstDim : idx_secondDim;
        SymbolicScalar axisMin = std::min(axisSrcDim, axisIdxDim);
        const int64_t axisViewShape = viewShape[axis];
        const int64_t axisLoop = CeilDiv(axisMin, axisViewShape);

        LOOP("LOOP_L0", FunctionType::DYNAMIC_LOOP, aIdx, LoopRange(0, axisLoop, 1))
        {
            SymbolicScalar axisOffset = aIdx * axisViewShape;
            SymbolicScalar srcOff0 = (axis == 0) ? axisOffset : SymbolicScalar(0);
            SymbolicScalar srcOff1 = (axis == 1) ? axisOffset : SymbolicScalar(0);
            SymbolicScalar idxOff0 = (axis == 0) ? axisOffset : SymbolicScalar(0);
            SymbolicScalar idxOff1 = (axis == 1) ? axisOffset : SymbolicScalar(0);

            SymbolicScalar srcValid0 = (axis == 0) ? std::min(src_firstDim - axisOffset, SymbolicScalar(firstViewShape))
                                                    : src_firstDim;
            SymbolicScalar srcValid1 = (axis == 1) ? std::min(src_secondDim - axisOffset, SymbolicScalar(secondViewShape))
                                                    : src_secondDim;
            SymbolicScalar idxValid0 = (axis == 0) ? std::min(idx_firstDim - axisOffset, SymbolicScalar(firstViewShape))
                                                    : idx_firstDim;
            SymbolicScalar idxValid1 = (axis == 1) ? std::min(idx_secondDim - axisOffset, SymbolicScalar(secondViewShape))
                                                    : idx_secondDim;

            auto srcTensor = View(inputs[2], viewShape, {srcValid0, srcValid1}, {srcOff0, srcOff1});
            auto idxTensor = View(inputs[1], viewShape, {idxValid0, idxValid1}, {idxOff0, idxOff1});
            TileShape::Current().SetVecTile(args->tileShape_);
            Scatter(outputs[0], idxTensor, srcTensor, args->axis_, args->reduce_);
        }
    }
}

static void ScatterTensorOperationExeFunc3Dims(
    const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs, const OpFuncArgs* opArgs)
{
    FUNCTION("main", {inputs[0], inputs[1], inputs[2]}, {outputs[0]})
    {
        SymbolicScalar src_firstDim = inputs[2].GetShape()[0];
        SymbolicScalar src_secondDim = inputs[2].GetShape()[1];
        SymbolicScalar src_thirdDim = inputs[2].GetShape()[2];
        SymbolicScalar idx_firstDim = inputs[1].GetShape()[0];
        SymbolicScalar idx_secondDim = inputs[1].GetShape()[1];
        SymbolicScalar idx_thirdDim = inputs[1].GetShape()[2];
        auto args = static_cast<const ScatterTensorOpFuncArgs*>(opArgs);
        int axis = args->axis_;
        axis = axis >= 0 ? axis : axis + 3;
        std::vector<int64_t> viewShape = args->viewShape_;
        const int64_t firstViewShape = viewShape[0];
        const int64_t secondViewShape = viewShape[1];
        const int64_t thirdViewShape = viewShape[2];

        SymbolicScalar srcDims[] = {src_firstDim, src_secondDim, src_thirdDim};
        SymbolicScalar idxDims[] = {idx_firstDim, idx_secondDim, idx_thirdDim};
        SymbolicScalar axisMin = std::min(srcDims[axis], idxDims[axis]);
        const int64_t axisViewShape = viewShape[axis];
        const int64_t axisLoop = CeilDiv(axisMin, axisViewShape);

        LOOP("LOOP_L0", FunctionType::DYNAMIC_LOOP, aIdx, LoopRange(0, axisLoop, 1))
        {
            SymbolicScalar axisOffset = aIdx * axisViewShape;
            SymbolicScalar srcOff0 = (axis == 0) ? axisOffset : SymbolicScalar(0);
            SymbolicScalar srcOff1 = (axis == 1) ? axisOffset : SymbolicScalar(0);
            SymbolicScalar srcOff2 = (axis == 2) ? axisOffset : SymbolicScalar(0);
            SymbolicScalar idxOff0 = (axis == 0) ? axisOffset : SymbolicScalar(0);
            SymbolicScalar idxOff1 = (axis == 1) ? axisOffset : SymbolicScalar(0);
            SymbolicScalar idxOff2 = (axis == 2) ? axisOffset : SymbolicScalar(0);

            SymbolicScalar srcValid0 = (axis == 0) ? std::min(src_firstDim - axisOffset, SymbolicScalar(firstViewShape))
                                                    : src_firstDim;
            SymbolicScalar srcValid1 = (axis == 1) ? std::min(src_secondDim - axisOffset, SymbolicScalar(secondViewShape))
                                                    : src_secondDim;
            SymbolicScalar srcValid2 = (axis == 2) ? std::min(src_thirdDim - axisOffset, SymbolicScalar(thirdViewShape))
                                                    : src_thirdDim;
            SymbolicScalar idxValid0 = (axis == 0) ? std::min(idx_firstDim - axisOffset, SymbolicScalar(firstViewShape))
                                                    : idx_firstDim;
            SymbolicScalar idxValid1 = (axis == 1) ? std::min(idx_secondDim - axisOffset, SymbolicScalar(secondViewShape))
                                                    : idx_secondDim;
            SymbolicScalar idxValid2 = (axis == 2) ? std::min(idx_thirdDim - axisOffset, SymbolicScalar(thirdViewShape))
                                                    : idx_thirdDim;

            auto srcTensor =
                View(inputs[2], viewShape, {srcValid0, srcValid1, srcValid2}, {srcOff0, srcOff1, srcOff2});
            auto idxTensor =
                View(inputs[1], viewShape, {idxValid0, idxValid1, idxValid2}, {idxOff0, idxOff1, idxOff2});
            TileShape::Current().SetVecTile(args->tileShape_);
            Scatter(outputs[0], idxTensor, srcTensor, args->axis_, args->reduce_);
        }
    }
}

static void ScatterTensorOperationExeFunc4Dims(
    const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs, const OpFuncArgs* opArgs)
{
    FUNCTION("main", {inputs[0], inputs[1], inputs[2]}, {outputs[0]})
    {
        SymbolicScalar src_firstDim = inputs[2].GetShape()[0];
        SymbolicScalar src_secondDim = inputs[2].GetShape()[1];
        SymbolicScalar src_thirdDim = inputs[2].GetShape()[2];
        SymbolicScalar src_fourthDim = inputs[2].GetShape()[3];
        SymbolicScalar idx_firstDim = inputs[1].GetShape()[0];
        SymbolicScalar idx_secondDim = inputs[1].GetShape()[1];
        SymbolicScalar idx_thirdDim = inputs[1].GetShape()[2];
        SymbolicScalar idx_fourthDim = inputs[1].GetShape()[3];
        auto args = static_cast<const ScatterTensorOpFuncArgs*>(opArgs);
        int axis = args->axis_;
        axis = axis >= 0 ? axis : axis + 4;
        std::vector<int64_t> viewShape = args->viewShape_;
        const int64_t firstViewShape = viewShape[0];
        const int64_t secondViewShape = viewShape[1];
        const int64_t thirdViewShape = viewShape[2];
        const int64_t fourthViewShape = viewShape[3];

        SymbolicScalar srcDims[] = {src_firstDim, src_secondDim, src_thirdDim, src_fourthDim};
        SymbolicScalar idxDims[] = {idx_firstDim, idx_secondDim, idx_thirdDim, idx_fourthDim};
        SymbolicScalar axisMin = std::min(srcDims[axis], idxDims[axis]);
        const int64_t axisViewShape = viewShape[axis];
        const int64_t axisLoop = CeilDiv(axisMin, axisViewShape);

        LOOP("LOOP_L0", FunctionType::DYNAMIC_LOOP, aIdx, LoopRange(0, axisLoop, 1))
        {
            SymbolicScalar axisOffset = aIdx * axisViewShape;
            SymbolicScalar srcOff0 = (axis == 0) ? axisOffset : SymbolicScalar(0);
            SymbolicScalar srcOff1 = (axis == 1) ? axisOffset : SymbolicScalar(0);
            SymbolicScalar srcOff2 = (axis == 2) ? axisOffset : SymbolicScalar(0);
            SymbolicScalar srcOff3 = (axis == 3) ? axisOffset : SymbolicScalar(0);
            SymbolicScalar idxOff0 = (axis == 0) ? axisOffset : SymbolicScalar(0);
            SymbolicScalar idxOff1 = (axis == 1) ? axisOffset : SymbolicScalar(0);
            SymbolicScalar idxOff2 = (axis == 2) ? axisOffset : SymbolicScalar(0);
            SymbolicScalar idxOff3 = (axis == 3) ? axisOffset : SymbolicScalar(0);

            SymbolicScalar srcValid0 = (axis == 0) ? std::min(src_firstDim - axisOffset, SymbolicScalar(firstViewShape))
                                                    : src_firstDim;
            SymbolicScalar srcValid1 = (axis == 1) ? std::min(src_secondDim - axisOffset, SymbolicScalar(secondViewShape))
                                                    : src_secondDim;
            SymbolicScalar srcValid2 = (axis == 2) ? std::min(src_thirdDim - axisOffset, SymbolicScalar(thirdViewShape))
                                                    : src_thirdDim;
            SymbolicScalar srcValid3 = (axis == 3) ? std::min(src_fourthDim - axisOffset, SymbolicScalar(fourthViewShape))
                                                    : src_fourthDim;
            SymbolicScalar idxValid0 = (axis == 0) ? std::min(idx_firstDim - axisOffset, SymbolicScalar(firstViewShape))
                                                    : idx_firstDim;
            SymbolicScalar idxValid1 = (axis == 1) ? std::min(idx_secondDim - axisOffset, SymbolicScalar(secondViewShape))
                                                    : idx_secondDim;
            SymbolicScalar idxValid2 = (axis == 2) ? std::min(idx_thirdDim - axisOffset, SymbolicScalar(thirdViewShape))
                                                    : idx_thirdDim;
            SymbolicScalar idxValid3 = (axis == 3) ? std::min(idx_fourthDim - axisOffset, SymbolicScalar(fourthViewShape))
                                                    : idx_fourthDim;

            auto srcTensor = View(inputs[2], viewShape,
                {srcValid0, srcValid1, srcValid2, srcValid3}, {srcOff0, srcOff1, srcOff2, srcOff3});
            auto idxTensor = View(inputs[1], viewShape,
                {idxValid0, idxValid1, idxValid2, idxValid3}, {idxOff0, idxOff1, idxOff2, idxOff3});
            TileShape::Current().SetVecTile(args->tileShape_);
            Scatter(outputs[0], idxTensor, srcTensor, args->axis_, args->reduce_);
        }
    }
}

class ScatterTensorOperationTest
    : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac_param<ScatterTensorOpMetaData> {};

INSTANTIATE_TEST_SUITE_P(
    TestScatterTensor, ScatterTensorOperationTest,
    ::testing::ValuesIn(
        GetOpMetaData<ScatterTensorOpMetaData>(
            {ScatterTensorOperationExeFunc2Dims, ScatterTensorOperationExeFunc3Dims,
             ScatterTensorOperationExeFunc4Dims},
            "ScatterTensor")));

TEST_P(ScatterTensorOperationTest, TestScatterTensor)
{
    auto test_data = GetParam().test_data_;
    auto axis = GetValueByName<int>(test_data, "axis");
    auto reduce =
        GetMapValByName(ScatterOperation::GetScatterModeMap(), GetValueByName<std::string>(test_data, "reduce"));
    auto args = ScatterTensorOpFuncArgs(GetViewShape(test_data), GetTileShape(test_data), axis, reduce);
    auto testCase = CreateTestCaseDesc<ScatterTensorOpMetaData>(GetParam(), &args);
    TestExecutor::runTest(testCase);
}
} // namespace ScatterTensorOperation
