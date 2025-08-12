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
 * \file tilefwk_op.h
 * \brief
 */

#pragma once

#include "tilefwk/tensor.h"
#include "tilefwk/element.h"
#include "common/pre_def.h"

namespace npu::tile_fwk {
constexpr const int TILE_VEC_DIMS = 2;
constexpr const int TILE_CUBE_DIMS = 6;
Tensor View(const Tensor &operand, const std::vector<int> &shapes, const std::vector<int> &offsets);
Tensor DView(const Tensor &operand, const std::vector<int> &shapes, const std::vector<SymbolicScalar> &newOffsets);
Tensor DViewPad(const Tensor &operand, const std::vector<int> &shapes,
    const std::vector<SymbolicScalar> &newValidShapes, const std::vector<SymbolicScalar> &newOffsets);

Tensor VectorDuplicate(const Element &src, DataType dtype, std::vector<int> dstShape,
    std::vector<SymbolicScalar> validShape = {});
Tensor VectorDuplicate(const SymbolicScalar &src, DataType dtype, std::vector<int> dstShape,
    std::vector<SymbolicScalar> validShape = {});
Tensor Transpose(const Tensor &operand, std::vector<int> transposeShape);
Tensor Cast(const Tensor &operand, DataType newDataType, CastMode mode = CAST_NONE);

Tensor Exp(const Tensor &operand);
Tensor Sqrt(const Tensor &operand);
Tensor Reciprocal(const Tensor &operand);
Tensor Abs(const Tensor &operand);

Tensor Reshape(const Tensor &operand, const std::vector<int> &dstshape, const std::vector<SymbolicScalar> &validShape={});
Tensor Duplicate(const Tensor &operand);
Tensor Gather(const Tensor &params, const Tensor &indices, int axis);
Tensor GatherElement(const Tensor &params, const Tensor &indices, int axis);
Tensor ScatterElement(const Tensor &src, const Tensor &idx, const Element &scalar, int axis);
Tensor IndexPut(const Tensor &src, std::vector<Tensor> indices, const Tensor &values);

Tensor RowSumExpand(const Tensor &operand);
Tensor RowMaxExpand(const Tensor &operand);

Tensor RowSumSingle(const Tensor &operand, int axis = -1);
Tensor RowMaxSingle(const Tensor &operand, int axis = -1);
Tensor RowMinSingle(const Tensor &operand, int axis = -1);

Tensor Compact(const Tensor &operand);

Tensor Add(const Tensor &operand1, const Tensor &operand2);
Tensor Sub(const Tensor &operand1, const Tensor &operand2);
Tensor Div(const Tensor &operand1, const Tensor &operand2);
Tensor Mul(const Tensor &operand1, const Tensor &operand2);
Tensor Maximum(const Tensor &operand1, const Tensor &operand2);
Tensor AddS(const Tensor &operand1, const Element &operand2);
Tensor SubS(const Tensor &operand1, const Element &operand2);
Tensor DivS(const Tensor &operand1, const Element &operand2);
Tensor MulS(const Tensor &operand1, const Element &operand2);

Tensor Unsqueeze(const Tensor &old, int unsqueezeDimNum);

Tensor TensorIndex(const Tensor &params, const Tensor &indices);
Tensor ScatterUpdate(const Tensor &dst, const Tensor &index, const Tensor &src, int axis = -2,
    std::string cacheMode = "PA_BNSD", int blockSize = 1);

Tensor Expand(const Tensor &operand, const std::vector<int> &dstShape);
Tensor Expand(const Tensor &operand, DataType dataType, const std::vector<int> &shape);

Tensor Assemble(const std::vector<std::pair<Tensor, std::vector<int>>> &tensors);
void DAssemble(const Tensor &tensor, const std::vector<SymbolicScalar> &dynOffset, Tensor &dest);

Tensor Sin(Tensor operand);
Tensor Cos(Tensor operand);
Tensor Softmax(const Tensor &operand);
Tensor RmsNorm(const Tensor &operand);
Tensor RmsNorm(const Tensor &operand, const Tensor &gamma, float epsilon = 1e-05f);
Tensor Concat(const std::vector<Tensor> &tensorList, int axis);
Tensor NewCompact(const Tensor &operand);
Tensor Pad(const Tensor &old, const std::vector<int> &newShape);
Tensor LogicalNot(const Tensor &operand);

Tensor Assign(const Tensor &operand);

std::tuple<Tensor, Tensor> TopK(const Tensor &operand, const int &k, int axis, bool isLargest = true);
Tensor ArgSort(const Tensor &operand, int axis, bool isLargest = true);

Tensor SoftmaxNew(const Tensor &operand);

Tensor RotateHalf(const Tensor &input);

// moe
Tensor Sigmoid(Tensor &input);

std::tuple<Tensor, Tensor> Quant(
    const Tensor &input, bool isSymmetry = true, bool hasSmoothFactor = false, const Tensor &smoothFactor = Tensor());

Tensor ScalarDivS(const Tensor &operand, const Element &value, bool reverseOperand = false);
Tensor ScalarAddS(const Tensor &operand, const Element &value, bool reverseOperand = false);
Tensor ScalarMaxS(const Tensor &operand, const Element &value, bool reverseOperand = false);
Tensor ScalarSubS(const Tensor &operand, const Element &value, bool reverseOperand = false);
Tensor ScalarMulS(const Tensor &operand, const Element &value, bool reverseOperand = false);

Tensor ScalarSub(const Tensor &operand1, const Tensor &operand2);
Tensor ScalarDiv(const Tensor &operand1, const Tensor &operand2);

struct PaTileShapeConfig {
    int headNumQTile;
    std::array<int, TILE_VEC_DIMS> v0TileShape;
    std::array<int, TILE_CUBE_DIMS> c1TileShape; // (m, M), (k, K), (n, N)
    std::array<int, TILE_VEC_DIMS> v1TileShape;
    std::array<int, TILE_CUBE_DIMS> c2TileShape; // (m, M), (k, K), (n, N)
    std::array<int, TILE_VEC_DIMS> v2TileShape;
};

enum class ReduceMode {
    ATOMIC_ADD,
};
// template <ReduceMode reduceMode>
Tensor Reduce(const std::vector<Tensor> &aggregation, const ReduceMode reduceMode);

Tensor Maxpool(const Tensor &operand, const std::vector<int> &pools, const std::vector<int> &strides,
    const std::vector<int> &paddings);

struct IfaTileShapeConfig {
    int blockSize;
    int headNumQTile;
    std::array<int, TILE_VEC_DIMS> v0TileShape;
    std::array<int, TILE_CUBE_DIMS> c1TileShape; // (m, M), (k, K), (n, N)
    std::array<int, TILE_VEC_DIMS> v1TileShape;
    std::array<int, TILE_CUBE_DIMS> c2TileShape; // (m, M), (k, K), (n, N)
    std::array<int, TILE_VEC_DIMS> v2TileShape;
};

struct RoPETileShapeConfig {
    std::vector<int> twoDimsTileShape;
    std::vector<int> threeDimsTileShape;
    std::vector<int> fourDimsTileShape;
    std::vector<int> fiveDimsTileShape;
};

struct RoPETileShapeConfigNew {
    std::vector<int> threeDimsTileShape;
    std::vector<int> fourDimsTileShapeQ;
    std::vector<int> fourDimsTileShapeK;
    std::vector<int> fiveDimsTileShape;
};

void ApplyRotaryPosEmb(const Tensor &q, const Tensor &k, const Tensor &cos, const Tensor &sin,
    const Tensor &positionIds, Tensor &qEmbed, Tensor &kEmbed, const int unsqueezeDim = 1,
    const RoPETileShapeConfig &ropeTileShapeConfig = {});

void ApplyRotaryPosEmbV2(const Tensor &q, const Tensor &k, const Tensor &cos, const Tensor &sin, Tensor &qEmbed,
    Tensor &kEmbed, const int unsqueezeDim = 2, const RoPETileShapeConfigNew &ropeTileShapeConfig = {});

void IncreFlashAttention(Tensor &qNope, Tensor &kNopeCache, Tensor &vNopeCache, Tensor &qRope, Tensor &kRopeCache,
    std::vector<std::vector<int>> &blockTable, std::vector<int> &actSeqs, float softmaxScale, Tensor &attentionOut,
    IfaTileShapeConfig &tileConfig);

void PageAttentionAddS(Tensor &qNope, Tensor &kNopeCache, Tensor &vNopeCache, Tensor &qRope, Tensor &kRopeCache,
    Tensor &blockTable, Tensor &actSeqs, int blockSize, float softmaxScale, Tensor &attentionOut, Tensor &postOut,
    PaTileShapeConfig &tileConfig, int maxUnrollTimes = 1);

void PageAttentionAddSSingleOutput(Tensor &qNope, Tensor &kNopeCache, Tensor &vNopeCache, Tensor &qRope, Tensor &kRopeCache,
    Tensor &blockTable, Tensor &actSeqs, int blockSize, float softmaxScale, Tensor &attentionOut, Tensor &postOut,
    PaTileShapeConfig &tileConfig, int maxUnrollTimes = 1);

void PrologPost(Tensor &qNope, Tensor &kNopeCache, Tensor &vNopeCache, Tensor &qRope, Tensor &kRopeCache,
    Tensor &blockTable, Tensor &actSeqs, Tensor &weightUV, Tensor &weightO, int blockSize, float softmaxScale,
    Tensor &postOut, PaTileShapeConfig &tileConfig);

namespace Matrix {
// regular intf: c = a * b
template <bool isATrans = false, bool isBTrans = false, bool isCMatrixNZ = false>
Tensor Matmul(DataType outType, const Tensor &aMatrix, const Tensor &bMatrix);

// intf: k spilt
template <bool isATrans = false, bool isBTrans = false, bool isCMatrixNZ = false>
Tensor Matmul(DataType outType, const Tensor &aMatrix, const Tensor &bMatrix, const Tensor &cMatrix);

// matmul extend intf with MatmulParams(Bias/Quant)
template <typename ScaleT>
struct MatmulParams {
    Tensor biasTensor;
    ScaleT quantScale;
};

/*****************batch matmul intf *************/
// intf: c = a * b
template <bool isATrans = false, bool isBTrans = false>
Tensor BatchMatmul(DataType dataType, const Tensor &aMatrix, const Tensor &bMatrix);

// batch mamtul extend intf with MatmulParams(Bias/Quant)
template <typename ScaleT, bool isATrans = false, bool isBTrans = false>
Tensor Matmul(DataType outType, const Tensor &aMatrix, const Tensor &bMatrix, const Tensor &cMatrix,
    const MatmulParams<ScaleT> &params);

Tensor QuantMM(const Tensor &operand1, const Tensor &operand2, const Tensor &dequantScaleW);
} // namespace Matrix

namespace Distributed {
enum class DistReduceType {
    DIST_REDUCE_ADD,
    DIST_REDUCE_MAX,
    DIST_REDUCE_MIN,
};
void AllGather(const Tensor &in, std::vector<Tensor> &out, const char *group);
Tensor AllGather(const Tensor &in, const char *group);
Tensor ReduceScatter(const std::vector<Tensor> &in, const char *group, DistReduceType reduceType);
Tensor ReduceScatter(const Tensor &in, const char *group, DistReduceType reduceType);
Tensor MoeDispatch(const Tensor &tokenTensor, const Tensor &tokenExpertTable, Tensor &validCnt, const char *group);
Tensor MoeCombine(const Tensor &in, const Tensor &scale, const Tensor &combineInfo, const char *group);
} // namespace Distributed
} // namespace npu::tile_fwk
