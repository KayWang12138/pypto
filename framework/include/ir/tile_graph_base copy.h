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
 * \file tile_graph_base.h
 * \brief
 */

#pragma once

#include <vector>
#include <string>
#include <sstream>
#include <map>
#include <unordered_set>
#include "type.h"
#include "value.h"
#include "operation.h"

namespace pto {

class TileBaseOp : public Operation {
public:
    TileBaseOp(Opcode opcode, std::vector<ValuePtr> iops, std::vector<ValuePtr> oops)
     : Operation(opcode, iops, oops) {}

    std::shared_ptr<TileValue> GetInOperand(size_t index) const;
    std::shared_ptr<TileValue> GetOutOperand(size_t index) const;
private:
};

class ElementWiseTileBaseOp : public TileBaseOp {
public:
    ElementWiseTileBaseOp(Opcode opcode, std::vector<ValuePtr> iops, std::vector<ValuePtr> oops)
     : TileBaseOp(opcode, iops, oops) {}
};

class ElementWiseUnaryTileBaseOp : public ElementWiseTileBaseOp {
public:
    ElementWiseUnaryTileBaseOp(Opcode opcode, TileValuePtr input, TileValuePtr output)
        : ElementWiseTileBaseOp(opcode, {ValueCast<Value>(input)}, {ValueCast<Value>(output)}) {}
};

class ElementWiseBinaryTileBaseOp : public ElementWiseTileBaseOp {
public:
    ElementWiseBinaryTileBaseOp(Opcode opcode,
                                ValuePtr lhs,
                                ValuePtr rhs,
                                ValuePtr out)
        : ElementWiseTileBaseOp(opcode, {lhs, rhs}, {out}) {
    }
};

class ElementWiseScalarMixBinaryTileBaseOp : public ElementWiseTileBaseOp {
public:
    ElementWiseScalarMixBinaryTileBaseOp(Opcode opcode, TileValuePtr lhs, ScalarValuePtr rhs, TileValuePtr output)
        : ElementWiseTileBaseOp(opcode, {ValueCast<Value>(lhs), ValueCast<Value>(rhs)}, {ValueCast<Value>(output)}) {}
};

class ElementWiseTernaryTileBaseOp : public ElementWiseTileBaseOp {
public:
    // 对应 OP_WHERE_TT
    ElementWiseTernaryTileBaseOp(Opcode opcode, TileValuePtr cond, TileValuePtr input, TileValuePtr other, TileValuePtr output, TileValuePtr temp)
        : ElementWiseTileBaseOp(opcode, 
                                {ValueCast<Value>(cond), ValueCast<Value>(input), ValueCast<Value>(other)}, 
                                {ValueCast<Value>(output), ValueCast<Value>(temp)}) {}
};

class ElementWiseScalarMixTernaryTileBaseOp : public ElementWiseTileBaseOp {
public:
    // 对应 OP_WHERE_ST: Condition(Tile), Input(Scalar), Other(Tile)
    ElementWiseScalarMixTernaryTileBaseOp(Opcode opcode, TileValuePtr cond, ScalarValuePtr input, TileValuePtr other, TileValuePtr output, TileValuePtr temp)
        : ElementWiseTileBaseOp(opcode, 
                                {ValueCast<Value>(cond), ValueCast<Value>(input), ValueCast<Value>(other)}, 
                                {ValueCast<Value>(output), ValueCast<Value>(temp)}) {}

    // 对应 OP_WHERE_TS: Condition(Tile), Input(Tile), Other(Scalar)
    ElementWiseScalarMixTernaryTileBaseOp(Opcode opcode, TileValuePtr cond, TileValuePtr input, ScalarValuePtr other, TileValuePtr output, TileValuePtr temp)
        : ElementWiseTileBaseOp(opcode, 
                                {ValueCast<Value>(cond), ValueCast<Value>(input), ValueCast<Value>(other)}, 
                                {ValueCast<Value>(output), ValueCast<Value>(temp)}) {}

    // 对应 OP_WHERE_SS: Condition(Tile), Input(Scalar), Other(Scalar)
    ElementWiseScalarMixTernaryTileBaseOp(Opcode opcode, TileValuePtr cond, ScalarValuePtr input, ScalarValuePtr other, TileValuePtr output, TileValuePtr temp)
        : ElementWiseTileBaseOp(opcode, 
                                {ValueCast<Value>(cond), ValueCast<Value>(input), ValueCast<Value>(other)}, 
                                {ValueCast<Value>(output), ValueCast<Value>(temp)}) {}
    
    // 对应 OP_INDEX_ADD: Input1(Tile), Input2(Tile), Input3(Tile), Axis(Scalar), Alpha(Scalar)
    ElementWiseScalarMixTernaryTileBaseOp(Opcode opcode, TileValuePtr in1, TileValuePtr in2, TileValuePtr in3, ScalarValuePtr axis, ScalarValuePtr alpha, TileValuePtr output)
        : ElementWiseTileBaseOp(opcode,
                                {ValueCast<Value>(in1), ValueCast<Value>(in2), ValueCast<Value>(in3), ValueCast<Value>(axis), ValueCast<Value>(alpha)},
                                {ValueCast<Value>(output)}) {}
};

// 注意：DEF_OP中未直接使用SortTileOp，通常使用SortTileBaseOp，这里作为通用预留
class SortTileOp : public TileBaseOp { 
public:
    SortTileOp(Opcode opcode, std::vector<ValuePtr> iops, std::vector<ValuePtr> oops)
     : TileBaseOp(opcode, iops, oops) {}
};

// 注意：DEF_OP中未直接使用DataCopyTileOp，通常使用DataCopyTileBaseOp，这里作为通用预留
class DataCopyTileOp : public TileBaseOp {
public:
    DataCopyTileOp(Opcode opcode, std::vector<ValuePtr> iops, std::vector<ValuePtr> oops)
     : TileBaseOp(opcode, iops, oops) {}
};

class ReduceTileBaseOp : public TileBaseOp {
public:
    // 对应 ReduceOp
    ReduceTileBaseOp(Opcode opcode, TileValuePtr input, TileValuePtr output)
        : TileBaseOp(opcode, {ValueCast<Value>(input)}, {ValueCast<Value>(output)}) {}
};

class ReduceWithTempTileBaseOp : public TileBaseOp {
public:
    // 对应 ReduceTempOp
    ReduceWithTempTileBaseOp(Opcode opcode, TileValuePtr input, TileValuePtr output, TileValuePtr TempTensor)
        : TileBaseOp(opcode, {ValueCast<Value>(input)}, 
                             {ValueCast<Value>(output), ValueCast<Value>(TempTensor)}) {}
    // 构造函数重载，兼容你给出的原始代码段（双Src输入）
    ReduceWithTempTileBaseOp(Opcode opcode, TileValuePtr Src0, TileValuePtr Src1, TileValuePtr output, TileValuePtr TempTensor)
        : TileBaseOp(opcode, {ValueCast<Value>(Src0), ValueCast<Value>(Src1)},
                             {ValueCast<Value>(output), ValueCast<Value>(TempTensor)}) {}
};

class LogicalAndTileBaseOp : public TileBaseOp {
public:
    // 对应 BinaryOp (OP_LOGICALAND)
    LogicalAndTileBaseOp(Opcode opcode, TileValuePtr lhs, TileValuePtr rhs, TileValuePtr output, TileValuePtr TempTensor)
        : TileBaseOp(opcode, {ValueCast<Value>(lhs), ValueCast<Value>(rhs)}, 
                             {ValueCast<Value>(output), ValueCast<Value>(TempTensor)}) {}
};

class CastTileBaseOp : public TileBaseOp {
public:
    // 对应 CastOp
    CastTileBaseOp(Opcode opcode, TileValuePtr input, TileValuePtr output)
        : TileBaseOp(opcode, {ValueCast<Value>(input)}, {ValueCast<Value>(output)}) {}
};

class VecDupTileBaseOp : public TileBaseOp {
public:
    // 对应 VecDupOp
    VecDupTileBaseOp(Opcode opcode, ScalarValuePtr Scalar, TileValuePtr output)
        : TileBaseOp(opcode, {ValueCast<Value>(Scalar)}, {ValueCast<Value>(output)}) {}
};

class RangeTileBaseOp : public TileBaseOp {
public:
    // 对应 RangeOp
    RangeTileBaseOp(Opcode opcode, ScalarValuePtr START, ScalarValuePtr STEP, ScalarValuePtr SIZE, TileValuePtr output)
        : TileBaseOp(opcode, {ValueCast<Value>(START), ValueCast<Value>(STEP),ValueCast<Value>(SIZE)},
                             {ValueCast<Value>(output)}) {}
};

class ScatterTileBaseOp : public TileBaseOp {
public:
    // 对应 ScatterOp
    ScatterTileBaseOp(Opcode opcode, TileValuePtr Src0, TileValuePtr Src1, TileValuePtr Src2, TileValuePtr output)
        : TileBaseOp(opcode, {ValueCast<Value>(Src0), ValueCast<Value>(Src1), ValueCast<Value>(Src2)}, {ValueCast<Value>(output)}) {}
};

class ScatetrElementsTileBaseOp : public TileBaseOp {
public:
    // 对应 ScatterElementsOp
    ScatetrElementsTileBaseOp(Opcode opcode, TileValuePtr Src0, TileValuePtr Src1, ScalarValuePtr Scatter, TileValuePtr output)
        : TileBaseOp(opcode, {ValueCast<Value>(Src0), ValueCast<Value>(Src1), ValueCast<Value>(Scatter)}, {ValueCast<Value>(output)}) {}
};

class GatherTileBaseOp : public TileBaseOp {
public:
    // 对应 GatherOp 和 GatherExtendedOp
    GatherTileBaseOp(Opcode opcode, TileValuePtr lhs, TileValuePtr rhs, TileValuePtr output)
        : TileBaseOp(opcode, {ValueCast<Value>(lhs), ValueCast<Value>(rhs)}, {ValueCast<Value>(output)}) {}
};

class ExpandTileBaseOp : public TileBaseOp {
public:
    // 对应 ExpandOp
    ExpandTileBaseOp(Opcode opcode, TileValuePtr input, TileValuePtr output)
        : TileBaseOp(opcode, {ValueCast<Value>(input)}, {ValueCast<Value>(output)}) {}
};

class OnehotTileBaseOp : public TileBaseOp {
public:
    // 对应 OnehotOp
    OnehotTileBaseOp(Opcode opcode, TileValuePtr input, TileValuePtr output)
        : TileBaseOp(opcode, {ValueCast<Value>(input)}, {ValueCast<Value>(output)}) {}
};

class ConcatTileBaseOp : public TileBaseOp {
public:
    // Concat通常输入是列表，这里假设用vector传入
    ConcatTileBaseOp(Opcode opcode, std::vector<ValuePtr> inputs, TileValuePtr output)
        : TileBaseOp(opcode, inputs, {ValueCast<Value>(output)}) {}
};

class TransposeTileBaseOp : public TileBaseOp {
public:
    // 对应 TransposeOp 和 TransposeVnchwConvOp
    TransposeTileBaseOp(Opcode opcode, TileValuePtr input, TileValuePtr output)
        : TileBaseOp(opcode, {ValueCast<Value>(input)}, {ValueCast<Value>(output)}) {}
};

class PowTileBaseOp : public TileBaseOp {
public:
    // 对应 PowOp (Input: Tile, Scalar)
    PowTileBaseOp(Opcode opcode, TileValuePtr lhs, ScalarValuePtr rhs, TileValuePtr output)
        : TileBaseOp(opcode, {ValueCast<Value>(lhs), ValueCast<Value>(rhs)}, {ValueCast<Value>(output)}) {}
};

class CumSumTileBaseOp : public TileBaseOp {
public:
    // 对应 CumSumOp
    CumSumTileBaseOp(Opcode opcode, TileValuePtr input, TileValuePtr output)
        : TileBaseOp(opcode, {ValueCast<Value>(input)}, {ValueCast<Value>(output)}) {}
};

class SortTileBaseOp : public TileBaseOp {
public:
    // 对应 SortOp (OP_BITSORT 等): Input(Tile) -> Output(Tile)
    SortTileBaseOp(Opcode opcode, TileValuePtr lhs, TileValuePtr output)
        : TileBaseOp(opcode, {ValueCast<Value>(lhs)}, {ValueCast<Value>(output)}) {}

    // 对应 TopKOp: Input(Tile, Scalar, Scalar, Scalar) -> Output(Tile, Tile)
    SortTileBaseOp(Opcode opcode, TileValuePtr lhs, ScalarValuePtr axis, ScalarValuePtr kVal, ScalarValuePtr order, TileValuePtr out1, TileValuePtr out2)
        : TileBaseOp(opcode, 
                     {ValueCast<Value>(lhs), ValueCast<Value>(axis), ValueCast<Value>(kVal), ValueCast<Value>(order)}, 
                     {ValueCast<Value>(out1), ValueCast<Value>(out2)}) {}
    
    // 对应 TileSortOp: Input(Tile*4, Scalar) -> Output(Tile, Tile)
    SortTileBaseOp(Opcode opcode, TileValuePtr in1, TileValuePtr in2, TileValuePtr in3, TileValuePtr in4, ScalarValuePtr kVal, TileValuePtr out, TileValuePtr temp)
        : TileBaseOp(opcode,
                     {ValueCast<Value>(in1), ValueCast<Value>(in2), ValueCast<Value>(in3), ValueCast<Value>(in4), ValueCast<Value>(kVal)},
                     {ValueCast<Value>(out), ValueCast<Value>(temp)}) {}
};

class CompareScalarTileBaseOp : public TileBaseOp {
public:
    // 对应 CompareScalarOp: Input(Tile, Scalar) -> Output(Tile, Tile)
    CompareScalarTileBaseOp(Opcode opcode, TileValuePtr in1, ScalarValuePtr in2, TileValuePtr output, TileValuePtr temp)
        : TileBaseOp(opcode, {ValueCast<Value>(in1), ValueCast<Value>(in2)}, {ValueCast<Value>(output), ValueCast<Value>(temp)}) {}
};

class CompareTileBaseOp : public TileBaseOp {
public:
    // 对应 CompareOp: Input(Tile, Tile) -> Output(Tile, Tile)
    CompareTileBaseOp(Opcode opcode, TileValuePtr in1, TileValuePtr in2, TileValuePtr output, TileValuePtr temp)
        : TileBaseOp(opcode, {ValueCast<Value>(in1), ValueCast<Value>(in2)}, {ValueCast<Value>(output), ValueCast<Value>(temp)}) {}
};

class BroadcastTileBaseOp : public TileBaseOp {
public:
    // 对应 FusedOp
    BroadcastTileBaseOp(Opcode opcode, TileValuePtr lhs, TileValuePtr rhs, TileValuePtr output)
        : TileBaseOp(opcode, {ValueCast<Value>(lhs), ValueCast<Value>(rhs)}, {ValueCast<Value>(output)}) {}
};

class BroadcastWithTempTileBaseOp : public TileBaseOp {
public:
    // 对应 BroadcastOp (OP_MAXIMUM 等)
    BroadcastWithTempTileBaseOp(Opcode opcode, TileValuePtr lhs, TileValuePtr rhs, TileValuePtr output, TileValuePtr temp)
        : TileBaseOp(opcode, {ValueCast<Value>(lhs), ValueCast<Value>(rhs)}, {ValueCast<Value>(output), ValueCast<Value>(temp)}) {}
};

class BroadcastBinaryTileOp : public TileBaseOp {
public:
    // 假设类似于普通Binary: Tile, Tile -> Tile
    BroadcastBinaryTileOp(Opcode opcode, TileValuePtr lhs, TileValuePtr rhs, TileValuePtr output)
        : TileBaseOp(opcode, {ValueCast<Value>(lhs), ValueCast<Value>(rhs)}, {ValueCast<Value>(output)}) {}
};

class DataCopyTileBaseOp : public TileBaseOp {
public:
    // 对应 MoveOp: Input(Tile) -> Output(Tile)
    DataCopyTileBaseOp(Opcode opcode, TileValuePtr input, TileValuePtr output)
        : TileBaseOp(opcode, {ValueCast<Value>(input)}, {ValueCast<Value>(output)}) {}

    // 对应 LoadOp, UBMoveOp: Input(Tile, Tile) -> Output(Tile)
    DataCopyTileBaseOp(Opcode opcode, TileValuePtr lhs, TileValuePtr rhs, TileValuePtr output)
        : TileBaseOp(opcode, {ValueCast<Value>(lhs), ValueCast<Value>(rhs)}, {ValueCast<Value>(output)}) {}

    // 对应 ExperimentOp: Input(Tile, Tile, Tile) -> Output(Tile)
    DataCopyTileBaseOp(Opcode opcode, TileValuePtr in1, TileValuePtr in2, TileValuePtr in3, TileValuePtr output)
        : TileBaseOp(opcode, {ValueCast<Value>(in1), ValueCast<Value>(in2), ValueCast<Value>(in3)}, {ValueCast<Value>(output)}) {}

    // 对应 IndexOp: Input(Tile*3, Scalar*3) -> Output(Tile)
    DataCopyTileBaseOp(Opcode opcode, TileValuePtr src, TileValuePtr idx, TileValuePtr dst, ScalarValuePtr axis, ScalarValuePtr bsize, ScalarValuePtr cmode, TileValuePtr output)
        : TileBaseOp(opcode,
                     {ValueCast<Value>(src), ValueCast<Value>(idx), ValueCast<Value>(dst), ValueCast<Value>(axis), ValueCast<Value>(bsize), ValueCast<Value>(cmode)},
                     {ValueCast<Value>(output)}) {}
};

class MatmulTileBaseOp : public TileBaseOp {
public:
    // 基础Matmul: Lhs, Rhs -> Output
    MatmulTileBaseOp(Opcode opcode, TileValuePtr lhs, TileValuePtr rhs, TileValuePtr output)
        : TileBaseOp(opcode, {ValueCast<Value>(lhs), ValueCast<Value>(rhs)}, {ValueCast<Value>(output)}) {}
};

class SysBaseOp : public ScalarBaseOp {
public:
    std::string GetName() const;
private:
    std::string name_;
};

class CustomTileBaseOp : public TileBaseOp {
public:
    CustomTileBaseOp(Opcode opcode, std::vector<ValuePtr> iops, std::vector<ValuePtr> oops)
     : TileBaseOp(opcode, iops, oops) {}
};

} // namespace pto