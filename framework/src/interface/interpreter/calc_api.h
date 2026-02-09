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
 * \file calc_api.h
 * \brief Calculator API
 */

#pragma once

#include <cstdint>
#include <ostream>
#include "tilefwk/data_type.h"
#include "tilefwk/element.h"

namespace npu::tile_fwk {
struct MatMulParam {
    bool aTrans = false;
    bool bTrans = false;
    int64_t kStep = 0;
};

enum class CmpOperationType {
    EQ,
    NE,
    LT,
    LE,
    GT,
    GE,
};
enum class CmpModeType {
    BOOL,
    BIT,
};

struct CalcTensorData {
    void *dataPtr = nullptr;
    std::vector<int64_t> rawShape;
    std::vector<int64_t> shape;
    std::vector<int64_t> stride;
    int64_t storageOffset;
    DataType dtype;
    bool isAxisCombine = false;
};

struct CalcOps {
    void (*Random)(const CalcTensorData &);
    bool (*AllClose)(const CalcTensorData &, const CalcTensorData &, double, double);

    void (*Cast)(const CalcTensorData &, const CalcTensorData &, CastMode);
    void (*Exp)(const CalcTensorData &, const CalcTensorData &);
    void (*Neg)(const CalcTensorData &, const CalcTensorData &);
    void (*Rsqrt)(const CalcTensorData &, const CalcTensorData &);
    void (*Sqrt)(const CalcTensorData &, const CalcTensorData &);
    void (*Ceil)(const CalcTensorData &, const CalcTensorData &);
    void (*Floor)(const CalcTensorData &, const CalcTensorData &);
    void (*Trunc)(const CalcTensorData &, const CalcTensorData &);
    void (*Round)(const CalcTensorData &, const CalcTensorData &, int);
    void (*Reciprocal)(const CalcTensorData &, const CalcTensorData &);
    void (*BitwiseNot)(const CalcTensorData &, const CalcTensorData &);
    void (*Abs)(const CalcTensorData &, const CalcTensorData &);
    void (*Brcb)(const CalcTensorData &, const CalcTensorData &);  
    void (*WhereTT)(const CalcTensorData &, const CalcTensorData &, const CalcTensorData &, const CalcTensorData &);
    void (*WhereTS)(const CalcTensorData &, const CalcTensorData &, const CalcTensorData &, const Element &);
    void (*WhereST)(const CalcTensorData &, const CalcTensorData &, const Element &, const CalcTensorData &);
    void (*WhereSS)(const CalcTensorData &, const CalcTensorData &, const Element &, const Element &);
    void (*Ln)(const CalcTensorData &, const CalcTensorData &);
    void (*LogicalNot)(const CalcTensorData &, const CalcTensorData &);
    void (*Range)(const CalcTensorData &, const Element &, const Element &, const Element &);
    void (*Compare)(const CalcTensorData &, const CalcTensorData &, const CalcTensorData &, CmpOperationType, CmpModeType);
    void (*Cmps)(const CalcTensorData &, const CalcTensorData &, const Element &, CmpOperationType, CmpModeType);
    void (*LogicalAnd)(const CalcTensorData &, const CalcTensorData &, const CalcTensorData &);

    void (*AddS)(const CalcTensorData &, const CalcTensorData &, const Element &, bool);
    void (*SubS)(const CalcTensorData &, const CalcTensorData &, const Element &, bool);
    void (*MulS)(const CalcTensorData &, const CalcTensorData &, const Element &, bool);
    void (*DivS)(const CalcTensorData &, const CalcTensorData &, const Element &, bool);
    void (*FmodS)(const CalcTensorData &, const CalcTensorData &, const Element &, bool);
    void (*BitwiseAndS)(const CalcTensorData &, const CalcTensorData &, const Element &, bool);
    void (*BitwiseOrS)(const CalcTensorData &, const CalcTensorData &, const Element &, bool);
    void (*BitwiseXorS)(const CalcTensorData &, const CalcTensorData &, const Element &, bool);

    void (*Add)(const CalcTensorData &, const CalcTensorData &, const CalcTensorData &);
    void (*Sub)(const CalcTensorData &, const CalcTensorData &, const CalcTensorData &);
    void (*Mul)(const CalcTensorData &, const CalcTensorData &, const CalcTensorData &);
    void (*Div)(const CalcTensorData &, const CalcTensorData &, const CalcTensorData &);
    void (*Fmod)(const CalcTensorData &, const CalcTensorData &, const CalcTensorData &);
    void (*BitwiseAnd)(const CalcTensorData &, const CalcTensorData &, const CalcTensorData &);
    void (*BitwiseOr)(const CalcTensorData &, const CalcTensorData &, const CalcTensorData &);
    void (*BitwiseXor)(const CalcTensorData &, const CalcTensorData &, const CalcTensorData &);
    void (*CopySign)(const CalcTensorData &, const CalcTensorData &, const CalcTensorData &);

    void (*PairSum)(const CalcTensorData &, const CalcTensorData &, const CalcTensorData &);
    void (*PairMax)(const CalcTensorData &, const CalcTensorData &, const CalcTensorData &);
    void (*PairMin)(const CalcTensorData &, const CalcTensorData &, const CalcTensorData &);

    void (*Min)(const CalcTensorData &, const CalcTensorData &, const CalcTensorData &);
    void (*Max)(const CalcTensorData &, const CalcTensorData &, const CalcTensorData &);
    void (*MinS)(const CalcTensorData &, const CalcTensorData &, const Element &);
    void (*MaxS)(const CalcTensorData &, const CalcTensorData &, const Element &);

    void (*RowSumExpand)(const CalcTensorData &, const CalcTensorData &, int);
    void (*RowMinExpand)(const CalcTensorData &, const CalcTensorData &, int);
    void (*RowMaxExpand)(const CalcTensorData &, const CalcTensorData &, int);

    void (*RowSumSingle)(const CalcTensorData &, const CalcTensorData &, int);
    void (*RowMinSingle)(const CalcTensorData &, const CalcTensorData &, int);
    void (*RowMaxSingle)(const CalcTensorData &, const CalcTensorData &, int);

    void (*RowMinLine)(const CalcTensorData &, const CalcTensorData &, int);
    void (*RowMaxLine)(const CalcTensorData &, const CalcTensorData &, int);

    void (*OneHot)(const CalcTensorData &, const CalcTensorData &, int);
    void (*ExpandS)(const CalcTensorData &, const Element &);
    void (*Expand)(const CalcTensorData &, const CalcTensorData &);
    void (*GatherElements)(const CalcTensorData &, const CalcTensorData &, const CalcTensorData &, int);
    void (*IndexAdd)(const CalcTensorData &, const CalcTensorData &, const CalcTensorData &, const CalcTensorData &, int, const Element &);
    void (*TriU)(const CalcTensorData &, const CalcTensorData &, int);
    void (*TriL)(const CalcTensorData &, const CalcTensorData &, int);
    void (*CumSum)(const CalcTensorData &, const CalcTensorData &, int);
    void (*IndexPut)(const CalcTensorData &, const CalcTensorData &, const std::vector<CalcTensorData> &, const CalcTensorData &, bool);

    void (*Reshape)(const CalcTensorData &, const CalcTensorData &);
    void (*Permute)(const CalcTensorData &, const CalcTensorData &, const std::vector<int64_t> &);
    void (*Transpose)(const CalcTensorData &, const CalcTensorData &, int64_t, int64_t);

    void (*ReduceAcc)(const CalcTensorData &, const std::vector<CalcTensorData> &);
    void (*Copy)(const CalcTensorData &, const CalcTensorData &, bool);
    void (*ScatterUpdate)(const CalcTensorData &, const CalcTensorData &, const CalcTensorData &, const CalcTensorData &, int, std::string, int);
    void (*ScatterElement)(const CalcTensorData &, const CalcTensorData &, const CalcTensorData &, const Element &, int, int);
    void (*Scatter)(const CalcTensorData &, const CalcTensorData &, const CalcTensorData &, const CalcTensorData &,
        int, int);
    void (*FormatND2NZ)(const CalcTensorData &, const CalcTensorData &);
    void (*FormatNZ2ND)(const CalcTensorData &, const CalcTensorData &);
    void (*MatMul)(const CalcTensorData &, const CalcTensorData &, const CalcTensorData &, const CalcTensorData *, MatMulParam &);

    void (*BitSort)(const CalcTensorData &, const CalcTensorData &, int64_t, bool, int64_t);
    void (*TiledMrgSort)(const CalcTensorData &, const CalcTensorData &, const CalcTensorData &, const CalcTensorData &, const CalcTensorData &, int, int);
    void (*Extract)(const CalcTensorData &, const CalcTensorData &, int, bool);
    void (*Topk)(const CalcTensorData &, const CalcTensorData &, int64_t, int64_t, bool);
    void (*TopK)(const CalcTensorData &, const CalcTensorData &, const CalcTensorData &, int, int, bool);
    void (*TopkSort)(const CalcTensorData &, const CalcTensorData &, const CalcTensorData &, int);
    void (*TopkMerge)(const CalcTensorData &, const CalcTensorData &, int);
    void (*TopkExtract)(const CalcTensorData &, const CalcTensorData &, int, bool);
    void (*TwoTileMrgSort)(const CalcTensorData &, const CalcTensorData &);
    void (*Sort)(const CalcTensorData &, const CalcTensorData &, const CalcTensorData &, int64_t, bool);
    void (*Gather)(const CalcTensorData &, const CalcTensorData &, const CalcTensorData &, int64_t);
    void (*GatherINUB)(
        const CalcTensorData &, const CalcTensorData &, const CalcTensorData &, const CalcTensorData &, int64_t, int64_t);
    void (*BitwiseRightShift)(const CalcTensorData &, const CalcTensorData &, const CalcTensorData &);
    void (*BitwiseLeftShift)(const CalcTensorData &, const CalcTensorData &, const CalcTensorData &);
    void (*BitwiseRightShiftS)(const CalcTensorData &, const CalcTensorData &, const Element &);
    void (*BitwiseLeftShiftS)(const CalcTensorData &, const CalcTensorData &, const Element &);
    void (*SBitwiseRightShift)(const CalcTensorData &, const Element &, const CalcTensorData &);
    void (*SBitwiseLeftShift)(const CalcTensorData &, const Element &, const CalcTensorData &);
};

extern "C" struct CalcOps *GetCalcOps();
} // namespace npu::tile_fwk
