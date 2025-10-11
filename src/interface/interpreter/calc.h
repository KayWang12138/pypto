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
  * \file calc.h
  * \brief
  */

#pragma once

#include <cstdint>
#include <vector>
#include <ostream>

#include "tilefwk/data_type.h"
#include "tilefwk/element.h"
#include "raw_tensor_data.h"

namespace npu::tile_fwk::calc {

struct MatMulParam {
    bool aTrans = false;
    bool bTrans = false;
    int64_t kStep = 0;
};

extern "C" {
const char *Model();
void Dump(std::ostream &os, LogicalTensorDataPtr self);

bool AllClose(LogicalTensorDataPtr self, LogicalTensorDataPtr other, double atol = 1e-8, double rtol = 1e-5);
void Cast(LogicalTensorDataPtr out, LogicalTensorDataPtr self, CastMode mode = CAST_NONE);
void Exp(LogicalTensorDataPtr out, LogicalTensorDataPtr self);
void Neg(LogicalTensorDataPtr out, LogicalTensorDataPtr self);
void Rsqrt(LogicalTensorDataPtr out, LogicalTensorDataPtr self);
void Sqrt(LogicalTensorDataPtr out, LogicalTensorDataPtr self);
void Abs(LogicalTensorDataPtr out, LogicalTensorDataPtr self);
void Ln(LogicalTensorDataPtr out, LogicalTensorDataPtr self);
void LogicalNot(LogicalTensorDataPtr out, LogicalTensorDataPtr self);

void AddS(LogicalTensorDataPtr out, LogicalTensorDataPtr self, const Element &scalar, bool reverse = false);
void SubS(LogicalTensorDataPtr out, LogicalTensorDataPtr self, const Element &scalar, bool reverse = false);
void MulS(LogicalTensorDataPtr out, LogicalTensorDataPtr self, const Element &scalar, bool reverse = false);
void DivS(LogicalTensorDataPtr out, LogicalTensorDataPtr self, const Element &scalar, bool reverse = false);

void Add(LogicalTensorDataPtr out, LogicalTensorDataPtr self, LogicalTensorDataPtr other);
void Sub(LogicalTensorDataPtr out, LogicalTensorDataPtr self, LogicalTensorDataPtr other);
void Mul(LogicalTensorDataPtr out, LogicalTensorDataPtr self, LogicalTensorDataPtr other);
void Div(LogicalTensorDataPtr out, LogicalTensorDataPtr self, LogicalTensorDataPtr other);

void Min(LogicalTensorDataPtr out, LogicalTensorDataPtr self, LogicalTensorDataPtr other);
void Max(LogicalTensorDataPtr out, LogicalTensorDataPtr self, LogicalTensorDataPtr other);
void MinS(LogicalTensorDataPtr out, LogicalTensorDataPtr self, const Element &scalar);
void MaxS(LogicalTensorDataPtr out, LogicalTensorDataPtr self, const Element &scalar);

/* used by reducc op, if shape are not same, need masked */
void PairSum(LogicalTensorDataPtr out, LogicalTensorDataPtr self, LogicalTensorDataPtr other);
void PairMax(LogicalTensorDataPtr out, LogicalTensorDataPtr self, LogicalTensorDataPtr other);
void PairMin(LogicalTensorDataPtr out, LogicalTensorDataPtr self, LogicalTensorDataPtr other);

void RowSumExpand(LogicalTensorDataPtr out, LogicalTensorDataPtr self, int dim);
void RowMinExpand(LogicalTensorDataPtr out, LogicalTensorDataPtr self, int dim);
void RowMaxExpand(LogicalTensorDataPtr out, LogicalTensorDataPtr self, int dim);

void RowSumSingle(LogicalTensorDataPtr out, LogicalTensorDataPtr self, int dim);
void RowMinSingle(LogicalTensorDataPtr out, LogicalTensorDataPtr self, int dim);
void RowMaxSingle(LogicalTensorDataPtr out, LogicalTensorDataPtr self, int dim);

void ExpandS(LogicalTensorDataPtr out, const Element &scalar);
void Expand(LogicalTensorDataPtr out, LogicalTensorDataPtr self);

void Reshape(LogicalTensorDataPtr out, LogicalTensorDataPtr self);
void Permute(LogicalTensorDataPtr out, LogicalTensorDataPtr self, const std::vector<int64_t> &dim);
void Transpose(LogicalTensorDataPtr out, LogicalTensorDataPtr self, int64_t dim0, int64_t dim1);

void ReduceAcc(LogicalTensorDataPtr out, const std::vector<LogicalTensorDataPtr> &tdatas);

void Copy(LogicalTensorDataPtr out, LogicalTensorDataPtr self, bool trans = false);
void ScatterUpdate(LogicalTensorDataPtr out, LogicalTensorDataPtr self, LogicalTensorDataPtr index, int axis = -2,
    std::string cacheMode = "BSND", int blockSize = 1);
void Scatter(LogicalTensorDataPtr out, LogicalTensorDataPtr self, LogicalTensorDataPtr index, const Element &src, 
    int axis, std::string reduce = "");
void BitSort(LogicalTensorDataPtr out, LogicalTensorDataPtr self, int64_t axis, bool descending);
void Gather(LogicalTensorDataPtr out, LogicalTensorDataPtr params, LogicalTensorDataPtr indices, int64_t axis);

void Extract(LogicalTensorDataPtr out, LogicalTensorDataPtr self, int mod, bool descending);

void Topk(LogicalTensorDataPtr out, LogicalTensorDataPtr self, int64_t axis, int64_t k, bool descending);

// matmul
void FormatNZ2ND(LogicalTensorDataPtr out, LogicalTensorDataPtr self);
void FormatND2NZ(LogicalTensorDataPtr out, LogicalTensorDataPtr self);

void MatMul(LogicalTensorDataPtr out, LogicalTensorDataPtr self, LogicalTensorDataPtr other, LogicalTensorDataPtr acc,
    MatMulParam &param);
}

#ifndef ENABLE_VERIFIER
#include "calc_stub.h"
#endif

inline std::ostream &operator<<(std::ostream &os, LogicalTensorDataPtr self) {
    Dump(os, self);
    return os;
}

template <bool aTrans = false, bool bTrans = false>
inline void MatMul(LogicalTensorDataPtr out, LogicalTensorDataPtr self, LogicalTensorDataPtr other, int64_t kStep = 0) {
    MatMulParam param = {aTrans, bTrans, kStep};
    MatMul(out, self, other, nullptr, param);
}

template <bool aTrans = false, bool bTrans = false>
inline void AccMatMul(LogicalTensorDataPtr out, LogicalTensorDataPtr self, LogicalTensorDataPtr other,
                      LogicalTensorDataPtr acc = nullptr, int64_t kStep = 0) {
    MatMulParam param = {aTrans, bTrans, kStep};
    MatMul(out, self, other, acc, param);
}

} // namespace npu::tile_fwk::calc