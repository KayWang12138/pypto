/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once

#include <cstdint>
#include <vector>
#include <ostream>

#include "tilefwk/data_type.h"
#include "tilefwk/element.h"
#include "raw_tensor_data.h"

namespace npu::tile_fwk::calc {

extern "C" {
void Dump(std::ostream &os, LogicalTensorDataPtr self);

bool AllClose(LogicalTensorDataPtr self, LogicalTensorDataPtr other, double atol = 1e-8, double rtol = 1e-5);
void Cast(LogicalTensorDataPtr out, LogicalTensorDataPtr self, CastMode mode = CAST_NONE);
void Exp(LogicalTensorDataPtr out, LogicalTensorDataPtr self);
void Sqrt(LogicalTensorDataPtr out, LogicalTensorDataPtr self);
void Abs(LogicalTensorDataPtr out, LogicalTensorDataPtr self);

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

// matmul
void MatMul(LogicalTensorDataPtr out, LogicalTensorDataPtr self, LogicalTensorDataPtr other, bool atrans, bool btrans,
    bool acc, int64_t kstep);
}

#ifndef ENABLE_VERIFIER
#include "calc_stub.h"
#endif

inline std::ostream &operator<<(std::ostream &os, LogicalTensorDataPtr self) {
    Dump(os, self);
    return os;
}

template <bool aTrans = false, bool bTrans = false>
inline void MatMul(LogicalTensorDataPtr out, LogicalTensorDataPtr self, LogicalTensorDataPtr other, int64_t kstep = 0) {
    MatMul(out, self, other, aTrans, bTrans, false, kstep);
}

template <bool aTrans = false, bool bTrans = false>
inline void AccMatMul(
    LogicalTensorDataPtr out, LogicalTensorDataPtr self, LogicalTensorDataPtr other, int64_t kstep = 0) {
    MatMul(out, self, other, aTrans, bTrans, true, kstep);
}

} // namespace npu::tile_fwk::calc