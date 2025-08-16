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

inline void Dump(std::ostream &, LogicalTensorDataPtr) {}
inline bool AllClose(LogicalTensorDataPtr, LogicalTensorDataPtr, double, double) { return true; }

inline void Cast(LogicalTensorDataPtr, LogicalTensorDataPtr, CastMode) {}
inline void Exp(LogicalTensorDataPtr, LogicalTensorDataPtr) {}
inline void Sqrt(LogicalTensorDataPtr, LogicalTensorDataPtr) {}
inline void Abs(LogicalTensorDataPtr, LogicalTensorDataPtr) {}

inline void AddS(LogicalTensorDataPtr, LogicalTensorDataPtr, const Element &, bool) {}
inline void SubS(LogicalTensorDataPtr, LogicalTensorDataPtr, const Element &, bool) {}
inline void MulS(LogicalTensorDataPtr, LogicalTensorDataPtr, const Element &, bool) {}
inline void DivS(LogicalTensorDataPtr, LogicalTensorDataPtr, const Element &, bool) {}

inline void Add(LogicalTensorDataPtr, LogicalTensorDataPtr, LogicalTensorDataPtr) {}
inline void Sub(LogicalTensorDataPtr, LogicalTensorDataPtr, LogicalTensorDataPtr) {}
inline void Mul(LogicalTensorDataPtr, LogicalTensorDataPtr, LogicalTensorDataPtr) {}
inline void Div(LogicalTensorDataPtr, LogicalTensorDataPtr, LogicalTensorDataPtr) {}

inline void Min(LogicalTensorDataPtr, LogicalTensorDataPtr, LogicalTensorDataPtr) {}
inline void Max(LogicalTensorDataPtr, LogicalTensorDataPtr, LogicalTensorDataPtr) {}
inline void MinS(LogicalTensorDataPtr, LogicalTensorDataPtr, const Element &) {}
inline void MaxS(LogicalTensorDataPtr, LogicalTensorDataPtr, const Element &) {}

inline void RowSumExpand(LogicalTensorDataPtr, LogicalTensorDataPtr, int) {}
inline void RowMinExpand(LogicalTensorDataPtr, LogicalTensorDataPtr, int) {}
inline void RowMaxExpand(LogicalTensorDataPtr, LogicalTensorDataPtr, int) {}

inline void RowSumSingle(LogicalTensorDataPtr, LogicalTensorDataPtr, int) {}
inline void RowMinSingle(LogicalTensorDataPtr, LogicalTensorDataPtr, int) {}
inline void RowMaxSingle(LogicalTensorDataPtr, LogicalTensorDataPtr, int) {}

inline void ExpandS(LogicalTensorDataPtr, const Element &) {}
inline void Expand(LogicalTensorDataPtr, LogicalTensorDataPtr) {}

inline void Reshape(LogicalTensorDataPtr, LogicalTensorDataPtr) {}
inline void Permute(LogicalTensorDataPtr, LogicalTensorDataPtr, const std::vector<int64_t> &) {}
inline void Transpose(LogicalTensorDataPtr, LogicalTensorDataPtr, int64_t, int64_t) {}

inline void ReduceAcc(LogicalTensorDataPtr, const std::vector<LogicalTensorDataPtr> &) {}
inline void Copy(LogicalTensorDataPtr, LogicalTensorDataPtr, bool) {}

inline void MatMul(LogicalTensorDataPtr, LogicalTensorDataPtr, LogicalTensorDataPtr, bool, bool, bool, int64_t) {}